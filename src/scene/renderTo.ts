// Scene-level offscreen rendering — Aardvark.Dom's
// `IRuntime.RenderTo` / `RenderToPickable` ported (RenderTo.fs,
// commits 0e9d698 / 3844127).
//
//   renderSceneTo(runtime, device, scene, opts)
//     → offscreen textures for an SgNode scene. The color / depth
//       attachments come back as `AdaptiveResource<ITexture>`; pulling
//       one during an outer frame runs the inner render task first, so
//       compositing an offscreen scene is just "bind the texture on a
//       downstream leaf". All post passes (blur / AO / warp /
//       composite) build on this.
//
//   renderToPickable(runtime, device, scene, opts)
//     → the same through a `PickProducer`: the offscreen pass carries
//       the pickId attachment, and the result additionally exposes
//       `pick: IRenderPickContext` — the recursion handle that lets
//       Sg.onClick etc. work INSIDE the offscreen scene when its color
//       texture is composited onto host geometry (see
//       `picking/pickContext.ts` and the `PickContext` scene
//       attribute).
//
// Lifetime: the returned textures are ref-counted AdaptiveResources
// (acquire → allocates the FBO, last release → frees it); `dispose()`
// tears down the compiled task, the pick artifacts, and the acquired
// pick ids regardless of outstanding texture refs.

import { AVal, type AdaptiveToken, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V2i } from "@aardworx/wombat.base";
import {
  AdaptiveResource,
  ITexture,
  type ClearValues,
  type Effect,
  type FramebufferSignature,
  type IFramebuffer,
} from "@aardworx/wombat.rendering/core";
import {
  allocateFramebuffer,
  createFramebufferSignature,
  type Runtime,
} from "@aardworx/wombat.rendering";

import { compileScene } from "./compile.js";
import { sniffViewProj } from "./renderControl.js";
import type { SgNode } from "./sg.js";
import { TraversalState } from "./traversalState.js";
import type { OitMode } from "./transparency.js";
import { globalTime } from "./ambient.js";
import { arbitratePick, resolveThroughPortals } from "./picking/pickArbitrate.js";
import type { IRenderPickContext, PortalPickHit } from "./picking/pickContext.js";
import { createPickProducer, type PickProducer } from "./picking/pickProducer.js";

// ---------------------------------------------------------------------------
// Options / results
// ---------------------------------------------------------------------------

export interface RenderSceneToOptions {
  /** Offscreen render-target size (device pixels). */
  readonly size: aval<{ readonly width: number; readonly height: number }>;
  /** View trafo. Falls back to the scene's outermost `<Sg View=…>`,
   *  then identity. Required in spirit for `renderToPickable` — the
   *  pick unprojection is meaningless without a real camera. */
  readonly view?: aval<Trafo3d>;
  readonly proj?: aval<Trafo3d>;
  /** Color format of the offscreen target. Default `"rgba8unorm"`. */
  readonly format?: GPUTextureFormat;
  /** Depth format. Default `"depth24plus"`; `null` for no depth. */
  readonly depthFormat?: GPUTextureFormat | null;
  readonly sampleCount?: number;
  /** Color-attachment name. Default `"Colors"` (the wombat.shader
   *  convention — user fragment outputs are named `Colors`). */
  readonly colorAttachmentName?: string;
  /** Clear values; color / depth (and pickId, when pickable) defaults
   *  are injected exactly like RenderControl's. */
  readonly clear?: ClearValues;
  readonly defaultEffect?: Effect;
  /** Order-independent transparency for `Sg.transparent` subtrees. */
  readonly transparency?: boolean | OitMode;
  /** Per-frame clock threaded into the traversal state; defaults to
   *  the ambient RenderControl clock. */
  readonly time?: aval<number>;
  readonly label?: string;
}

export interface RenderSceneToResult {
  readonly signature: FramebufferSignature;
  readonly size: aval<{ readonly width: number; readonly height: number }>;
  /** The offscreen framebuffer AFTER this frame's inner render. */
  readonly framebuffer: AdaptiveResource<IFramebuffer>;
  /** Named color attachment (default: the main color). Acquiring it
   *  activates the whole offscreen pipeline. */
  color(name?: string): AdaptiveResource<ITexture>;
  /** Depth attachment (throws when the target has no depth). */
  depthStencil(): AdaptiveResource<ITexture>;
  dispose(): void;
}

export interface RenderToPickableResult extends RenderSceneToResult {
  /** The pick half — recursion handle + camera avals + registry. */
  readonly pick: IRenderPickContext;
  /**
   * The single-sample rgba32float pick attachment — doubles as a free
   * G-buffer: slot 0 pickId, slot 1 oct24 view-space normal, slot 2
   * NDC depth (Mode-A pixels; cleared pixels are 0). Sample with
   * `textureLoad` (unfilterable float). Pull it AFTER this frame's
   * `framebuffer` so the inner render has run.
   */
  readonly pickTexture: aval<GPUTexture>;
}

// ---------------------------------------------------------------------------
// TaskFramebuffer — an AdaptiveResource<IFramebuffer> that runs a
// render callback on compute. The dom-side twin of wombat.rendering's
// `RenderToFramebuffer`, but task-based: the inner IRenderTask encodes
// and submits its own passes, so no RenderContext.encoder threading is
// needed. Queue order stays correct — the inner submission lands
// before the outer frame's encoder is finished and submitted.
// ---------------------------------------------------------------------------

class TaskFramebuffer extends AdaptiveResource<IFramebuffer> {
  constructor(
    private readonly fbo: AdaptiveResource<IFramebuffer>,
    private readonly renderFrame: (token: AdaptiveToken) => void,
    private readonly isDisposed: () => boolean,
  ) { super(); }

  protected override create(): void { this.fbo.acquire(); }
  protected override destroy(): void { this.fbo.release(); }

  override compute(token: AdaptiveToken): IFramebuffer {
    const fb = this.fbo.getValue(token);
    if (!this.isDisposed()) this.renderFrame(token);
    return fb;
  }
}

function textureAccessors(
  base: AdaptiveResource<IFramebuffer>,
  signature: FramebufferSignature,
  defaultColorName: string,
): Pick<RenderSceneToResult, "color" | "depthStencil"> {
  return {
    color(name?: string): AdaptiveResource<ITexture> {
      const n = name ?? defaultColorName;
      return base.derive<ITexture>((fb) => {
        const tex = fb.colorTextures?.tryFind(n);
        if (tex === undefined) {
          throw new Error(`renderSceneTo.color: framebuffer has no color attachment "${n}"`);
        }
        return ITexture.fromGPU(tex);
      });
    },
    depthStencil(): AdaptiveResource<ITexture> {
      return base.derive<ITexture>((fb) => {
        if (fb.depthStencilTexture === undefined) {
          throw new Error("renderSceneTo.depthStencil: framebuffer has no depth-stencil attachment");
        }
        return ITexture.fromGPU(fb.depthStencilTexture);
      });
    },
  };
}

function resolveCamera(
  scene: SgNode,
  opts: RenderSceneToOptions,
  what: string,
): { view: aval<Trafo3d>; proj: aval<Trafo3d> } {
  const sniffed = sniffViewProj(scene);
  const view = opts.view ?? sniffed.view;
  const proj = opts.proj ?? sniffed.proj;
  if (view === undefined || proj === undefined) {
    console.warn(`${what}: no ${view === undefined ? "View" : "Proj"} provided (prop or outermost scene scope); defaulting to identity.`);
  }
  return {
    view: view ?? AVal.constant(Trafo3d.identity),
    proj: proj ?? AVal.constant(Trafo3d.identity),
  };
}

// ---------------------------------------------------------------------------
// renderSceneTo — non-pick offscreen primitive
// ---------------------------------------------------------------------------

export function renderSceneTo(
  runtime: Runtime,
  device: GPUDevice,
  scene: SgNode,
  opts: RenderSceneToOptions,
): RenderSceneToResult {
  const colorName = opts.colorAttachmentName ?? "Colors";
  const format = opts.format ?? "rgba8unorm";
  const depthFormat = opts.depthFormat === undefined ? "depth24plus" : opts.depthFormat;
  const signature = createFramebufferSignature({
    colors: { [colorName]: format },
    ...(depthFormat !== null ? { depthStencil: { format: depthFormat } } : {}),
    ...(opts.sampleCount !== undefined ? { sampleCount: opts.sampleCount } : {}),
  });
  const fbo = allocateFramebuffer(device, signature, opts.size, {
    ...(opts.label !== undefined ? { labelPrefix: opts.label } : {}),
  });

  const { view, proj } = resolveCamera(scene, opts, "renderSceneTo");
  const initial = TraversalState.empty
    .withViewport(opts.size)
    .withCamera(view, proj)
    .withTime(opts.time ?? globalTime());

  const task = runtime.compile(signature, compileScene(scene, {
    initialState: initial,
    ...(opts.defaultEffect !== undefined ? { defaultEffect: opts.defaultEffect } : {}),
    ...(opts.clear !== undefined ? { clear: opts.clear } : {}),
  }));

  let disposed = false;
  const base = new TaskFramebuffer(
    fbo,
    (token) => task.run(fbo.getValue(token), token),
    () => disposed,
  );

  return {
    signature,
    size: opts.size,
    framebuffer: base,
    ...textureAccessors(base, signature, colorName),
    dispose(): void {
      if (disposed) return;
      disposed = true;
      task.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// renderToPickable — the same through a PickProducer
// ---------------------------------------------------------------------------

export function renderToPickable(
  runtime: Runtime,
  device: GPUDevice,
  scene: SgNode,
  opts: RenderSceneToOptions,
): RenderToPickableResult {
  const colorName = opts.colorAttachmentName ?? "Colors";
  const format = opts.format ?? "rgba8unorm";
  const depthFormat = opts.depthFormat === undefined ? "depth24plus" : opts.depthFormat;
  const baseSignature = createFramebufferSignature({
    colors: { [colorName]: format },
    ...(depthFormat !== null ? { depthStencil: { format: depthFormat } } : {}),
    ...(opts.sampleCount !== undefined ? { sampleCount: opts.sampleCount } : {}),
  });
  const fbo = allocateFramebuffer(device, baseSignature, opts.size, {
    labelPrefix: opts.label ?? "renderToPickable",
  });

  const { view, proj } = resolveCamera(scene, opts, "renderToPickable");

  // The PickProducer wraps the offscreen target exactly like it wraps
  // the canvas attachment in RenderControl — the "target" here is the
  // base FBO; the producer adds the pickId attachment on top.
  const producer: PickProducer = createPickProducer(runtime, device, {
    framebuffer: fbo,
    size: opts.size,
    signature: baseSignature,
  }, scene, {
    view, proj,
    time: opts.time ?? globalTime(),
    ...(opts.transparency !== undefined ? { transparency: opts.transparency } : {}),
    ...(opts.defaultEffect !== undefined ? { defaultEffect: opts.defaultEffect } : {}),
    ...(opts.clear !== undefined ? { clear: opts.clear } : {}),
    colorAttachmentName: colorName,
  });

  let disposed = false;
  const base = new TaskFramebuffer(
    fbo,
    (token) => producer.run(token),
    () => disposed,
  );

  const pick: IRenderPickContext = {
    size: opts.size,
    view, proj,
    registry: producer.registry,
    async pickAt(x: number, y: number): Promise<PortalPickHit | undefined> {
      if (disposed) return undefined;
      const result = await producer.pickPixel(x, y, false);
      if (disposed) return undefined;
      // AVal.force OK: pick-time snapshot — "now" is the caller's tick
      // (same policy as the dispatcher's resolve).
      const v = AVal.force(view);
      const p = AVal.force(proj);
      const sz = AVal.force(opts.size);
      const viewportSize = new V2i(sz.width, sz.height);
      // Arbitrate in THIS scene, then recurse — the inner scene may
      // itself contain portals (arbitrary nesting, F# parity).
      const hit = await resolveThroughPortals(
        arbitratePick(result, { devX: x, devY: y }, producer.registry, v, p, viewportSize),
      );
      if (hit === undefined || disposed) return undefined;
      return { hit, registry: hit.registry ?? producer.registry };
    },
  };

  return {
    signature: baseSignature,
    size: opts.size,
    framebuffer: base,
    ...textureAccessors(base, baseSignature, colorName),
    pick,
    pickTexture: producer.readbackPickTexture,
    dispose(): void {
      if (disposed) return;
      disposed = true;
      producer.dispose();
    },
  };
}
