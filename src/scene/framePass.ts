// Chainable screen-space frame passes — the extension point for
// post-scene effects. After the scene renders (color + pick G-buffer),
// the RenderControl walks an ordered chain of passes; each may encode
// GPU work against the frame and may return a REPLACEMENT pick-shaped
// texture (rgba32float, .z = NDC depth) that downstream passes see —
// e.g. voronoi fill hands plane-fit shading a hole-filled depth buffer.
//
// Pixel picking is NOT routed through the chain: pointer/readback picks
// always use the producer's own pick texture, so a pass can reshape
// what SHADING sees without breaking interaction.
//
// Liveness: `enabled` and `version` are read through the frame token.
// Toggling `enabled` or transacting `version` marks the frame loop —
// keep a settings object, mutate a field, bump `version`, and the next
// frame renders with the new value (the built-in passes re-read their
// settings every frame).

import { type aval } from "@aardworx/wombat.adaptive";
import type { Trafo3d } from "@aardworx/wombat.base";
import type { IFramebuffer } from "@aardworx/wombat.rendering/core";
import { createGtaoPass, gtaoConfig, type GtaoOption } from "./gtao.js";
import { createPlaneFitPass, planeFitConfig, type PlaneFitOption } from "./planeFit.js";
import { createVoronoiFillPass, voronoiFillConfig, type VoronoiFillOption } from "./voronoiFill.js";

/** Creation-time context handed to `FramePass.create`. */
export interface FramePassContext {
  readonly device: GPUDevice;
  /** Format of the color attachment the chain composites into. */
  readonly colorFormat: GPUTextureFormat;
  readonly colorName: string;
}

/** Per-frame inputs handed to `FramePassInstance.run`. */
export interface FramePassFrame {
  /** The frame's framebuffer: canvas color view(s) + pick attachment;
   *  `colorTextures` carries sampleable textures where available. */
  readonly framebuffer: IFramebuffer;
  /** Pick-shaped G-buffer texture — the previous pass's output, or the
   *  producer's readback pick texture at the head of the chain. */
  readonly pick: GPUTexture;
  readonly proj: Trafo3d;
  readonly colorName: string;
  /** True when a LATER enabled pass will read the pick texture this pass
   *  returns. When false a pass can skip producing it — e.g. the voronoi
   *  fill writes a full-res rgba32float G-buffer that only exists for
   *  downstream shading, which is pure bandwidth if nothing shades. */
  readonly pickConsumed: boolean;
}

export interface FramePassInstance {
  /** Encode the pass. Return a replacement pick-shaped texture for
   *  downstream passes, or undefined/void to pass the current one on. */
  run(frame: FramePassFrame): GPUTexture | undefined | void;
  dispose(): void;
}

/** One entry of `RenderControlProps.framePasses`. */
export interface FramePass {
  /** Off entries skip `run` but keep their instance. Default true. */
  readonly enabled?: boolean | aval<boolean>;
  /** Read through the frame token: transact on it after mutating pass
   *  settings to schedule a repaint (live tuning). */
  readonly version?: aval<unknown>;
  /** Called lazily on the first enabled frame. */
  create(ctx: FramePassContext): FramePassInstance;
}

/** Voronoi fill as a chainable frame pass (see `createVoronoiFillPass`). */
export function voronoiFillFramePass(opt: VoronoiFillOption): FramePass {
  const cfg = voronoiFillConfig(opt);
  const version = (cfg.settings as { version?: aval<unknown> }).version;
  return {
    enabled: cfg.enabled,
    ...(version !== undefined ? { version } : {}),
    create: (ctx) => {
      const p = createVoronoiFillPass(ctx.device, ctx.colorFormat, cfg.settings);
      return {
        run: (f) => p.run(f.framebuffer, f.pick, f.colorName, f.pickConsumed),
        dispose: () => p.dispose(),
      };
    },
  };
}

/** Plane-fit shading (+AO) as a chainable frame pass. */
export function planeFitFramePass(opt: PlaneFitOption): FramePass {
  const cfg = planeFitConfig(opt);
  const version = (cfg.settings as { version?: aval<unknown> }).version;
  return {
    enabled: cfg.enabled,
    ...(version !== undefined ? { version } : {}),
    create: (ctx) => {
      const p = createPlaneFitPass(ctx.device, ctx.colorFormat, cfg.settings);
      return {
        run: (f) => { p.run(f.framebuffer, f.pick, f.proj, f.colorName); },
        dispose: () => p.dispose(),
      };
    },
  };
}

/** GTAO as a chainable frame pass. */
export function gtaoFramePass(opt: GtaoOption): FramePass {
  const cfg = gtaoConfig(opt);
  const version = (cfg.settings as { version?: aval<unknown> }).version;
  return {
    enabled: cfg.enabled,
    ...(version !== undefined ? { version } : {}),
    create: (ctx) => {
      const p = createGtaoPass(ctx.device, ctx.colorFormat, cfg.settings);
      return {
        run: (f) => { p.run(f.framebuffer, f.pick, f.proj, f.colorName); },
        dispose: () => p.dispose(),
      };
    },
  };
}
