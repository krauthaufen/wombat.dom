// `<RenderControl>` — drop a WebGPU canvas into the DOM and wire a
// scene-graph tree to it. Mirrors Aardvark.Dom's `renderControl { }`
// CE: the canvas owns a `Runtime`, an `attachCanvas` attachment, a
// per-frame loop, and the compiled scene.
//
// Prop-shape choices:
//   - `scene` is a prop, not children. Children are reserved for
//     HTML overlay UI (HUDs, overlays placed on top of the canvas);
//     the scene tree is plain SgNode data.
//   - `view` / `proj` are optional props. They're also auto-sniffed
//     from any `<Sg View=… Proj=…>` scopes inside the scene — so
//     you can either set them on the control or set them inside
//     the scene; both are picked up. (Mirrors RenderControlBuilder
//     `view <- atts |> List.tryPick ...` in Aardvark.Dom.)
//   - `device` and `runtime` are optional. If not provided, the
//     component requests its own; on dispose it tears them down.
//     Pass them in to share across multiple controls.
//
// Lifecycle is wired through `useScope()`. Async device acquisition
// is guarded against scope disposal mid-flight.
//
// Note: the component renders a `<canvas>` immediately. GPU setup
// runs in a `ref` callback on canvas mount and is asynchronous —
// the first frame appears one rAF after the device is ready.

import { AVal, avalAddCallback, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V2i } from "@aardworx/wombat.base";
import {
  Runtime, attachCanvas, runFrame,
  type AttachCanvasOptions,
} from "@aardworx/wombat.rendering";
import type {
  ClearValues,
  Effect,
} from "@aardworx/wombat.rendering/core";

import type { Child } from "../vnode.js";
import { useScope } from "../scope.js";
import { Sg, collectSgChildren } from "./constructors.js";
import type { OitMode } from "./transparency.js";
import { createGtaoPass, gtaoConfig, type GtaoOption, type GtaoPass } from "./gtao.js";
import type { SgNode } from "./sg.js";
import { TraversalState } from "./traversalState.js";
import { PickDispatcher, type TapThresholds } from "./picking/dispatcher.js";
import type { PickRegistry } from "./picking/registry.js";
import { arbitratePick, resolveThroughPortals } from "./picking/pickArbitrate.js";
import type { PortalPickHit } from "./picking/pickContext.js";
import { createPickProducer } from "./picking/pickProducer.js";
import { setAmbient, clearAmbient } from "./ambient.js";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface RenderControlProps {
  /**
   * The scene to render. Either:
   *   - **JSX children** (preferred):
   *       ```tsx
   *       <RenderControl>
   *         <Sg Trafo={[...]} Shader={DefaultSurfaces.basic()}>
   *           <Sg.Box />
   *         </Sg>
   *       </RenderControl>
   *       ```
   *     Children pass through `collectSgChildren`, which extracts
   *     SgNodes from JSX components, alists, asets, avals, etc.
   *   - **`scene` prop**: a pre-built `SgNode` for code-generated
   *     trees. Wins over JSX children when both are set.
   */
  readonly scene?: SgNode;
  readonly children?: Child | Child[] | SgNode | ReadonlyArray<Child | SgNode>;

  /**
   * Opt into order-independent transparency for `Sg.transparent` subtrees.
   * `true` uses the default mode (WBOIT); a string picks the mode explicitly.
   * The scene is rendered through `transparencyTask` (opaque → OIT → composite →
   * transparent-pick), so transparent objects blend order-independently and stay
   * pickable. Default off (single forward pass, unchanged behaviour).
   */
  readonly transparency?: boolean | OitMode;

  /**
   * Screen-space ambient occlusion (GTAO-style horizon search), applied as a
   * post-pass over the rendered frame. Costs no extra geometry pass — it reads
   * the view-space normal + depth the pick attachment already carries — and
   * multiplies the result into the canvas colour.
   *
   * `true` / an `aval<boolean>` (so it can be toggled live) turn it on with the
   * defaults; an object supplies `radius` / `intensity` as well. Default off.
   * Ignored under MSAA (`sampleCount > 1`).
   */
  readonly ambientOcclusion?: GtaoOption;

  /**
   * View trafo (world → view). When unset, the scene's outermost
   * `<Sg View=…>` scope is used. Picking-related; M7-M8 stash a
   * snapshot for the picker too.
   */
  readonly view?: aval<Trafo3d>;
  /** Projection trafo (view → clip). Same fallback rules as `view`. */
  readonly proj?: aval<Trafo3d>;

  /**
   * Fallback effect for leaves outside any `<Sg Shader=…>` scope.
   * Without this, leaves with no enclosing shader silently drop.
   */
  readonly defaultEffect?: Effect;

  /** Optional clear values issued before each frame's render. */
  readonly clear?: ClearValues;

  /**
   * Existing GPU device. When omitted, the control requests one
   * via `navigator.gpu.requestAdapter() → requestDevice()` on
   * mount and destroys it on dispose.
   */
  readonly device?: GPUDevice;
  /**
   * Existing Runtime. When omitted, one is constructed from the
   * device and disposed on unmount.
   */
  readonly runtime?: Runtime;

  /**
   * §7 derived uniforms — when true, ModelView / ViewProj /
   * ModelViewProj / *Inv / NormalMatrix are produced by a df32
   * compute pre-pass instead of CPU-packed per-RO every frame.
   * Sg-driven scenes are the primary beneficiary: `autoInjectedUniforms`
   * supplies 16 trafo-derived uniforms per leaf; in derived mode only
   * Model / View / Proj are uploaded, the GPU computes the rest.
   *
   * Default: `true` when RenderControl constructs its own Runtime.
   * Ignored when `runtime` is supplied (caller controls the option).
   * Set to `false` here to opt out (e.g. for backends without df32
   * compute, or to A/B-compare).
   */
  readonly enableDerivedUniforms?: boolean;

  /**
   * §3 — per-arena-chunk byte cap for the heap path. Pass a small
   * value (e.g. 4 MB) to exercise multi-chunk routing without
   * needing massive workloads. Threaded through to Runtime →
   * RuntimeContext → HybridScene → heapScene's arena setup.
   * Default: adapter's `maxStorageBufferBindingSize` (capped at
   * `DEFAULT_MAX_BUFFER_BYTES = 256 MB`).
   */
  readonly maxChunkBytes?: number;

  /** Forwarded to `attachCanvas` — see wombat.rendering. */
  readonly attach?: Omit<AttachCanvasOptions, "format" | "depthFormat" | "sampleCount">;
  readonly format?: GPUTextureFormat;
  readonly depthFormat?: GPUTextureFormat;
  readonly sampleCount?: number;

  /**
   * Called once after the runtime + scene are live. Useful for
   * grabbing handles to the runtime / loop / canvas attachment.
   */
  readonly onReady?: (info: RenderControlReadyInfo) => void;

  /**
   * Called each frame immediately before the render task runs. Runs
   * inside the frame eval — do NOT transact() here (marks would race
   * the render loop's dirty tracking).
   */
  readonly onBeforeRender?: (info: RenderControlFrameInfo) => void;
  /**
   * Called after each rendered frame, on a microtask off the frame
   * eval stack — transact() here is safe (the canonical use is a
   * camera controller stepping its physics per frame, which marks the
   * camera and thereby schedules the next frame).
   */
  readonly onRendered?: (info: RenderControlFrameInfo) => void;
  /** Called whenever the framebuffer size changes. */
  readonly onResize?: (info: RenderControlFrameInfo) => void;

  /**
   * Per-instance overrides for the tap / long-press / double-tap /
   * drag / hover thresholds. Any unset field falls back to the module
   * default.
   */
  readonly tapThresholds?: TapThresholds;

  // HTML pass-through props (canvas attributes). Kept loose so
  // callers can pass anything the canvas DOM element accepts; the
  // wombat.dom attribute binder routes by attribute kind.
  readonly style?: string | Record<string, string | number | boolean | null | undefined>;
  readonly class?: string | aval<string>;
  readonly id?: string;
  readonly tabIndex?: number;
  readonly width?: number | string;
  readonly height?: number | string;
}

/** Per-frame statistics handed to onBeforeRender/onRendered/onResize —
 *  mirrors Aardvark.Dom's RenderControlEventInfo (sans IFramebufferSignature,
 *  which has no wombat counterpart). Times are milliseconds. */
export interface RenderControlFrameInfo {
  readonly size: { readonly width: number; readonly height: number };
  readonly frameIndex: number;
  /** ms since the control's loop started. */
  readonly time: number;
  /** ms duration since the previous frame began (0 on the first frame). */
  readonly frameTime: number;
}

export interface RenderControlReadyInfo {
  readonly canvas: HTMLCanvasElement;
  readonly device: GPUDevice;
  readonly runtime: Runtime;
  readonly viewport: aval<{ width: number; height: number }>;
  /** Canvas size in CSS pixels (clientWidth/Height; ResizeObserver-fed). */
  readonly clientSize: aval<{ width: number; height: number }>;
  /** The view aval used for picking (from prop or sniffed from scene). */
  readonly view: aval<Trafo3d>;
  /** The proj aval used for picking. */
  readonly proj: aval<Trafo3d>;
  /** Per-frame `performance.now()` clock; ticks once per rAF. Phase 7. */
  readonly time: aval<number>;
  /**
   * The internal `PickRegistry`. Picking is always-on; callers can use
   * this to drive `setFocus`, query scope membership, etc. Mirrors
   * Aardvark.Dom's `IEventHandler` surface.
   */
  readonly picking: PickRegistry;
  /**
   * Programmatic pick at device-pixel coords — the same resolution
   * pointer events get (argmin + portal recursion), so hits inside
   * offscreen `PickContext` scenes come back in THEIR frame. Useful
   * for pick-anchored camera controllers
   * (`OrbitController.attach(..., { picker })`).
   */
  readonly pickAt: (x: number, y: number) => Promise<PortalPickHit | undefined>;
}

// ---------------------------------------------------------------------------
// Phase 7 — global time clock. Lazy-init on first read; ticked from
// every active RenderControl's frame loop. Exposed both as `RenderControl.time`
// and via `info.time` from `onReady`.
// ---------------------------------------------------------------------------

// The global per-frame time clock lives in `./ambient.ts` so the
// ambient `RenderControl.time` getter stays valid even with no
// control mounted (tests can poke `globalTime()` directly).
import { globalTime as getGlobalTime, tickGlobalTime } from "./ambient.js";
let _globalSubs = 0;
let _globalRaf: number | undefined;

// (rAF loop is unused for now — we tick from each control's
// `runFrame` callback. Kept-references suppressed below.)
void _globalSubs; void _globalRaf;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function RenderControl(props: RenderControlProps): import("../jsx-runtime.js").JsxResult {
  const scope = useScope();

  // Resolve the scene tree at component-call time. The `scene`
  // prop (if set) wins over JSX children — useful for code-
  // generated scenes; otherwise children flow through
  // collectSgChildren.
  const sceneTree: SgNode = props.scene ?? collectSgChildren(props.children);

  // Ref callback — fires once the canvas is in the DOM. Kicks off
  // async device acquisition and returns immediately. Errors
  // (no GPU, adapter request rejected, ...) get logged rather than
  // thrown out of the unhandled-rejection handler — the canvas
  // stays mounted but never receives frames. Tests under happy-dom
  // hit this path on purpose.
  const onCanvasMount = (canvas: Element): void => {
    // Each DEVICE GENERATION lives in its own child scope. A lost device
    // (mobile Safari drops it under memory pressure, Chrome recycles the GPU
    // process) makes `Runtime` dispose every render task — but the render loop
    // knew nothing about that and kept running them, throwing
    // "RenderTask: run after dispose" forever: a permanently frozen viewer on a
    // page that otherwise still responds. Recovery is to tear the generation
    // down and build a fresh one (device, runtime, attachment, producer, loop);
    // the scene and every user cval are device-agnostic and survive untouched.
    let generation = 0;
    const start = (): void => {
      if (scope.isDisposed) return;
      const gen = ++generation;
      const genScope = scope.child();
      // Generations after the first must build their own device/runtime — any
      // that were handed in via props died with the original.
      initialise(canvas as HTMLCanvasElement, sceneTree, props, genScope, () => {
        // device lost — rebuild, unless this generation is already superseded
        // or the control went away.
        if (scope.isDisposed || gen !== generation) return;
        console.warn("[RenderControl] device lost — rebuilding");
        genScope.dispose();
        start();
      }, gen > 1).catch((err) => {
        if (!scope.isDisposed) {
          console.error("[RenderControl] init failed:", err);
        }
      });
    };
    start();
  };

  // Build the canvas VNode. The `ref` is consumed by mount.ts's
  // bindElementProps; everything else passes through to the DOM
  // attribute binder.
  const { scene, children, view, proj, defaultEffect, clear,
          device, runtime, attach, format, depthFormat, sampleCount,
          onReady, onBeforeRender, onRendered, onResize,
          tapThresholds, style: userStyle,
          ...htmlProps } = props;
  void scene; void children; void view; void proj; void defaultEffect; void clear;
  void device; void runtime; void attach;
  void format; void depthFormat; void sampleCount;
  void onReady; void tapThresholds;
  void onBeforeRender; void onRendered; void onResize;

  // Phase 5 — when no tabindex is supplied, set tabindex=0 so the
  // canvas can receive keyboard focus (otherwise key events do not
  // route to it). User-supplied tabindex wins.
  if (htmlProps.tabIndex === undefined) (htmlProps as { tabIndex?: number }).tabIndex = 0;

  // Default styles defend against the browser eating pointer
  // events that should reach the controller:
  //   - `touchAction: "none"` disables iOS Safari's pan/pinch/
  //     double-tap-zoom on the canvas; without this `pointermove`
  //     gets cancelled by the browser's gesture detector.
  //   - `userSelect: "none"` + `-webkit-touch-callout: none`
  //     prevents long-press selection / context menus.
  //   - `overscrollBehavior: "contain"` stops drag from chaining
  //     into pull-to-refresh on the parent scroller.
  // User-supplied `style` values override.
  const defaultStyle: Record<string, string | number | boolean | null | undefined> = {
    touchAction: "none",
    userSelect: "none",
    "-webkit-user-select": "none",
    "-webkit-touch-callout": "none",
    overscrollBehavior: "contain",
  };
  const mergedStyle =
    typeof userStyle === "string"
      ? userStyle
      : { ...defaultStyle, ...(userStyle ?? {}) };

  // Use the JSX runtime indirectly via the global jsx factory —
  // this file is `.tsx` so the build emits a jsx call.
  return <canvas ref={onCanvasMount} style={mergedStyle} {...htmlProps}/>;
}

// ---------------------------------------------------------------------------
// Initialisation — runs on canvas mount.
// ---------------------------------------------------------------------------

async function initialise(
  canvas: HTMLCanvasElement,
  sceneTree: SgNode,
  props: RenderControlProps,
  scope: import("../scope.js").Scope,
  onDeviceLost?: () => void,
  /** Rebuild after a device loss: the caller-supplied device (if any) is dead,
   *  so this generation must request its own. */
  ignoreProvidedDevice = false,
): Promise<void> {
  if (scope.isDisposed) return;

  // Acquire a device — either supplied or freshly requested.
  let device: GPUDevice;
  let ownsDevice = false;
  if (props.device !== undefined && !ignoreProvidedDevice) {
    device = props.device;
  } else {
    if (!("gpu" in navigator)) {
      throw new Error("RenderControl: navigator.gpu unavailable — WebGPU not enabled in this browser");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter === null) throw new Error("RenderControl: no GPU adapter available");
    if (scope.isDisposed) return;
    // Raise the storage-buffer limits toward what the adapter supports.
    // The heap binds 4 arena views (+3 megacall, vertex-only) per stage,
    // so the WebGPU default of 8 leaves only 4 for user storage — an OIT
    // A-buffer alone wants 5. Ask for the adapter's max (typically 16);
    // adapters that only offer the default are unchanged.
    const wantStorage = adapter.limits.maxStorageBuffersPerShaderStage;
    const requiredLimits: Record<string, number> = {};
    if (typeof wantStorage === "number" && wantStorage > 8) {
      requiredLimits.maxStorageBuffersPerShaderStage = wantStorage;
    }
    device = await adapter.requestDevice(
      Object.keys(requiredLimits).length > 0 ? { requiredLimits } : {},
    );
    if (scope.isDisposed) { device.destroy(); return; }
    // Pin the adapter to the device: Chrome/Dawn drops the wgpu
    // instance when the LAST JS reference to the GPUAdapter is
    // GC'd, losing the device with "A valid external Instance
    // reference no longer exists". Small demos never notice; the
    // first big allocation burst triggers a GC and kills the
    // device one frame in.
    (device as unknown as { __adapter?: GPUAdapter }).__adapter = adapter;
    ownsDevice = true;
  }

  // Runtime — supplied or constructed. A supplied runtime is bound to the
  // supplied (now dead) device, so a rebuild always constructs its own.
  const derivedUniforms = props.enableDerivedUniforms ?? true;
  const useProvidedRuntime = props.runtime !== undefined && !ignoreProvidedDevice;
  const runtime = useProvidedRuntime ? props.runtime! : new Runtime({
    device,
    ...(derivedUniforms ? { enableDerivedUniforms: true } : {}),
    ...(props.maxChunkBytes !== undefined ? { maxChunkBytes: props.maxChunkBytes } : {}),
  });
  const ownsRuntime = !useProvidedRuntime;

  // Canvas attachment — picks the right format + sample count;
  // exposes `framebuffer: aval<IFramebuffer>` and `size:
  // aval<{w,h}>`.
  //
  // Defaults:
  //   - `colorAttachmentName: "Colors"` aligns the framebuffer
  //     signature with the de-facto wombat.shader convention (the
  //     fragment output is named `Colors` in
  //     `DefaultSurfaces.basic` and in the renderto-real coverage
  //     in wombat.rendering).
  //   - `depthFormat: "depth24plus"` so the pipeline's `depth: {
  //     write, compare: "less" }` state actually has a buffer to
  //     test against. Without it, draws happen in vertex order —
  //     visibly wrong for opaque 3D geometry.
  //
  // `attach`/`format`/`depthFormat`/`sampleCount` props override.
  const attachment = attachCanvas(device, canvas, {
    colorAttachmentName: "Colors",
    depthFormat: "depth24plus",
    ...(props.attach ?? {}),
    ...(props.format !== undefined ? { format: props.format } : {}),
    ...(props.depthFormat !== undefined ? { depthFormat: props.depthFormat } : {}),
    ...(props.sampleCount !== undefined ? { sampleCount: props.sampleCount } : {}),
  });
  scope.onDispose(() => attachment.dispose());

  if (scope.isDisposed) return;

  // Resolve view / proj — props take precedence; otherwise sniff
  // the scene's outermost View / Proj scopes; otherwise default
  // to identity (with a console.warn — picking won't work).
  const sniffed = sniffViewProj(sceneTree);
  const view = props.view ?? sniffed.view ?? warnDefault("view");
  const proj = props.proj ?? sniffed.proj ?? warnDefault("proj");

  // Picking is always-on. The PickProducer bundles the combined FB
  // (canvas color + pickId + canvas depth), the id registry, the GPU
  // metadata mirror, the argmin resolve, and the compiled render task
  // — see `picking/pickProducer.ts`. The registry is created inside
  // and exposed back via `onReady`.
  const producer = createPickProducer(runtime, device, attachment, sceneTree, {
    view, proj,
    time: getGlobalTime(),
    ...(props.transparency !== undefined ? { transparency: props.transparency } : {}),
    ...(props.defaultEffect !== undefined ? { defaultEffect: props.defaultEffect } : {}),
    ...(props.clear !== undefined ? { clear: props.clear } : {}),
    colorAttachmentName: "Colors",
  });
  const registry = producer.registry;
  scope.onDispose(() => producer.dispose());

  // Ambient occlusion (opt-in). The pipelines are built on first ENABLED frame
  // — a control that never turns AO on pays nothing. MSAA would need the pass
  // to render into the multisampled colour target; not worth it, so skip.
  let ao: { readonly enabled: aval<boolean>; readonly pass: () => GtaoPass } | undefined;
  if (props.ambientOcclusion !== undefined && props.ambientOcclusion !== false) {
    const cfg = gtaoConfig(props.ambientOcclusion);
    if (attachment.signature.sampleCount > 1) {
      console.warn("[RenderControl] ambientOcclusion is ignored under MSAA (sampleCount > 1)");
    } else {
      let pass: GtaoPass | undefined;
      ao = {
        enabled: cfg.enabled,
        pass: () => {
          if (pass === undefined) {
            pass = createGtaoPass(device, attachment.signature.colors.get("Colors"), cfg.settings);
          }
          return pass;
        },
      };
      scope.onDispose(() => pass?.dispose());
    }
  }

  const dispatcher = new PickDispatcher(
    registry,
    () => AVal.force(view),
    () => AVal.force(proj),
    () => canvas.getBoundingClientRect(),
    props.tapThresholds,
  );
  const detach = dispatcher.attach(canvas, producer.pickPixel);
  scope.onDispose(detach);

  // Programmatic pick — same path as pointer events (arbitrate +
  // portal recursion), exposed via onReady for camera controllers etc.
  const pickAt = async (x: number, y: number): Promise<PortalPickHit | undefined> => {
    if (scope.isDisposed) return undefined;
    const result = await producer.pickPixel(x, y, false);
    if (scope.isDisposed) return undefined;
    // AVal.force OK: pick-time snapshot at an API boundary.
    const v = AVal.force(view);
    const p = AVal.force(proj);
    const sz = AVal.force(attachment.size);
    const hit = await resolveThroughPortals(
      arbitratePick(result, { devX: x, devY: y }, registry, v, p, new V2i(sz.width, sz.height)),
    );
    if (hit === undefined) return undefined;
    return { hit, registry: hit.registry ?? registry };
  };
  // Frame statistics for the onBeforeRender/onRendered/onResize hooks.
  let frameIndex = 0;
  let loopStart = -1;
  let lastFrameStart = -1;
  const frameInfo = (now: number): RenderControlFrameInfo => ({
    size: AVal.force(attachment.size),
    frameIndex,
    time: loopStart < 0 ? 0 : now - loopStart,
    frameTime: lastFrameStart < 0 ? 0 : now - lastFrameStart,
  });
  const loop = runFrame(attachment, (token) => {
    if (scope.isDisposed) return;
    const now = performance.now();
    if (loopStart < 0) loopStart = now;
    const info = (props.onBeforeRender !== undefined || props.onRendered !== undefined)
      ? frameInfo(now) : undefined;
    if (info !== undefined) props.onBeforeRender?.(info);
    producer.run(token);
    // AO post-pass. Reads the pick attachment the frame just wrote (normal +
    // depth) and multiplies the occlusion into the canvas colour. `enabled` is
    // pulled through the token, so flipping the toggle marks the frame loop.
    if (ao !== undefined && ao.enabled.getValue(token)) {
      ao.pass().run(
        producer.framebuffer.getValue(token),
        producer.readbackPickTexture.getValue(token),
        producer.proj.getValue(token),
        "Colors",
      );
    }
    // onRendered handlers may transact() (the Aardvark.Dom pattern of a
    // camera controller stepping physics per frame). Marks fired inside
    // this frame eval would race the wrapper aval's outOfDate reset and
    // stall the loop permanently, so defer the handler to a microtask —
    // it runs right after the eval unwinds, before the next rAF.
    if (info !== undefined && props.onRendered !== undefined) {
      queueMicrotask(() => {
        if (!scope.isDisposed) props.onRendered?.(info);
      });
    }
    lastFrameStart = now;
    frameIndex++;
  }, {
    // Tick the global time clock AFTER each frame's eval. This is
    // the dirty-tracking-loop equivalent of Aardvark.Rendering's
    // pattern: marks emitted here propagate post-eval, so any aval
    // that depends on time (animated controllers, time-tied
    // uniforms) marks `runFrame`'s wrapper aval and schedules the
    // next rAF. Scenes with no time-dependent reads (static cameras,
    // no animation) ignore the tick → no rAF scheduled → loop
    // sleeps until something else marks (cval edit, canvas resize,
    // input handler, etc.).
    onAfterFrame: tickGlobalTime,
    // GPU-rate pacing. `queue.submit` is fire-and-forget — without
    // this JS will encode 60 frames/s into a GPU that's actually
    // running at e.g. 10 fps, the compositor drops frames silently,
    // and perceived FPS has nothing to do with rAF rate. Awaiting
    // `onSubmittedWorkDone()` before scheduling the next rAF caps
    // encode rate to GPU completion rate so the queue stays at
    // depth 1. Critical during boot-phase pipeline-compile stalls
    // (without it, 50k addDraws queue up while the first frame is
    // still compiling).
    pacer: () => device.queue.onSubmittedWorkDone(),
  });
  scope.onDispose(() => loop.stop());

  if (ownsRuntime) scope.onDispose(() => runtime.disposeAll());
  if (ownsDevice)  scope.onDispose(() => device.destroy());

  // DEVICE LOSS. `Runtime` reacts by disposing every task it compiled, so the
  // very next frame would encode against dead tasks ("RenderTask: run after
  // dispose"). Stop the loop FIRST — before anything can run them — then hand
  // control back so the caller can rebuild this generation from scratch. A
  // control given a device/runtime from outside doesn't own the recovery: it
  // just stops, and its owner decides what to do.
  runtime.deviceLost.then((info) => {
    if (scope.isDisposed) return;
    loop.stop();
    console.warn(`[RenderControl] GPU device lost (${info?.reason ?? "unknown"}): ${info?.message ?? ""}`);
    onDeviceLost?.();
  }, () => { /* never rejects */ });

  // Publish this control's avals as the ambient context so callers
  // can read `viewport` / `view` / `proj` / `time` from
  // `@aardworx/wombat.dom/scene/ambient` without threading state.
  setAmbient({ viewport: attachment.size, view, proj, time: getGlobalTime(), registry });
  scope.onDispose(() => clearAmbient());

  // onResize: fire on framebuffer-size transitions (skipping the initial
  // value — Aardvark's Resize event fires on CHANGES).
  if (props.onResize !== undefined) {
    let first = true;
    const sub = avalAddCallback(attachment.size, () => {
      if (first) { first = false; return; }
      if (scope.isDisposed) return;
      props.onResize?.(frameInfo(performance.now()));
    });
    scope.onDispose(() => sub.dispose());
  }

  // clientSize: CSS-pixel canvas size as an adaptive value.
  const clientSize = cval<{ width: number; height: number }>({
    width: canvas.clientWidth, height: canvas.clientHeight,
  });
  const ro = new ResizeObserver(() => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const cur = AVal.force(clientSize as aval<{ width: number; height: number }>);
    // transact: ResizeObserver fires outside any adaptive scope; a bare cval
    // write throws "cannot mark object without transaction". Phones hit this
    // constantly (URL-bar collapse / overlay layout resize the canvas).
    if (cur.width !== w || cur.height !== h) {
      transact(() => { clientSize.value = { width: w, height: h }; });
    }
  });
  ro.observe(canvas);
  scope.onDispose(() => ro.disconnect());

  // Debug handle for the CURRENT device generation. The device-loss/rebuild
  // path is otherwise only reachable by starving a real GPU; a harness can
  // `__wombatDevice.destroy()` to exercise it.
  (globalThis as Record<string, unknown>).__wombatDevice = device;
  // Debug handle for the compiled render task: `__wombatRenderTask
  // .validateHeap()` is the GPU-verified liveness probe for streaming
  // scenes (dead-draw forensics on devices without a console).
  (globalThis as Record<string, unknown>).__wombatRenderTask = producer.task;

  props.onReady?.({
    canvas, device, runtime,
    viewport: attachment.size,
    clientSize: clientSize as aval<{ width: number; height: number }>,
    view, proj,
    time: getGlobalTime(),
    picking: registry,
    pickAt,
  });
}

/**
 * Static accessors that flatten the *active* RenderControl's avals
 * — `viewport` / `view` / `proj` / `time`. Each tracks the most-
 * recently-mounted control via the ambient context (`./ambient.ts`);
 * unmount restores the fallback (1×1 viewport, identity view+proj,
 * frozen time). Multiple controls mounted simultaneously: the LAST
 * mount wins. For per-control isolation, read from the JSX function
 * child's `state` (which carries the actual TraversalState).
 */
import {
  viewport as ambViewport,
  view as ambView,
  proj as ambProj,
  time as ambTime,
  query as ambQuery,
} from "./ambient.js";
import type { SceneQuery } from "./picking/sceneQuery.js";

// eslint-disable-next-line @typescript-eslint/no-namespace
export namespace RenderControl {
  export const viewport: aval<{ width: number; height: number }> = ambViewport;
  export const view: aval<Trafo3d> = ambView;
  export const proj: aval<Trafo3d> = ambProj;
  export const time: aval<number> = ambTime;
  /**
   * Reactive scene query bound to the active control's
   * view/proj/viewport + pick registry. Mirrors the single-active-
   * control assumption documented for the other ambient avals: the
   * LAST mounted RenderControl wins.
   */
  export const query: SceneQuery = ambQuery;
}

/** Manually tick the global clock — useful for tests with fake timers. */
export function _tickRenderControlTime(): void { tickGlobalTime(); }

// ---------------------------------------------------------------------------
// Sniff View / Proj from the scene's outermost scopes.
//
// Walks down attribute scopes (Trafo / Shader / Uniform / etc.)
// looking for the outermost View and Proj wrappers. Stops at the
// first non-attribute node (Group, Leaf, …). Mirrors
// RenderControlBuilderState.Append's `tryPick` behaviour.
// ---------------------------------------------------------------------------

export interface SniffResult {
  view: aval<Trafo3d> | undefined;
  proj: aval<Trafo3d> | undefined;
}

export function sniffViewProj(node: SgNode): SniffResult {
  const out: SniffResult = { view: undefined, proj: undefined };
  let cur: SgNode = node;
  // Walk through scope nodes that pass to a single child. `Delay`
  // is force-expanded with `TraversalState.empty` so View/Proj
  // hidden inside an outermost `Sg.delay` (the viewport-aware
  // perspective pattern) are still found. The expansion is
  // discarded after sniffing — the actual render path runs delay
  // with the real state.
  while (true) {
    switch (cur.kind) {
      case "View":
        if (out.view === undefined) out.view = cur.view;
        cur = cur.child;
        continue;
      case "Proj":
        if (out.proj === undefined) out.proj = cur.proj;
        cur = cur.child;
        continue;
      case "Trafo":
      case "Shader":
      case "Uniform":
      case "BlendMode":
      case "Cursor":
      case "PickThrough":
      case "PixelSnapRadius":
      case "On":
      case "Active":
      case "NoEvents":
      case "PickContext":
      case "ForcePixelPicking":
      case "CanFocus":
      case "DepthTest":
      case "DepthMask":
      case "DepthBias":
      case "DepthClamp":
      case "CullMode":
      case "FrontFace":
      case "FillMode":
      case "BlendConstant":
      case "ColorMask":
      case "StencilMode":
      case "Pass":
      case "VertexAttributes":
      case "InstanceAttributes":
      case "Index":
      case "Mode":
      case "Intersectable":
      case "Instanced":
        cur = cur.child;
        continue;
      case "Delay":
        try { cur = cur.create(TraversalState.empty); }
        catch { return out; }
        continue;
      default:
        return out;
    }
  }
}

function warnDefault(name: "view" | "proj"): aval<Trafo3d> {
  console.warn(
    `RenderControl: no ${name === "view" ? "View" : "Proj"} provided ` +
    `(neither as a prop nor as an outermost <Sg ${name === "view" ? "View" : "Proj"}=…> scope). ` +
    `Picking will not work; defaulting to identity.`,
  );
  return AVal.constant(Trafo3d.identity);
}
