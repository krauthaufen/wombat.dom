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

import { AVal, HashMap, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V4f } from "@aardworx/wombat.base";
import {
  Runtime, attachCanvas, runFrame,
  type AttachCanvasOptions,
} from "@aardworx/wombat.rendering";
import type {
  ClearValues,
  ClearColor,
  Effect,
} from "@aardworx/wombat.rendering/core";

import type { Child } from "../vnode.js";
import { useScope } from "../scope.js";
import { Sg, collectSgChildren } from "./constructors.js";
import { compileScene } from "./compile.js";
import type { SgNode } from "./sg.js";
import { TraversalState } from "./traversalState.js";
import { PickRegistry } from "./picking/registry.js";
import { createPickFramebuffer, PICK_NAME } from "./picking/pickFramebuffer.js";
import { PickDispatcher, type TapThresholds } from "./picking/dispatcher.js";
import { readPickRegion, type PickRegion } from "./picking/readback.js";
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

export interface RenderControlReadyInfo {
  readonly canvas: HTMLCanvasElement;
  readonly device: GPUDevice;
  readonly runtime: Runtime;
  readonly viewport: aval<{ width: number; height: number }>;
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

export function RenderControl(props: RenderControlProps): import("../vnode.js").VNode {
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
    initialise(canvas as HTMLCanvasElement, sceneTree, props, scope).catch((err) => {
      if (!scope.isDisposed) {
        console.error("[RenderControl] init failed:", err);
      }
    });
  };

  // Build the canvas VNode. The `ref` is consumed by mount.ts's
  // bindElementProps; everything else passes through to the DOM
  // attribute binder.
  const { scene, children, view, proj, defaultEffect, clear,
          device, runtime, attach, format, depthFormat, sampleCount,
          onReady, tapThresholds, style: userStyle,
          ...htmlProps } = props;
  void scene; void children; void view; void proj; void defaultEffect; void clear;
  void device; void runtime; void attach;
  void format; void depthFormat; void sampleCount;
  void onReady; void tapThresholds;

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
): Promise<void> {
  if (scope.isDisposed) return;

  // Acquire a device — either supplied or freshly requested.
  let device: GPUDevice;
  let ownsDevice = false;
  if (props.device !== undefined) {
    device = props.device;
  } else {
    if (!("gpu" in navigator)) {
      throw new Error("RenderControl: navigator.gpu unavailable — WebGPU not enabled in this browser");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter === null) throw new Error("RenderControl: no GPU adapter available");
    if (scope.isDisposed) return;
    device = await adapter.requestDevice();
    if (scope.isDisposed) { device.destroy(); return; }
    ownsDevice = true;
  }

  // Runtime — supplied or constructed.
  const runtime = props.runtime ?? new Runtime({ device });
  const ownsRuntime = props.runtime === undefined;

  // Canvas attachment — picks the right format + sample count;
  // exposes `framebuffer: aval<IFramebuffer>` and `size:
  // aval<{w,h}>`.
  //
  // Defaults:
  //   - `colorAttachmentName: "outColor"` aligns the framebuffer
  //     signature with the de-facto wombat.shader convention (the
  //     fragment output is named `outColor` in
  //     `DefaultSurfaces.basic` and in the renderto-real coverage
  //     in wombat.rendering).
  //   - `depthFormat: "depth24plus"` so the pipeline's `depth: {
  //     write, compare: "less" }` state actually has a buffer to
  //     test against. Without it, draws happen in vertex order —
  //     visibly wrong for opaque 3D geometry.
  //
  // `attach`/`format`/`depthFormat`/`sampleCount` props override.
  const attachment = attachCanvas(device, canvas, {
    colorAttachmentName: "outColor",
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

  // Initial traversal state — viewport + camera populated up front.
  const initial = TraversalState.empty
    .withViewport(attachment.size)
    .withCamera(view, proj)
    .withTime(getGlobalTime());

  // Picking is always-on. Allocate a combined FB (canvas color +
  // pickId + canvas depth), feed that to compileScene, and wire a
  // PickDispatcher to the canvas. Handlers fire on cursor events
  // after a 1-pixel readback decodes a registered pickId. The
  // registry is created internally per RenderControl instance and
  // exposed back via `onReady`.
  const registry = new PickRegistry();
  const pickFb = createPickFramebuffer(device, attachment, { colorAttachmentName: "outColor" });
  const outputFb = pickFb.pickFramebuffer;
  scope.onDispose(() => pickFb.dispose());

  // Always inject a per-frame clear for the pickId attachment.
  // Without this, fragments NOT covered by any drawn geometry
  // retain last-frame pickIds (loadOp defaults to "load") and the
  // hit-test readback decodes stale data — wrong NDC depth and a
  // bogus registered ID. Picking can't rely on the user's clear
  // config to know about its own attachments. (See
  // ~/claude/wombat-todo.md: phase 4 / item 11.)
  const userClear = props.clear;
  const userColors = userClear?.colors ?? HashMap.empty<string, ClearColor>();
  const clearWithPick: ClearValues = {
    ...(userClear ?? {}),
    colors: userColors.add(PICK_NAME, new V4f(0, 0, 0, 0)),
  };
  // Lower the scene; compile into the runtime; drive the loop.
  const commands = compileScene(sceneTree, outputFb, {
    initialState: initial,
    ...(props.defaultEffect !== undefined ? { defaultEffect: props.defaultEffect } : {}),
    clear: clearWithPick,
    picking: { registry },
  });
  const task = runtime.compile(commands);
  scope.onDispose(() => task.dispose());

  const dispatcher = new PickDispatcher(
    registry,
    () => AVal.force(view),
    () => AVal.force(proj),
    () => canvas.getBoundingClientRect(),
    props.tapThresholds,
  );
  const readRegion = async (x: number, y: number): Promise<PickRegion | undefined> => {
    const tex = AVal.force(pickFb.readbackPickTexture);
    return readPickRegion(device, tex, x, y);
  };
  const detach = dispatcher.attach(canvas, readRegion);
  scope.onDispose(detach);

  // For MSAA picking we need a custom resolve compute pass after the
  // render pass. wombat.rendering's `runFrame` does not expose its
  // per-frame command encoder, so we submit a SEPARATE encoder right
  // after `task.run`. This adds one queue submission per frame; an
  // upstream hook for piggy-backing on the render encoder would be
  // cleaner (see docs/FUTURE.md).
  const runResolve = pickFb.maybeRunResolve;
  const loop = runFrame(attachment, (token) => {
    if (scope.isDisposed) return;
    task.run(token);
    if (runResolve !== undefined) {
      const enc = device.createCommandEncoder({ label: "pick.resolve.frame" });
      runResolve(enc);
      device.queue.submit([enc.finish()]);
    }
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
  });
  scope.onDispose(() => loop.stop());

  if (ownsRuntime) scope.onDispose(() => runtime.disposeAll());
  if (ownsDevice)  scope.onDispose(() => device.destroy());

  // Publish this control's avals as the ambient context so callers
  // can read `viewport` / `view` / `proj` / `time` from
  // `@aardworx/wombat.dom/scene/ambient` without threading state.
  setAmbient({ viewport: attachment.size, view, proj, time: getGlobalTime(), registry });
  scope.onDispose(() => clearAmbient());

  props.onReady?.({
    canvas, device, runtime,
    viewport: attachment.size,
    view, proj,
    time: getGlobalTime(),
    picking: registry,
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

interface SniffResult {
  view: aval<Trafo3d> | undefined;
  proj: aval<Trafo3d> | undefined;
}

function sniffViewProj(node: SgNode): SniffResult {
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
