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

import { AVal, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";
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
import { compileScene } from "./compile.js";
import type { SgNode } from "./sg.js";
import { TraversalState } from "./traversalState.js";

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
}

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
          onReady,
          ...htmlProps } = props;
  void scene; void children; void view; void proj; void defaultEffect; void clear;
  void device; void runtime; void attach;
  void format; void depthFormat; void sampleCount;
  void onReady;

  // Use the JSX runtime indirectly via the global jsx factory —
  // this file is `.tsx` so the build emits a jsx call.
  return <canvas ref={onCanvasMount} {...htmlProps}/>;
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
  const attachment = attachCanvas(device, canvas, {
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
    .withCamera(view, proj);

  // Lower the scene; compile into the runtime; drive the loop.
  const commands = compileScene(sceneTree, attachment.framebuffer, {
    initialState: initial,
    ...(props.defaultEffect !== undefined ? { defaultEffect: props.defaultEffect } : {}),
    ...(props.clear !== undefined ? { clear: props.clear } : {}),
  });
  const task = runtime.compile(commands);
  scope.onDispose(() => task.dispose());

  const loop = runFrame(attachment, (token) => {
    if (scope.isDisposed) return;
    task.run(token);
  });
  scope.onDispose(() => loop.stop());

  if (ownsRuntime) scope.onDispose(() => runtime.disposeAll());
  if (ownsDevice)  scope.onDispose(() => device.destroy());

  props.onReady?.({
    canvas, device, runtime,
    viewport: attachment.size,
    view, proj,
  });
}

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
  // Walk through scope nodes that pass to a single child.
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
      case "On":
      case "Active":
        cur = cur.child;
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
