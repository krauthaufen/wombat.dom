// Ambient avals — drop-in references to the currently-mounted
// `<RenderControl>`'s state. Lets callers read `viewport`, `view`,
// `proj`, `time` etc. anywhere in a scene graph WITHOUT routing them
// through the surrounding `Sg.delay` / function-child plumbing.
//
// Single-active-control assumption: when multiple `<RenderControl>`s
// are mounted simultaneously, the LAST one to mount wins for these
// ambient bindings. Apps that need per-control isolation should use
// the function-child form `<Sg>{state => …}</Sg>` instead, which
// reads the actually-accumulated `TraversalState`.

import { AVal, ChangeableValue, cval, transact, type aval } from "@aardworx/wombat.adaptive";
type Cval<T> = ReturnType<typeof cval<T>>;
import { Trafo3d, V2i } from "@aardworx/wombat.base";

import { PickRegistry } from "./picking/registry.js";
import { createSceneQuery, type SceneQuery } from "./picking/sceneQuery.js";

/**
 * Snapshot of a RenderControl's exposed avals. Updated on mount /
 * unmount; ambient avals below project individual fields out.
 */
export interface AmbientContext {
  readonly viewport: aval<{ width: number; height: number }>;
  readonly view: aval<Trafo3d>;
  readonly proj: aval<Trafo3d>;
  readonly time: aval<number>;
  /**
   * Active control's pick registry — exposed so `RenderControl.query`
   * can lazy-build a `SceneQuery` bound to the ambient view/proj/
   * viewport without threading state through every reader. Optional
   * for the fallback context (no control mounted yet).
   */
  readonly registry?: PickRegistry;
}

// The global per-frame time clock. Ticked by every active
// RenderControl's frame loop; lives here (not in renderControl.tsx)
// so the ambient `time` aval stays valid even when no control is
// mounted yet — tests can `globalTime.value = …` directly.
let _globalTime: Cval<number> | undefined;
export function globalTime(): Cval<number> {
  if (_globalTime === undefined) {
    _globalTime = cval(typeof performance !== "undefined" ? performance.now() : 0);
  }
  return _globalTime;
}
/** Tick the global clock. Called from `runFrame` per frame. */
export function tickGlobalTime(): void {
  if (_globalTime === undefined) return;
  transact(() => { _globalTime!.value = typeof performance !== "undefined" ? performance.now() : 0; });
}

const fallback: AmbientContext = {
  viewport: AVal.constant({ width: 1, height: 1 }),
  view: AVal.constant(Trafo3d.identity),
  proj: AVal.constant(Trafo3d.identity),
  time: globalTime(),
};

const current = new ChangeableValue<AmbientContext>(fallback);

/** Set the ambient context. Called by `<RenderControl>` on mount.
 *  Transacted — consumers may have subscribed to the ambient avals
 *  BEFORE the control mounts (e.g. an offscreen `renderToPickable`
 *  sized by `RenderControl.viewport`), and marking with dependents
 *  requires a transaction. */
export function setAmbient(ctx: AmbientContext): void {
  transact(() => { current.value = ctx; });
}

/** Restore fallback. Called on `<RenderControl>` unmount. */
export function clearAmbient(): void {
  transact(() => { current.value = fallback; });
}

/** Ambient viewport size — tracks the active control. */
export const viewport: aval<{ width: number; height: number }> = current.bind(c => c.viewport);
/** Ambient view trafo (world → view) of the active control. */
export const view: aval<Trafo3d> = current.bind(c => c.view);
/** Ambient projection trafo (view → clip) of the active control. */
export const proj: aval<Trafo3d> = current.bind(c => c.proj);
/** Ambient frame clock (`performance.now()`-style ms) of the active control. */
export const time: aval<number> = current.bind(c => c.time);

/**
 * Ambient scene-query bound to the active control's view/proj/viewport
 * and pick registry. Lazy-built per ambient-context tick. When no
 * control is mounted, returns a query against an empty registry —
 * `intersect` / `pickAt` then always resolve to `undefined`.
 *
 * Single-active-control assumption mirrors `view`/`proj`/`viewport`
 * above: the LAST mounted RenderControl wins.
 */
let _emptyRegistry: PickRegistry | undefined;
function emptyRegistry(): PickRegistry {
  if (_emptyRegistry === undefined) _emptyRegistry = new PickRegistry();
  return _emptyRegistry;
}

export const query: SceneQuery = {
  intersect: (ray) => current.bind(c => {
    const reg = c.registry ?? emptyRegistry();
    const viewportV2i: aval<V2i> = c.viewport.map(v => new V2i(v.width, v.height));
    return createSceneQuery(reg, c.view, c.proj, viewportV2i).intersect(ray);
  }),
  pickAt: (cursorPixel) => current.bind(c => {
    const reg = c.registry ?? emptyRegistry();
    const viewportV2i: aval<V2i> = c.viewport.map(v => new V2i(v.width, v.height));
    return createSceneQuery(reg, c.view, c.proj, viewportV2i).pickAt(cursorPixel);
  }),
};
