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

import { AVal, ChangeableValue, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";

/**
 * Snapshot of a RenderControl's exposed avals. Updated on mount /
 * unmount; ambient avals below project individual fields out.
 */
export interface AmbientContext {
  readonly viewport: aval<{ width: number; height: number }>;
  readonly view: aval<Trafo3d>;
  readonly proj: aval<Trafo3d>;
  readonly time: aval<number>;
}

const fallback: AmbientContext = {
  viewport: AVal.constant({ width: 1, height: 1 }),
  view: AVal.constant(Trafo3d.identity),
  proj: AVal.constant(Trafo3d.identity),
  time: AVal.constant(0),
};

const current = new ChangeableValue<AmbientContext>(fallback);

/** Set the ambient context. Called by `<RenderControl>` on mount. */
export function setAmbient(ctx: AmbientContext): void {
  current.value = ctx;
}

/** Restore fallback. Called on `<RenderControl>` unmount. */
export function clearAmbient(): void {
  current.value = fallback;
}

/** Ambient viewport size — tracks the active control. */
export const viewport: aval<{ width: number; height: number }> = current.bind(c => c.viewport);
/** Ambient view trafo (world → view) of the active control. */
export const view: aval<Trafo3d> = current.bind(c => c.view);
/** Ambient projection trafo (view → clip) of the active control. */
export const proj: aval<Trafo3d> = current.bind(c => c.proj);
/** Ambient frame clock (`performance.now()`-style ms) of the active control. */
export const time: aval<number> = current.bind(c => c.time);
