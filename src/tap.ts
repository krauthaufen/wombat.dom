// Global `tap` / `dbltap` synthesis — the DOM-level counterpart of the scene
// dispatcher's OnTap (which only fires when a scene node is actually hit).
//
// Same dance Aardvark.Dom does in `aardvark-dom.js`, and deliberately the same
// thresholds, so `OnTap` means the same thing in both stacks:
//
//   · pointerdown / pointerup are watched on `window` in the CAPTURE phase and
//     paired by `pointerId` — a tap is recognised no matter which element
//     handles (or stops) the events, so an overlay that preventDefaults touch
//     cannot suppress it, and no synthesized `click` is required.
//   · EVERY pointer type qualifies (mouse, pen, touch); consumers branch on
//     `pointerType` / `button` themselves.
//   · tap    — press <= 400 ms with <= 20 px euclidean travel (client coords)
//   · dbltap — a second tap within 600 ms and 30 px of the previous one
//   · the synthesized event is a bubbling PointerEvent dispatched on the
//     pointerup's `target`, carrying `movementX/Y` (travel) and `deltaTime`.
//
// Installed once, lazily, by `mount` — so any wombat.dom app can just write
// `<div onTap={…}>` (or `Dom.OnTap` from F#) and it works.

/** Press duration (ms) below which a press counts as a tap. */
export const TAP_MAX_DURATION_MS = 400;
/** Euclidean travel (css px) below which a press counts as a tap. */
export const TAP_MAX_MOVE_PX = 20;
/** Tap-to-tap gap (ms) below which two taps count as a double tap. */
export const DOUBLE_TAP_GAP_MS = 600;
/** Tap-to-tap distance (css px) below which two taps count as a double tap. */
export const DOUBLE_TAP_MOVE_PX = 30;

/** Extra fields carried on the synthesized events. */
export interface TapEventExtras {
  /** Net down→up travel, css px. */
  readonly movementX: number;
  readonly movementY: number;
  /** Press duration (tap) / tap-to-tap gap (dbltap), ms. */
  readonly deltaTime: number;
}

export type TapEvent = PointerEvent & TapEventExtras;

let installed = false;

/**
 * Install the global tap/dbltap synthesis. Idempotent; `mount` calls it, so
 * apps normally don't have to. Safe to call in a non-DOM context (no-op).
 */
export function installTapEvents(): void {
  if (installed || typeof window === "undefined") return;
  installed = true;

  const down = new Map<number, PointerEvent>();

  // GHOST SUPPRESSION. After a touch, Safari/WebKit (and Chrome for
  // touch-emulating mice) replay the interaction as compatibility MOUSE
  // events, which surface here as a SECOND tap with pointerType "mouse" a
  // few hundred ms later, at the same spot. A consumer that branches on
  // pointerType (touch → open a menu, mouse → dismiss it) then sees the menu
  // open and instantly close again. Swallow mouse taps that shadow a recent
  // touch tap — the standard ghost-click guard, at the layer that owns tap.
  const GHOST_WINDOW_MS = 700;
  const GHOST_MOVE_PX = 30;
  let lastTouch: { t: number; x: number; y: number } | null = null;

  const dispatch = (
    type: "tap" | "dbltap", source: PointerEvent, target: EventTarget,
    dx: number, dy: number, dt: number,
  ): void => {
    // Clone the SOURCE event's fields (the pointerdown for `tap`, the previous
    // tap for `dbltap`) so clientX/Y, button, buttons, pointerType and the
    // modifier keys all read like the originating press.
    const init: Record<string, unknown> = {};
    for (const k in source) init[k] = (source as unknown as Record<string, unknown>)[k];
    init.bubbles = true;
    // movementX/Y are read-only accessors on the event object — they must go
    // through the init dict (they are legitimate MouseEventInit members).
    // Assigning them afterwards throws in strict mode (module code), which is
    // how a tap silently disappears.
    init.movementX = dx;
    init.movementY = dy;
    const evt = new PointerEvent(type, init as PointerEventInit);
    Object.defineProperty(evt, "deltaTime", { value: dt, enumerable: true });
    target.dispatchEvent(evt);
  };

  window.addEventListener("pointerdown", (e: PointerEvent) => {
    down.set(e.pointerId, e);
  }, true);

  window.addEventListener("pointerup", (e: PointerEvent) => {
    const start = down.get(e.pointerId);
    if (start === undefined) return;
    down.delete(e.pointerId);
    const dt = e.timeStamp - start.timeStamp;
    const dx = e.clientX - start.clientX;
    const dy = e.clientY - start.clientY;
    if (dt > TAP_MAX_DURATION_MS || Math.hypot(dx, dy) > TAP_MAX_MOVE_PX) return;
    if (start.pointerType === "touch") {
      lastTouch = { t: e.timeStamp, x: e.clientX, y: e.clientY };
    } else if (lastTouch !== null) {
      const ghost =
        e.timeStamp - lastTouch.t < GHOST_WINDOW_MS &&
        Math.hypot(e.clientX - lastTouch.x, e.clientY - lastTouch.y) < GHOST_MOVE_PX;
      if (ghost) return; // compatibility-mouse replay of the touch tap
    }
    dispatch("tap", start, e.target ?? window, dx, dy, dt);
  }, true);

  window.addEventListener("pointercancel", (e: PointerEvent) => {
    down.delete(e.pointerId);
  }, true);

  let lastTap: PointerEvent | null = null;
  window.addEventListener("tap", (ev: Event) => {
    const e = ev as PointerEvent;
    if (lastTap !== null) {
      const dt = e.timeStamp - lastTap.timeStamp;
      const dx = e.clientX - lastTap.clientX;
      const dy = e.clientY - lastTap.clientY;
      if (dt < DOUBLE_TAP_GAP_MS && Math.hypot(dx, dy) < DOUBLE_TAP_MOVE_PX) {
        dispatch("dbltap", lastTap, e.target ?? window, dx, dy, dt);
        lastTap = null;
        return;
      }
    }
    lastTap = e;
  });
}
