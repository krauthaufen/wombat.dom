// PickDispatcher — wires pointer events on a canvas to handler
// dispatch via the pick registry. Now does Aardvark.Dom-style
// pixel-snap: a 33×33 region readback around the cursor, then a
// spiral walk over `SNAP_OFFSETS` in d²-ascending order. The first
// offset whose pickId belongs to a registered, active, non-pick-
// through scope AND falls within that scope's own
// `pixelSnapRadius²` wins.
//
// On top of the spiral resolve we run a capture/bubble dispatch
// across the scope's `handlers` chain (outer→inner). The chain is
// stored on `LeafPickScope` as the path from root to leaf; index 0
// is the outermost `<Sg On=...>` scope that wrapped this leaf.
//
// Differential enter/leave: when a pointermove hits a different
// scope than `lastHit`, we diff the two paths. Because we don't
// have a tree (we have an array per leaf), we approximate the
// "common ancestor" by the longest shared prefix of the OLD and
// NEW paths. Equality is reference equality on the shared
// `EventHandlers` objects — those are the identity carriers for
// `<Sg On=...>` scope nodes (`SgOn.handlers`), and the registry
// captures them by reference from the surrounding TraversalState.
// Two leaves under the same `<Sg On={H}>` scope therefore share
// the same H reference, so the prefix walk skips H during a
// sibling-to-sibling transition. This matches the F# common-
// ancestor pruning closely enough for our DAG-shaped event scopes.
//
// PointerCapture: while a pointer is captured, all of its events
// route to the captured scope INSTEAD of the spiral hit; lastHit
// is NOT updated (mirrors `SceneHandler.SetPointerCapture` / F#'s
// "no lastOver update while captured"). Releasing fires a synthetic
// pointermove using the last cursor position to re-establish hover.

import { AVal, avalAddCallback } from "@aardworx/wombat.adaptive";
import { Ray3d, Trafo3d, V3d, V4d } from "@aardworx/wombat.base";

import type { LeafPickScope, PickRegistry } from "./registry.js";
import { decodePick, readSlotsAt, type DecodedPick, type PickRegion } from "./readback.js";
import {
  SceneEvent,
  type SceneEventDispatch,
  type SceneEventKind,
} from "./sceneEvent.js";
import { SNAP_OFFSETS, SNAP_RADIUS_MAX } from "./snapOffsets.js";

// --- Tap / long-press / double-tap detection thresholds ----------------
// Sensible defaults. Exported so callers can read or — eventually —
// override them at the RenderControl level (see docs/FUTURE.md).

/** Max pointerdown→pointerup duration to count as a tap (ms). */
export const TAP_MAX_DURATION_MS = 250;
/** Max total movement during the press, in CSS pixels. */
export const TAP_MAX_MOVE_PX = 10;
/** Max gap between consecutive taps' pointerup times (ms). */
export const DOUBLE_TAP_GAP_MS = 300;
/** Max distance between the two taps' down positions (CSS px). */
export const DOUBLE_TAP_MOVE_PX = 20;
/** A pointerdown held still for this long fires OnLongPress (ms). */
export const LONG_PRESS_MS = 500;
/** Distance moved (CSS px) past which a press becomes a drag. Mirrors Aardvark's drag threshold. */
export const DRAG_THRESHOLD_PX = 5;
/** Cursor must rest still over the same scope this long before OnHover fires (ms). */
export const HOVER_DELAY_MS = 500;

/**
 * Per-RenderControl override of the tap/long-press/double-tap/drag/hover
 * thresholds. Any unset field falls back to the corresponding module
 * default constant.
 */
export interface TapThresholds {
  readonly tapMaxDurationMs?: number;
  readonly tapMaxMovePx?: number;
  readonly doubleTapGapMs?: number;
  readonly doubleTapMovePx?: number;
  readonly longPressMs?: number;
  readonly dragThresholdPx?: number;
  readonly hoverDelayMs?: number;
}

/**
 * Reader for the 33×33 pick region centred on `(x, y)` in device
 * pixels. The renderControl wires this to `readPickRegion`; tests
 * mock it with a canned region.
 */
export type ReadRegion = (x: number, y: number) => Promise<PickRegion | undefined>;

interface SpiralHit {
  readonly scope: LeafPickScope;
  readonly decoded: DecodedPick;
  readonly hitPxX: number;
  readonly hitPxY: number;
  readonly d2: number;
}

/**
 * Result of the BVH ray fall-through: a non-pickThrough scope whose
 * world-space intersectable was hit by the cursor ray, plus the
 * world-space hit point + outward normal. The dispatcher unprojects
 * those into view space using the scope's `view` to feed
 * `SceneEvent.viewPos` / `viewNormal`.
 */
interface BvhFallthrough {
  readonly scope: LeafPickScope;
  readonly worldPoint: V3d;
  readonly worldNormal: V3d;
}

/**
 * Snapshot of the most-recent move for synthetic-move replay on
 * capture release. Stores enough to reconstruct an event without
 * re-reading the framebuffer.
 */
interface LastMoveInfo {
  readonly raw: PointerEvent;
  readonly cssX: number;
  readonly cssY: number;
  readonly rect: DOMRect;
  readonly sx: number;
  readonly sy: number;
  readonly hit: SpiralHit | undefined;
}

/**
 * Per-pointer state tracked from pointerdown onward, used to
 * synthesise OnTap / OnLongPress. Cleared on pointerup, pointercancel,
 * or when movement breaks the tap-distance threshold.
 *
 * `consumed` flips true when OnLongPress fires — the trailing
 * pointerup then suppresses OnTap.
 */
interface PressState {
  readonly downAt: number;
  readonly downX: number;
  readonly downY: number;
  /** Resolved at pointerdown via spiral readback. May be 0 (no hit). */
  readonly hitPickId: number;
  longPressTimer: ReturnType<typeof setTimeout> | undefined;
  consumed: boolean;
  movedTooFar: boolean;
  /** Phase 6 — drag-state machine. Once `dragging`, the press routes
   * pointermove events as `OnDrag` to `dragScopeId`. */
  dragging: boolean;
  dragScopeId: number;
}

/** Trailing-tap state for double-tap detection. */
interface LastTapInfo {
  readonly at: number;
  readonly x: number;
  readonly y: number;
  readonly pickId: number;
}

export class PickDispatcher implements SceneEventDispatch {
  private lastHit: number = 0;
  private lastPath: ReadonlyArray<import("../sg.js").EventHandlers> = [];
  private seq: number = 0;
  private lastSettledSeq: number = 0;

  private readonly capturedScopes: Map<number, LeafPickScope> = new Map();
  private lastMove: LastMoveInfo | undefined;

  // Tap / long-press synthesis. Per-pointer press state created on
  // pointerdown, finalised at pointerup (or cancelled on big move /
  // pointercancel). `lastTap` is shared across pointers — double-tap
  // is only fired when the second tap lands on the SAME pickId, so a
  // multi-touch second finger on a different scope won't match.
  private readonly presses: Map<number, PressState> = new Map();
  private lastTap: LastTapInfo | undefined;

  // Multi-touch gesture state. Tracks every active pointer's last
  // CSS-pixel position; OnPinch / OnTwoFingerPan / OnTwoFingerRotate
  // are synthesised from this map on every pointermove with ≥2 active
  // pointers. The "primary" gesture scope is captured at the moment
  // the second pointer goes down (whichever scope the first pointer
  // was over), so subsequent finger movement keeps routing to the
  // same target even if individual fingers wander.
  private readonly activePointers: Map<number, { x: number; y: number }> = new Map();
  private gestureScope: LeafPickScope | undefined;
  private gestureLastDist: number | undefined;
  private gestureLastAngle: number | undefined;
  private gestureLastCenter: { x: number; y: number } | undefined;

  // Hover dwell. Each move that resolves to a different scope cancels
  // the pending timer and schedules a new one for the new scope; on
  // fire we dispatch OnHover via capture/bubble. Cancelled on leave.
  private hoverTimer: ReturnType<typeof setTimeout> | undefined;
  private hoverPickId: number | undefined;

  // Effective thresholds (instance copy of the module defaults; can
  // be overridden via the ctor to support per-RenderControl tuning).
  private readonly tTapMaxDuration: number;
  private readonly tTapMaxMove: number;
  private readonly tDoubleTapGap: number;
  private readonly tDoubleTapMove: number;
  private readonly tLongPress: number;
  private readonly tDragThreshold: number;
  private readonly tHoverDelay: number;

  constructor(
    private readonly registry: PickRegistry,
    /** Used by BVH ray fall-through to construct the world-space cursor ray. */
    private readonly _getView: () => Trafo3d,
    private readonly _getProj: () => Trafo3d,
    private readonly getCanvasRect: () => DOMRect,
    thresholds?: TapThresholds,
  ) {
    this.tTapMaxDuration = thresholds?.tapMaxDurationMs ?? TAP_MAX_DURATION_MS;
    this.tTapMaxMove     = thresholds?.tapMaxMovePx     ?? TAP_MAX_MOVE_PX;
    this.tDoubleTapGap   = thresholds?.doubleTapGapMs   ?? DOUBLE_TAP_GAP_MS;
    this.tDoubleTapMove  = thresholds?.doubleTapMovePx  ?? DOUBLE_TAP_MOVE_PX;
    this.tLongPress      = thresholds?.longPressMs      ?? LONG_PRESS_MS;
    this.tDragThreshold  = thresholds?.dragThresholdPx  ?? DRAG_THRESHOLD_PX;
    this.tHoverDelay     = thresholds?.hoverDelayMs     ?? HOVER_DELAY_MS;
  }

  /**
   * Wire pointer listeners to the canvas. Returns a disposer.
   */
  attach(canvas: HTMLCanvasElement, readRegion: ReadRegion): () => void {
    const handle = (ev: PointerEvent, kind: SceneEventKind): void => {
      const seq = ++this.seq;
      const rect = this.getCanvasRect();
      const cssX = ev.clientX - rect.left;
      const cssY = ev.clientY - rect.top;

      const sx = rect.width  > 0 ? canvas.width  / rect.width  : 1;
      const sy = rect.height > 0 ? canvas.height / rect.height : 1;
      const devX = Math.floor(cssX * sx);
      const devY = Math.floor(cssY * sy);

      void readRegion(devX, devY).then((region) => {
        if (seq < this.lastSettledSeq) return;
        this.lastSettledSeq = seq;
        const hit = region !== undefined ? this.spiralResolve(region, devX, devY) : undefined;
        const bvhFall = hit === undefined ? this.bvhFallthrough(devX, devY, rect, sx, sy) : undefined;
        this.dispatch(ev, kind, cssX, cssY, hit, rect, sx, sy, bvhFall);
      });
    };

    const onDown   = (e: PointerEvent): void => handle(e, "OnPointerDown");
    const onUp     = (e: PointerEvent): void => handle(e, "OnPointerUp");
    const onMove   = (e: PointerEvent): void => handle(e, "OnPointerMove");
    const onClick  = (e: PointerEvent): void => handle(e, "OnClick");
    const onEnter  = (e: PointerEvent): void => handle(e, "OnPointerEnter");
    const onCancel = (e: PointerEvent): void => {
      // Phase 6 — fire OnDragEnd if a drag was active. We don't have
      // fresh cursor coords, so reuse the press's original down pos.
      const press = this.presses.get(e.pointerId);
      if (press !== undefined && press.dragging) {
        const scope = this.registry.lookup(press.dragScopeId);
        if (scope !== undefined) this.dispatchDrag("OnDragEnd", e, press.downX, press.downY, scope, press);
        press.dragging = false;
      }
      this.cancelPress(e.pointerId);
    };
    const onLeave  = (e: PointerEvent): void => {
      this.cancelHover();
      if (this.lastHit !== 0) {
        const scope = this.registry.lookup(this.lastHit);
        if (scope !== undefined && !scope.pickThrough && AVal.force(scope.active)) {
          const rect = this.getCanvasRect();
          const cssX = e.clientX - rect.left;
          const cssY = e.clientY - rect.top;
          const ev = this.makeEvent("OnPointerLeave", e, cssX, cssY, scope, scope.pickId, false, undefined);
          this.runUpAll(this.lastPath, [], ev);
        }
        this.lastHit = 0;
        this.lastPath = [];
      }
    };

    // Why { passive: false }: SceneEvent.preventDefault() / stopPropagation()
    // delegate to the underlying PointerEvent. Passive listeners can't
    // call preventDefault — and the browser default is `passive: true`
    // for touchmove / wheel on a canvas. Force non-passive so handlers
    // actually take effect.
    const opts: AddEventListenerOptions = { passive: false };
    canvas.addEventListener("pointerdown", onDown, opts);
    canvas.addEventListener("pointerup", onUp, opts);
    canvas.addEventListener("pointermove", onMove, opts);
    canvas.addEventListener("pointercancel", onCancel as unknown as EventListener, opts);
    canvas.addEventListener("click", onClick as unknown as EventListener, opts);
    canvas.addEventListener("pointerenter", onEnter, opts);
    canvas.addEventListener("pointerleave", onLeave, opts);

    // Phase 4 — wheel events. Read pixel under cursor (same spiral
    // as pointer events), then dispatch via capture/bubble through
    // the hit's path. Non-passive so handlers can preventDefault().
    const onWheel = (ev: WheelEvent): void => {
      const seq = ++this.seq;
      const rect = this.getCanvasRect();
      const cssX = ev.clientX - rect.left;
      const cssY = ev.clientY - rect.top;
      const sx = rect.width  > 0 ? canvas.width  / rect.width  : 1;
      const sy = rect.height > 0 ? canvas.height / rect.height : 1;
      const devX = Math.floor(cssX * sx);
      const devY = Math.floor(cssY * sy);
      void readRegion(devX, devY).then((region) => {
        if (seq < this.lastSettledSeq) return;
        this.lastSettledSeq = seq;
        const hit = region !== undefined ? this.spiralResolve(region, devX, devY) : undefined;
        const bvhFall = hit === undefined ? this.bvhFallthrough(devX, devY, rect, sx, sy) : undefined;
        this.dispatchWheel(ev, cssX, cssY, hit, rect, sx, sy, bvhFall);
      });
    };
    canvas.addEventListener("wheel", onWheel as unknown as EventListener, opts);

    // Phase 5 — keyboard. Routed to focused scope's handler chain.
    // The canvas needs to be focusable for a real DOM `focus` to
    // direct keys to it; the renderControl sets `tabindex=0`.
    const onKey = (kind: SceneEventKind) => (ev: KeyboardEvent): void => {
      const focused = AVal.force(this.registry.focusedPickId);
      if (focused === undefined) return;
      const scope = this.registry.lookup(focused);
      if (scope === undefined) return;
      if (!AVal.force(scope.active)) return;
      const sceneEv = new SceneEvent({
        kind,
        clientX: 0, clientY: 0,
        pickId: scope.pickId,
        modeB: false,
        pointerId: 0, pointerType: "",
        raw: ev,
        key: ev.key, code: ev.code, repeat: ev.repeat,
        ctrlKey: ev.ctrlKey, shiftKey: ev.shiftKey, altKey: ev.altKey, metaKey: ev.metaKey,
        scope, dispatch: this,
      });
      this.runCaptureBubble(scope.handlers, sceneEv);
    };
    const onKeyDown  = onKey("OnKeyDown");
    const onKeyUp    = onKey("OnKeyUp");
    const onKeyPress = onKey("OnKeyPress");
    canvas.addEventListener("keydown",  onKeyDown  as EventListener, opts);
    canvas.addEventListener("keyup",    onKeyUp    as EventListener, opts);
    canvas.addEventListener("keypress", onKeyPress as EventListener, opts);

    // Phase 5 — focus subscription: when focusedPickId changes,
    // dispatch OnBlur on the old scope's path then OnFocus on the
    // new one's. Both fire capture+bubble together (like enter/leave).
    let prevFocus: number | undefined;
    const focusUnsub = avalAddCallback(this.registry.focusedPickId, (next) => {
      const old = prevFocus;
      prevFocus = next;
      if (old !== undefined) {
        const oldScope = this.registry.lookup(old);
        if (oldScope !== undefined) {
          const blur = new SceneEvent({
            kind: "OnBlur", clientX: 0, clientY: 0,
            pickId: oldScope.pickId, modeB: false,
            pointerId: 0, pointerType: "",
            raw: new Event("blur") as FocusEvent,
            ...(next !== undefined ? { relatedTarget: next } : {}),
            scope: oldScope, dispatch: this,
          });
          this.runUpAll(oldScope.handlers, [], blur);
        }
      }
      if (next !== undefined) {
        const newScope = this.registry.lookup(next);
        if (newScope !== undefined) {
          const focus = new SceneEvent({
            kind: "OnFocus", clientX: 0, clientY: 0,
            pickId: newScope.pickId, modeB: false,
            pointerId: 0, pointerType: "",
            raw: new Event("focus") as FocusEvent,
            ...(old !== undefined ? { relatedTarget: old } : {}),
            scope: newScope, dispatch: this,
          });
          this.runDownAll(newScope.handlers, [], focus);
        }
      }
    });

    return () => {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointercancel", onCancel as unknown as EventListener);
      canvas.removeEventListener("click", onClick as unknown as EventListener);
      canvas.removeEventListener("pointerenter", onEnter);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("wheel", onWheel as unknown as EventListener);
      canvas.removeEventListener("keydown",  onKeyDown  as EventListener);
      canvas.removeEventListener("keyup",    onKeyUp    as EventListener);
      canvas.removeEventListener("keypress", onKeyPress as EventListener);
      focusUnsub.dispose();
      // Cancel any pending long-press timers so the disposer is clean.
      for (const id of [...this.presses.keys()]) this.cancelPress(id);
      this.cancelHover();
    };
  }

  private dispatchWheel(
    ev: WheelEvent,
    cssX: number,
    cssY: number,
    hit: SpiralHit | undefined,
    rect: DOMRect,
    sx: number,
    sy: number,
    bvhFall: BvhFallthrough | undefined,
  ): void {
    let scope: LeafPickScope | undefined;
    let viewPos: V3d | undefined;
    let viewNormal: V3d | undefined;
    let modeB = false;
    if (hit !== undefined) {
      scope = hit.scope;
      modeB = hit.decoded.modeB;
      viewPos = this.viewPosFor(scope, hit, rect, sx, sy);
    } else if (bvhFall !== undefined) {
      scope = bvhFall.scope;
      const v = AVal.force(scope.view);
      const p4 = v.forward.transform(new V4d(bvhFall.worldPoint.x, bvhFall.worldPoint.y, bvhFall.worldPoint.z, 1));
      const pw = p4.w !== 0 ? p4.w : 1;
      viewPos = new V3d(p4.x / pw, p4.y / pw, p4.z / pw);
      const n4 = v.forward.transform(new V4d(bvhFall.worldNormal.x, bvhFall.worldNormal.y, bvhFall.worldNormal.z, 0));
      viewNormal = new V3d(n4.x, n4.y, n4.z);
    }
    if (scope === undefined) return;
    const sceneEv = new SceneEvent({
      kind: "OnWheel",
      clientX: cssX,
      clientY: cssY,
      pickId: scope.pickId,
      modeB,
      ...(viewPos !== undefined ? { viewPos } : {}),
      ...(viewNormal !== undefined ? { viewNormal } : {}),
      pointerId: 0,
      pointerType: "",
      raw: ev,
      deltaX: ev.deltaX,
      deltaY: ev.deltaY,
      deltaZ: ev.deltaZ,
      deltaMode: (ev.deltaMode as 0 | 1 | 2),
      ctrlKey: ev.ctrlKey, shiftKey: ev.shiftKey, altKey: ev.altKey, metaKey: ev.metaKey,
      scope,
      dispatch: this,
    });
    this.runCaptureBubble(scope.handlers, sceneEv);
  }

  // --- tap / long-press synthesis -----------------------------------------

  private cancelPress(pointerId: number): void {
    const p = this.presses.get(pointerId);
    if (p === undefined) return;
    if (p.longPressTimer !== undefined) clearTimeout(p.longPressTimer);
    this.presses.delete(pointerId);
  }

  /**
   * Dispatch a synthesised SceneEvent through the same capture/bubble
   * pipeline as a real pointer event. `kind` ∈ {OnTap, OnDoubleTap,
   * OnLongPress}. `raw` is the originating pointer event (pointerup
   * for tap/double-tap, pointerdown for long-press).
   */
  private dispatchSynthetic(
    kind: SceneEventKind,
    raw: PointerEvent,
    cssX: number,
    cssY: number,
    scope: LeafPickScope,
    modeB: boolean,
    viewPos: V3d | undefined,
  ): SceneEvent {
    const sceneEv = this.makeEvent(kind, raw, cssX, cssY, scope, scope.pickId, modeB, viewPos);
    this.runCaptureBubble(scope.handlers, sceneEv);
    return sceneEv;
  }

  // --- pointer capture API -------------------------------------------------

  setPointerCapture(scope: LeafPickScope, pointerId: number): void {
    this.capturedScopes.set(pointerId, scope);
  }

  releasePointerCapture(scope: LeafPickScope, pointerId: number): void {
    const cur = this.capturedScopes.get(pointerId);
    if (cur === undefined || cur !== scope) return;
    this.capturedScopes.delete(pointerId);

    // Synthetic move to re-establish hover state at the last cursor
    // position. Matches `SceneHandler.ReleasePointerCapture`.
    const lm = this.lastMove;
    if (lm === undefined) return;
    this.dispatch(lm.raw, "OnPointerMove", lm.cssX, lm.cssY, lm.hit, lm.rect, lm.sx, lm.sy);
  }

  hasPointerCapture(scope: LeafPickScope, pointerId: number): boolean {
    return this.capturedScopes.get(pointerId) === scope;
  }

  // --- spiral hit-test -----------------------------------------------------

  private spiralResolve(region: PickRegion, centerX: number, centerY: number): SpiralHit | undefined {
    // Per-scope snap-r² cache, lazily populated as we walk. Avoids
    // forcing the same `pixelSnapRadius` aval for every offset that
    // touches the same scope.
    const r2Cache = new Map<number, number>();
    const snapR2 = (scope: LeafPickScope): number => {
      const cached = r2Cache.get(scope.pickId);
      if (cached !== undefined) return cached;
      const raw = AVal.force(scope.pixelSnapRadius);
      const clamped = Math.min(SNAP_RADIUS_MAX, Math.max(0, Math.floor(raw)));
      const r2 = clamped * clamped;
      r2Cache.set(scope.pickId, r2);
      return r2;
    };

    for (let i = 0; i < SNAP_OFFSETS.length; i++) {
      const off = SNAP_OFFSETS[i]!;
      const lx = (centerX + off.dx) - region.originX;
      const ly = (centerY + off.dy) - region.originY;
      const slots = readSlotsAt(region, lx, ly);
      const decoded = decodePick(slots);
      if (decoded.pickId === 0) continue;
      const scope = this.registry.lookup(decoded.pickId);
      if (scope === undefined) continue;
      // pickThrough on a pixel hit: skip dispatch entirely. Mirrors
      // F# `SceneHandler.fs:1814` — it warns and keeps the winner;
      // we don't have BVH fall-through, so we just bail.
      if (scope.pickThrough) continue;
      if (!AVal.force(scope.active)) continue;
      if (off.d2 > snapR2(scope)) continue;
      return {
        scope,
        decoded,
        hitPxX: centerX + off.dx,
        hitPxY: centerY + off.dy,
        d2: off.d2,
      };
    }
    return undefined;
  }

  // --- core dispatch -------------------------------------------------------

  private dispatch(
    ev: PointerEvent,
    kind: SceneEventKind,
    cssX: number,
    cssY: number,
    hit: SpiralHit | undefined,
    rect: DOMRect,
    sx: number,
    sy: number,
    bvhFall?: BvhFallthrough,
  ): void {
    const pointerId = ev.pointerId ?? 0;

    // ---- Multi-touch gesture tracking ---------------------------------
    // Maintain `activePointers` and synthesise pinch / two-finger
    // pan / two-finger rotate events when ≥2 pointers are active.
    // Pointer "type" check is intentionally absent — works for touch,
    // pen, and even mouse (rarely useful for the latter, but cheap).
    if (kind === "OnPointerDown") {
      this.activePointers.set(pointerId, { x: cssX, y: cssY });
      // On the second-finger-down, latch the gesture's target scope
      // to whatever is currently under the cursor (the spiral hit
      // produced by THIS down event).
      if (this.activePointers.size === 2 && hit !== undefined) {
        this.gestureScope = hit.scope;
      }
      this.refreshGestureRefs();
    } else if (kind === "OnPointerMove") {
      const prev = this.activePointers.get(pointerId);
      this.activePointers.set(pointerId, { x: cssX, y: cssY });
      if (prev !== undefined && this.activePointers.size >= 2) {
        this.dispatchGestureEvents(ev);
      }
    } else if (kind === "OnPointerUp") {
      this.activePointers.delete(pointerId);
      if (this.activePointers.size < 2) {
        this.gestureScope = undefined;
        this.gestureLastDist = undefined;
        this.gestureLastAngle = undefined;
        this.gestureLastCenter = undefined;
      } else {
        this.refreshGestureRefs();
      }
    }

    // Always record the latest cursor position for synthetic-move
    // replay on capture release. Track `hit` from the spiral so we
    // can replay against the geometry under the cursor at release
    // time (close enough — the framebuffer may have changed, but we
    // don't have a fresh region read in hand).
    this.lastMove = { raw: ev, cssX, cssY, rect, sx, sy, hit };

    let captured = this.capturedScopes.get(pointerId);
    // If the captured scope was removed from the registry (e.g.
    // scene-graph subtree unmounted), the registry's `lookup` either
    // returns undefined or a different scope under the recycled id.
    // Either way we drop the stale capture so the event falls back to
    // the spiral hit.
    if (captured !== undefined) {
      const fresh = this.registry.lookup(captured.pickId);
      if (fresh !== captured) {
        this.capturedScopes.delete(pointerId);
        captured = undefined;
      }
    }
    if (captured !== undefined) {
      // Route to captured scope; do NOT update lastHit/lastPath. F#
      // `SceneHandler.fs:1700–` skips lastOver updates while a
      // pointer is captured, then re-fires move on release.
      const viewPos = hit !== undefined ? this.viewPosFor(captured, hit, rect, sx, sy) : undefined;
      const sceneEv = this.makeEvent(kind, ev, cssX, cssY, captured, captured.pickId, hit?.decoded.modeB ?? false, viewPos);
      this.runCaptureBubble(captured.handlers, sceneEv);
      return;
    }

    // BVH fall-through: when the spiral didn't yield a non-pickThrough
    // hit but the registry has intersectables, use the cursor ray's
    // closest non-pickThrough scope as the dispatch target. The
    // viewPos/viewNormal come from the world-space BVH hit transformed
    // into the scope's view space.
    let bvhScope: LeafPickScope | undefined;
    let bvhViewPos: V3d | undefined;
    let bvhViewNormal: V3d | undefined;
    if (hit === undefined && bvhFall !== undefined) {
      bvhScope = bvhFall.scope;
      const v = AVal.force(bvhScope.view);
      const p4 = v.forward.transform(new V4d(bvhFall.worldPoint.x, bvhFall.worldPoint.y, bvhFall.worldPoint.z, 1));
      const pw = p4.w !== 0 ? p4.w : 1;
      bvhViewPos = new V3d(p4.x / pw, p4.y / pw, p4.z / pw);
      // Normal: transform as a direction (w=0). Strictly correct for
      // affine view trafos; acceptable for the camera trafos we use.
      const n4 = v.forward.transform(new V4d(bvhFall.worldNormal.x, bvhFall.worldNormal.y, bvhFall.worldNormal.z, 0));
      bvhViewNormal = new V3d(n4.x, n4.y, n4.z);
    }

    const hitScope = hit?.scope ?? bvhScope;
    const hitId = hitScope?.pickId ?? 0;
    const newPath = hitScope?.handlers ?? [];

    if (kind === "OnPointerMove") {
      // Hover dwell: any move cancels the pending timer; we re-schedule
      // when the move resolves to a (still-current) scope below.
      if (hitId !== this.hoverPickId) this.cancelHover();
    }

    if (kind === "OnPointerMove" && hitId !== this.lastHit) {
      // Differential enter/leave. Diff oldPath vs newPath by
      // longest shared prefix (reference equality on EventHandlers
      // objects — see file header).
      const prefix = sharedPrefixLength(this.lastPath, newPath);

      if (this.lastHit !== 0) {
        const oldScope = this.registry.lookup(this.lastHit);
        if (oldScope !== undefined && !oldScope.pickThrough && AVal.force(oldScope.active)) {
          const leaveEv = this.makeEvent("OnPointerLeave", ev, cssX, cssY, oldScope, oldScope.pickId, false, undefined);
          // Walk the OLD path (above prefix) firing OnPointerLeave
          // — F# `runUp` fires capture+bubble together and ignores
          // continue/stop. We mirror that.
          this.runUpAll(this.lastPath, this.lastPath.slice(0, prefix), leaveEv);
        }
      }

      if (hitScope !== undefined) {
        const viewPos = hit !== undefined
          ? this.viewPosFor(hitScope, hit, rect, sx, sy)
          : bvhViewPos;
        const modeB = hit?.decoded.modeB ?? false;
        const enterEv = this.makeEvent("OnPointerEnter", ev, cssX, cssY, hitScope, hitScope.pickId, modeB, viewPos, bvhViewNormal);
        this.runDownAll(newPath, newPath.slice(0, prefix), enterEv);
      }

      this.lastHit = hitId;
      this.lastPath = newPath;
    }

    if (hitScope === undefined) {
      if (kind === "OnPointerMove") this.cancelHover();
      // No scope under cursor: still let pointermove cancel a press
      // that drifted off-target (movement check below). Pointerdown /
      // pointerup with no hit can't synthesise a tap (no scope to
      // dispatch to), so we just return.
      if (kind === "OnPointerMove") {
        this.checkPressMove(pointerId, cssX, cssY);
        // A drag in progress still gets OnDrag fired on its press scope
        // even if the cursor leaves the geometry — Aardvark / DOM
        // semantics.
        this.maybeDispatchDrag(pointerId, ev, cssX, cssY);
      }
      if (kind === "OnPointerUp") {
        this.maybeFinishDrag(pointerId, ev, cssX, cssY);
      }
      // A click landing on no scope clears focus.
      if (kind === "OnClick") this.registry.clearFocus();
      return;
    }

    const viewPos = hit !== undefined
      ? this.viewPosFor(hitScope, hit, rect, sx, sy)
      : bvhViewPos;
    const modeB = hit?.decoded.modeB ?? false;
    const sceneEv = this.makeEvent(kind, ev, cssX, cssY, hitScope, hitScope.pickId, modeB, viewPos, bvhViewNormal);
    this.runCaptureBubble(hitScope.handlers, sceneEv);

    // Hover dwell: a move that resolves to a known scope re-arms the
    // hover timer. The cancel above handles transitions to a different
    // scope; here we (re)schedule for the current scope.
    if (kind === "OnPointerMove") {
      this.scheduleHover(hitScope, cssX, cssY, ev);
    }

    // Phase 5 — auto-focus on click. Mirrors DOM mousedown→focus
    // behaviour but on `OnClick` (primary button) so it composes with
    // the existing tap/double-tap logic. CanFocus=true scopes win
    // focus; clicks outside any focusable scope clear it.
    if (kind === "OnClick") {
      if (hitScope.canFocus !== undefined && AVal.force(hitScope.canFocus)) {
        this.registry.setFocus(hitScope.pickId);
      } else {
        this.registry.clearFocus();
      }
    }

    // ---- tap / long-press synthesis -------------------------------------
    if (kind === "OnPointerDown") {
      // Replace any stale press for this pointer (extremely rare: a
      // missed pointerup / cancel).
      this.cancelPress(pointerId);
      const downAt = Date.now();
      const press: PressState = {
        downAt,
        downX: cssX,
        downY: cssY,
        hitPickId: hitScope.pickId,
        longPressTimer: undefined,
        consumed: false,
        movedTooFar: false,
        dragging: false,
        dragScopeId: hitScope.pickId,
      };
      press.longPressTimer = setTimeout(() => {
        const cur = this.presses.get(pointerId);
        if (cur === undefined || cur !== press) return;
        if (cur.movedTooFar) return;
        cur.consumed = true;
        cur.longPressTimer = undefined;
        // Re-resolve the scope by pickId — it may have been released
        // mid-press; if so, drop the long-press silently.
        const lpScope = this.registry.lookup(cur.hitPickId);
        if (lpScope === undefined || !AVal.force(lpScope.active) || lpScope.pickThrough) return;
        this.dispatchSynthetic("OnLongPress", ev, cssX, cssY, lpScope, modeB, viewPos);
      }, this.tLongPress);
      this.presses.set(pointerId, press);
    } else if (kind === "OnPointerMove") {
      this.checkPressMove(pointerId, cssX, cssY);
      this.maybeDispatchDrag(pointerId, ev, cssX, cssY);
    } else if (kind === "OnPointerUp") {
      // Phase 6 — DragEnd suppresses any trailing tap. We finalise
      // drag state here, BEFORE the tap detection below.
      const wasDrag = this.maybeFinishDrag(pointerId, ev, cssX, cssY);
      const press = this.presses.get(pointerId);
      this.cancelPress(pointerId);
      if (press === undefined) return;
      if (press.consumed) return; // long-press already fired
      if (wasDrag) return;
      if (press.movedTooFar) return;
      const dt = Date.now() - press.downAt;
      if (dt > this.tTapMaxDuration) return;
      // Movement check on the up position too — we may not have had a
      // pointermove between down and up.
      const mdx = cssX - press.downX;
      const mdy = cssY - press.downY;
      if (mdx * mdx + mdy * mdy > this.tTapMaxMove * this.tTapMaxMove) return;
      // Tap fires against the scope under the cursor at pointerup.
      this.dispatchSynthetic("OnTap", ev, cssX, cssY, hitScope, modeB, viewPos);

      // Double-tap detection: SAME pickId, gap ≤ DOUBLE_TAP_GAP_MS,
      // down-position move ≤ DOUBLE_TAP_MOVE_PX.
      const now = Date.now();
      const prev = this.lastTap;
      if (
        prev !== undefined
        && prev.pickId === hitScope.pickId
        && now - prev.at <= this.tDoubleTapGap
      ) {
        const dx = press.downX - prev.x;
        const dy = press.downY - prev.y;
        if (dx * dx + dy * dy <= this.tDoubleTapMove * this.tDoubleTapMove) {
          this.dispatchSynthetic("OnDoubleTap", ev, cssX, cssY, hitScope, modeB, viewPos);
          // Consume — a third quick tap should not pair with this
          // tap as the "previous" one.
          this.lastTap = undefined;
          return;
        }
      }
      this.lastTap = { at: now, x: press.downX, y: press.downY, pickId: hitScope.pickId };
    }
  }

  /**
   * Fire OnDragStart (once) / OnDrag (subsequent) on the press scope
   * if movement past `DRAG_THRESHOLD_PX` was reached. Drag dispatches
   * route to the press scope, NOT the current hover.
   */
  private maybeDispatchDrag(pointerId: number, ev: PointerEvent, cssX: number, cssY: number): void {
    const press = this.presses.get(pointerId);
    if (press === undefined) return;
    const scope = this.registry.lookup(press.dragScopeId);
    if (scope === undefined) return;
    const dx = cssX - press.downX;
    const dy = cssY - press.downY;
    const d2 = dx * dx + dy * dy;
    if (!press.dragging) {
      if (d2 < this.tDragThreshold * this.tDragThreshold) return;
      press.dragging = true;
      // Cancel pending long-press — drag wins.
      if (press.longPressTimer !== undefined) {
        clearTimeout(press.longPressTimer);
        press.longPressTimer = undefined;
      }
      this.dispatchDrag("OnDragStart", ev, cssX, cssY, scope, press);
      return;
    }
    this.dispatchDrag("OnDrag", ev, cssX, cssY, scope, press);
  }

  /**
   * If a drag is in progress for this pointer, fire OnDragEnd on the
   * press scope. Returns whether a drag had been active (caller uses
   * this to suppress trailing OnTap).
   */
  private maybeFinishDrag(pointerId: number, ev: PointerEvent, cssX: number, cssY: number): boolean {
    const press = this.presses.get(pointerId);
    if (press === undefined || !press.dragging) return false;
    const scope = this.registry.lookup(press.dragScopeId);
    if (scope !== undefined) {
      this.dispatchDrag("OnDragEnd", ev, cssX, cssY, scope, press);
    }
    press.dragging = false;
    return true;
  }

  private dispatchDrag(
    kind: SceneEventKind,
    ev: PointerEvent,
    cssX: number,
    cssY: number,
    scope: LeafPickScope,
    press: PressState,
  ): void {
    const sceneEv = new SceneEvent({
      kind,
      clientX: cssX,
      clientY: cssY,
      pickId: scope.pickId,
      modeB: false,
      ...(ev.button !== undefined ? { button: ev.button } : {}),
      ...(ev.buttons !== undefined ? { buttons: ev.buttons } : {}),
      pointerId: ev.pointerId ?? 0,
      pointerType: ev.pointerType ?? "",
      raw: ev,
      dragStartX: press.downX,
      dragStartY: press.downY,
      scope, dispatch: this,
    });
    this.runCaptureBubble(scope.handlers, sceneEv);
  }

  private checkPressMove(pointerId: number, cssX: number, cssY: number): void {
    const press = this.presses.get(pointerId);
    if (press === undefined || press.movedTooFar) return;
    const dx = cssX - press.downX;
    const dy = cssY - press.downY;
    if (dx * dx + dy * dy > this.tTapMaxMove * this.tTapMaxMove) {
      press.movedTooFar = true;
      if (press.longPressTimer !== undefined) {
        clearTimeout(press.longPressTimer);
        press.longPressTimer = undefined;
      }
    }
  }

  // --- capture/bubble walk -------------------------------------------------

  /**
   * Capture phase outer-first, bubble phase inner-first. A handler
   * returning `false` OR calling `stopPropagation()` halts further
   * dispatch. Capture-stop also skips the bubble phase entirely
   * (matches F# `runCapture` short-circuit).
   *
   * Why try/catch around each handler: defensive — a buggy scene-
   * graph handler shouldn't take down the runtime. Errors are
   * logged, then the loop continues. (Aardvark.Dom doesn't isolate
   * handlers, but TS userland tends to throw more freely.)
   */
  private runCaptureBubble(path: ReadonlyArray<import("../sg.js").EventHandlers>, ev: SceneEvent): void {
    // Capture phase: outer → inner
    for (let i = 0; i < path.length; i++) {
      const h = path[i]!.capture?.[ev.kind];
      if (h !== undefined) {
        let r: boolean | void;
        try { r = h(ev); } catch (err) { console.error(`[PickDispatcher] capture ${ev.kind} threw:`, err); continue; }
        if (r === false || ev.propagationStopped) return;
      }
    }
    // Bubble phase: inner → outer
    for (let i = path.length - 1; i >= 0; i--) {
      const h = path[i]!.bubble?.[ev.kind];
      if (h !== undefined) {
        let r: boolean | void;
        try { r = h(ev); } catch (err) { console.error(`[PickDispatcher] bubble ${ev.kind} threw:`, err); continue; }
        if (r === false || ev.propagationStopped) return;
      }
    }
  }

  /**
   * Fire on every scope along `path` (excluding any in `exclude`,
   * which is a prefix slice), collapsing capture+bubble — outer-first.
   * Matches F# `runUp`: "fire all, ignore continue/stop". We use it
   * for OnPointerLeave on the OLD path. Direction is leaf→root
   * (inner-first) per F#.
   */
  private runUpAll(
    path: ReadonlyArray<import("../sg.js").EventHandlers>,
    exclude: ReadonlyArray<import("../sg.js").EventHandlers>,
    ev: SceneEvent,
  ): void {
    for (let i = path.length - 1; i >= exclude.length; i--) {
      this.fireBoth(path[i]!, ev);
    }
  }

  private runDownAll(
    path: ReadonlyArray<import("../sg.js").EventHandlers>,
    exclude: ReadonlyArray<import("../sg.js").EventHandlers>,
    ev: SceneEvent,
  ): void {
    for (let i = exclude.length; i < path.length; i++) {
      this.fireBoth(path[i]!, ev);
    }
  }

  private fireBoth(h: import("../sg.js").EventHandlers, ev: SceneEvent): void {
    const cap = h.capture?.[ev.kind];
    if (cap !== undefined) {
      try { cap(ev); } catch (err) { console.error(`[PickDispatcher] capture ${ev.kind} threw:`, err); }
    }
    const bub = h.bubble?.[ev.kind];
    if (bub !== undefined) {
      try { bub(ev); } catch (err) { console.error(`[PickDispatcher] bubble ${ev.kind} threw:`, err); }
    }
  }

  // --- BVH fall-through ----------------------------------------------------

  /**
   * Build a world-space ray from the cursor pixel and walk the
   * registry's BVH for the closest hit on a non-pickThrough scope.
   * Returns undefined when the BVH is empty, no scope matched, or
   * only pickThrough scopes were intersected.
   */
  private bvhFallthrough(
    devX: number,
    devY: number,
    rect: DOMRect,
    sx: number,
    sy: number,
  ): BvhFallthrough | undefined {
    const bvh = this.registry.buildBvh();
    if (bvh === undefined) return undefined;
    if (rect.width <= 0 || rect.height <= 0) return undefined;

    const view = this._getView();
    const proj = this._getProj();
    const ray = this.unprojectWorldRay(devX, devY, rect, sx, sy, view, proj);

    const hit = bvh.closestHit(ray, 0, Number.POSITIVE_INFINITY, (key, value) => {
      const scope = this.registry.lookup(key);
      if (scope === undefined || scope.pickThrough) return undefined;
      if (!AVal.force(scope.active)) return undefined;
      // ForcePixelPicking opts a scope out of BVH ray fall-through.
      if (scope.forcePixelPicking !== undefined && AVal.force(scope.forcePixelPicking)) return undefined;
      return value.intersects(ray, 0, Number.POSITIVE_INFINITY);
    });
    if (hit === undefined) return undefined;
    const scope = this.registry.lookup(hit.key);
    if (scope === undefined) return undefined;
    return { scope, worldPoint: hit.point, worldNormal: hit.normal };
  }

  /**
   * Mode-A unprojection generalised: produce a world-space ray from
   * a device pixel through the scene at the supplied view+proj.
   * Reuses the same NDC math as `viewPosFor` (search "ndcX = (cssHitX
   * / rect.width)..." above) — z=-1 is the near plane, z=+1 the far
   * plane, then view⁻¹ pulls the two view-space points back into
   * world.
   */
  private unprojectWorldRay(
    devX: number,
    devY: number,
    rect: DOMRect,
    sx: number,
    sy: number,
    view: Trafo3d,
    proj: Trafo3d,
  ): Ray3d {
    const cssHitX = (devX + 0.5) / (sx > 0 ? sx : 1);
    const cssHitY = (devY + 0.5) / (sy > 0 ? sy : 1);
    const ndcX = (cssHitX / rect.width) * 2 - 1;
    const ndcY = 1 - (cssHitY / rect.height) * 2;
    const near = unprojectNdc(ndcX, ndcY, -1, proj, view);
    const far  = unprojectNdc(ndcX, ndcY,  1, proj, view);
    return Ray3d.fromPoints(near, far);
  }

  // --- helpers -------------------------------------------------------------

  private viewPosFor(
    scope: LeafPickScope,
    hit: SpiralHit,
    rect: DOMRect,
    sx: number,
    sy: number,
  ): V3d | undefined {
    const decoded = hit.decoded;
    if (decoded.modeB) return decoded.viewPos;
    if (rect.width <= 0 || rect.height <= 0) return undefined;
    // Use the spiral hit pixel (in device coords) for NDC, not the
    // raw cursor pixel — the snap may have moved us a few px.
    const cssHitX = (hit.hitPxX + 0.5) / (sx > 0 ? sx : 1);
    const cssHitY = (hit.hitPxY + 0.5) / (sy > 0 ? sy : 1);
    const ndcX = (cssHitX / rect.width) * 2 - 1;
    const ndcY = 1 - (cssHitY / rect.height) * 2;
    const ndcZ = decoded.raw.slot2;
    const proj: Trafo3d = AVal.force(scope.proj);
    const v4 = new V4d(ndcX, ndcY, ndcZ, 1);
    const viewSpace = proj.backward.transform(v4);
    const w = viewSpace.w !== 0 ? viewSpace.w : 1;
    return new V3d(viewSpace.x / w, viewSpace.y / w, viewSpace.z / w);
  }

  private makeEvent(
    kind: SceneEventKind,
    ev: PointerEvent,
    cssX: number,
    cssY: number,
    scope: LeafPickScope,
    pickId: number,
    modeB: boolean,
    viewPos: V3d | undefined,
    viewNormal?: V3d,
  ): SceneEvent {
    return new SceneEvent({
      kind,
      clientX: cssX,
      clientY: cssY,
      pickId,
      modeB,
      ...(ev.button !== undefined ? { button: ev.button } : {}),
      ...(ev.buttons !== undefined ? { buttons: ev.buttons } : {}),
      ...(viewPos !== undefined ? { viewPos } : {}),
      ...(viewNormal !== undefined ? { viewNormal } : {}),
      pointerId: ev.pointerId ?? 0,
      pointerType: ev.pointerType ?? "",
      raw: ev,
      scope,
      dispatch: this,
    });
  }

  // --- Multi-touch gesture synthesis ---------------------------------------

  private refreshGestureRefs(): void {
    if (this.activePointers.size < 2) {
      this.gestureLastDist = undefined;
      this.gestureLastAngle = undefined;
      this.gestureLastCenter = undefined;
      return;
    }
    const arr = Array.from(this.activePointers.values());
    const a = arr[0]!, b = arr[1]!;
    this.gestureLastDist = Math.hypot(a.x - b.x, a.y - b.y);
    this.gestureLastAngle = Math.atan2(b.y - a.y, b.x - a.x);
    let cx = 0, cy = 0;
    for (const p of arr) { cx += p.x; cy += p.y; }
    this.gestureLastCenter = { x: cx / arr.length, y: cy / arr.length };
  }

  private dispatchGestureEvents(raw: PointerEvent): void {
    if (this.gestureScope === undefined) return;
    const scope = this.registry.lookup(this.gestureScope.pickId);
    if (scope !== this.gestureScope) {
      // Captured-scope-style invalidation: the original gesture target
      // disappeared. Drop and let the next pointer-down latch a new one.
      this.gestureScope = undefined;
      return;
    }
    if (!AVal.force(scope.active)) return;

    const arr = Array.from(this.activePointers.values());
    const a = arr[0]!, b = arr[1]!;
    const newDist = Math.hypot(a.x - b.x, a.y - b.y);
    const newAngle = Math.atan2(b.y - a.y, b.x - a.x);
    let cx = 0, cy = 0;
    for (const p of arr) { cx += p.x; cy += p.y; }
    cx /= arr.length; cy /= arr.length;

    if (this.gestureLastDist !== undefined && this.gestureLastDist > 0) {
      const scale = newDist / this.gestureLastDist;
      if (Math.abs(scale - 1) > 1e-6) {
        const ev = new SceneEvent({
          kind: "OnPinch",
          clientX: cx, clientY: cy,
          pickId: scope.pickId, modeB: false,
          pointerId: raw.pointerId ?? 0,
          pointerType: raw.pointerType ?? "",
          raw,
          pinchScale: scale,
          pinchCenter: { x: cx, y: cy },
          scope, dispatch: this,
        });
        this.runCaptureBubble(scope.handlers, ev);
      }
    }
    if (this.gestureLastCenter !== undefined) {
      const dx = cx - this.gestureLastCenter.x;
      const dy = cy - this.gestureLastCenter.y;
      if (dx !== 0 || dy !== 0) {
        const ev = new SceneEvent({
          kind: "OnTwoFingerPan",
          clientX: cx, clientY: cy,
          pickId: scope.pickId, modeB: false,
          pointerId: raw.pointerId ?? 0,
          pointerType: raw.pointerType ?? "",
          raw,
          panDeltaX: dx, panDeltaY: dy,
          scope, dispatch: this,
        });
        this.runCaptureBubble(scope.handlers, ev);
      }
    }
    if (this.gestureLastAngle !== undefined) {
      let dA = newAngle - this.gestureLastAngle;
      while (dA > Math.PI) dA -= 2 * Math.PI;
      while (dA < -Math.PI) dA += 2 * Math.PI;
      if (Math.abs(dA) > 1e-6) {
        const ev = new SceneEvent({
          kind: "OnTwoFingerRotate",
          clientX: cx, clientY: cy,
          pickId: scope.pickId, modeB: false,
          pointerId: raw.pointerId ?? 0,
          pointerType: raw.pointerType ?? "",
          raw,
          rotateRadians: dA,
          scope, dispatch: this,
        });
        this.runCaptureBubble(scope.handlers, ev);
      }
    }

    this.gestureLastDist = newDist;
    this.gestureLastAngle = newAngle;
    this.gestureLastCenter = { x: cx, y: cy };
  }

  // --- Hover dwell ----------------------------------------------------------

  private cancelHover(): void {
    if (this.hoverTimer !== undefined) {
      clearTimeout(this.hoverTimer);
      this.hoverTimer = undefined;
    }
    this.hoverPickId = undefined;
  }

  private scheduleHover(scope: LeafPickScope, cssX: number, cssY: number, raw: PointerEvent): void {
    if (this.hoverPickId === scope.pickId && this.hoverTimer !== undefined) return;
    this.cancelHover();
    this.hoverPickId = scope.pickId;
    this.hoverTimer = setTimeout(() => {
      this.hoverTimer = undefined;
      const fresh = this.registry.lookup(scope.pickId);
      if (fresh !== scope) return;
      if (!AVal.force(scope.active)) return;
      const ev = new SceneEvent({
        kind: "OnHover",
        clientX: cssX, clientY: cssY,
        pickId: scope.pickId, modeB: false,
        pointerId: raw.pointerId ?? 0,
        pointerType: raw.pointerType ?? "",
        raw,
        scope, dispatch: this,
      });
      this.runCaptureBubble(scope.handlers, ev);
    }, this.tHoverDelay);
  }
}

function unprojectNdc(ndcX: number, ndcY: number, ndcZ: number, proj: Trafo3d, view: Trafo3d): V3d {
  // proj.backward: clip → view; view.backward: view → world.
  const v4 = proj.backward.transform(new V4d(ndcX, ndcY, ndcZ, 1));
  const w = v4.w !== 0 ? v4.w : 1;
  const viewSpace = new V4d(v4.x / w, v4.y / w, v4.z / w, 1);
  const w4 = view.backward.transform(viewSpace);
  const ww = w4.w !== 0 ? w4.w : 1;
  return new V3d(w4.x / ww, w4.y / ww, w4.z / ww);
}

function sharedPrefixLength<T>(a: ReadonlyArray<T>, b: ReadonlyArray<T>): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

// Suppress unused-import noise — kept for future getter-based proj
// override paths.
void Trafo3d;
