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
//
// AVal.force policy: every force in this file runs inside a pointer
// /key/focus event handler or a programmatic API entry called from
// user code — "now" is the user's tick. The force does not enter
// any reader's dependency set. Each call site below is in one of
// those contexts.

import { AVal, avalAddCallback } from "@aardworx/wombat.adaptive";
import { Trafo3d, V2d, V2i, V3d, V4d } from "@aardworx/wombat.base";

import type { LeafPickEntry, LeafPickScope, PickRegistry } from "./registry.js";
import { type PickRegion } from "./readback.js";
import {
  SceneEvent,
  type SceneEventDispatch,
  type SceneEventKind,
} from "./sceneEvent.js";
import { SceneEventLocation } from "./sceneEventLocation.js";
import { spiralHitTest, pointHitTest, type ResolvedHit } from "./spiralHitTest.js";

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
  readonly hit: ResolvedHit | undefined;
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
  private lastPath: ReadonlyArray<LeafPickEntry> = [];
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

  /** Canvas backing-store size in device pixels — used as `viewportSize` when building the SceneEventLocation. Set on `attach`. */
  private canvasSize: () => V2i = () => V2i.zero;

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
    this.canvasSize = () => new V2i(canvas.width, canvas.height);
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
        const hit = region !== undefined ? this.resolve(region, devX, devY) : undefined;
        this.dispatch(ev, kind, cssX, cssY, hit, rect, sx, sy, region, devX, devY);
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
        const hit = region !== undefined ? this.resolve(region, devX, devY) : undefined;
        this.dispatchWheel(ev, cssX, cssY, hit, rect, sx, sy);
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
      if (scope.noEvents !== undefined && AVal.force(scope.noEvents)) return;
      const sceneEv = new SceneEvent({
        kind,
        location: this.buildEmptyLocation(scope),
        pickId: scope.pickId,
        raw: ev,
        key: ev.key, code: ev.code, repeat: ev.repeat,
        ctrl: ev.ctrlKey, shift: ev.shiftKey, alt: ev.altKey, meta: ev.metaKey,
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
            kind: "OnBlur",
            location: this.buildEmptyLocation(oldScope),
            pickId: oldScope.pickId,
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
            kind: "OnFocus",
            location: this.buildEmptyLocation(newScope),
            pickId: newScope.pickId,
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
    hit: ResolvedHit | undefined,
    _rect: DOMRect,
    _sx: number,
    _sy: number,
  ): void {
    if (hit === undefined) return;
    const { scope, viewPos, viewNormal, partIndex, isPixel } = hit;
    const location = this.buildLocation(
      scope, cssX, cssY,
      viewPos,
      viewNormal,
      partIndex,
    );
    const sceneEv = new SceneEvent({
      kind: "OnWheel",
      location,
      pickId: scope.pickId,
      // The pickId's registered mode determines modeB (Mode-B → true).
      modeB: !isPixel ? false : (this.registry.modeOf(scope.pickId) === "B"),
      raw: ev,
      deltaX: ev.deltaX,
      deltaY: ev.deltaY,
      deltaZ: ev.deltaZ,
      deltaMode: (ev.deltaMode as 0 | 1 | 2),
      ctrl: ev.ctrlKey, shift: ev.shiftKey, alt: ev.altKey, meta: ev.metaKey,
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

  /**
   * Single-call merge of the pixel + BVH candidates that mirrors F#'s
   * SceneHandler step-for-step. Replaces the previous separate
   * `spiralResolve` + `bvhFallthrough` paths.
   */
  private resolve(region: PickRegion, centerX: number, centerY: number): ResolvedHit | undefined {
    return spiralHitTest(
      region,
      { devX: centerX, devY: centerY },
      this.registry,
      this._getView(),
      this._getProj(),
      this.canvasSize(),
    );
  }

  // --- core dispatch -------------------------------------------------------

  private dispatch(
    ev: PointerEvent,
    kind: SceneEventKind,
    cssX: number,
    cssY: number,
    hit: ResolvedHit | undefined,
    rect: DOMRect,
    sx: number,
    sy: number,
    region?: PickRegion,
    devX?: number,
    devY?: number,
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
      // AVal.force OK: pointer event handler — see file-top policy.
      if (captured.noEvents !== undefined && AVal.force(captured.noEvents)) {
        // The captured scope flipped to noEvents while held; drop the
        // capture so future events fall back to spiral resolution.
        this.capturedScopes.delete(pointerId);
        return;
      }
      // Route to captured scope; do NOT update lastHit/lastPath. F#
      // `SceneHandler.fs:1700–` skips lastOver updates while a
      // pointer is captured, then re-fires move on release.
      //
      // For the dispatched `viewPos` / `worldPos`: query the cursor's
      // current pixel + BVH ray (no snap) so handlers see the world
      // position UNDER the cursor regardless of where the captured
      // geometry is. Strictly more expressive than nothing — same
      // query the spiral runs at offset (0,0).
      let viewPos: V3d | undefined;
      let viewNormal: V3d | undefined;
      if (region !== undefined && devX !== undefined && devY !== undefined && this.canvasSize !== undefined) {
        const point = pointHitTest(
          region, { devX, devY }, this.registry,
          AVal.force(captured.view), AVal.force(captured.proj), this.canvasSize(),
        );
        if (point !== undefined) {
          viewPos = point.viewPos;
          viewNormal = point.viewNormal;
        }
      }
      const modeB = this.registry.modeOf(captured.pickId) === "B";
      const sceneEv = this.makeEvent(kind, ev, cssX, cssY, captured, captured.pickId, modeB, viewPos, viewNormal);
      this.runCaptureBubble(captured.handlers, sceneEv);
      return;
    }

    const hitScope = hit?.scope;
    const hitId = hitScope?.pickId ?? 0;
    const newPath = hitScope?.handlers ?? [];
    void rect; void sx; void sy;

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

      if (hitScope !== undefined && hit !== undefined) {
        const modeB = !hit.isPixel ? false : (this.registry.modeOf(hitScope.pickId) === "B");
        const enterEv = this.makeEvent("OnPointerEnter", ev, cssX, cssY, hitScope, hitScope.pickId, modeB, hit.viewPos, hit.viewNormal, hit.partIndex);
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

    const viewPos = hit?.viewPos;
    const viewNormal = hit?.viewNormal;
    const partIndex = hit?.partIndex ?? 0;
    const modeB = hit !== undefined && hit.isPixel
      ? this.registry.modeOf(hitScope.pickId) === "B"
      : false;
    const sceneEv = this.makeEvent(kind, ev, cssX, cssY, hitScope, hitScope.pickId, modeB, viewPos, viewNormal, partIndex);
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
    const location = this.buildLocation(scope, cssX, cssY, V3d.zero, V3d.zero, 0);
    const sceneEv = new SceneEvent({
      kind,
      location,
      pickId: scope.pickId,
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
  private runCaptureBubble(path: ReadonlyArray<LeafPickEntry>, ev: SceneEvent): void {
    // Capture phase: outer → inner
    for (let i = 0; i < path.length; i++) {
      const h = path[i]!.handlers.capture?.[ev.kind];
      if (h !== undefined) {
        const local = AVal.force(path[i]!.local2World);
        const localEv = ev.transformed(local);
        let r: boolean | void;
        try { r = h(localEv); } catch (err) { console.error(`[PickDispatcher] capture ${ev.kind} threw:`, err); continue; }
        if (r === false || ev.propagationStopped) return;
      }
    }
    // Bubble phase: inner → outer
    for (let i = path.length - 1; i >= 0; i--) {
      const h = path[i]!.handlers.bubble?.[ev.kind];
      if (h !== undefined) {
        const local = AVal.force(path[i]!.local2World);
        const localEv = ev.transformed(local);
        let r: boolean | void;
        try { r = h(localEv); } catch (err) { console.error(`[PickDispatcher] bubble ${ev.kind} threw:`, err); continue; }
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
    path: ReadonlyArray<LeafPickEntry>,
    exclude: ReadonlyArray<LeafPickEntry>,
    ev: SceneEvent,
  ): void {
    for (let i = path.length - 1; i >= exclude.length; i--) {
      this.fireBoth(path[i]!, ev);
    }
  }

  private runDownAll(
    path: ReadonlyArray<LeafPickEntry>,
    exclude: ReadonlyArray<LeafPickEntry>,
    ev: SceneEvent,
  ): void {
    for (let i = exclude.length; i < path.length; i++) {
      this.fireBoth(path[i]!, ev);
    }
  }

  private fireBoth(entry: LeafPickEntry, ev: SceneEvent): void {
    const cap = entry.handlers.capture?.[ev.kind];
    const bub = entry.handlers.bubble?.[ev.kind];
    if (cap === undefined && bub === undefined) return;
    const local = AVal.force(entry.local2World);
    const localEv = ev.transformed(local);
    if (cap !== undefined) {
      try { cap(localEv); } catch (err) { console.error(`[PickDispatcher] capture ${ev.kind} threw:`, err); }
    }
    if (bub !== undefined) {
      try { bub(localEv); } catch (err) { console.error(`[PickDispatcher] bubble ${ev.kind} threw:`, err); }
    }
  }

  // --- helpers -------------------------------------------------------------

  /**
   * Build a `SceneEventLocation` for a hit-bearing event. `viewPos` /
   * `viewNormal` come from the spiral hit (`viewPosFor`) or the BVH
   * fall-through; `partIndex` from the decoded pick slot's slot3
   * channel (Mode-A only — BVH / Mode-B / synthetic hits use 0).
   *
   * Why `AVal.force` here: dispatcher runs in a non-adaptive event
   * handler — the forces don't enter any reader's dependency set.
   */
  private buildLocation(
    scope: LeafPickScope | undefined,
    cssX: number,
    cssY: number,
    viewPos: V3d,
    viewNormal: V3d,
    partIndex: number,
  ): SceneEventLocation {
    const view = scope !== undefined ? AVal.force(scope.view) : Trafo3d.identity;
    const proj = scope !== undefined ? AVal.force(scope.proj) : Trafo3d.identity;
    const model = scope?.model ?? AVal.constant(Trafo3d.identity);
    // Convert CSS-px cursor coords to device pixels using the live
    // canvas rect — keeps `pixel/viewportSize` in the same unit space
    // for the NDC math inside SceneEventLocation.
    const rect = this.getCanvasRect();
    const size = this.canvasSize();
    const sx = rect.width  > 0 ? size.x / rect.width  : 1;
    const sy = rect.height > 0 ? size.y / rect.height : 1;
    const pixel = new V2d(cssX * sx, cssY * sy);
    return new SceneEventLocation(
      model,
      Trafo3d.identity,
      view,
      proj,
      pixel,
      size,
      viewPos,
      viewNormal,
      partIndex,
    );
  }

  /**
   * Build a non-spatial location used for keyboard / focus / "no
   * hit" synthetic events. View/proj come from the scope when one
   * exists (so `worldPos` etc. are still well-typed), and viewPos /
   * viewNormal are zero. Handlers checking `e.depth` on a key event
   * get `0` — they shouldn't be reading it. (See "Why" in step 4 of
   * the SceneEventLocation refactor brief.)
   */
  private buildEmptyLocation(scope: LeafPickScope | undefined): SceneEventLocation {
    return this.buildLocation(scope, 0, 0, V3d.zero, V3d.zero, 0);
  }

  private makeEvent(
    kind: SceneEventKind,
    ev: PointerEvent,
    devX: number,
    devY: number,
    scope: LeafPickScope,
    pickId: number,
    modeB: boolean,
    viewPos: V3d | undefined,
    viewNormal?: V3d,
    partIndex: number = 0,
  ): SceneEvent {
    const vp = viewPos ?? V3d.zero;
    const vn = viewNormal ?? V3d.zero;
    const location = this.buildLocation(scope, devX, devY, vp, vn, partIndex);
    return new SceneEvent({
      kind,
      location,
      pickId,
      modeB,
      ...(ev.button !== undefined ? { button: ev.button } : {}),
      ...(ev.buttons !== undefined ? { buttons: ev.buttons } : {}),
      pointerId: ev.pointerId ?? 0,
      pointerType: ev.pointerType ?? "",
      ctrl: ev.ctrlKey, shift: ev.shiftKey, alt: ev.altKey, meta: ev.metaKey,
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

    const gestureLocation = (): SceneEventLocation =>
      this.buildLocation(scope, cx, cy, V3d.zero, V3d.zero, 0);
    if (this.gestureLastDist !== undefined && this.gestureLastDist > 0) {
      const scale = newDist / this.gestureLastDist;
      if (Math.abs(scale - 1) > 1e-6) {
        const ev = new SceneEvent({
          kind: "OnPinch",
          location: gestureLocation(),
          pickId: scope.pickId,
          pointerId: raw.pointerId ?? 0,
          pointerType: raw.pointerType ?? "",
          raw,
          pinchScale: scale,
          pinchCenter: new V2d(cx, cy),
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
          location: gestureLocation(),
          pickId: scope.pickId,
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
          location: gestureLocation(),
          pickId: scope.pickId,
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
        location: this.buildLocation(scope, cssX, cssY, V3d.zero, V3d.zero, 0),
        pickId: scope.pickId,
        pointerId: raw.pointerId ?? 0,
        pointerType: raw.pointerType ?? "",
        raw,
        scope, dispatch: this,
      });
      this.runCaptureBubble(scope.handlers, ev);
    }, this.tHoverDelay);
  }
}

function sharedPrefixLength(
  a: ReadonlyArray<LeafPickEntry>,
  b: ReadonlyArray<LeafPickEntry>,
): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  // Reference equality on the inner `handlers` payload (the
  // identity carrier from `<Sg On=…>` — `pushHandlers` allocates a
  // fresh entry wrapper per traversal but copies the same handlers
  // ref, so two leaves under the same On scope share the same
  // `entry.handlers` identity).
  while (i < n && a[i]!.handlers === b[i]!.handlers) i++;
  return i;
}
