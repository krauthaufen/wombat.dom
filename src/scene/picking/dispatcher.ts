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

import { AVal } from "@aardworx/wombat.adaptive";
import { Trafo3d, V4d } from "@aardworx/wombat.base";

import type { LeafPickScope, PickRegistry } from "./registry.js";
import { decodePick, readSlotsAt, type DecodedPick, type PickRegion } from "./readback.js";
import {
  SceneEvent,
  type SceneEventDispatch,
  type SceneEventKind,
  type SceneEventViewPos,
} from "./sceneEvent.js";
import { SNAP_OFFSETS, SNAP_RADIUS_MAX } from "./snapOffsets.js";

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

export class PickDispatcher implements SceneEventDispatch {
  private lastHit: number = 0;
  private lastPath: ReadonlyArray<import("../sg.js").EventHandlers> = [];
  private seq: number = 0;
  private lastSettledSeq: number = 0;

  private readonly capturedScopes: Map<number, LeafPickScope> = new Map();
  private lastMove: LastMoveInfo | undefined;

  constructor(
    private readonly registry: PickRegistry,
    /** Kept for future rev when global view/proj override per-scope's. */
    private readonly _getView: () => Trafo3d,
    private readonly _getProj: () => Trafo3d,
    private readonly getCanvasRect: () => DOMRect,
  ) {
    void this._getView; void this._getProj;
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
        this.dispatch(ev, kind, cssX, cssY, hit, rect, sx, sy);
      });
    };

    const onDown   = (e: PointerEvent): void => handle(e, "OnPointerDown");
    const onUp     = (e: PointerEvent): void => handle(e, "OnPointerUp");
    const onMove   = (e: PointerEvent): void => handle(e, "OnPointerMove");
    const onClick  = (e: PointerEvent): void => handle(e, "OnClick");
    const onEnter  = (e: PointerEvent): void => handle(e, "OnPointerEnter");
    const onLeave  = (e: PointerEvent): void => {
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

    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("click", onClick as unknown as EventListener);
    canvas.addEventListener("pointerenter", onEnter);
    canvas.addEventListener("pointerleave", onLeave);

    return () => {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("click", onClick as unknown as EventListener);
      canvas.removeEventListener("pointerenter", onEnter);
      canvas.removeEventListener("pointerleave", onLeave);
    };
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
  ): void {
    const pointerId = ev.pointerId ?? 0;

    // Always record the latest cursor position for synthetic-move
    // replay on capture release. Track `hit` from the spiral so we
    // can replay against the geometry under the cursor at release
    // time (close enough — the framebuffer may have changed, but we
    // don't have a fresh region read in hand).
    this.lastMove = { raw: ev, cssX, cssY, rect, sx, sy, hit };

    const captured = this.capturedScopes.get(pointerId);
    if (captured !== undefined) {
      // Route to captured scope; do NOT update lastHit/lastPath. F#
      // `SceneHandler.fs:1700–` skips lastOver updates while a
      // pointer is captured, then re-fires move on release.
      const viewPos = hit !== undefined ? this.viewPosFor(captured, hit, rect, sx, sy) : undefined;
      const sceneEv = this.makeEvent(kind, ev, cssX, cssY, captured, captured.pickId, hit?.decoded.modeB ?? false, viewPos);
      this.runCaptureBubble(captured.handlers, sceneEv);
      return;
    }

    const hitScope = hit?.scope;
    const hitId = hitScope?.pickId ?? 0;
    const newPath = hitScope?.handlers ?? [];

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
        const viewPos = this.viewPosFor(hitScope, hit, rect, sx, sy);
        const enterEv = this.makeEvent("OnPointerEnter", ev, cssX, cssY, hitScope, hitScope.pickId, hit.decoded.modeB, viewPos);
        this.runDownAll(newPath, newPath.slice(0, prefix), enterEv);
      }

      this.lastHit = hitId;
      this.lastPath = newPath;
    }

    if (hitScope === undefined || hit === undefined) return;

    const viewPos = this.viewPosFor(hitScope, hit, rect, sx, sy);
    const sceneEv = this.makeEvent(kind, ev, cssX, cssY, hitScope, hitScope.pickId, hit.decoded.modeB, viewPos);
    this.runCaptureBubble(hitScope.handlers, sceneEv);
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

  // --- helpers -------------------------------------------------------------

  private viewPosFor(
    scope: LeafPickScope,
    hit: SpiralHit,
    rect: DOMRect,
    sx: number,
    sy: number,
  ): SceneEventViewPos | undefined {
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
    return { x: viewSpace.x / w, y: viewSpace.y / w, z: viewSpace.z / w };
  }

  private makeEvent(
    kind: SceneEventKind,
    ev: PointerEvent,
    cssX: number,
    cssY: number,
    scope: LeafPickScope,
    pickId: number,
    modeB: boolean,
    viewPos: SceneEventViewPos | undefined,
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
      pointerId: ev.pointerId ?? 0,
      pointerType: ev.pointerType ?? "",
      raw: ev,
      scope,
      dispatch: this,
    });
  }
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
