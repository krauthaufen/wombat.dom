// PickDispatcher — wires pointer events on a canvas to handler
// dispatch via the pick registry. Now does Aardvark.Dom-style
// pixel-snap: a 33×33 region readback around the cursor, then a
// spiral walk over `SNAP_OFFSETS` in d²-ascending order. The first
// offset whose pickId belongs to a registered, active, non-pick-
// through scope AND falls within that scope's own
// `pixelSnapRadius²` wins.
//
// Why spiral walk: Aardvark.Dom's `SceneHandler.fs:1700–1720`. In
// practice this lets touch UIs widen the cursor's effective pick
// area to ~16 px without the snap ever overshooting the pickee's
// own preferred slop (so a tiny 1-px pickee won't capture the
// cursor from 16 px away).
//
// `pickThrough` simplification: F# only falls through for BVH-
// (intersector-) backed scopes; for pixel hits it just emits a
// log warning and still selects the winner (`SceneHandler.fs:1814–
// 1817`). We have no BVH path, so we treat pickThrough as "skip the
// whole spiral hit" — no fall-through to the next offset.

import { AVal } from "@aardworx/wombat.adaptive";
import { Trafo3d, V4d } from "@aardworx/wombat.base";

import type { LeafPickScope, PickRegistry } from "./registry.js";
import { decodePick, readSlotsAt, type DecodedPick, type PickRegion } from "./readback.js";
import type { SceneEvent, SceneEventKind, SceneEventViewPos } from "./sceneEvent.js";
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

export class PickDispatcher {
  private lastHit: number = 0;
  private seq: number = 0;
  private lastSettledSeq: number = 0;

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
          this.fire(scope, {
            kind: "OnPointerLeave",
            clientX: cssX, clientY: cssY,
            pickId: scope.pickId, modeB: false,
            ...(e.button !== undefined ? { button: e.button } : {}),
            buttons: e.buttons,
            raw: e,
          });
        }
        this.lastHit = 0;
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
    const hitScope = hit?.scope;
    const hitId = hitScope?.pickId ?? 0;

    if (kind === "OnPointerMove" && hitId !== this.lastHit) {
      const oldScope = this.lastHit !== 0 ? this.registry.lookup(this.lastHit) : undefined;
      if (oldScope !== undefined && !oldScope.pickThrough && AVal.force(oldScope.active)) {
        this.fire(oldScope, this.makeEvent("OnPointerLeave", ev, cssX, cssY, oldScope.pickId, false, undefined));
      }
      if (hitScope !== undefined && hit !== undefined) {
        const viewPos = this.viewPosFor(hitScope, hit, rect, sx, sy);
        this.fire(hitScope, this.makeEvent("OnPointerEnter", ev, cssX, cssY, hitScope.pickId, hit.decoded.modeB, viewPos));
      }
      this.lastHit = hitId;
    }

    if (hitScope === undefined || hit === undefined) return;

    const viewPos = this.viewPosFor(hitScope, hit, rect, sx, sy);
    this.fire(hitScope, this.makeEvent(kind, ev, cssX, cssY, hitScope.pickId, hit.decoded.modeB, viewPos));
  }

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
    pickId: number,
    modeB: boolean,
    viewPos: SceneEventViewPos | undefined,
  ): SceneEvent {
    return {
      kind,
      clientX: cssX,
      clientY: cssY,
      pickId,
      modeB,
      ...(ev.button !== undefined ? { button: ev.button } : {}),
      buttons: ev.buttons,
      ...(viewPos !== undefined ? { viewPos } : {}),
      raw: ev,
    };
  }

  private fire(scope: LeafPickScope, event: SceneEvent): void {
    for (const map of scope.handlers) {
      const fn = map[event.kind];
      if (typeof fn === "function") {
        try {
          fn(event);
        } catch (err) {
          console.error(`[PickDispatcher] handler for ${event.kind} threw:`, err);
        }
      }
    }
  }
}

// Suppress unused-import noise — these are kept for future
// integrations (e.g. a getter-based proj override path).
void Trafo3d;
