// Per-leaf pick registry — allocates `PickId`s and stores the slice
// of `TraversalState` that hit-dispatch will need (event handlers,
// cursor, pick-through, active gating, camera matrices).
//
// Mirrors the F# `Aardvark.Dom.SceneHandler.acquireId` /
// `pickBuffer` capture, minus the recycling. See file footer.

import {
  AVal,
  avalAddCallback,
  cval,
  transact,
  type aval,
  type IDisposable,
} from "@aardworx/wombat.adaptive";
import { Bvh, type IIntersectable, type Trafo3d } from "@aardworx/wombat.base";

import type { EventHandlers } from "../sg.js";

/**
 * A pick identifier. We use a uint that fits the f32 mantissa
 * (`< 2^24`) so the rgba32f pick target can carry it without
 * precision loss when it's MSAA-resolved by averaging. `0` is
 * reserved as "no hit" — the framebuffer is cleared to 0.
 */
export type PickId = number;

/** Inclusive upper bound — pickIds beyond this can lose precision in f32. */
export const PICK_ID_MAX: PickId = (1 << 24) - 1;

/**
 * One entry in the leaf's path-of-scopes. `handlers` is the
 * `EventHandlers` value from the surrounding `<Sg On=…>` scope —
 * carried by reference, so two leaves sharing the same On scope
 * also share the same entry's `handlers` identity (used for the
 * dispatcher's prefix diff). `local2World` is the model trafo
 * accumulated UP TO AND INCLUDING this scope (snapshotted from
 * `state.model` when the scope was pushed in TraversalState).
 *
 * The dispatcher applies `local2World.inverse()` (via
 * `SceneEvent.transformed`) when invoking the handlers, so each
 * level's handler sees `e.position` / `e.normal` / `e.pickRay` in
 * its own local frame. F# parity: `event.Transformed(model)` —
 * see `Aardvark.Dom/SceneGraph/TraversalState.fs runCapture/runBubble`.
 */
export interface LeafPickEntry {
  readonly handlers: EventHandlers;
  readonly local2World: aval<Trafo3d>;
}

/**
 * The scope information captured per leaf at compile time. The
 * picking dispatcher (next milestone) reads this to:
 *   - run the right event handlers when a pixel hit lands here,
 *   - apply the cursor while hovered,
 *   - decide whether to skip this leaf for click-through,
 *   - gate dispatch on `active`,
 *   - unproject the hit point using `view` / `proj` / `model`.
 */
export interface LeafPickScope {
  readonly pickId: PickId;
  readonly handlers: ReadonlyArray<LeafPickEntry>;
  readonly cursor: string | aval<string> | undefined;
  readonly pickThrough: boolean;
  readonly active: aval<boolean>;
  readonly view: aval<Trafo3d>;
  readonly proj: aval<Trafo3d>;
  readonly model: aval<Trafo3d>;
  /**
   * Pixel-snap-radius (device pixels) for the spiral hit-test.
   * Clamped to `[0, SNAP_RADIUS_MAX]` at dispatch time. Default
   * (when no `<Sg PixelSnapRadius>` scope was entered) is 1.
   */
  readonly pixelSnapRadius: aval<number>;
  /**
   * Optional per-scope intersectable (world-space). Used by the
   * dispatcher to build a BVH and ray-fall-through past pickThrough
   * scopes when the pixel-pick lands on a "transparent" hit.
   */
  readonly intersectable?: aval<IIntersectable>;
  /**
   * When true, the dispatcher's BVH ray fall-through is suppressed
   * for hits on this scope: the pixel-pick result is final.
   */
  readonly forcePixelPicking?: aval<boolean>;
  /** When true, this scope can receive keyboard / focus events. */
  readonly canFocus?: aval<boolean>;
}

/**
 * Allocator + lookup table for leaf pick scopes. One instance per
 * scene compilation (typically owned by a `<RenderControl>`).
 *
 * Why no recycling: F# `SceneHandler` recycles freed ids, which is
 * worth the bookkeeping there because long-lived scenes churn leaves.
 * For v1 we lean on the 24-bit space (16M ids) and just rebuild the
 * registry on full recompiles. Revisit if a long-running app
 * actually exhausts the range — the next signal would be `acquire`
 * needing to fail or wrap, neither of which is correct silently.
 */
export type PickMode = "A" | "B";

export class PickRegistry {
  private next: PickId = 1;
  private readonly entries = new Map<PickId, LeafPickScope>();
  // Mode the leaf was registered with — Mode-A writes `+pickId`,
  // Mode-B writes `-pickId`. The spiral hit-test rejects any pixel
  // whose sign disagrees with the registered mode (MSAA averaging
  // at silhouettes can otherwise surface a valid pickId with the
  // wrong layout).
  private readonly modes = new Map<PickId, PickMode>();

  // BVH cache invalidation: `_dirtyVersion` bumps on every `acquire`,
  // on `clear`, AND every time any registered scope's `intersectable`
  // aval ticks (callback below). `buildBvh` rebuilds (and stamps
  // `_bvhVersion = _dirtyVersion`) only when the two diverge.
  //
  // Why a callback rather than incremental rebuild: every acquire
  // bumps `_dirtyVersion`, every intersectable tick bumps
  // `_dirtyVersion`, and the next `buildBvh()` does a full rebuild.
  // This is intentionally coarse — the registry today is rebuilt
  // wholesale per scene compile, so the leaf count is bounded; an
  // incremental refit would only matter for streaming geometry
  // changes inside a single compile, which we don't have today.
  private _dirtyVersion: number = 0;
  private _bvh: Bvh<PickId, IIntersectable> | undefined = undefined;
  private _bvhVersion: number = -1;
  // Per-pickId subscription on `intersectable`. Disposed on
  // `clear` (the only path that invalidates a pickId today —
  // see "Why no recycling" above).
  private readonly _intersectableSubs = new Map<PickId, IDisposable>();

  // Phase 5 — focus model. `focusedPickId` is observable via aval;
  // dispatcher subscribes to drive OnFocus / OnBlur.
  private readonly _focused = cval<PickId | undefined>(undefined);
  /** Currently focused scope's PickId, or `undefined`. Observable. */
  readonly focusedPickId: aval<PickId | undefined> = this._focused;

  acquire(scope: Omit<LeafPickScope, "pickId">, mode: PickMode = "A"): PickId {
    const pickId = this.next++;
    const full: LeafPickScope = { pickId, ...scope };
    this.entries.set(pickId, full);
    this.modes.set(pickId, mode);
    this._dirtyVersion++;
    if (scope.intersectable !== undefined && !scope.intersectable.isConstant) {
      // Tick the dirty counter on every change so `buildBvh()` will
      // rebuild on its next call. Constants never tick, so we skip
      // the subscription for them (avoids ConstantObject's no-op
      // marking-callback path). The version was already bumped by
      // the acquire above, so the first build picks up the current
      // bbox without needing a primer call here.
      const sub = avalAddCallback(scope.intersectable, () => {
        this._dirtyVersion++;
        this._bvh = undefined;
      });
      this._intersectableSubs.set(pickId, sub);
    }
    return pickId;
  }

  lookup(id: PickId): LeafPickScope | undefined {
    return this.entries.get(id);
  }

  /**
   * Mode the given pickId was acquired with — `"A"` means Mode-A
   * (slot0 written as `+pickId`), `"B"` means Mode-B (`-pickId`).
   * `undefined` for unknown ids.
   */
  modeOf(id: PickId): PickMode | undefined {
    return this.modes.get(id);
  }

  clear(): void {
    for (const sub of this._intersectableSubs.values()) sub.dispose();
    this._intersectableSubs.clear();
    this.entries.clear();
    this.modes.clear();
    this.next = 1;
    this._dirtyVersion++;
    this._bvh = undefined;
    this._bvhVersion = -1;
    transact(() => { this._focused.value = undefined; });
  }

  size(): number {
    return this.entries.size;
  }

  /**
   * Move focus to the given scope. Validates that the target's
   * `canFocus = true`; silently rejects otherwise. Pass `undefined`
   * (or call {@link clearFocus}) to drop focus.
   */
  setFocus(pickId: PickId | undefined): void {
    if (pickId === undefined) {
      transact(() => { this._focused.value = undefined; });
      return;
    }
    const scope = this.entries.get(pickId);
    if (scope === undefined) return;
    // Why force here: `setFocus` is invoked from event-handler code
    // paths (click, programmatic), not from inside an adaptive
    // computation; reading the latest value point-in-time is fine.
    if (scope.canFocus === undefined || !AVal.force(scope.canFocus)) return;
    transact(() => { this._focused.value = pickId; });
  }

  /** Drop focus. Equivalent to `setFocus(undefined)`. */
  clearFocus(): void {
    transact(() => { this._focused.value = undefined; });
  }

  /**
   * Build (or return the cached) world-space BVH over every scope
   * that has an `intersectable`. Returns `undefined` when no scope
   * has one — callers can short-circuit ray fall-through. Forces
   * each `aval<IIntersectable>` to extract its `boundingBox`.
   *
   * Tradeoff (no incremental rebuild): a single `intersectable` tick
   * invalidates the entire BVH and the next call rebuilds it from
   * scratch. The bbox stored in the BVH is in `intersectable`-local
   * space at acquire/tick time and does NOT track per-scope
   * `model` trafo changes — `spiralHitTest` / `pointHitTest` still
   * fetch `AVal.force(scope.model)` at query time and re-test
   * intersection against every cull-set entry, so a stale bbox only
   * affects culling efficiency, not correctness.
   */
  buildBvh(): Bvh<PickId, IIntersectable> | undefined {
    if (this._bvhVersion === this._dirtyVersion) return this._bvh;
    const items: { key: PickId; box: import("@aardworx/wombat.base").Box3d; value: IIntersectable }[] = [];
    for (const [pickId, scope] of this.entries) {
      if (scope.intersectable === undefined) continue;
      const it = AVal.force(scope.intersectable);
      items.push({ key: pickId, box: it.boundingBox, value: it });
    }
    this._bvh = items.length === 0 ? undefined : Bvh.build(items);
    this._bvhVersion = this._dirtyVersion;
    return this._bvh;
  }
}
