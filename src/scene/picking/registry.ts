// Per-leaf pick registry — allocates `PickId`s and stores the slice
// of `TraversalState` that hit-dispatch will need (event handlers,
// cursor, pick-through, active gating, camera matrices).
//
// Mirrors the F# `Aardvark.Dom.SceneHandler.acquireId` /
// `pickBuffer` capture, minus the recycling. See file footer.

import { AVal, type aval } from "@aardworx/wombat.adaptive";
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
  readonly handlers: ReadonlyArray<EventHandlers>;
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
export class PickRegistry {
  private next: PickId = 1;
  private readonly entries = new Map<PickId, LeafPickScope>();

  // BVH cache invalidation: every `acquire` bumps `_dirtyVersion`;
  // `buildBvh` rebuilds (and stamps `_bvhVersion = _dirtyVersion`)
  // only when the two diverge. This is intentionally coarse — a full
  // rebuild on any acquire — because the registry today is rebuilt
  // wholesale per scene compile, so churn is bounded by leaf count.
  private _dirtyVersion: number = 0;
  private _bvh: Bvh<PickId, IIntersectable> | undefined = undefined;
  private _bvhVersion: number = -1;

  acquire(scope: Omit<LeafPickScope, "pickId">): PickId {
    const pickId = this.next++;
    const full: LeafPickScope = { pickId, ...scope };
    this.entries.set(pickId, full);
    this._dirtyVersion++;
    return pickId;
  }

  lookup(id: PickId): LeafPickScope | undefined {
    return this.entries.get(id);
  }

  clear(): void {
    this.entries.clear();
    this.next = 1;
    this._dirtyVersion++;
    this._bvh = undefined;
    this._bvhVersion = -1;
  }

  size(): number {
    return this.entries.size;
  }

  /**
   * Build (or return the cached) world-space BVH over every scope
   * that has an `intersectable`. Returns `undefined` when no scope
   * has one — callers can short-circuit ray fall-through. Forces
   * each `aval<IIntersectable>` to extract its `boundingBox`.
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
