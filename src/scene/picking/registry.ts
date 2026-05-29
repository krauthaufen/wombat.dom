// Per-leaf pick registry — allocates `PickId`s and stores the slice
// of `TraversalState` that hit-dispatch will need (event handlers,
// cursor, pick-through, active gating, camera matrices).
//
// Mirrors the F# `Aardvark.Dom.SceneHandler.acquireId` /
// `pickBuffer` capture, minus the recycling. See file footer.
//
// AVal.force policy (this file): `setFocus` and the dispatcher's
// reads of `bvhAval` happen in event-handler / programmatic-API
// context — "now" is the user's tick. `acquire`, `clear`, and the
// `bvhAval` AVal.custom body all run during compileScene or as
// reactive deltas (no force needed).

import {
  AVal,
  ChangeableHashSet,
  cset,
  cval,
  transact,
  type aset,
  type aval,
} from "@aardworx/wombat.adaptive";
import {
  Box3d,
  Bvh,
  V3d,
  type IIntersectable,
  type Trafo3d,
} from "@aardworx/wombat.base";

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
   * Pick path chosen at compile-scene time. Pixel and BVH are
   * complementary: a leaf takes ONE path so the picker doesn't
   * pay both costs.
   *
   *   - "pixel" — pick-chain rendered to pickFb; the dispatcher
   *               resolves hits via the spiral readback. BVH-add is
   *               skipped (the scene-wide BVH stays small).
   *   - "bvh"   — pick-chain rendered AND the intersectable joins
   *               the BVH. `sceneQuery.intersect` and the
   *               dispatcher's pickThrough fall-through both work.
   *
   * Default is "bvh" (matching pre-change behaviour) so existing
   * `sceneQuery` callers stay green. Dense scenes that don't need
   * ray-cast queries should set `Sg.ForcePixelPicking={true}` on a
   * surrounding scope to opt into the cheaper pixel path.
   */
  readonly pickPath?: "pixel" | "bvh";
  /** When true, this scope can receive keyboard / focus events. */
  readonly canFocus?: aval<boolean>;
  /**
   * When true, the dispatcher SKIPS this scope's events entirely —
   * the leaf was registered (so its pickId is allocated and the BVH
   * can still see it for fall-through), but no handlers fire and no
   * cursor/focus interaction happens. Only set when the compile-time
   * `state.noEvents` was non-constant — constant-true leaves don't
   * register at all (see `compile.ts` registration policy).
   */
  readonly noEvents?: aval<boolean>;
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

/**
 * Item stored in the BVH — pickId-keyed scope plus the (now-applied)
 * trafo and intersectable. The trafo is snapshotted when the BVH
 * delta was applied; the BVH bbox is in WORLD space (intersectable's
 * local bbox transformed by `trafo`). `spiralHitTest` reads the
 * scope's *current* `model` aval at hit time and re-tests
 * intersection in local space — small staleness vs. trafo ticks
 * shows up only as an inflated bbox during the (very brief) gap
 * between an aval tick and the next `bvhAval` re-evaluation, never
 * as missed hits or wrong intersections.
 */
export interface BvhEntry {
  readonly scope: LeafPickScope;
  readonly intersectable: IIntersectable;
  readonly trafo: Trafo3d;
}

/**
 * Pick object as stored in the aset — mirrors F# `PickObject`. Each
 * `acquire` of an intersectable-bearing scope adds one of these to
 * `_pickObjects`; `clear` empties the set. The aset's `mapA` chains
 * intersectable-aval AND trafo-aval ticks into the BVH delta stream.
 */
interface PickObject {
  readonly scope: LeafPickScope;
  readonly intersectable: aval<IIntersectable>;
  readonly trafo: aval<Trafo3d>;
}

/**
 * Observer the picker's GPU metadata buffer implements so it can mirror
 * scope register/deregister without the registry depending on a
 * `GPUDevice`. {@link PickMetadata} satisfies this shape.
 */
export interface PickRegistryObserver {
  register(scope: LeafPickScope, mode: PickMode): void;
  deregister(id: PickId): void;
  clear(): void;
}

export class PickRegistry {
  private next: PickId = 1;
  private observer: PickRegistryObserver | undefined;
  private readonly entries = new Map<PickId, LeafPickScope>();
  // Mode the leaf was registered with — Mode-A writes `+pickId`,
  // Mode-B writes `-pickId`. The spiral hit-test rejects any pixel
  // whose sign disagrees with the registered mode (MSAA averaging
  // at silhouettes can otherwise surface a valid pickId with the
  // wrong layout).
  private readonly modes = new Map<PickId, PickMode>();
  // Side map for the BVH path so `release(pickId)` can drop the exact
  // PickObject reference from `_pickObjects`. Without this we'd have
  // no way to identify which entry to remove (the underlying cset is
  // identity-keyed, not pickId-keyed).
  private readonly bvhEntries = new Map<PickId, PickObject>();

  // ------------------------------------------------------------------
  // Reactive BVH (mirrors Aardvark.Dom SceneHandler.fs:1444-1468).
  //
  // Shape:
  //   _pickObjects : cset<PickObject>           — driven by acquire/clear
  //   transformed  : aset<BvhEntry>             — _pickObjects.mapA chain
  //   bvhAval      : aval<Bvh<PickId, BvhEntry>> — incremental, AVal.custom
  //
  // The aset's `mapA` projects each PickObject through its
  // intersectable AND trafo avals (via two nested .map's). On any
  // tick of either, the projected BvhEntry value's identity changes,
  // which the inner reader surfaces as a Rem(old) + Add(new) delta
  // pair. The AVal.custom body folds those deltas into a persistent
  // BVH (Rem then Add, mirroring F#'s ordering at line 1455-1466).
  // The bbox stored in the BVH is WORLD-space (intersectable's local
  // bbox transformed through the current trafo).
  // ------------------------------------------------------------------
  private readonly _pickObjects: ChangeableHashSet<PickObject> = cset<PickObject>();
  /** Underlying aset of pick-objects; exposed for tests. */
  get pickObjects(): aset<PickObject> { return this._pickObjects; }
  /**
   * Reactive world-space BVH over every scope that has an
   * `intersectable`. The dispatcher force-reads this in pointer-event
   * context (legal — see file-top policy). The returned BVH is empty
   * when no intersectable scopes are registered.
   */
  readonly bvhAval: aval<Bvh<PickId, BvhEntry>>;

  // Phase 5 — focus model. `focusedPickId` is observable via aval;
  // dispatcher subscribes to drive OnFocus / OnBlur.
  private readonly _focused = cval<PickId | undefined>(undefined);
  /** Currently focused scope's PickId, or `undefined`. Observable. */
  readonly focusedPickId: aval<PickId | undefined> = this._focused;

  constructor() {
    // Project each PickObject through its intersectable + trafo avals.
    // Two-level chain: intersectable.bind(i => trafo.map(t =>
    // BvhEntry)). Each tick of either aval surfaces a fresh BvhEntry
    // identity (different object reference) so the inner reader emits
    // Rem(old) + Add(new) on the next bvhAval recomputation.
    const transformed: aset<BvhEntry> = this._pickObjects.mapA((o) =>
      o.intersectable.bind((i) =>
        o.trafo.map((t): BvhEntry => ({ scope: o.scope, intersectable: i, trafo: t })),
      ),
    );
    const reader = transformed.getReader();
    let tree: Bvh<PickId, BvhEntry> = Bvh.empty();
    // Pending-Add map keyed by pickId — mapA emits Rem(old) + Add(new)
    // for the SAME key when an inner aval ticks. We collect all Rem's
    // first (so the BVH drops the stale bbox before re-adding), then
    // all Add's (with the fresh world bbox).
    this.bvhAval = AVal.custom((token) => {
      const ops = reader.getChanges(token);
      // Apply Rem first, then Add — same ordering as F# SceneHandler.fs.
      // CountingHashSet deltas: count===-1 is Rem, count===1 is Add.
      // Each entry surfaces a Rem(old BvhEntry) + Add(new BvhEntry)
      // with the SAME scope.pickId but a DIFFERENT entry-object
      // reference, so we must remove by old key and re-add with new
      // bbox.
      for (const op of ops) {
        if (op.count < 0) {
          tree = tree.remove(op.value.scope.pickId);
        }
      }
      for (const op of ops) {
        if (op.count > 0) {
          const e = op.value;
          const worldBox = transformBox(e.intersectable.boundingBox, e.trafo);
          tree = tree.add(e.scope.pickId, worldBox, e);
        }
      }
      return tree;
    });
  }

  /**
   * Attach the GPU metadata mirror. Replays all current entries so a
   * picker attached after a compile still sees every live scope.
   */
  attachObserver(obs: PickRegistryObserver | undefined): void {
    this.observer = obs;
    if (obs !== undefined) {
      for (const [id, scope] of this.entries) {
        obs.register(scope, this.modes.get(id) ?? "A");
      }
    }
  }

  acquire(scope: Omit<LeafPickScope, "pickId">, mode: PickMode = "A"): PickId {
    const pickId = this.next++;
    const full: LeafPickScope = { pickId, ...scope };
    this.entries.set(pickId, full);
    this.modes.set(pickId, mode);
    this.observer?.register(full, mode);
    // BVH-add gate. Pixel and BVH ray-cast are complementary — pixel-
    // path scopes resolve through the pickFb readback alone, so the
    // BVH only needs ray-cast-path entries. Without this gate a 10k-
    // leaf scene's BVH grows to 10k entries and `spiralHitTest` walks
    // them for every pointermove (800 spiral offsets × cullCount =
    // multi-second stalls).
    const path = scope.pickPath ?? "bvh";
    if (path === "bvh" && scope.intersectable !== undefined) {
      const po: PickObject = {
        scope: full,
        intersectable: scope.intersectable!,
        trafo: scope.model,
      };
      this.bvhEntries.set(pickId, po);
      transact(() => { this._pickObjects.add(po); });
    }
    return pickId;
  }

  lookup(id: PickId): LeafPickScope | undefined {
    return this.entries.get(id);
  }

  /**
   * Drop a previously-acquired pickId. Removes the scope record from
   * `entries`/`modes` and, when the scope landed in the BVH path,
   * removes its `PickObject` from `_pickObjects` (which transitively
   * removes the entry from the live BVH on the next `bvhAval` pull).
   *
   * Called by the scene-graph layer when a lowered leaf is dropped
   * from an adaptive container (aset/alist/aval) — without this, every
   * `cset.remove(...)` leaked one pickId, and any toggle/restream
   * workload grew `entries`/`modes` unboundedly.
   */
  release(id: PickId): void {
    const scope = this.entries.get(id);
    if (scope === undefined) return;
    this.entries.delete(id);
    this.modes.delete(id);
    this.observer?.deregister(id);
    // Drop focus if we just removed the focused scope.
    // AVal.force OK: isConstant guard avoided — _focused is a cval,
    // so `force()` is one-shot here at the API boundary (same shape as
    // setFocus/clearFocus, which also call force).
    if (AVal.force(this._focused) === id) {
      transact(() => { this._focused.value = undefined; });
    }
    // BVH-path scopes also live in `_pickObjects`. The path is decided
    // at acquire-time and stored on the scope; mirror that here so we
    // don't blindly call `_pickObjects.remove` for pixel/none paths
    // (would be a no-op but wastes a transact + Set lookup).
    const po = this.bvhEntries.get(id);
    if (po !== undefined) {
      this.bvhEntries.delete(id);
      transact(() => { this._pickObjects.remove(po); });
    }
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
    transact(() => {
      this._pickObjects.clear();
      this._focused.value = undefined;
    });
    this.entries.clear();
    this.modes.clear();
    this.bvhEntries.clear();
    this.next = 1;
    this.observer?.clear();
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
    // AVal.force OK: programmatic API entry / event handler — see
    // file-top policy.
    if (scope.canFocus === undefined || !AVal.force(scope.canFocus)) return;
    transact(() => { this._focused.value = pickId; });
  }

  /** Drop focus. Equivalent to `setFocus(undefined)`. */
  clearFocus(): void {
    transact(() => { this._focused.value = undefined; });
  }
}

/**
 * Compute the world-space AABB of a local AABB transformed by `t`.
 * Mirrors `IIntersectable.transformed` in wombat.base — kept as a
 * small inline helper here so the BVH delta path doesn't allocate a
 * `TransformedIntersectable` wrapper just to read its `boundingBox`.
 */
function transformBox(local: Box3d, t: Trafo3d): Box3d {
  if (!local.isValid() || local.isEmpty()) return Box3d.empty;
  const fwd = t.forward;
  const corners: V3d[] = [
    new V3d(local.min.x, local.min.y, local.min.z),
    new V3d(local.max.x, local.min.y, local.min.z),
    new V3d(local.min.x, local.max.y, local.min.z),
    new V3d(local.max.x, local.max.y, local.min.z),
    new V3d(local.min.x, local.min.y, local.max.z),
    new V3d(local.max.x, local.min.y, local.max.z),
    new V3d(local.min.x, local.max.y, local.max.z),
    new V3d(local.max.x, local.max.y, local.max.z),
  ];
  return Box3d.fromPoints(corners.map((c) => fwd.transformPos(c)));
}
