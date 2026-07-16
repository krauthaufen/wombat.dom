// Spiral hit-test — pixel + BVH merge, mirroring Aardvark.Dom's
// `SceneHandler.fs` lines 1670–1885 step for step.
//
// For each spiral offset (sorted by d²) we compute TWO candidates:
//  - Pixel: the framebuffer slot 0 at the offset; validated by mode
//    sign, registry membership, snap radius, and a 3×3 same-id
//    neighbour count (rejects MSAA silhouette averages).
//  - BVH: ray-trace the cull set (scopes with `intersectable`),
//    each in its model space, taking the closest hit by ray-t.
//
// Per-offset winner: pixel wins iff valid AND (no BVH or pixDepth
// ≤ bvhDepth). Else BVH wins. Else next offset. Hover capture at
// the centre offset latches `hoverPickId` even when the pixel
// candidate is rejected (cursor / hover updates that don't need a
// real scope still want a stable id).
//
// PickThrough handling on the winner mirrors F#:
//  - winner-by-pixel + scope.pickThrough → warn, keep winner.
//  - winner-by-BVH + scope.pickThrough → re-trace this offset's ray
//    through scopes that are active AND not pickThrough; if any hit
//    take the closest, else keep the original.
//
// AVal.force policy: this file runs from the dispatcher's pointer
// event handlers — "now" is the user's tick. Forces of
// `scope.active`, `scope.model`, `scope.pixelSnapRadius`,
// `scope.noEvents` are all permitted here.

import { AVal } from "@aardworx/wombat.adaptive";
import { Plane3d, Ray3d, Trafo3d, V2i, V3d, V4d } from "@aardworx/wombat.base";

import { n24DecodeF32 } from "./normal24.js";
import type { PickRegion } from "./readback.js";
import { readSlotsAt } from "./readback.js";
import type { LeafPickScope, PickRegistry } from "./registry.js";
import { SNAP_OFFSETS, SNAP_RADIUS_MAX } from "./snapOffsets.js";

export interface ResolvedHit {
  readonly scope: LeafPickScope;
  readonly viewPos: V3d;
  readonly viewNormal: V3d;
  readonly partIndex: number;
  /** True for a pixel-pick winner; false for a BVH-pick winner. */
  readonly isPixel: boolean;
  /** Decoded `abs(slot0)` at the centre offset, even if mapping/validation failed. */
  readonly hoverPickId: number;
  /**
   * For pickThrough re-trace: the scope behind the original
   * pickThrough winner (the one we re-traced into). Mirrors F#'s
   * tuple `(state, ..., Some nextScope)` — exposed mainly for
   * symmetry; the dispatcher only needs the resolved `scope`.
   */
  readonly nextScope?: LeafPickScope;
  /**
   * Portal ("offscreen pick context") support. When the winning
   * pixel belongs to a scope carrying a `pickSubContext`, the pick
   * attachment's slots 1-2 hold the SOURCE UV the composite sampled
   * (not normal/depth) — recorded here so the resolver can recurse
   * into the inner scene (`resolveThroughPortals`).
   */
  readonly portalUv?: { readonly x: number; readonly y: number };
  /**
   * The registry the winning scope's pickId belongs to. `undefined`
   * means the dispatcher's own registry; set to the INNER producer's
   * registry when the hit came through a portal recursion — pickIds
   * from different producers live in independent id spaces.
   */
  readonly registry?: PickRegistry;
}

interface PointerLoc {
  readonly devX: number;
  readonly devY: number;
}

/**
 * `scope.noEvents` is set when the compile-time `state.noEvents` was
 * non-constant — the leaf is registered (so its pickId still exists
 * and the BVH can prune against its bbox), but the dispatcher should
 * treat it as if it weren't there. A static `NoEvents=true` leaf
 * doesn't reach the registry at all (compile.ts collapses it), so
 * scope.noEvents is `undefined` for those.
 *
 * AVal.force OK: spiralHitTest runs from the dispatcher in pointer-
 * event-handler context — "now" is the user's tick.
 */
function scopeNoEvents(scope: LeafPickScope): boolean {
  return scope.noEvents !== undefined && AVal.force(scope.noEvents);
}

/**
 * Cursor-only world-pos resolution. Same query as the spiral merge
 * but at the centre offset only AND without the per-scope
 * `pixelSnapRadius` gating — used by the captured-pointer branch to
 * keep `worldPos` live as the cursor wanders over the scene.
 *
 * Pixel wins if its validated id depth ≤ BVH depth; else BVH wins;
 * else returns `undefined`. Mode-sign + 3×3 neighbour validation
 * mirror the spiral path.
 */
export function pointHitTest(
  region: PickRegion,
  pointer: PointerLoc,
  registry: PickRegistry,
  view: Trafo3d,
  proj: Trafo3d,
  viewportSize: V2i,
): { scope: LeafPickScope | undefined; viewPos: V3d; viewNormal: V3d; partIndex: number; isPixel: boolean } | undefined {
  const sX = viewportSize.x;
  const sY = viewportSize.y;
  if (sX <= 0 || sY <= 0) return undefined;

  const vFwd = view.forward;
  const vBwd = view.backward;
  const pBwd = proj.backward;
  const pFwd = proj.forward;
  // Reversed-Z projections flip the meaning of "closer" in NDC depth —
  // arbitration must follow.
  const revZ = isReversedZ(pBwd);
  const pxX = pointer.devX;
  const pxY = pointer.devY;
  const lx = pxX - region.originX;
  const ly = pxY - region.originY;

  // Pixel candidate at the centre.
  const pixIdRaw = readSlotsAt(region, lx, ly).slot0;
  let pixOk = false;
  let pixScope: LeafPickScope | undefined;
  let pixDepth = 0;
  let pixVp = V3d.zero;
  let pixN = V3d.zero;
  let pixPi = 0;
  if (pixIdRaw !== 0) {
    const absId = pixIdRaw < 0 ? -pixIdRaw : pixIdRaw;
    const scope = registry.lookup(absId);
    const regMode = registry.modeOf(absId);
    const modeOk = regMode !== undefined && (regMode === "A" ? pixIdRaw > 0 : pixIdRaw < 0);
    if (modeOk && scope !== undefined && AVal.force(scope.active) && !scopeNoEvents(scope)) {
      let matches = 0;
      outer: for (let ddy = -1; ddy <= 1; ddy++) {
        for (let ddx = -1; ddx <= 1; ddx++) {
          if (ddx === 0 && ddy === 0) continue;
          if (readSlotsAt(region, lx + ddx, ly + ddy).slot0 === pixIdRaw) {
            matches++;
            if (matches >= 3) break outer;
          }
        }
      }
      if (matches >= 3) {
        const slots = readSlotsAt(region, lx, ly);
        if (pixIdRaw > 0) {
          const ndcZ = slots.slot2;
          pixDepth = ndcZ;
          const tcX = (pxX + 0.5) / sX;
          const tcY = (pxY + 0.5) / sY;
          const ndcX = 2 * tcX - 1;
          const ndcY = 1 - 2 * tcY;
          pixVp = transformPosProj(pBwd, ndcX, ndcY, ndcZ);
          if ((slots.slot1 | 0) === 0) {
            pixN = V3d.zero;
          } else {
            const [nx, ny, nz] = n24DecodeF32(slots.slot1);
            pixN = new V3d(nx, ny, nz);
          }
          pixPi = slots.slot3 | 0;
        } else {
          pixVp = new V3d(slots.slot1, slots.slot2, slots.slot3);
          pixDepth = transformPosProjZ(pFwd, pixVp);
          pixN = V3d.zero;
        }
        pixScope = scope;
        pixOk = true;
      }
    }
  }

  // BVH candidate at the centre ray. No snap-radius gating.
  let bvhOk = false;
  let bvhScope: LeafPickScope | undefined;
  let bvhDepth = 0;
  let bvhVp = V3d.zero;
  let bvhN = V3d.zero;
  // AVal.force OK: dispatcher / pointHitTest run in pointer event
  // handler context — see file-top policy.
  const bvh = AVal.force(registry.bvhAval);
  if (bvh.count > 0) {
    const ndcX = (2 * (pxX + 0.5) / sX) - 1;
    const ndcY = 1 - (2 * (pxY + 0.5) / sY);
    const near = unprojClipToWorld(ndcX, ndcY, -1, pBwd, vBwd);
    const far  = unprojClipToWorld(ndcX, ndcY,  1, pBwd, vBwd);
    const ray = Ray3d.fromPoints(near, far);
    let bestT = Number.POSITIVE_INFINITY;
    for (const item of bvh.items()) {
      const scope = item.value.scope;
      if (!AVal.force(scope.active)) continue;
      if (scopeNoEvents(scope)) continue;
      const trafo = AVal.force(scope.model);
      const localRay = ray.transformed(trafo.inverse());
      const hit = item.value.intersectable.intersects(localRay, 0, bestT);
      if (hit !== undefined && hit.t < bestT) {
        bestT = hit.t;
        const worldPoint = trafo.transformPos(hit.point);
        const viewPoint = vFwd.transformPos(worldPoint);
        bvhDepth = transformPosProjZ(pFwd, viewPoint);
        bvhVp = viewPoint;
        const worldN = transposedTransformDir(trafo.backward, hit.normal);
        bvhN = normalize(transposedTransformDir(vBwd, worldN));
        bvhScope = scope;
        bvhOk = true;
      }
    }
  }

  if (pixOk && pixScope !== undefined && (!bvhOk || (revZ ? pixDepth >= bvhDepth : pixDepth <= bvhDepth))) {
    return { scope: pixScope, viewPos: pixVp, viewNormal: pixN, partIndex: pixPi, isPixel: true };
  }
  if (bvhOk && bvhScope !== undefined) {
    return { scope: bvhScope, viewPos: bvhVp, viewNormal: bvhN, partIndex: 0, isPixel: false };
  }
  return undefined;
}

export function spiralHitTest(
  region: PickRegion,
  pointer: PointerLoc,
  registry: PickRegistry,
  view: Trafo3d,
  proj: Trafo3d,
  viewportSize: V2i,
): ResolvedHit | undefined {
  const sX = viewportSize.x;
  const sY = viewportSize.y;
  if (sX <= 0 || sY <= 0) return undefined;

  // Per-scope snap-r² cache (matches F#'s `snapR2` closure with its
  // memo dict).
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

  // Ray for a given device pixel. The ray runs along the inverse
  // projection through `(ndcX, ndcY, ±1)`, then back through view
  // into world space. (`vBwd · pBwd · ndc → world`.)
  const vFwd = view.forward;
  const vBwd = view.backward;
  const pBwd = proj.backward;
  const pFwd = proj.forward;
  // Reversed-Z projections flip the meaning of "closer" in NDC depth —
  // arbitration must follow.
  const revZ = isReversedZ(pBwd);
  const rayFor = (pxX: number, pxY: number): Ray3d => {
    const ndcX = (2 * (pxX + 0.5) / sX) - 1;
    const ndcY = 1 - (2 * (pxY + 0.5) / sY);
    const near = unprojClipToWorld(ndcX, ndcY, -1, pBwd, vBwd);
    const far  = unprojClipToWorld(ndcX, ndcY,  1, pBwd, vBwd);
    return Ray3d.fromPoints(near, far);
  };

  // Cull set — scopes with an intersectable whose world-space bbox
  // touches the cursor disc's frustum cone. F# Aardvark.Dom pre-
  // filters with `bvh.getIntersecting(frustumAabb)`; we do the same.
  // Without this, a 10k-leaf scene flattens the entire BVH and the
  // 800-spiral-offset inner loop becomes O(800 × N) per pointer
  // event — turning every hover into a multi-second stall.
  interface Cull {
    readonly scope: LeafPickScope;
    readonly intersectable: import("@aardworx/wombat.base").IIntersectable;
    readonly trafo: Trafo3d;
  }
  const cullSet: Cull[] = [];
  // AVal.force OK: dispatcher pointer event handler context — see
  // file-top policy.
  const bvh = AVal.force(registry.bvhAval);
  if (bvh.count > 0) {
    // Unproject the 4 corners of the 33×33 pick disc through near
    // and far → 8 world-space frustum corners. From those, build the
    // 6 inward-facing planes (left/right/bottom/top + near/far) and
    // use `bvh.getIntersectingFrustum` to walk the tree with a tight
    // convex-hull test instead of an AABB. AABB-of-corners would be
    // arbitrarily loose along the camera-diagonal — entire scene
    // can collapse into one box for a typical orbit angle. The
    // plane test stays tight (~6 SAT-style classifications per
    // node) and prunes whole subtrees.
    const pxMinX = pointer.devX - SNAP_RADIUS_MAX;
    const pxMaxX = pointer.devX + SNAP_RADIUS_MAX;
    const pxMinY = pointer.devY - SNAP_RADIUS_MAX;
    const pxMaxY = pointer.devY + SNAP_RADIUS_MAX;
    const ndcAt = (px: number, py: number): { x: number; y: number } => ({
      x: (2 * (px + 0.5) / sX) - 1,
      y: 1 - (2 * (py + 0.5) / sY),
    });
    // Order: bl, br, tr, tl. Pair (nearN, farN) per corner.
    const cN = [
      ndcAt(pxMinX, pxMaxY), ndcAt(pxMaxX, pxMaxY),
      ndcAt(pxMaxX, pxMinY), ndcAt(pxMinX, pxMinY),
    ];
    const nearW = cN.map(c => unprojClipToWorld(c.x, c.y, -1, pBwd, vBwd));
    const farW  = cN.map(c => unprojClipToWorld(c.x, c.y,  1, pBwd, vBwd));
    // Inward-facing planes. fromThreePoints' normal uses RH on
    // (b-a, c-a); pick triplets so the normal points into the
    // frustum interior. We always flip to enforce: a sanity
    // anchor inside the frustum should have positive signedDistance.
    // Use the disc-centre near point as the anchor.
    const ndcC = ndcAt(pointer.devX, pointer.devY);
    const anchor = unprojClipToWorld(ndcC.x, ndcC.y, 0, pBwd, vBwd);
    const inward = (p: Plane3d): Plane3d =>
      p.signedDistance(anchor) >= 0 ? p : p.flipped();
    const planes: Plane3d[] = [
      // bottom: near0-far0-near1 (y- side of disc)
      inward(Plane3d.fromThreePoints(nearW[0]!, farW[0]!, nearW[1]!)),
      // right:  near1-far1-near2
      inward(Plane3d.fromThreePoints(nearW[1]!, farW[1]!, nearW[2]!)),
      // top:    near2-far2-near3
      inward(Plane3d.fromThreePoints(nearW[2]!, farW[2]!, nearW[3]!)),
      // left:   near3-far3-near0
      inward(Plane3d.fromThreePoints(nearW[3]!, farW[3]!, nearW[0]!)),
      // near:   near0-near1-near2
      inward(Plane3d.fromThreePoints(nearW[0]!, nearW[1]!, nearW[2]!)),
      // far:    far2-far1-far0
      inward(Plane3d.fromThreePoints(farW[2]!,  farW[1]!,  farW[0]!)),
    ];
    const candidates = bvh.getIntersectingFrustum(planes);
    for (const item of candidates) {
      const scope = item.value.scope;
      const trafo = AVal.force(scope.model);
      cullSet.push({ scope, intersectable: item.value.intersectable, trafo });
    }
  }
  const cullCount = cullSet.length;

  let winnerHoverIdValue = 0;

  // Priority support: instead of returning at the FIRST valid offset,
  // keep the best candidate by (priority desc); offsets are distance-
  // sorted, so the first candidate AT a given priority is its closest.
  // Early-exit once the best reaches the registry's priority watermark
  // (all-default scenes exit at the first candidate — old behaviour).
  const prioOf = (scope: LeafPickScope): number => {
    if (scope.pickPriority === undefined) return 0;
    const p = AVal.force(scope.pickPriority);
    return Math.max(-8, Math.min(7, Math.floor(p)));
  };
  const prioMax = (registry as { maxPickPriority?: number }).maxPickPriority ?? 0;
  let bestPrio = -Infinity;
  let bestWinner: PreliminaryWinner | undefined;

  // readIdAt: helper for the 3×3 neighbour count + the centre slot
  // read. Mirrors F# `readIdAt`.
  const readIdAt = (lx: number, ly: number): number => {
    return readSlotsAt(region, lx, ly).slot0;
  };

  for (let oi = 0; oi < SNAP_OFFSETS.length; oi++) {
    const off = SNAP_OFFSETS[oi]!;
    const d2 = off.d2;
    const pxX = pointer.devX + off.dx;
    const pxY = pointer.devY + off.dy;
    const lx = pxX - region.originX;
    const ly = pxY - region.originY;

    // ---- Pixel candidate ----
    const pixIdRaw = readIdAt(lx, ly);
    let pixOk = false;
    let pixScope: LeafPickScope | undefined;
    let pixDepth = 0;
    let pixVp = V3d.zero;
    let pixN = V3d.zero;
    let pixPi = 0;

    if (pixIdRaw !== 0) {
      const absId = pixIdRaw < 0 ? -pixIdRaw : pixIdRaw;
      // Hover capture even when we can't validate. Only at the
      // centre — F# matches off.X = 0 && off.Y = 0.
      if (winnerHoverIdValue === 0 && off.dx === 0 && off.dy === 0) {
        winnerHoverIdValue = absId;
      }
      const scope = registry.lookup(absId);
      const regMode = registry.modeOf(absId);
      // Mode-vs-sign: scope was registered with a specific mode;
      // an averaged silhouette pixel can land on the wrong one.
      const modeOk = regMode !== undefined
        && (regMode === "A" ? pixIdRaw > 0 : pixIdRaw < 0);

      if (modeOk && scope !== undefined && AVal.force(scope.active) && !scopeNoEvents(scope) && d2 <= snapR2(scope)) {
        // 3×3 same-id neighbour count, ≥ 3 to validate.
        let matches = 0;
        outer: for (let ddy = -1; ddy <= 1; ddy++) {
          for (let ddx = -1; ddx <= 1; ddx++) {
            if (ddx === 0 && ddy === 0) continue;
            if (readIdAt(lx + ddx, ly + ddy) === pixIdRaw) {
              matches++;
              if (matches >= 3) break outer;
            }
          }
        }
        if (matches >= 3) {
          const slots = readSlotsAt(region, lx, ly);
          if (pixIdRaw > 0) {
            // Mode A: slot1 = encoded normal, slot2 = NDC depth,
            // slot3 = part index.
            const ndcZ = slots.slot2;
            pixDepth = ndcZ;
            const tcX = (pxX + 0.5) / sX;
            const tcY = (pxY + 0.5) / sY;
            const ndcX = 2 * tcX - 1;
            const ndcY = 1 - 2 * tcY;
            pixVp = transformPosProj(pBwd, ndcX, ndcY, ndcZ);
            const enc = slots.slot1 | 0;
            if (enc === 0) {
              pixN = V3d.zero;
            } else {
              const [nx, ny, nz] = n24DecodeF32(slots.slot1);
              pixN = new V3d(nx, ny, nz);
            }
            pixPi = slots.slot3 | 0;
          } else {
            // Mode B: viewPos is in slots 1..3 directly.
            pixVp = new V3d(slots.slot1, slots.slot2, slots.slot3);
            // Project back to NDC for depth comparison against BVH.
            const ndcZ = transformPosProjZ(pFwd, pixVp);
            pixDepth = ndcZ;
            pixN = V3d.zero;
            pixPi = 0;
          }
          pixScope = scope;
          pixOk = true;
        }
      }
    }

    // ---- BVH candidate ----
    let bvhOk = false;
    let bvhScope: LeafPickScope | undefined;
    let bvhDepth = 0;
    let bvhVp = V3d.zero;
    let bvhN = V3d.zero;
    if (cullCount > 0) {
      const ray = rayFor(pxX, pxY);
      let bestT = Number.POSITIVE_INFINITY;
      for (let k = 0; k < cullCount; k++) {
        const entry = cullSet[k]!;
        if (!AVal.force(entry.scope.active)) continue;
        if (scopeNoEvents(entry.scope)) continue;
        if (d2 > snapR2(entry.scope)) continue;
        const localRay = ray.transformed(entry.trafo.inverse());
        const hit = entry.intersectable.intersects(localRay, 0, bestT);
        if (hit !== undefined && hit.t < bestT) {
          bestT = hit.t;
          const worldPoint = entry.trafo.transformPos(hit.point);
          const viewPoint = vFwd.transformPos(worldPoint);
          const wp4 = vFwd.transform(new V4d(worldPoint.x, worldPoint.y, worldPoint.z, 1));
          // Depth in NDC for compare against pixDepth.
          bvhDepth = transformPosProjZ(pFwd, viewPoint);
          bvhVp = viewPoint;
          // Normal: world n = trafo.backward.transposed.transformDir(localN);
          // viewN = view.backward.transposed.transformDir(worldN); then normalize.
          const localN = hit.normal;
          const worldN = transposedTransformDir(entry.trafo.backward, localN);
          const viewNormal = transposedTransformDir(vBwd, worldN);
          bvhN = normalize(viewNormal);
          bvhScope = entry.scope;
          bvhOk = true;
          void wp4;
        }
      }
    }

    // ---- Decide winner for this offset ----
    // Same-offset pixel-vs-BVH: higher priority wins; tie → depth rule.
    let cand: PreliminaryWinner | undefined;
    let candPrio = -Infinity;
    const pixPrio = pixOk && pixScope !== undefined ? prioOf(pixScope) : -Infinity;
    const bvhPrio = bvhOk && bvhScope !== undefined ? prioOf(bvhScope) : -Infinity;
    const pickPixel =
      pixOk && (!bvhOk
        || pixPrio > bvhPrio
        || (pixPrio === bvhPrio && (revZ ? pixDepth >= bvhDepth : pixDepth <= bvhDepth)));
    if (pickPixel && pixScope !== undefined) {
      cand = {
        scope: pixScope, viewPos: pixVp, viewNormal: pixN, partIndex: pixPi,
        isPixel: true, offX: off.dx, offY: off.dy, hoverPickId: pixScope.pickId,
      };
      candPrio = pixPrio;
    } else if (bvhOk && bvhScope !== undefined) {
      cand = {
        scope: bvhScope, viewPos: bvhVp, viewNormal: bvhN, partIndex: 0,
        isPixel: false, offX: off.dx, offY: off.dy, hoverPickId: winnerHoverIdValue,
      };
      candPrio = bvhPrio;
    }
    if (cand !== undefined && candPrio > bestPrio) {
      bestPrio = candPrio;
      bestWinner = cand;
      if (bestPrio >= prioMax) break; // nothing higher can exist
    }
  }

  if (bestWinner !== undefined) {
    winnerHoverIdValue = bestWinner.hoverPickId !== 0 ? bestWinner.hoverPickId : winnerHoverIdValue;
    return finalizeWinner(
      { ...bestWinner, hoverPickId: winnerHoverIdValue },
      pointer, cullSet, rayFor);
  }

  // No spiral offset produced a winner. F# returns None here; we
  // return undefined to mirror that. (`hoverPickId` is lost on
  // miss — F#'s `hoverId` is set inside the same scope but only
  // reflected externally on a real winner, so dropping it on a
  // total miss matches.)
  return undefined;
}

interface PreliminaryWinner {
  readonly scope: LeafPickScope;
  readonly viewPos: V3d;
  readonly viewNormal: V3d;
  readonly partIndex: number;
  readonly isPixel: boolean;
  readonly offX: number;
  readonly offY: number;
  readonly hoverPickId: number;
}

function finalizeWinner(
  w: PreliminaryWinner,
  pointer: PointerLoc,
  cullSet: ReadonlyArray<{ scope: LeafPickScope; intersectable: import("@aardworx/wombat.base").IIntersectable; trafo: Trafo3d }>,
  rayFor: (px: number, py: number) => Ray3d,
): ResolvedHit {
  const scope = w.scope;
  if (!scope.pickThrough) {
    return {
      scope,
      viewPos: w.viewPos,
      viewNormal: w.viewNormal,
      partIndex: w.partIndex,
      isPixel: w.isPixel,
      hoverPickId: w.hoverPickId,
    };
  }

  if (w.isPixel) {
    // Match F#: pixel-picked pickThrough scopes can't be re-traced
    // (we don't have a depth from the next geometry behind). Warn
    // and keep.
    // eslint-disable-next-line no-console
    console.warn("[picking] cannot pick-through pixel-picked objects");
    return {
      scope,
      viewPos: w.viewPos,
      viewNormal: w.viewNormal,
      partIndex: w.partIndex,
      isPixel: w.isPixel,
      hoverPickId: w.hoverPickId,
    };
  }

  // BVH winner + pickThrough → re-trace through active, non-pickThrough
  // scopes only. Take the closest hit. If none, keep the original
  // winner (mirrors F# `Some (scope, viewPos, ..., None)`).
  const ray = rayFor(pointer.devX + w.offX, pointer.devY + w.offY);
  let bestT = Number.POSITIVE_INFINITY;
  let nextScope: LeafPickScope | undefined;
  let nextVp = V3d.zero;
  let nextN = V3d.zero;
  for (const entry of cullSet) {
    if (!AVal.force(entry.scope.active)) continue;
    if (scopeNoEvents(entry.scope)) continue;
    if (entry.scope.pickThrough) continue;
    const localRay = ray.transformed(entry.trafo.inverse());
    const hit = entry.intersectable.intersects(localRay, 0, bestT);
    if (hit !== undefined && hit.t < bestT) {
      bestT = hit.t;
      const worldPoint = entry.trafo.transformPos(hit.point);
      // We can't easily get a fresh view trafo here — `viewPos` is
      // computed in the same view as the original BVH winner used.
      // Caller (dispatcher) gets `nextScope`, treats it as the
      // dispatch target, and re-uses the winner's view-space hit if
      // it needs view coords. To stay close to F# we write the
      // intersection's world point into `nextVp` slot — callers
      // reading `viewPos` should multiply through their own view if
      // needed. For this codebase the dispatcher already runs an
      // unproject from the spiral hit pixel, so this is fine.
      nextVp = worldPoint;
      const localN = hit.normal;
      nextN = transposedTransformDir(entry.trafo.backward, localN);
      nextScope = entry.scope;
    }
  }
  if (nextScope === undefined) {
    return {
      scope,
      viewPos: w.viewPos,
      viewNormal: w.viewNormal,
      partIndex: w.partIndex,
      isPixel: w.isPixel,
      hoverPickId: w.hoverPickId,
    };
  }
  return {
    scope: nextScope,
    viewPos: nextVp,
    viewNormal: normalize(nextN),
    partIndex: 0,
    isPixel: false,
    hoverPickId: w.hoverPickId,
    nextScope,
  };
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/**
 * Does this projection map SMALLER NDC depth to points FARTHER from the
 * camera (reversed-Z)? Pixel-vs-BVH arbitration has to know, because
 * `pixDepth`/`bvhDepth` are NDC depths and "closer" flips sign.
 *
 * Probed, not sniffed off a matrix element: unproject two depths well
 * inside the valid NDC range and compare their distance from the camera
 * (which is the origin of view space — an invariant of any view trafo).
 * The old test `pFwd.M22 >= 0` matched the real reversed perspectives
 * but also fired on degenerate projections (identity has M22 = 1), which
 * inverted arbitration for orthonormal / test projections where +z is
 * genuinely farther. Both probe points sit on the same side of the
 * projection's pole (w = 0), so the depth→distance map is monotone
 * between them and one comparison settles the direction.
 */
export function isReversedZ(pBwd: import("@aardworx/wombat.base").M44d): boolean {
  const a = transformPosProj(pBwd, 0, 0, 0.25);
  const b = transformPosProj(pBwd, 0, 0, 0.75);
  const da = a.x * a.x + a.y * a.y + a.z * a.z;
  const db = b.x * b.x + b.y * b.y + b.z * b.z;
  return da > db;
}

function unprojClipToWorld(
  ndcX: number, ndcY: number, ndcZ: number,
  pBwd: import("@aardworx/wombat.base").M44d,
  vBwd: import("@aardworx/wombat.base").M44d,
): V3d {
  const v4 = pBwd.transform(new V4d(ndcX, ndcY, ndcZ, 1));
  const w = v4.w !== 0 ? v4.w : 1;
  const view = new V4d(v4.x / w, v4.y / w, v4.z / w, 1);
  const w4 = vBwd.transform(view);
  const ww = w4.w !== 0 ? w4.w : 1;
  return new V3d(w4.x / ww, w4.y / ww, w4.z / ww);
}

function transformPosProj(m: import("@aardworx/wombat.base").M44d, x: number, y: number, z: number): V3d {
  const v = m.transform(new V4d(x, y, z, 1));
  const w = v.w !== 0 ? v.w : 1;
  return new V3d(v.x / w, v.y / w, v.z / w);
}

function transformPosProjZ(m: import("@aardworx/wombat.base").M44d, p: V3d): number {
  const v = m.transform(new V4d(p.x, p.y, p.z, 1));
  const w = v.w !== 0 ? v.w : 1;
  return v.z / w;
}

function transposedTransformDir(m: import("@aardworx/wombat.base").M44d, d: V3d): V3d {
  // (Mᵀ · d) treating d as direction (w=0).
  // Equivalent to dot products with rows transposed = columns.
  return new V3d(
    m.M00 * d.x + m.M10 * d.y + m.M20 * d.z,
    m.M01 * d.x + m.M11 * d.y + m.M21 * d.z,
    m.M02 * d.x + m.M12 * d.y + m.M22 * d.z,
  );
}

function normalize(v: V3d): V3d {
  const l = Math.hypot(v.x, v.y, v.z);
  if (l === 0) return v;
  return new V3d(v.x / l, v.y / l, v.z / l);
}
