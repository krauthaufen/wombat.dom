// Pick arbitration — the CPU half of the single-result pick path.
//
// Consumes the GPU argmin kernel's single winning pixel
// (`PickArgminResult`) plus ONE BVH centre ray, and merges them into a
// `ResolvedHit` the dispatcher can fire. Replaces the spiral walk's
// 33×33 region readback + ~800-offset CPU merge with: argmin (GPU) →
// one ray (CPU, O(log N) via `bvh.closestHit`) → a 5-case arbitration.
//
// 5-case arbitration (pixel = argmin winner, bvh = centre-ray hit):
//   pixel exactly at cursor + bvh hit  → whichever is closer in depth
//   pixel off-centre        + bvh hit  → bvh
//   no pixel                + bvh hit  → bvh
//   any pixel               + bvh miss → pixel as-is (no occlusion check)
//   no pixel                + bvh miss → nothing
// "exactly at cursor" is `result.dist2 === 0` (the kernel reports the
// winner's screen-dist² to the cursor; the in-kernel snap-radius gate
// means an off-centre winner is the nearest VALID pixel within radius).
//
// The argmin kernel already did the pixel-side validity gating
// (slot0≠0, per-id snap radius, mode-sign, 3×3 MSAA neighbour count),
// so here we only decode the winner's slots — Mode-A (normal/depth/
// part) or Mode-B (viewPos) — exactly as the spiral path did.
//
// pickThrough on the chosen winner mirrors the spiral path:
//   pixel winner + pickThrough → warn, keep (no depth behind it).
//   bvh winner   + pickThrough → re-trace the centre ray through
//     active, non-pickThrough scopes; take the closest, else keep.
//
// AVal.force policy: every force here runs from the dispatcher's
// pointer-event handlers — "now" is the user's tick (see dispatcher
// file-top policy). Forces of scope.active / model / noEvents are fine.

import { AVal } from "@aardworx/wombat.adaptive";
import { Ray3d, Trafo3d, V2i, V3d, V4d } from "@aardworx/wombat.base";
import type { IIntersectHit, M44d } from "@aardworx/wombat.base";

import { n24DecodeF32 } from "./normal24.js";
import type { PickArgminResult } from "./pickArgminCompute.js";
import type { LeafPickScope, PickRegistry } from "./registry.js";
import { isReversedZ } from "./spiralHitTest.js";
import type { ResolvedHit } from "./spiralHitTest.js";

interface PointerLoc {
  readonly devX: number;
  readonly devY: number;
}

/** A decoded BVH centre-ray hit, all in WORLD space. */
interface BvhHit {
  readonly scope: LeafPickScope;
  readonly worldPoint: V3d;
  readonly worldNormal: V3d;
}

function scopeNoEvents(scope: LeafPickScope): boolean {
  return scope.noEvents !== undefined && AVal.force(scope.noEvents);
}

/**
 * Resolve the pick under `pointer` from the GPU argmin `result` and one
 * BVH centre ray. Drop-in replacement for `spiralHitTest`'s return.
 *
 * `result === undefined` means the GPU read failed/was skipped — we
 * fall back to a BVH-only resolve (still useful for proxy geometry).
 */
export function arbitratePick(
  result: PickArgminResult | undefined,
  pointer: PointerLoc,
  registry: PickRegistry,
  view: Trafo3d,
  proj: Trafo3d,
  viewportSize: V2i,
): ResolvedHit | undefined {
  const sX = viewportSize.x;
  const sY = viewportSize.y;
  if (sX <= 0 || sY <= 0) return undefined;

  const vFwd = view.forward;
  const vBwd = view.backward;
  const pBwd = proj.backward;
  const pFwd = proj.forward;

  // Hover id: the raw centre pixel, even when it's not a valid winner
  // (mirrors the spiral path's centre-offset hover latch).
  const hoverPickId = result !== undefined ? Math.abs(result.centerSlot0) | 0 : 0;

  // ---- Pixel candidate (decode the argmin winner) ----
  let pixScope: LeafPickScope | undefined;
  let pixVp = V3d.zero;
  let pixN = V3d.zero;
  let pixPi = 0;
  let pixDepth = 0;
  let portalUv: { x: number; y: number } | undefined;
  const pixCentered = result !== undefined && result.found && result.dist2 === 0;
  if (result !== undefined && result.found) {
    const s0 = result.slot0;
    const absId = Math.abs(s0) | 0;
    const scope = registry.lookup(absId);
    if (scope !== undefined) {
      if (s0 > 0 && scope.pickSubContext !== undefined) {
        // Portal: slots 1-2 = the sampled source-uv, slot 3 = the
        // portal GEOMETRY's own NDC depth (so pixel-vs-BVH arbitration
        // below still compares real depths).
        const ndcZ = result.slot3;
        pixDepth = ndcZ;
        const ndcX = 2 * ((result.px + 0.5) / sX) - 1;
        const ndcY = 1 - 2 * ((result.py + 0.5) / sY);
        pixVp = transformPosProj(pBwd, ndcX, ndcY, ndcZ);
        pixN = V3d.zero;
        pixPi = 0;
        portalUv = { x: result.slot1, y: result.slot2 };
      } else if (s0 > 0) {
        // Mode A: slot1 = encoded normal, slot2 = NDC depth, slot3 = part.
        const ndcZ = result.slot2;
        pixDepth = ndcZ;
        const ndcX = 2 * ((result.px + 0.5) / sX) - 1;
        const ndcY = 1 - 2 * ((result.py + 0.5) / sY);
        pixVp = transformPosProj(pBwd, ndcX, ndcY, ndcZ);
        pixN = (result.slot1 | 0) === 0 ? V3d.zero : n24ToV3d(result.slot1);
        pixPi = result.slot3 | 0;
      } else {
        // Mode B: viewPos in slots 1..3 directly.
        pixVp = new V3d(result.slot1, result.slot2, result.slot3);
        pixDepth = transformPosProjZ(pFwd, pixVp);
        pixN = V3d.zero;
      }
      pixScope = scope;
    }
  }

  // ---- BVH centre-ray candidate ----
  const ray = rayFor(pointer.devX, pointer.devY, sX, sY, pBwd, vBwd);
  const bvh = bvhClosest(ray, registry, () => true);
  let bvhDepth = 0;
  if (bvh !== undefined) {
    bvhDepth = transformPosProjZ(pFwd, vFwd.transformPos(bvh.worldPoint));
  }

  // ---- 5-case arbitration ----
  let winnerIsPixel: boolean;
  if (bvh !== undefined) {
    // Pixel wins only when it's exactly under the cursor AND in front.
    // "In front" flips under a reversed-Z projection (near = 1, far = 0).
    const revZ = isReversedZ(pBwd);
    winnerIsPixel = pixScope !== undefined && pixCentered
      && (revZ ? pixDepth >= bvhDepth : pixDepth <= bvhDepth);
  } else {
    winnerIsPixel = pixScope !== undefined;
  }

  if (winnerIsPixel && pixScope !== undefined) {
    // Pixel winner. pickThrough can't re-trace (no geometry depth
    // behind a pixel hit) — warn and keep, matching the spiral path.
    if (pixScope.pickThrough) {
      // eslint-disable-next-line no-console
      console.warn("[picking] cannot pick-through pixel-picked objects");
    }
    return {
      scope: pixScope, viewPos: pixVp, viewNormal: pixN, partIndex: pixPi,
      isPixel: true, hoverPickId,
      ...(portalUv !== undefined ? { portalUv } : {}),
    };
  }

  if (bvh !== undefined) {
    // BVH winner. If pickThrough, re-trace the SAME centre ray through
    // active, non-pickThrough scopes and take the closest; else keep.
    let chosen = bvh;
    let nextScope: LeafPickScope | undefined;
    if (bvh.scope.pickThrough) {
      const behind = bvhClosest(ray, registry, (s) => !s.pickThrough);
      if (behind !== undefined) {
        chosen = behind;
        nextScope = behind.scope;
      }
    }
    const viewPos = vFwd.transformPos(chosen.worldPoint);
    const viewNormal = normalize(transposedTransformDir(vBwd, chosen.worldNormal));
    return {
      scope: chosen.scope,
      viewPos,
      viewNormal,
      partIndex: 0,
      isPixel: false,
      hoverPickId,
      ...(nextScope !== undefined ? { nextScope } : {}),
    };
  }

  return undefined;
}

function n24ToV3d(enc: number): V3d {
  const [nx, ny, nz] = n24DecodeF32(enc);
  return new V3d(nx, ny, nz);
}

// ---------------------------------------------------------------------------
// BVH centre-ray query
// ---------------------------------------------------------------------------

/**
 * Closest BVH hit along `ray`, in WORLD space. `accept(scope)` filters
 * candidate scopes (used for pickThrough re-trace). Returns undefined
 * when no scope is hit. Uses `bvh.closestHit`, so it's O(log N), not the
 * O(N) scan the old `pointHitTest` did.
 */
function bvhClosest(
  ray: Ray3d,
  registry: PickRegistry,
  accept: (scope: LeafPickScope) => boolean,
): BvhHit | undefined {
  const bvh = AVal.force(registry.bvhAval);
  if (bvh.count === 0) return undefined;
  const hit = bvh.closestHit(ray, 0, Number.POSITIVE_INFINITY, (_key, value) => {
    const scope = value.scope;
    if (!AVal.force(scope.active)) return undefined;
    if (scopeNoEvents(scope)) return undefined;
    if (!accept(scope)) return undefined;
    const trafo = AVal.force(scope.model);
    const localRay = ray.transformed(trafo.inverse());
    const lh = value.intersectable.intersects(localRay, 0, Number.POSITIVE_INFINITY);
    if (lh === undefined) return undefined;
    // Re-express in WORLD space so `closestHit`'s bestT pruning (world-
    // ray slab) and the hit's t are in the same space.
    const worldPoint = trafo.transformPos(lh.point);
    const worldNormal = transposedTransformDir(trafo.backward, lh.normal);
    const t = worldPoint.sub(ray.origin).dot(ray.direction); // direction is unit
    const out: IIntersectHit = { t, point: worldPoint, normal: worldNormal };
    return out;
  });
  if (hit === undefined) return undefined;
  return { scope: hit.value.scope, worldPoint: hit.point, worldNormal: hit.normal };
}

// ---------------------------------------------------------------------------
// Math helpers (kept identical to spiralHitTest's, so behaviour matches)
// ---------------------------------------------------------------------------

function rayFor(pxX: number, pxY: number, sX: number, sY: number, pBwd: M44d, vBwd: M44d): Ray3d {
  const ndcX = (2 * (pxX + 0.5) / sX) - 1;
  const ndcY = 1 - (2 * (pxY + 0.5) / sY);
  const near = unprojClipToWorld(ndcX, ndcY, -1, pBwd, vBwd);
  const far = unprojClipToWorld(ndcX, ndcY, 1, pBwd, vBwd);
  return Ray3d.fromPoints(near, far);
}

function unprojClipToWorld(ndcX: number, ndcY: number, ndcZ: number, pBwd: M44d, vBwd: M44d): V3d {
  const v4 = pBwd.transform(new V4d(ndcX, ndcY, ndcZ, 1));
  const w = v4.w !== 0 ? v4.w : 1;
  const viewp = new V4d(v4.x / w, v4.y / w, v4.z / w, 1);
  const w4 = vBwd.transform(viewp);
  const ww = w4.w !== 0 ? w4.w : 1;
  return new V3d(w4.x / ww, w4.y / ww, w4.z / ww);
}

function transformPosProj(m: M44d, x: number, y: number, z: number): V3d {
  const v = m.transform(new V4d(x, y, z, 1));
  const w = v.w !== 0 ? v.w : 1;
  return new V3d(v.x / w, v.y / w, v.z / w);
}

function transformPosProjZ(m: M44d, p: V3d): number {
  const v = m.transform(new V4d(p.x, p.y, p.z, 1));
  const w = v.w !== 0 ? v.w : 1;
  return v.z / w;
}

function transposedTransformDir(m: M44d, d: V3d): V3d {
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

// ---------------------------------------------------------------------------
// Portal recursion — shared by the dispatcher and by nested
// `IPickSubContext.pickAt` implementations, so arbitrary nesting works
// through ONE code path.
// ---------------------------------------------------------------------------

/**
 * If `hit` landed on a portal scope (a `pickSubContext`-bearing leaf
 * whose pick pixel carried the sampled source-uv), forward the pick
 * into the inner scene and return the INNERMOST hit; an inner miss
 * falls through to the portal scope itself (hover the "window
 * background" — its own cursor / handlers apply). Non-portal hits
 * pass through unchanged.
 *
 * uv → pixel mapping Y-FLIPS: texture coordinates have their origin
 * bottom-left, the pick buffer's pixel space is top-left (F# parity:
 * SceneHandler's portal resolve).
 *
 * The returned hit's `registry` identifies the id space it belongs to
 * (`undefined` = the caller's own registry).
 */
export async function resolveThroughPortals(
  hit: ResolvedHit | undefined,
): Promise<ResolvedHit | undefined> {
  if (hit === undefined) return undefined;
  const sub = hit.scope.pickSubContext;
  if (sub === undefined || hit.portalUv === undefined || !hit.isPixel) return hit;
  // AVal.force OK: pick-time snapshot in event/pick context.
  const sz = AVal.force(sub.size);
  const ix = Math.floor(hit.portalUv.x * sz.width);
  const iy = Math.floor((1 - hit.portalUv.y) * sz.height);
  if (ix < 0 || iy < 0 || ix >= sz.width || iy >= sz.height) return hit;
  const inner = await sub.pickAt(ix, iy);
  if (inner === undefined) return hit;
  // Keep the OUTER hover latch — the outer pick buffer is what the
  // dispatcher's hover diff reads centre pixels from.
  return { ...inner.hit, registry: inner.registry, hoverPickId: hit.hoverPickId };
}
