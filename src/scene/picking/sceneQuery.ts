// Reactive scene queries — toolkit-level "what's currently under
// this point/ray?" answers as avals. Backed by the registry's
// reactive BVH, so queries re-fire when any scope's intersectable,
// model trafo, active flag, or noEvents flag ticks.
//
// Pixel-pick is intentionally NOT part of this API — it's async
// (framebuffer readback) while every other adaptive value here is
// sync. For cursor-based "tooltip / ghost-cursor / snap-to-geometry"
// consumers a live BVH ray-trace is exactly what's wanted.
//
// AVal policy: inside `AVal.custom`'s body we use `getValue(token)`
// only — every dependency must enter the reader's set so subsequent
// ticks re-fire the query.

import { AVal, type aval } from "@aardworx/wombat.adaptive";
import { Ray3d, Trafo3d, V2d, V2i, V3d, V4d } from "@aardworx/wombat.base";

import type { LeafPickScope, PickRegistry, PickId } from "./registry.js";

export interface HitInfo {
  readonly pickId: PickId;
  readonly worldPos: V3d;
  readonly viewPos: V3d;
  readonly worldNormal: V3d;
  /** 0 for BVH-only paths. */
  readonly partIndex: number;
  readonly scope: LeafPickScope;
}

export interface SceneQuery {
  /**
   * Closest BVH-ray intersection. Pixel-pick is not part of this API
   * (async-only); for cursor-based queries the result is identical to
   * the live BVH ray-trace, which is exactly what tooltip /
   * ghost-cursor / snap-to-geometry consumers want.
   */
  intersect(ray: aval<Ray3d>): aval<HitInfo | undefined>;

  /** Convenience: turn a cursor pixel + camera into a ray, then call `intersect`. */
  pickAt(cursorPixel: aval<V2d>): aval<HitInfo | undefined>;
}

export function createSceneQuery(
  registry: PickRegistry,
  view: aval<Trafo3d>,
  proj: aval<Trafo3d>,
  viewportSize: aval<V2i>,
): SceneQuery {
  const intersect = (rayAval: aval<Ray3d>): aval<HitInfo | undefined> =>
    AVal.custom<HitInfo | undefined>((token) => {
      const ray = rayAval.getValue(token);
      const bvh = registry.bvhAval.getValue(token);
      const v = view.getValue(token);
      if (bvh.count === 0) return undefined;

      const vFwd = v.forward;
      const vBwd = v.backward;

      let bestT = Number.POSITIVE_INFINITY;
      let bestHit: HitInfo | undefined;
      for (const item of bvh.items()) {
        const scope = item.value.scope;
        if (!scope.active.getValue(token)) continue;
        if (scope.noEvents !== undefined && scope.noEvents.getValue(token)) continue;
        const trafo = scope.model().getValue(token);
        const localRay = ray.transformed(trafo.inverse());
        const hit = item.value.intersectable.intersects(localRay, 0, bestT);
        if (hit !== undefined && hit.t < bestT) {
          bestT = hit.t;
          const worldPos = trafo.transformPos(hit.point);
          const viewPos = vFwd.transformPos(worldPos);
          const worldN = transposedTransformDir(trafo.backward, hit.normal);
          const worldNormal = normalize(worldN);
          bestHit = {
            pickId: scope.pickId,
            worldPos,
            viewPos,
            worldNormal,
            partIndex: 0,
            scope,
          };
          void viewPos; void vBwd;
        }
      }
      return bestHit;
    });

  const pickAt = (cursorPixel: aval<V2d>): aval<HitInfo | undefined> => {
    const rayAval = AVal.zip(cursorPixel, view, proj, viewportSize).map(
      (px, v, p, size) => {
        const sX = size.x;
        const sY = size.y;
        if (sX <= 0 || sY <= 0) {
          return new Ray3d(V3d.zero, new V3d(0, 0, 1));
        }
        const ndcX = (2 * (px.x + 0.5) / sX) - 1;
        const ndcY = 1 - (2 * (px.y + 0.5) / sY);
        const pBwd = p.backward;
        const vBwd = v.backward;
        const near = unprojClipToWorld(ndcX, ndcY, -1, pBwd, vBwd);
        const far  = unprojClipToWorld(ndcX, ndcY,  1, pBwd, vBwd);
        return Ray3d.fromPoints(near, far);
      },
    );
    return intersect(rayAval);
  };

  return { intersect, pickAt };
}

// ---------------------------------------------------------------------------
// Math helpers — kept local to avoid a public re-export from spiralHitTest.
// ---------------------------------------------------------------------------

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

function transposedTransformDir(m: import("@aardworx/wombat.base").M44d, d: V3d): V3d {
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
