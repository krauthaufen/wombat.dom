// Reactive BVH test — pinning Section 3 of the registry refactor.
//
// The PickRegistry now exposes `bvhAval: aval<Bvh<PickId, BvhEntry>>`
// driven by an aset-of-PickObject + `mapA` chain over the
// intersectable AND model trafo avals. World-space bbox stored in
// the BVH (via `transformBox`) means trafo-aval ticks reactively
// invalidate the bbox without registry rebuild — fixes the staleness
// the previous commit only mitigated. Mirror of Aardvark.Dom
// SceneHandler.fs:1444-1468.

import { describe, expect, it } from "vitest";
import { AVal, cval, transact } from "@aardworx/wombat.adaptive";
import { Box3d, Intersectable, Ray3d, Trafo3d, V3d } from "@aardworx/wombat.base";

import { PickRegistry } from "../src/scene/picking/registry.js";

function unitBox(): import("@aardworx/wombat.base").IIntersectable {
  return Intersectable.box(Box3d.fromMinMax(new V3d(-1, -1, -1), new V3d(1, 1, 1)));
}

describe("PickRegistry — reactive BVH (Aardvark.Dom parity)", () => {
  it("trafo aval tick updates the BVH's stored world-space bbox", () => {
    const reg = new PickRegistry();
    const trafo = cval(Trafo3d.identity);
    const intersectable = AVal.constant(unitBox());
    const id = reg.acquire({
      handlers: [], cursor: undefined, pickThrough: false,
      active: AVal.constant(true), view: AVal.constant(Trafo3d.identity),
      proj: AVal.constant(Trafo3d.identity), model: () => (trafo),
      pixelSnapRadius: AVal.constant(1),
      intersectable,
    });

    // Initial BVH: world bbox = local bbox (identity trafo).
    const bvh0 = AVal.force(reg.bvhAval);
    const items0 = [...bvh0.items()];
    expect(items0).toHaveLength(1);
    expect(items0[0]!.key).toBe(id);
    expect(items0[0]!.box.min.x).toBeCloseTo(-1);
    expect(items0[0]!.box.max.x).toBeCloseTo(1);

    // Ray from (0,0,-10) along +z: must HIT the unit box.
    const ray = new Ray3d(new V3d(0, 0, -10), new V3d(0, 0, 1));
    const hit0 = bvh0.closestHit(ray, 0, Number.POSITIVE_INFINITY,
      (_k, e) => e.intersectable.intersects(ray, 0, Number.POSITIVE_INFINITY));
    expect(hit0).toBeDefined();

    // Move the trafo so the box is centred at (10, 0, 0).
    transact(() => { trafo.value = Trafo3d.translation(new V3d(10, 0, 0)); });

    const bvh1 = AVal.force(reg.bvhAval);
    const items1 = [...bvh1.items()];
    expect(items1).toHaveLength(1);
    // World-space bbox must reflect translation.
    expect(items1[0]!.box.min.x).toBeCloseTo(9);
    expect(items1[0]!.box.max.x).toBeCloseTo(11);

    // Centre ray at x=10 still hits via the entry's local-space
    // intersection through the (now-current) trafo.
    const rayShift = new Ray3d(new V3d(10, 0, -10), new V3d(0, 0, 1));
    const hit1 = bvh1.closestHit(rayShift, 0, Number.POSITIVE_INFINITY,
      (_k, entry) => {
        const t = AVal.force(entry.scope.model());
        const localRay = rayShift.transformed(t.inverse());
        return entry.intersectable.intersects(localRay, 0, Number.POSITIVE_INFINITY);
      });
    expect(hit1).toBeDefined();
    // Original ray now MISSES the moved box.
    const hit0b = bvh1.closestHit(ray, 0, Number.POSITIVE_INFINITY,
      (_k, entry) => {
        const t = AVal.force(entry.scope.model());
        const localRay = ray.transformed(t.inverse());
        return entry.intersectable.intersects(localRay, 0, Number.POSITIVE_INFINITY);
      });
    expect(hit0b).toBeUndefined();
  });

  it("intersectable aval swap updates the BVH entry", () => {
    const reg = new PickRegistry();
    const small = Intersectable.box(Box3d.fromMinMax(new V3d(-0.1, -0.1, -0.1), new V3d(0.1, 0.1, 0.1)));
    const big   = Intersectable.box(Box3d.fromMinMax(new V3d(-5, -5, -5), new V3d(5, 5, 5)));
    const it = cval(small);
    reg.acquire({
      handlers: [], cursor: undefined, pickThrough: false,
      active: AVal.constant(true), view: AVal.constant(Trafo3d.identity),
      proj: AVal.constant(Trafo3d.identity), model: () => (AVal.constant(Trafo3d.identity)),
      pixelSnapRadius: AVal.constant(1),
      intersectable: it,
    });

    const items0 = [...AVal.force(reg.bvhAval).items()];
    expect(items0).toHaveLength(1);
    expect(items0[0]!.box.max.x).toBeCloseTo(0.1);

    transact(() => { it.value = big; });
    const items1 = [...AVal.force(reg.bvhAval).items()];
    expect(items1).toHaveLength(1);
    expect(items1[0]!.box.max.x).toBeCloseTo(5);
  });

  it("clear() empties the BVH; subsequent acquires re-populate", () => {
    const reg = new PickRegistry();
    reg.acquire({
      handlers: [], cursor: undefined, pickThrough: false,
      active: AVal.constant(true), view: AVal.constant(Trafo3d.identity),
      proj: AVal.constant(Trafo3d.identity), model: () => (AVal.constant(Trafo3d.identity)),
      pixelSnapRadius: AVal.constant(1),
      intersectable: AVal.constant(unitBox()),
    });
    expect(AVal.force(reg.bvhAval).count).toBe(1);
    reg.clear();
    expect(AVal.force(reg.bvhAval).count).toBe(0);
    reg.acquire({
      handlers: [], cursor: undefined, pickThrough: false,
      active: AVal.constant(true), view: AVal.constant(Trafo3d.identity),
      proj: AVal.constant(Trafo3d.identity), model: () => (AVal.constant(Trafo3d.identity)),
      pixelSnapRadius: AVal.constant(1),
      intersectable: AVal.constant(unitBox()),
    });
    expect(AVal.force(reg.bvhAval).count).toBe(1);
  });
});
