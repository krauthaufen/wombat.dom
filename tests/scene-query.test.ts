// Reactive scene queries — `createSceneQuery` returns avals tracking
// the registry's BVH + each scope's model trafo. Trafo ticks must
// re-evaluate the query.

import { describe, expect, it } from "vitest";
import { AVal, cval, transact } from "@aardworx/wombat.adaptive";
import {
  Box3d, Intersectable, Ray3d, Trafo3d, V2d, V2i, V3d,
} from "@aardworx/wombat.base";

import { PickRegistry } from "../src/scene/picking/registry.js";
import { createSceneQuery } from "../src/scene/picking/sceneQuery.js";

function unitBox(): import("@aardworx/wombat.base").IIntersectable {
  return Intersectable.box(Box3d.fromMinMax(new V3d(-1, -1, -1), new V3d(1, 1, 1)));
}

function acquireBoxAt(reg: PickRegistry, model: import("@aardworx/wombat.adaptive").aval<Trafo3d>): number {
  return reg.acquire({
    handlers: [], cursor: undefined, pickThrough: false,
    active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity),
    proj: AVal.constant(Trafo3d.identity),
    model: () => model,
    pixelSnapRadius: AVal.constant(1),
    intersectable: AVal.constant(unitBox()),
  });
}

describe("createSceneQuery", () => {
  it("intersect returns the closer cube's HitInfo", () => {
    const reg = new PickRegistry();
    const trafoA = cval(Trafo3d.translation(new V3d(0, 0, 5)));   // closer
    const trafoB = cval(Trafo3d.translation(new V3d(0, 0, 20)));  // farther
    const idA = acquireBoxAt(reg, trafoA);
    const idB = acquireBoxAt(reg, trafoB);
    void idB;

    const q = createSceneQuery(
      reg,
      AVal.constant(Trafo3d.identity),
      AVal.constant(Trafo3d.identity),
      AVal.constant(new V2i(100, 100)),
    );
    const ray = new Ray3d(new V3d(0, 0, -10), new V3d(0, 0, 1));
    const hit = AVal.force(q.intersect(AVal.constant(ray)));
    expect(hit).toBeDefined();
    expect(hit!.pickId).toBe(idA);
  });

  it("ticking the second cube's model re-evaluates the aval and re-picks", () => {
    const reg = new PickRegistry();
    const trafoA = cval(Trafo3d.translation(new V3d(0, 0, 10)));
    const trafoB = cval(Trafo3d.translation(new V3d(0, 0, 20)));
    const idA = acquireBoxAt(reg, trafoA);
    const idB = acquireBoxAt(reg, trafoB);

    const q = createSceneQuery(
      reg,
      AVal.constant(Trafo3d.identity),
      AVal.constant(Trafo3d.identity),
      AVal.constant(new V2i(100, 100)),
    );
    const ray = new Ray3d(new V3d(0, 0, -10), new V3d(0, 0, 1));
    const hitAval = q.intersect(AVal.constant(ray));

    expect(AVal.force(hitAval)!.pickId).toBe(idA);

    // Move B in front of A.
    transact(() => { trafoB.value = Trafo3d.translation(new V3d(0, 0, 2)); });
    expect(AVal.force(hitAval)!.pickId).toBe(idB);
  });

  it("pickAt converts cursor pixel + camera into a ray and dispatches to the BVH", () => {
    const reg = new PickRegistry();
    const trafoA = cval(Trafo3d.translation(new V3d(0, 0, 5)));
    const idA = acquireBoxAt(reg, trafoA);

    // Identity proj wouldn't unproject to anything useful — use a
    // simple perspective-like inverse via Trafo3d.identity here too,
    // but with a viewport of (1, 1) so NDC ≈ pixel and the unproject
    // produces a centered ray pointing along +z.
    const q = createSceneQuery(
      reg,
      AVal.constant(Trafo3d.identity),
      AVal.constant(Trafo3d.identity),
      AVal.constant(new V2i(200, 200)),
    );

    // Cursor at the centre — NDC (0, 0). With identity view/proj that
    // unprojects to a ray running along world +z through origin, which
    // hits cube A at z=5.
    const cursor = AVal.constant(new V2d(99.5, 99.5));
    const hit = AVal.force(q.pickAt(cursor));
    expect(hit).toBeDefined();
    expect(hit!.pickId).toBe(idA);
  });

  it("intersect returns undefined when the BVH is empty", () => {
    const reg = new PickRegistry();
    const q = createSceneQuery(
      reg,
      AVal.constant(Trafo3d.identity),
      AVal.constant(Trafo3d.identity),
      AVal.constant(new V2i(100, 100)),
    );
    const ray = new Ray3d(new V3d(0, 0, -10), new V3d(0, 0, 1));
    expect(AVal.force(q.intersect(AVal.constant(ray)))).toBeUndefined();
  });
});
