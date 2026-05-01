// Camera helpers (lookAt / perspective / orthographic / aspect-from-viewport)
// — math + reactivity tests.
//
// Controller behaviour lives in `freefly-controller.test.ts` and
// `orbit-controller.test.ts` against the new FreeFlyController /
// OrbitController API. The old `freeFlyController` /
// `orbitController` factories were replaced wholesale; this file
// keeps only the math/aval shape tests that survived the rewrite.

import { describe, expect, it } from "vitest";
import { AVal, cval, transact } from "@aardworx/wombat.adaptive";
import { V3d } from "@aardworx/wombat.base";

import {
  aspectFromViewport,
  lookAt,
  orthographic,
  perspective,
} from "../src/scene/index.js";

describe("lookAt", () => {
  it("places origin at eye + looks toward target", () => {
    const view = lookAt({
      eye: new V3d(0, 0, 5),
      target: V3d.zero,
      up: new V3d(0, 1, 0),
    });
    const t = AVal.force(view);
    const eyeInView = t.transform(new V3d(0, 0, 5));
    expect(eyeInView.x).toBeCloseTo(0, 6);
    expect(eyeInView.y).toBeCloseTo(0, 6);
    expect(eyeInView.z).toBeCloseTo(0, 6);
    const targetInView = t.transform(V3d.zero);
    expect(targetInView.z).toBeCloseTo(-5, 6);
  });

  it("reactive eye → reactive view", () => {
    const eye = cval(new V3d(0, 0, 5));
    const view = lookAt({ eye, target: V3d.zero, up: new V3d(0, 1, 0) });
    const t1 = AVal.force(view);
    transact(() => { eye.value = new V3d(0, 0, 10); });
    const t2 = AVal.force(view);
    expect(t1).not.toBe(t2);
    const targetInView = t2.transform(V3d.zero);
    expect(targetInView.z).toBeCloseTo(-10, 6);
  });
});

describe("perspective", () => {
  it("maps near plane to NDC z=0 and far plane to z=1 (RH/WebGPU)", () => {
    const p = AVal.force(perspective({
      fovInRadians: Math.PI / 2,
      aspect: 1,
      near: 1,
      far: 100,
    }));
    const near = p.transform(new V3d(0, 0, -1));
    expect(near.z).toBeCloseTo(0, 6);
    const far = p.transform(new V3d(0, 0, -100));
    expect(far.z).toBeCloseTo(1, 6);
  });

  it("aspect changes update the projection reactively", () => {
    const aspect = cval(1);
    const p = perspective({ fovInRadians: Math.PI / 2, aspect, near: 1, far: 100 });
    const t1 = AVal.force(p);
    transact(() => { aspect.value = 2; });
    const t2 = AVal.force(p);
    expect(t1).not.toBe(t2);
  });
});

describe("orthographic", () => {
  it("maps the corner of the box to NDC (-1, -1, 0)", () => {
    const p = AVal.force(orthographic({
      left: -1, right: 1, bottom: -1, top: 1, near: 1, far: 100,
    }));
    const ll = p.transform(new V3d(-1, -1, -1));
    expect(ll.x).toBeCloseTo(-1, 6);
    expect(ll.y).toBeCloseTo(-1, 6);
    expect(ll.z).toBeCloseTo(0, 6);
  });
});

describe("aspectFromViewport", () => {
  it("clamps height to ≥1 to avoid div-by-0", () => {
    const v = cval({ width: 100, height: 0 });
    const aspect = aspectFromViewport(v);
    expect(AVal.force(aspect)).toBe(100);
  });

  it("re-derives the projection when the viewport changes", () => {
    const viewport = cval({ width: 800, height: 400 });
    const proj = perspective({
      fovInRadians: Math.PI / 2,
      aspect: aspectFromViewport(viewport),
      near: 1, far: 100,
    });
    const before = AVal.force(proj);
    transact(() => { viewport.value = { width: 400, height: 800 }; });
    const after = AVal.force(proj);
    expect(before).not.toBe(after);
  });
});
