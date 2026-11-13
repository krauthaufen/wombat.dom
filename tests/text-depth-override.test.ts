// `Sg.Text` with aa "alpha-blending" sets its own depth state inside its tree
// (inner state wins), so a caller's DepthTest / DepthMask on the text used to
// be shadowed. Labels drawn over everything ("always", no depth writes) need
// them to replace the text's own (`sdfTextDepth`).

import { describe, expect, it } from "vitest";
import { AVal, cval, transact } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";
import { sdfTextDepth } from "../src/scene/text-sdf.js";
import { depthByProjection } from "../src/scene/constructors.js";
import { TraversalState } from "../src/scene/traversalState.js";

describe("SDF text depth state", () => {
  it("defaults to the projection's test with depth writes", () => {
    const d = sdfTextDepth(undefined);
    expect(d.depthTest).toBe(depthByProjection("less-equal"));
    expect(d.depthMask).toBe(true);
    expect(d.scope).toBeUndefined();
  });

  it("takes the caller's DepthTest and DepthMask out of the wrapping scope", () => {
    const d = sdfTextDepth({ DepthTest: "always", DepthMask: false, NoEvents: true } as never);
    expect(d.depthTest).toBe("always");
    expect(d.depthMask).toBe(false);
    expect(d.scope).toEqual({ NoEvents: true });
  });
});

describe("a depth test that follows the projection", () => {
  const standard = Trafo3d.perspectiveProjectionRH(-1, 1, -1, 1, 0.1, 100);
  // reversed-Z, finite and infinite far plane
  const reversed = Trafo3d.perspectiveProjectionReversedRH(-1, 1, -1, 1, 0.1, 100);
  const reversedInf = Trafo3d.perspectiveProjectionReversedRH(-1, 1, -1, 1, 0.1, Infinity);
  const under = (proj: Trafo3d) =>
    TraversalState.empty.withCamera(AVal.constant(Trafo3d.identity), AVal.constant(proj));

  it("is the given test under a standard projection, flipped under a reversed one", () => {
    const t = depthByProjection("less-equal");
    expect(AVal.force(under(standard).pushDepthTest(t).depthTest)).toBe("less-equal");
    expect(AVal.force(under(reversed).pushDepthTest(t).depthTest)).toBe("greater-equal");
    expect(AVal.force(under(reversedInf).pushDepthTest(t).depthTest)).toBe("greater-equal");
    expect(AVal.force(under(reversed).pushDepthTest(depthByProjection("greater")).depthTest)).toBe("less");
    expect(AVal.force(under(reversed).pushDepthTest(depthByProjection("always")).depthTest)).toBe("always");
  });

  it("is the given test with no projection (an offscreen scene's default)", () => {
    expect(AVal.force(TraversalState.empty.pushDepthTest(depthByProjection("less-equal")).depthTest)).toBe("less-equal");
  });

  it("is the state's projection's, follows it, and is shared by scopes under it", () => {
    const proj = cval(standard);
    const s = TraversalState.empty.withCamera(AVal.constant(Trafo3d.identity), proj);
    const a = s.pushDepthTest(depthByProjection("less-equal")).depthTest;
    const b = s.pushDepthTest(depthByProjection("less-equal")).depthTest;
    expect(b).toBe(a);
    expect(AVal.force(a)).toBe("less-equal");
    transact(() => { proj.value = reversed; });
    expect(AVal.force(a)).toBe("greater-equal");
  });
});
