// `Sg.Text` with aa "alpha-blending" sets its own depth state inside its tree
// (inner state wins), so a caller's DepthTest / DepthMask on the text used to
// be shadowed. Labels drawn over everything ("always", no depth writes) need
// them to replace the text's own (`sdfTextDepth`).

import { describe, expect, it } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { sdfTextDepth } from "../src/scene/text-sdf.js";

describe("SDF text depth state", () => {
  it("defaults to the projection's test with depth writes", () => {
    const d = sdfTextDepth(undefined);
    expect(["less-equal", "greater-equal"]).toContain(AVal.force(d.depthTest as never));
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
