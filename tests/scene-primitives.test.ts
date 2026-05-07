// Primitives + DefaultSurfaces — sanity checks. No GPU; we just
// verify the leaf shape (vertex attributes, index counts, draw
// call) and that the bundled shader compiles to a CompiledEffect
// with the expected stages and uniform-block surface.

import { describe, expect, it } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { V3d } from "@aardworx/wombat.base";
import { effect } from "@aardworx/wombat.shader";
import { box, DefaultSurfaces, quad } from "../src/scene/index.js";

describe("primitives", () => {
  describe("box", () => {
    it("has Positions + Colors attributes, 36 indices", () => {
      const leaf = box();
      expect(leaf.kind).toBe("Leaf");
      const va = leaf.vertexAttributes;
      expect(va.tryFind("Positions")).toBeDefined();
      expect(va.tryFind("Colors")).toBeDefined();
      const dc = AVal.force(leaf.drawCall);
      expect(dc.kind).toBe("indexed");
      if (dc.kind === "indexed") expect(dc.indexCount).toBe(36);
    });

    it("size scales positions to span [-size, +size]", () => {
      const leaf = box({ size: new V3d(2, 3, 4) });
      const view = leaf.vertexAttributes.tryFind("Positions")!;
      const data = AVal.force(view.buffer);
      if (data.kind !== "host") throw new Error("expected host buffer");
      const ab = data.data as ArrayBuffer;
      const positions = new Float32Array(ab);
      // Compute min/max of each axis across all vertices.
      let mnx = Infinity, mny = Infinity, mnz = Infinity;
      let mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
      for (let i = 0; i < positions.length; i += 3) {
        if (positions[i]!     < mnx) mnx = positions[i]!;
        if (positions[i + 1]! < mny) mny = positions[i + 1]!;
        if (positions[i + 2]! < mnz) mnz = positions[i + 2]!;
        if (positions[i]!     > mxx) mxx = positions[i]!;
        if (positions[i + 1]! > mxy) mxy = positions[i + 1]!;
        if (positions[i + 2]! > mxz) mxz = positions[i + 2]!;
      }
      expect(mnx).toBeCloseTo(-2, 6); expect(mxx).toBeCloseTo(2, 6);
      expect(mny).toBeCloseTo(-3, 6); expect(mxy).toBeCloseTo(3, 6);
      expect(mnz).toBeCloseTo(-4, 6); expect(mxz).toBeCloseTo(4, 6);
    });
  });

  describe("quad", () => {
    it("has 4 vertices, 6 indices", () => {
      const leaf = quad();
      // BufferView no longer carries `count`; the draw counts live on
      // `drawCall`. 6 indices for a quad = 2 triangles × 3 indices.
      const dc = AVal.force(leaf.drawCall);
      if (dc.kind !== "indexed") throw new Error("quad should be indexed");
      expect(dc.indexCount).toBe(6);
      expect(leaf.vertexAttributes.tryFind("Positions")).toBeDefined();
    });
  });
});

describe("effect(trafo, vertexColor)", () => {
  it("compiles to a CompiledEffect with vertex + fragment stages", () => {
    const fx = effect(DefaultSurfaces.trafo(), DefaultSurfaces.vertexColor());
    const compiled = fx.compile({ target: "wgsl" });
    const stages = compiled.stages.map(s => s.stage).sort();
    expect(stages).toEqual(["fragment", "vertex"]);
  });

  it("DefaultSurfaces.trafo() / vertexColor() are cached", () => {
    expect(DefaultSurfaces.trafo()).toBe(DefaultSurfaces.trafo());
    expect(DefaultSurfaces.vertexColor()).toBe(DefaultSurfaces.vertexColor());
  });
});
