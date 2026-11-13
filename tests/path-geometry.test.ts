// `Sg.Path`'s fill geometry (scene/path.ts): curve triangles for curved
// edges (exact at every zoom), fill rules, SVG d against segments, and
// an empty path drawing nothing. Pixels in tests-browser/path.browser.test.ts.

import { describe, expect, it } from "vitest";
import { ArcSegment, LineSegment, V2d, VERTEX_KIND_ARC, VERTEX_KIND_BEZIER2 } from "@aardworx/wombat.base";
import { pathGeometry } from "../src/scene/path.js";

const kinds = (g: ReturnType<typeof pathGeometry>): number[] => {
  const ks = new Set<number>();
  for (let i = 0; i < g.indexCount; i++) ks.add(g.klmKinds[g.indices[i]! * 4 + 3]!);
  return [...ks].sort();
};

describe("path fill geometry", () => {
  it("an arc band: interior triangles plus arc triangles", () => {
    const g = pathGeometry("M 10 0 A 10 10 0 0 1 0 10 L 0 8 A 8 8 0 0 0 8 0 Z");
    expect(g.indexCount).toBeGreaterThan(0);
    expect(kinds(g)).toContain(VERTEX_KIND_ARC);
  });

  it("a quadratic: a bezier triangle", () => {
    expect(kinds(pathGeometry("M 0 0 Q 5 10 10 0 Z"))).toContain(VERTEX_KIND_BEZIER2);
  });

  it("segments and the same path as SVG d give the same geometry", () => {
    const d = pathGeometry("M 1 0 A 1 1 0 0 1 0 1 L 0 0 Z");
    const s = pathGeometry([
      ArcSegment.circular(new V2d(0, 0), 1, 0, Math.PI / 2),
      new LineSegment(new V2d(0, 1), new V2d(0, 0)),
      new LineSegment(new V2d(0, 0), new V2d(1, 0)),
    ]);
    expect(s.indexCount).toBe(d.indexCount);
    expect(kinds(s)).toEqual(kinds(d));
  });

  it("fill rules: a same-winding hole is filled under nonzero, empty under evenodd", () => {
    const sq = "M 0 0 L 10 0 L 10 10 L 0 10 Z M 3 3 L 7 3 L 7 7 L 3 7 Z";
    const area = (rule: "nonzero" | "evenodd"): number => {
      const g = pathGeometry(sq, rule); let a = 0;
      for (let t = 0; t < g.indexCount; t += 3) {
        const p = (k: number) => [g.positions[g.indices[t + k]! * 2]!, g.positions[g.indices[t + k]! * 2 + 1]!];
        const [a0, a1, a2] = [p(0), p(1), p(2)];
        // interior triangles only (ribbons have no area here, curves none in a polygon)
        a += Math.abs((a1[0]! - a0[0]!) * (a2[1]! - a0[1]!) - (a2[0]! - a0[0]!) * (a1[1]! - a0[1]!)) / 2;
      }
      return a;
    };
    expect(area("nonzero")).toBeCloseTo(100, 6);
    expect(area("evenodd")).toBeCloseTo(84, 6);
  });

  it("an empty path draws nothing (one dummy vertex, count 0)", () => {
    for (const p of ["", [] as const]) {
      const g = pathGeometry(p);
      expect(g.indexCount).toBe(0);
      expect(g.positions.length).toBeGreaterThan(0);
    }
  });
});
