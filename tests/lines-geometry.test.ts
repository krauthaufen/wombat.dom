// Sg.Lines / Sg.Points: the instanced shapes are closed, unit-sized and
// index only their own vertices.
import { describe, expect, it } from "vitest";
import { cylinderGeometry, sphereGeometry } from "../src/scene/lines.js";

describe("the unit cylinder of Sg.Lines", () => {
  const g = cylinderGeometry(8);
  it("has a ring at each end and a centre for each cap", () => {
    expect(g.positions.length / 3).toBe(18);
    for (let i = 0; i < 16; i++) {
      expect(Math.hypot(g.positions[i * 3]!, g.positions[i * 3 + 1]!)).toBeCloseTo(1, 6);
      expect(g.positions[i * 3 + 2]).toBe(i < 8 ? 0 : 1);
    }
  });
  it("is closed: sides and two caps, every index a vertex", () => {
    expect(g.indices.length).toBe(8 * 12);
    expect(Math.max(...g.indices)).toBe(17);
    // Every edge is shared by exactly two triangles.
    const edges = new Map<string, number>();
    for (let t = 0; t < g.indices.length; t += 3) {
      for (const [a, b] of [[0, 1], [1, 2], [2, 0]] as const) {
        const i = g.indices[t + a]!, j = g.indices[t + b]!;
        const key = i < j ? `${i}-${j}` : `${j}-${i}`;
        edges.set(key, (edges.get(key) ?? 0) + 1);
      }
    }
    expect([...edges.values()].every((count) => count === 2)).toBe(true);
  });
});

describe("the unit sphere of Sg.Points", () => {
  const g = sphereGeometry(12);
  it("lies on the unit sphere", () => {
    for (let i = 0; i < g.positions.length; i += 3) {
      expect(Math.hypot(g.positions[i]!, g.positions[i + 1]!, g.positions[i + 2]!)).toBeCloseTo(1, 6);
    }
    expect(Math.max(...g.indices)).toBeLessThan(g.positions.length / 3);
  });
});
