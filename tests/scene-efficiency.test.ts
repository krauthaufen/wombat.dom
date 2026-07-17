// Quantified efficiency accounting: counts + byte estimates per bail
// reason, thresholds gate reporting (3 inefficient objects = silence).

import { beforeEach, describe, expect, it } from "vitest";
import {
  sceneEfficiency, resetEfficiency,
} from "../src/scene/index.js";
import { recordRowBail, recordRowLowered, resetTemplates } from "../src/scene/template.js";

beforeEach(() => { resetEfficiency(); resetTemplates(); });

describe("scene efficiency accounting", () => {
  it("aggregates counts and byte estimates per reason", () => {
    for (let i = 0; i < 1200; i++) recordRowBail("dynamic-uniform-bag");
    for (let i = 0; i < 3; i++) recordRowBail("multi-leaf-subtree");
    for (let i = 0; i < 10; i++) recordRowLowered();
    const r = sceneEfficiency();
    expect(r.rowsLowered).toBe(10);
    const dyn = r.bails.find((b) => b.reason === "dynamic-uniform-bag")!;
    expect(dyn.count).toBe(1200);
    expect(dyn.estBytes).toBe(1200 * 700);
    const multi = r.bails.find((b) => b.reason === "multi-leaf-subtree")!;
    expect(multi.count).toBe(3);
    // sorted by impact
    expect(r.bails[0]!.reason).toBe("dynamic-uniform-bag");
  });

  it("same-effect enrichment names the shared effect id", () => {
    for (let i = 0; i < 800; i++) recordRowBail("per-leaf-effect-scope", "eff-X");
    const r = sceneEfficiency();
    const b = r.bails.find((x) => x.reason === "per-leaf-effect-scope")!;
    expect(b.hint).toContain("SAME effect eff-X");
    expect(b.hint).toContain("800");
  });

  it("mixed effect ids drop the enrichment", () => {
    recordRowBail("per-leaf-effect-scope", "a");
    recordRowBail("per-leaf-effect-scope", "b");
    const r = sceneEfficiency();
    const b = r.bails.find((x) => x.reason === "per-leaf-effect-scope")!;
    expect(b.hint).not.toContain("SAME");
  });
});
