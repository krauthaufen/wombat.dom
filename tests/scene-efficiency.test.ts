// Quantified efficiency accounting: counts + byte estimates per bail
// reason, thresholds gate reporting (3 inefficient objects = silence).

import { beforeEach, describe, expect, it, vi } from "vitest";
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

  it("aggregates call-site locations per reason, heaviest first", () => {
    for (let i = 0; i < 50; i++) recordRowBail("dynamic-uniform-bag", undefined, "App.fs:947");
    for (let i = 0; i < 9; i++) recordRowBail("dynamic-uniform-bag", undefined, "App.fs:12");
    recordRowBail("dynamic-uniform-bag"); // untagged: counted, no source
    const b = sceneEfficiency().bails.find((x) => x.reason === "dynamic-uniform-bag")!;
    expect(b.count).toBe(60);
    expect(b.sources).toEqual([
      { loc: "App.fs:947", count: 50 },
      { loc: "App.fs:12", count: 9 },
    ]);
  });

  it("caps distinct locations (existing sites keep counting)", () => {
    for (let i = 0; i < 40; i++) recordRowBail("multi-leaf-subtree", undefined, `F.fs:${i}`);
    for (let i = 0; i < 5; i++) recordRowBail("multi-leaf-subtree", undefined, "F.fs:0");
    const b = sceneEfficiency().bails.find((x) => x.reason === "multi-leaf-subtree")!;
    expect(b.count).toBe(45);
    expect(b.sources.length).toBe(16);
    expect(b.sources[0]).toEqual({ loc: "F.fs:0", count: 6 });
  });

  it("the thresholded console line names the top call sites", () => {
    vi.useFakeTimers();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      // 1500 × 700B ≈ 1.05MB — crosses BOTH thresholds.
      for (let i = 0; i < 1500; i++) recordRowBail("dynamic-uniform-bag", undefined, "App.fs:947");
      vi.advanceTimersByTime(1100);
      expect(warn).toHaveBeenCalledTimes(1);
      const line = warn.mock.calls[0]![0] as string;
      expect(line).toContain("1500 scene items");
      expect(line).toContain("— at App.fs:947 ×1500");
    } finally {
      warn.mockRestore();
      vi.useRealTimers();
    }
  });
});
