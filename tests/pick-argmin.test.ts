// Tests for the single-pass GPU argmin pick kernel.
//
// Two slices that need no real GPUDevice:
//   1. WGSL string template — bindings + workgroup size + radius baked in.
//   2. JS reference impl of the argmin — the spec the WGSL mirrors.
// The compute pipeline itself runs only on real hardware (browser test).

import { describe, expect, it } from "vitest";
import {
  argminPickReference,
  buildPickArgminWgsl,
  type PickArgminResult,
} from "../src/scene/picking/pickArgminCompute.js";
import { SNAP_RADIUS_MAX } from "../src/scene/picking/snapOffsets.js";

// Build a pixelAt over a sparse map; out-of-map pixels read as no-hit.
function gridFrom(
  pixels: Map<string, readonly [number, number, number, number]>,
): (x: number, y: number) => readonly [number, number, number, number] {
  return (x, y) => pixels.get(`${x},${y}`) ?? [0, 0, 0, 0];
}

// Fill a 3×3 block of identical slots. (A single pixel works too now —
// the 3×3 neighbour gate was dropped; MSAA ids are resolved upstream.)
function block(
  pixels: Map<string, readonly [number, number, number, number]>,
  cx: number,
  cy: number,
  slots: readonly [number, number, number, number],
): void {
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) pixels.set(`${cx + dx},${cy + dy}`, slots);
  }
}

describe("buildPickArgminWgsl", () => {
  const src = buildPickArgminWgsl();
  it("takes the radius as a runtime uniform (not baked)", () => {
    expect(src).toContain("radius: i32");
    expect(src).not.toContain("const R: i32 = ");
  });
  it("declares the five bindings", () => {
    expect(src).toContain("var pickTex: texture_2d<f32>");
    expect(src).toContain("var<uniform> params: Params");
    expect(src).toContain("var<storage, read> metadata: array<vec2<f32>>");
    expect(src).toContain("var<storage, read_write> bestKey: atomic<u32>");
    expect(src).toContain("var<storage, read_write> result: Result");
  });
  it("does the argmin via a lock-free atomicMin across workgroups", () => {
    expect(src).toContain("@workgroup_size(256)");
    expect(src).toContain("atomicMin(&bestKey");
    expect(src).toContain("fn findMin");
    expect(src).toContain("fn decode");
  });
});

describe("argminPickReference", () => {
  const W = 200;
  const H = 200;
  const CX = 100;
  const CY = 100;
  // metadata[id] = [effectiveRadius, modeSign]
  const meta: Array<readonly [number, number]> = [];
  const setMeta = (id: number, r: number, sign: number) => { meta[id] = [r, sign]; };

  it("returns found=false when the disc is empty", () => {
    const r = argminPickReference(CX, CY, W, H, gridFrom(new Map()), meta);
    expect(r.found).toBe(false);
  });

  it("picks the pixel nearest the cursor", () => {
    const px = new Map<string, readonly [number, number, number, number]>();
    setMeta(7, SNAP_RADIUS_MAX, 1);
    setMeta(9, SNAP_RADIUS_MAX, 1);
    block(px, CX + 3, CY, [7, 0, 0.5, 0]); // nearest filled px at CX+2 → dist² = 4
    block(px, CX + 6, CY, [9, 0, 0.5, 0]); // nearest filled px at CX+5 → dist² = 25
    const r = argminPickReference(CX, CY, W, H, gridFrom(px), meta);
    expect(r.found).toBe(true);
    expect(r.slot0).toBe(7);
    expect(r.px).toBe(CX + 2);
    expect(r.dist2).toBe(4);
  });

  it("rejects pixels beyond their per-id radius and falls to the next valid", () => {
    const px = new Map<string, readonly [number, number, number, number]>();
    setMeta(7, 0, 1); // radius 0 → only an exact-cursor pixel is valid
    setMeta(9, SNAP_RADIUS_MAX, 1);
    block(px, CX + 3, CY, [7, 0, 0, 0]); // nearest filled px at CX+2, dist² = 4 > 0
    block(px, CX + 6, CY, [9, 0, 0, 0]); // nearest filled px at CX+5, dist² = 25 ≤ 16²
    const r = argminPickReference(CX, CY, W, H, gridFrom(px), meta);
    expect(r.found).toBe(true);
    expect(r.slot0).toBe(9);
  });

  it("treats effectiveRadius<0 (inactive/noEvents) as never-valid", () => {
    const px = new Map<string, readonly [number, number, number, number]>();
    setMeta(7, -1, 1); // inactive
    setMeta(9, SNAP_RADIUS_MAX, 1);
    block(px, CX + 2, CY, [7, 0, 0, 0]); // nearer but inactive
    block(px, CX + 5, CY, [9, 0, 0, 0]);
    const r = argminPickReference(CX, CY, W, H, gridFrom(px), meta);
    expect(r.found).toBe(true);
    expect(r.slot0).toBe(9);
  });

  it("rejects a pixel whose sign disagrees with its registered mode", () => {
    const px = new Map<string, readonly [number, number, number, number]>();
    setMeta(7, SNAP_RADIUS_MAX, 1); // registered Mode-A (+)
    block(px, CX + 2, CY, [-7, 1, 2, 3]); // pixel is negative → Mode-B layout, mismatch
    const r = argminPickReference(CX, CY, W, H, gridFrom(px), meta);
    expect(r.found).toBe(false);
  });

  it("accepts an isolated pixel (no 3×3 guard — MSAA handled upstream)", () => {
    const px = new Map<string, readonly [number, number, number, number]>();
    setMeta(7, SNAP_RADIUS_MAX, 1);
    px.set(`${CX + 2},${CY}`, [7, 0, 0, 0]); // single pixel, no neighbours
    const r = argminPickReference(CX, CY, W, H, gridFrom(px), meta);
    expect(r.found).toBe(true);
    expect(r.slot0).toBe(7);
    expect(r.px).toBe(CX + 2);
    expect(r.dist2).toBe(4);
  });

  it("reports centerSlot0 for hover even when the center pixel is invalid", () => {
    const px = new Map<string, readonly [number, number, number, number]>();
    setMeta(7, -1, 1); // inactive → no valid winner
    block(px, CX, CY, [7, 0, 0, 0]); // sits under the cursor
    const r = argminPickReference(CX, CY, W, H, gridFrom(px), meta);
    expect(r.found).toBe(false);
    expect(r.centerSlot0).toBe(7);
  });

  it("carries Mode-B slots through verbatim", () => {
    const px = new Map<string, readonly [number, number, number, number]>();
    setMeta(4, SNAP_RADIUS_MAX, -1); // Mode-B
    block(px, CX, CY, [-4, 1.5, -2.5, 3.5]); // slots 1/2/3 = PickViewPosition
    const r = argminPickReference(CX, CY, W, H, gridFrom(px), meta);
    expect(r.found).toBe(true);
    const out: PickArgminResult = r;
    expect(out.slot0).toBe(-4);
    expect(out.slot1).toBe(1.5);
    expect(out.slot2).toBe(-2.5);
    expect(out.slot3).toBe(3.5);
  });
});
