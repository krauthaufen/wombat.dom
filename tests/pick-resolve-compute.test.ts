// Tests for the MSAA pickId resolve compute pass.
//
// We cover two slices that don't need a real GPUDevice:
//
//   1. WGSL string template — sampleCount baked in, workgroup size,
//      storage format, multisampled binding declared correctly.
//   2. JS reference impl of the majority vote — exhaustive cases.
//
// The actual compute pipeline only runs on real hardware; document
// in docs/FUTURE.md.

import { describe, expect, it } from "vitest";
import {
  buildPickResolveWgsl,
  majorityVoteReference,
  PICK_RESOLVE_WORKGROUP_X,
  PICK_RESOLVE_WORKGROUP_Y,
} from "../src/scene/picking/pickResolveCompute.js";

describe("buildPickResolveWgsl", () => {
  it("rejects sampleCount < 2", () => {
    expect(() => buildPickResolveWgsl(1)).toThrow();
    expect(() => buildPickResolveWgsl(0)).toThrow();
  });

  for (const n of [2, 4, 8]) {
    it(`generates a resolve WGSL with sampleCount=${n} baked in`, () => {
      const src = buildPickResolveWgsl(n);
      // sampleCount baked into the constant + the local array sizes.
      expect(src).toContain(`const SAMPLE_COUNT: u32 = ${n}u;`);
      expect(src).toContain(`array<f32, ${n}>`);
      expect(src).toContain(`array<u32, ${n}>`);
      // Workgroup size constants used unchanged.
      expect(src).toContain(`@workgroup_size(${PICK_RESOLVE_WORKGROUP_X}, ${PICK_RESOLVE_WORKGROUP_Y}, 1)`);
      // Storage texture format.
      expect(src).toContain("texture_storage_2d<rgba32float, write>");
      // Multisampled source.
      expect(src).toContain("texture_multisampled_2d<f32>");
      // Bitcast-based equality (sign-aware integer match).
      expect(src).toContain("bitcast<u32>");
    });
  }

  it("workgroup size is 8x8", () => {
    expect(PICK_RESOLVE_WORKGROUP_X).toBe(8);
    expect(PICK_RESOLVE_WORKGROUP_Y).toBe(8);
  });
});

describe("majorityVoteReference", () => {
  const samples = (...xs: number[]): Float32Array => Float32Array.from(xs);

  it("all-agree: returns sample 0", () => {
    const r = majorityVoteReference(samples(7, 7, 7, 7), 4);
    expect(r.winnerIdx).toBe(0);
    expect(r.winnerValue).toBe(7);
  });

  it("simple 3-1 majority: returns first occurrence of the majority", () => {
    // sample 0 = 5, samples 1..3 = 13. Majority is 13, first at idx 1.
    const r = majorityVoteReference(samples(5, 13, 13, 13), 4);
    expect(r.winnerIdx).toBe(1);
    expect(r.winnerValue).toBe(13);
  });

  it("2-2 tie: lowest sample index wins", () => {
    // First-occurrence rule: value at sample 0 ties on count with
    // value at sample 2; winner is sample 0.
    const r = majorityVoteReference(samples(7, 7, 13, 13), 4);
    expect(r.winnerIdx).toBe(0);
    expect(r.winnerValue).toBe(7);
  });

  it("two pairs reversed: still picks lowest first-occurrence", () => {
    const r = majorityVoteReference(samples(13, 13, 7, 7), 4);
    expect(r.winnerIdx).toBe(0);
    expect(r.winnerValue).toBe(13);
  });

  it("four distinct values: sample 0 wins (all tied at 1)", () => {
    const r = majorityVoteReference(samples(1, 2, 3, 4), 4);
    expect(r.winnerIdx).toBe(0);
    expect(r.winnerValue).toBe(1);
  });

  it("sign-aware: positive and negative pickIds with same magnitude are distinct", () => {
    // Mode-A pickId 7 (=> +7) vs Mode-B pickId 7 (=> -7).
    // Three -7s should beat one +7.
    const r = majorityVoteReference(samples(7, -7, -7, -7), 4);
    expect(r.winnerIdx).toBe(1);
    expect(r.winnerValue).toBe(-7);
  });

  it("works for sampleCount = 2", () => {
    const r = majorityVoteReference(samples(11, 22), 2);
    expect(r.winnerIdx).toBe(0);
    expect(r.winnerValue).toBe(11);
  });

  it("works for sampleCount = 8 with 5-3 majority", () => {
    const r = majorityVoteReference(samples(1, 1, 2, 2, 2, 2, 2, 1), 8);
    expect(r.winnerIdx).toBe(2);
    expect(r.winnerValue).toBe(2);
  });

  it("zero (no-hit) majority is preserved", () => {
    const r = majorityVoteReference(samples(0, 0, 0, 42), 4);
    expect(r.winnerIdx).toBe(0);
    expect(r.winnerValue).toBe(0);
  });

  it("throws when samples shorter than sampleCount", () => {
    expect(() => majorityVoteReference(samples(1, 2), 4)).toThrow();
  });
});
