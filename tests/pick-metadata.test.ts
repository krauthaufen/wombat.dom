// Tests for the pick metadata buffer's pure logic — the upload planner
// (coalesced runs + full-upload threshold) and the effectiveRadius fold.
// The GPU buffer maintenance itself needs a device (browser test).

import { describe, expect, it } from "vitest";
import {
  effectiveRadius,
  planMetadataUploads,
} from "../src/scene/picking/pickMetadata.js";
import { PICK_SNAP_RADIUS } from "../src/scene/picking/pickArgminCompute.js";

describe("effectiveRadius", () => {
  it("folds inactive / noEvents to -1", () => {
    expect(effectiveRadius(false, false, 8)).toBe(-1);
    expect(effectiveRadius(true, true, 8)).toBe(-1);
  });
  it("clamps to [0, PICK_SNAP_RADIUS]", () => {
    expect(effectiveRadius(true, false, -3)).toBe(0);
    expect(effectiveRadius(true, false, 999)).toBe(PICK_SNAP_RADIUS);
    expect(effectiveRadius(true, false, 5)).toBe(5);
  });
});

describe("planMetadataUploads", () => {
  it("returns nothing for an empty dirty set", () => {
    expect(planMetadataUploads([], 100, 0.5)).toEqual([]);
  });

  it("coalesces contiguous ids into one run", () => {
    expect(planMetadataUploads([3, 4, 5, 6], 100, 0.5)).toEqual([
      { startId: 3, countIds: 4 },
    ]);
  });

  it("splits disjoint ids into separate runs", () => {
    expect(planMetadataUploads([3, 4, 10, 11, 12, 50], 100, 0.5)).toEqual([
      { startId: 3, countIds: 2 },
      { startId: 10, countIds: 3 },
      { startId: 50, countIds: 1 },
    ]);
  });

  it("sorts before coalescing", () => {
    expect(planMetadataUploads([6, 3, 5, 4], 100, 0.5)).toEqual([
      { startId: 3, countIds: 4 },
    ]);
  });

  it("uploads the whole buffer once past the fraction threshold", () => {
    // 6 dirty of size 10, fraction 0.5 → 6 ≥ 5 → full upload.
    expect(planMetadataUploads([1, 2, 3, 4, 5, 6], 10, 0.5)).toEqual([
      { startId: 0, countIds: 10 },
    ]);
  });

  it("stays in run mode just under the threshold", () => {
    // 4 dirty of size 10, fraction 0.5 → 4 < 5 → coalesced runs.
    const runs = planMetadataUploads([1, 2, 3, 9], 10, 0.5);
    expect(runs).toEqual([
      { startId: 1, countIds: 3 },
      { startId: 9, countIds: 1 },
    ]);
  });
});

describe("PickMetadata — dynamic scopes are re-read only when they change", () => {
  it("a flush re-reads only the scopes whose inputs were marked", async () => {
    const { PickMetadata } = await import("../src/scene/picking/pickMetadata.js");
    const { AVal, cval, transact } = await import("@aardworx/wombat.adaptive");
    const { vi } = await import("vitest");
    let writes = 0;
    (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage ??= { STORAGE: 128, COPY_DST: 8 };
    const device = {
      createBuffer: () => ({ destroy() {} }),
      queue: { writeBuffer: () => { writes++; } },
    } as unknown as GPUDevice;
    const pm = new PickMetadata(device);
    const flags = Array.from({ length: 1000 }, () => cval(false));
    flags.forEach((a, i) => pm.register({
      pickId: i + 1, active: a, pixelSnapRadius: AVal.constant(4),
    } as never, "A"));
    pm.flush();
    const reads = vi.spyOn(PickMetadata.prototype as never as { computeEffR: () => number }, "computeEffR");
    writes = 0;
    pm.flush(); // nothing changed: nothing re-read, nothing written
    expect(reads).toHaveBeenCalledTimes(0);
    expect(writes).toBe(0);
    transact(() => { flags[41]!.value = true; });
    pm.flush(); // one switch flipped: that scope only
    expect(reads).toHaveBeenCalledTimes(1);
    expect(writes).toBe(1);
    pm.flush();
    expect(reads).toHaveBeenCalledTimes(1);
    reads.mockRestore();
  });
});
