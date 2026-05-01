// pixel + BVH merge tests for the spiral hit-test. Mirrors the
// per-offset behaviour of Aardvark.Dom's SceneHandler — see
// `src/scene/picking/spiralHitTest.ts`.

import { describe, expect, it, vi } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Box3d, Intersectable, Trafo3d, V2i, V3d } from "@aardworx/wombat.base";

import { spiralHitTest } from "../src/scene/picking/spiralHitTest.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import type { PickRegion } from "../src/scene/picking/readback.js";
import { SNAP_RADIUS_MAX, SNAP_REGION_SIZE } from "../src/scene/picking/snapOffsets.js";

// Identity view + identity proj over a 200×100 viewport. Cursor at
// (devX=50, devY=50). The same shape pick-bvh.test.ts uses.
const VP = new V2i(200, 100);

function makeRegion(
  centerX: number,
  centerY: number,
  stamps: ReadonlyArray<{ dx: number; dy: number; pickId: number; modeB?: boolean }>,
  expand = false,
): PickRegion {
  const size = SNAP_REGION_SIZE;
  const originX = centerX - SNAP_RADIUS_MAX;
  const originY = centerY - SNAP_RADIUS_MAX;
  const data = new Float32Array(size * size * 4);
  const r = expand ? 1 : 0;
  for (const s of stamps) {
    const slot0 = s.modeB ? -s.pickId : s.pickId;
    for (let ddy = -r; ddy <= r; ddy++) {
      for (let ddx = -r; ddx <= r; ddx++) {
        const lx = (centerX + s.dx + ddx) - originX;
        const ly = (centerY + s.dy + ddy) - originY;
        if (lx < 0 || ly < 0 || lx >= size || ly >= size) continue;
        data[(ly * size + lx) * 4] = slot0;
      }
    }
  }
  return { data, originX, originY, sizeX: size, sizeY: size };
}

interface AcquireOpts {
  pickThrough?: boolean;
  pixelSnapRadius?: number;
  intersectable?: ReturnType<typeof Intersectable.box>;
  mode?: "A" | "B";
}

function acquire(reg: PickRegistry, opts: AcquireOpts = {}): number {
  return reg.acquire({
    handlers: [],
    cursor: undefined,
    pickThrough: opts.pickThrough ?? false,
    active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity),
    proj: AVal.constant(Trafo3d.identity),
    model: AVal.constant(Trafo3d.identity),
    pixelSnapRadius: AVal.constant(opts.pixelSnapRadius ?? SNAP_RADIUS_MAX),
    ...(opts.intersectable !== undefined ? { intersectable: AVal.constant(opts.intersectable) } : {}),
  }, opts.mode ?? "A");
}

const boxNear = Intersectable.box(Box3d.fromMinMax(new V3d(-1, -1, -0.5), new V3d(1, 1, -0.3)));
const boxFar  = Intersectable.box(Box3d.fromMinMax(new V3d(-1, -1,  0.3), new V3d(1, 1,  0.5)));

describe("spiralHitTest pixel + BVH merge", () => {
  it("pixel hit at center, BVH miss → pixel wins", () => {
    const reg = new PickRegistry();
    const id = acquire(reg);
    const region = makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }], true);
    const hit = spiralHitTest(region, { devX: 50, devY: 50 }, reg, Trafo3d.identity, Trafo3d.identity, VP);
    expect(hit).toBeDefined();
    expect(hit!.scope.pickId).toBe(id);
    expect(hit!.isPixel).toBe(true);
  });

  it("pixel hit + BVH hit closer (smaller depth) → BVH wins", () => {
    // Pixel hit at center carrying ndcZ=+0.9 (far). BVH boxNear hits
    // closer (smaller ndcZ). pickPixel = (pixOk && pixDepth ≤ bvhDepth)
    // → false → BVH wins.
    const reg = new PickRegistry();
    const idPx  = acquire(reg);
    const idBvh = acquire(reg, { intersectable: boxNear });
    const region = makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: idPx }], true);
    // Stamp ndcZ at the centre as +0.9 (slot2).
    const lx = 50 - region.originX;
    const ly = 50 - region.originY;
    region.data[(ly * region.sizeX + lx) * 4 + 2] = 0.9;
    const hit = spiralHitTest(region, { devX: 50, devY: 50 }, reg, Trafo3d.identity, Trafo3d.identity, VP);
    expect(hit).toBeDefined();
    expect(hit!.scope.pickId).toBe(idBvh);
    expect(hit!.isPixel).toBe(false);
  });

  it("pixel garbage (single non-zero pixel) → 3-neighbour check rejects, falls through to BVH", () => {
    const reg = new PickRegistry();
    const idPx  = acquire(reg);
    const idBvh = acquire(reg, { intersectable: boxNear });
    // No expand → only one pixel of idPx at the centre.
    const region = makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: idPx }], false);
    const hit = spiralHitTest(region, { devX: 50, devY: 50 }, reg, Trafo3d.identity, Trafo3d.identity, VP);
    expect(hit).toBeDefined();
    expect(hit!.scope.pickId).toBe(idBvh);
    expect(hit!.isPixel).toBe(false);
  });

  it("mode mismatch (registered Mode-A, slot0 negative) → pixel rejected", () => {
    const reg = new PickRegistry();
    const idPx  = acquire(reg, { mode: "A" });
    const idBvh = acquire(reg, { intersectable: boxNear });
    const region = makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: idPx, modeB: true }], true);
    const hit = spiralHitTest(region, { devX: 50, devY: 50 }, reg, Trafo3d.identity, Trafo3d.identity, VP);
    expect(hit).toBeDefined();
    // Pixel rejected on mode mismatch → BVH wins.
    expect(hit!.scope.pickId).toBe(idBvh);
    expect(hit!.isPixel).toBe(false);
  });

  it("pickThrough scope wins by pixel → console.warn fires, scope still dispatched", () => {
    const reg = new PickRegistry();
    const idPx = acquire(reg, { pickThrough: true });
    const region = makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: idPx }], true);
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const hit = spiralHitTest(region, { devX: 50, devY: 50 }, reg, Trafo3d.identity, Trafo3d.identity, VP);
    expect(hit).toBeDefined();
    expect(hit!.scope.pickId).toBe(idPx);
    expect(hit!.isPixel).toBe(true);
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it("pickThrough scope wins by BVH → re-trace excluding pickThrough finds the scope behind", () => {
    const reg = new PickRegistry();
    const idA = acquire(reg, { pickThrough: true, intersectable: boxNear });
    const idB = acquire(reg, { intersectable: boxFar });
    // No pixel stamps → BVH winner. Closest BVH hit is boxNear (idA),
    // which is pickThrough → re-trace excluding pickThrough finds idB.
    const region = makeRegion(50, 50, [], false);
    const hit = spiralHitTest(region, { devX: 50, devY: 50 }, reg, Trafo3d.identity, Trafo3d.identity, VP);
    expect(hit).toBeDefined();
    expect(hit!.scope.pickId).toBe(idB);
    expect(hit!.isPixel).toBe(false);
  });

  it("hover: pixel garbage at centre but registered absId → ResolvedHit not produced (and dispatcher's centre-id capture handled separately)", () => {
    // F# captures `winnerHoverIdValue` at the centre even on a
    // mapping/validation miss; spiralHitTest preserves it on the
    // winning ResolvedHit. With a single (un-validatable) pixel and
    // no BVH, the function returns undefined — matching F# `None`.
    const reg = new PickRegistry();
    const idPx = acquire(reg);
    const region = makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: idPx }], false);
    const hit = spiralHitTest(region, { devX: 50, devY: 50 }, reg, Trafo3d.identity, Trafo3d.identity, VP);
    // No BVH and pixel rejected by 3-neighbour → undefined (matches F# None).
    expect(hit).toBeUndefined();
  });
});
