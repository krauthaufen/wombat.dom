// Unit tests for the 5-case pick arbitration (pickArbitrate.ts),
// exercised directly (no dispatcher/DOM). Verifies the merge of the
// GPU argmin winner with one BVH centre ray:
//
//   pixel exactly at cursor + bvh hit  → closer-in-depth wins
//   pixel off-centre        + bvh hit  → bvh
//   no pixel                + bvh hit  → bvh
//   any pixel               + bvh miss → pixel as-is
//   no pixel                + bvh miss → undefined
//
// plus pickThrough re-trace on a BVH winner.

import { describe, expect, it, vi } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Box3d, Intersectable, Trafo3d, V2i, V3d } from "@aardworx/wombat.base";

import { arbitratePick } from "../src/scene/picking/pickArbitrate.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import type { EventHandlers } from "../src/scene/sg.js";
import { noPixel, pixelWinner } from "./pickArgminTestUtil.js";

const VP = new V2i(200, 100);
const POINTER = { devX: 50, devY: 50 };

function noHandlers(): EventHandlers {
  return {};
}

interface AcquireOpts {
  pickThrough?: boolean;
  intersectable?: ReturnType<typeof Intersectable.box>;
}

function acquire(reg: PickRegistry, opts: AcquireOpts = {}): number {
  return reg.acquire({
    handlers: [{ handlers: noHandlers(), local2World: AVal.constant(Trafo3d.identity) }],
    cursor: undefined,
    pickThrough: opts.pickThrough ?? false,
    active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity),
    proj: AVal.constant(Trafo3d.identity),
    model: AVal.constant(Trafo3d.identity),
    pixelSnapRadius: AVal.constant(1),
    ...(opts.intersectable !== undefined ? { intersectable: AVal.constant(opts.intersectable) } : {}),
  });
}

// Near box: z ∈ [0.1, 0.2]. Far box: z ∈ [0.6, 0.7]. Identity view/proj
// ⇒ NDC == world, depth compare is on world z.
const boxNear = Intersectable.box(Box3d.fromMinMax(new V3d(-1, -1, 0.1), new V3d(1, 1, 0.2)));
const boxFar = Intersectable.box(Box3d.fromMinMax(new V3d(-1, -1, 0.6), new V3d(1, 1, 0.7)));

describe("arbitratePick — 5-case merge", () => {
  it("pixel centred, no BVH → pixel wins", () => {
    const reg = new PickRegistry();
    const id = acquire(reg);
    const hit = arbitratePick(pixelWinner(id), POINTER, reg, Trafo3d.identity, Trafo3d.identity, VP);
    expect(hit?.scope.pickId).toBe(id);
    expect(hit?.isPixel).toBe(true);
  });

  it("no pixel, BVH hit → BVH wins", () => {
    const reg = new PickRegistry();
    const id = acquire(reg, { intersectable: boxNear });
    const hit = arbitratePick(noPixel(), POINTER, reg, Trafo3d.identity, Trafo3d.identity, VP);
    expect(hit?.scope.pickId).toBe(id);
    expect(hit?.isPixel).toBe(false);
  });

  it("pixel off-centre + BVH hit → BVH wins", () => {
    const reg = new PickRegistry();
    const idPix = acquire(reg);
    const idBvh = acquire(reg, { intersectable: boxNear });
    // Pixel winner off-centre (dist²=4) on idPix; BVH ray hits idBvh.
    const hit = arbitratePick(
      pixelWinner(idPix, { px: 52, dist2: 4 }),
      POINTER, reg, Trafo3d.identity, Trafo3d.identity, VP,
    );
    expect(hit?.scope.pickId).toBe(idBvh);
    expect(hit?.isPixel).toBe(false);
  });

  it("pixel centred IN FRONT of BVH → pixel wins (depth)", () => {
    const reg = new PickRegistry();
    const idPix = acquire(reg);
    acquire(reg, { intersectable: boxFar });
    // pixel ndcZ (slot2) = 0.0 < far box z 0.6 ⇒ pixel closer.
    const hit = arbitratePick(
      pixelWinner(idPix, { slot2: 0.0 }),
      POINTER, reg, Trafo3d.identity, Trafo3d.identity, VP,
    );
    expect(hit?.scope.pickId).toBe(idPix);
    expect(hit?.isPixel).toBe(true);
  });

  it("pixel centred BEHIND BVH → BVH wins (depth)", () => {
    const reg = new PickRegistry();
    const idPix = acquire(reg);
    const idBvh = acquire(reg, { intersectable: boxNear });
    // pixel ndcZ (slot2) = 0.9 > near box z 0.1 ⇒ BVH closer.
    const hit = arbitratePick(
      pixelWinner(idPix, { slot2: 0.9 }),
      POINTER, reg, Trafo3d.identity, Trafo3d.identity, VP,
    );
    expect(hit?.scope.pickId).toBe(idBvh);
    expect(hit?.isPixel).toBe(false);
  });

  it("pixel + BVH miss → pixel as-is (no occlusion check)", () => {
    const reg = new PickRegistry();
    const id = acquire(reg); // no intersectable ⇒ empty BVH
    const hit = arbitratePick(
      pixelWinner(id, { px: 53, dist2: 9 }),
      POINTER, reg, Trafo3d.identity, Trafo3d.identity, VP,
    );
    expect(hit?.scope.pickId).toBe(id);
    expect(hit?.isPixel).toBe(true);
  });

  it("no pixel + BVH miss → undefined", () => {
    const reg = new PickRegistry();
    acquire(reg); // no intersectable
    const hit = arbitratePick(noPixel(), POINTER, reg, Trafo3d.identity, Trafo3d.identity, VP);
    expect(hit).toBeUndefined();
  });

  it("undefined result (GPU read failed) + BVH hit → BVH-only resolve", () => {
    const reg = new PickRegistry();
    const id = acquire(reg, { intersectable: boxNear });
    const hit = arbitratePick(undefined, POINTER, reg, Trafo3d.identity, Trafo3d.identity, VP);
    expect(hit?.scope.pickId).toBe(id);
    expect(hit?.isPixel).toBe(false);
  });

  it("BVH winner that is pickThrough re-traces to the scope behind", () => {
    const reg = new PickRegistry();
    const idThrough = acquire(reg, { pickThrough: true, intersectable: boxNear });
    const idBehind = acquire(reg, { intersectable: boxFar });
    void idThrough;
    const hit = arbitratePick(noPixel(), POINTER, reg, Trafo3d.identity, Trafo3d.identity, VP);
    expect(hit?.scope.pickId).toBe(idBehind);
    expect(hit?.nextScope?.pickId).toBe(idBehind);
  });

  it("pixel-picked pickThrough is kept with a warning", () => {
    const reg = new PickRegistry();
    const id = acquire(reg, { pickThrough: true });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const hit = arbitratePick(pixelWinner(id), POINTER, reg, Trafo3d.identity, Trafo3d.identity, VP);
    expect(hit?.scope.pickId).toBe(id);
    expect(hit?.isPixel).toBe(true);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});
