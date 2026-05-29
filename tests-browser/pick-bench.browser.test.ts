// Benchmark: old spiral pick vs new GPU-argmin pick, on a real GPU.
//
// Compares, per simulated pointermove:
//   old_cpu    — spiralHitTest over a 33×33 region. Its per-move cost is
//                dominated by the O(N) cull-set build (getIntersectingFrustum
//                + forcing every scope's model trafo), which runs even when
//                a pixel wins at the centre offset (the per-scope snap-radius
//                gate prunes the actual BVH ray tests to a few offsets).
//   new_cpu    — arbitratePick (decode winner + ONE bvh.closestHit ray
//                + 5-case arbitration), given a precomputed argmin result
//   new_total  — the full new per-move path: argmin compute dispatch +
//                40-byte readback + arbitratePick (incl. GPU round-trip)
//
// Two scenes: a dense BVH under the cursor (the worst case the new path
// targets) and a plain pixel hit (common case). Prints a table; asserts
// only a loose, non-flaky win in the BVH-heavy case.
//
// NOTE: timings are indicative (one machine, warm caches), not a
// contract. The structural facts are: readback shrinks 33×33 rgba32f
// (~25 KB) → 40 B, and the CPU cost drops from O(N) (cull build) to
// O(log N) (one tree-pruned ray) per move.

import { beforeAll, afterAll, describe, expect, it } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Box3d, Intersectable, Trafo3d, V2i, V3d } from "@aardworx/wombat.base";

import { arbitratePick } from "../src/scene/picking/pickArbitrate.js";
import {
  createPickArgminCompute,
  type PickArgminCompute,
  type PickArgminResult,
} from "../src/scene/picking/pickArgminCompute.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import { spiralHitTest } from "../src/scene/picking/spiralHitTest.js";
import type { PickRegion } from "../src/scene/picking/readback.js";
import { SNAP_RADIUS_MAX, SNAP_REGION_SIZE } from "../src/scene/picking/snapOffsets.js";
import { requestRealDevice } from "./_realGpu.js";

const W = 200, H = 100;
const CX = 100, CY = 50;
const VP = new V2i(W, H);
const ID_VIEW = Trafo3d.identity;

// Cluster N small boxes inside the cursor's pick-disc frustum cone, at
// staggered depths, so BOTH the spiral's cull set and closestHit's
// candidate set are large (the pathological "many proxies under the
// cursor" case the registry comment flagged).
function buildRegistry(n: number): PickRegistry {
  const reg = new PickRegistry();
  for (let i = 0; i < n; i++) {
    // deterministic jitter in NDC≈world units (identity view/proj)
    const jx = ((i * 2654435761) % 1000) / 1000 - 0.5; // [-0.5,0.5)
    const jy = ((i * 40503) % 1000) / 1000 - 0.5;
    const cx = 0.005 + jx * 0.06;
    const cy = 0.01 + jy * 0.06;
    const z = 0.1 + (i / n) * 0.8;
    const box = Intersectable.box(Box3d.fromMinMax(new V3d(cx - 0.04, cy - 0.04, z), new V3d(cx + 0.04, cy + 0.04, z + 0.02)));
    reg.acquire({
      handlers: [{ handlers: {}, local2World: AVal.constant(Trafo3d.identity) }],
      cursor: undefined,
      pickThrough: false,
      active: AVal.constant(true),
      view: AVal.constant(ID_VIEW),
      proj: AVal.constant(ID_VIEW),
      model: AVal.constant(Trafo3d.identity),
      pixelSnapRadius: AVal.constant(1),
      intersectable: AVal.constant(box),
    });
  }
  // Force the BVH once so neither path pays the build cost in the loop.
  AVal.force(reg.bvhAval);
  return reg;
}

function emptyRegion(): PickRegion {
  const sz = SNAP_REGION_SIZE;
  return { data: new Float32Array(sz * sz * 4), originX: CX - SNAP_RADIUS_MAX, originY: CY - SNAP_RADIUS_MAX, sizeX: sz, sizeY: sz };
}
function pixelRegion(id: number): PickRegion {
  const sz = SNAP_REGION_SIZE;
  const data = new Float32Array(sz * sz * 4);
  const ox = CX - SNAP_RADIUS_MAX, oy = CY - SNAP_RADIUS_MAX;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    const lx = CX + dx - ox, ly = CY + dy - oy;
    data[(ly * sz + lx) * 4] = id; // slot0=+id, slot2(ndcZ)=0 ⇒ closest
  }
  return { data, originX: ox, originY: oy, sizeX: sz, sizeY: sz };
}
function noPixel(): PickArgminResult {
  return { found: false, px: 0, py: 0, dist2: 0, slot0: 0, slot1: 0, slot2: 0, slot3: 0, centerSlot0: 0 };
}
function pixelWinner(id: number): PickArgminResult {
  return { found: true, px: CX, py: CY, dist2: 0, slot0: id, slot1: 0, slot2: 0, slot3: 0, centerSlot0: id };
}

function bench(fn: (i: number) => void, iters: number, warm: number): number {
  for (let i = 0; i < warm; i++) fn(i);
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn(i);
  return (performance.now() - t0) / iters;
}
async function benchAsync(fn: (i: number) => Promise<void>, iters: number, warm: number): Promise<number> {
  for (let i = 0; i < warm; i++) await fn(i);
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) await fn(i);
  return (performance.now() - t0) / iters;
}
// Jitter the cursor by ±`j` px (vary branches/caches across iterations).
const px = (i: number, j = 3): number => CX + (j === 0 ? 0 : (i % (2 * j + 1)) - j);

describe("pick benchmark (real WebGPU)", () => {
  let device: GPUDevice;
  let argmin: PickArgminCompute;

  beforeAll(async () => {
    device = await requestRealDevice();
    argmin = createPickArgminCompute(device);
  });
  afterAll(() => { argmin?.dispose(); device?.destroy(); });

  // Build a real pick texture + metadata buffer for the new_total path.
  function makeGpu(reg: PickRegistry, pixelId: number | undefined): { tex: GPUTexture; metaBuf: GPUBuffer } {
    const buf = new Float32Array(W * H * 4);
    if (pixelId !== undefined) {
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) buf[((CY + dy) * W + (CX + dx)) * 4] = pixelId;
    }
    const tex = device.createTexture({ size: { width: W, height: H }, format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: tex }, buf, { bytesPerRow: W * 16, rowsPerImage: H }, { width: W, height: H });
    const maxId = reg.size() + 2;
    const meta = new Float32Array(maxId * 2).fill(-1);
    if (pixelId !== undefined) { meta[pixelId * 2] = 16; meta[pixelId * 2 + 1] = 1; }
    const metaBuf = device.createBuffer({ size: meta.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(metaBuf, 0, meta);
    return { tex, metaBuf };
  }

  async function newTotal(reg: PickRegistry, tex: GPUTexture, metaBuf: GPUBuffer, cx: number, radius?: number): Promise<void> {
    const enc = device.createCommandEncoder();
    argmin.compute(enc, tex.createView(), metaBuf, cx, CY, W, H, radius);
    argmin.copyResult(enc);
    device.queue.submit([enc.finish()]);
    const r = await argmin.read();
    arbitratePick(r, { devX: cx, devY: CY }, reg, ID_VIEW, ID_VIEW, VP);
  }

  for (const N of [200, 1000]) {
    it(`BVH-heavy scene, N=${N}`, async () => {
      const reg = buildRegistry(N);
      const region = emptyRegion();         // no pixel ⇒ forces BVH work
      const res = noPixel();
      const { tex, metaBuf } = makeGpu(reg, undefined);

      const oldCpu = bench((i) => { spiralHitTest(region, { devX: px(i), devY: CY }, reg, ID_VIEW, ID_VIEW, VP); }, 30, 5);
      const newCpu = bench((i) => { arbitratePick(res, { devX: px(i), devY: CY }, reg, ID_VIEW, ID_VIEW, VP); }, 500, 50);
      const newTot = await benchAsync((i) => newTotal(reg, tex, metaBuf, px(i)), 100, 20);
      tex.destroy(); metaBuf.destroy();

      // eslint-disable-next-line no-console
      console.log(`[bench] BVH-heavy N=${N}  old_cpu=${oldCpu.toFixed(3)}ms  new_cpu=${newCpu.toFixed(4)}ms  new_total=${newTot.toFixed(3)}ms  speedup_cpu=${(oldCpu / newCpu).toFixed(0)}x  readback: 25KB→40B`);
      expect(newCpu).toBeLessThan(oldCpu); // CPU resolve must be cheaper
    });
  }

  it("pixel-hit scene (common case), N=1000", async () => {
    const reg = buildRegistry(1000);
    const PIXID = 3;
    const region = pixelRegion(PIXID);
    const res = pixelWinner(PIXID);
    const { tex, metaBuf } = makeGpu(reg, PIXID);

    // Cursor stays on the stamped pixel (j=0): the pixel wins at the
    // centre offset, but the old path STILL pays the O(N) cull-set build
    // (getIntersectingFrustum + forcing N model trafos) every move.
    const oldCpu = bench((i) => { spiralHitTest(region, { devX: px(i, 0), devY: CY }, reg, ID_VIEW, ID_VIEW, VP); }, 200, 20);
    const newCpu = bench((i) => { arbitratePick(res, { devX: px(i, 0), devY: CY }, reg, ID_VIEW, ID_VIEW, VP); }, 500, 50);
    const newTot = await benchAsync((i) => newTotal(reg, tex, metaBuf, px(i, 0)), 100, 20);
    tex.destroy(); metaBuf.destroy();

    // eslint-disable-next-line no-console
    console.log(`[bench] pixel-hit N=1000  old_cpu=${oldCpu.toFixed(4)}ms  new_cpu=${newCpu.toFixed(4)}ms  new_total=${newTot.toFixed(3)}ms  speedup_cpu=${(oldCpu / newCpu).toFixed(0)}x  readback: 25KB→40B`);
    expect(newTot).toBeGreaterThan(0);
  });

  // Flexible R is a runtime uniform now (no recompile). The multi-
  // workgroup atomicMin keeps large radii cheap, unlike the old single-
  // workgroup design (R=48 there cost ~1ms; see git history).
  it("flexible radius sweep (runtime R, multi-workgroup)", async () => {
    const reg = buildRegistry(1000);
    const { tex, metaBuf } = makeGpu(reg, undefined); // BVH-heavy, no pixel
    const times: string[] = [];
    for (const R of [16, 32, 48, 96]) {
      const t = await benchAsync((i) => newTotal(reg, tex, metaBuf, px(i), R), 100, 20);
      times.push(`R=${R}:${t.toFixed(3)}ms`);
    }
    tex.destroy(); metaBuf.destroy();
    // eslint-disable-next-line no-console
    console.log(`[bench] radius sweep N=1000 new_total  ${times.join("  ")}`);
    expect(times.length).toBe(4);
  });
});
