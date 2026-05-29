// Real-WebGPU test for the single-pass pick argmin kernel.
//
// Runs the ACTUAL WGSL (the node/happy-dom suite can't — no
// navigator.gpu) and asserts the GPU output matches the JS reference
// `argminPickReference` across the validity/arbitration scenarios. The
// reference itself is unit-tested in tests/pick-argmin.test.ts; this
// test pins the WGSL to it, catching shader compile/semantics drift
// (e.g. the `meta` reserved-keyword bug a JS-only test cannot see).

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import {
  argminPickReference,
  createPickArgminCompute,
  type PickArgminCompute,
  type PickArgminResult,
} from "../src/scene/picking/pickArgminCompute.js";
import { requestRealDevice } from "./_realGpu.js";

const W = 80;
const H = 80;
const CX = 40;
const CY = 40;

type Pixels = (x: number, y: number) => readonly [number, number, number, number];
type Meta = ReadonlyArray<readonly [number, number]>;

function makeTex(): Float32Array {
  return new Float32Array(W * H * 4);
}
function setPx(buf: Float32Array, x: number, y: number, s0: number, s2 = 0): void {
  if (x < 0 || y < 0 || x >= W || y >= H) return;
  const i = (y * W + x) * 4;
  buf[i] = s0; buf[i + 1] = 0; buf[i + 2] = s2; buf[i + 3] = 0;
}
// A 3×3 stamp (a single pixel also works now — the 3×3 MSAA guard was
// dropped; MSAA ids are resolved upstream by pickResolveCompute).
function block(buf: Float32Array, cx: number, cy: number, id: number, sign = 1, s2 = 0): void {
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) setPx(buf, cx + dx, cy + dy, sign * id, s2);
}
function pixelAt(buf: Float32Array): Pixels {
  return (x, y) => {
    if (x < 0 || y < 0 || x >= W || y >= H) return [0, 0, 0, 0];
    const i = (y * W + x) * 4;
    return [buf[i]!, buf[i + 1]!, buf[i + 2]!, buf[i + 3]!];
  };
}
// metadata[id] = [effectiveRadius, modeSign]; index 0 and gaps = [-1, 1].
function meta(pairs: ReadonlyArray<readonly [number, number, number]>): Meta {
  let maxId = 0;
  for (const [id] of pairs) maxId = Math.max(maxId, id);
  const arr: Array<[number, number]> = [];
  for (let i = 0; i <= maxId; i++) arr.push([-1, 1]);
  for (const [id, r, sign] of pairs) arr[id] = [r, sign];
  return arr;
}

interface Scenario {
  readonly name: string;
  readonly build: (b: Float32Array) => void;
  readonly meta: Meta;
}

const scenarios: Scenario[] = [
  { name: "center hit (block, r16)", build: (b) => block(b, CX, CY, 7), meta: meta([[7, 16, 1]]) },
  { name: "off-center within radius", build: (b) => block(b, CX + 5, CY, 7), meta: meta([[7, 16, 1]]) },
  { name: "outside snap radius → reject", build: (b) => block(b, CX + 10, CY, 7), meta: meta([[7, 4, 1]]) },
  { name: "single pixel accepted (no 3×3 guard)", build: (b) => setPx(b, CX, CY, 7), meta: meta([[7, 16, 1]]) },
  { name: "mode-sign mismatch → reject", build: (b) => block(b, CX, CY, 7, +1), meta: meta([[7, 16, -1]]) },
  { name: "mode-B (negative id) accepted", build: (b) => block(b, CX, CY, 7, -1), meta: meta([[7, 16, -1]]) },
  { name: "two ids — nearer wins", build: (b) => { block(b, CX + 3, CY, 5); block(b, CX + 8, CY, 9); }, meta: meta([[5, 16, 1], [9, 16, 1]]) },
  { name: "inactive (effR<0) → reject", build: (b) => block(b, CX, CY, 7), meta: meta([[7, -1, 1]]) },
  { name: "empty → no winner", build: () => { /* all zero */ }, meta: meta([[7, 16, 1]]) },
  { name: "diagonal tie-break (linear index)", build: (b) => { block(b, CX + 2, CY + 2, 5); block(b, CX - 2, CY - 2, 9); }, meta: meta([[5, 16, 1], [9, 16, 1]]) },
  { name: "depth slot carried through (slot2)", build: (b) => block(b, CX, CY, 7, +1, 0.42), meta: meta([[7, 16, 1]]) },
];

describe("pick argmin kernel (real WebGPU)", () => {
  let device: GPUDevice;
  let argmin: PickArgminCompute;

  beforeAll(async () => {
    device = await requestRealDevice();
    argmin = createPickArgminCompute(device);
  });
  afterAll(() => {
    argmin?.dispose();
    device?.destroy();
  });

  async function runOnGpu(sc: Scenario): Promise<{ gpu: PickArgminResult; ref: PickArgminResult; buf: Float32Array }> {
    const buf = makeTex();
    sc.build(buf);
    const tex = device.createTexture({
      size: { width: W, height: H }, format: "rgba32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: tex }, buf, { bytesPerRow: W * 16, rowsPerImage: H }, { width: W, height: H });
    const metaFlat = new Float32Array(sc.meta.flatMap((p) => [p[0], p[1]]));
    const metaBuf = device.createBuffer({ size: Math.max(8, metaFlat.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(metaBuf, 0, metaFlat);

    const enc = device.createCommandEncoder();
    argmin.compute(enc, tex.createView(), metaBuf, CX, CY, W, H);
    argmin.copyResult(enc);
    device.queue.submit([enc.finish()]);
    const gpu = await argmin.read();
    tex.destroy();
    metaBuf.destroy();
    if (gpu === undefined) throw new Error("argmin.read() returned undefined");
    const ref = argminPickReference(CX, CY, W, H, pixelAt(buf), sc.meta);
    return { gpu, ref, buf };
  }

  it("navigator.gpu is available", () => {
    expect("gpu" in navigator).toBe(true);
  });

  for (const sc of scenarios) {
    it(sc.name, async () => {
      const { gpu, ref } = await runOnGpu(sc);
      expect(gpu.found).toBe(ref.found);
      expect(gpu.centerSlot0).toBeCloseTo(ref.centerSlot0, 5);
      if (ref.found) {
        expect(gpu.px).toBe(ref.px);
        expect(gpu.py).toBe(ref.py);
        expect(gpu.dist2).toBeCloseTo(ref.dist2, 5);
        expect(gpu.slot0).toBeCloseTo(ref.slot0, 5);
        expect(gpu.slot1).toBeCloseTo(ref.slot1, 5);
        expect(gpu.slot2).toBeCloseTo(ref.slot2, 5);
        expect(gpu.slot3).toBeCloseTo(ref.slot3, 5);
      }
    });
  }
});
