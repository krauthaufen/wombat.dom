// Single-pass GPU argmin pick kernel.
//
// Replaces the CPU spiral walk's PIXEL side: instead of reading back a
// (2·SNAP_RADIUS_MAX+1)² disc and walking ~800 offsets on the CPU, one
// compute dispatch finds THE single best pixel — nearest to the cursor
// (screen-dist²), among pixels that are valid per their own scope — and
// writes one tiny result the host maps back (~32 B).
//
// Validity, per candidate pixel (all in-kernel so the argmin lands on a
// pixel the host will accept, never one it would reject and have to
// fall past):
//   - slot0 != 0                       (0 = "no hit" clear value)
//   - dist² ≤ effectiveRadius²         (per-id snap radius)
//   - pixel sign == registered mode    (rejects MSAA silhouette averages
//                                        that surface a valid id, wrong layout)
//   - ≥3 of the 8 neighbours share the exact slot0 (further MSAA guard,
//     mirrors spiralHitTest's 3×3 same-id count)
//
// Per-id metadata (`metadata[absId] = vec2(effectiveRadius, modeSign)`):
//   - effectiveRadius < 0  ⇒ the scope is inactive / noEvents / unknown
//     ⇒ never a candidate. This folds the host's `active`/`noEvents`
//     gating INTO the radius test, so a single argmin reproduces the
//     spiral's "skip invalid, take next-nearest valid" without iterating.
//   - modeSign = +1 for Mode-A (slot0 = +id), -1 for Mode-B (slot0 = -id).
//
// The kernel is MODE-AGNOSTIC beyond the sign check: it copies the
// winner's raw slot0..slot3 verbatim, so the host's existing `decodePick`
// reconstructs Mode-A (normal/depth/part) and Mode-B (PickViewPosition)
// identically to the spiral path.
//
// One workgroup, grid-strided over the disc; a shared-memory reduction
// picks the winner (key: dist² asc, then linear pixel index asc for a
// deterministic tie-break — matching the spiral's stable d²-order).

import { SNAP_RADIUS_MAX } from "./snapOffsets.js";

/** Bytes of the result struct mapped back to the host. See `RESULT_*` offsets. */
export const PICK_ARGMIN_RESULT_BYTES = 40;

// Result struct field offsets (in f32/u32/i32 units within the buffer).
const R_FOUND = 0; // u32: 1 ⇒ a valid winner was written
const R_PX = 1; // i32: winner device-pixel x
const R_PY = 2; // i32: winner device-pixel y
const R_DIST2 = 3; // f32: screen-dist² to the cursor
const R_SLOT0 = 4; // f32×4: raw winning pixel slots 0..3
const R_CENTER_SLOT0 = 8; // f32: raw slot0 exactly under the cursor (hover, even if invalid)
// index 9 is padding

const WORKGROUP_X = 16;
const WORKGROUP_Y = 16;
const THREADS = WORKGROUP_X * WORKGROUP_Y;

/** Decoded host-side view of the kernel's result buffer. */
export interface PickArgminResult {
  readonly found: boolean;
  readonly px: number;
  readonly py: number;
  readonly dist2: number;
  readonly slot0: number;
  readonly slot1: number;
  readonly slot2: number;
  readonly slot3: number;
  /** Raw slot0 exactly under the cursor — for hover latching even when no valid winner. */
  readonly centerSlot0: number;
}

/** Decode the result buffer the kernel wrote. */
export function decodeArgminResult(buf: ArrayBuffer): PickArgminResult {
  const n = PICK_ARGMIN_RESULT_BYTES / 4;
  const f = new Float32Array(buf, 0, n);
  const i = new Int32Array(buf, 0, n);
  const u = new Uint32Array(buf, 0, n);
  return {
    found: u[R_FOUND] !== 0,
    px: i[R_PX]!,
    py: i[R_PY]!,
    dist2: f[R_DIST2]!,
    slot0: f[R_SLOT0]!,
    slot1: f[R_SLOT0 + 1]!,
    slot2: f[R_SLOT0 + 2]!,
    slot3: f[R_SLOT0 + 3]!,
    centerSlot0: f[R_CENTER_SLOT0]!,
  };
}

export function buildPickArgminWgsl(): string {
  const R = SNAP_RADIUS_MAX;
  const SIDE = 2 * R + 1;
  const DISC = SIDE * SIDE;
  return `// Auto-generated single-pass pick argmin. radius=${R}.
@group(0) @binding(0) var pickTex: texture_2d<f32>;

struct Params {
  cx: i32,        // cursor device-pixel x
  cy: i32,        // cursor device-pixel y
  width: i32,     // pick texture width
  height: i32,    // pick texture height
};
@group(0) @binding(1) var<uniform> params: Params;

// metadata[absId] = vec2(effectiveRadius, modeSign). radius<0 ⇒ invalid.
@group(0) @binding(2) var<storage, read> metadata: array<vec2<f32>>;

struct Result {
  found: u32,
  px: i32,
  py: i32,
  dist2: f32,
  slot0: f32,
  slot1: f32,
  slot2: f32,
  slot3: f32,
  centerSlot0: f32,
  _pad: f32,
};
@group(0) @binding(3) var<storage, read_write> result: Result;

const R: i32 = ${R};
const SIDE: i32 = ${SIDE};
const DISC: i32 = ${DISC};
const THREADS: u32 = ${THREADS}u;
// Sentinel "no candidate": dist² larger than any real disc distance.
const NO_CAND: f32 = 1.0e30;

var<workgroup> wgDist2: array<f32, ${THREADS}>;
var<workgroup> wgIndex: array<i32, ${THREADS}>; // linear pixel index, tie-break
var<workgroup> wgPx:    array<i32, ${THREADS}>;
var<workgroup> wgPy:    array<i32, ${THREADS}>;
var<workgroup> wgS0:    array<f32, ${THREADS}>;
var<workgroup> wgS1:    array<f32, ${THREADS}>;
var<workgroup> wgS2:    array<f32, ${THREADS}>;
var<workgroup> wgS3:    array<f32, ${THREADS}>;

// load slot0 at (x,y); out-of-bounds reads as 0 (= no hit).
fn loadSlot0(x: i32, y: i32) -> f32 {
  if (x < 0 || y < 0 || x >= params.width || y >= params.height) { return 0.0; }
  return textureLoad(pickTex, vec2<i32>(x, y), 0).x;
}

// "a beats b": smaller dist², ties broken by smaller linear index.
fn beats(da: f32, ia: i32, db: f32, ib: i32) -> bool {
  return da < db || (da == db && ia < ib);
}

@compute @workgroup_size(${WORKGROUP_X}, ${WORKGROUP_Y}, 1)
fn main(@builtin(local_invocation_index) lid: u32) {
  var bestD: f32 = NO_CAND;
  var bestI: i32 = 0x7fffffff;
  var bPx: i32 = 0; var bPy: i32 = 0;
  var bS0: f32 = 0.0; var bS1: f32 = 0.0; var bS2: f32 = 0.0; var bS3: f32 = 0.0;

  // grid-stride over the disc's bounding square; reject corners by radius.
  var k: i32 = i32(lid);
  loop {
    if (k >= DISC) { break; }
    let dx: i32 = (k % SIDE) - R;
    let dy: i32 = (k / SIDE) - R;
    let d2i: i32 = dx * dx + dy * dy;
    if (d2i <= R * R) {
      let px: i32 = params.cx + dx;
      let py: i32 = params.cy + dy;
      let s = textureLoad(pickTex, vec2<i32>(clamp(px, 0, params.width - 1), clamp(py, 0, params.height - 1)), 0);
      let s0: f32 = select(0.0, s.x, px >= 0 && py >= 0 && px < params.width && py < params.height);
      if (s0 != 0.0) {
        let absId: u32 = u32(abs(s0));
        let md: vec2<f32> = metadata[absId];
        let effR: f32 = md.x;
        let d2f: f32 = f32(d2i);
        // radius gate (also encodes active/noEvents: effR<0 ⇒ never valid)
        if (effR >= 0.0 && d2f <= effR * effR) {
          // mode-sign gate
          let pixSign: f32 = select(1.0, -1.0, s0 < 0.0);
          if (pixSign == md.y) {
            // 3×3 same-id neighbour count ≥ 3 (MSAA silhouette guard)
            var matches: i32 = 0;
            for (var ddy: i32 = -1; ddy <= 1; ddy = ddy + 1) {
              for (var ddx: i32 = -1; ddx <= 1; ddx = ddx + 1) {
                if (!(ddx == 0 && ddy == 0)) {
                  if (loadSlot0(px + ddx, py + ddy) == s0) { matches = matches + 1; }
                }
              }
            }
            if (matches >= 3) {
              let li: i32 = py * params.width + px;
              if (beats(d2f, li, bestD, bestI)) {
                bestD = d2f; bestI = li; bPx = px; bPy = py;
                bS0 = s.x; bS1 = s.y; bS2 = s.z; bS3 = s.w;
              }
            }
          }
        }
      }
    }
    k = k + i32(THREADS);
  }

  wgDist2[lid] = bestD; wgIndex[lid] = bestI;
  wgPx[lid] = bPx; wgPy[lid] = bPy;
  wgS0[lid] = bS0; wgS1[lid] = bS1; wgS2[lid] = bS2; wgS3[lid] = bS3;
  workgroupBarrier();

  // tree reduction over the workgroup
  var stride: u32 = THREADS / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lid < stride) {
      let j = lid + stride;
      if (beats(wgDist2[j], wgIndex[j], wgDist2[lid], wgIndex[lid])) {
        wgDist2[lid] = wgDist2[j]; wgIndex[lid] = wgIndex[j];
        wgPx[lid] = wgPx[j]; wgPy[lid] = wgPy[j];
        wgS0[lid] = wgS0[j]; wgS1[lid] = wgS1[j]; wgS2[lid] = wgS2[j]; wgS3[lid] = wgS3[j];
      }
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lid == 0u) {
    result.centerSlot0 = loadSlot0(params.cx, params.cy);
    if (wgDist2[0] < NO_CAND) {
      result.found = 1u;
      result.px = wgPx[0]; result.py = wgPy[0];
      result.dist2 = wgDist2[0];
      result.slot0 = wgS0[0]; result.slot1 = wgS1[0];
      result.slot2 = wgS2[0]; result.slot3 = wgS3[0];
    } else {
      result.found = 0u;
      result.px = 0; result.py = 0; result.dist2 = 0.0;
      result.slot0 = 0.0; result.slot1 = 0.0; result.slot2 = 0.0; result.slot3 = 0.0;
    }
  }
}
`;
}

interface CacheEntry {
  module: GPUShaderModule;
  pipeline: GPUComputePipeline;
  layout: GPUBindGroupLayout;
}

const pipelineCache: WeakMap<GPUDevice, CacheEntry> = new WeakMap();

function getOrBuildEntry(device: GPUDevice): CacheEntry {
  const cached = pipelineCache.get(device);
  if (cached !== undefined) return cached;
  const code = buildPickArgminWgsl();
  const module = device.createShaderModule({ code, label: "pick.argmin" });
  const layout = device.createBindGroupLayout({
    label: "pick.argmin.bgl",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d", multisampled: false } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipeline = device.createComputePipeline({
    label: "pick.argmin.pipeline",
    layout: pipelineLayout,
    compute: { module, entryPoint: "main" },
  });
  const entry: CacheEntry = { module, pipeline, layout };
  pipelineCache.set(device, entry);
  return entry;
}

export interface PickArgminCompute {
  /**
   * Encode one argmin dispatch into `encoder`. `metadataBuffer` is a
   * read-only storage buffer of `vec2<f32>(effectiveRadius, modeSign)`
   * indexed by absolute pickId (see module header); it must be at least
   * `(maxPickId+1) * 8` bytes. The result lands in the internal result
   * buffer; call `read()` after the submit to map it back.
   */
  compute(
    encoder: GPUCommandEncoder,
    pickView: GPUTextureView,
    metadataBuffer: GPUBuffer,
    cx: number,
    cy: number,
    width: number,
    height: number,
  ): void;
  /** Copy the result into the mappable buffer (encode before submit). */
  copyResult(encoder: GPUCommandEncoder): void;
  /** Map the readback buffer and decode. Call after the submit resolves. */
  read(): Promise<PickArgminResult | undefined>;
  dispose(): void;
}

export function createPickArgminCompute(device: GPUDevice): PickArgminCompute {
  const { pipeline, layout } = getOrBuildEntry(device);
  const params = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: "pick.argmin.params",
  });
  const result = device.createBuffer({
    size: PICK_ARGMIN_RESULT_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: "pick.argmin.result",
  });
  const readback = device.createBuffer({
    size: PICK_ARGMIN_RESULT_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: "pick.argmin.readback",
  });

  return {
    compute(encoder, pickView, metadataBuffer, cx, cy, width, height): void {
      device.queue.writeBuffer(params, 0, new Int32Array([cx | 0, cy | 0, width | 0, height | 0]));
      const bg = device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: pickView },
          { binding: 1, resource: { buffer: params } },
          { binding: 2, resource: { buffer: metadataBuffer } },
          { binding: 3, resource: { buffer: result } },
        ],
      });
      const pass = encoder.beginComputePass({ label: "pick.argmin.pass" });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(1, 1, 1);
      pass.end();
    },
    copyResult(encoder): void {
      encoder.copyBufferToBuffer(result, 0, readback, 0, PICK_ARGMIN_RESULT_BYTES);
    },
    async read(): Promise<PickArgminResult | undefined> {
      try {
        await readback.mapAsync(GPUMapMode.READ);
        const out = decodeArgminResult(readback.getMappedRange().slice(0));
        readback.unmap();
        return out;
      } catch {
        return undefined;
      }
    },
    dispose(): void {
      try { params.destroy(); } catch { /* gone */ }
      try { result.destroy(); } catch { /* gone */ }
      try { readback.destroy(); } catch { /* gone */ }
    },
  };
}

/**
 * Host-side reference of the kernel's argmin — used by unit tests and as
 * the spec the WGSL mirrors. `pixelAt(x,y)` returns the 4 slot floats at
 * a device pixel (0,0,0,0 when out of bounds). `metadata[absId] =
 * [effectiveRadius, modeSign]`.
 */
export function argminPickReference(
  cx: number,
  cy: number,
  width: number,
  height: number,
  pixelAt: (x: number, y: number) => readonly [number, number, number, number],
  metadata: ReadonlyArray<readonly [number, number]>,
): PickArgminResult {
  const R = SNAP_RADIUS_MAX;
  let bestD = Infinity;
  let bestI = Infinity;
  const slot0At = (x: number, y: number): number => {
    if (x < 0 || y < 0 || x >= width || y >= height) return 0;
    return pixelAt(x, y)[0];
  };
  const centerSlot0 = slot0At(cx, cy);
  let best: PickArgminResult = {
    found: false, px: 0, py: 0, dist2: 0, slot0: 0, slot1: 0, slot2: 0, slot3: 0, centerSlot0,
  };
  for (let dy = -R; dy <= R; dy++) {
    for (let dx = -R; dx <= R; dx++) {
      const d2 = dx * dx + dy * dy;
      if (d2 > R * R) continue;
      const px = cx + dx;
      const py = cy + dy;
      if (px < 0 || py < 0 || px >= width || py >= height) continue;
      const s = pixelAt(px, py);
      const s0 = s[0];
      if (s0 === 0) continue;
      const absId = Math.abs(s0) | 0;
      const meta = metadata[absId];
      if (meta === undefined) continue;
      const effR = meta[0];
      if (effR < 0 || d2 > effR * effR) continue;
      const pixSign = s0 < 0 ? -1 : 1;
      if (pixSign !== meta[1]) continue;
      let matches = 0;
      for (let ddy = -1; ddy <= 1; ddy++) {
        for (let ddx = -1; ddx <= 1; ddx++) {
          if (ddx === 0 && ddy === 0) continue;
          if (slot0At(px + ddx, py + ddy) === s0) matches++;
        }
      }
      if (matches < 3) continue;
      const li = py * width + px;
      if (d2 < bestD || (d2 === bestD && li < bestI)) {
        bestD = d2;
        bestI = li;
        best = { found: true, px, py, dist2: d2, slot0: s[0], slot1: s[1], slot2: s[2], slot3: s[3], centerSlot0 };
      }
    }
  }
  return best;
}
