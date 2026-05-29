// Multi-workgroup GPU argmin pick kernel.
//
// Finds THE single best pixel under the cursor — nearest (screen-dist²),
// among pixels valid per their own scope — and writes one ~40 B result
// the host maps back. Replaces the old 33×33 region readback + CPU spiral
// walk: the readback is constant (40 B) at any radius.
//
// Two compute passes over a shared `bestKey: atomic<u32>`:
//   findMin — one invocation per disc pixel, across as many workgroups as
//     needed (full GPU utilisation; NOT a single workgroup). Each VALID
//     candidate does `atomicMin(bestKey, key)` where
//         key = (dist² << SHIFT) | discLocalIndex
//     dist² dominates; the disc-local index is the tie-break. Within the
//     disc, ascending disc-local index == ascending (py·W+px), so this
//     matches the JS reference's tie-break exactly.
//   decode — one invocation: unpacks the winning (dist², index), decodes
//     the pixel, reads its 4 slots + the centre pixel (hover), writes the
//     result struct.
//
// Validity, per candidate pixel:
//   - slot0 != 0                    (0 = "no hit" clear value)
//   - dist² ≤ effectiveRadius²      (per-id snap radius; effR<0 ⇒ the
//                                     scope is inactive/noEvents/unknown)
//   - pixel sign == registered mode (+id Mode-A / −id Mode-B)
//
// There is NO 3×3 neighbour (MSAA) check: MSAA pickIds are resolved
// upstream by `pickResolveCompute`'s majority vote (averaging ids is
// nonsense), and the non-MSAA path is single-sample, so averaged-id
// pixels never reach here. Dropping the check makes each thread a single
// texture load + atomicMin — embarrassingly parallel, so a large radius
// stays cheap. (It does make 1px slivers pickable, which the old 3×3
// guard rejected — an intentional behaviour change.)
//
// RADIUS is a runtime uniform (the search window), capped at RMAX by the
// packing budget: dist² ≤ RMAX² must fit in (32-SHIFT) bits and the disc-
// local index ≤ (2·RMAX+1)² must fit in SHIFT bits. SHIFT=17, RMAX=127:
// (255)² = 65025 < 2^17, and 127² = 16129 < 2^15. One pipeline serves any
// radius ≤ 127 with no recompile. Per-scope radius stays fully flexible
// via the metadata buffer; this is just the window ceiling.

import { SNAP_RADIUS_MAX } from "./snapOffsets.js";

/** Largest radius the packed-key budget supports at runtime. */
export const PICK_ARGMIN_MAX_RADIUS = 127;

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

// Packing: key = (dist² << SHIFT) | discLocalIndex.
const SHIFT = 17;

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
  return `// Auto-generated multi-workgroup pick argmin. SHIFT=${SHIFT}, RMAX=${PICK_ARGMIN_MAX_RADIUS}.
@group(0) @binding(0) var pickTex: texture_2d<f32>;

struct Params {
  cx: i32,        // cursor device-pixel x
  cy: i32,        // cursor device-pixel y
  width: i32,     // pick texture width
  height: i32,    // pick texture height
  radius: i32,    // search-window radius (≤ RMAX)
};
@group(0) @binding(1) var<uniform> params: Params;

// metadata[absId] = vec2(effectiveRadius, modeSign). radius<0 ⇒ invalid.
@group(0) @binding(2) var<storage, read> metadata: array<vec2<f32>>;

// Packed winner key: (dist² << SHIFT) | discLocalIndex. Init to 0xffffffff.
@group(0) @binding(3) var<storage, read_write> bestKey: atomic<u32>;

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
@group(0) @binding(4) var<storage, read_write> result: Result;

const SHIFT: u32 = ${SHIFT}u;
const NO_WINNER: u32 = 0xffffffffu;

// load slot0 at (x,y); out-of-bounds reads as 0 (= no hit).
fn loadSlot0(x: i32, y: i32) -> f32 {
  if (x < 0 || y < 0 || x >= params.width || y >= params.height) { return 0.0; }
  return textureLoad(pickTex, vec2<i32>(x, y), 0).x;
}

@compute @workgroup_size(256)
fn findMin(@builtin(global_invocation_id) gid: vec3<u32>) {
  let R: i32 = params.radius;
  let SIDE: i32 = 2 * R + 1;
  let k: i32 = i32(gid.x);
  if (k >= SIDE * SIDE) { return; }
  let dx: i32 = (k % SIDE) - R;
  let dy: i32 = (k / SIDE) - R;
  let d2i: i32 = dx * dx + dy * dy;
  if (d2i > R * R) { return; }                 // reject the bounding square's corners
  let s0: f32 = loadSlot0(params.cx + dx, params.cy + dy);
  if (s0 == 0.0) { return; }
  let absId: u32 = u32(abs(s0));
  let md: vec2<f32> = metadata[absId];
  let effR: f32 = md.x;
  let d2f: f32 = f32(d2i);
  if (effR < 0.0 || d2f > effR * effR) { return; }      // snap radius (folds active/noEvents)
  let pixSign: f32 = select(1.0, -1.0, s0 < 0.0);
  if (pixSign != md.y) { return; }                       // mode-sign gate
  // disc-local index is monotonic in (dy, dx) ⇒ matches a (py·W+px) tie-break.
  let discIdx: u32 = u32((dy + R) * SIDE + (dx + R));
  atomicMin(&bestKey, (u32(d2i) << SHIFT) | discIdx);
}

@compute @workgroup_size(1)
fn decode() {
  let R: i32 = params.radius;
  let SIDE: i32 = 2 * R + 1;
  result.centerSlot0 = loadSlot0(params.cx, params.cy);
  let key: u32 = atomicLoad(&bestKey);
  if (key == NO_WINNER) {
    result.found = 0u;
    result.px = 0; result.py = 0; result.dist2 = 0.0;
    result.slot0 = 0.0; result.slot1 = 0.0; result.slot2 = 0.0; result.slot3 = 0.0;
    return;
  }
  let discIdx: i32 = i32(key & ((1u << SHIFT) - 1u));
  let d2: i32 = i32(key >> SHIFT);
  let dx: i32 = (discIdx % SIDE) - R;
  let dy: i32 = (discIdx / SIDE) - R;
  let px: i32 = params.cx + dx;
  let py: i32 = params.cy + dy;
  let s: vec4<f32> = textureLoad(pickTex, vec2<i32>(clamp(px, 0, params.width - 1), clamp(py, 0, params.height - 1)), 0);
  result.found = 1u;
  result.px = px; result.py = py; result.dist2 = f32(d2);
  result.slot0 = s.x; result.slot1 = s.y; result.slot2 = s.z; result.slot3 = s.w;
}
`;
}

interface CacheEntry {
  module: GPUShaderModule;
  findMin: GPUComputePipeline;
  decode: GPUComputePipeline;
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
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const findMin = device.createComputePipeline({ label: "pick.argmin.findMin", layout: pipelineLayout, compute: { module, entryPoint: "findMin" } });
  const decode = device.createComputePipeline({ label: "pick.argmin.decode", layout: pipelineLayout, compute: { module, entryPoint: "decode" } });
  const entry: CacheEntry = { module, findMin, decode, layout };
  pipelineCache.set(device, entry);
  return entry;
}

export interface PickArgminCompute {
  /**
   * Encode one argmin (findMin + decode) into `encoder`. `metadataBuffer`
   * is a read-only storage buffer of `vec2<f32>(effectiveRadius, modeSign)`
   * indexed by absolute pickId; it must be ≥ `(maxPickId+1)*8` bytes.
   * `radius` is the search-window radius (clamped to `PICK_ARGMIN_MAX_RADIUS`;
   * defaults to `SNAP_RADIUS_MAX`). Call `read()` after the submit.
   */
  compute(
    encoder: GPUCommandEncoder,
    pickView: GPUTextureView,
    metadataBuffer: GPUBuffer,
    cx: number,
    cy: number,
    width: number,
    height: number,
    radius?: number,
  ): void;
  /** Copy the result into the mappable buffer (encode before submit). */
  copyResult(encoder: GPUCommandEncoder): void;
  /** Map the readback buffer and decode. Call after the submit resolves. */
  read(): Promise<PickArgminResult | undefined>;
  dispose(): void;
}

export function createPickArgminCompute(device: GPUDevice): PickArgminCompute {
  const { findMin, decode, layout } = getOrBuildEntry(device);
  const params = device.createBuffer({
    size: 32, // 5×i32, padded to a 16-byte multiple
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: "pick.argmin.params",
  });
  const bestKey = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "pick.argmin.bestKey",
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
    compute(encoder, pickView, metadataBuffer, cx, cy, width, height, radius): void {
      const r = Math.max(0, Math.min(PICK_ARGMIN_MAX_RADIUS, Math.floor(radius ?? SNAP_RADIUS_MAX)));
      device.queue.writeBuffer(params, 0, new Int32Array([cx | 0, cy | 0, width | 0, height | 0, r]));
      device.queue.writeBuffer(bestKey, 0, new Uint32Array([0xffffffff]));
      const bg = device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: pickView },
          { binding: 1, resource: { buffer: params } },
          { binding: 2, resource: { buffer: metadataBuffer } },
          { binding: 3, resource: { buffer: bestKey } },
          { binding: 4, resource: { buffer: result } },
        ],
      });
      const side = 2 * r + 1;
      const groups = Math.ceil((side * side) / 256);
      const find = encoder.beginComputePass({ label: "pick.argmin.findMin" });
      find.setPipeline(findMin);
      find.setBindGroup(0, bg);
      find.dispatchWorkgroups(groups, 1, 1);
      find.end();
      const dec = encoder.beginComputePass({ label: "pick.argmin.decode" });
      dec.setPipeline(decode);
      dec.setBindGroup(0, bg);
      dec.dispatchWorkgroups(1, 1, 1);
      dec.end();
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
      try { bestKey.destroy(); } catch { /* gone */ }
      try { result.destroy(); } catch { /* gone */ }
      try { readback.destroy(); } catch { /* gone */ }
    },
  };
}

/**
 * Host-side reference of the kernel's argmin — used by unit tests and as
 * the spec the WGSL mirrors. `pixelAt(x,y)` returns the 4 slot floats at
 * a device pixel (0,0,0,0 when out of bounds). `metadata[absId] =
 * [effectiveRadius, modeSign]`. `radius` is the search window (defaults
 * to `SNAP_RADIUS_MAX`, matching `compute`'s default).
 */
export function argminPickReference(
  cx: number,
  cy: number,
  width: number,
  height: number,
  pixelAt: (x: number, y: number) => readonly [number, number, number, number],
  metadata: ReadonlyArray<readonly [number, number]>,
  radius: number = SNAP_RADIUS_MAX,
): PickArgminResult {
  const R = Math.max(0, Math.min(PICK_ARGMIN_MAX_RADIUS, Math.floor(radius)));
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
      // disc-local index tie-break (== ascending py·W+px within the disc).
      const li = (dy + R) * (2 * R + 1) + (dx + R);
      if (d2 < bestD || (d2 === bestD && li < bestI)) {
        bestD = d2;
        bestI = li;
        best = { found: true, px, py, dist2: d2, slot0: s[0], slot1: s[1], slot2: s[2], slot3: s[3], centerSlot0 };
      }
    }
  }
  return best;
}
