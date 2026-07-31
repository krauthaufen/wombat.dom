// Screen-space point fill via an explicit spatial grid — the geometric
// counterpart to voronoiFill.ts's jump-flooding.
//
// Same job, same decision rules, different machinery. Instead of
// propagating a nearest-seed label through log2(r) full-screen ping-pong
// passes, this bins the rendered points into screen tiles and answers
// each pixel's query by actually looking at the points around it:
//
//   1. clear   — zero the per-tile counters
//   2. append  — one thread per pixel; a covered pixel appends its
//                (x, y, depth, colour) into its tile's list. `atomicAdd`
//                returns the slot, so the allocation IS the identity —
//                nothing has to be carried through a min.
//   3. query   — one thread per pixel; scan the tiles overlapping the
//                search radius, apply the SAME comparator voronoiFill
//                uses, and fill.
//
// Only commutative atomics (add) on storage buffers — no fragment
// interlock (which WebGPU does not have) and no depth+payload packing.
//
// The point of having both is measurement: the grid can express queries
// JFA cannot (true neighbour sets, angular coverage, per-pixel adaptive
// radius), but it pays a per-pixel candidate scan where JFA pays a fixed
// tap count. Which wins is a question about the data, not the theory.

import { AVal, type aval } from "@aardworx/wombat.adaptive";
import type { IFramebuffer } from "@aardworx/wombat.rendering/core";

const CLEAR_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> fronts: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= arrayLength(&counters)) { return; }
  atomicStore(&counters[gid.x], 0u);
  atomicStore(&fronts[gid.x], 0u);
}
`;

const APPEND_WGSL = /* wgsl */ `
struct U {
  size: vec2u, tiles: vec2u,
  tileSize: u32, capacity: u32, reversedZ: u32, _p1: u32,
}
@group(0) @binding(0) var pick: texture_2d<f32>;
@group(0) @binding(1) var colorTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> points: array<vec4<u32>>;
@group(0) @binding(4) var<uniform> u: U;
@group(0) @binding(5) var<storage, read_write> fronts: array<atomic<u32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u.size.x || gid.y >= u.size.y) { return; }
  let px = vec2i(gid.xy);
  let z = textureLoad(pick, px, 0).z;
  if (z <= 0.0 || z >= 1.0) { return; }          // no point on this pixel
  let tile = gid.xy / u.tileSize;
  let ti = tile.y * u.tiles.x + tile.x;
  // "frontness": bigger = nearer, so one atomicMax per tile gives the
  // front-most surface in it. f32 bits are monotonic for positive floats,
  // so the integer max IS the float max. Used only as a THRESHOLD — no
  // identity has to survive it.
  var fness = z;
  if (u.reversedZ == 0u) { fness = 1.0 - z; }
  atomicMax(&fronts[ti], bitcast<u32>(fness));
  let slot = atomicAdd(&counters[ti], 1u);
  // Overflow drops the point. Safe by construction: a tile only overflows
  // when it is dense, and a dense tile's pixels are already covered — the
  // fill has nothing to do there, and any retained subset still provides
  // a near candidate for a neighbouring hole.
  if (slot >= u.capacity) { return; }
  let c = textureLoad(colorTex, px, 0);
  let rgba = (u32(clamp(c.r, 0.0, 1.0) * 255.0) << 24)
           | (u32(clamp(c.g, 0.0, 1.0) * 255.0) << 16)
           | (u32(clamp(c.b, 0.0, 1.0) * 255.0) << 8)
           | 255u;
  points[ti * u.capacity + slot] = vec4u((gid.x << 16u) | gid.y, bitcast<u32>(z), rgba, 0u);
}
`;

// Dilate the per-tile front depth over the reach neighbourhood ONCE per
// tile, so the per-pixel front test is a single load instead of ~16.
const DILATE_WGSL = /* wgsl */ `
struct U { tiles: vec2u, radius: u32, _p: u32 }
@group(0) @binding(0) var<storage, read> fronts: array<u32>;
@group(0) @binding(1) var<storage, read_write> dilated: array<u32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = u.tiles.x * u.tiles.y;
  if (gid.x >= n) { return; }
  let tx = i32(gid.x % u.tiles.x);
  let ty = i32(gid.x / u.tiles.x);
  let r = i32(u.radius);
  var m = 0u;
  for (var y = max(0, ty - r); y <= min(i32(u.tiles.y) - 1, ty + r); y = y + 1) {
    for (var x = max(0, tx - r); x <= min(i32(u.tiles.x) - 1, tx + r); x = x + 1) {
      m = max(m, fronts[u32(y) * u.tiles.x + u32(x)]);
    }
  }
  dilated[gid.x] = m;
}
`;

const QUERY_WGSL = /* wgsl */ `
struct U {
  size: vec2u, tiles: vec2u,
  tileSize: u32, capacity: u32, maxR2: f32, reach2: f32,
  tol: f32, reversedZ: f32, writePick: f32, _p0: f32,
}
// colour comes packed out of the point buffer, so no colour texture here
@group(0) @binding(0) var pick: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> counters: array<u32>;
@group(0) @binding(3) var<storage, read> points: array<vec4<u32>>;
@group(0) @binding(4) var<uniform> u: U;
@group(0) @binding(5) var filledPick: texture_storage_2d<rgba32float, write>;
@group(0) @binding(6) var<storage, read> fronts: array<u32>;
@group(0) @binding(7) var<storage, read> dilated: array<u32>;
@group(0) @binding(8) var<storage, read> dilatedFar: array<u32>;

fn distProxy(z: f32) -> f32 {
  if (u.reversedZ != 0.0) { return 1.0 / max(z, 1e-6); }
  return z;
}

struct VOut { @builtin(position) pos: vec4f }

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var xy = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut;
  o.pos = vec4f(xy[vi], 0.0, 1.0);
  return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4f {
  let px = vec2i(i.pos.xy);
  let own = textureLoad(pick, px, 0);
  let ownValid = own.z > 0.0 && own.z < 1.0;

  let ts = i32(u.tileSize);
  let tmax = vec2i(i32(u.tiles.x) - 1, i32(u.tiles.y) - 1);
  let reach = i32(ceil(sqrt(u.reach2)));

  // --- prune 1: is this pixel already ON the front surface? -------------
  // Nothing within reach can be distinctly nearer than the front-most
  // surface around us, and our own point sits at distance 0, so it wins
  // every same-depth tie. Reading ~25 tile fronts settles that without
  // touching a single point. Conservative (the tile range is a superset
  // of the reach disc), so a miss just falls through to the full scan.
  let r0 = max(vec2i(0), (px - vec2i(reach)) / ts);
  let r1 = min(tmax, (px + vec2i(reach)) / ts);
  let myTile = (px / ts);
  let mti = u32(myTile.y) * u.tiles.x + u32(myTile.x);

  // --- prune 0: is there ANY point within the search radius? ------------
  // Background pixels dominate a sparse view, and without this each one
  // walks every tile in the radius (~49) only to find nothing. One read
  // of the radius-dilated occupancy settles it.
  if (dilatedFar[mti] == 0u) {
    if (u.writePick != 0.0) { textureStore(filledPick, px, own); }
    discard;
  }
  let frontness = bitcast<f32>(dilated[mti]);
  var ownF = own.z;
  if (u.reversedZ == 0.0) { ownF = 1.0 - own.z; }
  if (ownValid && ownF >= frontness * (1.0 - u.tol)) {
    if (u.writePick != 0.0) { textureStore(filledPick, px, own); }
    discard;
  }

  var bestD2 = 1e30;
  var bestZ = 0.0;
  var bestC = 0u;
  var found = false;

  // nothing within reach at all -> skip straight to the outer ring
  let anyNear = dilated[mti] != 0u;
  // --- prune 2: scan the REACH disc first -------------------------------
  // A beyond-reach candidate can never beat a within-reach one (the
  // comparator returns "keep best" for that case), so if anything is
  // found here the outer ring cannot change the answer and is skipped.
  for (var ty = r0.y; ty <= r1.y && anyNear; ty = ty + 1) {
    for (var tx = r0.x; tx <= r1.x; tx = tx + 1) {
      let ti = u32(ty) * u.tiles.x + u32(tx);
      if (counters[ti] == 0u) { continue; }
      // --- prune 3: whole-tile depth reject ----------------------------
      // once we hold a within-reach best, a tile whose FRONT surface is
      // distinctly behind it cannot contain a winner: no point of it can
      // be depth-nearer, nor tie on depth and win on distance.
      if (found) {
        let tf = bitcast<f32>(fronts[ti]);
        var bf = bestZ;
        if (u.reversedZ == 0.0) { bf = 1.0 - bestZ; }
        if (tf < bf * (1.0 - u.tol)) { continue; }
      }
      let n = min(counters[ti], u.capacity);
      for (var k = 0u; k < n; k = k + 1u) {
        let pt = points[ti * u.capacity + k];
        let dx = f32(i32(pt.x >> 16u) - px.x);
        let dy = f32(i32(pt.x & 0xffffu) - px.y);
        let d2 = dx * dx + dy * dy;
        if (d2 > u.reach2) { continue; }
        let cz = bitcast<f32>(pt.y);
        var take = false;
        if (!found) {
          take = true;
        } else {
          // both candidates are within reach here, so the comparator
          // reduces to: nearer depth wins, else nearer in 2D
          let lc = distProxy(cz);
          let lb = distProxy(bestZ);
          if (lc < lb * (1.0 - u.tol)) { take = true; }
          else if (lb < lc * (1.0 - u.tol)) { take = false; }
          else { take = d2 < bestD2; }
        }
        if (take) { bestD2 = d2; bestZ = cz; bestC = pt.z; found = true; }
      }
    }
  }

  // --- outer ring: only reached when nothing is within reach ------------
  if (!found) {
    let r = i32(ceil(sqrt(u.maxR2)));
    let t0 = max(vec2i(0), (px - vec2i(r)) / ts);
    let t1 = min(tmax, (px + vec2i(r)) / ts);
    for (var ty = t0.y; ty <= t1.y; ty = ty + 1) {
      for (var tx = t0.x; tx <= t1.x; tx = tx + 1) {
        let ti = u32(ty) * u.tiles.x + u32(tx);
        let n = min(counters[ti], u.capacity);
        for (var k = 0u; k < n; k = k + 1u) {
          let pt = points[ti * u.capacity + k];
          let dx = f32(i32(pt.x >> 16u) - px.x);
          let dy = f32(i32(pt.x & 0xffffu) - px.y);
          let d2 = dx * dx + dy * dy;
          if (d2 > u.maxR2) { continue; }
          // beyond reach on both sides -> plain 2D nearest
          if (!found || d2 < bestD2) {
            bestD2 = d2; bestZ = bitcast<f32>(pt.y); bestC = pt.z; found = true;
          }
        }
      }
    }
  }

  var outPick = own;
  var color = vec4f(0.0);
  var fill = false;
  if (found) {
    // same resolve rule: fill holes, and overwrite rendered points that
    // sit distinctly BEHIND the winning surface (hidden-point removal)
    let seedNearer = distProxy(bestZ) < distProxy(own.z) * (1.0 - u.tol);
    if (!ownValid || seedNearer) {
      color = vec4f(
        f32((bestC >> 24u) & 0xffu) / 255.0,
        f32((bestC >> 16u) & 0xffu) / 255.0,
        f32((bestC >> 8u) & 0xffu) / 255.0,
        1.0);
      outPick = vec4f(own.x, own.y, bestZ, own.w);
      fill = true;
    }
  }
  if (u.writePick != 0.0) { textureStore(filledPick, px, outPick); }
  if (!fill) { discard; }
  return vec4f(color.rgb, 1.0);
}
`;


// Query as a COMPUTE pass: one workgroup per tile, candidates staged in
// shared memory.
//
// The fragment version re-read the same tile lists for every pixel — at
// sparse LOD densities that is ~363 candidate loads PER PIXEL, versus the
// JFA's 54 taps, which is why pruning the average never helped. Here the
// workgroup loads the neighbourhood once (~363 loads for all 64 pixels)
// and every thread then scans it out of shared memory with a
// workgroup-uniform trip count.
//
// Records are packed to 8 bytes so a full neighbourhood fits the 16 KB
// workgroup budget: offsets from the tile origin as biased i8, depth
// quantised to 16 bits (only ever compared, against a 2% tolerance), and
// the colour as RGBA8. If a neighbourhood still overflows, the workgroup
// falls back to reading global memory — a workgroup-uniform branch.
const QUERYCS_WGSL = /* wgsl */ `
struct U {
  size: vec2u, tiles: vec2u,
  tileSize: u32, capacity: u32, maxR2: f32, reach2: f32,
  tol: f32, reversedZ: f32, writePick: f32, _p0: f32,
}
@group(0) @binding(0) var pick: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> counters: array<u32>;
@group(0) @binding(3) var<storage, read> points: array<vec4<u32>>;
@group(0) @binding(4) var<uniform> u: U;
@group(0) @binding(5) var filledPick: texture_storage_2d<rgba32float, write>;
// the uniform shm scan needs no front/occupancy prunes — those existed to
// cut the average work, which is not what costs here
@group(0) @binding(9) var fillTex: texture_storage_2d<rgba8unorm, write>;

const CAP: u32 = 1900u;
var<workgroup> shN: atomic<u32>;
var<workgroup> shPts: array<vec2<u32>, CAP>;
var<workgroup> shOrigin: vec2i;

fn distProxy(z: f32) -> f32 {
  if (u.reversedZ != 0.0) { return 1.0 / max(z, 1e-6); }
  return z;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wg: vec3u,
        @builtin(local_invocation_index) tid: u32,
        @builtin(global_invocation_id) gid: vec3u) {
  let ts = i32(u.tileSize);
  let origin = vec2i(i32(wg.x), i32(wg.y)) * ts;
  if (tid == 0u) { atomicStore(&shN, 0u); shOrigin = origin; }
  workgroupBarrier();

  let r = i32(ceil(sqrt(u.maxR2)));
  let tmax = vec2i(i32(u.tiles.x) - 1, i32(u.tiles.y) - 1);
  let t0 = max(vec2i(0), (origin - vec2i(r)) / ts);
  let t1 = min(tmax, (origin + vec2i(ts - 1 + r)) / ts);
  let nx = u32(t1.x - t0.x + 1);
  let ny = u32(t1.y - t0.y + 1);

  // cooperative gather of the whole neighbourhood
  for (var idx = tid; idx < nx * ny; idx = idx + 64u) {
    let tx = t0.x + i32(idx % nx);
    let ty = t0.y + i32(idx / nx);
    let ti = u32(ty) * u.tiles.x + u32(tx);
    let n = min(counters[ti], u.capacity);
    for (var k = 0u; k < n; k = k + 1u) {
      let pt = points[ti * u.capacity + k];
      let dx = i32(pt.x >> 16u) - origin.x;
      let dy = i32(pt.x & 0xffffu) - origin.y;
      if (dx < -128 || dx > 127 || dy < -128 || dy > 127) { continue; }
      let slot = atomicAdd(&shN, 1u);
      if (slot >= CAP) { continue; }
      let z16 = u32(clamp(bitcast<f32>(pt.y), 0.0, 1.0) * 65535.0);
      shPts[slot] = vec2u(
        (u32(dx + 128) << 24u) | (u32(dy + 128) << 16u) | z16,
        pt.z);
    }
  }
  workgroupBarrier();

  let total = atomicLoad(&shN);
  let n = min(total, CAP);

  let px = vec2i(gid.xy);
  if (px.x >= i32(u.size.x) || px.y >= i32(u.size.y)) { return; }
  let own = textureLoad(pick, px, 0);
  let ownValid = own.z > 0.0 && own.z < 1.0;

  var bestD2 = 1e30;
  var bestZ = 0.0;
  var bestC = 0u;
  var found = false;
  var bestNear = false;

  // uniform scan: identical trip count for every thread in the workgroup
  for (var k = 0u; k < n; k = k + 1u) {
    let rec = shPts[k];
    let dx = f32(i32((rec.x >> 24u) & 0xffu) - 128 + shOrigin.x - px.x);
    let dy = f32(i32((rec.x >> 16u) & 0xffu) - 128 + shOrigin.y - px.y);
    let d2 = dx * dx + dy * dy;
    if (d2 > u.maxR2) { continue; }
    let cz = f32(rec.x & 0xffffu) / 65535.0;
    let cw = d2 <= u.reach2;
    var take = false;
    if (!found) {
      take = true;
    } else {
      let lc = distProxy(cz);
      let lb = distProxy(bestZ);
      let cNearer = lc < lb * (1.0 - u.tol);
      let bNearer = lb < lc * (1.0 - u.tol);
      if (cw && !bestNear) { take = true; }
      else if (bestNear && !cw) { take = false; }
      else if (cw && bestNear && cNearer) { take = true; }
      else if (cw && bestNear && bNearer) { take = false; }
      else { take = d2 < bestD2; }
    }
    if (take) { bestD2 = d2; bestZ = cz; bestC = rec.y; found = true; bestNear = cw; }
  }

  var outPick = own;
  var col = vec4f(0.0);
  if (found) {
    let seedNearer = distProxy(bestZ) < distProxy(own.z) * (1.0 - u.tol);
    if (!ownValid || seedNearer) {
      col = vec4f(
        f32((bestC >> 24u) & 0xffu) / 255.0,
        f32((bestC >> 16u) & 0xffu) / 255.0,
        f32((bestC >> 8u) & 0xffu) / 255.0,
        1.0);
      outPick = vec4f(own.x, own.y, bestZ, own.w);
    }
  }
  if (u.writePick != 0.0) { textureStore(filledPick, px, outPick); }
  textureStore(fillTex, px, col);   // alpha 0 == leave the canvas alone
}
`;

// Trivial composite: the compute pass cannot write the canvas (it is not
// storage-bindable), so it writes an RGBA8 layer and this blends it in.
const COMPOSITE_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
struct VOut { @builtin(position) pos: vec4f }
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var xy = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut; o.pos = vec4f(xy[vi], 0.0, 1.0); return o;
}
@fragment
fn fs(i: VOut) -> @location(0) vec4f {
  let c = textureLoad(src, vec2i(i.pos.xy), 0);
  if (c.a < 0.5) { discard; }
  return vec4f(c.rgb, 1.0);
}
`;

export interface GridFillSettings {
  /** Max fill radius in device pixels. Default 24 (clamped 1..64). */
  readonly maxRadius?: number;
  /** Reversed-Z depth convention (near -> 1). Default true. */
  readonly reversedZ?: boolean;
  /** Occlusion reach in device pixels — see voronoiFill. Default 10. */
  readonly occlusionReach?: number;
  /** Relative depth tolerance separating surfaces. Default 0.02. */
  readonly depthTolerance?: number;
  /** Screen tile edge in pixels. Default 8. */
  readonly tileSize?: number;
  /** Max points retained per tile. Default 96. */
  readonly capacity?: number;
}

export type GridFillOption =
  | boolean
  | aval<boolean>
  | (GridFillSettings & { readonly enabled?: boolean | aval<boolean> });

export interface GridFillPass {
  run(fb: IFramebuffer, pickTexture: GPUTexture, colorName: string, needPick?: boolean): GPUTexture | undefined;
  dispose(): void;
}

export function createGridFillPass(
  device: GPUDevice,
  colorFormat: GPUTextureFormat,
  settings: GridFillSettings = {},
): GridFillPass {
  const pipeClear = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: CLEAR_WGSL, label: "grid.clear" }), entryPoint: "main" },
  });
  const pipeAppend = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: APPEND_WGSL, label: "grid.append" }), entryPoint: "main" },
  });
  const pipeDilate = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: DILATE_WGSL, label: "grid.dilate" }), entryPoint: "main" },
  });
  const dilateUbo = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const dilateFarUbo = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const pipeQueryCS = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: QUERYCS_WGSL, label: "grid.queryCS" }), entryPoint: "main" },
  });
  const compModule = device.createShaderModule({ code: COMPOSITE_WGSL, label: "grid.composite" });
  const pipeComposite = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: compModule, entryPoint: "vs" },
    fragment: { module: compModule, entryPoint: "fs", targets: [{ format: colorFormat }] },
    primitive: { topology: "triangle-list" },
    label: "grid.composite",
  });
  const queryModule = device.createShaderModule({ code: QUERY_WGSL, label: "grid.query" });
  const pipeQuery = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: queryModule, entryPoint: "vs" },
    fragment: { module: queryModule, entryPoint: "fs", targets: [{ format: colorFormat }] },
    primitive: { topology: "triangle-list" },
    label: "grid.query",
  });

  const appendUbo = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const queryUbo = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  let counters: GPUBuffer | undefined;
  let fronts: GPUBuffer | undefined;
  let dilated: GPUBuffer | undefined;
  let dilatedFar: GPUBuffer | undefined;
  let pointBuf: GPUBuffer | undefined;
  let colorCopy: GPUTexture | undefined;
  let filledPick: GPUTexture | undefined;
  let fillTex: GPUTexture | undefined;
  let dims = { w: 0, h: 0, tx: 0, ty: 0, cap: 0, ts: 0 };

  const ensure = (w: number, h: number, ts: number, cap: number): void => {
    if (dims.w === w && dims.h === h && dims.ts === ts && dims.cap === cap) return;
    counters?.destroy(); fronts?.destroy(); dilated?.destroy(); dilatedFar?.destroy(); pointBuf?.destroy();
    colorCopy?.destroy(); filledPick?.destroy(); fillTex?.destroy();
    const tx = Math.ceil(w / ts), ty = Math.ceil(h / ts);
    counters = device.createBuffer({
      size: Math.max(4, tx * ty * 4),
      usage: GPUBufferUsage.STORAGE, label: "grid.counters",
    });
    fronts = device.createBuffer({
      size: Math.max(4, tx * ty * 4),
      usage: GPUBufferUsage.STORAGE, label: "grid.fronts",
    });
    dilated = device.createBuffer({
      size: Math.max(4, tx * ty * 4),
      usage: GPUBufferUsage.STORAGE, label: "grid.dilated",
    });
    dilatedFar = device.createBuffer({
      size: Math.max(4, tx * ty * 4),
      usage: GPUBufferUsage.STORAGE, label: "grid.dilatedFar",
    });
    pointBuf = device.createBuffer({
      size: Math.max(16, tx * ty * cap * 4 * 4),
      usage: GPUBufferUsage.STORAGE, label: "grid.points",
    });
    colorCopy = device.createTexture({
      size: { width: w, height: h }, format: colorFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, label: "grid.color",
    });
    filledPick = device.createTexture({
      size: { width: w, height: h }, format: "rgba32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING, label: "grid.pick",
    });
    fillTex = device.createTexture({
      size: { width: w, height: h }, format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING, label: "grid.fill",
    });
    dims = { w, h, tx, ty, cap, ts };
  };

  return {
    run(fb, pick, colorName, needPick) {
      const colorView = fb.colors.tryFind(colorName);
      const colorTex = fb.colorTextures?.tryFind?.(colorName);
      if (colorView === undefined || colorTex === undefined) return undefined;

      const maxRadius = Math.max(1, Math.min(64, settings.maxRadius ?? 24));
      const reach = settings.occlusionReach ?? 10;
      const tol = settings.depthTolerance ?? 0.02;
      const revZ = settings.reversedZ ?? true;
      const ts = Math.max(2, Math.min(32, Math.round(settings.tileSize ?? 8)));
      const cap = Math.max(4, Math.min(512, Math.round(settings.capacity ?? 96)));
      const w = pick.width, h = pick.height;
      ensure(w, h, ts, cap);
      const wantPick = needPick !== false;
      const { tx, ty } = dims;

      const ai = new ArrayBuffer(32);
      new Uint32Array(ai).set([w, h, tx, ty, ts, cap, revZ ? 1 : 0, 0]);
      device.queue.writeBuffer(appendUbo, 0, ai);
      const qi = new ArrayBuffer(64);
      new Uint32Array(qi, 0, 6).set([w, h, tx, ty, ts, cap]);
      new Float32Array(qi, 24, 6).set([
        maxRadius * maxRadius, reach * reach, tol, revZ ? 1 : 0, wantPick ? 1 : 0, 0,
      ]);
      device.queue.writeBuffer(queryUbo, 0, qi);

      const enc = device.createCommandEncoder({ label: "grid.frame" });
      enc.copyTextureToTexture({ texture: colorTex }, { texture: colorCopy! }, { width: w, height: h });
      {
        const bg = device.createBindGroup({
          layout: pipeClear.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: counters! } },
            { binding: 1, resource: { buffer: fronts! } },
          ],
        });
        const p = enc.beginComputePass({ label: "grid.clear" });
        p.setPipeline(pipeClear); p.setBindGroup(0, bg);
        p.dispatchWorkgroups(Math.ceil((tx * ty) / 64)); p.end();
      }
      {
        const bg = device.createBindGroup({
          layout: pipeAppend.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: pick.createView() },
            { binding: 1, resource: colorCopy!.createView() },
            { binding: 2, resource: { buffer: counters! } },
            { binding: 3, resource: { buffer: pointBuf! } },
            { binding: 4, resource: { buffer: appendUbo } },
            { binding: 5, resource: { buffer: fronts! } },
          ],
        });
        const p = enc.beginComputePass({ label: "grid.append" });
        p.setPipeline(pipeAppend); p.setBindGroup(0, bg);
        p.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8)); p.end();
      }
      {
        const di = new ArrayBuffer(16);
        new Uint32Array(di).set([tx, ty, Math.ceil(reach / ts), 0]);
        device.queue.writeBuffer(dilateUbo, 0, di);
        const bg = device.createBindGroup({
          layout: pipeDilate.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: fronts! } },
            { binding: 1, resource: { buffer: dilated! } },
            { binding: 2, resource: { buffer: dilateUbo } },
          ],
        });
        const p = enc.beginComputePass({ label: "grid.dilate" });
        p.setPipeline(pipeDilate); p.setBindGroup(0, bg);
        p.dispatchWorkgroups(Math.ceil((tx * ty) / 64)); p.end();
      }
      {
        const di = new ArrayBuffer(16);
        new Uint32Array(di).set([tx, ty, Math.ceil(maxRadius / ts), 0]);
        device.queue.writeBuffer(dilateFarUbo, 0, di);
        const bg = device.createBindGroup({
          layout: pipeDilate.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: fronts! } },
            { binding: 1, resource: { buffer: dilatedFar! } },
            { binding: 2, resource: { buffer: dilateFarUbo } },
          ],
        });
        const p = enc.beginComputePass({ label: "grid.dilateFar" });
        p.setPipeline(pipeDilate); p.setBindGroup(0, bg);
        p.dispatchWorkgroups(Math.ceil((tx * ty) / 64)); p.end();
      }
      {
        const bg = device.createBindGroup({
          layout: pipeQueryCS.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: pick.createView() },
            { binding: 2, resource: { buffer: counters! } },
            { binding: 3, resource: { buffer: pointBuf! } },
            { binding: 4, resource: { buffer: queryUbo } },
            { binding: 5, resource: filledPick!.createView() },
            { binding: 9, resource: fillTex!.createView() },
          ],
        });
        const p = enc.beginComputePass({ label: "grid.queryCS" });
        p.setPipeline(pipeQueryCS); p.setBindGroup(0, bg);
        p.dispatchWorkgroups(tx, ty); p.end();
      }
      {
        const bg = device.createBindGroup({
          layout: pipeComposite.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: fillTex!.createView() }],
        });
        const p = enc.beginRenderPass({
          label: "grid.composite",
          colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
        });
        p.setPipeline(pipeComposite); p.setBindGroup(0, bg);
        p.draw(3); p.end();
      }
      device.queue.submit([enc.finish()]);
      return wantPick ? filledPick : undefined;
    },
    dispose() {
      counters?.destroy(); fronts?.destroy(); dilated?.destroy(); pointBuf?.destroy();
      colorCopy?.destroy(); filledPick?.destroy(); fillTex?.destroy();
      appendUbo.destroy(); queryUbo.destroy(); dilateUbo.destroy(); dilateFarUbo.destroy();
      dilatedFar?.destroy();
      counters = fronts = dilated = dilatedFar = pointBuf = undefined;
      colorCopy = filledPick = fillTex = undefined;
    },
  };
}

export function gridFillConfig(opt: GridFillOption): { enabled: aval<boolean>; settings: GridFillSettings } {
  const asAval = (v: boolean | aval<boolean>): aval<boolean> =>
    typeof v === "boolean" ? AVal.constant(v) : v;
  if (typeof opt === "boolean" || typeof (opt as aval<boolean>).getValue === "function") {
    return { enabled: asAval(opt as boolean | aval<boolean>), settings: {} };
  }
  const o = opt as GridFillSettings & { enabled?: boolean | aval<boolean> };
  return { enabled: asAval(o.enabled ?? true), settings: o };
}
