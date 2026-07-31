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
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= arrayLength(&counters)) { return; }
  atomicStore(&counters[gid.x], 0u);
}
`;

const APPEND_WGSL = /* wgsl */ `
struct U {
  size: vec2u, tiles: vec2u,
  tileSize: u32, capacity: u32, _p0: u32, _p1: u32,
}
@group(0) @binding(0) var pick: texture_2d<f32>;
@group(0) @binding(1) var colorTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> points: array<u32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u.size.x || gid.y >= u.size.y) { return; }
  let px = vec2i(gid.xy);
  let z = textureLoad(pick, px, 0).z;
  if (z <= 0.0 || z >= 1.0) { return; }          // no point on this pixel
  let tile = gid.xy / u.tileSize;
  let ti = tile.y * u.tiles.x + tile.x;
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
  let base = (ti * u.capacity + slot) * 3u;
  points[base + 0u] = (gid.x << 16u) | gid.y;
  points[base + 1u] = bitcast<u32>(z);
  points[base + 2u] = rgba;
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
@group(0) @binding(3) var<storage, read> points: array<u32>;
@group(0) @binding(4) var<uniform> u: U;
@group(0) @binding(5) var filledPick: texture_storage_2d<rgba32float, write>;

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

  let r = i32(ceil(sqrt(u.maxR2)));
  let ts = i32(u.tileSize);
  let t0 = max(vec2i(0), (px - vec2i(r)) / ts);
  let t1 = min(vec2i(i32(u.tiles.x) - 1, i32(u.tiles.y) - 1), (px + vec2i(r)) / ts);

  var bestD2 = 1e30;
  var bestZ = 0.0;
  var bestC = 0u;
  var found = false;

  for (var ty = t0.y; ty <= t1.y; ty = ty + 1) {
    for (var tx = t0.x; tx <= t1.x; tx = tx + 1) {
      let ti = u32(ty) * u.tiles.x + u32(tx);
      let n = min(counters[ti], u.capacity);
      for (var k = 0u; k < n; k = k + 1u) {
        let base = (ti * u.capacity + k) * 3u;
        let xy = points[base + 0u];
        let dx = f32(i32(xy >> 16u) - px.x);
        let dy = f32(i32(xy & 0xffffu) - px.y);
        let d2 = dx * dx + dy * dy;
        if (d2 > u.maxR2) { continue; }
        let cz = bitcast<f32>(points[base + 1u]);
        var take = false;
        if (!found) {
          take = true;
        } else {
          // identical rule set to voronoiFill's STEP comparator: a
          // candidate within the occlusion reach represents a continuous
          // surface, so the NEARER one wins outright there; a
          // within-reach candidate beats any beyond-reach one; only when
          // both are beyond reach does plain 2D distance decide.
          let cw = d2 <= u.reach2;
          let bw = bestD2 <= u.reach2;
          let lc = distProxy(cz);
          let lb = distProxy(bestZ);
          let cNearer = lc < lb * (1.0 - u.tol);
          let bNearer = lb < lc * (1.0 - u.tol);
          if (cw && !bw) { take = true; }
          else if (bw && !cw) { take = false; }
          else if (cw && bw && cNearer) { take = true; }
          else if (cw && bw && bNearer) { take = false; }
          else { take = d2 < bestD2; }
        }
        if (take) {
          bestD2 = d2; bestZ = cz; bestC = points[base + 2u]; found = true;
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
  let pointBuf: GPUBuffer | undefined;
  let colorCopy: GPUTexture | undefined;
  let filledPick: GPUTexture | undefined;
  let dims = { w: 0, h: 0, tx: 0, ty: 0, cap: 0, ts: 0 };

  const ensure = (w: number, h: number, ts: number, cap: number): void => {
    if (dims.w === w && dims.h === h && dims.ts === ts && dims.cap === cap) return;
    counters?.destroy(); pointBuf?.destroy();
    colorCopy?.destroy(); filledPick?.destroy();
    const tx = Math.ceil(w / ts), ty = Math.ceil(h / ts);
    counters = device.createBuffer({
      size: Math.max(4, tx * ty * 4),
      usage: GPUBufferUsage.STORAGE, label: "grid.counters",
    });
    pointBuf = device.createBuffer({
      size: Math.max(16, tx * ty * cap * 3 * 4),
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
      new Uint32Array(ai).set([w, h, tx, ty, ts, cap, 0, 0]);
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
          entries: [{ binding: 0, resource: { buffer: counters! } }],
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
          ],
        });
        const p = enc.beginComputePass({ label: "grid.append" });
        p.setPipeline(pipeAppend); p.setBindGroup(0, bg);
        p.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8)); p.end();
      }
      {
        const bg = device.createBindGroup({
          layout: pipeQuery.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: pick.createView() },
            { binding: 2, resource: { buffer: counters! } },
            { binding: 3, resource: { buffer: pointBuf! } },
            { binding: 4, resource: { buffer: queryUbo } },
            { binding: 5, resource: filledPick!.createView() },
          ],
        });
        const p = enc.beginRenderPass({
          label: "grid.query",
          colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
        });
        p.setPipeline(pipeQuery); p.setBindGroup(0, bg);
        p.draw(3); p.end();
      }
      device.queue.submit([enc.finish()]);
      return wantPick ? filledPick : undefined;
    },
    dispose() {
      counters?.destroy(); pointBuf?.destroy();
      colorCopy?.destroy(); filledPick?.destroy();
      appendUbo.destroy(); queryUbo.destroy();
      counters = pointBuf = undefined;
      colorCopy = filledPick = undefined;
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
