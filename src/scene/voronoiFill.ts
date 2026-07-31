// Screen-space Voronoi fill for 1px point rendering, via the Jump
// Flooding Algorithm: each rendered point is a seed; O(log r) ping-pong
// passes propagate the nearest seed to every pixel within `maxRadius`.
// Hole pixels then take their seed's color (Voronoi cells — splats that
// grow with local point spacing and clip at bisectors), and a filled
// pick-shaped depth texture is produced so the planeFit/AO pass shades
// the FILLED surface.
//
// Cost: init + ceil(log2(maxRadius)) + 1 JFA steps (9 taps each) + a
// resolve blit. maxRadius 32 -> 7 compute passes.
//
// Same G-buffer contract as gtao/planeFit: pickId slot .z = NDC depth.

import { AVal, type aval } from "@aardworx/wombat.adaptive";
import type { IFramebuffer } from "@aardworx/wombat.rendering/core";

const INIT_WGSL = /* wgsl */ `
@group(0) @binding(0) var pick: texture_2d<f32>;
@group(0) @binding(1) var seedOut: texture_storage_2d<rg32uint, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dim = textureDimensions(seedOut);
  if (gid.x >= dim.x || gid.y >= dim.y) { return; }
  let z = textureLoad(pick, vec2i(gid.xy), 0).z;
  if (z > 0.0 && z < 1.0) {
    textureStore(seedOut, vec2i(gid.xy), vec4u((gid.x << 16) | gid.y, bitcast<u32>(z), 0, 0));
  } else {
    textureStore(seedOut, vec2i(gid.xy), vec4u(0xffffffffu, 0, 0, 0));
  }
}
`;

const STEP_WGSL = /* wgsl */ `
struct U { step: i32, reversedZ: i32, reach2: f32, tol: f32 }
@group(0) @binding(0) var seedIn: texture_2d<u32>;
@group(0) @binding(1) var seedOut: texture_storage_2d<rg32uint, write>;
@group(0) @binding(2) var<uniform> u: U;

// monotone proxy for view distance from NDC depth: reversed-Z -> 1/z
// grows with distance; classic -> z grows with distance. We only ever
// COMPARE, so any monotone map works.
fn distProxy(z: f32) -> f32 {
  if (u.reversedZ != 0) { return 1.0 / max(z, 1e-6); }
  return z;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dim = vec2i(textureDimensions(seedOut));
  if (gid.x >= u32(dim.x) || gid.y >= u32(dim.y)) { return; }
  let px = vec2i(gid.xy);
  var bestKey = 0xffffffffu;
  var bestDepth = 0u;
  var bestD2 = 0x7fffffff;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let q = px + vec2i(dx, dy) * u.step;
      if (q.x < 0 || q.y < 0 || q.x >= dim.x || q.y >= dim.y) { continue; }
      let cand = textureLoad(seedIn, q, 0);
      if (cand.x == 0xffffffffu) { continue; }
      let sx = i32(cand.x >> 16);
      let sy = i32(cand.x & 0xffffu);
      let ddx = sx - px.x;
      let ddy = sy - px.y;
      let d2 = ddx * ddx + ddy * ddy;
      var take = false;
      if (bestKey == 0xffffffffu) {
        take = true;
      } else {
        // DEPTH-AWARE comparison. A seed within the occlusion reach
        // plausibly represents a continuous surface at this pixel, so
        // among within-reach seeds the NEARER surface wins outright —
        // regardless of 2D distance (a close-up surface projects its
        // points sparsely while the background behind it is dense; pure
        // 2D-nearest would make it see-through). A within-reach seed
        // also beats any beyond-reach one. Only when both are beyond
        // reach (spacing coarser than the reach, e.g. distant overview)
        // does plain 2D Voronoi apply.
        let cw = f32(d2) <= u.reach2;
        let bw = f32(bestD2) <= u.reach2;
        let lc = distProxy(bitcast<f32>(cand.y));
        let lb = distProxy(bitcast<f32>(bestDepth));
        let cNearer = lc < lb * (1.0 - u.tol);
        let bNearer = lb < lc * (1.0 - u.tol);
        if (cw && !bw) { take = true; }
        else if (bw && !cw) { take = false; }
        else if (cw && bw && cNearer) { take = true; }
        else if (cw && bw && bNearer) { take = false; }
        else { take = d2 < bestD2; }
      }
      if (take) {
        bestD2 = d2;
        bestKey = cand.x;
        bestDepth = cand.y;
      }
    }
  }
  textureStore(seedOut, px, vec4u(bestKey, bestDepth, 0, 0));
}
`;

// Resolve: fill hole pixels' colour from their seed; also write the
// filled pick-shaped depth texture (z slot) for downstream shading.
const RESOLVE_WGSL = /* wgsl */ `
struct U { maxR2: f32, tol: f32, reversedZ: f32, writePick: f32,
           support: f32, _p0: f32, _p1: f32, _p2: f32 }
@group(0) @binding(0) var seed: texture_2d<u32>;
@group(0) @binding(1) var colorCopy: texture_2d<f32>;
@group(0) @binding(2) var pick: texture_2d<f32>;
@group(0) @binding(3) var filledPick: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<uniform> u: U;

struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f }

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var xy = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let p = xy[vi];
  var o: VOut;
  o.pos = vec4f(p, 0.0, 1.0);
  o.uv = vec2f(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
  return o;
}

fn distProxy(z: f32) -> f32 {
  if (u.reversedZ != 0.0) { return 1.0 / max(z, 1e-6); }
  return z;
}

// Is this hole INSIDE the point set, or off its edge?
//
// A pixel is interior iff the directions to the seeds around it do NOT
// all fit in a half-plane — i.e. they span >180deg. If they all sit on
// one side, the pixel is beyond the silhouette and filling it would
// EXTRAPOLATE, which is what blooms an isolated point into a full-radius
// disc ("exterior blur").
//
// Distinct seeds come free from the finished Voronoi map: sampling it at
// an offset returns whichever seed owns THAT position, which on the far
// side of a gap is a different one. 8 probes, and only for pixels far
// enough from their seed to be at risk — near ones fill unconditionally.
fn interior(px: vec2i, dim: vec2i, R: f32) -> bool {
  var sectors: u32 = 0u;
  for (var k = 0; k < 8; k = k + 1) {
    let a = f32(k) * 0.78539816;
    let q = vec2i(round(vec2f(px) + vec2f(cos(a), sin(a)) * R));
    if (q.x < 0 || q.y < 0 || q.x >= dim.x || q.y >= dim.y) { continue; }
    let sd = textureLoad(seed, q, 0);
    if (sd.x == 0xffffffffu) { continue; }
    let d = vec2f(f32(i32(sd.x >> 16)), f32(i32(sd.x & 0xffffu))) - vec2f(px);
    let len = length(d);
    if (len < 0.5) { return true; }        // a seed sits on us
    var ang = atan2(d.y, d.x);
    if (ang < 0.0) { ang = ang + 6.28318531; }
    sectors = sectors | (1u << (u32(ang / 0.78539816) & 7u));
  }
  if (sectors == 0u) { return false; }
  // any 180deg window (4 of 8 sectors) containing every occupied sector
  // means the seeds are all on one side -> extrapolating
  for (var r = 0u; r < 8u; r = r + 1u) {
    let w = ((0xFu << r) | (0xFu >> (8u - r))) & 0xFFu;
    if ((sectors & ~w & 0xFFu) == 0u) { return false; }
  }
  return true;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4f {
  let px = vec2i(i.pos.xy);
  let dim = vec2i(textureDimensions(seed));
  let own = textureLoad(pick, px, 0);
  let s = textureLoad(seed, px, 0);
  let ownValid = own.z > 0.0 && own.z < 1.0;
  var outPick = own;
  var color = vec4f(0.0);
  var fill = false;
  if (s.x != 0xffffffffu) {
    let sx = i32(s.x >> 16);
    let sy = i32(s.x & 0xffffu);
    let dx = f32(sx - px.x);
    let dy = f32(sy - px.y);
    let d2 = dx * dx + dy * dy;
    let dd = sqrt(d2);
    // near seeds fill unconditionally; only far ones can be extrapolation
    let ok = u.support <= 0.0 || dd <= u.support || interior(px, dim, dd);
    if (d2 <= u.maxR2 && ok) {
      let sz = bitcast<f32>(s.y);
      // Fill holes; ALSO overwrite rendered points that lie distinctly
      // BEHIND the winning surface (screen-space hidden-point removal —
      // background points otherwise speckle through close-up surfaces).
      let seedNearer = distProxy(sz) < distProxy(own.z) * (1.0 - u.tol);
      if (!ownValid || seedNearer) {
        color = textureLoad(colorCopy, vec2i(sx, sy), 0);
        outPick = vec4f(own.x, own.y, sz, own.w);
        fill = true;
      }
    }
  }
  // Only pay the full-res rgba32float write when something downstream
  // actually shades from it (16 B/px adds up fast on mobile).
  if (u.writePick != 0.0) { textureStore(filledPick, px, outPick); }
  if (!fill) { discard; }
  return vec4f(color.rgb, 1.0);
}
`;

export interface VoronoiFillSettings {
  /** Max fill radius in device pixels. Default 24 (clamped 1..64). */
  readonly maxRadius?: number;
  /** Reversed-Z depth convention (near -> 1). Default true. */
  readonly reversedZ?: boolean;
  /** Occlusion reach in device pixels: within this radius a nearer seed
   *  represents a continuous surface and occludes farther seeds outright.
   *  Choose a small multiple of the LOD target point spacing. Default 10. */
  readonly occlusionReach?: number;
  /** Relative depth tolerance separating "same surface" from
   *  "occluding surface". Default 0.02. */
  readonly depthTolerance?: number;
  /** Silhouette tightness, in pixels. Holes whose nearest seed is
   *  farther than this run the 180deg interior test (see `interior`) and
   *  are only filled when seeds surround them — so isolated points stop
   *  blooming into full-radius discs. Smaller = tighter outlines and
   *  more probes; 0 disables the test. Default 0. */
  readonly exteriorSupport?: number;
}

export type VoronoiFillOption =
  | boolean
  | aval<boolean>
  | (VoronoiFillSettings & { readonly enabled?: boolean | aval<boolean> });

export interface VoronoiFillPass {
  /** Fill the canvas holes. Returns the filled pick-shaped texture
   *  (z = filled NDC depth) for downstream passes, or undefined if the
   *  pass could not run. Call AFTER the scene render, BEFORE shading. */
  run(fb: IFramebuffer, pickTexture: GPUTexture, colorName: string, needPick?: boolean): GPUTexture | undefined;
  dispose(): void;
}

export function createVoronoiFillPass(
  device: GPUDevice,
  colorFormat: GPUTextureFormat,
  settings: VoronoiFillSettings = {},
): VoronoiFillPass {
  const pipeInit = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: INIT_WGSL, label: "vfill.init" }), entryPoint: "main" },
  });
  const pipeStep = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: STEP_WGSL, label: "vfill.step" }), entryPoint: "main" },
  });
  const resolveModule = device.createShaderModule({ code: RESOLVE_WGSL, label: "vfill.resolve" });
  const pipeResolve = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: resolveModule, entryPoint: "vs" },
    fragment: { module: resolveModule, entryPoint: "fs", targets: [{ format: colorFormat }] },
    primitive: { topology: "triangle-list" },
    label: "vfill.resolve",
  });
  const stepUbos: GPUBuffer[] = [];
  const resolveUbo = device.createBuffer({
    size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "vfill.u",
  });

  let seedA: GPUTexture | undefined;
  let seedB: GPUTexture | undefined;
  let colorCopy: GPUTexture | undefined;
  let filledPick: GPUTexture | undefined;

  const ensure = (w: number, h: number): void => {
    if (seedA !== undefined && seedA.width === w && seedA.height === h) return;
    for (const t of [seedA, seedB, colorCopy, filledPick]) t?.destroy();
    const mkSeed = (label: string): GPUTexture => device.createTexture({
      size: { width: w, height: h }, format: "rg32uint",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING, label,
    });
    seedA = mkSeed("vfill.seedA");
    seedB = mkSeed("vfill.seedB");
    colorCopy = device.createTexture({
      size: { width: w, height: h }, format: colorFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, label: "vfill.color",
    });
    filledPick = device.createTexture({
      size: { width: w, height: h }, format: "rgba32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING, label: "vfill.pick",
    });
  };

  return {
    run(fb, pick, colorName, needPick) {
      const colorView = fb.colors.tryFind(colorName);
      const colorTex = fb.colorTextures?.tryFind?.(colorName);
      if (colorView === undefined || colorTex === undefined) return undefined;
      // Settings are read per frame so callers can mutate the object live.
      const maxRadius = Math.max(1, Math.min(64, settings.maxRadius ?? 24));
      const reach = settings.occlusionReach ?? 10;
      const tol = settings.depthTolerance ?? 0.02;
      const revZ = settings.reversedZ ?? true;
      const wantPick = needPick !== false;
      device.queue.writeBuffer(resolveUbo, 0, new Float32Array([
        maxRadius * maxRadius, tol, revZ ? 1 : 0, wantPick ? 1 : 0,
        settings.exteriorSupport ?? 0, 0, 0, 0,
      ]));
      const w = pick.width;
      const h = pick.height;
      ensure(w, h);

      const enc = device.createCommandEncoder({ label: "vfill.frame" });
      enc.copyTextureToTexture({ texture: colorTex }, { texture: colorCopy! }, { width: w, height: h });
      {
        const bg = device.createBindGroup({
          layout: pipeInit.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: pick.createView() },
            { binding: 1, resource: seedA!.createView() },
          ],
        });
        const pass = enc.beginComputePass({ label: "vfill.init" });
        pass.setPipeline(pipeInit);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
        pass.end();
      }
      // JFA: steps maxRadius -> 1, halving
      let steps: number[] = [];
      for (let s = 1 << Math.ceil(Math.log2(maxRadius)); s >= 1; s >>= 1) steps.push(s);
      while (stepUbos.length < steps.length) {
        const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        stepUbos.push(b);
      }
      let src = seedA!;
      let dst = seedB!;
      steps.forEach((s, i) => {
        const ub = new ArrayBuffer(16);
        new Int32Array(ub, 0, 2).set([s, revZ ? 1 : 0]);
        new Float32Array(ub, 8, 2).set([reach * reach, tol]);
        const ubo = stepUbos[i]!;
        device.queue.writeBuffer(ubo, 0, ub);
        const bg = device.createBindGroup({
          layout: pipeStep.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: src.createView() },
            { binding: 1, resource: dst.createView() },
            { binding: 2, resource: { buffer: ubo } },
          ],
        });
        const pass = enc.beginComputePass({ label: "vfill.step" });
        pass.setPipeline(pipeStep);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
        pass.end();
        const t = src; src = dst; dst = t;
      });
      {
        const bg = device.createBindGroup({
          layout: pipeResolve.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: src.createView() },
            { binding: 1, resource: colorCopy!.createView() },
            { binding: 2, resource: pick.createView() },
            { binding: 3, resource: filledPick!.createView() },
            { binding: 4, resource: { buffer: resolveUbo } },
          ],
        });
        const pass = enc.beginRenderPass({
          label: "vfill.resolve",
          colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
        });
        pass.setPipeline(pipeResolve);
        pass.setBindGroup(0, bg);
        pass.draw(3);
        pass.end();
      }
      device.queue.submit([enc.finish()]);
      return wantPick ? filledPick : undefined;
    },
    dispose() {
      for (const t of [seedA, seedB, colorCopy, filledPick]) t?.destroy();
      seedA = seedB = colorCopy = filledPick = undefined;
      resolveUbo.destroy();
      for (const b of stepUbos) b.destroy();
    },
  };
}

export function voronoiFillConfig(opt: VoronoiFillOption): { enabled: aval<boolean>; settings: VoronoiFillSettings } {
  const asAval = (v: boolean | aval<boolean>): aval<boolean> =>
    typeof v === "boolean" ? AVal.constant(v) : v;
  if (typeof opt === "boolean" || typeof (opt as aval<boolean>).getValue === "function") {
    return { enabled: asAval(opt as boolean | aval<boolean>), settings: {} };
  }
  const o = opt as VoronoiFillSettings & { enabled?: boolean | aval<boolean> };
  return { enabled: asAval(o.enabled ?? true), settings: o };
}
