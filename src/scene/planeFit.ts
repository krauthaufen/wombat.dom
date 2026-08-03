// Screen-space plane-fit shading driven by the pick attachment (same
// G-buffer contract as gtao.ts: pickId slot .z carries NDC depth).
//
// Port of Aardvark.Rendering.PointSet's `blitPlaneFit` idea: for every
// pixel, reconstruct view-space positions of a jittered pixel
// neighbourhood, fit a plane (covariance + smallest eigenvector), and
// light the pixel from the FITTED normal — the smoothing this produces
// is what makes 1px/splat point clouds read as surfaces. Output is a
// shade factor multiplied into the canvas colour (same pure-post-effect
// contract as the AO pass; the two compose by multiplication).
//
// Reversed-Z safe for the same reason gtao is: reconstruction goes
// through projInv, background depth rejected by the z guard.

import { AVal, type aval } from "@aardworx/wombat.adaptive";
import type { Trafo3d } from "@aardworx/wombat.base";
import type { IFramebuffer } from "@aardworx/wombat.rendering/core";

const FIT_WGSL = /* wgsl */ `
struct U {
  projInv: mat4x4f,
  radiusPx: f32,
  tolerance: f32,   // view-space gate, relative to |z|
  ambient: f32,     // shade floor
  strength: f32,    // 0..1 blend between flat and fitted shading
  aoRadius: f32,    // world-space AO radius (0 = AO off)
  aoIntensity: f32,
  projM11: f32,
  _pad: f32,
  size: vec2u,       // pick texture (device px)
  shadeSize: vec2u,  // output texture (may be scaled down)
}
@group(0) @binding(0) var pick: texture_2d<f32>;
@group(0) @binding(1) var outShade: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> u: U;

fn viewPosAt(px: vec2i) -> vec4f {
  let s = textureLoad(pick, px, 0);
  let z = s.z;
  if (z <= 0.0 || z >= 1.0) { return vec4f(0.0); }
  let uv = (vec2f(px) + 0.5) / vec2f(u.size);
  let ndc = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
  let v = u.projInv * ndc;
  return vec4f(v.xyz / v.w, 1.0);
}

// 24-sample Vogel disk (unit radius).
const N = 24u;
fn diskOffset(i: u32) -> vec2f {
  let r = sqrt((f32(i) + 0.5) / f32(N));
  let a = f32(i) * 2.3999632; // golden angle
  return vec2f(cos(a), sin(a)) * r;
}

// Smallest eigenvector of a symmetric 3x3 matrix (Smith's trig method
// for the eigenvalue, then row-cross for the vector).
fn smallestEigenvector(a: mat3x3f) -> vec3f {
  let p1 = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
  if (p1 < 1e-12) {
    // diagonal: pick the axis of the smallest diagonal entry
    var mn = a[0][0]; var idx = 0;
    if (a[1][1] < mn) { mn = a[1][1]; idx = 1; }
    if (a[2][2] < mn) { idx = 2; }
    if (idx == 0) { return vec3f(1.0, 0.0, 0.0); }
    if (idx == 1) { return vec3f(0.0, 1.0, 0.0); }
    return vec3f(0.0, 0.0, 1.0);
  }
  let q = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
  let d0 = a[0][0] - q; let d1 = a[1][1] - q; let d2 = a[2][2] - q;
  let p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * p1;
  let p = sqrt(p2 / 6.0);
  let bm = (1.0 / p) * mat3x3f(
    vec3f(d0, a[0][1], a[0][2]),
    vec3f(a[0][1], d1, a[1][2]),
    vec3f(a[0][2], a[1][2], d2));
  let detB = bm[0][0] * (bm[1][1] * bm[2][2] - bm[1][2] * bm[2][1])
           - bm[0][1] * (bm[1][0] * bm[2][2] - bm[1][2] * bm[2][0])
           + bm[0][2] * (bm[1][0] * bm[2][1] - bm[1][1] * bm[2][0]);
  let r = clamp(detB / 2.0, -1.0, 1.0);
  let phi = acos(r) / 3.0;
  let eig1 = q + 2.0 * p * cos(phi);                       // largest
  let eig3 = q + 2.0 * p * cos(phi + 2.0943951);           // smallest
  let eig2 = 3.0 * q - eig1 - eig3;
  // eigenvector for eig3: cross of two independent rows of (A - eig3*I)
  let m0 = vec3f(a[0][0] - eig3, a[0][1], a[0][2]);
  let m1 = vec3f(a[0][1], a[1][1] - eig3, a[1][2]);
  let m2 = vec3f(a[0][2], a[1][2], a[2][2] - eig3);
  var v = cross(m0, m1);
  var l = dot(v, v);
  let v2 = cross(m0, m2);
  let l2 = dot(v2, v2);
  if (l2 > l) { v = v2; l = l2; }
  let v3 = cross(m1, m2);
  let l3 = dot(v3, v3);
  if (l3 > l) { v = v3; l = l3; }
  if (l < 1e-16) { return vec3f(0.0, 0.0, 1.0); }
  return normalize(v);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u.shadeSize.x || gid.y >= u.shadeSize.y) { return; }
  let opx = vec2i(gid.xy);
  // map shade pixel -> pick pixel (identity at scale 1)
  let sc = vec2f(u.size) / vec2f(u.shadeSize);
  let px = vec2i((vec2f(gid.xy) + 0.5) * sc);
  let p = viewPosAt(px);
  if (p.w == 0.0) { textureStore(outShade, opx, vec4f(1.0, 1.0, 1.0, 0.0)); return; }

  let tol = max(0.05, u.tolerance * abs(p.z));

  // gather neighbourhood
  var sum = p.xyz;
  var cnt = 1.0;
  var pts: array<vec3f, 24>;
  var np = 0u;
  for (var i = 0u; i < N; i++) {
    let off = diskOffset(i) * u.radiusPx;
    let q = vec2i(vec2f(px) + off);
    if (q.x < 0 || q.y < 0 || q.x >= i32(u.size.x) || q.y >= i32(u.size.y)) { continue; }
    let vq = viewPosAt(q);
    if (vq.w == 0.0) { continue; }
    if (abs(vq.z - p.z) > tol) { continue; }
    pts[np] = vq.xyz;
    np++;
    sum += vq.xyz;
    cnt += 1.0;
  }

  var shade = 1.0;
  var nrm = vec3f(0.0);
  var haveN = false;
  if (np >= 5u) {
    let c = sum / cnt;
    var cov = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
    let d0 = p.xyz - c;
    cov[0] += d0 * d0.x; cov[1] += d0 * d0.y; cov[2] += d0 * d0.z;
    for (var i = 0u; i < np; i++) {
      let d = pts[i] - c;
      cov[0] += d * d.x; cov[1] += d * d.y; cov[2] += d * d.z;
    }
    var n = smallestEigenvector(cov);
    let toCam = normalize(-p.xyz);
    if (dot(n, toCam) < 0.0) { n = -n; }   // orient toward the camera
    nrm = n;
    haveN = true;
    let ndl = abs(dot(n, toCam));
    shade = u.ambient + (1.0 - u.ambient) * ndl;
  }
  shade = mix(1.0, shade, u.strength);

  // AO against the FITTED normal (the pick attachment's normal slot is
  // effect-dependent and may be absent — the fit is the ground truth here).
  if (u.aoRadius > 0.0 && haveN) {
    let pixR = u.aoRadius * u.projM11 * f32(u.size.y) / (2.0 * max(0.1, abs(p.z)));
    let stepPx = max(1.0, pixR / 4.0);
    var occl = 0.0;
    let rot = fract(sin(f32(gid.x) * 12.9898 + f32(gid.y) * 78.233) * 43758.5453) * 6.2831853;
    for (var d = 0u; d < 8u; d++) {
      let a = rot + f32(d) * (6.2831853 / 8.0);
      let dir = vec2f(cos(a), sin(a));
      var best = 0.0;
      for (var st = 1u; st <= 4u; st++) {
        let q = vec2i(vec2f(px) + dir * stepPx * f32(st));
        if (q.x < 0 || q.y < 0 || q.x >= i32(u.size.x) || q.y >= i32(u.size.y)) { break; }
        let vq = viewPosAt(q);
        if (vq.w == 0.0) { continue; }
        let dl = vq.xyz - p.xyz;
        let dist = length(dl);
        if (dist < 1e-4 || dist > u.aoRadius) { continue; }
        let h = dot(nrm, dl / dist) - 0.08;
        if (h > best) { best = h; }
      }
      occl += max(0.0, best);
    }
    let ao = clamp(1.0 - occl / 8.0 * u.aoIntensity, 0.0, 1.0);
    shade = shade * ao;
  }
  textureStore(outShade, opx, vec4f(shade, shade, shade, 1.0));
}
`;

const APPLY_WGSL = /* wgsl */ `
@group(0) @binding(0) var shade: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

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

@fragment
fn fs(i: VOut) -> @location(0) vec4f {
  let t = textureSample(shade, samp, i.uv);
  // alpha 0 = no geometry here. In multiply mode writing 1 is a no-op,
  // but white mode REPLACES the colour, so painting the background with
  // the default shade of 1 would turn the sky white.
  if (t.w < 0.5) { discard; }
  return vec4f(t.x, t.x, t.x, 1.0);
}
`;

export interface PlaneFitSettings {
  /** Neighbourhood radius in device pixels. Default 7. */
  readonly radius?: number;
  /** View-space depth gate relative to |z|. Default 0.01 (min 5 cm). */
  readonly tolerance?: number;
  /** Shade floor (ambient). Default 0.35. */
  readonly ambient?: number;
  /** Replace surface colour with a white diffuse headlight instead of
   *  modulating the point colours. Default false. */
  readonly white?: boolean;
  /** Blend between unshaded and fitted shading, 0..1. Default 1. */
  readonly strength?: number;
  /** World-space AO radius computed against the FITTED normal.
   *  0 disables. Default 0.6. */
  readonly aoRadius?: number;
  /** AO strength. Default 2. */
  readonly aoIntensity?: number;
  /** Shade-buffer resolution as a fraction of the framebuffer. The
   *  fit/AO are low-frequency — 0.5 is 4x cheaper and visually near-
   *  identical (bilinear upsample on apply). Default 0.5. */
  readonly resolutionScale?: number;
}

export type PlaneFitOption =
  | boolean
  | aval<boolean>
  | (PlaneFitSettings & { readonly enabled?: boolean | aval<boolean> });

export interface PlaneFitPass {
  /** Call AFTER the scene rendered into `fb` (and typically before AO). */
  run(fb: IFramebuffer, pickTexture: GPUTexture, proj: Trafo3d, colorName: string): void;
  dispose(): void;
}

export function createPlaneFitPass(
  device: GPUDevice,
  colorFormat: GPUTextureFormat,
  settings: PlaneFitSettings = {},
): PlaneFitPass {
  const pipeFit = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: FIT_WGSL, label: "planeFit.fit" }), entryPoint: "main" },
  });
  const applyModule = device.createShaderModule({ code: APPLY_WGSL, label: "planeFit.apply" });
  const pipeApply = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: applyModule, entryPoint: "vs" },
    fragment: {
      module: applyModule,
      entryPoint: "fs",
      targets: [{
        format: colorFormat,
        // colour := colour × shade (same darkening contract as gtao.apply)
        blend: {
          color: { srcFactor: "zero", dstFactor: "src", operation: "add" },
          alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
        },
      }],
    },
    primitive: { topology: "triangle-list" },
    label: "planeFit.apply",
  });
  // White mode: REPLACE the colour with the shading term instead of
  // multiplying into it — a plain white surface lit by a headlight,
  // which reads geometry without the point colours competing with it.
  // Blend state is pipeline state, so this needs its own pipeline.
  const pipeApplyWhite = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: applyModule, entryPoint: "vs" },
    fragment: {
      module: applyModule,
      entryPoint: "fs",
      targets: [{ format: colorFormat }],
    },
    primitive: { topology: "triangle-list" },
    label: "planeFit.apply.white",
  });
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", label: "planeFit.samp" });
  const ubo = device.createBuffer({
    size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "planeFit.u",
  });

  let tex: GPUTexture | undefined;
  const ensureTexture = (w: number, h: number): void => {
    if (tex !== undefined && tex.width === w && tex.height === h) return;
    tex?.destroy();
    tex = device.createTexture({
      size: { width: w, height: h },
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      label: "planeFit.shade",
    });
  };

  return {
    run(fb, pick, proj, colorName) {
      const colorView = fb.colors.tryFind(colorName);
      if (colorView === undefined) return;
      const w = pick.width;
      const h = pick.height;
      const scale = Math.min(1, Math.max(0.2, settings.resolutionScale ?? 0.5));
      const sw = Math.max(1, Math.round(w * scale));
      const sh = Math.max(1, Math.round(h * scale));
      ensureTexture(sw, sh);

      const m = proj.backward;
      const buf = new ArrayBuffer(112); // 64 mat + 32 scalars + 16 sizes
      const f = new Float32Array(buf, 0, 24);
      f.set([
        m.M00, m.M10, m.M20, m.M30,
        m.M01, m.M11, m.M21, m.M31,
        m.M02, m.M12, m.M22, m.M32,
        m.M03, m.M13, m.M23, m.M33,
      ], 0);
      f[16] = settings.radius ?? 7;
      f[17] = settings.tolerance ?? 0.01;
      f[18] = settings.ambient ?? 0.35;
      f[19] = settings.strength ?? 1;
      f[20] = settings.aoRadius ?? 0.6;
      f[21] = settings.aoIntensity ?? 2;
      f[22] = proj.forward.M11;
      f[23] = 0;
      new Uint32Array(buf, 96, 4).set([w, h, sw, sh]);
      device.queue.writeBuffer(ubo, 0, buf);

      const enc = device.createCommandEncoder({ label: "planeFit.frame" });
      {
        const bg = device.createBindGroup({
          layout: pipeFit.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: pick.createView() },
            { binding: 1, resource: tex!.createView() },
            { binding: 2, resource: { buffer: ubo } },
          ],
        });
        const pass = enc.beginComputePass({ label: "planeFit.fit" });
        pass.setPipeline(pipeFit);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(sw / 8), Math.ceil(sh / 8));
        pass.end();
      }
      {
        // Both apply pipelines use layout:"auto", so each has its OWN
        // bind group layout — the group must come from the pipeline that
        // will actually be set, or it is rejected as incompatible.
        const applyPipe = settings.white === true ? pipeApplyWhite : pipeApply;
        const bg = device.createBindGroup({
          layout: applyPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: tex!.createView() },
            { binding: 1, resource: sampler },
          ],
        });
        const pass = enc.beginRenderPass({
          label: "planeFit.apply",
          colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
        });
        pass.setPipeline(applyPipe);
        pass.setBindGroup(0, bg);
        pass.draw(3);
        pass.end();
      }
      device.queue.submit([enc.finish()]);
    },
    dispose() {
      tex?.destroy();
      tex = undefined;
      ubo.destroy();
    },
  };
}

/** Normalize the `planeFitShading` prop (same shape as `gtaoConfig`). */
export function planeFitConfig(opt: PlaneFitOption): { enabled: aval<boolean>; settings: PlaneFitSettings } {
  const asAval = (v: boolean | aval<boolean>): aval<boolean> =>
    typeof v === "boolean" ? AVal.constant(v) : v;
  if (typeof opt === "boolean" || typeof (opt as aval<boolean>).getValue === "function") {
    return { enabled: asAval(opt as boolean | aval<boolean>), settings: {} };
  }
  const o = opt as PlaneFitSettings & { enabled?: boolean | aval<boolean> };
  return { enabled: asAval(o.enabled ?? true), settings: o };
}
