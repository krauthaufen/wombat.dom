// Screen-space ambient occlusion (GTAO-style horizon search) driven by the
// pick attachment — the G-buffer we already pay for.
//
// The RenderControl's pick producer renders canvas colour, depth AND a
// rgba32float `pickId` attachment in ONE pass, where slot .y carries the
// oct24-packed view-space normal and slot .z the NDC depth. So AO needs no
// extra geometry pass at all: two compute dispatches (horizon search + a 4×4
// box denoise) over that texture, then one fullscreen quad multiplied into the
// canvas colour.
//
// Ported from `examples/cityview/src/gtao.ts`, which composited through an
// offscreen colour texture. Here the scene has already been drawn into the
// canvas, so instead of re-compositing we darken in place with a multiply
// blend (`src·0 + dst·src` = colour × AO). That keeps the pass a pure
// post-effect: nothing about the scene, OIT or picking changes.
//
// Reversed-Z safe: background pixels carry the depth CLEAR value (0 under
// reversed-Z, 1 otherwise) and both are rejected by the `z <= 0 || z >= 1`
// guard, so the sky is never occluded. The view-space reconstruction goes
// through the projection's inverse, which encodes the depth convention.

import { AVal, type aval } from "@aardworx/wombat.adaptive";
import type { Trafo3d } from "@aardworx/wombat.base";
import type { IFramebuffer } from "@aardworx/wombat.rendering/core";

const AO_WGSL = /* wgsl */ `
struct U {
  projInv: mat4x4f,
  radiusWorld: f32,
  intensity: f32,
  projM11: f32,
  _pad: f32,
  size: vec2u,
}
@group(0) @binding(0) var pick: texture_2d<f32>;
@group(0) @binding(1) var outAo: texture_storage_2d<rgba8unorm, write>;
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

fn n24decode(e: f32) -> vec3f {
  let i = i32(e);
  let xi = (i >> 12) & 4095;
  let yi = i & 4095;
  var x = f32(xi) / 4095.0 * 2.0 - 1.0;
  var y = f32(yi) / 4095.0 * 2.0 - 1.0;
  let z0 = 1.0 - abs(x) - abs(y);
  if (z0 < 0.0) {
    let ox = x;
    x = (1.0 - abs(y)) * select(-1.0, 1.0, ox >= 0.0);
    y = (1.0 - abs(ox)) * select(-1.0, 1.0, y >= 0.0);
  }
  return normalize(vec3f(x, y, z0));
}

const DIRS = 8u;
const STEPS = 4u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u.size.x || gid.y >= u.size.y) { return; }
  let px = vec2i(gid.xy);
  let p = viewPosAt(px);
  if (p.w == 0.0) { textureStore(outAo, px, vec4f(1.0)); return; }
  let s = textureLoad(pick, px, 0);
  let n = n24decode(s.y);

  // Screen-space step length for the world radius at this depth.
  let pixR = u.radiusWorld * u.projM11 * f32(u.size.y) / (2.0 * max(0.1, abs(p.z)));
  let stepPx = max(1.0, pixR / f32(STEPS));

  var occl = 0.0;
  // Per-pixel rotation to decorrelate direction sets (cheap hash).
  let rot = fract(sin(f32(gid.x) * 12.9898 + f32(gid.y) * 78.233) * 43758.5453) * 6.2831853;
  for (var d = 0u; d < DIRS; d++) {
    let a = rot + f32(d) * (6.2831853 / f32(DIRS));
    let dir = vec2f(cos(a), sin(a));
    var best = 0.0;
    for (var st = 1u; st <= STEPS; st++) {
      let q = vec2i(vec2f(px) + dir * stepPx * f32(st));
      if (q.x < 0 || q.y < 0 || q.x >= i32(u.size.x) || q.y >= i32(u.size.y)) { break; }
      let vq = viewPosAt(q);
      if (vq.w == 0.0) { continue; }
      let dl = vq.xyz - p.xyz;
      let dist = length(dl);
      if (dist < 1e-4 || dist > u.radiusWorld) { continue; }
      let h = dot(n, dl / dist) - 0.08;           // bias against self-shadowing
      if (h > best) { best = h; }
    }
    occl += max(0.0, best);
  }
  let ao = clamp(1.0 - occl / f32(DIRS) * u.intensity, 0.0, 1.0);
  textureStore(outAo, px, vec4f(ao, ao, ao, 1.0));
}
`;

const BLUR_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dim = textureDimensions(src);
  if (gid.x >= dim.x || gid.y >= dim.y) { return; }
  var sum = 0.0;
  var cnt = 0.0;
  for (var dy = -2; dy <= 1; dy++) {
    for (var dx = -2; dx <= 1; dx++) {
      let q = vec2i(gid.xy) + vec2i(dx, dy);
      if (q.x < 0 || q.y < 0 || q.x >= i32(dim.x) || q.y >= i32(dim.y)) { continue; }
      sum += textureLoad(src, q, 0).x;
      cnt += 1.0;
    }
  }
  let v = sum / max(1.0, cnt);
  textureStore(dst, vec2i(gid.xy), vec4f(v, v, v, 1.0));
}
`;

// Fullscreen triangle, multiplied into the canvas colour.
const APPLY_WGSL = /* wgsl */ `
@group(0) @binding(0) var ao: texture_2d<f32>;
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
  let a = textureSample(ao, samp, i.uv).x;
  return vec4f(a, a, a, 1.0);
}
`;

export interface GtaoSettings {
  /** World-space AO radius, in scene units. Default 4. */
  readonly radius?: number;
  /** Occlusion strength. Default 1.4. */
  readonly intensity?: number;
}

/** `ambientOcclusion` prop of `RenderControl`: on/off (adaptive so it can be
 *  toggled from a checkbox) plus the two tunables. */
export type GtaoOption =
  | boolean
  | aval<boolean>
  | (GtaoSettings & { readonly enabled?: boolean | aval<boolean> });

export interface GtaoPass {
  /**
   * Run AO for this frame: two compute dispatches over `pickTexture`, then a
   * multiply-blended fullscreen quad into the framebuffer's colour attachment.
   * Call AFTER the scene has been rendered into `fb` — the encoder is
   * submitted separately, so queue order does the sequencing.
   */
  run(fb: IFramebuffer, pickTexture: GPUTexture, proj: Trafo3d, colorName: string): void;
  dispose(): void;
}

export function createGtaoPass(
  device: GPUDevice,
  colorFormat: GPUTextureFormat,
  settings: GtaoSettings = {},
): GtaoPass {
  const pipeAo = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: AO_WGSL, label: "gtao.ao" }), entryPoint: "main" },
  });
  const pipeBlur = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: BLUR_WGSL, label: "gtao.blur" }), entryPoint: "main" },
  });
  const applyModule = device.createShaderModule({ code: APPLY_WGSL, label: "gtao.apply" });
  const pipeApply = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: applyModule, entryPoint: "vs" },
    fragment: {
      module: applyModule,
      entryPoint: "fs",
      targets: [{
        format: colorFormat,
        // colour := src·0 + dst·src  =  colour × AO. A pure darkening pass;
        // nothing else in the framebuffer is touched (alpha kept as-is).
        blend: {
          color: { srcFactor: "zero", dstFactor: "src", operation: "add" },
          alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
        },
      }],
    },
    primitive: { topology: "triangle-list" },
    label: "gtao.apply",
  });
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", label: "gtao.samp" });
  const ubo = device.createBuffer({
    size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "gtao.u",
  });

  let texA: GPUTexture | undefined;
  let texB: GPUTexture | undefined;

  const ensureTextures = (w: number, h: number): void => {
    if (texA !== undefined && texA.width === w && texA.height === h) return;
    texA?.destroy();
    texB?.destroy();
    const mk = (label: string): GPUTexture => device.createTexture({
      size: { width: w, height: h },
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      label,
    });
    texA = mk("gtao.raw");
    texB = mk("gtao.blurred");
  };

  return {
    run(fb, pick, proj, colorName) {
      const colorView = fb.colors.tryFind(colorName);
      if (colorView === undefined) return;
      const w = pick.width;
      const h = pick.height;
      ensureTextures(w, h);

      // projInv (column-major — wombat's M44d is row-major), radius, intensity,
      // projM11 (vertical focal scale: turns the world radius into pixels), size.
      const m = proj.backward;
      const buf = new ArrayBuffer(96);
      const f = new Float32Array(buf, 0, 20);
      f.set([
        m.M00, m.M10, m.M20, m.M30,
        m.M01, m.M11, m.M21, m.M31,
        m.M02, m.M12, m.M22, m.M32,
        m.M03, m.M13, m.M23, m.M33,
      ], 0);
      f[16] = settings.radius ?? 4;
      f[17] = settings.intensity ?? 1.4;
      f[18] = proj.forward.M11;
      f[19] = 0;
      new Uint32Array(buf, 80, 2).set([w, h]);
      device.queue.writeBuffer(ubo, 0, buf);

      const enc = device.createCommandEncoder({ label: "gtao.frame" });
      {
        const bg = device.createBindGroup({
          layout: pipeAo.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: pick.createView() },
            { binding: 1, resource: texA!.createView() },
            { binding: 2, resource: { buffer: ubo } },
          ],
        });
        const pass = enc.beginComputePass({ label: "gtao.ao" });
        pass.setPipeline(pipeAo);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
        pass.end();
      }
      {
        const bg = device.createBindGroup({
          layout: pipeBlur.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: texA!.createView() },
            { binding: 1, resource: texB!.createView() },
          ],
        });
        const pass = enc.beginComputePass({ label: "gtao.blur" });
        pass.setPipeline(pipeBlur);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
        pass.end();
      }
      {
        const bg = device.createBindGroup({
          layout: pipeApply.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: texB!.createView() },
            { binding: 1, resource: sampler },
          ],
        });
        const pass = enc.beginRenderPass({
          label: "gtao.apply",
          colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
        });
        pass.setPipeline(pipeApply);
        pass.setBindGroup(0, bg);
        pass.draw(3);
        pass.end();
      }
      device.queue.submit([enc.finish()]);
    },
    dispose() {
      texA?.destroy();
      texB?.destroy();
      texA = undefined;
      texB = undefined;
      ubo.destroy();
    },
  };
}

/** Normalize the `ambientOcclusion` prop into an enabled-aval + settings. */
export function gtaoConfig(opt: GtaoOption): { enabled: aval<boolean>; settings: GtaoSettings } {
  const asAval = (v: boolean | aval<boolean>): aval<boolean> =>
    typeof v === "boolean" ? AVal.constant(v) : v;
  if (typeof opt === "boolean" || typeof (opt as aval<boolean>).getValue === "function") {
    return { enabled: asAval(opt as boolean | aval<boolean>), settings: {} };
  }
  const o = opt as GtaoSettings & { enabled?: boolean | aval<boolean> };
  return { enabled: asAval(o.enabled ?? true), settings: o };
}
