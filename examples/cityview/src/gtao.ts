// GTAO-style screen-space AO from the pick attachment (the free
// G-buffer: slot 1 = oct24 view-space normal, slot 2 = NDC depth).
// Two raw-WGSL compute passes: horizon sampling + a 4×4 box denoise.
//
// Exposed as an `AdaptiveResource<ITexture>`: compute() first pulls the
// inner render's framebuffer (running this frame's city pass), then
// dispatches AO + blur on its own encoder. The composite quad samples
// the result — pulling it during the outer frame keeps everything
// queue-ordered.

import { AVal, type AdaptiveToken, type aval } from "@aardworx/wombat.adaptive";
import type { Trafo3d } from "@aardworx/wombat.base";
import { AdaptiveResource, ITexture, type IFramebuffer } from "@aardworx/wombat.rendering/core";

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

export interface GtaoOptions {
  /** World-space AO radius (meters for the city). Default 4. */
  readonly radius?: number;
  /** Occlusion strength. Default 1.4. */
  readonly intensity?: number;
}

/**
 * AO texture resource. Pulling it (during the outer frame) runs the
 * inner city render (via `innerFramebuffer`), then the AO + denoise
 * kernels over `pickTexture`. Returns the blurred AO as an ITexture.
 */
export class GtaoResource extends AdaptiveResource<ITexture> {
  private pipeAo: GPUComputePipeline | undefined;
  private pipeBlur: GPUComputePipeline | undefined;
  private ubo: GPUBuffer | undefined;
  private texA: GPUTexture | undefined;
  private texB: GPUTexture | undefined;

  constructor(
    private readonly device: GPUDevice,
    private readonly innerFramebuffer: AdaptiveResource<IFramebuffer>,
    private readonly pickTexture: aval<GPUTexture>,
    private readonly proj: aval<Trafo3d>,
    private readonly opts: GtaoOptions = {},
  ) { super(); }

  protected override create(): void {
    this.innerFramebuffer.acquire();
  }

  /** Pipelines/ubo created lazily — `compute` can be called for
   *  classification before the first acquire. */
  private ensurePipelines(): void {
    if (this.pipeAo !== undefined) return;
    const d = this.device;
    this.pipeAo = d.createComputePipeline({
      layout: "auto",
      compute: { module: d.createShaderModule({ code: AO_WGSL, label: "gtao.ao" }), entryPoint: "main" },
    });
    this.pipeBlur = d.createComputePipeline({
      layout: "auto",
      compute: { module: d.createShaderModule({ code: BLUR_WGSL, label: "gtao.blur" }), entryPoint: "main" },
    });
    this.ubo = d.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "gtao.u" });
  }

  protected override destroy(): void {
    this.innerFramebuffer.release();
    this.texA?.destroy(); this.texB?.destroy(); this.ubo?.destroy();
    this.texA = undefined; this.texB = undefined;
    this.ubo = undefined; this.pipeAo = undefined; this.pipeBlur = undefined;
  }

  override compute(token: AdaptiveToken): ITexture {
    // Run this frame's inner render first (ordering!).
    this.ensurePipelines();
    this.innerFramebuffer.getValue(token);
    const pick = this.pickTexture.getValue(token);
    const d = this.device;
    const w = pick.width, h = pick.height;
    if (this.texA === undefined || this.texA.width !== w || this.texA.height !== h) {
      this.texA?.destroy(); this.texB?.destroy();
      const mk = (label: string): GPUTexture => d.createTexture({
        size: { width: w, height: h },
        format: "rgba8unorm",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        label,
      });
      this.texA = mk("gtao.raw");
      this.texB = mk("gtao.blurred");
    }

    // Uniforms: projInv (column-major f32×16), radius, intensity, projM11, size.
    // AVal.force OK: per-frame snapshot inside compute.
    const projT = AVal.force(this.proj);
    const inv = projT.backward;
    const u = new ArrayBuffer(96);
    const f = new Float32Array(u, 0, 20);
    // wombat M44d is row-major (M00..M03 = first row); WGSL mat4x4f is
    // column-major — write columns.
    const m = inv;
    f.set([
      m.M00, m.M10, m.M20, m.M30,
      m.M01, m.M11, m.M21, m.M31,
      m.M02, m.M12, m.M22, m.M32,
      m.M03, m.M13, m.M23, m.M33,
    ], 0);
    f[16] = this.opts.radius ?? 4;
    f[17] = this.opts.intensity ?? 1.4;
    f[18] = projT.forward.M11; // vertical focal scale
    f[19] = 0;
    new Uint32Array(u, 80, 2).set([w, h]);
    d.queue.writeBuffer(this.ubo!, 0, u);

    const enc = d.createCommandEncoder({ label: "gtao.frame" });
    {
      const bg = d.createBindGroup({
        layout: this.pipeAo!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: pick.createView() },
          { binding: 1, resource: this.texA!.createView() },
          { binding: 2, resource: { buffer: this.ubo! } },
        ],
      });
      const pass = enc.beginComputePass({ label: "gtao.ao" });
      pass.setPipeline(this.pipeAo!);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
      pass.end();
    }
    {
      const bg = d.createBindGroup({
        layout: this.pipeBlur!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.texA!.createView() },
          { binding: 1, resource: this.texB!.createView() },
        ],
      });
      const pass = enc.beginComputePass({ label: "gtao.blur" });
      pass.setPipeline(this.pipeBlur!);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
      pass.end();
    }
    d.queue.submit([enc.finish()]);
    return ITexture.fromGPU(this.texB!);
  }
}
