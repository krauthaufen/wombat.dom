// Surfel gather — reconstruction as oriented-disc coverage rather than as
// a nearest-seed fill.
//
// One pass, no ping-pong. For each pixel we gather the points nearby and
// ask a geometric question of each: does its DISC cover this pixel? The
// test is a ray/plane intersection — take the pixel's view ray, hit the
// surfel's plane, and accept if the hit lies within the disc radius. That
// single operation gives coverage AND the depth the surfel predicts here,
// and foreshortening is automatic: a grazing disc covers fewer pixels
// along the tilt direction because the ray meets its plane further out.
//
// Consequences worth stating, because they remove machinery rather than
// adding it. How far a near surface may extend past its own silhouette is
// no longer tunable — it is the disc's footprint, so the occlusion reach,
// the dilation ratio and the depth window all stop being needed. What
// remains is three parameters, two of them unitless:
//
//   radius      disc size, in pixels of screen extent
//   planeTol    agreement between predicted depths, as a fraction of R
//   enclosure   how much of the circle the covering set must span
//
// The covering surfels are clustered by predicted depth, the front-most
// cluster wins (that is the visible surface), and the pixel is filled only
// when that cluster surrounds it — which also rejects lone noise points,
// since one point can never surround anything.

import { AVal, type aval } from "@aardworx/wombat.adaptive";
import type { Trafo3d } from "@aardworx/wombat.base";
import type { IFramebuffer } from "@aardworx/wombat.rendering/core";

const WGSL = /* wgsl */ `
struct U {
  size: vec2f, radius: f32, planeTol: f32,
  enclosure: f32, reversedZ: f32, nearPlane: f32, writePick: f32,
  projInv: mat4x4<f32>,
}
@group(0) @binding(0) var pick: texture_2d<f32>;
@group(0) @binding(1) var colorCopy: texture_2d<f32>;
@group(0) @binding(2) var<uniform> u: U;
@group(0) @binding(3) var filledPick: texture_storage_2d<rgba32float, write>;

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

fn viewPos(px: vec2f, z: f32) -> vec3f {
  let uv = (px + 0.5) / u.size;
  let ndc = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
  let v = u.projInv * ndc;
  return v.xyz / v.w;
}

struct VOut { @builtin(position) pos: vec4f }

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var xy = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut; o.pos = vec4f(xy[vi], 0.0, 1.0); return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4f {
  let px = vec2i(i.pos.xy);
  let dim = vec2i(u.size);
  let own = textureLoad(pick, px, 0);
  let ownValid = own.z > 0.0 && own.z < 1.0;
  if (ownValid) {
    // the pixel has its own point; nothing to reconstruct
    if (u.writePick != 0.0) { textureStore(filledPick, px, own); }
    discard;
  }

  // this pixel's view ray (direction only; depth is solved per surfel)
  let ray = normalize(viewPos(vec2f(px), 0.5));
  let R = i32(ceil(u.radius));

  // At most ONE accepted surfel per octant. Rejected ones never need
  // counting, and keeping every coverer would bias the interpolation
  // toward whichever direction happens to hold more points — one per
  // sector gives a spread, bounded set (8 entries, all in registers).
  var sz: array<f32, 8>;
  var sw: array<f32, 8>;
  var sc: array<vec3f, 8>;
  var have: u32 = 0u;

  for (var dy = -R; dy <= R; dy = dy + 1) {
    for (var dx = -R; dx <= R; dx = dx + 1) {
      let d2 = f32(dx * dx + dy * dy);
      if (d2 > u.radius * u.radius || d2 < 0.25) { continue; }
      let q = px + vec2i(dx, dy);
      if (q.x < 0 || q.y < 0 || q.x >= dim.x || q.y >= dim.y) { continue; }
      let s = textureLoad(pick, q, 0);
      if (s.z <= 0.0 || s.z >= 1.0) { continue; }

      // does this surfel's DISC cover us? ray/plane hit, then radius test
      let P = viewPos(vec2f(q), s.z);
      let n = n24decode(s.y);
      let denom = dot(n, ray);
      if (abs(denom) < 1e-6) { continue; }
      let t = dot(n, P) / denom;
      if (t <= 0.0) { continue; }
      let H = ray * t;
      // world size of one pixel AT THIS DEPTH, measured rather than
      // approximated: unproject a neighbouring pixel at the same depth.
      // (The earlier length(P)/height guess was off by the fov factor,
      // which shrank every disc and starved the coverage test.)
      let pw = length(viewPos(vec2f(q) + vec2f(1.0, 0.0), s.z) - P);
      let rw = u.radius * pw;
      let dr = length(H - P);
      if (dr > rw) { continue; }
      let zv = -H.z;
      if (zv <= 1e-6) { continue; }
      let zndc = select(s.z, clamp(u.nearPlane / zv, 1e-7, 0.9999999), u.reversedZ != 0.0);

      var ang = atan2(f32(dy), f32(dx));
      if (ang < 0.0) { ang = ang + 6.28318531; }
      let k = u32(ang * 1.27323954) & 7u;
      let bit = 1u << k;
      // front-most wins its sector (reversed-Z: larger is nearer)
      if ((have & bit) == 0u || zndc > sz[k]) {
        sz[k] = zndc;
        sw[k] = 1.0 - dr / max(rw, 1e-6);
        sc[k] = textureLoad(colorCopy, q, 0).rgb;
        have = have | bit;
      }
    }
  }

  if (have == 0u) {
    if (u.writePick != 0.0) { textureStore(filledPick, px, own); }
    discard;
  }

  // front-most sector entry defines the visible surface
  var bestZ = 0.0;
  for (var k = 0u; k < 8u; k = k + 1u) {
    if ((have & (1u << k)) != 0u && sz[k] > bestZ) { bestZ = sz[k]; }
  }

  // keep only the sectors whose plane agrees with it — those are "the
  // planes taken", and the enclosure test runs over exactly these
  var kept: u32 = 0u;
  var wsum = 0.0;
  var csum = vec3f(0.0);
  var zsum = 0.0;
  for (var k = 0u; k < 8u; k = k + 1u) {
    if ((have & (1u << k)) == 0u) { continue; }
    if (abs(sz[k] - bestZ) > bestZ * u.planeTol) { continue; }
    kept = kept | (1u << k);
    let w = max(sw[k], 1e-3);
    csum = csum + sc[k] * w;
    zsum = zsum + sz[k] * w;
    wsum = wsum + w;
  }

  var span = 0u;
  for (var k = 0u; k < 8u; k = k + 1u) { span = span + ((kept >> k) & 1u); }
  var enclosed = f32(span) >= u.enclosure;
  if (u.enclosure > 0.0) {
    // and they must not all sit inside one half-plane
    for (var r = 0u; r < 8u; r = r + 1u) {
      let hw = ((0xFu << r) | (0xFu >> (8u - r))) & 0xFFu;
      if ((kept & ~hw & 0xFFu) == 0u) { enclosed = false; }
    }
  }
  if (wsum <= 0.0 || !enclosed) {
    if (u.writePick != 0.0) { textureStore(filledPick, px, own); }
    discard;
  }

  let outZ = zsum / wsum;
  if (u.writePick != 0.0) {
    textureStore(filledPick, px, vec4f(own.x, own.y, outZ, own.w));
  }
  return vec4f(csum / wsum, 1.0);
}
`;

export interface SurfelFillSettings {
  /** Disc radius in screen pixels. Default 6. */
  readonly radius?: number;
  /** Depth agreement between covering surfels, as a fraction. Default 0.02. */
  readonly planeTolerance?: number;
  /** Minimum number of the 8 angular sectors that must contain a
   *  contributing surfel; the >180deg half-plane test applies on top.
   *  0 disables the enclosure requirement. Default 3. */
  readonly enclosure?: number;
  readonly reversedZ?: boolean;
  readonly nearPlane?: number;
}

export type SurfelFillOption =
  | boolean
  | aval<boolean>
  | (SurfelFillSettings & { readonly enabled?: boolean | aval<boolean> });

export interface SurfelFillPass {
  run(fb: IFramebuffer, pickTexture: GPUTexture, colorName: string, needPick?: boolean, proj?: Trafo3d): GPUTexture | undefined;
  dispose(): void;
}

export function createSurfelFillPass(
  device: GPUDevice,
  colorFormat: GPUTextureFormat,
  settings: SurfelFillSettings = {},
): SurfelFillPass {
  const mod = device.createShaderModule({ code: WGSL, label: "surfel.fill" });
  const pipe = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: mod, entryPoint: "vs" },
    fragment: { module: mod, entryPoint: "fs", targets: [{ format: colorFormat }] },
    primitive: { topology: "triangle-list" },
    label: "surfel.fill",
  });
  const ubo = device.createBuffer({
    size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "surfel.u",
  });
  let colorCopy: GPUTexture | undefined;
  let filledPick: GPUTexture | undefined;
  let dims = { w: 0, h: 0 };

  const ensure = (w: number, h: number): void => {
    if (dims.w === w && dims.h === h) return;
    colorCopy?.destroy(); filledPick?.destroy();
    colorCopy = device.createTexture({
      size: { width: w, height: h }, format: colorFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST, label: "surfel.color",
    });
    filledPick = device.createTexture({
      size: { width: w, height: h }, format: "rgba32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING, label: "surfel.pick",
    });
    dims = { w, h };
  };

  return {
    run(fb, pick, colorName, needPick, proj) {
      const colorView = fb.colors.tryFind(colorName);
      const colorTex = fb.colorTextures?.tryFind?.(colorName);
      if (colorView === undefined || colorTex === undefined || proj === undefined) return undefined;
      const w = pick.width, h = pick.height;
      ensure(w, h);
      const wantPick = needPick !== false;

      const f = new Float32Array(28);
      f.set([
        w, h,
        Math.max(1, Math.min(24, settings.radius ?? 6)),
        settings.planeTolerance ?? 0.02,
        settings.enclosure ?? 3,
        (settings.reversedZ ?? true) ? 1 : 0,
        settings.nearPlane ?? 0.05,
        wantPick ? 1 : 0,
      ], 0);
      const m = proj.backward;
      // mat4x4 aligns to 16 bytes: the 8 scalars above occupy 0..31, so the
      // matrix starts at byte 32 = float index 8 (NOT 12 — writing it at 12
      // fed viewPos a garbage matrix and nothing ever covered anything).
      f.set([
        m.M00, m.M10, m.M20, m.M30,
        m.M01, m.M11, m.M21, m.M31,
        m.M02, m.M12, m.M22, m.M32,
        m.M03, m.M13, m.M23, m.M33,
      ], 8);
      device.queue.writeBuffer(ubo, 0, f);

      const enc = device.createCommandEncoder({ label: "surfel.frame" });
      enc.copyTextureToTexture({ texture: colorTex }, { texture: colorCopy! }, { width: w, height: h });
      const bg = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: pick.createView() },
          { binding: 1, resource: colorCopy!.createView() },
          { binding: 2, resource: { buffer: ubo } },
          { binding: 3, resource: filledPick!.createView() },
        ],
      });
      const p = enc.beginRenderPass({
        label: "surfel.fill",
        colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
      });
      p.setPipeline(pipe); p.setBindGroup(0, bg); p.draw(3); p.end();
      device.queue.submit([enc.finish()]);
      return wantPick ? filledPick : undefined;
    },
    dispose() {
      colorCopy?.destroy(); filledPick?.destroy(); ubo.destroy();
      colorCopy = filledPick = undefined;
    },
  };
}

export function surfelFillConfig(opt: SurfelFillOption): { enabled: aval<boolean>; settings: SurfelFillSettings } {
  const asAval = (v: boolean | aval<boolean>): aval<boolean> =>
    typeof v === "boolean" ? AVal.constant(v) : v;
  if (typeof opt === "boolean" || typeof (opt as aval<boolean>).getValue === "function") {
    return { enabled: asAval(opt as boolean | aval<boolean>), settings: {} };
  }
  const o = opt as SurfelFillSettings & { enabled?: boolean | aval<boolean> };
  return { enabled: asAval(o.enabled ?? true), settings: o };
}
