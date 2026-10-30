// Eye-Dome Lighting — screen-space shape cues from depth alone.
//
// For every pixel, compare its eye distance against a ring of
// neighbours and darken it by how much FARTHER it is than they are.
// Creases, the far side of a step, and the gap between two surfaces all
// go dark; a flat surface, where every neighbour sits at the same
// distance, is untouched. The result reads as shape without a single
// normal being computed.
//
// WHY THIS AND NOT THE PLANE FIT. They answer different questions and
// are worth having side by side:
//
//   planeFit  estimates a SURFACE (least-squares normal + AO) and
//             shades it. Physical, needs a neighbourhood that really is
//             one surface, and goes uncertain at silhouettes — which is
//             what its confidence term exists to hide.
//   EDL       estimates nothing. It is a depth-contrast operator, so it
//             cannot be wrong about a surface it never modelled, and it
//             is at its strongest exactly where the fit is weakest: at
//             depth discontinuities.
//
// The comparison is on the LOG of eye distance, which is what makes one
// strength work across a scene spanning metres to kilometres: a 1 cm
// step at 10 m and a 1 m step at 1 km produce the same log ratio, so
// near and far geometry get the same amount of relief. On raw depth the
// far half of the scene would be flat.
//
// Multiply-blended, like the AO pass: a pure darkening of what is
// already in the colour attachment, with alpha untouched.

import { Trafo3d } from "@aardworx/wombat.base";
import type { IFramebuffer } from "@aardworx/wombat.rendering/core";

const EDL_WGSL = /* wgsl */ `
struct U {
  projInv: mat4x4f,
  radiusPx: f32,
  strength: f32,
  pickGroup: f32,   // >= 0 = shade only this group (floor of pick .w)
  _pad: f32,
  size: vec2u,
}
@group(0) @binding(0) var pick: texture_2d<f32>;
@group(0) @binding(1) var<uniform> u: U;

// Eye distance at a pixel, or 0 where this pass has no business.
fn eyeDist(pxIn: vec2i) -> f32 {
  // CLAMP to the edge rather than reporting "no geometry" outside the
  // frame, as the reference's own sampler does. Reporting empty would
  // make the one-pixel border a false discontinuity and ring the whole
  // frame edge dark.
  let px = clamp(pxIn, vec2i(0, 0), vec2i(i32(u.size.x) - 1, i32(u.size.y) - 1));
  let s = textureLoad(pick, px, 0);
  let z = s.z;
  if (z <= 0.0 || z >= 1.0) { return 0.0; }
  // Group filter, for the same reason every other screen-space pass has
  // one: the pass belongs to a DATASET, not to the frame. Unfiltered, a
  // point cloud would be darkened against the terrain behind it and
  // every silhouette would gain a black fringe that belongs to neither.
  if (u.pickGroup >= 0.0 && floor(s.w) != u.pickGroup) { return 0.0; }
  let uv = (vec2f(px) + 0.5) / vec2f(u.size);
  let ndc = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
  let v = u.projInv * ndc;
  let vp = v.xyz / v.w;
  return max(length(vp), 1e-6);
}

struct VOut { @builtin(position) pos: vec4f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(-1.0, 1.0), vec2f(3.0, 1.0));
  var o: VOut; o.pos = vec4f(p[i], 0.0, 1.0); return o;
}

@fragment fn fs(v: VOut) -> @location(0) vec4f {
  let px = vec2i(v.pos.xy);
  let d0 = eyeDist(px);
  // White = multiply by one. A pixel this pass does not own has to come
  // through the blend unchanged, so there is no discard here: a
  // discarded fragment would leave the colour alone too, but writing
  // the identity keeps the pass uniform and the blend state honest.
  if (d0 <= 0.0) { return vec4f(1.0, 1.0, 1.0, 1.0); }

  let l0 = log(d0);
  // Eight neighbours on a SQUARE: the four axes plus the four diagonals,
  // at integer offsets. Eight, not four, because a four-tap cross misses
  // every diagonal crease.
  let o = max(1, i32(floor(u.radiusPx + 0.5)));
  var sum = 0.0;
  // A neighbour with no geometry contributes NOTHING, but it is still
  // counted in the divisor -- the denominator is a fixed 8. Dividing by
  // the number of valid neighbours instead inflates the response exactly
  // where fewest of them exist, which is at silhouettes, and paints a
  // dark fringe around every object. This is the reference's convention
  // and it is the right one.
  for (var k = 0; k < 8; k++) {
    var q = vec2i(o, 0);
    if (k == 1) { q = vec2i(-o, 0); }
    else if (k == 2) { q = vec2i(0, o); }
    else if (k == 3) { q = vec2i(0, -o); }
    else if (k == 4) { q = vec2i(o, o); }
    else if (k == 5) { q = vec2i(-o, o); }
    else if (k == 6) { q = vec2i(o, -o); }
    else if (k == 7) { q = vec2i(-o, -o); }
    let d = eyeDist(px + q);
    if (d > 0.0) { sum = sum + max(0.0, l0 - log(d)); }
  }

  // The 32 is the reference's: it puts a useful amount of relief at
  // strength 1, so the knob reads the same here as it does there.
  let shade = exp(-32.0 * u.strength * (sum / 8.0));
  return vec4f(shade, shade, shade, 1.0);
}
`;

export interface EdlSettings {
  /** Neighbour ring radius in device pixels. Default 2. */
  readonly radius?: number;
  /** How hard to darken. Multiplies the MEAN LOG-DISTANCE RATIO, so it
   *  is scale-free — the same value works near and far.
   *
   *  Default 1. The shader folds in a factor of 32, matching the
   *  reference pipeline this mirrors, so the useful range is roughly
   *  0..2 and 1 is a sensible picture rather than a number to discover.
   *  Measured on an indoor scan at 1: 14% of the frame darkens, every
   *  changed pixel darkens (never brightens), the median drop is ~2/255
   *  on flat surfaces and the 95th percentile ~92/255 at depth
   *  discontinuities — subtle where there is no shape to show and
   *  strong where there is. */
  readonly strength?: number;
  /** Shade only the fragments tagged with this group number — the
   *  integer part of the pick G-buffer's `.w`. Omit (or -1) to shade
   *  everything, which is right only for a single-dataset frame. */
  readonly pickGroup?: number;
}

export interface EdlPass {
  /** Call AFTER the scene (and any fill) has written the colour. */
  run(fb: IFramebuffer, pickTexture: GPUTexture, proj: Trafo3d, colorName: string): void;
  dispose(): void;
}

export function createEdlPass(
  device: GPUDevice,
  colorFormat: GPUTextureFormat,
  settings: EdlSettings = {},
): EdlPass {
  const mod = device.createShaderModule({ code: EDL_WGSL, label: "edl" });
  const pipe = device.createRenderPipeline({
    label: "edl",
    layout: "auto",
    vertex: { module: mod, entryPoint: "vs" },
    fragment: {
      module: mod,
      entryPoint: "fs",
      targets: [{
        format: colorFormat,
        // colour := src·0 + dst·src = colour × shade. Pure darkening,
        // alpha untouched — the same contract as the AO pass.
        blend: {
          color: { srcFactor: "zero", dstFactor: "src", operation: "add" },
          alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
        },
      }],
    },
    primitive: { topology: "triangle-list" },
  });
  const ubo = device.createBuffer({
    size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "edl.u",
  });

  return {
    run(fb, pick, proj, colorName) {
      const colorView = fb.colors.tryFind(colorName);
      if (colorView === undefined) return;
      const w = pick.width, h = pick.height;

      const m = proj.backward;
      const buf = new ArrayBuffer(96);
      const f = new Float32Array(buf);
      // projInv, COLUMN-MAJOR: wombat's M44d is row-major and exposes
      // named fields, so the transpose is written out rather than
      // indexed — exactly as gtao.ts does it. Getting this wrong yields
      // NaN distances, and every comparison against a NaN is false, so
      // the pass would silently do nothing at all.
      f.set([m.M00, m.M10, m.M20, m.M30,
             m.M01, m.M11, m.M21, m.M31,
             m.M02, m.M12, m.M22, m.M32,
             m.M03, m.M13, m.M23, m.M33], 0);
      f[16] = Math.max(0.5, settings.radius ?? 2);
      f[17] = Math.max(0, settings.strength ?? 1);
      f[18] = settings.pickGroup ?? -1;
      f[19] = 0;
      new Uint32Array(buf, 80, 2).set([w, h]);
      device.queue.writeBuffer(ubo, 0, buf);

      const bg = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: pick.createView() },
          { binding: 1, resource: { buffer: ubo } },
        ],
      });
      const enc = device.createCommandEncoder({ label: "edl.frame" });
      const pass = enc.beginRenderPass({
        label: "edl",
        colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
      });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
      device.queue.submit([enc.finish()]);
    },
    dispose() {
      ubo.destroy();
    },
  };
}
