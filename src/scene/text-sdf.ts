// Per-pixel SDF text rendering for `aa = "alpha-blending"`.
//
// Geometry comes from `GlyphCache` and consists of two layers:
//
//   1. BODY (kind = 0 flat / kind = 1,2 Loop-Blinn curves) — fills
//      the glyph interior. Body triangles draw with α = 1 (kind = 0)
//      or with the standard k²-l / k²+l²-1 implicit-curve discard
//      (kinds 1, 2). No SDF work; no overdraw beyond the contour.
//
//   2. BAND (kind = 4) — a triangulated annulus around every contour
//      at ±halo_em, built CPU-side via Clipper2 polygon offsetting
//      (see `band-builder.ts`). Band fragments compute pixel-space
//      distance to the nearest curve in the glyph's full SSBO range,
//      ramp α from 1 down to 0 across `AaWidthPx` pixels of distance,
//      and discard when α reaches 0.
//
// Distance computation per band fragment:
//   - Iterate the glyph's SSBO range [triFirst, triFirst+triCount)
//     once. Each entry stores three control points (a, b, c) plus
//     a kind flag in the 4th vec4's .w slot:
//       * kind = 5 (line sentinel: p1 == p0): closed-form
//         point-segment distance from the projected line p0..p2.
//       * kind = 1, 2 (real bezier2 / arc): rational Newton on the
//         pixel-space projected quadratic — 5 seeds × 8 iterations
//         + endpoint check.
//       * kind = 0 (body flat tri): iteration is currently NOT
//         skipped, but the FS guards against NaNs from these
//         degenerate-quadratic Newton paths via clamps.
//   - The min over all SSBO entries is the fragment-to-contour
//     distance. Cost per band fragment is O(triCount × 5 seeds × 8
//     iters) for curves; lines are constant time. Mobile-GPU heavy
//     on glyphs with many curves; per-tri SSBO range clipping is on
//     the followup list (see `~/claude/wombat-todo.md`).
//
// Per-glyph indexed instanced draw: the SDF index buffer slice
// covers band + body tris; instance attribute supplies (cx, by) per
// glyph occurrence in the laid-out string.
//
// Fragment dispatch by interpolated kind:
//   kind = 0     → α = 1, body interior
//   kind = 1, 2  → Loop-Blinn klm discard, then α = 1
//   kind = 3     → ribbon (NOT included in the SDF index buffer)
//   kind = 4     → band: SDF distance + AA ramp
//   kind = 5     → not rendered as geometry (line-sentinel SSBO
//                  entry only; vertex tri is degenerate / zero area)

import { AVal, HashMap, type aval } from "@aardworx/wombat.adaptive";
import { V2f, V3d, V3f, V4f, Trafo3d } from "@aardworx/wombat.base";
import {
  type Font, GlyphCache, layoutText, GLYPH_FLOATS_PER_VERTEX,
} from "@aardworx/wombat.base/font";
import {
  IBuffer, type BufferView, type DrawCall, type BlendState,
  ElementType,
} from "@aardworx/wombat.rendering/core";
import type { Effect } from "@aardworx/wombat.shader";
import { effect, vertex, fragment } from "@aardworx/wombat.shader";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { clamp, discard, fract, max, min, sqrt } from "@aardworx/wombat.shader/types";
import type { f32, u32, Storage } from "@aardworx/wombat.shader/types";

import type { VNode } from "../vnode.js";
import { Sg } from "./constructors.js";
import type { SgNode } from "./sg.js";
import type { SgScopeProps } from "./constructors.js";
import { viewport as ambViewport } from "./ambient.js";

// `DebugMode` lives only in this file; PathColor/Viewport/AaWidthPx
// are augmented in text.ts and shared.
declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope {
    readonly DebugMode: number;
  }
}

// Storage buffer of (a.xy, klm0.xy), (b.xy, klm1.xy)…—one quad-of-V4f
// per candidate curve, indexed by the candidate's tri-id from v_cands.
declare const tris: Storage<V4f[], "read">;

// Per-band-triangle candidate CSR. `bandCandOffsets[triId]` ..
// `bandCandOffsets[triId+1]` is the slice of `bandCandIndices` holding
// this band triangle's candidate curve-triangle SSBO indices. Replaces
// the old per-vertex cand slots; the band FS loops the actual range.
declare const bandCandOffsets: Storage<u32[], "read">;
declare const bandCandIndices: Storage<u32[], "read">;

// SSBO entry layout per candidate:
//   tris[base + 0] = (a.xy, klm0.xy)
//   tris[base + 1] = (b.xy, klm1.xy)   ← b == a marks line sentinel
//   tris[base + 2] = (c.xy, klm2.xy)
//   tris[base + 3] = (klm0.z, klm1.z, klm2.z, kind)
//
// Returns the squared pixel distance from `(fragPxX, fragPxY)` to
// the closest point on this candidate's curve (or 1e20 if `candF`
// is sentinel `-1`). Linear segments are handled by point-on-segment
// projection; quadratic Béziers by 5-seed Newton iteration on the
// projected curve.
function processCand(
  candF: f32,
  sx: f32, fragPxX: f32, fragPxY: f32,
  vInstX: f32, vInstY: f32,
): f32 {
  if (candF < 0.0) return 1.0e20;
  const idx = (candF + 0.5) as u32;
  const ZERO_U: u32 = 0 as u32;
  const ONE_U:  u32 = 1 as u32;
  const TWO_U:  u32 = 2 as u32;
  const FOUR_U: u32 = 4 as u32;
  const SEEDS_U: u32 = 5 as u32;
  const ITER_U:  u32 = 8 as u32;
  const base = idx * FOUR_U;
  const a: V4f = tris[base] as V4f;
  const b: V4f = tris[base + ONE_U] as V4f;
  const c: V4f = tris[base + TWO_U] as V4f;
  const cT0 = new V4f((vInstX + a.x) * sx, vInstY + a.y, 0.0, 1.0);
  const cT2 = new V4f((vInstX + c.x) * sx, vInstY + c.y, 0.0, 1.0);
  const cC0 = uniform.ProjTrafo.mul(uniform.ViewTrafo.mul(uniform.ModelTrafo.mul(cT0)));
  const cC2 = uniform.ProjTrafo.mul(uniform.ViewTrafo.mul(uniform.ModelTrafo.mul(cT2)));
  const e0px = (cC0.x / cC0.w + 1.0) * 0.5 * uniform.Viewport.x;
  const e0py = (cC0.y / cC0.w + 1.0) * 0.5 * uniform.Viewport.y;
  const e2px = (cC2.x / cC2.w + 1.0) * 0.5 * uniform.Viewport.x;
  const e2py = (cC2.y / cC2.w + 1.0) * 0.5 * uniform.Viewport.y;
  let bestPx2: f32 = 1.0e20;
  const isLine = (a.x == b.x) && (a.y == b.y);
  if (isLine) {
    const sxL = e2px - e0px;
    const syL = e2py - e0py;
    const ll = sxL * sxL + syL * syL;
    const dx0 = fragPxX - e0px;
    const dy0 = fragPxY - e0py;
    const tL = clamp((dx0 * sxL + dy0 * syL) / max(ll, 1.0e-9), 0.0, 1.0);
    const cx = e0px + tL * sxL - fragPxX;
    const cy = e0py + tL * syL - fragPxY;
    bestPx2 = cx * cx + cy * cy;
  } else {
    const cT1 = new V4f((vInstX + b.x) * sx, vInstY + b.y, 0.0, 1.0);
    const cC1 = uniform.ProjTrafo.mul(uniform.ViewTrafo.mul(uniform.ModelTrafo.mul(cT1)));
    for (let s: u32 = ZERO_U; s < SEEDS_U; s = s + ONE_U) {
      // u32 → f32 cast for the seed multiplier.
      const sNum: number = s as number;
      const sf: f32 = sNum as f32;
      let t: f32 = sf * 0.25;
      for (let n: u32 = ZERO_U; n < ITER_U; n = n + ONE_U) {
        const oneT = 1.0 - t;
        const aw = oneT * oneT;
        const bw = 2.0 * oneT * t;
        const cw = t * t;
        const daw = -2.0 * oneT;
        const dbw = 2.0 - 4.0 * t;
        const dcw = 2.0 * t;
        const Nx = aw * cC0.x + bw * cC1.x + cw * cC2.x;
        const Ny = aw * cC0.y + bw * cC1.y + cw * cC2.y;
        const Wv = aw * cC0.w + bw * cC1.w + cw * cC2.w;
        const dNx = daw * cC0.x + dbw * cC1.x + dcw * cC2.x;
        const dNy = daw * cC0.y + dbw * cC1.y + dcw * cC2.y;
        const dWv = daw * cC0.w + dbw * cC1.w + dcw * cC2.w;
        const invW = 1.0 / max(Wv, 1.0e-9);
        const Px = (Nx * invW + 1.0) * 0.5 * uniform.Viewport.x;
        const Py = (Ny * invW + 1.0) * 0.5 * uniform.Viewport.y;
        const dBx = (dNx * Wv - Nx * dWv) * invW * invW;
        const dBy = (dNy * Wv - Ny * dWv) * invW * invW;
        const dPx = dBx * 0.5 * uniform.Viewport.x;
        const dPy = dBy * 0.5 * uniform.Viewport.y;
        const rdx = Px - fragPxX;
        const rdy = Py - fragPxY;
        const Fv = rdx * dPx + rdy * dPy;
        const Fp = dPx * dPx + dPy * dPy;
        const dt = Fv / max(Fp, 1.0e-9);
        t = clamp(t - dt, 0.0, 1.0);
      }
      const oneTf = 1.0 - t;
      const awf = oneTf * oneTf;
      const bwf = 2.0 * oneTf * t;
      const cwf = t * t;
      const Nxf = awf * cC0.x + bwf * cC1.x + cwf * cC2.x;
      const Nyf = awf * cC0.y + bwf * cC1.y + cwf * cC2.y;
      const Wvf = awf * cC0.w + bwf * cC1.w + cwf * cC2.w;
      const invWf = 1.0 / max(Wvf, 1.0e-9);
      const Pxf = (Nxf * invWf + 1.0) * 0.5 * uniform.Viewport.x;
      const Pyf = (Nyf * invWf + 1.0) * 0.5 * uniform.Viewport.y;
      const rdxf = Pxf - fragPxX;
      const rdyf = Pyf - fragPxY;
      bestPx2 = min(bestPx2, rdxf * rdxf + rdyf * rdyf);
    }
    const e0dx = e0px - fragPxX;
    const e0dy = e0py - fragPxY;
    const e2dx = e2px - fragPxX;
    const e2dy = e2py - fragPxY;
    bestPx2 = min(bestPx2, e0dx * e0dx + e0dy * e0dy);
    bestPx2 = min(bestPx2, e2dx * e2dx + e2dy * e2dy);
  }
  return bestPx2;
}

// Original string-template fragment kept here for reference; the
// inline-marker effect below replaces it. Unused but readable.

let sdfTextEffectMemo: Effect | undefined;
function buildSdfTextEffect(): Effect {
  if (sdfTextEffectMemo) return sdfTextEffectMemo;

  const vsMain = vertex((input: {
    a_pos:        V2f;
    a_klmKind:    V4f;
    a_lensCands:  V3f;
    a_instOffset: V2f;
  }) => {
    const pPlus  = uniform.ProjTrafo.mul(uniform.ViewTrafo.mul(uniform.ModelTrafo.mul(new V4f( 1.0, 0.0, 0.0, 1.0))));
    const pMinus = uniform.ProjTrafo.mul(uniform.ViewTrafo.mul(uniform.ModelTrafo.mul(new V4f(-1.0, 0.0, 0.0, 1.0))));
    const sx = (pPlus.x / pPlus.w) < (pMinus.x / pMinus.w) ? -1.0 : 1.0;
    const wx = (input.a_instOffset.x + input.a_pos.x) * sx;
    const wy = input.a_instOffset.y + input.a_pos.y;
    const text = new V4f(wx, wy, 0.0, 1.0);
    const clip = uniform.ProjTrafo.mul(uniform.ViewTrafo.mul(uniform.ModelTrafo.mul(text)));
    return {
      gl_Position: clip,
      v_kind:      input.a_klmKind.w,
      v_klm:       new V3f(input.a_klmKind.x, input.a_klmKind.y, input.a_klmKind.z),
      v_lensCands: input.a_lensCands,
      v_inst:      input.a_instOffset,
      v_sx:        sx,
      v_pos:       input.a_pos,
    };
  });

  const fsMain = fragment((input: {
    v_kind:      f32;
    v_klm:       V3f;
    v_lensCands: V3f;
    v_inst:      V2f;
    v_sx:        f32;
    v_pos:       V2f;
  }) => {
    // Dispatch by kind:
    //   0      → body flat fill (alpha = 1).
    //   1      → bezier2 lens. Loop-Blinn implicit (k*k - l) * m
    //            decides solid (<= 0) vs discard (> 0) in the lens.
    //   2      → arc lens, same recipe with (k*k + l*l - 1) * m.
    //   3      → ribbon (NOT in the SDF index buffer).
    //   4      → band quad (run Newton on up to 6 candidate curves,
    //            take min pixel distance, ramp 1 → 0 across AaWidthPx).
    const sx = input.v_sx;
    const useDebug = uniform.DebugMode > 0.5;
    let alpha: f32 = 1.0;
    let lensOutside: f32 = 0.0;
    if (input.v_kind > 0.5 && input.v_kind < 2.5) {
      const k = input.v_klm.x;
      const l = input.v_klm.y;
      const m = input.v_klm.z;
      const isArc = input.v_kind > 1.5;
      const f = isArc ? (k * k + l * l - 1.0) * m : (k * k - l) * m;
      if (f > 0.0) lensOutside = 1.0;
    }
    const isLens = input.v_kind > 0.5 && input.v_kind < 2.5;
    const isBand = input.v_kind > 3.5 && input.v_kind < 4.5;
    const isFill = input.v_kind > 4.5;
    if (isBand || isFill || isLens) {
      // Symmetric AA ramp centred on the outline:
      //   α = clamp(0.5 + sign·distPx/aaW, 0, 1)
      // sign = +1 INSIDE the outline, −1 OUTSIDE. The band is always
      // outside; the interior shell (fill ramp) is always inside; a lens
      // fragment is inside the curve when the Loop-Blinn implicit f ≤ 0
      // (lensOutside == 0), outside when f > 0. At distPx = 0 (on the
      // outline) every kind gives α = 0.5, so the perceived edge sits on
      // the true boundary instead of ½·aaW outside it — no fattening.
      let sign: f32 = 1.0;
      if (isBand) { sign = -1.0; }
      else if (isLens && lensOutside > 0.5) { sign = -1.0; }
      const aaW = max(uniform.AaWidthPx, 1.0e-3);
      const fragText = new V4f(
        (input.v_inst.x + input.v_pos.x) * sx,
        input.v_inst.y + input.v_pos.y,
        0.0, 1.0,
      );
      const fragClip = uniform.ProjTrafo.mul(uniform.ViewTrafo.mul(uniform.ModelTrafo.mul(fragText)));
      const fragPxX = (fragClip.x / fragClip.w + 1.0) * 0.5 * uniform.Viewport.x;
      const fragPxY = (fragClip.y / fragClip.w + 1.0) * 0.5 * uniform.Viewport.y;

      let minDistPx2: f32 = 1.0e20;
      if (isBand || isFill) {
        // Band + interior shell: loop this triangle's candidate CSR
        // range. triId lives in klm.x (slot 2); curve indices are global
        // SSBO indices.
        const ONE_U: u32 = 1 as u32;
        const triId = (input.v_klm.x + 0.5) as u32;
        const lo = bandCandOffsets[triId] as u32;
        const hi = bandCandOffsets[triId + ONE_U] as u32;
        for (let i: u32 = lo; i < hi; i = i + ONE_U) {
          const ciU = bandCandIndices[i] as u32;
          const ci: f32 = (ciU as number) as f32;
          minDistPx2 = min(minDistPx2, processCand(ci, sx, fragPxX, fragPxY, input.v_inst.x, input.v_inst.y));
        }
      } else {
        // Lens fragment (inside OR outside the curve): the 3 inline
        // candidates (this curve + its prev/next neighbours) in slots 6..8.
        minDistPx2 = min(minDistPx2, processCand(input.v_lensCands.x, sx, fragPxX, fragPxY, input.v_inst.x, input.v_inst.y));
        minDistPx2 = min(minDistPx2, processCand(input.v_lensCands.y, sx, fragPxX, fragPxY, input.v_inst.x, input.v_inst.y));
        minDistPx2 = min(minDistPx2, processCand(input.v_lensCands.z, sx, fragPxX, fragPxY, input.v_inst.x, input.v_inst.y));
      }
      const minDistPx = sqrt(minDistPx2);
      alpha = clamp(0.5 + sign * minDistPx / aaW, 0.0, 1.0);
      if (alpha <= 0.0 && !useDebug) discard();
    }
    // Real-mode color: PathColor with computed alpha.
    const aa_ = alpha * uniform.PathColor.w;
    const realR = uniform.PathColor.x * aa_;
    const realG = uniform.PathColor.y * aa_;
    const realB = uniform.PathColor.z * aa_;
    const realA = aa_;

    // Debug-mode color: hash the band-tri id (parked in v_klm.x —
    // band tris don't use klm at all, so we reuse the slot) into a
    // deterministic colour. Acts as a primitive id since WGSL has
    // no fragment-stage primitive_index.
    const tid = input.v_klm.x + 1.0;
    const bandR = 0.25 + 0.75 * fract(tid * 0.6180339887);
    const bandG = 0.25 + 0.75 * fract(tid * 0.3819660113);
    const bandB = 0.25 + 0.75 * fract(tid * 0.7548776662);

    const isRampDbg = isBand || isFill;
    const debugR = isRampDbg ? bandR : realR;
    const debugG = isRampDbg ? bandG : realG;
    const debugB = isRampDbg ? bandB : realB;
    const debugA = isRampDbg ? 1.0   : realA;

    const outR = useDebug ? debugR : realR;
    const outG = useDebug ? debugG : realG;
    const outB = useDebug ? debugB : realB;
    const outA = useDebug ? debugA : realA;
    return { Colors: new V4f(outR, outG, outB, outA) };
  });

  sdfTextEffectMemo = effect(vsMain, fsMain);
  return sdfTextEffectMemo;
}

let alphaOverBlend: BlendState | undefined;
function alphaOverBlendState(): BlendState {
  if (alphaOverBlend) return alphaOverBlend;
  // Standard premultiplied alpha-over. The band geometry now tiles
  // exactly (perpendicular trapezoids per chunk + corner-fan tris at
  // joins) so band tris don't overlap and a normal blend composes
  // correctly with any background.
  alphaOverBlend = {
    color: {
      operation: AVal.constant<GPUBlendOperation>("add"),
      srcFactor: AVal.constant<GPUBlendFactor>("one"),
      dstFactor: AVal.constant<GPUBlendFactor>("one-minus-src-alpha"),
    },
    alpha: {
      operation: AVal.constant<GPUBlendOperation>("add"),
      srcFactor: AVal.constant<GPUBlendFactor>("one"),
      dstFactor: AVal.constant<GPUBlendFactor>("one-minus-src-alpha"),
    },
    writeMask: AVal.constant(0xf),
  };
  return alphaOverBlend;
}

export interface SdfTextArgs {
  font: Font;
  text: string;
  align: "left" | "center" | "right";
  kerning: boolean;
  cache: GlyphCache;
  color: aval<V4f>;
  aaWidth: aval<number>;
  scope?: SgScopeProps;
}

export function buildSdfTextScene(args: SdfTextArgs): SgNode {
  const { font, text, align, kerning, cache, color, aaWidth, scope } = args;
  const layout = layoutText(font, text, { kerning });
  const emScale = 1 / (font.unitsPerEm || 1);
  const totalAdvance = layout.advance * emScale;

  // Group instances by codepoint — one indexed-instanced draw per
  // unique glyph; instance attribute carries (cx, by) per occurrence.
  type Group = { rec: ReturnType<GlyphCache["get"]>; insts: number[] };
  const groups = new Map<number, Group>();
  for (const g of layout.glyphs) {
    const rec = cache.get(g.codepoint);
    if (rec.empty || rec.sdfIndexCount === 0) continue;
    const cx = g.x * emScale + rec.advance * 0.5 - totalAdvance * 0.5;
    const by = g.y * emScale;
    let group = groups.get(g.codepoint);
    if (!group) { group = { rec, insts: [] }; groups.set(g.codepoint, group); }
    group.insts.push(cx, by);
  }

  // Snapshot the cache's atlases. Shared across all per-glyph leaves
  // in this run; each leaf differs only in the per-instance offset
  // buffer and the (firstIndex, indexCount, baseVertex) draw range.
  const interleaved = cache.vertexBuffer();
  const sdfIndices  = cache.sdfIndexBuffer();
  const triPacked   = cache.trianglePackedBuffer();

  const totalVerts = interleaved.length / GLYPH_FLOATS_PER_VERTEX;
  const STRIDE_BYTES = GLYPH_FLOATS_PER_VERTEX * 4;
  void totalVerts;

  const vertBuf = IBuffer.fromHost(interleaved);
  const idxBuf  = IBuffer.fromHost(sdfIndices);
  const triBuf  = IBuffer.fromHost(triPacked);

  // Per-glyph vertex attribute views — same buffer, different offsets.
  const vertBufAval = AVal.constant(vertBuf);
  const vertexAttrs = HashMap.empty<string, BufferView>()
    .add("a_pos",     { buffer: vertBufAval, elementType: ElementType.V2f, offset: 0,  stride: STRIDE_BYTES })
    .add("a_klmKind", { buffer: vertBufAval, elementType: ElementType.V4f, offset: 8,  stride: STRIDE_BYTES })
    .add("a_lensCands", { buffer: vertBufAval, elementType: ElementType.V3f, offset: 24, stride: STRIDE_BYTES });

  const candOffBuf = IBuffer.fromHost(cache.bandCandOffsetBuffer());
  const candIdxBuf = IBuffer.fromHost(cache.bandCandIndexBuffer());
  const storageBuffers = HashMap.empty<string, aval<IBuffer>>()
    .add("tris", AVal.constant<IBuffer>(triBuf))
    .add("bandCandOffsets", AVal.constant<IBuffer>(candOffBuf))
    .add("bandCandIndices", AVal.constant<IBuffer>(candIdxBuf));

  const effect = buildSdfTextEffect();
  const viewportV2: aval<V2f> = ambViewport.map(
    (vp) => new V2f(vp.width, vp.height),
  );

  const colorAval: aval<V4f> = color;

  const leafChildren: SgNode[] = [];
  // Whole SDF index buffer view shared across leaves; per-glyph slice
  // lives in the DrawCall's firstIndex / indexCount.
  const sharedIndexBV: BufferView = {
    buffer: AVal.constant(idxBuf),
    elementType: ElementType.U32,
  };
  for (const [, { rec, insts }] of groups) {
    const instCount = insts.length / 2;
    const instArr = new Float32Array(insts);
    const instBuf = IBuffer.fromHost(instArr);
    const instanceAttrs = HashMap.empty<string, BufferView>()
      .add("a_instOffset", { buffer: AVal.constant(instBuf), elementType: ElementType.V2f });
    const drawCall: DrawCall = {
      kind: "indexed",
      indexCount:    rec.sdfIndexCount,
      instanceCount: instCount,
      firstIndex:    rec.sdfFirstIndex,
      baseVertex:    rec.baseVertex,
      firstInstance: 0,
    };
    leafChildren.push((Sg.leaf({
      vertexAttributes: vertexAttrs,
      instanceAttributes: instanceAttrs,
      indices: sharedIndexBV,
      drawCall: AVal.constant<DrawCall>(drawCall),
      storageBuffers,
    })));
  }

  const alignDx
    = align === "left"  ? +totalAdvance * 0.5
    : align === "right" ? -totalAdvance * 0.5
    : 0;
  const alignTrafo = alignDx === 0
    ? undefined
    : Trafo3d.translation(new V3d(alignDx, 0, 0));

  const tree = Sg({
    Shader: effect,
    Uniform: {
      PathColor: colorAval,
      Viewport: viewportV2,
      AaWidthPx: aaWidth,
    },
    CullMode: "none",
    BlendMode: alphaOverBlendState(),
    // Body fill and band geometry now meet at the body's interior
    // approximation polyline at exact em-space coords (band-builder
    // feeds the inner polyline straight into libtess; only the
    // Clipper-inflated outer side is quantized). Body+band share
    // their boundary bit-for-bit, so depth writes can stay on with
    // less-equal — no z-fight on the seam. The remaining overlap is
    // the Loop-Blinn lens triangles for outward-bulging beziers
    // (band overlaps lens; lens already paints α=1 so the band's
    // SDF result there is harmlessly covered by the lens fragment).
    DepthTest: "less-equal",
    DepthMask: true,
    ...(alignTrafo !== undefined ? { Trafo: alignTrafo } : {}),
    children: leafChildren,
  } as never);

  return scope === undefined
    ? tree
    : Sg({ ...(scope as object), children: [tree] } as never);
}
