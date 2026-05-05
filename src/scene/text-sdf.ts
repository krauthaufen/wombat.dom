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
import { V2f, V3d, V4f, Trafo3d } from "@aardworx/wombat.base";
import {
  type Font, GlyphCache, layoutText, GLYPH_FLOATS_PER_VERTEX,
} from "@aardworx/wombat.base/font";
import {
  IBuffer, type BufferView, type DrawCall, type BlendState,
} from "@aardworx/wombat.rendering/core";
import type { Effect } from "@aardworx/wombat.shader";
import { stage } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import {
  Mat, Tf32, Tu32, Vec, type Type, type ValueDef, type Module,
} from "@aardworx/wombat.shader/ir";

import type { VNode } from "../vnode.js";
import { Sg } from "./constructors.js";
import { sgVNode } from "./sgVNode.js";
import type { SgScopeProps } from "./constructors.js";
import { viewport as ambViewport } from "./ambient.js";

const Tvec2f: Type = Vec(Tf32, 2);
const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);
const TM44f:  Type = Mat(Tf32, 4, 4);
const Tvec4Array: Type = { kind: "Array", element: Tvec4f, length: "runtime" };
void Tu32;

// (Removed per-candidate Newton template — see fsMain below; the FS
//  loops over the glyph's full SSBO range from triFirst..triCount.)

let sdfTextEffectMemo: Effect | undefined;
function buildSdfTextEffect(): Effect {
  if (sdfTextEffectMemo) return sdfTextEffectMemo;

  const source = `
    declare const ModelTrafo: M44f;
    declare const ViewTrafo:  M44f;
    declare const ProjTrafo:  M44f;
    declare const PathColor:  V4f;
    declare const Viewport:   V2f;
    declare const AaWidthPx:  f32;

    function vsMain(input: {
      a_pos:        V2f;
      a_klmKind:    V4f;     // klm.xyz, kind
      a_triRange:   V2f;     // band only: (triFirst, triCount) into SSBO
      a_instOffset: V2f;     // (cx, by)
    }): {
      gl_Position: V4f;
      v_kind:      f32;
      v_klm:       V3f;
      v_triRange:  V2f;
      v_inst:      V2f;
      v_sx:        f32;
      v_pos:       V2f;
    } {
      const pPlus  = ProjTrafo.mul(ViewTrafo.mul(ModelTrafo.mul(new V4f( 1.0, 0.0, 0.0, 1.0))));
      const pMinus = ProjTrafo.mul(ViewTrafo.mul(ModelTrafo.mul(new V4f(-1.0, 0.0, 0.0, 1.0))));
      const sx = (pPlus.x / pPlus.w) < (pMinus.x / pMinus.w) ? -1.0 : 1.0;
      const wx = (input.a_instOffset.x + input.a_pos.x) * sx;
      const wy = input.a_instOffset.y + input.a_pos.y;
      const text = new V4f(wx, wy, 0.0, 1.0);
      const clip = ProjTrafo.mul(ViewTrafo.mul(ModelTrafo.mul(text)));
      return {
        gl_Position: clip,
        v_kind:      input.a_klmKind.w,
        v_klm:       new V3f(input.a_klmKind.x, input.a_klmKind.y, input.a_klmKind.z),
        v_triRange:  input.a_triRange,
        v_inst:      input.a_instOffset,
        v_sx:        sx,
        v_pos:       input.a_pos,
      };
    }

    function fsMain(input: {
      v_kind: f32;
      v_klm:  V3f;
      v_triRange: V2f;
      v_inst: V2f;
      v_sx: f32;
      v_pos:  V2f;
    }): { outColor: V4f } {
      // Dispatch by kind:
      //   0      → body flat fill (alpha = 1).
      //   1      → bezier2 lens. Loop-Blinn implicit (k*k - l) * m
      //            decides solid (<= 0) vs discard (> 0) in the lens.
      //   2      → arc lens, same recipe with (k*k + l*l - 1) * m.
      //   3      → ribbon (NOT in the SDF index buffer).
      //   4      → band quad (run Newton on up to 6 candidate
      //            curves, take min pixel distance, ramp 1 → 0
      //            across AaWidthPx).
      const sx = input.v_sx;
      var alpha: f32 = 1.0;
      if (input.v_kind > 0.5 && input.v_kind < 2.5) {
        const k = input.v_klm.x;
        const l = input.v_klm.y;
        const m = input.v_klm.z;
        const isArc = input.v_kind > 1.5;
        const f = isArc ? (k*k + l*l - 1.0) * m : (k*k - l) * m;
        if (f > 0.0) discard;
      }
      if (input.v_kind > 3.5) {
        const Mvp = ProjTrafo.mul(ViewTrafo.mul(ModelTrafo));
        const aaW = max(AaWidthPx, 1.0e-3);

        // Re-project this fragment's em-space pos to pixel coords.
        const fragText = new V4f(
          (input.v_inst.x + input.v_pos.x) * sx,
          input.v_inst.y + input.v_pos.y,
          0.0, 1.0,
        );
        const fragClip = Mvp.mul(fragText);
        const fragPxX = (fragClip.x / fragClip.w + 1.0) * 0.5 * Viewport.x;
        const fragPxY = (fragClip.y / fragClip.w + 1.0) * 0.5 * Viewport.y;

        const ZERO_U: u32 = 0 as u32;
        const ONE_U:  u32 = 1 as u32;
        const TWO_U:  u32 = 2 as u32;
        const FOUR_U: u32 = 4 as u32;
        // Newton config matches the original bbox-quad shader that
        // "just worked": 5 seeds at t = 0, 0.25, 0.5, 0.75, 1; 8
        // iterations of Gauss-Newton on each candidate's rational
        // pixel-space bezier. The denser seed grid and longer iter
        // budget reliably find the global minimum even for high-
        // curvature beziers; explicit endpoint checks below guarantee
        // we never lose the t=0 / t=1 distance to a Newton seed that
        // converges to a local maximum or stalls at the clamp.
        const SEEDS_U: u32 = 5 as u32;
        const ITER_U:  u32 = 8 as u32;

        // Iterate every curve in this glyph's SSBO range. With the
        // band geometry covering only the halo strip, far-away
        // background pixels never reach this loop, so the per-glyph
        // Newton cost is paid only on visibly-covered fragments.
        const triFirst = (input.v_triRange.x + 0.5) as u32;
        const triCount = (input.v_triRange.y + 0.5) as u32;
        var minDistPx2: f32 = 1.0e20;
        for (let ti: u32 = ZERO_U; ti < triCount; ti = ti + ONE_U) {
          const idx = triFirst + ti;
          const base = idx * FOUR_U;
          const a = tris[base];
          const b = tris[base + ONE_U];
          const c = tris[base + TWO_U];
          const cT0 = new V4f((input.v_inst.x + a.x) * sx, input.v_inst.y + a.y, 0.0, 1.0);
          const cT2 = new V4f((input.v_inst.x + c.x) * sx, input.v_inst.y + c.y, 0.0, 1.0);
          const cC0 = Mvp.mul(cT0);
          const cC2 = Mvp.mul(cT2);
          const e0px = (cC0.x / cC0.w + 1.0) * 0.5 * Viewport.x;
          const e0py = (cC0.y / cC0.w + 1.0) * 0.5 * Viewport.y;
          const e2px = (cC2.x / cC2.w + 1.0) * 0.5 * Viewport.x;
          const e2py = (cC2.y / cC2.w + 1.0) * 0.5 * Viewport.y;
          var bestPx2: f32 = 1.0e20;
          // Line sentinel: glyph-cache emits line edges as quadratic
          // entries with p1 = p0. Closed-form point-segment distance.
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
            minDistPx2 = min(minDistPx2, bestPx2);
            continue;
          }
          const cT1 = new V4f((input.v_inst.x + b.x) * sx, input.v_inst.y + b.y, 0.0, 1.0);
          const cC1 = Mvp.mul(cT1);
          for (let s: u32 = ZERO_U; s < SEEDS_U; s = s + ONE_U) {
            var t: f32 = (s as f32) * 0.25;
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
              const Px = (Nx * invW + 1.0) * 0.5 * Viewport.x;
              const Py = (Ny * invW + 1.0) * 0.5 * Viewport.y;
              const dBx = (dNx * Wv - Nx * dWv) * invW * invW;
              const dBy = (dNy * Wv - Ny * dWv) * invW * invW;
              const dPx = dBx * 0.5 * Viewport.x;
              const dPy = dBy * 0.5 * Viewport.y;
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
            const Pxf = (Nxf * invWf + 1.0) * 0.5 * Viewport.x;
            const Pyf = (Nyf * invWf + 1.0) * 0.5 * Viewport.y;
            const rdxf = Pxf - fragPxX;
            const rdyf = Pyf - fragPxY;
            bestPx2 = min(bestPx2, rdxf * rdxf + rdyf * rdyf);
          }
          // Always consider the true endpoints t=0 (= P0) and t=1
          // (= P2) — at segment-to-segment junctions the closest
          // contour point IS the shared endpoint, and Newton seeds
          // can stall at the [0, 1] clamp giving a slightly-off
          // distance. e0px/e0py/e2px/e2py were hoisted above the
          // line-sentinel branch.
          const e0dx = e0px - fragPxX;
          const e0dy = e0py - fragPxY;
          const e2dx = e2px - fragPxX;
          const e2dy = e2py - fragPxY;
          bestPx2 = min(bestPx2, e0dx*e0dx + e0dy*e0dy);
          bestPx2 = min(bestPx2, e2dx*e2dx + e2dy*e2dy);
          minDistPx2 = min(minDistPx2, bestPx2);
        }
        const minDistPx = sqrt(minDistPx2);
        alpha = clamp(1.0 - minDistPx / aaW, 0.0, 1.0);
        if (alpha <= 0.0) discard;
      }
      const aa_ = alpha * PathColor.w;
      return { outColor: new V4f(PathColor.x * aa_, PathColor.y * aa_, PathColor.z * aa_, aa_) };
    }
  `;

  const entries: EntryRequest[] = [
    {
      name: "vsMain", stage: "vertex",
      inputs: [
        { name: "a_pos",         type: Tvec2f, semantic: "Pos",        decorations: [{ kind: "Location", value: 0 }] },
        { name: "a_klmKind",     type: Tvec4f, semantic: "KlmKind",    decorations: [{ kind: "Location", value: 1 }] },
        { name: "a_triRange",    type: Tvec2f, semantic: "TriRange",   decorations: [{ kind: "Location", value: 2 }] },
        { name: "a_instOffset",  type: Tvec2f, semantic: "InstOffset", decorations: [{ kind: "Location", value: 3 }] },
      ],
      outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "v_kind",      type: Tf32,   semantic: "Kind",     decorations: [{ kind: "Location", value: 0 }] },
        { name: "v_triRange",  type: Tvec2f, semantic: "TriRange", decorations: [{ kind: "Location", value: 1 }] },
        { name: "v_inst",      type: Tvec2f, semantic: "Inst",     decorations: [{ kind: "Location", value: 3 }] },
        { name: "v_sx",        type: Tf32,   semantic: "Sx",       decorations: [{ kind: "Location", value: 4 }] },
        { name: "v_pos",       type: Tvec2f, semantic: "PosEm",    decorations: [{ kind: "Location", value: 5 }] },
        { name: "v_klm",       type: Tvec3f, semantic: "Klm",      decorations: [{ kind: "Location", value: 6 }] },
      ],
    },
    {
      name: "fsMain", stage: "fragment",
      outputs: [
        { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
  ];

  const externalTypes = new Map<string, Type>();
  externalTypes.set("ModelTrafo", TM44f);
  externalTypes.set("ViewTrafo",  TM44f);
  externalTypes.set("ProjTrafo",  TM44f);
  externalTypes.set("PathColor",  Tvec4f);
  externalTypes.set("Viewport",   Tvec2f);
  externalTypes.set("AaWidthPx",  Tf32);
  externalTypes.set("tris",       Tvec4Array);

  const camUBO: ValueDef = {
    kind: "Uniform",
    uniforms: [
      { name: "ModelTrafo", type: TM44f,  group: 0, slot: 0, buffer: "Camera" },
      { name: "ViewTrafo",  type: TM44f,  group: 0, slot: 0, buffer: "Camera" },
      { name: "ProjTrafo",  type: TM44f,  group: 0, slot: 0, buffer: "Camera" },
      { name: "PathColor",  type: Tvec4f, group: 0, slot: 0, buffer: "Camera" },
      { name: "Viewport",   type: Tvec2f, group: 0, slot: 0, buffer: "Camera" },
      { name: "AaWidthPx",  type: Tf32,   group: 0, slot: 0, buffer: "Camera" },
    ],
  };
  const trisSSBO: ValueDef = {
    kind: "StorageBuffer",
    binding: { group: 1, slot: 0 },
    name: "tris",
    layout: Tvec4Array,
    access: "read",
  };

  const parsed = parseShader({ source, entries, externalTypes });
  const merged: Module = { ...parsed, values: [camUBO, trisSSBO, ...parsed.values] };
  sdfTextEffectMemo = stage(merged);
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

export function buildSdfTextScene(args: SdfTextArgs): VNode {
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
  const vertexAttrs = HashMap.empty<string, aval<BufferView>>()
    .add("a_pos", AVal.constant<BufferView>({
      buffer: vertBuf, offset: 0,  count: totalVerts, stride: STRIDE_BYTES, format: "float32x2",
    }))
    .add("a_klmKind", AVal.constant<BufferView>({
      buffer: vertBuf, offset: 8,  count: totalVerts, stride: STRIDE_BYTES, format: "float32x4",
    }))
    .add("a_triRange", AVal.constant<BufferView>({
      buffer: vertBuf, offset: 24, count: totalVerts, stride: STRIDE_BYTES, format: "float32x2",
    }));

  const storageBuffers = HashMap.empty<string, aval<IBuffer>>()
    .add("tris", AVal.constant<IBuffer>(triBuf));

  const effect = buildSdfTextEffect();
  const viewportV2: aval<V2f> = ambViewport.map(
    (vp) => new V2f(vp.width, vp.height),
  );

  const colorAval: aval<V4f> = color;

  const leafChildren: VNode[] = [];
  // Whole SDF index buffer view shared across leaves; per-glyph slice
  // lives in the DrawCall's firstIndex / indexCount.
  const sharedIndexBV: BufferView = {
    buffer: idxBuf, offset: 0, count: sdfIndices.length, stride: 4, format: "uint32",
  };
  for (const [, { rec, insts }] of groups) {
    const instCount = insts.length / 2;
    const instArr = new Float32Array(insts);
    const instBuf = IBuffer.fromHost(instArr);
    const instanceAttrs = HashMap.empty<string, aval<BufferView>>()
      .add("a_instOffset", AVal.constant<BufferView>({
        buffer: instBuf, offset: 0, count: instCount, stride: 8, format: "float32x2",
      }));
    const drawCall: DrawCall = {
      kind: "indexed",
      indexCount:    rec.sdfIndexCount,
      instanceCount: instCount,
      firstIndex:    rec.sdfFirstIndex,
      baseVertex:    rec.baseVertex,
      firstInstance: 0,
    };
    leafChildren.push(sgVNode(Sg.leaf({
      vertexAttributes: vertexAttrs,
      instanceAttributes: instanceAttrs,
      indices: AVal.constant<BufferView>(sharedIndexBV),
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
    DepthTest: "less-equal",
    // Body and band overlap on a thin strip along the contour; both
    // would write the same depth value but the rasteriser doesn't
    // guarantee the band fragment lands at the same projected z as
    // the body (rounding differences after Mvp), so leaving write on
    // produces z-fighting that hides the body. Read-only depth is
    // fine here because the run is rendered front-to-back implicitly
    // by glyph order.
    DepthMask: false,
    ...(alignTrafo !== undefined ? { Trafo: alignTrafo } : {}),
    children: leafChildren,
  } as never);

  return scope === undefined
    ? tree
    : Sg({ ...(scope as object), children: [tree] } as never);
}
