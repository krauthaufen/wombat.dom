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

/**
 * Emit one candidate iteration. `candFExpr` evaluates to a candidate's
 * SSBO index (or -1 to skip). Wrapped in its own block so locals
 * are scoped per call. SSBO entry layout per candidate:
 *   tris[base + 0] = (a.xy, klm0.xy)
 *   tris[base + 1] = (b.xy, klm1.xy)   ← b == a marks line sentinel
 *   tris[base + 2] = (c.xy, klm2.xy)
 *   tris[base + 3] = (klm0.z, klm1.z, klm2.z, kind)
 */
function candidateLoopBody(candFExpr: string): string {
  return `
        {
          const candF = ${candFExpr};
          if (candF >= 0.0) {
            const idx = (candF + 0.5) as u32;
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
              const e0dx = e0px - fragPxX;
              const e0dy = e0py - fragPxY;
              const e2dx = e2px - fragPxX;
              const e2dy = e2py - fragPxY;
              bestPx2 = min(bestPx2, e0dx*e0dx + e0dy*e0dy);
              bestPx2 = min(bestPx2, e2dx*e2dx + e2dy*e2dy);
            }
            minDistPx2 = min(minDistPx2, bestPx2);
          }
        }
  `;
}

// Build stamp — bumped on every meaningful rebuild so you can verify
// you're seeing the current code in the browser. Print this from the
// console (or read the message logged on first effect build) and
// compare with what I tell you the current stamp is.
const SDF_BUILD_STAMP = "v13-band-solid-red-2026-05-05";

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
    declare const DebugMode:  f32;  // 0 = real SDF; 1 = candidate-count viz

    function vsMain(input: {
      a_pos:        V2f;
      a_klmKind:    V4f;     // klm.xyz, kind  (band reuses klm.x as triId)
      a_cands:      V4f;     // band only: candidates c0..c3
      a_cands2:     V2f;     // band only: candidates c4, c5
      a_instOffset: V2f;     // (cx, by)
    }): {
      gl_Position: V4f;
      v_kind:      f32;
      v_klm:       V3f;
      v_cands:     V4f;
      v_cands2:    V2f;
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
        v_cands:     input.a_cands,
        v_cands2:    input.a_cands2,
        v_inst:      input.a_instOffset,
        v_sx:        sx,
        v_pos:       input.a_pos,
      };
    }

    function fsMain(input: {
      v_kind: f32;
      v_klm:  V3f;
      v_cands: V4f;
      v_cands2: V2f;
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
      const useDebug = DebugMode > 0.5;
      var alpha: f32 = 1.0;
      // Lens / band classification. A "lens-outside" fragment is one
      // inside the bezier/arc lens triangle but on the wrong side of
      // the actual curve — same job as a band fragment, just with 3
      // candidate curves (this curve + prev + next adjacent) instead
      // of up to 6.
      var lensOutside: f32 = 0.0;
      if (input.v_kind > 0.5 && input.v_kind < 2.5) {
        const k = input.v_klm.x;
        const l = input.v_klm.y;
        const m = input.v_klm.z;
        const isArc = input.v_kind > 1.5;
        const f = isArc ? (k*k + l*l - 1.0) * m : (k*k - l) * m;
        if (f > 0.0) lensOutside = 1.0;
      }
      const isBand = input.v_kind > 3.5;
      if (isBand || lensOutside > 0.5) {
        const Mvp = ProjTrafo.mul(ViewTrafo.mul(ModelTrafo));
        const aaW = max(AaWidthPx, 1.0e-3);
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
        const SEEDS_U: u32 = 5 as u32;
        const ITER_U:  u32 = 8 as u32;

        // Both lens-outside and band run cands [0..2] (lens fixed
        // 3-tuple = self/prev/next; band's first 3 of pass-1+2 list).
        // Band additionally runs [3..5]; lens has -1 in those slots
        // and the candidate-loop body is a no-op for negative ids.
        var minDistPx2: f32 = 1.0e20;
        ${candidateLoopBody("input.v_cands.x")}
        ${candidateLoopBody("input.v_cands.y")}
        ${candidateLoopBody("input.v_cands.z")}
        ${candidateLoopBody("input.v_cands.w")}
        ${candidateLoopBody("input.v_cands2.x")}
        ${candidateLoopBody("input.v_cands2.y")}
        const minDistPx = sqrt(minDistPx2);
        alpha = clamp(1.0 - minDistPx / aaW, 0.0, 1.0);
        if (alpha <= 0.0 && !useDebug) discard;
      }
      // Real-mode color: PathColor with computed alpha.
      const aa_ = alpha * PathColor.w;
      const realR = PathColor.x * aa_;
      const realG = PathColor.y * aa_;
      const realB = PathColor.z * aa_;
      const realA = aa_;

      // Debug-mode color: hash the band-tri id (parked in v_klm.x —
      // band tris don't use klm at all, so we reuse the slot) into
      // a deterministic colour. Same value on all 3 verts → flat
      // across the tri. Acts as a primitive id since WebGPU/WGSL
      // has no fragment-stage primitive_index.
      const tid = input.v_klm.x + 1.0;
      const bandR = 0.25 + 0.75 * fract(tid * 0.6180339887);
      const bandG = 0.25 + 0.75 * fract(tid * 0.3819660113);
      const bandB = 0.25 + 0.75 * fract(tid * 0.7548776662);

      const debugR = isBand ? bandR : realR;
      const debugG = isBand ? bandG : realG;
      const debugB = isBand ? bandB : realB;
      const debugA = isBand ? 1.0   : realA;

      // Single Uniform-controlled mux. DebugMode > 0.5 → debug ramp.
      const outR = useDebug ? debugR : realR;
      const outG = useDebug ? debugG : realG;
      const outB = useDebug ? debugB : realB;
      const outA = useDebug ? debugA : realA;
      return { outColor: new V4f(outR, outG, outB, outA) };
    }
  `;

  const entries: EntryRequest[] = [
    {
      name: "vsMain", stage: "vertex",
      inputs: [
        { name: "a_pos",         type: Tvec2f, semantic: "Pos",        decorations: [{ kind: "Location", value: 0 }] },
        { name: "a_klmKind",     type: Tvec4f, semantic: "KlmKind",    decorations: [{ kind: "Location", value: 1 }] },
        { name: "a_cands",       type: Tvec4f, semantic: "Cands",      decorations: [{ kind: "Location", value: 2 }] },
        { name: "a_cands2",      type: Tvec2f, semantic: "Cands2",     decorations: [{ kind: "Location", value: 4 }] },
        { name: "a_instOffset",  type: Tvec2f, semantic: "InstOffset", decorations: [{ kind: "Location", value: 5 }] },
      ],
      outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "v_kind",      type: Tf32,   semantic: "Kind",     decorations: [{ kind: "Location", value: 0 }] },
        { name: "v_cands",     type: Tvec4f, semantic: "Cands",    decorations: [{ kind: "Location", value: 1 }] },
        { name: "v_inst",      type: Tvec2f, semantic: "Inst",     decorations: [{ kind: "Location", value: 3 }] },
        { name: "v_sx",        type: Tf32,   semantic: "Sx",       decorations: [{ kind: "Location", value: 4 }] },
        { name: "v_pos",       type: Tvec2f, semantic: "PosEm",    decorations: [{ kind: "Location", value: 5 }] },
        { name: "v_klm",       type: Tvec3f, semantic: "Klm",      decorations: [{ kind: "Location", value: 6 }] },
        { name: "v_cands2",    type: Tvec2f, semantic: "Cands2",   decorations: [{ kind: "Location", value: 7 }] },
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
  externalTypes.set("DebugMode",  Tf32);
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
      { name: "DebugMode",  type: Tf32,   group: 0, slot: 0, buffer: "Camera" },
    ],
  };
  const trisSSBO: ValueDef = {
    kind: "StorageBuffer",
    binding: { group: 1, slot: 0 },
    name: "tris",
    layout: Tvec4Array,
    access: "read",
  };

  let parsed;
  try {
    parsed = parseShader({ source, entries, externalTypes });
  } catch (e: unknown) {
    if (e && typeof e === "object" && "diagnostics" in e) {
      // eslint-disable-next-line no-console
      console.error("[sdf-text] parseShader diagnostics:",
        JSON.stringify((e as { diagnostics: unknown }).diagnostics, null, 2));
    }
    throw e;
  }
  const merged: Module = { ...parsed, values: [camUBO, trisSSBO, ...parsed.values] };
  sdfTextEffectMemo = stage(merged);
  // eslint-disable-next-line no-console
  console.log(`[sdf-text] ${SDF_BUILD_STAMP} effect ready`);
  try {
    const compiled = sdfTextEffectMemo.compile({ target: "wgsl" });
    for (const s of compiled.stages) {
      // eslint-disable-next-line no-console
      console.log(`[sdf-text] ${s.entryName} (${s.stage}):\n${s.source}`);
    }
  } catch (e) {
    // eslint-disable-next-line no-console
    console.warn(`[sdf-text] compile dump failed:`, e);
  }
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
    .add("a_cands", AVal.constant<BufferView>({
      buffer: vertBuf, offset: 24, count: totalVerts, stride: STRIDE_BYTES, format: "float32x4",
    }))
    .add("a_cands2", AVal.constant<BufferView>({
      buffer: vertBuf, offset: 40, count: totalVerts, stride: STRIDE_BYTES, format: "float32x2",
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
