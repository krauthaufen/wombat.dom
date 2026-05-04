// Per-pixel triangle-SDF text rendering for `aa = "alpha-blending"`.
//
// Pipeline (per Sg.Text run):
//   1. Layout the text as usual.
//   2. For each glyph instance: emit a quad covering the glyph's
//      bbox + a small world-space pad.
//   3. Bind the per-font packed triangle buffer (built lazily by
//      `GlyphCache`) as a read-only storage buffer. Each triangle is
//      4 × vec4 = 16 floats = 64 bytes carrying its 3 verts, per-vert
//      klm, and kind. Ribbons (kind=3) are excluded.
//   4. Vertex shader emits the bbox quad and forwards glyph-local XY.
//   5. Fragment shader linear-scans triangles in this glyph's range.
//      For each triangle that contains the fragment (barycentric
//      inside-test), it interpolates klm and evaluates alpha by kind:
//        - kind = 0 (interior flat): α = 1.
//        - kind = 1 (bezier2): f = (k²−l)·m, α = clamp(0.5 − f / (|∇f|·AaWidthPx)).
//        - kind = 2 (arc):     f = (k²+l²−1)·m, same recipe.
//      We take max(α) across overlapping triangles.

import { AVal, HashMap, type aval } from "@aardworx/wombat.adaptive";
import { V2f, V3d, V4f, Trafo3d } from "@aardworx/wombat.base";
import {
  type Font, GlyphCache, layoutText, TRI_FLOATS_PER_TRI,
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
const Tvec4f: Type = Vec(Tf32, 4);
const TM44f:  Type = Mat(Tf32, 4, 4);
const Tvec4Array: Type = { kind: "Array", element: Tvec4f, length: "runtime" };
void Tu32;

const QUAD_PAD_EM = 0.05;

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
      a_localPos:    V2f;   // (uv ∈ {0,1}^2)
      a_instOffset:  V2f;   // (centerX, baseY)
      a_instTri:     V2f;   // (triFirst, triCount) as floats
      a_instBboxMin: V2f;
      a_instBboxMax: V2f;
    }): {
      gl_Position: V4f;
      v_world:     V2f;
      v_tri:       V2f;
      v_inst:      V2f;
      v_sx:        f32;
    } {
      const pPlus  = ProjTrafo.mul(ViewTrafo.mul(ModelTrafo.mul(new V4f( 1.0, 0.0, 0.0, 1.0))));
      const pMinus = ProjTrafo.mul(ViewTrafo.mul(ModelTrafo.mul(new V4f(-1.0, 0.0, 0.0, 1.0))));
      const sx = (pPlus.x / pPlus.w) < (pMinus.x / pMinus.w) ? -1.0 : 1.0;
      const local = new V2f(
        (input.a_instBboxMin.x + input.a_localPos.x * (input.a_instBboxMax.x - input.a_instBboxMin.x)) * sx,
        input.a_instBboxMin.y + input.a_localPos.y * (input.a_instBboxMax.y - input.a_instBboxMin.y),
      );
      const text = new V4f(
        input.a_instOffset.x * sx + local.x,
        input.a_instOffset.y + local.y,
        0.0, 1.0,
      );
      const clip = ProjTrafo.mul(ViewTrafo.mul(ModelTrafo.mul(text)));
      return {
        gl_Position: clip,
        v_world:     new V2f(local.x, local.y),
        v_tri:       input.a_instTri,
        v_inst:      input.a_instOffset,
        v_sx:        sx,
      };
    }

    function fsMain(input: { v_world: V2f; v_tri: V2f; v_inst: V2f; v_sx: f32 }): { outColor: V4f } {
      const triFirst = (input.v_tri.x + 0.5) as u32;
      const triCount = (input.v_tri.y + 0.5) as u32;
      const pp: V2f = new V2f(input.v_world.x, input.v_world.y);
      const sx = input.v_sx;
      
      // Compose the per-fragment Mvp once. Rational projection of
      // bezier control points then uses Mvp · (instance + P_em).
      const Mvp = ProjTrafo.mul(ViewTrafo.mul(ModelTrafo));
      // pp's pixel coords (origin at viewport bottom-left, NDC→pixel).
      const ppText = new V4f(
        input.v_inst.x * input.v_sx + pp.x,
        input.v_inst.y + pp.y,
        0.0, 1.0,
      );
      const ppClip = Mvp.mul(ppText);
      const ppPxX = (ppClip.x / ppClip.w + 1.0) * 0.5 * Viewport.x;
      const ppPxY = (ppClip.y / ppClip.w + 1.0) * 0.5 * Viewport.y;

      // DEBUG VISUALIZATION:
      //   insideFill = 1.0 when pp is inside any bezier triangle on
      //                the fill side OR inside any interior triangle.
      //                In that case we render solid blue ignoring the
      //                distance ramp.
      //   else minDistPx = pixel distance to nearest bezier curve.
      var minDistPx: f32 = 1.0e10;
      var insideFill: f32 = 0.0;
      // Pixel-space search radius. Triangles whose closest point
      // lands further than this away in pixels can't contribute to
      // the visible AA ramp.
      const searchPx = max(AaWidthPx, 1.0) * 4.0;
      // Helper: project a glyph-em coord to pixel coords via Mvp.
      // Reuses the same instance offset and sx-flip as the VS.
      // (No way to factor this into a real function in the DSL —
      // each call site inlines.)
      // toPxX(em) = (Mvp · vec4(sx*(inst.x + em.x), inst.y + em.y, 0, 1)).x / .w * 0.5 * Viewport.x + 0.5*Viewport.x

      const ZERO_U: u32 = 0 as u32;
      const ONE_U:  u32 = 1 as u32;
      const TWO_U:  u32 = 2 as u32;
      const THREE_U:u32 = 3 as u32;
      const FOUR_U: u32 = 4 as u32;
      for (let i: u32 = ZERO_U; i < triCount; i = i + ONE_U) {
        const base = (triFirst + i) * FOUR_U;
        const a = tris[base];
        const b = tris[base + ONE_U];
        const c = tris[base + TWO_U];
        const v0 = new V2f(a.x, a.y);
        const v1 = new V2f(b.x, b.y);
        const v2 = new V2f(c.x, c.y);

        // 2D point-to-segment distance for each of 3 edges.
        // Edge 0: v0→v1.
        const e0x = v1.x - v0.x; const e0y = v1.y - v0.y;
        const p0x = pp.x - v0.x; const p0y = pp.y - v0.y;
        const t0 = clamp((p0x * e0x + p0y * e0y) / max(e0x*e0x + e0y*e0y, 1.0e-18), 0.0, 1.0);
        const cp0x = v0.x + t0 * e0x - pp.x;
        const cp0y = v0.y + t0 * e0y - pp.y;
        const dE0 = sqrt(cp0x*cp0x + cp0y*cp0y);
        // Edge 1: v1→v2.
        const e1x = v2.x - v1.x; const e1y = v2.y - v1.y;
        const p1x = pp.x - v1.x; const p1y = pp.y - v1.y;
        const t1 = clamp((p1x * e1x + p1y * e1y) / max(e1x*e1x + e1y*e1y, 1.0e-18), 0.0, 1.0);
        const cp1x = v1.x + t1 * e1x - pp.x;
        const cp1y = v1.y + t1 * e1y - pp.y;
        const dE1 = sqrt(cp1x*cp1x + cp1y*cp1y);
        // Edge 2: v2→v0.
        const e2x = v0.x - v2.x; const e2y = v0.y - v2.y;
        const p2x = pp.x - v2.x; const p2y = pp.y - v2.y;
        const t2 = clamp((p2x * e2x + p2y * e2y) / max(e2x*e2x + e2y*e2y, 1.0e-18), 0.0, 1.0);
        const cp2x = v2.x + t2 * e2x - pp.x;
        const cp2y = v2.y + t2 * e2y - pp.y;
        const dE2 = sqrt(cp2x*cp2x + cp2y*cp2y);
        const dEdgeMin = min(dE0, min(dE1, dE2));

        // Inside-test: barycentric coords ≥ 0.
        const T2 = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
        const invT2 = 1.0 / T2;
        const w1 = ((pp.x - v0.x) * (v2.y - v0.y) - (pp.y - v0.y) * (v2.x - v0.x)) * invT2;
        const w2 = ((v1.x - v0.x) * (pp.y - v0.y) - (v1.y - v0.y) * (pp.x - v0.x)) * invT2;
        const w0 = 1.0 - w1 - w2;
        const inside = step(0.0, w0) * step(0.0, w1) * step(0.0, w2);
        const d = tris[base + THREE_U];   // (klm0.z, klm1.z, klm2.z, kind)
        const kind = d.w;
        // Interior (kind=0): if pp is inside, solid fill. Else
        // compute pixel distance via projection of the closest point
        // on the closest edge through Mvp.
        if (kind < 0.5) {
          if (inside > 0.5) { insideFill = 1.0; continue; }
          // Closest point in world (em) on the closest of the 3 edges.
          // Pick the world point of whichever edge gave dEdgeMin.
          const cWx = (dE0 <= dEdgeMin) ? (v0.x + t0 * e0x) : ((dE1 <= dEdgeMin) ? (v1.x + t1 * e1x) : (v2.x + t2 * e2x));
          const cWy = (dE0 <= dEdgeMin) ? (v0.y + t0 * e0y) : ((dE1 <= dEdgeMin) ? (v1.y + t1 * e1y) : (v2.y + t2 * e2y));
          const cText = new V4f(input.v_inst.x * input.v_sx + cWx, input.v_inst.y + cWy, 0.0, 1.0);
          const cClip = Mvp.mul(cText);
          const cPxX = (cClip.x / cClip.w + 1.0) * 0.5 * Viewport.x;
          const cPxY = (cClip.y / cClip.w + 1.0) * 0.5 * Viewport.y;
          const dpx = cPxX - ppPxX;
          const dpy = cPxY - ppPxY;
          const distPx = sqrt(dpx * dpx + dpy * dpy);
          if (distPx < searchPx) {
            minDistPx = min(minDistPx, distPx);
          }
          continue;
        }
        // Curve (kind=1 or 2). Closest-point on the rational quadratic
        // bezier in PIXEL space via Newton iteration.
        //
        // Project the 3 control points to clip space ONCE per
        // triangle. The bezier in clip space is
        //   N(t) = (1−t)²·clip0 + 2(1−t)t·clip1 + t²·clip2
        // and the projected pixel position is
        //   P(t) = (N(t).xy / N(t).w + 1) · 0.5 · Viewport
        // Distance² to pp_pixel: D(t) = |P(t) − pp_pixel|².
        // We solve D'(t) = 0 ↔ (P(t)−pp) · P'(t) = 0 with Newton
        // starting from 5 evenly-spaced seeds in [0, 1]; F'(t) is
        // approximated by |P'(t)|² (drops the 2nd-order term, which
        // is small near the minimum). Final distance is min over all
        // converged seeds (each clamped to [0, 1]).
        if (kind > 1.5) { continue; }   // arcs not handled here
        // Coarse cull: project parent centroid; reject if obviously far.
        const cenX = (v0.x + v1.x + v2.x) * (1.0 / 3.0);
        const cenY = (v0.y + v1.y + v2.y) * (1.0 / 3.0);
        const cenText = new V4f(input.v_inst.x * input.v_sx + cenX, input.v_inst.y + cenY, 0.0, 1.0);
        const cenClip = Mvp.mul(cenText);
        const cenPxX = (cenClip.x / cenClip.w + 1.0) * 0.5 * Viewport.x;
        const cenPxY = (cenClip.y / cenClip.w + 1.0) * 0.5 * Viewport.y;
        const cenDx = cenPxX - ppPxX;
        const cenDy = cenPxY - ppPxY;
        if (sqrt(cenDx * cenDx + cenDy * cenDy) > searchPx + 200.0) { continue; }

        // Project P0/P1/P2 (in canonical start/control/end order) once.
        const cT0 = new V4f(input.v_inst.x * input.v_sx + v0.x, input.v_inst.y + v0.y, 0.0, 1.0);
        const cT1 = new V4f(input.v_inst.x * input.v_sx + v1.x, input.v_inst.y + v1.y, 0.0, 1.0);
        const cT2 = new V4f(input.v_inst.x * input.v_sx + v2.x, input.v_inst.y + v2.y, 0.0, 1.0);
        const cC0 = Mvp.mul(cT0);
        const cC1 = Mvp.mul(cT1);
        const cC2 = Mvp.mul(cT2);

        var bestPx2: f32 = 1.0e20;
        const FIVE_U: u32 = 5 as u32;
        const ITER_U: u32 = 8 as u32;
        for (let s: u32 = ZERO_U; s < FIVE_U; s = s + ONE_U) {
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
            // P(t) in pixels.
            const Bx = Nx * invW;
            const By = Ny * invW;
            const Px = (Bx + 1.0) * 0.5 * Viewport.x;
            const Py = (By + 1.0) * 0.5 * Viewport.y;
            // P'(t) = (N'·W − N·W') / W², then scaled to pixel space.
            const dBx = (dNx * Wv - Nx * dWv) * invW * invW;
            const dBy = (dNy * Wv - Ny * dWv) * invW * invW;
            const dPx = dBx * 0.5 * Viewport.x;
            const dPy = dBy * 0.5 * Viewport.y;
            // F(t) = (P − pp) · P'(t).  Newton step with F'(t) ≈ |P'|².
            const rdx = Px - ppPxX;
            const rdy = Py - ppPxY;
            const Fv = rdx * dPx + rdy * dPy;
            const Fp = dPx * dPx + dPy * dPy;
            const dt = Fv / max(Fp, 1.0e-9);
            t = clamp(t - dt, 0.0, 1.0);
          }
          // Final evaluation at converged t.
          const oneT = 1.0 - t;
          const aw = oneT * oneT;
          const bw = 2.0 * oneT * t;
          const cw = t * t;
          const Nx = aw * cC0.x + bw * cC1.x + cw * cC2.x;
          const Ny = aw * cC0.y + bw * cC1.y + cw * cC2.y;
          const Wv = aw * cC0.w + bw * cC1.w + cw * cC2.w;
          const invW = 1.0 / max(Wv, 1.0e-9);
          const Px = (Nx * invW + 1.0) * 0.5 * Viewport.x;
          const Py = (Ny * invW + 1.0) * 0.5 * Viewport.y;
          const rdx = Px - ppPxX;
          const rdy = Py - ppPxY;
          bestPx2 = min(bestPx2, rdx * rdx + rdy * rdy);
        }
        // Also consider true endpoints t=0 (= v0 = start) and t=1
        // (= v2 = end) directly — these are already projected in cC0
        // and cC2 — to guarantee endpoint wins at junctions.
        const endP0x = (cC0.x / cC0.w + 1.0) * 0.5 * Viewport.x;
        const endP0y = (cC0.y / cC0.w + 1.0) * 0.5 * Viewport.y;
        const endP2x = (cC2.x / cC2.w + 1.0) * 0.5 * Viewport.x;
        const endP2y = (cC2.y / cC2.w + 1.0) * 0.5 * Viewport.y;
        const endD0 = (endP0x - ppPxX)*(endP0x - ppPxX) + (endP0y - ppPxY)*(endP0y - ppPxY);
        const endD2 = (endP2x - ppPxX)*(endP2x - ppPxX) + (endP2y - ppPxY)*(endP2y - ppPxY);
        bestPx2 = min(bestPx2, min(endD0, endD2));
        const distCurvePx_dir = sqrt(bestPx2);
        // Inside/outside via bezier implicit f at pp (klm interpolated):
        //   f<0 means pp is on the FILL side of the bezier (with the
        //   triangulator's m-flip, this works for both inward and
        //   outward curves).
        //   f>0 means pp is on the halo side.
        // Reuse the w0/w1/w2 already computed above.
        const k = w0 * a.z + w1 * b.z + w2 * c.z;
        const l = w0 * a.w + w1 * b.w + w2 * c.w;
        const mAt = w0 * d.x + w1 * d.y + w2 * d.z;
        const fAt = (k * k - l) * mAt;
        // Inside this bezier triangle's parent AND on the fill side
        // (fAt ≤ 0): the whole region is solid fill (lens for outward,
        // control-side for inward). Mark as fill, no distance gradient.
        if ((inside > 0.5) && (fAt <= 0.0)) {
          insideFill = 1.0;
          continue;
        }
        // Otherwise: distance to bezier curve in PIXELS (direction-
        // aware via inverse Jacobian computed from the offset vector).
        if (distCurvePx_dir < abs(minDistPx)) {
          minDistPx = distCurvePx_dir;
        }
      }

      // Alpha gradient:
      //   insideFill = 1            → α = 1 (solid fill).
      //   minDistPx ≤ AaWidthPx/2   → α ramps 1 → 0 over AaWidthPx px,
      //                                centred such that α=0.5 at
      //                                distance = AaWidthPx/2 from the
      //                                bezier (matches the rasterised
      //                                fwidth path).
      //   beyond                    → α = 0 → discard.
      const aaW = max(AaWidthPx, 1.0e-3);
      // dist 0 (on the bezier) → α = 1, dist aaW → α = 0. Bezier is
      // the visible fill boundary, ramp extends into halo only.
      const alphaHalo = clamp(1.0 - minDistPx / aaW, 0.0, 1.0);
      const alpha = mix(alphaHalo, 1.0, insideFill);
      if (alpha <= 0.0) discard;
      const aa_ = alpha * PathColor.w;
      return { outColor: new V4f(PathColor.x * aa_, PathColor.y * aa_, PathColor.z * aa_, aa_) };
    }
  `;

  const entries: EntryRequest[] = [
    {
      name: "vsMain", stage: "vertex",
      inputs: [
        { name: "a_localPos",    type: Tvec2f, semantic: "Position",   decorations: [{ kind: "Location", value: 0 }] },
        { name: "a_instOffset",  type: Tvec2f, semantic: "InstOffset", decorations: [{ kind: "Location", value: 1 }] },
        { name: "a_instTri",     type: Tvec2f, semantic: "InstTri",    decorations: [{ kind: "Location", value: 2 }] },
        { name: "a_instBboxMin", type: Tvec2f, semantic: "InstBboxMin",decorations: [{ kind: "Location", value: 3 }] },
        { name: "a_instBboxMax", type: Tvec2f, semantic: "InstBboxMax",decorations: [{ kind: "Location", value: 4 }] },
      ],
      outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "v_world",     type: Tvec2f, semantic: "World",    decorations: [{ kind: "Location", value: 0 }] },
        { name: "v_tri",       type: Tvec2f, semantic: "Tri",      decorations: [{ kind: "Location", value: 1 }] },
        { name: "v_inst",      type: Tvec2f, semantic: "Inst",     decorations: [{ kind: "Location", value: 2 }] },
        { name: "v_sx",        type: Tf32,   semantic: "Sx",       decorations: [{ kind: "Location", value: 3 }] },
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

  const quadVerts = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
  const quadIndices = new Uint32Array([0, 1, 2, 0, 2, 3]);
  const quadVertBuf = IBuffer.fromHost(quadVerts);
  const quadIdxBuf  = IBuffer.fromHost(quadIndices);

  type Inst = { cx: number; by: number; tf: number; tc: number; xmin: number; ymin: number; xmax: number; ymax: number };
  const instances: Inst[] = [];
  for (const g of layout.glyphs) {
    const rec = cache.get(g.codepoint);
    if (rec.empty || rec.triCount === 0) continue;
    const cx = g.x * emScale + rec.advance * 0.5 - totalAdvance * 0.5;
    const by = g.y * emScale;
    const xmin = rec.sdfBbox.x0 - QUAD_PAD_EM;
    const ymin = rec.sdfBbox.y0 - QUAD_PAD_EM;
    const xmax = rec.sdfBbox.x1 + QUAD_PAD_EM;
    const ymax = rec.sdfBbox.y1 + QUAD_PAD_EM;
    instances.push({
      cx, by,
      tf: rec.triFirst, tc: rec.triCount,
      xmin, ymin, xmax, ymax,
    });
  }

  const instCount = instances.length;
  const instArr = new Float32Array(instCount * 8);
  for (let i = 0; i < instCount; i++) {
    const it = instances[i]!;
    instArr[i*8 + 0] = it.cx;     instArr[i*8 + 1] = it.by;
    instArr[i*8 + 2] = it.tf;     instArr[i*8 + 3] = it.tc;
    instArr[i*8 + 4] = it.xmin;   instArr[i*8 + 5] = it.ymin;
    instArr[i*8 + 6] = it.xmax;   instArr[i*8 + 7] = it.ymax;
  }
  const instBuf = IBuffer.fromHost(instArr);

  const vertexAttrs = HashMap.empty<string, aval<BufferView>>()
    .add("a_localPos", AVal.constant<BufferView>({
      buffer: quadVertBuf, offset: 0, count: 4, stride: 8, format: "float32x2",
    }));
  const instanceAttrs = HashMap.empty<string, aval<BufferView>>()
    .add("a_instOffset", AVal.constant<BufferView>({
      buffer: instBuf, offset: 0, count: instCount, stride: 32, format: "float32x2",
    }))
    .add("a_instTri", AVal.constant<BufferView>({
      buffer: instBuf, offset: 8, count: instCount, stride: 32, format: "float32x2",
    }))
    .add("a_instBboxMin", AVal.constant<BufferView>({
      buffer: instBuf, offset: 16, count: instCount, stride: 32, format: "float32x2",
    }))
    .add("a_instBboxMax", AVal.constant<BufferView>({
      buffer: instBuf, offset: 24, count: instCount, stride: 32, format: "float32x2",
    }));
  const indexBV: BufferView = {
    buffer: quadIdxBuf, offset: 0, count: 6, stride: 4, format: "uint32",
  };
  const drawCall: DrawCall = {
    kind: "indexed",
    indexCount:    6,
    instanceCount: instCount,
    firstIndex:    0,
    baseVertex:    0,
    firstInstance: 0,
  };

  const triPacked = cache.trianglePackedBuffer();
  const triBuf = IBuffer.fromHost(triPacked);
  void TRI_FLOATS_PER_TRI;
  const storageBuffers = HashMap.empty<string, aval<IBuffer>>()
    .add("tris", AVal.constant<IBuffer>(triBuf));

  const effect = buildSdfTextEffect();
  const viewportV2: aval<V2f> = ambViewport.map(
    (vp) => new V2f(vp.width, vp.height),
  );

  const leafNode = sgVNode(Sg.leaf({
    vertexAttributes: vertexAttrs,
    instanceAttributes: instanceAttrs,
    indices: AVal.constant<BufferView>(indexBV),
    drawCall: AVal.constant<DrawCall>(drawCall),
    storageBuffers,
  }));

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
      PathColor: color,
      Viewport: viewportV2,
      AaWidthPx: aaWidth,
    },
    CullMode: "none",
    BlendMode: alphaOverBlendState(),
    DepthTest: "less-equal",
    ...(alignTrafo !== undefined ? { Trafo: alignTrafo } : {}),
    children: [leafNode],
  } as never);

  return scope === undefined
    ? tree
    : Sg({ ...(scope as object), children: [tree] } as never);
}
