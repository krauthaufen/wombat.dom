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
      };
    }

    function fsMain(input: { v_world: V2f; v_tri: V2f }): { outColor: V4f } {
      const triFirst = (input.v_tri.x + 0.5) as u32;
      const triCount = (input.v_tri.y + 0.5) as u32;
      const pp: V2f = new V2f(input.v_world.x, input.v_world.y);
      const wpxx = dFdx(pp.x);
      const wpxy = dFdx(pp.y);
      const wpyx = dFdy(pp.x);
      const wpyy = dFdy(pp.y);
      const worldPerPx = sqrt(0.5 * (wpxx*wpxx + wpxy*wpxy + wpyx*wpyx + wpyy*wpyy)) + 1.0e-9;

      // DEBUG VISUALIZATION:
      //   insideFill = 1.0 when pp is inside any bezier triangle on
      //                the fill side OR inside any interior triangle.
      //                In that case we render solid blue ignoring the
      //                distance ramp.
      //   else minDistPx = pixel distance to nearest bezier curve.
      var minDistPx: f32 = 1.0e10;
      var insideFill: f32 = 0.0;
      const searchWorld = max(AaWidthPx, 1.0) * 4.0 * worldPerPx;

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
        const distSolidWorld = (1.0 - inside) * dEdgeMin;
        // Interior (kind=0): if pp is inside, it's solid fill.
        if (kind < 0.5) {
          if (inside > 0.5) { insideFill = 1.0; }
          else if (distSolidWorld < searchWorld) {
            minDistPx = min(minDistPx, distSolidWorld / worldPerPx);
          }
          continue;
        }
        // Curve (kind=1 or 2). Closest-point on the quadratic bezier:
        //   B(t) = P0 + t·L + t²·Q, with L = 2(P1−P0), Q = P0−2P1+P2.
        //   F(t) = (B(t) − p) · B'(t)
        //        = D·L + (2 D·Q + L·L) t + 3 (L·Q) t² + 2 |Q|² t³
        // is a cubic in t. Solve via the trig / Cardano formula for
        // real roots, clamp to [0, 1], also test endpoints.
        if (distSolidWorld > searchWorld) { continue; }
        if (kind > 1.5) { continue; }   // arcs not handled in debug viz
        const Lx = 2.0 * (v1.x - v0.x);
        const Ly = 2.0 * (v1.y - v0.y);
        const Qx = v0.x - 2.0 * v1.x + v2.x;
        const Qy = v0.y - 2.0 * v1.y + v2.y;
        const Dx = v0.x - pp.x;
        const Dy = v0.y - pp.y;
        const cubA = 2.0 * (Qx * Qx + Qy * Qy);
        const cubB = 3.0 * (Lx * Qx + Ly * Qy);
        const cubC = (Lx * Lx + Ly * Ly) + 2.0 * (Dx * Qx + Dy * Qy);
        const cubD = Dx * Lx + Dy * Ly;
        // Depressed cubic y³ + p·y + q = 0 via t = y − cubB/(3·cubA).
        const aSafe = sign(cubA) * max(abs(cubA), 1.0e-18);
        const bN = cubB / aSafe;
        const cN = cubC / aSafe;
        const dN = cubD / aSafe;
        const pCoef = cN - bN * bN / 3.0;
        const qCoef = 2.0 * bN * bN * bN / 27.0 - bN * cN / 3.0 + dN;
        const disc = qCoef * qCoef * 0.25 + pCoef * pCoef * pCoef / 27.0;
        const shift = -bN / 3.0;
        // Three candidate roots (some may be NaN — guarded by clamp + endpoint check).
        var r0: f32 = -1.0;
        var r1: f32 = -1.0;
        var r2: f32 = -1.0;
        const PI23 = 2.094395102;   // 2π/3
        const useTrig = step(disc, 0.0);
        // Trig form (3 real roots when disc <= 0):
        const negP3 = max(-pCoef / 3.0, 0.0);
        const m = 2.0 * sqrt(negP3);
        const cosArg = clamp(-qCoef * 0.5 / max(pow(negP3, 1.5), 1.0e-18), -1.0, 1.0);
        const phi = acos(cosArg) / 3.0;
        const tr0 = shift + m * cos(phi);
        const tr1 = shift + m * cos(phi - PI23);
        const tr2 = shift + m * cos(phi + PI23);
        // Cardano form (1 real root when disc > 0):
        const sd = sqrt(max(disc, 0.0));
        const u = sign(-qCoef * 0.5 + sd) * pow(abs(-qCoef * 0.5 + sd), 1.0/3.0);
        const v = sign(-qCoef * 0.5 - sd) * pow(abs(-qCoef * 0.5 - sd), 1.0/3.0);
        const tc0 = shift + u + v;
        r0 = mix(tc0, tr0, useTrig);
        r1 = mix(tc0, tr1, useTrig);   // duplicate when Cardano (only 1 real)
        r2 = mix(tc0, tr2, useTrig);
        // Test 5 candidates: endpoints t=0, t=1, plus the (up to 3)
        // roots clamped to [0, 1]. Manually unrolled — the shader DSL
        // doesn't accept WGSL array<f32, N>(…) constructors here.
        const u0 = 0.0;
        const u1 = 1.0;
        const u2 = clamp(r0, 0.0, 1.0);
        const u3 = clamp(r1, 0.0, 1.0);
        const u4 = clamp(r2, 0.0, 1.0);
        const Bt0x = v0.x + u0 * Lx + u0 * u0 * Qx; const Bt0y = v0.y + u0 * Ly + u0 * u0 * Qy;
        const Bt1x = v0.x + u1 * Lx + u1 * u1 * Qx; const Bt1y = v0.y + u1 * Ly + u1 * u1 * Qy;
        const Bt2x = v0.x + u2 * Lx + u2 * u2 * Qx; const Bt2y = v0.y + u2 * Ly + u2 * u2 * Qy;
        const Bt3x = v0.x + u3 * Lx + u3 * u3 * Qx; const Bt3y = v0.y + u3 * Ly + u3 * u3 * Qy;
        const Bt4x = v0.x + u4 * Lx + u4 * u4 * Qx; const Bt4y = v0.y + u4 * Ly + u4 * u4 * Qy;
        const cd0 = (Bt0x - pp.x)*(Bt0x - pp.x) + (Bt0y - pp.y)*(Bt0y - pp.y);
        const cd1 = (Bt1x - pp.x)*(Bt1x - pp.x) + (Bt1y - pp.y)*(Bt1y - pp.y);
        const cd2 = (Bt2x - pp.x)*(Bt2x - pp.x) + (Bt2y - pp.y)*(Bt2y - pp.y);
        const cd3 = (Bt3x - pp.x)*(Bt3x - pp.x) + (Bt3y - pp.y)*(Bt3y - pp.y);
        const cd4 = (Bt4x - pp.x)*(Bt4x - pp.x) + (Bt4y - pp.y)*(Bt4y - pp.y);
        const bestD2 = min(min(min(cd0, cd1), min(cd2, cd3)), cd4);
        const distCurveWorld = sqrt(bestD2);
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
        // Otherwise: distance to bezier curve, positive (halo side).
        const distCurvePx = distCurveWorld / worldPerPx;
        if (distCurvePx < abs(minDistPx)) {
          minDistPx = distCurvePx;
        }
      }

      // HSV mapping:
      //   insideFill = 1 → solid blue (deep fill).
      //   else minDistPx ∈ [0, radius] → green (0) → red (radius).
      const radiusPx = max(AaWidthPx, 1.0) * 4.0;
      const haloT = clamp(minDistPx / radiusPx, 0.0, 1.0);
      const hHalo = clamp(0.33 - haloT * 0.33, 0.0, 1.0);
      const hFill = 0.66;
      const h = mix(hHalo, hFill, insideFill);
      const h6 = h * 6.0;
      const ff = h6 - floor(h6);
      const hi = floor(h6);
      const r = (hi < 0.5) ? 1.0 : ((hi < 1.5) ? 1.0 - ff : ((hi < 2.5) ? 0.0 : ((hi < 3.5) ? 0.0 : ((hi < 4.5) ? ff : 1.0))));
      const g = (hi < 0.5) ? ff : ((hi < 1.5) ? 1.0 : ((hi < 2.5) ? 1.0 : ((hi < 3.5) ? 1.0 - ff : ((hi < 4.5) ? 0.0 : 0.0))));
      const bl = (hi < 0.5) ? 0.0 : ((hi < 1.5) ? 0.0 : ((hi < 2.5) ? ff : ((hi < 3.5) ? 1.0 : ((hi < 4.5) ? 1.0 : 1.0 - ff))));
      // Dark when no triangle was within the search radius AND not
      // inside any fill region.
      const beyond = step(searchWorld, minDistPx * worldPerPx) * (1.0 - insideFill);
      const rr = mix(r, 0.05, beyond);
      const gg = mix(g, 0.05, beyond);
      const bb = mix(bl, 0.10, beyond);
      return { outColor: new V4f(rr, gg, bb, 1.0) };
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
