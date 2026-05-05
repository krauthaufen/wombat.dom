// `<Sg.Text font={…} text="…" align="center" aa="none" Color={…}>`
//
// Text rendering follows Aardvark.Rendering.Text's conventions:
//
//   - Coordinate frame: `y = 0` is the baseline of the first row,
//     `+x` right, `+y` up. Font-size is 1 unit per em (so a single em
//     spans 1 world unit; scale via `Trafo` for other sizes).
//
//   - Geometry is always centred on `x = 0` regardless of `align`.
//     Alignment is a translation absorbed into `ModelTrafo`:
//       left   → +totalAdvance/2 (x = 0 ends up at the left edge)
//       center → 0                (x = 0 in the middle of the run)
//       right  → -totalAdvance/2 (x = 0 ends up at the right edge)
//     This keeps the flip-pivot at `x = 0` unconditionally.
//
//   - Glyphs are tessellated once per code-point per Font (via the
//     shared `GlyphCache`). Each unique glyph in a run becomes ONE
//     draw call; each occurrence becomes one INSTANCE of that draw.
//     The instance buffer carries `(centerX, baseY)` — where this
//     occurrence's glyph-centre sits in text-frame.
//
//   - The vertex shader detects a back-facing view by projecting two
//     text-local probes (`±x` on the baseline) and comparing their
//     screen-x. When the text is viewed from behind, both per-instance
//     `offsetX` and per-vertex `localX` are negated. Because both
//     frames are centred around 0, the double mirror is symmetric and
//     the text reads correctly from either side without any CPU
//     bookkeeping.
//
// `aa` controls the anti-aliasing pipeline:
//   "none"             — current behaviour, hard pixel edges.
//   "alpha-blending"   — outline-ribbon expansion + analytic alpha (TODO).
//   "sample-shading"   — per-sample fragment frequency on a 4×-MSAA
//                        target (requires the parent `RenderControl`
//                        to opt into MSAA, TODO).

import { AVal, HashMap, type aval } from "@aardworx/wombat.adaptive";
import { V2d, V2f, V3d, V4f, Trafo3d } from "@aardworx/wombat.base";
import {
  Font, GlyphCache, GLYPH_FLOATS_PER_VERTEX, layoutText,
} from "@aardworx/wombat.base/font";
import {
  IBuffer, type BufferView, type DrawCall, type BlendState,
} from "@aardworx/wombat.rendering/core";
import type { Effect } from "@aardworx/wombat.shader";
import { stage } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import {
  Mat, Tf32, Vec, type Type, type ValueDef, type Module,
} from "@aardworx/wombat.shader/ir";

import type { VNode } from "../vnode.js";
import { Sg } from "./constructors.js";
import { sgVNode } from "./sgVNode.js";
import type { SgScopeProps, SgNamespace } from "./constructors.js";
import { viewport as ambViewport } from "./ambient.js";
import { buildSdfTextScene } from "./text-sdf.js";

// Module augmentation so callers can write `<Sg.Text .../>` and the
// TypeScript types know about it. The runtime attachment lives in
// `scene/index.ts` to avoid the constructors.ts ↔ text.ts cycle.
declare module "./constructors.js" {
  interface SgNamespace {
    Text: (props: SgTextProps & SgScopeProps) => VNode;
  }
}

// ─────────────────────────────────────────────────────────────────
// Per-Font cache memo
// ─────────────────────────────────────────────────────────────────

const cacheByFont = new WeakMap<Font, GlyphCache>();
function cacheFor(font: Font): GlyphCache {
  let c = cacheByFont.get(font);
  if (!c) { c = new GlyphCache(font); cacheByFont.set(font, c); }
  return c;
}

// ─────────────────────────────────────────────────────────────────
// Loop-Blinn surface effect, with auto-flip + per-instance offset
// ─────────────────────────────────────────────────────────────────

const Tvec2f: Type = Vec(Tf32, 2);
const Tvec4f: Type = Vec(Tf32, 4);
const TM44f:  Type = Mat(Tf32, 4, 4);

let pathTextEffectAaNone: Effect | undefined;
let pathTextEffectAaAlphaBlending: Effect | undefined;
let pathTextEffectWireframe: Effect | undefined;

// Vertex shader source factory. `expandRibbon` = whether to apply
// clip-space expansion for `kind = 3` (line-ribbon) outer vertices.
// For aa="none" the expansion is skipped, so ribbon outer vertices
// collapse to their inner pair and the GPU rasterises them as
// zero-area triangles (no visible contribution, no overdraw).
function vsSource(expandRibbon: boolean): string {
  const expansion = expandRibbon ? `
    // Line-ribbon expansion: for kind = 3 vertices, the klm slot
    // carries (outwardX, outwardY, isOuter). The polygon edge sits
    // at the CENTRE of the AA ramp — inner verts (isOuter=0) get
    // pushed INWARD by AaWidthPx/2 and outer verts (isOuter=1) get
    // pushed OUTWARD by AaWidthPx/2, so the linear m=isOuter ramp
    // (1→0 in the FS) makes α=0.5 land exactly on the polygon edge.
    if (input.a_klmKind.w > 2.5) {
      const outX  = input.a_klmKind.x * sx;
      const outY  = input.a_klmKind.y;
      const isOut = input.a_klmKind.z;
      const outClip = ProjTrafo.mul(ViewTrafo.mul(ModelTrafo.mul(new V4f(outX, outY, 0.0, 0.0))));
      const outNdc = new V2f(outClip.x, outClip.y).div(max(clip.w, 1e-8));
      const outPx  = new V2f(outNdc.x * Viewport.x, outNdc.y * Viewport.y);
      const len    = max(outPx.length(), 1e-8);
      const stepPx = new V2f(outPx.x / len, outPx.y / len);
      // Map isOuter ∈ {0, 1} to signedHalf ∈ {-0.5, +0.5}.
      const signedHalf = isOut - 0.5;
      const stepNdc = new V2f(
        stepPx.x / Viewport.x * 2.0 * AaWidthPx,
        stepPx.y / Viewport.y * 2.0 * AaWidthPx,
      );
      clip = new V4f(
        clip.x + stepNdc.x * clip.w * signedHalf,
        clip.y + stepNdc.y * clip.w * signedHalf,
        clip.z, clip.w,
      );
    }
` : "";
  return `
    declare const ModelTrafo:    M44f;
    declare const ViewTrafo:     M44f;
    declare const ProjTrafo:     M44f;
    declare const PathColor:     V4f;
    declare const Viewport:      V2f;
    /** Ribbon expansion width in framebuffer pixels. Production
     *  rendering uses 1; bump to 5–10 to debug AA gradient direction
     *  / shape. */
    declare const AaWidthPx: f32;

    function vsMain(input: {
      a_localPos:   V2f;
      a_klmKind:    V4f;
      a_instOffset: V2f;
    }): { gl_Position: V4f; v_klmKind: V4f } {
      // Flip detection: project ±x baseline probes through MVP and
      // compare their screen-x. If +x lands LEFT of -x in screen
      // space, the text is viewed from behind → mirror around x=0.
      const pPlus  = ProjTrafo.mul(ViewTrafo.mul(ModelTrafo.mul(new V4f( 1.0, 0.0, 0.0, 1.0))));
      const pMinus = ProjTrafo.mul(ViewTrafo.mul(ModelTrafo.mul(new V4f(-1.0, 0.0, 0.0, 1.0))));
      const sx = (pPlus.x / pPlus.w) < (pMinus.x / pMinus.w) ? -1.0 : 1.0;
      // Per-instance offset (glyph centre in text-frame) and per-
      // vertex local position (glyph-centred coords) both mirror
      // around 0 with the same sign.
      const ofsX = input.a_instOffset.x * sx;
      const locX = input.a_localPos.x * sx;
      const text = new V4f(ofsX + locX, input.a_instOffset.y + input.a_localPos.y, 0.0, 1.0);
      let clip = ProjTrafo.mul(ViewTrafo.mul(ModelTrafo.mul(text)));
${expansion}
      return {
        gl_Position: clip,
        v_klmKind: input.a_klmKind,
      };
    }
  `;
}

function buildPathTextEffectAaNone(): Effect {
  if (pathTextEffectAaNone) return pathTextEffectAaNone;
  const source = `${vsSource(false)}

    function fsMain(input: { v_klmKind: V4f }): { outColor: V4f } {
      // Loop-Blinn implicit test, m carries the inside/outside sign
      // for inward-bulging curves. Mirrors Aardvark's pathFragment.
      // kind=3 (line ribbons) collapse to zero-area triangles when
      // expansion is off — the FS won't run for them.
      if (input.v_klmKind.w > 1.7 && input.v_klmKind.w < 2.5) {
        if ((input.v_klmKind.x * input.v_klmKind.x + input.v_klmKind.y * input.v_klmKind.y - 1.0) * input.v_klmKind.z > 0.0) discard;
      } else if (input.v_klmKind.w > 0.7 && input.v_klmKind.w < 1.5) {
        if ((input.v_klmKind.x * input.v_klmKind.x - input.v_klmKind.y) * input.v_klmKind.z > 0.0) discard;
      }
      return { outColor: PathColor };
    }
  `;

  pathTextEffectAaNone = compilePathTextEffect(source);
  return pathTextEffectAaNone;
}

/**
 * Analytic anti-aliasing via the screen-space gradient of the
 * Loop-Blinn implicit. For each fragment in a curve triangle:
 *
 *   f(k,l,m) = (k² − l) · m       (bezier2)
 *   f(k,l,m) = (k² + l² − 1) · m  (arc)
 *
 * The screen-space gradient `fwidth(f)` ≈ how much `f` changes
 * across one pixel. Signed distance in pixels ≈ `f / fwidth(f)`.
 * Alpha = `clamp(0.5 − sd, 0, 1)` gives a smooth 1px ramp at the
 * curve boundary (alpha=0.5 exactly on the curve, fully filled
 * one pixel inside, fully transparent one pixel outside).
 *
 * Caveats / TODO for v1:
 *
 *   - Line edges of the polygon are NOT yet anti-aliased — they
 *     still have hard pixel boundaries. Adding line ribbons
 *     requires tessellator changes; tracked separately.
 *
 *   - On the chord-side of a curve, AA only fades pixels within
 *     the curve triangle's coverage. Pixels just inside the
 *     polygon (covered by flat-fill, not the curve triangle)
 *     stay opaque — so the fade is only on the bulge side. This
 *     is acceptable for thin glyph strokes where both sides of
 *     the stroke are curves, but visible on isolated boundaries.
 */
function buildPathTextEffectAaAlphaBlending(): Effect {
  if (pathTextEffectAaAlphaBlending) return pathTextEffectAaAlphaBlending;
  const source = `${vsSource(true)}

    function fsMain(input: { v_klmKind: V4f }): { outColor: V4f } {
      // Take screen-space derivatives of the BARE interpolated
      // attributes (k, l, m) using the implementation-defined
      // \`dFdx\`/\`dFdy\`. The explicit \`Fine\` variant (dpdxFine)
      // wasn't accepted by WebKit's WGSL→MSL backend (curves came
      // back alpha=0 on iOS Safari) — plain \`dpdx\` works there
      // (verified by the diagnostic-shader pass that visualised
      // |dpdx(k)| as a yellow halo on every WebKit build we tried).
      //
      //   bez2: f = (k²-l)·m
      //         ∂f = (2k·∂k - ∂l)·m + (k²-l)·∂m
      //   arc:  f = (k²+l²-1)·m
      //         ∂f = (2k·∂k + 2l·∂l)·m + (k²+l²-1)·∂m
      const k = input.v_klmKind.x;
      const l = input.v_klmKind.y;
      const m = input.v_klmKind.z;
      const kind = input.v_klmKind.w;
      const dkx = dFdx(k); const dky = dFdy(k);
      const dlx = dFdx(l); const dly = dFdy(l);
      const dmx = dFdx(m); const dmy = dFdy(m);
      const fBez = (k * k - l) * m;
      const fArc = (k * k + l * l - 1.0) * m;
      const dfBezX = (2.0 * k * dkx - dlx) * m + (k * k - l) * dmx;
      const dfBezY = (2.0 * k * dky - dly) * m + (k * k - l) * dmy;
      const dfArcX = (2.0 * k * dkx + 2.0 * l * dlx) * m + (k * k + l * l - 1.0) * dmx;
      const dfArcY = (2.0 * k * dky + 2.0 * l * dly) * m + (k * k + l * l - 1.0) * dmy;
      const mBez    = step(0.7, kind) - step(1.5, kind);
      const mArc    = step(1.7, kind) - step(2.5, kind);
      const mRibbon = step(2.5, kind) - step(3.5, kind);
      const mDist   = step(3.5, kind);   // kind=4: signed-distance halo
      // For kind=4 the linear distance field is klm.x directly,
      // and its screen-space gradient is fwidth(klm.x). We mix
      // that into the bezier f so the same 0.5-f/w AA recipe gives
      // the right ramp without a separate code path.
      const f   = fBez * mBez + fArc * mArc + k * mDist + (-1.0) * (1.0 - mBez - mArc - mRibbon - mDist);
      const dfX = dfBezX * mBez + dfArcX * mArc + dkx * mDist;
      const dfY = dfBezY * mBez + dfArcY * mArc + dky * mDist;
      const w   = sqrt(dfX * dfX + dfY * dfY) + 1e-6;
      // 1-pixel ramp scaled by AaWidthPx, centred on the curve
      // (α=0.5 exactly on f=0). Same uniform also drives the line-
      // ribbon geometric expansion in the VS and the halo distance-
      // field encoding, so all three share one width knob.
      const curveAlpha = clamp(0.5 - f / (w * AaWidthPx), 0.0, 1.0);
      const ribbonAlpha = clamp(1.0 - m, 0.0, 1.0);
      const alpha = curveAlpha * (1.0 - mRibbon) + ribbonAlpha * mRibbon;
      // Discard fully-transparent fragments BEFORE the blend stage.
      // The Loop-Blinn curve triangle covers (start, control, end);
      // pixels in the wedge BETWEEN the curve and the control vertex
      // get α=0 because they're outside the curve's filled half.
      // That wedge often extends into an adjacent glyph's polygon,
      // so blending α=0 over a neighbour's pixel WOULD preserve dst
      // mathematically — but discarding here also keeps depth-write
      // honest: the wedge area shouldn't claim any pixels at all.
      if (alpha <= 0.0) discard;
      // Premultiplied-alpha output (RGB pre-scaled by alpha).
      // iOS Safari WebGPU silently dropped curve fragments when we
      // used the conventional non-premultiplied form
      // \`new V4f(PathColor.x, PathColor.y, PathColor.z, alpha)\`,
      // even though the alpha calc was correct (visualising alpha
      // as RGB rendered fine). Referencing alpha in every channel
      // keeps WebKit's optimiser honest and matches the premultiplied
      // blend factors (src=ONE, dst=ONE_MINUS_SRC_ALPHA).
      const aa_ = alpha * PathColor.w;
      return { outColor: new V4f(PathColor.x * aa_, PathColor.y * aa_, PathColor.z * aa_, aa_) };
    }
  `;
  pathTextEffectAaAlphaBlending = compilePathTextEffect(source);
  return pathTextEffectAaAlphaBlending;
}

/**
 * Wireframe debug effect: no discard, no blend, no AA math. Each
 * fragment is coloured by its triangle's `kind`:
 *
 *   kind = 0 (interior)        → green
 *   kind = 1 (bezier2 / halo)  → red
 *   kind = 2 (arc)             → blue
 *   kind = 3 (line ribbon)     → yellow
 *
 * Used with `Mode = "line-list"` and a 3-line-per-triangle index
 * buffer to draw the raw mesh structure for diagnosis. Ribbon
 * verts still get expanded by the VS so collapsed (zero-area) lines
 * don't disappear in screen space.
 */
function buildPathTextEffectWireframe(): Effect {
  if (pathTextEffectWireframe) return pathTextEffectWireframe;
  const source = `${vsSource(true)}

    function fsMain(input: { v_klmKind: V4f }): { outColor: V4f } {
      const k    = input.v_klmKind.w;
      const m0   = step(-0.5, k) - step(0.5, k);     // 1 if kind=0
      const m1   = step(0.5,  k) - step(1.5, k);     // 1 if kind=1
      const m2   = step(1.5,  k) - step(2.5, k);     // 1 if kind=2
      const m3   = step(2.5,  k);                    // 1 if kind=3
      const r = 0.20 * m0 + 0.95 * m1 + 0.20 * m2 + 0.95 * m3;
      const g = 0.85 * m0 + 0.30 * m1 + 0.40 * m2 + 0.85 * m3;
      const b = 0.30 * m0 + 0.30 * m1 + 0.95 * m2 + 0.20 * m3;
      return { outColor: new V4f(r, g, b, 1.0) };
    }
  `;
  pathTextEffectWireframe = compilePathTextEffect(source);
  return pathTextEffectWireframe;
}

function compilePathTextEffect(source: string): Effect {
  const entries: EntryRequest[] = [
    {
      name: "vsMain", stage: "vertex",
      inputs: [
        { name: "a_localPos",   type: Tvec2f, semantic: "Position",  decorations: [{ kind: "Location", value: 0 }] },
        { name: "a_klmKind",    type: Tvec4f, semantic: "KLMKind",   decorations: [{ kind: "Location", value: 1 }] },
        { name: "a_instOffset", type: Tvec2f, semantic: "InstOffset",decorations: [{ kind: "Location", value: 2 }] },
      ],
      outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin",  value: "position" }] },
        { name: "v_klmKind",   type: Tvec4f, semantic: "KLMKind",  decorations: [{ kind: "Location", value: 0 }] },
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
  externalTypes.set("ModelTrafo",    TM44f);
  externalTypes.set("ViewTrafo",     TM44f);
  externalTypes.set("ProjTrafo",     TM44f);
  externalTypes.set("PathColor",     Tvec4f);
  externalTypes.set("Viewport",      Tvec2f);
  externalTypes.set("AaWidthPx", Tf32);

  const camUBO: ValueDef = {
    kind: "Uniform",
    uniforms: [
      { name: "ModelTrafo",    type: TM44f,  group: 0, slot: 0, buffer: "Camera" },
      { name: "ViewTrafo",     type: TM44f,  group: 0, slot: 0, buffer: "Camera" },
      { name: "ProjTrafo",     type: TM44f,  group: 0, slot: 0, buffer: "Camera" },
      { name: "PathColor",     type: Tvec4f, group: 0, slot: 0, buffer: "Camera" },
      { name: "Viewport",      type: Tvec2f, group: 0, slot: 0, buffer: "Camera" },
      { name: "AaWidthPx", type: Tf32,   group: 0, slot: 0, buffer: "Camera" },
    ],
  };

  const parsed = parseShader({ source, entries, externalTypes });
  const merged: Module = { ...parsed, values: [camUBO, ...parsed.values] };
  return stage(merged);
}

// Standard alpha-over blending state for the analytic-AA mode.
// Cached at module level — every `<Sg.Text aa="alpha-blending"/>`
// reuses the same BlendState instance.
let alphaOverBlend: BlendState | undefined;
function alphaOverBlendState(): BlendState {
  if (alphaOverBlend) return alphaOverBlend;
  alphaOverBlend = {
    color: {
      operation: AVal.constant<GPUBlendOperation>("add"),
      // PREMULTIPLIED-alpha source: shader outputs RGB * alpha
      // already, so source factor is ONE (don't multiply by alpha
      // again).
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

// ─────────────────────────────────────────────────────────────────
// SgText component
// ─────────────────────────────────────────────────────────────────

export type TextAlign = "left" | "center" | "right";
export type TextAa = "none" | "alpha-blending" | "sample-shading";

export interface SgTextProps {
  /** Parsed font (e.g. via `Font.load(url)`). */
  font: Font;
  /** The text to render. Codepoint-based shaping (no GSUB). */
  text: string;
  /** Horizontal alignment relative to the local origin. Default:
   *  `"left"` (origin at the left edge of the first glyph). */
  align?: TextAlign;
  /** Anti-aliasing mode. Default: `"none"`. */
  aa?: TextAa;
  /** Apply opentype.js KERN-table pairs between consecutive glyphs.
   *  Default: `true`. */
  kerning?: boolean;
  /** Fill colour, vec4 (rgba). Default: opaque white. */
  Color?: V4f | aval<V4f>;
  /** Width of the AA ramp in framebuffer pixels (only consulted when
   *  `aa === "alpha-blending"`). Drives BOTH the curve-AA ramp width
   *  and the line-ribbon geometric expansion — the polygon edge sits
   *  at the centre, ramp 1→0 over `aaWidthPx` pixels. Default: 1.
   *  Bump to 5–10 to debug AA gradient direction / shape. */
  aaWidthPx?: number | aval<number>;
  /** Render every triangle as a wireframe outline (three lines per
   *  triangle). Debug aid only. Default: `false`. */
  wireframe?: boolean;
}

/** Build a self-contained `<Sg.Text/>` JSX element. */
export function SgText(
  props: SgTextProps & SgScopeProps,
): VNode {
  const {
    font, text, align = "left", aa = "none", kerning = true, Color,
    aaWidthPx, wireframe = false,
    ...scope
  } = props;

  const cache = cacheFor(font);
  // Triangle-SDF alpha-blending path: per-pixel barycentric inside-
  // test against the cached triangulation, evaluating klm + kind
  // directly. Reuses the existing curve implicit; no segment Newton.
  if (aa === "alpha-blending" && !wireframe) {
    const colorAvalSdf: aval<V4f> = Color === undefined
      ? AVal.constant(new V4f(1, 1, 1, 1))
      : (Color instanceof V4f ? AVal.constant(Color) : Color);
    const aaWidthAvalSdf: aval<number> = aaWidthPx === undefined
      ? AVal.constant(1)
      : (typeof aaWidthPx === "number"
          ? AVal.constant(aaWidthPx)
          : aaWidthPx);
    return buildSdfTextScene({
      font, text, align, kerning,
      cache, color: colorAvalSdf, aaWidth: aaWidthAvalSdf,
      scope: scope as SgScopeProps,
    });
  }
  const layout = layoutText(font, text, { kerning });
  // GlyphCache stores em-scaled geometry (1 em = 1 world unit), so
  // layout positions (font units) need the same conversion before
  // they're used as per-instance offsets.
  const emScale = 1 / (font.unitsPerEm || 1);

  // Build, per unique glyph, an instance buffer of (centerX, baseY)
  // for every occurrence in the run. Geometry stays centred around
  // x = 0; alignment is a ModelTrafo translation applied below.
  const totalAdvance = layout.advance * emScale;
  const groups = new Map<number, { record: ReturnType<GlyphCache["get"]>; offsets: number[] }>();
  for (const g of layout.glyphs) {
    const record = cache.get(g.codepoint);
    if (record.empty) continue; // whitespace contributes nothing visible
    const centerX = g.x * emScale + record.advance * 0.5 - totalAdvance * 0.5;
    let entry = groups.get(g.codepoint);
    if (!entry) { entry = { record, offsets: [] }; groups.set(g.codepoint, entry); }
    entry.offsets.push(centerX);
    entry.offsets.push(g.y * emScale);
  }

  // Snapshot the cache's atlas. Shared across all per-glyph leaves
  // in this run: same vbo / ibo, sliced by per-leaf draw call
  // (firstIndex / baseVertex / indexCount).
  const interleaved = cache.vertexBuffer();
  const indices = cache.indexBuffer();
  const totalVerts = interleaved.length / GLYPH_FLOATS_PER_VERTEX;
  const positions = new Float32Array(totalVerts * 2);
  const klmKinds  = new Float32Array(totalVerts * 4);
  const STRIDE = GLYPH_FLOATS_PER_VERTEX;
  for (let i = 0; i < totalVerts; i++) {
    positions[i * 2 + 0] = interleaved[i * STRIDE + 0]!;
    positions[i * 2 + 1] = interleaved[i * STRIDE + 1]!;
    klmKinds[i * 4 + 0]  = interleaved[i * STRIDE + 2]!;
    klmKinds[i * 4 + 1]  = interleaved[i * STRIDE + 3]!;
    klmKinds[i * 4 + 2]  = interleaved[i * STRIDE + 4]!;
    klmKinds[i * 4 + 3]  = interleaved[i * STRIDE + 5]!;
  }
  const posBuf = IBuffer.fromHost(positions);
  const klmBuf = IBuffer.fromHost(klmKinds);
  // For wireframe, expand each triangle (3 indices) into 3 line
  // segments (6 indices: a,b, b,c, c,a). Per-glyph drawcall ranges
  // map firstIndex/indexCount × 2 to address the line-list buffer.
  const wireIndices = wireframe ? new Uint32Array(indices.length * 2) : undefined;
  if (wireIndices !== undefined) {
    for (let t = 0; t < indices.length; t += 3) {
      const a = indices[t]!, b = indices[t + 1]!, c = indices[t + 2]!;
      const w = t * 2;
      wireIndices[w + 0] = a; wireIndices[w + 1] = b;
      wireIndices[w + 2] = b; wireIndices[w + 3] = c;
      wireIndices[w + 4] = c; wireIndices[w + 5] = a;
    }
  }
  const idxBuf = IBuffer.fromHost(wireframe ? wireIndices! : indices);

  const vertexAttrs = HashMap.empty<string, aval<BufferView>>()
    .add("a_localPos", AVal.constant<BufferView>({
      buffer: posBuf, offset: 0, count: totalVerts, stride: 8, format: "float32x2",
    }))
    .add("a_klmKind", AVal.constant<BufferView>({
      buffer: klmBuf, offset: 0, count: totalVerts, stride: 16, format: "float32x4",
    }));
  const indexBV: BufferView = {
    buffer: idxBuf, offset: 0,
    count: wireframe ? wireIndices!.length : indices.length,
    stride: 4, format: "uint32",
  };

  // One `<Sg.Leaf>` per unique glyph in the run; all share the
  // atlas vbo / ibo and differ only in the per-instance offsets and
  // the draw range.
  const colorAval: aval<V4f> = Color === undefined
    ? AVal.constant(new V4f(1, 1, 1, 1))
    : (Color instanceof V4f ? AVal.constant(Color) : Color);

  const leafChildren: VNode[] = [];
  for (const [, { record, offsets }] of groups) {
    if (record.empty) continue;
    const instCount = offsets.length / 2;
    const instArr = new Float32Array(offsets);
    const instBuf = IBuffer.fromHost(instArr);
    const instAttrs = HashMap.empty<string, aval<BufferView>>()
      .add("a_instOffset", AVal.constant<BufferView>({
        buffer: instBuf, offset: 0, count: instCount, stride: 8, format: "float32x2",
      }));
    const draw: DrawCall = {
      kind: "indexed",
      indexCount:    wireframe ? record.indexCount * 2 : record.indexCount,
      instanceCount: instCount,
      firstIndex:    wireframe ? record.firstIndex * 2 : record.firstIndex,
      baseVertex:    record.baseVertex,
      firstInstance: 0,
    };
    leafChildren.push(
      sgVNode(Sg.leaf({
        vertexAttributes: vertexAttrs,
        instanceAttributes: instAttrs,
        indices: AVal.constant<BufferView>(indexBV),
        drawCall: AVal.constant<DrawCall>(draw),
      })),
    );
  }

  // Alignment: shift the centred-around-0 geometry by half-advance
  // so the local origin (x = 0) lands at the requested anchor.
  const alignDx
    = align === "left"  ? +totalAdvance * 0.5
    : align === "right" ? -totalAdvance * 0.5
    : 0;
  const alignTrafo = alignDx === 0
    ? undefined
    : Trafo3d.translation(new V3d(alignDx, 0, 0));

  const effect
    = wireframe                 ? buildPathTextEffectWireframe()
    : aa === "alpha-blending"   ? buildPathTextEffectAaAlphaBlending()
    : /* sample-shading uses the same shader as none; the AA work
         happens via per-sample frequency at pipeline-state level. */
      buildPathTextEffectAaNone();

  // Compose under one Sg scope: shader, color uniform, default
  // CullMode none (path triangles are math-CCW = framebuffer-CW),
  // alpha-blend BlendMode for the alpha-blending AA mode, user
  // scope props on top.
  const viewportV2: aval<V2f> = ambViewport.map(
    (vp) => new V2f(vp.width, vp.height),
  );
  const aaWidthAval: aval<number> = aaWidthPx === undefined
    ? AVal.constant(1)
    : (typeof aaWidthPx === "number"
        ? AVal.constant(aaWidthPx)
        : aaWidthPx);

  const tree = Sg({
    Shader: effect,
    Uniform: {
      PathColor:     colorAval,
      Viewport:      viewportV2,
      AaWidthPx: aaWidthAval,
    },
    CullMode: "none",
    // alpha-blending: depth-test=less-equal so curve triangles drawn
    // after flat triangles at the same z don't get rejected; depth
    // write stays on so opaque paths still occlude geometry behind.
    // alpha-blending: depth-test=less-equal lets curve and flat
    // triangles inside a single glyph (and adjacent glyphs at the
    // same z) compose without z-fighting; depth-write stays ON so
    // text properly occludes geometry behind it. Fully-transparent
    // fragments (α=0) are discarded in the FS so they don't write
    // depth either — that prevents a curve triangle's wedge-outside-
    // curve area (α=0 by construction) from "hole-punching" through
    // an overlapping neighbour glyph.
    ...(wireframe ? {
      Mode: "line-list" as const,
    } : aa === "alpha-blending" ? {
      BlendMode: alphaOverBlendState(),
      DepthTest: "less-equal" as const,
    } : {}),
    ...(alignTrafo !== undefined ? { Trafo: alignTrafo } : {}),
    children: leafChildren,
  } as never);

  // Wrap the whole thing in the user-supplied scope (Trafo, OnDoubleTap,
  // PixelSnapRadius, …) so the alignment Trafo composes inside it.
  return scope === undefined
    ? tree
    : Sg({ ...(scope as object), children: [tree] } as never);
}
