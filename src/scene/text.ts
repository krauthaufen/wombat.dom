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
import { V2d, V3d, V4f, Trafo3d } from "@aardworx/wombat.base";
import {
  Font, GlyphCache, GLYPH_FLOATS_PER_VERTEX, layoutText,
} from "@aardworx/wombat.base/font";
import {
  IBuffer, type BufferView, type DrawCall,
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

function buildPathTextEffectAaNone(): Effect {
  if (pathTextEffectAaNone) return pathTextEffectAaNone;
  const source = `
    declare const ModelTrafo: M44f;
    declare const ViewTrafo:  M44f;
    declare const ProjTrafo:  M44f;
    declare const PathColor:  V4f;

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
      return {
        gl_Position: ProjTrafo.mul(ViewTrafo.mul(ModelTrafo.mul(text))),
        v_klmKind: input.a_klmKind,
      };
    }

    function fsMain(input: { v_klmKind: V4f }): { outColor: V4f } {
      // Loop-Blinn implicit test, m carries the inside/outside sign
      // for inward-bulging curves. Mirrors Aardvark's pathFragment.
      if (input.v_klmKind.w > 1.7) {
        if ((input.v_klmKind.x * input.v_klmKind.x + input.v_klmKind.y * input.v_klmKind.y - 1.0) * input.v_klmKind.z > 0.0) discard;
      } else if (input.v_klmKind.w > 0.7) {
        if ((input.v_klmKind.x * input.v_klmKind.x - input.v_klmKind.y) * input.v_klmKind.z > 0.0) discard;
      }
      return { outColor: PathColor };
    }
  `;

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
  externalTypes.set("ModelTrafo", TM44f);
  externalTypes.set("ViewTrafo",  TM44f);
  externalTypes.set("ProjTrafo",  TM44f);
  externalTypes.set("PathColor",  Tvec4f);

  const camUBO: ValueDef = {
    kind: "Uniform",
    uniforms: [
      { name: "ModelTrafo", type: TM44f,  group: 0, slot: 0, buffer: "Camera" },
      { name: "ViewTrafo",  type: TM44f,  group: 0, slot: 0, buffer: "Camera" },
      { name: "ProjTrafo",  type: TM44f,  group: 0, slot: 0, buffer: "Camera" },
      { name: "PathColor",  type: Tvec4f, group: 0, slot: 0, buffer: "Camera" },
    ],
  };

  const parsed = parseShader({ source, entries, externalTypes });
  const merged: Module = { ...parsed, values: [camUBO, ...parsed.values] };
  pathTextEffectAaNone = stage(merged);
  return pathTextEffectAaNone;
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
}

/** Build a self-contained `<Sg.Text/>` JSX element. */
export function SgText(
  props: SgTextProps & SgScopeProps,
): VNode {
  const {
    font, text, align = "left", aa = "none", kerning = true, Color,
    ...scope
  } = props;

  const cache = cacheFor(font);
  const layout = layoutText(font, text, { kerning });

  // Build, per unique glyph, an instance buffer of (centerX, baseY)
  // for every occurrence in the run. Geometry stays centred around
  // x = 0; alignment is a ModelTrafo translation applied below.
  const totalAdvance = layout.advance;
  const groups = new Map<number, { record: ReturnType<GlyphCache["get"]>; offsets: number[] }>();
  for (const g of layout.glyphs) {
    const record = cache.get(g.codepoint);
    if (record.empty) continue; // whitespace contributes nothing visible
    const centerX = g.x + record.advance * 0.5 - totalAdvance * 0.5;
    let entry = groups.get(g.codepoint);
    if (!entry) { entry = { record, offsets: [] }; groups.set(g.codepoint, entry); }
    entry.offsets.push(centerX);
    entry.offsets.push(g.y);
  }

  // Snapshot the cache's atlas. Shared across all per-glyph leaves
  // in this run: same vbo / ibo, sliced by per-leaf draw call
  // (firstIndex / baseVertex / indexCount).
  const interleaved = cache.vertexBuffer();
  const indices = cache.indexBuffer();
  const totalVerts = interleaved.length / GLYPH_FLOATS_PER_VERTEX;
  const positions = new Float32Array(totalVerts * 2);
  const klmKinds  = new Float32Array(totalVerts * 4);
  for (let i = 0; i < totalVerts; i++) {
    positions[i * 2 + 0] = interleaved[i * 6 + 0]!;
    positions[i * 2 + 1] = interleaved[i * 6 + 1]!;
    klmKinds[i * 4 + 0]  = interleaved[i * 6 + 2]!;
    klmKinds[i * 4 + 1]  = interleaved[i * 6 + 3]!;
    klmKinds[i * 4 + 2]  = interleaved[i * 6 + 4]!;
    klmKinds[i * 4 + 3]  = interleaved[i * 6 + 5]!;
  }
  const posBuf = IBuffer.fromHost(positions);
  const klmBuf = IBuffer.fromHost(klmKinds);
  const idxBuf = IBuffer.fromHost(indices);

  const vertexAttrs = HashMap.empty<string, aval<BufferView>>()
    .add("a_localPos", AVal.constant<BufferView>({
      buffer: posBuf, offset: 0, count: totalVerts, stride: 8, format: "float32x2",
    }))
    .add("a_klmKind", AVal.constant<BufferView>({
      buffer: klmBuf, offset: 0, count: totalVerts, stride: 16, format: "float32x4",
    }));
  const indexBV: BufferView = {
    buffer: idxBuf, offset: 0, count: indices.length, stride: 4, format: "uint32",
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
      indexCount:    record.indexCount,
      instanceCount: instCount,
      firstIndex:    record.firstIndex,
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

  const effect = aa === "none"
    ? buildPathTextEffectAaNone()
    : buildPathTextEffectAaNone(); // TODO: alpha-blending / sample-shading variants

  // Compose under one Sg scope: shader, color uniform, default
  // CullMode none (path triangles are math-CCW = framebuffer-CW),
  // user scope props on top.
  const tree = Sg({
    Shader: effect,
    Uniform: { PathColor: colorAval },
    CullMode: "none",
    ...(alignTrafo !== undefined ? { Trafo: alignTrafo } : {}),
    children: leafChildren,
  } as never);

  // Wrap the whole thing in the user-supplied scope (Trafo, OnDoubleTap,
  // PixelSnapRadius, …) so the alignment Trafo composes inside it.
  return scope === undefined
    ? tree
    : Sg({ ...(scope as object), children: [tree] } as never);
}
