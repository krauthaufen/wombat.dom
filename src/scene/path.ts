// `Sg.Path`: a filled 2D path, drawn exactly at every zoom.
//
// The path — line, quadratic, cubic and arc segments, or an SVG `d`
// string — is tessellated with wombat.base's planar-graph fill (the same
// Loop-Blinn geometry the text backend draws glyphs with): interior
// triangles plus one curve triangle per curved edge, whose fragment
// shader keeps exactly the inside of the curve. So arcs stay arcs at any
// distance, with no polyline to refine. A stroke of a line or an arc is
// itself a closed outline (outer arc, end, inner arc back, end), so a
// band such as a connection between two points is a path too.
//
// Coordinates are the path's own x and y in the local z = 0 plane (an
// SVG `d` is taken as written: its y points down only if the trafo above
// says so). An adaptive path rebuilds its geometry when it changes; the
// draw call follows the new size.

import { AVal, HashMap, type aval } from "@aardworx/wombat.adaptive";
import {
  V2f, V4f, FillRules, compileTessellation, tessellatePath, triangulateFilledFaces,
  VERTEX_BYTE_SIZE, type PathSegment,
} from "@aardworx/wombat.base";
import { pathFromSvgD } from "@aardworx/wombat.base/svg";
import {
  IBuffer, type BlendState, type BufferView, type DrawCall, ElementType,
} from "@aardworx/wombat.rendering/core";
import { effect, vertex, type Effect } from "@aardworx/wombat.shader";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { max } from "@aardworx/wombat.shader/types";
import { Sg } from "./constructors.js";
import type { SgNode } from "./sg.js";
import type { SgScopeProps } from "./constructors.js";
import type { TraversalState } from "./traversalState.js";
import { markHeapSafeDrawAVal, markHostBufferAVal } from "./hostBuffers.js";
import { pathFsAaAlpha, pathFsAaNone } from "./text.js";

// `<Sg.Path/>` on the Sg namespace (attached in index.ts, as Sg.Text).
declare module "./constructors.js" {
  interface SgNamespace {
    Path: (props: SgPathProps & SgScopeProps) => SgNode;
  }
}

/** A path: its segments, or an SVG path `d` string. */
export type PathInput = string | ReadonlyArray<PathSegment>;

export type PathFillRule = "nonzero" | "evenodd";

export interface SgPathProps {
  readonly path: PathInput | aval<PathInput>;
  /** Which regions are inside (default "nonzero", as SVG). */
  readonly fillRule?: PathFillRule;
  /** Fill colour (default white). */
  readonly Color?: V4f | aval<V4f>;
  /** "none": hard, exact edges. "alpha-blending": a one-pixel analytic
   *  ramp on every edge, blended over what is behind. */
  readonly aa?: "none" | "alpha-blending";
  /** Width of the anti-aliasing ramp in pixels (default 1). */
  readonly aaWidthPx?: number | aval<number>;
}

// ── geometry ───────────────────────────────────────────────────────────

interface PathGeometry {
  readonly positions: Float32Array;  // x, y per vertex
  readonly klmKinds: Float32Array;   // k, l, m, kind per vertex
  readonly indices: Uint32Array;
  readonly indexCount: number;
}

const FLOATS = VERTEX_BYTE_SIZE / 4;

/** The fill geometry of `p` (vertex layout of `compileTessellation`). */
export function pathGeometry(p: PathInput, fillRule: PathFillRule = "nonzero"): PathGeometry {
  const segs = typeof p === "string" ? pathFromSvgD(p) : p;
  if (segs.length > 0) {
    const t = tessellatePath(segs, fillRule === "evenodd" ? FillRules.evenOdd : FillRules.nonZero);
    if (t.filledFaces.length > 0) {
      const b = compileTessellation(triangulateFilledFaces(t.filledFaces, t.extraction, t.graph));
      const n = b.vertices.length / FLOATS;
      const positions = new Float32Array(n * 2), klmKinds = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) {
        positions[i * 2] = b.vertices[i * FLOATS]!;
        positions[i * 2 + 1] = b.vertices[i * FLOATS + 1]!;
        for (let c = 0; c < 4; c++) klmKinds[i * 4 + c] = b.vertices[i * FLOATS + 2 + c]!;
      }
      if (b.indices.length > 0) return { positions, klmKinds, indices: b.indices, indexCount: b.indices.length };
    }
  }
  // nothing to fill: one dummy vertex and index (a buffer is never empty), drawn 0 times
  return { positions: new Float32Array(2), klmKinds: new Float32Array(4), indices: new Uint32Array(1), indexCount: 0 };
}

// ── effects ────────────────────────────────────────────────────────────

// The path's plane through the model-view trafo (composed CPU-side, see
// text.ts) — no mirroring and no instance offset, unlike a glyph run.
const vsPath = vertex((input: { a_localPos: V2f; a_klmKind: V4f }) => {
  const clip = uniform.ProjTrafo.mul(uniform.ModelViewTrafo.mul(new V4f(input.a_localPos.x, input.a_localPos.y, 0.0, 1.0)));
  return { gl_Position: clip, v_klmKind: input.a_klmKind };
});

// As `vsPath`, with the one-pixel ribbon of kind-3 vertices pushed out
// (inner ones in) along the outline's screen-space normal (text.ts).
const vsPathRibbon = vertex((input: { a_localPos: V2f; a_klmKind: V4f }) => {
  let clip = uniform.ProjTrafo.mul(uniform.ModelViewTrafo.mul(new V4f(input.a_localPos.x, input.a_localPos.y, 0.0, 1.0)));
  if (input.a_klmKind.w > 2.5) {
    const outClip = uniform.ProjTrafo.mul(uniform.ModelViewTrafo.mul(new V4f(input.a_klmKind.x, input.a_klmKind.y, 0.0, 0.0)));
    const outNdc = new V2f(outClip.x, outClip.y).div(max(clip.w, 1e-8));
    const outPx = new V2f(outNdc.x * uniform.Viewport.x, outNdc.y * uniform.Viewport.y);
    const len = max(outPx.length(), 1e-8);
    const signedHalf = input.a_klmKind.z - 0.5;
    const stepNdc = new V2f(
      outPx.x / len / uniform.Viewport.x * 2.0 * uniform.AaWidthPx,
      outPx.y / len / uniform.Viewport.y * 2.0 * uniform.AaWidthPx,
    );
    clip = new V4f(clip.x + stepNdc.x * clip.w * signedHalf, clip.y + stepNdc.y * clip.w * signedHalf, clip.z, clip.w);
  }
  return { gl_Position: clip, v_klmKind: input.a_klmKind };
});

let fillNone: Effect | undefined;
let fillAlpha: Effect | undefined;
const fillEffect = (aa: "none" | "alpha-blending"): Effect =>
  aa === "alpha-blending"
    ? (fillAlpha ??= effect(vsPathRibbon, pathFsAaAlpha))
    : (fillNone ??= effect(vsPath, pathFsAaNone));

let alphaOver: BlendState | undefined;
const alphaOverBlend = (): BlendState => (alphaOver ??= {
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
  writeMask: AVal.constant(0xF),
} as unknown as BlendState);

// ── node ───────────────────────────────────────────────────────────────

const isAval = (x: unknown): x is aval<unknown> =>
  x !== null && typeof x === "object" && typeof (x as { getValue?: unknown }).getValue === "function";

/** `<Sg.Path path="M 0 0 L 1 0 A 1 1 0 0 1 0 1 Z" Color={…}/>`. */
export function SgPath(props: SgPathProps & SgScopeProps): SgNode {
  const { path, fillRule = "nonzero", aa = "none", Color, aaWidthPx, ...scope } = props;
  const pathA: aval<PathInput> = isAval(path) ? (path as aval<PathInput>) : AVal.constant(path as PathInput);
  const geo = pathA.map((p) => pathGeometry(p, fillRule));
  const buffer = <T extends Float32Array | Uint32Array>(f: (g: PathGeometry) => T) =>
    markHostBufferAVal(geo.map((g) => IBuffer.fromHost(f(g))));
  const vertexAttributes = HashMap.empty<string, BufferView>()
    .add("a_localPos", { buffer: buffer((g) => g.positions), elementType: ElementType.V2f })
    .add("a_klmKind", { buffer: buffer((g) => g.klmKinds), elementType: ElementType.V4f });
  const indices: BufferView = { buffer: buffer((g) => g.indices), elementType: ElementType.U32 };
  const drawCall = markHeapSafeDrawAVal(geo.map((g): DrawCall => ({
    kind: "indexed", indexCount: g.indexCount, instanceCount: 1, firstIndex: 0, baseVertex: 0, firstInstance: 0,
  })), "indexed");
  const leaf = Sg.leaf({ vertexAttributes, indices, drawCall });
  const color: aval<V4f> = Color === undefined
    ? AVal.constant(new V4f(1, 1, 1, 1))
    : (isAval(Color) ? (Color as aval<V4f>) : AVal.constant(Color as V4f));
  const aaWidth: aval<number> = aaWidthPx === undefined
    ? AVal.constant(1)
    : (typeof aaWidthPx === "number" ? AVal.constant(aaWidthPx) : aaWidthPx);
  // The viewport (for the AA ramp's width in pixels) and the depth
  // convention come from the traversal state this path is drawn in —
  // not the ambient render control, which an offscreen scene has not.
  const tree = Sg.delay((state: TraversalState) => Sg({
    Shader: fillEffect(aa),
    Uniform: {
      PathColor: color,
      Viewport: state.viewport.map((vp) => new V2f(vp.width, vp.height)),
      AaWidthPx: aaWidth,
    },
    // the fill's triangles come in either orientation
    CullMode: "none",
    ...(aa === "alpha-blending" ? {
      BlendMode: alphaOverBlend(),
      // curve and interior triangles of one path meet at equal depth (text.ts);
      // reversed-Z flips the test
      DepthTest: state.proj.map((p) => (p.forward.M22 >= 0 ? "greater-equal" : "less-equal") as GPUCompareFunction),
    } : {}),
    children: [leaf],
  } as never));
  return Object.keys(scope).length === 0 ? tree : Sg({ ...(scope as object), children: [tree] } as never);
}
