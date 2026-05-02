// path-rendering — head-on diagnostic.
//
// Goal: render ONE simple closed path with the path tessellator and
// the Loop-Blinn shader, with the camera looking straight down −Z at
// the XY plane and an orthographic projection. Beside the canvas,
// render the SAME path as an inline SVG with `viewBox` matching the
// orthographic frustum, so the two should look identical pixel-for-
// pixel modulo aliasing.

import { mount } from "@aardworx/wombat.dom";
import {
  RenderControl,
  Sg,
  lookAt,
  orthographic,
} from "@aardworx/wombat.dom/scene";
import {
  AVal, HashMap, type aval,
} from "@aardworx/wombat.adaptive";
import {
  V2d, V3d, V4f,
  LineSegment, Bezier2Segment, Bezier3Segment, ArcSegment, Path,
  type PathSegment,
  tessellatePath, triangulateFilledFaces, compileTessellation,
} from "@aardworx/wombat.base";
import {
  IBuffer,
  type BufferView, type DrawCall,
} from "@aardworx/wombat.rendering/core";
import type { ClearValues } from "@aardworx/wombat.rendering/core";
import type { Effect } from "@aardworx/wombat.shader";
import { stage } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import {
  Tf32, Vec, type Type, type ValueDef, type Module,
} from "@aardworx/wombat.shader/ir";

// ---------------------------------------------------------------------------
// Loop-Blinn surface — string-source via parseShader+stage, same
// pattern as wombat.dom's defaultSurfaces.basic. Manual UBO layout
// (a single Camera UBO at group 0 slot 0 with all uniforms) so Sg's
// auto-injection by name and the leaf's Uniform={{ PathColor }} both
// land in the same buffer.
//
// Why not the inline marker plugin: at the time of writing the plugin
// has multiple bugs that block this combination — capture-detector
// sibling-binding (fixed locally), generic-typed marker regex (fixed
// locally), and an unresolved cross-namespace bind-group-layout
// collision when more than one ambient namespace is used. The
// string-source path is well-tested (defaultSurfaces) and avoids
// every one of those.
// ---------------------------------------------------------------------------

const Tvec2f: Type = Vec(Tf32, 2);
const Tvec4f: Type = Vec(Tf32, 4);
const TM44f:  Type = { kind: "Matrix", element: Tf32, rows: 4, cols: 4 } as Type;

function buildLoopBlinnEffect(): Effect {
  const source = `
    declare const ModelTrafo: M44f;
    declare const ViewTrafo:  M44f;
    declare const ProjTrafo:  M44f;
    declare const PathColor:  V4f;

    function vsMain(input: { a_position: V2f; a_klmKind: V4f })
      : { gl_Position: V4f; v_klmKind: V4f } {
      const world = ModelTrafo.mul(new V4f(input.a_position.x, input.a_position.y, 0.0, 1.0));
      const view  = ViewTrafo.mul(world);
      const clip  = ProjTrafo.mul(view);
      return { gl_Position: clip, v_klmKind: input.a_klmKind };
    }

    function fsMain(input: { v_klmKind: V4f }): { outColor: V4f } {
      // Loop-Blinn implicit test, with the M-component carrying a
      // ±1 sign that flips inside/outside discard for curves whose
      // "extra" vertex (bez2 control, arc apex) lies inside the
      // chord polygon (= curve bulges into the solid → subtractive).
      // Mirrors Aardvark.Rendering.Text's pathFragment shader.
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
        { name: "a_position", type: Tvec2f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "a_klmKind",  type: Tvec4f, semantic: "KLMKind",  decorations: [{ kind: "Location", value: 1 }] },
      ],
      outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
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
  return stage(merged);
}

const loopBlinnEffect = buildLoopBlinnEffect();

// ---------------------------------------------------------------------------
// Path → SgLeaf bridge
// ---------------------------------------------------------------------------

interface CompiledPath {
  readonly vertexAttrs: HashMap<string, aval<BufferView>>;
  readonly indices: aval<BufferView>;
  readonly drawCall: aval<DrawCall>;
  readonly triangleCount: number;
}

function compilePathSegments(input: ReadonlyArray<Path | PathSegment>): CompiledPath {
  const r = tessellatePath(input);
  const tri = triangulateFilledFaces(r.filledFaces, r.extraction, r.graph);
  const bufs = compileTessellation(tri);
  const vCount = bufs.vertices.length / 6;

  // De-interleave: wombat.rendering's vertex-attribute binding doesn't
  // handle multiple attributes in the same buffer correctly (or my
  // setup of it is wrong); separate buffers per attribute, like
  // hello-triangle, is the proven pattern.
  const positions = new Float32Array(vCount * 2);
  const klmKinds  = new Float32Array(vCount * 4);
  for (let i = 0; i < vCount; i++) {
    positions[i * 2 + 0] = bufs.vertices[i * 6 + 0]!;
    positions[i * 2 + 1] = bufs.vertices[i * 6 + 1]!;
    klmKinds[i * 4 + 0] = bufs.vertices[i * 6 + 2]!;
    klmKinds[i * 4 + 1] = bufs.vertices[i * 6 + 3]!;
    klmKinds[i * 4 + 2] = bufs.vertices[i * 6 + 4]!;
    klmKinds[i * 4 + 3] = bufs.vertices[i * 6 + 5]!;
  }
  const posBuf = IBuffer.fromHost(positions);
  const klmBuf = IBuffer.fromHost(klmKinds);
  const iBuf = IBuffer.fromHost(bufs.indices);

  return {
    vertexAttrs: HashMap.empty<string, aval<BufferView>>()
      .add("a_position", AVal.constant<BufferView>({
        buffer: posBuf, offset: 0, count: vCount, stride: 8, format: "float32x2",
      }))
      .add("a_klmKind", AVal.constant<BufferView>({
        buffer: klmBuf, offset: 0, count: vCount, stride: 16, format: "float32x4",
      })),
    indices: AVal.constant<BufferView>({
      buffer: iBuf, offset: 0, count: bufs.indices.length, stride: 4, format: "uint32",
    }),
    drawCall: AVal.constant<DrawCall>({
      kind: "indexed",
      indexCount: bufs.indices.length,
      instanceCount: 1, firstIndex: 0, baseVertex: 0, firstInstance: 0,
    }),
    triangleCount: bufs.indices.length / 3,
  };
}

// ---------------------------------------------------------------------------
// Test path — a CCW unit square traced as 4 line segments. Convex,
// no holes, no curves: the simplest possible filled polygon.
// ---------------------------------------------------------------------------

// Test paths.
//
//   Row of 4 primitive shapes across the top — one per segment kind:
//     lines (triangle), bezier2 (lens), bezier3 (leaf), arc (circle).
//   Bottom half: a Great Vibes script-font ampersand glyph, lowered
//     from the font's quadratic-Bezier outline. Exercises real-world
//     multi-subpath topology (outer body + interior loop).
//
// Coordinate system: (-2..2)² math y-up. Top row sits at y≈1.4, the
// glyph occupies the lower ~2.4 vertical units centred at (0, -0.5).

import glyphAmpData from "./glyph-amp.json";

// Affine transform of a list of segments. Endpoints shared by V2d
// identity in the input are preserved in the output (required by the
// planar-graph spatial-hash for arcs whose start / end are computed
// once via cos/sin and reused across two halves).
function transformSegs(
  segs: ReadonlyArray<PathSegment>,
  dx: number, dy: number,
  sx: number, sy: number = sx,
): PathSegment[] {
  const cache = new Map<V2d, V2d>();
  const t = (p: V2d): V2d => {
    let q = cache.get(p);
    if (!q) { q = new V2d(p.x * sx + dx, p.y * sy + dy); cache.set(p, q); }
    return q;
  };
  return segs.map((s): PathSegment => {
    switch (s.kind) {
      case "line":    return new LineSegment(t(s.start), t(s.end));
      case "bezier2": return new Bezier2Segment(t(s.start), t(s.control), t(s.end));
      case "bezier3": return new Bezier3Segment(t(s.start), t(s.control1), t(s.control2), t(s.end));
      case "arc":     return new ArcSegment(
        t(s.start), t(s.end), t(s.center),
        new V2d(s.axis0.x * sx, s.axis0.y * sy),
        new V2d(s.axis1.x * sx, s.axis1.y * sy),
        s.startAngle, s.deltaAngle,
      );
    }
  });
}

// Translate a path so its bbox-centre lands at (newCx, newCy) and is
// uniformly scaled by `scale`.
function place(
  segs: ReadonlyArray<PathSegment>,
  oldCx: number, oldCy: number,
  newCx: number, newCy: number,
  scale: number, scaleY: number = scale,
): PathSegment[] {
  return transformSegs(segs, newCx - oldCx * scale, newCy - oldCy * scaleY, scale, scaleY);
}

// --- Top row: 4 primitive kinds, half-size, packed at y≈1.4 -------

const lineTri0: ReadonlyArray<PathSegment> = (() => {
  const a = new V2d(-1.6,  0.4);
  const b = new V2d(-0.4,  0.4);
  const c = new V2d(-1.0,  1.6);
  return [
    new LineSegment(a, b),
    new LineSegment(b, c),
    new LineSegment(c, a),
  ];
})();
const lineTri = place(lineTri0, -1, 1, -1.5, 1.4, 0.40);

const bez2Lens0: ReadonlyArray<PathSegment> = (() => {
  const l = new V2d(0.4, 1);
  const r = new V2d(1.6, 1);
  return [
    new Bezier2Segment(l, new V2d(1, 1.7), r),
    new Bezier2Segment(r, new V2d(1, 0.3), l),
  ];
})();
const bez2Lens = place(bez2Lens0, 1, 1, -0.5, 1.4, 0.40);

const bez3Leaf0: ReadonlyArray<PathSegment> = (() => {
  const top = new V2d(-1, -0.4);
  const bot = new V2d(-1, -1.6);
  return [
    new Bezier3Segment(top, new V2d(-0.3, -0.5), new V2d(-0.3, -1.5), bot),
    new Bezier3Segment(bot, new V2d(-1.7, -1.5), new V2d(-1.7, -0.5), top),
  ];
})();
const bez3Leaf = place(bez3Leaf0, -1, -1, 0.5, 1.4, 0.40);

const arcCircle0: ReadonlyArray<PathSegment> = (() => {
  const cx = 1, cy = -1, r = 0.6;
  return [
    ArcSegment.circular(new V2d(cx, cy), r, 0,        Math.PI),
    ArcSegment.circular(new V2d(cx, cy), r, Math.PI,  Math.PI),
  ];
})();
const arcCircle = place(arcCircle0, 1, -1, 1.5, 1.4, 0.40);

// --- Glyph: Great Vibes ampersand ----------------------------------

// Convert opentype.js path commands to PathSegments.
//   - opentype emits coords in screen-y-down; flip to math y-up.
//   - L commands of zero length (very common artefact in this font's
//     command stream) are dropped.
//   - Z closes back to the current sub-path's M anchor with a
//     LineSegment if needed.
type GlyphCmd = ["M", number, number] | ["L", number, number]
  | ["Q", number, number, number, number]
  | ["C", number, number, number, number, number, number]
  | ["Z"];

function commandsToSegments(
  commands: ReadonlyArray<GlyphCmd>,
  xform: (x: number, y: number) => V2d,
): PathSegment[] {
  const out: PathSegment[] = [];
  let pen: V2d | undefined;
  let anchor: V2d | undefined;
  const closeIfOpen = (): void => {
    if (pen && anchor && (Math.abs(pen.x - anchor.x) > 1e-12 || Math.abs(pen.y - anchor.y) > 1e-12)) {
      out.push(new LineSegment(pen, anchor));
    }
  };
  for (const c of commands) {
    if (c[0] === "M") {
      const p = xform(c[1], c[2]);
      pen = p; anchor = p;
    } else if (c[0] === "L") {
      const p = xform(c[1], c[2]);
      if (pen && (Math.abs(pen.x - p.x) > 1e-12 || Math.abs(pen.y - p.y) > 1e-12)) {
        out.push(new LineSegment(pen, p));
        pen = p;
      }
    } else if (c[0] === "Q") {
      const ctrl = xform(c[1], c[2]);
      const p = xform(c[3], c[4]);
      if (pen) { out.push(new Bezier2Segment(pen, ctrl, p)); pen = p; }
    } else if (c[0] === "C") {
      const c1 = xform(c[1], c[2]);
      const c2 = xform(c[3], c[4]);
      const p = xform(c[5], c[6]);
      if (pen) { out.push(new Bezier3Segment(pen, c1, c2, p)); pen = p; }
    } else if (c[0] === "Z") {
      closeIfOpen();
      pen = anchor;
    }
  }
  closeIfOpen();
  return out;
}

const ampGlyph: ReadonlyArray<PathSegment> = (() => {
  // Glyph bbox in opentype.js coords (y-down): x∈[11,749], y∈[-706,45].
  // After y flip → math y-up: x∈[11,749], y∈[-45,706]. Centre (380,330.5),
  // size 738×751.
  const oldCx = 380, oldCy = 330.5;
  const newCx = 0, newCy = -0.5;
  const targetH = 2.2;
  const scale = targetH / 751;
  return commandsToSegments(
    glyphAmpData as unknown as ReadonlyArray<GlyphCmd>,
    (x, y) => new V2d((x - oldCx) * scale + newCx, (-y - oldCy) * scale + newCy),
  );
})();

const TEST_PATH: ReadonlyArray<PathSegment> = [
  ...lineTri, ...bez2Lens, ...bez3Leaf, ...arcCircle,
  ...ampGlyph,
];

// SVG mirror — emit a `M…Z` subpath per primitive group so each
// closed contour stays closed. Supports L / Q / C / A directives.
// SVG's arc-sweep flag is inverted relative to math-CCW because the
// outer <svg> wears `transform="scale(1,-1)"` to flip y-down → y-up.
function pathSegmentsToSvgD(segs: ReadonlyArray<PathSegment>): string {
  const groups: PathSegment[][] = [];
  let cur: PathSegment[] = [];
  for (const s of segs) {
    const prev = cur[cur.length - 1];
    const close = prev !== undefined
      && Math.abs(prev.end.x - s.start.x) < 1e-9
      && Math.abs(prev.end.y - s.start.y) < 1e-9;
    if (cur.length === 0 || close) {
      cur.push(s);
    } else {
      groups.push(cur);
      cur = [s];
    }
  }
  if (cur.length > 0) groups.push(cur);

  const out: string[] = [];
  for (const g of groups) {
    out.push(`M ${g[0]!.start.x} ${g[0]!.start.y}`);
    for (const s of g) {
      if (s.kind === "line") {
        out.push(`L ${s.end.x} ${s.end.y}`);
      } else if (s.kind === "bezier2") {
        out.push(`Q ${s.control.x} ${s.control.y} ${s.end.x} ${s.end.y}`);
      } else if (s.kind === "bezier3") {
        out.push(`C ${s.control1.x} ${s.control1.y} ${s.control2.x} ${s.control2.y} ${s.end.x} ${s.end.y}`);
      } else if (s.kind === "arc") {
        const rx = Math.hypot(s.axis0.x, s.axis0.y);
        const ry = Math.hypot(s.axis1.x, s.axis1.y);
        const rot = Math.atan2(s.axis0.y, s.axis0.x) * 180 / Math.PI;
        const largeArc = Math.abs(s.deltaAngle) > Math.PI ? 1 : 0;
        // sweep-flag in SVG's native y-down: CCW=0, CW=1. Our
        // viewer flips y, so invert: math-CCW (deltaAngle>0) → 1.
        const sweep = s.deltaAngle > 0 ? 1 : 0;
        out.push(`A ${rx} ${ry} ${rot} ${largeArc} ${sweep} ${s.end.x} ${s.end.y}`);
      }
    }
    out.push("Z");
  }
  return out.join(" ");
}

function testPathToSvg(): string {
  return pathSegmentsToSvgD(TEST_PATH);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

const root = document.getElementById("app")!;
const status = document.getElementById("status")!;
status.textContent = "starting…";

window.addEventListener("error", (e) => {
  status.textContent = "error: " + (e.error?.message ?? e.message);
  status.style.color = "#ff7777";
});
window.addEventListener("unhandledrejection", (e) => {
  const msg = e.reason?.message ?? String(e.reason);
  status.textContent = "promise rejected: " + msg;
  status.style.color = "#ff7777";
});

// Reference SVG overlay — bottom-right corner so it doesn't clobber
// the WebGPU canvas's layout. Same `viewBox` as the orthographic
// frustum (4×4 world units centred at origin) so the path should
// look identical between the two.
const svgNS = "http://www.w3.org/2000/svg";
const svg = document.createElementNS(svgNS, "svg");
svg.setAttribute("viewBox", "-2 -2 4 4");
svg.setAttribute("style",
  "position:absolute; right:8px; bottom:36px;"
  + "width:384px; height:384px; background:#001828;"
  + "border:1px solid #555; pointer-events:none;");
const svgPath = document.createElementNS(svgNS, "path");
svgPath.setAttribute("d", testPathToSvg());
svgPath.setAttribute("fill", "rgb(229, 130, 65)");
svgPath.setAttribute("transform", "scale(1, -1)"); // SVG y-down → math y-up
svg.appendChild(svgPath);
document.body.appendChild(svg);

const clear: ClearValues = {
  colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0.0, 0.09, 0.16, 1.0)),
  depth: 1.0,
};

const test = compilePathSegments(TEST_PATH);
console.log("test path tris:", test.triangleCount);

// DEBUG: dump the actual triangulated vertex/index data.
{
  const r = tessellatePath(TEST_PATH);
  const tri = triangulateFilledFaces(r.filledFaces, r.extraction, r.graph);
  const bufs = compileTessellation(tri);
  console.log("buffer floats:", Array.from(bufs.vertices).map(x => x.toFixed(2)).join(","));
  console.log("buffer bytes:", bufs.vertices.byteLength,
    "= 6 f32 *", bufs.vertices.length / 6, "verts");
  console.log("indices:", Array.from(bufs.indices).join(","));
}

mount(root, (
  <RenderControl
    clear={clear}
    onReady={() => {
      status.textContent = "ready — head-on orthographic; webgpu (left) vs svg (right)";
    }}
  >
    <Sg
      View={lookAt({
        eye:    new V3d(0, 0, 5),
        target: new V3d(0, 0, 0),
        up:     new V3d(0, 1, 0),
      })}
      Proj={orthographic({ left: -2, right: 2, bottom: -2, top: 2, near: 0.1, far: 100 })}
      Shader={loopBlinnEffect}
    >
      {/* CullMode="none" because path triangles are CCW in math
          coords (y-up) which WebGPU sees as CW in framebuffer space
          (y-down) under the default `frontFace="ccw"`, which would
          cull them. The right long-term fix is to flip the path
          triangulation to emit framebuffer-CCW order, or to set
          FrontFace="cw" at the demo level. Disabling culling is the
          cheapest correct option for a 2D path renderer. */}
      <Sg.Leaf
        Uniform={{ PathColor: new V4f(0.9, 0.51, 0.255, 1) }}
        CullMode="none"
        vertexAttributes={test.vertexAttrs}
        indices={test.indices}
        drawCall={test.drawCall}
      />
    </Sg>
  </RenderControl>
));
