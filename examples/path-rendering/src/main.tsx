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
  OrbitController,
  aspectFromViewport,
  perspective,
} from "@aardworx/wombat.dom/scene";
import type { SceneEvent } from "@aardworx/wombat.dom/scene";
import {
  AVal, HashMap, type aval,
} from "@aardworx/wombat.adaptive";
import {
  V2d, V3d, V4f,
  LineSegment, Bezier2Segment, Bezier3Segment, ArcSegment, Path,
  type PathSegment,
  tessellatePath, triangulateFilledFaces, compileTessellation,
} from "@aardworx/wombat.base";
import { Font, textToSegments } from "@aardworx/wombat.base/font";

// Vite asset URL — emitted at build time, loaded at runtime.
import greatVibesUrl from "./great-vibes.ttf?url";
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
//   Bottom half: a Great Vibes script-font ampersand glyph, parsed
//     live at boot via `wombat.base/font` (Stage 7 TTF lowering)
//     and lowered to PathSegments. Exercises real-world multi-
//     subpath topology (outer body + interior loop).
//
// Coordinate system: (-2..2)² math y-up. Top row sits at y≈1.4, the
// glyph occupies the lower ~2.4 vertical units centred at (0, -0.5).

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

// --- Glyph: Great Vibes ampersand (live TTF lowering) -------------

const font = await Font.load(greatVibesUrl);
const ampGlyph: ReadonlyArray<PathSegment> = (() => {
  // Lower the glyph in raw font units, then place its bbox-centre
  // around y=0 scaled to fit a target height of 1.5 viewport units.
  const raw = font.charToSegments("&");
  const bb = font.charBoundingBox("&");
  const oldCx = (bb.min.x + bb.max.x) * 0.5;
  const oldCy = (bb.min.y + bb.max.y) * 0.5;
  const targetH = 1.5;
  const scale = targetH / (bb.max.y - bb.min.y);
  return place(raw, oldCx, oldCy, 0, 0, scale);
})();

// --- Text run: "Hello" via wombat.base/font layout ----------------

const textRun: ReadonlyArray<PathSegment> = (() => {
  const { segments, layout } = textToSegments(font, "Hello");
  // Centre the laid-out run horizontally and place its baseline at
  // y ≈ -1.5 with a target em-height of 0.7 viewport units.
  const targetH = 0.7;
  const scale = targetH / font.unitsPerEm;
  // `place` interprets oldCx in SOURCE units (pre-scale), so the
  // run's mid-x in font units is `layout.advance * 0.5`.
  return place(segments, layout.advance * 0.5, 0, 0, -1.5, scale);
})();

const TEST_PATH: ReadonlyArray<PathSegment> = [
  ...lineTri, ...bez2Lens, ...bez3Leaf, ...arcCircle,
  ...ampGlyph,
  ...textRun,
];

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

const clear: ClearValues = {
  colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0.0, 0.09, 0.16, 1.0)),
  depth: 1.0,
};

const test = compilePathSegments(TEST_PATH);

// Orbit camera. Wombat.dom's default sky is +Z (z-up), so the path's
// xy plane already lies on the floor and reads "naturally from above".
// phi = -π/2 puts the camera on the -y side of the floor, so the
// path's reading direction (+y in path-frame, i.e. "up" of each
// glyph) points away from the camera and the text reads naturally.
const ctl = OrbitController.create({
  radius: 6,
  phi: -Math.PI / 2,
  theta: Math.PI / 4,
});

const flyToHit = (e: SceneEvent): void => {
  ctl.flyTo(e.worldPos);
};

mount(root, (
  <RenderControl
    clear={clear}
    onReady={({ canvas, time }) => {
      ctl.attach(canvas, time);
      status.textContent = "ready — drag to rotate, wheel zoom, double-tap a glyph to fly to it";
    }}
  >
    <Sg
      View={ctl.view}
      Proj={perspective({
        fovInRadians: Math.PI / 3,
        aspect: aspectFromViewport(RenderControl.viewport),
        near: 0.05,
        far: 200,
      })}
      Shader={loopBlinnEffect}
      OnDoubleTap={flyToHit}
      PixelSnapRadius={8}
    >
      {/* CullMode="none" because path triangles are CCW in math
          coords (y-up) which WebGPU sees as CW in framebuffer space
          under the default `frontFace="ccw"`. Disabling culling is
          the cheapest correct option until the orientation is
          baked into the triangulator. */}
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
