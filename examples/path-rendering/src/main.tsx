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
  LineSegment, Bezier2Segment, ArcSegment, Path,
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
      // Inline klmKind component access to dodge the frontend's
      // sibling-binding bug — using \`const k = …; const l = …\` and
      // referencing them in nested ifs makes them unresolved.
      if (input.v_klmKind.w > 1.7) {
        if (input.v_klmKind.x * input.v_klmKind.x + input.v_klmKind.y * input.v_klmKind.y - 1.0 > 0.0) discard;
      } else if (input.v_klmKind.w > 0.7) {
        if (input.v_klmKind.x * input.v_klmKind.x - input.v_klmKind.y > 0.0) discard;
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

// Rounded square: 4 lines + 4 quadratic-Bezier corners. Exercises
// both the flat triangulation and the Loop-Blinn curve test.
const R = 0.4;        // corner radius
const E = 1.0;        // half-edge of the square
const TEST_PATH: ReadonlyArray<PathSegment> = [
  // bottom edge (left → right)
  new LineSegment(new V2d(-E + R, -E), new V2d( E - R, -E)),
  // bottom-right corner
  new Bezier2Segment(new V2d( E - R, -E), new V2d( E, -E), new V2d( E, -E + R)),
  // right edge
  new LineSegment(new V2d( E, -E + R), new V2d( E,  E - R)),
  // top-right corner
  new Bezier2Segment(new V2d( E,  E - R), new V2d( E,  E), new V2d( E - R,  E)),
  // top edge
  new LineSegment(new V2d( E - R,  E), new V2d(-E + R,  E)),
  // top-left corner
  new Bezier2Segment(new V2d(-E + R,  E), new V2d(-E,  E), new V2d(-E,  E - R)),
  // left edge
  new LineSegment(new V2d(-E,  E - R), new V2d(-E, -E + R)),
  // bottom-left corner
  new Bezier2Segment(new V2d(-E, -E + R), new V2d(-E, -E), new V2d(-E + R, -E)),
];

// SVG mirror — same path traced as `M / L / Q / Z` directives.
function testPathToSvg(): string {
  const head = TEST_PATH[0]!;
  let d = `M ${head.start.x} ${head.start.y}`;
  for (const s of TEST_PATH) {
    if (s.kind === "line") {
      d += ` L ${s.end.x} ${s.end.y}`;
    } else if (s.kind === "bezier2") {
      d += ` Q ${s.control.x} ${s.control.y} ${s.end.x} ${s.end.y}`;
    }
  }
  d += " Z";
  return d;
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
  + "width:200px; height:200px; background:#001828;"
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
