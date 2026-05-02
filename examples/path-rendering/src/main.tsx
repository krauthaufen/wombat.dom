// path-rendering — first end-to-end demo of the Stage 0–6 path
// tessellator. A handful of paths (a glyph-like shape, a star, a
// disc, a Bezier blob) are tessellated via wombat.base and rendered
// with a Loop-Blinn fragment shader written using wombat.shader's
// inline `effect(vertex(arrow), fragment(arrow))` markers.
//
// The shader uses a real `discard` — the vite plugin walks the
// arrow body, lifts `if (...) discard();` to the IR, and the WGSL
// emitter outputs `discard();`. Multisampling + discard gives clean
// edges on the curve boundaries.

import { mount } from "@aardworx/wombat.dom";
import {
  OrbitController,
  RenderControl,
  Sg,
  aspectFromViewport,
  perspective,
} from "@aardworx/wombat.dom/scene";
import {
  AVal, HashMap, type aval,
} from "@aardworx/wombat.adaptive";
import {
  V2d, V3d, V4f, M44f, V2f,
  ArcSegment, Bezier2Segment, Bezier3Segment, LineSegment, Path,
  type PathSegment,
  tessellatePath, triangulateFilledFaces, compileTessellation,
} from "@aardworx/wombat.base";
import {
  IBuffer,
  type BufferView, type DrawCall,
} from "@aardworx/wombat.rendering/core";
import type { ClearValues } from "@aardworx/wombat.rendering/core";
import { effect, fragment, vertex } from "@aardworx/wombat.shader";

// `discard` is a shader-only intrinsic recognised by the wombat.shader
// vite plugin (it's in SHIPPED_INTRINSIC_NAMES). The local ambient
// declaration here just satisfies the TS type-checker; the plugin
// emits a proper WGSL `discard();` statement.
declare function discard(): never;

// Ambient uniforms — the Sg compileScene auto-injects ModelTrafo /
// ViewTrafo / ProjTrafo by matching name. The wombat.shader vite
// plugin sees these `declare const`s and emits matching Uniform
// ValueDefs in the IR. PathColor follows the same convention,
// supplied per leaf via the scope's `Uniform={{ PathColor: … }}`.
declare const ModelTrafo: M44f;
declare const ViewTrafo:  M44f;
declare const ProjTrafo:  M44f;
declare const PathColor:  V4f;

// ---------------------------------------------------------------------------
// Loop-Blinn surface
// ---------------------------------------------------------------------------
//
// Vertex format produced by `compileTessellation`:
//   a_position : V2f   — path-local 2D xy
//   a_klmKind  : V4f   — Loop-Blinn (k, l, m) + kind:
//                          kind = 0 → interior  (no curve test)
//                          kind = 1 → bezier2   (k² − l > 0 → discard)
//                          kind = 2 → arc       (k² + l² − 1 > 0 → discard)
//
// Each triangle has the same kind on all 3 vertices, so the
// interpolated value is constant across the triangle interior — no
// flat-interpolation needed.
// Vertex body uses expression form (no `const` locals). The plugin's
// capture detector was patched (it now threads VariableStatement
// bindings across sibling stmts), but a downstream frontend pass
// still mis-resolves the second const in a chain — yak deferred.
const loopBlinnEffect = effect(
  vertex<{ a_position: V2f; a_klmKind: V4f }>(input => ({
    gl_Position: ProjTrafo.mul(ViewTrafo.mul(ModelTrafo.mul(
      new V4f(input.a_position.x, input.a_position.y, 0.0, 1.0),
    ))),
    v_klmKind: input.a_klmKind,
  })),
  fragment<{ v_klmKind: V4f }>(input => {
    if (input.v_klmKind.w > 1.5) {
      // Arc: ellipse-local unit circle. Outside circle → discard.
      if (input.v_klmKind.x * input.v_klmKind.x
        + input.v_klmKind.y * input.v_klmKind.y - 1.0 > 0.0) discard();
    } else if (input.v_klmKind.w > 0.5) {
      // Bezier2: standard Loop-Blinn k² − l test.
      if (input.v_klmKind.x * input.v_klmKind.x - input.v_klmKind.y > 0.0) discard();
    }
    // DEBUG: hardcoded red so we can verify geometry without
    // depending on the PathColor uniform binding (which may not yet
    // be wired through Sg.Leaf's `Uniform` scope prop).
    return new V4f(1.0, 0.3, 0.3, 1.0);
  }),
);

// ---------------------------------------------------------------------------
// Path → SgLeaf bridge
// ---------------------------------------------------------------------------

interface CompiledPath {
  /** Vertex attributes ready for the Sg leaf. */
  readonly vertexAttrs: HashMap<string, aval<BufferView>>;
  /** Index buffer view. */
  readonly indices: aval<BufferView>;
  /** Indexed draw call covering interior + curve triangles. */
  readonly drawCall: aval<DrawCall>;
}

function compilePathSegments(input: ReadonlyArray<Path | PathSegment>): CompiledPath {
  const r = tessellatePath(input);
  const tri = triangulateFilledFaces(r.filledFaces, r.extraction, r.graph);
  const bufs = compileTessellation(tri);

  const vBuf = IBuffer.fromHost(bufs.vertices);
  const iBuf = IBuffer.fromHost(bufs.indices);
  const vCount = bufs.vertices.length / 6;

  // Two attributes interleaved into one buffer.
  // a_position: float32x2 at offset 0
  // a_klmKind:  float32x4 at offset 8
  const positionView: BufferView = {
    buffer: vBuf, offset: 0, count: vCount, stride: 24, format: "float32x2",
  };
  const klmKindView: BufferView = {
    buffer: vBuf, offset: 8, count: vCount, stride: 24, format: "float32x4",
  };
  const indexView: BufferView = {
    buffer: iBuf, offset: 0, count: bufs.indices.length, stride: 4, format: "uint32",
  };
  const drawCall: DrawCall = {
    kind: "indexed",
    indexCount: bufs.indices.length,
    instanceCount: 1,
    firstIndex: 0,
    baseVertex: 0,
    firstInstance: 0,
  };

  return {
    vertexAttrs: HashMap.empty<string, aval<BufferView>>()
      .add("a_position", AVal.constant(positionView))
      .add("a_klmKind",  AVal.constant(klmKindView)),
    indices:  AVal.constant(indexView),
    drawCall: AVal.constant(drawCall),
  };
}

// ---------------------------------------------------------------------------
// Demo paths
// ---------------------------------------------------------------------------

// A "glyph-like" shape — outer rounded rectangle + inner hole
// (annulus topology, exercises the bridge-edge stage).
function glyphLikePath(): ReadonlyArray<PathSegment> {
  const outer = [
    new LineSegment(new V2d(-1, -1.2), new V2d(1, -1.2)),
    new Bezier2Segment(new V2d(1, -1.2), new V2d(1.5, -1.2), new V2d(1.5, -0.7)),
    new LineSegment(new V2d(1.5, -0.7), new V2d(1.5, 0.7)),
    new Bezier2Segment(new V2d(1.5, 0.7), new V2d(1.5, 1.2), new V2d(1, 1.2)),
    new LineSegment(new V2d(1, 1.2), new V2d(-1, 1.2)),
    new Bezier2Segment(new V2d(-1, 1.2), new V2d(-1.5, 1.2), new V2d(-1.5, 0.7)),
    new LineSegment(new V2d(-1.5, 0.7), new V2d(-1.5, -0.7)),
    new Bezier2Segment(new V2d(-1.5, -0.7), new V2d(-1.5, -1.2), new V2d(-1, -1.2)),
  ];
  // Inner hole — a CW circle (3 quarter arcs would also work).
  const innerCenter = new V2d(0, 0);
  const innerR = 0.4;
  const innerHole = [
    ArcSegment.circular(innerCenter, innerR, Math.PI / 2, -Math.PI / 2),
    ArcSegment.circular(innerCenter, innerR, 0, -Math.PI / 2),
    ArcSegment.circular(innerCenter, innerR, -Math.PI / 2, -Math.PI / 2),
    ArcSegment.circular(innerCenter, innerR, -Math.PI, -Math.PI / 2),
  ];
  return [...outer, ...innerHole];
}

// A 5-pointed star traced as 10 line segments.
function starPath(): ReadonlyArray<PathSegment> {
  const out: LineSegment[] = [];
  const n = 5;
  const pts: V2d[] = [];
  for (let i = 0; i < 2 * n; i++) {
    const r = i % 2 === 0 ? 1.2 : 0.5;
    const t = i * Math.PI / n - Math.PI / 2;
    pts.push(new V2d(r * Math.cos(t), r * Math.sin(t)));
  }
  for (let i = 0; i < pts.length; i++) {
    out.push(new LineSegment(pts[i]!, pts[(i + 1) % pts.length]!));
  }
  return out;
}

// A Bezier blob — closed shape with 4 cubic-Bezier sides.
function blobPath(): ReadonlyArray<PathSegment> {
  return [
    new Bezier3Segment(new V2d(1, 0),  new V2d(1, 1.2),  new V2d(0.4, 1.2),  new V2d(0, 1)),
    new Bezier3Segment(new V2d(0, 1),  new V2d(-0.6, 1), new V2d(-1.2, 0.4), new V2d(-1, 0)),
    new Bezier3Segment(new V2d(-1, 0), new V2d(-1, -1),  new V2d(-0.4, -1),  new V2d(0, -0.8)),
    new Bezier3Segment(new V2d(0, -0.8), new V2d(0.6, -1), new V2d(1.2, -0.4), new V2d(1, 0)),
  ];
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
const origConsoleError = console.error.bind(console);
console.error = (...args) => {
  origConsoleError(...args);
  const text = args.map(a => a instanceof Error ? a.message : String(a)).join(" ");
  status.textContent = "console.error: " + text.slice(0, 500);
  status.style.color = "#ff7777";
};

const ctl = OrbitController.create({
  radius: 6,
  phi: Math.PI / 5,
  theta: 0.5,
});

const clear: ClearValues = {
  colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0.07, 0.07, 0.08, 1.0)),
  depth: 1.0,
};

// Tessellate each demo path once at startup.
const glyph = compilePathSegments(glyphLikePath());
const star  = compilePathSegments(starPath());
const blob  = compilePathSegments(blobPath());

mount(root, (
  <RenderControl
    clear={clear}
    onReady={({ canvas, time }) => {
      ctl.attach(canvas, time);
      status.textContent = "ready — drag to rotate, wheel to zoom";
    }}
  >
    <Sg
      View={ctl.view}
      Shader={loopBlinnEffect}
      Proj={perspective({
        fovInRadians: Math.PI / 3,
        aspect: aspectFromViewport(RenderControl.viewport),
        near: 0.1,
        far: 100,
      })}
    >
      {/* PathColor is a per-leaf uniform passed via the scope's
          Uniform bag. wombat.shader's vite plugin sees the
          `declare const PathColor: V4f` and emits a Uniform
          ValueDef; the Sg compiler binds the scope value to it. */}
      <Sg.Leaf
        Trafo={Sg.translate(new V3d(-3, 0, 0))}
        Uniform={{ PathColor: new V4f(1.0, 0.55, 0.25, 1) }}
        vertexAttributes={glyph.vertexAttrs}
        indices={glyph.indices}
        drawCall={glyph.drawCall}
      />
      <Sg.Leaf
        Uniform={{ PathColor: new V4f(0.45, 0.75, 1.0, 1) }}
        vertexAttributes={star.vertexAttrs}
        indices={star.indices}
        drawCall={star.drawCall}
      />
      <Sg.Leaf
        Trafo={Sg.translate(new V3d(3, 0, 0))}
        Uniform={{ PathColor: new V4f(0.55, 0.95, 0.55, 1) }}
        vertexAttributes={blob.vertexAttrs}
        indices={blob.indices}
        drawCall={blob.drawCall}
      />
    </Sg>
  </RenderControl>
));
