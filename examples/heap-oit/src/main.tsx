// heap-oit — reversed-Z OIT over a heap scene: the dev harness for
// heap ↔ transparency integration. Fixed camera, known layout, pixel
// checks via screenshot; `?mode=off|wboit|abuffer` selects the path.
//
// Layout (world units, camera at (0,0,10) looking -z, y up):
//   opaque RED quad,   z = 0, x ∈ [-4, +4]         — backdrop
//   heap GREEN quad,   z = 2, x ∈ [-4,  0], α 0.5  — Sg.transparent (heap-eligible)
//   MVP  BLUE quad,    z = 4, x ∈ [-2, +2], α 0.5  — Sg.transparent, custom
//        shader reading uniform.ModelViewProjTrafo (the FShade-style path)
//
// Expected (mode=abuffer): left = green·red, center = blue·green·red,
// center-right = blue·red, far right = pure red.

import { mount } from "@aardworx/wombat.dom";
import {
  RenderControl,
  Sg,
  lookAt,
  quad,
} from "@aardworx/wombat.dom/scene";
import { effect, fragment, vertex } from "@aardworx/wombat.shader";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { AVal, HashMap } from "@aardworx/wombat.adaptive";
import { M44f, Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import { IBuffer, type ClearValues } from "@aardworx/wombat.rendering/core";


const root = document.getElementById("app")!;
const status = document.getElementById("status")!;
status.textContent = "starting…";

window.addEventListener("error", (e) => {
  status.textContent = "error: " + (e.error?.message ?? e.message);
  (globalThis as any).__err = String(e.error?.message ?? e.message);
});
window.addEventListener("unhandledrejection", (e) => {
  const msg = e.reason?.message ?? String(e.reason);
  status.textContent = "promise rejected: " + msg;
  (globalThis as any).__err = msg;
});
const origWarn = console.warn.bind(console);
console.warn = (...args: unknown[]) => {
  origWarn(...args);
  const t = args.map(String).join(" ");
  if (/error|invalid|unresolved/i.test(t)) {
    (globalThis as any).__gpuWarn = ((globalThis as any).__gpuWarn ?? "") + t.slice(0, 300) + "\n";
  }
};

const mode = new URLSearchParams(location.search).get("mode") ?? "abuffer";

// ── camera: fixed, reversed-Z infinite perspective ──
const view = lookAt({ eye: new V3d(0, 0, 10), target: new V3d(0, 0, 0), up: new V3d(0, 1, 0) });
const proj = AVal.constant(Trafo3d.perspectiveProjectionReversedRHFov((60 * Math.PI) / 180, 4 / 3, 0.5));

// ── custom shader reading ModelViewProjTrafo (FShade-style path) ──
declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope {
    readonly ModelViewProjTrafo: M44f;
    readonly u_color: V4f;
  }
}
const mvpVS = vertex((v: { Positions: V4f }) => ({
  gl_Position: uniform.ModelViewProjTrafo.mul(v.Positions),
}));
const mvpFS = fragment((_i: {}) => ({ Colors: uniform.u_color }));
const mvpEffect = effect(mvpVS, mvpFS);

// minimal Model/ViewProj shader with vertex-color passthrough (avoids
// DefaultSurfaces' inv-transpose row-vec trick, which trips the emit here)
const baseVS = vertex((v: { Positions: V4f; Colors: V4f }) => ({
  gl_Position: uniform.ViewProjTrafo.mul(uniform.ModelTrafo.mul(v.Positions)),
  Colors: v.Colors,
}));
const baseFS = fragment((i: { Colors: V4f }) => ({ Colors: i.Colors }));
const baseEffect = effect(baseVS, baseFS);

// clear: transparent black (page background shows through), depth 0 (reversed)
const clear: ClearValues = {
  colors: HashMap.empty<string, V4f>().add("Colors", new V4f(0, 0, 0, 0)),
  depth: 0.0,
};

const transparency = mode === "off" ? undefined : (mode as "wboit" | "abuffer");

// force-CLASSIC leaf (a dummy storage buffer fails heap eligibility): a
// magenta quad on the MVP-uniform shader, main pass — isolates whether the
// classic (ScenePass) path serves uniform.ModelViewProjTrafo at all.
const classicLeaf = {
  ...quad({ color: new V4f(1, 1, 1, 1) }),
  storageBuffers: HashMap.empty<string, any>().add(
    "dummyBuf",
    AVal.constant(IBuffer.fromHost(new Uint32Array([0, 0, 0, 0]))),
  ),
};

mount(root, (
  <RenderControl
    clear={clear}
    {...(transparency !== undefined ? { transparency } : {})}
    onReady={() => {
      status.textContent = `ready (mode=${mode})`;
      (globalThis as any).__ready = true;
    }}
  >
    <Sg
      View={view}
      Proj={proj}
      DepthTest="greater"
      Shader={baseEffect}
    >
      {/* opaque backdrop */}
      <Sg.Quad
        Trafo={[Sg.translate(new V3d(0, 0, 0)), Sg.scale(new V3d(4, 3, 1))]}
        Color={new V4f(1, 0, 0, 1)}
      />
      {/* heap-eligible transparent quad */}
      {!new URLSearchParams(location.search).has("noGreen") && Sg.transparent(
        <Sg.Quad
          Trafo={[Sg.translate(new V3d(-2, 0, 2)), Sg.scale(new V3d(2, 2.4, 1))]}
          Color={new V4f(0, 1, 0, 0.5)}
        />,
      )}
      {/* force-classic MVP quad, MAIN pass (magenta, top-right corner) */}
      <Sg
        Shader={mvpEffect}
        Uniform={{ u_color: AVal.constant(new V4f(1, 0, 1, 1)) }}
        Trafo={[Sg.translate(new V3d(3, 2.4, 5)), Sg.scale(new V3d(0.7, 0.6, 1))]}
      >
        {classicLeaf as any}
      </Sg>
      {/* transparent quad via the ModelViewProjTrafo uniform path */}
      {!new URLSearchParams(location.search).has("noBlue") && Sg.transparent(
        <Sg Shader={new URLSearchParams(location.search).has("bluebase") ? baseEffect : mvpEffect} Uniform={{ u_color: AVal.constant(new V4f(0, 0.3, 1, 0.5)) }}>
          <Sg.Quad
            Trafo={[Sg.translate(new V3d(0, 0, 4)), Sg.scale(new V3d(2, 1.8, 1))]}
            Color={new V4f(0, 0.3, 1, 0.5)}
          />
        </Sg>,
      )}
    </Sg>
  </RenderControl>
));
