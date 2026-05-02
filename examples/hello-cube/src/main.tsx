// hello-cube — first end-to-end demo of wombat.dom + the scene
// layer. Renders a vertex-coloured rainbow cube using
// `DefaultSurfaces.basic`, with an orbit controller wired to the
// canvas for drag-rotate / wheel-zoom interaction.
//
// `<Sg.Adaptive value={…}>` would also work for the projection
// swap, but `Sg.delay` reads cleaner here — we just want the
// fully-accumulated viewport at this scope and to build the
// `Proj` from it.

import { mount } from "@aardworx/wombat.dom";
import {
  DefaultSurfaces,
  OrbitController,
  RenderControl,
  Sg,
  aspectFromViewport,
  perspective,
} from "@aardworx/wombat.dom/scene";
import { HashMap } from "@aardworx/wombat.adaptive";
import { Box3d, Intersectable, V3d, V4f } from "@aardworx/wombat.base";
import type { ClearValues } from "@aardworx/wombat.rendering/core";
import type { SceneEvent } from "@aardworx/wombat.dom/scene";

const root = document.getElementById("app")!;
const status = document.getElementById("status")!;
status.textContent = "starting…";

// Surface unhandled errors / warnings into the on-page status
// div — handy when the dev tools aren't reachable (e.g. mobile).
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

// Orbit controller — listeners + per-frame integration are attached
// after the canvas exists (in `onReady`); the controller is driven
// by `info.time` (the per-frame clock).
const ctl = OrbitController.create({
  radius: 5,
  phi: Math.PI / 6,
  theta: 0.4,
});

// Double-tap on the cube re-centres the orbit on the actual pick
// point and zooms in / out with the Tanh easing. `e.worldPos` is the
// canonical world-space hit position from the SceneEventLocation.
const flyTarget = (e: SceneEvent): void => {
  ctl.flyTo(e.worldPos);
};

// Clear the framebuffer to a dark grey before rendering each frame.
// Attachment name matches the canvas signature configured by
// `<RenderControl>` (which uses `colorAttachmentName: "outColor"`
// by default to match wombat.shader's fragment-output convention).
const clear: ClearValues = {
  colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0.07, 0.07, 0.08, 1.0)),
  depth: 1.0,
};

mount(root, (
  <RenderControl
    clear={clear}
    onReady={({ canvas, time }) => {
      ctl.attach(canvas, time);
      status.textContent = "ready — drag to rotate, wheel to zoom, double-tap to fly";
    }}
  >
    <Sg
      View={ctl.view}
      Shader={DefaultSurfaces.basic()}
      Proj={perspective({
        fovInRadians: Math.PI / 3,
        aspect: aspectFromViewport(RenderControl.viewport),
        near: 0.1,
        far: 100,
      })}
      OnDoubleTap={flyTarget}
    >
      {/* Cube A — picked via the rgba32f pickId attachment. */}
      <Sg.Box Trafo={Sg.translate(new V3d(-1.5, 0, 0))} />
      {/* Cube B — picked geometrically via an Intersectable. The
          intersectable is local to its scope (bbox `[-1,1]³`); the
          enclosing Trafo translates it to (+1.5, 0, 0) and the
          dispatcher transforms the world-space pick ray into local
          space before testing — no global pre-computed world bbox
          needed. */}
      <Sg.Box
        Trafo={Sg.translate(new V3d(1.5, 0, 0))}
        Intersectable={Intersectable.box(Box3d.fromMinMax(new V3d(-1, -1, -1), new V3d(1, 1, 1)))}
      />
    </Sg>
  </RenderControl>
));
