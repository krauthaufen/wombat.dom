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
import { V4f } from "@aardworx/wombat.base";
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
let near = false;
const flyTarget = (e: SceneEvent): void => {
  near = !near;
  ctl.flyTo(e.worldPos, near ? 2.2 : 5.5);
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
    <Sg View={ctl.view} Shader={DefaultSurfaces.basic()}>
      {/* Build a viewport-tracking perspective once we know the
          canvas size — Sg.delay receives the fully-accumulated
          TraversalState (including viewport from the
          `<RenderControl>` shell). */}
      {Sg.delay(state =>
        Sg.proj(
          perspective({
            fovInRadians: Math.PI / 3,
            aspect: aspectFromViewport(state.viewport),
            near: 0.1,
            far: 100,
          }),
          <Sg OnDoubleTap={flyTarget}>
            {Sg.box()}
          </Sg>,
        ),
      )}
    </Sg>
  </RenderControl>
));
