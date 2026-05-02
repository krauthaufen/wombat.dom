// primitives-showcase — every built-in primitive rendered with the
// camera-headlight Blinn-Phong surface. Orbit-controlled, with
// double-tap fly-to on every primitive (auto-Intersectable wires the
// canonical local-space shape, so picking is geometric — no rgba32f
// readback required for any of the primitives below).

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
import { V3d, V4f } from "@aardworx/wombat.base";
import type { ClearValues } from "@aardworx/wombat.rendering/core";
import type { SceneEvent } from "@aardworx/wombat.dom/scene";

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
  radius: 16,
  phi: Math.PI / 5,
  theta: 0.5,
});

const flyTarget = (e: SceneEvent): void => {
  ctl.flyTo(e.worldPos);
};

const clear: ClearValues = {
  colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0.07, 0.07, 0.08, 1.0)),
  depth: 1.0,
};

// Six primitives, evenly spaced along X. Solids on z = 0, wireframes
// on z = 3. Cylinders/cones extend up the +Z axis in their local
// space; pre-translate them so each one is centred on its slot.
const xBox = -7.5, xSph = -4.5, xCyl = -1.5, xCon = 1.5, xTet = 4.5, xOct = 7.5;

const cBox = new V4f(1.0, 0.55, 0.25, 1);
const cSph = new V4f(0.45, 0.75, 1.0, 1);
const cCyl = new V4f(0.95, 0.85, 0.35, 1);
const cCon = new V4f(0.55, 0.95, 0.55, 1);
const cTet = new V4f(0.95, 0.45, 0.75, 1);
const cOct = new V4f(0.65, 0.55, 0.95, 1);
const dim = (c: V4f): V4f => new V4f(c.x * 0.85, c.y * 0.85, c.z * 0.85, 1);

mount(root, (
  <RenderControl
    clear={clear}
    onReady={({ canvas, time }) => {
      ctl.attach(canvas, time);
      status.textContent = "ready — drag to rotate, wheel to zoom, double-tap a primitive to fly to it";
    }}
  >
    <Sg
      View={ctl.view}
      Shader={DefaultSurfaces.headlight()}
      Proj={perspective({
        fovInRadians: Math.PI / 3,
        aspect: aspectFromViewport(RenderControl.viewport),
        near: 0.1,
        far: 200,
      })}
      OnDoubleTap={flyTarget}
    >
      {/* ---- solids on z=0 ---- */}
      <Sg.Box           Trafo={Sg.translate(new V3d(xBox, 0, 0))} Color={cBox} />
      <Sg.Sphere        Trafo={Sg.translate(new V3d(xSph, 0, 0))} Color={cSph} />
      {/* Cylinders/cones live along the +Z axis; translate down by
          half so each one is centred on its slot. */}
      <Sg.Cylinder      Trafo={Sg.translate(new V3d(xCyl, 0, -0.5))} Color={cCyl} />
      <Sg.Cone          Trafo={Sg.translate(new V3d(xCon, 0, -0.5))} Color={cCon} />
      <Sg.Tetrahedron   Trafo={Sg.translate(new V3d(xTet, 0, 0))} Color={cTet} />
      <Sg.Octahedron    Trafo={Sg.translate(new V3d(xOct, 0, 0))} Color={cOct} />

      {/* ---- wireframes on z=3 ---- */}
      <Sg.WireBox         Trafo={Sg.translate(new V3d(xBox, 0, 3))} Color={dim(cBox)} />
      <Sg.WireSphere      Trafo={Sg.translate(new V3d(xSph, 0, 3))} Color={dim(cSph)} />
      <Sg.WireCylinder    Trafo={Sg.translate(new V3d(xCyl, 0, 2.5))} Color={dim(cCyl)} />
      <Sg.WireCone        Trafo={Sg.translate(new V3d(xCon, 0, 2.5))} Color={dim(cCon)} />
      <Sg.WireTetrahedron Trafo={Sg.translate(new V3d(xTet, 0, 3))} Color={dim(cTet)} />
      <Sg.WireOctahedron  Trafo={Sg.translate(new V3d(xOct, 0, 3))} Color={dim(cOct)} />
    </Sg>
  </RenderControl>
));
