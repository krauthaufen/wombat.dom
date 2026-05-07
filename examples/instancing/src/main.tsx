// Auto-instancing demo — port of Aardvark's
// `25 - AutomaticInstancing/Program.fs` to wombat.dom.
//
// Builds a "coordinate cross" gizmo (red X, green Y, blue Z axis,
// each a cylinder + cone tip, plus a white centre sphere). The
// gizmo gets replicated 1024 times at random trafos via
// `Sg.instancedTrafos`. A toggle button switches between INSTANCED
// (one draw call per primitive geometry) and NON-INSTANCED (1024
// individual nodes — same trafos, same lighting, identical pixels).

import { mount } from "@aardworx/wombat.dom";
import { effect } from "@aardworx/wombat.shader";
import {
  DefaultSurfaces,
  OrbitController,
  RenderControl,
  Sg,
  aspectFromViewport,
  perspective,
} from "@aardworx/wombat.dom/scene";
import { AVal, HashMap, cval, transact } from "@aardworx/wombat.adaptive";
import { V3d, V3f, V4f, Trafo3d } from "@aardworx/wombat.base";
import type { ClearValues } from "@aardworx/wombat.rendering/core";

const root = document.getElementById("app")!;
const status = document.getElementById("status")!;

// ─── error reporting in the status bar ───────────────────────────────
window.addEventListener("error", (e) => {
  status.textContent = "error: " + (e.error?.message ?? e.message);
  status.style.color = "#ff7777";
});
window.addEventListener("unhandledrejection", (e) => {
  status.textContent = "promise rejected: " + (e.reason?.message ?? String(e.reason));
  status.style.color = "#ff7777";
});

// ─── deterministic RNG so instanced/non-instanced run on the same
// trafo set (pixel-for-pixel comparable) ─────────────────────────────
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}
const rnd = lcg(0x42424242);

const COUNT = 1024;
const RANGE = 6.0;

function unitDir(): V3d {
  for (;;) {
    const x = rnd() * 2 - 1;
    const y = rnd() * 2 - 1;
    const s = x * x + y * y;
    if (s >= 1) continue;
    const z = 1 - 2 * s;
    const r = 2 * Math.sqrt(1 - s);
    return new V3d(x * r, y * r, z);
  }
}
const trafos: Trafo3d[] = [];
for (let i = 0; i < COUNT; i++) {
  const axis  = unitDir();
  const angle = rnd() * Math.PI;
  const pos   = new V3d(
    (rnd() * 2 - 1) * RANGE,
    (rnd() * 2 - 1) * RANGE,
    (rnd() * 2 - 1) * RANGE,
  );
  trafos.push(Trafo3d.rotation(axis, angle).mul(Trafo3d.translation(pos)));
}

// ─── one coordinate-cross gizmo ──────────────────────────────────────
//
// wombat.dom primitives are unit-sized:
//   * `<Sg.Cylinder/>` — radius 1, axis (0,0,0)→(0,0,1)
//   * `<Sg.Cone/>`     — apex (0,0,0), base disc at z=1
//   * `<Sg.Sphere/>`   — radius 1, centred at origin
// Scale + translate via `Trafo` to get the desired arrow shape.

const AXIS_LEN  = 1.0;
const AXIS_RAD  = 0.05;
const TIP_LEN   = 0.2;
const TIP_RAD   = 0.10;
const CENTER_RAD = 0.10;

function arrow(color: V4f, intoZ: V3d): JSX.Element {
  const orient = Trafo3d.rotateInto(new V3d(0, 0, 1), intoZ);
  // Cone built apex-down with base at z=1; we want apex up at z=AXIS_LEN+TIP_LEN.
  // Trafo: scale(TIP_RAD, TIP_RAD, -TIP_LEN) then translate(0,0,AXIS_LEN+TIP_LEN).
  const tipTrafo = Trafo3d.scaling(TIP_RAD, TIP_RAD, -TIP_LEN)
    .mul(Trafo3d.translation(new V3d(0, 0, AXIS_LEN + TIP_LEN)));
  return (
    <Sg Trafo={orient}>
      <Sg.Cylinder Color={color} Trafo={Trafo3d.scaling(AXIS_RAD, AXIS_RAD, AXIS_LEN)} />
      <Sg.Cone Color={color} Trafo={tipTrafo} />
    </Sg>
  );
}

const RED   = new V4f(1.0, 0.20, 0.20, 1);
const GREEN = new V4f(0.20, 1.0, 0.20, 1);
const BLUE  = new V4f(0.20, 0.40, 1.0, 1);
const WHITE = new V4f(0.95, 0.95, 0.95, 1);

const coordinateCross = (
  <Sg>
    {arrow(RED,   new V3d(1, 0, 0))}
    {arrow(GREEN, new V3d(0, 1, 0))}
    {arrow(BLUE,  new V3d(0, 0, 1))}
    <Sg.Sphere Trafo={Trafo3d.scaling(CENTER_RAD, CENTER_RAD, CENTER_RAD)} Color={WHITE} />
  </Sg>
);

// ─── two scene variants ──────────────────────────────────────────────
const trafosAval = AVal.constant<readonly Trafo3d[]>(trafos);

const instancedScene = Sg.instancedTrafos(trafosAval)(coordinateCross);

const nonInstancedChildren = trafos.map((t, i) => (
  <Sg key={i} Trafo={t}>{coordinateCross}</Sg>
));

// ─── UI: toggle ──────────────────────────────────────────────────────
const isInstanced = cval(true);
const btn = document.createElement("button");
btn.style.cssText = "position:fixed; top:12px; right:12px; z-index:10; padding:6px 10px; font: 12px system-ui; background:#222; color:#ddd; border:1px solid #444; border-radius:6px; cursor:pointer;";
const setBtn = (): void => {
  btn.textContent = `mode: ${isInstanced.value ? `instanced (~6 draws)` : `non-instanced (${COUNT * 7} draws)`} (toggle)`;
};
setBtn();
btn.onclick = () => {
  transact(() => { isInstanced.value = !isInstanced.value; });
  setBtn();
};
document.body.appendChild(btn);

// ─── camera + render ─────────────────────────────────────────────────
const ctl = OrbitController.create({
  radius: 18,
  phi: Math.PI / 5,
  theta: 0.5,
});
const clear: ClearValues = {
  colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0.06, 0.07, 0.10, 1)),
  depth: 1,
};

mount(root, (
  <RenderControl
    clear={clear}
    onReady={({ canvas, time }) => {
      ctl.attach(canvas, time);
      status.textContent = "ready — drag to rotate, wheel to zoom; toggle modes top-right";
    }}
  >
    <Sg
      View={ctl.view}
      Uniform={{ LightLocation: AVal.constant(new V3f(20, 20, 20)) }}
      Shader={effect(DefaultSurfaces.trafo(), DefaultSurfaces.simpleLighting())}
      Proj={perspective({
        fovInRadians: Math.PI / 3,
        aspect: aspectFromViewport(RenderControl.viewport),
        near: 0.1,
        far: 200,
      })}
    >
      <Sg Active={isInstanced}>{instancedScene}</Sg>
      <Sg Active={isInstanced.map(v => !v)}>{nonInstancedChildren}</Sg>
    </Sg>
  </RenderControl>
));
