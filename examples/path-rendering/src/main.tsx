// path-rendering demo — Sg.Text on the floor with the orbit camera.
//
// `<Sg.Text>` does the lifting: TTF → PathSegment lowering, glyph
// caching, per-instance layout, the Loop-Blinn surface effect and
// the auto-flip vertex shader. We stack three runs of varying
// alignment + a small primitive-shape line so we can sanity-check
// scaling, kerning, and the per-instance offset path.

import { mount } from "@aardworx/wombat.dom";
import {
  RenderControl,
  Sg,
  OrbitController,
  aspectFromViewport,
  perspective,
} from "@aardworx/wombat.dom/scene";
import type { SceneEvent } from "@aardworx/wombat.dom/scene";
import { HashMap } from "@aardworx/wombat.adaptive";
import { V3d, V4f } from "@aardworx/wombat.base";
import { Font } from "@aardworx/wombat.base/font";
import type { ClearValues } from "@aardworx/wombat.rendering/core";

import greatVibesUrl from "./great-vibes.ttf?url";

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
  colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0.04, 0.07, 0.10, 1)),
  depth: 1.0,
};

const font = await Font.load(greatVibesUrl);

const ctl = OrbitController.create({
  radius: 5,
  phi: -Math.PI / 2,
  theta: Math.PI / 4,
});

const flyToHit = (e: SceneEvent): void => {
  ctl.flyTo(e.worldPos);
};

// 1 em → 1 world unit; user controls visual size via Trafo.
const orange = new V4f(0.9, 0.51, 0.255, 1);
const cream  = new V4f(0.95, 0.88, 0.78, 1);

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
      OnDoubleTap={flyToHit}
      PixelSnapRadius={8}
    >
      {/* Three text runs lying on the floor. Scale shrinks the
          em-height; alignment translation is absorbed into ModelTrafo
          inside Sg.Text. */}
      <Sg.Text
        font={font} text="Hello" align="center" aa="none"
        Color={orange}
        Trafo={[Sg.translate(new V3d(0, 0.5, 0)), Sg.scale(0.0015)]}
      />
      <Sg.Text
        font={font} text="& wombat" align="center" aa="none"
        Color={cream}
        Trafo={[Sg.translate(new V3d(0, -0.5, 0)), Sg.scale(0.0015)]}
      />
      <Sg.Text
        font={font} text="left-anchored" align="left" aa="none"
        Color={orange}
        Trafo={[Sg.translate(new V3d(-1.5, -1.5, 0)), Sg.scale(0.0008)]}
      />
    </Sg>
  </RenderControl>
));
