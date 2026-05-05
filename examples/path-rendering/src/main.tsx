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
import { HashMap, cval, transact } from "@aardworx/wombat.adaptive";
import { V3d, V4f } from "@aardworx/wombat.base";
import { Font } from "@aardworx/wombat.base/font";
import type { ClearValues } from "@aardworx/wombat.rendering/core";

import greatVibesUrl from "./great-vibes.ttf?url";
// Lato exercises plenty of L commands (straight stems, crossbars,
// sharp corners) — much better than Great Vibes for debugging the
// line-edge AA ribbons.
import latoUrl from "./lato.ttf?url";
import robotoMonoUrl from "./roboto-mono.ttf?url";

// ---------------------------------------------------------------------------

const root = document.getElementById("app")!;
const status = document.getElementById("status")!;
status.textContent = "starting…";

// Debug AA mode toggle: render both subtrees, gate each with an
// `Active={cval}` so we just flip booleans on click — no remount,
// camera state survives.
const aaIsBlend = cval(true);
const aaBtn = document.createElement("button");
aaBtn.style.cssText = "position:fixed; top:12px; right:12px; z-index:10; padding:6px 10px; font: 12px system-ui; background:#222; color:#ddd; border:1px solid #444; border-radius:6px; cursor:pointer;";
const setAaBtn = (): void => { aaBtn.textContent = `aa: ${aaIsBlend.value ? "alpha-blending" : "none"} (toggle)`; };
setAaBtn();
aaBtn.onclick = () => {
  transact(() => { aaIsBlend.value = !aaIsBlend.value; });
  setAaBtn();
};
document.body.appendChild(aaBtn);

// AA ramp width slider (only meaningful in alpha-blending mode).
const aaWidthPx = cval(1.0);
const aaWidthBox = document.createElement("div");
aaWidthBox.style.cssText = "position:fixed; top:50px; right:12px; z-index:10; padding:6px 10px; font: 12px system-ui; background:#222; color:#ddd; border:1px solid #444; border-radius:6px; display:flex; gap:8px; align-items:center;";
const aaWidthLbl = document.createElement("span");
const slider = document.createElement("input");
slider.type = "range";
slider.min = "0.5";
slider.max = "10";
slider.step = "0.1";
slider.value = "1";
slider.style.cssText = "width:140px;";
const setAaWidthLbl = (): void => {
  aaWidthLbl.textContent = `AA: ${aaWidthPx.value.toFixed(1)} px`;
};
setAaWidthLbl();
slider.oninput = () => {
  transact(() => { aaWidthPx.value = parseFloat(slider.value); });
  setAaWidthLbl();
};
aaWidthBox.appendChild(aaWidthLbl);
aaWidthBox.appendChild(slider);
document.body.appendChild(aaWidthBox);

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

const font           = await Font.load(greatVibesUrl);
const latoFont       = await Font.load(latoUrl);
const robotoMonoFont = await Font.load(robotoMonoUrl);

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

// Stress paragraph: long Lorem-Ipsum-style English to exercise full
// glyph cache + per-fragment SDF cost across many glyphs at once.
const STRESS_LINES = [
  "The quick brown fox jumps over the lazy dog 0123456789",
  "Sphinx of black quartz, judge my vow! Pack my box with",
  "five dozen liquor jugs. How vexingly quick daft zebras",
  "jump. Bright vixens jump; dozy fowl quack. Waltz, bad",
  "nymph, for quick jigs vex. Glib jocks quiz nymph to vex",
  "dwarf. Two driven jocks help fax my big quiz. The five",
  "boxing wizards jump quickly. Jaded zombies acted quaintly",
  "but kept driving their oxen forward. A wizards job is to",
  "vex chumps quickly in fog. Watch Jeopardy! Alex Trebek's",
  "fun TV quiz game. Mr. Jock, TV quiz PhD., bags few lynx.",
  "Bawds jog, flick quartz, vex nymphs. Big fjords vex quick",
  "waltz nymph, for jigs vex chubd. Foxy parsons quiz and jam",
];

// Each line scales to 0.3 em-units tall and stacks at 1.25 × that
// (typical typographic line height) for proper Lato spacing.
const TEXT_SCALE = 0.3;
const LINE_STEP  = TEXT_SCALE * 1.25;
const FIRST_Y    = ((STRESS_LINES.length - 1) / 2) * LINE_STEP;
const rows = (aa: "none" | "alpha-blending") =>
  STRESS_LINES.map((line, i) => (
    <Sg.Text
      key={`stress-${i}`}
      font={latoFont}
      text={line}
      align="center"
      aa={aa}
      aaWidthPx={aaWidthPx}
      Color={i % 2 === 0 ? cream : orange}
      Trafo={[
        Sg.translate(new V3d(0, FIRST_Y - i * LINE_STEP, 0)),
        Sg.scale(TEXT_SCALE),
      ]}
    />
  ));

const aaIsNone = aaIsBlend.map((b) => !b);

mount(root, (
  <RenderControl
    clear={clear}
    attach={{ devicePixelRatio: typeof window !== "undefined" ? window.devicePixelRatio : 1 }}
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
      <Sg Active={aaIsBlend}>{rows("alpha-blending")}</Sg>
      <Sg Active={aaIsNone}>{rows("none")}</Sg>
    </Sg>
  </RenderControl>
));
