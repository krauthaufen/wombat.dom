// warp-portal — the DemoCubes-style acceptance demo for offscreen
// ("portal") picking.
//
// An INNER scene (3×3 grid of flat-colored clickable boxes, its own
// camera) renders offscreen via `renderToPickable`. The OUTER scene is
// one fullscreen quad whose fragment shader warps the uv with an
// animated ripple, samples the offscreen texture, and writes the
// sampled uv as `PickContextCoord`. `PickContext={pick}` mounts the
// inner pick handle on the quad — so hover / tap / click on the WARPED
// image resolve to the inner boxes (hover highlights, tap deletes);
// pixels where the inner scene misses fall through to the quad itself
// ("portal-bg").
//
// URL params:
//   ?amp=0.035   warp amplitude in uv units (0 = undistorted)
//   ?freeze=1    freeze the warp phase (deterministic for the driver)
//   ?t0=0.7      frozen phase value
//
// The page exposes hooks for the check.mjs driver:
//   window.__ready              — scene live
//   window.__log                — [{kind, target}] event records
//   window.__boxScreen()        — {name: [cssX, cssY]} outer screen
//                                 coords of live box centers (amp=0
//                                 mapping — analytic, no readback)
//   window.__live()             — live box names

import { mount } from "@aardworx/wombat.dom";
import {
  RenderControl,
  Sg,
  box,
  quad,
  lookAt,
  perspective,
  renderToPickable,
} from "@aardworx/wombat.dom/scene";
import { AVal, cset, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4d, V4f } from "@aardworx/wombat.base";
import { Runtime } from "@aardworx/wombat.rendering";
import type { SgNode } from "@aardworx/wombat.dom/scene";

import { flatSurface, portalSurface } from "./effects.js";

const root = document.getElementById("app")!;
const status = document.getElementById("status")!;
const logEl = document.getElementById("log")!;
const setStatus = (m: string, err = false): void => {
  status.textContent = m;
  status.style.color = err ? "#ff7777" : "#888";
};
window.addEventListener("error", (e) => setStatus("error: " + (e.error?.message ?? e.message), true));
window.addEventListener("unhandledrejection", (e) => setStatus("rejected: " + (e.reason?.message ?? String(e.reason)), true));

const params = new URLSearchParams(location.search);
const AMP = params.get("amp") !== null ? parseFloat(params.get("amp")!) : 0.035;
const FREEZE = params.get("freeze") === "1";
const T0 = params.get("t0") !== null ? parseFloat(params.get("t0")!) : 0.7;

// ─── driver hooks ───────────────────────────────────────────────────────

interface LogEntry { kind: string; target: string }
const log: LogEntry[] = [];
const record = (kind: string, target: string): void => {
  log.push({ kind, target });
  logEl.textContent = log.slice(-12).map(l => `${l.kind} ${l.target}`).join("\n");
};
(window as unknown as { __log: LogEntry[] }).__log = log;

// ─── inner scene: 3×3 clickable boxes ───────────────────────────────────

const GRID = 3;
const SPACING = 2.6;
const BASE_COLORS: V4f[] = [
  new V4f(0.9, 0.2, 0.2, 1), new V4f(0.2, 0.9, 0.2, 1), new V4f(0.2, 0.35, 0.95, 1),
  new V4f(0.95, 0.85, 0.2, 1), new V4f(0.85, 0.3, 0.9, 1), new V4f(0.25, 0.9, 0.9, 1),
  new V4f(0.95, 0.55, 0.2, 1), new V4f(0.6, 0.6, 0.6, 1), new V4f(0.5, 0.25, 0.9, 1),
];
const HILITE = new V4f(1, 1, 1, 1);

const innerSize = cval({ width: 512, height: 512 });
const innerView = lookAt({ eye: new V3d(0, 0, 14), target: V3d.zero, up: new V3d(0, 1, 0) });
const innerProj = perspective({ fovInRadians: Math.PI / 3, aspect: 1, near: 0.1, far: 100 });

interface BoxInfo { name: string; center: V3d; node: SgNode }
const liveBoxes = new Map<string, BoxInfo>();
const boxSet = cset<SgNode>();

for (let j = 0; j < GRID; j++) {
  for (let i = 0; i < GRID; i++) {
    const idx = j * GRID + i;
    const name = `box-${idx}`;
    const base = BASE_COLORS[idx]!;
    const color = cval(base);
    const center = new V3d((i - 1) * SPACING, (j - 1) * SPACING, 0);
    const holder: { node?: SgNode } = {};
    const node = (
      <Sg
        Trafo={Trafo3d.translation(center)}
        OnPointerEnter={() => { record("enter", name); transact(() => { color.value = HILITE; }); }}
        OnPointerLeave={() => { record("leave", name); transact(() => { color.value = base; }); }}
        OnTap={() => {
          record("tap", name);
          const info = liveBoxes.get(name);
          if (info !== undefined) {
            liveBoxes.delete(name);
            transact(() => boxSet.remove(info.node));
          }
        }}
      >
        {box({ size: new V3d(0.9, 0.9, 0.9), color: color as aval<V4f> })}
      </Sg>
    ) as SgNode;
    holder.node = node;
    liveBoxes.set(name, { name, center, node });
    transact(() => boxSet.add(node));
  }
}

const innerScene = (
  <Sg View={innerView} Proj={innerProj} Shader={flatSurface}>
    {Sg.unordered(boxSet)}
  </Sg>
) as SgNode;

// ─── analytic outer-screen positions (amp = 0 mapping) ─────────────────
// Box center → inner NDC → (via the portal's identity uv chain) outer
// CSS coords. Used by the driver to click exactly on box centers.

function boxScreen(): Record<string, [number, number]> {
  const canvas = document.querySelector("canvas");
  if (canvas === null) return {};
  const W = canvas.clientWidth, H = canvas.clientHeight;
  const view = AVal.force(innerView).forward;
  const proj = AVal.force(innerProj).forward;
  const out: Record<string, [number, number]> = {};
  for (const info of liveBoxes.values()) {
    const vp = view.transform(new V4d(info.center.x, info.center.y, info.center.z, 1));
    const cp = proj.transform(vp);
    const nx = cp.x / cp.w, ny = cp.y / cp.w;
    out[info.name] = [(nx * 0.5 + 0.5) * W, (0.5 - 0.5 * ny) * H];
  }
  return out;
}
(window as unknown as { __boxScreen: typeof boxScreen }).__boxScreen = boxScreen;
(window as unknown as { __live: () => string[] }).__live = () => [...liveBoxes.keys()];

// ─── boot: shared device/runtime, offscreen pickable, outer scene ──────

async function main(): Promise<void> {
  if (!("gpu" in navigator)) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter === null) throw new Error("no GPU adapter");
  const device = await adapter.requestDevice();
  const runtime = new Runtime({ device });

  const pickable = renderToPickable(runtime, device, innerScene, {
    size: innerSize,
    view: innerView,
    proj: innerProj,
    label: "portal",
  });

  const timeU: aval<number> = FREEZE
    ? AVal.constant(T0)
    : RenderControl.time.map((t) => t / 1000);

  const outerScene = (
    <Sg View={AVal.constant(Trafo3d.identity)} Proj={AVal.constant(Trafo3d.identity)}>
      <Sg
        Shader={portalSurface}
        Uniform={{
          PortalTex: pickable.color(),
          Time: timeU,
          WarpAmp: AVal.constant(AMP),
        }}
        PickContext={pickable.pick}
        OnPointerEnter={() => record("enter", "portal-bg")}
        OnPointerLeave={() => record("leave", "portal-bg")}
        OnTap={() => record("tap", "portal-bg")}
      >
        {quad()}
      </Sg>
    </Sg>
  ) as SgNode;

  mount(root, (
    <RenderControl
      device={device}
      runtime={runtime}
      scene={outerScene}
      onReady={() => {
        setStatus(`portal live — amp=${AMP}${FREEZE ? ` frozen t=${T0}` : ""}. Hover to highlight, tap to delete.`);
        (window as unknown as { __ready: boolean }).__ready = true;
      }}
    />
  ));
}

void main().catch((e) => setStatus(String(e), true));
