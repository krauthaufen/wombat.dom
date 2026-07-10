// cityview — CadSceneDemo-class viewer: vienna d1-9 through the Sg/heap
// path with picking (hover highlight + tap info), an orbit camera with
// fly-to-on-double-tap, and an fps overlay.
//
// URL params:
//   ?districts=N or ?districts=1,4,7   which districts to load (default 1)
//
// Data: /vienna/d01..d09 (symlinked from ~/arcbench/pkg-linux/vienna) —
// oct32 normals + decoupled C4b colors, expanded to f32 on load (the
// desktop demo path; packed BufferViews are the phone-scale follow-up).

import { mount } from "@aardworx/wombat.dom";
import {
  OrbitController,
  RenderControl,
  Sg,
  aspectFromViewport,
  perspective,
  quad,
  renderToPickable,
  type SceneEvent,
  type SgNode,
} from "@aardworx/wombat.dom/scene";
import { Runtime } from "@aardworx/wombat.rendering";
import { Trafo3d } from "@aardworx/wombat.base";
import { AVal, HashMap, cset, cval, transact } from "@aardworx/wombat.adaptive";
import { V3d } from "@aardworx/wombat.base";
import type { BufferView, DrawCall } from "@aardworx/wombat.rendering/core";
import { ElementType, IBuffer } from "@aardworx/wombat.rendering/core";

import { citySurface, compositeSurface } from "./effects.js";
import { GtaoResource } from "./gtao.js";

const statusEl = document.getElementById("status")!;
const fpsEl = document.getElementById("fps")!;
const infoEl = document.getElementById("info")!;
const setStatus = (m: string, err = false): void => {
  statusEl.textContent = m;
  statusEl.style.color = err ? "#ff7777" : "#8a9";
};
window.addEventListener("error", (e) => setStatus("error: " + (e.error?.message ?? e.message), true));
window.addEventListener("unhandledrejection", (e) => setStatus("rejected: " + (e.reason?.message ?? String(e.reason)), true));

// ─── page-zoom suppression (iOS Safari) ────────────────────────────────
// The controller already guards the canvas, but Safari page-zooms when a
// pinch STARTS partly outside it (or via its proprietary gesture events,
// which fire on the ancestor). The app is a fullscreen canvas with no
// scrolling, so kill every browser zoom/scroll gesture document-wide:
//   - gesture* events: Safari's pinch pipeline (preventDefault = no zoom)
//   - multi-touch / scale!=1 touchmove: pinch fallback path
//   - dblclick: double-tap zoom remnants
//   - ctrl+wheel: desktop pinch-to-zoom trackpad gesture
for (const ev of ["gesturestart", "gesturechange", "gestureend"]) {
  document.addEventListener(ev, (e) => e.preventDefault(), { passive: false });
}
document.addEventListener("touchstart", (e) => {
  if (e.touches.length > 1) e.preventDefault();
}, { passive: false });
document.addEventListener("touchmove", (e) => {
  if (e.touches.length > 1 || ((e as unknown as { scale?: number }).scale ?? 1) !== 1) e.preventDefault();
}, { passive: false });
document.addEventListener("dblclick", (e) => e.preventDefault(), { passive: false });
document.addEventListener("wheel", (e) => { if (e.ctrlKey) e.preventDefault(); }, { passive: false });

const P = new URLSearchParams(location.search);
const PART_CAP = P.get("parts") !== null ? parseInt(P.get("parts")!, 10) | 0 : 0;
const AO = P.get("ao") !== "0";
// Mobile caps at dpr=2 (native 3x is 2.25x the pixels of 2x for barely
// visible gain; dpr=1 is noticeably soft) — desktop stays native.
// ?dpr= overrides either way.
const DPR = P.get("dpr") !== null
  ? Math.max(0.25, parseFloat(P.get("dpr")!))
  : (typeof window !== "undefined" && window.matchMedia("(pointer: coarse)").matches
      ? Math.min(window.devicePixelRatio ?? 1, 2)
      : (window.devicePixelRatio ?? 1));
const AO_RADIUS = P.get("aor") !== null ? parseFloat(P.get("aor")!) : 4;
const AO_INTENSITY = P.get("aoi") !== null ? parseFloat(P.get("aoi")!) : 1.4;
const DISTRICT_LIST: number[] = (() => {
  const s = P.get("districts");
  if (s === null) return [1];
  if (s.includes(",")) return [...new Set(s.split(",").map(x => Math.min(9, Math.max(1, parseInt(x, 10) | 0))))];
  const n = Math.min(9, Math.max(1, parseInt(s, 10) | 0));
  return Array.from({ length: n }, (_, i) => i + 1);
})();

// ─── loader (renderbench's loadD19, expanded-f32 path) ─────────────────

interface Part { readonly v0: number; readonly vn: number; readonly c0?: number; readonly cn?: number }
interface Manifest {
  readonly radius: number;
  readonly buildings?: readonly Part[];
  readonly trees?: readonly Part[];
  readonly ground?: readonly Part[];
  readonly water?: readonly Part[];
}
const KINDS = ["buildings", "trees", "ground", "water"] as const;
type Kind = typeof KINDS[number];

interface CityPart { v0: number; vn: number; c0: number; cn: number; kind: Kind; district: number }

// Data-version cache-buster: the vienna files keep stable URLs, so a
// redeployed ASSET (same names, new bytes) would otherwise be served
// from the browser's HTTP cache indefinitely (nginx sends no
// cache-control). Bump on every data redeploy.
const DATA_V = "?v=3";

async function gunzipMaybe(r: Response): Promise<ArrayBuffer> {
  const buf = await r.arrayBuffer();
  const u8 = new Uint8Array(buf, 0, 2);
  if (u8[0] === 0x1f && u8[1] === 0x8b) {
    const ds = new Response(buf).body!.pipeThrough(new DecompressionStream("gzip"));
    return new Response(ds).arrayBuffer();
  }
  return buf;
}
// SPA dev servers answer missing files with index.html (200) — treat
// an html content-type as a miss so the raw fallback still runs.
const isReal = (r: Response): boolean => r.ok && !(r.headers.get("content-type") ?? "").includes("html");
async function fetchBin(url: string): Promise<ArrayBuffer> {
  const gz = await fetch(`${url}.gz${DATA_V}`);
  if (isReal(gz)) return gunzipMaybe(gz);
  const r = await fetch(`${url}${DATA_V}`);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.arrayBuffer();
}
async function fetchJson<T>(url: string): Promise<T> {
  const gz = await fetch(`${url}.gz${DATA_V}`);
  if (isReal(gz)) {
    const buf = await gunzipMaybe(gz);
    return JSON.parse(new TextDecoder().decode(buf)) as T;
  }
  return fetch(`${url}${DATA_V}`).then(r => r.json()) as Promise<T>;
}

/** One district's payload — arrays are district-LOCAL (parts index into
 *  them via v0/c0). They become the leaf BufferViews' backing directly:
 *  no merged copies, so the load peak is ONE district's temporaries. */
interface DistrictData {
  district: number;
  parts: CityPart[];
  positions: Float32Array;
  normalsOct: Uint32Array;
  colorsC4b: Uint32Array;
}

async function fetchManifests(): Promise<{ manifests: Manifest[]; totalVerts: number; radius: number }> {
  const manifests: Manifest[] = [];
  for (const n of DISTRICT_LIST) manifests.push(await fetchJson<Manifest>(`vienna/d0${n}/manifest.json`));
  const totalVerts = manifests
    .map((m) => KINDS.flatMap(k => [...(m[k] ?? [])]).reduce((a, p) => Math.max(a, p.v0 + p.vn), 0))
    .reduce((a, b) => a + b, 0);
  const radius = manifests.reduce((a, m) => Math.max(a, m.radius), 0);
  return { manifests, totalVerts, radius };
}

async function fetchDistrict(di: number, manifest: Manifest): Promise<DistrictData> {
  const n = DISTRICT_LIST[di]!;
  const d = `vienna/d0${n}`;
  const positions = new Float32Array(await fetchBin(`${d}/positions.bin`));
  const normalsOct = new Uint32Array(await fetchBin(`${d}/normals.bin`));
  const colorsC4b = new Uint32Array(await fetchBin(`${d}/colors.bin`));
  const parts: CityPart[] = [];
  for (const kind of KINDS) {
    for (const p of manifest[kind] ?? []) {
      parts.push({ v0: p.v0, vn: p.vn, c0: p.c0 ?? p.v0, cn: p.cn ?? p.vn, kind, district: n });
    }
  }
  return { district: n, parts, positions, normalsOct, colorsC4b };
}

// ─── scene ──────────────────────────────────────────────────────────────

const selectedPick = cval(0);
const fmt = new Intl.NumberFormat("en-US");

function districtLeaves(dd: DistrictData): SgNode[] {
  // Per-part attribute slices (subarray views over the district
  // arrays; zero copies host-side). firstVertex stays 0 so every leaf
  // is heap-ELIGIBLE — the heap stages each part once into the arena
  // and the megacall collapses all draws into one indirect call.
  const showInfo = (p: CityPart, pickId: number): void => {
    infoEl.style.display = "block";
    infoEl.textContent =
      `${p.kind.replace(/s$/, "")}  (district ${p.district})\n` +
      `vertices  ${fmt.format(p.vn)}\n` +
      `pickId    ${pickId}`;
  };

  const useParts = PART_CAP > 0 ? dd.parts.slice(0, PART_CAP) : dd.parts;
  return useParts.map((p) => {
    const dc: DrawCall = {
      kind: "non-indexed",
      vertexCount: p.vn, instanceCount: 1,
      firstVertex: 0, firstInstance: 0,
    };
    const vertexAttributes = HashMap.empty<string, BufferView>()
      .add("Positions", {
        buffer: AVal.constant(IBuffer.fromHost(dd.positions.subarray(p.v0 * 3, (p.v0 + p.vn) * 3))),
        elementType: ElementType.V3f,
      })
      .add("Normals", {
        // raw oct32 — 4 B/vertex, decoded by the heap VS typeId arm
        buffer: AVal.constant(IBuffer.fromHost(dd.normalsOct.subarray(p.v0, p.v0 + p.vn))),
        elementType: ElementType.Oct32,
      })
      .add("Colors", {
        // raw C4b, decoupled: cn=1 = 4-byte singleton broadcast
        buffer: AVal.constant(IBuffer.fromHost(dd.colorsC4b.subarray(p.c0, p.c0 + p.cn))),
        elementType: ElementType.C4b,
      });
    return (
      <Sg
        OnPointerEnter={(e: SceneEvent) => { transact(() => { selectedPick.value = e.pickId; }); }}
        OnPointerLeave={() => { transact(() => { selectedPick.value = 0; }); }}
        OnTap={(e: SceneEvent) => showInfo(p, e.pickId)}
      >
        {Sg.leaf({ vertexAttributes, drawCall: AVal.constant(dc) })}
      </Sg>
    ) as SgNode;
  });
}

function cityRoot(leafSet: import("@aardworx/wombat.adaptive").aset<SgNode>, ctl: OrbitController, proj: import("@aardworx/wombat.adaptive").aval<Trafo3d>): SgNode {
  return (
    <Sg
      View={ctl.view}
      Proj={proj}
      Shader={citySurface}
      Uniform={{ SelectedPick: selectedPick.map(x => x) }}
      ForcePixelPicking={AVal.constant(true)}
      OnDoubleTap={(e: SceneEvent) => ctl.flyTo(e.worldPos)}
      OnTap={() => { /* background tap: no panel change */ }}
    >
      {Sg.unordered(leafSet)}
    </Sg>
  ) as SgNode;
}

// ─── boot ───────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  if (!("gpu" in navigator)) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter === null) throw new Error("no GPU adapter");
  const device = await adapter.requestDevice();
  // Pin the adapter — Chrome loses the device when the last adapter
  // reference is GC'd ("external Instance reference no longer exists").
  (device as unknown as { __adapter: GPUAdapter }).__adapter = adapter;
  const runtime = new Runtime({ device, enableDerivedUniforms: true });

  const t0 = performance.now();
  const { manifests, totalVerts, radius } = await fetchManifests();
  setStatus(`streaming districts ${DISTRICT_LIST.join(",")} — ${(totalVerts / 1e6).toFixed(1)} M verts…`);

  // The camera can't know the data bbox before the geometry streams
  // in; seed it from the first district's manifest radius and refine
  // once the first district's positions arrive.
  const ctl = OrbitController.create({
    radius: radius * 1.6,
    phi: Math.PI / 4,
    theta: 0.9,
    // Wheel zooms the camera toward the FIXED orbit center.
    config: { wheelZoom: "radius" },
  });

  const proj = perspective({
    fovInRadians: Math.PI / 3,
    aspect: aspectFromViewport(RenderControl.viewport),
    near: Math.max(0.5, radius * 1e-3),
    far: radius * 8,
  });

  const leafSet = cset<SgNode>();
  const cityScene = cityRoot(leafSet, ctl, proj);
  let leafCount = 0;

  // AO path: render the city offscreen through renderToPickable — the
  // pick attachment doubles as the AO G-buffer (oct24 normal + NDC
  // depth), and ALL picking rides the portal quad's PickContext into
  // the offscreen scene.
  let scene: SgNode = cityScene;
  if (AO) {
    const pickable = renderToPickable(runtime, device, cityScene, {
      size: RenderControl.viewport,
      view: ctl.view,
      proj,
      label: "city",
    });
    const ao = new GtaoResource(device, pickable.framebuffer, pickable.pickTexture, proj, {
      radius: AO_RADIUS, intensity: AO_INTENSITY,
    });
    scene = (
      <Sg View={AVal.constant(Trafo3d.identity)} Proj={AVal.constant(Trafo3d.identity)}>
        <Sg
          Shader={compositeSurface}
          Uniform={{ SceneTex: pickable.color(), AoTex: ao }}
          PickContext={pickable.pick}
        >
          {quad()}
        </Sg>
      </Sg>
    ) as SgNode;
  }

  // fps: exponential moving average over onRendered frame times.
  let ema = 0;
  let frames = 0;
  const onRendered = (info: { frameTime: number }): void => {
    frames++;
    if (info.frameTime > 0) ema = ema === 0 ? info.frameTime : ema * 0.9 + info.frameTime * 0.1;
    if (ema > 0) fpsEl.textContent = `${(1000 / ema).toFixed(0)} fps  ${ema.toFixed(1)} ms`;
  };
  // debug hooks for the driver
  (window as unknown as { __dbg: unknown }).__dbg = {
    frames: () => frames,
    center: () => { const c = ctl.state.value.center; return [c.x, c.y, c.z]; },
    radius: () => ctl.state.value.radius,
    flyTo: (x: number, y: number, z: number) => ctl.flyTo(new V3d(x, y, z)),
    setRadius: (r: number) => { transact(() => { ctl.state.value = { ...ctl.state.value, radius: r, targetRadius: r }; }); },
    sceneRadius: radius,
    view: () => AVal.force(ctl.view).forward.toString(),
  };

  // ── streaming ingest — one district at a time (renderbench recipe):
  // fetch → build leaves → add to the live set → yield a few frames so
  // the heap stages this batch before the next district's temporaries
  // exist. Host peak ≈ ONE district's decode buffers.
  const ingest = async (): Promise<void> => {
    const tStart = performance.now();
    let firstCentered = false;
    for (let di = 0; di < DISTRICT_LIST.length; di++) {
      const dd = await fetchDistrict(di, manifests[di]!);
      const leaves = districtLeaves(dd);
      leafCount += leaves.length;
      // add in slices so a huge district doesn't stall one frame
      const SLICE = 4096;
      for (let o = 0; o < leaves.length; o += SLICE) {
        const batch = leaves.slice(o, o + SLICE);
        transact(() => { for (const l of batch) leafSet.add(l); });
        await new Promise((r) => requestAnimationFrame(() => r(undefined)));
      }
      if (!firstCentered) {
        // center the orbit on the first district's bbox
        firstCentered = true;
        let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
        const ps = dd.positions;
        for (let i = 0; i < ps.length; i += 3) {
          const x = ps[i]!, y = ps[i + 1]!, z = ps[i + 2]!;
          if (x < mnx) mnx = x; if (x > mxx) mxx = x;
          if (y < mny) mny = y; if (y > mxy) mxy = y;
          if (z < mnz) mnz = z; if (z > mxz) mxz = z;
        }
        transact(() => {
          ctl.state.value = {
            ...ctl.state.value,
            center: new V3d((mnx + mxx) / 2, (mny + mxy) / 2, (mnz + mxz) / 2),
          };
        });
      }
      setStatus(`d0${dd.district} in (${fmt.format(leafCount)} parts) — ${di + 1}/${DISTRICT_LIST.length}`);
      // give GC + staging a breather between districts
      await new Promise((r) => setTimeout(r, 50));
    }
    setStatus(
      `${fmt.format(leafCount)} parts, ${(totalVerts / 1e6).toFixed(1)} M verts — streamed in ${((performance.now() - tStart) / 1000).toFixed(1)} s. ` +
      `${AO ? "GTAO on (?ao=0 to disable)" : "GTAO off"} · drag orbit · wheel/pinch zoom · double-tap fly-to · tap = part info`,
    );
    (window as unknown as { __ready: boolean }).__ready = true;
  };

  mount(document.getElementById("app")!, (
    <RenderControl
      device={device}
      runtime={runtime}
      scene={scene}
      attach={{ devicePixelRatio: DPR }}
      onRendered={onRendered}
      onReady={({ canvas, time, device, pickAt }) => {
        void device.lost.then((info) => setStatus(`DEVICE LOST: ${info.reason} ${info.message}`, true));
        const gpuErrors: string[] = [];
        (window as unknown as { __gpuErrors: string[] }).__gpuErrors = gpuErrors;
        device.onuncapturederror = (e): void => {
          const msg = (e as GPUUncapturedErrorEvent).error.message;
          gpuErrors.push(msg);
          if (gpuErrors.length === 1) setStatus(`GPU ERROR: ${msg.slice(0, 300)}`, true);
        };
        // Pick-anchored navigation: rotate pivots around the point you
        // grab, pan keeps it glued to the pointer (portal hits come
        // back in the inner city frame — scope.view is the city view).
        const picker = async (clientX: number, clientY: number): Promise<V3d | undefined> => {
          const rect = canvas.getBoundingClientRect();
          if (rect.width <= 0 || rect.height <= 0) return undefined;
          const sx = canvas.width / rect.width, sy = canvas.height / rect.height;
          const h = await pickAt(Math.floor((clientX - rect.left) * sx), Math.floor((clientY - rect.top) * sy));
          if (h === undefined) return undefined;
          const w = AVal.force(h.hit.scope.view).backward.transformPos(h.hit.viewPos);
          return new V3d(w.x, w.y, w.z);
        };
        ctl.attach(canvas, time, { picker, proj });
        void ingest().catch((e) => setStatus(`ingest failed: ${e}`, true));
      }}
    />
  ));
}

void main().catch((e) => { console.error(e); setStatus(String(e), true); });
