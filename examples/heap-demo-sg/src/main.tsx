// Naive scene-graph heap demo. Counterpart to
// `wombat.rendering/examples/heap-demo`, but written the way a user
// would actually write it — a plain loop of `<Sg.*>` primitive nodes,
// each declaring its own Shader / Uniform / Trafo / instance scope.
// No manual deduplication, no pre-shared PipelineState, no hand-rolled
// RenderObject construction. The heap renderer's job is to collapse
// this naive tree into one bucket per Effect via the heap-everything
// path (decoder VS composed with the user effect, one pipeline per
// effect — no family-merge wrapper).
//
// `?count=N` controls how many leaves we emit (default 64). Each leaf
// picks an effect from a 5-effect table; instanced effects also wrap
// the primitive in `Sg.instanced(...)` to stack 6 copies along +Z.

import { mount, type VNode } from "@aardworx/wombat.dom";
import {
  OrbitController,
  RenderControl,
  Sg,
  aspectFromViewport,
  perspective,
  type SceneEvent,
} from "@aardworx/wombat.dom/scene";
import { AVal, HashMap, cval, transact } from "@aardworx/wombat.adaptive";
import { V3d, V4f } from "@aardworx/wombat.base";
import { BufferView, ElementType, IBuffer, ITexture } from "@aardworx/wombat.rendering/core";
import type { ClearValues } from "@aardworx/wombat.rendering/core";
import {
  surface,
  instancedSurface,
  tintedSurface,
  pulsingSurface,
  wobblingInstancedSurface,
  texturedSurface,
  texturedInstancedSurface,
} from "./effects.js";

// ─── Page chrome ───────────────────────────────────────────────────────

const root = document.getElementById("app")!;
const status = document.getElementById("status")!;
status.textContent = "starting…";
const setStatus = (m: string, err = false): void => {
  status.textContent = m;
  status.style.color = err ? "#ff7777" : "#888";
};
window.addEventListener("error", (e) => setStatus("error: " + (e.error?.message ?? e.message), true));
window.addEventListener("unhandledrejection", (e) => setStatus("rejected: " + (e.reason?.message ?? String(e.reason)), true));

// ─── Demo knobs ────────────────────────────────────────────────────────

const params = new URLSearchParams(location.search);
const ROCount = (() => {
  const v = params.get("count");
  return v !== null ? Math.max(1, parseInt(v, 10) | 0) : 64;
})();
const instCount = (() => {
  const v = params.get("inst");
  return v !== null ? Math.max(2, parseInt(v, 10) | 0) : 6;
})();
const forcedFx = (() => {
  const v = params.get("fx");
  if (v === null) return -1;
  const n = parseInt(v, 10) | 0;
  return Math.max(0, Math.min(6, n));
})();

// ─── Camera ────────────────────────────────────────────────────────────

const ctl = OrbitController.create({ radius: 22, phi: Math.PI / 5, theta: 0.5 });
// `RenderControl.time` is the post-frame clock — ticked in `onAfterFrame`
// AFTER the encode + pacer-await complete. Using it directly avoids a
// second per-frame transact and keeps animations sync'd with the
// frames the GPU actually finished.
const time = RenderControl.time;

// ─── Per-leaf parameter tables ─────────────────────────────────────────

const colors: readonly V4f[] = [
  new V4f(1.00, 0.55, 0.25, 1),
  new V4f(0.45, 0.75, 1.00, 1),
  new V4f(0.95, 0.85, 0.35, 1),
  new V4f(0.55, 0.95, 0.55, 1),
  new V4f(0.95, 0.45, 0.75, 1),
  new V4f(0.65, 0.55, 0.95, 1),
];

const tints: readonly V4f[] = [
  new V4f(1.0, 0.5, 0.5, 1),
  new V4f(0.5, 1.0, 0.5, 1),
  new V4f(0.5, 0.5, 1.0, 1),
  new V4f(1.0, 1.0, 0.4, 1),
];

type FxKind =
  | "surface" | "instanced" | "tinted" | "pulsing" | "wobbling"
  | "textured" | "texturedInstanced";
const fxTable: readonly FxKind[] = [
  "surface", "instanced", "tinted", "pulsing", "wobbling",
  "textured", "texturedInstanced",
];

// ─── Texture sources ───────────────────────────────────────────────────
//
// One URL texture (resolves via the Sg layer's placeholder-then-swap
// path — 8×8 magenta/grey checker while loading) and one procedurally
// generated canvas texture. Both flow through the same uniform-as-
// texture pipeline as any real ITexture would.

function makeCanvasTexture(): ITexture {
  const W = 128;
  const cv = document.createElement("canvas");
  cv.width = W; cv.height = W;
  const ctx = cv.getContext("2d")!;
  // Gradient + ring pattern — visually distinguishable from the URL one.
  const g = ctx.createLinearGradient(0, 0, W, W);
  g.addColorStop(0,   "#1a3a5f");
  g.addColorStop(0.5, "#3aa0d0");
  g.addColorStop(1,   "#f0e060");
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, W);
  ctx.strokeStyle = "rgba(255,255,255,0.7)";
  ctx.lineWidth = 4;
  for (let r = 8; r < W; r += 14) {
    ctx.beginPath(); ctx.arc(W/2, W/2, r, 0, Math.PI*2); ctx.stroke();
  }
  return ITexture.fromExternal(cv);
}

// Pick a small set of remote URLs the demo can load. They're CC0 from
// picsum (deterministic seed → cacheable). The placeholder shows
// while the fetch is in flight.
const urlTextures: readonly ITexture[] = [
  ITexture.fromUrl("https://picsum.photos/seed/wombat1/256/256"),
  ITexture.fromUrl("https://picsum.photos/seed/wombat2/256/256"),
  ITexture.fromUrl("https://picsum.photos/seed/wombat3/256/256"),
];
const canvasTexture: ITexture = makeCanvasTexture();
const allTextures: readonly ITexture[] = [...urlTextures, canvasTexture];

// 6 primitive shapes — direct function references so we can call
// them naturally (`<Shape Trafo Color/>`) without `as unknown` casts.
const shapes = [
  Sg.Box, Sg.Sphere, Sg.Cylinder, Sg.Cone, Sg.Tetrahedron, Sg.Octahedron,
] as const;

// Per-instance offsets for instanced effects — stacked along +Z.
function makeInstanceOffsets(count: number, dz: number): BufferView {
  const offsets = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) offsets[i * 3 + 2] = i * dz;
  return {
    buffer: AVal.constant(IBuffer.fromHost(offsets)),
    elementType: ElementType.V3f,
  };
}
const instanceOffsetsView = makeInstanceOffsets(instCount, 0.7);

// ─── Build the scene as a naive flat list of nodes ─────────────────────

const WHITE = new V4f(1, 1, 1, 1);

function makeLeaf(k: number) {
  const fxIdx = forcedFx >= 0 ? forcedFx : k % fxTable.length;
  const fx    = fxTable[fxIdx]!;
  const Shape = shapes[k % shapes.length]!;
  const baseColor = colors[k % colors.length]!;
  const tint  = tints[(k >> 3) % tints.length]!;
  const tex   = allTextures[k % allTextures.length]!;

  // Per-leaf reactive colour. Hover handlers below flip it to white;
  // pointer-leave flips back to `baseColor`. Only the hovered leaf's
  // cval ticks per pointer event → one heap-arena slot repacks +
  // one writeBuffer to the GPU. Everything else stays put.
  const color = cval<V4f>(baseColor);

  // Grid placement.
  const side = Math.ceil(Math.sqrt(ROCount));
  const spacing = 2.4;
  const center = (side - 1) * 0.5;
  const ix = k % side, iy = Math.floor(k / side);
  const trafo = Sg.translate(new V3d((ix - center) * spacing, (iy - center) * spacing, 0));

  const onEnter = (): void => { transact(() => { color.value = WHITE; }); };
  const onLeave = (): void => { transact(() => { color.value = baseColor; }); };

  const prim = <Shape Trafo={trafo} Color={color}
    OnPointerEnter={onEnter} OnPointerLeave={onLeave} />;

  switch (fx) {
    case "surface":
      return <Sg Shader={surface}>{prim}</Sg>;
    case "tinted":
      return <Sg Shader={tintedSurface} Uniform={{ Tint: AVal.constant(tint) }}>{prim}</Sg>;
    case "pulsing":
      return <Sg Shader={pulsingSurface} Uniform={{ Time: time }}>{prim}</Sg>;
    case "instanced": {
      const instanced = Sg.instanced({
        count: instCount,
        attributes: HashMap.empty<string, BufferView>().add("InstanceOffset", instanceOffsetsView),
      });
      return <Sg Shader={instancedSurface}>{instanced(prim)}</Sg>;
    }
    case "wobbling": {
      const instanced = Sg.instanced({
        count: instCount,
        attributes: HashMap.empty<string, BufferView>().add("InstanceOffset", instanceOffsetsView),
      });
      return (
        <Sg Shader={wobblingInstancedSurface} Uniform={{ Time: time }}>
          {instanced(prim)}
        </Sg>
      );
    }
    case "textured":
      return (
        <Sg Shader={texturedSurface} Uniform={{ DiffuseTex: AVal.constant(tex) }}>
          {prim}
        </Sg>
      );
    case "texturedInstanced": {
      const instanced = Sg.instanced({
        count: instCount,
        attributes: HashMap.empty<string, BufferView>().add("InstanceOffset", instanceOffsetsView),
      });
      return (
        <Sg Shader={texturedInstancedSurface} Uniform={{ DiffuseTex: AVal.constant(tex) }}>
          {instanced(prim)}
        </Sg>
      );
    }
  }
}

const leaves: VNode[] = [];
for (let k = 0; k < ROCount; k++) leaves.push(makeLeaf(k));

// ─── Mount ─────────────────────────────────────────────────────────────

const clear: ClearValues = {
  colors: HashMap.empty<string, V4f>().add("Colors", new V4f(0.07, 0.07, 0.08, 1)),
  depth: 1.0,
};

mount(root, (
  <RenderControl
    clear={clear}
    onReady={({ canvas, time: rcTime }) => {
      ctl.attach(canvas, rcTime);
      const inst = fxTable.filter(f => f === "instanced" || f === "wobbling" || f === "texturedInstanced").length;
      setStatus(`ready — ${ROCount} naive leaves (~${ROCount + Math.round(ROCount * inst / fxTable.length * (instCount - 1))} draws collapsed by heap)`);
    }}
  >
    <Sg
      View={ctl.view}
      Proj={perspective({
        fovInRadians: Math.PI / 3,
        aspect: aspectFromViewport(RenderControl.viewport),
        near: 0.1,
        far: 200,
      })}
      ForcePixelPicking={AVal.constant(true)}
      OnDoubleTap={(e: SceneEvent) => ctl.flyTo(e.worldPos)}
    >
      {leaves}
    </Sg>
  </RenderControl>
));
