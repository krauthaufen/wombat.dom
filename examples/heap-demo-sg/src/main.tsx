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
  type CullValue,
  type SceneEvent,
} from "@aardworx/wombat.dom/scene";
import { AVal, HashMap, cset, cval, transact } from "@aardworx/wombat.adaptive";
import { V3d, V4f, Trafo3d } from "@aardworx/wombat.base";
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
  tintBGR,
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
  return v !== null ? Math.max(2, parseInt(v, 10) | 0) : 2;
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
  ITexture.fromUrl("https://picsum.photos/seed/wombat4/256/256"),
  ITexture.fromUrl("https://picsum.photos/seed/wombat5/256/256"),
  ITexture.fromUrl("https://picsum.photos/seed/wombat6/256/256"),
  ITexture.fromUrl("https://picsum.photos/seed/wombat7/256/256"),
  ITexture.fromUrl("https://picsum.photos/seed/wombat8/256/256"),
  ITexture.fromUrl("https://picsum.photos/seed/wombat9/256/256"),
  ITexture.fromUrl("https://picsum.photos/seed/wombatA/256/256"),
  ITexture.fromUrl("https://picsum.photos/seed/wombatB/256/256"),
  ITexture.fromUrl("https://picsum.photos/seed/wombatC/256/256"),
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

  // Per-leaf reactive colour. Hover handlers flip it to white;
  // pointer-leave flips back. One cval tick per pointer event →
  // one heap-arena slot repacks, no scene rebuild.
  const color = cval<V4f>(baseColor);

  // Grid placement.
  const side = Math.ceil(Math.sqrt(ROCount));
  const spacing = 2.4;
  const center = (side - 1) * 0.5;
  const ix = k % side, iy = Math.floor(k / side);
  // (Earlier iterations tried alternating x-mirror trafos so the
  // determinant rule could be visually demonstrated. With the heap
  // demo's primitive Shape geometry the winding doesn't actually
  // reverse under a -x scale the way the rule's det check assumed —
  // the visual case is a 3D-math/winding-convention issue independent
  // of the renderer's correctness. Reverted to plain placement.)
  const trafo = Sg.translate(new V3d((ix - center) * spacing, (iy - center) * spacing, 0));

  const onEnter = (): void => { transact(() => { color.value = WHITE; }); };
  const onLeave = (): void => { transact(() => { color.value = baseColor; }); };

  const prim = <Shape Trafo={trafo} Color={color}
    OnPointerEnter={onEnter} OnPointerLeave={onLeave} />;

  switch (fx) {
    case "surface":
      return <Sg Shader={surface}>{prim}</Sg>;
    case "tinted":
      return <Sg Shader={tintedSurface} Uniform={{ Tint: AVal.constant(tint), TintBGR: tintBGR }}>{prim}</Sg>;
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

// Build the JSX leaves once and stash them in a `cset<VNode>`. The
// `<Sg>{cset}</Sg>` form is fully supported — `collectSgChildren`
// maps each VNode → SgNode through the same path JSX-children take,
// and the heap path consumes the resulting aset deltas as-is
// (one heap-arena alloc/release + one pool-entry ref tweak per
// delta, no scene rebuild, no shader recompile).
const allLeafNodes: VNode[] = [];
for (let k = 0; k < ROCount; k++) allLeafNodes.push(makeLeaf(k));
const liveLeaves = cset<VNode>(allLeafNodes);

// Toggle: tap the floating "toggle ½" button to remove a deterministic
// half of the leaves from the cset; tap again to put them back. The
// heap renderer takes the aset deltas as-is — one heap-arena
// release + one pool-ref tweak per removed leaf, no shader compile,
// no scene-wide rebuild. Watch the FPS counter — it stays at 60.
let halfOut = false;
const toggleBtn = document.createElement("button");
toggleBtn.textContent = "toggle ½";
toggleBtn.style.cssText =
  "position:fixed;top:8px;right:8px;z-index:10;padding:8px 14px;" +
  "font:14px system-ui,sans-serif;background:#222;color:#ddd;" +
  "border:1px solid #555;border-radius:6px;cursor:pointer;" +
  "-webkit-tap-highlight-color:transparent;";
toggleBtn.addEventListener("click", () => {
  halfOut = !halfOut;
  transact(() => {
    if (halfOut) {
      for (let k = 0; k < allLeafNodes.length; k += 2) liveLeaves.remove(allLeafNodes[k]!);
    } else {
      for (let k = 0; k < allLeafNodes.length; k += 2) liveLeaves.add(allLeafNodes[k]!);
    }
  });
});
document.body.appendChild(toggleBtn);

// ─── Reactive cullMode demo ────────────────────────────────────────────
//
// One cval shared across the whole scene drives the cullMode for every
// RO via `<Sg CullMode={cullModeC}>`. Tapping the "cull" button cycles
// "back" → "front" → "none". Pre-wombat.rendering@0.9.15 this was
// silently broken: the bucket key hashed the cval BY IDENTITY, so the
// value change never moved any RO into the right bucket and pixels
// rendered with the OLD cull mode. As of 0.9.15 the key is value-
// based AND a per-RO ModeKeyTracker triggers a rebucket on aval mark
// — the next frame's pixels reflect the new cull mode.
const cullModes: readonly CullValue[] = ["back", "front", "none"];
let cullIdx = 0;
const cullModeC = cval<CullValue>(cullModes[cullIdx]!);
const cullBtn = document.createElement("button");
cullBtn.textContent = `cull: ${cullModes[cullIdx]}`;
cullBtn.style.cssText =
  "position:fixed;top:8px;right:120px;z-index:10;padding:8px 14px;" +
  "font:14px system-ui,sans-serif;background:#222;color:#ddd;" +
  "border:1px solid #555;border-radius:6px;cursor:pointer;" +
  "-webkit-tap-highlight-color:transparent;";
cullBtn.addEventListener("click", () => {
  cullIdx = (cullIdx + 1) % cullModes.length;
  const next = cullModes[cullIdx]!;
  cullBtn.textContent = `cull: ${next}`;
  transact(() => { cullModeC.value = next; });
});
document.body.appendChild(cullBtn);

// ─── GPU-eval derived-mode rule (Task 2 Phase 5) ───────────────────────
//
// A derived-mode rule authored via the `rule(...)` marker from
// @aardworx/wombat.shader. The marker lowers the closure body to
// shader IR at build time; the wombat.rendering runtime extracts
// the body, runs the symbolic output-set analysis + per-declared
// concrete evaluator, sizes the bucket's pipeline slots, and
// codegens the partition kernel WGSL.
//
// This rule body is the trivial pass-through (`return declared`) —
// equivalent in routing to NO rule, but exercises the full GPU
// routing pipeline end-to-end (rule extraction → analysis →
// codegen → kernel → declared-mark reactivity). Richer rule bodies
// (e.g. determinant-flip-cull) need the shader package's intrinsic
// catalogue to grow `determinant` + matrix subscripting in rule
// scope; that lands when the analysis catches up.
import { rule } from "@aardworx/wombat.shader";
import { derivedMode } from "@aardworx/wombat.rendering/runtime";

declare const declared: number;

const cullExpr = rule(() => declared);
const cullRule = derivedMode("cull", cullExpr);  // SG fills in `declared` from cullModeC

const enableGpuRule = params.get("gpurule") === "1";
const cullModeOrRule = enableGpuRule ? cullRule : cullModeC;

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
      CullMode={cullModeOrRule}
      ForcePixelPicking={AVal.constant(true)}
      OnDoubleTap={(e: SceneEvent) => ctl.flyTo(e.worldPos)}
    >
      {liveLeaves}
    </Sg>
  </RenderControl>
));
