# cityview — CadSceneDemo-class viewer on wombat.dom (plan)

Goal: a full demo app (like aardvark's CadSceneDemo) rendering the vienna
d1-9 asset through the heap, with picking, a proper camera controller, and
GTAO. Built HERE (wombat.dom) as `examples/cityview`.

Prior art this builds on (2026-07, wombat.rendering session):
- The asset + streaming loader exist in
  `wombat.rendering/examples/renderbench` (gz transfer + browser gunzip,
  per-district streaming, oct32/c4b packed attributes, arena pre-sizing).
  Asset: `~/arcbench/pkg-linux/vienna/d01..d09` (53.07 M verts, 246,655
  parts); v2 subset at `CadSceneDemo/assets/vienna_v2`.
- Measured: RTX 5060 heap/baked 1.05× (packed); iPhone renders d1-9
  streamed at 40-60 fps.

## 0. Prerequisite — workspace wombat.rendering

wombat.dom depends on PUBLISHED @aardworx/wombat.rendering ^0.19.13. The
viewer needs this session's unpublished work (paged shared HeapStorage,
mirror-less arena, oct32/c4b packed encodings, releaseConstantAttributes,
initialArenaBytes). Options: publish 0.20.0-prerelease, or `file:` /
`npm link` the workspace package for development. Local commits live on
wombat.rendering master (3fdff48, 9773012 — unpushed as of writing).

## 1. Scaffold (template: examples/tiles-demo)

- `<RenderControl>` fullscreen + `OrbitController` (`create()` +
  `attach(canvas, time)`; `OnDoubleTap={e => ctl.flyTo(e.worldPos)}`).
- Scene-wide `ForcePixelPicking={true}` (mandatory at 246k parts — keeps
  everything off the BVH; registry.ts:288 explains the stall otherwise).
- Vienna loader ported from renderbench: manifests → radius/parts; stream
  districts (fetch → leaves → drop). Parts become Sg leaves with
  Positions/Normals/Colors BufferViews + per-part trafo; `Active` gates,
  not add/remove, for any visibility toggling (tiles-demo pattern).
- NOTE packed attributes: `oct32()`/`c4b()` markers are HeapDrawSpec-level
  API in wombat.rendering; the Sg path goes RenderObject →
  renderObjectToHeapSpec. Either (a) start f32 BufferViews (desktop demo:
  memory fine, 1.26× perf fine), or (b) teach the heap adapter to map
  oct32/C4b-typed BufferViews (`ElementType` extension) to the packed
  markers — the right long-term move, needed for the phone.

## 2. Picking (mostly already works)

- Pixel path is heap-compatible today: pick chain composes onto the user
  effect; `PickId: u32` rides the drawHeader automatically. heap-demo-sg
  proves hover + OnDoubleTap on heap-rendered leaves.
- Per-part metadata: keep the manifest entries (bauTyp, height, year,
  solar…) indexed by part; `OnTap` → info panel; hover → highlight via a
  scene-level `selectedPickId` cval uniform compared in the FS (per-part
  colors may be released constants — do NOT recolor via the Color aval).
- Known gaps (defer unless hit): MSAA+OIT produces no pick output
  (transparency.ts note); pick-id == heap-draw-id GPU compaction design in
  wombat.rendering/TODO.md.

## 3. GTAO

- Template: `transparency.ts transparencyTask` — a wrapping IRenderTask
  sequencing inner tasks against offscreen IFramebuffers + fullscreen
  composite RO; RenderControl already switches task impls at init.
- v1 G-buffer for free: the pick attachment (rgba32float, TEXTURE_BINDING
  when non-MSAA) already holds oct24 view-space normal (slot1, decode via
  normal24.ts) + NDC depth (slot2). GTAO compute (raw WGSL, per-device
  pipeline cache like pickArgminCompute.ts): half-res horizon-based AO from
  depth + normal, spatial denoise, multiply into color in the composite.
  Caveats: Mode-B pixels lack normals; no-normal geometry writes 0 —
  acceptable for the city (all Mode A with normals).
- v2 (cleaner): dedicated normals/depth prepass via `composeEffect` +
  interSig-style signature (odepthWriter pattern, transparency.ts:105).
- Toggle via URL param + keyboard; add gpuMs overlay (timestamp-query)
  so the AO cost is visible.

## 4. Order of work

1. Prereq 0 (link workspace rendering) + scaffold app, f32 attributes,
   static camera → city on screen via Sg/heap.
2. OrbitController + flyTo-on-double-tap + fps overlay.
3. Picking: hover highlight (selectedPickId uniform) + tap info panel
   from manifest metadata.
4. GTAO v1 off the pick attachment + toggle + timing.
5. (Then) packed BufferViews through the adapter for phone-scale; GTAO v2
   prepass; compaction/pick-id-unification per wombat.rendering TODO.

## Risks / notes

- 246k Sg leaves: compile.ts lowering cost per leaf — heap-demo-sg does 5k;
  vienna needs ~50× that. May need a flat-array Sg node or chunked mounting
  (stream districts as batches of leaves; measure lowering time early —
  this is the main unknown of the whole plan).
- wombat.dom examples use published packages: cityview should follow
  tiles-demo's vite config but with the file:-linked rendering.
- The renderbench numbers/protocol stay in wombat.rendering as the perf
  reference; cityview is the showcase, not the benchmark.
