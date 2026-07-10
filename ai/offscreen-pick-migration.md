# Offscreen ("portal") picking — aardvark.dom design → wombat.dom migration

Source of truth read 2026-07: aardvark.dom `src/Aardvark.Dom/SceneGraph/`
{RenderTo.fs, SceneHandler.fs, TraversalState.fs, SceneFrontend.fs},
DemoCubes/Program.fs. Key commits: 8bb820b (PickProducer split), 0e9d698
(IRuntime.RenderTo/RenderToPickable + IRenderPickContext), 3844127 (portal
picking), 85c3883 (portal fixes), d45240d (semantics incl. depth → GTAO),
047871a (ONE HeapStorage shared by render + pick heaps), 5bd6d9c/a0f5db7
(bit-exact encoding + majority resolve), ed3fc7a (readback/resize race).

## The design (verified against code)

Picking stays inline-MRT (extra `pickId` rgba32float attachment on the pass,
exactly like wombat.dom today). The NEW capability: **Sg.onClick etc. work in
scene graphs rendered OFFSCREEN**, composited onto arbitrary geometry, through
arbitrary warps. Mechanism, end to end:

1. **PickProducer** (SceneHandler split, 8bb820b): the reusable "scene →
   RenderTask + pick artifacts" half (pick attachment on the signature, id
   registry, pick readback, resolve) — no window/dispatcher. The window's
   SceneHandler AND offscreen renders both wrap it. Its task is a concrete
   `PickRenderTask : AbstractRenderTask` reporting runtime+signature so a
   generic renderTo can drive it offscreen.

2. **`IRuntime.RenderTo(scene, size, clear, semantics)`** — the non-pick
   offscreen primitive: compile scene → requested semantic textures. ALL post
   passes (blur/AO/warp/composite) build on it.
   **`RenderToPickable(scene, view, proj, size, clear, semantics)`** — same
   through the PickProducer; returns `{ Textures; Pick: IRenderPickContext }`.
   Depth semantic surfaces for free (the blit copies depth when the output has
   a depth attachment) → single-pass GTAO input (d45240d).

3. **`IRenderPickContext`** — ref-counted (NOT adaptive; hold it or a returned
   texture → render stays live), `View/Proj` (live avals),
   `Pick(px) -> PickResult voption`. Extends **`IPickSubContext`** — the
   minimal recursion handle `{ Size: aval<V2i>; PickAt(px) }`, declared early
   for compile order. `PickResult` = world pos + Model/View/Proj **forced at
   pick time** + the innermost TraversalState (world is meaningless without
   the frame that produced it).

4. **`Sg.PickContext`** scene attribute → `TraversalState.PickSubContext`.
   The app composites the offscreen color texture onto any geometry and
   mounts the inner render's pick context on that node.

5. **`PickContextCoord`**: the composite's FRAGMENT shader writes the
   source-uv it sampled as a semantic output. Because the rasterizer
   interpolates whatever uv the surface emits, picking is correct for ANY
   tilt/scale/warp of the host geometry (DemoCubes warps with animated
   pincushion+ripple; picking follows the warp).

6. **Portal pick final** (`pickFinalPortal`): when the pick lowering sees
   `PickSubContext` on a leaf, it appends this final instead of A/B: packs
   `{ +PickId, uv.x, uv.y, quadOwnNdcDepth }` into the pick attachment.
   +id passes the mode-A sign gate untouched; slots 1-2 are uv (NOT
   normal/depth); slot 3 carries the PORTAL GEOMETRY's own depth so the
   pixel-vs-BVH tie-break still has a real depth. Registers
   `portalScopes[pickId] = subContext`.

7. **Recursive resolve** (resolver third branch): winning pixel's scope has a
   PickSubContext → read uv from slots 1-2 of the readback, map tc→pixel
   against `sub.Size` (Y-FLIP: tc origin bottom-left, pick buffer top-left),
   `sub.PickAt(innerPx)` → innermost (world, M/V/P forced, TraversalState);
   dispatch events to the INNER node's handler chain, positions in the inner
   frame. Inner MISS → fall through to the portal node itself (hover the
   "window background", its own Cursor/handlers apply). Nesting = natural
   recursion (inner scenes can contain portals).

8. **Heap tie-in** (047871a): render heap + pick heap share ONE HeapStorage —
   the pick pass is a second heap over the same arena with pick-final
   effects. HeapNode brokers ids via the minimal `IPickContext`
   (Register/Deregister; distinct interface from IRenderPickContext,
   deliberately unmerged). d179325: mixed pickable / NoEvents heaps.

9. **Robustness learned the hard way**: pick readback runs off the render
   thread (mouse-lag, b3be49e) but disposal must be raced-guarded vs resize
   (ed3fc7a); MSAA pick resolve must be a compute MAJORITY VOTE (averaging
   blit corrupts ids; also MoltenVK blit-resolve crash) — wombat.dom already
   has this; bit-exact encodings (intBitsToFloat ids) remove the 2^24 id
   ceiling where compute exists — wombat.dom still uses f32 ids (2^24 cap,
   fine for now); pick-effect cache tags must be bumped when finals change
   (pickv4/portalv2 — same lesson as wombat.rendering's
   HEAP_PERSIST_VERSION).

## Migration map (wombat.dom names)

| aardvark.dom | wombat.dom today | work |
|---|---|---|
| PickProducer | inline in renderControl.tsx `initialise()` (pickFb + registry + argmin + metadata) | **extract `PickProducer`**: scene→task + pick artifacts, reusable by RenderControl and renderToPickable |
| IRuntime.RenderTo | wombat.rendering `runtime/renderTo.ts` (check shape) + transparency.ts's hand-built offscreen fbs | **`renderTo(sceneNode, size, clear, semantics)`** scene-level wrapper returning semantic textures |
| RenderToPickable / IRenderPickContext | — | new: producer-backed, `{textures, pick}`; ref-count via texture acquire |
| IPickSubContext | — | new minimal interface `{ size: aval<V2i>; pickAt(px): InnerHit \| undefined }` |
| Sg.PickContext attr | SceneAttribute set (traversalState.ts) | new attribute → `TraversalState.pickSubContext` |
| PickContextCoord + pickFinalPortal | pickShaders.ts finals (Mode A/B) | new portal final: `{+id, uv, quadDepth}`; compose when leaf has subContext; `portalScopes: Map<int, IPickSubContext>` |
| resolver third branch | pickArbitrate.ts + dispatcher.ts | portal branch: uv from winner slots 1-2 (argmin result already carries the 4 slots!), Y-flip, `sub.pickAt`, dispatch inner scope; miss → portal node fallback |
| PickResult forced frames | sceneEventLocation.ts lazy positions | inner-frame variant: event location built from inner view/proj |
| shared-storage pick heap | heap runs implicitly via hybridScene | later: pick pass as second heap over shared HeapStorage (wombat.rendering has createHeapStorage + buildHeapScene({storage}) since 3fdff48) |

Order: (1) PickProducer extraction (pure refactor, tests stay green) →
(2) renderTo/renderToPickable + IPickSubContext → (3) Sg.PickContext +
portal final + resolver recursion → (4) DemoCubes-style warp example as the
regression test → (5) GTAO consumes RenderTo's depth/normal semantics
(cityview-plan.md §3). Steps 1-3 are the "whole infrastructure" the city
viewer's GTAO + minimap/portal needs.

## Cross-validated details (second full source pass)

- **Two-FBO shared-attachment split** (SceneHandler.fs:1612-1668): pickable
  and NoEvents objects render into two framebuffers SHARING the same backing
  color+depth attachments — pickable first (with PickId), NoEvents after
  through a signature WITHOUT it. NoEvents geometry occludes color/depth but
  never overwrites ids = pick-through-by-routing. The migration needs the
  same two-pass split (WebGPU: two render passes over the same textures,
  second one without the pick color attachment — load, not clear).
- **aardvark's shipped resolver is the CPU spiral over a 33×33 region read**;
  their PickArgmin GPU kernel is staged but UNWIRED. wombat.dom already ships
  the argmin compute — it is AHEAD here; keep argmin, ignore the spiral.
- **Portal hits do not propagate normal/partIndex today** (only inner
  viewPos + cameras + state) — acceptable v1 scope for the migration too.
- **Per-producer id spaces**: each producer has its own registry (ids from 1,
  smallest-first free-list reuse, ref-counted acquire/release via a delta
  reader pumped per frame). Outer/inner collisions impossible — each pick
  buffer is only decoded against its own maps. Unknown/stale id → skipped.
- **WebGPU async readback**: aardvark's Pick is synchronous (own queue +
  pickLock vs the resolve; ed3fc7a guards resize disposal under the same
  lock). WebGPU's copyTextureToBuffer + mapAsync is genuinely async ⇒
  `pickAt` returns a Promise; wombat.dom's resolvePixel already coalesces
  one-in-flight — portal recursion adds one await per nesting level.
- **Encoding**: take only the bit-exact path (intBitsToFloat ids — no 2^24
  ceiling, Normal32, majority-vote compute resolve; wombat.dom currently
  uses f32 ids + oct24 — upgrading is optional but removes the ceiling and
  the near-integer validation).
- **Heap side, precise contracts** (HeapPool.fs / HeapNode.fs):
  `ofRenderObjectsPicking storage deregister objects` PARTITIONS inputs by
  presence of the `HeapPickId` marker uniform into TWO heap builds over ONE
  HeapStorage (pickable → PickId-extended signature; NoEvents → base).
  Per-bucket `pickIds: int[]` slot array (−1 unpickable) flushed to a
  `HeapPickIds` SSBO; `uniform.HeapPickId` reads are rewritten BY NAME to
  `HeapPickIds[slot]`. Slot free → `deregister(pickIds[slot])` → dom's
  ref-counted releaseId. `IsPickable` is decided BEFORE signature-dependent
  expansion so the producer can route deferred heap bundles to the right
  pass. Bucket-aware linkDCE write-masks attachments a bucket's effect
  doesn't write (wombat gap: heapScene's colorTargets is one list for all
  families — needs per-family writeMask/null targets).
- **Effect substitution plan** (planChain/planSemantic): route-vs-synthesize
  per requested output semantic via effect-dependency introspection; the
  depth seed comes from fragCoord.z (never declare Depth as an FS input);
  vsn-from-Normals injector when the user effect lacks a view-space normal.
- **Acceptance test**: port DemoCubes (warped portal + heap + NoEvents mix,
  hover-highlight + click-to-delete through an animated warp).
