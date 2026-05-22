# Order-Independent Transparency (OIT) — design

Status: **Shipped in wombat.dom 0.8.0 — both techniques, reactive.**
- `Sg.transparent` / `Sg.opaque` + `transparencyTask` (the OIT render-task
  wrapper, aardvark's `WrappedTask` analog) with **two modes**:
  - `"wboit"` — weighted-blended OIT (approximate, the default).
  - `"abuffer"` — exact, lock-free per-pixel linked list (atomicAdd /
    atomicExchange, sorted resolve, opaque-depth occlusion).
- **Global toggle:** `setOitMode("wboit" | "abuffer")` / `getOitMode()` — apps
  flip every `transparencyTask` at once; a per-task `mode` option overrides it.
- **Reactive sizing:** `size` is an `aval`; resize re-allocates (two managed
  framebuffers + a zip that shares the opaque depth into the OIT color targets;
  the A-buffer head buffer re-allocs on resize, node pool + heads cleared/frame).
- Validated on the real GPU through an `Sg.transparent` z-stack
  (`examples/transparency`): `wboit` → `(0.25, 0.375, 0.375)`, `abuffer` →
  exact `(0.25, 0.25, 0.5)`. Also validated as hand-built prototypes against
  aardvark's `zStackWithOccluder` (with correct pick=A(2) + depth ~0.1).

**Picking (WBOIT, wombat.dom 0.9.0):** when the output framebuffer has a
`PickData` attachment, the WBOIT path renders the opaque objects straight into
the output (`Colors`+`PickData`+depth), composites the transparent result OVER it
(with `PickData` masked write-mask-only so opaque pick survives), then renders the
transparent objects **once more** depth-tested into `PickData`+depth — aardvark's
`transformTransparentPick`. The normal pick readback then works; the picking
system stays oblivious. Requires the write-mask-only color targets added in
**wombat.rendering 0.19.6**. Validated: WBOIT `pick=A(2)`, depth ~0.1.

**A-buffer picking (wombat.dom 0.10.0):** done. A-buffer's color resolve still
samples the opaque from its intermediate, so the wrapper adds an opaque-pick pass
(opaque rendered once more into `PickData`+depth) plus the shared transparent-pick
pass; the resolve masks `PickData` write-mask-only. Both modes now pick.

**MSAA (wombat.dom 0.11.0):** `transparencyTask({ sampleCount })`. Opaque +
transparent render into one multisampled FBO `{Colors, accum, reveal, depth}`
(opaque masks accum/reveal, transparent masks Colors); the composite samples the
resolve, and the output stays single-sample. Both MSAA writers pass Colors
through (a stage that doesn't reference an upstream output drops it). WBOIT color
validated at 4×: `(0.25, 0.375, 0.375)`. (WBOIT only; picking is not produced in
the MSAA path — `pickId` isn't meaningfully MS-resolvable.)

**RenderControl auto-wiring (wombat.dom 0.11.0):** opt-in `<RenderControl
transparency={true | "wboit" | "abuffer"}>`. It routes the scene through
`transparencyTask`, threading the pick registry + traversal `initialState`. The
wrapper's pick attachment was renamed `PickData` → **`pickId`** to match
RenderControl's canvas pick framebuffer, so picking integrates with no registry
changes.

**MSAA picking — per-pixel majority vote (wombat.dom 0.12.0):** `pickId` can't be
hardware-resolved (ids aren't averageable), so the MSAA path renders `pickId`
into a multisampled `rgba16float` texture (its hardware resolve is ignored) and
resolves by **majority vote per pixel** — `texelFetch` every sample via
`Sampler2DMS`, the id covering the most samples wins, output its `(id, depth)`.
Needs multisampled texture bindings (**wombat.rendering 0.19.7**). Validated at
4×: color `(0.25, 0.375, 0.375)` + `pick=A(2)`, depth ~0.1.

Remaining (minor): an **MSAA canvas** through RenderControl (the opt-in assumes a
single-sample canvas today).

`transparencyTask` relies on five additive `compileScene` hooks (`passFilter`,
`composeEffect`, `pipelineOverride`, `injectStorage`, `injectUniforms`) — all
default-off, so existing `compileScene` callers are unaffected. See
`scene/compile.ts` and `scene/transparency.ts`.

### WebGPU gotchas found while building the prototype (read before implementing)
- **`maxColorAttachmentBytesPerSample` defaults to 32.** A 4-attachment FBO
  (e.g. Colors16f + PickData32f + accum16f + reveal16f = 34 B) silently fails
  pipeline validation. Request the adapter's max at `requestDevice`.
- **`rgba32float` is not blendable.** PickData (32f) must never be in the
  `blends` map (give it no blend; write-mask only). Use 16-float for blended
  attachments (accum/reveal/Colors).
- **Read-write storage must be FRAGMENT-only.** WebGPU forbids `read_write`
  storage in the vertex stage. Fixed in wombat.rendering **0.19.5**
  (`preparedRenderObject` binds read_write storage fragment-only).
- **Fragment storage writes ignore the depth test** (late depth test with side
  effects). So an A-buffer build pass inserts *occluded* fragments too. Occlude
  explicitly: write the opaque depth to an attachment and skip behind-nodes in
  the resolve (the prototype quantizes depth to u32 and compares against the
  sampled opaque depth). aardvark dodges this with early-depth-test + interlock.
- Texture binding split: `Sampler2D S` → texture `S_view` + sampler `S`.

Two techniques, one shared frame structure:
- **Weighted-Blended OIT (WBOIT)** — cheap, approximate, fully portable. The
  default. Direct port of aardvark's `WeightedBlendedOIT`.
- **Linked-list A-buffer** — exact, lock-free, unbounded storage. The
  "high quality / reference" mode. *Diverges* from aardvark's A-buffer (see §5).

## 0. The one WebGPU constraint that shapes everything

WebGPU has **no fragment-shader interlock / rasterizer-ordered views**. Aardvark's
exact path is an interlocked fixed-K k-buffer (per-pixel critical section). That
does not port. wombat keeps the *approximate* technique identical (WBOIT is just
MRT + blend) and replaces the exact technique with a **lock-free per-pixel linked
list** (`atomicAdd` to allocate, `atomicExchange` to prepend) sorted in the
resolve — needs only 32-bit atomics on storage buffers, which we have. See
`docs`/notes for the full reasoning (interlock simulation via spin-lock was
rejected: WGSL gives no forward-progress guarantee, riskiest on Apple/mobile
tilers — i.e. our iOS target).

## 1. API — an Sg attribute (mirrors aardvark)

```ts
Sg.transparent(child)   // mark subtree's render objects transparent
Sg.opaque(child)        // mark subtree opaque (the default)
```
- New traversal flag `isTransparent: boolean`, inherited down the scope, **reset
  to `false` at the root** (exactly aardvark's `TransparentSemantics`: root→false,
  `TransparentApplicator`→its value). Implement as an `SgTransparent` scope node
  alongside the existing `SgDepthTest` / `SgBlendMode` scopes (`scene/sg.ts`).
- Lowered onto a new `RenderObject.isTransparent?: boolean` flag (default false),
  threaded through `traversalState` like the other pipeline scopes.
- Distinct from the existing `Sg.pass` (main/transparent/overlay) which is only a
  *static draw-order* sort — OIT needs a real routing flag + separate passes.

## 2. Frame architecture — an extra pass group after the opaques

Mirrors aardvark's `TransparencyRenderTask` 5-step `Perform`, adapted to wombat's
`Command`/`RenderTask`/`renderTo` model. The closest existing template is
**picking** (`scene/picking/pickFramebuffer.ts`): offscreen MRT + a custom
resolve pass.

Reactive split (aardvark `ASet.filter`):
```
opaqueSet      = ros.filter(ro => !ro.isTransparent)
transparentSet = ros.filter(ro =>  ro.isTransparent)   // + per-technique transform
```

Per frame:
1. **Direct fast path** — if `transparentSet` is currently empty, render
   `opaqueSet` straight to the canvas FBO. No intermediate, no composite. (Read
   the set through the current adaptive token so the wrapper re-runs when
   transparency appears/disappears.)
2. Otherwise: `Copy` canvas → **intermediate FBO** (preserves any pre-clear).
3. **Opaque pass** → intermediate (writes shared depth).
4. **Transparent build** → offscreen (technique-specific, §4 / §5). Depth-test
   **on** against the opaque depth, depth-write **off**.
5. **Composite/resolve** → intermediate's `Colors` (premultiplied-over).
6. **Transparent depth+extras pass** (§6) → intermediate: re-render transparent
   surfaces depth-write **on**, `Colors` masked — the depth test selects the
   nearest transparent fragment so **depth + pickId** are correct for picking.
7. `Copy` intermediate → canvas.

Resources: FBO bundles cached per `(size, sampleCount)` with a small LRU (aardvark
uses cap 4) so size flips don't thrash. The current single-FBO `RenderControl`
loop (`renderControl.tsx`) must either grow a multi-pass hook or be bypassed by a
hand-built `alist<Command>` + sub-tasks/`renderTo` at the call site (the picking
subsystem already composes an offscreen pass + custom resolve this way).

### What exists vs. new (wombat building blocks)
Exists: MRT framebuffer signatures + `rgba16float`/`r32float`
(`resources/framebufferSignature.ts`), **per-attachment blend by name**
(`PipelineState.blends`), depth-test-on/write-off (`DepthState`), offscreen
render→sample (`runtime/renderTo.ts`), MSAA + resolve, fragment-stage atomics on
`read_write` storage buffers (`RenderObject.storageBuffers`, FRAGMENT visibility).
New: a **fullscreen-triangle composite helper** (none exists), the **multi-pass
frame orchestration** (the wrapper above), the **opaque/transparent split + Sg
attribute**, and a per-frame **storage clear** for the A-buffer.

## 3. Framebuffer signatures

- **Intermediate**: mirrors the canvas signature (all color attachments incl.
  `pickId`) + shared depth.
- **OIT (WBOIT)**: `{ accum: rgba16float, reveal: r16float|r32float }` + the
  *shared* depth (no extras bound here).
- **Composite**: `{ Colors }` + depth, backed by the *same* color texture as the
  intermediate so the composite writes land in it (aardvark's `compositeSig`
  trick — keeps FShade/our linker from synthesising varyings for unrelated
  attachments).

## 4. Technique 1 — WBOIT (default; port of aardvark)

Append a fragment "writer" to each transparent object's effect (after its normal
shading produces `color: V4f`, straight color + coverage alpha). Per-attachment
blend: **accum additive** `(one, one)`, **reveal multiplicative**
`(zero, one-minus-src)`; clears: `accum = 0`, `reveal = 1`.

```ts
// transparent build — appended writer (port of aardvark weightedBlend)
const weightedBlend = fragment((i: { color: V4f; coord: FragCoord<V4f> }) => {
  const a = i.color.w.mul(8.0).add(0.01);
  const b = i.coord.z.mul(-0.95).add(1.0);
  const w = clamp(a.mul(a).mul(a).mul(1e8).mul(b).mul(b).mul(b), 1e-2, 3e2);
  const alpha = i.color.w;
  return {
    accum:  new V4f(i.color.xyz.mul(alpha), alpha).mul(w),  // @location accum
    reveal: alpha,                                          // @location reveal
  };
});
```
```ts
// composite — fullscreen triangle, premultiplied-over onto Colors
const composite = fragment((i: { coord: FragCoord<V4f> }) => {
  let accum  = textureLoad(accumTex,  vec2i(i.coord.xy), 0);   // + per-sample avg if MSAA
  const reveal = textureLoad(revealTex, vec2i(i.coord.xy), 0).x;
  if (isInf(accum)) accum = new V4f(accum.w);
  return new V4f(accum.xyz.div(max(accum.w, 1e-5)), 1.0.sub(reveal));
});
```
MSAA: composite reads each sample and averages (aardvark loops `0..samples-1`).
Quality is the standard WBOIT approximation (good for low-to-moderate overlap;
hue shifts under heavy overdraw).

## 5. Technique 2 — exact A-buffer (linked list; wombat-specific)

Lock-free per-pixel linked list. Resources (storage buffers, not images):
- `nodes: array<Node>` — `Node = { depth: u32, color: u32, next: u32 }` (12 B; pad
  to 16). Global pool sized `W·H·avgLayers`.
- `counter: atomic<u32>` — node allocator.
- `head: array<atomic<u32>>` (W·H) — per-pixel list head, null = `0xffffffff`.

Per-frame **clear**: `head[] → 0xffffffff`, `counter → 0` (a `Custom`/compute
clear; picking's compute resolve is the template).

```ts
// build — appended writer; color writes masked, depth-test on / write off
const aBufferInsert = fragment((i: { color: V4f; coord: FragCoord<V4f> }) => {
  const n = atomicAdd(counter, 1u);
  if (n.lt(MAX_NODES)) {                       // overflow guard → graceful drop
    nodes[n].depth = bitcast<u32>(i.coord.z);
    nodes[n].color = pack4x8unorm(i.color);    // premultiplied
    nodes[n].next  = atomicExchange(head[pixel(i.coord)], n);  // lock-free prepend
  }
  // no color output (Colors masked)
});
```
Resolve (fullscreen, premultiplied-over). Two flavours:
- **capped**: copy list into `array<…, CAP>`, sort, composite — fast, but `CAP`
  re-caps exactness (CAP≈32–64 is typical).
- **exact / unbounded (N²)**: repeated min-extraction by depth threshold — walk
  the whole list each step picking the nearest `depth > lastDepth`, composite,
  repeat. No local array → truly unbounded + 100% correct, at O(N²) walks. Needs
  a tiebreak (node index) for equal depths to avoid skip/loop.

Costs (not capability gaps): node-pool memory (~200–300 MB at 1080p×8 layers — the
**iOS memory** risk), per-frame clear, and resolve divergence/sort cost. Keep it
as the opt-in exact/reference mode.

## 6. Transparent depth + extras ("pick") pass

Aardvark's step that we must keep, because wombat has a `pickId` attachment:
re-render the transparent set with the **original** effect, depth-write **on**,
`Colors` masked, blend off. The depth test alone selects the nearest transparent
fragment, so `pickId` (and shared depth) get the closest-fragment values
regardless of draw order → **picking works on transparent objects**.

## 7. Heap-renderer integration

- Opaque pass keeps using `HybridScene` (heap + legacy) unchanged.
- Transparent build can also go through the **heap**: static per-attachment blend
  resolves on the heap path (`modeKeyCpu.snapshotDescriptor`). The heap's
  *derived-mode* (uniform-driven) blend writes attachment 0 only, but OIT blend
  is static per pass, so that limitation doesn't bite.
- Shared depth attachment across opaque/transparent/pick passes.

## 8. aardvark → wombat mapping

| aardvark | wombat |
|---|---|
| `Sg.transparent` / `Sg.opaque`, `TransparentApplicator` | `Sg.transparent` / `Sg.opaque`, `SgTransparent` scope |
| `IsTransparent` Ag attribute (root→false) | `isTransparent` traversal flag (root→false) |
| `RenderObject.IsTransparent` | `RenderObject.isTransparent` |
| `TransparencyRenderTask.WrappedTask` (5-step `Perform`) | OIT orchestration wrapper / RenderControl multi-pass |
| `WeightedBlendedOIT` effects | WBOIT WGSL effects (§4) |
| `ABufferOIT` (interlock, fixed-K, storage images) | linked-list A-buffer (atomics, storage buffers) (§5) |
| `transformTransparentPick` pass | depth+extras pass (§6) |
| FBO bundle LRU per (size, samples) | same |
| `runtime.Copy` seed/blit | `Copy` command / `renderTo` |

## 9. Phasing

1. **Phase 0 — infra**: `Sg.transparent`/`isTransparent` + reactive split;
   fullscreen-composite helper; multi-pass orchestration (+ direct fast path,
   FBO cache); the depth+extras pass.
2. **Phase 1 — WBOIT** on that infra (§4). The shippable, iOS-safe default.
3. **Phase 2 — linked-list A-buffer** (§5), opt-in exact mode.
4. **Future**: MBOIT (same rails as WBOIT, better quality, bounded — strong
   candidate for the default later); spin-lock fixed-K (desktop-only experiment);
   `vec2u` 64-bit-atomic single-pass k-buffer once WebGPU ships 64-bit atomics
   (gpuweb#5071, currently Milestone 2 / not shipped).

## 10. Open questions / risks

- **Orchestration home**: extend `RenderControl` with a transparency-aware
  multi-pass loop, or compose at the call site? (Picking suggests the latter is
  feasible today.)
- **MSAA × storage buffers**: the A-buffer build under MSAA — per-sample vs
  per-fragment insert; start single-sample.
- **A-buffer memory on iOS**; resolve sort cost / divergence.
- **Composite helper** should be reusable (post-processing will want it too).
- Effort: Phase 0+1 ≈ 2–3 days; Phase 2 ≈ 1 week.
