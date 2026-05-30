# wombat.dom — TODO

Status: 🔄 active (0.6.1). Adaptive DOM + WebGPU scene-graph DSL (JSX). Shipped:
incremental DOM rendering; scene graph M1–M7 (Sg tagged-union, traversal/compose,
RenderObject lowering, `<RenderControl>`, camera, FreeFly + Orbit controllers,
picking framebuffer + pick effect, SceneEvent capture/bubble, focus/key routing);
auto-instancing (phases 1–4, pixel-perfect parity); symmetric SDF text AA.

Design doc kept: `docs/auto-instancing.md`. Architectural items live in
`~/claude/wombat-todo.md`.

Note: M8 primitive Sg nodes (`<Sg.Box/>` / `Quad` / `Sphere` / `Cylinder` /
`Cone` + DefaultSurfaces) and M10 (reactive BVH picking + pixel↔BVH fusion) are
both **shipped** — they're no longer open.

## Open

### Transparency / OIT (designed, not built)
- Order-independent transparency: `Sg.transparent`/`Sg.opaque` attribute + an
  extra pass group after the opaque pass. Default **WBOIT** (port of aardvark),
  opt-in exact **linked-list A-buffer** (WebGPU has no fragment interlock, so the
  exact path is a lock-free atomic linked list, not aardvark's interlocked
  k-buffer). Full design: `docs/transparency-oit.md`.

### Picking / focus (deferred)
- HTML/JS focus integration (tab-nav into scene scopes) + focus-ring outline
  rendering.
- GPU-side MSAA pick-resolve test (Playwright + WebGPU).
- Per-frame command-encoder hook on wombat.rendering; sparse/dirty-region pick
  readback.

### Auto-instancing gaps
- Generic matrix-typed plain attributes (only the `ModelTrafo` trafo convenience
  is shipped; generic `inst.attributes` matrices are passed through verbatim).
  (Mid-frame re-validation on `SgAdaptiveGroup` swaps inside an instanced subtree
  is already handled — per-swap `validateInstancingSubtree`.)

### Text rendering
- **Coverage probes** — neither a body-side gap probe (sample points inside each
  glyph contour, assert coverage by ≥1 fill/curve triangle) nor a band-
  watertightness probe exists yet. Add both so triangulator refactors can't
  silently regress fill. (Lives in wombat.base `tests/font/`.)
- **Halo as a runtime / zoom-aware knob** — `BAND_HALO_EM` is a static 0.05;
  rebuild per zoom-bucket or store multiple halo widths so the AA ramp neither
  clips (zoom-in) nor over-draws (zoom-out).
- **Bezier3 / arc in the band builder** — `expandContour` falls back to a chord
  for non-bezier2; arcs lose offset accuracy (cubics are pre-split, rarely hit).

## Out of scope (FUTURE.md, folded here)

- Auto-fusing leaves without an explicit instancing wrapper.
- MDI without manual setup.
- Storage-buffer instance data.

### Sampler state from shaders (shader-defined filter/address)
- FShade-style sampler builders (`sampler2d { texture uniform?Bla; filter …;
  addressU … }`) are now recognised by the F# shader plugin
  (Wombat.Fable.Shader.Plugin), which recovers the **texture-uniform binding
  name** (`Bla`) — a scene then supplies the texture under that name
  (`Sg.DiffuseTexture` / `Sg.Texture(name, tex)`), and textured rendering works.
  BUT the **sampler STATE** (filter, addressU/V/W) the shader specifies is NOT
  yet threaded to the GPU sampler descriptor:
  - the plugin walks the builder chain but `IR.VSampler` (binding * name * type)
    has **no state slot**, so filter/address are dropped at the IR boundary;
  - consequently `wombat.shader`'s WGSL `SamplerBinding` and
    `wombat.rendering`'s `HeapSamplerBinding` / GPUSampler descriptor fall back
    to a default sampler.
  TODO: add a sampler-state field to `IR.VSampler` (+ IR JSON in IREncoder),
  carry it through `wombat.shader` (SamplerBinding) into `wombat.rendering`
  (build the `GPUSamplerDescriptor` from it), so shader-defined sampler state
  actually reaches the GPU instead of the default.

  PROGRESS (sampler state threading):
  - DONE (F# plugin, Wombat.Fable.Shader.Plugin): `IR.VSampler` gained a
    `SamplerState option` (Filter/AddressU/AddressV); `Hash.fs` emits it as the
    Sampler ValueDef's `state`; `Translator.fs` recovers it from the sampler
    builder (SamplerCache, keyed by binding name) and threads it into both
    VSampler construction sites.
  - DONE (wombat.shader): IR `ir/types.ts` Sampler ValueDef + `SamplerState`
    type; `runtime/interface.ts` `SamplerInfo.state` + `collectSamplersAndTextures`
    reads `v.state`. Type-checks; additive.
  - REMAINING (wombat.rendering, the deep bit): a texture-valued uniform is
    split into a texture binding + a DEFAULT sampler upstream of
    `preparedRenderObject.ts` (which at ~L733 already requires `obj.samplers` to
    contain the sampler). Find where that default sampler is created (the
    scene→render-object/heap adapter — heapAdapter / heapEffect, possibly in
    wombat.dom's scene layer) and build its `GPUSamplerDescriptor` from the
    effect's `iface.samplers[name].state`: map Filter→{magFilter,minFilter,
    mipmapFilter} (MinMagMipLinear→all "linear"; MinMagMipPoint→all "nearest";
    MinMagLinearMipPoint→mag/min "linear", mip "nearest") and AddressU/V→
    addressModeU/V ("repeat"/"clamp-to-edge"/"mirror-repeat"). Then rebuild
    wombat.shader + wombat.rendering, reinstall into wombat.fable/node_modules
    (they're installed copies, not symlinks), and validate visually
    (Filter.MinMagMipPoint → blocky checker vs MinMagMipLinear → smooth).

  UPDATE (end-to-end attempt): the contract is now threaded F#→wombat.shader→
  wombat.rendering: SamplerInfo.state → HeapSamplerBinding.state, and BOTH the
  legacy (`preparedRenderObject.ts`) and heap (`heapAdapter.ts`) sampler paths
  build a GPUSamplerDescriptor from it. All repos build + dists propagate, and
  the emitted IR carries `state` (verified). BUT the visible filter change does
  NOT appear: a runtime `console.log` in heapAdapter's sampler-folding block
  NEVER fires for the textured cube — so that texture/sampler is bound via a
  render path neither override touches. NEXT: instrument the actual dispatch
  (which heap/standalone branch binds the sampler for a single-texture RO) to
  find where the GPUSampler is really created, and apply the state THERE. Also:
  (a) small textures atlas-route (AtlasPool) and may sample via a shared/page
  sampler, so per-binding state won't apply to atlas textures — confirm whether
  the test cube atlas-routes; (b) rule out vite dep pre-bundling serving a stale
  wombat.rendering (clear ALL .vite caches / force re-optimize) before trusting
  a "no visible change" result.
