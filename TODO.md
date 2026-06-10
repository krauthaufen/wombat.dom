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
  RESOLVED ✅ — shader-defined sampler **state** (filter, addressU/V) now reaches
  the GPU end-to-end on the Fable/web backend, validated visually
  (Filter.MinMagMipPoint → blocky checker vs MinMagMipLinear → smooth).

  The full chain, repo by repo:
  - F# plugin (Wombat.Fable.Shader.Plugin): `IR.VSampler` carries a
    `SamplerState option` (Filter/AddressU/AddressV); `Hash.fs` emits it as the
    Sampler ValueDef's `state`; `Translator.fs` recovers it from the sampler
    builder (SamplerCache, keyed by binding name). (commit `c0171a3`)
  - wombat.shader: IR `ir/types.ts` Sampler ValueDef + `SamplerState` type;
    `runtime/interface.ts` `SamplerInfo.state` + `collectSamplersAndTextures`
    reads `v.state`. (commit `53dc90c`)
  - wombat.rendering: `heapAdapter.ts` / `preparedRenderObject.ts` build a
    `GPUSamplerDescriptor` from the binding's `state` (`samplerDescriptorFromState`);
    `heapScene.ts` `samplerStateBits` packs mag/min-filter + address into the
    atlas `formatBits`. (commit `0892adf`)

  THE LAST-MILE FIX (the part that made the pixels actually change):
  small textures atlas-route through the shared AtlasPool page, which has ONE
  hardware-linear sampler across all sub-rects — so per-binding GPUSampler state
  can't apply. The atlas does software filtering, and its WGSL (`atlasSample`)
  decoded `formatBits` for format/mips/address but **ignored the mag/min filter
  bits** → every atlas-routed texture sampled bilinear regardless of the shader.
  Fix: decode the filter bit and, for nearest, snap the atlas-pixel coord to the
  texel center before the (still hardware-linear) fetch — bilinear at a texel
  center == nearest. Bumped `HEAP_PERSIST_VERSION` h1→h2 (WGSL emitter changed,
  invalidate cached compiled heap shaders). (commit `45d04c3`)

  Two debugging traps worth remembering: (a) vite dep pre-bundling served a
  stale wombat.rendering — must `vite --force`; (b) the heap compile cache
  persists compiled WGSL to **localStorage** — clear it (or bump
  HEAP_PERSIST_VERSION) or shader-emitter changes stay invisible.

  This matches the .NET/FShade path (already GPU-validated on aardvark.dom).

### aval<Child> mid-siblings reconciler hiccup
`bindings/child` flush throws a caught `NotFoundError: insertBefore … is not
a child of this node` when an `aval<VNode>` child SWAPS its node while sibling
lists churn in the same flush (seen: puresg-generated conditional empty-state
`aval<DomNode>` between static siblings + an alist sibling, wombat.fable
HelloTodo). The DOM converges correctly afterwards (state is right, error is
caught + logged), so this is reconciler noise, not corruption — but the anchor
bookkeeping for aval-children should be fixed. Repro: HelloTodo add/remove
across the empty-state boundary.
