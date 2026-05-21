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
