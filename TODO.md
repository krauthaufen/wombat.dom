# wombat.dom — TODO

Status: 🔄 active (0.6.1). Adaptive DOM + WebGPU scene-graph DSL (JSX). Shipped:
incremental DOM rendering; scene graph M1–M7 (Sg tagged-union, traversal/compose,
RenderObject lowering, `<RenderControl>`, camera, FreeFly + Orbit controllers,
picking framebuffer + pick effect, SceneEvent capture/bubble, focus/key routing);
auto-instancing (phases 1–4, pixel-perfect parity); symmetric SDF text AA.

Design doc kept: `docs/auto-instancing.md`. Architectural items live in
`~/claude/wombat-todo.md`.

## Open

### Scene graph (M8+)
- **M8 — default surfaces / primitives**: `<Sg.Box/>`, `<Sg.Sphere/>`, etc.
- **M10 — BVH picking + fusion**.

### Picking / focus (deferred)
- HTML/JS focus integration (tab-nav into scene scopes) + focus-ring outline
  rendering.
- GPU-side MSAA pick-resolve test (Playwright + WebGPU).
- Per-frame command-encoder hook on wombat.rendering; sparse/dirty-region pick
  readback.

### Auto-instancing gaps
- Generic matrix-typed plain attributes (only the `ModelTrafo` trafo convenience
  is shipped).
- Re-validation when an `SgAdaptiveGroup` inside an instanced subtree swaps in an
  `instanceCount > 1` leaf mid-frame (validation is one-shot at compile today).

### Text rendering
- **Body-side gap probe** — CPU test sampling points inside each glyph contour to
  assert coverage by ≥1 fill/curve triangle (mirror of the band-watertightness
  probe), so triangulator refactors can't silently regress fill.
- **Halo as a runtime / zoom-aware knob** — `BAND_HALO_EM` is a static 0.05;
  rebuild per zoom-bucket or store multiple halo widths so the AA ramp neither
  clips (zoom-in) nor over-draws (zoom-out).
- **Bezier3 / arc in the band builder** — `expandContour` falls back to a chord
  for non-bezier2; arcs lose offset accuracy (cubics are pre-split, rarely hit).

## Out of scope (FUTURE.md, folded here)

- Auto-fusing leaves without an explicit instancing wrapper.
- MDI without manual setup.
- Storage-buffer instance data.
