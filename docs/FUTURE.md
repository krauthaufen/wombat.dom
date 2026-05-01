# Future ideas

Tracking deferred work and stretch ideas. Living document — entries get crossed off (or refined into TODOs) as they're picked up.

## Picking & input
- [ ] HTML/JS focus integration: scene scopes participate in DOM focus order; tab-navigation reaches Sg.Focusable nodes; OnFocus / OnBlur as scene events synthesised when focus enters/leaves a scope.
- [ ] Key events (OnKeyDown / OnKeyUp / OnKeyPress) routed to the focused scene scope, with capture/bubble. Modifier-key state on SceneEvent.
- [ ] Wheel events (OnWheel) routed to the spiral hit scope; deltaY inverted-scroll handling hint.
- [ ] Touch gestures beyond tap: pinch / two-finger pan / rotate as synthesised events.
- [ ] Focus ring rendering — outline around the focused scope (UI only; uses pickId attachment).
- [x] Custom MSAA pick-resolve (compute pass, majority vote on PickId).
- [ ] GPU-side test for MSAA pick resolve — currently only the JS reference impl + WGSL template are tested in vitest; the actual compute pipeline runs on real hardware. A browser-side test (Playwright + WebGPU) would close this loop.
- [ ] Per-frame command-encoder hook on `wombat.rendering`'s `runFrame` — would let the pickId resolve compute pass piggy-back on the render encoder instead of submitting a separate one.
- [ ] BVH + IIntersectable in wombat.base + BVH-based fall-through after pixel pickThrough.
- [ ] PointerCapture release on scope unmount.
- [ ] Configurable tap / long-press thresholds at the RenderControl level.

## Scene graph
- [ ] OnDrag / OnDrop synthesis (drag = pointerdown → move beyond drag-threshold → up).
- [ ] Hover delay / tooltip helper (synthesised OnHover after PointerEnter + idle).

## Rendering
- [ ] Multi-output post-processing chains beyond the pickId attachment.
- [ ] Sparse / dirty-region picking readback (skip readback when no relevant pointer event).
