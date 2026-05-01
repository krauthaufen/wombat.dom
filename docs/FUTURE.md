# Future ideas

Tracking deferred work and stretch ideas. Living document — entries get crossed off (or refined into TODOs) as they're picked up.

## Picking & input
- [x] OnFocus / OnBlur as scene events synthesised when focus enters/leaves a scope (Phase 5 — auto-focus on click; programmatic via `PickRegistry.setFocus`).
- [ ] HTML/JS focus integration: scene scopes participate in DOM focus order; tab-navigation reaches Sg.CanFocus nodes via DOM tabindex routing.
- [x] Key events (OnKeyDown / OnKeyUp / OnKeyPress) routed to the focused scene scope, with capture/bubble. Modifier-key state on SceneEvent (Phase 5).
- [x] Wheel events (OnWheel) routed to the spiral hit scope (Phase 4).
- [ ] Touch gestures beyond tap: pinch / two-finger pan / rotate as synthesised events.
- [ ] Focus ring rendering — outline around the focused scope (UI only; uses pickId attachment).
- [x] Custom MSAA pick-resolve (compute pass, majority vote on PickId).
- [ ] GPU-side test for MSAA pick resolve — currently only the JS reference impl + WGSL template are tested in vitest; the actual compute pipeline runs on real hardware. A browser-side test (Playwright + WebGPU) would close this loop.
- [ ] Per-frame command-encoder hook on `wombat.rendering`'s `runFrame` — would let the pickId resolve compute pass piggy-back on the render encoder instead of submitting a separate one.
- [ ] BVH + IIntersectable in wombat.base + BVH-based fall-through after pixel pickThrough.
- [ ] PointerCapture release on scope unmount.
- [ ] Configurable tap / long-press thresholds at the RenderControl level.

## Scene graph
- [x] OnDragStart / OnDrag / OnDragEnd synthesis (Phase 6 — pointerdown → move past `DRAG_THRESHOLD_PX` → up; suppresses trailing tap).
- [ ] Hover delay / tooltip helper (synthesised OnHover after PointerEnter + idle).

## Render state
- [x] Render-state scopes (DepthTest/Mask/Bias/Clamp, CullMode, FrontFace, FillMode, Multisample, BlendConstant, ColorMask, StencilMode, Pass) — Phase 1.
- [ ] **Reactive PipelineState**: render-state scope avals are forced once at compile time. Dynamic state changes require swapping the subtree via `Sg.adaptive` so a fresh RenderObject is produced. A direct aval pipeline state would avoid the reflow.
- [ ] **BlendConstant** — wombat.rendering does not expose a per-RenderObject `setBlendConstant`. The constant is captured but not wired to the encoder.
- [ ] **Multisample / DepthClamp** — gated on adapter features (`unclippedDepth`); falls back to no-op with a once-only console warning.
- [ ] **FillMode line/point** — approximated by topology override; no real polygon-mode wireframe (WebGPU limitation).

## Time-driven adaptivity
- [x] Per-frame `time` clock exposed via `RenderControl.time` and `info.time` (Phase 7). Active controls tick the global `cval<number>` once per frame.

## Rendering
- [ ] Multi-output post-processing chains beyond the pickId attachment.
- [ ] Sparse / dirty-region picking readback (skip readback when no relevant pointer event).

## Controllers (FreeFly / Orbit)
- [ ] Gamepad input for FreeFlyController — F# wires `OnGamepadAxisChange` / `OnGamepadButtonDown/Up` to MoveVec/TurnVec; the TS port currently omits this. Re-add via the GamepadAPI once the scene-event side surfaces gamepad events.
- [ ] FreeFlyController `FlyTo` animation — F# branch present (yaw/pitch deltas + global target move); the TS port leaves this for now (no test consumer yet).
- [ ] Virtual-touch-stick UI overlay for FreeFly — the original F# resource drew on-canvas joystick widgets. The TS port replaces this with multi-pointer gestures; an explicit overlay (with thumbstick rendering + dead-zones) is still nice-to-have.
- [ ] Orbit pick-aware MMB-up snap to picked depth (F# uses `model.pick V2d.Half` after pan release to recenter; TS port has the hook but no PickRegistry threading yet).
- [ ] OrbitController free-pan vs scroll-pan toggle (F# has `freeMovePan = true` hard-coded; expose as config).
- [ ] Damping/spring tuning surface for OrbitController (currently `speed` controls integration gain, but no per-axis spring constants).
