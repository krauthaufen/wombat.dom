# Future ideas

Tracking deferred work and stretch ideas. Living document — entries get crossed off (or refined into TODOs) as they're picked up.

## Picking & input
- [x] OnFocus / OnBlur as scene events synthesised when focus enters/leaves a scope (Phase 5 — auto-focus on click; programmatic via `PickRegistry.setFocus`).
- [ ] HTML/JS focus integration: scene scopes participate in DOM focus order; tab-navigation reaches Sg.CanFocus nodes via DOM tabindex routing.
- [x] Key events (OnKeyDown / OnKeyUp / OnKeyPress) routed to the focused scene scope, with capture/bubble. Modifier-key state on SceneEvent (Phase 5).
- [x] Wheel events (OnWheel) routed to the spiral hit scope (Phase 4).
- [x] Touch gestures beyond tap: pinch / two-finger pan / two-finger rotate as synthesised events (`OnPinch`, `OnTwoFingerPan`, `OnTwoFingerRotate`).
- [ ] Focus ring rendering — outline around the focused scope (UI only; uses pickId attachment).
- [x] Custom MSAA pick-resolve (compute pass, majority vote on PickId).
- [ ] GPU-side test for MSAA pick resolve — currently only the JS reference impl + WGSL template are tested in vitest; the actual compute pipeline runs on real hardware. A browser-side test (Playwright + WebGPU) would close this loop.
- [ ] Per-frame command-encoder hook on `wombat.rendering`'s `runFrame` — would let the pickId resolve compute pass piggy-back on the render encoder instead of submitting a separate one.
- [ ] BVH + IIntersectable in wombat.base + BVH-based fall-through after pixel pickThrough.
- [x] PointerCapture release on scope unmount (dispatcher invalidates capture when registry lookup no longer matches).
- [x] Configurable tap / long-press thresholds at the RenderControl level (`<RenderControl tapThresholds={…}>`).
- [x] Implicit picking: `<RenderControl>` always allocates a per-instance `PickRegistry` and pickId attachment; the registry is exposed via `onReady({ picking })`. (Previously gated on a `picking` prop.)
- [x] Per-scope `local2World` in capture/bubble: each `LeafPickEntry` carries a snapshot of `state.model` taken at `pushHandlers` time; the dispatcher applies it via `event.transformed(...)` before invoking each handler so `e.position` / `e.normal` / `e.pickRay` are in that scope's own local frame.

## Scene graph
- [x] OnDragStart / OnDrag / OnDragEnd synthesis (Phase 6 — pointerdown → move past `DRAG_THRESHOLD_PX` → up; suppresses trailing tap).
- [x] Hover delay / tooltip helper — `OnHover` fires after `HOVER_DELAY_MS` of idle hover on the same scope.

## Render state
- [x] Render-state scopes (DepthTest/Mask/Bias/Clamp, CullMode, FrontFace, FillMode, Multisample, BlendConstant, ColorMask, StencilMode, Pass) — Phase 1.
- [ ] **Reactive PipelineState**: render-state scope avals are forced once at compile time. Dynamic state changes require swapping the subtree via `Sg.adaptive` so a fresh RenderObject is produced. A direct aval pipeline state would avoid the reflow.
- [ ] **DepthClamp** — gated on the `unclippedDepth` adapter feature; consumed via `primitive.unclippedDepth` in the snapshot path. Pipeline compile fails when the feature is absent (no silent fallback).
- [ ] **FillMode line/point** — approximated by topology override; no real polygon-mode wireframe (WebGPU limitation).

## Time-driven adaptivity
- [x] Per-frame `time` clock exposed via `RenderControl.time` and `info.time` (Phase 7). Active controls tick the global `cval<number>` once per frame.

## Rendering
- [ ] Multi-output post-processing chains beyond the pickId attachment.
- [ ] Sparse / dirty-region picking readback (skip readback when no relevant pointer event).

## Controllers (FreeFly / Orbit)
- [x] Gamepad input for FreeFlyController — `attach()` polls the GamepadAPI on each time-aval tick and translates axes/buttons into SetMoveVec / SetTurnVec / AddMoveVec / AdjustMoveSpeed.
- [x] FreeFlyController `FlyTo(location, forward)` — yaw/pitch deltas + global target move, ported from F#.
- [x] Virtual-touch-stick UI overlay for FreeFly — opt-in `attach(target, time, { virtualSticks: true })`; auto-hides on non-touch devices.
- [x] Orbit pick-aware MMB-up snap to picked depth — opt-in `attach(target, time, { picker })`; centre re-targets via QuadInOut animation.
- [x] OrbitController free-pan vs scroll-pan toggle — `OrbitConfig.freeMovePan` (default true).
- [x] Damping/spring tuning surface for OrbitController — `OrbitConfig.springConstants: { phi, theta, radius, center }`.
