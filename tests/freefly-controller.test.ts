// FreeFlyController — behaviour tests.
//
// We drive integration manually via `tick(dt)` (no rAF needed). DOM
// events are synthesised against happy-dom elements; the time aval
// passed to attach() is irrelevant in these tests because we only
// rely on tick() for per-frame steps.

import { describe, expect, it } from "vitest";
import { AVal, cval } from "@aardworx/wombat.adaptive";
import { V2d, V3d } from "@aardworx/wombat.base";

import {
  FreeFlyController,
  freeFlyIsAnimating,
} from "../src/scene/index.js";

function makeTarget(): HTMLElement {
  const el = document.createElement("div");
  (el as unknown as { setPointerCapture?: () => void }).setPointerCapture = () => {};
  (el as unknown as { releasePointerCapture?: () => void }).releasePointerCapture = () => {};
  return el;
}

describe("FreeFlyController", () => {
  it("starts idle (IsAnimating = false)", () => {
    const ctl = FreeFlyController.create({ position: new V3d(0, 0, 5) });
    expect(freeFlyIsAnimating(ctl.state.value)).toBe(false);
  });

  it("SetMoveVec(W) marks animating; tick advances Position along Forward", () => {
    const ctl = FreeFlyController.create({
      position: V3d.zero,
      forward: new V3d(0, 0, 1),  // forward along +Z
      sky: new V3d(0, 1, 0),
    });
    ctl.setMoveVec("W", new V3d(0, 0, 1));
    expect(freeFlyIsAnimating(ctl.state.value)).toBe(true);
    const before = ctl.state.value.Position;
    ctl.tick(0.5);
    const after = ctl.state.value.Position;
    expect(after.z).toBeGreaterThan(before.z);
  });

  it("Releasing W returns IsAnimating to false after damping settles", () => {
    const ctl = FreeFlyController.create();
    ctl.setMoveVec("W", new V3d(0, 0, 1));
    ctl.tick(0.1);
    ctl.setMoveVec("W", V3d.zero);
    // Run several ticks; with no momentum and no MoveVec there's nothing to damp.
    for (let i = 0; i < 5; i++) ctl.tick(0.1);
    expect(freeFlyIsAnimating(ctl.state.value)).toBe(false);
  });

  it("addTargetTurn rotates Forward via tick", () => {
    const ctl = FreeFlyController.create({
      forward: new V3d(0, 0, -1),
      sky: new V3d(0, 1, 0),
    });
    const fwd0 = ctl.state.value.Forward;
    ctl.addTargetTurn(new V2d(0.5, 0));   // yaw around sky
    // tick large enough to consume all of TargetTurn (turnFactor*dt clamps to 1)
    ctl.tick(0.5);
    const fwd1 = ctl.state.value.Forward;
    expect(fwd1.sub(fwd0).length()).toBeGreaterThan(0.1);
  });

  it("Wheel adds momentum on Z; momentum decays with damping", () => {
    const ctl = FreeFlyController.create();
    ctl.addMomentum(new V3d(0, 0, 1));
    const m0 = ctl.state.value.Momentum.length();
    expect(m0).toBeCloseTo(1, 6);
    // tick: Damping=20 → 0.5 ** (20*0.1) = 0.25; after 1s should be tiny.
    ctl.tick(1.0);
    const m1 = ctl.state.value.Momentum.length();
    expect(m1).toBeLessThan(m0);
    expect(m1).toBeLessThan(1e-4);
  });

  it("Two-finger pinch synthesised: Z target-move grows with pinch delta", () => {
    const ctl = FreeFlyController.create();
    const target = makeTarget();
    const time = cval(0);
    const detach = ctl.attach(target, time);

    firePointer(target, "pointerdown", { pointerType: "touch", pointerId: 1, clientX: 100, clientY: 100 });
    firePointer(target, "pointerdown", { pointerType: "touch", pointerId: 2, clientX: 200, clientY: 100 });
    // Spread fingers apart: from dist=100 to dist=200.
    firePointer(target, "pointermove", { pointerType: "touch", pointerId: 1, clientX: 50,  clientY: 100 });
    firePointer(target, "pointermove", { pointerType: "touch", pointerId: 2, clientX: 250, clientY: 100 });

    expect(ctl.state.value.TargetMoveLocal.z).not.toBe(0);

    detach();
    void AVal;
  });
});

// ---------------------------------------------------------------------------

interface PointerInit {
  pointerId?: number;
  pointerType?: string;
  button?: number;
  clientX?: number;
  clientY?: number;
  movementX?: number;
  movementY?: number;
  shiftKey?: boolean;
}

function firePointer(
  target: EventTarget,
  type: "pointerdown" | "pointermove" | "pointerup" | "pointercancel",
  init: PointerInit,
): void {
  const ev = new Event(type, { bubbles: false, cancelable: true }) as unknown as PointerEvent;
  Object.defineProperty(ev, "pointerId", { value: init.pointerId ?? 1 });
  Object.defineProperty(ev, "pointerType", { value: init.pointerType ?? "mouse" });
  Object.defineProperty(ev, "button", { value: init.button ?? 0 });
  Object.defineProperty(ev, "clientX", { value: init.clientX ?? 0 });
  Object.defineProperty(ev, "clientY", { value: init.clientY ?? 0 });
  Object.defineProperty(ev, "movementX", { value: init.movementX ?? 0 });
  Object.defineProperty(ev, "movementY", { value: init.movementY ?? 0 });
  Object.defineProperty(ev, "shiftKey", { value: init.shiftKey ?? false });
  Object.defineProperty(ev, "target", { value: target });
  target.dispatchEvent(ev);
}
