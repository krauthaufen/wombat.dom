// OrbitController — behaviour + animation easing tests.

import { describe, expect, it } from "vitest";
import { AVal, cval } from "@aardworx/wombat.adaptive";
import { V2d, V3d } from "@aardworx/wombat.base";

import {
  Anim,
  OrbitController,
  deriveView,
  getParameter,
  interpolateV3,
  type Animation,
} from "../src/scene/index.js";

function makeTarget(): HTMLElement {
  const el = document.createElement("div");
  (el as unknown as { setPointerCapture?: () => void }).setPointerCapture = () => {};
  (el as unknown as { releasePointerCapture?: () => void }).releasePointerCapture = () => {};
  return el;
}

describe("OrbitController — initial pose", () => {
  it("eye at distance R from center on the unit-direction sphere", () => {
    const ctl = OrbitController.create({
      center: V3d.zero,
      radius: 7,
      phi: 0,
      theta: 0,
      sky: new V3d(0, 0, 1),
    });
    const view = deriveView(ctl.state.value);
    expect(view.eye.sub(V3d.zero).length()).toBeCloseTo(7, 5);
    // phi=0, theta=0 with sky=Z → eye at (R, 0, 0).
    expect(view.eye.x).toBeCloseTo(7, 5);
    expect(view.eye.z).toBeCloseTo(0, 5);
  });

  it("view aval gives the expected lookAt — eye world-pos lands on view origin", () => {
    const ctl = OrbitController.create({ center: V3d.zero, radius: 5, phi: 0, theta: 0, sky: new V3d(0, 0, 1) });
    const t = AVal.force(ctl.view);
    const eyeWorld = deriveView(ctl.state.value).eye;
    const eyeView = t.transform(eyeWorld);
    expect(eyeView.x).toBeCloseTo(0, 5);
    expect(eyeView.y).toBeCloseTo(0, 5);
    expect(eyeView.z).toBeCloseTo(0, 5);
    // Center should be in front of the eye → -Z in view space.
    const centerView = t.transform(V3d.zero);
    expect(centerView.z).toBeCloseTo(-5, 5);
  });
});

describe("OrbitController — drag to rotate (LMB)", () => {
  it("LMB drag of dx px produces dphi = dx * -0.01 * moveSensitivity", () => {
    const ctl = OrbitController.create({ center: V3d.zero, radius: 5, phi: 0, theta: 0 });
    const target = makeTarget();
    const time = cval(0);
    const detach = ctl.attach(target, time);

    firePointer(target, "pointerdown", { pointerId: 1, clientX: 0, clientY: 0, button: 0, pointerType: "mouse" });
    firePointer(target, "pointermove", { pointerId: 1, clientX: 100, clientY: 0, button: 0, pointerType: "mouse" });

    const dphiExpected = 100 * -0.01 * ctl.state.value.config.moveSensitivity;
    expect(ctl.state.value.targetPhi).toBeCloseTo(dphiExpected, 6);

    detach();
  });
});

describe("OrbitController — wheel zoom", () => {
  it("wheel scrolls grow targetRadius (when shift+ortho fallback path active)", () => {
    const ctl = OrbitController.create({ center: V3d.zero, radius: 5, phi: 0, theta: 0 });
    const target = makeTarget();
    const time = cval(0);
    const detach = ctl.attach(target, time);
    const r0 = ctl.state.value.targetRadius;

    fireWheel(target, 120, true /* shift = lockedToScene path */);
    expect(ctl.state.value.targetRadius).toBeGreaterThan(r0);

    fireWheel(target, -120, true);
    expect(ctl.state.value.targetRadius).toBeCloseTo(r0, 5);

    detach();
  });
});

describe("OrbitController — clamping", () => {
  it("setTargetTheta clamps to thetaRange", () => {
    const ctl = OrbitController.create({ theta: 0 });
    ctl.setTargetTheta(10);
    expect(ctl.state.value.targetTheta).toBeLessThanOrEqual(ctl.state.value.config.thetaRange.y);
    ctl.setTargetTheta(-10);
    expect(ctl.state.value.targetTheta).toBeGreaterThanOrEqual(ctl.state.value.config.thetaRange.x);
  });
  it("setTargetRadius clamps to radiusRange", () => {
    const ctl = OrbitController.create({ radius: 5 });
    ctl.setTargetRadius(1e12);
    expect(ctl.state.value.targetRadius).toBeLessThanOrEqual(ctl.state.value.config.radiusRange.y);
    ctl.setTargetRadius(-1);
    expect(ctl.state.value.targetRadius).toBeGreaterThanOrEqual(ctl.state.value.config.radiusRange.x);
  });
});

describe("OrbitController — two-finger pinch", () => {
  it("spreading two touches reduces targetRadius (zoom in)", () => {
    const ctl = OrbitController.create({ radius: 10 });
    const target = makeTarget();
    const time = cval(0);
    const detach = ctl.attach(target, time);
    const r0 = ctl.state.value.targetRadius;

    firePointer(target, "pointerdown", { pointerId: 1, pointerType: "touch", clientX: 100, clientY: 100 });
    firePointer(target, "pointerdown", { pointerId: 2, pointerType: "touch", clientX: 200, clientY: 100 });
    // Spread: scale > 1 → r/scale shrinks.
    firePointer(target, "pointermove", { pointerId: 1, pointerType: "touch", clientX: 50,  clientY: 100 });

    expect(ctl.state.value.targetRadius).toBeLessThan(r0);
    detach();
  });
});

describe("OrbitController — animation easings", () => {
  it("Linear/QuadInOut/Cubic produce the documented values", () => {
    expect(getParameter(Anim.Linear, 0.5)).toBeCloseTo(0.5, 10);
    expect(getParameter(Anim.QuadInOut, 0.25)).toBeCloseTo(0.125, 10);   // 2 * 0.0625
    expect(getParameter(Anim.QuadInOut, 0.5)).toBeCloseTo(0.5, 10);      // -1 - 2*(0.5-2)*0.5 = 0.5
    expect(getParameter(Anim.Cubic, 0.5)).toBeCloseTo(0.5, 10);          // -2*0.125 + 3*0.25
    expect(getParameter(Anim.Cubic, 0)).toBe(0);
    expect(getParameter(Anim.Cubic, 1)).toBe(1);
    expect(getParameter(Anim.Exp, 0)).toBe(0);
  });

  it("animation completes — value equals stopValue at/after stopTime", () => {
    const a: Animation<V3d> = {
      kind: Anim.Linear,
      startTimeMs: 0,
      stopTimeMs: 1000,
      startValue: V3d.zero,
      stopValue: new V3d(10, 20, 30),
    };
    const v = interpolateV3(2000, a);
    expect(v.x).toBe(10); expect(v.y).toBe(20); expect(v.z).toBe(30);
    const v0 = interpolateV3(0, a);
    expect(v0.x).toBe(0); expect(v0.y).toBe(0); expect(v0.z).toBe(0);
    const half = interpolateV3(500, a);
    expect(half.x).toBeCloseTo(5, 6);
  });
});

void V2d;

// ---------------------------------------------------------------------------

interface PointerInit {
  pointerId?: number;
  pointerType?: string;
  button?: number;
  clientX?: number;
  clientY?: number;
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
  Object.defineProperty(ev, "shiftKey", { value: init.shiftKey ?? false });
  Object.defineProperty(ev, "target", { value: target });
  target.dispatchEvent(ev);
}

function fireWheel(target: EventTarget, deltaY: number, shiftKey = false): void {
  const ev = new Event("wheel", { bubbles: false, cancelable: true }) as unknown as WheelEvent;
  Object.defineProperty(ev, "deltaY", { value: deltaY });
  Object.defineProperty(ev, "deltaX", { value: 0 });
  Object.defineProperty(ev, "shiftKey", { value: shiftKey });
  Object.defineProperty(ev, "target", { value: target });
  target.dispatchEvent(ev);
}
