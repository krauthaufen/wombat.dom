// Camera helpers + controllers — math + behaviour tests.
//
// All math runs without GPU. Controllers use `loop: "manual"` and
// `tick(dt)` for deterministic stepping; we synthesise pointer +
// keyboard events against happy-dom elements to drive them.

import { describe, expect, it } from "vitest";
import { AVal, cval, transact } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d } from "@aardworx/wombat.base";

import {
  freeFlyController,
  lookAt,
  orbitController,
  orthographic,
  perspective,
  perspectiveCamera,
} from "../src/scene/index.js";

// ---------------------------------------------------------------------------
// View / projection math
// ---------------------------------------------------------------------------

describe("lookAt", () => {
  it("places origin at eye + looks toward target", () => {
    const view = lookAt({
      eye: new V3d(0, 0, 5),
      target: V3d.zero,
      up: new V3d(0, 1, 0),
    });
    const t = AVal.force(view);
    // Eye at (0,0,5) → in view space, the eye is at origin.
    const eyeInView = t.transform(new V3d(0, 0, 5));
    expect(eyeInView.x).toBeCloseTo(0, 6);
    expect(eyeInView.y).toBeCloseTo(0, 6);
    expect(eyeInView.z).toBeCloseTo(0, 6);
    // World-origin (the target) should land in front of the eye:
    // RH view → looking down -Z, so target.z < 0 in view space.
    const targetInView = t.transform(V3d.zero);
    expect(targetInView.z).toBeCloseTo(-5, 6);
  });

  it("reactive eye → reactive view", () => {
    const eye = cval(new V3d(0, 0, 5));
    const view = lookAt({ eye, target: V3d.zero, up: new V3d(0, 1, 0) });
    const t1 = AVal.force(view);
    transact(() => { eye.value = new V3d(0, 0, 10); });
    const t2 = AVal.force(view);
    expect(t1).not.toBe(t2);
    const targetInView = t2.transform(V3d.zero);
    expect(targetInView.z).toBeCloseTo(-10, 6);
  });
});

describe("perspective", () => {
  it("maps near plane to NDC z=0 and far plane to z=1 (RH/WebGPU)", () => {
    const p = AVal.force(perspective({
      fovInRadians: Math.PI / 2,  // 90° horizontal
      aspect: 1,
      near: 1,
      far: 100,
    }));
    // Project a point on the near plane (eye-space z = -near).
    const near = p.transform(new V3d(0, 0, -1));
    expect(near.z).toBeCloseTo(0, 6);
    // ...and one on the far plane.
    const far = p.transform(new V3d(0, 0, -100));
    expect(far.z).toBeCloseTo(1, 6);
  });

  it("aspect changes update the projection reactively", () => {
    const aspect = cval(1);
    const p = perspective({ fovInRadians: Math.PI / 2, aspect, near: 1, far: 100 });
    const t1 = AVal.force(p);
    transact(() => { aspect.value = 2; });
    const t2 = AVal.force(p);
    expect(t1).not.toBe(t2);
  });
});

describe("orthographic", () => {
  it("maps the corner of the box to NDC (-1, -1, 0)", () => {
    const p = AVal.force(orthographic({
      left: -1, right: 1, bottom: -1, top: 1, near: 1, far: 100,
    }));
    const ll = p.transform(new V3d(-1, -1, -1));
    expect(ll.x).toBeCloseTo(-1, 6);
    expect(ll.y).toBeCloseTo(-1, 6);
    expect(ll.z).toBeCloseTo(0, 6);
  });
});

describe("perspectiveCamera", () => {
  it("auto-tracks viewport aspect", () => {
    const view = AVal.constant(Trafo3d.identity);
    const viewport = cval({ width: 800, height: 400 });
    const cam = perspectiveCamera({ view, viewport, fovInRadians: Math.PI / 2, near: 1, far: 100 });
    const before = AVal.force(cam.proj);
    transact(() => { viewport.value = { width: 400, height: 800 }; });
    const after = AVal.force(cam.proj);
    expect(before).not.toBe(after);
  });
});

// ---------------------------------------------------------------------------
// Free-fly controller
// ---------------------------------------------------------------------------

describe("freeFlyController", () => {
  it("WASD moves the eye along the local basis after tick", () => {
    const ctl = freeFlyController({
      eye: new V3d(0, 0, 5),
      forward: new V3d(0, 0, -1),
      up: new V3d(0, 1, 0),
      moveSpeed: 1,
      loop: "manual",
    });
    const target = makeKeyTarget();
    const att = ctl.attach(target);

    // Press W → forward (negative-Z in this setup).
    fireKey(target, "keydown", "w");
    ctl.tick(1);  // 1 second
    expect(AVal.force(ctl.eye).z).toBeCloseTo(4, 5);

    // Release W, press D → strafe right (+X).
    fireKey(target, "keyup", "w");
    fireKey(target, "keydown", "d");
    ctl.tick(1);
    expect(AVal.force(ctl.eye).x).toBeCloseTo(1, 5);

    att.dispose();
    ctl.dispose();
  });

  it("drag accumulates yaw/pitch and applies on the next tick", () => {
    const ctl = freeFlyController({
      eye: new V3d(0, 0, 5),
      forward: new V3d(0, 0, -1),
      up: new V3d(0, 1, 0),
      turnSensitivity: 0.01,
      loop: "manual",
    });
    const target = makeKeyTarget();
    ctl.attach(target);

    const fwd0 = AVal.force(ctl.target).sub(AVal.force(ctl.eye)).normalize();
    expect(fwd0.x).toBeCloseTo(0, 6);

    firePointer(target, "pointerdown", { button: 0, pointerId: 1 });
    firePointer(target, "pointermove", { button: 0, pointerId: 1, movementX: 100, movementY: 0 });
    ctl.tick(0);
    const fwd1 = AVal.force(ctl.target).sub(AVal.force(ctl.eye)).normalize();
    // Yaw rotated forward — x component should now be non-zero.
    expect(Math.abs(fwd1.x)).toBeGreaterThan(0.5);

    ctl.dispose();
  });
});

// ---------------------------------------------------------------------------
// Orbit controller
// ---------------------------------------------------------------------------

describe("orbitController", () => {
  it("starts with the eye at distance D from the target", () => {
    const ctl = orbitController({
      target: V3d.zero,
      distance: 7,
      yaw: 0,
      pitch: 0,
      up: new V3d(0, 1, 0),
      loop: "manual",
    });
    const eye = AVal.force(ctl.eye);
    const tgt = AVal.force(ctl.target);
    const d = eye.sub(tgt).length();
    expect(d).toBeCloseTo(7, 5);
    ctl.dispose();
  });

  it("wheel scroll changes distance multiplicatively", () => {
    const ctl = orbitController({
      target: V3d.zero,
      distance: 5,
      up: new V3d(0, 1, 0),
      zoomFactor: 1.5,
      loop: "manual",
    });
    const target = makeKeyTarget();
    ctl.attach(target);

    fireWheel(target, 100);  // deltaY > 0 → zoom out
    ctl.tick(0);
    let d = AVal.force(ctl.eye).sub(AVal.force(ctl.target)).length();
    expect(d).toBeCloseTo(5 * 1.5, 5);

    fireWheel(target, -100); // zoom in
    ctl.tick(0);
    d = AVal.force(ctl.eye).sub(AVal.force(ctl.target)).length();
    expect(d).toBeCloseTo(5, 5);  // back where we started

    ctl.dispose();
  });

  it("drag rotates yaw — eye moves around target on the same orbit sphere", () => {
    const ctl = orbitController({
      target: V3d.zero,
      distance: 5,
      yaw: 0,
      pitch: 0,
      up: new V3d(0, 1, 0),
      turnSensitivity: 0.01,
      loop: "manual",
    });
    const target = makeKeyTarget();
    ctl.attach(target);
    const eye0 = AVal.force(ctl.eye);

    firePointer(target, "pointerdown", { button: 0, pointerId: 1 });
    firePointer(target, "pointermove", { button: 0, pointerId: 1, movementX: 100, movementY: 0 });
    firePointer(target, "pointerup",   { button: 0, pointerId: 1 });
    ctl.tick(0);
    const eye1 = AVal.force(ctl.eye);
    expect(eye1.sub(eye0).length()).toBeGreaterThan(0.1);
    // Distance to target preserved.
    expect(eye1.length()).toBeCloseTo(5, 5);

    ctl.dispose();
  });

  it("dispose stops listening and is idempotent", () => {
    const ctl = orbitController({ loop: "manual" });
    const target = makeKeyTarget();
    ctl.attach(target);
    ctl.dispose();
    expect(() => ctl.dispose()).not.toThrow();

    // Events after dispose should NOT change state.
    const eye0 = AVal.force(ctl.eye);
    firePointer(target, "pointerdown", { button: 0, pointerId: 1 });
    firePointer(target, "pointermove", { button: 0, pointerId: 1, movementX: 100, movementY: 0 });
    firePointer(target, "pointerup",   { button: 0, pointerId: 1 });
    ctl.tick(0);
    expect(AVal.force(ctl.eye)).toEqual(eye0);
  });
});

// ---------------------------------------------------------------------------
// Test helpers — synthetic events for happy-dom
// ---------------------------------------------------------------------------

function makeKeyTarget(): HTMLElement {
  const el = document.createElement("div");
  // happy-dom accepts addEventListener calls fine, but pointer-capture
  // methods may be missing on the underlying element. Stub them.
  (el as unknown as { setPointerCapture?: () => void }).setPointerCapture = () => {};
  (el as unknown as { releasePointerCapture?: () => void }).releasePointerCapture = () => {};
  return el;
}

function fireKey(target: EventTarget, type: "keydown" | "keyup", key: string): void {
  // happy-dom prefixes event handlers via the global Event constructor.
  const ev = new Event(type) as unknown as KeyboardEvent;
  Object.defineProperty(ev, "key", { value: key });
  // In happy-dom a window-level listener fires from window dispatch; for
  // simplicity, dispatch on the window so the controller's
  // window.addEventListener("keydown") handler picks it up.
  window.dispatchEvent(ev);
  void target;
}

interface PointerInit {
  button?: number;
  pointerId?: number;
  movementX?: number;
  movementY?: number;
}

function firePointer(
  target: EventTarget,
  type: "pointerdown" | "pointermove" | "pointerup" | "pointercancel",
  init: PointerInit,
): void {
  const ev = new Event(type, { bubbles: false, cancelable: true }) as unknown as PointerEvent;
  Object.defineProperty(ev, "button", { value: init.button ?? 0 });
  Object.defineProperty(ev, "pointerId", { value: init.pointerId ?? 1 });
  Object.defineProperty(ev, "movementX", { value: init.movementX ?? 0 });
  Object.defineProperty(ev, "movementY", { value: init.movementY ?? 0 });
  Object.defineProperty(ev, "target", { value: target });
  target.dispatchEvent(ev);
}

function fireWheel(target: EventTarget, deltaY: number): void {
  const ev = new Event("wheel", { bubbles: false, cancelable: true }) as unknown as WheelEvent;
  Object.defineProperty(ev, "deltaY", { value: deltaY });
  Object.defineProperty(ev, "target", { value: target });
  target.dispatchEvent(ev);
}
