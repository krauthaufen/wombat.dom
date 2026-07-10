// Coverage for the controller additions:
//   - FreeFlyController.flyTo(location, forward)
//   - virtual-stick UI overlay (opt-in)
//   - OrbitController MMB pick-aware snap
//   - OrbitConfig.freeMovePan toggle
//   - OrbitConfig.springConstants per-axis tuning

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cval, transact } from "@aardworx/wombat.adaptive";
import { V2d, V3d } from "@aardworx/wombat.base";

import {
  FreeFlyController,
  freeFlyIsAnimating,
  OrbitController,
  deriveView,
} from "../src/scene/index.js";

function makeTarget(): HTMLElement {
  const el = document.createElement("div");
  (el as unknown as { setPointerCapture?: () => void }).setPointerCapture = () => {};
  (el as unknown as { releasePointerCapture?: () => void }).releasePointerCapture = () => {};
  document.body.appendChild(el);
  el.getBoundingClientRect = (): DOMRect => {
    const r = { x: 0, y: 0, top: 0, left: 0, right: 200, bottom: 100, width: 200, height: 100 };
    return { ...r, toJSON: () => r } as DOMRect;
  };
  return el;
}

describe("FreeFlyController.flyTo", () => {
  it("calling flyTo while idle starts an animation; ticks converge Position to location and Forward to forward", () => {
    const ctl = FreeFlyController.create({
      position: V3d.zero,
      forward: new V3d(1, 0, 0),       // looking along +X
      sky: new V3d(0, 0, 1),
    });
    expect(freeFlyIsAnimating(ctl.state.value)).toBe(false);

    const target = new V3d(2, 0, 3);
    // Rotate ~30° in horizontal plane and tilt up a bit; both deltas are small enough for the planar dx/dy math.
    const fwd = new V3d(Math.cos(0.5), Math.sin(0.5), 0.2).normalize();
    ctl.flyTo(target, fwd);
    expect(freeFlyIsAnimating(ctl.state.value)).toBe(true);

    for (let i = 0; i < 200; i++) ctl.tick(0.05);

    const pos = ctl.state.value.Position;
    expect(pos.sub(target).length()).toBeLessThan(0.05);
    const f = ctl.state.value.Forward.normalize();
    expect(f.sub(fwd.normalize()).length()).toBeLessThan(0.05);
  });
});

describe("FreeFlyController.attach({ virtualSticks })", () => {
  // Force the matchMedia (pointer: coarse) check to true so happy-dom is treated as a touch device.
  beforeEach(() => {
    if (typeof window !== "undefined") {
      window.matchMedia = ((q: string) => ({
        matches: q.includes("coarse"),
        media: q, onchange: null,
        addListener: () => {}, removeListener: () => {},
        addEventListener: () => {}, removeEventListener: () => {}, dispatchEvent: () => false,
      })) as unknown as typeof window.matchMedia;
    }
  });

  it("simulating pointerdown+pointermove on the left stick produces a non-zero MoveVec", () => {
    const ctl = FreeFlyController.create();
    const target = makeTarget();
    const time = cval(0);
    const detach = ctl.attach(target, time, { virtualSticks: true });

    const left = target.querySelector('[data-virtual-stick="left"]') as HTMLDivElement | null;
    expect(left).not.toBeNull();
    // Stick rect: 24px from left, 24px from bottom. body has no layout in
    // happy-dom; we just call the listeners with a synthetic offset.
    left!.getBoundingClientRect = (): DOMRect => {
      const r = { x: 24, y: 0, top: 0, left: 24, right: 144, bottom: 120, width: 120, height: 120 };
      return { ...r, toJSON: () => r } as DOMRect;
    };
    const fire = (type: string, x: number, y: number): void => {
      const Ctor = (globalThis as { PointerEvent?: typeof PointerEvent }).PointerEvent
                ?? (globalThis as { MouseEvent: typeof MouseEvent }).MouseEvent;
      const ev = new Ctor(type, { clientX: x, clientY: y, bubbles: true, cancelable: true });
      Object.defineProperty(ev, "pointerId", { value: 1, configurable: true });
      left!.dispatchEvent(ev);
    };
    fire("pointerdown", 84, 60);  // centre of the stick
    fire("pointermove", 130, 60); // push right
    // Move should now have non-zero X.
    let total = V3d.zero;
    for (const v of ctl.state.value.MoveVectors.values()) total = total.add(v);
    expect(Math.abs(total.x)).toBeGreaterThan(0);

    // The right stick mirrors the same setup for TurnVec.
    const right = target.querySelector('[data-virtual-stick="right"]') as HTMLDivElement | null;
    expect(right).not.toBeNull();
    right!.getBoundingClientRect = (): DOMRect => {
      const r = { x: 56, y: 0, top: 0, left: 56, right: 176, bottom: 120, width: 120, height: 120 };
      return { ...r, toJSON: () => r } as DOMRect;
    };
    const fireR = (type: string, x: number, y: number): void => {
      const Ctor = (globalThis as { PointerEvent?: typeof PointerEvent }).PointerEvent
                ?? (globalThis as { MouseEvent: typeof MouseEvent }).MouseEvent;
      const ev = new Ctor(type, { clientX: x, clientY: y, bubbles: true, cancelable: true });
      Object.defineProperty(ev, "pointerId", { value: 2, configurable: true });
      right!.dispatchEvent(ev);
    };
    fireR("pointerdown", 116, 60);
    fireR("pointermove", 116, 30);  // push up
    let totalT = new V2d(0, 0);
    for (const v of ctl.state.value.TurnVectors.values()) totalT = totalT.add(v);
    expect(Math.abs(totalT.x) + Math.abs(totalT.y)).toBeGreaterThan(0);

    detach();
    // Detach should remove the overlays.
    expect(target.querySelector('[data-virtual-stick="left"]')).toBeNull();
    void transact;
  });
});

describe("OrbitController — pick-anchored navigation", () => {
  it("rotate drag pivots the rig around the picked point; center only swings, no snap", async () => {
    const ctl = OrbitController.create({ center: V3d.zero, radius: 5, phi: 0.3, theta: 0.4 });
    const target = makeTarget();
    const time = cval(performance.now());
    const hit = new V3d(1.5, -0.5, 0.25);
    const detach = ctl.attach(target, time, {
      picker: async () => hit,
    });

    const before = ctl.state.value;
    const eyeBefore = deriveView(before).eye;
    const pivotDistBefore = eyeBefore.sub(hit).length();

    // Rotate-button down (L). Anchor resolves on a microtask; the DOWN
    // itself must not touch the state (no snap).
    firePointer(target, "pointerdown", { pointerId: 1, button: 0, clientX: 100, clientY: 50, pointerType: "mouse" });
    for (let i = 0; i < 5; i++) await Promise.resolve();
    expect(ctl.state.value.center.sub(before.center).length()).toBeLessThan(1e-12);
    expect(deriveView(ctl.state.value).eye.sub(eyeBefore).length()).toBeLessThan(1e-12);

    // Drag: the rig rotates rigidly around the pivot — the eye keeps
    // its distance to the pivot AND to the center (radius unchanged);
    // the center swings around the pivot.
    firePointer(target, "pointermove", { pointerId: 1, button: 0, clientX: 160, clientY: 70, pointerType: "mouse" });
    const s = ctl.state.value;
    const eyeAfter = deriveView(s).eye;
    expect(Math.abs(eyeAfter.sub(hit).length() - pivotDistBefore)).toBeLessThan(1e-9);
    expect(Math.abs(s.radius - before.radius)).toBeLessThan(1e-9);
    expect(s.center.sub(before.center).length()).toBeGreaterThan(1e-3);
    expect(s.phi).not.toBeCloseTo(before.phi, 6);

    firePointer(target, "pointerup", { pointerId: 1, button: 0, clientX: 160, clientY: 70, pointerType: "mouse" });
    detach();
  });

  it("pan-down anchors without re-centring; background rotate keeps the center", async () => {
    const ctl = OrbitController.create({ center: new V3d(2, 2, 2), radius: 5, phi: 0, theta: 0.3 });
    const target = makeTarget();
    const time = cval(performance.now());
    let answer: V3d | undefined = new V3d(9, 9, 9);
    const detach = ctl.attach(target, time, {
      picker: async () => answer,
    });

    // Pan-button down (MMB): the hit becomes the pan anchor — the
    // orbit center must NOT jump.
    firePointer(target, "pointerdown", { pointerId: 1, button: 1, clientX: 100, clientY: 50, pointerType: "mouse" });
    for (let i = 0; i < 5; i++) await Promise.resolve();
    expect(ctl.state.value.center.sub(new V3d(2, 2, 2)).length()).toBeLessThan(1e-9);
    firePointer(target, "pointerup", { pointerId: 1, button: 1, clientX: 100, clientY: 50, pointerType: "mouse" });

    // Background rotate (picker misses): center unchanged.
    answer = undefined;
    firePointer(target, "pointerdown", { pointerId: 2, button: 0, clientX: 100, clientY: 50, pointerType: "mouse" });
    for (let i = 0; i < 5; i++) await Promise.resolve();
    expect(ctl.state.value.center.sub(new V3d(2, 2, 2)).length()).toBeLessThan(1e-9);
    firePointer(target, "pointerup", { pointerId: 2, button: 0, clientX: 100, clientY: 50, pointerType: "mouse" });
    detach();
  });
});

describe("OrbitController — freeMovePan toggle", () => {
  it("freeMovePan = false suppresses pan along screen-Y", () => {
    const ctl = OrbitController.create({
      center: V3d.zero, radius: 5,
      config: { freeMovePan: false },
    });
    const target = makeTarget();
    const time = cval(0);
    const detach = ctl.attach(target, time);

    // MMB drag with vertical-only mouse motion. The world-space center
    // shift along the up-axis (right=Y in our derived view) should be
    // zero with freeMovePan=false.
    firePointer(target, "pointerdown", { pointerId: 1, button: 1, clientX: 100, clientY: 50, pointerType: "mouse" });
    firePointer(target, "pointermove", { pointerId: 1, button: 1, clientX: 100, clientY: 80, pointerType: "mouse" });
    const c1 = ctl.state.value.center;
    // Reference: the same drag with freeMovePan=true should change center.
    detach();

    const ctl2 = OrbitController.create({
      center: V3d.zero, radius: 5,
      config: { freeMovePan: true },
    });
    const t2 = makeTarget();
    const tm2 = cval(0);
    const d2 = ctl2.attach(t2, tm2);
    firePointer(t2, "pointerdown", { pointerId: 1, button: 1, clientX: 100, clientY: 50, pointerType: "mouse" });
    firePointer(t2, "pointermove", { pointerId: 1, button: 1, clientX: 100, clientY: 80, pointerType: "mouse" });
    const c2 = ctl2.state.value.center;
    d2();

    // freeMovePan=false → no Y shift; with default sky=Z, "up" = Z axis.
    // freeMovePan=true → non-zero center change.
    expect(c1.length()).toBeLessThan(1e-9);
    expect(c2.length()).toBeGreaterThan(1e-6);
  });
});

describe("OrbitController — springConstants", () => {
  it("high spring on phi closes faster than low spring on theta with equal target deltas", () => {
    const ctl = OrbitController.create({
      center: V3d.zero, radius: 5, phi: 0, theta: 0,
      // (we still want a Partial<OrbitConfig>; OrbitInitial.config is partial)
    });
    // Re-create with partial config — the discriminator above also wants
    // `config` to push us into the OrbitState branch, so we use the
    // setter API instead.
    transact(() => {
      ctl.state.value = {
        ...ctl.state.value,
        config: {
          ...ctl.state.value.config,
          springConstants: { phi: 5, theta: 0.05, radius: 1, center: 1 },
        },
      };
    });
    // Equal deltas of 0.4 rad on both axes (within thetaRange).
    ctl.setTargetPhi(0.4);
    ctl.setTargetTheta(0.4);

    // Tick: nowMs starts at lastRenderMs=undefined → 0 motion; second
    // tick uses dt = 16 ms.
    ctl.tick(0);
    ctl.tick(16);

    const dPhi = Math.abs(ctl.state.value.phi - 0);
    const dTheta = Math.abs(ctl.state.value.theta - 0);
    expect(dPhi).toBeGreaterThan(dTheta);
  });
});

// ---------------------------------------------------------------------------

interface PointerInit {
  pointerId?: number;
  pointerType?: string;
  button?: number;
  clientX?: number;
  clientY?: number;
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
  Object.defineProperty(ev, "shiftKey", { value: false });
  Object.defineProperty(ev, "target", { value: target });
  target.dispatchEvent(ev);
}
