// FreeFlyController joining the unified DOM↔scene walk via
// `attach(..., { input })` — the camera is a DOM-world bubble participant,
// so a scene handler that stops propagation suppresses it, and an orbit
// drag skips the scene pick. See docs/unified-event-propagation.md.

import { describe, expect, it } from "vitest";
import { AVal, cval } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";

import { PickDispatcher } from "../src/scene/picking/dispatcher.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import { pixelWinner, noPixel } from "./pickArgminTestUtil.js";
import { FreeFlyController } from "../src/scene/controllers/freefly.js";
import type { EventHandlers } from "../src/scene/sg.js";

function makeCanvas(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = 200; c.height = 100;
  document.body.appendChild(c);
  c.getBoundingClientRect = (): DOMRect => {
    const r = { x: 0, y: 0, top: 0, left: 0, right: 200, bottom: 100, width: 200, height: 100 };
    return { ...r, toJSON: () => r } as DOMRect;
  };
  return c;
}

function makeDispatcher(reg: PickRegistry, canvas: HTMLCanvasElement): PickDispatcher {
  return new PickDispatcher(
    reg,
    () => Trafo3d.identity,
    () => Trafo3d.identity,
    () => canvas.getBoundingClientRect(),
  );
}

function acquire(reg: PickRegistry, h: EventHandlers): number {
  return reg.acquire({
    handlers: [{ handlers: h, local2World: () => AVal.constant(Trafo3d.identity) }],
    cursor: undefined, pickThrough: false, active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity), proj: AVal.constant(Trafo3d.identity),
    model: () => AVal.constant(Trafo3d.identity), pixelSnapRadius: AVal.constant(1),
  });
}

function pevent(canvas: HTMLCanvasElement, type: string, x: number, y: number, extra: Record<string, unknown> = {}): PointerEvent {
  const Ctor = (globalThis as { PointerEvent?: typeof PointerEvent }).PointerEvent
            ?? (globalThis as { MouseEvent: typeof MouseEvent }).MouseEvent;
  const ev = new Ctor(type, { clientX: x, clientY: y, bubbles: true, cancelable: true, button: 0 }) as PointerEvent;
  Object.defineProperty(ev, "pointerId", { value: 1, configurable: true });
  Object.defineProperty(ev, "pointerType", { value: "mouse", configurable: true });
  for (const [k, v] of Object.entries(extra)) Object.defineProperty(ev, k, { value: v, configurable: true });
  canvas.dispatchEvent(ev);
  return ev;
}

async function flush(): Promise<void> {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

describe("FreeFlyController — unified input", () => {
  it("orbits on a drag over empty sky (camera bubble runs; moves skip the pick)", async () => {
    const reg = new PickRegistry();
    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const ctrl = new FreeFlyController();
    const before = ctrl.state.value.TargetTurn;
    const detach = ctrl.attach(canvas, cval(0), { input: d });
    d.attach(canvas, async () => noPixel());

    pevent(canvas, "pointerdown", 50, 50);
    await flush();                                   // camera claims the pointer
    pevent(canvas, "pointermove", 60, 50, { movementX: 10, movementY: 0 });
    await flush();

    const after = ctrl.state.value.TargetTurn;
    expect(after.x).not.toBe(before.x);              // rotated
    detach();
  });

  it("a scene handler that stops propagation suppresses the camera", async () => {
    const reg = new PickRegistry();
    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    // Marker claims the press: stops the down so the camera never starts.
    const id = acquire(reg, { bubble: { OnPointerDown: (e) => { e.stopPropagation(); } } });
    const ctrl = new FreeFlyController();
    const detach = ctrl.attach(canvas, cval(0), { input: d });
    d.attach(canvas, async () => pixelWinner(id, { px: 50, py: 50, dist2: 0 }));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();
    pevent(canvas, "pointermove", 70, 50, { movementX: 20, movementY: 0 });
    await flush();

    // Camera never received the down → rotDown stayed false → no rotation.
    expect(ctrl.state.value.TargetTurn.x).toBe(0);
    expect(ctrl.state.value.TargetTurn.y).toBe(0);
    detach();
  });
});
