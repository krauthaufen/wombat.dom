// Discrete pointer events (click, down, up) are never dropped as stale:
// only moves are lossy. A click whose pick resolves after a newer move's
// (a coalesced-away move settles at once, ahead of a click still queued
// behind the in-flight pick) must still reach its handlers.

import { describe, expect, it } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";

import { PickDispatcher } from "../src/scene/picking/dispatcher.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import type { PickArgminResult } from "../src/scene/picking/pickArgminCompute.js";
import type { EventHandlers } from "../src/scene/sg.js";
import { pixelWinner } from "./pickArgminTestUtil.js";

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

function pevent(canvas: HTMLCanvasElement, type: string, x: number, y: number): void {
  const Ctor = (globalThis as { PointerEvent?: typeof PointerEvent }).PointerEvent
            ?? (globalThis as { MouseEvent: typeof MouseEvent }).MouseEvent;
  const ev = new Ctor(type, { clientX: x, clientY: y, bubbles: true, cancelable: true, button: 0 }) as PointerEvent;
  Object.defineProperty(ev, "pointerId", { value: 1, configurable: true });
  Object.defineProperty(ev, "pointerType", { value: "mouse", configurable: true });
  canvas.dispatchEvent(ev);
}

async function flush(): Promise<void> {
  for (let i = 0; i < 8; i++) await Promise.resolve();
}

function acquire(reg: PickRegistry, handlers: EventHandlers): number {
  return reg.acquire({
    handlers: [{ handlers, local2World: () => AVal.constant(Trafo3d.identity) }],
    cursor: undefined,
    pickThrough: false,
    active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity),
    proj: AVal.constant(Trafo3d.identity),
    model: () => AVal.constant(Trafo3d.identity),
    pixelSnapRadius: AVal.constant(1),
  });
}

describe("PickDispatcher — discrete events survive newer moves", () => {
  it("a click resolving after a newer move still dispatches", async () => {
    const reg = new PickRegistry();
    let clicks = 0;
    const id = acquire(reg, { bubble: { OnClick: () => { clicks++; } } });
    const canvas = makeCanvas();
    const d = new PickDispatcher(reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect());

    // Picks resolve when the test says so: the click's first, held back,
    // while the move after it resolves at once.
    const held: Array<(r: PickArgminResult) => void> = [];
    const resolvePixel = (_x: number, _y: number, coalesce?: boolean): Promise<PickArgminResult> =>
      coalesce === true
        ? Promise.resolve(pixelWinner(id, { px: 60, py: 50, dist2: 0 }))
        : new Promise((r) => { held.push(r); });
    const detach = d.attach(canvas, resolvePixel as never);

    pevent(canvas, "click", 50, 50);
    pevent(canvas, "pointermove", 60, 50);
    await flush();
    expect(clicks).toBe(0);          // the click's pick is still out
    held.forEach((r) => r(pixelWinner(id, { px: 50, py: 50, dist2: 0 })));
    await flush();
    expect(clicks).toBe(1);
    detach();
  });

  it("an overtaken move is still dropped", async () => {
    const reg = new PickRegistry();
    let moves = 0;
    const id = acquire(reg, { bubble: { OnPointerMove: () => { moves++; } } });
    const canvas = makeCanvas();
    const d = new PickDispatcher(reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect());
    const held: Array<(r: PickArgminResult) => void> = [];
    let n = 0;
    const resolvePixel = (): Promise<PickArgminResult> =>
      n++ === 0 ? new Promise((r) => { held.push(r); }) : Promise.resolve(pixelWinner(id, { px: 50, py: 50, dist2: 0 }));
    const detach = d.attach(canvas, resolvePixel as never);

    pevent(canvas, "pointermove", 50, 50);  // held
    pevent(canvas, "pointermove", 51, 50);  // settles first
    await flush();
    held.forEach((r) => r(pixelWinner(id, { px: 50, py: 50, dist2: 0 })));
    await flush();
    expect(moves).toBe(1);
    detach();
  });
});
