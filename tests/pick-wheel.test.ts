// Phase 4 — OnWheel dispatch through capture/bubble. Mirrors
// pick-dispatch.test.ts shape.

import { describe, expect, it } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";

import { PickDispatcher } from "../src/scene/picking/dispatcher.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import type { PickArgminResult } from "../src/scene/picking/pickArgminCompute.js";
import { noPixel, pixelWinner } from "./pickArgminTestUtil.js";
import type { SceneEvent, SceneEventKind } from "../src/scene/picking/sceneEvent.js";
import type { EventHandlers, SceneEventHandler } from "../src/scene/sg.js";

function makeCanvas(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = 200; c.height = 100; document.body.appendChild(c);
  c.getBoundingClientRect = (): DOMRect => {
    const r = { x: 0, y: 0, top: 0, left: 0, right: 200, bottom: 100, width: 200, height: 100 };
    return { ...r, toJSON: () => r } as DOMRect;
  };
  return c;
}

function makeRegion(centerX: number, centerY: number, stamps: ReadonlyArray<{ dx: number; dy: number; pickId: number }>): PickArgminResult {
  // Argmin verdict: the nearest stamp to the cursor is the winner
  // (the per-pixel snap/MSAA gating now lives in the GPU kernel).
  if (stamps.length === 0) return noPixel();
  let best = stamps[0]!;
  let bestD = best.dx * best.dx + best.dy * best.dy;
  for (const s of stamps) {
    const d = s.dx * s.dx + s.dy * s.dy;
    if (d < bestD) { best = s; bestD = d; }
  }
  return pixelWinner(best.pickId, { px: centerX + best.dx, py: centerY + best.dy, dist2: bestD });
}

function bubbleOf(rec: Record<string, (e: SceneEvent) => unknown>): EventHandlers {
  const bubble: Partial<Record<SceneEventKind, SceneEventHandler>> = {};
  for (const [k, v] of Object.entries(rec)) bubble[k as SceneEventKind] = v as SceneEventHandler;
  return { bubble };
}

function acquire(reg: PickRegistry, handlers: ReadonlyArray<EventHandlers>): number {
  return reg.acquire({
    handlers: handlers.map(h => ({ handlers: h, local2World: () => AVal.constant(Trafo3d.identity) })),
    cursor: undefined, pickThrough: false,
    active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity), proj: AVal.constant(Trafo3d.identity),
    model: () => (AVal.constant(Trafo3d.identity)), pixelSnapRadius: AVal.constant(1),
  });
}

async function flush(): Promise<void> {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

describe("PickDispatcher OnWheel", () => {
  it("dispatches OnWheel on the spiral hit scope with deltaX/Y/mode populated", async () => {
    const reg = new PickRegistry();
    const calls: SceneEvent[] = [];
    const id = acquire(reg, [bubbleOf({ OnWheel: (e) => calls.push(e) })]);
    const canvas = makeCanvas();
    const d = new PickDispatcher(reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect());
    const detach = d.attach(canvas, async () => makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }]));

    const ev = new WheelEvent("wheel", { deltaX: 1, deltaY: 5, deltaZ: 0, deltaMode: 0, bubbles: true, cancelable: true });
    Object.defineProperty(ev, "clientX", { value: 50 });
    Object.defineProperty(ev, "clientY", { value: 50 });
    canvas.dispatchEvent(ev);
    await flush();

    expect(calls).toHaveLength(1);
    expect(calls[0]!.kind).toBe("OnWheel");
    expect(calls[0]!.deltaX).toBe(1);
    expect(calls[0]!.deltaY).toBe(5);
    expect(calls[0]!.deltaMode).toBe(0);
    expect(calls[0]!.pickId).toBe(id);
    detach();
  });

  it("preventDefault propagates to the underlying WheelEvent", async () => {
    const reg = new PickRegistry();
    const id = acquire(reg, [bubbleOf({ OnWheel: (e) => { e.preventDefault(); } })]);
    const canvas = makeCanvas();
    const d = new PickDispatcher(reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect());
    const detach = d.attach(canvas, async () => makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }]));

    const ev = new WheelEvent("wheel", { deltaY: 1, bubbles: true, cancelable: true });
    Object.defineProperty(ev, "clientX", { value: 50 });
    Object.defineProperty(ev, "clientY", { value: 50 });
    canvas.dispatchEvent(ev);
    await flush();
    expect(ev.defaultPrevented).toBe(true);
    detach();
  });

  it("no hit ⇒ no dispatch", async () => {
    const reg = new PickRegistry();
    const calls: SceneEvent[] = [];
    acquire(reg, [bubbleOf({ OnWheel: (e) => calls.push(e) })]);
    const canvas = makeCanvas();
    const d = new PickDispatcher(reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect());
    const detach = d.attach(canvas, async () => makeRegion(50, 50, []));
    canvas.dispatchEvent(new WheelEvent("wheel", { clientX: 50, clientY: 50, deltaY: 1 }));
    await flush();
    expect(calls).toHaveLength(0);
    detach();
  });
});
