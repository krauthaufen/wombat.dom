// Phase 6 — drag/drop synthesis (OnDragStart / OnDrag / OnDragEnd).
// Pin: threshold, suppression of tap/click after drag, drag fires on
// the PRESS scope (not the current hover).

import { describe, expect, it } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";

import { PickDispatcher, DRAG_THRESHOLD_PX } from "../src/scene/picking/dispatcher.js";
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

function bubbleOf(rec: Record<string, (e: SceneEvent) => unknown>): EventHandlers {
  const bubble: Partial<Record<SceneEventKind, SceneEventHandler>> = {};
  for (const [k, v] of Object.entries(rec)) bubble[k as SceneEventKind] = v as SceneEventHandler;
  return { bubble };
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

function pevent(canvas: HTMLCanvasElement, type: string, x: number, y: number): PointerEvent {
  const Ctor = (globalThis as { PointerEvent?: typeof PointerEvent }).PointerEvent
            ?? (globalThis as { MouseEvent: typeof MouseEvent }).MouseEvent;
  const ev = new Ctor(type, { clientX: x, clientY: y, bubbles: true, cancelable: true, button: 0, pointerId: 1 });
  canvas.dispatchEvent(ev);
  return ev as PointerEvent;
}

async function flush(): Promise<void> { for (let i=0;i<5;i++) await Promise.resolve(); }

function acquire(reg: PickRegistry, handlers: EventHandlers[]): number {
  return reg.acquire({
    handlers: handlers.map(h => ({ handlers: h, local2World: () => AVal.constant(Trafo3d.identity) })),
    cursor: undefined, pickThrough: false,
    active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity), proj: AVal.constant(Trafo3d.identity),
    model: () => (AVal.constant(Trafo3d.identity)), pixelSnapRadius: AVal.constant(1),
  });
}

describe("PickDispatcher — drag synthesis", () => {
  it("Movement under threshold ⇒ no drag, tap fires", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const id = acquire(reg, [bubbleOf({
      OnDragStart: () => log.push("dragStart"),
      OnDrag: () => log.push("drag"),
      OnDragEnd: () => log.push("dragEnd"),
      OnTap: () => log.push("tap"),
    })]);
    const canvas = makeCanvas();
    const d = new PickDispatcher(reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect());
    const detach = d.attach(canvas, async () => makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }]));

    pevent(canvas, "pointerdown", 50, 50); await flush();
    pevent(canvas, "pointermove", 51, 50); await flush();  // <5 px
    pevent(canvas, "pointerup",   51, 50); await flush();

    expect(log).toEqual(["tap"]);
    detach();
  });

  it("Movement past threshold ⇒ DragStart, then Drag, then DragEnd; micro-drag still taps (Aardvark parity)", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const id = acquire(reg, [bubbleOf({
      OnDragStart: () => log.push("dragStart"),
      OnDrag: () => log.push("drag"),
      OnDragEnd: () => log.push("dragEnd"),
      OnTap: () => log.push("tap"),
    })]);
    const canvas = makeCanvas();
    const d = new PickDispatcher(reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect());
    const detach = d.attach(canvas, async () => makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }]));

    pevent(canvas, "pointerdown", 50, 50); await flush();
    // First move triggers DragStart.
    pevent(canvas, "pointermove", 50 + DRAG_THRESHOLD_PX + 1, 50); await flush();
    // Subsequent moves are OnDrag.
    pevent(canvas, "pointermove", 50 + DRAG_THRESHOLD_PX + 5, 50); await flush();
    pevent(canvas, "pointerup",   50 + DRAG_THRESHOLD_PX + 5, 50); await flush();

    // Net movement (10px) is within TAP_MAX_MOVE_PX (20px), so the tap
    // ALSO fires — tap detection is independent of the drag machinery
    // (Aardvark parity). A real drag exceeds 20px and taps nothing.
    expect(log).toEqual(["dragStart", "drag", "dragEnd", "tap"]);
    detach();
  });

  it("Drag fires on the press scope even when the cursor leaves geometry", async () => {
    const reg = new PickRegistry();
    const events: SceneEvent[] = [];
    const id = acquire(reg, [bubbleOf({
      OnDragStart: (e) => events.push(e),
      OnDrag: (e) => events.push(e),
      OnDragEnd: (e) => events.push(e),
    })]);
    const canvas = makeCanvas();
    const d = new PickDispatcher(reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect());
    let pressActive = false;
    const detach = d.attach(canvas, async () => {
      // Once the press is active, return a region with NO hit (cursor
      // off geometry). The drag should still fire on the press scope.
      if (pressActive) return makeRegion(50, 50, []);
      return makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }]);
    });

    pevent(canvas, "pointerdown", 50, 50); await flush();
    pressActive = true;
    pevent(canvas, "pointermove", 50 + DRAG_THRESHOLD_PX + 2, 50); await flush();
    pevent(canvas, "pointerup",   50 + DRAG_THRESHOLD_PX + 2, 50); await flush();

    // dragStart + dragEnd minimum (could include a Drag depending on flush ordering)
    expect(events.length).toBeGreaterThanOrEqual(2);
    for (const e of events) expect(e.pickId).toBe(id);
    detach();
  });
});
