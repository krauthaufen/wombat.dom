// Coverage for the input/picking additions:
//   - configurable tap/long-press thresholds via PickDispatcher ctor
//   - PointerCapture release on scope unmount
//   - synthesised pinch / two-finger pan / two-finger rotate events
//   - hover dwell (OnHover) timer

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";

import { PickDispatcher, LONG_PRESS_MS, HOVER_DELAY_MS } from "../src/scene/picking/dispatcher.js";
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

function makeRegion(centerX: number, centerY: number, pickId: number): PickArgminResult {
  // Argmin verdict: pickId is the centre-pixel winner (0 ⇒ no hit).
  return pickId === 0 ? noPixel() : pixelWinner(pickId, { px: centerX, py: centerY });
}

function pevent(canvas: HTMLCanvasElement, type: string, x: number, y: number, pointerId = 1): PointerEvent {
  const Ctor = (globalThis as { PointerEvent?: typeof PointerEvent }).PointerEvent
            ?? (globalThis as { MouseEvent: typeof MouseEvent }).MouseEvent;
  const ev = new Ctor(type, { clientX: x, clientY: y, bubbles: true, cancelable: true, button: 0 });
  Object.defineProperty(ev, "pointerId", { value: pointerId, configurable: true });
  canvas.dispatchEvent(ev);
  return ev as PointerEvent;
}

async function flush(): Promise<void> { for (let i = 0; i < 5; i++) await Promise.resolve(); }

function acquire(reg: PickRegistry, handlers: EventHandlers[]): number {
  return reg.acquire({
    handlers: handlers.map(h => ({ handlers: h, local2World: () => AVal.constant(Trafo3d.identity) })),
    cursor: undefined, pickThrough: false,
    active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity), proj: AVal.constant(Trafo3d.identity),
    model: () => (AVal.constant(Trafo3d.identity)), pixelSnapRadius: AVal.constant(1),
  });
}

describe("PickDispatcher — configurable thresholds", () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); });

  it("override longPressMs to 1000 — a 600 ms hold does NOT fire OnLongPress", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const id = acquire(reg, [bubbleOf({
      OnLongPress: () => log.push("long"),
      OnTap: () => log.push("tap"),
    })]);
    const canvas = makeCanvas();
    const d = new PickDispatcher(
      reg,
      () => Trafo3d.identity, () => Trafo3d.identity,
      () => canvas.getBoundingClientRect(),
      { longPressMs: 1000 },
    );
    const detach = d.attach(canvas, async () => makeRegion(50, 50, id));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();
    vi.advanceTimersByTime(600);
    expect(log).toEqual([]);  // longPress would have fired at 500 with the default
    pevent(canvas, "pointerup", 50, 50);
    await flush();
    // 600 ms hold: still well past tap-max-duration (250) — no tap.
    expect(log).toEqual([]);

    // For comparison, the default constant is shorter: 600 ms > 500 ms default.
    expect(LONG_PRESS_MS).toBeLessThan(1000);
    detach();
  });
});

describe("PickDispatcher — pointer capture released on scope unmount", () => {
  it("after registry.clear(), captured pointer's events route via spiral instead of the captured scope", async () => {
    const reg = new PickRegistry();
    const onA: string[] = [];
    const onB: string[] = [];
    const idA = acquire(reg, [bubbleOf({
      OnPointerDown: (ev: SceneEvent) => { onA.push("downA"); ev.setPointerCapture(); },
      OnPointerMove: () => onA.push("moveA"),
    })]);

    const canvas = makeCanvas();
    const d = new PickDispatcher(
      reg, () => Trafo3d.identity, () => Trafo3d.identity,
      () => canvas.getBoundingClientRect(),
    );
    let curId = idA;
    const detach = d.attach(canvas, async (x, y) => makeRegion(x, y, curId));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();
    expect(onA).toEqual(["downA"]);

    // Move while captured: routes to A despite no spiral hit consideration.
    pevent(canvas, "pointermove", 60, 50);
    await flush();
    expect(onA).toEqual(["downA", "moveA"]);

    // Wipe the registry — emulates the scope being unmounted.
    reg.clear();
    // Re-register a NEW scope for the spiral hit to land on.
    const idB = reg.acquire({
      handlers: [{ handlers: bubbleOf({ OnPointerMove: () => onB.push("moveB") }), local2World: () => AVal.constant(Trafo3d.identity) }],
      cursor: undefined, pickThrough: false,
      active: AVal.constant(true),
      view: AVal.constant(Trafo3d.identity), proj: AVal.constant(Trafo3d.identity),
      model: () => (AVal.constant(Trafo3d.identity)), pixelSnapRadius: AVal.constant(1),
    });
    curId = idB;

    // Next move: capture must be dropped, event routes via spiral to B.
    pevent(canvas, "pointermove", 70, 50);
    await flush();
    expect(onA).toEqual(["downA", "moveA"]);  // A unchanged
    expect(onB).toEqual(["moveB"]);            // B got it
    detach();
  });
});

describe("PickDispatcher — multi-touch gestures", () => {
  it("two pointers spreading + rotating fires OnPinch / OnTwoFingerPan / OnTwoFingerRotate", async () => {
    const reg = new PickRegistry();
    const log: { kind: string; pinchScale?: number | undefined; rotate?: number | undefined; panX?: number | undefined; panY?: number | undefined }[] = [];
    const id = acquire(reg, [bubbleOf({
      OnPinch: (e: SceneEvent) => log.push({ kind: "pinch", pinchScale: e.pinchScale }),
      OnTwoFingerPan: (e: SceneEvent) => log.push({ kind: "pan", panX: e.panDeltaX, panY: e.panDeltaY }),
      OnTwoFingerRotate: (e: SceneEvent) => log.push({ kind: "rotate", rotate: e.rotateRadians }),
    })]);

    const canvas = makeCanvas();
    const d = new PickDispatcher(reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect());
    // The pixel-pick should resolve to `id` regardless of pointer position
    // — easier than adjusting the canned region per-event.
    const detach = d.attach(canvas, async (cx, cy) => makeRegion(cx, cy, id));

    // Two fingers down — both report the same pickId.
    pevent(canvas, "pointerdown", 80, 50, 1);
    await flush();
    pevent(canvas, "pointerdown", 120, 50, 2);
    await flush();

    // Spread + rotate: finger 2 moves diagonally outward and up.
    pevent(canvas, "pointermove", 80, 50, 1);
    await flush();
    pevent(canvas, "pointermove", 140, 30, 2);
    await flush();

    const kinds = log.map(e => e.kind);
    expect(kinds).toContain("pinch");
    expect(kinds).toContain("pan");
    expect(kinds).toContain("rotate");
    const pinch = log.find(e => e.kind === "pinch");
    expect(pinch!.pinchScale).toBeGreaterThan(1);
    detach();
  });
});

describe("PickDispatcher — hover dwell", () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); });

  it("Cursor still over the same scope for HOVER_DELAY_MS fires OnHover once", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const id = acquire(reg, [bubbleOf({ OnHover: () => log.push("hover") })]);

    const canvas = makeCanvas();
    const d = new PickDispatcher(reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect());
    const detach = d.attach(canvas, async () => makeRegion(50, 50, id));

    pevent(canvas, "pointermove", 50, 50);
    await flush();
    vi.advanceTimersByTime(HOVER_DELAY_MS + 50);
    expect(log).toEqual(["hover"]);
    detach();
  });

  it("Movement that resolves to a different pickId cancels the pending hover timer", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const idA = acquire(reg, [bubbleOf({ OnHover: () => log.push("hoverA") })]);
    const idB = acquire(reg, [bubbleOf({ OnHover: () => log.push("hoverB") })]);

    const canvas = makeCanvas();
    const d = new PickDispatcher(reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect());
    let cur = idA;
    const detach = d.attach(canvas, async () => makeRegion(50, 50, cur));

    pevent(canvas, "pointermove", 50, 50);
    await flush();
    // Half the dwell elapses, then we move to B — A's timer must be cancelled.
    vi.advanceTimersByTime(HOVER_DELAY_MS / 2);
    cur = idB;
    pevent(canvas, "pointermove", 51, 50);
    await flush();
    // Advance past the original schedule; B should fire eventually.
    vi.advanceTimersByTime(HOVER_DELAY_MS + 50);
    expect(log).toEqual(["hoverB"]);
    detach();
  });

  it("override hoverDelayMs via constructor", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const id = acquire(reg, [bubbleOf({ OnHover: () => log.push("hover") })]);
    const canvas = makeCanvas();
    const d = new PickDispatcher(
      reg, () => Trafo3d.identity, () => Trafo3d.identity,
      () => canvas.getBoundingClientRect(),
      { hoverDelayMs: 100 },
    );
    const detach = d.attach(canvas, async () => makeRegion(50, 50, id));
    pevent(canvas, "pointermove", 50, 50);
    await flush();
    vi.advanceTimersByTime(50);
    expect(log).toEqual([]);
    vi.advanceTimersByTime(100);
    expect(log).toEqual(["hover"]);
    detach();
  });
});
