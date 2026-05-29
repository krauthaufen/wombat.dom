// Synthesised tap / double-tap / long-press dispatch tests.
//
// These cover the timing-based synthesis layer in `dispatcher.ts`.
// Real GPU readback is mocked with a canned `PickRegion`. Vitest fake
// timers drive `Date.now()` and `setTimeout` so we can advance time
// deterministically.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";

import {
  PickDispatcher,
  TAP_MAX_DURATION_MS,
  TAP_MAX_MOVE_PX,
  DOUBLE_TAP_GAP_MS,
  LONG_PRESS_MS,
} from "../src/scene/picking/dispatcher.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import type { PickArgminResult } from "../src/scene/picking/pickArgminCompute.js";
import { noPixel, pixelWinner } from "./pickArgminTestUtil.js";
import type { SceneEvent, SceneEventKind } from "../src/scene/picking/sceneEvent.js";
import type { EventHandlers, SceneEventHandler } from "../src/scene/sg.js";

function makeCanvas(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = 200;
  c.height = 100;
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

function makeRegion(centerX: number, centerY: number, pickId: number): PickArgminResult {
  // Argmin verdict: pickId is the centre-pixel winner (0 ⇒ no hit).
  return pickId === 0 ? noPixel() : pixelWinner(pickId, { px: centerX, py: centerY });
}

function pevent(canvas: HTMLCanvasElement, type: string, x: number, y: number, pointerId = 1): PointerEvent {
  const Ctor = (globalThis as { PointerEvent?: typeof PointerEvent }).PointerEvent
            ?? (globalThis as { MouseEvent: typeof MouseEvent }).MouseEvent;
  const ev = new Ctor(type, { clientX: x, clientY: y, bubbles: true, cancelable: true, button: 0 });
  // Pointer id is read-only on PointerEvent in jsdom; force it.
  Object.defineProperty(ev, "pointerId", { value: pointerId, configurable: true });
  canvas.dispatchEvent(ev);
  return ev as PointerEvent;
}

// We have to await microtasks even with fake timers — readback resolves
// via Promise.resolve. Microtasks run independently of fake-timer time.
async function flush(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

function bubbleOf(rec: Record<string, (e: SceneEvent) => unknown>): EventHandlers {
  const bubble: Partial<Record<SceneEventKind, SceneEventHandler>> = {};
  for (const [k, v] of Object.entries(rec)) bubble[k as SceneEventKind] = v as SceneEventHandler;
  return { bubble };
}

interface AcquireOpts { pixelSnapRadius?: number; }

function acquire(reg: PickRegistry, handlers: Record<string, (e: SceneEvent) => unknown>, opts: AcquireOpts = {}): number {
  return reg.acquire({
    handlers: [{ handlers: bubbleOf(handlers), local2World: AVal.constant(Trafo3d.identity) }],
    cursor: undefined,
    pickThrough: false,
    active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity),
    proj: AVal.constant(Trafo3d.identity),
    model: AVal.constant(Trafo3d.identity),
    pixelSnapRadius: AVal.constant(opts.pixelSnapRadius ?? 1),
  });
}

/**
 * Drive a tap-cycle (down→up) and advance fake time between the two
 * pointer events by `holdMs`. The readback for each event is awaited
 * via `flush()` AFTER the time-advance, since the synthesis is driven
 * by Date.now() inside `dispatch()` which fires on the awaited
 * microtask.
 */
async function downUp(canvas: HTMLCanvasElement, x: number, y: number, holdMs: number, pointerId = 1): Promise<void> {
  pevent(canvas, "pointerdown", x, y, pointerId);
  await flush();
  vi.advanceTimersByTime(holdMs);
  pevent(canvas, "pointerup", x, y, pointerId);
  await flush();
}

describe("PickDispatcher tap synthesis", () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); });

  it("quick down+up within TAP_MAX_DURATION_MS, no move → fires OnTap", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const id = acquire(reg, {
      OnTap: () => log.push("tap"),
      OnLongPress: () => log.push("long"),
    });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, async () => makeRegion(50, 50, id));

    await downUp(canvas, 50, 50, TAP_MAX_DURATION_MS - 10);

    expect(log).toEqual(["tap"]);
    detach();
  });

  it("down for >LONG_PRESS_MS, no move → fires OnLongPress, then up does NOT fire OnTap", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const id = acquire(reg, {
      OnTap: () => log.push("tap"),
      OnLongPress: () => log.push("long"),
    });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, async () => makeRegion(50, 50, id));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();
    vi.advanceTimersByTime(LONG_PRESS_MS + 100);
    // Long-press timer should have fired.
    expect(log).toEqual(["long"]);
    pevent(canvas, "pointerup", 50, 50);
    await flush();
    // No additional tap.
    expect(log).toEqual(["long"]);
    detach();
  });

  it("move > TAP_MAX_MOVE_PX during press → no tap, no long press", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const id = acquire(reg, {
      OnTap: () => log.push("tap"),
      OnLongPress: () => log.push("long"),
    }, { pixelSnapRadius: 50 });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    let cx = 50, cy = 50;
    const detach = d.attach(canvas, async () => makeRegion(cx, cy, id));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();
    // Drag well beyond TAP_MAX_MOVE_PX.
    cx = 50 + TAP_MAX_MOVE_PX + 5; cy = 50;
    pevent(canvas, "pointermove", cx, cy);
    await flush();
    // Hold long enough that long-press WOULD fire if not cancelled.
    vi.advanceTimersByTime(LONG_PRESS_MS + 100);
    pevent(canvas, "pointerup", cx, cy);
    await flush();

    expect(log).toEqual([]);
    detach();
  });

  it("two taps within DOUBLE_TAP_GAP_MS on same pickId → 2× OnTap + OnDoubleTap", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const id = acquire(reg, {
      OnTap: () => log.push("tap"),
      OnDoubleTap: () => log.push("double"),
    });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, async () => makeRegion(50, 50, id));

    await downUp(canvas, 50, 50, 50);
    vi.advanceTimersByTime(DOUBLE_TAP_GAP_MS - 100);
    await downUp(canvas, 50, 50, 50);

    expect(log).toEqual(["tap", "tap", "double"]);
    detach();
  });

  it("two taps separated by more than DOUBLE_TAP_GAP_MS → 2× OnTap, no OnDoubleTap", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const id = acquire(reg, {
      OnTap: () => log.push("tap"),
      OnDoubleTap: () => log.push("double"),
    });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, async () => makeRegion(50, 50, id));

    await downUp(canvas, 50, 50, 50);
    vi.advanceTimersByTime(DOUBLE_TAP_GAP_MS + 200);
    await downUp(canvas, 50, 50, 50);

    expect(log).toEqual(["tap", "tap"]);
    detach();
  });

  it("two taps on different pickIds → 2× OnTap, no OnDoubleTap", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const idA = acquire(reg, {
      OnTap: () => log.push("tapA"),
      OnDoubleTap: () => log.push("doubleA"),
    });
    const idB = acquire(reg, {
      OnTap: () => log.push("tapB"),
      OnDoubleTap: () => log.push("doubleB"),
    });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    let curId = idA;
    const detach = d.attach(canvas, async (cx, cy) => makeRegion(cx, cy, curId));

    await downUp(canvas, 50, 50, 50);
    vi.advanceTimersByTime(50);
    curId = idB;
    await downUp(canvas, 60, 50, 50);

    expect(log).toEqual(["tapA", "tapB"]);
    detach();
  });
});
