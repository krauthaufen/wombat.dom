// Capture/bubble + PointerCapture + differential enter/leave
// behaviour for the PickDispatcher. Mirrors Aardvark.Dom's
// `TraversalState.handleEvent` / `handleDifferential` and the
// pointer-capture map in `SceneHandler`.

import { describe, expect, it, vi } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";

import { PickDispatcher } from "../src/scene/picking/dispatcher.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import type { PickRegion } from "../src/scene/picking/readback.js";
import type { SceneEvent } from "../src/scene/picking/sceneEvent.js";
import { SNAP_RADIUS_MAX, SNAP_REGION_SIZE } from "../src/scene/picking/snapOffsets.js";
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

function makeRegion(centerX: number, centerY: number, stamps: ReadonlyArray<{ dx: number; dy: number; pickId: number }>): PickRegion {
  const sizeX = SNAP_REGION_SIZE;
  const sizeY = SNAP_REGION_SIZE;
  const originX = centerX - SNAP_RADIUS_MAX;
  const originY = centerY - SNAP_RADIUS_MAX;
  const data = new Float32Array(sizeX * sizeY * 4);
  for (const s of stamps) {
    // Expand each stamp to a 3×3 block so the spiral validator's
    // ≥ 3 same-id neighbour check passes. Mirrors how a real
    // multi-sample pick buffer would write a contiguous patch.
    for (let ddy = -1; ddy <= 1; ddy++) {
      for (let ddx = -1; ddx <= 1; ddx++) {
        const lx = (centerX + s.dx + ddx) - originX;
        const ly = (centerY + s.dy + ddy) - originY;
        if (lx < 0 || ly < 0 || lx >= sizeX || ly >= sizeY) continue;
        const i = (ly * sizeX + lx) * 4;
        data[i] = s.pickId;
      }
    }
  }
  return { data, originX, originY, sizeX, sizeY };
}

function pevent(canvas: HTMLCanvasElement, type: string, x: number, y: number, pointerId = 1): PointerEvent {
  const Ctor = (globalThis as { PointerEvent?: typeof PointerEvent }).PointerEvent
            ?? (globalThis as { MouseEvent: typeof MouseEvent }).MouseEvent;
  const ev = new Ctor(type, { clientX: x, clientY: y, bubbles: true, cancelable: true, button: 0 }) as PointerEvent;
  // jsdom MouseEvent fallback may lack pointerId; force it.
  Object.defineProperty(ev, "pointerId", { value: pointerId, configurable: true });
  Object.defineProperty(ev, "pointerType", { value: "mouse", configurable: true });
  canvas.dispatchEvent(ev);
  return ev;
}

async function flush(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

function acquireWith(
  reg: PickRegistry,
  handlers: ReadonlyArray<EventHandlers>,
  localTrafos?: ReadonlyArray<Trafo3d>,
): number {
  return reg.acquire({
    handlers: handlers.map((h, i) => ({
      handlers: h,
      local2World: AVal.constant(localTrafos?.[i] ?? Trafo3d.identity),
    })),
    cursor: undefined,
    pickThrough: false,
    active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity),
    proj: AVal.constant(Trafo3d.identity),
    model: AVal.constant(Trafo3d.identity),
    pixelSnapRadius: AVal.constant(1),
  });
}

describe("PickDispatcher — capture/bubble", () => {
  it("capture fires outer-first, bubble fires inner-first", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const A: EventHandlers = {
      capture: { OnClick: () => { log.push("A.cap"); } },
      bubble:  { OnClick: () => { log.push("A.bub"); } },
    };
    const B: EventHandlers = {
      capture: { OnClick: () => { log.push("B.cap"); } },
      bubble:  { OnClick: () => { log.push("B.bub"); } },
    };
    const C: EventHandlers = {
      capture: { OnClick: () => { log.push("C.cap"); } },
      bubble:  { OnClick: () => { log.push("C.bub"); } },
    };
    const id = acquireWith(reg, [A, B, C]);

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, async () => makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }]));

    pevent(canvas, "click", 50, 50);
    await flush();

    expect(log).toEqual(["A.cap", "B.cap", "C.cap", "C.bub", "B.bub", "A.bub"]);
    detach();
  });

  it("stopPropagation in capture stops the bubble phase", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const A: EventHandlers = { bubble: { OnClick: () => { log.push("A.bub"); } } };
    const B: EventHandlers = {
      capture: { OnClick: (e) => { log.push("B.cap"); e.stopPropagation(); } },
      bubble:  { OnClick: () => { log.push("B.bub"); } },
    };
    const id = acquireWith(reg, [A, B]);

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, async () => makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }]));

    pevent(canvas, "click", 50, 50);
    await flush();

    expect(log).toEqual(["B.cap"]);
    detach();
  });

  it("returning false from capture stops bubble phase", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const A: EventHandlers = {
      capture: { OnClick: () => { log.push("A.cap"); return false; } },
      bubble:  { OnClick: () => { log.push("A.bub"); } },
    };
    const B: EventHandlers = {
      capture: { OnClick: () => { log.push("B.cap"); } },
      bubble:  { OnClick: () => { log.push("B.bub"); } },
    };
    const id = acquireWith(reg, [A, B]);

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, async () => makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }]));

    pevent(canvas, "click", 50, 50);
    await flush();

    expect(log).toEqual(["A.cap"]);
    detach();
  });

  it("a thrown handler does not crash dispatch — others still fire", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const A: EventHandlers = { bubble: { OnClick: () => { throw new Error("boom"); } } };
    const B: EventHandlers = { bubble: { OnClick: () => { log.push("B.bub"); } } };
    const id = acquireWith(reg, [A, B]);

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, async () => makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }]));

    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    pevent(canvas, "click", 50, 50);
    await flush();
    errSpy.mockRestore();

    // Bubble runs inner-first: B fires first, A throws but is logged.
    expect(log).toEqual(["B.bub"]);
    detach();
  });
});

describe("PickDispatcher — differential enter/leave", () => {
  it("scope-chain [A,B,C] → [A,B,D]: leave on C, enter on D, A and B unchanged", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const A: EventHandlers = {
      bubble: {
        OnPointerEnter: () => { log.push("A.enter"); },
        OnPointerLeave: () => { log.push("A.leave"); },
      },
    };
    const B: EventHandlers = {
      bubble: {
        OnPointerEnter: () => { log.push("B.enter"); },
        OnPointerLeave: () => { log.push("B.leave"); },
      },
    };
    const C: EventHandlers = {
      bubble: {
        OnPointerEnter: () => { log.push("C.enter"); },
        OnPointerLeave: () => { log.push("C.leave"); },
      },
    };
    const D: EventHandlers = {
      bubble: {
        OnPointerEnter: () => { log.push("D.enter"); },
        OnPointerLeave: () => { log.push("D.leave"); },
      },
    };
    const idABC = acquireWith(reg, [A, B, C]);
    const idABD = acquireWith(reg, [A, B, D]);

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);

    let next = idABC;
    const detach = d.attach(canvas, async (cx, cy) => makeRegion(cx, cy, [{ dx: 0, dy: 0, pickId: next }]));

    pevent(canvas, "pointermove", 10, 10);
    await flush();
    log.length = 0;

    next = idABD;
    pevent(canvas, "pointermove", 50, 50);
    await flush();

    // Leave fires on the OLD-only scopes (C); enter on NEW-only (D);
    // shared prefix [A,B] is skipped.
    expect(log).toEqual(["C.leave", "D.enter"]);
    detach();
  });

  it("pointerenter fires capture outer-first then bubble inner-first across the chain", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const A: EventHandlers = {
      capture: { OnPointerEnter: () => { log.push("A.cap"); } },
      bubble:  { OnPointerEnter: () => { log.push("A.bub"); } },
    };
    const B: EventHandlers = {
      capture: { OnPointerEnter: () => { log.push("B.cap"); } },
      bubble:  { OnPointerEnter: () => { log.push("B.bub"); } },
    };
    const id = acquireWith(reg, [A, B]);

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, async (cx, cy) => makeRegion(cx, cy, [{ dx: 0, dy: 0, pickId: id }]));

    pevent(canvas, "pointermove", 10, 10);
    await flush();

    // The synthetic enter walks `runDownAll`: per-scope cap+bub
    // outer→inner. So expect A.cap, A.bub, B.cap, B.bub.
    expect(log).toEqual(["A.cap", "A.bub", "B.cap", "B.bub"]);
    detach();
  });
});

describe("PickDispatcher — PointerCapture", () => {
  it("setPointerCapture routes subsequent moves to captured scope; release fires synthetic move", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    let captureSelf: ((e: SceneEvent) => void) | undefined;
    const A: EventHandlers = {
      bubble: {
        OnPointerDown: (e) => { log.push("A.down"); captureSelf?.(e); },
        OnPointerMove: () => { log.push("A.move"); },
        OnPointerEnter: () => { log.push("A.enter"); },
        OnPointerLeave: () => { log.push("A.leave"); },
      },
    };
    const B: EventHandlers = {
      bubble: {
        OnPointerMove: () => { log.push("B.move"); },
        OnPointerEnter: () => { log.push("B.enter"); },
        OnPointerLeave: () => { log.push("B.leave"); },
      },
    };
    const idA = acquireWith(reg, [A]);
    const idB = acquireWith(reg, [B]);

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);

    captureSelf = (e): void => { e.setPointerCapture(); };

    let next = idA;
    const detach = d.attach(canvas, async (cx, cy) => makeRegion(cx, cy, [{ dx: 0, dy: 0, pickId: next }]));

    // 1) Hover over A → A.enter
    pevent(canvas, "pointermove", 10, 10);
    await flush();
    // 2) Down on A → A.down (which captures the pointer)
    pevent(canvas, "pointerdown", 10, 10);
    await flush();
    log.length = 0;

    // 3) Move over B's region; spiral would resolve B but capture
    //    routes the move to A. lastHit (which is A) must NOT update.
    next = idB;
    pevent(canvas, "pointermove", 60, 60);
    await flush();

    expect(log).toEqual(["A.move"]);
    log.length = 0;

    // 4) Release capture from A; synthetic move replays at the
    //    last cursor position (60,60), which now hits B → A.leave
    //    then B.enter, B.move. Release is invoked via the
    //    dispatcher API (handlers do this via ev.releasePointerCapture()).
    const scopeA = reg.lookup(idA)!;
    d.releasePointerCapture(scopeA, 1);
    await flush();

    expect(log).toEqual(["A.leave", "B.enter", "B.move"]);
    detach();
  });

  it("hasPointerCapture reports whether this scope holds the pointer", async () => {
    const reg = new PickRegistry();
    const flags: boolean[] = [];
    const A: EventHandlers = {
      bubble: {
        OnPointerDown: (e) => {
          flags.push(e.hasPointerCapture()); // false before
          e.setPointerCapture();
          flags.push(e.hasPointerCapture()); // true after
        },
      },
    };
    const id = acquireWith(reg, [A]);

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, async (cx, cy) => makeRegion(cx, cy, [{ dx: 0, dy: 0, pickId: id }]));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(flags).toEqual([false, true]);
    detach();
  });
});

