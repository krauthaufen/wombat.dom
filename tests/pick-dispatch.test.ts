// PickDispatcher behaviour tests — no GPU. We mock the region
// reader with a canned `PickRegion` and assert which scope wins
// the spiral hit-test under various pixelSnapRadius / pickThrough
// / active configurations.
//
// Why no real GPU: the dispatcher's contract is "given a region of
// decoded slots, do the right thing"; the actual GPU readback is
// covered separately by browser-mode tests.

import { describe, expect, it, vi } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";

import { PickDispatcher } from "../src/scene/picking/dispatcher.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import type { SceneEvent } from "../src/scene/picking/sceneEvent.js";
import { noPixel, pixelWinner, resolverOf } from "./pickArgminTestUtil.js";
import { TraversalState } from "../src/scene/traversalState.js";
import type { EventHandlers, SceneEventHandler } from "../src/scene/sg.js";
import type { SceneEventKind } from "../src/scene/picking/sceneEvent.js";

// Canvas size: 200×100. Place the cursor at (50, 50) — that lands
// at device pixel (50, 50) since CSS px = device px in our stub.
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

function pevent(canvas: HTMLCanvasElement, type: string, x: number, y: number): PointerEvent {
  const Ctor = (globalThis as { PointerEvent?: typeof PointerEvent }).PointerEvent
            ?? (globalThis as { MouseEvent: typeof MouseEvent }).MouseEvent;
  const ev = new Ctor(type, { clientX: x, clientY: y, bubbles: true, cancelable: true, button: 0 });
  canvas.dispatchEvent(ev);
  return ev as PointerEvent;
}

async function flush(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

interface AcquireOpts {
  pickThrough?: boolean;
  active?: boolean;
  pixelSnapRadius?: number;
}

function bubbleOf(rec: Record<string, (e: SceneEvent) => unknown>): EventHandlers {
  const bubble: Partial<Record<SceneEventKind, SceneEventHandler>> = {};
  for (const [k, v] of Object.entries(rec)) bubble[k as SceneEventKind] = v as SceneEventHandler;
  return { bubble };
}

function acquire(reg: PickRegistry, handlers: ReadonlyArray<Record<string, (e: SceneEvent) => unknown>>, opts: AcquireOpts = {}): number {
  return reg.acquire({
    handlers: handlers.map(h => ({ handlers: bubbleOf(h), local2World: () => AVal.constant(Trafo3d.identity) })),
    cursor: undefined,
    pickThrough: opts.pickThrough ?? false,
    active: AVal.constant(opts.active ?? true),
    view: AVal.constant(Trafo3d.identity),
    proj: AVal.constant(Trafo3d.identity),
    model: () => (AVal.constant(Trafo3d.identity)),
    pixelSnapRadius: AVal.constant(opts.pixelSnapRadius ?? 1),
  });
}

describe("PickDispatcher", () => {
  it("center-pixel hit dispatches that scope", async () => {
    const reg = new PickRegistry();
    const calls: SceneEvent[] = [];
    const id = acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], { pixelSnapRadius: 1 });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, resolverOf(pixelWinner(id)));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(calls.length).toBe(1);
    expect(calls[0]!.pickId).toBe(id);
    detach();
  });

  it("does nothing when no pickIds in region (all zeros)", async () => {
    const reg = new PickRegistry();
    const calls: SceneEvent[] = [];
    acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }]);

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, resolverOf(noPixel()));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(calls.length).toBe(0);
    detach();
  });

  it("center empty, neighbour within snap radius wins", async () => {
    const reg = new PickRegistry();
    const calls: SceneEvent[] = [];
    // Scope X with snap radius = 5 (r² = 25). Kernel's nearest valid
    // pixel is at (52, 50) → d² = 4 ≤ 25, off-centre.
    const id = acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], { pixelSnapRadius: 5 });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, resolverOf(pixelWinner(id, { px: 52, dist2: 4 })));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(calls.length).toBe(1);
    expect(calls[0]!.pickId).toBe(id);
    detach();
  });

  it("neighbour outside its own snap radius does not dispatch", async () => {
    const reg = new PickRegistry();
    const calls: SceneEvent[] = [];
    // pixelSnapRadius = 1 → r² = 1. The nearest stamped pixel is at
    // d²=4 > 1, so the kernel rejects it (no valid winner).
    const id = acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], { pixelSnapRadius: 1 });
    void id;

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, resolverOf(noPixel()));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(calls.length).toBe(0);
    detach();
  });

  it("two scopes at different distances — closer wins", async () => {
    const reg = new PickRegistry();
    const calls: SceneEvent[] = [];
    const idNear = acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], { pixelSnapRadius: 5 });
    const idFar  = acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], { pixelSnapRadius: 5 });
    void idFar;

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    // far at d²=9, near at d²=1; the kernel's argmin returns the
    // nearest valid pixel, idNear at d²=1.
    const detach = d.attach(canvas, resolverOf(pixelWinner(idNear, { px: 51, dist2: 1 })));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(calls.length).toBe(1);
    expect(calls[0]!.pickId).toBe(idNear);
    detach();
  });

  it("pixel-picked pickThrough scope is kept (warn, no fall-through)", async () => {
    // F# `SceneHandler.fs:1814` warns and keeps the winner when a
    // pickThrough scope was selected by the pixel path — only BVH
    // winners fall through to the next non-pickThrough scope.
    const reg = new PickRegistry();
    const calls: SceneEvent[] = [];
    const idThrough = acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], {
      pickThrough: true, pixelSnapRadius: 5,
    });
    const idBehind = acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], { pixelSnapRadius: 5 });
    void idBehind;

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    // The pickThrough scope is the centre pixel winner. A pixel-picked
    // pickThrough can't re-trace (no depth behind a pixel) → warn+keep.
    const detach = d.attach(canvas, resolverOf(pixelWinner(idThrough)));

    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(calls.length).toBe(1);
    expect(calls[0]!.pickId).toBe(idThrough);
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
    detach();
  });

  it("inactive scope does not dispatch", async () => {
    const reg = new PickRegistry();
    const calls: SceneEvent[] = [];
    const id = acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], { active: false });
    void id;

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    // Inactive scope → the kernel's per-id metadata gates it out
    // (effectiveRadius < 0) → no valid winner.
    const detach = d.attach(canvas, resolverOf(noPixel()));

    pevent(canvas, "pointerdown", 10, 10);
    await flush();

    expect(calls.length).toBe(0);
    detach();
  });

  it("default pixelSnapRadius is 0 (pixel-exact) in TraversalState", () => {
    expect(AVal.force(TraversalState.empty.pixelSnapRadius)).toBe(0);
  });

  it("pointermove changing pickId fires leave on old + enter on new", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const idA = acquire(reg, [{
      OnPointerEnter: () => log.push("A.enter"),
      OnPointerLeave: () => log.push("A.leave"),
      OnPointerMove:  () => log.push("A.move"),
    }]);
    const idB = acquire(reg, [{
      OnPointerEnter: () => log.push("B.enter"),
      OnPointerLeave: () => log.push("B.leave"),
      OnPointerMove:  () => log.push("B.move"),
    }]);

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);

    let nextId = idA;
    const detach = d.attach(canvas, async (cx, cy) => pixelWinner(nextId, { px: cx, py: cy }));

    pevent(canvas, "pointermove", 10, 10);
    await flush();
    nextId = idB;
    pevent(canvas, "pointermove", 50, 50);
    await flush();

    expect(log).toEqual(["A.enter", "A.move", "A.leave", "B.enter", "B.move"]);
    detach();
  });

  it("enter/leave also fires on pointerdown/up target changes (differential on every pointer event)", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const idA = acquire(reg, [{
      OnPointerEnter: () => log.push("A.enter"),
      OnPointerLeave: () => log.push("A.leave"),
    }]);
    const idB = acquire(reg, [{
      OnPointerEnter: () => log.push("B.enter"),
      OnPointerLeave: () => log.push("B.leave"),
    }]);

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);

    let nextId = idA;
    const detach = d.attach(canvas, async (cx, cy) => pixelWinner(nextId, { px: cx, py: cy }));

    // A pointerdown (no prior move) already establishes hover on A —
    // Aardvark runs handleMove for EVERY pointer event, not just moves.
    pevent(canvas, "pointerdown", 10, 10);
    await flush();
    expect(log).toEqual(["A.enter"]);

    // A pointerup whose pick resolves to B transfers hover A→B.
    nextId = idB;
    pevent(canvas, "pointerup", 50, 50);
    await flush();
    expect(log).toEqual(["A.enter", "A.leave", "B.enter"]);
    detach();
  });

  it("dblclick dispatches OnDoubleClick; events carry clientRect/context/target/self", async () => {
    const reg = new PickRegistry();
    let seen: SceneEvent | undefined;
    const id = acquire(reg, [{
      OnDoubleClick: (e: SceneEvent) => { seen = e; },
    }]);

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, async (cx, cy) => pixelWinner(id, { px: cx, py: cy }));

    const Ctor = (globalThis as { MouseEvent: typeof MouseEvent }).MouseEvent;
    canvas.dispatchEvent(new Ctor("dblclick", { clientX: 30, clientY: 30, bubbles: true, button: 0 }));
    await flush();

    expect(seen).toBeDefined();
    expect(seen!.kind).toBe("OnDoubleClick");
    // Aardvark-parity surface: ClientRect / Context / Target / This.
    expect(seen!.clientRect).toBeDefined();
    expect(seen!.context).toBeDefined();
    expect(seen!.context!.size.x).toBe(200);
    expect(seen!.target).toBeDefined();
    expect(seen!.self).toBeDefined();
    detach();
  });
});
