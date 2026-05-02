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
import type { PickRegion } from "../src/scene/picking/readback.js";
import type { SceneEvent } from "../src/scene/picking/sceneEvent.js";
import { SNAP_RADIUS_MAX, SNAP_REGION_SIZE } from "../src/scene/picking/snapOffsets.js";
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

/**
 * Build a region centred on `(centerX, centerY)` of full
 * `SNAP_REGION_SIZE` × `SNAP_REGION_SIZE`, with all-zero slots, then
 * stamp pickIds at given (dx, dy) offsets relative to the centre.
 */
function makeRegion(centerX: number, centerY: number, stamps: ReadonlyArray<{ dx: number; dy: number; pickId: number }>): PickRegion {
  const sizeX = SNAP_REGION_SIZE;
  const sizeY = SNAP_REGION_SIZE;
  const originX = centerX - SNAP_RADIUS_MAX;
  const originY = centerY - SNAP_RADIUS_MAX;
  const data = new Float32Array(sizeX * sizeY * 4);
  for (const s of stamps) {
    for (let ddy = -1; ddy <= 1; ddy++) {
      for (let ddx = -1; ddx <= 1; ddx++) {
        const lx = (centerX + s.dx + ddx) - originX;
        const ly = (centerY + s.dy + ddy) - originY;
        if (lx < 0 || ly < 0 || lx >= sizeX || ly >= sizeY) continue;
        const i = (ly * sizeX + lx) * 4;
        data[i] = s.pickId;       // slot0 = +id (mode-A)
        data[i + 1] = 0;
        data[i + 2] = 0;           // ndcZ
        data[i + 3] = 0;
      }
    }
  }
  return { data, originX, originY, sizeX, sizeY };
}

function regionOf(region: PickRegion): (x: number, y: number) => Promise<PickRegion> {
  return async () => region;
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
    handlers: handlers.map(h => ({ handlers: bubbleOf(h), local2World: AVal.constant(Trafo3d.identity) })),
    cursor: undefined,
    pickThrough: opts.pickThrough ?? false,
    active: AVal.constant(opts.active ?? true),
    view: AVal.constant(Trafo3d.identity),
    proj: AVal.constant(Trafo3d.identity),
    model: AVal.constant(Trafo3d.identity),
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
    const detach = d.attach(canvas, regionOf(makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }])));

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
    const detach = d.attach(canvas, regionOf(makeRegion(50, 50, [])));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(calls.length).toBe(0);
    detach();
  });

  it("center empty, neighbour within snap radius wins", async () => {
    const reg = new PickRegistry();
    const calls: SceneEvent[] = [];
    // Scope X with snap radius = 5 (r² = 25). Hit at (dx=2, dy=0) → d² = 4 ≤ 25.
    const id = acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], { pixelSnapRadius: 5 });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, regionOf(makeRegion(50, 50, [{ dx: 2, dy: 0, pickId: id }])));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(calls.length).toBe(1);
    expect(calls[0]!.pickId).toBe(id);
    detach();
  });

  it("neighbour outside its own snap radius does not dispatch", async () => {
    const reg = new PickRegistry();
    const calls: SceneEvent[] = [];
    // pixelSnapRadius = 1 → r² = 1. Stamp the 3×3 patch at (dx=3, dy=0):
    // the closest neighbour pixel within the patch is at (2, 0), d²=4 > 1.
    const id = acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], { pixelSnapRadius: 1 });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, regionOf(makeRegion(50, 50, [{ dx: 3, dy: 0, pickId: id }])));

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

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    // far at (3,0) d²=9, near at (1,0) d²=1; spiral visits (1,0) first.
    const region = makeRegion(50, 50, [
      { dx: 1, dy: 0, pickId: idNear },
      { dx: 3, dy: 0, pickId: idFar },
    ]);
    const detach = d.attach(canvas, regionOf(region));

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

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const region = makeRegion(50, 50, [
      { dx: 0, dy: 0, pickId: idThrough }, // pickThrough at the centre
      { dx: 2, dy: 0, pickId: idBehind  },
    ]);
    const detach = d.attach(canvas, regionOf(region));

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

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, regionOf(makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }])));

    pevent(canvas, "pointerdown", 10, 10);
    await flush();

    expect(calls.length).toBe(0);
    detach();
  });

  it("default pixelSnapRadius is 1 in TraversalState", () => {
    expect(AVal.force(TraversalState.empty.pixelSnapRadius)).toBe(1);
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
    const reader = async (cx: number, cy: number): Promise<PickRegion> =>
      makeRegion(cx, cy, [{ dx: 0, dy: 0, pickId: nextId }]);
    const detach = d.attach(canvas, reader);

    pevent(canvas, "pointermove", 10, 10);
    await flush();
    nextId = idB;
    pevent(canvas, "pointermove", 50, 50);
    await flush();

    expect(log).toEqual(["A.enter", "A.move", "A.leave", "B.enter", "B.move"]);
    detach();
  });
});
