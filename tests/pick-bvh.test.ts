// BVH ray fall-through tests for the PickDispatcher. Mocks the
// region reader the same way pick-dispatch.test.ts does, registers a
// per-scope intersectable on each pickId, and asserts the BVH path's
// behaviour for pickThrough / no-pixel-hit cases.

import { describe, expect, it } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Box3d, Intersectable, Trafo3d, V3d } from "@aardworx/wombat.base";

import { PickDispatcher } from "../src/scene/picking/dispatcher.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import type { PickRegion } from "../src/scene/picking/readback.js";
import type { SceneEvent } from "../src/scene/picking/sceneEvent.js";
import { SNAP_RADIUS_MAX, SNAP_REGION_SIZE } from "../src/scene/picking/snapOffsets.js";
import type { EventHandlers, SceneEventHandler } from "../src/scene/sg.js";
import type { SceneEventKind } from "../src/scene/picking/sceneEvent.js";

// Canvas 200×100 — same shape as pick-dispatch.test.ts.
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

// Identity view + identity proj => NDC == world. Cursor at (50, 50)
// on a 200×100 canvas gives NDC (-0.495, +0.01, *) once the half-
// pixel centre offset is applied, so the world ray runs along
// (x ≈ -0.495, y ≈ 0.01, dir +z).
function makeDispatcher(reg: PickRegistry, canvas: HTMLCanvasElement): PickDispatcher {
  return new PickDispatcher(
    reg,
    () => Trafo3d.identity,
    () => Trafo3d.identity,
    () => canvas.getBoundingClientRect(),
  );
}

function makeRegion(
  centerX: number,
  centerY: number,
  stamps: ReadonlyArray<{ dx: number; dy: number; pickId: number }>,
  /** When true, expand each stamp to a 3×3 patch so the spiral validator's neighbour count passes. */
  expand = false,
): PickRegion {
  const sizeX = SNAP_REGION_SIZE;
  const sizeY = SNAP_REGION_SIZE;
  const originX = centerX - SNAP_RADIUS_MAX;
  const originY = centerY - SNAP_RADIUS_MAX;
  const data = new Float32Array(sizeX * sizeY * 4);
  const r = expand ? 1 : 0;
  for (const s of stamps) {
    for (let ddy = -r; ddy <= r; ddy++) {
      for (let ddx = -r; ddx <= r; ddx++) {
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

function bubbleOf(rec: Record<string, (e: SceneEvent) => unknown>): EventHandlers {
  const bubble: Partial<Record<SceneEventKind, SceneEventHandler>> = {};
  for (const [k, v] of Object.entries(rec)) bubble[k as SceneEventKind] = v as SceneEventHandler;
  return { bubble };
}

interface AcquireOpts {
  pickThrough?: boolean;
  intersectable?: ReturnType<typeof Intersectable.box>;
}

function acquire(reg: PickRegistry, handlers: ReadonlyArray<Record<string, (e: SceneEvent) => unknown>>, opts: AcquireOpts = {}): number {
  return reg.acquire({
    handlers: handlers.map(bubbleOf),
    cursor: undefined,
    pickThrough: opts.pickThrough ?? false,
    active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity),
    proj: AVal.constant(Trafo3d.identity),
    model: AVal.constant(Trafo3d.identity),
    pixelSnapRadius: AVal.constant(1),
    ...(opts.intersectable !== undefined ? { intersectable: AVal.constant(opts.intersectable) } : {}),
  });
}

// Box A: near box covering the ray (x=-0.5, y=0). Hit at t≈0.7.
// Box B: far box at z>0. Hit at t≈1.3.
const boxA = Intersectable.box(Box3d.fromMinMax(new V3d(-1, -1, -0.5), new V3d(1, 1, -0.3)));
const boxB = Intersectable.box(Box3d.fromMinMax(new V3d(-1, -1,  0.3), new V3d(1, 1,  0.5)));

describe("PickDispatcher BVH fall-through", () => {
  it("BVH returns the next non-pickThrough scope when spiral hits a pickThrough", async () => {
    const reg = new PickRegistry();
    const calls: SceneEvent[] = [];
    const idA = acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], {
      pickThrough: true, intersectable: boxA,
    });
    const idB = acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], {
      intersectable: boxB,
    });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    // Region only stamps the pickThrough scope at the centre; spiral
    // skips it (no non-pickThrough pixel hit) → BVH fall-through.
    const detach = d.attach(canvas, regionOf(makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: idA }])));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(calls.length).toBe(1);
    expect(calls[0]!.pickId).toBe(idB);
    // viewPos: world hit on box B's near face (z = 0.3) at the
    // unprojected cursor pixel.
    expect(calls[0]!.viewPos).toBeDefined();
    expect(calls[0]!.viewPos!.x).toBeCloseTo(-0.495, 5);
    expect(calls[0]!.viewPos!.y).toBeCloseTo(-0.01, 5);
    expect(calls[0]!.viewPos!.z).toBeCloseTo(0.3, 5);
    // Outward normal on box B's near face is (0, 0, -1).
    expect(calls[0]!.viewNormal).toBeDefined();
    expect(calls[0]!.viewNormal!.z).toBeCloseTo(-1, 5);
    detach();
  });

  it("BVH winner that is pickThrough with no fallthrough scope keeps the pickThrough scope", async () => {
    // F#: pickThrough BVH winner re-traces excluding pickThrough; on
    // empty re-trace it keeps the original (Some(scope, ..., None)).
    const reg = new PickRegistry();
    const calls: SceneEvent[] = [];
    const idA = acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], {
      pickThrough: true, intersectable: boxA,
    });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    // Single-pixel stamp: 3-neighbour check rejects the pixel
    // candidate, so BVH on idA wins. idA is pickThrough → re-trace
    // with no other scopes finds nothing → keep idA.
    const detach = d.attach(canvas, regionOf(makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: idA }])));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(calls.length).toBe(1);
    expect(calls[0]!.pickId).toBe(idA);
    detach();
  });

  it("spiral non-pickThrough hit wins over BVH (BVH not consulted)", async () => {
    const reg = new PickRegistry();
    const calls: SceneEvent[] = [];
    // A is non-pickThrough at the cursor pixel. B has a near box that
    // would otherwise win the ray, but the spiral hit on A short-
    // circuits before BVH is touched.
    const idA = acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], {
      intersectable: boxB, // doesn't matter — non-pickThrough wins via spiral
    });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, regionOf(makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: idA }], true)));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(calls.length).toBe(1);
    expect(calls[0]!.pickId).toBe(idA);
    // Pixel-pick path leaves viewNormal at its synthetic zero default
    // — there is no per-fragment normal in Mode-A pick output.
    expect(calls[0]!.viewNormal.x).toBe(0);
    expect(calls[0]!.viewNormal.y).toBe(0);
    expect(calls[0]!.viewNormal.z).toBe(0);
    detach();
  });

  it("BVH fall-through fires when there is no pixel hit at all (region all zeros)", async () => {
    const reg = new PickRegistry();
    const calls: SceneEvent[] = [];
    acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], {
      pickThrough: true, intersectable: boxA,
    });
    const idB = acquire(reg, [{ OnPointerDown: (e) => calls.push(e) }], {
      intersectable: boxB,
    });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, regionOf(makeRegion(50, 50, [])));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(calls.length).toBe(1);
    expect(calls[0]!.pickId).toBe(idB);
    detach();
  });
});
