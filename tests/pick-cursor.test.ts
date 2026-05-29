// Cursor wiring — `state.cursor` from a hit scope drives
// `canvas.style.cursor`. Mirrors the dispatcher's existing pixel-pick
// resolution path; we mock `readRegion` with a canned scope-stamped
// region.

import { describe, expect, it } from "vitest";
import { AVal, cval, transact } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";

import { PickDispatcher } from "../src/scene/picking/dispatcher.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import type { PickArgminResult } from "../src/scene/picking/pickArgminCompute.js";
import { noPixel, pixelWinner } from "./pickArgminTestUtil.js";

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

function pevent(canvas: HTMLCanvasElement, type: string, x: number, y: number): void {
  const Ctor = (globalThis as { PointerEvent?: typeof PointerEvent }).PointerEvent
            ?? (globalThis as { MouseEvent: typeof MouseEvent }).MouseEvent;
  const ev = new Ctor(type, { clientX: x, clientY: y, bubbles: true, cancelable: true, button: 0 });
  canvas.dispatchEvent(ev);
}

async function flush(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

interface AcquireOpts {
  cursor?: string | import("@aardworx/wombat.adaptive").aval<string>;
}

function acquire(reg: PickRegistry, opts: AcquireOpts = {}): number {
  return reg.acquire({
    handlers: [],
    cursor: opts.cursor,
    pickThrough: false,
    active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity),
    proj: AVal.constant(Trafo3d.identity),
    model: AVal.constant(Trafo3d.identity),
    pixelSnapRadius: AVal.constant(1),
  });
}

describe("PickDispatcher — cursor wiring", () => {
  it("plain string cursor on hit writes canvas.style.cursor", async () => {
    const reg = new PickRegistry();
    const id = acquire(reg, { cursor: "pointer" });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, async () => makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }]));

    pevent(canvas, "pointermove", 50, 50);
    await flush();

    expect(canvas.style.cursor).toBe("pointer");
    detach();
  });

  it("hit scope with cursor=undefined resets canvas.style.cursor to \"\"", async () => {
    const reg = new PickRegistry();
    const idWithCursor = acquire(reg, { cursor: "crosshair" });
    const idNoCursor   = acquire(reg, {});

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);

    let nextId = idWithCursor;
    const detach = d.attach(canvas, async (cx, cy) =>
      makeRegion(cx, cy, [{ dx: 0, dy: 0, pickId: nextId }]),
    );

    pevent(canvas, "pointermove", 30, 30);
    await flush();
    expect(canvas.style.cursor).toBe("crosshair");

    nextId = idNoCursor;
    pevent(canvas, "pointermove", 60, 60);
    await flush();
    expect(canvas.style.cursor).toBe("");

    detach();
  });

  it("aval<string> cursor — flipping the cval's value updates canvas.style.cursor on the next move", async () => {
    const reg = new PickRegistry();
    const cur = cval("grab");
    const id = acquire(reg, { cursor: cur });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, async () => makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }]));

    pevent(canvas, "pointermove", 50, 50);
    await flush();
    expect(canvas.style.cursor).toBe("grab");

    transact(() => { cur.value = "grabbing"; });
    pevent(canvas, "pointermove", 50, 50);
    await flush();
    expect(canvas.style.cursor).toBe("grabbing");

    detach();
  });

  it("detach resets canvas.style.cursor to \"\"", async () => {
    const reg = new PickRegistry();
    const id = acquire(reg, { cursor: "pointer" });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, async () => makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }]));

    pevent(canvas, "pointermove", 50, 50);
    await flush();
    expect(canvas.style.cursor).toBe("pointer");

    detach();
    expect(canvas.style.cursor).toBe("");
  });
});
