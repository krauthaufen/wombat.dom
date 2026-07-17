// Phase 5 — focus model + OnFocus/OnBlur. Tests live without the
// dispatcher (registry-only) for the focus state machinery, and with
// the dispatcher for click-driven auto-focus + event flow.

import { describe, expect, it } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";

import { PickDispatcher } from "../src/scene/picking/dispatcher.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import type { PickArgminResult } from "../src/scene/picking/pickArgminCompute.js";
import { noPixel, pixelWinner } from "./pickArgminTestUtil.js";
import type { SceneEvent, SceneEventKind } from "../src/scene/picking/sceneEvent.js";
import type { EventHandlers, SceneEventHandler } from "../src/scene/sg.js";

function bubbleOf(rec: Record<string, (e: SceneEvent) => unknown>): EventHandlers {
  const bubble: Partial<Record<SceneEventKind, SceneEventHandler>> = {};
  for (const [k, v] of Object.entries(rec)) bubble[k as SceneEventKind] = v as SceneEventHandler;
  return { bubble };
}

function acquire(reg: PickRegistry, opts: {
  handlers?: EventHandlers[];
  canFocus?: boolean;
} = {}): number {
  return reg.acquire({
    handlers: (opts.handlers ?? []).map(h => ({ handlers: h, local2World: () => AVal.constant(Trafo3d.identity) })),
    cursor: undefined, pickThrough: false,
    active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity), proj: AVal.constant(Trafo3d.identity),
    model: () => (AVal.constant(Trafo3d.identity)), pixelSnapRadius: AVal.constant(1),
    ...(opts.canFocus !== undefined ? { canFocus: AVal.constant(opts.canFocus) } : {}),
  });
}

describe("PickRegistry — focus model", () => {
  it("setFocus on canFocus=true scope updates focusedPickId", () => {
    const reg = new PickRegistry();
    const id = acquire(reg, { canFocus: true });
    reg.setFocus(id);
    expect(AVal.force(reg.focusedPickId)).toBe(id);
  });

  it("setFocus on canFocus=false scope is rejected silently", () => {
    const reg = new PickRegistry();
    const id = acquire(reg, { canFocus: false });
    reg.setFocus(id);
    expect(AVal.force(reg.focusedPickId)).toBeUndefined();
  });

  it("setFocus on a scope with no canFocus is rejected", () => {
    const reg = new PickRegistry();
    const id = acquire(reg);
    reg.setFocus(id);
    expect(AVal.force(reg.focusedPickId)).toBeUndefined();
  });

  it("clearFocus drops focus", () => {
    const reg = new PickRegistry();
    const id = acquire(reg, { canFocus: true });
    reg.setFocus(id);
    reg.clearFocus();
    expect(AVal.force(reg.focusedPickId)).toBeUndefined();
  });
});

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

async function flush(): Promise<void> { for (let i=0;i<5;i++) await Promise.resolve(); }

describe("PickDispatcher — auto-focus on click", () => {
  it("Click on canFocus=true scope sets focus + fires OnFocus", async () => {
    const reg = new PickRegistry();
    const focusEvents: SceneEvent[] = [];
    const id = acquire(reg, {
      canFocus: true,
      handlers: [bubbleOf({ OnFocus: (e) => focusEvents.push(e) })],
    });
    const canvas = makeCanvas();
    const d = new PickDispatcher(reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect());
    const detach = d.attach(canvas, async () => makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: id }]));

    canvas.dispatchEvent(new MouseEvent("click", { clientX: 50, clientY: 50, bubbles: true, cancelable: true, button: 0 }));
    await flush();

    expect(AVal.force(reg.focusedPickId)).toBe(id);
    expect(focusEvents).toHaveLength(1);
    detach();
  });

  it("Click on canFocus=false scope clears focus + fires OnBlur on previous", async () => {
    const reg = new PickRegistry();
    const blurEvents: SceneEvent[] = [];
    const focusable = acquire(reg, {
      canFocus: true,
      handlers: [bubbleOf({ OnBlur: (e) => blurEvents.push(e) })],
    });
    const nonFocus = acquire(reg, { canFocus: false });
    const canvas = makeCanvas();
    const d = new PickDispatcher(reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect());

    // Drive focus first via API then re-attach dispatcher to start
    // observing changes; then click on non-focus scope to clear.
    reg.setFocus(focusable);
    const detach = d.attach(canvas, async () => makeRegion(50, 50, [{ dx: 0, dy: 0, pickId: nonFocus }]));
    canvas.dispatchEvent(new MouseEvent("click", { clientX: 50, clientY: 50, bubbles: true, cancelable: true, button: 0 }));
    await flush();

    expect(AVal.force(reg.focusedPickId)).toBeUndefined();
    expect(blurEvents).toHaveLength(1);
    detach();
  });
});

describe("PickDispatcher — keyboard events", () => {
  it("Key events route to the focused scope only", async () => {
    const reg = new PickRegistry();
    const keys: string[] = [];
    const id = acquire(reg, {
      canFocus: true,
      handlers: [bubbleOf({ OnKeyDown: (e) => keys.push(e.key ?? "") })],
    });
    const canvas = makeCanvas();
    const d = new PickDispatcher(reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect());
    const detach = d.attach(canvas, async () => makeRegion(50, 50, []));

    // No focus → key dropped.
    canvas.dispatchEvent(new KeyboardEvent("keydown", { key: "a", code: "KeyA", bubbles: true }));
    await flush();
    expect(keys).toHaveLength(0);

    // Set focus → key dispatched.
    reg.setFocus(id);
    canvas.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", code: "ArrowLeft", bubbles: true }));
    await flush();
    expect(keys).toEqual(["ArrowLeft"]);
    detach();
  });
});
