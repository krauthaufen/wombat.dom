// Per-scope local2World on event dispatch.
//
// Each entry in a leaf's path-of-scopes carries the model trafo
// accumulated UP TO AND INCLUDING that scope. The dispatcher applies
// it (via `event.transformed(local2World)`) before invoking each
// handler, so handlers see `e.position` etc. in their OWN local frame.
// F# parity: `event.Transformed(model)` —
// `Aardvark.Dom/SceneGraph/TraversalState.fs runCapture/runBubble`.

import { describe, expect, it } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d } from "@aardworx/wombat.base";

import { PickDispatcher } from "../src/scene/picking/dispatcher.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import type { PickArgminResult } from "../src/scene/picking/pickArgminCompute.js";
import { noPixel, pixelWinner } from "./pickArgminTestUtil.js";
import type { SceneEvent } from "../src/scene/picking/sceneEvent.js";
import type { EventHandlers } from "../src/scene/sg.js";
import { TraversalState } from "../src/scene/traversalState.js";

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

function makeRegion(centerX: number, centerY: number, pickId: number): PickArgminResult {
  // Argmin verdict: pickId is the centre-pixel winner (0 ⇒ no hit).
  return pickId === 0 ? noPixel() : pixelWinner(pickId, { px: centerX, py: centerY });
}

function pevent(canvas: HTMLCanvasElement, type: string, x: number, y: number, pointerId = 1): PointerEvent {
  const Ctor = (globalThis as { PointerEvent?: typeof PointerEvent }).PointerEvent
            ?? (globalThis as { MouseEvent: typeof MouseEvent }).MouseEvent;
  const ev = new Ctor(type, { clientX: x, clientY: y, bubbles: true, cancelable: true, button: 0 }) as PointerEvent;
  Object.defineProperty(ev, "pointerId", { value: pointerId, configurable: true });
  Object.defineProperty(ev, "pointerType", { value: "mouse", configurable: true });
  canvas.dispatchEvent(ev);
  return ev;
}

async function flush(): Promise<void> { await Promise.resolve(); await Promise.resolve(); }

describe("PickDispatcher — per-scope local2World on dispatch", () => {
  it("each handler sees e.position in its OWN local frame", async () => {
    // Two-level scope: outer translates +(10,0,0), inner translates
    // an additional +(0,5,0). Inner local2World = T(10,5,0).
    const outerT = Trafo3d.translation(new V3d(10, 0, 0));
    const innerT = Trafo3d.translation(new V3d(10, 5, 0));

    const reg = new PickRegistry();
    let outerPos: V3d | undefined;
    let innerPos: V3d | undefined;
    let outerWorld: V3d | undefined;
    let innerWorld: V3d | undefined;
    const outer: EventHandlers = {
      bubble: { OnClick: (e) => { outerPos = e.position; outerWorld = e.worldPos; } },
    };
    const inner: EventHandlers = {
      bubble: { OnClick: (e) => { innerPos = e.position; innerWorld = e.worldPos; } },
    };

    const id = reg.acquire({
      handlers: [
        { handlers: outer, local2World: AVal.constant(outerT) },
        { handlers: inner, local2World: AVal.constant(innerT) },
      ],
      cursor: undefined,
      pickThrough: false,
      active: AVal.constant(true),
      view: AVal.constant(Trafo3d.identity),
      proj: AVal.constant(Trafo3d.identity),
      model: AVal.constant(innerT),
      pixelSnapRadius: AVal.constant(1),
    });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const detach = d.attach(canvas, async () => makeRegion(50, 50, id));

    pevent(canvas, "click", 50, 50);
    await flush();

    expect(outerWorld).toBeDefined();
    expect(innerWorld).toBeDefined();
    // World positions are identical across handlers (the world hit is
    // shared) — only the local frame changes.
    expect(innerWorld!.x).toBeCloseTo(outerWorld!.x, 6);
    expect(innerWorld!.y).toBeCloseTo(outerWorld!.y, 6);
    expect(innerWorld!.z).toBeCloseTo(outerWorld!.z, 6);

    // e.position = worldPos transformed by the handler's local2World⁻¹.
    const outerExpected = outerT.backward.transformPos(outerWorld!);
    const innerExpected = innerT.backward.transformPos(innerWorld!);

    expect(outerPos).toBeDefined();
    expect(innerPos).toBeDefined();
    expect(outerPos!.x).toBeCloseTo(outerExpected.x, 6);
    expect(outerPos!.y).toBeCloseTo(outerExpected.y, 6);
    expect(outerPos!.z).toBeCloseTo(outerExpected.z, 6);
    expect(innerPos!.x).toBeCloseTo(innerExpected.x, 6);
    expect(innerPos!.y).toBeCloseTo(innerExpected.y, 6);
    expect(innerPos!.z).toBeCloseTo(innerExpected.z, 6);

    // Sanity: the inner offset is non-zero so the two e.position
    // values must differ.
    const dx = outerPos!.x - innerPos!.x;
    const dy = outerPos!.y - innerPos!.y;
    const dz = outerPos!.z - innerPos!.z;
    expect(dx * dx + dy * dy + dz * dz).toBeGreaterThan(1e-6);
  });

  it("siblings under a shared On scope see coords in the shared scope's frame", async () => {
    // Shared parent T(7,0,0). Two leaf-level scopes (child A: T(7,0,0); child B: T(7,0,0)
    // — leaves in the same parent scope, no per-leaf trafo).
    const parentT = Trafo3d.translation(new V3d(7, 0, 0));
    const reg = new PickRegistry();
    const seen: { which: string; pos: V3d }[] = [];
    const parent: EventHandlers = {
      bubble: { OnClick: (e) => { seen.push({ which: "parent", pos: e.position }); } },
    };
    const childA: EventHandlers = { bubble: { OnClick: (e) => { seen.push({ which: "A", pos: e.position }); } } };
    const childB: EventHandlers = { bubble: { OnClick: (e) => { seen.push({ which: "B", pos: e.position }); } } };
    const idA = reg.acquire({
      handlers: [
        { handlers: parent, local2World: AVal.constant(parentT) },
        { handlers: childA, local2World: AVal.constant(parentT) },
      ],
      cursor: undefined, pickThrough: false,
      active: AVal.constant(true),
      view: AVal.constant(Trafo3d.identity),
      proj: AVal.constant(Trafo3d.identity),
      model: AVal.constant(parentT),
      pixelSnapRadius: AVal.constant(1),
    });
    const idB = reg.acquire({
      handlers: [
        { handlers: parent, local2World: AVal.constant(parentT) },
        { handlers: childB, local2World: AVal.constant(parentT) },
      ],
      cursor: undefined, pickThrough: false,
      active: AVal.constant(true),
      view: AVal.constant(Trafo3d.identity),
      proj: AVal.constant(Trafo3d.identity),
      model: AVal.constant(parentT),
      pixelSnapRadius: AVal.constant(1),
    });

    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    let next = idA;
    const detach = d.attach(canvas, async (cx, cy) => makeRegion(cx, cy, next));

    pevent(canvas, "click", 50, 50);
    await flush();

    // Click on leaf A: parent fires (capture+bubble), childA fires.
    // Both must see e.position in the same parentT-local frame.
    const parentSeen = seen.filter(s => s.which === "parent");
    const aSeen = seen.filter(s => s.which === "A");
    expect(parentSeen.length).toBe(1);
    expect(aSeen.length).toBe(1);
    expect(parentSeen[0]!.pos.x).toBeCloseTo(aSeen[0]!.pos.x, 6);
    expect(parentSeen[0]!.pos.y).toBeCloseTo(aSeen[0]!.pos.y, 6);

    seen.length = 0;
    next = idB;
    pevent(canvas, "click", 60, 60);
    await flush();

    const parentSeen2 = seen.filter(s => s.which === "parent");
    const bSeen = seen.filter(s => s.which === "B");
    expect(parentSeen2.length).toBe(1);
    expect(bSeen.length).toBe(1);
    expect(parentSeen2[0]!.pos.x).toBeCloseTo(bSeen[0]!.pos.x, 6);
    expect(parentSeen2[0]!.pos.y).toBeCloseTo(bSeen[0]!.pos.y, 6);

    detach();
  });
});

describe("TraversalState.pushHandlers — local2World snapshot", () => {
  it("snapshots the model accumulated UP TO AND INCLUDING the scope", () => {
    const t1 = Trafo3d.translation(new V3d(1, 0, 0));
    const t2 = Trafo3d.translation(new V3d(0, 2, 0));
    const aHandlers: EventHandlers = { bubble: { OnClick: (): void => {} } };
    const bHandlers: EventHandlers = { bubble: { OnClick: (): void => {} } };

    // Outer Trafo t1, then On A, then Trafo t2, then On B.
    const s = TraversalState.empty
      .pushTrafo(t1)
      .pushHandlers(aHandlers)
      .pushTrafo(t2)
      .pushHandlers(bHandlers);

    expect(s.handlers).toHaveLength(2);
    // Entry A's local2World captures only t1.
    const aL = AVal.force(s.handlers[0]!.local2World);
    expect(aL.forward.toArray()).toEqual(t1.forward.toArray());
    // Entry B's local2World captures t1 ∘ t2 = the full state.model
    // at scope-B entry.
    const bL = AVal.force(s.handlers[1]!.local2World);
    expect(bL.forward.toArray()).toEqual(AVal.force(s.model).forward.toArray());
  });
});
