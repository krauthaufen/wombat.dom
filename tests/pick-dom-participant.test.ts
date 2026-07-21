// Unified DOM ↔ scene event propagation (docs/unified-event-propagation.md).
//
// A DOM-world participant (canonically the camera controller) registers
// with the dispatcher instead of installing its own native listener, and
// joins ONE walk per raw event:
//     DOM capture (outer→inner) → scene capture/bubble → DOM bubble (inner→outer)
// So a scene handler that stops propagation halts the outer DOM bubble —
// "stop the camera during a drag" falls out of the model, no special case.

import { describe, expect, it, vi } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";

import { PickDispatcher, type DomParticipant, type DomParticipantHandle } from "../src/scene/picking/dispatcher.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import type { PickArgminResult } from "../src/scene/picking/pickArgminCompute.js";
import { noPixel, pixelWinner } from "./pickArgminTestUtil.js";
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

function region(pickId: number): PickArgminResult {
  return pixelWinner(pickId, { px: 50, py: 50, dist2: 0 });
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

function wheel(canvas: HTMLCanvasElement, x: number, y: number, deltaY = 10): WheelEvent {
  const ev = new WheelEvent("wheel", { clientX: x, clientY: y, deltaY, bubbles: true, cancelable: true });
  canvas.dispatchEvent(ev);
  return ev;
}

async function flush(): Promise<void> {
  // A few extra microtasks over the usual 2 — a vi.fn()-wrapped resolver
  // adds a tick to the settle chain, and the DOM bubble runs one hop
  // after the scene dispatch.
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

function acquire(reg: PickRegistry, handlers: EventHandlers): number {
  return reg.acquire({
    handlers: [{ handlers, local2World: () => AVal.constant(Trafo3d.identity) }],
    cursor: undefined,
    pickThrough: false,
    active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity),
    proj: AVal.constant(Trafo3d.identity),
    model: () => AVal.constant(Trafo3d.identity),
    pixelSnapRadius: AVal.constant(1),
  });
}

describe("PickDispatcher — unified DOM participant walk", () => {
  it("a bubble participant fires after the scene (camera acts over empty sky)", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const part: DomParticipant = { bubble: { pointerdown: () => { log.push("cam.down"); } } };
    d.registerDomParticipant(part);
    // No scene hit — resolvePixel yields nothing.
    const detach = d.attach(canvas, async () => noPixel());

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(log).toEqual(["cam.down"]);
    detach();
  });

  it("scene bubble stopPropagation() suppresses the DOM bubble (stops the camera)", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const scene: EventHandlers = {
      bubble: { OnPointerDown: (e) => { log.push("marker"); e.stopPropagation(); } },
    };
    const id = acquire(reg, scene);
    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    d.registerDomParticipant({ bubble: { pointerdown: () => { log.push("cam.down"); } } });
    const detach = d.attach(canvas, async () => region(id));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(log).toEqual(["marker"]); // camera never sees it
    detach();
  });

  it("scene bubble returning false also suppresses the DOM bubble", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const id = acquire(reg, { bubble: { OnPointerDown: () => { log.push("marker"); return false; } } });
    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    d.registerDomParticipant({ bubble: { pointerdown: () => { log.push("cam.down"); } } });
    const detach = d.attach(canvas, async () => region(id));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(log).toEqual(["marker"]);
    detach();
  });

  it("a scene handler that does NOT stop lets the camera bubble run too", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const id = acquire(reg, { bubble: { OnPointerDown: () => { log.push("marker"); } } });
    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    d.registerDomParticipant({ bubble: { pointerdown: () => { log.push("cam.down"); } } });
    const detach = d.attach(canvas, async () => region(id));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(log).toEqual(["marker", "cam.down"]);
    detach();
  });

  it("a DOM capture participant fires before the scene and can suppress it", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const id = acquire(reg, { bubble: { OnPointerDown: () => { log.push("scene"); } } });
    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    d.registerDomParticipant({ capture: { pointerdown: () => { log.push("overlay.cap"); return false; } } });
    const detach = d.attach(canvas, async () => region(id));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(log).toEqual(["overlay.cap"]); // scene never dispatched
    detach();
  });

  it("capturePointer routes moves straight to the participant, skipping the pick", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    let handle: DomParticipantHandle;
    const part: DomParticipant = {
      bubble: {
        pointerdown: (e) => { log.push("down"); handle.capturePointer((e as PointerEvent).pointerId); },
        pointermove: () => { log.push("move"); },
        pointerup: () => { log.push("up"); },
      },
    };
    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    handle = d.registerDomParticipant(part);
    const resolve = vi.fn(async () => noPixel());
    const detach = d.attach(canvas, resolve);

    pevent(canvas, "pointerdown", 50, 50);
    await flush();
    expect(resolve).toHaveBeenCalledTimes(1); // the down still resolves a pick

    resolve.mockClear();
    pevent(canvas, "pointermove", 60, 55);
    pevent(canvas, "pointermove", 70, 60);
    await flush();
    expect(resolve).not.toHaveBeenCalled();   // captured moves skip the pick entirely
    expect(log).toEqual(["down", "move", "move"]);

    pevent(canvas, "pointerup", 70, 60);
    await flush();
    expect(log).toEqual(["down", "move", "move", "up"]);
    // capture released on up → next move resolves a pick again
    resolve.mockClear();
    pevent(canvas, "pointermove", 80, 65);
    await flush();
    expect(resolve).toHaveBeenCalledTimes(1);
    detach();
  });

  it("a captured pointer still delivers its UP to the scene (click preserved through an orbit claim)", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    let handle: DomParticipantHandle;
    // Camera-style participant: claims the pointer on down (orbit), so
    // moves skip the pick — but the up must still reach the scene.
    const cam: DomParticipant = {
      bubble: {
        pointerdown: (e) => { log.push("cam.down"); handle.capturePointer((e as PointerEvent).pointerId); },
        pointermove: () => { log.push("cam.move"); },
        pointerup: () => { log.push("cam.up"); },
      },
    };
    const id = acquire(reg, { bubble: { OnPointerUp: () => { log.push("scene.up"); } } });
    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    handle = d.registerDomParticipant(cam);
    const detach = d.attach(canvas, async () => region(id));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();               // down → camera claims the pointer
    pevent(canvas, "pointermove", 55, 52);
    await flush();               // captured move → camera only, no scene
    pevent(canvas, "pointerup", 55, 52);
    await flush();               // up → releases, scene AND camera see it once

    expect(log).toEqual(["cam.down", "cam.move", "scene.up", "cam.up"]);
    detach();
  });

  it("wheel: bubble participant fires and default is prevented; scene stop suppresses it", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    d.registerDomParticipant({ bubble: { wheel: () => { log.push("cam.zoom"); } } });
    const detach = d.attach(canvas, async () => noPixel());

    const w = wheel(canvas, 50, 50);
    expect(w.defaultPrevented).toBe(true); // page-scroll suppressed synchronously
    await flush();
    expect(log).toEqual(["cam.zoom"]);

    // A scene wheel handler that stops propagation keeps the camera out.
    log.length = 0;
    const id = acquire(reg, { bubble: { OnWheel: (e) => { log.push("scene.wheel"); e.stopPropagation(); } } });
    detach();
    const d2 = makeDispatcher(reg, canvas);
    d2.registerDomParticipant({ bubble: { wheel: () => { log.push("cam.zoom"); } } });
    const detach2 = d2.attach(canvas, async () => region(id));
    wheel(canvas, 50, 50);
    await flush();
    expect(log).toEqual(["scene.wheel"]);
    detach2();
  });

  it("no participants → behaviour is unchanged (no takeover, pick still runs)", async () => {
    const reg = new PickRegistry();
    const log: string[] = [];
    const id = acquire(reg, { bubble: { OnPointerDown: () => { log.push("scene"); } } });
    const canvas = makeCanvas();
    const d = makeDispatcher(reg, canvas);
    const resolve = vi.fn(async () => region(id));
    const detach = d.attach(canvas, resolve);

    const e = pevent(canvas, "pointerdown", 50, 50);
    await flush();
    expect(log).toEqual(["scene"]);
    expect(e.defaultPrevented).toBe(false); // no takeover when nobody registered
    expect(resolve).toHaveBeenCalledTimes(1);
    detach();
  });
});
