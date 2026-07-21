// Increment 2: the RenderControl canvas as the async scene-leaf of the
// RegionRouter. An ancestor DOM handler captures before / bubbles after
// the scene, under one stop model — the full "DOM capture → scene →
// DOM bubble" walk across the boundary. See docs/unified-event-propagation.md.

import { describe, expect, it } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";

import { RegionRouter } from "../src/eventRouter.js";
import { PickDispatcher } from "../src/scene/picking/dispatcher.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import { pixelWinner, noPixel } from "./pickArgminTestUtil.js";
import type { EventHandlers } from "../src/scene/sg.js";
import type { DomParticipant } from "../src/scene/picking/dispatcher.js";

function makeCanvas(parent: Element): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = 200; c.height = 100;
  parent.appendChild(c);
  c.getBoundingClientRect = (): DOMRect => {
    const r = { x: 0, y: 0, top: 0, left: 0, right: 200, bottom: 100, width: 200, height: 100 };
    return { ...r, toJSON: () => r } as DOMRect;
  };
  return c;
}

function acquire(reg: PickRegistry, h: EventHandlers): number {
  return reg.acquire({
    handlers: [{ handlers: h, local2World: () => AVal.constant(Trafo3d.identity) }],
    cursor: undefined, pickThrough: false, active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity), proj: AVal.constant(Trafo3d.identity),
    model: () => AVal.constant(Trafo3d.identity), pixelSnapRadius: AVal.constant(1),
    canFocus: AVal.constant(true),
  });
}

function pevent(el: Element, type: string, x = 50, y = 50): PointerEvent {
  const Ctor = (globalThis as { PointerEvent?: typeof PointerEvent }).PointerEvent
            ?? (globalThis as { MouseEvent: typeof MouseEvent }).MouseEvent;
  const ev = new Ctor(type, { clientX: x, clientY: y, bubbles: true, cancelable: true, button: 0 }) as PointerEvent;
  Object.defineProperty(ev, "pointerId", { value: 1, configurable: true });
  Object.defineProperty(ev, "pointerType", { value: "mouse", configurable: true });
  el.dispatchEvent(ev);
  return ev;
}

async function flush(): Promise<void> {
  for (let i = 0; i < 6; i++) await Promise.resolve();
}

/** root > ancestorDiv > canvas, with a router on root and the dispatcher
 *  attached to the canvas in region mode. */
function harness() {
  const root = document.createElement("div");
  const ancestor = document.createElement("div");
  root.appendChild(ancestor);
  document.body.appendChild(root);
  const canvas = makeCanvas(ancestor);
  const router = new RegionRouter(root);
  const reg = new PickRegistry();
  const d = new PickDispatcher(
    reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect(),
  );
  return { root, ancestor, canvas, router, reg, d };
}

describe("RenderControl canvas as the router's scene leaf", () => {
  it("ancestor DOM handler bubbles AFTER the scene", async () => {
    const { ancestor, canvas, router, reg, d } = harness();
    const log: string[] = [];
    const id = acquire(reg, { bubble: { OnPointerDown: () => { log.push("scene"); } } });
    router.registerHandler(ancestor, "bubble", "pointerdown", () => { log.push("ancestor.bub"); });
    router.registerHandler(ancestor, "capture", "pointerdown", () => { log.push("ancestor.cap"); });
    const detach = d.attach(canvas, async () => pixelWinner(id, { px: 50, py: 50, dist2: 0 }), { region: router });

    pevent(canvas, "pointerdown");
    await flush();

    expect(log).toEqual(["ancestor.cap", "scene", "ancestor.bub"]);
    detach();
  });

  it("a scene stopPropagation() suppresses the ancestor DOM bubble", async () => {
    const { ancestor, canvas, router, reg, d } = harness();
    const log: string[] = [];
    const id = acquire(reg, { bubble: { OnPointerDown: (e) => { log.push("scene"); e.stopPropagation(); } } });
    router.registerHandler(ancestor, "bubble", "pointerdown", () => { log.push("ancestor.bub"); });
    const detach = d.attach(canvas, async () => pixelWinner(id, { px: 50, py: 50, dist2: 0 }), { region: router });

    pevent(canvas, "pointerdown");
    await flush();

    expect(log).toEqual(["scene"]); // ancestor bubble suppressed across the boundary
    detach();
  });

  it("an ancestor CAPTURE handler can stop the event before the scene ever picks", async () => {
    const { ancestor, canvas, router, reg, d } = harness();
    const log: string[] = [];
    let picked = false;
    const id = acquire(reg, { bubble: { OnPointerDown: () => { log.push("scene"); } } });
    router.registerHandler(ancestor, "capture", "pointerdown", (e) => { log.push("ancestor.cap"); e.stopPropagation(); });
    const detach = d.attach(canvas, async () => { picked = true; return pixelWinner(id, { px: 50, py: 50, dist2: 0 }); }, { region: router });

    pevent(canvas, "pointerdown");
    await flush();

    expect(log).toEqual(["ancestor.cap"]);
    expect(picked).toBe(false);  // scene never dispatched
    detach();
  });

  it("the camera (dispatcher participant) still stops via a scene handler in region mode", async () => {
    const { canvas, router, reg, d } = harness();
    const log: string[] = [];
    const cam: DomParticipant = { bubble: { pointerdown: () => { log.push("cam"); } } };
    d.registerDomParticipant(cam);
    const id = acquire(reg, { bubble: { OnPointerDown: (e) => { log.push("scene"); e.stopPropagation(); } } });
    const detach = d.attach(canvas, async () => pixelWinner(id, { px: 50, py: 50, dist2: 0 }), { region: router });

    pevent(canvas, "pointerdown");
    await flush();

    expect(log).toEqual(["scene"]); // camera suppressed
    detach();
  });

  it("keyboard on the focused canvas routes to the focused scene scope through the leaf", async () => {
    const { ancestor, canvas, router, reg, d } = harness();
    const log: string[] = [];
    const id = acquire(reg, { bubble: { OnKeyDown: (e) => { log.push(`scene:${e.key}`); } } });
    reg.setFocus(id);
    router.registerHandler(ancestor, "bubble", "keydown", () => { log.push("ancestor.bub"); });
    const detach = d.attach(canvas, async () => noPixel(), { region: router });

    const ev = new KeyboardEvent("keydown", { key: "a", bubbles: true, cancelable: true });
    canvas.dispatchEvent(ev);
    await flush();

    // Scene scope gets the key; the ancestor keydown bubbles after it.
    expect(log).toEqual(["scene:a", "ancestor.bub"]);
    detach();
  });

  it("an overlay sibling click does NOT reach the scene leaf", async () => {
    const { root, canvas, router, reg, d } = harness();
    const log: string[] = [];
    const overlay = document.createElement("button");
    root.appendChild(overlay);
    router.registerHandler(overlay, "bubble", "click", () => { log.push("overlay"); });
    let picked = false;
    d.attach(canvas, async () => { picked = true; return noPixel(); }, { region: router });

    pevent(overlay, "click");
    await flush();

    expect(log).toEqual(["overlay"]);
    expect(picked).toBe(false); // overlay is a separate DOM branch, no scene dive
  });
});
