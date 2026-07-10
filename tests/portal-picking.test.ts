// Portal ("offscreen pick context") resolution — no GPU.
//
// The dispatcher's contract: an argmin winner on a scope carrying a
// `pickSubContext` has uv in slots 1-2 (portal final); the resolver
// maps uv → inner pixel (Y-flip) and recurses `pickAt` into the inner
// scene. Inner hit → events land on the INNER scope (its own id
// space); inner miss → the portal scope itself. Nested portals recurse
// naturally. The GPU half (portal final writes uv, argmin reads it) is
// covered by the warp-portal example / browser validation.

import { describe, expect, it } from "vitest";
import { AVal } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d } from "@aardworx/wombat.base";

import { PickDispatcher } from "../src/scene/picking/dispatcher.js";
import { PickRegistry, type LeafPickScope } from "../src/scene/picking/registry.js";
import type { IPickSubContext, PortalPickHit } from "../src/scene/picking/pickContext.js";
import { resolveThroughPortals } from "../src/scene/picking/pickArbitrate.js";
import type { ResolvedHit } from "../src/scene/picking/spiralHitTest.js";
import type { SceneEvent, SceneEventKind } from "../src/scene/picking/sceneEvent.js";
import type { EventHandlers, SceneEventHandler } from "../src/scene/sg.js";
import { pixelWinner, resolverOf } from "./pickArgminTestUtil.js";

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

function pevent(canvas: HTMLCanvasElement, type: string, x: number, y: number): void {
  const Ctor = (globalThis as { PointerEvent?: typeof PointerEvent }).PointerEvent
            ?? (globalThis as { MouseEvent: typeof MouseEvent }).MouseEvent;
  const ev = new Ctor(type, { clientX: x, clientY: y, bubbles: true, cancelable: true, button: 0 });
  canvas.dispatchEvent(ev);
}

async function flush(): Promise<void> {
  // The portal path has one extra await (sub.pickAt) per nesting level.
  for (let i = 0; i < 8; i++) await Promise.resolve();
}

function bubbleOf(rec: Record<string, (e: SceneEvent) => unknown>): EventHandlers {
  const bubble: Partial<Record<SceneEventKind, SceneEventHandler>> = {};
  for (const [k, v] of Object.entries(rec)) bubble[k as SceneEventKind] = v as SceneEventHandler;
  return { bubble };
}

function acquire(
  reg: PickRegistry,
  handlers: ReadonlyArray<Record<string, (e: SceneEvent) => unknown>>,
  extra: { pickSubContext?: IPickSubContext } = {},
): number {
  return reg.acquire({
    handlers: handlers.map(h => ({ handlers: bubbleOf(h), local2World: AVal.constant(Trafo3d.identity) })),
    cursor: undefined,
    pickThrough: false,
    active: AVal.constant(true),
    view: AVal.constant(Trafo3d.identity),
    proj: AVal.constant(Trafo3d.identity),
    model: AVal.constant(Trafo3d.identity),
    pixelSnapRadius: AVal.constant(1),
    ...extra,
  });
}

/** A fake inner producer: registry + a canned pickAt. */
function makeSubContext(
  innerReg: PickRegistry,
  innerScopeId: () => number | undefined,
  size = { width: 100, height: 100 },
  onPickAt?: (x: number, y: number) => void,
): IPickSubContext {
  return {
    size: AVal.constant(size),
    async pickAt(x: number, y: number): Promise<PortalPickHit | undefined> {
      onPickAt?.(x, y);
      const id = innerScopeId();
      if (id === undefined) return undefined;
      const scope = innerReg.lookup(id);
      if (scope === undefined) return undefined;
      const hit: ResolvedHit = {
        scope,
        viewPos: new V3d(1, 2, 3),
        viewNormal: new V3d(0, 0, 1),
        partIndex: 0,
        isPixel: true,
        hoverPickId: id,
      };
      return { hit, registry: innerReg };
    },
  };
}

describe("portal picking", () => {
  it("recursion: click on a portal pixel dispatches the INNER scope", async () => {
    const outer = new PickRegistry();
    const inner = new PickRegistry();

    const innerCalls: SceneEvent[] = [];
    const innerId = acquire(inner, [{ OnPointerDown: (e) => innerCalls.push(e) }]);

    const portalCalls: SceneEvent[] = [];
    const sub = makeSubContext(inner, () => innerId);
    const portalId = acquire(outer, [{ OnPointerDown: (e) => portalCalls.push(e) }], { pickSubContext: sub });

    const canvas = makeCanvas();
    const d = makeDispatcher(outer, canvas);
    // Portal winner: +id, slots 1-2 = uv, slot3 = own depth.
    const detach = d.attach(canvas, resolverOf(pixelWinner(portalId, { slot1: 0.5, slot2: 0.5, slot3: 0.3 })));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(innerCalls.length).toBe(1);
    expect(innerCalls[0]!.pickId).toBe(innerId);
    expect(portalCalls.length).toBe(0);
    detach();
  });

  it("uv → inner pixel mapping Y-flips against the inner size", async () => {
    const outer = new PickRegistry();
    const inner = new PickRegistry();
    const innerId = acquire(inner, [{}]);

    const coords: Array<{ x: number; y: number }> = [];
    const sub = makeSubContext(inner, () => innerId, { width: 400, height: 200 }, (x, y) => coords.push({ x, y }));
    const portalId = acquire(outer, [{}], { pickSubContext: sub });

    const canvas = makeCanvas();
    const d = makeDispatcher(outer, canvas);
    // uv = (0.25, 0.75): tc origin bottom-left → pixel-space y = (1-0.75)*200.
    const detach = d.attach(canvas, resolverOf(pixelWinner(portalId, { slot1: 0.25, slot2: 0.75, slot3: 0.3 })));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(coords.length).toBe(1);
    expect(coords[0]!.x).toBe(Math.floor(0.25 * 400));
    expect(coords[0]!.y).toBe(Math.floor(0.25 * 200));
    detach();
  });

  it("inner miss falls through to the portal scope itself", async () => {
    const outer = new PickRegistry();
    const inner = new PickRegistry();

    const portalCalls: SceneEvent[] = [];
    const sub = makeSubContext(inner, () => undefined); // always miss
    const portalId = acquire(outer, [{ OnPointerDown: (e) => portalCalls.push(e) }], { pickSubContext: sub });

    const canvas = makeCanvas();
    const d = makeDispatcher(outer, canvas);
    const detach = d.attach(canvas, resolverOf(pixelWinner(portalId, { slot1: 0.5, slot2: 0.5, slot3: 0.3 })));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    expect(portalCalls.length).toBe(1);
    expect(portalCalls[0]!.pickId).toBe(portalId);
    detach();
  });

  it("id collision across registries dispatches the right scope", async () => {
    // Outer scope and inner scope share pickId 1 numerically — inner
    // must win when the recursion resolves, decoded against ITS registry.
    const outer = new PickRegistry();
    const inner = new PickRegistry();

    const innerCalls: SceneEvent[] = [];
    const innerId = acquire(inner, [{ OnPointerDown: (e) => innerCalls.push(e) }]);

    const sub = makeSubContext(inner, () => innerId);
    const portalId = acquire(outer, [{}], { pickSubContext: sub });
    expect(innerId).toBe(1);
    expect(portalId).toBe(1); // deliberate collision

    const canvas = makeCanvas();
    const d = makeDispatcher(outer, canvas);
    const detach = d.attach(canvas, resolverOf(pixelWinner(portalId, { slot1: 0.5, slot2: 0.5, slot3: 0.3 })));

    pevent(canvas, "pointerdown", 50, 50);
    await flush();

    // Only the INNER scope's handler pushes into innerCalls — the
    // outer scope with the same numeric id has none.
    expect(innerCalls.length).toBe(1);
    expect(innerCalls[0]!.pickId).toBe(innerId);
    detach();
  });

  it("nested portals: resolveThroughPortals recurses to the innermost hit", async () => {
    const outer = new PickRegistry();
    const mid = new PickRegistry();
    const innermost = new PickRegistry();

    const deepId = acquire(innermost, [{}]);
    const deepSub = makeSubContext(innermost, () => deepId);

    // The mid-level portal scope: its pickAt arbitrates in mid space,
    // finds ITS portal scope, and recurses — mirroring renderToPickable's
    // pickAt (arbitrate → resolveThroughPortals).
    const midPortalId = acquire(mid, [{}], { pickSubContext: deepSub });
    const midSub: IPickSubContext = {
      size: AVal.constant({ width: 100, height: 100 }),
      async pickAt(): Promise<PortalPickHit | undefined> {
        const scope = mid.lookup(midPortalId)!;
        const hit: ResolvedHit = {
          scope,
          viewPos: V3d.zero, viewNormal: V3d.zero, partIndex: 0,
          isPixel: true, hoverPickId: midPortalId,
          portalUv: { x: 0.5, y: 0.5 },
        };
        const resolved = await resolveThroughPortals(hit);
        if (resolved === undefined) return undefined;
        return { hit: resolved, registry: resolved.registry ?? mid };
      },
    };
    const outerPortalId = acquire(outer, [{}], { pickSubContext: midSub });

    const outerScope = outer.lookup(outerPortalId)!;
    const outerHit: ResolvedHit = {
      scope: outerScope,
      viewPos: V3d.zero, viewNormal: V3d.zero, partIndex: 0,
      isPixel: true, hoverPickId: outerPortalId,
      portalUv: { x: 0.5, y: 0.5 },
    };
    const resolved = await resolveThroughPortals(outerHit);
    expect(resolved).toBeDefined();
    expect(resolved!.scope).toBe(innermost.lookup(deepId));
    expect(resolved!.registry).toBe(innermost);
  });

  it("hover transitions across the portal boundary fire enter/leave on the inner scope", async () => {
    const outer = new PickRegistry();
    const inner = new PickRegistry();

    const events: string[] = [];
    const innerId = acquire(inner, [{
      OnPointerEnter: () => events.push("inner-enter"),
      OnPointerLeave: () => events.push("inner-leave"),
    }]);
    let innerHitOn = true;
    const sub = makeSubContext(inner, () => (innerHitOn ? innerId : undefined));
    const portalId = acquire(outer, [{
      OnPointerEnter: () => events.push("portal-enter"),
      OnPointerLeave: () => events.push("portal-leave"),
    }], { pickSubContext: sub });

    const canvas = makeCanvas();
    const d = makeDispatcher(outer, canvas);
    const detach = d.attach(canvas, resolverOf(pixelWinner(portalId, { slot1: 0.5, slot2: 0.5, slot3: 0.3 })));

    pevent(canvas, "pointermove", 50, 50);
    await flush();
    expect(events).toEqual(["inner-enter"]);

    // Same portal pixel, but the inner scene now misses → the hit
    // becomes the portal scope: leave inner, enter portal.
    innerHitOn = false;
    pevent(canvas, "pointermove", 51, 50);
    await flush();
    expect(events).toEqual(["inner-enter", "inner-leave", "portal-enter"]);
    detach();
  });
});
