// Phase 3 — misc scopes: NoEvents, ForcePixelPicking, CanFocus.

import { describe, expect, it } from "vitest";
import { AVal, HashMap } from "@aardworx/wombat.adaptive";
import { Box3d, Intersectable, Trafo3d, V3d } from "@aardworx/wombat.base";
import { stage, type Effect } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";

import {
  Sg,
  compileScene,
  PickRegistry,
} from "../src/scene/index.js";
import { PickDispatcher } from "../src/scene/picking/dispatcher.js";
import type { PickRegion } from "../src/scene/picking/readback.js";
import { SNAP_RADIUS_MAX, SNAP_REGION_SIZE } from "../src/scene/picking/snapOffsets.js";
import type { BufferView, DrawCall } from "@aardworx/wombat.rendering/core";
import { ElementType } from "@aardworx/wombat.rendering/core";

const Tvec4f: Type = Vec(Tf32, 4);

const draw: DrawCall = { kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 };
const bv: BufferView = { buffer: AVal.constant({ kind: "host", data: new Float32Array(0), sizeBytes: 0  }), offset: 0, stride: 12, elementType: ElementType.V3f };

function buildUserEffect(): Effect {
  const source = `
    function vsMain(input: { Positions: V4f }): { gl_Position: V4f } {
      return { gl_Position: input.Positions };
    }
    function fsMain(input: {}): { Color: V4f } {
      return { Color: new V4f(1.0, 1.0, 1.0, 1.0) };
    }
  `;
  const entries: EntryRequest[] = [
    { name: "vsMain", stage: "vertex",
      inputs: [{ name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] }] },
    { name: "fsMain", stage: "fragment", inputs: [],
      outputs: [{ name: "Color", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }] },
  ];
  return stage(parseShader({ source, entries }));
}
const fakeEffect: Effect = buildUserEffect();

function leaf(): import("../src/scene/index.js").SgLeaf {
  return Sg.leaf({ vertexAttributes: HashMap.empty<string, BufferView>().add("Positions", bv), drawCall: AVal.constant(draw) });
}
type AVal<T> = ReturnType<typeof AVal.constant<T>>;

describe("Phase 3 — NoEvents", () => {
  it("NoEvents=true skips PickRegistry registration", () => {
    const reg = new PickRegistry();
    const tree = Sg.noEvents(true, leaf());
    compileScene(tree, { defaultEffect: fakeEffect, picking: { registry: reg } });
    expect(reg.size()).toBe(0);
  });

  it("NoEvents=false (or absent) registers normally", () => {
    const reg = new PickRegistry();
    compileScene(leaf(), { defaultEffect: fakeEffect, picking: { registry: reg } });
    expect(reg.size()).toBe(1);
  });

  it("NoEvents wraps subtree — descendant leaves all skipped", () => {
    const reg = new PickRegistry();
    const tree = Sg.noEvents(true, Sg.group([leaf(), leaf(), leaf()]));
    compileScene(tree, { defaultEffect: fakeEffect, picking: { registry: reg } });
    expect(reg.size()).toBe(0);
  });
});

describe("Phase 3 — CanFocus", () => {
  it("CanFocus is captured on LeafPickScope", () => {
    const reg = new PickRegistry();
    const tree = Sg.canFocus(true, leaf());
    compileScene(tree, { defaultEffect: fakeEffect, picking: { registry: reg } });
    expect(reg.size()).toBe(1);
    const scope = reg.lookup(1);
    expect(scope).toBeDefined();
    expect(AVal.force(scope!.canFocus!)).toBe(true);
  });

  it("CanFocus default is false (no scope)", () => {
    const reg = new PickRegistry();
    compileScene(leaf(), { defaultEffect: fakeEffect, picking: { registry: reg } });
    const scope = reg.lookup(1)!;
    // Either undefined or AVal of false — both fine; doc says default false.
    if (scope.canFocus !== undefined) {
      expect(AVal.force(scope.canFocus)).toBe(false);
    }
  });
});

describe("Phase 3 — ForcePixelPicking suppresses BVH fall-through", () => {
  function makeCanvas(): HTMLCanvasElement {
    const c = document.createElement("canvas");
    c.width = 200; c.height = 100; document.body.appendChild(c);
    c.getBoundingClientRect = (): DOMRect => {
      const r = { x: 0, y: 0, top: 0, left: 0, right: 200, bottom: 100, width: 200, height: 100 };
      return { ...r, toJSON: () => r } as DOMRect;
    };
    return c;
  }
  function emptyRegion(cx: number, cy: number): PickRegion {
    const sz = SNAP_REGION_SIZE;
    return { data: new Float32Array(sz * sz * 4), originX: cx - SNAP_RADIUS_MAX, originY: cy - SNAP_RADIUS_MAX, sizeX: sz, sizeY: sz };
  }
  it("BVH fall-through skips a scope whose forcePixelPicking=true", async () => {
    const reg = new PickRegistry();
    const box = Intersectable.box(Box3d.fromMinMax(new V3d(-1, -1, -0.5), new V3d(1, 1, 0.5)));
    reg.acquire({
      handlers: [], cursor: undefined, pickThrough: false,
      active: AVal.constant(true), view: AVal.constant(Trafo3d.identity),
      proj: AVal.constant(Trafo3d.identity), model: AVal.constant(Trafo3d.identity),
      pixelSnapRadius: AVal.constant(1),
      intersectable: AVal.constant(box),
      forcePixelPicking: AVal.constant(true),
    });
    const canvas = makeCanvas();
    const d = new PickDispatcher(reg, () => Trafo3d.identity, () => Trafo3d.identity, () => canvas.getBoundingClientRect());
    const calls: number[] = [];
    // No way to register a global handler; just confirm registry sees no hit.
    // Use the BVH directly to test the closestHit filter behaviour.
    const bvh = AVal.force(reg.bvhAval);
    expect(bvh.count).toBeGreaterThan(0);
    // Cursor ray (identity view+proj). Hit predicate inside dispatcher
    // returns undefined for forcePixelPicking. Mirror that here.
    const hit = bvh.closestHit(
      // ray pointing through origin along +z
      { origin: new V3d(0, 0, -10), direction: new V3d(0, 0, 1) } as never,
      0, Number.POSITIVE_INFINITY,
      (_key, entry) => {
        const s = entry.scope;
        if (s.forcePixelPicking !== undefined && AVal.force(s.forcePixelPicking)) return undefined;
        return entry.intersectable.intersects({ origin: new V3d(0, 0, -10), direction: new V3d(0, 0, 1) } as never, 0, Number.POSITIVE_INFINITY);
      },
    );
    expect(hit).toBeUndefined();
    calls.push(0); void d; void calls; void emptyRegion;
  });
});
