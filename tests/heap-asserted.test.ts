// Producer-asserted heap eligibility on row-lowered leaves.
// Evidence rules under test: `isConstant` (forceable) or a
// construction-level proof (`__sgHeapSafeDraw` from instancing,
// `markHostBufferAVal`); anything else must NOT assert and keeps
// the reactive predicate.

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AVal, HashMap, cval } from "@aardworx/wombat.adaptive";
import { stage, type Effect } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import { IBuffer, ElementType } from "@aardworx/wombat.rendering/core";
import type { BufferView, DrawCall, RenderObject, RenderTree } from "@aardworx/wombat.rendering/core";

import { Sg, compileScene, __setRowLowering, markHostBufferAVal } from "../src/scene/index.js";
import { resetTemplates } from "../src/scene/template.js";

const dummyDraw: DrawCall = {
  kind: "non-indexed",
  vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0,
};
function view(): BufferView {
  return {
    buffer: AVal.constant(IBuffer.fromHost(new Float32Array(9).buffer)),
    elementType: ElementType.V3f,
  };
}
/** An adaptive buffer aval WITHOUT the construction proof. */
function unprovenAdaptiveView(): BufferView {
  const b = cval(IBuffer.fromHost(new Float32Array(9).buffer));
  return { buffer: b, elementType: ElementType.V3f };
}
/** An adaptive buffer aval WITH the construction proof. */
function provenAdaptiveView(): BufferView {
  const src = cval(new Float32Array(9));
  return {
    buffer: markHostBufferAVal(src.map((a) => IBuffer.fromHost(a.buffer))),
    elementType: ElementType.V3f,
  };
}
function leafWith(v: BufferView): ReturnType<typeof Sg.leaf> {
  return Sg.leaf({
    vertexAttributes: HashMap.empty<string, BufferView>().add("Positions", v),
    drawCall: AVal.constant(dummyDraw),
  });
}
const Tvec4f: Type = Vec(Tf32, 4);
function eff(): Effect {
  const source = `
    function vsMain(input: { Positions: V4f }): { gl_Position: V4f } {
      return { gl_Position: input.Positions };
    }
    function fsMain(input: {}): { Color: V4f } {
      return { Color: new V4f(1.0, 1.0, 1.0, 1.0) };
    }
  `;
  const entries: EntryRequest[] = [
    {
      name: "vsMain", stage: "vertex",
      inputs: [{ name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] }],
    },
    { name: "fsMain", stage: "fragment", inputs: [], outputs: [{ name: "Color", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }] },
  ];
  return stage(parseShader({ source, entries }));
}

function loweredLeaves(scene: ReturnType<typeof Sg.shader>): RenderObject[] {
  const cmds = compileScene(scene, {});
  const list = AVal.force(cmds.content);
  const render = list.toArray().find((c: { kind: string }) => c.kind === "Render") as { tree: RenderTree };
  const out: RenderObject[] = [];
  const walk = (t: RenderTree): void => {
    if (t.kind === "Leaf") { out.push(t.object); return; }
    if (t.kind === "UnorderedFromSet") {
      const content = AVal.force((t as unknown as { children: { content: never } }).children.content) as Iterable<RenderTree>;
      for (const c of content) walk(c);
    }
  };
  walk(render.tree);
  return out;
}

function rowChild(v: BufferView, i: number): ReturnType<typeof Sg.uniform> {
  return Sg.uniform({ LineColor: AVal.init(i) }, leafWith(v));
}

beforeEach(() => { resetTemplates(); __setRowLowering(true); });
afterEach(() => __setRowLowering(true));

describe("heapAsserted on row-lowered leaves", () => {
  it("constant host buffers + constant drawCall → asserted", () => {
    const scene = Sg.shader(eff(), Sg.unordered([rowChild(view(), 1), rowChild(view(), 2)]));
    const leaves = loweredLeaves(scene);
    expect(leaves.length).toBe(2);
    for (const ro of leaves) expect(ro.heapAsserted).toBe(true);
  });

  it("proof-marked ADAPTIVE buffer → asserted (values stay reactive)", () => {
    const scene = Sg.shader(eff(), Sg.unordered([rowChild(provenAdaptiveView(), 1)]));
    const leaves = loweredLeaves(scene);
    expect(leaves.length).toBe(1);
    expect(leaves[0]!.heapAsserted).toBe(true);
  });

  it("UNPROVEN adaptive buffer → not asserted (reactive predicate kept)", () => {
    const scene = Sg.shader(eff(), Sg.unordered([rowChild(unprovenAdaptiveView(), 1)]));
    const leaves = loweredLeaves(scene);
    expect(leaves.length).toBe(1);
    expect(leaves[0]!.heapAsserted).toBeUndefined();
  });

  it("rows off → nothing asserted", () => {
    __setRowLowering(false);
    const scene = Sg.shader(eff(), Sg.unordered([rowChild(view(), 1)]));
    const leaves = loweredLeaves(scene);
    expect(leaves[0]!.heapAsserted).toBeUndefined();
  });
});
