// Render passes are found while lowering (compile.ts `PassSplit`), not by
// an eager pre-walk of the scene: compileScene evaluates no adaptive
// container, a pass that first appears at runtime is drawn in order, and
// a pass nobody names any more loses its lowering.

import { describe, expect, it } from "vitest";
import { AList, AVal, HashMap, cval, clist, transact } from "@aardworx/wombat.adaptive";

import { Sg, compileScene, type SgLeaf, type SgNode } from "../src/scene/index.js";
import type { BufferView, DrawCall, RenderObject, RenderTree } from "@aardworx/wombat.rendering/core";
import { ElementType } from "@aardworx/wombat.rendering/core";
import type { alist } from "@aardworx/wombat.adaptive";
import type { Command } from "@aardworx/wombat.rendering/core";

const dummyDraw: DrawCall = { kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 };
const dummyView: BufferView = {
  buffer: AVal.constant({ kind: "host", data: new Float32Array(0), sizeBytes: 0 }),
  offset: 0,
  stride: 12,
  elementType: ElementType.V3f,
};
function leaf(): SgLeaf {
  return Sg.leaf({
    vertexAttributes: HashMap.empty<string, BufferView>().add("Positions", dummyView),
    drawCall: AVal.constant(dummyDraw),
  });
}
/** A leaf identified by its effect's id. */
const named = (id: string): SgNode => Sg.shader({ id } as never, leaf());

function walk(t: RenderTree, out: RenderObject[]): void {
  switch (t.kind) {
    case "Empty": return;
    case "Leaf": out.push(t.object); return;
    case "Ordered": case "Unordered": for (const c of t.children) walk(c, out); return;
    case "OrderedFromList": for (const c of AVal.force(t.children.content)) walk(c, out); return;
    case "UnorderedFromSet": for (const c of AVal.force(t.children.content)) walk(c, out); return;
    case "Adaptive": walk(AVal.force(t.tree), out); return;
  }
}
/** The ids in draw order after one evaluation of the compiled scene. */
function order(cmds: alist<Command>): string[] {
  const out: RenderObject[] = [];
  for (const cmd of AVal.force(cmds.content)) if (cmd.kind === "Render") walk(cmd.tree, out);
  return out.map((o) => (o.effect as never as { id: string }).id);
}
/** Number of passes of the compiled scene: one Render command each. */
function passCount(cmds: alist<Command>): number {
  return AVal.force(cmds.content).toArray().filter((c) => c.kind === "Render").length;
}

describe("Sg.pass — reactive pass discovery", () => {
  it("compileScene evaluates no adaptive container (no pre-walk)", () => {
    let mapped = 0;
    const items = clist<string>(["a", "b", "c"]);
    const children = AList.map((id: string) => { mapped++; return named(id); }, items);
    const cmds = compileScene(Sg.group(children), { defaultEffect: { id: "FX" } as never });
    expect(mapped).toBe(0);
    expect(order(cmds)).toEqual(["a", "b", "c"]);
    // each child lowered once: no second evaluation for pass discovery
    expect(mapped).toBe(3);
  });

  it("passes in constant structure are drawn from the first evaluation", () => {
    const cmds = compileScene(Sg.group([Sg.pass(10, named("A")), named("B"), Sg.pass(-1, named("C"))]));
    expect(order(cmds)).toEqual(["C", "B", "A"]);
  });

  it("a linear pass's Render command is marked linear, in constant structure and when it appears at runtime", () => {
    const linear = (cmds: alist<Command>): boolean[] =>
      AVal.force(cmds.content).toArray().filter((c) => c.kind === "Render").map((c) => (c as { linear?: boolean }).linear === true);
    const constant = compileScene(Sg.group([Sg.linearPass(10, named("A")), named("B"), Sg.pass(5, named("C"))]));
    expect(order(constant)).toEqual(["B", "C", "A"]);
    expect(linear(constant)).toEqual([false, false, true]);
    const slot = cval<SgNode>(Sg.empty);
    const late = compileScene(Sg.group([named("B"), Sg.adaptive(slot)]));
    expect(order(late)).toEqual(["B"]);
    expect(linear(late)).toEqual([false]);
    transact(() => { slot.value = Sg.linearPass(10, named("A")); });
    order(late);
    expect(order(late)).toEqual(["B", "A"]);
    expect(linear(late)).toEqual([false, true]);
  });

  it("a pass that appears at runtime is drawn, in pass order", () => {
    const slot = cval<SgNode>(Sg.empty);
    const cmds = compileScene(Sg.group([Sg.adaptive(slot), named("A")]));
    expect(order(cmds)).toEqual(["A"]);
    transact(() => { slot.value = Sg.pass(10, named("B")); });
    // met while evaluating: lowered right after, drawn from the next evaluation
    order(cmds);
    expect(order(cmds)).toEqual(["A", "B"]);
  });

  it("a subtree whose pass flips keeps rendering, in the new order", () => {
    const slot = cval<SgNode>(Sg.pass(10, named("B")));
    const cmds = compileScene(Sg.group([Sg.adaptive(slot), named("A")]));
    order(cmds);
    expect(order(cmds)).toEqual(["A", "B"]);
    expect(passCount(cmds)).toBe(2);
    transact(() => { slot.value = Sg.pass(-5, named("B")); });
    order(cmds);
    expect(order(cmds)).toEqual(["B", "A"]);
    // pass 10 is named by nothing any more: its lowering is gone
    expect(passCount(cmds)).toBe(2);
    transact(() => { slot.value = named("B"); });
    order(cmds);
    expect(order(cmds)).toEqual(["B", "A"]);
    expect(passCount(cmds)).toBe(1);
  });

  it("a pass appearing in a scene that had none at compile time", () => {
    const items = clist<SgNode>([named("A")]);
    const cmds = compileScene(Sg.group(items));
    expect(order(cmds)).toEqual(["A"]);
    transact(() => { items.prepend(Sg.transparent(named("T"))); });
    order(cmds);
    expect(order(cmds)).toEqual(["A", "T"]);
  });

  it("a pass the user filter rejects gets no lowering", () => {
    const cmds = compileScene(Sg.group([Sg.pass(10, named("A")), named("B")]), { passFilter: (p) => p < 5 });
    expect(order(cmds)).toEqual(["B"]);
    expect(passCount(cmds)).toBe(1);
  });

  it("each pass is a Render command of its own, in pass order: the renderer keeps no order inside one", () => {
    const cmds = compileScene(Sg.group([Sg.pass(3000, named("over")), named("base"), Sg.pass(1000, named("mid"))]), { clear: {} as never });
    const list = AVal.force(cmds.content).toArray();
    expect(list.map((c) => c.kind)).toEqual(["Clear", "Render", "Render", "Render"]);
    const ids = list.filter((c) => c.kind === "Render").map((c) => { const out: RenderObject[] = []; walk((c as { tree: RenderTree }).tree, out); return out.map((o) => (o.effect as never as { id: string }).id); });
    expect(ids).toEqual([["base"], ["mid"], ["over"]]);
  });
});
