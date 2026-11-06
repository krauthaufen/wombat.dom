// Lifetimes of lowered subtrees: an `Sg.delay` creator's scope ends when its
// lowering leaves the render tree, on every removal path (ordered group,
// unordered group, adaptive swap, dynamic uniform swap, whole-scene
// disposal); pick ids ride the same mechanism.

import { describe, expect, it } from "vitest";
import {
  AList, ASet, AVal, HashMap, cmap, clist, cset, cval, transact, type aval,
} from "@aardworx/wombat.adaptive";
import { ElementType, type BufferView, type DrawCall, type RenderTree } from "@aardworx/wombat.rendering/core";
import { stage, type Effect } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { Tf32, Vec } from "@aardworx/wombat.shader/ir";

import { Sg, compileScene, PickRegistry, TraversalState, type SgLeaf, type SgNode } from "../src/scene/index.js";
import { LoweringScope } from "../src/scene/lifetime.js";

const dummyDraw: DrawCall = { kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 };
const dummyView: BufferView = {
  buffer: AVal.constant({ kind: "host", data: new Float32Array(0), sizeBytes: 0 }),
  elementType: ElementType.V3f,
};
function leaf(): SgLeaf {
  return Sg.leaf({
    vertexAttributes: HashMap.empty<string, BufferView>().add("Positions", dummyView),
    drawCall: AVal.constant(dummyDraw),
  });
}

/** Delay nodes counting their live (traversal) scopes by name; probes are counted apart. */
function tracker() {
  const counts = new Map<string, number>();
  const live = {
    add: (n: string) => { counts.set(n, (counts.get(n) ?? 0) + 1); },
    delete: (n: string) => { const c = (counts.get(n) ?? 0) - 1; if (c > 0) counts.set(n, c); else counts.delete(n); },
    has: (n: string) => counts.has(n),
    get size() { return counts.size; },
    [Symbol.iterator]: () => counts.keys(),
  };
  let probes = 0;
  let ended = 0;
  const delay = (name: string): SgNode =>
    Sg.delay((state, scope) => {
      if (state === TraversalState.empty) {
        probes++;
        scope.onDispose(() => { probes--; });
        return leaf();
      }
      live.add(name);
      scope.onDispose(() => { live.delete(name); ended++; });
      return leaf();
    });
  return { delay, live, probes: () => probes, ended: () => ended };
}

/** Reads the compiled tree like a renderer does: every adaptive part, again after each change. */
function drain(cmds: ReturnType<typeof compileScene>): void {
  const walk = (t: RenderTree): void => {
    switch (t.kind) {
      case "Leaf": case "Empty": return;
      case "Ordered": case "Unordered": t.children.forEach(walk); return;
      case "Adaptive": walk(AVal.force(t.tree)); return;
      case "OrderedFromList": AList.force(t.children).toArray().forEach(walk); return;
      case "UnorderedFromSet": for (const c of ASet.force(t.children)) walk(c); return;
      case "Rows": ASet.force(t.set.rows); return;
    }
  };
  for (const c of AList.force(cmds).toArray()) if (c.kind === "Render") walk(c.tree);
}

describe("Sg.delay scope lifetimes", () => {
  it("one-argument creators keep working", () => {
    let calls = 0;
    const cmds = compileScene(Sg.delay(() => { calls++; return leaf(); }));
    drain(cmds);
    expect(calls).toBeGreaterThan(0);
  });

  it("probe scopes have ended when the creator returns", () => {
    const t = tracker();
    const cmds = compileScene(Sg.group([t.delay("a"), Sg.pass(1, leaf())]));
    drain(cmds);
    expect(t.probes()).toBe(0);
    expect(t.live.has("a")).toBe(true);
  });

  it("disposing the compiled scene ends every scope", () => {
    const t = tracker();
    const life = new LoweringScope();
    const swap = cval<SgNode>(t.delay("adaptive"));
    const set = cset<SgNode>([t.delay("unordered")]);
    const list = clist<SgNode>([t.delay("ordered")]);
    const scene = Sg.group([t.delay("root"), Sg.adaptive(swap), Sg.unordered(set), Sg.group(list as never)]);
    drain(compileScene(scene, { lifetime: life }));
    expect([...t.live].sort()).toEqual(["adaptive", "ordered", "root", "unordered"]);
    life.dispose();
    expect(t.live.size).toBe(0);
  });

  it("unordered group: removing a child ends its scope only", () => {
    const t = tracker();
    const a = t.delay("a"), b = t.delay("b");
    const set = cset<SgNode>([a, b]);
    const cmds = compileScene(Sg.unordered(set));
    drain(cmds);
    expect([...t.live].sort()).toEqual(["a", "b"]);
    transact(() => { set.remove(a); });
    drain(cmds);
    expect([...t.live]).toEqual(["b"]);
  });

  it("ordered group: removing or replacing a child ends its scope", () => {
    const t = tracker();
    const list = clist<SgNode>([t.delay("a"), t.delay("b")]);
    const cmds = compileScene(Sg.group(list as never));
    drain(cmds);
    expect([...t.live].sort()).toEqual(["a", "b"]);
    transact(() => { list.removeAt(0); });
    drain(cmds);
    expect([...t.live]).toEqual(["b"]);
    transact(() => { list.setAt(0, t.delay("c")); });
    drain(cmds);
    expect([...t.live]).toEqual(["c"]);
  });

  it("adaptive group: a swap ends the previous subtree's scope", () => {
    const t = tracker();
    const swap = cval<SgNode>(t.delay("a"));
    const cmds = compileScene(Sg.adaptive(swap));
    drain(cmds);
    expect([...t.live]).toEqual(["a"]);
    transact(() => { swap.value = t.delay("b"); });
    drain(cmds);
    expect([...t.live]).toEqual(["b"]);
  });

  it("dynamic uniforms: a key change ends the previous lowering's scope", () => {
    const t = tracker();
    const bag = cmap<string, aval<unknown>>([["A", AVal.constant(1)]]);
    const cmds = compileScene(Sg.uniform(bag, t.delay("u")));
    drain(cmds);
    expect(t.live.has("u")).toBe(true);
    const before = t.ended();
    transact(() => { bag.set("B", AVal.constant(2)); });
    drain(cmds);
    expect(t.ended()).toBe(before + 1);
    expect(t.live.has("u")).toBe(true);
  });

  it("each traversal gets its own scope: the same node in two places", () => {
    const t = tracker();
    const shared = t.delay("x");
    let scopes = 0;
    const counting = Sg.delay((state, scope) => {
      if (state !== TraversalState.empty) { scopes++; scope.onDispose(() => { scopes--; }); }
      return shared;
    });
    const life = new LoweringScope();
    drain(compileScene(Sg.group([counting, counting]), { lifetime: life }));
    expect(scopes).toBe(2);
    life.dispose();
    expect(scopes).toBe(0);
  });
});

/** A pass-through effect (as in pick-compile.test.ts): leaves under it acquire pick ids. */
function effect(): Effect {
  const v4 = Vec(Tf32, 4);
  const source = `
    function vsMain(input: { Positions: V4f }): { gl_Position: V4f } { return { gl_Position: input.Positions }; }
    function fsMain(input: {}): { Color: V4f } { return { Color: new V4f(1.0, 1.0, 1.0, 1.0) }; }
  `;
  const entries: EntryRequest[] = [
    { name: "vsMain", stage: "vertex",
      inputs: [{ name: "Positions", type: v4, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "gl_Position", type: v4, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] }] },
    { name: "fsMain", stage: "fragment", inputs: [],
      outputs: [{ name: "Color", type: v4, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }] },
  ];
  return stage(parseShader({ source, entries }));
}

describe("pick ids share the lifetime", () => {
  const eff = effect();
  const picked = (): SgNode => Sg.shader(eff, Sg.on({ bubble: { OnClick: () => {} } }, leaf()));

  it("an adaptive swap releases the previous subtree's pick ids", () => {
    const registry = new PickRegistry();
    const swap = cval<SgNode>(Sg.group([picked(), picked()]));
    const cmds = compileScene(Sg.adaptive(swap), { picking: { registry } });
    drain(cmds);
    expect(registry.size()).toBe(2);
    transact(() => { swap.value = picked(); });
    drain(cmds);
    expect(registry.size()).toBe(1);
  });

  it("ordered group removal and scene disposal release pick ids", () => {
    const registry = new PickRegistry();
    const life = new LoweringScope();
    const list = clist<SgNode>([picked(), picked()]);
    const cmds = compileScene(Sg.group([picked(), Sg.group(list as never)]), { picking: { registry }, lifetime: life });
    drain(cmds);
    expect(registry.size()).toBe(3);
    transact(() => { list.removeAt(1); });
    drain(cmds);
    expect(registry.size()).toBe(2);
    life.dispose();
    expect(registry.size()).toBe(0);
  });
});
