// Split rows (docs/split-rows.md): a collection child that is row-safe
// scopes over groups of leaves contributes each leaf as a row of the
// collection, a row set per plan; switches on the way are the row's flag
// and their modes its membership.

import { __basePassTree } from "../src/scene/compile.js";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AVal, ASet, HashMap, cset, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { stage, type Effect } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import { IBuffer, ElementType } from "@aardworx/wombat.rendering/core";
import type { BufferView, DrawCall, RenderRow, RenderTree } from "@aardworx/wombat.rendering/core";
import {
  Sg, compileScene, sceneEfficiency, resetEfficiency, __setRowLowering, PickRegistry, idleRegistration,
  markHostBufferAVal, markHeapSafeDrawAVal,
  type SgNode,
} from "../src/scene/index.js";
import { resetTemplates } from "../src/scene/template.js";

const dummyDraw: DrawCall = { kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 };
const view = (): BufferView => ({
  buffer: AVal.constant(IBuffer.fromHost(new Float32Array(9).buffer)),
  elementType: ElementType.V3f,
});
const leaf = (): SgNode => Sg.leaf({
  vertexAttributes: HashMap.empty<string, BufferView>().add("Positions", view()),
  drawCall: AVal.constant(dummyDraw),
});
const Tvec4f: Type = Vec(Tf32, 4);
function eff(green = 1.0): Effect {
  const entries: EntryRequest[] = [
    { name: "vsMain", stage: "vertex",
      inputs: [{ name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] }] },
    { name: "fsMain", stage: "fragment", inputs: [],
      outputs: [{ name: "Color", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }] },
  ];
  return stage(parseShader({
    source: `
      function vsMain(input: { Positions: V4f }): { gl_Position: V4f } { return { gl_Position: input.Positions }; }
      function fsMain(input: {}): { Color: V4f } { return { Color: new V4f(1.0, ${green.toFixed(1)}, 1.0, 1.0) }; }
    `,
    entries,
  }));
}

type RowsTree = RenderTree & { kind: "Rows" };
const content = <T>(s: unknown): T[] => [...(AVal.force((s as { content: never }).content) as Iterable<T>)];

/** The group's row sets (primary first) and its classic trees. */
function lowered(scene: SgNode, opts: Parameters<typeof compileScene>[1] = {}): () => { sets: RowsTree[]; rows: RenderRow[]; classic: RenderTree[] } {
  const cmds = compileScene(scene, opts);
  const render = AVal.force(cmds.content).toArray().find((c: { kind: string }) => c.kind === "Render") as { tree: RenderTree };
  const tree = __basePassTree(render.tree) as RenderTree & { kind: "Unordered" };
  return () => {
    const primary = tree.children.find((k) => k.kind === "Rows") as RowsTree;
    const setNode = tree.children.find((k) => k.kind === "UnorderedFromSet")!;
    const inSet = content<RenderTree>((setNode as unknown as { children: unknown }).children).filter((t) => t.kind !== "Empty");
    const more = inSet.filter((t) => t.kind === "Rows") as RowsTree[];
    const sets = [primary, ...more].filter((s) => content(s.set.rows).length > 0);
    const rows = sets.flatMap((s) => content<RenderRow>(s.set.rows));
    return { sets, rows, classic: inSet.filter((t) => t.kind !== "Rows") };
  };
}
const multiLeafBails = (): number => sceneEfficiency().bails.find((b) => b.reason === "multi-leaf-subtree")?.count ?? 0;

beforeEach(() => { resetTemplates(); resetEfficiency(); __setRowLowering(true); });
afterEach(() => { idleRegistration.enabled = true; });

describe("split rows", () => {
  it("a multi-leaf child's leaves are rows of the collection", () => {
    const child = (i: number) => Sg.uniform({ LineColor: AVal.init(i) }, Sg.group([leaf(), leaf()]));
    const read = lowered(Sg.shader(eff(), Sg.unordered([child(1), child(2), child(3)])));
    const { sets, rows, classic } = read();
    expect(rows.length).toBe(6);
    expect(sets.length).toBe(1);
    expect(classic.length).toBe(0);
    expect(multiLeafBails()).toBe(0);
  });

  it("nested groups split all the way down", () => {
    const child = (i: number) => Sg.uniform({ LineColor: AVal.init(i) },
      Sg.group([Sg.pickThrough(true, Sg.group([leaf(), leaf()])), Sg.unordered([leaf()])]));
    const { rows, classic } = lowered(Sg.shader(eff(), Sg.unordered([child(1), child(2)])))();
    expect(rows.length).toBe(6);
    expect(classic.length).toBe(0);
  });

  it("mixed plans in one collection: a row set each", () => {
    // a line and a point per item, each its own shader (as an annotation's parts)
    const e1 = eff(1.0), e2 = eff(0.0);
    const child = (i: number) => Sg.uniform({ LineColor: AVal.init(i) },
      Sg.group([Sg.shader(e1, leaf()), Sg.shader(e2, leaf())]));
    const { sets, rows, classic } = lowered(Sg.unordered([child(1), child(2), child(3)]))();
    expect(sets.length).toBe(2);
    expect(sets.map((s) => content(s.set.rows).length)).toEqual([3, 3]);
    expect(rows.length).toBe(6);
    expect(classic.length).toBe(0);
  });

  it("a dynamic inner group's elements come and go as rows, releasing their pick ids", () => {
    const registry = new PickRegistry();
    const parts = cset<SgNode>([leaf()]);
    const child = Sg.uniform({ LineColor: AVal.init(1) }, Sg.unordered(parts));
    const read = lowered(Sg.shader(eff(), Sg.unordered([child, Sg.uniform({ LineColor: AVal.init(2) }, leaf())])), { picking: { registry } });
    expect(read().rows.length).toBe(2);
    const size0 = registry.size();
    const extra = [leaf(), leaf()];
    transact(() => { for (const l of extra) parts.add(l); });
    expect(read().rows.length).toBe(4);
    expect(registry.size()).toBe(size0 + 2);
    transact(() => { parts.remove(extra[0]!); });
    expect(read().rows.length).toBe(3);
    expect(registry.size()).toBe(size0 + 1);
    expect(read().classic.length).toBe(0);
  });

  it("a switch in the spine is the row's flag (eager, no idle time: a member at once)", () => {
    idleRegistration.enabled = false;
    const on = [cval(false), cval(true)];
    const child = (i: number) => Sg.uniform({ LineColor: AVal.init(i) }, Sg.active(on[i]!, Sg.group([leaf()]), { eager: true }));
    const read = lowered(Sg.shader(eff(), Sg.unordered([child(0), child(1)])));
    const { rows, sets } = read();
    expect(rows.length).toBe(2);
    expect(sets.length).toBe(1); // the flag is a hole: one plan for both
    const flags = () => read().rows.map((r) => AVal.force(r.active!)).sort();
    expect(flags()).toEqual([false, true]);
    transact(() => { on[0]!.value = true; });
    expect(flags()).toEqual([true, true]);
  });

  it("eager in idle time: a member once released (or shown)", () => {
    const on = cval(false);
    const read = lowered(Sg.shader(eff(), Sg.unordered([
      Sg.uniform({ LineColor: AVal.init(1) }, Sg.active(on, Sg.group([leaf(), leaf()]), { eager: true })),
    ])));
    expect(read().rows.length).toBe(0);
    idleRegistration.runNow();
    expect(read().rows.length).toBe(2);
    expect(read().rows.every((r) => AVal.force(r.active!) === false)).toBe(true);
  });

  it("nothing below an idle-time switch is read before its release", () => {
    // the item's parts (a dynamic set the app computes) stay unread at load
    let reads = 0;
    const parts = AVal.custom(() => { reads++; return [leaf(), leaf()]; });
    const read = lowered(Sg.shader(eff(), Sg.unordered([
      Sg.uniform({ LineColor: AVal.init(1) }, Sg.active(cval(false), Sg.unordered(ASet.ofAVal(parts)), { eager: "guarded" })),
    ])));
    expect(read().rows.length).toBe(0);
    expect(reads).toBe(0);
    idleRegistration.runNow();
    expect(read().rows.length).toBe(2);
    expect(reads).toBe(1);
  });

  it("no eager: a member from the first show, kept when hidden", () => {
    const on = cval(false);
    const read = lowered(Sg.shader(eff(), Sg.unordered([
      Sg.uniform({ LineColor: AVal.init(1) }, Sg.active(on, Sg.group([leaf(), leaf()]))),
    ])));
    expect(read().rows.length).toBe(0);
    transact(() => { on.value = true; });
    expect(read().rows.length).toBe(2);
    transact(() => { on.value = false; });
    expect(read().rows.length).toBe(2);
  });

  it("guarded: a leaf whose values raise while hidden is left out until shown", () => {
    idleRegistration.enabled = false;
    const on = cval(false);
    const valid = cval(false);
    // geometry only valid while shown (a branch reading what its guard checked)
    const positions: BufferView = {
      buffer: valid.map((v) => { if (!v) throw new Error("not valid while hidden"); return IBuffer.fromHost(new Float32Array(9).buffer); }),
      elementType: ElementType.V3f,
    };
    const raising = Sg.leaf({ vertexAttributes: HashMap.empty<string, BufferView>().add("Positions", positions), drawCall: AVal.constant(dummyDraw) });
    const read = lowered(Sg.shader(eff(), Sg.unordered([
      Sg.active(on, Sg.uniform({ LineColor: AVal.init(2) }, Sg.group([raising, leaf()])), { eager: "guarded" }),
      Sg.active(on, Sg.uniform({ LineColor: AVal.init(1) }, Sg.group([leaf()])), { eager: "guarded" }),
    ])));
    const drawn = (): number => { const r = read(); return r.rows.length + r.classic.length; };
    // the valid leaves are registered hidden; the raising one left out
    expect(drawn()).toBe(2);
    transact(() => { valid.value = true; on.value = true; });
    expect(drawn()).toBe(3);
  });


  it("a mesh whose data changes is a row when its buffers and draw call are proven", () => {
    const data = cval(new Float32Array(9));
    const mesh = (proven: boolean): SgNode => {
      const buffer = data.map((a) => IBuffer.fromHost(a.buffer));
      const drawCall = data.map((a): DrawCall => ({ ...dummyDraw, vertexCount: a.length / 3 }));
      if (proven) { markHostBufferAVal(buffer); markHeapSafeDrawAVal(drawCall, "non-indexed"); }
      return Sg.leaf({ vertexAttributes: HashMap.empty<string, BufferView>().add("Positions", { buffer, elementType: ElementType.V3f }), drawCall });
    };
    const child = (i: number, proven: boolean) => Sg.uniform({ LineColor: AVal.init(i) }, Sg.group([mesh(proven), mesh(proven)]));
    const proven = lowered(Sg.shader(eff(), Sg.unordered([child(1, true), child(2, true), child(3, true)])))();
    expect(proven.rows.length).toBe(6);
    expect(proven.classic.length).toBe(0);
    resetTemplates();
    const unproven = lowered(Sg.shader(eff(), Sg.unordered([child(1, false), child(2, false)])))();
    expect(unproven.rows.length).toBe(0);
    expect(unproven.classic.length).toBe(4);
  });

  it("a single-leaf child takes the unchanged path", () => {
    const child = (i: number) => Sg.uniform({ LineColor: AVal.init(i) }, leaf());
    const { rows, sets } = lowered(Sg.shader(eff(), Sg.unordered([child(1), child(2)])))();
    expect(rows.length).toBe(2);
    expect(sets.length).toBe(1);
  });

  it("rows under PickThrough are pickable, with the scope's flag and the switch's", () => {
    idleRegistration.enabled = false;
    const registry = new PickRegistry();
    const on = cval(true);
    const child = (i: number) => Sg.uniform({ LineColor: AVal.init(i) },
      Sg.active(on, Sg.pickThrough(true, Sg.group([leaf(), leaf()])), { eager: true }));
    const { rows } = lowered(Sg.shader(eff(), Sg.unordered([child(1), child(2), child(3)])), { picking: { registry } })();
    expect(rows.length).toBe(6);
    const scopes = rows.map((r) => registry.lookup(r.pickId!)!);
    expect(scopes.every((s) => s !== undefined)).toBe(true);
    expect(scopes.every((s) => (s as unknown as { pickThrough: boolean }).pickThrough === true)).toBe(true);
    transact(() => { on.value = false; });
    expect(scopes.every((s) => AVal.force((s as unknown as { active: aval<boolean> }).active) === false)).toBe(true);
  });
});
