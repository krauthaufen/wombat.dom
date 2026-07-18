// rowWrap / SgRow — the build-time M3 construction bypass, runtime
// side. The Fable plugin rewrites collection mapping functions into
// `rowWrap(loc, f)`; these gates pin the wrapper's value-driven
// behavior and the Row node's lowering paths.

import { beforeEach, describe, expect, it } from "vitest";
import { AVal, HashMap } from "@aardworx/wombat.adaptive";
import { stage, type Effect } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import { IBuffer, ElementType, materializeRow } from "@aardworx/wombat.rendering/core";
import type { BufferView, DrawCall, RenderObject, RenderTree } from "@aardworx/wombat.rendering/core";
import {
  Sg, compileScene, __setRowLowering,
  rowWrap, rowNodesConstructed, resetRowNodesConstructed,
} from "../src/scene/index.js";
import type { SgNode, SgRow } from "../src/scene/sg.js";
import { resetTemplates, resetEfficiency } from "../src/scene/template.js";

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
function leaf(): ReturnType<typeof Sg.leaf> {
  return Sg.leaf({
    vertexAttributes: HashMap.empty<string, BufferView>().add("Positions", view()),
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

/** The row-eligible per-item factory (same shape as an annotation). */
function rowChild(i: number): SgNode {
  return Sg.uniform({ LineColor: AVal.init(i) }, leaf()) as SgNode;
}
/** Per-row On handler → fastRow-unsafe template. */
function classicChild(i: number): SgNode {
  return Sg.on({ bubble: { OnClick: () => void i } }, rowChild(i)) as SgNode;
}

function renderTreeOf(scene: SgNode): RenderTree {
  const cmds = compileScene(scene, {});
  const list = AVal.force(cmds.content);
  const render = list.toArray().find((c: { kind: string }) => c.kind === "Render") as { tree: RenderTree };
  return render.tree;
}

function parts(tree: RenderTree): { rows: unknown[]; classic: RenderTree[] } {
  expect(tree.kind).toBe("Unordered");
  const kids = (tree as RenderTree & { kind: "Unordered" }).children;
  const rowsNode = kids.find((k) => k.kind === "Rows") as RenderTree & { kind: "Rows" };
  const setNode = kids.find((k) => k.kind === "UnorderedFromSet") as RenderTree & { kind: "UnorderedFromSet" };
  const rows = [...(AVal.force((rowsNode.set.rows as unknown as { content: never }).content) as Iterable<unknown>)];
  const classic = [...(AVal.force((setNode as unknown as { children: { content: never } }).children.content) as Iterable<RenderTree>)]
    .filter((t) => t.kind !== "Empty");
  return { rows, classic };
}

function leavesOf(tree: RenderTree, out: RenderObject[] = []): RenderObject[] {
  if (tree.kind === "Leaf") { out.push(tree.object); return out; }
  if (tree.kind === "Ordered" || tree.kind === "Unordered") {
    for (const c of (tree as { children: readonly RenderTree[] }).children) leavesOf(c, out);
    return out;
  }
  if (tree.kind === "Rows") {
    const set = (tree as RenderTree & { kind: "Rows" }).set;
    const rows = AVal.force((set.rows as unknown as { content: never }).content) as Iterable<never>;
    for (const r of rows) out.push(materializeRow(set, r));
    return out;
  }
  if (tree.kind === "UnorderedFromSet") {
    const content = AVal.force((tree as unknown as { children: { content: never } }).children.content) as Iterable<RenderTree>;
    for (const c of content) leavesOf(c, out);
    return out;
  }
  if (tree.kind === "OrderedFromList") {
    const content = AVal.force((tree as unknown as { children: { content: never } }).children.content) as { toArray(): RenderTree[] };
    for (const c of content.toArray()) leavesOf(c, out);
    return out;
  }
  return out;
}

beforeEach(() => {
  resetTemplates(); resetEfficiency(); resetRowNodesConstructed();
  __setRowLowering(true);
});

describe("rowWrap — value-driven wrapping", () => {
  it("eligible nodes become thin SgRow (tree dropped), stats count", () => {
    const f = rowWrap("Test.fs:1", 1, (i) => rowChild(i as number));
    const r = f(1) as SgRow;
    expect(r.kind).toBe("Row");
    expect(r.staged.template.fastRow).toBeDefined();
    expect(r.tree).toBeUndefined();
    expect(r.srcLoc).toBe("Test.fs:1");
    expect(rowNodesConstructed()).toBe(1);
  });

  it("fastRow-unsafe nodes pass through verbatim", () => {
    const f = rowWrap("Test.fs:2", 1, (i) => classicChild(i as number));
    const r = f(1) as SgNode;
    expect(r.kind).toBe("On");
    expect(rowNodesConstructed()).toBe(0);
  });

  it("non-node values pass through untouched", () => {
    const f = rowWrap("Test.fs:3", 1, (i) => (i as number) * 2);
    expect(f(21)).toBe(42);
    const obj = { kind: "SomethingElse" };
    const g = rowWrap("Test.fs:3", 1, () => obj);
    expect(g(0)).toBe(obj);
  });

  it("already-wrapped Rows pass through (no double wrap)", () => {
    const inner = rowWrap("Test.fs:4", 1, (i) => rowChild(i as number));
    const outer = rowWrap("Test.fs:5", 1, (v) => v);
    const r = inner(1) as SgRow;
    expect(outer(r)).toBe(r);
    expect(rowNodesConstructed()).toBe(1);
  });

  it("handles curried mapping functions (k => v => node)", () => {
    const curried = (k: unknown) => (v: unknown) => rowChild((k as number) + (v as number));
    const f = rowWrap("Test.fs:6", 2, curried as (...args: unknown[]) => unknown);
    const r = f(1, 2) as SgRow;
    expect(r.kind).toBe("Row");
  });

  it("wrapper invoked one argument at a time (curry adapter) still works", () => {
    const f = rowWrap("Test.fs:8", 2, ((k: number, v: number) => rowChild(k + v)) as (...args: unknown[]) => unknown);
    const r = (f(1) as (...a: unknown[]) => unknown)(2) as SgRow;
    expect(r.kind).toBe("Row");
  });

  it("function-valued mapping results are NOT misread as partial application", () => {
    const cb = (): number => 42;
    const f = rowWrap("Test.fs:9", 1, () => cb);
    expect(f("key")).toBe(cb);
  });

  it("build rebuilds an equivalent subtree", () => {
    const f = rowWrap("Test.fs:7", 1, (i) => rowChild(i as number));
    const r = f(7) as SgRow;
    const t1 = r.build();
    expect(t1.kind).toBe("Uniform");
  });
});

describe("SgRow lowering", () => {
  it("wrapped children land in the RowSet exactly like unwrapped ones", () => {
    const f = rowWrap("Test.fs:10", 1, (i) => rowChild(i as number));
    const wrapped = [f(1), f(2), f(3)] as SgNode[];
    // first child of a group always lowers classically (anchor seed) —
    // it materializes; the rest stay thin.
    const scene = Sg.shader(eff(), Sg.unordered(wrapped)) as SgNode;
    const { rows, classic } = parts(renderTreeOf(scene));
    expect(rows.length).toBe(3);
    expect(classic.length).toBe(0);
    // rows 2+ never materialized
    const thin = wrapped.filter((w) => (w as SgRow).tree === undefined);
    expect(thin.length).toBeGreaterThanOrEqual(2);
  });

  it("row-parity: wrapped and unwrapped scenes produce identical uniform values", () => {
    const mk = (wrap: boolean): RenderObject[] => {
      resetTemplates();
      const f = rowWrap("Test.fs:11", 1, (i) => rowChild(i as number));
      const children = [1, 2, 3].map((i) => (wrap ? (f(i) as SgNode) : rowChild(i)));
      const scene = Sg.shader(eff(), Sg.unordered(children)) as SgNode;
      return leavesOf(renderTreeOf(scene));
    };
    const a = mk(true);
    const b = mk(false);
    expect(a.length).toBe(b.length);
    const colors = (ros: RenderObject[]) =>
      ros
        .map((ro) => AVal.force((ro.uniforms as unknown as { tryGet(n: string): never }).tryGet("LineColor")))
        .sort();
    expect(colors(a)).toEqual(colors(b));
  });

  it("a Row in an ORDERED context materializes and renders", () => {
    const f = rowWrap("Test.fs:12", 1, (i) => rowChild(i as number));
    const scene = Sg.shader(eff(), Sg.group([f(5) as SgNode])) as SgNode;
    const leaves = leavesOf(renderTreeOf(scene));
    expect(leaves.length).toBe(1);
    expect(AVal.force((leaves[0]!.uniforms as unknown as { tryGet(n: string): never }).tryGet("LineColor"))).toBe(5);
  });

  it("rows off → Rows materialize through the classic path", () => {
    __setRowLowering(false);
    const f = rowWrap("Test.fs:13", 1, (i) => rowChild(i as number));
    const scene = Sg.shader(eff(), Sg.unordered([f(1) as SgNode, f(2) as SgNode])) as SgNode;
    const leaves = leavesOf(renderTreeOf(scene));
    expect(leaves.length).toBe(2);
    __setRowLowering(true);
  });
});
