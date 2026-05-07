// Phase 2 — geometry attribute scopes (VertexAttributes,
// InstanceAttributes, Index, Mode). Pin: scope inheritance, compose
// vs override, leaf-wins, mixed nesting.

import { describe, expect, it } from "vitest";
import { AVal, HashMap } from "@aardworx/wombat.adaptive";

import {
  Sg,
  compileScene,
  collectSgChildren,
  type SgNode,
} from "../src/scene/index.js";
import type {
  BufferView, DrawCall, RenderObject, RenderTree,
} from "@aardworx/wombat.rendering/core";

const draw: DrawCall = { kind: "non-indexed", vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 };
function bv(name: string): BufferView {
  return {
    buffer: AVal.constant({ kind: "host", data: new Float32Array(0), sizeBytes: 0  }),
    offset: 0, stride: 12, elementType: "v3f",
    // tag the buffer so we can identify it later
    label: name,
  } as BufferView & { label?: string };
}

const fakeEffect = { id: "FX" } as never;

function compile(node: SgNode | import("../src/vnode.js").VNode): RenderObject[] {
  const sg: SgNode = (node as { _tag?: string })._tag !== undefined
    ? collectSgChildren(node)
    : (node as SgNode);
  const cmds = compileScene(sg, AVal.constant({} as never), { defaultEffect: fakeEffect });
  const out: RenderObject[] = [];
  for (const c of AVal.force(cmds.content)) {
    if (c.kind !== "Render") continue;
    walk(c.tree, out);
  }
  return out;
}

function walk(t: RenderTree, out: RenderObject[]): void {
  switch (t.kind) {
    case "Empty": return;
    case "Leaf": out.push(t.object); return;
    case "Ordered": case "Unordered":
      for (const c of t.children) walk(c, out); return;
    case "OrderedFromList":
      for (const c of AVal.force(t.children.content)) walk(c, out); return;
    case "UnorderedFromSet":
      for (const c of AVal.force(t.children.content)) walk(c, out); return;
    case "Adaptive": walk(AVal.force(t.tree), out); return;
  }
}

const minimalLeaf = (): SgNode => Sg.leaf({
  // Truly minimal: empty vertexAttributes — they'll be supplied by scope.
  vertexAttributes: HashMap.empty<string, BufferView>().add("placeholder", bv("placeholder")),
  drawCall: AVal.constant(draw),
});
type AVal<T> = ReturnType<typeof AVal.constant<T>>;

describe("Phase 2 — geometry-attribute scopes", () => {
  it("VertexAttributes scope flows into a leaf without its own attribute", () => {
    const a = bv("scope-a");
    const scoped: HashMap<string, BufferView> =
      HashMap.empty<string, BufferView>().add("scope-pos", a);
    const tree = Sg.vertexAttributes(scoped)(minimalLeaf());
    const [obj] = compile(tree);
    expect(obj!.vertexAttributes.tryFind("scope-pos")).toBeDefined();
  });

  it("VertexAttributes COMPOSE (per-key map merge, leaf-wins on conflict)", () => {
    const outer = bv("outer");
    const leafBuf = bv("leaf");
    const scoped: HashMap<string, BufferView> = HashMap.empty<string, BufferView>().add("Positions", outer);
    const leaf = Sg.leaf({
      vertexAttributes: HashMap.empty<string, BufferView>().add("Positions", leafBuf),
      drawCall: AVal.constant(draw),
    });
    const tree = Sg.vertexAttributes(scoped)(leaf);
    const [obj] = compile(tree);
    // Leaf wins.
    const pos = obj!.vertexAttributes.tryFind("Positions");
    const view = (pos! as BufferView & { label?: string });
    expect(view.label).toBe("leaf");
  });

  it("Index scope used when leaf has no indices", () => {
    const idx = bv("idx");
    const tree = Sg.index(idx)(Sg.leaf({
      vertexAttributes: HashMap.empty<string, BufferView>().add("p", bv("p")),
      drawCall: AVal.constant(draw),
    }));
    const [obj] = compile(tree);
    expect(obj!.indices).toBeDefined();
  });

  it("Index scope is overridden by leaf's own indices (leaf-wins)", () => {
    const scopeIdx = bv("scope-idx");
    const leafIdx = bv("leaf-idx");
    const tree = Sg.index(scopeIdx)(Sg.leaf({
      vertexAttributes: HashMap.empty<string, BufferView>().add("p", bv("p")),
      indices: leafIdx,
      drawCall: AVal.constant(draw),
    }));
    const [obj] = compile(tree);
    const it = (obj!.indices! as BufferView & { label?: string });
    expect(it.label).toBe("leaf-idx");
  });

  it("InstanceAttributes scope flows into the leaf", () => {
    const inst: HashMap<string, BufferView> = HashMap.empty<string, BufferView>().add("inst-color", bv("inst"));
    const tree = Sg.instanceAttributes(inst)(minimalLeaf());
    const [obj] = compile(tree);
    expect(obj!.instanceAttributes!.tryFind("inst-color")).toBeDefined();
  });

  it("Mode scope flows into pipelineState.rasterizer.topology", () => {
    const tree = Sg.mode("line-list")(minimalLeaf());
    const [obj] = compile(tree);
    expect(AVal.force(obj!.pipelineState.rasterizer.topology)).toBe("line-list");
  });

  it("nested Vertex scopes — inner-wins on key collision, outer keys kept", () => {
    const outerOnly = bv("outer-only");
    const innerOnly = bv("inner-only");
    const conflictOuter = bv("conflict-outer");
    const conflictInner = bv("conflict-inner");
    const o: HashMap<string, BufferView> = HashMap.empty<string, BufferView>()
      .add("outer-only", outerOnly)
      .add("conflict", conflictOuter);
    const i: HashMap<string, BufferView> = HashMap.empty<string, BufferView>()
      .add("inner-only", innerOnly)
      .add("conflict", conflictInner);
    const tree = Sg.vertexAttributes(o)(Sg.vertexAttributes(i)(minimalLeaf()));
    const [obj] = compile(tree);
    expect(obj!.vertexAttributes.tryFind("outer-only")).toBeDefined();
    expect(obj!.vertexAttributes.tryFind("inner-only")).toBeDefined();
    const conflict = (obj!.vertexAttributes.tryFind("conflict")! as BufferView & { label?: string });
    expect(conflict.label).toBe("conflict-inner");
  });
});
