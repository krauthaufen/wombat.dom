// Phase 1 — render-state scopes (DepthTest/Mask/Bias/Clamp, Cull,
// FrontFace, Fill, BlendConstant, ColorMask, Stencil,
// Pass). Pin: pipeline-state derivation, override semantics, Pass
// grouping.

import { describe, expect, it } from "vitest";
import { AList, AVal, HashMap } from "@aardworx/wombat.adaptive";

import {
  Sg,
  TraversalState,
  compileScene,
  type SgLeaf,
} from "../src/scene/index.js";
import type { BufferView, DrawCall, RenderObject, RenderTree } from "@aardworx/wombat.rendering/core";
import { ElementType } from "@aardworx/wombat.rendering/core";

const dummyDraw: DrawCall = {
  kind: "non-indexed",
  vertexCount: 3,
  instanceCount: 1,
  firstVertex: 0,
  firstInstance: 0,
};
const dummyView: BufferView = {
  buffer: AVal.constant({ kind: "host", data: new Float32Array(0), sizeBytes: 0  }),
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

const fakeEffect = { id: "FX" } as never;

import { collectSgChildren } from "../src/scene/index.js";
import type { VNode } from "../src/vnode.js";

function compileToObjects(input: import("../src/scene/index.js").SgNode | VNode): RenderObject[] {
  const node = (input as { _tag?: string })._tag !== undefined
    ? collectSgChildren(input)
    : (input as import("../src/scene/index.js").SgNode);
  const out: RenderObject[] = [];
  const cmds = compileScene(node, AVal.constant({} as never), { defaultEffect: fakeEffect });
  for (const cmd of AVal.force(cmds.content)) {
    if (cmd.kind !== "Render") continue;
    walk(cmd.tree, out);
  }
  return out;
}

function walk(t: RenderTree, out: RenderObject[]): void {
  switch (t.kind) {
    case "Empty": return;
    case "Leaf": out.push(t.object); return;
    case "Ordered":
    case "Unordered":
      for (const c of t.children) walk(c, out); return;
    case "OrderedFromList":
      for (const c of AVal.force(t.children.content)) walk(c, out); return;
    case "UnorderedFromSet":
      for (const c of AVal.force(t.children.content)) walk(c, out); return;
    case "Adaptive":
      walk(AVal.force(t.tree), out); return;
  }
}

describe("Phase 1 — render-state scopes", () => {
  it("DepthTest overrides default 'less' on the leaf's PipelineState", () => {
    const tree = Sg({ DepthTest: "greater", children: leaf() });
    const [obj] = compileToObjects(tree);
    expect(AVal.force(obj!.pipelineState.depth!.compare)).toBe("greater");
  });

  it("DepthTest innermost wins (override)", () => {
    const tree = Sg({ DepthTest: "always", children: Sg({ DepthTest: "equal", children: leaf() }) });
    const [obj] = compileToObjects(tree);
    expect(AVal.force(obj!.pipelineState.depth!.compare)).toBe("equal");
  });

  it("DepthMask propagates to depth.write", () => {
    const tree = Sg({ DepthMask: false, children: leaf() });
    const [obj] = compileToObjects(tree);
    expect(AVal.force(obj!.pipelineState.depth!.write)).toBe(false);
  });

  it("DepthBias maps to rasterizer.depthBias", () => {
    const tree = Sg({ DepthBias: { constant: 1, slopeScale: 2, clamp: 0 }, children: leaf() });
    const [obj] = compileToObjects(tree);
    expect(AVal.force(obj!.pipelineState.rasterizer.depthBias!)).toEqual({ constant: 1, slopeScale: 2, clamp: 0 });
  });

  it("CullMode + FrontFace propagate", () => {
    const tree = Sg({ CullMode: "back", FrontFace: "cw", children: leaf() });
    const [obj] = compileToObjects(tree);
    expect(AVal.force(obj!.pipelineState.rasterizer.cullMode)).toBe("back");
    expect(AVal.force(obj!.pipelineState.rasterizer.frontFace)).toBe("cw");
  });

  it("FillMode='line' overrides topology to line-list (with warning)", () => {
    const tree = Sg({ FillMode: "line", children: leaf() });
    const [obj] = compileToObjects(tree);
    expect(AVal.force(obj!.pipelineState.rasterizer.topology)).toBe("line-list");
  });

  it("StencilMode produces a stencil state when enabled", () => {
    const tree = Sg({
      StencilMode: {
        enabled: true,
        reference: 1,
        readMask: 0xff,
        writeMask: 0xff,
        front: { compare: "always", fail: "keep", depthFail: "keep", pass: "replace" },
        back:  { compare: "always", fail: "keep", depthFail: "keep", pass: "replace" },
      },
      children: leaf(),
    });
    const [obj] = compileToObjects(tree);
    expect(obj!.pipelineState.stencil).toBeDefined();
    expect(AVal.force(obj!.pipelineState.stencil!.front.passOp)).toBe("replace");
  });

  it("ColorMask single-attachment shorthand produces an 'Colors' blend with channel mask", () => {
    // The shorthand attaches to the framebuffer's `Colors` slot —
    // the canonical name primitives + DefaultSurfaces use. The
    // earlier `"color"` spelling was historical (matched a test
    // fixture from before the DefaultSemantic rename pass).
    const tree = Sg({ ColorMask: { r: true, g: false, b: true, a: false }, children: leaf() });
    const [obj] = compileToObjects(tree);
    const blends = AVal.force(obj!.pipelineState.blends!);
    const cb = blends.tryFind("Colors");
    expect(cb).toBeDefined();
    // R=1, B=4 → mask = 5
    expect(AVal.force(cb!.writeMask)).toBe(5);
  });

  it("Pass groups leaves: pass=10 leaf comes after pass=0 leaf regardless of scene order", () => {
    const a = leaf();
    const b = leaf();
    // Tree order: a (pass=10) FIRST, b (pass=0) SECOND.
    const tree = Sg.group([
      Sg.pass(10, a),
      Sg.pass(0, b),
    ]);
    const objs = compileToObjects(tree);
    expect(objs).toHaveLength(2);
    // After grouping, pass=0 (b) emits before pass=10 (a). Identify
    // objects via their leaf reference is awkward; check ordering by
    // re-running with distinguishable shaders.
    const aFx = { id: "A" } as never;
    const bFx = { id: "B" } as never;
    const tree2 = Sg.group([
      Sg.pass(10, Sg.shader(aFx, leaf())),
      Sg.pass(0, Sg.shader(bFx, leaf())),
    ]);
    const objs2 = compileToObjects(tree2);
    expect(objs2.map(o => (o.effect as never as { id: string }).id)).toEqual(["B", "A"]);
  });

  it("Pass-less scene preserves scene-graph order (no pass-grouping side-effects)", () => {
    const aFx = { id: "A" } as never;
    const bFx = { id: "B" } as never;
    const tree = Sg.group([Sg.shader(aFx, leaf()), Sg.shader(bFx, leaf())]);
    const objs = compileToObjects(tree);
    expect(objs.map(o => (o.effect as never as { id: string }).id)).toEqual(["A", "B"]);
  });

  it("TraversalState push* methods are available and non-mutating", () => {
    const s = TraversalState.empty.pushDepthTest(AVal.constant("equal"));
    expect(AVal.force(s.depthTest)).toBe("equal");
    expect(AVal.force(TraversalState.empty.depthTest)).toBe("less");
  });
});

void AList;
