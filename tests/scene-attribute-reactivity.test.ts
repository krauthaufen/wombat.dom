// Phase B audit: per-buffer aval reactivity through compileScene. The
// outer attribute map is structural; only inner BufferView avals are
// reactive. compile.ts itself does not call `.force()` on attributes —
// verified by the scene-no-force audit; here we verify functional
// correctness end-to-end.

import { describe, expect, it } from "vitest";
import { AVal, HashMap, cval, transact } from "@aardworx/wombat.adaptive";
import type {
  BufferView, DrawCall, Effect, IFramebuffer, RenderTree,
} from "@aardworx/wombat.rendering/core";
import {
  Sg, compileScene,
  type SgLeaf,
} from "../src/scene/index.js";

const fbo: ReturnType<typeof AVal.constant<IFramebuffer>> = AVal.constant({} as IFramebuffer);

const dummyDraw: DrawCall = {
  kind: "non-indexed",
  vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0,
};

function bv(label: string): BufferView {
  return {
    buffer: { kind: "host", data: new Float32Array(0), sizeBytes: 0, label } as unknown as BufferView["buffer"],
    offset: 0, count: 3, stride: 12, format: "float32x3",
  };
}

const fakeEffect = { id: "fakeEffect" } as unknown as Effect;

function singleRender(cmds: ReturnType<typeof compileScene>): RenderTree {
  const list = AVal.force(cmds.content);
  const r = list.toArray().find(c => c.kind === "Render");
  if (r === undefined || r.kind !== "Render") throw new Error("no render command");
  return r.tree;
}

function leafOf(rt: RenderTree): RenderTree & { kind: "Leaf" } {
  if (rt.kind === "Leaf") return rt as RenderTree & { kind: "Leaf" };
  throw new Error(`expected Leaf, got ${rt.kind}`);
}

describe("Phase B — per-buffer reactive vertex attributes through compileScene", () => {
  it("flipping an inner BufferView aval flows through to RenderObject.vertexAttributes without recompiling the scene", () => {
    const a = bv("A");
    const b = bv("B");
    const positionView = cval<BufferView>(a);
    const map = HashMap.empty<string, ReturnType<typeof AVal.constant<BufferView>>>().add("a_position", positionView);

    const sg: SgLeaf = Sg.leaf({
      vertexAttributes: map,
      drawCall: AVal.constant(dummyDraw),
    });
    const cmds = compileScene(Sg.shader(fakeEffect, sg), fbo);
    const rt = leafOf(singleRender(cmds));
    const obj = rt.object;

    // The outer map is plain — the same HashMap reference is threaded
    // through. Per-key avals are still reactive on the leaf.
    const found1 = obj.vertexAttributes.tryFind("a_position");
    expect(found1).toBeDefined();
    expect((AVal.force(found1!).buffer as { label?: string }).label).toBe("A");

    transact(() => { positionView.value = b; });
    const found2 = obj.vertexAttributes.tryFind("a_position");
    expect((AVal.force(found2!).buffer as { label?: string }).label).toBe("B");
  });

  it("Index aval flips between defined and undefined", () => {
    const idx = bv("idx");
    const present: BufferView | undefined = idx;
    const absent: BufferView | undefined = undefined;
    const dynIdx = cval<BufferView | undefined>(present);

    const sg: SgLeaf = Sg.leaf({
      vertexAttributes: HashMap.empty<string, ReturnType<typeof AVal.constant<BufferView>>>().add("a_position", AVal.constant(bv("p"))),
      indices: dynIdx,
      drawCall: AVal.constant(dummyDraw),
    });
    const cmds = compileScene(Sg.shader(fakeEffect, sg), fbo);
    const rt = leafOf(singleRender(cmds));
    const obj = rt.object;
    expect(obj.indices).toBeDefined();

    transact(() => { dynIdx.value = absent; });
    const view = AVal.force(obj.indices!);
    expect(view).toBeUndefined();

    transact(() => { dynIdx.value = present; });
    const view2 = AVal.force(obj.indices!);
    expect(view2).toBeDefined();
  });
});
