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
import { ElementType } from "@aardworx/wombat.rendering/core";
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
  // The label rides on the IBuffer kind:"host" object so callers can
  // round-trip identity through `AVal.force(view.buffer).label`.
  return {
    buffer: AVal.constant({
      kind: "host", data: new Float32Array(0), sizeBytes: 0, label,
    } as never),
    elementType: ElementType.V3f,
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
  it("flipping the inner aval<IBuffer> flows through to RenderObject.vertexAttributes without recompiling the scene", () => {
    const ibA = { kind: "host", data: new Float32Array(0), sizeBytes: 0, label: "A" } as never;
    const ibB = { kind: "host", data: new Float32Array(0), sizeBytes: 0, label: "B" } as never;
    const bufCval = cval(ibA);
    const view: BufferView = { buffer: bufCval, elementType: ElementType.V3f };
    const map = HashMap.empty<string, BufferView>().add("Positions", view);

    const sg: SgLeaf = Sg.leaf({
      vertexAttributes: map,
      drawCall: AVal.constant(dummyDraw),
    });
    const cmds = compileScene(Sg.shader(fakeEffect, sg), fbo);
    const rt = leafOf(singleRender(cmds));
    const obj = rt.object;

    const found1 = obj.vertexAttributes.tryFind("Positions");
    expect(found1).toBeDefined();
    expect((AVal.force(found1!.buffer) as { label?: string }).label).toBe("A");

    transact(() => { bufCval.value = ibB; });
    const found2 = obj.vertexAttributes.tryFind("Positions");
    expect((AVal.force(found2!.buffer) as { label?: string }).label).toBe("B");
  });
});
