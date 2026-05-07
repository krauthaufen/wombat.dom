// M2 — Sg core sanity. Pure data + traversal; no GPU, no JSX.

import { describe, expect, it } from "vitest";
import { AList, AVal, HashMap, cval, transact } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d } from "@aardworx/wombat.base";

import {
  Sg,
  TraversalState,
  composeTrafoValue,
  collectLeaves,
  countLeaves,
  type SgLeaf,
  type SgNode,
} from "../src/scene/index.js";

import type { BufferView, DrawCall } from "@aardworx/wombat.rendering/core";
import { ElementType } from "@aardworx/wombat.rendering/core";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Trafo composition
// ---------------------------------------------------------------------------

describe("composeTrafoValue", () => {
  it("identity on empty array", () => {
    const t = AVal.force(composeTrafoValue([]));
    expect(t).toBe(Trafo3d.identity);
  });

  it("single value passes through", () => {
    const T = Trafo3d.translation(new V3d(1, 0, 0));
    expect(AVal.force(composeTrafoValue(T))).toBe(T);
  });

  it("array order: index 0 outermost — point hits index n-1 first", () => {
    // [translate(1,0,0), scale(2)] applied to the origin should produce
    // (2 * 0, 0, 0) + (1, 0, 0) = (1, 0, 0)? No — the rule is "scale
    // first, then translate", so origin → scale=origin → translate=(1,0,0).
    const arr = [
      Sg.translate(new V3d(1, 0, 0)) as Trafo3d,
      Sg.scale(2) as Trafo3d,
    ];
    const composed = AVal.force(composeTrafoValue(arr));
    const out = composed.transform(V3d.zero);
    expect(out.x).toBeCloseTo(1, 6);
    expect(out.y).toBeCloseTo(0, 6);
    expect(out.z).toBeCloseTo(0, 6);

    // A different starting point exercises the scale: (1,0,0) → (2,0,0) → (3,0,0)
    const out2 = composed.transform(new V3d(1, 0, 0));
    expect(out2.x).toBeCloseTo(3, 6);
  });

  it("propagates aval changes", () => {
    const t = cval(Trafo3d.translation(new V3d(1, 0, 0)));
    const arr = [t, Sg.scale(1) as Trafo3d];
    const composed = composeTrafoValue(arr);
    const before = AVal.force(composed).transform(V3d.zero);
    expect(before.x).toBeCloseTo(1, 6);

    transact(() => { t.value = Trafo3d.translation(new V3d(5, 0, 0)); });
    const after = AVal.force(composed).transform(V3d.zero);
    expect(after.x).toBeCloseTo(5, 6);
  });
});

// ---------------------------------------------------------------------------
// TraversalState composition
// ---------------------------------------------------------------------------

describe("TraversalState — attribute composition", () => {
  it("empty state has identity model and active=true", () => {
    const s = TraversalState.empty;
    expect(AVal.force(s.model)).toBe(Trafo3d.identity);
    expect(AVal.force(s.active)).toBe(true);
    expect(s.shader).toBeUndefined();
    expect(s.uniforms.count).toBe(0);
    expect(s.handlers).toHaveLength(0);
  });

  it("Trafo composes by left-mul: outer applied last", () => {
    // descend through Translate(1,0,0) then Scale(2);
    // a leaf-local origin should land at (1,0,0) after both:
    // local=0 → scale=(0,0,0) → translate=(1,0,0).
    const s = TraversalState.empty
      .pushTrafo(Sg.translate(new V3d(1, 0, 0)) as Trafo3d)
      .pushTrafo(Sg.scale(2) as Trafo3d);
    const out = AVal.force(s.model).transform(V3d.zero);
    expect(out.x).toBeCloseTo(1, 6);

    const out2 = AVal.force(s.model).transform(new V3d(1, 0, 0));
    expect(out2.x).toBeCloseTo(3, 6);
  });

  it("Shader overrides (innermost wins)", () => {
    const fakeA = { id: "A" } as unknown as Parameters<typeof TraversalState.empty.pushShader>[0];
    const fakeB = { id: "B" } as unknown as Parameters<typeof TraversalState.empty.pushShader>[0];
    const s = TraversalState.empty.pushShader(fakeA).pushShader(fakeB);
    expect(s.shader).toBe(fakeB);
  });

  it("Uniform composes per-key with inner-wins", () => {
    const outer = HashMap.empty<string, ReturnType<typeof AVal.constant<unknown>>>()
      .add("tint", AVal.constant("red"))
      .add("sharpness", AVal.constant(0.7));
    const inner = HashMap.empty<string, ReturnType<typeof AVal.constant<unknown>>>()
      .add("tint", AVal.constant("blue"));
    const s = TraversalState.empty.pushUniforms(outer).pushUniforms(inner);
    expect(AVal.force(s.uniforms.tryFind("tint")!)).toBe("blue");
    expect(AVal.force(s.uniforms.tryFind("sharpness")!)).toBe(0.7);
  });

  it("handlers append to chain (outermost first)", () => {
    const a = { bubble: { OnClick: (): void => {} } };
    const b = { bubble: { OnClick: (): void => {} } };
    const s = TraversalState.empty.pushHandlers(a).pushHandlers(b);
    expect(s.handlers).toHaveLength(2);
    expect(s.handlers[0]!.handlers).toBe(a);
    expect(s.handlers[1]!.handlers).toBe(b);
    // Each entry snapshots the model trafo at scope-push time.
    expect(s.handlers[0]!.local2World).toBeDefined();
    expect(s.handlers[1]!.local2World).toBeDefined();
  });

  it("active AND-composes", () => {
    const a = cval(true);
    const b = cval(true);
    const s = TraversalState.empty.pushActive(a).pushActive(b);
    expect(AVal.force(s.active)).toBe(true);
    transact(() => { b.value = false; });
    expect(AVal.force(s.active)).toBe(false);
    transact(() => { b.value = true; a.value = false; });
    expect(AVal.force(s.active)).toBe(false);
  });

  it("with* methods are non-mutating — original state unchanged", () => {
    const s = TraversalState.empty;
    s.pushTrafo(Sg.scale(7) as Trafo3d);
    expect(AVal.force(s.model)).toBe(Trafo3d.identity);
  });
});

// ---------------------------------------------------------------------------
// Visitors
// ---------------------------------------------------------------------------

describe("forEachLeaf / collectLeaves", () => {
  it("counts leaves through Group + nested attribute scopes", () => {
    const tree: SgNode = Sg.group([
      Sg.trafo(Sg.translate(new V3d(1, 0, 0)) as Trafo3d, leaf()),
      Sg.shader({ id: "X" } as never, Sg.group([leaf(), leaf()])),
    ]);
    expect(countLeaves(tree)).toBe(3);
  });

  it("propagates accumulated state to leaves", () => {
    const tree = Sg.trafo(
      Sg.translate(new V3d(2, 0, 0)) as Trafo3d,
      Sg.group([leaf(), leaf()]),
    );
    const collected = collectLeaves(tree);
    expect(collected).toHaveLength(2);
    for (const { state } of collected) {
      const out = AVal.force(state.model).transform(V3d.zero);
      expect(out.x).toBeCloseTo(2, 6);
    }
  });

  it("alist children flow through (forced once for the static visitor)", () => {
    const list = AList.ofList<SgNode>([leaf(), leaf(), leaf()]);
    const tree = Sg.group(list);
    expect(countLeaves(tree)).toBe(3);
  });

  it("Empty contributes zero leaves", () => {
    expect(countLeaves(Sg.empty)).toBe(0);
    expect(countLeaves(Sg.group([Sg.empty, leaf(), Sg.empty]))).toBe(1);
  });

  it("View / Proj scopes set TraversalState.view / .proj for nested leaves", () => {
    const v = AVal.constant(Trafo3d.translation(new V3d(0, 0, -5)));
    const p = AVal.constant(Trafo3d.scaling(0.5));
    const tree = Sg.view(v, Sg.proj(p, leaf()));
    const got = collectLeaves(tree)[0]!;
    expect(AVal.force(got.state.view)).toBe(AVal.force(v));
    expect(AVal.force(got.state.proj)).toBe(AVal.force(p));
  });

  it("Sg.delay receives the accumulated state and produces a sub-tree", () => {
    let captured: { hit: boolean } = { hit: false };
    const tree = Sg.trafo(
      Sg.translate(new V3d(7, 0, 0)) as Trafo3d,
      Sg.delay(state => {
        captured.hit = true;
        // Confirm we see the accumulated model trafo here.
        const at = AVal.force(state.model).transform(V3d.zero);
        expect(at.x).toBeCloseTo(7, 6);
        return leaf();
      }),
    );
    expect(countLeaves(tree)).toBe(1);
    expect(captured.hit).toBe(true);
  });
});
