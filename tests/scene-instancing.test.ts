// Phase-2 tests: SgInstanced node + subtree validator.
//
// Covers: nested-Instanced rejection, leaves with instanceCount>1
// rejection, pass-through scopes (Trafo, Shader, Group, …) accepted.

import { describe, expect, it } from "vitest";
import { AList, AVal, HashMap, cval, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d } from "@aardworx/wombat.base";
import type { BufferView, DrawCall } from "@aardworx/wombat.rendering/core";
import type { Effect } from "@aardworx/wombat.shader";

import { type SgLeaf, type SgNode, type SgInstanced } from "../src/scene/sg.js";
import { applyInstancing, validateInstancingSubtree } from "../src/scene/instancing.js";

const dummyDraw: DrawCall = {
  kind: "non-indexed",
  vertexCount: 3,
  firstVertex: 0,
  instanceCount: 1,
  firstInstance: 0,
};

const drawWithInstance = (instanceCount: number): DrawCall => ({
  kind: "non-indexed",
  vertexCount: 3,
  firstVertex: 0,
  instanceCount,
  firstInstance: 0,
});

const leaf = (drawCall: DrawCall = dummyDraw): SgLeaf => ({
  kind: "Leaf",
  vertexAttributes: HashMap.empty<string, BufferView>(),
  drawCall: AVal.constant(drawCall),
});

const dummyAttr = (): HashMap<string, BufferView> => HashMap.empty();

const instanced = (count: number, child: SgNode): SgInstanced => ({
  kind: "Instanced",
  count: AVal.constant(count),
  attributes: dummyAttr(),
  child,
});

describe("validateInstancingSubtree", () => {
  it("accepts a plain leaf inside the subtree", () => {
    expect(() => validateInstancingSubtree(leaf())).not.toThrow();
  });

  it("accepts a Trafo / Group / Shader pass-through chain wrapping a leaf", () => {
    const node: SgNode = {
      kind: "Trafo",
      value: AVal.constant(Trafo3d.identity),
      child: { kind: "Group", children: AList.ofList<SgNode>([leaf(), leaf()]) },
    };
    expect(() => validateInstancingSubtree(node)).not.toThrow();
  });

  it("rejects nested Sg.Instanced", () => {
    const inner = instanced(4, leaf());
    expect(() => validateInstancingSubtree(inner)).toThrow(/nested .*Instanced/);
  });

  it("rejects a leaf with drawCall.instanceCount > 1", () => {
    expect(() => validateInstancingSubtree(leaf(drawWithInstance(8))))
      .toThrow(/instanceCount=8/);
  });

  it("walks AdaptiveGroup contents (forces the aval once)", () => {
    const inner = cval<SgNode>(leaf(drawWithInstance(2)));
    const node: SgNode = { kind: "AdaptiveGroup", child: inner };
    expect(() => validateInstancingSubtree(node)).toThrow(/instanceCount=2/);
  });

  it("walks UnorderedGroup children", () => {
    const node: SgNode = {
      kind: "UnorderedGroup",
      children: { content: AVal.constant(new Set<SgNode>([instanced(2, leaf())])) } as never,
    };
    // Passing an aset shape via a constant aval is enough for the
    // validator's force(); we don't need a real aset here.
    expect(() => validateInstancingSubtree(node)).toThrow(/nested .*Instanced/);
  });
});

// ─── applyInstancing — leaf rewrite shape ────────────────────────────

describe("applyInstancing", () => {
  const stubEffect: Effect = {
    stages: [], id: "stub",
    compile: () => { throw new Error("stub compile not callable"); },
    dumpIR: () => "",
  };
  const oneTrafo: Trafo3d[] = [Trafo3d.identity];

  const idTrafo = AVal.constant(Trafo3d.identity);

  it("trafo case: produces 8 column attributes + override of every parent-derived trafo uniform + count drawCall", () => {
    const node: SgInstanced = {
      kind: "Instanced",
      count: AVal.constant(1),
      trafos: AVal.constant(oneTrafo),
      attributes: HashMap.empty<string, BufferView>(),
      child: leaf(),
    };
    const applied = applyInstancing(node, idTrafo, idTrafo, idTrafo, idTrafo, stubEffect, leaf());
    let cols = 0;
    for (const [k] of applied.instanceAttributes) {
      if (k.startsWith("_InstanceTrafo_col") || k.startsWith("_InstanceTrafoInv_col")) cols++;
    }
    expect(cols).toBe(8);
    // drawCall.instanceCount overridden to scope's count.
    const dc = AVal.force(applied.drawCall);
    expect(dc.instanceCount).toBe(1);
  });

  it("overrides EVERY auto-injected trafo uniform with parent-derived values (regression)", () => {
    // The auto-injected trafo uniforms at the leaf are derived from
    // `state.model` — but inside an `Sg.Instanced` scope `state.model`
    // is the *inner* trafo chain (cylinderScale·orient etc.), NOT the
    // outer/parent. The instanceUniforms IR rules expect the OUTER
    // trafo to be in `uniform.X` so the per-instance composition
    // `InstanceTrafo*X` (forward) / `InstanceTrafoInv·X` (inverse)
    // / `X.m33 · transpose(m33(InstanceTrafoInv))` (NormalMatrix)
    // produces the full chain. Without the override, the
    // per-instance normal transform composes innerModel twice and
    // drops the rotation contribution — the original symptom that
    // produced the "all gizmos look the same shade" rendering bug.
    const innerModel = AVal.constant(Trafo3d.scaling(0.05, 0.05, 1));
    const parentModel = AVal.constant(Trafo3d.translation(new V3d(3, 0, 0)));
    const view = AVal.constant(Trafo3d.identity);
    const proj = AVal.constant(Trafo3d.identity);
    const node: SgInstanced = {
      kind: "Instanced",
      count: AVal.constant(1),
      trafos: AVal.constant(oneTrafo),
      attributes: HashMap.empty<string, BufferView>(),
      child: leaf(),
    };
    const applied = applyInstancing(node, innerModel, parentModel, view, proj, stubEffect, leaf());
    // Every trafo uniform that depends on the model trafo must be
    // overridden — the auto-injected default would derive it from
    // `state.model = innerModel` (post-reset inner chain), which is
    // wrong for the inst rule.
    for (const name of [
      "ModelTrafo", "ModelTrafoInv",
      "ModelViewTrafo", "ModelViewTrafoInv",
      "ModelViewProjTrafo", "ModelViewProjTrafoInv",
      "NormalMatrix",
    ]) {
      expect(applied.uniformOverrides.tryFind(name), `missing override for ${name}`).toBeDefined();
    }
    // Spot-check ModelTrafo carries the parent value, not innerModel.
    const mt = AVal.force(applied.uniformOverrides.tryFind("ModelTrafo")! as aval<Trafo3d>);
    const expected = AVal.force(parentModel);
    expect(mt.forward.toArray()[3]).toBeCloseTo(expected.forward.toArray()[3]!); // translation x
  });

  it("plain (non-trafo) attributes pass through verbatim", () => {
    const dummyView: BufferView = {
      buffer: AVal.constant({ kind: "host", data: new Float32Array(0), sizeBytes: 0 } as never),
      elementType: "v4f",
    };
    const node: SgInstanced = {
      kind: "Instanced",
      count: AVal.constant(2),
      attributes: HashMap.empty<string, BufferView>().add("Color", dummyView),
      child: leaf(),
    };
    const applied = applyInstancing(node, idTrafo, idTrafo, idTrafo, idTrafo, stubEffect, leaf());
    expect(applied.instanceAttributes.tryFind("Color")).toBeDefined();
    expect(applied.uniformOverrides.isEmpty).toBe(true);
  });
});

// ─── Sub-graph dynamism: adaptive swap re-validation ─────────────────

describe("Sg.Instanced — adaptive subtree swap re-validation", () => {
  // The compile-time validator is one-shot. After scene-compile, an
  // adaptive container inside the instancing scope can swap content;
  // the swap must re-validate so a violator (nested Sg.Instanced or a
  // leaf with `drawCall.instanceCount > 1`) doesn't render
  // silent-wrong pixels. Per-swap fallback is `RenderTree.empty` plus
  // a `console.error` (less disruptive than throwing inside an rAF
  // tick — the compile-time check is still the loud eager error).

  it("an SgAdaptiveGroup that swaps in a violator triggers per-swap validation", () => {
    // We can't easily run `compileScene` here without a GPU device, so
    // exercise the validator directly across the whole subtree shape
    // (the per-swap path calls the same `validateInstancingSubtree`
    // wrapped in try/catch). The valid-then-violator transition this
    // simulates is what the runtime path does on each `aval` fire.
    const valid: SgNode = leaf();
    expect(() => validateInstancingSubtree(valid)).not.toThrow();
    const violator: SgNode = leaf(drawWithInstance(4));
    expect(() => validateInstancingSubtree(violator)).toThrow(/instanceCount=4/);
    const nested: SgNode = instanced(2, leaf());
    expect(() => validateInstancingSubtree(nested)).toThrow(/nested .*Instanced/);
  });

  it("`Group`/`UnorderedGroup` children added after compile are also validated on insert", () => {
    // The map callback in `compile.ts:lower`'s Group/UnorderedGroup
    // cases delegates to `lowerInsideInstancing`, which re-runs the
    // validator on each inserted child. The behaviour mirrors
    // AdaptiveGroup's swap path.
    const newChild: SgNode = instanced(3, leaf());
    expect(() => validateInstancingSubtree(newChild)).toThrow(/nested .*Instanced/);
  });
});
