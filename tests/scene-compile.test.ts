// M3 — scene → RenderTree lowering. Verifies that:
//   - SgNode kinds map to the expected RenderTree variants;
//   - leaves carry RenderObjects whose uniforms include the
//     auto-injected Model/View/Proj and the user-provided overrides
//     (with user-wins on key conflict);
//   - active gating drops dead leaves;
//   - alist-/aset-/aval-shaped children produce the corresponding
//     delta-driven RenderTree node;
//   - effect resolution falls back through `defaultEffect`.
//
// No real GPU here — RenderTree is a pure data structure.

import { describe, expect, it } from "vitest";
import { AList, AVal, HashMap, cval, transact } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d } from "@aardworx/wombat.base";
import type {
  BufferView, DrawCall, Effect, IFramebuffer, RenderTree,
} from "@aardworx/wombat.rendering/core";
import { ElementType } from "@aardworx/wombat.rendering/core";

import {
  Sg, TraversalState, compileScene,
  type SgLeaf, type SgNode,
} from "../src/scene/index.js";

// ---------------------------------------------------------------------------
// Fakes — leaf + framebuffer + effect placeholders. None of the
// rendering machinery needs to inspect them at this layer.
// ---------------------------------------------------------------------------

const fbo: ReturnType<typeof AVal.constant<IFramebuffer>> = AVal.constant({} as IFramebuffer);

const dummyDraw: DrawCall = {
  kind: "non-indexed",
  vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0,
};

const dummyView: BufferView = {
  buffer: AVal.constant({ kind: "host", data: new Float32Array(0), sizeBytes: 0  }),
  offset: 0, stride: 12, elementType: ElementType.V3f,
};

function leaf(): SgLeaf {
  return Sg.leaf({
    vertexAttributes: HashMap.empty<string, BufferView>().add("Positions", dummyView),
    drawCall: AVal.constant(dummyDraw),
  });
}

const fakeEffect = { id: "fakeEffect" } as unknown as Effect;
const fakeEffect2 = { id: "fakeEffect2" } as unknown as Effect;

function singleRender(cmds: ReturnType<typeof compileScene>): RenderTree {
  const list = AVal.force(cmds.content);
  // ofList yields a single Render command unless `clear` is set.
  const r = list.toArray().find(c => c.kind === "Render");
  if (r === undefined || r.kind !== "Render") throw new Error("no render command");
  return r.tree;
}

function getLeafObject(tree: RenderTree): RenderTree extends infer _ ? RenderTree & { kind: "Leaf" } : never {
  if (tree.kind !== "Leaf") throw new Error(`expected Leaf, got ${tree.kind}`);
  return tree as RenderTree & { kind: "Leaf" };
}

// ---------------------------------------------------------------------------
// Lowering basics
// ---------------------------------------------------------------------------

describe("compileScene — lowering basics", () => {
  it("Empty SG → Empty RenderTree", () => {
    const cmds = compileScene(Sg.empty);
    expect(singleRender(cmds).kind).toBe("Empty");
  });

  it("Leaf without a Shader scope or defaultEffect drops silently", () => {
    const cmds = compileScene(leaf());
    expect(singleRender(cmds).kind).toBe("Empty");
  });

  it("Leaf with defaultEffect builds a RenderObject", () => {
    const cmds = compileScene(leaf(), { defaultEffect: fakeEffect });
    const lt = getLeafObject(singleRender(cmds));
    expect(lt.object.effect).toBe(fakeEffect);
    expect(lt.object.drawCall).toBeDefined();
  });

  it("Shader scope wins over defaultEffect", () => {
    const tree = Sg.shader(fakeEffect2, leaf());
    const cmds = compileScene(tree, { defaultEffect: fakeEffect });
    const lt = getLeafObject(singleRender(cmds));
    expect(lt.object.effect).toBe(fakeEffect2);
  });

  it("Group → OrderedFromList; UnorderedGroup → UnorderedFromSet", () => {
    const ordered = Sg.shader(fakeEffect, Sg.group([leaf(), leaf()]));
    expect(singleRender(compileScene(ordered)).kind).toBe("OrderedFromList");

    const unordered = Sg.shader(fakeEffect, Sg.unordered([leaf(), leaf()]));
    expect(singleRender(compileScene(unordered)).kind).toBe("UnorderedFromSet");
  });

  it("AdaptiveGroup → Adaptive(aval<RenderTree>)", () => {
    const c = cval<SgNode>(leaf());
    const tree = Sg.shader(fakeEffect, Sg.adaptive(c));
    const rt = singleRender(compileScene(tree));
    expect(rt.kind).toBe("Adaptive");
    if (rt.kind === "Adaptive") {
      const inner = AVal.force(rt.tree);
      expect(inner.kind).toBe("Leaf");
      // Swap → empty
      transact(() => { c.value = Sg.empty; });
      expect(AVal.force(rt.tree).kind).toBe("Empty");
    }
  });
});

// ---------------------------------------------------------------------------
// Auto-injected uniforms
// ---------------------------------------------------------------------------

describe("compileScene — auto-injected uniforms", () => {
  it("ModelTrafo / ViewTrafo / ProjTrafo / ViewProjTrafo are injected by default", () => {
    const cmds = compileScene(Sg.shader(fakeEffect, leaf()));
    const lt = getLeafObject(singleRender(cmds));
    for (const k of ["ModelTrafo", "ViewTrafo", "ProjTrafo", "ViewProjTrafo", "ViewportSize"]) {
      expect(lt.object.uniforms.tryFind(k)).toBeDefined();
    }
  });

  it("autoUniforms: false suppresses the injection", () => {
    const cmds = compileScene(Sg.shader(fakeEffect, leaf()), { autoUniforms: false });
    const lt = getLeafObject(singleRender(cmds));
    expect(lt.object.uniforms.tryFind("ModelTrafo")).toBeUndefined();
  });

  it("user-provided uniform with the same name overrides the auto-injected one", () => {
    const userTrafo = AVal.constant(Trafo3d.translation(new V3d(99, 0, 0)));
    const tree =
      Sg.shader(
        fakeEffect,
        Sg.uniform({ ModelTrafo: userTrafo }, leaf()),
      );
    const cmds = compileScene(tree);
    const lt = getLeafObject(singleRender(cmds));
    const got = lt.object.uniforms.tryFind("ModelTrafo")!;
    // After GPU-adapter: Trafo3d → M44f. Translation x is at row-
    // major index [3]. User wins over auto-injected identity.
    const m = AVal.force(got) as { _data: Float32Array };
    expect(m._data[3]).toBeCloseTo(99, 6);
  });

  it("ModelTrafo reflects accumulated Trafo scopes (adapted to M44f for the GPU)", () => {
    const tree =
      Sg.shader(
        fakeEffect,
        Sg.trafo(
          Sg.translate(new V3d(1, 0, 0)) as Trafo3d,
          Sg.trafo(Sg.scale(2) as Trafo3d, leaf()),
        ),
      );
    const cmds = compileScene(tree);
    const lt = getLeafObject(singleRender(cmds));
    const model = lt.object.uniforms.tryFind("ModelTrafo")!;
    const m = AVal.force(model) as { _data: Float32Array };
    // Row-major: m._data[0..3] is row 0; the translation column
    // sits at indices [3, 7, 11]. Outer trafo is translate(1,0,0)
    // applied last; inner is scale(2). Combined: scale then
    // translate; tx = 1, scale = 2.
    expect(m._data[3]).toBeCloseTo(1, 6);
    expect(m._data[0]).toBeCloseTo(2, 6);
    expect(m._data[5]).toBeCloseTo(2, 6);
    expect(m._data[10]).toBeCloseTo(2, 6);
  });
});

// ---------------------------------------------------------------------------
// Active gating
// ---------------------------------------------------------------------------

describe("compileScene — Active gating", () => {
  it("constantly-active leaf passes through as a Leaf", () => {
    const tree = Sg.shader(fakeEffect, Sg.active(AVal.constant(true), leaf()));
    expect(singleRender(compileScene(tree)).kind).toBe("Leaf");
  });

  it("constantly-inactive leaf becomes Empty", () => {
    const tree = Sg.shader(fakeEffect, Sg.active(AVal.constant(false), leaf()));
    expect(singleRender(compileScene(tree)).kind).toBe("Empty");
  });

  it("dynamic active wraps as Adaptive(aval<Leaf|Empty>)", () => {
    const a = cval(true);
    const tree = Sg.shader(fakeEffect, Sg.active(a, leaf()));
    const rt = singleRender(compileScene(tree));
    expect(rt.kind).toBe("Adaptive");
    if (rt.kind === "Adaptive") {
      expect(AVal.force(rt.tree).kind).toBe("Leaf");
      transact(() => { a.value = false; });
      expect(AVal.force(rt.tree).kind).toBe("Empty");
    }
  });
});

// ---------------------------------------------------------------------------
// Children flow + alist deltas
// ---------------------------------------------------------------------------

describe("compileScene — children", () => {
  it("Group with alist children produces a derived alist of RenderTrees", () => {
    const list = AList.ofList<SgNode>([leaf(), leaf()]);
    const tree = Sg.shader(fakeEffect, Sg.group(list));
    const rt = singleRender(compileScene(tree));
    expect(rt.kind).toBe("OrderedFromList");
    if (rt.kind === "OrderedFromList") {
      const arr = AVal.force(rt.children.content).toArray();
      expect(arr).toHaveLength(2);
      for (const child of arr) expect(child.kind).toBe("Leaf");
    }
  });
});

// ---------------------------------------------------------------------------
// Clear command emission
// ---------------------------------------------------------------------------

describe("compileScene — Clear", () => {
  it("emits a Clear command before Render when `clear` is set", () => {
    const cmds = compileScene(leaf(), {
      defaultEffect: fakeEffect,
      clear: { colors: HashMap.empty() } as never,
    });
    const arr = AVal.force(cmds.content).toArray();
    expect(arr).toHaveLength(2);
    expect(arr[0]!.kind).toBe("Clear");
    expect(arr[1]!.kind).toBe("Render");
  });

  it("omits Clear when not set", () => {
    const cmds = compileScene(leaf(), { defaultEffect: fakeEffect });
    const arr = AVal.force(cmds.content).toArray();
    expect(arr).toHaveLength(1);
    expect(arr[0]!.kind).toBe("Render");
  });
});

// ---------------------------------------------------------------------------
// TraversalState immutability after compile
// ---------------------------------------------------------------------------

describe("compileScene — View / Proj / Delay", () => {
  it("View / Proj scopes feed ViewTrafo / ProjTrafo auto-uniforms (M44f-adapted)", () => {
    const v = AVal.constant(Trafo3d.translation(new V3d(0, 0, -5)));
    const p = AVal.constant(Trafo3d.scaling(0.5));
    const tree = Sg.shader(fakeEffect, Sg.view(v, Sg.proj(p, leaf())));
    const lt = getLeafObject(singleRender(compileScene(tree)));
    const view = AVal.force(lt.object.uniforms.tryFind("ViewTrafo")!) as { _data: Float32Array };
    const proj = AVal.force(lt.object.uniforms.tryFind("ProjTrafo")!) as { _data: Float32Array };
    // Translate-z(-5): row-major [11] = -5
    expect(view._data[11]).toBeCloseTo(-5, 6);
    // Uniform scale 0.5: row-major [0,5,10] = 0.5
    expect(proj._data[0]).toBeCloseTo(0.5, 6);
  });

  it("Sg.camera sets both at once", () => {
    const v = AVal.constant(Trafo3d.translation(new V3d(1, 2, 3)));
    const p = AVal.constant(Trafo3d.scaling(2));
    const tree = Sg.shader(fakeEffect, Sg.camera(v, p, leaf()));
    const lt = getLeafObject(singleRender(compileScene(tree)));
    const view = AVal.force(lt.object.uniforms.tryFind("ViewTrafo")!) as { _data: Float32Array };
    const proj = AVal.force(lt.object.uniforms.tryFind("ProjTrafo")!) as { _data: Float32Array };
    expect(view._data[3]).toBeCloseTo(1, 6);   // tx
    expect(view._data[7]).toBeCloseTo(2, 6);   // ty
    expect(view._data[11]).toBeCloseTo(3, 6);  // tz
    expect(proj._data[0]).toBeCloseTo(2, 6);   // scale
  });

  it("Sg.delay produces a sub-tree from the accumulated state", () => {
    const tree = Sg.shader(
      fakeEffect,
      Sg.trafo(
        Sg.translate(new V3d(5, 0, 0)) as Trafo3d,
        Sg.delay(state => {
          // Build a leaf only when model.x > 0 — exercises the
          // "decide structure based on traversal" pattern.
          const at = AVal.force(state.model).transform(V3d.zero);
          return at.x > 0 ? leaf() : Sg.empty;
        }),
      ),
    );
    const rt = singleRender(compileScene(tree));
    expect(rt.kind).toBe("Leaf");
  });
});

describe("compileScene — non-mutating", () => {
  it("supplied initialState is not mutated", () => {
    const before = TraversalState.empty;
    const tree = Sg.shader(fakeEffect, Sg.trafo(Sg.translate(new V3d(7, 0, 0)) as Trafo3d, leaf()));
    compileScene(tree, { initialState: before });
    expect(AVal.force(before.model)).toBe(Trafo3d.identity);
    expect(before.shader).toBeUndefined();
  });
});
