// Instance-tables step 3 — the formal parity gate: the SAME scene
// lowered with row-providers ON vs OFF must resolve every uniform a
// shader could pull to the same VALUES (trafo family included, both
// plain and Instanced spines), register the same number of pick
// scopes, and shadow parent scopes identically.

import { __basePassTree } from "../src/scene/compile.js";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AVal, HashMap, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d } from "@aardworx/wombat.base";
import { stage, type Effect } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import type { BufferView, DrawCall, RenderTree } from "@aardworx/wombat.rendering/core";
import { ElementType, materializeRow } from "@aardworx/wombat.rendering/core";

import { Sg, compileScene, PickRegistry, __setRowLowering } from "../src/scene/index.js";
import { resetTemplates } from "../src/scene/template.js";

const dummyDraw: DrawCall = {
  kind: "non-indexed",
  vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0,
};
function view(): BufferView {
  return {
    buffer: AVal.constant({ kind: "host", data: new Float32Array(9), sizeBytes: 36 }),
    elementType: ElementType.V3f,
  };
}
function leaf(): ReturnType<typeof Sg.leaf> {
  return Sg.leaf({
    vertexAttributes: HashMap.empty<string, BufferView>().add("Positions", view()),
    drawCall: AVal.constant(dummyDraw),
  });
}
/** Real minimal effect (the pick chain runs real IR passes over it). */
const Tvec4f: Type = Vec(Tf32, 4);
function buildUserEffect(): Effect {
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
      inputs: [
        { name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
      ],
      outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
      ],
    },
    {
      name: "fsMain", stage: "fragment",
      inputs: [],
      outputs: [
        { name: "Color", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
  ];
  return stage(parseShader({ source, entries }));
}

function child(i: number): ReturnType<typeof Sg.uniform> {
  const handler = (): void => { /* per-row */ };
  return Sg.uniform(
    { LineColor: AVal.init(i), LineWidthPx: AVal.init(i * 2) },
    Sg.onTap(handler)(
      Sg.trafo(Trafo3d.translation(new V3d(i, 0, 0)), leaf()),
    ),
  );
}

function leafObjects(tree: RenderTree): Array<RenderTree & { kind: "Leaf" }> {
  const out: Array<RenderTree & { kind: "Leaf" }> = [];
  const walk = (t: RenderTree): void => {
    if (t.kind === "Leaf") { out.push(t as RenderTree & { kind: "Leaf" }); return; }
    if (t.kind === "Ordered" || t.kind === "Unordered") {
      for (const c of (t as { children: readonly RenderTree[] }).children) walk(c);
      return;
    }
    if (t.kind === "Rows") {
      const set = (t as RenderTree & { kind: "Rows" }).set;
      const rows = AVal.force((set.rows as unknown as { content: never }).content) as Iterable<never>;
      for (const r of rows) out.push({ kind: "Leaf", object: materializeRow(set, r) } as never);
      return;
    }
    if (t.kind === "UnorderedFromSet") {
      const content = AVal.force((t as unknown as { children: { content: never } }).children.content) as Iterable<RenderTree>;
      for (const c of content) walk(c);
    }
  };
  walk(tree);
  return out;
}

function lowerScene(rows: boolean): {
  leaves: Array<RenderTree & { kind: "Leaf" }>;
  registrySize: number;
} {
  resetTemplates();
  __setRowLowering(rows);
  const eff = buildUserEffect();
  const registry = new PickRegistry();
  const scene = Sg.shader(eff,
    Sg.uniform({ GlobalTint: AVal.init(7), LineColor: AVal.init(-1) },
      Sg.unordered([child(1), child(2), child(3)])));
  const cmds = compileScene(scene, { picking: { registry } });
  const list = AVal.force(cmds.content);
  const r = list.toArray().find((c: { kind: string }) => c.kind === "Render") as { tree: RenderTree };
  return { leaves: leafObjects(__basePassTree(r.tree)), registrySize: registry.size() };
}

beforeEach(() => __setRowLowering(true));
afterEach(() => __setRowLowering(true));

describe("row lowering — formal parity vs classic", () => {
  it("pass-level injected uniforms resolve identically (OIT shape)", () => {
    const inject = HashMap.empty<string, aval<unknown>>()
      .add("OitPassFlag", AVal.constant(7) as aval<unknown>)
      // must NOT override a scope entry:
      .add("GlobalTint", AVal.constant(-99) as aval<unknown>);
    const lower = (rows: boolean) => {
      resetTemplates();
      __setRowLowering(rows);
      const eff = buildUserEffect();
      const registry = new PickRegistry();
      const scene = Sg.shader(eff,
        Sg.uniform({ GlobalTint: AVal.init(7) },
          Sg.unordered([child(1), child(2)])));
      const cmds = compileScene(scene, { picking: { registry }, injectUniforms: inject });
      const list = AVal.force(cmds.content);
      const r = list.toArray().find((c: { kind: string }) => c.kind === "Render") as { tree: RenderTree };
      return leafObjects(__basePassTree(r.tree));
    };
    const classic = lower(false);
    const rows = lower(true);
    expect(rows.length).toBe(2);
    for (let i = 0; i < rows.length; i++) {
      const cu = classic[i]!.object.uniforms;
      const ru = rows[i]!.object.uniforms;
      // injected name: both resolve to the injected value
      expect(AVal.force(ru.tryGet("OitPassFlag") as never)).toBe(7);
      expect(AVal.force(cu.tryGet("OitPassFlag") as never)).toBe(7);
      // scope shadowing: the scope value wins over injection on both paths
      expect(AVal.force(ru.tryGet("GlobalTint") as never)).toBe(7);
      expect(AVal.force(cu.tryGet("GlobalTint") as never)).toBe(7);
    }
  });

  it("uniform resolution, shadowing, trafo family and pick counts match", () => {
    const classic = lowerScene(false);
    const rows = lowerScene(true);
    expect(rows.leaves.length).toBe(classic.leaves.length);
    expect(rows.leaves.length).toBe(3);
    expect(rows.registrySize).toBe(classic.registrySize);

    // order leaves deterministically by their LineColor value
    const key = (l: RenderTree & { kind: "Leaf" }): number =>
      AVal.force(l.object.uniforms.tryGet("LineColor") as never) as number;
    const cs = [...classic.leaves].sort((a, b) => key(a) - key(b));
    const rs = [...rows.leaves].sort((a, b) => key(a) - key(b));

    for (let i = 0; i < cs.length; i++) {
      const cu = cs[i]!.object.uniforms;
      const ru = rs[i]!.object.uniforms;
      // row-scoped values shadow the parent scope
      expect(AVal.force(ru.tryGet("LineColor") as never)).toBe(AVal.force(cu.tryGet("LineColor") as never));
      expect(AVal.force(ru.tryGet("LineWidthPx") as never)).toBe(AVal.force(cu.tryGet("LineWidthPx") as never));
      // parent scope resolves through the row provider
      expect(AVal.force(ru.tryGet("GlobalTint") as never)).toBe(7);
      // trafo family: per-row model must match the classic derivation
      const cm = AVal.force(cu.tryGet("ModelTrafo") as never) as Trafo3d;
      const rm = AVal.force(ru.tryGet("ModelTrafo") as never) as Trafo3d;
      expect(rm.forward.equals(cm.forward)).toBe(true);
      const cmv = AVal.force(cu.tryGet("ModelViewTrafo") as never) as Trafo3d;
      const rmv = AVal.force(ru.tryGet("ModelViewTrafo") as never) as Trafo3d;
      expect(rmv.forward.equals(cmv.forward)).toBe(true);
      // unknown names: both undefined
      expect(ru.tryGet("Nope")).toBeUndefined();
      expect(cu.tryGet("Nope")).toBeUndefined();
    }
  });
});

// Scopes INSIDE a row with constant values (a camera, a depth test) are
// folded into the template's key. A row must still be lowered under them:
// its derived trafos read ITS view and projection, not its parent's, and
// its pipeline state is its own.
describe("row lowering — constant scopes inside a row", () => {
  const proj = Trafo3d.scaling(new V3d(0.5, 0.25, 1));
  const view = Trafo3d.translation(new V3d(0, 0, -7));
  const lower = (rows: boolean): Array<RenderTree & { kind: "Leaf" }> => {
    resetTemplates();
    __setRowLowering(rows);
    const eff = buildUserEffect();
    const registry = new PickRegistry();
    const row = (i: number) =>
      Sg.uniform({ LineColor: AVal.init(i) },
        Sg.view(AVal.constant(view),
          Sg.proj(AVal.constant(proj),
            Sg.depthTest("always")(
              Sg.trafo(Trafo3d.translation(new V3d(i, 0, 0)), leaf())))));
    const scene = Sg.shader(eff, Sg.unordered([row(1), row(2), row(3)]));
    const cmds = compileScene(scene, { picking: { registry } });
    const list = AVal.force(cmds.content);
    const r = list.toArray().find((c: { kind: string }) => c.kind === "Render") as { tree: RenderTree };
    return leafObjects(__basePassTree(r.tree));
  };
  const matrix = (leaf: RenderTree & { kind: "Leaf" }, name: string): number[] =>
    (AVal.force(leaf.object.uniforms.tryGet(name) as never) as Trafo3d).forward.toArray();
  const key = (l: RenderTree & { kind: "Leaf" }): number => AVal.force(l.object.uniforms.tryGet("LineColor") as never) as number;

  it("the derived trafos are the row's own camera's, as in the classic lowering", () => {
    const classic = lower(false).sort((a, b) => key(a) - key(b));
    const rows = lower(true).sort((a, b) => key(a) - key(b));
    expect(rows.length).toBe(3);
    for (let i = 0; i < 3; i++) {
      for (const name of ["ProjTrafo", "ViewTrafo", "ModelViewTrafo", "ModelViewProjTrafo"]) {
        expect(matrix(rows[i]!, name)).toEqual(matrix(classic[i]!, name));
      }
      // And they are that camera: model · view · proj.
      const expected = Trafo3d.translation(new V3d(i + 1, 0, 0)).mul(view).mul(proj).forward.toArray();
      expect(matrix(rows[i]!, "ModelViewProjTrafo")).toEqual(expected);
    }
  });

  it("the depth test is the row's", () => {
    for (const rows of [false, true]) {
      for (const l of lower(rows)) expect(AVal.force(l.object.pipelineState.depth!.compare)).toBe("always");
    }
  });
});

// The default surfaces light in VIEW space: `ViewLightLocation` is the
// scope's LightLocation taken there in doubles, and exactly the eye when
// no light is set — on both lowering paths.
describe("ViewLightLocation — the light in view space", () => {
  // A camera at geodetic magnitude, where an f32 world position holds ~0.5 m.
  const eye = new V3d(4_193_790.25, 1_163_590.75, 4_647_190.5);
  const view = Trafo3d.translation(new V3d(-eye.x, -eye.y, -eye.z));
  const lower = (rows: boolean, light: V3d | undefined): Array<RenderTree & { kind: "Leaf" }> => {
    resetTemplates();
    __setRowLowering(rows);
    const eff = buildUserEffect();
    const registry = new PickRegistry();
    const row = (i: number) => Sg.uniform({ LineColor: AVal.init(i) }, Sg.trafo(Trafo3d.translation(new V3d(i, 0, 0)), leaf()));
    const body = Sg.unordered([row(1), row(2)]);
    const lit = light === undefined ? body : Sg.uniform({ LightLocation: AVal.constant(light) }, body);
    const scene = Sg.shader(eff, Sg.view(AVal.constant(view), lit));
    const cmds = compileScene(scene, { picking: { registry } });
    const list = AVal.force(cmds.content);
    const r = list.toArray().find((c: { kind: string }) => c.kind === "Render") as { tree: RenderTree };
    return leafObjects(__basePassTree(r.tree));
  };
  const lightOf = (l: RenderTree & { kind: "Leaf" }): number[] => {
    const v = AVal.force(l.object.uniforms.tryGet("ViewLightLocation") as never) as { x: number; y: number; z: number };
    return [v.x, v.y, v.z];
  };

  it("is exactly the eye without a light (the headlight)", () => {
    for (const rows of [false, true]) {
      const leaves = lower(rows, undefined);
      expect(leaves.length).toBe(2);
      for (const l of leaves) expect(lightOf(l)).toEqual([0, 0, 0]);
    }
  });

  it("is the scope's light relative to the eye, to the millimetre at geodetic coordinates", () => {
    const light = new V3d(eye.x + 3.125, eye.y - 2.5, eye.z + 10.375);
    for (const rows of [false, true]) {
      for (const l of lower(rows, light)) {
        const [x, y, z] = lightOf(l);
        expect(x).toBeCloseTo(3.125, 3);
        expect(y).toBeCloseTo(-2.5, 3);
        expect(z).toBeCloseTo(10.375, 3);
      }
    }
  });
});
