// pick-compile — integration of the pick chain + per-leaf registry
// into `compileScene`. Covers:
//   - leaf registration (one entry per leaf, distinct ids);
//   - the leaf's RenderObject effect is the composed pick chain
//     (more stages than the user effect, and still WGSL-compilable);
//   - the leaf's uniforms map carries `PickId`;
//   - opts.picking absent → no behavioural change.

import { describe, expect, it } from "vitest";
import { AVal, HashMap } from "@aardworx/wombat.adaptive";
import { stage, type Effect } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import type {
  BufferView, DrawCall, IFramebuffer, RenderTree,
} from "@aardworx/wombat.rendering/core";
import { ElementType } from "@aardworx/wombat.rendering/core";

import {
  Sg, compileScene, PickRegistry,
  type SgLeaf,
} from "../src/scene/index.js";

const Tvec4f: Type = Vec(Tf32, 4);

const fbo = AVal.constant({} as IFramebuffer);

const dummyDraw: DrawCall = {
  kind: "non-indexed",
  vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0,
};

const dummyView: BufferView = {
  buffer: AVal.constant({ kind: "host", data: new Float32Array(0), sizeBytes: 0 }),
  elementType: ElementType.V3f,
};

function leaf(): SgLeaf {
  return Sg.leaf({
    vertexAttributes: HashMap.empty<string, BufferView>().add("Positions", dummyView),
    drawCall: AVal.constant(dummyDraw),
  });
}

// Minimal real effect — passes Positions through, writes a Color
// fragment output. Compatible with the pick chain (no Vsn / Pi /
// Pvp; geom has no Normals → the chooser picks FinalANoNormalNoPi).
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

function singleRender(cmds: ReturnType<typeof compileScene>): RenderTree {
  const list = AVal.force(cmds.content);
  const r = list.toArray().find(c => c.kind === "Render");
  if (r === undefined || r.kind !== "Render") throw new Error("no render command");
  return r.tree;
}

function getLeaf(tree: RenderTree): RenderTree & { kind: "Leaf" } {
  if (tree.kind !== "Leaf") throw new Error(`expected Leaf, got ${tree.kind}`);
  return tree as RenderTree & { kind: "Leaf" };
}

describe("compileScene — picking integration", () => {
  it("single leaf with On registers exactly one PickScope and wraps the effect", () => {
    const userEff = buildUserEffect();
    const registry = new PickRegistry();
    const onClick = (_e: unknown): void => { /* noop test handler */ };

    const tree = Sg.shader(userEff, Sg.on({ bubble: { OnClick: onClick } }, leaf()));
    const cmds = compileScene(tree, { picking: { registry } });
    const lt = getLeaf(singleRender(cmds));

    // exactly one entry
    expect(registry.size()).toBe(1);

    // the registered handlers contain our onClick
    const id = AVal.force(lt.object.uniforms.tryFind("PickId")!) as number;
    const scope = registry.lookup(id);
    expect(scope).toBeDefined();
    expect(scope!.handlers.length).toBe(1);
    expect(scope!.handlers[0]!.handlers.bubble!.OnClick).toBe(onClick);

    // composed effect has more stages than the user effect, and
    // compiles to WGSL successfully.
    expect(lt.object.effect.stages.length).toBeGreaterThan(userEff.stages.length);
    const compiled = lt.object.effect.compile({ target: "wgsl" });
    expect(compiled.stages.length).toBeGreaterThan(0);
    for (const s of compiled.stages) expect(s.source.length).toBeGreaterThan(0);

    // PickId uniform is present and equals the registered id (>= 1).
    expect(id).toBeGreaterThanOrEqual(1);
  });

  it("two leaves under different On scopes get distinct pickIds and matching handlers", () => {
    const userEff = buildUserEffect();
    const registry = new PickRegistry();
    const onA = (_: unknown): void => {};
    const onB = (_: unknown): void => {};

    const tree = Sg.shader(
      userEff,
      Sg.group([
        Sg.on({ bubble: { OnClick: onA } }, leaf()),
        Sg.on({ bubble: { OnClick: onB } }, leaf()),
      ]),
    );
    const cmds = compileScene(tree, { picking: { registry } });
    const rt = singleRender(cmds);
    if (rt.kind !== "OrderedFromList") throw new Error("expected OrderedFromList");
    const children = AVal.force(rt.children.content).toArray();
    expect(children).toHaveLength(2);

    const id0 = AVal.force(getLeaf(children[0]!).object.uniforms.tryFind("PickId")!) as number;
    const id1 = AVal.force(getLeaf(children[1]!).object.uniforms.tryFind("PickId")!) as number;
    expect(id0).not.toBe(id1);
    expect(registry.size()).toBe(2);

    expect(registry.lookup(id0)!.handlers[0]!.handlers.bubble!.OnClick).toBe(onA);
    expect(registry.lookup(id1)!.handlers[0]!.handlers.bubble!.OnClick).toBe(onB);
  });

  it("no picking option → effect is the user's, no PickId uniform", () => {
    const userEff = buildUserEffect();
    const tree = Sg.shader(userEff, leaf());
    const cmds = compileScene(tree);
    const lt = getLeaf(singleRender(cmds));
    expect(lt.object.effect).toBe(userEff);
    expect(lt.object.uniforms.tryFind("PickId")).toBeUndefined();
  });
});
