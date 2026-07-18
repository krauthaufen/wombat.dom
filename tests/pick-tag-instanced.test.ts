// PickTag + ROOT-level handler + INSTANCED leaf — the collection-tap
// shape the annotation demo uses. The tag must reach the registry
// scope even when the handler lives on an ancestor scope and the leaf
// is an Sg.Instanced rewrite.
import { describe, expect, it } from "vitest";
import { AVal, HashMap } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d } from "@aardworx/wombat.base";
import type { BufferView, DrawCall } from "@aardworx/wombat.rendering/core";
import { ElementType, IBuffer } from "@aardworx/wombat.rendering/core";
import { stage } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import { Sg, compileScene, PickRegistry } from "../src/scene/index.js";

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
function eff() {
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

function forceTree(t: { kind: string }): void {
  const a = t as unknown as { children?: unknown; tree?: unknown };
  switch (t.kind) {
    case "UnorderedFromSet": case "OrderedFromList": {
      const content = AVal.force((a.children as { content: never }).content) as Iterable<{ kind: string }>;
      for (const c of content) forceTree(c);
      return;
    }
    case "Ordered": case "Unordered":
      for (const c of a.children as Array<{ kind: string }>) forceTree(c);
      return;
    case "Adaptive":
      forceTree(AVal.force(a.tree as never) as { kind: string });
      return;
    default: return;
  }
}

describe("PickTag + instanced leaf + root handler", () => {
  it("the tag reaches the registry scope", () => {
    const registry = new PickRegistry();
    const tagged = (id: number) =>
      Sg.uniform({ LineColor: AVal.constant(id) },
        Sg.pickTag(id)(
          Sg.trafo(Trafo3d.translation(new V3d(id, 0, 0)),
            Sg.instanced(
              { count: AVal.constant(2),
                attributes: HashMap.empty<string, BufferView>()
                  .add("SegA", view()).add("SegB", view()) })(
              leaf()))));
    const scene = Sg.shader(eff(),
      Sg.forcePixelPicking(true,
        Sg.onTap(() => {})(
          Sg.unordered([tagged(41), tagged(42)]))));
    const cmds = compileScene(scene, { picking: { registry } });
    const list = AVal.force(cmds.content);
    const render = list.toArray().find((c: { kind: string }) => c.kind === "Render") as { tree: { kind: string } };
    forceTree(render.tree);
    expect(registry.size()).toBe(2);
    const tags = new Set<unknown>();
    for (let id = 1; id <= 2; id++) {
      const scope = registry.lookup(id);
      if (scope !== undefined) tags.add(scope.tag);
    }
    expect(tags).toEqual(new Set([41, 42]));
  });
});

describe("bisect", () => {
  it("root handler + PLAIN leaf registers", () => {
    const registry = new PickRegistry();
    const scene = Sg.shader(eff(),
      Sg.onTap(() => {})(
        Sg.unordered([Sg.pickTag(7)(leaf())])));
    const cmds = compileScene(scene, { picking: { registry } });
    const list = AVal.force(cmds.content);
    const render = list.toArray().find((c: { kind: string }) => c.kind === "Render") as { tree: { kind: string } };
    forceTree(render.tree);
    expect(registry.size()).toBe(1);
    expect(registry.lookup(1)?.tag).toBe(7);
  });

  it("leaf handler + INSTANCED leaf registers", () => {
    const registry = new PickRegistry();
    const scene = Sg.shader(eff(),
      Sg.unordered([
        Sg.pickTag(9)(Sg.onTap(() => {})(
          Sg.instanced(
            { count: AVal.constant(2),
              attributes: HashMap.empty<string, BufferView>()
                .add("SegA", view()).add("SegB", view()) })(
            leaf()))),
      ]));
    const cmds = compileScene(scene, { picking: { registry } });
    const list = AVal.force(cmds.content);
    const render = list.toArray().find((c: { kind: string }) => c.kind === "Render") as { tree: { kind: string } };
    forceTree(render.tree);
    expect(registry.size()).toBe(1);
    expect(registry.lookup(1)?.tag).toBe(9);
  });
});
