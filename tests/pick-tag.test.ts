// `<Sg PickTag>` — an opaque app row key carried onto the pick scope
// and surfaced as `SceneEvent.pickTag`. Enables ONE collection-level
// handler instead of a closure per item.

import { describe, expect, it } from "vitest";
import { AVal, HashMap } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";
import type { BufferView, DrawCall } from "@aardworx/wombat.rendering/core";
import { ElementType } from "@aardworx/wombat.rendering/core";
import { stage } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";

import { Sg, compileScene, PickRegistry } from "../src/scene/index.js";
import { stageNode, resetTemplates } from "../src/scene/template.js";

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

describe("Sg.PickTag", () => {
  it("the tag lands on the registered pick scope (innermost wins)", () => {
    const registry = new PickRegistry();
    const scene = Sg.shader(eff(),
      Sg.pickTag("outer")(
        Sg.unordered([
          Sg.pickTag(41)(Sg.onTap(() => {})(leaf())),
          Sg.pickTag(42)(Sg.onTap(() => {})(leaf())),
          Sg.onTap(() => {})(leaf()), // inherits "outer"
        ])));
    const cmds = compileScene(scene, { picking: { registry } });
    // registration happens lazily when the group's children set is
    // pulled (as the renderer would) — force the tree once.
    const list = AVal.force(cmds.content);
    const render = list.toArray().find((c: { kind: string }) => c.kind === "Render") as { tree: unknown };
    const forceTree = (t: { kind: string }): void => {
      const a = t as unknown as {
        children?: unknown; tree?: { getValue?: unknown };
      };
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
    };
    forceTree(render.tree as { kind: string });
    expect(registry.size()).toBe(3);
    const tags = new Set<unknown>();
    for (let id = 1; id <= 3; id++) {
      const scope = registry.lookup(id);
      if (scope !== undefined) tags.add(scope.tag);
    }
    expect(tags).toEqual(new Set([41, 42, "outer"]));
  });

  it("staging holes the tag value — distinct tags share one template", () => {
    resetTemplates();
    const a = stageNode(Sg.pickTag(1)(leaf()));
    const b = stageNode(Sg.pickTag(2)(leaf()));
    expect(a.template.id).toBe(b.template.id);
    expect(a.holes).toContain(1);
    expect(b.holes).toContain(2);
  });
});
