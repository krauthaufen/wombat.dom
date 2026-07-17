// `withSgSource` (M3 front-end) — the Fable plugin wraps every `Sg.*`
// call site in `withSgSource(value, "File.fs:line")`. Locations stamp
// the PRODUCED node (ops are wrapped, innermost construct wins), never
// enter template keys, and flow into the efficiency report's per-site
// counts.

import { beforeEach, describe, expect, it } from "vitest";
import { AVal, HashMap } from "@aardworx/wombat.adaptive";
import { stage, type Effect } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import type { BufferView, DrawCall } from "@aardworx/wombat.rendering/core";
import { ElementType } from "@aardworx/wombat.rendering/core";

import {
  Sg, compileScene, sceneEfficiency, resetEfficiency,
  withSgSource, sgSourceOf,
} from "../src/scene/index.js";
import { resetTemplates, stageNode } from "../src/scene/template.js";

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
function eff(): Effect {
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

beforeEach(() => { resetEfficiency(); resetTemplates(); });

describe("withSgSource", () => {
  it("stamps nodes; wraps ops so the produced node is stamped", () => {
    const n = withSgSource(leaf(), "A.fs:1");
    expect(sgSourceOf(n)).toBe("A.fs:1");

    const child = leaf();
    const op = withSgSource(Sg.onTap(() => {}), "A.fs:2");
    const produced = op(child);
    expect(sgSourceOf(produced)).toBe("A.fs:2");
    // the pre-existing child keeps its own (lack of) stamp
    expect(sgSourceOf(child)).toBeUndefined();
  });

  it("first stamp wins — an identity op never overwrites the inner site", () => {
    const inner = withSgSource(leaf(), "Inner.fs:5");
    const identityOp = withSgSource(<T>(x: T): T => x, "Outer.fs:9");
    expect(sgSourceOf(identityOp(inner))).toBe("Inner.fs:5");
  });

  it("passes primitives through untouched and never throws", () => {
    expect(withSgSource(undefined, "X.fs:1")).toBeUndefined();
    expect(withSgSource(3, "X.fs:1")).toBe(3);
    expect(withSgSource("s", "X.fs:1")).toBe("s");
  });

  it("locations never enter template keys", () => {
    const a = stageNode(withSgSource(Sg.pickTag(1)(leaf()), "A.fs:1"));
    const b = stageNode(withSgSource(Sg.pickTag(2)(leaf()), "B.fs:99"));
    expect(a.template.id).toBe(b.template.id);
  });

  it("flows into the efficiency report's per-site counts", () => {
    // Each unordered child lowers to TWO leaves → multi-leaf-subtree
    // bail; the stamped location must show up against that reason.
    const twoLeaves = (loc: string): unknown =>
      withSgSource(Sg.group([leaf(), leaf()]), loc);
    const scene = Sg.shader(eff(), Sg.unordered([
      twoLeaves("App.fs:947"), twoLeaves("App.fs:947"), twoLeaves("App.fs:12"),
    ] as never));
    const cmds = compileScene(scene, {});
    const list = AVal.force(cmds.content);
    const render = list.toArray().find((c: { kind: string }) => c.kind === "Render") as { tree: { kind: string } };
    const forceTree = (t: { kind: string }): void => {
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
    };
    forceTree(render.tree);
    const b = sceneEfficiency().bails.find((x) => x.reason === "multi-leaf-subtree");
    expect(b).toBeDefined();
    expect(b!.sources).toEqual([
      { loc: "App.fs:947", count: 2 },
      { loc: "App.fs:12", count: 1 },
    ]);
  });
});
