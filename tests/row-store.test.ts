// Row-store admission: asserted plan-matching children become THIN
// rows in the group's RowSet; a SECOND shape in the same group lowers
// classically and the efficiency report says so (mixed-plan-group).

import { beforeEach, describe, expect, it } from "vitest";
import { AVal, HashMap } from "@aardworx/wombat.adaptive";
import { stage, type Effect } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import { IBuffer, ElementType } from "@aardworx/wombat.rendering/core";
import type { BufferView, DrawCall, RenderTree } from "@aardworx/wombat.rendering/core";
import {
  Sg, compileScene, sceneEfficiency, resetEfficiency, __setRowLowering,
} from "../src/scene/index.js";
import { resetTemplates } from "../src/scene/template.js";

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

function parts(tree: RenderTree): { rows: unknown[]; classic: RenderTree[] } {
  expect(tree.kind).toBe("Unordered");
  const kids = (tree as RenderTree & { kind: "Unordered" }).children;
  const rowsNode = kids.find((k) => k.kind === "Rows") as RenderTree & { kind: "Rows" };
  const setNode = kids.find((k) => k.kind === "UnorderedFromSet") as RenderTree & { kind: "UnorderedFromSet" };
  const rows = [...(AVal.force((rowsNode.set.rows as unknown as { content: never }).content) as Iterable<unknown>)];
  const classic = [...(AVal.force((setNode as unknown as { children: { content: never } }).children.content) as Iterable<RenderTree>)]
    .filter((t) => t.kind !== "Empty");
  return { rows, classic };
}

beforeEach(() => { resetTemplates(); resetEfficiency(); __setRowLowering(true); });

describe("row-store admission", () => {
  it("same-shape asserted children all land in the RowSet", () => {
    const child = (i: number) => Sg.uniform({ LineColor: AVal.init(i) }, leaf());
    const scene = Sg.shader(eff(), Sg.unordered([child(1), child(2), child(3)]));
    const cmds = compileScene(scene, {});
    const list = AVal.force(cmds.content);
    const render = list.toArray().find((c: { kind: string }) => c.kind === "Render") as { tree: RenderTree };
    const { rows, classic } = parts(render.tree);
    expect(rows.length).toBe(3);
    expect(classic.length).toBe(0);
  });

  it("a second shape lowers classically and is reported", () => {
    const a = (i: number) => Sg.uniform({ LineColor: AVal.init(i) }, leaf());
    const b = (i: number) => Sg.uniform({ OtherName: AVal.init(i) }, leaf()); // different template
    const scene = Sg.shader(eff(), Sg.unordered([a(1), a(2), b(3)]));
    const cmds = compileScene(scene, {});
    const list = AVal.force(cmds.content);
    const render = list.toArray().find((c: { kind: string }) => c.kind === "Render") as { tree: RenderTree };
    const { rows, classic } = parts(render.tree);
    expect(rows.length).toBe(2);
    expect(classic.length).toBe(1);
    const bail = sceneEfficiency().bails.find((x) => x.reason === "mixed-plan-group");
    expect(bail?.count).toBe(1);
  });
});
