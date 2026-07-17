// Template plans (instance-tables step 1): the RowProvider over a
// staged subtree's holes must resolve uniform names to the IDENTICAL
// aval instances that the classic per-leaf lowering resolves — i.e.
// pushing the same Uniform scopes onto a TraversalState.

import { beforeEach, describe, expect, it } from "vitest";
import { AVal, HashMap } from "@aardworx/wombat.adaptive";
import type { aval } from "@aardworx/wombat.adaptive";
import type { Effect } from "@aardworx/wombat.shader";

import { Sg, TraversalState } from "../src/scene/index.js";
import { stageNode, resetTemplates } from "../src/scene/template.js";
import { getPlan, rowProvider } from "../src/scene/templatePlan.js";

/** Effect stand-in reading the given uniforms (see scene-template tests). */
function effectReading(uniforms: string[]): Effect {
  return {
    id: `eff-${uniforms.join("-")}`,
    stages: [{
      template: {
        types: [],
        values: [{
          kind: "Entry",
          entry: {
            name: "fake", stage: "vertex", inputs: [], outputs: [],
            body: {
              kind: "Sequential",
              body: uniforms.map((name) => ({
                kind: "Expr",
                value: { kind: "ReadInput", scope: "Uniform", name },
              })),
            },
          },
        }],
      },
      holes: {}, avalHoles: {}, id: "stage-fake",
    }],
  } as unknown as Effect;
}

beforeEach(() => resetTemplates());

describe("template plans — resolution parity with pushUniforms", () => {
  it("row holes shadow parent scopes exactly like nested Uniform pushes", () => {
    const eff = effectReading(["LineColor", "LineWidthPx", "GlobalTint"]);

    const tint = AVal.init("tint");
    const outerColor = AVal.init("outer");
    // outer scope provides GlobalTint and a LineColor that the
    // per-row scope must shadow
    const parent = TraversalState.empty.pushUniforms(
      HashMap.empty<string, aval<unknown>>()
        .add("GlobalTint", tint)
        .add("LineColor", outerColor),
    );

    const rowColor = AVal.init("row");
    const rowWidth = AVal.init(3);
    const node = Sg.uniform({ LineColor: rowColor, LineWidthPx: rowWidth }, Sg.empty);
    const staged = stageNode(node);

    // classic lowering: push the SAME bag onto the parent state
    const bag = (node as unknown as { bag: { entries: HashMap<string, aval<unknown>> } }).bag.entries;
    const classic = parent.pushUniforms(bag);

    const plan = getPlan(staged.template, parent, eff);
    const rp = rowProvider(plan, staged);

    for (const name of ["LineColor", "LineWidthPx", "GlobalTint"]) {
      expect(rp.tryGet(name)).toBe(classic.uniforms.tryFind(name));
    }
    // unknown names fall through to the parent (undefined here)
    expect(rp.tryGet("Nope")).toBe(classic.uniforms.tryFind("Nope"));
  });

  it("plans are cached per (parent, template, effect)", () => {
    const eff = effectReading(["X"]);
    const parent = TraversalState.empty;
    const s1 = stageNode(Sg.uniform({ X: AVal.init(1) }, Sg.empty));
    const s2 = stageNode(Sg.uniform({ X: AVal.init(2) }, Sg.empty));
    expect(s1.template.id).toBe(s2.template.id);
    const p1 = getPlan(s1.template, parent, eff);
    const p2 = getPlan(s2.template, parent, eff);
    expect(p1).toBe(p2);
    // rows differ only by holes
    expect(rowProvider(p1, s1).tryGet("X")).not.toBe(rowProvider(p2, s2).tryGet("X"));
  });

  it("inner scope wins over outer within one template", () => {
    const eff = effectReading(["A"]);
    const outerA = AVal.init("outer");
    const innerA = AVal.init("inner");
    const node = Sg.uniform({ A: outerA }, Sg.uniform({ A: innerA }, Sg.empty));
    const staged = stageNode(node);
    const plan = getPlan(staged.template, TraversalState.empty, eff);
    expect(rowProvider(plan, staged).tryGet("A")).toBe(innerA);
  });
});
