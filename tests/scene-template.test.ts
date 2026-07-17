// Scene templates (M0/M1, docs/scene-templates.md): structural
// staging of SgNode spines into interned templates + holes, and
// template×effect uniform validation.

import { beforeEach, describe, expect, it, vi } from "vitest";
import { AVal, HashMap } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d } from "@aardworx/wombat.base";
import type { Effect } from "@aardworx/wombat.shader";
import type { BufferView, DrawCall } from "@aardworx/wombat.rendering/core";
import { ElementType } from "@aardworx/wombat.rendering/core";

import { Sg } from "../src/scene/index.js";
import {
  stageNode, templateStats, resetTemplates,
  effectUniformNames, validateTemplateEffect,
} from "../src/scene/template.js";

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

/** Mimics one `annotationLeaf` instance: per-instance avals + handler. */
function annotationish(i: number): ReturnType<typeof Sg.uniform> {
  const color = AVal.init({ r: i });
  const handler = (): void => { /* per-instance closure */ };
  return Sg.uniform(
    { LineColor: color, LineWidthPx: 3.0 + i },
    Sg.onTap(handler)(Sg.trafo(Trafo3d.translation(new V3d(i, 0, 0)), leaf())),
  );
}

/** Minimal Effect stand-in: `effectUniformNames`/`validateTemplateEffect`
 *  read only `id` and `stages[].template.values` (Uniform decls), and
 *  `stageNode` reads only `id`. */
function effectWithUniforms(): Effect {
  const mkStage = (uniforms: string[]): unknown => ({
    template: {
      types: [],
      values: [
        { kind: "Uniform", uniforms: uniforms.map((name) => ({ name })) },
        {
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
        },
      ],
    },
    holes: {}, avalHoles: {}, id: "stage-fake",
  });
  return {
    id: "eff-fake",
    stages: [mkStage(["ModelViewProjTrafo"]), mkStage(["LineColor"])],
  } as unknown as Effect;
}

beforeEach(() => resetTemplates());

describe("stageNode — structural interning", () => {
  it("same shape from different instances shares one template", () => {
    const a = stageNode(annotationish(1));
    const b = stageNode(annotationish(2));
    expect(a.template.id).toBe(b.template.id);
    expect(a.holes).not.toEqual(b.holes);
    const s = templateStats();
    expect(s.templates).toBe(1);
    expect(s.instances).toBe(2);
  });

  it("different uniform NAMES give different templates", () => {
    const a = stageNode(Sg.uniform({ Foo: AVal.init(1) }, leaf()));
    const b = stageNode(Sg.uniform({ Bar: AVal.init(1) }, leaf()));
    expect(a.template.id).not.toBe(b.template.id);
  });

  it("uniform VALUES are holes, never folded (identity semantics)", () => {
    const c1 = AVal.constant(5);
    const c2 = AVal.constant(5);
    const a = stageNode(Sg.uniform({ X: c1 }, leaf()));
    const b = stageNode(Sg.uniform({ X: c2 }, leaf()));
    expect(a.template.id).toBe(b.template.id);
    expect(a.holes).toContain(c1);
    expect(b.holes).toContain(c2);
  });

  it("constant primitive scope values fold into the key", () => {
    const a = stageNode(Sg.pickPriority(1)(leaf()));
    const b = stageNode(Sg.pickPriority(1)(leaf()));
    const c = stageNode(Sg.pickPriority(2)(leaf()));
    expect(a.template.id).toBe(b.template.id);
    expect(a.template.id).not.toBe(c.template.id);
    // folded → not a hole
    expect(a.template.holeCount).toBe(b.template.holeCount);
  });

  it("non-constant scope avals are holes", () => {
    const active1 = AVal.init(true);
    const active2 = AVal.init(true);
    const a = stageNode(Sg.active(active1, leaf()));
    const b = stageNode(Sg.active(active2, leaf()));
    expect(a.template.id).toBe(b.template.id);
    expect(a.holes).toContain(active1);
    expect(b.holes).toContain(active2);
  });

  it("constant groups inline children; leaf attr names are static", () => {
    const g1 = stageNode(Sg.group([leaf(), leaf()]));
    const g2 = stageNode(Sg.group([leaf(), leaf()]));
    const g3 = stageNode(Sg.group([leaf()]));
    expect(g1.template.id).toBe(g2.template.id);
    expect(g1.template.id).not.toBe(g3.template.id);
  });

  it("handler kinds are static, closures are holes", () => {
    const h1 = (): void => {};
    const h2 = (): void => {};
    const a = stageNode(Sg.onTap(h1)(leaf()));
    const b = stageNode(Sg.onTap(h2)(leaf()));
    expect(a.template.id).toBe(b.template.id);
    expect(a.holes).toContain(h1);
    const c = stageNode(Sg.onClick(h1)(leaf()));
    expect(c.template.id).not.toBe(a.template.id);
  });

  it("effect identity is static (same effect → same template)", () => {
    const e = effectWithUniforms();
    const a = stageNode(Sg.shader(e, leaf()));
    const b = stageNode(Sg.shader(e, leaf()));
    expect(a.template.id).toBe(b.template.id);
  });
});

describe("template × effect validation (M1)", () => {
  it("collects declared uniform names from the effect IR", () => {
    const e = effectWithUniforms();
    const names = effectUniformNames(e);
    expect(names.has("LineColor")).toBe(true);
    expect(names.has("ModelViewProjTrafo")).toBe(true);
  });

  it("warns on uniforms with no provider, once per (template, effect)", () => {
    const e = effectWithUniforms();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      // LineColor NOT provided anywhere on the spine.
      const staged = stageNode(Sg.shader(e, leaf()));
      const missing = validateTemplateEffect(staged, e);
      expect(missing).toEqual(["LineColor"]);
      expect(warn).toHaveBeenCalledTimes(1);
      // memoised: second call for the same pair is silent
      const staged2 = stageNode(Sg.shader(e, leaf()));
      expect(validateTemplateEffect(staged2, e)).toEqual([]);
      expect(warn).toHaveBeenCalledTimes(1);
    } finally {
      warn.mockRestore();
    }
  });

  it("uniform scopes on the spine satisfy the effect", () => {
    const e = effectWithUniforms();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const staged = stageNode(
        Sg.uniform({ LineColor: AVal.init(0) }, Sg.shader(e, leaf())),
      );
      expect(validateTemplateEffect(staged, e)).toEqual([]);
      expect(warn).not.toHaveBeenCalled();
      // auto-injected trafo names never count as missing
    } finally {
      warn.mockRestore();
    }
  });

  it("extraProvided allowlist suppresses app-injected names", () => {
    const e = effectWithUniforms();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const staged = stageNode(Sg.shader(e, leaf()));
      expect(validateTemplateEffect(staged, e, ["LineColor"])).toEqual([]);
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });
});
