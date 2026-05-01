import { parseShader } from "@aardworx/wombat.shader/frontend";
import { stage, effect } from "@aardworx/wombat.shader";

const userMod = parseShader({
  source: `
    function vsMain(input: { a_position: V3f; a_color: V3f }): { gl_Position: V4f; v_color: V3f } {
      return { gl_Position: new V4f(input.a_position, 1.0), v_color: input.a_color };
    }
    function fsMain(input: { v_color: V3f }): { outColor: V4f } {
      return { outColor: new V4f(input.v_color, 1.0) };
    }
  `,
  entries: [
    { name: "vsMain", stage: "vertex",
      inputs: [
        { name: "a_position", type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 3 }, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "a_color", type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 3 }, semantic: "Color", decorations: [{ kind: "Location", value: 1 }] },
      ],
      outputs: [
        { name: "gl_Position", type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 }, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "v_color", type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 3 }, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
    { name: "fsMain", stage: "fragment",
      inputs: [{ name: "v_color", type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 3 }, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "outColor", type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 }, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    },
  ],
});
const userEff = stage(userMod);

import { pickFinalANoNormalNoPiEffect } from "@aardworx/wombat.dom/scene";
const pickEff = pickFinalANoNormalNoPiEffect();

const composed = effect(userEff, pickEff);
const compiled = composed.compile({ target: "wgsl" });
console.log("--- WGSL ---\n" + compiled.stages.find(s=>s.stage==="fragment")?.source);
