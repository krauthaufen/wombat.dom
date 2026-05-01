import { parseShader } from "@aardworx/wombat.shader/frontend";
import { stage } from "@aardworx/wombat.shader";
const m = parseShader({
  source: `
    function fsMain(input: { v_color: V3f }, b: FragmentBuiltinIn): { outColor: V4f } {
      return { outColor: new V4f(input.v_color, b.fragCoord.z) };
    }
  `,
  entries: [{
    name: "fsMain", stage: "fragment",
    inputs: [{ name: "v_color", type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 3 }, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
    outputs: [{ name: "outColor", type: { kind: "Vector", element: { kind: "Float", width: 32 }, dim: 4 }, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] }],
  }],
});
console.log(stage(m).compile({target:"wgsl"}).stages[0].source);
