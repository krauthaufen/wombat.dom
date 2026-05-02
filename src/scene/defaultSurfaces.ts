// `DefaultSurfaces` — bundled `Effect`s usable without writing
// shaders. Mirrors Aardvark's `DefaultSurfaces.trafo + …` family,
// scaled down to what the M9 hello-cube demo needs.
//
// Effects ship as raw shader source compiled at first use via
// `parseShader + stage`. This avoids requiring the wombat.shader
// vite plugin in consumer apps for built-in effects (the plugin
// is still the right path for app-defined inline effects).

import type { Effect } from "@aardworx/wombat.shader";
import { stage } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import {
  Mat, Tf32, Vec, type Type, type ValueDef,
  type Module,
} from "@aardworx/wombat.shader/ir";

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);
const TM44f:  Type = Mat(Tf32, 4, 4);

/**
 * Camera UBO declaration shared by all default surfaces. Members
 * match the names auto-injected by `compileScene`'s
 * `autoInjectedUniforms` (ModelTrafo / ViewTrafo / ProjTrafo).
 */
function cameraUniformBlock(): ValueDef {
  return {
    kind: "Uniform",
    uniforms: [
      { name: "ModelTrafo", type: TM44f, group: 0, slot: 0, buffer: "Camera" },
      { name: "ViewTrafo",  type: TM44f, group: 0, slot: 0, buffer: "Camera" },
      { name: "ProjTrafo",  type: TM44f, group: 0, slot: 0, buffer: "Camera" },
    ],
  };
}

// ---------------------------------------------------------------------------
// basic — vertex-color surface
// ---------------------------------------------------------------------------

let basicCache: Effect | undefined;

/**
 * Vertex-coloured pass-through with `ModelTrafo · ViewTrafo ·
 * ProjTrafo` applied per vertex. Expects:
 *
 *   - `a_position : V3f`
 *   - `a_color    : V3f`
 *
 * Outputs:
 *
 *   - `gl_Position : V4f` (clip-space)
 *   - `outColor    : V4f`
 *
 * Compiles once per process and caches the result.
 */
export function basic(): Effect {
  if (basicCache !== undefined) return basicCache;

  // a_color is V4f (RGBA) — matches Aardvark.Dom's
  // `DefaultSemantic.Colors` convention. Primitives feed this with a
  // single-value (stride-0) vertex buffer carrying one V4f read by
  // every vertex.
  const source = `
    declare const ModelTrafo: M44f;
    declare const ViewTrafo:  M44f;
    declare const ProjTrafo:  M44f;

    function vsMain(input: { a_position: V3f; a_color: V4f }): { gl_Position: V4f; v_color: V4f } {
      const world = ModelTrafo.mul(new V4f(input.a_position.x, input.a_position.y, input.a_position.z, 1.0));
      const view  = ViewTrafo.mul(world);
      const clip  = ProjTrafo.mul(view);
      return { gl_Position: clip, v_color: input.a_color };
    }

    function fsMain(input: { v_color: V4f }): { outColor: V4f } {
      return { outColor: input.v_color };
    }
  `;

  const entries: EntryRequest[] = [
    {
      name: "vsMain", stage: "vertex",
      inputs: [
        { name: "a_position", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "a_color",    type: Tvec4f, semantic: "Color",    decorations: [{ kind: "Location", value: 1 }] },
      ],
      outputs: [
        { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "v_color",     type: Tvec4f, semantic: "Color",    decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
    {
      name: "fsMain", stage: "fragment",
      outputs: [
        { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
  ];

  // Register the camera UBO members in externalTypes so the
  // frontend resolves bare `ModelTrafo` etc. references in the
  // source body.
  const externalTypes = new Map<string, Type>();
  externalTypes.set("ModelTrafo", TM44f);
  externalTypes.set("ViewTrafo",  TM44f);
  externalTypes.set("ProjTrafo",  TM44f);

  const camUBO = cameraUniformBlock();
  const parsed = parseShader({ source, entries, externalTypes });
  const merged: Module = { ...parsed, values: [camUBO, ...parsed.values] };
  basicCache = stage(merged);
  return basicCache;
}

// ---------------------------------------------------------------------------
// Combined namespace
// ---------------------------------------------------------------------------

export const DefaultSurfaces = {
  basic,
} as const;
