// `DefaultSurfaces` — bundled `Effect`s usable without writing
// shaders. Mirrors Aardvark's `DefaultSurfaces.trafo + …` family,
// scaled down to what the M9 hello-cube demo needs.
//
// Effects ship as raw shader source compiled at first use via
// `parseShader + stage`. This avoids requiring the wombat.shader
// vite plugin in consumer apps for built-in effects (the plugin
// is still the right path for app-defined inline effects).

import type { Effect } from "@aardworx/wombat.shader";
import { stage, vertex, fragment, effect } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { V3f, V4f, type V2f, type M44f } from "@aardworx/wombat.base";
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
// headlight — Blinn-Phong with a single light coincident with the eye
// ---------------------------------------------------------------------------

let headlightCache: Effect | undefined;

/**
 * Vertex-coloured Blinn-Phong with a "camera headlight": the light
 * is fixed at the eye in view space, so `L = V = normalize(-viewPos)`
 * and the halfway vector collapses to `L`. Expects:
 *
 *   - `a_position : V3f`
 *   - `a_normal   : V3f`
 *   - `a_color    : V4f` (base / diffuse colour, supplied per primitive)
 *
 * Assumes the model+view trafo has uniform scale (translations +
 * rotations only) so the upper-3×3 transports normals without an
 * explicit normal matrix — true for the primitive layouts produced
 * by `Sg.translate` / `Sg.scale` with uniform scale factors.
 */
export function headlight(): Effect {
  if (headlightCache !== undefined) return headlightCache;

  const source = `
    declare const ModelTrafo: M44f;
    declare const ViewTrafo:  M44f;
    declare const ProjTrafo:  M44f;

    function vsMain(input: { a_position: V3f; a_normal: V3f; a_color: V4f })
      : { gl_Position: V4f; v_viewPos: V3f; v_viewNormal: V3f; v_color: V4f } {
      const world    = ModelTrafo.mul(new V4f(input.a_position.x, input.a_position.y, input.a_position.z, 1.0));
      const view4    = ViewTrafo.mul(world);
      const clip     = ProjTrafo.mul(view4);
      const nWorld4  = ModelTrafo.mul(new V4f(input.a_normal.x, input.a_normal.y, input.a_normal.z, 0.0));
      const nView4   = ViewTrafo.mul(new V4f(nWorld4.x, nWorld4.y, nWorld4.z, 0.0));
      return {
        gl_Position: clip,
        v_viewPos:   new V3f(view4.x, view4.y, view4.z),
        v_viewNormal: new V3f(nView4.x, nView4.y, nView4.z),
        v_color:     input.a_color,
      };
    }

    function fsMain(input: { v_viewPos: V3f; v_viewNormal: V3f; v_color: V4f }): { outColor: V4f } {
      const N = normalize(input.v_viewNormal);
      const L = normalize(new V3f(-input.v_viewPos.x, -input.v_viewPos.y, -input.v_viewPos.z));
      const ndotl = max(N.dot(L), 0.0);
      const ambient = 0.15;
      const diffuse = ndotl;
      const spec    = pow(ndotl, 48.0) * 0.4;
      const r = input.v_color.x * (ambient + diffuse) + spec;
      const g = input.v_color.y * (ambient + diffuse) + spec;
      const b = input.v_color.z * (ambient + diffuse) + spec;
      return { outColor: new V4f(r, g, b, input.v_color.w) };
    }
  `;

  const entries: EntryRequest[] = [
    {
      name: "vsMain", stage: "vertex",
      inputs: [
        { name: "a_position", type: Tvec3f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "a_normal",   type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
        { name: "a_color",    type: Tvec4f, semantic: "Color",    decorations: [{ kind: "Location", value: 2 }] },
      ],
      outputs: [
        { name: "gl_Position",  type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "v_viewPos",    type: Tvec3f, semantic: "ViewPosition", decorations: [{ kind: "Location", value: 0 }] },
        { name: "v_viewNormal", type: Tvec3f, semantic: "ViewNormal",   decorations: [{ kind: "Location", value: 1 }] },
        { name: "v_color",      type: Tvec4f, semantic: "Color",        decorations: [{ kind: "Location", value: 2 }] },
      ],
    },
    {
      name: "fsMain", stage: "fragment",
      outputs: [
        { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
  ];

  const externalTypes = new Map<string, Type>();
  externalTypes.set("ModelTrafo", TM44f);
  externalTypes.set("ViewTrafo",  TM44f);
  externalTypes.set("ProjTrafo",  TM44f);

  const camUBO = cameraUniformBlock();
  const parsed = parseShader({ source, entries, externalTypes });
  const merged: Module = { ...parsed, values: [camUBO, ...parsed.values] };
  headlightCache = stage(merged);
  return headlightCache;
}

// ---------------------------------------------------------------------------
// trafo — port of Aardvark.Rendering's `DefaultSurfaces.trafo`
// (`Effects/Default/Impl/Trafo.fs`). Vertex-only effect that
// transforms the standard Aardvark vertex bundle:
//
//     pos → ViewProjTrafo · ModelTrafo · pos             (clip)
//     wp  → ModelTrafo · pos                             (world position)
//     n   → NormalMatrix · n                             (transformed normal)
//     b/t → ModelTrafo · (b/t, 0)                        (transformed tangents)
//     c   → c                                            (pass-through)
//     tc  → tc                                           (pass-through)
//
// Vertex inputs use the canonical `DefaultSemantic` names. Outputs
// use the same names with `v_` prefix as varyings. Composes with
// any fragment effect that consumes those varyings (vertex-color
// pass-through, lighting, diffuse-texture, ...).
//
// LIMITATION (until phase 3 helper-extraction lands): all six
// outputs are emitted unconditionally, so a pipeline composed only
// with a vertex-colour FS still pays the normal/tangent matmuls.
// Once the auto-link + global-DCE pass arrives, the speculative
// outputs become free in pipelines that don't read them.
// ---------------------------------------------------------------------------

declare const ModelTrafo:    M44f;
declare const ViewProjTrafo: M44f;
declare const NormalMatrix:  M44f;

// Shader intrinsics — stub TypeScript types for the wombat.shader-vite
// inline-marker plugin's recognised intrinsic names. The plugin's own
// SHIPPED_INTRINSIC_NAMES table drives the actual IR translation;
// these declarations exist purely so tsc / vite-plugin-dts type-check
// the inline-marker bodies. (TODO: ship these from
// @aardworx/wombat.shader as a single ambient .d.ts so consumers
// don't have to re-declare per file.)
declare function normalize(v: V3f): V3f;
declare function abs(x: number): number;

let trafoCache: Effect | undefined;

export function trafo(): Effect {
  if (trafoCache !== undefined) return trafoCache;
  trafoCache = vertex((v: {
    Positions:               V3f;
    Normals:                 V3f;
    DiffuseColorUTangents:   V3f;
    DiffuseColorVTangents:   V3f;
    Colors:                  V4f;
    DiffuseColorCoordinates: V2f;
  }) => {
    const wp = ModelTrafo.mul(new V4f(v.Positions.x, v.Positions.y, v.Positions.z, 1.0));
    // Direction transforms — w=0 so the translation column drops out.
    const n4 = NormalMatrix.mul(new V4f(v.Normals.x,               v.Normals.y,               v.Normals.z,               0.0));
    const t4 = ModelTrafo.mul(  new V4f(v.DiffuseColorUTangents.x, v.DiffuseColorUTangents.y, v.DiffuseColorUTangents.z, 0.0));
    const b4 = ModelTrafo.mul(  new V4f(v.DiffuseColorVTangents.x, v.DiffuseColorVTangents.y, v.DiffuseColorVTangents.z, 0.0));
    return {
      // `gl_Position` is the rasterizer's built-in clip-space output —
      // separate concern from the vertex-attribute `Positions` semantic
      // even though they're the same logical concept. The
      // wombat.shader-vite plugin maps `gl_Position` to
      // `@builtin(position)`; everything else gets an inter-stage
      // varying location auto-assigned at WGSL emit time.
      gl_Position:             ViewProjTrafo.mul(wp),
      WorldPositions:          wp,
      Normals:                 new V3f(n4.x, n4.y, n4.z),
      DiffuseColorUTangents:   new V3f(t4.x, t4.y, t4.z),
      DiffuseColorVTangents:   new V3f(b4.x, b4.y, b4.z),
      Colors:                  v.Colors,
      DiffuseColorCoordinates: v.DiffuseColorCoordinates,
    };
  });
  return trafoCache;
}

// ---------------------------------------------------------------------------
// simpleLighting — port of Aardvark.Rendering's
// `DefaultSurfaces.simpleLighting`
// (`Effects/Default/Impl/SimpleLighting.fs`).
//
// Per-pixel Lambertian + ambient using a single point light. Reads
// `Normals`, `Colors`, `WorldPositions` from the inter-stage carrier
// (so it composes naturally after `trafo`); reads `LightLocation`
// from the auto-injected uniform set (default = (10,10,10), override
// via `<Sg Uniform={{LightLocation: cval(...)}}>`).
//
// Output is `outColor` — the framebuffer's `Colors` attachment.
// Once phase-3 8a (DefaultSemantic registry) lands the output field
// can be renamed `Colors` and the `outColor` alias dropped.
//
// LIMITATION: pre-phase-3 we don't have the helper-extraction +
// universal pass-through pipeline yet, so this effect ships as a
// stand-alone fragment that still requires a vertex side feeding
// it. Compose with `trafo` via `effect(trafo, simpleLighting)` —
// the wombat.shader `effect(...)` combinator already flattens
// stage lists in argument order.
// ---------------------------------------------------------------------------

declare const LightLocation: V3f;

let simpleLightingCache: Effect | undefined;

export function simpleLighting(): Effect {
  if (simpleLightingCache !== undefined) return simpleLightingCache;
  simpleLightingCache = fragment((v: {
    Normals:        V3f;
    Colors:         V4f;
    WorldPositions: V4f;
  }) => {
    const n = normalize(v.Normals);
    // World-space surface → light direction. WorldPositions is V4f
    // (homogeneous); we drop the .w component. Pre-swizzles, that's
    // a fresh V3f.
    const wp = new V3f(v.WorldPositions.x, v.WorldPositions.y, v.WorldPositions.z);
    const c = normalize(new V3f(
      LightLocation.x - wp.x,
      LightLocation.y - wp.y,
      LightLocation.z - wp.z,
    ));
    const ambient = 0.2;
    const diffuse = abs(c.dot(n));
    const l = ambient + (1.0 - ambient) * diffuse;
    return {
      outColor: new V4f(
        v.Colors.x * l,
        v.Colors.y * l,
        v.Colors.z * l,
        v.Colors.w,
      ),
    };
  });
  return simpleLightingCache;
}

// ---------------------------------------------------------------------------
// vertexColor — port of Aardvark.Rendering's
// `DefaultSurfaces.vertexColor` (`Effects/Default/Impl/VertexColor.fs`).
//
// One-line fragment effect: emits the interpolated `Colors` varying
// as the framebuffer's `outColor`. Composes with `trafo` —
// `effect(trafo, vertexColor)` is the post-phase-3 replacement for
// today's `DefaultSurfaces.basic`.
// ---------------------------------------------------------------------------

let vertexColorCache: Effect | undefined;

export function vertexColor(): Effect {
  if (vertexColorCache !== undefined) return vertexColorCache;
  vertexColorCache = fragment((v: { Colors: V4f }) => ({
    outColor: v.Colors,
  }));
  return vertexColorCache;
}

// ---------------------------------------------------------------------------
// Combined namespace
// ---------------------------------------------------------------------------

export const DefaultSurfaces = {
  basic,
  headlight,
  trafo,
  simpleLighting,
  vertexColor,
} as const;
