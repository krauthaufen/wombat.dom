// `DefaultSurfaces` — bundled `Effect`s usable without writing
// shaders. Mirrors Aardvark's `DefaultSurfaces.trafo + …` family,
// scaled down to what the M9 hello-cube demo needs.
//
// Effects ship as raw shader source compiled at first use via
// `parseShader + stage`. This avoids requiring the wombat.shader
// vite plugin in consumer apps for built-in effects (the plugin
// is still the right path for app-defined inline effects).

import type { Effect } from "@aardworx/wombat.shader";
import { vertex, fragment } from "@aardworx/wombat.shader";
import { abs } from "@aardworx/wombat.shader/types";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { V3f, V4f, type V2f } from "@aardworx/wombat.base";


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
    const wp = uniform.ModelTrafo.mul(new V4f(v.Positions.xyz, 1.0));
    // Direction transforms — w=0 so the translation column drops out.
    const n4 = uniform.NormalMatrix.mul(new V4f(v.Normals.xyz, 0.0));
    const t4 = uniform.ModelTrafo.mul(new V4f(v.DiffuseColorUTangents.xyz, 0.0));
    const b4 = uniform.ModelTrafo.mul(new V4f(v.DiffuseColorVTangents.xyz, 0.0));
    return {
      // `gl_Position` is the rasterizer's built-in clip-space output —
      // separate concern from the vertex-attribute `Positions` semantic
      // even though they're the same logical concept. The
      // wombat.shader-vite plugin maps `gl_Position` to
      // `@builtin(position)`; everything else gets an inter-stage
      // varying location auto-assigned at WGSL emit time.
      gl_Position:             uniform.ViewProjTrafo.mul(wp),
      WorldPositions:          wp,
      Normals:                 n4.xyz,
      DiffuseColorUTangents:   t4.xyz,
      DiffuseColorVTangents:   b4.xyz,
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

let simpleLightingCache: Effect | undefined;

export function simpleLighting(): Effect {
  if (simpleLightingCache !== undefined) return simpleLightingCache;
  simpleLightingCache = fragment((v: {
    Normals:        V3f;
    Colors:         V4f;
    WorldPositions: V4f;
  }) => {
    const n = v.Normals.normalize();
    // World-space surface → light direction. WorldPositions is V4f
    // (homogeneous); .xyz drops the .w component.
    const wp = v.WorldPositions.xyz;
    const c = uniform.LightLocation.sub(wp).normalize();
    const ambient = 0.2;
    // `dot` returns plain number; use `abs()` from wombat.shader's
    // shipped intrinsic table (recognised by name in the
    // SHIPPED_INTRINSIC_NAMES set, mapped to WGSL `abs(...)`).
    const diffuse = abs(c.dot(n));
    const l = ambient + (1.0 - ambient) * diffuse;
    return {
      outColor: new V4f(v.Colors.xyz.mul(l), v.Colors.w),
    };
  });
  return simpleLightingCache;
}

// ---------------------------------------------------------------------------
// constantColor — port of Aardvark.Rendering's
// `DefaultSurfaces.constantColor` (`Effects/Default/Impl/ConstantColor.fs`).
//
// Parameterised fragment effect: emits the captured colour as the
// framebuffer's `outColor`, ignoring all per-fragment varyings.
// Demonstrates closure-capture support in the inline marker form —
// each call to `constantColor(c)` produces a fresh `Effect` with
// `c` baked in as a `ReadInput("Closure", ...)` IR hole, so two
// calls with different colours produce distinct effects.
// ---------------------------------------------------------------------------

export function constantColor(c: V4f): Effect {
  return fragment(() => ({ outColor: c }));
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
  trafo,
  simpleLighting,
  vertexColor,
  constantColor,
} as const;
