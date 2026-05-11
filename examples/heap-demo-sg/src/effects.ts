// Effect zoo for the naive-SG heap demo.
//
// All effects use Sg's standard uniform conventions:
//   - `ModelTrafo`, `ModelTrafoInv` (auto-injected by Sg.Trafo scopes)
//   - `ViewProjTrafo` (auto-derived from Sg.View + Sg.Proj)
//   - `LightLocation`               (default uniform in Sg)
// Custom uniforms `Time` and `Tint` are user-supplied via `Sg.uniform({...})`.
//
// Effects intentionally vary in vertex/fragment chain so the heap
// renderer has 5 distinct buckets to merge — the whole point is to
// show that a naive SG (one shader per RO, no manual sharing) gets
// collapsed by the heap+family path.

import { effect, vertex, fragment } from "@aardworx/wombat.shader";
import { abs, asin, atan2, sin, type Sampler2D, texture } from "@aardworx/wombat.shader/types";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { V2f, V3f, V4f } from "@aardworx/wombat.base";

declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope {
    readonly Time: number;
    readonly Tint: V4f;
    // Declared as a uniform; `Sg.instanced(...)` rewrites this read into
    // a per-instance vertex-attribute fetch at effect-compile time.
    readonly InstanceOffset: V3f;
  }
}

// Module-scope sampler declaration. The wombat.shader-vite plugin
// lowers `texture(DiffuseTex, uv)` against this `const` into the IR's
// Sampler binding; the Sg compile layer matches it by name to the
// `Uniform={{ DiffuseTex: aval<ITexture> }}` value the user supplies.
const DiffuseTex: Sampler2D = null as unknown as Sampler2D;

// ─── Vertex stages ─────────────────────────────────────────────────────

// Object → clip in a single stage (object→world via ModelTrafo, world→clip
// via ViewProjTrafo). Mirrors `DefaultSurfaces.trafo()`. Normals via
// row-vec mul against ModelTrafoInv = inv-transpose normal transform under
// wombat.shader's row/col-major convention.
const modelVS = vertex((v: {
  Positions: V4f;
  Normals:   V3f;
  Colors:    V4f;
}) => {
  const wp = uniform.ModelTrafo.mul(v.Positions);
  const n4 = new V4f(v.Normals.xyz, 0.0).mul(uniform.ModelTrafoInv);
  return {
    gl_Position:    uniform.ViewProjTrafo.mul(wp),
    WorldPositions: wp,
    Normals:        n4.xyz,
    Colors:         v.Colors,
  };
});

// Per-instance world-space offset. We read it as `uniform.InstanceOffset`;
// `Sg.instanced({ attributes: { InstanceOffset: ... }})` rewrites this
// to a per-instance vertex-attribute fetch.
const instanceOffsetVS = vertex((v: {
  WorldPositions: V4f;
  Normals:        V3f;
  Colors:         V4f;
}) => ({
  WorldPositions: new V4f(v.WorldPositions.xyz.add(uniform.InstanceOffset), v.WorldPositions.w),
  Normals:        v.Normals,
  Colors:         v.Colors,
}));

// Z-wobble in world space.
const wobbleVS = vertex((v: {
  WorldPositions: V4f;
  Normals:        V3f;
  Colors:         V4f;
}) => {
  const t = uniform.Time;
  const wob = sin(t.mul(0.002).add(v.WorldPositions.x)).mul(0.3);
  return {
    WorldPositions: new V4f(
      v.WorldPositions.x,
      v.WorldPositions.y,
      v.WorldPositions.z + wob,
      v.WorldPositions.w,
    ),
    Normals: v.Normals,
    Colors:  v.Colors,
  };
});

// World → clip — used after `wobbleVS` / `instanceOffsetVS` perturb
// the WorldPositions carrier and we need to redo the clip-space step.
const clipVS = vertex((v: {
  WorldPositions: V4f;
  Normals:        V3f;
  Colors:         V4f;
}) => ({
  gl_Position:    uniform.ViewProjTrafo.mul(v.WorldPositions),
  WorldPositions: v.WorldPositions,
  Normals:        v.Normals,
  Colors:         v.Colors,
}));

// Spherical UV mapping from the object-space normal. Primitives in
// this demo don't carry a `Uv` attribute, so we derive one
// per-vertex from `Normals` (which the primitive helpers DO
// supply). Equirectangular: u = atan2(n.z, n.x) / 2π + 0.5,
// v = asin(n.y) / π + 0.5. Looks reasonable on spheres / cylinders /
// cones / tetrahedra / octahedra and on boxes (it maps each face to
// a wedge of the texture). A real app would pass an explicit `Uv`
// attribute.
const ONE_OVER_TWO_PI = 1 / (2 * Math.PI);
const ONE_OVER_PI     = 1 / Math.PI;
const sphericalUvVS = vertex((v: {
  Positions:      V4f;
  WorldPositions: V4f;
  Normals:        V3f;
  Colors:         V4f;
}) => {
  const n = v.Normals.normalize();
  const u = atan2(n.z, n.x).mul(ONE_OVER_TWO_PI).add(0.5);
  const w = asin(n.y).mul(ONE_OVER_PI).add(0.5);
  return {
    WorldPositions: v.WorldPositions,
    Normals:        v.Normals,
    Colors:         v.Colors,
    Uv:             new V2f(u, w),
  };
});

// ─── Fragment stages ───────────────────────────────────────────────────

// World-space Lambert with the Sg-default `LightLocation` point light.
const lambertFS = fragment((v: {
  Normals:        V3f;
  Colors:         V4f;
  WorldPositions: V4f;
}) => {
  const n = v.Normals.normalize();
  const l = uniform.LightLocation.sub(v.WorldPositions.xyz).normalize();
  const ambient = 0.2;
  const diffuse = abs(l.dot(n));
  const k = ambient + (1.0 - ambient) * diffuse;
  return {
    Colors: new V4f(v.Colors.xyz.mul(k), v.Colors.w),
  };
});

// FS-side tint via a uniform.
const tintFS = fragment((v: { Colors: V4f }) => ({
  Colors: new V4f(v.Colors.xyz.mul(uniform.Tint.xyz), v.Colors.w),
}));

// FS-side time-driven pulse.
const pulseFS = fragment((v: { Colors: V4f }) => {
  const t = uniform.Time.mul(0.003);
  const k = sin(t).mul(0.4).add(0.6);
  return {
    Colors: new V4f(v.Colors.xyz.mul(k), v.Colors.w),
  };
});

// Sampled-texture fragment stage: lit base × texture. The Sg compile
// layer binds the `DiffuseTex` Uniform entry (aval<ITexture>) to the
// shader's module-scope sampler of the same name, supplying a default
// `linear / repeat` sampler. URL-deferred textures resolve to a
// placeholder checker until the bitmap finishes loading.
const texturedLitFS = fragment((v: {
  Normals:        V3f;
  Colors:         V4f;
  WorldPositions: V4f;
  Uv:             V2f;
}) => {
  const n = v.Normals.normalize();
  const l = uniform.LightLocation.sub(v.WorldPositions.xyz).normalize();
  const ambient = 0.25;
  const diffuse = abs(l.dot(n));
  const k = ambient + (1.0 - ambient) * diffuse;
  const tex = texture(DiffuseTex, v.Uv);
  return {
    Colors: new V4f(v.Colors.xyz.mul(tex.xyz).mul(k), v.Colors.w),
  };
});

// ─── Composed Effects ──────────────────────────────────────────────────

export const surface                  = effect(modelVS, lambertFS);
export const instancedSurface         = effect(modelVS, instanceOffsetVS, clipVS, lambertFS);
export const tintedSurface            = effect(modelVS, clipVS, lambertFS, tintFS);
export const pulsingSurface           = effect(modelVS, clipVS, lambertFS, pulseFS);
export const wobblingInstancedSurface = effect(
  modelVS, wobbleVS, instanceOffsetVS, clipVS, lambertFS,
);
export const texturedSurface          = effect(modelVS, sphericalUvVS, texturedLitFS);
export const texturedInstancedSurface = effect(
  modelVS, instanceOffsetVS, clipVS, sphericalUvVS, texturedLitFS,
);
