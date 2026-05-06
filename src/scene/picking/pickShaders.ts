// Pick-chain effect implementations.
//
// Five fragment "final" variants + one depth-seed fragment + one
// view-space-normal vertex producer. Mirrors
// `Aardvark.Dom.SceneHandler.PickShader` but expressed in
// inline-marker form (`vertex(...)` / `fragment(...)`) — the build
// plugin lifts each marker to an IR template at vite-transform time.

import type { Effect } from "@aardworx/wombat.shader";
import { effect, fragment, vertex } from "@aardworx/wombat.shader";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { abs, clamp, floor } from "@aardworx/wombat.shader/types";
import type { f32, i32, u32, FragmentBuiltinIn } from "@aardworx/wombat.shader/types";
import { V3f, V4f } from "@aardworx/wombat.base";
// Silence TS "imported but unused" — these only show up inside marker
// bodies, which the inline plugin replaces at build time.
void (null as unknown as f32 | i32 | u32 | FragmentBuiltinIn);

// `PickId` is bound per-render-object by the runtime; augment the
// shared UniformScope so `uniform.PickId` type-checks inside marker
// bodies. ModelViewTrafoInv is already in the core scope.
declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope {
    readonly PickId: u32;
  }
}

// ---------------------------------------------------------------------------
// Normal24 octahedron pack/unpack — TS helpers translated by the
// inline plugin into IR Function ValueDefs and called from each
// pickFinal* fragment via a typed `Call(FunctionRef)`.
// ---------------------------------------------------------------------------

function n24Decode(e: f32): V3f {
  const i = (e as number) as i32;
  const xi = (i >> 12) & 4095;
  const yi = i & 4095;
  const ex = (xi as f32) / 4095.0 * 2.0 - 1.0;
  const ey = (yi as f32) / 4095.0 * 2.0 - 1.0;
  const z0 = 1.0 - abs(ex) - abs(ey);
  const sx = ex >= 0.0 ? 1.0 : -1.0;
  const sy = ey >= 0.0 ? 1.0 : -1.0;
  const fx = z0 < 0.0 ? (1.0 - abs(ey)) * sx : ex;
  const fy = z0 < 0.0 ? (1.0 - abs(ex)) * sy : ey;
  const v = new V3f(fx, fy, z0);
  return v.normalize();
}
void n24Decode;

function n24Encode(v: V3f): f32 {
  const inv = 1.0 / (abs(v.x) + abs(v.y) + abs(v.z));
  const px0 = v.x * inv;
  const py0 = v.y * inv;
  const sx = px0 >= 0.0 ? 1.0 : -1.0;
  const sy = py0 >= 0.0 ? 1.0 : -1.0;
  const fx = v.z <= 0.0 ? (1.0 - abs(py0)) * sx : px0;
  const fy = v.z <= 0.0 ? (1.0 - abs(px0)) * sy : py0;
  const cx = clamp(fx, -1.0, 1.0);
  const cy = clamp(fy, -1.0, 1.0);
  const x0 = floor((cx * 0.5 + 0.5) * 4095.0) as i32;
  const y0 = floor((cy * 0.5 + 0.5) * 4095.0) as i32;
  const e = ((x0 << 12) | y0) as f32;
  return e;
}

// ---------------------------------------------------------------------------
// viewSpaceNormalVertex
//
// Reads `Normals` and writes `ViewSpaceNormal` — nothing else. Runs
// before the user's vertex effect; the actual clip-space transform is
// the user's responsibility, so `gl_Position` here is the unmodified
// `Positions` attribute (gets overwritten downstream).
// ---------------------------------------------------------------------------

let viewSpaceNormalVertexCache: Effect | undefined;

export function viewSpaceNormalVertexEffect(): Effect {
  if (viewSpaceNormalVertexCache !== undefined) return viewSpaceNormalVertexCache;
  const vs = vertex((input: { Positions: V4f; Normals: V3f }) => {
    const t = uniform.ModelViewTrafoInv.transpose();
    const v4 = new V4f(input.Normals.x, input.Normals.y, input.Normals.z, 0.0);
    const vn = t.mul(v4);
    const n = new V3f(vn.x, vn.y, vn.z);
    return { gl_Position: input.Positions, ViewSpaceNormal: n.normalize() };
  });
  viewSpaceNormalVertexCache = effect(vs);
  return viewSpaceNormalVertexCache;
}

// ---------------------------------------------------------------------------
// pickDepthBefore
//
// Writes `Depth = gl_FragCoord.z` so downstream pickFinal* sees the
// natural depth even when the user fragment never touches it. Reads
// `gl_FragCoord` via `FragmentBuiltinIn`.
// ---------------------------------------------------------------------------

let pickDepthBeforeCache: Effect | undefined;

export function pickDepthBeforeEffect(): Effect {
  if (pickDepthBeforeCache !== undefined) return pickDepthBeforeCache;
  const fs = fragment((_input: {}, b: FragmentBuiltinIn) => ({
    Depth: b.fragCoord.z,
  }));
  pickDepthBeforeCache = effect(fs);
  return pickDepthBeforeCache;
}

// ---------------------------------------------------------------------------
// pickFinalA — mode A, with normal, with PartIndex
// ---------------------------------------------------------------------------

let pickFinalACache: Effect | undefined;

export function pickFinalAEffect(): Effect {
  if (pickFinalACache !== undefined) return pickFinalACache;
  const fs = fragment((input: {
    outColor: V4f;
    ViewSpaceNormal: V3f;
    PickPartIndex: f32;
  }, b: FragmentBuiltinIn) => {
    const n24 = n24Encode(input.ViewSpaceNormal.normalize());
    const id = new V4f((uniform.PickId as number) as f32, n24, b.fragCoord.z, input.PickPartIndex);
    return { outColor: input.outColor, pickId: id, Depth: b.fragCoord.z };
  });
  pickFinalACache = effect(fs);
  return pickFinalACache;
}

// ---------------------------------------------------------------------------
// pickFinalANoPi — mode A, with normal, no PartIndex (slot 3 = 0)
// ---------------------------------------------------------------------------

let pickFinalANoPiCache: Effect | undefined;

export function pickFinalANoPiEffect(): Effect {
  if (pickFinalANoPiCache !== undefined) return pickFinalANoPiCache;
  const fs = fragment((input: {
    outColor: V4f;
    ViewSpaceNormal: V3f;
  }, b: FragmentBuiltinIn) => {
    const n24 = n24Encode(input.ViewSpaceNormal.normalize());
    const id = new V4f((uniform.PickId as number) as f32, n24, b.fragCoord.z, 0.0);
    return { outColor: input.outColor, pickId: id, Depth: b.fragCoord.z };
  });
  pickFinalANoPiCache = effect(fs);
  return pickFinalANoPiCache;
}

// ---------------------------------------------------------------------------
// pickFinalANoNormal — mode A, no normal, with PartIndex
// (slot 1 stays 0; downstream decoder reads it as a zero-length normal)
// ---------------------------------------------------------------------------

let pickFinalANoNormalCache: Effect | undefined;

export function pickFinalANoNormalEffect(): Effect {
  if (pickFinalANoNormalCache !== undefined) return pickFinalANoNormalCache;
  const fs = fragment((input: {
    outColor: V4f;
    PickPartIndex: f32;
  }, b: FragmentBuiltinIn) => {
    const id = new V4f((uniform.PickId as number) as f32, 0.0, b.fragCoord.z, input.PickPartIndex);
    return { outColor: input.outColor, pickId: id, Depth: b.fragCoord.z };
  });
  pickFinalANoNormalCache = effect(fs);
  return pickFinalANoNormalCache;
}

// ---------------------------------------------------------------------------
// pickFinalANoNormalNoPi — mode A, no normal, no PartIndex
// ---------------------------------------------------------------------------

let pickFinalANoNormalNoPiCache: Effect | undefined;

export function pickFinalANoNormalNoPiEffect(): Effect {
  if (pickFinalANoNormalNoPiCache !== undefined) return pickFinalANoNormalNoPiCache;
  const fs = fragment((input: {
    outColor: V4f;
  }, b: FragmentBuiltinIn) => {
    const id = new V4f((uniform.PickId as number) as f32, 0.0, b.fragCoord.z, 0.0);
    return { outColor: input.outColor, pickId: id, Depth: b.fragCoord.z };
  });
  pickFinalANoNormalNoPiCache = effect(fs);
  return pickFinalANoNormalNoPiCache;
}

// ---------------------------------------------------------------------------
// pickFinalB — mode B (real position). Tags mode via NEGATIVE PickId
// in slot 0; pvp components fill 1/2/3. No depth slot — pvp is the
// view-space position, so the host doesn't need to unproject. We do
// not write `Depth` here either; the user's effect (which produced
// `PickViewPosition`) is responsible for any custom depth handling.
// ---------------------------------------------------------------------------

let pickFinalBCache: Effect | undefined;

export function pickFinalBEffect(): Effect {
  if (pickFinalBCache !== undefined) return pickFinalBCache;
  const fs = fragment((input: {
    outColor: V4f;
    PickViewPosition: V3f;
  }) => {
    const id = new V4f(
      -((uniform.PickId as number) as f32),
      input.PickViewPosition.x, input.PickViewPosition.y, input.PickViewPosition.z,
    );
    return { outColor: input.outColor, pickId: id };
  });
  pickFinalBCache = effect(fs);
  return pickFinalBCache;
}
