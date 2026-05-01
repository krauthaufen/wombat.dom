// Pick-chain effect implementations.
//
// Five fragment "final" variants + one depth-seed fragment + one
// view-space-normal vertex producer. Mirrors
// `Aardvark.Dom.SceneHandler.PickShader` but expressed against
// wombat.shader's `parseShader + stage` entry. See `pickChain.ts`
// for the chooser that decides which variant to plug in.
//
// Slot layout (rgba32float pickId target, slot 1 alongside the
// regular colour target on slot 0):
//   Mode A (positive PickId in slot 0):
//     slot 0 : +f32 PickId
//     slot 1 : f32 of an n24 octahedron normal (0 = no normal)
//     slot 2 : NDC depth in [-1, 1]   (= 2 * gl_FragCoord.z - 1)
//     slot 3 : f32 of int part-index (< 2^24), or 0
//   Mode B (negative PickId in slot 0):
//     slot 0 : -f32 PickId
//     slot 1 : pvp.x   (plain f32)
//     slot 2 : pvp.y
//     slot 3 : pvp.z
//
// All four rgba32f channels are PLAIN floats — never bit-cast — so
// they survive MSAA resolve-average. Integer payloads are kept
// strictly under the 24-bit f32 mantissa.

import type { Effect } from "@aardworx/wombat.shader";
import { stage } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import type { EntryParameter } from "@aardworx/wombat.shader/ir";
import {
  Mat, Tf32, Tu32, Vec, type Type, type ValueDef, type Module,
} from "@aardworx/wombat.shader/ir";

import { n24ShaderHelpers } from "./normal24.js";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);
const TM44f:  Type = Mat(Tf32, 4, 4);

// ---------------------------------------------------------------------------
// Uniform blocks
// ---------------------------------------------------------------------------

function pickUniformBlock(): ValueDef {
  return {
    kind: "Uniform",
    uniforms: [
      { name: "PickId", type: Tu32, group: 0, slot: 1, buffer: "Pick" },
    ],
  };
}

function modelViewInvUniformBlock(): ValueDef {
  return {
    kind: "Uniform",
    uniforms: [
      { name: "ModelViewTrafoInv", type: TM44f, group: 0, slot: 0, buffer: "Camera" },
    ],
  };
}

// ---------------------------------------------------------------------------
// viewSpaceNormalVertex
//
// Pure normal pass: reads `Normals` and writes `ViewSpaceNormal` —
// nothing more, so PickPartIndex / PickViewPosition can never get
// pulled through this stage as vertex attributes. `Positions` is
// passed through as `gl_Position`; the actual clip-space transform
// is the caller's responsibility (we run before the user's vertex).
// ---------------------------------------------------------------------------

let viewSpaceNormalVertexCache: Effect | undefined;

export function viewSpaceNormalVertexEffect(): Effect {
  if (viewSpaceNormalVertexCache !== undefined) return viewSpaceNormalVertexCache;

  const source = `
    declare const ModelViewTrafoInv: M44f;

    function vsMain(input: { Positions: V4f; Normals: V3f }): { gl_Position: V4f; ViewSpaceNormal: V3f } {
      const t = ModelViewTrafoInv.transpose();
      const v4 = new V4f(input.Normals.x, input.Normals.y, input.Normals.z, 0.0);
      const vn = t.mul(v4);
      const n = new V3f(vn.x, vn.y, vn.z);
      return { gl_Position: input.Positions, ViewSpaceNormal: n.normalize() };
    }
  `;

  const entries: EntryRequest[] = [
    {
      name: "vsMain", stage: "vertex",
      inputs: [
        { name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
        { name: "Normals",   type: Tvec3f, semantic: "Normal",   decorations: [{ kind: "Location", value: 1 }] },
      ],
      outputs: [
        { name: "gl_Position",     type: Tvec4f, semantic: "Position",        decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "ViewSpaceNormal", type: Tvec3f, semantic: "ViewSpaceNormal", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
  ];

  const externalTypes = new Map<string, Type>();
  externalTypes.set("ModelViewTrafoInv", TM44f);

  const parsed = parseShader({ source, entries, externalTypes });
  const merged: Module = { ...parsed, values: [modelViewInvUniformBlock(), ...parsed.values] };
  viewSpaceNormalVertexCache = stage(merged);
  return viewSpaceNormalVertexCache;
}

// ---------------------------------------------------------------------------
// pickDepthBefore
//
// Writes `Depth = gl_FragCoord.z` so downstream pickFinal* sees the
// natural depth even when the user fragment never touches it. Reads
// `gl_FragCoord` via wombat.shader's `FragmentBuiltinIn` builtin —
// declaring it as a normal input would back-route Depth as a
// vertex attribute.
// ---------------------------------------------------------------------------

let pickDepthBeforeCache: Effect | undefined;

export function pickDepthBeforeEffect(): Effect {
  if (pickDepthBeforeCache !== undefined) return pickDepthBeforeCache;

  const source = `
    function fsMain(b: FragmentBuiltinIn): { Depth: f32 } {
      return { Depth: b.fragCoord.z };
    }
  `;

  const entries: EntryRequest[] = [
    {
      name: "fsMain", stage: "fragment",
      inputs: [],
      outputs: [
        { name: "Depth", type: Tf32, semantic: "Depth", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
  ];

  const parsed = parseShader({ source, entries });
  pickDepthBeforeCache = stage(parsed);
  return pickDepthBeforeCache;
}

// ---------------------------------------------------------------------------
// Shared helpers for the five pickFinal* fragments
// ---------------------------------------------------------------------------

const COLOR_OUT = {
  name: "outColor",
  type: Tvec4f,
  semantic: "Color",
  decorations: [{ kind: "Location", value: 0 } as const],
} as const;

const PICKID_OUT = {
  name: "pickId",
  type: Tvec4f,
  semantic: "PickId",
  decorations: [{ kind: "Location", value: 1 } as const],
} as const;

const DEPTH_OUT = {
  name: "Depth",
  type: Tf32,
  semantic: "Depth",
  decorations: [{ kind: "Builtin", value: "frag_depth" } as const],
} as const;

function buildFinal(
  source: string,
  entryName: string,
  inputs: readonly EntryParameter[],
  withDepth: boolean,
): Effect {
  const fullSource = `${n24ShaderHelpers()}\n${source}`;
  const outputs: EntryRequest["outputs"] = withDepth
    ? [COLOR_OUT, PICKID_OUT, DEPTH_OUT]
    : [COLOR_OUT, PICKID_OUT];
  const entries: EntryRequest[] = [
    {
      name: entryName, stage: "fragment",
      inputs: [...inputs],
      outputs,
    },
  ];
  const externalTypes = new Map<string, Type>();
  externalTypes.set("PickId", Tu32);
  const parsed = parseShader({ source: fullSource, entries, externalTypes });
  const merged: Module = { ...parsed, values: [pickUniformBlock(), ...parsed.values] };
  return stage(merged);
}

// ---------------------------------------------------------------------------
// pickFinalA — mode A, with normal, with PartIndex
// ---------------------------------------------------------------------------

let pickFinalACache: Effect | undefined;

export function pickFinalAEffect(): Effect {
  if (pickFinalACache !== undefined) return pickFinalACache;

  const source = `
    function fsMain(input: {
      outColor: V4f;
      ViewSpaceNormal: V3f;
      PickPartIndex: i32;
    }, b: FragmentBuiltinIn): { outColor: V4f; pickId: V4f; Depth: f32 } {
      const n24 = n24Encode(input.ViewSpaceNormal.normalize());
      const id = new V4f(PickId as f32, n24, b.fragCoord.z, input.PickPartIndex as f32);
      return { outColor: input.outColor, pickId: id, Depth: b.fragCoord.z };
    }
  `;

  pickFinalACache = buildFinal(source, "fsMain", [
    { name: "outColor", type: Tvec4f, semantic: "Color",           decorations: [{ kind: "Location", value: 0 }] },
    { name: "ViewSpaceNormal", type: Tvec3f, semantic: "ViewSpaceNormal", decorations: [{ kind: "Location", value: 1 }] },
    { name: "PickPartIndex",   type: Tf32,   semantic: "PickPartIndex",   decorations: [{ kind: "Location", value: 2 }] },
  ], true);
  return pickFinalACache;
}

// ---------------------------------------------------------------------------
// pickFinalANoPi — mode A, with normal, no PartIndex (slot 3 = 0)
// ---------------------------------------------------------------------------

let pickFinalANoPiCache: Effect | undefined;

export function pickFinalANoPiEffect(): Effect {
  if (pickFinalANoPiCache !== undefined) return pickFinalANoPiCache;

  const source = `
    function fsMain(input: {
      outColor: V4f;
      ViewSpaceNormal: V3f;
    }, b: FragmentBuiltinIn): { outColor: V4f; pickId: V4f; Depth: f32 } {
      const n24 = n24Encode(input.ViewSpaceNormal.normalize());
      const id = new V4f(PickId as f32, n24, b.fragCoord.z, 0.0);
      return { outColor: input.outColor, pickId: id, Depth: b.fragCoord.z };
    }
  `;

  pickFinalANoPiCache = buildFinal(source, "fsMain", [
    { name: "outColor", type: Tvec4f, semantic: "Color",           decorations: [{ kind: "Location", value: 0 }] },
    { name: "ViewSpaceNormal", type: Tvec3f, semantic: "ViewSpaceNormal", decorations: [{ kind: "Location", value: 1 }] },
  ], true);
  return pickFinalANoPiCache;
}

// ---------------------------------------------------------------------------
// pickFinalANoNormal — mode A, no normal, with PartIndex
// (slot 1 stays 0; downstream decoder reads it as a zero-length normal)
// ---------------------------------------------------------------------------

let pickFinalANoNormalCache: Effect | undefined;

export function pickFinalANoNormalEffect(): Effect {
  if (pickFinalANoNormalCache !== undefined) return pickFinalANoNormalCache;

  const source = `
    function fsMain(input: {
      outColor: V4f;
      PickPartIndex: i32;
    }, b: FragmentBuiltinIn): { outColor: V4f; pickId: V4f; Depth: f32 } {
      const id = new V4f(PickId as f32, 0.0, b.fragCoord.z, input.PickPartIndex as f32);
      return { outColor: input.outColor, pickId: id, Depth: b.fragCoord.z };
    }
  `;

  pickFinalANoNormalCache = buildFinal(source, "fsMain", [
    { name: "outColor", type: Tvec4f, semantic: "Color",         decorations: [{ kind: "Location", value: 0 }] },
    { name: "PickPartIndex", type: Tf32,   semantic: "PickPartIndex", decorations: [{ kind: "Location", value: 1 }] },
  ], true);
  return pickFinalANoNormalCache;
}

// ---------------------------------------------------------------------------
// pickFinalANoNormalNoPi — mode A, no normal, no PartIndex
// ---------------------------------------------------------------------------

let pickFinalANoNormalNoPiCache: Effect | undefined;

export function pickFinalANoNormalNoPiEffect(): Effect {
  if (pickFinalANoNormalNoPiCache !== undefined) return pickFinalANoNormalNoPiCache;

  const source = `
    function fsMain(input: {
      outColor: V4f;
    }, b: FragmentBuiltinIn): { outColor: V4f; pickId: V4f; Depth: f32 } {
      const id = new V4f(PickId as f32, 0.0, b.fragCoord.z, 0.0);
      return { outColor: input.outColor, pickId: id, Depth: b.fragCoord.z };
    }
  `;

  pickFinalANoNormalNoPiCache = buildFinal(source, "fsMain", [
    { name: "outColor", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: 0 }] },
  ], true);
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

  const source = `
    function fsMain(input: {
      outColor: V4f;
      PickViewPosition: V3f;
    }): { outColor: V4f; pickId: V4f } {
      const id = new V4f(-(PickId as f32), input.PickViewPosition.x, input.PickViewPosition.y, input.PickViewPosition.z);
      return { outColor: input.outColor, pickId: id };
    }
  `;

  pickFinalBCache = buildFinal(source, "fsMain", [
    { name: "outColor", type: Tvec4f, semantic: "Color",            decorations: [{ kind: "Location", value: 0 }] },
    { name: "PickViewPosition", type: Tvec3f, semantic: "PickViewPosition", decorations: [{ kind: "Location", value: 1 }] },
  ], false);
  return pickFinalBCache;
}
