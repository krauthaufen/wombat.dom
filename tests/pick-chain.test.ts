// pick-chain — chooseChain decision matrix + compile sanity for the
// composed effect. The decision-matrix block builds eight minimal
// user effects covering all combinations of:
//
//   canCarryVsn := userVsn || (geomHas Normals)
//   userPi      := user effect produces PickPartIndex
//   userPvp     := user effect produces PickViewPosition
//   hasNormals  := geometry exposes Normals
//
// (Two of those are derived: canCarryVsn from userVsn / hasNormals;
// `injectVsn` from hasNormals && !userVsn.) We pin the chooser
// output for every cell.

import { describe, expect, it } from "vitest";
import { stage, type Effect } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { Tf32, Vec, type Type, type EntryParameter } from "@aardworx/wombat.shader/ir";

import {
  chooseChain, composePickChain,
  type PickChainChoice,
} from "../src/scene/picking/index.js";

const Tvec3f: Type = Vec(Tf32, 3);
const Tvec4f: Type = Vec(Tf32, 4);

// ---------------------------------------------------------------------------
// Helpers — build minimal user effects.
//
// Every effect has a vertex stage that copies `Positions` →
// `gl_Position`; the variant-flag inputs (`Normals`, `vColor`) are
// carried straight through to the fragment, where the requested
// outputs are written. By making the dataflow explicit and trivial
// the dep walker reports the intended deps without surprise.
// ---------------------------------------------------------------------------

interface UserEffectFlags {
  readonly produceVsn: boolean;
  readonly producePi:  boolean;
  readonly producePvp: boolean;
}

function buildUserEffect(flags: UserEffectFlags): Effect {
  const vsInputs = ["Positions: V4f"];
  const vsOutputs = ["gl_Position: V4f"];
  const fsInputs: string[] = [];
  const fsOutputs = ["Colors: V4f"];
  const fsBody: string[] = [];

  const vsBodyExtra: string[] = [];
  const vsReturnExtra: string[] = [];

  // To keep the geometry-attribute axes orthogonal in the matrix
  // we synthesise Vsn / Pvp from Positions rather than introducing
  // extra vertex attributes — otherwise "user produces Vsn, geom
  // has no Normals" is a contradiction (the user shader couldn't
  // run) and the dep walker correctly reports `canEffProduce` =
  // false.
  if (flags.produceVsn) {
    vsOutputs.push("vNormal: V3f");
    vsReturnExtra.push("vNormal: new V3f(input.Positions.x, input.Positions.y, input.Positions.z)");
    fsInputs.push("vNormal: V3f");
    fsOutputs.push("ViewSpaceNormal: V3f");
  }
  if (flags.producePi) {
    fsOutputs.push("PickPartIndex: f32");
  }
  if (flags.producePvp) {
    vsOutputs.push("vPvp: V3f");
    vsReturnExtra.push("vPvp: new V3f(input.Positions.x, input.Positions.y, input.Positions.z)");
    fsInputs.push("vPvp: V3f");
    fsOutputs.push("PickViewPosition: V3f");
  }

  const vsRet = ["gl_Position: input.Positions", ...vsReturnExtra].join(", ");
  const fsRetParts = ["Colors: new V4f(1.0, 1.0, 1.0, 1.0)"];
  if (flags.produceVsn) fsRetParts.push("ViewSpaceNormal: input.vNormal");
  if (flags.producePi)  fsRetParts.push("PickPartIndex: 7.0");
  if (flags.producePvp) fsRetParts.push("PickViewPosition: input.vPvp");

  const fsInputDecl = fsInputs.length === 0 ? "{}" : `{ ${fsInputs.join("; ")} }`;
  const source = `
    function vsMain(input: { ${vsInputs.join("; ")} }): { ${vsOutputs.join("; ")} } {
      ${vsBodyExtra.join("\n")}
      return { ${vsRet} };
    }
    function fsMain(input: ${fsInputDecl}): { ${fsOutputs.join("; ")} } {
      ${fsBody.join("\n")}
      return { ${fsRetParts.join(", ")} };
    }
  `;

  const vsInputDescs: EntryParameter[] = [
    { name: "Positions", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Location", value: 0 }] },
  ];

  let vsOutLoc = 0;
  const vsOutputDescs: EntryParameter[] = [
    { name: "gl_Position", type: Tvec4f, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
  ];
  if (flags.produceVsn) vsOutputDescs.push({ name: "vNormal", type: Tvec3f, semantic: "Normal", decorations: [{ kind: "Location", value: vsOutLoc++ }] });
  if (flags.producePvp) vsOutputDescs.push({ name: "vPvp",    type: Tvec3f, semantic: "ViewPos", decorations: [{ kind: "Location", value: vsOutLoc++ }] });

  let fsInLoc = 0;
  const fsInputDescs: EntryParameter[] = [];
  if (flags.produceVsn) fsInputDescs.push({ name: "vNormal", type: Tvec3f, semantic: "Normal", decorations: [{ kind: "Location", value: fsInLoc++ }] });
  if (flags.producePvp) fsInputDescs.push({ name: "vPvp",    type: Tvec3f, semantic: "ViewPos", decorations: [{ kind: "Location", value: fsInLoc++ }] });

  let fsOutLoc = 0;
  const fsOutputDescs: EntryParameter[] = [
    { name: "Colors", type: Tvec4f, semantic: "Color", decorations: [{ kind: "Location", value: fsOutLoc++ }] },
  ];
  if (flags.produceVsn) fsOutputDescs.push({ name: "ViewSpaceNormal",  type: Tvec3f, semantic: "ViewSpaceNormal",  decorations: [{ kind: "Location", value: fsOutLoc++ }] });
  if (flags.producePi)  fsOutputDescs.push({ name: "PickPartIndex",   type: Tf32,   semantic: "PickPartIndex",   decorations: [{ kind: "Location", value: fsOutLoc++ }] });
  if (flags.producePvp) fsOutputDescs.push({ name: "PickViewPosition", type: Tvec3f, semantic: "PickViewPosition", decorations: [{ kind: "Location", value: fsOutLoc++ }] });

  const entries: EntryRequest[] = [
    { name: "vsMain", stage: "vertex",   inputs: vsInputDescs, outputs: vsOutputDescs },
    { name: "fsMain", stage: "fragment", inputs: fsInputDescs, outputs: fsOutputDescs },
  ];
  return stage(parseShader({ source, entries }));
}

function geomFromSet(attrs: ReadonlySet<string>): (s: string) => boolean {
  return (s) => attrs.has(s);
}

// ---------------------------------------------------------------------------
// Decision matrix
// ---------------------------------------------------------------------------

interface MatrixCase {
  readonly name: string;
  readonly user: UserEffectFlags;
  readonly geom: ReadonlyArray<string>;
  readonly expected: PickChainChoice;
}

const baseGeom = ["Positions"];

const matrix: readonly MatrixCase[] = [
  // 1. user produces nothing, geom has no Normals
  //    canCarryVsn = false, userPi = false, userPvp = false
  //    -> FinalANoNormalNoPi, no inject
  {
    name: "no user, no Normals -> FinalANoNormalNoPi",
    user: { produceVsn: false, producePi: false, producePvp: false },
    geom: baseGeom,
    expected: { final: "FinalANoNormalNoPi", injectVsn: false },
  },
  // 2. user produces nothing, geom has Normals
  //    canCarryVsn = true (synthesise), userPi = false, userPvp = false
  //    -> FinalANoPi + inject
  {
    name: "no user, geom Normals -> FinalANoPi + inject",
    user: { produceVsn: false, producePi: false, producePvp: false },
    geom: [...baseGeom, "Normals"],
    expected: { final: "FinalANoPi", injectVsn: true },
  },
  // 3. user produces Pi, geom has no Normals
  //    -> FinalANoNormal, no inject
  {
    name: "user Pi, no Normals -> FinalANoNormal",
    user: { produceVsn: false, producePi: true, producePvp: false },
    geom: baseGeom,
    expected: { final: "FinalANoNormal", injectVsn: false },
  },
  // 4. user produces Pi, geom has Normals
  //    canCarryVsn = true, userPi = true -> FinalA + inject
  {
    name: "user Pi, geom Normals -> FinalA + inject",
    user: { produceVsn: false, producePi: true, producePvp: false },
    geom: [...baseGeom, "Normals"],
    expected: { final: "FinalA", injectVsn: true },
  },
  // 5. user produces Vsn (so canCarryVsn = true regardless), no Pi
  //    geom has no Normals -> no inject (user already produces vsn)
  //    -> FinalANoPi
  {
    name: "user Vsn, no Pi, no Normals -> FinalANoPi (no inject)",
    user: { produceVsn: true, producePi: false, producePvp: false },
    geom: baseGeom,
    expected: { final: "FinalANoPi", injectVsn: false },
  },
  // 6. user produces Vsn AND Pi, no Normals on geometry -> FinalA, no inject
  {
    name: "user Vsn + Pi -> FinalA (no inject)",
    user: { produceVsn: true, producePi: true, producePvp: false },
    geom: baseGeom,
    expected: { final: "FinalA", injectVsn: false },
  },
  // 7. user produces Vsn + geom Normals: canCarryVsn = true, no inject
  //    (userVsn already covers it)
  {
    name: "user Vsn + geom Normals -> FinalANoPi (no inject)",
    user: { produceVsn: true, producePi: false, producePvp: false },
    geom: [...baseGeom, "Normals"],
    expected: { final: "FinalANoPi", injectVsn: false },
  },
  // 8. user produces Pvp -> FinalB regardless of vsn / pi flags.
  //    Pick a non-trivial case: geom has Normals (so injectVsn flips
  //    on for the synthesise-but-not-userVsn axis even though we
  //    don't need vsn for Mode B).
  {
    name: "user Pvp + geom Normals -> FinalB + inject",
    user: { produceVsn: false, producePi: false, producePvp: true },
    geom: [...baseGeom, "Normals"],
    expected: { final: "FinalB", injectVsn: true },
  },
];

describe("chooseChain decision matrix", () => {
  for (const c of matrix) {
    it(c.name, () => {
      const eff = buildUserEffect(c.user);
      const choice = chooseChain(eff, geomFromSet(new Set(c.geom)));
      expect(choice).toEqual(c.expected);
    });
  }
});

// ---------------------------------------------------------------------------
// Sanity — compose + compile (no GPU)
// ---------------------------------------------------------------------------

describe("composePickChain compile sanity", () => {
  it("Mode A simple chain produces WGSL source", () => {
    const eff = buildUserEffect({ produceVsn: false, producePi: false, producePvp: false });
    const composed = composePickChain(eff, geomFromSet(new Set(["Positions"])));
    const compiled = composed.compile({ target: "wgsl" });
    expect(compiled.stages.length).toBeGreaterThan(0);
    for (const s of compiled.stages) {
      expect(s.source.length).toBeGreaterThan(0);
    }
  });

  it("Mode B chain produces WGSL source", () => {
    const eff = buildUserEffect({ produceVsn: false, producePi: false, producePvp: true });
    const composed = composePickChain(eff, geomFromSet(new Set(["Positions"])));
    const compiled = composed.compile({ target: "wgsl" });
    expect(compiled.stages.length).toBeGreaterThan(0);
    for (const s of compiled.stages) {
      expect(s.source.length).toBeGreaterThan(0);
    }
  });
});
