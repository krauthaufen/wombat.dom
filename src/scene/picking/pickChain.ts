// Pick chain selector + composer.
//
// Mirror of `Aardvark.Dom.SceneHandler.PickShader.chooseChain /
// composePickChain`. Driven entirely by wombat.shader's
// `effectDependencies` — no speculative `compile()` calls. The user
// effect's per-output dep map answers "can this effect produce
// `<sem>` given that the geometry exposes `<attrs>`?", and the
// answer picks one of the five fragment variants.
//
// Resolution rules:
//   * `PickViewPosition` produced by the user effect → mode B
//   * else → mode A; pick with-normal / no-normal x with-pi /
//     no-pi according to whether the user can produce
//     `ViewSpaceNormal` (or geometry has `Normals` for synthesis)
//     and whether the user can produce `PickPartIndex`.
//
// `injectVsn` is set iff we need to synthesise vsn ourselves —
// geometry has `Normals` but the user's effect doesn't write it.

import type { Effect } from "@aardworx/wombat.shader";
import { effect, effectDependencies } from "@aardworx/wombat.shader";

import {
  pickDepthBeforeEffect,
  pickFinalAEffect,
  pickFinalANoNormalEffect,
  pickFinalANoNormalNoPiEffect,
  pickFinalANoPiEffect,
  pickFinalBEffect,
  viewSpaceNormalVertexEffect,
} from "./pickShaders.js";

export type PickFinalTag =
  | "FinalA"
  | "FinalANoPi"
  | "FinalANoNormal"
  | "FinalANoNormalNoPi"
  | "FinalB";

export interface PickChainChoice {
  readonly final: PickFinalTag;
  readonly injectVsn: boolean;
}

// FShade renames `Position` to `Positions0` (or `Positions1`, ...)
// when a fragment shader reads `v.pos` directly — to disambiguate
// the fragment-input copy from the vertex-attribute output. wombat's
// frontend uses the bare name everywhere, but we keep the same
// canonicalisation in case the user feeds us a hand-built effect
// that mimics FShade's rename. Geometry only ever exposes
// `Positions`, so any `Positions[0-9]+` collapses to it.
function canonicalSemantic(s: string): string {
  if (!s.startsWith("Positions")) return s;
  const suffix = s.slice("Positions".length);
  if (suffix.length === 0) return s;
  for (let i = 0; i < suffix.length; i++) {
    const c = suffix.charCodeAt(i);
    if (c < 48 || c > 57) return s;
  }
  return "Positions";
}

export function chooseChain(
  eff: Effect,
  geomHas: (semantic: string) => boolean,
): PickChainChoice {
  const resolved = effectDependencies(eff);

  const canEffProduce = (sem: string): boolean => {
    const dep = resolved.get(sem);
    if (dep === undefined) return false;
    for (const name of dep.inputs.keys()) {
      if (!geomHas(canonicalSemantic(name))) return false;
    }
    return true;
  };

  const userVsn       = canEffProduce("ViewSpaceNormal");
  const userPvp       = canEffProduce("PickViewPosition");
  const userPi        = canEffProduce("PickPartIndex");
  const canSynthesise = geomHas("Normals");
  const canCarryVsn   = userVsn || canSynthesise;
  const needInjectVsn = canSynthesise && !userVsn;

  if (userPvp) {
    return { final: "FinalB", injectVsn: needInjectVsn };
  }

  let final: PickFinalTag;
  if (canCarryVsn && userPi)        final = "FinalA";
  else if (canCarryVsn && !userPi)  final = "FinalANoPi";
  else if (!canCarryVsn && userPi)  final = "FinalANoNormal";
  else                              final = "FinalANoNormalNoPi";

  return { final, injectVsn: needInjectVsn };
}

function finalEffect(tag: PickFinalTag): Effect {
  switch (tag) {
    case "FinalA":             return pickFinalAEffect();
    case "FinalANoPi":         return pickFinalANoPiEffect();
    case "FinalANoNormal":     return pickFinalANoNormalEffect();
    case "FinalANoNormalNoPi": return pickFinalANoNormalNoPiEffect();
    case "FinalB":             return pickFinalBEffect();
  }
}

/**
 * Build the full composed pick effect for a user effect + geometry
 * callback. Chain order:
 *
 *   (viewSpaceNormalVertex if injectVsn) +
 *   pickDepthBefore +
 *   userEffect +
 *   pickFinal<chosen variant>
 *
 * Composition uses `effect(...)` — the runtime's `composeStages`
 * pass handles the v+v / f+f fuse at compile time.
 */
export function composePickChain(
  eff: Effect,
  geomHas: (semantic: string) => boolean,
): Effect {
  const choice = chooseChain(eff, geomHas);
  const stages: Effect[] = [];
  if (choice.injectVsn) stages.push(viewSpaceNormalVertexEffect());
  stages.push(eff);
  stages.push(finalEffect(choice.final));
  void pickDepthBeforeEffect;
  return effect(...stages);
}
