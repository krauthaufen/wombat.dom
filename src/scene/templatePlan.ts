// Template plans — instance-tables step 1 (docs/instance-tables.md).
//
// A plan resolves, ONCE per (template, parent-state, effect), where
// every uniform the effect needs comes from: a HOLE of the template's
// own Uniform scopes (per-row values), or the shared parent context
// (scope chain + auto-derived). Rows then answer uniform requests via
// `RowProvider` — one tiny object over `(plan, holes)` — instead of a
// per-leaf provider/state graph.
//
// Resolution parity contract: for any staged subtree, a RowProvider
// over its holes must return the IDENTICAL aval instances that the
// classic lowering's provider composition (scope-chain union over the
// leaf's TraversalState) would return. `tests/template-plan.test.ts`
// pins this by construction against `pushUniforms`.

import type { aval } from "@aardworx/wombat.adaptive";
import type { Effect } from "@aardworx/wombat.shader";
import type { IUniformProvider } from "@aardworx/wombat.rendering/core";
import type { TraversalState } from "./traversalState.js";
import { effectUniformNames, type SceneTemplate, type StagedNode } from "./template.js";

export type UniformSlot =
  | { readonly kind: "hole"; readonly index: number }
  /** Resolved by the shared parent context (scope chain or auto-derived). */
  | { readonly kind: "parent" };

export interface TemplatePlan {
  readonly template: SceneTemplate;
  readonly effect: Effect;
  /** Shared parent context — the state at the group, NOT per row. */
  readonly parent: TraversalState;
  readonly slots: ReadonlyMap<string, UniformSlot>;
}

const HOLE: (i: number) => UniformSlot = (index) => ({ kind: "hole", index });
const PARENT: UniformSlot = { kind: "parent" };

function computePlan(
  template: SceneTemplate,
  parent: TraversalState,
  effect: Effect,
): TemplatePlan {
  const slots = new Map<string, UniformSlot>();
  for (const name of effectUniformNames(effect)) {
    const hole = template.uniformHoles.get(name);
    slots.set(name, hole !== undefined ? HOLE(hole) : PARENT);
  }
  return { template, effect, parent, slots };
}

// Plans cached per parent state (weak) × (template, effect).
const _plans = new WeakMap<TraversalState, Map<string, TemplatePlan>>();

export function getPlan(
  template: SceneTemplate,
  parent: TraversalState,
  effect: Effect,
): TemplatePlan {
  let byKey = _plans.get(parent);
  if (byKey === undefined) {
    byKey = new Map();
    _plans.set(parent, byKey);
  }
  const key = `${template.id}|${effect.id}`;
  let plan = byKey.get(key);
  if (plan === undefined) {
    plan = computePlan(template, parent, effect);
    byKey.set(key, plan);
  }
  return plan;
}

/**
 * Per-row uniform provider: plan + this row's holes. The parent state
 * (an `IUniformProvider` itself) handles everything the template's own
 * scopes don't supply — shared across every row of the group.
 */
export class RowProvider implements IUniformProvider {
  constructor(
    private readonly plan: TemplatePlan,
    private readonly holes: ReadonlyArray<unknown>,
  ) {}

  tryGet(name: string): aval<unknown> | undefined {
    const slot = this.plan.slots.get(name);
    if (slot !== undefined && slot.kind === "hole") {
      return this.holes[slot.index] as aval<unknown>;
    }
    return (this.plan.parent as unknown as IUniformProvider).tryGet(name);
  }

  *names(): Iterable<string> {
    yield* this.plan.slots.keys();
  }
}

/** Convenience: provider for one staged row under a plan's parent. */
export function rowProvider(plan: TemplatePlan, staged: StagedNode): RowProvider {
  return new RowProvider(plan, staged.holes);
}
