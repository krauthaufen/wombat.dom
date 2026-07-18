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

import { AVal, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";
import type { Effect } from "@aardworx/wombat.shader";
import type { IUniformProvider } from "@aardworx/wombat.rendering/core";
import {
  composeModel, composeTrafoValue, deriveAutoUniform,
  type TraversalState,
} from "./traversalState.js";
import type { TrafoValue } from "./sg.js";
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
  /** Pass-level injected uniforms (e.g. the OIT build's) — identical
   *  for every row of a pass BY CONSTRUCTION, hence plan state. A
   *  subtree rendered in N passes holds N plans; that's a few shared
   *  objects per scene, never per row. */
  readonly injected: IUniformProvider | undefined;
  readonly slots: ReadonlyMap<string, UniformSlot>;
}

const HOLE: (i: number) => UniformSlot = (index) => ({ kind: "hole", index });
const PARENT: UniformSlot = { kind: "parent" };

function computePlan(
  template: SceneTemplate,
  parent: TraversalState,
  effect: Effect,
  injected: IUniformProvider | undefined,
): TemplatePlan {
  const slots = new Map<string, UniformSlot>();
  for (const name of effectUniformNames(effect)) {
    const hole = template.uniformHoles.get(name);
    slots.set(name, hole !== undefined ? HOLE(hole) : PARENT);
  }
  return { template, effect, parent, injected, slots };
}

// Plans cached per parent state (weak) × (template, effect, injected).
const _plans = new WeakMap<TraversalState, Map<string, TemplatePlan>>();
// Stable small ids for injected providers (cache-key axis).
const _injectedIds = new WeakMap<object, number>();
let _nextInjectedId = 1;
function injectedIdOf(p: IUniformProvider | undefined): number {
  if (p === undefined) return 0;
  let id = _injectedIds.get(p);
  if (id === undefined) { id = _nextInjectedId++; _injectedIds.set(p, id); }
  return id;
}

export function getPlan(
  template: SceneTemplate,
  parent: TraversalState,
  effect: Effect,
  injected?: IUniformProvider,
): TemplatePlan {
  let byKey = _plans.get(parent);
  if (byKey === undefined) {
    byKey = new Map();
    _plans.set(parent, byKey);
  }
  const key = `${template.id}|${effect.id}|${injectedIdOf(injected)}`;
  let plan = byKey.get(key);
  if (plan === undefined) {
    plan = computePlan(template, parent, effect, injected);
    byKey.set(key, plan);
  }
  return plan;
}

/**
 * Per-row uniform provider: plan + this row's holes. The parent state
 * (an `IUniformProvider` itself) handles everything the template's own
 * scopes don't supply — shared across every row of the group.
 */
const IDENTITY_MODEL: aval<Trafo3d> = AVal.constant(Trafo3d.identity);

export class RowProvider implements IUniformProvider {
  /** Lazily reconstructed row model (see rowModel()). */
  private _model: aval<Trafo3d> | undefined;

  constructor(
    private readonly plan: TemplatePlan,
    private readonly holes: ReadonlyArray<unknown>,
  ) {}

  /**
   * The row's model aval, reconstructed from the template's Trafo
   * holes exactly as the traversal would have composed it:
   *   - spine WITHOUT Instanced: fold pre-trafos over the parent model;
   *   - spine WITH Instanced: `pushInstancing` resets the model to
   *     identity, so only post-Instanced trafos compose (over identity).
   * Only ever built when a trafo-family uniform is actually pulled —
   * the heap path (modelChain + recipes) never asks.
   */
  private rowModel(): aval<Trafo3d> {
    if (this._model !== undefined) return this._model;
    const t = this.plan.template;
    let m: aval<Trafo3d>;
    if (t.hasInstancing) {
      m = IDENTITY_MODEL;
      for (const i of t.postTrafoHoles) {
        m = composeModel(m, composeTrafoValue(this.holes[i] as TrafoValue));
      }
    } else {
      m = this.plan.parent.model;
      for (const i of t.preTrafoHoles) {
        m = composeModel(m, composeTrafoValue(this.holes[i] as TrafoValue));
      }
    }
    this._model = m;
    return m;
  }

  tryGet(name: string): aval<unknown> | undefined {
    // Row-scoped uniforms shadow for ANY name (not only effect-declared
    // ones — derived-mode rules and diagnostics pull arbitrary names).
    const hole = this.plan.template.uniformHoles.get(name);
    if (hole !== undefined) return this.holes[hole] as aval<unknown>;
    // parent uniform scopes (shared chain), then auto-derived over the
    // ROW's model — same code path the TraversalState uses, extracted
    // into `deriveAutoUniform` so the two cannot drift.
    const p = this.plan.parent;
    const scoped = p.uniforms.tryFind(name);
    if (scoped !== undefined) return scoped;
    // Pass-level injected uniforms sit BETWEEN scopes and autos —
    // exactly the classic provider's shadowing order (scopes win,
    // injected shadows the auto-derived names).
    if (this.plan.injected !== undefined) {
      const inj = this.plan.injected.tryGet(name);
      if (inj !== undefined) return inj as aval<unknown>;
    }
    return deriveAutoUniform(name, this.rowModel(), p.view, p.proj, p.viewport);
  }

  *names(): Iterable<string> {
    yield* this.plan.slots.keys();
  }
}

/** Convenience: provider for one staged row under a plan's parent. */
export function rowProvider(plan: TemplatePlan, staged: StagedNode): RowProvider {
  return new RowProvider(plan, staged.holes);
}
