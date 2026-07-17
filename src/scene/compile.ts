// compileScene — turn a `SgNode` tree into an `alist<Command>` that
// `wombat.rendering`'s `Runtime.compile` consumes. Walks the tree
// under a `TraversalState`, producing a `RenderTree` in which:
//
//   - every `SgLeaf` becomes a `RenderTree.leaf` carrying a
//     `RenderObject` built from the leaf + accumulated state;
//   - `SgGroup` / `SgUnorderedGroup` / `SgAdaptiveGroup` map onto
//     `RenderTree.orderedFromList` / `unorderedFromSet` / `adaptive`
//     so structural changes in the SG flow through as deltas;
//   - attribute-scope nodes (`Trafo`, `Shader`, `Uniform`, ...) push
//     their attribute onto the state and recurse into the child.
//
// The returned `alist<Command>` has a single `Render` command. M9+
// will add multi-output scenes (post-processing chains) that emit
// more than one command from a single SG by introducing
// `<Sg.RenderTo>`-style attribute scopes.
//
// **Trafo / matrix-order convention reminder.** All Trafo
// composition uses `Trafo3d.mul`, which is `a.mul(b)` =
// "apply a, then b" (i.e. `(a.mul(b)).forward = b.forward ·
// a.forward`). M44d.mul, in contrast, is plain matrix-on-vector
// math (`a · b · v` applies b first, a last). Don't mix the two
// conventions in this file or you'll trip yourself up.
//
// AVal.force policy (this file): `force` is permitted ONLY at sites
// that are either (a) provably structural one-shots run at
// compile-scene time, or (b) `isConstant`-guarded reads of avals
// known to never tick. Every surviving call site carries an
// `AVal.force OK:` (or `Why force here:`) comment naming the
// category. Anything else MUST go through reactive plumbing
// (AVal.custom / map / bind / RenderTree.adaptive). The
// `tests/scene-no-force.test.ts` audit allowlists each line by
// substring — adding a new force requires both an explanatory
// comment and an allowlist entry.

import {
  AList, ASet, AVal,
  HashMap,
  type alist, type aset, type aval,
} from "@aardworx/wombat.adaptive";
import { M44f, Trafo3d, V3d, V3f } from "@aardworx/wombat.base";
import { RenderTree, UniformProvider, AttributeProvider } from "@aardworx/wombat.rendering/core";
import { isDerivedRule } from "@aardworx/wombat.rendering/runtime";
import type {
  BlendState, BufferView,
  Command, ClearValues, DepthBiasState, DepthState,
  IAttributeProvider, IBuffer, IFramebuffer, ISampler, ITexture, IUniformProvider,
  PipelineState, PlainRasterizerState, RasterizerState, RenderObject,
  StencilState, Topology,
} from "@aardworx/wombat.rendering/core";
import type { Effect } from "@aardworx/wombat.shader";

import type {
  ColorMaskValue,
  ModeValue,
  SgLeaf, SgNode,
} from "./sg.js";
import { TraversalState, UniformScopeChain } from "./traversalState.js";
import { applyInstancing, validateInstancingSubtree } from "./instancing.js";
import { isITexture, resolveTextureAval } from "./textureResolver.js";
import { ISampler as ISamplerImpl } from "@aardworx/wombat.rendering/core";
import { composePickChainWithChoiceCached } from "./picking/pickChain.js";
import type { PickRegistry } from "./picking/registry.js";
import { stageNode, warnUnresolvedUniforms } from "./template.js";
import { getPlan, RowProvider } from "./templatePlan.js";

// ---------------------------------------------------------------------------
// Options + entry point
// ---------------------------------------------------------------------------

export interface CompileSceneOptions {
  /** Initial state — mostly for tests. Defaults to `TraversalState.empty`. */
  readonly initialState?: TraversalState;
  /**
   * Default effect used when no enclosing `<Sg Shader=...>` scope
   * has been entered. If both this and the scope shader are absent,
   * leaves are silently dropped (no draw possible without an
   * effect). M9 ships a default lit effect to plug in here.
   */
  readonly defaultEffect?: Effect;
  /** Optional clear pass before the render. */
  readonly clear?: ClearValues;
  /** Default rasterizer state. */
  /** Default rasterizer state (plain values). Wraps to avals at compile time. */
  readonly rasterizer?: PlainRasterizerState;
  /** Whether to inject `ModelTrafo` / `ViewTrafo` / `ProjTrafo` / `ViewProjTrafo` uniforms. Default `true`. */
  readonly autoUniforms?: boolean;
  /**
   * If present, every leaf is wrapped with the pick chain (see
   * `picking/pickChain.ts`) and registered with `picking.registry`.
   * Absent ⇒ zero pick overhead, identical output to before.
   */
  readonly picking?: PickingOptions;
  /**
   * Optional per-leaf filter by render-pass ordinal (`state.renderPass`,
   * set by `Sg.pass` / `Sg.transparent` / `Sg.opaque`). When it returns
   * false the leaf is dropped (`RenderTree.empty`). Used by the OIT
   * `transparencyTask` to lower the same scene into separate opaque and
   * transparent sub-trees. Absent ⇒ all leaves kept (unchanged).
   */
  readonly passFilter?: (renderPass: number) => boolean;
  /**
   * Optional transform applied to each kept leaf's final effect (after
   * any pick-chain composition). Used by the OIT path to append the
   * weighted-blend / A-buffer fragment writer onto transparent leaves
   * (mirrors aardvark's `composeSurface`). Absent ⇒ effect unchanged.
   */
  readonly composeEffect?: (effect: Effect) => Effect;
  /**
   * Optional transform applied to each kept leaf's derived
   * `PipelineState`. The OIT path uses this to force depth-write-off +
   * the accum/reveal blend modes on the transparent (WBOIT) pass without
   * the scene objects having to declare them. Absent ⇒ unchanged.
   */
  readonly pipelineOverride?: (ps: PipelineState, renderPass: number) => PipelineState;
  /**
   * Optional storage buffers merged into every kept leaf's
   * `storageBuffers` (leaf entries win on key conflict). The A-buffer
   * OIT path uses this to bind its node-pool / head / counter buffers
   * onto the transparent build pass without the scene declaring them.
   */
  readonly injectStorage?: HashMap<string, aval<IBuffer>>;
  /**
   * Optional uniforms merged into every kept leaf's uniform set (leaf
   * entries win on key conflict). The A-buffer path uses this to bind
   * the framebuffer width onto the transparent build pass.
   */
  readonly injectUniforms?: HashMap<string, aval<unknown>>;
}

export interface PickingOptions {
  readonly registry: PickRegistry;
  /**
   * Predicate over geometry semantics, called per leaf. When omitted
   * we fall back to a per-leaf set built from the leaf's vertex- /
   * instance-attribute keys (see `defaultGeomHas`).
   */
  readonly geomHas?: (semantic: string) => boolean;
  /**
   * Optional cache key for `geomHas`. When supplied alongside
   * `geomHas`, `composePickChainWithChoice` results are memoised
   * across leaves sharing the same `(userEffect, geomKey)`. Required
   * if you want caching on the user-supplied predicate path —
   * otherwise the call falls back to a sentinel key, defeating the
   * cache when the predicate semantically varies across leaves.
   */
  readonly geomKey?: string;
  /**
   * Notification hook called once per `registry.acquire()` performed
   * during `lower` (i.e. once per leaf with `pickPath !== "none"`).
   * The reactive structural cases (Group / UnorderedGroup /
   * AdaptiveGroup) override this hook on their recursive `lower`
   * call so the pickIds produced under each child are collected into
   * a cleanup list keyed by the child's mapped result; when the
   * adaptive collection evicts that result, the cleanup releases all
   * pickIds it acquired.
   */
  readonly onAcquire?: (pickId: number) => void;
}

/**
 * Compile a scene-graph tree into the `alist<Command>` that
 * `Runtime.compile` consumes. Returns one `Render` command (and
 * optionally one `Clear` ahead of it). The framebuffer is no longer
 * part of the Command — it's supplied at `task.run(framebuffer, token)`.
 */
export function compileScene(
  sg: SgNode,
  opts: CompileSceneOptions = {},
): alist<Command> {
  const base = opts.initialState ?? TraversalState.empty;
  // Derive the root `PipelineState` once and stash it on the state.
  // Descendants inherit it unchanged until a render-state scope clears
  // it (then the leaf path re-derives for that subtree). For a scene
  // with no render-state scopes — the common case, and the heap-demo —
  // every leaf shares this one object, so the heap path's bucket-key
  // cache computes the pipeline-state content key just once instead of
  // once per leaf, and `derivePipelineState` runs once instead of N×.
  const initial = base.withPipelineState(derivePipelineState(base, opts));
  // If the scene uses any `Sg.pass` scope, fall through to a flat
  // lowering that orders leaves by pass first, scene-graph order
  // second. Otherwise the structural lower path is identical to the
  // pre-Phase-1 behaviour.
  const tree = sceneUsesPassStatic(sg)
    ? lowerByPass(sg, initial, opts)
    : lower(sg, initial, opts);
  const cmds: Command[] = [];
  if (opts.clear !== undefined) {
    cmds.push({ kind: "Clear", values: opts.clear });
  }
  cmds.push({ kind: "Render", tree });
  return AList.ofList(cmds);
}

/**
 * Static one-shot scan: does this SG tree contain a `Pass` scope
 * anywhere? Runs ONCE at `compileScene` time to decide whether to
 * bucket leaves by pass at all. The answer is constant for a given
 * SG tree shape, so the `.force()` calls below are not on the live
 * render path — they're construction-boundary reads. Any subsequent
 * structural reshuffle of the SG flows through `lower`'s reactive
 * paths (Group / UnorderedGroup / AdaptiveGroup → orderedFromList /
 * unorderedFromSet / RenderTree.adaptive); the pass split itself is
 * not re-evaluated, but the contents of each bucket remain reactive.
 *
 * Why force here: see above — STATIC at compile-scene time, allowed.
 */
function sceneUsesPassStatic(node: SgNode): boolean {
  switch (node.kind) {
    case "Pass": return true;
    case "Empty": case "Leaf": return false;
    case "Group": {
      for (const c of node.children.content.force()) if (sceneUsesPassStatic(c)) return true;
      return false;
    }
    case "UnorderedGroup": {
      for (const c of node.children.content.force()) if (sceneUsesPassStatic(c)) return true;
      return false;
    }
    case "AdaptiveGroup": return sceneUsesPassStatic(node.child.force());
    case "Trafo": case "Shader": case "Uniform": case "BlendMode":
    case "Cursor": case "PickThrough": case "Intersectable":
    case "PixelSnapRadius": case "PickPriority": case "On": case "Active": case "View": case "Proj":
    case "DepthTest": case "DepthMask": case "DepthBias": case "DepthClamp":
    case "CullMode": case "FrontFace": case "FillMode":
    case "BlendConstant": case "ColorMask": case "StencilMode":
    case "VertexAttributes": case "InstanceAttributes": case "Index": case "Mode":
    case "NoEvents": case "ForcePixelPicking": case "CanFocus":
    case "PickContext":
      return sceneUsesPassStatic(node.child);
    case "Delay":
      try { return sceneUsesPassStatic(node.create(TraversalState.empty)); }
      catch { return false; }
    case "Instanced":
      return sceneUsesPassStatic(node.child);
  }
}

/**
 * Pass-grouped lowering — REACTIVE.
 *
 * The tree is lowered once PER DISTINCT PASS (ascending), each time
 * through the ordinary reactive `lower` path with a pass-scoped leaf
 * filter, and the resulting sub-trees are concatenated in pass order.
 * Ordering between passes therefore comes from the concatenation, while
 * everything INSIDE each pass keeps its normal reactivity — adaptive
 * groups, asets/alists, streamed tile leaves and `Active` flips all still
 * propagate.
 *
 * The previous implementation eagerly force-walked the whole SG once and
 * bucketed the lowered leaves. That froze every adaptive container in a
 * Pass-using scene: a streaming scene (tiles arriving in a clist) simply
 * stopped updating — and, because nothing marked any more, the on-demand
 * render loop stopped too (the "Sg.Pass makes the scene vanish/freeze"
 * bug). Lowering N times costs one extra traversal per pass (N is 2–3 in
 * practice) and buys back full reactivity.
 *
 * The SET of distinct passes is still discovered by a static scan
 * (`collectPassValues`): a pass value that only appears later, inside an
 * adaptive subtree that was empty at compile time, is not picked up.
 * Leaves in such a pass are dropped rather than mis-ordered.
 */
function lowerByPass(node: SgNode, state: TraversalState, opts: CompileSceneOptions): RenderTree {
  const passes = [...collectPassValues(node, state.renderPass, new Set<number>())].sort((a, b) => a - b);
  if (passes.length <= 1) return lower(node, state, opts);
  const userFilter = opts.passFilter;
  const trees: RenderTree[] = [];
  for (const p of passes) {
    const passOpts: CompileSceneOptions = {
      ...opts,
      passFilter: (rp: number) => rp === p && (userFilter === undefined || userFilter(rp)),
    };
    trees.push(lower(node, state, passOpts));
  }
  return RenderTree.ordered(...trees);
}

/**
 * Static scan for the distinct `renderPass` values a tree can produce.
 * Structural containers are forced ONCE here (construction-boundary read,
 * not on the live path) — only the SET of passes is snapshotted; the
 * contents of each pass stay fully reactive (see `lowerByPass`).
 */
function collectPassValues(node: SgNode, pass: number, out: Set<number>): Set<number> {
  switch (node.kind) {
    case "Empty": return out;
    case "Leaf": out.add(pass); return out;
    case "Pass": return collectPassValues(node.child, node.pass, out);
    case "Group":
    case "UnorderedGroup": {
      // Why force here: STATIC pass-set discovery at compileScene time (see
      // the doc above) — the per-pass CONTENTS stay reactive.
      for (const c of node.children.content.force()) collectPassValues(c, pass, out);
      return out;
    }
    // Why force here: same static pass-set scan.
    case "AdaptiveGroup": return collectPassValues(node.child.force(), pass, out);
    case "Delay": {
      try { return collectPassValues(node.create(TraversalState.empty), pass, out); }
      catch { return out; }
    }
    default: {
      // every remaining node kind is a single-child scope
      const child = (node as { child?: SgNode }).child;
      return child !== undefined ? collectPassValues(child, pass, out) : out;
    }
  }
}

function lower(
  node: SgNode,
  state: TraversalState,
  opts: CompileSceneOptions,
): RenderTree {
  switch (node.kind) {
    case "Empty":
      return RenderTree.empty;

    case "Leaf":
      return lowerLeaf(node, state, opts);

    case "Group": {
      // alist<SgNode> → alist<RenderTree>; outer is OrderedFromList.
      // `.map` fires per inserted child — re-validate the new
      // subtree if we're inside an `Sg.Instanced` scope.
      const children: alist<RenderTree> = node.children.map(
        child => lowerInsideInstancing(child, state, opts),
      );
      return RenderTree.orderedFromList(children);
    }

    case "UnorderedGroup": {
      // aset<SgNode> → aset<RenderTree>; outer is UnorderedFromSet.
      // `mapUse` (rather than `map`) ties the lowering's acquired
      // pickIds to the source entry's lifetime: when a child is
      // removed from the underlying aset, the cleanup callback
      // releases every `pickId` registered while lowering that
      // child. Without this, every `cset.remove` leaked pickIds in
      // `PickRegistry` (and a corresponding `_pickObjects` slot for
      // BVH-path scopes).
      const children: aset<RenderTree> = ASet.mapUse(
        (child: SgNode) => lowerRowOrClassic(child, state, opts),
        (wc) => wc.dispose(),
        node.children,
      ).map(wc => wc.tree);
      return RenderTree.unorderedFromSet(children);
    }

    case "AdaptiveGroup":
      // single-slot subtree swap: aval<SgNode> → aval<RenderTree>.
      // TODO(adaptive-cleanup): mirror UnorderedGroup's mapUse here —
      // an `aval.mapUse` (release-on-tick) variant is needed so a
      // swap of the inner SgNode releases the previous lowering's
      // pickIds.
      return RenderTree.adaptive(node.child.map(
        child => lowerInsideInstancing(child, state, opts),
      ));

    case "Trafo":
      return lower(node.child, state.pushTrafo(node.value), opts);

    case "Shader":
      return lower(node.child, state.pushShader(node.effect), opts);

    case "Uniform": {
      const entries =
        node.bag.kind === "Static"
          ? AVal.constant(node.bag.entries)
          : node.bag.entries.content;
      // Static path: known map at compile time → push once.
      // Dynamic path: amap.content gives us aval<HashMap>; lift.
      if (node.bag.kind === "Static") {
        return lower(node.child, state.pushUniforms(node.bag.entries), opts);
      }
      // Dynamic uniform set — re-evaluate the whole child each time
      // the uniform-key set changes. Per-key avals already update
      // independently inside the leaf.
      return RenderTree.adaptive(entries.map(map => lower(node.child, state.pushUniforms(map), opts)));
    }

    case "BlendMode":
      return lower(node.child, state.pushBlendMode(node.mode), opts);

    case "Cursor":
      return lower(node.child, state.pushCursor(node.cursor), opts);

    case "PickThrough":
      return lower(node.child, state.pushPickThrough(node.value), opts);

    case "Intersectable":
      return lower(node.child, state.pushIntersectable(node.intersectable), opts);

    case "PixelSnapRadius":
      return lower(node.child, state.pushPixelSnapRadius(node.radius), opts);

    case "PickPriority":
      return lower(node.child, state.pushPickPriority(node.value), opts);

    case "On":
      return lower(node.child, state.pushHandlers(node.handlers), opts);

    case "Active":
      return lower(node.child, state.pushActive(node.active), opts);

    case "View":
      return lower(node.child, state.withCamera(node.view, state.proj), opts);

    case "Proj":
      return lower(node.child, state.withCamera(state.view, node.proj), opts);

    case "Delay":
      // Run the creator with the accumulated state. The returned
      // sub-tree is lowered with the SAME state — Delay isn't a
      // scope that pushes anything; it just builds the child from
      // what it sees.
      return lower(node.create(state), state, opts);

    case "DepthTest":   return lower(node.child, state.pushDepthTest(node.mode), opts);
    case "DepthMask":   return lower(node.child, state.pushDepthMask(node.write), opts);
    case "DepthBias":   return lower(node.child, state.pushDepthBias(node.bias), opts);
    case "DepthClamp":  return lower(node.child, state.pushDepthClamp(node.clamp), opts);
    case "CullMode":    return lower(node.child, state.pushCullMode(node.mode), opts);
    case "FrontFace":   return lower(node.child, state.pushFrontFace(node.mode), opts);
    case "FillMode":    return lower(node.child, state.pushFillMode(node.mode), opts);
    case "BlendConstant": return lower(node.child, state.pushBlendConstant(node.value), opts);
    case "ColorMask":   return lower(node.child, state.pushColorMask(node.mask), opts);
    case "StencilMode": return lower(node.child, state.pushStencilMode(node.mode), opts);
    case "Pass": {
      // When opts.passBuckets is present (top-level lowering with
      // pass-grouping requested) the leaves below this scope route
      // into the corresponding bucket. Otherwise, ignore — leaves
      // emit as ordered in the surrounding tree.
      return lower(node.child, state.pushRenderPass(node.pass), opts);
    }
    case "VertexAttributes":
      return lower(node.child, state.pushVertexAttributes(node.attributes), opts);
    case "InstanceAttributes":
      return lower(node.child, state.pushInstanceAttributes(node.attributes), opts);
    case "Index":
      return lower(node.child, state.pushIndex(node.index), opts);
    case "Mode":
      return lower(node.child, state.pushMode(node.mode), opts);

    case "NoEvents":
      return lower(node.child, state.pushNoEvents(node.value), opts);
    case "ForcePixelPicking":
      return lower(node.child, state.pushForcePixelPicking(node.value), opts);
    case "CanFocus":
      return lower(node.child, state.pushCanFocus(node.value), opts);
    case "PickContext":
      return lower(node.child, state.pushPickContext(node.value), opts);
    case "Instanced":
      // Validate the subtree once at scene-compile (no nested
      // SgInstanced, no leaves with `instanceCount > 1`, no indirect
      // draws), then push the scope. `lowerLeaf` consults the
      // accumulated `state.instancing` and applies the rewrite.
      validateInstancingSubtree(node.child);
      return lower(node.child, state.pushInstancing(node), opts);
  }
}

/**
 * `lower` wrapper that re-validates the subtree if we're inside an
 * `Sg.Instanced` scope. Called from the adaptive-boundary cases of
 * `lower` (`Group` / `UnorderedGroup` / `AdaptiveGroup`'s `.map(...)`)
 * so that a child swapped in *after* scene-compile gets the same
 * invariant check the eager scene-compile pass ran. A violator
 * (nested `Sg.Instanced` or a leaf with `drawCall.instanceCount > 1`)
 * renders as `RenderTree.empty` plus a `console.error` — louder than
 * the original silent-wrong-pixels behaviour, less disruptive than
 * throwing inside a render-tick.
 *
 * Outside an instancing scope this is a thin pass-through to `lower`.
 */
/**
 * Recurse into a child SgNode while collecting every pickId
 * `lowerLeaf` acquires under it. Returns the lowered `RenderTree`
 * plus a `dispose` closure that releases all collected pickIds —
 * called by the adaptive-collection cleanup (see `mapUse` in the
 * UnorderedGroup case of `lower`) when the source entry is removed.
 *
 * Re-validates `Sg.Instanced` invariants on swap-in via
 * `lowerInsideInstancing` to preserve the existing behaviour.
 */
interface LoweredChild {
  readonly tree: RenderTree;
  readonly dispose: () => void;
}
const _NOOP_DISPOSE = (): void => {};
// One shared child-options object per (opts) — the old code spread the
// full CompileSceneOptions + picking objects PER CHILD, which at heap
// scale (hundreds of thousands of leaves) was measurable JS memory.
// The shared onAcquire pushes into whichever ids array is currently on
// the sink stack (lowering is synchronous, so plain save/restore keeps
// nested collections correct with the same replace-semantics).
let _idSink: number[] | undefined;
const _childOptsCache = new WeakMap<CompileSceneOptions, CompileSceneOptions>();
// Instance-tables step 2 (docs/instance-tables.md): unordered-group
// children that lower to a SINGLE leaf and stage to a row-eligible
// template get their uniform provider swapped for the shared-plan
// `RowProvider` — per-row retention drops from (leaf state + split
// result + provider union) to (plan ref + holes). Trafo-family names
// reconstruct the row model from the template's trafo holes through
// the same `deriveAutoUniform` code path the state uses. Anything
// ineligible keeps the classic result untouched — correctness never
// depends on staging.
let _rowLowering = true;
/** TEST-ONLY: toggle row lowering to compare against the classic path. */
export function __setRowLowering(enabled: boolean): void {
  _rowLowering = enabled;
}

function lowerRowOrClassic(
  child: SgNode,
  state: TraversalState,
  opts: CompileSceneOptions,
): LoweredChild {
  const wc = lowerWithCleanup(child, state, opts);
  if (!_rowLowering) return wc;
  const t = wc.tree;
  if (t.kind !== "Leaf") return wc;
  // Injected uniforms ride between scope and state in the classic
  // provider; rows don't model them — and autoUniforms:false disables
  // the auto derivation rows assume. Both are rare, pass-level options.
  if (opts.injectUniforms !== undefined || opts.autoUniforms === false) return wc;
  try {
    const staged = stageNode(child);
    if (staged.template.hasDynamicUniforms) return wc;
    const obj = t.object as RenderObject & { uniforms: IUniformProvider };
    const plan = getPlan(staged.template, state, obj.effect);
    (obj as { uniforms: IUniformProvider }).uniforms = new RowProvider(plan, staged.holes);
  } catch {
    // staging must never break lowering — keep the classic result
  }
  return wc;
}

function lowerWithCleanup(
  child: SgNode,
  state: TraversalState,
  opts: CompileSceneOptions,
): LoweredChild {
  if (opts.picking === undefined) {
    return { tree: lowerInsideInstancing(child, state, opts), dispose: _NOOP_DISPOSE };
  }
  let shared = _childOptsCache.get(opts);
  if (shared === undefined) {
    shared = { ...opts, picking: { ...opts.picking, onAcquire: (pid) => { _idSink?.push(pid); } } };
    _childOptsCache.set(opts, shared);
  }
  const ids: number[] = [];
  const prev = _idSink;
  _idSink = ids;
  let tree: RenderTree;
  try { tree = lowerInsideInstancing(child, state, shared); }
  finally { _idSink = prev; }
  if (ids.length === 0) return { tree, dispose: _NOOP_DISPOSE };
  const registry = opts.picking.registry;
  return { tree, dispose: () => { for (const id of ids) registry.release(id); } };
}

function lowerInsideInstancing(
  child: SgNode,
  state: TraversalState,
  opts: CompileSceneOptions,
): RenderTree {
  if (state.instancing !== undefined) {
    try {
      validateInstancingSubtree(child);
    } catch (e) {
      console.error(
        "Sg.Instanced: a sub-graph swap brought in a child that violates " +
        "the instancing invariants — rendering as Empty. " +
        "(Compile-time validator is one-shot; this is the per-swap re-check.)",
        e,
      );
      return RenderTree.empty;
    }
  }
  return lower(child, state, opts);
}

// ---------------------------------------------------------------------------
// Leaf lowering — build a RenderObject from leaf + TraversalState.
// ---------------------------------------------------------------------------

function lowerLeaf(
  leaf: SgLeaf,
  state: TraversalState,
  opts: CompileSceneOptions,
): RenderTree {
  const userEffect = state.shader ?? opts.defaultEffect;
  if (userEffect === undefined) {
    // No shader at scope and no default — nothing to draw. Silent
    // drop matches Aardvark.Dom's "DirectDrawNode requires a
    // surface" rule: explicit setup failures rather than implicit
    // garbage on screen.
    return RenderTree.empty;
  }

  // OIT pass split: drop leaves not selected by the pass filter.
  if (opts.passFilter !== undefined && !opts.passFilter(state.renderPass)) {
    return RenderTree.empty;
  }

  // Merge geometry-attribute scopes into the leaf — leaf-supplied
  // values win on per-key conflict. This is the Phase-2 contract:
  // VertexAttributes/InstanceAttributes/Index/Mode scope nodes flow
  // through state, and the leaf's own (more specific) entries take
  // precedence. The attribute key sets are structural (plain HashMap
  // merge); per-attribute BufferView avals stay reactive. Index
  // remains aval<BufferView | undefined>.
  const merged = mergeLeafGeometry(leaf, state);

  let effect: Effect = userEffect;
  let pickId: number | undefined;
  // Pick-registration policy:
  //   - state.noEvents is a CONSTANT aval (the common case — `<Sg
  //     NoEvents={true}>` with a static boolean): force-collapse at
  //     compile time. The force is on a constant aval (no upstream
  //     dep, never ticks) — analogous to the state.active.isConstant
  //     fast-path below.
  //   - state.noEvents is dynamic: register unconditionally and carry
  //     `noEvents` on the scope. The dispatcher consults it per-event
  //     (its forces run in event-handler context, where "now" is the
  //     user's tick and AVal.force is permitted by the policy).
  // Pick path selection — chosen at compile-scene time from constant
  // snapshots of `noEvents` and `forcePixelPicking`. Pixel and BVH
  // ray-cast are mutually exclusive: every registered leaf takes one
  // of three paths.
  //
  //   noEvents=const true     → "none"  — not registered. Invisible to
  //                                       picking.
  //   forcePixelPicking       → "pixel" — pick-chain in FS (writes
  //   =const true                         pickId), NOT in BVH.
  //   otherwise (default)     → "bvh"   — pick-chain in FS AND
  //                                       intersectable in BVH (when
  //                                       provided). Ray-cast queries
  //                                       (sceneQuery, fall-through)
  //                                       work as today.
  //
  // The flags must resolve to constants here. Reactive avals are
  // accepted but only the constant branch counts — a non-constant
  // aval defaults to BVH path.
  // AVal.force OK: constant-aval one-shot reads at compile-scene time.
  const noEventsConst = state.noEvents.isConstant && state.noEvents.force();
  const fppConst = state.forcePixelPicking.isConstant && state.forcePixelPicking.force();
  const pickPath: "none" | "pixel" | "bvh" =
    noEventsConst ? "none" :
    fppConst      ? "pixel" :
                    "bvh";
  if (opts.picking !== undefined && pickPath !== "none") {
    const geom = opts.picking.geomHas !== undefined
      ? { has: opts.picking.geomHas, key: opts.picking.geomKey ?? "__user" }
      : defaultGeomHas(merged);
    // Portal leaves (under `<Sg PickContext=…>`) compose the portal
    // final — pick slots carry the sampled source-uv so the resolver
    // can recurse into the offscreen scene. Only attach the sub-
    // context to the scope when the portal final was actually chosen
    // (fallback finals write normal/depth in slots 1-2, which must
    // NOT be decoded as uv).
    const portal = state.pickSubContext !== undefined;
    const composed = composePickChainWithChoiceCached(userEffect, geom.key, geom.has, portal);
    effect = composed.effect;
    const mode = composed.choice.final === "FinalB" ? "B" : "A";
    const pid = opts.picking.registry.acquire({
      handlers: state.handlers,
      cursor: state.cursor,
      pickThrough: state.pickThrough,
      active: state.active,
      view: state.view,
      proj: state.proj,
      model: state.modelLazy(),
      pixelSnapRadius: state.pixelSnapRadius,
      pickPriority: state.pickPriority,
      canFocus: state.canFocus,
      pickPath,
      ...(state.intersectable !== undefined ? { intersectable: state.intersectable } : {}),
      ...(composed.choice.final === "FinalPortal" && state.pickSubContext !== undefined
        ? { pickSubContext: state.pickSubContext } : {}),
    }, mode);
    pickId = pid;
    if (opts.picking.onAcquire !== undefined) opts.picking.onAcquire(pid);
  }

  // OIT: append the weighted-blend / A-buffer fragment writer onto the
  // (transparent) leaf's effect. Mirrors aardvark's composeSurface.
  if (opts.composeEffect !== undefined) {
    effect = composeEffectCached(opts.composeEffect, effect);
  }

  // If an `Sg.Instanced` scope is in effect, rewrite the effect via
  // `instanceUniforms` and patch the leaf with per-instance attributes
  // + identity-trafo uniform overrides + an instance-counted draw call.
  let leafForBuild = merged;
  let stateForBuild = state;
  if (state.instancing !== undefined) {
    const parentModel = state.instancingParentModel ?? state.modelLazy();
    const applied = applyInstancing(
      state.instancing, state.modelLazy(), parentModel,
      state.view, state.proj, effect, merged,
    );
    effect = applied.effect;
    leafForBuild = {
      ...merged,
      instanceAttributes: applied.instanceAttributes,
      drawCall: applied.drawCall,
    };
    if (!applied.uniformOverrides.isEmpty) {
      stateForBuild = stateForBuild.pushUniforms(applied.uniformOverrides);
    }
  }
  // Active gating: when state.active is structurally CONSTANT we can
  // collapse at compile time. `false` ⇒ skip the leaf entirely (no
  // RO ever built). `true` ⇒ behave as if there was no Active scope.
  // (This force is not on the live path — `isConstant` avals never
  // re-fire.)
  if (state.active.isConstant) {
    // AVal.force OK: isConstant guard — value is by definition immutable.
    if (!state.active.force()) return RenderTree.empty;
    const obj: RenderObject = buildRenderObject(leafForBuild, stateForBuild, effect, opts, pickId);
    return RenderTree.leaf(obj);
  }

  // Reactive Active: attach the aval to the RO so the runtime gates
  // it in place (heap path skips the draw, legacy path returns 0
  // vertices) WITHOUT removing the RO from the scene. This avoids
  // the pool-churn cycle the old `RenderTree.adaptive(empty/baseTree)`
  // approach caused when a tile's visibility flipped many times per
  // camera move.
  const baseObj: RenderObject = buildRenderObject(leafForBuild, stateForBuild, effect, opts, pickId);
  const obj: RenderObject = { ...baseObj, active: state.active };
  return RenderTree.leaf(obj);
}

// Applicator hoists (scene-templates M2): anything built from a
// SHARED input collapses to one instance via identity memos. A scene
// with 10k leaves referencing one geometry node gets ONE attribute
// provider, one composed OIT effect per user effect, one overridden
// PipelineState per (state, pass) — not 10k of each.
const _attrProviderMemo = new WeakMap<object, IAttributeProvider>();
function attrProviderOf(m: HashMap<string, BufferView>): IAttributeProvider {
  let p = _attrProviderMemo.get(m);
  if (p === undefined) {
    p = AttributeProvider.ofMap(m);
    _attrProviderMemo.set(m, p);
  }
  return p;
}

const _composeEffectMemo = new WeakMap<object, WeakMap<Effect, Effect>>();
function composeEffectCached(fn: (e: Effect) => Effect, e: Effect): Effect {
  let byEffect = _composeEffectMemo.get(fn);
  if (byEffect === undefined) {
    byEffect = new WeakMap();
    _composeEffectMemo.set(fn, byEffect);
  }
  let r = byEffect.get(e);
  if (r === undefined) {
    r = fn(e);
    byEffect.set(e, r);
  }
  return r;
}

const _psOverrideMemo = new WeakMap<object, WeakMap<PipelineState, Map<number, PipelineState>>>();
function pipelineOverrideCached(
  fn: (ps: PipelineState, renderPass: number) => PipelineState,
  ps: PipelineState,
  renderPass: number,
): PipelineState {
  let byPs = _psOverrideMemo.get(fn);
  if (byPs === undefined) {
    byPs = new WeakMap();
    _psOverrideMemo.set(fn, byPs);
  }
  let byPass = byPs.get(ps);
  if (byPass === undefined) {
    byPass = new Map();
    byPs.set(ps, byPass);
  }
  let r = byPass.get(renderPass);
  if (r === undefined) {
    r = fn(ps, renderPass);
    byPass.set(renderPass, r);
  }
  return r;
}

// Provider over `opts.injectUniforms` — memoised per map so every
// leaf shares one instance.
const _injectedProviderMemo = new WeakMap<object, IUniformProvider>();
function getInjectedProvider(m: HashMap<string, aval<unknown>>): IUniformProvider {
  let p = _injectedProviderMemo.get(m);
  if (p === undefined) {
    p = UniformProvider.ofMap(m);
    _injectedProviderMemo.set(m, p);
  }
  return p;
}

// Names supplied via `opts.injectUniforms` — memoised per map so the
// M1 uniform check doesn't rebuild the set for every leaf.
const injectedNamesMemo = new WeakMap<object, ReadonlySet<string>>();
function injectedNames(opts: CompileSceneOptions): ReadonlySet<string> | undefined {
  const m = opts.injectUniforms;
  if (m === undefined) return undefined;
  let s = injectedNamesMemo.get(m);
  if (s === undefined) {
    const names = new Set<string>();
    for (const [k] of m) names.add(k);
    injectedNamesMemo.set(m, names);
    s = names;
  }
  return s;
}

function buildRenderObject(
  leaf: SgLeaf,
  state: TraversalState,
  effect: Effect,
  opts: CompileSceneOptions,
  pickId: number | undefined,
): RenderObject {
  // M1 (docs/scene-templates.md): name effect uniforms nothing in scope
  // resolves — today they silently read zero at draw time. Deduped per
  // (effect, missing-set); per-leaf cost is a few Set probes.
  warnUnresolvedUniforms(effect, state.uniforms, injectedNames(opts), leaf.instanceAttributes);
  // `<Sg Uniform={…}>` scope entries, split into scalars vs textures/
  // samplers — memoised PER CHAIN NODE, extending the parent's cached
  // split (scene-templates M2): leaves reuse ancestor texture/sampler
  // maps + scalar providers by reference; a per-leaf uniform scope
  // costs one provider node reusing the bag's own entry map, never a
  // merged HashMap.
  const { textures, samplers, scalarProvider, scalarCount } = getSplitScopeUniforms(state.uniforms);
  // PickId is a FIRST-CLASS RenderObject field (`ro.pickId`) — the
  // heap writes it inline into the drawHeader, the classic path
  // synthesizes the uniform. It must NOT ride the uniform provider: a
  // unique per-leaf aval/provider measured ~6.5 KB of JS heap PER LEAF
  // at scale (see ~/claude/pickid-inline-plan.md).
  //
  // Provider shadowing order (same as the old overlay-union): scope
  // scalars (inner-most first via the chain) → injected uniforms →
  // state-derived autos. `injectUniforms` must not override scope
  // entries (old code only added missing keys) but MUST shadow the
  // auto-derived names — placing it between chain and state does both.
  const injectedP = opts.injectUniforms !== undefined
    ? getInjectedProvider(opts.injectUniforms)
    : undefined;
  let uniforms: IUniformProvider;
  if (opts.autoUniforms ?? true) {
    let tail: IUniformProvider = state;
    if (injectedP !== undefined) tail = new UnionOf2(injectedP, state);
    uniforms = scalarCount === 0 || scalarProvider === undefined
      ? tail
      : new UnionOf2(scalarProvider, tail);
  } else {
    const scopeOnly = scalarProvider ?? UniformProvider.empty;
    uniforms = injectedP !== undefined ? new UnionOf2(scopeOnly, injectedP) : scopeOnly;
  }
  // Use the `PipelineState` the traversal already derived for this
  // subtree (shared across all leaves under the same render-state
  // scopes); only fall back to deriving here when there isn't one
  // (e.g. `TraversalState.empty` used directly, or a render-state
  // scope cleared it — descendants of such a scope re-derive once
  // each, same as before).
  let pipelineState = state.pipelineState ?? derivePipelineState(state, opts);
  if (opts.pipelineOverride !== undefined) {
    pipelineState = pipelineOverrideCached(opts.pipelineOverride, pipelineState, state.renderPass);
  }

  // RenderObject.indices is `aval<BufferView>` (no undefined). The
  // leaf carries `aval<BufferView | undefined>`; lift to a
  // possibly-absent indices field here. We treat
  // "constantly undefined" as no indices; everything else stays
  // reactive (an aval that flips between undefined and a real view
  // is not yet supported by RenderObject — the runtime would have to
  // re-prepare for index format changes; we narrow by mapping the
  // undefined case to a "draw nothing"-style sentinel only if needed).
  const obj: RenderObject = {
    effect,
    ...(pickId !== undefined ? { pickId } : {}),
    pipelineState,
    vertexAttributes: attrProviderOf(leaf.vertexAttributes),
    ...(leaf.instanceAttributes !== undefined
      ? { instanceAttributes: attrProviderOf(leaf.instanceAttributes) }
      : {}),
    uniforms,
    textures,
    samplers,
    // Leaf-own storage stays `storageBuffers` (heap-disqualifying);
    // scene-level injections (the OIT node pool) ride `injectedStorage`,
    // which the heap binds at bucket level instead.
    ...(leaf.storageBuffers !== undefined ? { storageBuffers: leaf.storageBuffers } : {}),
    ...(opts.injectStorage !== undefined ? { injectedStorage: opts.injectStorage } : {}),
    ...(leaf.indices !== undefined ? { indices: leaf.indices } : {}),
    drawCall: leaf.drawCall,
    // GPU transform propagation: hand the heap the Model ancestor chain so it
    // composes per-RO Model on the GPU (a shared root trafo over N descendants
    // stays O(1) on the CPU). The legacy path ignores this and uses the eager
    // `ModelTrafo` uniform instead.
    ...(state.modelChain.length > 0 ? { modelChain: state.modelChain } : {}),
    // Route any derived-mode rule the traversal accumulated onto the
    // RO so the heap path picks it up. The heap runtime evaluates the
    // rule's CPU closure (or runs the GPU kernel if the rule carries
    // a `gpu` spec) per RO each frame the rule's inputs are dirty.
    ...((state.cullModeRule !== undefined
      || state.frontFaceRule !== undefined
      || state.modeRule !== undefined
      || state.depthTestRule !== undefined
      || state.depthMaskRule !== undefined)
      ? {
          modeRules: {
            ...(state.cullModeRule  !== undefined ? { cull:          state.cullModeRule  } : {}),
            ...(state.frontFaceRule !== undefined ? { frontFace:     state.frontFaceRule } : {}),
            ...(state.modeRule      !== undefined ? { topology:      state.modeRule      } : {}),
            ...(state.depthTestRule !== undefined ? { depthCompare:  state.depthTestRule } : {}),
            ...(state.depthMaskRule !== undefined ? { depthWrite:    state.depthMaskRule } : {}),
          },
        }
      : {}),
  };
  return obj;
}

/**
 * Merge state's geometry-attribute scopes into the leaf — leaf-
 * supplied values win on per-key conflict for vertex / instance
 * attributes, and on whole-value override for index. The attribute
 * key sets are structural (plain HashMap merge); inner per-attribute
 * BufferView avals stay reactive. Index remains aval<BufferView |
 * undefined>.
 */
function mergeLeafGeometry(leaf: SgLeaf, state: TraversalState): SgLeaf {
  // Hoist: when no geometry scope contributes anything, the leaf passes
  // through UNCHANGED — preserving its identity so every downstream
  // per-map memo (attribute providers, pick geometry keys) collapses
  // across all occurrences of a shared leaf (one provider for all
  // cylinders, not one per cylinder).
  if (state.vertexAttributes.count === 0
    && state.instanceAttributes.count === 0
    && (leaf.indices !== undefined || state.index === undefined)) {
    return leaf;
  }
  const vertex: HashMap<string, BufferView> = (() => {
    if (state.vertexAttributes.count === 0) return leaf.vertexAttributes;
    let m = state.vertexAttributes;
    for (const [k, v] of leaf.vertexAttributes) m = m.add(k, v);
    return m;
  })();

  let instance: HashMap<string, BufferView> | undefined;
  if (leaf.instanceAttributes !== undefined) {
    if (state.instanceAttributes.count === 0) {
      instance = leaf.instanceAttributes;
    } else {
      let m = state.instanceAttributes;
      for (const [k, v] of leaf.instanceAttributes) m = m.add(k, v);
      instance = m;
    }
  } else if (state.instanceAttributes.count > 0) {
    instance = state.instanceAttributes;
  }

  // Index: leaf-supplied wins; otherwise use scope.
  const indices: BufferView | undefined =
    leaf.indices !== undefined ? leaf.indices : state.index;

  return {
    kind: "Leaf",
    vertexAttributes: vertex,
    ...(instance !== undefined ? { instanceAttributes: instance } : {}),
    indices,
    drawCall: leaf.drawCall,
    ...(leaf.storageBuffers !== undefined ? { storageBuffers: leaf.storageBuffers } : {}),
  };
}

// Default sampler — linear/linear/mip-linear/repeat. Shared across
// every Sg leaf that doesn't declare its own samplers. Cached so the
// runtime's value-equality keyed sampler cache collapses to one
// GPUSampler.
const _defaultSampler = ISamplerImpl.fromDescriptor({
  magFilter: "linear", minFilter: "linear", mipmapFilter: "linear",
  addressModeU: "repeat", addressModeV: "repeat",
});
const _defaultSamplerAval = AVal.constant(_defaultSampler);

/**
 * Split texture-valued uniforms out of the merged uniform map. The
 * Sg API treats textures as ordinary uniforms whose value happens to
 * be an `ITexture`; the rendering runtime needs them on separate
 * `textures` / `samplers` maps (paired by name per the
 * `legaliseTypes` IR pass — `Sampler X` binding + `Texture X_view`).
 *
 * We bind the texture aval under both `name` and `${name}_view` so
 * `prepareRenderObject` finds the binding whichever name the iface
 * uses, and supply a shared default sampler under `name`. URL-deferred
 * `ITexture` values pass through `resolveTextureAval` so they start
 * out as a checker placeholder until the fetch completes.
 *
 * Why force here: a structural one-shot read at compile-scene time —
 * we need to know each uniform's *shape* (texture vs scalar) to lay
 * out the RenderObject. The force fires once per leaf during
 * lowering; the render walk never reaches it. Same category as the
 * pass-bucketing static forces above.
 */
function splitTexturesFromUniforms(
  merged: HashMap<string, aval<unknown>>,
): {
  uniforms: HashMap<string, aval<unknown>>;
  textures: HashMap<string, aval<ITexture>>;
  samplers: HashMap<string, aval<ISampler>>;
} {
  let outU = HashMap.empty<string, aval<unknown>>();
  let outT = HashMap.empty<string, aval<ITexture>>();
  let outS = HashMap.empty<string, aval<ISampler>>();
  for (const [k, v] of merged) {
    // §7 derived-uniform rules pass straight through (they're not avals — no `.force()`);
    // the heap renderer routes them to the compute pre-pass.
    if (isDerivedRule(v)) { outU = outU.add(k, v as unknown as aval<unknown>); continue; }
    // Why force here: structural classification at compile-scene time.
    const current = v.force();
    if (isITexture(current)) {
      const texAval = resolveTextureAval(v as aval<ITexture>);
      outT = outT.add(k, texAval).add(`${k}_view`, texAval);
      outS = outS.add(k, _defaultSamplerAval);
    } else {
      outU = outU.add(k, v);
    }
  }
  return { uniforms: outU, textures: outT, samplers: outS };
}

// Memoise the texture/sampler/scalar classification PER CHAIN NODE and
// extend the parent's cached result incrementally (scene-templates M2):
// a leaf-level `<Sg.Uniform>` scope with two scalar entries reuses the
// parent's texture/sampler maps by REFERENCE (no per-leaf empty maps)
// and composes its scalar provider as `own → parent` without ever
// materializing a merged HashMap. Classification forces each entry
// once per SCOPE, not once per leaf.
interface SplitUniforms {
  readonly textures: HashMap<string, aval<ITexture>>;
  readonly samplers: HashMap<string, aval<ISampler>>;
  /** Chain-shaped scalar provider (own-first shadowing), or undefined
   *  when no scalars exist anywhere on the chain. */
  readonly scalarProvider?: IUniformProvider | undefined;
  /** Total scalar entries on the chain (fast emptiness check). */
  readonly scalarCount: number;
}
const _emptySplit: SplitUniforms = {
  textures: HashMap.empty<string, aval<ITexture>>(),
  samplers: HashMap.empty<string, aval<ISampler>>(),
  scalarCount: 0,
};
const _splitScopeUniformsCache = new WeakMap<UniformScopeChain, SplitUniforms>();
function getSplitScopeUniforms(chain: UniformScopeChain): SplitUniforms {
  const cached = _splitScopeUniformsCache.get(chain);
  if (cached !== undefined) return cached;
  let r: SplitUniforms;
  const parent = chain.parent !== undefined ? getSplitScopeUniforms(chain.parent) : _emptySplit;
  if (chain.entries.count === 0) {
    r = parent;
  } else {
    const own = splitTexturesFromUniforms(chain.entries);
    const textures = own.textures.count === 0
      ? parent.textures
      : mergeMaps(parent.textures, own.textures);
    const samplers = own.samplers.count === 0
      ? parent.samplers
      : mergeMaps(parent.samplers, own.samplers);
    let scalarProvider: IUniformProvider | undefined;
    if (own.uniforms.count === 0) scalarProvider = parent.scalarProvider;
    else {
      const ownP = UniformProvider.ofMap(own.uniforms);
      scalarProvider = parent.scalarProvider === undefined ? ownP : new UnionOf2(ownP, parent.scalarProvider);
    }
    r = { textures, samplers, scalarProvider, scalarCount: parent.scalarCount + own.uniforms.count };
  }
  _splitScopeUniformsCache.set(chain, r);
  return r;
}

function mergeMaps<V>(outer: HashMap<string, V>, inner: HashMap<string, V>): HashMap<string, V> {
  if (outer.count === 0) return inner;
  let m = outer;
  for (const [k, v] of inner) m = m.add(k, v);
  return m;
}

/** Two-provider union without the general variadic machinery. */
class UnionOf2 implements IUniformProvider {
  constructor(
    private readonly a: IUniformProvider,
    private readonly b: IUniformProvider,
  ) {}
  tryGet(name: string): aval<unknown> | undefined {
    return this.a.tryGet(name) ?? this.b.tryGet(name);
  }
  *names(): Iterable<string> {
    yield* this.a.names();
    yield* this.b.names();
  }
}

// ---------------------------------------------------------------------------
// Pipeline state — derived from TraversalState's render-state scopes.
//
// All Phase-1 scopes (DepthTest/Mask/Bias/Clamp, CullMode, FrontFace,
// FillMode, Multisample, BlendConstant, ColorMask, StencilMode) are
// declared as `aval<…>` for parity with Aardvark.Dom. PipelineState
// itself is a static value on RenderObject — the runtime caches
// pipelines by it. We `force` the avals once at compile time. For
// dynamic state changes, swap the subtree via `Sg.adaptive` so a new
// RenderObject (with fresh PipelineState) is produced. (See
// docs/FUTURE.md: "reactive PipelineState".)
// ---------------------------------------------------------------------------

const defaultRasterizer: PlainRasterizerState = {
  topology: "triangle-list",
  cullMode: "none",
  frontFace: "ccw",
};

function fillModeWarning(): void { /* no-op stub; logged-once below */ }
let warnedNonFillFillMode = false;
let warnedDepthClamp = false;
let warnedBlendConstant = false;

function topologyForMode(mode: ModeValue, fill: "fill" | "line" | "point"): Topology {
  // FillMode "fill" leaves topology to the leaf-supplied Mode scope.
  // FillMode "line" / "point" overrides topology to a wireframe-ish
  // approximation (WebGPU has no real polygonMode). DOC: limitation.
  if (fill === "line") {
    if (!warnedNonFillFillMode) {
      console.warn("[wombat.dom] FillMode='line' approximated by topology='line-list' — true polygon-mode wireframe is not exposed by WebGPU.");
      warnedNonFillFillMode = true; fillModeWarning();
    }
    return "line-list";
  }
  if (fill === "point") {
    if (!warnedNonFillFillMode) {
      console.warn("[wombat.dom] FillMode='point' approximated by topology='point-list'.");
      warnedNonFillFillMode = true;
    }
    return "point-list";
  }
  return mode;
}

function maskBits(mask: ColorMaskValue): number {
  let bits = 0;
  if (mask.r) bits |= 1;
  if (mask.g) bits |= 2;
  if (mask.b) bits |= 4;
  if (mask.a) bits |= 8;
  return bits;
}

function blendStateWithMask(b: BlendState, mask: ColorMaskValue | undefined): BlendState {
  if (mask === undefined) return b;
  return {
    ...(b.color !== undefined ? { color: b.color } : {}),
    ...(b.alpha !== undefined ? { alpha: b.alpha } : {}),
    writeMask: AVal.constant(maskBits(mask)),
  };
}

function defaultBlendForMask(mask: ColorMaskValue): BlendState {
  // No blending, just channel mask. Mirrors WebGPU's "passthrough".
  return {
    color: {
      operation: AVal.constant<GPUBlendOperation>("add"),
      srcFactor: AVal.constant<GPUBlendFactor>("one"),
      dstFactor: AVal.constant<GPUBlendFactor>("zero"),
    },
    alpha: {
      operation: AVal.constant<GPUBlendOperation>("add"),
      srcFactor: AVal.constant<GPUBlendFactor>("one"),
      dstFactor: AVal.constant<GPUBlendFactor>("zero"),
    },
    writeMask: AVal.constant(maskBits(mask)),
  };
}

function derivePipelineState(state: TraversalState, opts: CompileSceneOptions): PipelineState {
  const baseRast = opts.rasterizer ?? defaultRasterizer;

  // Topology = mode + fillMode (combine reactively). Calls
  // `topologyForMode` per evaluation so the FillMode warning still
  // triggers, but only as a side effect of the aval map.
  const topology: aval<Topology> = AVal.zip(state.mode, state.fillMode).map((m, f) =>
    topologyForMode(m, f),
  );

  // depthBias: state.depthBias (if present) is the source of truth;
  // otherwise fall back to the static base rasterizer's depthBias.
  // Use the namespace form `AVal.map(fn, src)` so the adaptive memo
  // plugin recognises this as a combinator call (it can't infer kind
  // from `state.depthBias` — that's a property access, opaque to the
  // plugin's lightweight scanner — but it does know `AVal.map`).
  const baseDepthBias = baseRast.depthBias;
  const _depthBiasSrc = state.depthBias;
  const depthBiasAval = _depthBiasSrc !== undefined
    ? AVal.map(_depthBiasSrc, (b: DepthBiasState | undefined) => b)
    : (baseDepthBias !== undefined ? AVal.constant<DepthBiasState | undefined>(baseDepthBias) : undefined);

  const rasterizer: RasterizerState = {
    topology,
    cullMode: state.cullMode,
    frontFace: state.frontFace,
    ...(depthBiasAval !== undefined ? { depthBias: depthBiasAval } : {}),
  };

  // ColorMask + BlendMode → blends aval. We construct the entire
  // blends HashMap reactively from state.colorMask (and the static
  // state.blendMode if present); per-entry BlendState fields stay
  // aval-typed.
  let blends: aval<HashMap<string, BlendState>> | undefined;
  const blendModeStatic = state.blendMode;
  const _colorMaskSrc = state.colorMask;
  if (blendModeStatic !== undefined || _colorMaskSrc !== undefined) {
    blends = AVal.map(_colorMaskSrc, (masks: HashMap<string, ColorMaskValue>) => {
      let m = HashMap.empty<string, BlendState>();
      if (blendModeStatic !== undefined) {
        m = m.add("Colors", blendStateWithMask(blendModeStatic, masks.tryFind("Colors")));
      }
      for (const [k, v] of masks) {
        if (m.tryFind(k) === undefined) m = m.add(k, defaultBlendForMask(v));
      }
      return m;
    });
    // If neither blendMode nor any colorMask entries, this evaluates
    // to an empty HashMap — fine to leave on PipelineState; the
    // prepared-RO snapshots empty as no-blend.
  }

  // Stencil: state.stencilMode is `aval<StencilModeValue> | undefined`.
  // The shape carries all fields in one aval; we lift each field.
  let stencil: StencilState | undefined;
  const sm = state.stencilMode;
  if (sm !== undefined) {
    stencil = {
      enabled: sm.map(v => v.enabled),
      reference: sm.map(v => v.reference),
      readMask: sm.map(v => v.readMask),
      writeMask: sm.map(v => v.writeMask),
      front: {
        compare:     sm.map(v => v.front.compare),
        failOp:      sm.map(v => v.front.fail),
        depthFailOp: sm.map(v => v.front.depthFail),
        passOp:      sm.map(v => v.front.pass),
      },
      back: {
        compare:     sm.map(v => v.back.compare),
        failOp:      sm.map(v => v.back.fail),
        depthFailOp: sm.map(v => v.back.depthFail),
        passOp:      sm.map(v => v.back.pass),
      },
    };
  }

  // Phase 1's compile-time warnings for DepthClamp / BlendConstant
  // were dropped when PipelineState became fully reactive: we no
  // longer force these avals at compile time, so the warning would
  // now fire lazily on first frame-eval rather than at scope entry.
  // Per-field status:
  //
  //   * BlendConstant — wired through to `pass.setBlendConstant` in
  //     `preparedRenderObject.record` when `PipelineState.blendConstant`
  //     is present (per-frame, no pipeline rebuild).
  //   * DepthClamp — folded into `DepthState.clamp` and consumed by
  //     `buildPipelineForSnap` via `primitive.unclippedDepth`. Falls
  //     back to "no-op" silently when the WebGPU adapter doesn't
  //     advertise the `unclippedDepth` feature; the pipeline compile
  //     itself surfaces the error in that case.
  void warnedDepthClamp; void warnedBlendConstant;

  const depth: DepthState = {
    write: state.depthMask,
    compare: state.depthTest,
    clamp: state.depthClamp,
  };

  const ps: PipelineState = {
    rasterizer,
    depth,
    ...(blends !== undefined ? { blends } : {}),
    ...(stencil !== undefined ? { stencil } : {}),
    ...(state.blendConstant !== undefined ? { blendConstant: state.blendConstant } : {}),
  };
  return ps;
}

// ---------------------------------------------------------------------------
// Default geomHas — pick-chain dependency probe over leaf attributes
// ---------------------------------------------------------------------------

function defaultGeomHas(leaf: SgLeaf): { has: (semantic: string) => boolean; key: string } {
  // Why: `BufferView` doesn't currently carry an explicit `semantic`
  // — wombat.dom binds vertex / instance attributes by their map
  // key directly to the shader's input by NAME (`Positions`,
  // `Normals`, ...), and that name is also what the pick chooser
  // queries via `effectDependencies`. So for v1 we treat the
  // attribute KEY as the semantic. If we ever introduce a separate
  // `semantic` on `BufferView`, fold it in here.
  //
  // Pick-chain composition is a STATIC operation baked into the
  // RenderObject's effect at compile time. The attribute key set is
  // now structural (plain HashMap), so we just enumerate it directly.
  const set = new Set<string>();
  for (const [k] of leaf.vertexAttributes) set.add(k);
  if (leaf.instanceAttributes !== undefined) {
    for (const [k] of leaf.instanceAttributes) set.add(k);
  }
  const key = [...set].sort().join("|");
  return { has: (sem) => set.has(sem), key };
}

// Avoid unused-import warning while ASet is reserved for the
// UnorderedGroup path's type inference.
void ASet;
void Trafo3d;
