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

import {
  AList, ASet, AVal,
  HashMap,
  type alist, type aset, type aval,
} from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";
import { RenderTree } from "@aardworx/wombat.rendering/core";
import type {
  Command, ClearValues, IFramebuffer, ISampler, ITexture,
  PipelineState, RasterizerState, RenderObject,
} from "@aardworx/wombat.rendering/core";
import type { Effect } from "@aardworx/wombat.shader";

import type { SgLeaf, SgNode } from "./sg.js";
import { TraversalState } from "./traversalState.js";

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
  readonly rasterizer?: RasterizerState;
  /** Whether to inject `ModelTrafo` / `ViewTrafo` / `ProjTrafo` / `ViewProjTrafo` uniforms. Default `true`. */
  readonly autoUniforms?: boolean;
}

/**
 * Compile a scene-graph tree into the `alist<Command>` that
 * `Runtime.compile` consumes. Returns one `Render` command (and
 * optionally one `Clear` ahead of it).
 */
export function compileScene(
  sg: SgNode,
  output: aval<IFramebuffer>,
  opts: CompileSceneOptions = {},
): alist<Command> {
  const initial = opts.initialState ?? TraversalState.empty;
  const tree = lower(sg, initial, opts);
  const cmds: Command[] = [];
  if (opts.clear !== undefined) {
    cmds.push({ kind: "Clear", output, values: opts.clear });
  }
  cmds.push({ kind: "Render", output, tree });
  return AList.ofList(cmds);
}

// ---------------------------------------------------------------------------
// Lowering — SgNode + TraversalState → RenderTree
// ---------------------------------------------------------------------------

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
      const children: alist<RenderTree> = node.children.map(child => lower(child, state, opts));
      return RenderTree.orderedFromList(children);
    }

    case "UnorderedGroup": {
      // aset<SgNode> → aset<RenderTree>; outer is UnorderedFromSet.
      const children: aset<RenderTree> = node.children.map(child => lower(child, state, opts));
      return RenderTree.unorderedFromSet(children);
    }

    case "AdaptiveGroup":
      // single-slot subtree swap: aval<SgNode> → aval<RenderTree>.
      return RenderTree.adaptive(node.child.map(child => lower(child, state, opts)));

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

    case "On":
      return lower(node.child, state.pushHandlers(node.handlers), opts);

    case "Active":
      return lower(node.child, state.pushActive(node.active), opts);
  }
}

// ---------------------------------------------------------------------------
// Leaf lowering — build a RenderObject from leaf + TraversalState.
// ---------------------------------------------------------------------------

function lowerLeaf(
  leaf: SgLeaf,
  state: TraversalState,
  opts: CompileSceneOptions,
): RenderTree {
  const effect = state.shader ?? opts.defaultEffect;
  if (effect === undefined) {
    // No shader at scope and no default — nothing to draw. Silent
    // drop matches Aardvark.Dom's "DirectDrawNode requires a
    // surface" rule: explicit setup failures rather than implicit
    // garbage on screen.
    return RenderTree.empty;
  }

  const obj: RenderObject = buildRenderObject(leaf, state, effect, opts);
  const baseTree: RenderTree = RenderTree.leaf(obj);

  // Active gating: skip when statically false, wrap as adaptive
  // when dynamic, pass-through when constantly true.
  if (state.active.isConstant) {
    return AVal.force(state.active) ? baseTree : RenderTree.empty;
  }
  return RenderTree.adaptive(state.active.map(a => (a ? baseTree : RenderTree.empty)));
}

function buildRenderObject(
  leaf: SgLeaf,
  state: TraversalState,
  effect: Effect,
  opts: CompileSceneOptions,
): RenderObject {
  const uniforms = mergeUniforms(leaf, state, opts);
  const pipelineState = derivePipelineState(state, opts);

  const obj: RenderObject = {
    effect,
    pipelineState,
    vertexAttributes: leaf.vertexAttributes,
    ...(leaf.instanceAttributes !== undefined ? { instanceAttributes: leaf.instanceAttributes } : {}),
    uniforms,
    textures: HashMap.empty<string, aval<ITexture>>(),
    samplers: HashMap.empty<string, aval<ISampler>>(),
    ...(leaf.indices !== undefined ? { indices: leaf.indices } : {}),
    drawCall: leaf.drawCall,
  };
  return obj;
}

// ---------------------------------------------------------------------------
// Auto-injected uniforms — ModelTrafo / ViewTrafo / ProjTrafo / ViewProjTrafo.
//
// User-supplied uniforms (state.uniforms) win on key conflict. The
// runtime's UBO packer ignores keys the shader's program interface
// doesn't declare, so the over-broad auto-inject costs nothing for
// shaders that don't reference these names.
// ---------------------------------------------------------------------------

function mergeUniforms(
  _leaf: SgLeaf,
  state: TraversalState,
  opts: CompileSceneOptions,
): HashMap<string, aval<unknown>> {
  const auto = (opts.autoUniforms ?? true) ? autoInjectedUniforms(state) : HashMap.empty<string, aval<unknown>>();
  // Inner-wins: state.uniforms entries override auto.
  let merged = auto;
  for (const [k, v] of state.uniforms) merged = merged.add(k, v);
  return merged;
}

function autoInjectedUniforms(state: TraversalState): HashMap<string, aval<unknown>> {
  // ViewProjTrafo: world → view → clip. In Trafo3d-land that's
  // view.mul(proj) (apply view, then proj). Forward of the result
  // is proj.forward · view.forward — i.e. proj after view, which
  // is what column-vector math expects.
  const viewProj = AVal.zip(state.view, state.proj).map((v, p) => v.mul(p));
  let map = HashMap.empty<string, aval<unknown>>();
  map = map.add("ModelTrafo", state.model);
  map = map.add("ViewTrafo", state.view);
  map = map.add("ProjTrafo", state.proj);
  map = map.add("ViewProjTrafo", viewProj);
  // Common derived helpers — cheap to expose; shaders that don't
  // bind them pay nothing at compile time.
  map = map.add("ModelTrafoInv", state.model.map(t => t.inverse()));
  map = map.add("ViewTrafoInv", state.view.map(t => t.inverse()));
  map = map.add("ViewportSize", state.viewport);
  return map;
}

// ---------------------------------------------------------------------------
// Pipeline state — sane defaults; BlendMode / DepthState scopes
// override pieces.
// ---------------------------------------------------------------------------

const defaultRasterizer: RasterizerState = {
  topology: "triangle-list",
  cullMode: "none",
  frontFace: "ccw",
};

function derivePipelineState(state: TraversalState, opts: CompileSceneOptions): PipelineState {
  const rasterizer = opts.rasterizer ?? defaultRasterizer;
  const ps: PipelineState = {
    rasterizer,
    depth: { write: true, compare: "less" },
    ...(state.blendMode !== undefined
      // Single-attachment default: apply the scope's blend to "color".
      // Multi-target scenes plug their own per-attachment map via
      // a dedicated attribute scope (M9+).
      ? { blends: HashMap.empty<string, typeof state.blendMode>().add("color", state.blendMode) }
      : {}),
  };
  return ps;
}

// Avoid unused-import warning while ASet is reserved for the
// UnorderedGroup path's type inference.
void ASet;
void Trafo3d;
