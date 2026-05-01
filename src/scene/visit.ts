// Tree-walking visitors over `SgNode`. Pure: input = scene tree
// + initial TraversalState; output = whatever the visitor builds.
//
// `forEachLeaf` is the foundation the rendering layer's lowering
// (M3) builds on — it produces an `(SgLeaf, TraversalState)`
// callback for every leaf reachable in the static (non-adaptive)
// tree. Adaptive containers (`Group` over alist, `UnorderedGroup`
// over aset, `AdaptiveGroup` over aval) require reader-driven
// emission and are handled by the rendering walker instead.
//
// The visitors here exist for:
//   * unit tests of attribute composition (no reader machinery)
//   * collecting `IIntersectable`s for the BVH path (M10)
//   * static analysis (e.g. counting leaves, dumping the tree)

import { AVal } from "@aardworx/wombat.adaptive";
import type { SgNode, SgLeaf } from "./sg.js";
import { TraversalState } from "./traversalState.js";

/**
 * Visit every leaf reachable in the **static** tree under `node`,
 * applying `state` accumulators along the way. Adaptive container
 * children (`alist`, `aset`, `aval`) are forced once at call time
 * — sufficient for tests, but production rendering must use the
 * delta-driven walker in M3 instead.
 */
export function forEachLeaf(
  node: SgNode,
  state: TraversalState,
  visit: (leaf: SgLeaf, leafState: TraversalState) => void,
): void {
  switch (node.kind) {
    case "Empty":
      return;
    case "Leaf":
      visit(node, state);
      return;
    case "Group":
      for (const child of AVal.force(node.children.content)) {
        forEachLeaf(child, state, visit);
      }
      return;
    case "UnorderedGroup":
      for (const child of AVal.force(node.children.content)) {
        forEachLeaf(child, state, visit);
      }
      return;
    case "AdaptiveGroup":
      forEachLeaf(AVal.force(node.child), state, visit);
      return;
    case "Trafo":
      forEachLeaf(node.child, state.pushTrafo(node.value), visit);
      return;
    case "Shader":
      forEachLeaf(node.child, state.pushShader(node.effect), visit);
      return;
    case "Uniform": {
      const entries =
        node.bag.kind === "Static"
          ? node.bag.entries
          : AVal.force(node.bag.entries.content);
      forEachLeaf(node.child, state.pushUniforms(entries), visit);
      return;
    }
    case "BlendMode":
      forEachLeaf(node.child, state.pushBlendMode(node.mode), visit);
      return;
    case "Cursor":
      forEachLeaf(node.child, state.pushCursor(node.cursor), visit);
      return;
    case "PickThrough":
      forEachLeaf(node.child, state.pushPickThrough(node.value), visit);
      return;
    case "PixelSnapRadius":
      forEachLeaf(node.child, state.pushPixelSnapRadius(node.radius), visit);
      return;
    case "On":
      forEachLeaf(node.child, state.pushHandlers(node.handlers), visit);
      return;
    case "Active":
      forEachLeaf(node.child, state.pushActive(node.active), visit);
      return;
    case "View":
      forEachLeaf(node.child, state.withCamera(node.view, state.proj), visit);
      return;
    case "Proj":
      forEachLeaf(node.child, state.withCamera(state.view, node.proj), visit);
      return;
    case "Delay":
      // Run the creator with the accumulated state and recurse
      // into whatever sub-tree it returns — same state.
      forEachLeaf(node.create(state), state, visit);
      return;
  }
}

/** Convenience: collect every leaf + its final TraversalState into an array. */
export function collectLeaves(
  node: SgNode,
  state: TraversalState = TraversalState.empty,
): Array<{ leaf: SgLeaf; state: TraversalState }> {
  const out: Array<{ leaf: SgLeaf; state: TraversalState }> = [];
  forEachLeaf(node, state, (leaf, leafState) => out.push({ leaf, state: leafState }));
  return out;
}

/** Count leaves in the static tree. Useful for sanity tests. */
export function countLeaves(node: SgNode): number {
  let n = 0;
  forEachLeaf(node, TraversalState.empty, () => { n++; });
  return n;
}
