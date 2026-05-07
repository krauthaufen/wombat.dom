// Symbol-tagged carrier VNode that smuggles `SgNode` values
// through the JSX children pipeline. Lets `<RenderControl>` accept
// scene trees as JSX children — `<RenderControl><Sg.Box/></RenderControl>` —
// without coupling the core VNode union to scene types.
//
// Usage:
//   - `sgVNode(node)` wraps an `SgNode` as a `FragmentVNode` whose
//     props carry the node under `SG_NODE_KEY`.
//   - The DOM mount path treats it as an empty fragment (no DOM
//     output) — render-anywhere safety: a stray `<Sg.Box/>` outside
//     `<RenderControl>` just disappears, doesn't render to text.
//   - The unified `Sg` namespace (in `constructors.ts`) walks JSX
//     children with `collectSgChildren` to extract the SgNodes.
//
// Kept minimal so it can be imported by both `sg.ts` (for the
// `SgNode` type) and the JSX-component layer in `constructors.ts`.

import type { FragmentVNode, Props } from "../vnode.js";
import { isVNode } from "../vnode.js";
import type { SgNode } from "./sg.js";

/** The Symbol property under which Fragment-carriers store their SgNode. */
export const SG_NODE_KEY: unique symbol = Symbol.for("wombat.dom.scene.sgNode");

interface SgFragmentProps extends Props {
  [SG_NODE_KEY]: SgNode;
}

/** Build a Fragment VNode that carries the given `SgNode` for later extraction. */
export function sgVNode(node: SgNode): FragmentVNode {
  const props: SgFragmentProps = { [SG_NODE_KEY]: node, children: [] };
  return { _tag: "fragment", props };
}

/** Cheap check: `value` is a Fragment carrier produced by `sgVNode`. */
export function isSgVNode(value: unknown): value is FragmentVNode {
  return (
    isVNode(value)
    && value._tag === "fragment"
    && SG_NODE_KEY in (value.props as object)
  );
}

/** Read the SgNode out of an `sgVNode`-tagged Fragment. */
export function extractSgNode(value: FragmentVNode): SgNode {
  return (value.props as SgFragmentProps)[SG_NODE_KEY];
}

/**
 * Set of `SgNode` `kind` values — used by `collectSgChildren` to
 * recognise raw scene nodes that snuck through JSX children
 * untagged (rare, but supported for code-generated trees).
 */
export const SG_KINDS: ReadonlySet<string> = new Set<string>([
  "Empty", "Group", "UnorderedGroup", "AdaptiveGroup", "Leaf",
  "Trafo", "Shader", "Uniform", "BlendMode", "Cursor", "PickThrough",
  "Intersectable", "PixelSnapRadius", "On", "Active", "View", "Proj", "Delay",
  // Phase 1
  "DepthTest", "DepthMask", "DepthBias", "DepthClamp",
  "CullMode", "FrontFace", "FillMode",
  "BlendConstant", "ColorMask", "StencilMode", "Pass",
  // Phase 2
  "VertexAttributes", "InstanceAttributes", "Index", "Mode",
  // Phase 3
  "NoEvents", "ForcePixelPicking", "CanFocus",
  // Auto-instancing
  "Instanced",
]);
