// Scene-node kind registry.
//
// (This file used to also host the `sgVNode` / `isSgVNode` /
// `extractSgNode` carrier-VNode bridge that smuggled `SgNode`s through
// the JSX children pipeline. That's gone — `<Sg ...>` / `<Sg.Box/>`
// now lower straight to `SgNode`s via the jsx-runtime's `__isSg`
// dispatch, so there's nothing to wrap or unwrap. Only `SG_KINDS`
// remains, used by `collectSgChildren` to recognise a raw scene node
// among JSX children.)

/**
 * Set of `SgNode` `kind` values — `collectSgChildren` uses it to
 * recognise a raw scene node passed as a JSX child.
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
