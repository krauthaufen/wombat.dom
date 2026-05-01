// @aardworx/wombat.dom/scene — adaptive 3D scene-graph layer.
//
// M2 surface:
//   - SgNode tagged-union data model
//   - immutable TraversalState with composition-by-attribute rules
//   - pure visitors over the static tree
//   - `Sg.*` constructors (data-only; no GPU)
//
// Importing from `@aardworx/wombat.dom/scene` is what pulls in
// wombat.rendering / wombat.shader / wombat.base via type-level
// dependencies declared as optional peer-deps. The DOM core
// (`@aardworx/wombat.dom`) does NOT load these modules — only
// consumers who actually import from `/scene` pay for them.

export type {
  SgNode,
  SgEmpty,
  SgGroup,
  SgUnorderedGroup,
  SgAdaptiveGroup,
  SgLeaf,
  SgTrafo,
  SgShader,
  SgUniform,
  SgBlendMode,
  SgCursor,
  SgPickThrough,
  SgOn,
  SgActive,
  SgView,
  SgProj,
  SgDelay,
  EventHandlers,
  SceneEventHandler,
  TrafoValue,
  UniformBag,
} from "./sg.js";

export {
  TraversalState,
  composeTrafoValue,
  composeModel,
} from "./traversalState.js";

export {
  forEachLeaf,
  collectLeaves,
  countLeaves,
} from "./visit.js";

export { Sg, collectSgChildren, type SgScopeProps } from "./constructors.js";
export { sgVNode, isSgVNode, extractSgNode, SG_NODE_KEY } from "./sgVNode.js";

export {
  compileScene,
  type CompileSceneOptions,
} from "./compile.js";

export {
  RenderControl,
  type RenderControlProps,
  type RenderControlReadyInfo,
} from "./renderControl.js";

export {
  type LookAtOptions,
  type PerspectiveOptions,
  type OrthographicOptions,
  lookAt,
  perspective,
  orthographic,
  aspectFromViewport,
} from "./camera.js";

export {
  FreeFlyController,
  FreeFlyConfigDefault,
  defaultFreeFlyState,
  freeFlyIsAnimating,
  OrbitController,
  OrbitConfigDefault,
  defaultOrbitState,
  deriveView,
  Anim,
  getParameter,
  interpolateV2,
  interpolateV3,
  type FreeFlyConfig,
  type FreeFlyState,
  type FreeFlyInitial,
  type FreeFlyAttachOptions,
  type OrbitConfig,
  type OrbitState,
  type OrbitInitial,
  type OrbitView,
  type OrbitAttachOptions,
  type AnimationKind,
  type Animation,
  type DragStart,
} from "./controllers/index.js";

export {
  box, quad,
  type BoxOptions, type QuadOptions,
} from "./primitives.js";

export {
  DefaultSurfaces,
  basic,
} from "./defaultSurfaces.js";

export {
  n24DecodeI32, n24EncodeI32,
  n24DecodeF32, n24EncodeF32,
  n24ShaderHelpers,
  N24_BITS_PER_AXIS,
  viewSpaceNormalVertexEffect,
  pickDepthBeforeEffect,
  pickFinalAEffect,
  pickFinalANoPiEffect,
  pickFinalANoNormalEffect,
  pickFinalANoNormalNoPiEffect,
  pickFinalBEffect,
  chooseChain,
  composePickChain,
  type PickFinalTag,
  type PickChainChoice,
  PickRegistry,
  PICK_ID_MAX,
  type PickId,
  type LeafPickScope,
} from "./picking/index.js";
