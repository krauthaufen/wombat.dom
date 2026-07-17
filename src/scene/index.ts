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

export {
  compileScene,
  __setRowLowering,
  type CompileSceneOptions,
} from "./compile.js";
export {
  stageNode, templateStats, resetTemplates,
  effectUniformNames, validateTemplateEffect, warnUnresolvedUniforms,
  sceneEfficiency, resetEfficiency,
  type EfficiencyReport, type RowBailReason,
  type SceneTemplate, type StagedNode, type TemplateStats,
} from "./template.js";

export {
  transparencyTask,
  setOitMode,
  getOitMode,
  type TransparencyTaskOptions,
  type OitMode,
} from "./transparency.js";

export {
  createGtaoPass,
  gtaoConfig,
  type GtaoOption,
  type GtaoPass,
  type GtaoSettings,
} from "./gtao.js";

export {
  RenderControl,
  type RenderControlProps,
  type RenderControlReadyInfo,
} from "./renderControl.js";

export {
  renderSceneTo,
  renderToPickable,
  type RenderSceneToOptions,
  type RenderSceneToResult,
  type RenderToPickableResult,
} from "./renderTo.js";

export type {
  IPickSubContext,
  IRenderPickContext,
  PortalPickHit,
} from "./picking/pickContext.js";

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

export { DefaultSurfaces } from "./defaultSurfaces.js";

export {
  SgText,
  type SgTextProps,
  type TextAlign,
  type TextAa,
} from "./text.js";

// Attach `<Sg.Text/>` onto the Sg namespace at module load. This is
// in `index.ts` rather than `constructors.ts` because `text.ts`
// itself imports from `constructors.ts`, so wiring it up there
// would be a cycle.
import { Sg as _SgForText } from "./constructors.js";
import { SgText as _SgTextImpl } from "./text.js";
(_SgForText as unknown as { Text: typeof _SgTextImpl }).Text = _SgTextImpl;
(_SgTextImpl as { __isSg?: boolean }).__isSg = true;

// Ambient context plumbing. The avals themselves live as static
// members on `RenderControl` (`RenderControl.viewport`, `.view`,
// `.proj`, `.time`); the helpers here let other layers (tests, etc.)
// publish/clear the context manually if they need to.
export {
  setAmbient, clearAmbient,
  type AmbientContext,
} from "./ambient.js";

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
  pickFinalPortalEffect,
  resolveThroughPortals,
  chooseChain,
  composePickChain,
  type PickFinalTag,
  type PickChainChoice,
  PickRegistry,
  PICK_ID_MAX,
  type PickId,
  type LeafPickScope,
  SceneEvent,
  SceneEventLocation,
  type SceneEventKind,
  type SceneEventInit,
  type SceneEventDispatch,
} from "./picking/index.js";

export type {
  CullValue, FrontFaceValue, FillModeValue,
  BlendConstantValue, ColorMaskValue, StencilModeValue,
  DepthBiasValue,
} from "./sg.js";
