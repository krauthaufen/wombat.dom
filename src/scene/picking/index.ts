// Pick-chain shader infrastructure. See `pickChain.ts` for the
// chooser + composer; `pickShaders.ts` for the seven Effects;
// `normal24.ts` for the 12+12 octahedron normal codec used by the
// mode-A fragments (and host-side decoders).

export {
  n24DecodeI32, n24EncodeI32,
  n24DecodeF32, n24EncodeF32,
  n24ShaderHelpers,
  N24_BITS_PER_AXIS,
} from "./normal24.js";

export {
  viewSpaceNormalVertexEffect,
  pickDepthBeforeEffect,
  pickFinalAEffect,
  pickFinalANoPiEffect,
  pickFinalANoNormalEffect,
  pickFinalANoNormalNoPiEffect,
  pickFinalBEffect,
  pickFinalPortalEffect,
} from "./pickShaders.js";

export {
  chooseChain,
  composePickChain,
  type PickFinalTag,
  type PickChainChoice,
} from "./pickChain.js";

export { arbitratePick, resolveThroughPortals } from "./pickArbitrate.js";

export {
  PickRegistry,
  PICK_ID_MAX,
  type PickId,
  type LeafPickScope,
} from "./registry.js";

export {
  SceneEvent,
  type SceneEventKind,
  type SceneEventInit,
  type SceneEventDispatch,
} from "./sceneEvent.js";

export { SceneEventLocation } from "./sceneEventLocation.js";

export type { EventHandlers, SceneEventHandler } from "../sg.js";

export {
  readPickPixel,
  readPickRegion,
  readSlotsAt,
  decodePick,
  type PickPixel,
  type DecodedPick,
  type PickRegion,
} from "./readback.js";

export {
  SNAP_OFFSETS,
  SNAP_RADIUS_MAX,
  SNAP_REGION_SIZE,
  type SnapOffset,
} from "./snapOffsets.js";

export {
  createPickFramebuffer,
  type PickFramebuffer,
  type CanvasLikeAttachment,
} from "./pickFramebuffer.js";

export {
  createPickResolveCompute,
  buildPickResolveWgsl,
  majorityVoteReference,
  PICK_RESOLVE_WORKGROUP_X,
  PICK_RESOLVE_WORKGROUP_Y,
  type PickResolveCompute,
} from "./pickResolveCompute.js";

export type {
  IPickSubContext,
  IRenderPickContext,
  PortalPickHit,
} from "./pickContext.js";

export {
  createPickProducer,
  type PickProducer,
  type PickProducerOptions,
} from "./pickProducer.js";

export {
  PickDispatcher,
  TAP_MAX_DURATION_MS,
  TAP_MAX_MOVE_PX,
  DOUBLE_TAP_GAP_MS,
  DOUBLE_TAP_MOVE_PX,
  LONG_PRESS_MS,
  DRAG_THRESHOLD_PX,
  HOVER_DELAY_MS,
  type ResolvePixel,
  type TapThresholds,
} from "./dispatcher.js";
