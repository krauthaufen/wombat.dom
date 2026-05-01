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
} from "./pickShaders.js";

export {
  chooseChain,
  composePickChain,
  type PickFinalTag,
  type PickChainChoice,
} from "./pickChain.js";

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

export {
  PickDispatcher,
  type ReadRegion,
} from "./dispatcher.js";
