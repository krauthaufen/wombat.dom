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
