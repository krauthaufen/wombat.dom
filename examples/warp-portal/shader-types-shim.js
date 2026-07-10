// Dev-server shim for "@aardworx/wombat.shader/types".
//
// The intrinsic helpers (`abs`, `sin`, `texture`, ...) are marker-only:
// they exist for the TS type layer and are erased by the wombatShader
// vite transform wherever they appear INSIDE vertex()/fragment()
// bodies -- but the import statements survive, and the package's
// runtime module doesn't export them (intrinsics.js is `export {}`).
// Rollup builds tolerate that; the vite DEV server's strict ESM
// linking does not. This shim re-exports the real module and adds
// inert stand-ins for every declared intrinsic so dev-serving the
// workspace src works. Generated from the package's
// `declare function` names.
//
// Calling one of these outside a shader body throws, same contract as
// the real (nonexistent) implementation.

// Relative path -- the package's exports map doesn't allow the deep
// specifier form.
export * from "../../node_modules/@aardworx/wombat.shader/dist/types/index.js";

const marker = (name) => () => {
  throw new Error(`wombat.shader intrinsic "${name}" called at runtime -- only valid inside vertex()/fragment() bodies`);
};

export const abs = marker("abs");
export const acos = marker("acos");
export const acosh = marker("acosh");
export const asin = marker("asin");
export const asinh = marker("asinh");
export const atan = marker("atan");
export const atan2 = marker("atan2");
export const atanh = marker("atanh");
export const atomicAdd = marker("atomicAdd");
export const atomicAnd = marker("atomicAnd");
export const atomicCompareExchangeWeak = marker("atomicCompareExchangeWeak");
export const atomicExchange = marker("atomicExchange");
export const atomicLoad = marker("atomicLoad");
export const atomicMax = marker("atomicMax");
export const atomicMin = marker("atomicMin");
export const atomicOr = marker("atomicOr");
export const atomicStore = marker("atomicStore");
export const atomicSub = marker("atomicSub");
export const atomicXor = marker("atomicXor");
export const ceil = marker("ceil");
export const clamp = marker("clamp");
export const cos = marker("cos");
export const cosh = marker("cosh");
export const countOneBits = marker("countOneBits");
export const degrees = marker("degrees");
export const dFdx = marker("dFdx");
export const dFdxCoarse = marker("dFdxCoarse");
export const dFdxFine = marker("dFdxFine");
export const dFdy = marker("dFdy");
export const dFdyCoarse = marker("dFdyCoarse");
export const dFdyFine = marker("dFdyFine");
export const discard = marker("discard");
export const exp = marker("exp");
export const exp2 = marker("exp2");
export const extractBits = marker("extractBits");
export const faceforward = marker("faceforward");
export const firstLeadingBit = marker("firstLeadingBit");
export const firstTrailingBit = marker("firstTrailingBit");
export const floor = marker("floor");
export const fract = marker("fract");
export const fwidth = marker("fwidth");
export const fwidthCoarse = marker("fwidthCoarse");
export const fwidthFine = marker("fwidthFine");
export const insertBits = marker("insertBits");
export const inversesqrt = marker("inversesqrt");
export const log = marker("log");
export const log2 = marker("log2");
export const max = marker("max");
export const min = marker("min");
export const mix = marker("mix");
export const mod = marker("mod");
export const pack2x16float = marker("pack2x16float");
export const pack2x16snorm = marker("pack2x16snorm");
export const pack2x16unorm = marker("pack2x16unorm");
export const pack4x8snorm = marker("pack4x8snorm");
export const pack4x8unorm = marker("pack4x8unorm");
export const pow = marker("pow");
export const radians = marker("radians");
export const reflect = marker("reflect");
export const refract = marker("refract");
export const reverseBits = marker("reverseBits");
export const round = marker("round");
export const sign = marker("sign");
export const sin = marker("sin");
export const sinh = marker("sinh");
export const smoothstep = marker("smoothstep");
export const sqrt = marker("sqrt");
export const step = marker("step");
export const storageBarrier = marker("storageBarrier");
export const tan = marker("tan");
export const tanh = marker("tanh");
export const texelFetch = marker("texelFetch");
export const texture = marker("texture");
export const textureGrad = marker("textureGrad");
export const textureLoad = marker("textureLoad");
export const textureLod = marker("textureLod");
export const textureSampleCompare = marker("textureSampleCompare");
export const textureSize = marker("textureSize");
export const textureStore = marker("textureStore");
export const trunc = marker("trunc");
export const unpack2x16float = marker("unpack2x16float");
export const unpack2x16snorm = marker("unpack2x16snorm");
export const unpack2x16unorm = marker("unpack2x16unorm");
export const unpack4x8snorm = marker("unpack4x8snorm");
export const unpack4x8unorm = marker("unpack4x8unorm");
export const workgroupBarrier = marker("workgroupBarrier");
