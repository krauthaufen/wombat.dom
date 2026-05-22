// Order-independent transparency as a wrapping IRenderTask — the wombat analog
// of aardvark's TransparencyRenderTask.WrappedTask.
//
// Given a scene tagged with `Sg.transparent` / `Sg.opaque`, `transparencyTask`
// lowers it (via compileScene's passFilter / composeEffect / pipelineOverride /
// injectStorage / injectUniforms hooks) into a multi-pass pipeline and returns a
// drop-in IRenderTask. Two modes:
//
//   "wboit"   — weighted-blended OIT (approximate, cheap, the default). Identical
//               math to aardvark's WeightedBlendedOIT.
//   "abuffer" — exact, lock-free per-pixel linked list (atomicAdd/atomicExchange),
//               sorted + composited in the resolve. Depth-occluded fragments are
//               dropped against the opaque depth at resolve time.
//
// Framebuffers are reactive (allocated against `size: aval`); resize re-allocates.
import { AList, AVal, AdaptiveToken, HashMap, type aval } from "@aardworx/wombat.adaptive";
import { V2f, V3f, V4f } from "@aardworx/wombat.base";
import {
  IBuffer, ITexture, ISampler, RenderTree, ElementType, PipelineState, asAttributeProvider,
  type BufferView, type Command, type DrawCall, type RenderObject, type IFramebuffer,
  type IRenderTask, type FramebufferSignature, type BlendState, type BlendComponentState, type PlainBlendState,
} from "@aardworx/wombat.rendering/core";
import { allocateFramebuffer, createFramebufferSignature, TextureUsage } from "@aardworx/wombat.rendering/resources";
import type { Runtime } from "@aardworx/wombat.rendering/runtime";
import { effect, fragment, vertex, type Effect } from "@aardworx/wombat.shader";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import {
  clamp, max, texture, atomicAdd, atomicExchange,
  type Sampler2D, type Storage, type FragmentBuiltinIn, type u32,
} from "@aardworx/wombat.shader/types";
import { compileScene, type CompileSceneOptions } from "./compile.js";
import { RenderPass, type SgNode } from "./sg.js";

declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope { readonly u_invSize: V2f; readonly u_width: u32; }
}

const MAXN = 1 << 20;        // A-buffer node-pool capacity (~24 MB)
const NULLU = 0xffffffff;
const DEPTH_Q = 16777215.0;  // depth quantisation (24-bit) for the A-buffer sort

const RAST = { topology: "triangle-list" as GPUPrimitiveTopology, cullMode: "none" as GPUCullMode, frontFace: "ccw" as GPUFrontFace };
const FS_QUAD = new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]);
const fsVS = vertex((v: { a_pos: V2f }) => ({ gl_Position: new V4f(v.a_pos.x, v.a_pos.y, 0.0, 1.0) }));

// ---------------------------------------------------------------- WBOIT shaders
const wboitWriter: Effect = effect(fragment((i: { Colors: V4f }, b: FragmentBuiltinIn) => {
  const c = i.Colors;
  const alpha = c.w;
  const a = alpha * 8.0 + 0.01;
  const bz = b.fragCoord.z * -0.95 + 1.0;
  const w = clamp(a * a * a * 1e8 * (bz * bz * bz), 1e-2, 3e2);
  return { accum: new V4f(c.xyz.mul(alpha), alpha).mul(w), reveal: alpha };
}));

const tAccum: Sampler2D = null as unknown as Sampler2D;
const tReveal: Sampler2D = null as unknown as Sampler2D;
// Over-blend composite: emits premultiplied transparent color (avg) with the
// transparent coverage (ta) as alpha; the pipeline blends it OVER the opaque
// already in the target (src-alpha / one-minus-src-alpha), so no opaque sampling.
// Two variants: plain, and one that also emits a (masked) pickId so the
// pipeline's color-target count matches a {Colors, pickId} output.
const wboitCompositeColor: Effect = effect(fsVS, fragment((_in: {}, b: FragmentBuiltinIn) => {
  const uv = b.fragCoord.xy.mul(uniform.u_invSize);
  const accum = texture(tAccum, uv);
  const reveal = texture(tReveal, uv).x;
  const avg = accum.xyz.div(max(accum.w, 1e-5));
  return { Colors: new V4f(avg, 1.0 - reveal) };
}));
const wboitCompositePick: Effect = effect(fsVS, fragment((_in: {}, b: FragmentBuiltinIn) => {
  const uv = b.fragCoord.xy.mul(uniform.u_invSize);
  const accum = texture(tAccum, uv);
  const reveal = texture(tReveal, uv).x;
  const avg = accum.xyz.div(max(accum.w, 1e-5));
  return { Colors: new V4f(avg, 1.0 - reveal), pickId: new V4f(0.0, 0.0, 0.0, 0.0) };
}));

function wboitPipelineOverride(ps: PipelineState): PipelineState {
  const bc = (s: GPUBlendFactor, d: GPUBlendFactor): BlendComponentState => ({
    operation: AVal.constant<GPUBlendOperation>("add"), srcFactor: AVal.constant(s), dstFactor: AVal.constant(d),
  });
  const blends: aval<HashMap<string, BlendState>> = AVal.constant(
    HashMap.empty<string, BlendState>()
      .add("Colors", { writeMask: AVal.constant(0) }) // ignored when no Colors attachment (non-MSAA); masks it in the combined MSAA FBO
      .add("accum", { color: bc("one", "one"), alpha: bc("one", "one"), writeMask: AVal.constant(0xf) })
      .add("reveal", { color: bc("zero", "one-minus-src"), alpha: bc("zero", "one-minus-src"), writeMask: AVal.constant(0xf) }),
  );
  return { ...ps, depth: { write: AVal.constant(false), compare: AVal.constant<GPUCompareFunction>("less"), clamp: ps.depth?.clamp ?? AVal.constant(false) }, blends };
}

// Transparent pick pass: depth-test + write ON (nearest transparent wins), color
// masked (write-mask-only — leaves the composited Colors alone), pickId (and
// depth) written. Mirrors aardvark's transformTransparentPick.
function wboitPickOverride(ps: PipelineState): PipelineState {
  const blends: aval<HashMap<string, BlendState>> = AVal.constant(
    HashMap.empty<string, BlendState>().add("Colors", { writeMask: AVal.constant(0) }),
  );
  return { ...ps, depth: { write: AVal.constant(true), compare: AVal.constant<GPUCompareFunction>("less"), clamp: ps.depth?.clamp ?? AVal.constant(false) }, blends };
}

// ------------------------------------------------------------- A-buffer shaders
// Opaque pass also writes its window-space depth into `odepth` so the resolve can
// occlude transparent fragments behind the opaque surface (fragment storage
// writes ignore the depth test, so build inserts everything; occlude at resolve).
const odepthWriter: Effect = effect(fragment((_in: {}, b: FragmentBuiltinIn) => ({ odepth: b.fragCoord.z })));

declare const headBuf: Storage<u32[], "read_write">;
declare const counterBuf: Storage<u32[], "read_write">;
declare const nodeDepth: Storage<u32[], "read_write">;
declare const nodeColor: Storage<V4f[], "read_write">;
declare const nodeNext: Storage<u32[], "read_write">;
const abBuildWriter: Effect = effect(fragment((i: { Colors: V4f }, b: FragmentBuiltinIn) => {
  const px = ((b.fragCoord.y as u32) * uniform.u_width + (b.fragCoord.x as u32)) as u32;
  const n = atomicAdd(counterBuf[0] as u32, 1 as u32);
  if (n < (MAXN as u32)) {
    nodeDepth[n] = (b.fragCoord.z * DEPTH_Q) as u32;
    nodeColor[n] = i.Colors;
    nodeNext[n] = atomicExchange(headBuf[px] as u32, n);
  }
  return { Colors: i.Colors, odepth: b.fragCoord.z };  // masked by pipelineOverride
}));

const tOpaque: Sampler2D = null as unknown as Sampler2D;
const tODepth: Sampler2D = null as unknown as Sampler2D;
declare const headBufR: Storage<u32[], "read">;
declare const nodeDepthR: Storage<u32[], "read">;
declare const nodeColorR: Storage<V4f[], "read">;
declare const nodeNextR: Storage<u32[], "read">;
const abResolveEffect: Effect = effect(fsVS, fragment((_in: {}, b: FragmentBuiltinIn) => {
  const uv = b.fragCoord.xy.mul(uniform.u_invSize);
  const opaque = texture(tOpaque, uv);
  const odepthQ = (texture(tODepth, uv).x * DEPTH_Q) as u32;
  const px = ((b.fragCoord.y as u32) * uniform.u_width + (b.fragCoord.x as u32)) as u32;
  let outRGB = new V3f(0.0, 0.0, 0.0);
  let T = 1.0;
  let lastD: u32 = 0 as u32;
  for (let layer = 0; layer < 16; layer = layer + 1) {
    // nearest node with depth strictly > lastD and in front of the opaque surface
    let bestD: u32 = NULLU as u32;
    let bestI: u32 = 0 as u32;
    let found = 0.0;
    let it: u32 = headBufR[px] as u32;
    for (let step = 0; step < 32; step = step + 1) {
      if (it !== (NULLU as u32)) {
        const d = nodeDepthR[it] as u32;
        if (d > lastD && d < bestD && d < odepthQ) { bestD = d; bestI = it; found = 1.0; }
        it = nodeNextR[it] as u32;
      }
    }
    if (found > 0.5) {
      const src = nodeColorR[bestI] as V4f;
      outRGB = outRGB.add(src.xyz.mul(src.w).mul(T));   // front-to-back over
      T = T * (1.0 - src.w);
      lastD = bestD;
    }
  }
  // pickId is emitted (dummy) so the pipeline matches a {Colors, pickId}
  // output; it's masked write-mask-only when present, and dropped otherwise.
  return { Colors: new V4f(outRGB.add(opaque.xyz.mul(T)), 1.0), pickId: new V4f(0.0, 0.0, 0.0, 0.0) };
}));

function abBuildPipelineOverride(ps: PipelineState): PipelineState {
  const off = (): BlendComponentState => ({ operation: AVal.constant<GPUBlendOperation>("add"), srcFactor: AVal.constant<GPUBlendFactor>("one"), dstFactor: AVal.constant<GPUBlendFactor>("zero") });
  const masked: BlendState = { color: off(), alpha: off(), writeMask: AVal.constant(0) };
  const blends: aval<HashMap<string, BlendState>> = AVal.constant(
    HashMap.empty<string, BlendState>().add("Colors", masked).add("odepth", masked),
  );
  return { ...ps, depth: { write: AVal.constant(false), compare: AVal.constant<GPUCompareFunction>("always"), clamp: ps.depth?.clamp ?? AVal.constant(false) }, blends };
}

// ---------------------------------------------------------------- shared bits
type CompileBase = Pick<CompileSceneOptions, "defaultEffect" | "rasterizer" | "autoUniforms" | "picking" | "initialState">;
const provider = <T>(m: HashMap<string, T>): unknown => m;
const nearestSampler = () => AVal.constant(ISampler.fromDescriptor({ magFilter: "nearest", minFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" }));

export type OitMode = "wboit" | "abuffer";

// Process-wide default OIT mode. Apps can flip this once (a mutable global) to
// switch every transparencyTask between approximate (wboit) and exact (abuffer)
// without threading an option through their scene code. A per-task `mode` option
// still overrides it.
let defaultOitMode: OitMode = "wboit";
/** Set the global default OIT mode used when `transparencyTask` is called without
 *  an explicit `mode`. */
export function setOitMode(mode: OitMode): void { defaultOitMode = mode; }
/** The current global default OIT mode. */
export function getOitMode(): OitMode { return defaultOitMode; }

export interface TransparencyTaskOptions {
  /** Override the global default ("wboit" approximate, or "abuffer" exact). */
  readonly mode?: OitMode;
  /** Forwarded to compileScene (defaultEffect, rasterizer, autoUniforms). */
  readonly compile?: CompileBase;
  /** Clear color for the opaque pass. Default opaque black. */
  readonly clearColor?: V4f;
  /** MSAA sample count for the internal OIT passes (1 = off). When > 1, opaque +
   *  transparent render multisampled into one resolved FBO and the composite
   *  samples the resolve; the output framebuffer stays single-sample. (WBOIT
   *  only; picking is not produced in the MSAA path.) */
  readonly sampleCount?: number;
}

/**
 * Build an `IRenderTask` that renders `scene` with order-independent
 * transparency. The output framebuffer's `Colors` attachment receives the result.
 */
export function transparencyTask(
  runtime: Runtime,
  device: GPUDevice,
  signature: FramebufferSignature,
  size: aval<{ width: number; height: number }>,
  scene: SgNode,
  opts: TransparencyTaskOptions = {},
): IRenderTask {
  const mode = opts.mode ?? defaultOitMode;
  // `picking` (the pick registry / chain) is threaded only to the opaque + pick
  // passes — NOT the WBOIT/build pass (which composes its own writer). `base`
  // therefore excludes it; opaque/pick passes spread `withPicking` back in.
  const { picking, ...compileRest } = opts.compile ?? {};
  const base: CompileSceneOptions = { autoUniforms: false, ...compileRest };
  const withPicking = picking !== undefined ? { picking } : {};
  const black = opts.clearColor ?? new V4f(0, 0, 0, 1);
  const quadBuf = AVal.constant(IBuffer.fromHost(FS_QUAD));
  const quadAttrs = asAttributeProvider(HashMap.empty<string, BufferView>().add("a_pos", { buffer: quadBuf, offset: 0, stride: 8, elementType: ElementType.V2f }));
  const quadDraw = AVal.constant<DrawCall>({ kind: "non-indexed", vertexCount: 6, instanceCount: 1, firstVertex: 0, firstInstance: 0 });
  const invSize = AVal.map(size, (s) => new V2f(1 / s.width, 1 / s.height));
  const widthU = AVal.map(size, (s) => s.width);
  const isOpaque = (p: number) => p < RenderPass.transparent;
  const isTransparent = (p: number) => p >= RenderPass.transparent;
  // The OIT passes inherit the sample count of the framebuffer handed to run():
  // a multisampled output ⇒ multisampled OIT passes, and the *owner* of the
  // framebuffer resolves (canvas resolves color, RenderControl's compute resolves
  // the pick ids by majority vote).
  const sampleCount = signature.sampleCount;

  if (mode === "wboit") {
    // Opaque renders straight into the output (so its pick + depth are there);
    // the transparent pass is the only offscreen part. The composite blends OVER
    // the opaque already in the output (no opaque sampling). When the output has
    // a pickId attachment we also re-render the transparent objects into it
    // (depth-tested) so the nearest transparent is pickable — aardvark's pattern.
    const hasPick = signature.colorNames.includes("pickId");
    // accum/reveal at the output's sample count; for MSAA they resolve to single
    // sample (colorTextures) which the composite samples and blends per output sample.
    const oitColorSig = createFramebufferSignature({ colors: { accum: "rgba16float", reveal: "r16float" }, sampleCount });
    const oitColorRes = allocateFramebuffer(device, oitColorSig, size, { extraUsage: TextureUsage.TEXTURE_BINDING });
    oitColorRes.acquire();
    const oitSig = createFramebufferSignature({ colors: { accum: "rgba16float", reveal: "r16float" }, depthStencil: { format: "depth32float" }, sampleCount });

    let opaqueClear = HashMap.empty<string, V4f>().add("Colors", black);
    if (hasPick) opaqueClear = opaqueClear.add("pickId", new V4f(0, 0, 0, 0));
    const opaqueTask = runtime.compile(signature, compileScene(scene, {
      ...base, ...withPicking, passFilter: isOpaque, clear: { colors: opaqueClear, depth: 1.0 },
    }));
    const wboitTask = runtime.compile(oitSig, compileScene(scene, {
      ...base, passFilter: isTransparent, composeEffect: (e) => effect(e, wboitWriter), pipelineOverride: wboitPipelineOverride,
      clear: { colors: HashMap.empty<string, V4f>().add("accum", new V4f(0, 0, 0, 0)).add("reveal", new V4f(1, 1, 1, 1)) },
    }));

    let compBlends = HashMap.empty<string, PlainBlendState>().add("Colors", {
      color: { operation: "add", srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
      alpha: { operation: "add", srcFactor: "zero", dstFactor: "one" }, writeMask: 0xf,
    });
    if (hasPick) compBlends = compBlends.add("pickId", { writeMask: 0 }); // write-mask-only: leave opaque pick intact
    const compRO: RenderObject = {
      effect: hasPick ? wboitCompositePick : wboitCompositeColor,
      pipelineState: PipelineState.constant({ rasterizer: RAST, depth: { write: false, compare: "always" }, blends: compBlends }),
      vertexAttributes: quadAttrs,
      uniforms: provider(HashMap.empty<string, aval<unknown>>().add("u_invSize", invSize)) as RenderObject["uniforms"],
      textures: provider(HashMap.empty<string, aval<ITexture>>()
        .add("tAccum_view", AVal.map(oitColorRes, (fb) => ITexture.fromGPU(fb.colorTextures!.tryFind("accum")!)))
        .add("tReveal_view", AVal.map(oitColorRes, (fb) => ITexture.fromGPU(fb.colorTextures!.tryFind("reveal")!)))) as RenderObject["textures"],
      samplers: provider(HashMap.empty<string, aval<ISampler>>().add("tAccum", nearestSampler()).add("tReveal", nearestSampler())) as RenderObject["samplers"],
      drawCall: quadDraw,
    };
    const compositeTask = runtime.compile(signature, AList.ofArray<Command>([{ kind: "Render", tree: RenderTree.leaf(compRO) }]));
    const pickTask = hasPick
      ? runtime.compile(signature, compileScene(scene, { ...base, ...withPicking, passFilter: isTransparent, pipelineOverride: wboitPickOverride }))
      : undefined;

    return {
      signature,
      run(userFb: IFramebuffer, token: AdaptiveToken): void {
        opaqueTask.run(userFb, token);
        const oc = oitColorRes.getValue(token);
        const oitFb: IFramebuffer = {
          signature: oitSig, colors: oc.colors,
          ...(oc.resolveColors !== undefined ? { resolveColors: oc.resolveColors } : {}),
          ...(oc.colorTextures !== undefined ? { colorTextures: oc.colorTextures } : {}),
          ...(userFb.depthStencil !== undefined ? { depthStencil: userFb.depthStencil } : {}),
          ...(userFb.depthStencilTexture !== undefined ? { depthStencilTexture: userFb.depthStencilTexture } : {}),
          width: userFb.width, height: userFb.height,
        };
        wboitTask.run(oitFb, token);
        compositeTask.run(userFb, token);
        if (pickTask !== undefined) pickTask.run(userFb, token);
      },
      dispose(): void {
        opaqueTask.dispose(); wboitTask.dispose(); compositeTask.dispose(); pickTask?.dispose();
        oitColorRes.release();
      },
    } as unknown as IRenderTask;
  }

  // ----- A-buffer (exact) -----
  const interSig = createFramebufferSignature({ colors: { Colors: "rgba16float", odepth: "r16float" }, depthStencil: { format: "depth32float" } });
  const interRes = allocateFramebuffer(device, interSig, size, { extraUsage: TextureUsage.TEXTURE_BINDING });
  interRes.acquire();

  // Device-owned node pool (fixed) + per-pixel head (re-allocated on resize).
  const su = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const counter = device.createBuffer({ size: 4, usage: su });
  const nDepth = device.createBuffer({ size: MAXN * 4, usage: su });
  const nColor = device.createBuffer({ size: MAXN * 16, usage: su });
  const nNext = device.createBuffer({ size: MAXN * 4, usage: su });
  const counterIB = AVal.constant(IBuffer.fromGPU(counter));
  const nDepthIB = AVal.constant(IBuffer.fromGPU(nDepth));
  const nColorIB = AVal.constant(IBuffer.fromGPU(nColor));
  const nNextIB = AVal.constant(IBuffer.fromGPU(nNext));
  let headBufGpu: GPUBuffer | undefined; let headIB: IBuffer | undefined; let headPx = 0; let clearArr = new Uint32Array(1);
  const zero1 = new Uint32Array([0]);
  const ensureHead = (px: number): void => {
    if (headBufGpu === undefined || px > headPx) {
      if (headBufGpu !== undefined) headBufGpu.destroy();
      headBufGpu = device.createBuffer({ size: Math.max(px, 1) * 4, usage: su });
      headIB = IBuffer.fromGPU(headBufGpu);
      headPx = px; clearArr = new Uint32Array(Math.max(px, 1)).fill(NULLU);
    }
  };
  const headAval = AVal.map(size, (s) => { ensureHead(s.width * s.height); return headIB!; });

  const buildStorage = HashMap.empty<string, aval<IBuffer>>()
    .add("headBuf", headAval).add("counterBuf", counterIB).add("nodeDepth", nDepthIB).add("nodeColor", nColorIB).add("nodeNext", nNextIB);
  const resolveStorage = HashMap.empty<string, aval<IBuffer>>()
    .add("headBufR", headAval).add("nodeDepthR", nDepthIB).add("nodeColorR", nColorIB).add("nodeNextR", nNextIB);

  const opaqueTask = runtime.compile(interSig, compileScene(scene, {
    ...base, passFilter: isOpaque, composeEffect: (e) => effect(e, odepthWriter),
    clear: { colors: HashMap.empty<string, V4f>().add("Colors", black).add("odepth", new V4f(1, 1, 1, 1)), depth: 1.0 },
  }));
  const buildTask = runtime.compile(interSig, compileScene(scene, {
    ...base, passFilter: isTransparent, composeEffect: (e) => effect(e, abBuildWriter),
    pipelineOverride: abBuildPipelineOverride, injectStorage: buildStorage,
    injectUniforms: HashMap.empty<string, aval<unknown>>().add("u_width", widthU),
  }));
  // When the output has a pickId attachment, the resolve writes Colors and
  // leaves pickId masked; opaque + transparent are rendered into pickId/depth
  // by dedicated pick passes (opaque rendered a second time, since A-buffer's
  // color path keeps the opaque in the intermediate it samples).
  const hasPick = signature.colorNames.includes("pickId");
  const resolveDepth = hasPick ? { write: false, compare: "always" as GPUCompareFunction } : undefined;
  const resolveBlends = hasPick
    ? HashMap.empty<string, PlainBlendState>().add("pickId", { writeMask: 0 })
    : undefined;
  const resolveRO: RenderObject = {
    effect: abResolveEffect,
    pipelineState: PipelineState.constant({ rasterizer: RAST, ...(resolveDepth !== undefined ? { depth: resolveDepth } : {}), ...(resolveBlends !== undefined ? { blends: resolveBlends } : {}) }),
    vertexAttributes: quadAttrs,
    uniforms: provider(HashMap.empty<string, aval<unknown>>().add("u_invSize", invSize).add("u_width", widthU)) as RenderObject["uniforms"],
    textures: provider(HashMap.empty<string, aval<ITexture>>()
      .add("tOpaque_view", AVal.map(interRes, (fb) => ITexture.fromGPU(fb.colorTextures!.tryFind("Colors")!)))
      .add("tODepth_view", AVal.map(interRes, (fb) => ITexture.fromGPU(fb.colorTextures!.tryFind("odepth")!)))) as RenderObject["textures"],
    samplers: provider(HashMap.empty<string, aval<ISampler>>().add("tOpaque", nearestSampler()).add("tODepth", nearestSampler())) as RenderObject["samplers"],
    storageBuffers: resolveStorage, drawCall: quadDraw,
  };
  const resolveTask = runtime.compile(signature, AList.ofArray<Command>([{ kind: "Render", tree: RenderTree.leaf(resolveRO) }]));

  // pick passes (only when the output carries pickId)
  let opaquePickClear = HashMap.empty<string, V4f>().add("Colors", black).add("pickId", new V4f(0, 0, 0, 0));
  const opaquePickTask = hasPick
    ? runtime.compile(signature, compileScene(scene, { ...base, ...withPicking, passFilter: isOpaque, pipelineOverride: wboitPickOverride, clear: { colors: opaquePickClear, depth: 1.0 } }))
    : undefined;
  const transparentPickTask = hasPick
    ? runtime.compile(signature, compileScene(scene, { ...base, ...withPicking, passFilter: isTransparent, pipelineOverride: wboitPickOverride }))
    : undefined;

  return {
    signature,
    run(userFb: IFramebuffer, token: AdaptiveToken): void {
      const s = size.getValue(token);
      ensureHead(s.width * s.height);
      device.queue.writeBuffer(headBufGpu!, 0, clearArr, 0, s.width * s.height); // reset linked-list heads
      device.queue.writeBuffer(counter, 0, zero1);                                // reset node allocator
      opaqueTask.run(interRes.getValue(token), token);
      if (opaquePickTask !== undefined) opaquePickTask.run(userFb, token); // opaque pick + depth into output
      buildTask.run(interRes.getValue(token), token);
      resolveTask.run(userFb, token);                                       // composite -> output Colors (pickId masked)
      if (transparentPickTask !== undefined) transparentPickTask.run(userFb, token);
    },
    dispose(): void {
      opaqueTask.dispose(); buildTask.dispose(); resolveTask.dispose();
      opaquePickTask?.dispose(); transparentPickTask?.dispose();
      interRes.release();
      counter.destroy(); nDepth.destroy(); nColor.destroy(); nNext.destroy();
      if (headBufGpu !== undefined) headBufGpu.destroy();
    },
  } as unknown as IRenderTask;
}
