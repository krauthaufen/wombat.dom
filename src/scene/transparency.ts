// Order-independent transparency as a wrapping IRenderTask — the wombat analog
// of aardvark's TransparencyRenderTask.WrappedTask.
//
// Given a scene tagged with `Sg.transparent` / `Sg.opaque`, `transparencyTask`
// lowers it (via compileScene's passFilter/composeEffect/pipelineOverride hooks)
// into three sub-tasks and sequences them per frame:
//   1. opaque pass     -> intermediate { Colors, depth }
//   2. WBOIT pass      -> { accum, reveal } sharing the opaque depth
//                         (depth-test, write off; accum additive, reveal mult)
//   3. composite pass  -> the output framebuffer (accum/reveal resolved over
//                         the opaque Colors).
// The composition is identical to aardvark's WeightedBlendedOIT.
import { AList, AVal, AdaptiveToken, HashMap, type aval } from "@aardworx/wombat.adaptive";
import { V2f, V4f } from "@aardworx/wombat.base";
import {
  IBuffer, ITexture, ISampler, RenderTree, ElementType, PipelineState, asAttributeProvider,
  type BufferView, type Command, type DrawCall, type RenderObject, type IFramebuffer,
  type IRenderTask, type FramebufferSignature, type BlendState, type BlendComponentState, type ClearValues,
} from "@aardworx/wombat.rendering/core";
import { allocateFramebuffer, createFramebufferSignature, TextureUsage } from "@aardworx/wombat.rendering/resources";
import type { Runtime } from "@aardworx/wombat.rendering/runtime";
import { effect, fragment, vertex, type Effect } from "@aardworx/wombat.shader";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { clamp, max, texture, type Sampler2D, type FragmentBuiltinIn } from "@aardworx/wombat.shader/types";
import { compileScene, type CompileSceneOptions } from "./compile.js";
import { RenderPass, type SgNode } from "./sg.js";

declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope { readonly u_invSize: V2f; }
}

// ---- WBOIT fragment writer, appended onto each transparent leaf's effect ----
// Reads the upstream `Colors` (straight color + alpha) and emits the
// weighted accumulation + revealage. Same weight as aardvark's WeightedBlendedOIT.
const wboitWriter: Effect = effect(fragment((i: { Colors: V4f }, b: FragmentBuiltinIn) => {
  const c = i.Colors;
  const alpha = c.w;
  const a = alpha * 8.0 + 0.01;
  const bz = b.fragCoord.z * -0.95 + 1.0;
  const w = clamp(a * a * a * 1e8 * (bz * bz * bz), 1e-2, 3e2);
  return {
    accum: new V4f(c.xyz.mul(alpha), alpha).mul(w),
    reveal: alpha,
  };
}));

// ---- composite effect: resolve accum/reveal over the opaque Colors ----
const tColors: Sampler2D = null as unknown as Sampler2D;
const tAccum: Sampler2D = null as unknown as Sampler2D;
const tReveal: Sampler2D = null as unknown as Sampler2D;
const compositeEffect: Effect = effect(
  vertex((v: { a_pos: V2f }) => ({ gl_Position: new V4f(v.a_pos.x, v.a_pos.y, 0.0, 1.0) })),
  fragment((_in: {}, b: FragmentBuiltinIn) => {
    const uv = b.fragCoord.xy.mul(uniform.u_invSize);
    const opaque = texture(tColors, uv);
    const accum = texture(tAccum, uv);
    const reveal = texture(tReveal, uv).x;
    const avg = accum.xyz.div(max(accum.w, 1e-5));
    const ta = 1.0 - reveal;
    return { Colors: new V4f(avg.mul(ta).add(opaque.xyz.mul(1.0 - ta)), 1.0) };
  }),
);

const RAST = { topology: "triangle-list" as GPUPrimitiveTopology, cullMode: "none" as GPUCullMode, frontFace: "ccw" as GPUFrontFace };
const FS_QUAD = new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]);

// WBOIT pipeline override: depth-test (less) but write off; accum additive,
// reveal multiplicative. These are 16-float attachments (blendable).
function wboitPipelineOverride(ps: PipelineState): PipelineState {
  const bc = (s: GPUBlendFactor, d: GPUBlendFactor): BlendComponentState => ({
    operation: AVal.constant<GPUBlendOperation>("add"), srcFactor: AVal.constant(s), dstFactor: AVal.constant(d),
  });
  const blends: aval<HashMap<string, BlendState>> = AVal.constant(
    HashMap.empty<string, BlendState>()
      .add("accum", { color: bc("one", "one"), alpha: bc("one", "one"), writeMask: AVal.constant(0xf) })
      .add("reveal", { color: bc("zero", "one-minus-src"), alpha: bc("zero", "one-minus-src"), writeMask: AVal.constant(0xf) }),
  );
  const prevClamp = ps.depth?.clamp ?? AVal.constant(false);
  return {
    ...ps,
    depth: { write: AVal.constant(false), compare: AVal.constant<GPUCompareFunction>("less"), clamp: prevClamp },
    blends,
  };
}

export interface TransparencyTaskOptions {
  /** Forwarded to compileScene (defaultEffect, rasterizer, autoUniforms, …). */
  readonly compile?: Pick<CompileSceneOptions, "defaultEffect" | "rasterizer" | "autoUniforms" | "picking">;
  /** Clear color for the opaque/intermediate pass. Default opaque black. */
  readonly clearColor?: V4f;
}

/**
 * Build an `IRenderTask` that renders `scene` with order-independent
 * transparency (WBOIT). Fixed-size: `size` must match the framebuffer passed to
 * `run`. The output framebuffer's first/`Colors` attachment receives the result.
 */
export function transparencyTask(
  runtime: Runtime,
  device: GPUDevice,
  signature: FramebufferSignature,
  size: { width: number; height: number },
  scene: SgNode,
  opts: TransparencyTaskOptions = {},
): IRenderTask {
  const { width: W, height: H } = size;
  const base: CompileSceneOptions = { autoUniforms: false, ...opts.compile };

  // intermediate { Colors, depth }
  const interSig = createFramebufferSignature({ colors: { Colors: "rgba16float" }, depthStencil: { format: "depth32float" } });
  const interRes = allocateFramebuffer(device, interSig, AVal.constant(size), { extraUsage: TextureUsage.TEXTURE_BINDING });
  interRes.acquire();
  const interFb = interRes.getValue(AdaptiveToken.top);

  // OIT { accum, reveal } sharing the intermediate depth
  const accumTex = device.createTexture({ size: [W, H], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const revealTex = device.createTexture({ size: [W, H], format: "r16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const oitSig = createFramebufferSignature({ colors: { accum: "rgba16float", reveal: "r16float" }, depthStencil: { format: "depth32float" } });
  const oitFb: IFramebuffer = {
    signature: oitSig,
    colors: HashMap.empty<string, GPUTextureView>().add("accum", accumTex.createView()).add("reveal", revealTex.createView()),
    colorTextures: HashMap.empty<string, GPUTexture>().add("accum", accumTex).add("reveal", revealTex),
    depthStencil: interFb.depthStencil!,
    depthStencilTexture: interFb.depthStencilTexture!,
    width: W, height: H,
  };

  // sub-tasks
  const black = opts.clearColor ?? new V4f(0, 0, 0, 1);
  const opaqueClear: ClearValues = { colors: HashMap.empty<string, V4f>().add("Colors", black), depth: 1.0 };
  const opaqueTask = runtime.compile(interSig, compileScene(scene, {
    ...base, passFilter: (p) => p < RenderPass.transparent, clear: opaqueClear,
  }));
  const wboitClear: ClearValues = { colors: HashMap.empty<string, V4f>().add("accum", new V4f(0, 0, 0, 0)).add("reveal", new V4f(1, 1, 1, 1)) };
  const wboitTask = runtime.compile(oitSig, compileScene(scene, {
    ...base,
    passFilter: (p) => p >= RenderPass.transparent,
    composeEffect: (e) => effect(e, wboitWriter),
    pipelineOverride: wboitPipelineOverride,
    clear: wboitClear,
  }));

  // composite
  const quadBuf = AVal.constant(IBuffer.fromHost(FS_QUAD));
  const nearest = AVal.constant(ISampler.fromDescriptor({ magFilter: "nearest", minFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" }));
  const compRO: RenderObject = {
    effect: compositeEffect,
    pipelineState: PipelineState.constant({ rasterizer: RAST }),
    vertexAttributes: asAttributeProvider(HashMap.empty<string, BufferView>().add("a_pos", { buffer: quadBuf, offset: 0, stride: 8, elementType: ElementType.V2f })),
    uniforms: HashMap.empty<string, unknown>().add("u_invSize", AVal.constant(new V2f(1 / W, 1 / H))) as unknown as RenderObject["uniforms"],
    textures: HashMap.empty<string, unknown>()
      .add("tColors_view", AVal.constant(ITexture.fromGPU(interFb.colorTextures!.tryFind("Colors")!)))
      .add("tAccum_view", AVal.constant(ITexture.fromGPU(accumTex)))
      .add("tReveal_view", AVal.constant(ITexture.fromGPU(revealTex))) as unknown as RenderObject["textures"],
    samplers: HashMap.empty<string, unknown>().add("tColors", nearest).add("tAccum", nearest).add("tReveal", nearest) as unknown as RenderObject["samplers"],
    drawCall: AVal.constant<DrawCall>({ kind: "non-indexed", vertexCount: 6, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
  };
  const compositeTask = runtime.compile(signature, AList.ofArray<Command>([{ kind: "Render", tree: RenderTree.leaf(compRO) }]));

  return {
    signature,
    run(userFb: IFramebuffer, token: AdaptiveToken): void {
      opaqueTask.run(interFb, token);
      wboitTask.run(oitFb, token);
      compositeTask.run(userFb, token);
    },
    dispose(): void {
      opaqueTask.dispose();
      wboitTask.dispose();
      compositeTask.dispose();
      interRes.release();
      accumTex.destroy();
      revealTex.destroy();
    },
  } as unknown as IRenderTask;
}
