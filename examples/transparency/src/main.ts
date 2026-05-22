// WBOIT z-stack test — wombat port of aardvark Transparency.zStackWithOccluder.
//
//   z=-0.8 A blue  a0.5 transparent   z=-0.3 B green a0.5 transparent
//   z= 0.0 SOLID red opaque
//   z= 0.3 C yellow a0.5 transparent (occluded)   z=0.7 D cyan a0.5 (occluded)
//
// FBO A {Colors16f, PickData32f, accum16f, reveal16f, depth32f}: clear ->
// opaque -> WBOIT transparent (depth-test, write off) -> pick pass. Composite
// (FBO B) samples A and resolves transparent over opaque.
import { AList, AdaptiveToken, AVal, HashMap, cval } from "@aardworx/wombat.adaptive";
import { V2f, V4f } from "@aardworx/wombat.base";
import {
  IBuffer, ITexture, ISampler, RenderTree, ElementType, PipelineState,
  type BufferView, type Command, type DrawCall, type RenderObject,
} from "@aardworx/wombat.rendering/core";
import { allocateFramebuffer, createFramebufferSignature, TextureUsage } from "@aardworx/wombat.rendering/resources";
import { Runtime } from "@aardworx/wombat.rendering/runtime";
import { effect, fragment, vertex } from "@aardworx/wombat.shader";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { clamp, max, texture, type Sampler2D, type FragmentBuiltinIn, type f32 } from "@aardworx/wombat.shader/types";

declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope { readonly u_z: f32; readonly u_color: V4f; readonly u_pickId: f32; }
}

const out = document.getElementById("out")!;
const log = (s: string) => { out.textContent += "\n" + s; };
const check = (name: string, ok: boolean, detail: string) => log(`[${ok ? "PASS" : "FAIL"}] ${name} — ${detail}`);
const SIZE = 8;

// ---- shaders ----
const mainVS = vertex((v: { a_pos: V2f }) => ({
  gl_Position: new V4f(v.a_pos.x, v.a_pos.y, uniform.u_z.mul(0.5).add(0.5), 1.0),
}));
const mainFS = fragment((_in: {}, b: FragmentBuiltinIn) => {
  const c = uniform.u_color;
  const alpha = c.w;
  const a = alpha.mul(8.0).add(0.01);
  const bz = b.fragCoord.z.mul(-0.95).add(1.0);
  const w = clamp(a.mul(a).mul(a).mul(1e8).mul(bz).mul(bz).mul(bz), 1e-2, 3e2);
  return {
    Colors: c,
    PickData: new V4f(uniform.u_pickId, b.fragCoord.z, 0.0, 0.0),
    accum: new V4f(c.xyz.mul(alpha), alpha).mul(w),
    reveal: alpha,
  };
});
const mainEffect = effect(mainVS, mainFS);

const tColors: Sampler2D = null as unknown as Sampler2D;
const tAccum: Sampler2D = null as unknown as Sampler2D;
const tReveal: Sampler2D = null as unknown as Sampler2D;
const compVS = vertex((v: { a_pos: V2f }) => ({ gl_Position: new V4f(v.a_pos.x, v.a_pos.y, 0.0, 1.0) }));
const compFS = fragment((_in: {}, b: FragmentBuiltinIn) => {
  const uv = new V2f(b.fragCoord.x.div(SIZE), b.fragCoord.y.div(SIZE));
  const opaque = texture(tColors, uv);
  const accum = texture(tAccum, uv);
  const reveal = texture(tReveal, uv).x;
  const avg = accum.xyz.div(max(accum.w, 1e-5));
  const ta = reveal.mul(-1.0).add(1.0);
  const rgb = avg.mul(ta).add(opaque.xyz.mul(ta.mul(-1.0).add(1.0)));
  return { finalColor: new V4f(rgb, 1.0) };
});
const compositeEffect = effect(compVS, compFS);

// ---- blends ----
type Comp = { operation: GPUBlendOperation; srcFactor: GPUBlendFactor; dstFactor: GPUBlendFactor };
const comp = (s: GPUBlendFactor, d: GPUBlendFactor): Comp => ({ operation: "add", srcFactor: s, dstFactor: d });
const blend = (color: Comp, alpha: Comp, writeMask: number) => ({ color, alpha, writeMask });
const masked = blend(comp("one", "zero"), comp("one", "zero"), 0);
const writeAll = blend(comp("one", "zero"), comp("one", "zero"), 0xf);
const additive = blend(comp("one", "one"), comp("one", "one"), 0xf);
const revealMul = blend(comp("zero", "one-minus-src"), comp("zero", "one-minus-src"), 0xf);

const RAST = { topology: "triangle-list" as GPUPrimitiveTopology, cullMode: "none" as GPUCullMode, frontFace: "ccw" as GPUFrontFace };
const FS_QUAD = new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]);

async function readPixel0(device: GPUDevice, tex: GPUTexture): Promise<Float32Array> {
  const bpr = Math.ceil((tex.width * 16) / 256) * 256;
  const buf = device.createBuffer({ size: bpr * tex.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: tex.height }, { width: tex.width, height: tex.height, depthOrArrayLayers: 1 });
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const f = new Float32Array(buf.getMappedRange().slice(0, 16));
  buf.unmap(); buf.destroy();
  return f;
}

async function main() {
  out.textContent = "init…";
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const device = await adapter.requestDevice({
    requiredLimits: { maxColorAttachmentBytesPerSample: adapter.limits.maxColorAttachmentBytesPerSample },
  });
  device.addEventListener("uncapturederror", (e) => log("GPU-ERROR: " + (e as GPUUncapturedErrorEvent).error.message));
  const runtime = new Runtime({ device });

  const quadBuf = AVal.constant(IBuffer.fromHost(FS_QUAD));
  const vattrs = HashMap.empty<string, BufferView>().add("a_pos", { buffer: quadBuf, offset: 0, stride: 8, elementType: ElementType.V2f });
  const draw = cval<DrawCall>({ kind: "non-indexed", vertexCount: 6, instanceCount: 1, firstVertex: 0, firstInstance: 0 });

  const quadRO = (z: number, color: V4f, pickId: number, blends: HashMap<string, ReturnType<typeof blend>>, depthWrite: boolean): RenderObject => ({
    effect: mainEffect,
    pipelineState: PipelineState.constant({ rasterizer: RAST, depth: { write: depthWrite, compare: "less" }, blends }),
    vertexAttributes: vattrs,
    uniforms: HashMap.empty<string, unknown>()
      .add("u_z", AVal.constant(z)).add("u_color", AVal.constant(color)).add("u_pickId", AVal.constant(pickId)) as unknown as RenderObject["uniforms"],
    textures: HashMap.empty(), samplers: HashMap.empty(), drawCall: draw,
  });

  const A = { z: -0.8, c: new V4f(0, 0, 1, 0.5), id: 2 };
  const Bq = { z: -0.3, c: new V4f(0, 1, 0, 0.5), id: 3 };
  const S = { z: 0.0, c: new V4f(1, 0, 0, 1.0), id: 1 };
  const Cq = { z: 0.3, c: new V4f(1, 1, 0, 0.5), id: 4 };
  const Dq = { z: 0.7, c: new V4f(0, 1, 1, 0.5), id: 5 };

  // PickData (rgba32float) is non-blendable, so it is NEVER placed in the blends
  // map (omitted => no blend, writeMask=all). It's written in every pass, but the
  // pick pass runs last with depth-test, so the final PickData is the nearest
  // transparent fragment regardless. Colors/accum/reveal are 16f (blendable).
  const opaqueBlends = HashMap.empty<string, ReturnType<typeof blend>>().add("accum", masked).add("reveal", masked);
  const wboitBlends = HashMap.empty<string, ReturnType<typeof blend>>().add("Colors", masked).add("accum", additive).add("reveal", revealMul);
  const pickBlends = HashMap.empty<string, ReturnType<typeof blend>>().add("Colors", masked).add("accum", masked).add("reveal", masked);
  void writeAll;

  const solidRO = quadRO(S.z, S.c, S.id, opaqueBlends, true);
  const wboitROs = [A, Bq, Cq, Dq].map((q) => quadRO(q.z, q.c, q.id, wboitBlends, false));
  const pickROs = [A, Bq, Cq, Dq].map((q) => quadRO(q.z, q.c, q.id, pickBlends, true));

  const sigA = createFramebufferSignature({
    colors: { Colors: "rgba16float", PickData: "rgba32float", accum: "rgba16float", reveal: "r16float" },
    depthStencil: { format: "depth32float" },
  });
  const fboA = allocateFramebuffer(device, sigA, cval({ width: SIZE, height: SIZE }), { extraUsage: TextureUsage.COPY_SRC });
  fboA.acquire();
  const taskA = runtime.compile(sigA, AList.ofArray<Command>([
    { kind: "Clear", values: { colors: HashMap.empty<string, V4f>()
        .add("Colors", new V4f(0, 0, 0, 1)).add("PickData", new V4f(0, 0, 0, 0))
        .add("accum", new V4f(0, 0, 0, 0)).add("reveal", new V4f(1, 1, 1, 1)), depth: 1.0 } },
    { kind: "Render", tree: RenderTree.leaf(solidRO) },
    { kind: "Render", tree: RenderTree.ordered(...wboitROs.map((r) => RenderTree.leaf(r))) },
    { kind: "Render", tree: RenderTree.ordered(...pickROs.map((r) => RenderTree.leaf(r))) },
  ]));
  taskA.run(fboA.getValue(AdaptiveToken.top), AdaptiveToken.top);
  await device.queue.onSubmittedWorkDone();
  log("pass A done");

  const ifbA = fboA.getValue(AdaptiveToken.top);
  const gpu = (n: string) => ifbA.colorTextures!.tryFind(n)! as unknown as GPUTexture;
  const nearest = AVal.constant(ISampler.fromDescriptor({ magFilter: "nearest", minFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" }));

  const sigB = createFramebufferSignature({ colors: { finalColor: "rgba32float" } });
  const fboB = allocateFramebuffer(device, sigB, cval({ width: SIZE, height: SIZE }), { extraUsage: TextureUsage.COPY_SRC });
  fboB.acquire();
  const compRO: RenderObject = {
    effect: compositeEffect,
    pipelineState: PipelineState.constant({ rasterizer: RAST }),
    vertexAttributes: vattrs,
    uniforms: HashMap.empty(),
    textures: HashMap.empty<string, unknown>()
      .add("tColors_view", AVal.constant(ITexture.fromGPU(gpu("Colors"))))
      .add("tAccum_view", AVal.constant(ITexture.fromGPU(gpu("accum"))))
      .add("tReveal_view", AVal.constant(ITexture.fromGPU(gpu("reveal")))) as unknown as RenderObject["textures"],
    samplers: HashMap.empty<string, unknown>().add("tColors", nearest).add("tAccum", nearest).add("tReveal", nearest) as unknown as RenderObject["samplers"],
    drawCall: draw,
  };
  const taskB = runtime.compile(sigB, AList.ofArray<Command>([
    { kind: "Clear", values: { colors: HashMap.empty<string, V4f>().add("finalColor", new V4f(0, 0, 0, 1)) } },
    { kind: "Render", tree: RenderTree.leaf(compRO) },
  ]));
  taskB.run(fboB.getValue(AdaptiveToken.top), AdaptiveToken.top);
  await device.queue.onSubmittedWorkDone();
  log("composite done");

  const pick = await readPixel0(device, gpu("PickData"));
  const fin = await readPixel0(device, fboB.getValue(AdaptiveToken.top).colorTextures!.tryFind("finalColor")! as unknown as GPUTexture);
  log(`PickData = ${[...pick].map((v) => v.toFixed(3)).join(", ")}`);
  log(`finalColor = ${[...fin].map((v) => v.toFixed(3)).join(", ")}`);

  check("pick = closest transparent (A=2)", Math.abs(pick[0]! - 2) < 0.5, `PickData.r=${pick[0]!.toFixed(2)}`);
  check("depth ~ A (0.1)", pick[1]! < 0.2, `PickData.g=${pick[1]!.toFixed(3)}`);
  check("red < 1 (transparents in front)", fin[0]! < 1.0, `r=${fin[0]!.toFixed(3)}`);
  check("green > 0 (B contributed)", fin[1]! > 0.0, `g=${fin[1]!.toFixed(3)}`);
  check("blue > 0 (A contributed)", fin[2]! > 0.0, `b=${fin[2]!.toFixed(3)}`);
  check("red < 0.95 (occluded yellow excluded)", fin[0]! < 0.95, `r=${fin[0]!.toFixed(3)}`);
}
main().catch((e) => { out.textContent += "\nERROR: " + (e?.stack || e); });
