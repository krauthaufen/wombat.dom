// OIT transparency tests on the real GPU — wombat port of aardvark's z-stack:
//   z=-0.8 A blue a0.5 (front)  z=-0.3 B green a0.5 (front)
//   z= 0.0 SOLID red opaque     z=0.3 C yellow / z=0.7 D cyan (occluded)
//
// Test 1: Weighted-Blended OIT (approximate)  -> ~(0.25, 0.375, 0.375)
// Test 2: linked-list A-buffer (EXACT)         -> (0.25, 0.25, 0.5)
// Both share an opaque pass + a depth/pick pass; both assert pick=A(2), depth~0.1.
import { AList, AdaptiveToken, AVal, HashMap, cval } from "@aardworx/wombat.adaptive";
import { V2f, V3f, V4f } from "@aardworx/wombat.base";
import {
  IBuffer, ITexture, ISampler, RenderTree, ElementType, PipelineState,
  type BufferView, type Command, type DrawCall, type RenderObject,
} from "@aardworx/wombat.rendering/core";
import { allocateFramebuffer, createFramebufferSignature, TextureUsage } from "@aardworx/wombat.rendering/resources";
import { Runtime } from "@aardworx/wombat.rendering/runtime";
import { effect, fragment, vertex } from "@aardworx/wombat.shader";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import {
  clamp, max, texture, atomicAdd, atomicExchange,
  type Sampler2D, type Storage, type FragmentBuiltinIn, type f32, type u32,
} from "@aardworx/wombat.shader/types";
import { Sg, transparencyTask, setOitMode, type OitMode } from "@aardworx/wombat.dom/scene";

declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope { readonly u_z: f32; readonly u_color: V4f; readonly u_pickId: f32; }
}

const out = document.getElementById("out")!;
const log = (s: string) => { out.textContent += "\n" + s; };
const check = (name: string, ok: boolean, detail: string) => log(`[${ok ? "PASS" : "FAIL"}] ${name} — ${detail}`);
const SIZE = 8;
const MAXN = 4096;
const NULLU = 0xffffffff;

// ---------- shared geometry / quad uniforms ----------
const RAST = { topology: "triangle-list" as GPUPrimitiveTopology, cullMode: "none" as GPUCullMode, frontFace: "ccw" as GPUFrontFace };
const FS_QUAD = new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]);

// main VS maps test-NDC z -> WebGPU clip [0,1]
const mainVS = vertex((v: { a_pos: V2f }) => ({
  gl_Position: new V4f(v.a_pos.x, v.a_pos.y, uniform.u_z.mul(0.5).add(0.5), 1.0),
}));

// ---------- WBOIT shaders ----------
const wboitFS = fragment((_in: {}, b: FragmentBuiltinIn) => {
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
const wboitEffect = effect(mainVS, wboitFS);

const tColors: Sampler2D = null as unknown as Sampler2D;
const tAccum: Sampler2D = null as unknown as Sampler2D;
const tReveal: Sampler2D = null as unknown as Sampler2D;
const fsVS = vertex((v: { a_pos: V2f }) => ({ gl_Position: new V4f(v.a_pos.x, v.a_pos.y, 0.0, 1.0) }));
const wboitCompFS = fragment((_in: {}, b: FragmentBuiltinIn) => {
  const uv = new V2f(b.fragCoord.x.div(SIZE), b.fragCoord.y.div(SIZE));
  const opaque = texture(tColors, uv);
  const accum = texture(tAccum, uv);
  const reveal = texture(tReveal, uv).x;
  const avg = accum.xyz.div(max(accum.w, 1e-5));
  const ta = reveal.mul(-1.0).add(1.0);
  return { finalColor: new V4f(avg.mul(ta).add(opaque.xyz.mul(ta.mul(-1.0).add(1.0))), 1.0) };
});
const wboitCompositeEffect = effect(fsVS, wboitCompFS);

// ---------- A-buffer shaders (linked list, atomics) ----------
declare const headBuf: Storage<u32[], "read_write">;
declare const counterBuf: Storage<u32[], "read_write">;
declare const nodeDepth: Storage<u32[], "read_write">;
declare const nodeColor: Storage<V4f[], "read_write">;
declare const nodeNext: Storage<u32[], "read_write">;
const abBuildFS = fragment((_in: {}, b: FragmentBuiltinIn) => {
  const fx = b.fragCoord.x as u32;
  const fy = b.fragCoord.y as u32;
  const px = fy.mul(8 as u32).add(fx);
  const n = atomicAdd(counterBuf[0], 1 as u32);
  if (n < (4096 as u32)) {
    nodeDepth[n] = b.fragCoord.z.mul(16777215.0) as u32;
    nodeColor[n] = uniform.u_color;
    nodeNext[n] = atomicExchange(headBuf[px], n);
  }
  return { Colors: uniform.u_color, PickData: new V4f(uniform.u_pickId, b.fragCoord.z, 0.0, 0.0), odepth: b.fragCoord.z };
});
const abBuildEffect = effect(mainVS, abBuildFS);

const tOpaque: Sampler2D = null as unknown as Sampler2D;
const tODepth: Sampler2D = null as unknown as Sampler2D;
declare const headBufR: Storage<u32[], "read">;
declare const nodeDepthR: Storage<u32[], "read">;
declare const nodeColorR: Storage<V4f[], "read">;
declare const nodeNextR: Storage<u32[], "read">;
const abResolveFS = fragment((_in: {}, b: FragmentBuiltinIn) => {
  const uv = new V2f(b.fragCoord.x.div(SIZE), b.fragCoord.y.div(SIZE));
  const opaque = texture(tOpaque, uv);
  const odepthQ = texture(tODepth, uv).x.mul(16777215.0) as u32; // opaque depth, quantized
  const fx = b.fragCoord.x as u32;
  const fy = b.fragCoord.y as u32;
  const px = fy.mul(8 as u32).add(fx);
  let outRGB = new V3f(0.0, 0.0, 0.0);
  let T = 1.0;        // remaining transmittance
  let lastD: u32 = 0 as u32;
  for (let layer = 0; layer < 8; layer = layer + 1) {
    // find nearest node with depth strictly > lastD and in front of the opaque
    // surface (storage writes ignore the depth test, so occlude here).
    let bestD: u32 = NULLU as u32;
    let bestI: u32 = 0 as u32;
    let found = 0.0;
    let i: u32 = headBufR[px];
    for (let step = 0; step < 16; step = step + 1) {
      if (i !== (NULLU as u32)) {
        const d = nodeDepthR[i];
        if (d > lastD && d < bestD && d < odepthQ) { bestD = d; bestI = i; found = 1.0; }
        i = nodeNextR[i];
      }
    }
    if (found > 0.5) {
      const src = nodeColorR[bestI];
      outRGB = outRGB.add(src.xyz.mul(src.w).mul(T)); // front-to-back over
      T = T * (1.0 - src.w);
      lastD = bestD;
    }
  }
  return { finalColor: new V4f(outRGB.add(opaque.xyz.mul(T)), 1.0) };
});
const abResolveEffect = effect(fsVS, abResolveFS);

// ---------- pick effect (depth-test selects nearest transparent) ----------
const pickFS = fragment((_in: {}, b: FragmentBuiltinIn) => ({
  Colors: uniform.u_color,
  PickData: new V4f(uniform.u_pickId, b.fragCoord.z, 0.0, 0.0),
  odepth: b.fragCoord.z,
}));
const pickEffect = effect(mainVS, pickFS);

// ---------- blends ----------
type Comp = { operation: GPUBlendOperation; srcFactor: GPUBlendFactor; dstFactor: GPUBlendFactor };
const comp = (s: GPUBlendFactor, d: GPUBlendFactor): Comp => ({ operation: "add", srcFactor: s, dstFactor: d });
const blend = (color: Comp, alpha: Comp, writeMask: number) => ({ color, alpha, writeMask });
const masked = blend(comp("one", "zero"), comp("one", "zero"), 0);
const additive = blend(comp("one", "one"), comp("one", "one"), 0xf);
const revealMul = blend(comp("zero", "one-minus-src"), comp("zero", "one-minus-src"), 0xf);
type Blends = HashMap<string, ReturnType<typeof blend>>;

// scene quads
const A = { z: -0.8, c: new V4f(0, 0, 1, 0.5), id: 2 };
const Bq = { z: -0.3, c: new V4f(0, 1, 0, 0.5), id: 3 };
const S = { z: 0.0, c: new V4f(1, 0, 0, 1.0), id: 1 };
const Cq = { z: 0.3, c: new V4f(1, 1, 0, 0.5), id: 4 };
const Dq = { z: 0.7, c: new V4f(0, 1, 1, 0.5), id: 5 };

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
  const nearest = AVal.constant(ISampler.fromDescriptor({ magFilter: "nearest", minFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" }));

  const ro = (eff: typeof wboitEffect, q: { z: number; c: V4f; id: number }, blends: Blends, depthWrite: boolean, extra?: Partial<RenderObject>): RenderObject => ({
    effect: eff,
    pipelineState: PipelineState.constant({ rasterizer: RAST, depth: { write: depthWrite, compare: "less" }, blends }),
    vertexAttributes: vattrs,
    uniforms: HashMap.empty<string, unknown>().add("u_z", AVal.constant(q.z)).add("u_color", AVal.constant(q.c)).add("u_pickId", AVal.constant(q.id)) as unknown as RenderObject["uniforms"],
    textures: HashMap.empty(), samplers: HashMap.empty(), drawCall: draw, ...extra,
  });

  const opaqueBlends: Blends = HashMap.empty<string, ReturnType<typeof blend>>().add("accum", masked).add("reveal", masked);
  const wboitBlends: Blends = HashMap.empty<string, ReturnType<typeof blend>>().add("Colors", masked).add("accum", additive).add("reveal", revealMul);
  const wboitPickBlends: Blends = HashMap.empty<string, ReturnType<typeof blend>>().add("Colors", masked).add("accum", masked).add("reveal", masked);
  const abPickBlends: Blends = HashMap.empty<string, ReturnType<typeof blend>>().add("Colors", masked);

  // ===================== Test 1: WBOIT =====================
  {
    const sigA = createFramebufferSignature({ colors: { Colors: "rgba16float", PickData: "rgba32float", accum: "rgba16float", reveal: "r16float" }, depthStencil: { format: "depth32float" } });
    const fboA = allocateFramebuffer(device, sigA, cval({ width: SIZE, height: SIZE }), { extraUsage: TextureUsage.COPY_SRC });
    fboA.acquire();
    const taskA = runtime.compile(sigA, AList.ofArray<Command>([
      { kind: "Clear", values: { colors: HashMap.empty<string, V4f>().add("Colors", new V4f(0, 0, 0, 1)).add("PickData", new V4f(0, 0, 0, 0)).add("accum", new V4f(0, 0, 0, 0)).add("reveal", new V4f(1, 1, 1, 1)), depth: 1.0 } },
      { kind: "Render", tree: RenderTree.leaf(ro(wboitEffect, S, opaqueBlends, true)) },
      { kind: "Render", tree: RenderTree.ordered(...[A, Bq, Cq, Dq].map((q) => RenderTree.leaf(ro(wboitEffect, q, wboitBlends, false)))) },
      { kind: "Render", tree: RenderTree.ordered(...[A, Bq, Cq, Dq].map((q) => RenderTree.leaf(ro(wboitEffect, q, wboitPickBlends, true)))) },
    ]));
    taskA.run(fboA.getValue(AdaptiveToken.top), AdaptiveToken.top);
    await device.queue.onSubmittedWorkDone();
    const ifbA = fboA.getValue(AdaptiveToken.top);
    const gpu = (n: string) => ifbA.colorTextures!.tryFind(n)! as unknown as GPUTexture;

    const sigB = createFramebufferSignature({ colors: { finalColor: "rgba32float" } });
    const fboB = allocateFramebuffer(device, sigB, cval({ width: SIZE, height: SIZE }), { extraUsage: TextureUsage.COPY_SRC });
    fboB.acquire();
    const compRO: RenderObject = {
      effect: wboitCompositeEffect, pipelineState: PipelineState.constant({ rasterizer: RAST }), vertexAttributes: vattrs, uniforms: HashMap.empty(),
      textures: HashMap.empty<string, unknown>().add("tColors_view", AVal.constant(ITexture.fromGPU(gpu("Colors")))).add("tAccum_view", AVal.constant(ITexture.fromGPU(gpu("accum")))).add("tReveal_view", AVal.constant(ITexture.fromGPU(gpu("reveal")))) as unknown as RenderObject["textures"],
      samplers: HashMap.empty<string, unknown>().add("tColors", nearest).add("tAccum", nearest).add("tReveal", nearest) as unknown as RenderObject["samplers"], drawCall: draw,
    };
    runtime.compile(sigB, AList.ofArray<Command>([
      { kind: "Clear", values: { colors: HashMap.empty<string, V4f>().add("finalColor", new V4f(0, 0, 0, 1)) } },
      { kind: "Render", tree: RenderTree.leaf(compRO) },
    ])).run(fboB.getValue(AdaptiveToken.top), AdaptiveToken.top);
    await device.queue.onSubmittedWorkDone();

    const pick = await readPixel0(device, gpu("PickData"));
    const fin = await readPixel0(device, fboB.getValue(AdaptiveToken.top).colorTextures!.tryFind("finalColor")! as unknown as GPUTexture);
    log(`\n== WBOIT ==  pick=${[...pick].slice(0, 2).map((v) => v.toFixed(2)).join(",")}  color=${[...fin].slice(0, 3).map((v) => v.toFixed(3)).join(",")}`);
    check("WBOIT pick=A(2)", Math.abs(pick[0]! - 2) < 0.5, `${pick[0]!.toFixed(2)}`);
    check("WBOIT depth~0.1", pick[1]! < 0.2, `${pick[1]!.toFixed(3)}`);
    check("WBOIT red<1 & green,blue>0 & red<0.95", fin[0]! < 0.95 && fin[1]! > 0 && fin[2]! > 0, `(${fin[0]!.toFixed(3)},${fin[1]!.toFixed(3)},${fin[2]!.toFixed(3)})`);
  }

  // ===================== Test 2: linked-list A-buffer (exact) =====================
  {
    // Device-owned storage so the build pass's writes persist into the resolve
    // pass (a host-backed IBuffer.fromHost is re-synced from host data).
    const sbuf = (data: Uint32Array | Float32Array): IBuffer => {
      const g = device.createBuffer({ size: Math.max(data.byteLength, 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(g, 0, data);
      return IBuffer.fromGPU(g);
    };
    const head = sbuf(new Uint32Array(SIZE * SIZE).fill(NULLU));
    const counter = sbuf(new Uint32Array([0]));
    const nDepth = sbuf(new Uint32Array(MAXN));
    const nColor = sbuf(new Float32Array(MAXN * 4));
    const nNext = sbuf(new Uint32Array(MAXN));
    const buildStorage = HashMap.empty<string, unknown>()
      .add("headBuf", AVal.constant(head)).add("counterBuf", AVal.constant(counter))
      .add("nodeDepth", AVal.constant(nDepth)).add("nodeColor", AVal.constant(nColor)).add("nodeNext", AVal.constant(nNext)) as unknown as RenderObject["storageBuffers"];

    const sigA = createFramebufferSignature({ colors: { Colors: "rgba16float", PickData: "rgba32float", odepth: "r16float" }, depthStencil: { format: "depth32float" } });
    const fboA = allocateFramebuffer(device, sigA, cval({ width: SIZE, height: SIZE }), { extraUsage: TextureUsage.COPY_SRC });
    fboA.acquire();
    const buildBlends = HashMap.empty<string, ReturnType<typeof blend>>().add("Colors", masked).add("odepth", masked);
    const abPick2 = HashMap.empty<string, ReturnType<typeof blend>>().add("Colors", masked).add("odepth", masked);
    const buildRO = (q: { z: number; c: V4f; id: number }) => ro(abBuildEffect, q, buildBlends, false, { storageBuffers: buildStorage });
    const taskA = runtime.compile(sigA, AList.ofArray<Command>([
      { kind: "Clear", values: { colors: HashMap.empty<string, V4f>().add("Colors", new V4f(0, 0, 0, 1)).add("PickData", new V4f(0, 0, 0, 0)).add("odepth", new V4f(1, 1, 1, 1)), depth: 1.0 } },
      { kind: "Render", tree: RenderTree.leaf(ro(pickEffect, S, HashMap.empty<string, ReturnType<typeof blend>>(), true)) }, // opaque solid (Colors+PickData+odepth+depth)
      { kind: "Render", tree: RenderTree.ordered(...[A, Bq, Cq, Dq].map((q) => RenderTree.leaf(buildRO(q)))) },              // A-buffer build (storage)
      { kind: "Render", tree: RenderTree.ordered(...[A, Bq, Cq, Dq].map((q) => RenderTree.leaf(ro(pickEffect, q, abPick2, true)))) }, // pick pass
    ]));
    taskA.run(fboA.getValue(AdaptiveToken.top), AdaptiveToken.top);
    await device.queue.onSubmittedWorkDone();
    const ifbA = fboA.getValue(AdaptiveToken.top);
    const gpu = (n: string) => ifbA.colorTextures!.tryFind(n)! as unknown as GPUTexture;

    const resolveStorage = HashMap.empty<string, unknown>()
      .add("headBufR", AVal.constant(head)).add("nodeDepthR", AVal.constant(nDepth)).add("nodeColorR", AVal.constant(nColor)).add("nodeNextR", AVal.constant(nNext)) as unknown as RenderObject["storageBuffers"];
    const sigB = createFramebufferSignature({ colors: { finalColor: "rgba32float" } });
    const fboB = allocateFramebuffer(device, sigB, cval({ width: SIZE, height: SIZE }), { extraUsage: TextureUsage.COPY_SRC });
    fboB.acquire();
    const resolveRO: RenderObject = {
      effect: abResolveEffect, pipelineState: PipelineState.constant({ rasterizer: RAST }), vertexAttributes: vattrs, uniforms: HashMap.empty(),
      textures: HashMap.empty<string, unknown>().add("tOpaque_view", AVal.constant(ITexture.fromGPU(gpu("Colors")))).add("tODepth_view", AVal.constant(ITexture.fromGPU(gpu("odepth")))) as unknown as RenderObject["textures"],
      samplers: HashMap.empty<string, unknown>().add("tOpaque", nearest).add("tODepth", nearest) as unknown as RenderObject["samplers"],
      storageBuffers: resolveStorage, drawCall: draw,
    };
    runtime.compile(sigB, AList.ofArray<Command>([
      { kind: "Clear", values: { colors: HashMap.empty<string, V4f>().add("finalColor", new V4f(0, 0, 0, 1)) } },
      { kind: "Render", tree: RenderTree.leaf(resolveRO) },
    ])).run(fboB.getValue(AdaptiveToken.top), AdaptiveToken.top);
    await device.queue.onSubmittedWorkDone();

    const pick = await readPixel0(device, gpu("PickData"));
    const fin = await readPixel0(device, fboB.getValue(AdaptiveToken.top).colorTextures!.tryFind("finalColor")! as unknown as GPUTexture);
    log(`\n== A-buffer (exact) ==  pick=${[...pick].slice(0, 2).map((v) => v.toFixed(2)).join(",")}  color=${[...fin].slice(0, 3).map((v) => v.toFixed(3)).join(",")}`);
    check("A-buffer pick=A(2)", Math.abs(pick[0]! - 2) < 0.5, `${pick[0]!.toFixed(2)}`);
    check("A-buffer depth~0.1", pick[1]! < 0.2, `${pick[1]!.toFixed(3)}`);
    check("A-buffer exact color ~(0.25,0.25,0.5)", Math.abs(fin[0]! - 0.25) < 0.05 && Math.abs(fin[1]! - 0.25) < 0.05 && Math.abs(fin[2]! - 0.5) < 0.05, `(${fin[0]!.toFixed(3)},${fin[1]!.toFixed(3)},${fin[2]!.toFixed(3)})`);
  }

  // ===================== Test 3: Sg.transparent via transparencyTask (library path) =====================
  {
    // A real scene-graph: opaque solid + 4 transparent quads, tagged with
    // Sg.opaque / Sg.transparent. transparencyTask lowers + multipasses it.
    const userEffect = effect(
      vertex((v: { a_pos: V2f }) => ({ gl_Position: new V4f(v.a_pos.x, v.a_pos.y, uniform.u_z.mul(0.5).add(0.5), 1.0) })),
      fragment(() => ({ Colors: uniform.u_color })),
    );
    const quadGeom = Sg.leaf({
      vertexAttributes: HashMap.empty<string, BufferView>().add("a_pos", { buffer: quadBuf, offset: 0, stride: 8, elementType: ElementType.V2f }),
      drawCall: cval<DrawCall>({ kind: "non-indexed", vertexCount: 6, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
    });
    const q = (z: number, color: V4f) => Sg.uniform({ u_z: z, u_color: color }, Sg.shader(userEffect, quadGeom));
    const scene = Sg.group([
      Sg.opaque(q(S.z, S.c)),
      Sg.transparent(Sg.group([q(A.z, A.c), q(Bq.z, Bq.c), q(Cq.z, Cq.c), q(Dq.z, Dq.c)])),
    ]);

    const outSig = createFramebufferSignature({ colors: { Colors: "rgba32float" } });
    const size = cval({ width: SIZE, height: SIZE });

    // Run the SAME Sg scene through both OIT modes (set via the global toggle).
    const runMode = async (mode: OitMode): Promise<Float32Array> => {
      setOitMode(mode);
      const outFbo = allocateFramebuffer(device, outSig, size, { extraUsage: TextureUsage.COPY_SRC });
      outFbo.acquire();
      const task = transparencyTask(runtime, device, outSig, size, scene);
      task.run(outFbo.getValue(AdaptiveToken.top), AdaptiveToken.top);
      await device.queue.onSubmittedWorkDone();
      const px = await readPixel0(device, outFbo.getValue(AdaptiveToken.top).colorTextures!.tryFind("Colors")! as unknown as GPUTexture);
      task.dispose(); outFbo.release();
      return px;
    };

    const w = await runMode("wboit");
    log(`\n== Sg.transparent + transparencyTask, mode=wboit ==  color=${[...w].slice(0, 3).map((v) => v.toFixed(3)).join(",")}`);
    check("Sg wboit ~(0.25,0.375,0.375)", Math.abs(w[0]! - 0.25) < 0.05 && Math.abs(w[1]! - 0.375) < 0.05 && Math.abs(w[2]! - 0.375) < 0.05, `(${w[0]!.toFixed(3)},${w[1]!.toFixed(3)},${w[2]!.toFixed(3)})`);

    const a = await runMode("abuffer");
    log(`== Sg.transparent + transparencyTask, mode=abuffer ==  color=${[...a].slice(0, 3).map((v) => v.toFixed(3)).join(",")}`);
    check("Sg abuffer exact ~(0.25,0.25,0.5)", Math.abs(a[0]! - 0.25) < 0.05 && Math.abs(a[1]! - 0.25) < 0.05 && Math.abs(a[2]! - 0.5) < 0.05, `(${a[0]!.toFixed(3)},${a[1]!.toFixed(3)},${a[2]!.toFixed(3)})`);
  }
}
main().catch((e) => { out.textContent += "\nERROR: " + (e?.stack || e); });
