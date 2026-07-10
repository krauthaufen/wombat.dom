// Memory harness — reproduce the cityview per-part pipeline headlessly
// (MockGPU) so Node's RELIABLE heap snapshot can attribute retained
// bytes by constructor. Runs the same structures the browser builds:
// packed BufferView leaves under a per-part <Sg On…> scope, cset →
// compileScene with picking → Runtime(heap) → task.run.
//
// Usage:
//   NODE_OPTIONS=--expose-gc npx vitest run tests-tmp/memharness.test.ts
// Env: PARTS=20000 PICK=1

import { describe, it } from "vitest";
import v8 from "node:v8";
import { AVal, AList, AdaptiveToken, HashMap, cset, cval, transact } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V3f, V4f } from "@aardworx/wombat.base";
import { effect, fragment, vertex } from "@aardworx/wombat.shader";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import {
  ElementType, IBuffer,
  type BufferView, type DrawCall,
} from "@aardworx/wombat.rendering/core";
import { Runtime, allocateFramebuffer, createFramebufferSignature } from "@aardworx/wombat.rendering";

import { Sg } from "../src/scene/constructors.js";
import { compileScene } from "../src/scene/compile.js";
import { TraversalState } from "../src/scene/traversalState.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import { PickMetadata } from "../src/scene/picking/pickMetadata.js";
import type { SgNode } from "../src/scene/sg.js";

// MockGPU straight from the rendering workspace test utils.
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore — outside the project root, resolved by vite-node
// Variable specifier: keeps tsc from type-following into the rendering
// workspace's test utils (different strictness flags); vite-node still
// resolves it at runtime.
import { MockGPU } from "/home/schorsch/projects/wombat.rendering/tests/_mockGpu.js";

const PARTS = parseInt(process.env.PARTS ?? "20000", 10);
const PICK = process.env.PICK !== "0";

const surf = effect(
  vertex((v: { Positions: V4f; Normals: V3f; Colors: V4f }) => {
    const t = uniform.ModelViewTrafoInv.transpose();
    const n4 = t.mul(new V4f(v.Normals.x, v.Normals.y, v.Normals.z, 0.0));
    return {
      gl_Position: uniform.ViewProjTrafo.mul(v.Positions),
      ViewSpaceNormal: new V3f(n4.x, n4.y, n4.z),
      VtxColor: new V3f(v.Colors.x, v.Colors.y, v.Colors.z),
    };
  }),
  fragment((v: { ViewSpaceNormal: V3f; VtxColor: V3f }) => {
    const n = v.ViewSpaceNormal.normalize();
    const d = 0.25 + 0.75 * (n.z < 0.0 ? 0.0 : n.z);
    return { Colors: new V4f(v.VtxColor.x * d, v.VtxColor.y * d, v.VtxColor.z * d, 1.0) };
  }),
);

describe.skipIf(process.env.PARTS === undefined)("memory harness", () => {
  it(`builds ${PARTS} parts pick=${PICK}`, async () => {
    const gpu = new MockGPU();
    const runtime = new Runtime({ device: gpu.device, enableDerivedUniforms: true });
    const registry = new PickRegistry();
    if (PICK) {
      const metadata = new PickMetadata(gpu.device);
      registry.attachObserver(metadata);
      (globalThis as { __md?: unknown }).__md = metadata; // retain
    }

    // shared "district" arrays (like a streamed district)
    const VPP = 30; // verts per part
    const positions = new Float32Array(PARTS * VPP * 3);
    const normalsOct = new Uint32Array(PARTS * VPP);
    const colorsC4b = new Uint32Array(PARTS);

    const leafSet = cset<SgNode>();
    const scene: SgNode = Sg(
      {
        View: AVal.constant(Trafo3d.identity),
        Proj: AVal.constant(Trafo3d.identity),
        Shader: surf,
        ForcePixelPicking: AVal.constant(true),
        children: Sg.unordered(leafSet),
      } as never,
    ) as unknown as SgNode;

    const initial = TraversalState.empty
      .withViewport(AVal.constant({ width: 64, height: 64 }))
      .withCamera(AVal.constant(Trafo3d.identity), AVal.constant(Trafo3d.identity));

    const sig = PICK
      ? createFramebufferSignature({ colors: { Colors: "rgba8unorm", pickId: "rgba32float" }, depthStencil: { format: "depth24plus" } })
      : createFramebufferSignature({ colors: { Colors: "rgba8unorm" } });
    const cmds = compileScene(scene, {
      initialState: initial,
      ...(PICK ? { picking: { registry } } : {}),
    });
    const task = runtime.compile(sig, cmds);
    const fbo = allocateFramebuffer(gpu.device, sig, cval({ width: 64, height: 64 }));
    fbo.acquire();

    const noop = (): void => {};
    const mkLeaf = (i: number): SgNode => {
      const v0 = i * VPP;
      const dc: DrawCall = { kind: "non-indexed", vertexCount: VPP, instanceCount: 1, firstVertex: 0, firstInstance: 0 };
      const vertexAttributes = HashMap.empty<string, BufferView>()
        .add("Positions", { buffer: AVal.constant(IBuffer.fromHost(positions.subarray(v0 * 3, (v0 + VPP) * 3))), elementType: ElementType.V3f })
        .add("Normals", { buffer: AVal.constant(IBuffer.fromHost(normalsOct.subarray(v0, v0 + VPP))), elementType: ElementType.Oct32 })
        .add("Colors", { buffer: AVal.constant(IBuffer.fromHost(colorsC4b.subarray(i, i + 1))), elementType: ElementType.C4b });
      const leaf = Sg.leaf({ vertexAttributes, drawCall: AVal.constant(dc) });
      return Sg({ OnPointerEnter: noop, OnPointerLeave: noop, OnTap: noop, children: leaf } as never) as unknown as SgNode;
    };

    // stream in slices like the demo
    const SLICE = 4096;
    for (let o = 0; o < PARTS; o += SLICE) {
      transact(() => {
        for (let i = o; i < Math.min(o + SLICE, PARTS); i++) leafSet.add(mkLeaf(i));
      });
      task.run(fbo.getValue(AdaptiveToken.top), AdaptiveToken.top);
      gpu.writeBufferCalls.length = 0; // don't let the mock retain staging
      gpu.copyBufferCalls.length = 0;
      gpu.renderPasses.length = 0;
    }
    task.run(fbo.getValue(AdaptiveToken.top), AdaptiveToken.top);
    gpu.writeBufferCalls.length = 0;
    gpu.copyBufferCalls.length = 0;
    gpu.renderPasses.length = 0;

    const g = (globalThis as { gc?: () => void }).gc;
    if (g) { g(); g(); }
    const mu = process.memoryUsage();
    console.log(`PARTS=${PARTS} PICK=${PICK} heapUsed=${(mu.heapUsed / 1e6).toFixed(0)}MB registry=${registry.size()}`);
    const file = `/tmp/memharness-${PICK ? "pick" : "nopick"}.heapsnapshot`;
    v8.writeHeapSnapshot(file);
    console.log("snapshot:", file);
  }, 600000);
});
