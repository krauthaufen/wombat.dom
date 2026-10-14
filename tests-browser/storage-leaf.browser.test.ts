// A scene-graph leaf the way a raymarcher draws: NO vertex attributes,
// per-instance data in storage buffers (indexed by instance_index), a
// 3D texture, and an adaptive draw call whose instance count changes.
// The texture and one buffer are wombat.rendering Mutable* resources
// written in place; the test checks that region writes, count changes
// and a buffer reallocation all reach the rendered image.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap, cval, transact } from "@aardworx/wombat.adaptive";
import { V4f } from "@aardworx/wombat.base";
import { stage, type Effect } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { Tf32, Vec, type Type, type ValueDef } from "@aardworx/wombat.shader/ir";
import type { BufferView, DrawCall, IBuffer } from "@aardworx/wombat.rendering/core";
import {
  MutableBuffer, MutableTexture, allocateFramebuffer, createFramebufferSignature, readTextureRegion,
} from "@aardworx/wombat.rendering/resources";
import { Runtime } from "@aardworx/wombat.rendering/runtime";

// the built package: native browser ESM needs the shader-marker imports the build strips
import { Sg, compileScene } from "../dist/scene/index.js";
import { requestRealDevice } from "./_realGpu.js";

const W = 32, H = 16;
const Tv4: Type = Vec(Tf32, 4);
const Tv3: Type = Vec(Tf32, 3);

// rects[i] = (x0, y0, x1, y1) in NDC; corners[v].xy in {0,1}; the 3D
// texture is sampled at layer (i + 0.5) / 2.
function effect(): Effect {
  const source = `
    declare const rects: V4f[];
    declare const corners: V4f[];
    declare const vol: Sampler3D;
    function vsMain(input: {}, b: VertexBuiltinIn): { gl_Position: V4f; Uvw: V3f } {
      const r = rects[b.instanceIndex as i32];
      const c = corners[b.vertexIndex as i32];
      const x = r.x + (r.z - r.x) * c.x;
      const y = r.y + (r.w - r.y) * c.y;
      return { gl_Position: new V4f(x, y, 0.0, 1.0), Uvw: new V3f(c.x * 0.5 + 0.25, c.y * 0.5 + 0.25, ((b.instanceIndex as f32) + 0.5) * 0.5) };
    }
    function fsMain(input: { Uvw: V3f }): { Colors: V4f } {
      const v = texture(vol, input.Uvw);
      return { Colors: new V4f(v.x, v.y, 0.0, 1.0) };
    }
  `;
  const arr = (name: string, slot: number): ValueDef => ({
    kind: "StorageBuffer", binding: { group: 0, slot }, name,
    layout: { kind: "Array", element: Tv4, length: "runtime" } as unknown as Type, access: "read",
  });
  const volType: Type = { kind: "Sampler", target: "3D", sampled: { kind: "Float" }, comparison: false };
  const vol: ValueDef = {
    kind: "Sampler", binding: { group: 0, slot: 2 }, name: "vol", type: volType,
    state: { filter: "MinMagPoint", addressU: "Clamp", addressV: "Clamp", addressW: "Clamp" },
  };
  const rects = arr("rects", 0), corners = arr("corners", 1);
  const externalTypes = new Map<string, Type>([
    ["rects", (rects as { layout: Type }).layout],
    ["corners", (corners as { layout: Type }).layout],
    ["vol", volType],
  ]);
  const entries: EntryRequest[] = [
    {
      name: "vsMain", stage: "vertex",
      inputs: [],
      outputs: [
        { name: "gl_Position", type: Tv4, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] },
        { name: "Uvw", type: Tv3, semantic: "Uvw", decorations: [{ kind: "Location", value: 0 }] },
      ],
    },
    {
      name: "fsMain", stage: "fragment",
      inputs: [{ name: "Uvw", type: Tv3, semantic: "Uvw", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "Colors", type: Tv4, semantic: "Colors", decorations: [{ kind: "Location", value: 0 }] }],
    },
  ];
  const parsed = parseShader({ source, entries, externalTypes });
  return stage({ ...parsed, values: [rects, corners, vol, ...parsed.values] });
}

describe("storage-buffer leaf without vertex attributes (real GPU)", () => {
  it("renders instances from storage buffers + a 3D MutableTexture; count, region writes and resizes flow through", async () => {
    const device = await requestRealDevice();
    const errors: string[] = [];
    device.onuncapturederror = (e) => errors.push(e.error.message);
    const warnings: string[] = [];
    const warn = console.warn;
    console.warn = (...args: unknown[]) => { warnings.push(args.map(String).join(" ")); warn(...args); };
    try {
      const runtime = new Runtime({ device });
      const vol = new MutableTexture(device, { size: [2, 2, 2], format: "rg8unorm", dimension: "3d" });
      const layer = (r: number, g: number) => new Uint8Array([r, g, r, g, r, g, r, g]);
      vol.write(layer(200, 0), { origin: [0, 0, 0], size: [2, 2, 1] });
      vol.write(layer(0, 200), { origin: [0, 0, 1], size: [2, 2, 1] });

      // two side-by-side rects: instance 0 left half, instance 1 right half
      const rects = new MutableBuffer(device, { size: 2 * 16 });
      rects.write(0, new Float32Array([-1, -1, 0, 1, 0, -1, 1, 1]));
      const corners = new MutableBuffer(device, { size: 6 * 16 });
      corners.write(0, new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]));

      const count = cval(2);
      const drawCall = count.map((n): DrawCall => ({ kind: "non-indexed", vertexCount: 6, instanceCount: n, firstVertex: 0, firstInstance: 0 }));
      const leaf = Sg.leaf({
        vertexAttributes: HashMap.empty<string, BufferView>(),
        drawCall,
        storageBuffers: HashMap.empty<string, AVal<IBuffer>>().add("rects", rects).add("corners", corners),
      });
      const scene = Sg.shader(effect(), Sg.uniform({ vol }, Sg.cullMode(AVal.constant("none"))(leaf)));

      const signature = createFramebufferSignature({ colors: { Colors: "rgba8unorm" } });
      const fbo = allocateFramebuffer(device, signature, AVal.constant({ width: W, height: H }), { extraUsage: GPUTextureUsage.COPY_SRC });
      fbo.acquire();
      const clear = { colors: HashMap.empty<string, V4f>().add("Colors", new V4f(0, 0, 0, 0)) };
      const task = runtime.compile(signature, compileScene(scene, { clear }));
      const pixel = async (x: number, y: number): Promise<number[]> => {
        task.run(fbo.getValue(AdaptiveToken.top), AdaptiveToken.top);
        const fb = fbo.getValue(AdaptiveToken.top);
        const tex = fb.colorTextures!.tryFind("Colors")!;
        return Array.from(new Uint8Array(await readTextureRegion(device, tex, { origin: [x, y], size: [1, 1] })));
      };

      expect(await pixel(4, 8)).toEqual([200, 0, 0, 255]);
      expect(await pixel(28, 8)).toEqual([0, 200, 0, 255]);

      // instance count follows the aval
      transact(() => { count.value = 1; });
      expect(await pixel(28, 8)).toEqual([0, 0, 0, 0]);
      expect(await pixel(4, 8)).toEqual([200, 0, 0, 255]);

      // a region write into the bound 3D texture shows up without rebinding
      vol.write(layer(50, 60), { origin: [0, 0, 0], size: [2, 2, 1] });
      await Promise.resolve();
      expect(await pixel(4, 8)).toEqual([50, 60, 0, 255]);

      // growing the instance buffer swaps the GPUBuffer behind the aval
      rects.ensureCapacity(3 * 16);
      rects.write(0, new Float32Array([-1, -1, -0.5, 1, -0.5, -1, 0.5, 1, 0.5, -1, 1, 1]));
      transact(() => { count.value = 3; });
      await Promise.resolve();
      expect(await pixel(2, 8)).toEqual([50, 60, 0, 255]);    // instance 0
      expect(await pixel(16, 8)).toEqual([0, 200, 0, 255]);   // instance 1
      // instance 2 samples past the last layer: clamped to layer 1
      expect(await pixel(30, 8)).toEqual([0, 200, 0, 255]);

      task.dispose();
      fbo.release();
      vol.dispose();
      rects.dispose();
      corners.dispose();
      await device.queue.onSubmittedWorkDone();
      expect(errors).toEqual([]);
      // leaf storage buffers satisfy the effect's buffer reads
      expect(warnings.filter(w => w.includes("no provider"))).toEqual([]);
    } finally {
      console.warn = warn;
      device.destroy();
    }
  }, 30000);
});
