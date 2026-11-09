// `Sg.active` at scale on the heap renderer: 10k subtrees, each an on/off
// node over a leaf. Shown once, a toggle must not lower,
// register or upload anything: the renderer flips one per-draw flag. The
// test checks the image (a hidden draw is not drawn) and prints the cost
// of toggling all 10k (one shared cell, and 10k cells in one transaction)
// against building and removing the same subtrees.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { V4f } from "@aardworx/wombat.base";
import { stage, type Effect } from "@aardworx/wombat.shader";
import { parseShader, type EntryRequest } from "@aardworx/wombat.shader/frontend";
import { Tf32, Vec, type Type } from "@aardworx/wombat.shader/ir";
import type { BufferView, DrawCall } from "@aardworx/wombat.rendering/core";
import { ElementType } from "@aardworx/wombat.rendering/core";
import { allocateFramebuffer, createFramebufferSignature, readTextureRegion } from "@aardworx/wombat.rendering/resources";
import { Runtime } from "@aardworx/wombat.rendering/runtime";

// the built package: native browser ESM needs the shader-marker imports the build strips
import { Sg, compileScene, type SgNode } from "../dist/scene/index.js";
import { requestRealDevice } from "./_realGpu.js";

const W = 64, H = 64;
const N = 10_000;
const Tv4: Type = Vec(Tf32, 4);

function effect(): Effect {
  const source = `
    function vsActiveToggle(input: { Positions: V4f }): { gl_Position: V4f } {
      return { gl_Position: input.Positions };
    }
    function fsActiveToggle(input: {}): { Colors: V4f } {
      return { Colors: new V4f(1.0, 1.0, 1.0, 1.0) };
    }
  `;
  const entries: EntryRequest[] = [
    {
      name: "vsActiveToggle", stage: "vertex",
      inputs: [{ name: "Positions", type: Tv4, semantic: "Positions", decorations: [{ kind: "Location", value: 0 }] }],
      outputs: [{ name: "gl_Position", type: Tv4, semantic: "Position", decorations: [{ kind: "Builtin", value: "position" }] }],
    },
    {
      name: "fsActiveToggle", stage: "fragment",
      inputs: [],
      outputs: [{ name: "Colors", type: Tv4, semantic: "Colors", decorations: [{ kind: "Location", value: 0 }] }],
    },
  ];
  return stage(parseShader({ source, entries }));
}

const draw: aval<DrawCall> = AVal.constant({ kind: "non-indexed", vertexCount: 6, instanceCount: 1, firstVertex: 0, firstInstance: 0 });

// Item i: a small quad (two triangles) of its own at a grid position in
// [-1, 1]²; item 0 covers the pixel the test reads.
function item(i: number, on: aval<boolean>, eager = false): SgNode {
  const x = -1 + 2 * ((i % 100) / 100), y = -1 + 2 * (Math.floor(i / 100) / 100), d = 0.04;
  const data = new Float32Array([x, y, 0, x + d, y, 0, x, y + d, 0, x + d, y, 0, x + d, y + d, 0, x, y + d, 0]);
  const positions: BufferView = { buffer: AVal.constant({ kind: "host", data, sizeBytes: data.byteLength }), elementType: ElementType.V3f };
  return Sg.active(on, Sg.leaf({ vertexAttributes: HashMap.empty<string, BufferView>().add("Positions", positions), drawCall: draw }), { eager });
}

describe("Sg.active over 10k subtrees (real GPU)", () => {
  it("toggles without rebuilding; hidden draws are not drawn", async () => {
    const device = await requestRealDevice();
    const errors: string[] = [];
    device.onuncapturederror = (e) => errors.push(e.error.message);
    try {
      const runtime = new Runtime({ device });
      const signature = createFramebufferSignature({ colors: { Colors: "rgba8unorm" } });
      const fbo = allocateFramebuffer(device, signature, AVal.constant({ width: W, height: H }), { extraUsage: GPUTextureUsage.COPY_SRC });
      fbo.acquire();
      const clear = { colors: HashMap.empty<string, V4f>().add("Colors", new V4f(0, 0, 0, 0)) };
      const ms = (f: () => void): number => { const t = performance.now(); f(); return performance.now() - t; };
      const frame = (task: { run: (f: never, t: AdaptiveToken) => void }) => ms(() => task.run(fbo.getValue(AdaptiveToken.top) as never, AdaptiveToken.top));
      const pixel = async (): Promise<number[]> => {
        await device.queue.onSubmittedWorkDone();
        const tex = fbo.getValue(AdaptiveToken.top).colorTextures!.tryFind("Colors")!;
        // item 0's quad covers pixel (0, H-1) .. (0, H-1) in framebuffer rows
        return Array.from(new Uint8Array(await readTextureRegion(device, tex, { origin: [0, H - 1], size: [1, 1] })));
      };
      const report: Record<string, number> = {};

      // (1) one shared cell for all 10k
      {
        const all = cval(false);
        const scene = Sg.shader(effect(), Sg.group(Array.from({ length: N }, (_, i) => item(i, all))));
        const task = runtime.compile(signature, compileScene(scene, { clear }));
        report["load hidden (ms)"] = frame(task);
        expect(await pixel()).toEqual([0, 0, 0, 0]);
        transact(() => { all.value = true; });
        report["first show, lowers and registers (ms)"] = frame(task);
        expect(await pixel()).toEqual([255, 255, 255, 255]);
        for (let k = 0; k < 3; k++) {
          report[`toggle off, shared cell #${k} (ms)`] = ms(() => transact(() => { all.value = false; })) + frame(task);
          expect(await pixel()).toEqual([0, 0, 0, 0]);
          report[`toggle on, shared cell #${k} (ms)`] = ms(() => transact(() => { all.value = true; })) + frame(task);
          expect(await pixel()).toEqual([255, 255, 255, 255]);
        }
        task.dispose();
      }

      // (1b) eager: registered inactive at load, the first show flips flags
      {
        const all = cval(false);
        const scene = Sg.shader(effect(), Sg.group(Array.from({ length: N }, (_, i) => item(i, all, true))));
        const task = runtime.compile(signature, compileScene(scene, { clear }));
        report["eager: load hidden, registers (ms)"] = frame(task);
        expect(await pixel()).toEqual([0, 0, 0, 0]);
        report["eager: first show (ms)"] = ms(() => transact(() => { all.value = true; })) + frame(task);
        expect(await pixel()).toEqual([255, 255, 255, 255]);
        report["eager: toggle off (ms)"] = ms(() => transact(() => { all.value = false; })) + frame(task);
        expect(await pixel()).toEqual([0, 0, 0, 0]);
        task.dispose();
      }

      // (2) a cell per subtree, all set in one transaction
      {
        const cells = Array.from({ length: N }, () => cval(true));
        const scene = Sg.shader(effect(), Sg.group(cells.map((c, i) => item(i, c))));
        const task = runtime.compile(signature, compileScene(scene, { clear }));
        report["load shown, cell per subtree (ms)"] = frame(task);
        expect(await pixel()).toEqual([255, 255, 255, 255]);
        for (let k = 0; k < 3; k++) {
          report[`toggle off, 10k cells #${k} (ms)`] = ms(() => transact(() => { for (const c of cells) c.value = false; })) + frame(task);
          expect(await pixel()).toEqual([0, 0, 0, 0]);
          report[`toggle on, 10k cells #${k} (ms)`] = ms(() => transact(() => { for (const c of cells) c.value = true; })) + frame(task);
          expect(await pixel()).toEqual([255, 255, 255, 255]);
        }
        // one subtree alone
        report["toggle one of 10k (ms)"] = ms(() => transact(() => { cells[0]!.value = false; })) + frame(task);
        expect(await pixel()).toEqual([0, 0, 0, 0]);
        task.dispose();
      }

      // (3) for contrast: the same subtrees built and removed (what a bind does)
      {
        const shown = cval(false);
        const on = AVal.constant(true);
        const scene = Sg.shader(effect(), Sg.adaptive(shown.map((v): SgNode => v ? Sg.group(Array.from({ length: N }, (_, i) => item(i, on))) : Sg.empty)));
        const task = runtime.compile(signature, compileScene(scene, { clear }));
        frame(task);
        for (let k = 0; k < 2; k++) {
          report[`build 10k (ms) #${k}`] = ms(() => transact(() => { shown.value = true; })) + frame(task);
          expect(await pixel()).toEqual([255, 255, 255, 255]);
          report[`remove 10k (ms) #${k}`] = ms(() => transact(() => { shown.value = false; })) + frame(task);
          expect(await pixel()).toEqual([0, 0, 0, 0]);
        }
        task.dispose();
      }

      // eslint-disable-next-line no-console
      console.log("[active-toggle] " + JSON.stringify(Object.fromEntries(Object.entries(report).map(([k, v]) => [k, Math.round(v * 10) / 10]))));
      fbo.release();
      await device.queue.onSubmittedWorkDone();
      expect(errors).toEqual([]);
    } finally {
      device.destroy();
    }
  }, 120000);
});
