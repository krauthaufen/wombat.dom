// Anti-aliased `Sg.Text` drawn offscreen (no render control mounted): its
// viewport (the AA ramp's width in pixels) and its depth convention must be
// those of the scene it is drawn in. They used to be the ambient render
// control's: with none mounted, a 1x1 viewport and the identity projection,
// under which the text's depth test came out reversed and nothing drew.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import { Font } from "@aardworx/wombat.base/font";
import { allocateFramebuffer, createFramebufferSignature, readTextureRegion } from "@aardworx/wombat.rendering/resources";
import { Runtime } from "@aardworx/wombat.rendering/runtime";

import { Sg, TraversalState, compileScene } from "../dist/scene/index.js";
import { requestRealDevice } from "./_realGpu.js";
// @ts-expect-error a vite url import
import latoUrl from "../examples/path-rendering/src/lato.ttf?url";

const W = 128, H = 128;

describe("Sg.Text offscreen (real GPU)", () => {
  it("anti-aliased text draws with a one-pixel ramp, whatever the ambient control", async () => {
    const device = await requestRealDevice();
    const errors: string[] = [];
    device.onuncapturederror = (e) => errors.push(e.error.message);
    try {
      const font = await Font.load(latoUrl);
      const runtime = new Runtime({ device });
      const signature = createFramebufferSignature({ colors: { Colors: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
      const fbo = allocateFramebuffer(device, signature, AVal.constant({ width: W, height: H }), { extraUsage: GPUTextureUsage.COPY_SRC });
      fbo.acquire();
      const clear = { colors: HashMap.empty<string, V4f>().add("Colors", new V4f(0, 0, 0, 0)), depth: 1 };
      const view = Trafo3d.viewTrafoRH(new V3d(0, 0, 1), new V3d(0, 1, 0), new V3d(0, 0, -1));
      // one em over the target: the "I" is a bar about 12 px wide and 90 px tall
      const proj = Trafo3d.orthoProjectionRH(-0.4, 0.6, -0.15, 0.85, 0.1, 10);
      const initial = TraversalState.empty.withViewport(AVal.constant({ width: W, height: H })).withCamera(AVal.constant(view), AVal.constant(proj));
      const scene = Sg.Text({ font, text: "I", aa: "alpha-blending", Color: new V4f(1, 0, 0, 1) }) as never;
      const task = runtime.compile(signature, compileScene(scene, { clear, initialState: initial }));
      task.run(fbo.getValue(AdaptiveToken.top) as never, AdaptiveToken.top);
      await device.queue.onSubmittedWorkDone();
      const tex = fbo.getValue(AdaptiveToken.top).colorTextures!.tryFind("Colors")!;
      const img = new Uint8Array(await readTextureRegion(device, tex, { origin: [0, 0], size: [W, H] }));
      let full = 0, ramp = 0;
      for (let i = 0; i < W * H; i++) {
        const a = img[i * 4 + 3]!;
        if (a > 250) full++; else if (a > 5) ramp++;
      }
      expect(full).toBeGreaterThan(500);
      // the ramp is the bar's outline, about two pixels wide
      expect(ramp).toBeGreaterThan(100);
      expect(ramp).toBeLessThan(full);
      task.dispose();
      expect(errors).toEqual([]);
    } finally {
      device.destroy();
    }
  }, 60000);
});
