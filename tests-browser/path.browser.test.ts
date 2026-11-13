// `Sg.Path` on a real GPU (scene/path.ts): a circle made of two arcs, seen
// through a tiny window on its edge at the 45° point, where any polygonal
// approximation is furthest off. Every pixel must be inside exactly when
// its centre is inside the circle (but for the pixels the edge crosses).
// A band's concave inner edge, zoomed in, likewise. Then the path changes:
// an empty one draws nothing, a square draws.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap, cval, transact } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4f } from "@aardworx/wombat.base";
import { allocateFramebuffer, createFramebufferSignature, readTextureRegion } from "@aardworx/wombat.rendering/resources";
import { Runtime } from "@aardworx/wombat.rendering/runtime";

// the built package: native browser ESM needs the shader-marker imports the build strips
import { Sg, TraversalState, compileScene } from "../dist/scene/index.js";
import { requestRealDevice } from "./_realGpu.js";

const W = 64, H = 64;
// the window: 0.01 wide around (cos 45°, sin 45°) — 64 px over 0.01 units
const c = Math.SQRT1_2, half = 0.005;

describe("Sg.Path (real GPU)", () => {
  it("a circle's edge is exact at any zoom; the path follows changes", async () => {
    const device = await requestRealDevice();
    const errors: string[] = [];
    device.onuncapturederror = (e) => errors.push(e.error.message);
    try {
      const runtime = new Runtime({ device });
      const signature = createFramebufferSignature({ colors: { Colors: "rgba8unorm" }, depthStencil: { format: "depth24plus" } });
      const fbo = allocateFramebuffer(device, signature, AVal.constant({ width: W, height: H }), { extraUsage: GPUTextureUsage.COPY_SRC });
      fbo.acquire();
      const clear = { colors: HashMap.empty<string, V4f>().add("Colors", new V4f(0, 0, 0, 0)), depth: 1 };
      const view = Trafo3d.viewTrafoRH(new V3d(0, 0, 1), new V3d(0, 1, 0), new V3d(0, 0, -1));
      const proj = Trafo3d.orthoProjectionRH(c - half, c + half, c - half, c + half, 0.1, 10);
      const initial = TraversalState.empty.withViewport(AVal.constant({ width: W, height: H })).withCamera(AVal.constant(view), AVal.constant(proj));
      const circle = "M 1 0 A 1 1 0 0 1 -1 0 A 1 1 0 0 1 1 0 Z";
      const path = cval(circle);
      const scene = Sg.Path({ path, Color: new V4f(1, 0, 0, 1) });
      const task = runtime.compile(signature, compileScene(scene, { clear, initialState: initial }));
      const image = async (): Promise<Uint8Array> => {
        task.run(fbo.getValue(AdaptiveToken.top) as never, AdaptiveToken.top);
        await device.queue.onSubmittedWorkDone();
        const tex = fbo.getValue(AdaptiveToken.top).colorTextures!.tryFind("Colors")!;
        return new Uint8Array(await readTextureRegion(device, tex, { origin: [0, 0], size: [W, H] }));
      };
      // pixel (x, y), row 0 at the top: its centre in the plane
      const at = (x: number, y: number): [number, number] =>
        [c - half + (x + 0.5) / W * 2 * half, c + half - (y + 0.5) / H * 2 * half];
      const px = 2 * half / W;

      let img = await image();
      let wrong = 0, inside = 0, outside = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const [u, v] = at(x, y);
        const r = Math.hypot(u, v);
        const red = img[(y * W + x) * 4]! > 127;
        if (Math.abs(r - 1) < px) continue;          // the edge crosses this pixel
        if (red !== r < 1) wrong++;
        if (red) inside++; else outside++;
      }
      expect(inside).toBeGreaterThan(1000);
      expect(outside).toBeGreaterThan(1000);
      expect(wrong).toBe(0);

      // a band (the stroke of an arc): its inner edge is concave, the outer arc's curve
      // triangle reaches under it (wombat.base separateCurveHulls); zoomed on the inner edge
      transact(() => { path.value = "M 10 0 A 10 10 0 0 1 0 10 L 0 9 A 9 9 0 0 0 9 0 Z"; });
      const ri = 9, ci = ri * Math.SQRT1_2;
      const bandTask = runtime.compile(signature, compileScene(scene, { clear, initialState: TraversalState.empty
        .withViewport(AVal.constant({ width: W, height: H }))
        .withCamera(AVal.constant(view), AVal.constant(Trafo3d.orthoProjectionRH(ci - half, ci + half, ci - half, ci + half, 0.1, 10))) }));
      bandTask.run(fbo.getValue(AdaptiveToken.top) as never, AdaptiveToken.top);
      await device.queue.onSubmittedWorkDone();
      {
        const tex = fbo.getValue(AdaptiveToken.top).colorTextures!.tryFind("Colors")!;
        const bi = new Uint8Array(await readTextureRegion(device, tex, { origin: [0, 0], size: [W, H] }));
        let bandWrong = 0, filled = 0, empty = 0;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const u = ci - half + (x + 0.5) / W * 2 * half, v = ci + half - (y + 0.5) / H * 2 * half;
          const r = Math.hypot(u, v);
          if (Math.abs(r - ri) < px) continue;
          const red = bi[(y * W + x) * 4]! > 127;
          if (red !== r > ri) bandWrong++;
          if (red) filled++; else empty++;
        }
        expect(filled).toBeGreaterThan(1000);
        expect(empty).toBeGreaterThan(1000);
        expect(bandWrong).toBe(0);
      }
      bandTask.dispose();

      transact(() => { path.value = ""; });
      img = await image();
      expect(img.some((v, i) => i % 4 === 0 && v > 0)).toBe(false);

      // a square over the whole window
      transact(() => { path.value = `M 0 0 L 1 0 L 1 1 L 0 1 Z`; });
      img = await image();
      expect(img.every((v, i) => i % 4 !== 0 || v > 127)).toBe(true);
      task.dispose();

      // analytic anti-aliasing: full more than 1.5 px inside, empty more than 1.5 px outside,
      // and a ramp across the edge
      const aaTask = runtime.compile(signature, compileScene(Sg.Path({ path: circle, aa: "alpha-blending", Color: new V4f(1, 0, 0, 1) }), { clear, initialState: initial }));
      aaTask.run(fbo.getValue(AdaptiveToken.top) as never, AdaptiveToken.top);
      await device.queue.onSubmittedWorkDone();
      const tex = fbo.getValue(AdaptiveToken.top).colorTextures!.tryFind("Colors")!;
      img = new Uint8Array(await readTextureRegion(device, tex, { origin: [0, 0], size: [W, H] }));
      let bad = 0, ramp = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const [u, v] = at(x, y);
        const d = (Math.hypot(u, v) - 1) / px;          // signed distance in pixels
        const red = img[(y * W + x) * 4]!;
        if (d < -1.5 && red < 250) bad++;
        if (d > 1.5 && red > 5) bad++;
        if (Math.abs(d) < 0.25 && red > 60 && red < 195) ramp++;
      }
      expect(bad).toBe(0);
      expect(ramp).toBeGreaterThan(20);
      aaTask.dispose();
      expect(errors).toEqual([]);
    } finally { device.destroy(); }
  });
});
