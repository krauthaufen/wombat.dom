// A pixel pick's readback carries the camera of the frame it read
// (pickProducer.ts): the pick buffer is the LAST frame rendered before the
// readback was submitted, and the camera may have moved on by the time the
// result is resolved. Real GPU: a plane under a camera that moves every
// frame; after each frame the camera moves on BEFORE the pick, and the
// picked world point must be the plane point under the pixel in the frame
// that was drawn (within pick precision) — taken through the moved camera
// it is off by the camera's motion.

import { describe, expect, it } from "vitest";
import { AVal, AdaptiveToken, HashMap, cval, transact } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4d, V4f } from "@aardworx/wombat.base";
import { effect } from "@aardworx/wombat.shader";
import type { BufferView, DrawCall } from "@aardworx/wombat.rendering/core";
import { ElementType } from "@aardworx/wombat.rendering/core";
import { Runtime } from "@aardworx/wombat.rendering/runtime";

// the built package: native browser ESM needs the shader-marker imports the build strips
import { DefaultSurfaces, Sg, renderToPickable } from "../dist/scene/index.js";
import { requestRealDevice } from "./_realGpu.js";

const W = 128, H = 64;
const proj = Trafo3d.perspectiveProjectionRHFov(Math.PI / 3, W / H, 1, 100);
/** Frame k: 10 above the plane z = 0, moved 0.5 along x per frame. */
const camera = (k: number): Trafo3d => Trafo3d.viewTrafoRH(new V3d(0.5 * k, 0, 10), new V3d(0, 1, 0), new V3d(0, 0, -1));
/** The plane point under device pixel (x, y) (its centre) in frame `cam`. */
function underPixel(cam: Trafo3d, x: number, y: number): V3d {
  const ndc = (z: number): V3d => {
    const v = proj.backward.transform(new V4d(2 * (x + 0.5) / W - 1, 1 - 2 * (y + 0.5) / H, z, 1));
    return cam.backward.transformPos(new V3d(v.x / v.w, v.y / v.w, v.z / v.w));
  };
  const a = ndc(0), b = ndc(1);
  const t = a.z / (a.z - b.z);
  return a.add(b.sub(a).mul(t));
}

describe("pick readback carries its frame's camera (real GPU)", () => {
  it("the picked world point is the drawn frame's, while the camera moves every frame", async () => {
    const device = await requestRealDevice();
    try {
      const runtime = new Runtime({ device });
      const s = 100;
      const data = new Float32Array([-s, -s, 0, s, -s, 0, -s, s, 0, -s, s, 0, s, -s, 0, s, s, 0]);
      const positions: BufferView = { buffer: AVal.constant({ kind: "host", data, sizeBytes: data.byteLength }), elementType: ElementType.V3f };
      const plane = Sg.leaf({
        vertexAttributes: HashMap.empty<string, BufferView>().add("Positions", positions),
        drawCall: AVal.constant<DrawCall>({ kind: "non-indexed", vertexCount: 6, instanceCount: 1, firstVertex: 0, firstInstance: 0 }),
      });
      const view = cval(camera(0));
      const r = renderToPickable(runtime, device, Sg.shader(effect(DefaultSurfaces.trafo(), DefaultSurfaces.constantColor(new V4f(1, 1, 1, 1))), Sg.onPointerMove(() => {})(plane)), {
        size: AVal.constant({ width: W, height: H }), view, proj: AVal.constant(proj),
      });
      const x = 90, y = 20;
      const errs: number[] = [], camErrs: number[] = [];
      for (let k = 0; k < 4; k++) {
        transact(() => { view.value = camera(k); });
        r.framebuffer.getValue(AdaptiveToken.top);           // frame k is drawn
        await device.queue.onSubmittedWorkDone();
        transact(() => { view.value = camera(k + 1); });     // ... and the camera moves on
        const hit = (await r.pick.pickAt(x, y))?.hit;
        expect(hit, `frame ${k}: a hit`).toBeDefined();
        const cam = hit!.camera?.view ?? camera(k + 1);
        camErrs.push(cam.backward.transformPos(V3d.zero).sub(new V3d(0.5 * k, 0, 10)).length());
        errs.push(cam.backward.transformPos(hit!.viewPos).sub(underPixel(camera(k), x, y)).length());
      }
      r.dispose();
      // the hit is relative to frame k's camera, and its point is frame k's
      for (const e of camErrs) expect(e).toBeLessThan(1e-9);
      // within depth precision (24-bit depth over near 1 .. far 100 at ~10)
      for (const e of errs) expect(e).toBeLessThan(0.02);
    } finally { device.destroy(); }
  });
});
