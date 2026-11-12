// A pixel pick belongs to the frame its pick buffer was drawn in. The
// buffer holds that frame's view-space position (Mode B) or depth (Mode A);
// taken through a camera that moved since, it is a different, wrong world
// point (FableCube dogfood/poetry.md: gestures "stuck, then twice the
// speed"). The producer records each frame's camera on the readback
// (`PickArgminResult.camera`); the event's world position, depth and eye
// are then all that frame's. A ray (BVH) pick is cast at dispatch and keeps
// the current camera.

import { describe, expect, it } from "vitest";
import { AVal, cval, transact } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V4d } from "@aardworx/wombat.base";

import { PickDispatcher } from "../src/scene/picking/dispatcher.js";
import { PickRegistry } from "../src/scene/picking/registry.js";
import type { PickArgminResult } from "../src/scene/picking/pickArgminCompute.js";
import { pixelWinner } from "./pickArgminTestUtil.js";
import type { SceneEvent } from "../src/scene/picking/sceneEvent.js";

const W = 200, H = 100;
function makeCanvas(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = W; c.height = H; document.body.appendChild(c);
  c.getBoundingClientRect = (): DOMRect => {
    const r = { x: 0, y: 0, top: 0, left: 0, right: W, bottom: H, width: W, height: H };
    return { ...r, toJSON: () => r } as DOMRect;
  };
  return c;
}
function pevent(canvas: HTMLCanvasElement, type: string, x: number, y: number): void {
  const Ctor = (globalThis as { PointerEvent?: typeof PointerEvent }).PointerEvent
            ?? (globalThis as { MouseEvent: typeof MouseEvent }).MouseEvent;
  canvas.dispatchEvent(new Ctor(type, { clientX: x, clientY: y, bubbles: true, cancelable: true, button: 0, buttons: 1, pointerId: 1 }));
}
async function flush(): Promise<void> { for (let i = 0; i < 8; i++) await Promise.resolve(); }

const proj = Trafo3d.perspectiveProjectionRHFov(Math.PI / 3, W / H, 0.1, 100);
/** The camera of frame k: 10 above the plane z = 0, moved 0.5 along x per frame. */
const camera = (k: number): Trafo3d => Trafo3d.viewTrafoRH(new V3d(0.5 * k, 0, 10), new V3d(0, 1, 0), new V3d(0, 0, -1));
/** A fixed world point on the plane. */
const P = new V3d(1.25, 0.5, 0);

/** Where `p` is in frame `cam`: device pixel and NDC depth. */
function project(cam: Trafo3d, p: V3d): { px: number; py: number; ndcZ: number } {
  const v = cam.forward.transformPos(p);
  const c4 = proj.forward.transform(new V4d(v.x, v.y, v.z, 1));
  const clip = new V3d(c4.x / c4.w, c4.y / c4.w, c4.z / c4.w);
  return { px: (clip.x + 1) / 2 * W - 0.5, py: (1 - clip.y) / 2 * H - 0.5, ndcZ: clip.z };
}

function setup() {
  const reg = new PickRegistry();
  const view = cval(camera(0));
  const projA = AVal.constant(proj);
  const seen: SceneEvent[] = [];
  const id = reg.acquire({
    handlers: [{ handlers: { bubble: { OnPointerMove: (e: SceneEvent) => { seen.push(e); } } }, local2World: () => AVal.constant(Trafo3d.identity) }],
    cursor: undefined, pickThrough: false,
    active: AVal.constant(true),
    view, proj: projA,
    model: () => AVal.constant(Trafo3d.identity), pixelSnapRadius: AVal.constant(1),
  });
  const canvas = makeCanvas();
  const d = new PickDispatcher(reg, () => AVal.force(view), () => proj, () => canvas.getBoundingClientRect());
  return { reg, view, projA, seen, id, canvas, d };
}

for (const mode of ["A", "B"] as const) {
  describe(`pixel pick taken from an earlier frame — mode ${mode}`, () => {
    for (const recorded of [true, false]) {
      it(recorded
        ? "with the frame's camera: the world point, depth and eye are that frame's on every event"
        : "without it (as before): the point drifts by the camera's motion", async () => {
        const { view, projA, seen, id, canvas, d } = setup();
        let next: PickArgminResult | undefined;
        const detach = d.attach(canvas, async () => next);
        for (let k = 0; k < 5; k++) {
          // frame k is drawn with camera k: P lies under (px, py) there ...
          const frame = camera(k);
          const at = project(frame, P);
          const px = Math.round(at.px), py = Math.round(at.py);
          const vp = frame.forward.transformPos(P);
          const r = pixelWinner(id, mode === "A"
            ? { px, py, mode, slot2: at.ndcZ }
            : { px, py, mode, slot1: vp.x, slot2: vp.y, slot3: vp.z });
          next = recorded ? { ...r, camera: { view: frame, proj, viewAval: view, projAval: projA } } : r;
          // ... and the camera moves on before the readback is resolved
          transact(() => { view.value = camera(k + 1); });
          pevent(canvas, "pointermove", px, py);
          await flush();
        }
        detach();
        expect(seen.length).toBe(5);
        // mode A rebuilds the point from the pixel's centre: within a pixel
        const tol = mode === "A" ? 0.06 : 1e-9;
        const err = seen.map((e) => e.worldPos.sub(P).length());
        if (recorded) {
          for (const [k, e] of seen.entries()) {
            expect(err[k], `event ${k}`).toBeLessThan(tol);
            // the eye and the depth are frame k's, the one the point came from
            expect(e.location.viewTrafo.backward.transformPos(V3d.zero).sub(new V3d(0.5 * k, 0, 10)).length()).toBeLessThan(1e-9);
            if (mode === "B") expect(e.depth).toBeCloseTo(project(camera(k), P).ndcZ, 9);
          }
        } else {
          // the old behaviour: off by the camera's motion (0.5) on every event
          for (const x of err) expect(x).toBeGreaterThan(0.4);
        }
      });
    }
  });
}
