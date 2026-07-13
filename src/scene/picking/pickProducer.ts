// PickProducer — the reusable "scene → RenderTask + pick artifacts"
// unit. Mirrors Aardvark.Dom's `PickProducer` split out of
// `SceneHandler` (commit 8bb820b): everything a pickable render needs
// EXCEPT the window/dispatcher half — the pick framebuffer (color +
// pickId + depth), the id registry, the per-id GPU metadata mirror,
// the argmin resolve kernel, and the compiled render task.
//
// Two consumers:
//   - `RenderControl` wraps one around the canvas attachment and wires
//     a `PickDispatcher` (DOM pointer events) to `pickPixel`.
//   - `renderToPickable` wraps one around an offscreen attachment and
//     exposes `pickPixel` through an `IRenderPickContext` so picking
//     recurses into offscreen ("portal") scenes.

import { AVal, HashMap, type AdaptiveToken, type aval } from "@aardworx/wombat.adaptive";
import { V4f, type Trafo3d } from "@aardworx/wombat.base";
import type { Runtime } from "@aardworx/wombat.rendering";
import type {
  ClearColor,
  ClearValues,
  Effect,
  FramebufferSignature,
  IFramebuffer,
  IRenderTask,
} from "@aardworx/wombat.rendering/core";

import { compileScene } from "../compile.js";
import type { SgNode } from "../sg.js";
import { TraversalState } from "../traversalState.js";
import { transparencyTask, type OitMode } from "../transparency.js";
import { createPickArgminCompute, type PickArgminResult } from "./pickArgminCompute.js";
import { createPickFramebuffer, PICK_NAME, type CanvasLikeAttachment } from "./pickFramebuffer.js";
import { PickMetadata } from "./pickMetadata.js";
import { PickRegistry } from "./registry.js";

export interface PickProducerOptions {
  /** View trafo (world → view) — threaded into the traversal state and
   *  exposed for pick unprojection. */
  readonly view: aval<Trafo3d>;
  /** Projection trafo (view → clip). */
  readonly proj: aval<Trafo3d>;
  /** Per-frame clock threaded into the traversal state. */
  readonly time: aval<number>;
  /** Order-independent transparency (`true` = default mode). */
  readonly transparency?: boolean | OitMode;
  /** Fallback effect for leaves outside any `<Sg Shader=…>` scope. */
  readonly defaultEffect?: Effect;
  /** User clear values; pickId / Colors / depth defaults are injected. */
  readonly clear?: ClearValues;
  /** Color-attachment name in the target signature. Default `"Colors"`. */
  readonly colorAttachmentName?: string;
}

export interface PickProducer {
  /** Per-producer pick-id space. Ids are only meaningful against THIS
   *  registry — nested producers each have their own. */
  readonly registry: PickRegistry;
  /** Combined signature (target color + pickId + target depth). */
  readonly signature: FramebufferSignature;
  /** The combined framebuffer the task renders into. */
  readonly framebuffer: aval<IFramebuffer>;
  /** Render-target size (device pixels). */
  readonly size: aval<{ readonly width: number; readonly height: number }>;
  readonly view: aval<Trafo3d>;
  readonly proj: aval<Trafo3d>;
  /** Single-sample rgba32float pick texture suitable for sampling/readback. */
  readonly readbackPickTexture: aval<GPUTexture>;
  /** The compiled render task (exposed for timing / inspection). */
  readonly task: IRenderTask;
  /**
   * Render one frame: run the task into the combined framebuffer and,
   * under MSAA, submit the majority-vote pick resolve right after.
   */
  run(token: AdaptiveToken): void;
  /**
   * Resolve the single nearest valid pick pixel at device coords
   * (x, y) via the GPU argmin kernel. At most one dispatch + readback
   * is in flight; further requests queue. `coalesce: true` (the
   * default, for pointer-move flooding) marks the request replaceable
   * — a newer request replaces a queued coalescable one, which
   * resolves `undefined`. Pass `false` for discrete events
   * (down/up/click): they always get a real resolution, in order.
   */
  pickPixel(x: number, y: number, coalesce?: boolean): Promise<PickArgminResult | undefined>;
  dispose(): void;
}

export function createPickProducer(
  runtime: Runtime,
  device: GPUDevice,
  target: CanvasLikeAttachment,
  sceneTree: SgNode,
  opts: PickProducerOptions,
): PickProducer {
  const colorName = opts.colorAttachmentName ?? "Colors";
  let disposed = false;

  const registry = new PickRegistry();
  const pickFb = createPickFramebuffer(device, target, { colorAttachmentName: colorName });

  // Initial traversal state — viewport + camera + clock up front.
  const initial = TraversalState.empty
    .withViewport(target.size)
    .withCamera(opts.view, opts.proj)
    .withTime(opts.time);

  // Always inject a per-frame clear for the pickId attachment; default
  // a color and depth clear when the user didn't supply one. Fragments
  // NOT covered by drawn geometry would otherwise retain last-frame
  // pickIds (loadOp defaults to "load") and the hit-test readback
  // decodes stale data. Depth defaults to 1.0 so the first frame has a
  // valid background to test against; color to transparent black so a
  // control over a styled background doesn't paint a hard rectangle.
  const userClear = opts.clear;
  const userColors = userClear?.colors ?? HashMap.empty<string, ClearColor>();
  const colorsWithDefaults = userColors.containsKey(colorName)
    ? userColors
    : userColors.add(colorName, new V4f(0, 0, 0, 0));
  const clearWithPick: ClearValues = {
    ...(userClear ?? {}),
    colors: colorsWithDefaults.add(PICK_NAME, new V4f(0, 0, 0, 0)),
    depth: userClear?.depth ?? 1.0,
  };

  // Lower + compile. When transparency is on, transparencyTask lowers
  // the scene itself via compileScene's pass hooks and threads the same
  // pick registry.
  const task = opts.transparency
    ? transparencyTask(runtime, device, pickFb.signature, target.size, sceneTree, {
        ...(typeof opts.transparency === "string" ? { mode: opts.transparency } : {}),
        // thread the user clear + depth convention: depth 0 = reversed-Z
        // (same signal the RenderControl's scene-root depthTest keys on)
        clearColor: colorsWithDefaults.get(colorName) as V4f, // always present (defaulted above)
        reversedZ: (userClear?.depth ?? 1.0) === 0,
        compile: {
          initialState: initial,
          autoUniforms: true,
          ...(opts.defaultEffect !== undefined ? { defaultEffect: opts.defaultEffect } : {}),
          picking: { registry },
        },
      })
    : runtime.compile(pickFb.signature, compileScene(sceneTree, {
        initialState: initial,
        ...(opts.defaultEffect !== undefined ? { defaultEffect: opts.defaultEffect } : {}),
        clear: clearWithPick,
        picking: { registry },
      }));

  // GPU argmin pick path. A per-id metadata buffer (snap radius + mode,
  // folding active/noEvents) mirrors the registry; the argmin kernel
  // finds THE single nearest valid pixel under the cursor and we read
  // back ~40 B.
  const pickMetadata = new PickMetadata(device);
  registry.attachObserver(pickMetadata);
  const argmin = createPickArgminCompute(device);

  // Serialized argmin queue: at most one dispatch+readback in flight
  // (the readback buffer is reused, so concurrent reads would clash).
  // Coalescable requests (pointer moves) get REPLACED by newer
  // coalescable ones — the replaced caller resolves `undefined`.
  // Non-coalescable requests (down/up/click, portal recursion) queue
  // FIFO so rapid event bursts (double-click!) never lose their
  // resolution.
  interface PickReq { x: number; y: number; coalesce: boolean; resolve: (r: PickArgminResult | undefined) => void }
  let pickInFlight = false;
  const pickQueue: PickReq[] = [];
  const runArgmin = async (req: PickReq): Promise<void> => {
    pickInFlight = true;
    try {
      if (disposed) { req.resolve(undefined); return; }
      pickMetadata.flush();
      const tex = AVal.force(pickFb.readbackPickTexture);
      const view = tex.createView({ label: "pick.argmin.src" });
      const enc = device.createCommandEncoder({ label: "pick.argmin.frame" });
      argmin.compute(enc, view, pickMetadata.buffer, req.x, req.y, tex.width, tex.height);
      argmin.copyResult(enc);
      device.queue.submit([enc.finish()]);
      req.resolve(await argmin.read());
    } catch {
      req.resolve(undefined);
    } finally {
      pickInFlight = false;
      const next = pickQueue.shift();
      if (next !== undefined) void runArgmin(next);
    }
  };
  const pickPixel = (x: number, y: number, coalesce = true): Promise<PickArgminResult | undefined> => {
    if (disposed) return Promise.resolve(undefined);
    return new Promise((resolve) => {
      const req: PickReq = { x, y, coalesce, resolve };
      if (!pickInFlight) { void runArgmin(req); return; }
      if (coalesce) {
        // Replace the newest queued coalescable request, if any.
        for (let i = pickQueue.length - 1; i >= 0; i--) {
          if (pickQueue[i]!.coalesce) {
            pickQueue[i]!.resolve(undefined);
            pickQueue[i] = req;
            return;
          }
        }
      }
      pickQueue.push(req);
    });
  };

  // For MSAA picking a custom compute resolve must run after the render
  // pass. wombat.rendering's `runFrame` does not expose its per-frame
  // command encoder, so we submit a SEPARATE encoder right after
  // `task.run`.
  const runResolve = pickFb.maybeRunResolve;

  return {
    registry,
    signature: pickFb.signature,
    framebuffer: pickFb.pickFramebuffer,
    size: target.size,
    view: opts.view,
    proj: opts.proj,
    readbackPickTexture: pickFb.readbackPickTexture,
    task,
    run(token: AdaptiveToken): void {
      if (disposed) return;
      task.run(pickFb.pickFramebuffer.getValue(token), token);
      if (runResolve !== undefined) {
        const enc = device.createCommandEncoder({ label: "pick.resolve.frame" });
        runResolve(enc);
        device.queue.submit([enc.finish()]);
      }
    },
    pickPixel,
    dispose(): void {
      if (disposed) return;
      disposed = true;
      registry.attachObserver(undefined);
      argmin.dispose();
      pickMetadata.dispose();
      task.dispose();
      pickFb.dispose();
    },
  };
}
