// Build a combined IFramebuffer that pairs the canvas's color +
// depth attachments with a freshly-allocated rgba32float `pickId`
// attachment. The pick chain shaders write to `outColor` (slot 0)
// and `pickId` (slot 1); both targets must be present in the
// framebuffer signature for pipeline creation to succeed.
//
// Lifecycle:
//   - `pickTexture: aval<GPUTexture>` — re-allocated whenever the
//     canvas size changes. Always sized to canvas device pixels.
//   - `pickFramebuffer: aval<IFramebuffer>` — depends on both the
//     canvas FB aval and the pick texture. Re-evaluating it pulls a
//     fresh swap-chain view + the current pick view.
//   - `dispose()` — destroys the latest pick texture and stops the
//     resize subscription.
//
// MSAA: v1 assumes sampleCount = 1 on the canvas. The pickId target
// is always single-sample; resolving multi-sample pickIds correctly
// needs a custom compute pass (averaging would smear two distinct
// ids into a fractional value that decodes to garbage). We document
// the future plan with a TODO instead of doing it half-right.

import {
  AVal,
  HashMap,
  type aval,
} from "@aardworx/wombat.adaptive";
import type {
  IFramebuffer,
  FramebufferSignature,
} from "@aardworx/wombat.rendering/core";
import { createFramebufferSignature } from "@aardworx/wombat.rendering";

export interface CanvasLikeAttachment {
  readonly framebuffer: aval<IFramebuffer>;
  readonly size: aval<{ readonly width: number; readonly height: number }>;
  readonly signature: FramebufferSignature;
}

export interface PickFramebuffer {
  readonly pickFramebuffer: aval<IFramebuffer>;
  /** Latest allocated pickId texture; resizes follow the canvas. */
  readonly pickTexture: aval<GPUTexture>;
  /** Combined signature (canvas color + pickId + canvas depth). */
  readonly signature: FramebufferSignature;
  dispose(): void;
}

const PICK_FORMAT: GPUTextureFormat = "rgba32float";
const PICK_NAME = "pickId";

export function createPickFramebuffer(
  device: GPUDevice,
  attachment: CanvasLikeAttachment,
  opts: {
    /** Color-attachment name in the canvas signature. Default `"outColor"`. */
    readonly colorAttachmentName?: string;
  } = {},
): PickFramebuffer {
  const colorName = opts.colorAttachmentName ?? "outColor";
  if (attachment.signature.sampleCount !== 1) {
    // TODO: MSAA pick attachment. WebGPU's render-pass resolve does
    // a per-sample average — fine for color, wrong for integer ids
    // packed as f32. The fix is a separate compute resolve that
    // picks the nearest-coverage sample, run after the pass.
    console.warn(
      "[pickFramebuffer] MSAA canvas detected; pick attachment will be single-sample, " +
      "which conflicts with WebGPU's requirement that all attachments share sampleCount. " +
      "Drop sampleCount to 1 on the RenderControl until the compute-resolve path lands.",
    );
  }

  // Build the combined signature once — its shape is fixed for the
  // lifetime of the framebuffer (only sizes change). Pipelines key
  // off `signature`, so it has to be a stable object across reads.
  let combinedColors: Record<string, GPUTextureFormat> = {};
  for (const [name, fmt] of attachment.signature.colors) {
    combinedColors[name] = fmt;
  }
  combinedColors[PICK_NAME] = PICK_FORMAT;
  const signature: FramebufferSignature = createFramebufferSignature({
    colors: combinedColors,
    sampleCount: 1,
    ...(attachment.signature.depthStencil !== undefined
      ? { depthStencil: { format: attachment.signature.depthStencil.format } }
      : {}),
  });

  // pickTexture: aval<GPUTexture> driven by canvas size. We track
  // the most recently allocated texture so dispose() can destroy it
  // even if no consumer ever forces the aval again.
  let latestPick: GPUTexture | undefined;
  let latestKey: string | undefined;
  const pickTexture: aval<GPUTexture> = attachment.size.map((sz) => {
    const w = Math.max(1, sz.width);
    const h = Math.max(1, sz.height);
    const key = `${w}x${h}`;
    if (latestPick !== undefined && latestKey === key) return latestPick;
    if (latestPick !== undefined) {
      try { latestPick.destroy(); } catch { /* already gone */ }
    }
    const tex = device.createTexture({
      size: { width: w, height: h, depthOrArrayLayers: 1 },
      format: PICK_FORMAT,
      // COPY_SRC is what the readback path needs;
      // RENDER_ATTACHMENT is what the render pass needs.
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      sampleCount: 1,
      label: "pick.id",
    });
    latestPick = tex;
    latestKey = key;
    return tex;
  });

  // Combined framebuffer: zip the canvas FB with the pick texture,
  // produce a fresh IFramebuffer that merges colors and forwards
  // depthStencil from the canvas FB.
  const pickFramebuffer: aval<IFramebuffer> = AVal.zip(attachment.framebuffer, pickTexture).map((fb, pick) => {
    const pickView = pick.createView({ label: "pick.id.view" });
    let colors = HashMap.empty<string, GPUTextureView>();
    for (const [name, view] of fb.colors) colors = colors.add(name, view);
    colors = colors.add(PICK_NAME, pickView);
    void colorName; // colorName is informational; we copy whatever the canvas FB exposes.
    const out: IFramebuffer = {
      signature,
      colors,
      ...(fb.depthStencil !== undefined ? { depthStencil: fb.depthStencil } : {}),
      ...(fb.depthStencilTexture !== undefined ? { depthStencilTexture: fb.depthStencilTexture } : {}),
      width: fb.width,
      height: fb.height,
    };
    return out;
  });

  return {
    pickFramebuffer,
    pickTexture,
    signature,
    dispose() {
      if (latestPick !== undefined) {
        try { latestPick.destroy(); } catch { /* already gone */ }
        latestPick = undefined;
        latestKey = undefined;
      }
    },
  };
}
