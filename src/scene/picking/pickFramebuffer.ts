// Build a combined IFramebuffer that pairs the canvas's color +
// depth attachments with a freshly-allocated rgba32float `pickId`
// attachment. The pick chain shaders write to `outColor` (slot 0)
// and `pickId` (slot 1); both targets must be present in the
// framebuffer signature for pipeline creation to succeed.
//
// Two paths:
//
//   sampleCount === 1:
//     - The pickId target is single-sample. The same texture acts
//       as render target AND as readback source.
//
//   sampleCount > 1:
//     - The pickId target is MSAA (so it can co-attach with the MSAA
//       canvas color). It is RENDER_ATTACHMENT | TEXTURE_BINDING so
//       the resolve compute can sample it via texture_multisampled_2d.
//     - A second, single-sample `resolvedPickTexture` is allocated
//       (STORAGE_BINDING | TEXTURE_BINDING | COPY_SRC). The custom
//       compute pass (see `pickResolveCompute.ts`) majority-votes
//       slot-0 across samples and writes the winning sample's full
//       vec4 into it. Readback copies from THIS texture, not the
//       MSAA one.
//
// The hardware `resolveTarget` average is wrong for pickIds: two
// distinct integer ids averaged at silhouettes produce a meaningless
// fractional value that decodes to a registered-but-unrelated id.

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

import { createPickResolveCompute, type PickResolveCompute } from "./pickResolveCompute.js";

export interface CanvasLikeAttachment {
  readonly framebuffer: aval<IFramebuffer>;
  readonly size: aval<{ readonly width: number; readonly height: number }>;
  readonly signature: FramebufferSignature;
}

export interface PickFramebuffer {
  readonly pickFramebuffer: aval<IFramebuffer>;
  /**
   * The texture suitable for readback (single-sample rgba32float).
   * sampleCount === 1: same as the render-target pick texture.
   * sampleCount  >  1: the resolved (post-compute) texture.
   */
  readonly readbackPickTexture: aval<GPUTexture>;
  /** Combined signature (canvas color + pickId + canvas depth). */
  readonly signature: FramebufferSignature;
  /**
   * When MSAA is in play, the caller must invoke this on a command
   * encoder AFTER the render pass and BEFORE submitting any pickId
   * readback copies. Undefined for sampleCount === 1.
   */
  readonly maybeRunResolve?: (encoder: GPUCommandEncoder) => void;
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
  const sampleCount = attachment.signature.sampleCount;
  const isMSAA = sampleCount > 1;

  // Build the combined signature once — its shape is fixed for the
  // lifetime of the framebuffer (only sizes change). Pipelines key
  // off `signature`, so it has to be a stable object across reads.
  let combinedColors: Record<string, GPUTextureFormat> = {};
  for (const name of attachment.signature.colorNames) {
    const fmt = attachment.signature.colors.tryFind(name);
    if (fmt === undefined) continue;
    combinedColors[name] = fmt;
  }
  combinedColors[PICK_NAME] = PICK_FORMAT;
  const signature: FramebufferSignature = createFramebufferSignature({
    colors: combinedColors,
    sampleCount,
    ...(attachment.signature.depthStencil !== undefined
      ? { depthStencil: { format: attachment.signature.depthStencil.format } }
      : {}),
  });

  // Lazy compute-resolve pipeline (one per sampleCount, cached on
  // device by pickResolveCompute itself). We hold the per-instance
  // wrapper so we can dispose its uniform buffer.
  let resolveCompute: PickResolveCompute | undefined;
  if (isMSAA) {
    resolveCompute = createPickResolveCompute(device, sampleCount);
  }

  // Track the latest allocations so dispose() always finds them and
  // resize-driven re-allocation can free the old textures.
  let latestRenderPick: GPUTexture | undefined;
  let latestResolvedPick: GPUTexture | undefined;
  let latestKey: string | undefined;
  let latestWidth = 0;
  let latestHeight = 0;

  const reallocate = (w: number, h: number): { renderTex: GPUTexture; resolvedTex: GPUTexture } => {
    const renderUsage = isMSAA
      ? GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
      : GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC;
    const renderTex = device.createTexture({
      size: { width: w, height: h, depthOrArrayLayers: 1 },
      format: PICK_FORMAT,
      usage: renderUsage,
      sampleCount,
      label: isMSAA ? `pick.id.ms${sampleCount}` : "pick.id",
    });
    let resolvedTex: GPUTexture;
    if (isMSAA) {
      resolvedTex = device.createTexture({
        size: { width: w, height: h, depthOrArrayLayers: 1 },
        format: PICK_FORMAT,
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC,
        sampleCount: 1,
        label: "pick.id.resolved",
      });
    } else {
      resolvedTex = renderTex;
    }
    return { renderTex, resolvedTex };
  };

  // Drive both textures off canvas size. We expose both as separate
  // avals (readbackPickTexture below) but realloc once per resize.
  const renderPickTexture: aval<GPUTexture> = attachment.size.map((sz) => {
    const w = Math.max(1, sz.width);
    const h = Math.max(1, sz.height);
    const key = `${w}x${h}`;
    if (latestRenderPick !== undefined && latestKey === key) return latestRenderPick;
    if (latestRenderPick !== undefined) {
      try { latestRenderPick.destroy(); } catch { /* already gone */ }
    }
    if (latestResolvedPick !== undefined && latestResolvedPick !== latestRenderPick) {
      try { latestResolvedPick.destroy(); } catch { /* already gone */ }
    }
    const { renderTex, resolvedTex } = reallocate(w, h);
    latestRenderPick = renderTex;
    latestResolvedPick = resolvedTex;
    latestKey = key;
    latestWidth = w;
    latestHeight = h;
    return renderTex;
  });

  const readbackPickTexture: aval<GPUTexture> = renderPickTexture.map(() => latestResolvedPick!);

  // Combined framebuffer: zip the canvas FB with the pick texture,
  // produce a fresh IFramebuffer that merges colors and forwards
  // depthStencil from the canvas FB.
  const pickFramebuffer: aval<IFramebuffer> = AVal.zip(attachment.framebuffer, renderPickTexture).map((fb, pick) => {
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

  const result: PickFramebuffer = {
    pickFramebuffer,
    readbackPickTexture,
    signature,
    dispose() {
      if (latestResolvedPick !== undefined && latestResolvedPick !== latestRenderPick) {
        try { latestResolvedPick.destroy(); } catch { /* already gone */ }
      }
      if (latestRenderPick !== undefined) {
        try { latestRenderPick.destroy(); } catch { /* already gone */ }
      }
      latestRenderPick = undefined;
      latestResolvedPick = undefined;
      latestKey = undefined;
      if (resolveCompute !== undefined) {
        resolveCompute.dispose();
        resolveCompute = undefined;
      }
    },
  };

  if (isMSAA) {
    (result as { maybeRunResolve?: (encoder: GPUCommandEncoder) => void }).maybeRunResolve = (encoder) => {
      if (resolveCompute === undefined) return;
      if (latestRenderPick === undefined || latestResolvedPick === undefined) return;
      if (latestRenderPick === latestResolvedPick) return;
      const srcView = latestRenderPick.createView({ label: "pick.id.ms.view" });
      const dstView = latestResolvedPick.createView({ label: "pick.id.resolved.view" });
      resolveCompute.resolve(encoder, srcView, dstView, latestWidth, latestHeight);
    };
  }

  return result;
}
