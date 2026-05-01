// Readback from the rgba32f pick attachment + slot decoder.
//
// Two readers:
//   - `readPickPixel` — single pixel (kept for tests / simple uses).
//   - `readPickRegion` — 33×33 disc-readback used by the spiral
//     hit-test (`dispatcher.ts`). Aligns bytesPerRow to 256 as
//     WebGPU requires for copyTextureToBuffer.
//
// Pooling is still future work — each readback allocates a fresh
// staging buffer. For one readback per pointer event that's fine;
// for sustained pointer-move under load we'd want a 2-3 buffer ring.

import type { SceneEventViewPos } from "./sceneEvent.js";
import { SNAP_RADIUS_MAX, SNAP_REGION_SIZE } from "./snapOffsets.js";

export interface PickPixel {
  readonly slot0: number;
  readonly slot1: number;
  readonly slot2: number;
  readonly slot3: number;
}

export interface DecodedPick {
  /** 0 ⇒ no hit. */
  readonly pickId: number;
  /** Mode-B fragment marker (slot0 was negative on the GPU). */
  readonly modeB: boolean;
  /**
   * Mode-B viewPos lifted straight from slots 1/2/3. Mode-A leaves
   * this undefined — its NDC depth lives in slot 2 and unprojection
   * needs the hit scope's view/proj, which the decoder doesn't see.
   * The dispatcher applies the unprojection.
   */
  readonly viewPos?: SceneEventViewPos;
  /** Raw decoded slot values, exposed for code that needs Mode-A's NDC depth (slot2) and normal/part (slot1, slot3). */
  readonly raw: PickPixel;
}

/**
 * Read one rgba32f pixel at (x, y) (canvas device-pixel coords) from
 * `pickTexture` and return the four float channels. Returns
 * `undefined` if the texture has been destroyed mid-flight or the
 * device was lost. Coordinates are clamped into bounds.
 */
export async function readPickPixel(
  device: GPUDevice,
  pickTexture: GPUTexture,
  x: number,
  y: number,
): Promise<PickPixel | undefined> {
  const w = pickTexture.width;
  const h = pickTexture.height;
  if (w <= 0 || h <= 0) return undefined;
  const cx = Math.max(0, Math.min(w - 1, x | 0));
  const cy = Math.max(0, Math.min(h - 1, y | 0));

  const BYTES_PER_ROW = 256;
  let buffer: GPUBuffer;
  try {
    buffer = device.createBuffer({
      size: BYTES_PER_ROW,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: "pick.readback",
    });
  } catch {
    return undefined;
  }

  try {
    const encoder = device.createCommandEncoder({ label: "pick.readback.encoder" });
    encoder.copyTextureToBuffer(
      { texture: pickTexture, origin: { x: cx, y: cy, z: 0 } },
      { buffer, bytesPerRow: BYTES_PER_ROW, rowsPerImage: 1 },
      { width: 1, height: 1, depthOrArrayLayers: 1 },
    );
    device.queue.submit([encoder.finish()]);

    await buffer.mapAsync(GPUMapMode.READ);
    const arr = new Float32Array(buffer.getMappedRange(0, 16).slice(0));
    buffer.unmap();
    return { slot0: arr[0]!, slot1: arr[1]!, slot2: arr[2]!, slot3: arr[3]! };
  } catch {
    return undefined;
  } finally {
    try { buffer.destroy(); } catch { /* already gone */ }
  }
}

// ---------------------------------------------------------------------------
// Region reader
// ---------------------------------------------------------------------------

/**
 * A 2D rectangle of decoded pixel slots. Stored as an interleaved
 * Float32Array of length `sizeX * sizeY * 4` (slot0..slot3 per
 * pixel). `originX` / `originY` are the texture-space coords of
 * the top-left of the region; the dispatcher converts an offset
 * `(dx, dy)` from the cursor into a region-local index by
 * subtracting the origin.
 */
export interface PickRegion {
  readonly data: Float32Array;
  readonly originX: number;
  readonly originY: number;
  readonly sizeX: number;
  readonly sizeY: number;
}

/**
 * Read a `(2*SNAP_RADIUS_MAX+1)`-sided square around `(centerX,
 * centerY)`, clamped to the texture bounds. Returns `undefined` if
 * the texture is empty / the device was lost. `bytesPerRow` is
 * aligned up to 256 (WebGPU requirement); the row stride in the
 * staging buffer can therefore be larger than `sizeX * 16`, and we
 * deinterleave back to a tight per-pixel layout on read.
 */
export async function readPickRegion(
  device: GPUDevice,
  pickTexture: GPUTexture,
  centerX: number,
  centerY: number,
): Promise<PickRegion | undefined> {
  const w = pickTexture.width;
  const h = pickTexture.height;
  if (w <= 0 || h <= 0) return undefined;

  const r = SNAP_RADIUS_MAX;
  const cx = centerX | 0;
  const cy = centerY | 0;
  const x0 = Math.max(0, cx - r);
  const y0 = Math.max(0, cy - r);
  const x1 = Math.min(w, cx + r + 1);
  const y1 = Math.min(h, cy + r + 1);
  const sizeX = x1 - x0;
  const sizeY = y1 - y0;
  if (sizeX <= 0 || sizeY <= 0) return undefined;

  // rgba32f = 16 B/px; bytesPerRow must be a multiple of 256.
  const tightRow = sizeX * 16;
  const bytesPerRow = Math.ceil(tightRow / 256) * 256;
  const totalBytes = bytesPerRow * sizeY;

  let buffer: GPUBuffer;
  try {
    buffer = device.createBuffer({
      size: totalBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: "pick.region.readback",
    });
  } catch {
    return undefined;
  }

  try {
    const encoder = device.createCommandEncoder({ label: "pick.region.readback.encoder" });
    encoder.copyTextureToBuffer(
      { texture: pickTexture, origin: { x: x0, y: y0, z: 0 } },
      { buffer, bytesPerRow, rowsPerImage: sizeY },
      { width: sizeX, height: sizeY, depthOrArrayLayers: 1 },
    );
    device.queue.submit([encoder.finish()]);

    await buffer.mapAsync(GPUMapMode.READ);
    // Deinterleave row-padded staging buffer → tight Float32Array.
    const staging = new Float32Array(buffer.getMappedRange(0, totalBytes).slice(0));
    const floatsPerStagingRow = bytesPerRow / 4;
    const floatsPerTightRow = sizeX * 4;
    const data = new Float32Array(sizeX * sizeY * 4);
    for (let y = 0; y < sizeY; y++) {
      const src = y * floatsPerStagingRow;
      const dst = y * floatsPerTightRow;
      data.set(staging.subarray(src, src + floatsPerTightRow), dst);
    }
    buffer.unmap();
    return { data, originX: x0, originY: y0, sizeX, sizeY };
  } catch {
    return undefined;
  } finally {
    try { buffer.destroy(); } catch { /* already gone */ }
  }
}

/**
 * Read 4 slot floats at region-local pixel `(lx, ly)`. Out-of-region
 * is treated as zero (the cleared "no hit" value), which lets the
 * dispatcher walk offsets without bounds-checking after the initial
 * region fetch.
 */
export function readSlotsAt(region: PickRegion, lx: number, ly: number): PickPixel {
  if (lx < 0 || ly < 0 || lx >= region.sizeX || ly >= region.sizeY) {
    return { slot0: 0, slot1: 0, slot2: 0, slot3: 0 };
  }
  const i = (ly * region.sizeX + lx) * 4;
  return {
    slot0: region.data[i]!,
    slot1: region.data[i + 1]!,
    slot2: region.data[i + 2]!,
    slot3: region.data[i + 3]!,
  };
}

/**
 * Decode the four float channels of a rgba32f pick pixel into a
 * `DecodedPick`. See `pickShaders.ts` for the producer-side slot
 * layout. The pickId was written as `f32(int)` (or `-f32(int)`),
 * always under the 24-bit f32 mantissa, so `Math.abs | 0` is exact.
 */
export function decodePick(pixel: PickPixel): DecodedPick {
  const { slot0 } = pixel;
  // pickId 0 ⇒ no hit (clear value). Either sign is fine to inspect.
  if (slot0 === 0) {
    return { pickId: 0, modeB: false, raw: pixel };
  }
  const modeB = slot0 < 0;
  const pickId = Math.abs(slot0) | 0;
  if (modeB) {
    return {
      pickId,
      modeB: true,
      viewPos: { x: pixel.slot1, y: pixel.slot2, z: pixel.slot3 },
      raw: pixel,
    };
  }
  // Mode-A: slot1 = encoded normal, slot2 = NDC depth in [-1,1],
  // slot3 = part index (or 0). We don't compute viewPos here — the
  // dispatcher does it once it has the hit scope's view / proj.
  return { pickId, modeB: false, raw: pixel };
}

// Re-export region constants for callers that want to size buffers /
// offsets identically.
export { SNAP_RADIUS_MAX, SNAP_REGION_SIZE };
