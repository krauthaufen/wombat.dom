// Readback from the rgba32f pick attachment + slot decoder.
//
// Three readers:
//   - `readPickPixel` — single pixel (kept for tests / simple uses).
//   - `readPickRegion` — 33×33 disc-readback. One-shot allocation
//     of a staging buffer; for one-off calls (tests, programmatic
//     ray queries) that's fine.
//   - `PickRegionReader` — pooled + coalescing variant used by
//     `dispatcher.ts` under sustained pointer-move. Owns a small
//     ring of staging buffers (always sized for the full disc) so
//     pointermoves don't allocate / destroy a GPU buffer per event,
//     and discards intermediate reads while one is in flight so the
//     dispatcher only resolves the LATEST cursor position.

import { V3d } from "@aardworx/wombat.base";

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
  readonly viewPos?: V3d;
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
 * Pooled + coalescing reader for the `(2·SNAP_RADIUS_MAX+1)²` pick
 * disc. Reuses a small ring of staging buffers across reads (avoids
 * a per-pointer-event `createBuffer` + `destroy`) and coalesces:
 * while a readback is in flight, additional `read(x, y)` calls
 * overwrite the pending target so the dispatcher only resolves the
 * latest cursor position. Drops intermediate samples — a sustained
 * pointer-move under load runs at most one readback in flight + one
 * pending, no matter the event rate.
 *
 * The staging buffers are always sized for the *full* disc
 * (`SNAP_REGION_SIZE²`), which is the worst case; this lets all
 * reads reuse the same buffer even as the cursor crosses the
 * texture-edge clamp region (the deinterleave step honours the
 * actual region the GPU produced).
 *
 * Lifetime is tied to the picker; call `dispose()` on unmount.
 */
export class PickRegionReader {
  private readonly device: GPUDevice;
  private readonly textureSrc: () => GPUTexture | undefined;
  private readonly pool: { buffer: GPUBuffer; size: number; inFlight: boolean }[] = [];
  private static readonly POOL_LIMIT = 3;
  // Sized for the full SNAP_REGION_SIZE × SNAP_REGION_SIZE disc with
  // bytesPerRow padded to 256. rgba32f = 16 B/px, so per-row tight
  // bytes are SNAP_REGION_SIZE * 16. Pad up.
  private static readonly BYTES_PER_ROW =
    Math.ceil(SNAP_REGION_SIZE * 16 / 256) * 256;
  private static readonly STAGING_BYTES =
    PickRegionReader.BYTES_PER_ROW * SNAP_REGION_SIZE;
  // Coalescer: if a request arrives while one is in flight, it
  // replaces `pending`. After the in-flight one resolves, we kick
  // off the latest pending coords (and discard everything that
  // arrived in between).
  private current: { resolveLatest: (r: PickRegion | undefined) => void } | undefined;
  private pending: { x: number; y: number; resolve: (r: PickRegion | undefined) => void } | undefined;
  private disposed = false;

  constructor(device: GPUDevice, textureSrc: () => GPUTexture | undefined) {
    this.device = device;
    this.textureSrc = textureSrc;
  }

  /**
   * Same shape as `readPickRegion`. Returns a Promise that resolves
   * with the region read at coords (x, y) — UNLESS a newer call
   * supersedes it before the GPU work finishes, in which case the
   * older promise resolves to `undefined` (caller filters via its
   * own seq counter the same way it always did).
   */
  read(x: number, y: number): Promise<PickRegion | undefined> {
    if (this.disposed) return Promise.resolve(undefined);
    if (this.current !== undefined) {
      // Already a readback in flight. Replace any prior pending
      // request — its caller gets `undefined` (their result was
      // superseded by the newer cursor position before the GPU
      // work for them ever started).
      if (this.pending !== undefined) this.pending.resolve(undefined);
      return new Promise<PickRegion | undefined>((resolve) => {
        this.pending = { x, y, resolve };
      });
    }
    return this.start(x, y);
  }

  private async start(x: number, y: number): Promise<PickRegion | undefined> {
    let resolveOuter!: (r: PickRegion | undefined) => void;
    const outer = new Promise<PickRegion | undefined>((res) => { resolveOuter = res; });
    this.current = { resolveLatest: resolveOuter };
    void this.runOne(x, y, resolveOuter);
    return outer;
  }

  private async runOne(
    x: number, y: number,
    resolveOuter: (r: PickRegion | undefined) => void,
  ): Promise<void> {
    const region = await this.execute(x, y);
    resolveOuter(region);
    this.current = undefined;
    // Drain the pending slot if anything queued.
    const next = this.pending;
    this.pending = undefined;
    if (next !== undefined && !this.disposed) {
      // Reuse the same code path: chain start() and hand the
      // pending caller's resolve into it.
      void this.start(next.x, next.y).then(next.resolve, () => next.resolve(undefined));
    }
  }

  private acquire(): { buffer: GPUBuffer; release: () => void } | undefined {
    // Reuse an idle buffer first.
    for (const e of this.pool) {
      if (!e.inFlight) {
        e.inFlight = true;
        return { buffer: e.buffer, release: () => { e.inFlight = false; } };
      }
    }
    // No idle entry. Grow the pool up to POOL_LIMIT.
    if (this.pool.length < PickRegionReader.POOL_LIMIT) {
      let buffer: GPUBuffer;
      try {
        buffer = this.device.createBuffer({
          size: PickRegionReader.STAGING_BYTES,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          label: `pick.region.readback.pool#${this.pool.length}`,
        });
      } catch {
        return undefined;
      }
      const entry = { buffer, size: PickRegionReader.STAGING_BYTES, inFlight: true };
      this.pool.push(entry);
      return { buffer, release: () => { entry.inFlight = false; } };
    }
    // Pool exhausted (3 in flight). With coalescing this shouldn't
    // normally happen — only one in-flight at a time per reader.
    // Fall back to one-shot allocation.
    let buffer: GPUBuffer;
    try {
      buffer = this.device.createBuffer({
        size: PickRegionReader.STAGING_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: "pick.region.readback.spill",
      });
    } catch {
      return undefined;
    }
    return { buffer, release: () => { try { buffer.destroy(); } catch { /* gone */ } } };
  }

  private async execute(centerX: number, centerY: number): Promise<PickRegion | undefined> {
    const tex = this.textureSrc();
    if (tex === undefined) return undefined;
    const w = tex.width;
    const h = tex.height;
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
    const bytesPerRow = PickRegionReader.BYTES_PER_ROW;
    const totalBytes = bytesPerRow * sizeY;
    const lease = this.acquire();
    if (lease === undefined) return undefined;
    const { buffer, release } = lease;
    try {
      const encoder = this.device.createCommandEncoder({ label: "pick.region.readback.encoder" });
      encoder.copyTextureToBuffer(
        { texture: tex, origin: { x: x0, y: y0, z: 0 } },
        { buffer, bytesPerRow, rowsPerImage: sizeY },
        { width: sizeX, height: sizeY, depthOrArrayLayers: 1 },
      );
      this.device.queue.submit([encoder.finish()]);
      await buffer.mapAsync(GPUMapMode.READ, 0, totalBytes);
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
      release();
    }
  }

  dispose(): void {
    this.disposed = true;
    if (this.pending !== undefined) {
      this.pending.resolve(undefined);
      this.pending = undefined;
    }
    for (const e of this.pool) {
      try { e.buffer.destroy(); } catch { /* gone */ }
    }
    this.pool.length = 0;
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
      viewPos: new V3d(pixel.slot1, pixel.slot2, pixel.slot3),
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
