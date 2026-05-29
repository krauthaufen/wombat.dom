// Shared test helpers for the GPU-argmin pick path.
//
// The dispatcher's injected seam is now `ResolvePixel` — a function
// returning the GPU argmin kernel's single winning pixel
// (`PickArgminResult`) instead of a 33×33 `PickRegion`. These builders
// stand in for the kernel's verdict so dispatcher tests can assert
// dispatch behaviour without a GPU:
//   - `noPixel()`        — the kernel found no valid pixel (cleared,
//                          out of snap radius, inactive, MSAA-rejected,
//                          or a single unsupported pixel).
//   - `pixelWinner(id)`  — the kernel's nearest VALID pixel is `id`.
//
// The per-pixel validity logic (snap radius / mode-sign / 3×3 MSAA
// neighbour count / active-noEvents folding) now lives IN the kernel
// and is unit-tested by `argminPickReference` (pick-argmin.test.ts) —
// the dispatcher only consumes the verdict.

import type { ResolvePixel } from "../src/scene/picking/dispatcher.js";
import type { PickArgminResult } from "../src/scene/picking/pickArgminCompute.js";

export function noPixel(): PickArgminResult {
  return { found: false, px: 0, py: 0, dist2: 0, slot0: 0, slot1: 0, slot2: 0, slot3: 0, centerSlot0: 0 };
}

export interface PixelOpts {
  /** Winner device-pixel x (default 50). */
  px?: number;
  /** Winner device-pixel y (default 50). */
  py?: number;
  /** Screen-dist² of the winner to the cursor (0 ⇒ exactly under the cursor). */
  dist2?: number;
  /** Mode A (+id, default) or B (−id). */
  mode?: "A" | "B";
  slot1?: number;
  slot2?: number;
  slot3?: number;
  /** Raw centre-pixel slot0 (hover); defaults to ±id when the winner is centred. */
  center?: number;
}

/** The kernel's verdict: `pickId` is the nearest valid pixel. */
export function pixelWinner(pickId: number, opts: PixelOpts = {}): PickArgminResult {
  const sign = (opts.mode ?? "A") === "A" ? 1 : -1;
  const dist2 = opts.dist2 ?? 0;
  const px = opts.px ?? 50;
  const py = opts.py ?? 50;
  return {
    found: true,
    px,
    py,
    dist2,
    slot0: sign * pickId,
    slot1: opts.slot1 ?? 0,
    slot2: opts.slot2 ?? 0,
    slot3: opts.slot3 ?? 0,
    centerSlot0: opts.center ?? (dist2 === 0 ? sign * pickId : 0),
  };
}

/** Wrap a fixed result as a `ResolvePixel` (ignores coords). */
export function resolverOf(r: PickArgminResult): ResolvePixel {
  return async () => r;
}
