// Pre-computed pixel-snap disc offsets, sorted ascending by d².
//
// Mirrors Aardvark.Dom's `PickSnap.radius = 16` + offsets array
// (SceneHandler.fs lines 999–1016): the dispatcher walks these in
// order so the closest valid hit always wins.
//
// Why precompute: the array has on the order of ~800 entries — way
// cheaper to build once at module init than per pointer event.

/** Hard cap on per-scope pixel-snap radius. Matches F# `PickSnap.radius`. */
export const SNAP_RADIUS_MAX = 16;

/** Side length of the readback square: 2 * SNAP_RADIUS_MAX + 1. */
export const SNAP_REGION_SIZE = SNAP_RADIUS_MAX * 2 + 1;

export interface SnapOffset {
  readonly dx: number;
  readonly dy: number;
  /** dx*dx + dy*dy. Precomputed so the dispatcher avoids redoing it. */
  readonly d2: number;
}

function buildOffsets(): SnapOffset[] {
  const r = SNAP_RADIUS_MAX;
  const r2 = r * r;
  const out: SnapOffset[] = [];
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const d2 = dx * dx + dy * dy;
      if (d2 <= r2) out.push({ dx, dy, d2 });
    }
  }
  // Stable sort by d² ascending. Center (0,0) ends up first.
  out.sort((a, b) => a.d2 - b.d2);
  return out;
}

/**
 * Disc offsets within `|o| <= SNAP_RADIUS_MAX`, sorted by d²
 * ascending. First entry is `(0, 0, 0)` (center). The dispatcher
 * stops at the first offset it can validate as a hit per the
 * scope's own `pixelSnapRadius`.
 */
export const SNAP_OFFSETS: ReadonlyArray<SnapOffset> = buildOffsets();
