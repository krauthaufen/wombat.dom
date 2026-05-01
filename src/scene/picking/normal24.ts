// Normal24 — 12+12 octahedron normal packed into a 24-bit integer
// stored as float32 (24-bit f32 mantissa round-trips exactly).
//
// Lower angular precision than the older Normal32 bit-cast
// (12 bits/axis ~ 0.05 deg vs 16 bits/axis ~ 0.003 deg), but
// bullet-proof under MSAA resolve-average: the encoded value stays
// representable in float32 and survives interpolation as a plain
// number rather than a re-bitcast.
//
// Shipped as shader-source SNIPPETS so we can inline them into each
// pick-fragment variant — composing them as separate Effects
// fragments is overkill, and inlining keeps the dep walker simple.
//
// Identifier conventions used by the snippets:
//   `n24Encode(v: V3f) : f32`  — assumes `v` is already normalised.
//   `n24Decode(e: f32) : V3f`  — returns the unit-length vector for
//                                a previously encoded value.
//
// The CPU-side mirror lives in `n24EncodeF32` / `n24DecodeF32`
// below — same algorithm, used by tests and for any host-side
// round-tripping (e.g. screen-space hover overlays).

const N24_SCALE = 4095; // 2^12 - 1

// ---------------------------------------------------------------------------
// CPU mirror (matches Aardvark.Dom's `Normal24.encode/decode` exactly)
// ---------------------------------------------------------------------------

function sgn2(x: number, y: number): [number, number] {
  return [x >= 0 ? 1 : -1, y >= 0 ? 1 : -1];
}

function clamp1(v: number): number {
  return v < -1 ? -1 : v > 1 ? 1 : v;
}

export function n24DecodeI32(v: number): [number, number, number] {
  if (v === 0) return [0, 0, 0];
  const ex = ((v >>> 12) & 0xFFF) / N24_SCALE * 2 - 1;
  const ey = (v & 0xFFF) / N24_SCALE * 2 - 1;
  let x = ex, y = ey, z = 1 - Math.abs(ex) - Math.abs(ey);
  if (z < 0) {
    const [sx, sy] = sgn2(x, y);
    const nx = (1 - Math.abs(y)) * sx;
    const ny = (1 - Math.abs(x)) * sy;
    x = nx; y = ny;
  }
  const len = Math.sqrt(x * x + y * y + z * z);
  return [x / len, y / len, z / len];
}

export function n24EncodeI32(x: number, y: number, z: number): number {
  if (x === 0 && y === 0 && z === 0) return 0;
  const inv = 1 / (Math.abs(x) + Math.abs(y) + Math.abs(z));
  let px = x * inv, py = y * inv;
  if (z <= 0) {
    const [sx, sy] = sgn2(px, py);
    const nx = (1 - Math.abs(py)) * sx;
    const ny = (1 - Math.abs(px)) * sy;
    px = clamp1(nx); py = clamp1(ny);
  } else {
    px = clamp1(px); py = clamp1(py);
  }
  const x0 = Math.floor((px * 0.5 + 0.5) * N24_SCALE) | 0;
  const y0 = Math.floor((py * 0.5 + 0.5) * N24_SCALE) | 0;
  let bestDot = 0, best = 0;
  for (let dx = 0; dx <= 1; dx++) {
    for (let dy = 0; dy <= 1; dy++) {
      const e = (((x0 + dx) << 12) | (y0 + dy)) >>> 0;
      const [vx, vy, vz] = n24DecodeI32(e);
      const d = vx * x + vy * y + vz * z;
      if (d > bestDot) { bestDot = d; best = e; }
    }
  }
  return best;
}

export function n24EncodeF32(x: number, y: number, z: number): number {
  return n24EncodeI32(x, y, z) >>> 0;
}

export function n24DecodeF32(e: number): [number, number, number] {
  return n24DecodeI32(e | 0);
}

// ---------------------------------------------------------------------------
// Shader source snippets
// ---------------------------------------------------------------------------

/**
 * Helper-function source for n24Encode / n24Decode. Concatenate
 * before the entry function in `parseShader({ source })`.
 *
 * Exposes:
 *   `function n24Decode(e: f32) : V3f`
 *   `function n24Encode(v: V3f)  : f32`
 *
 * Encoding picks the closest of the four neighbouring 24-bit cells
 * by max-dot, matching the CPU `n24EncodeI32` above bit-for-bit.
 */
export function n24ShaderHelpers(): string {
  return `
    function n24Decode(e: f32): V3f {
      const i = e as i32;
      const xi = (i >> 12) & 4095;
      const yi = i & 4095;
      const ex = (xi as f32) / 4095.0 * 2.0 - 1.0;
      const ey = (yi as f32) / 4095.0 * 2.0 - 1.0;
      const z0 = 1.0 - abs(ex) - abs(ey);
      const sx = ex >= 0.0 ? 1.0 : -1.0;
      const sy = ey >= 0.0 ? 1.0 : -1.0;
      const fx = z0 < 0.0 ? (1.0 - abs(ey)) * sx : ex;
      const fy = z0 < 0.0 ? (1.0 - abs(ex)) * sy : ey;
      const v = new V3f(fx, fy, z0);
      return v.normalize();
    }

    function n24Encode(v: V3f): f32 {
      const inv = 1.0 / (abs(v.x) + abs(v.y) + abs(v.z));
      const px0 = v.x * inv;
      const py0 = v.y * inv;
      const sx = px0 >= 0.0 ? 1.0 : -1.0;
      const sy = py0 >= 0.0 ? 1.0 : -1.0;
      const fx = v.z <= 0.0 ? (1.0 - abs(py0)) * sx : px0;
      const fy = v.z <= 0.0 ? (1.0 - abs(px0)) * sy : py0;
      const cx = clamp(fx, -1.0, 1.0);
      const cy = clamp(fy, -1.0, 1.0);
      const x0 = floor((cx * 0.5 + 0.5) * 4095.0) as i32;
      const y0 = floor((cy * 0.5 + 0.5) * 4095.0) as i32;
      const e = ((x0 << 12) | y0) as f32;
      return e;
    }
  `;
}

/** Bits per octahedron axis. Exposed so callers / docs stay in sync. */
export const N24_BITS_PER_AXIS = 12;
