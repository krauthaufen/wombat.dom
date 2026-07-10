// Effects for the warp-portal demo.
//
// - `flatSurface`: unlit color passthrough for the INNER scene (flat
//   colors make the click-vs-pixel-color validation exact).
// - `portalSurface`: the OUTER composite. Fullscreen quad; the
//   fragment warps the quad uv with an animated ripple, samples the
//   offscreen portal texture, and — the portal-picking contract —
//   writes the sampled source-uv as `PickContextCoord` so picks
//   follow the warp. `PickContextCoord` uses BOTTOM-LEFT-origin uv
//   (texture-coordinate convention, F# parity); the resolver Y-flips
//   to inner pixels.

import { effect, fragment, vertex } from "@aardworx/wombat.shader";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { sin, texture, type Sampler2D } from "@aardworx/wombat.shader/types";
import { V2f, V3f, V4f } from "@aardworx/wombat.base";

declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope {
    /** Warp phase (seconds). */
    readonly Time: number;
    /** Warp amplitude in uv units (0 = undistorted). */
    readonly WarpAmp: number;
  }
}

// Portal color texture — bound by name from the Sg Uniform scope.
const PortalTex: Sampler2D = null as unknown as Sampler2D;

// ─── inner: unlit flat color ────────────────────────────────────────────

const flatVS = vertex((v: { Positions: V4f; Colors: V4f }) => ({
  gl_Position: uniform.ViewProjTrafo.mul(uniform.ModelTrafo.mul(v.Positions)),
  Colors: v.Colors,
}));

const flatFS = fragment((v: { Colors: V4f }) => ({ Colors: v.Colors }));

export const flatSurface = effect(flatVS, flatFS);

// ─── outer: fullscreen warped composite ─────────────────────────────────

// The quad's positions span [-1,1]² — emit them as clip coords
// directly (no camera) and carry the raw xy for uv derivation.
const portalVS = vertex((v: { Positions: V4f }) => ({
  gl_Position: new V4f(v.Positions.x, v.Positions.y, 0.5, 1.0),
  RawPos: new V2f(v.Positions.x, v.Positions.y),
}));

const portalFS = fragment((v: { RawPos: V2f }) => {
  // uv0: bottom-left-origin quad uv (y = 1 at the top of the screen).
  const uv0 = new V2f(v.RawPos.x * 0.5 + 0.5, v.RawPos.y * 0.5 + 0.5);
  const t = uniform.Time;
  const amp = uniform.WarpAmp;
  const wx = uv0.x + amp * sin(uv0.y * 17.0 + t);
  const wy = uv0.y + amp * sin(uv0.x * 13.0 + t * 1.3);
  // Sampling coords are top-origin (WebGPU): flip y.
  const uvS = new V2f(wx, 1.0 - wy);
  const c = texture(PortalTex, uvS);
  // Contract: PickContextCoord is bottom-left-origin — i.e. the
  // warped uv itself.
  return { Colors: c, PickContextCoord: new V2f(wx, wy) };
});

export const portalSurface = effect(portalVS, portalFS);
