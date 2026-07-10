// cityview surface — renderbench's lit shader (identical light math so
// numbers stay comparable) + a SelectedPick highlight: the runtime
// binds `PickId` per draw for pick-composed effects, and the scene
// publishes the hovered id as `SelectedPick`; equality lights the part
// up without touching any per-part data.

import { effect, fragment, vertex } from "@aardworx/wombat.shader";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { max, texture, type Sampler2D } from "@aardworx/wombat.shader/types";
import type { u32 } from "@aardworx/wombat.shader/types";
import { V2f, V3f, V4f } from "@aardworx/wombat.base";
void (null as unknown as u32);

declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope {
    /** Bound per-draw by the picking runtime (drawHeader on the heap path). */
    readonly PickId: u32;
    /** Hovered pick id, scene-level cval. 0 = none. */
    readonly SelectedPick: number;
  }
}

// World-space geometry (no per-part model trafo) → one VP transform.
// The normal is carried in VIEW space for the headlight shading (and
// as a bonus the pick chain sees a user-provided `ViewSpaceNormal` —
// no injection needed).
// NOTE the inter-stage color rides as `VtxColor` — the pick-chain
// finals type the `Colors` port as V4f (the FS output), so a V3f
// varying under the same name would clash in the fused effect.
const cityVS = vertex((v: { Positions: V4f; Normals: V3f; Colors: V3f }) => {
  const t = uniform.ModelViewTrafoInv.transpose();
  const n4 = t.mul(new V4f(v.Normals.x, v.Normals.y, v.Normals.z, 0.0));
  return {
    gl_Position: uniform.ViewProjTrafo.mul(v.Positions),
    ViewSpaceNormal: new V3f(n4.x, n4.y, n4.z),
    VtxColor: v.Colors,
  };
});

// Headlight: the CAMERA is the light source — in view space the light
// direction is +Z, so d = 0.25 + 0.75·max(0, n̂.z).
const cityFS = fragment((v: { ViewSpaceNormal: V3f; VtxColor: V3f }) => {
  const n = v.ViewSpaceNormal.normalize();
  const d = 0.25 + 0.75 * max(0.0, n.z);
  const hot = (uniform.PickId as number) === uniform.SelectedPick && uniform.SelectedPick > 0.0 ? 1.0 : 0.0;
  const r = v.VtxColor.x * d * (1.0 - hot) + (v.VtxColor.x * 0.4 + 0.6) * hot;
  const g = v.VtxColor.y * d * (1.0 - hot) + (v.VtxColor.y * 0.4 + 0.45) * hot;
  const b = v.VtxColor.z * d * (1.0 - hot) + (v.VtxColor.z * 0.4 + 0.15) * hot;
  return { Colors: new V4f(r, g, b, 1.0) };
});

export const citySurface = effect(cityVS, cityFS);

// ─── AO composite (portal quad) ─────────────────────────────────────────
// Fullscreen quad sampling the offscreen city color × the AO texture.
// Writes `PickContextCoord` (bottom-left-origin uv) so ALL picking
// recurses into the offscreen city — hover/tap/fly-to ride the portal.

const SceneTex: Sampler2D = null as unknown as Sampler2D;
const AoTex: Sampler2D = null as unknown as Sampler2D;

const compositeVS = vertex((v: { Positions: V4f }) => ({
  gl_Position: new V4f(v.Positions.x, v.Positions.y, 0.5, 1.0),
  RawPos: new V2f(v.Positions.x, v.Positions.y),
}));

const compositeFS = fragment((v: { RawPos: V2f }) => {
  const uv0 = new V2f(v.RawPos.x * 0.5 + 0.5, v.RawPos.y * 0.5 + 0.5);
  const uvS = new V2f(uv0.x, 1.0 - uv0.y);
  const c = texture(SceneTex, uvS);
  const ao = texture(AoTex, uvS);
  return {
    Colors: new V4f(c.x * ao.x, c.y * ao.x, c.z * ao.x, 1.0),
    PickContextCoord: uv0,
  };
});

export const compositeSurface = effect(compositeVS, compositeFS);
