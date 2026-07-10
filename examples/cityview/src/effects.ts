// cityview surface — renderbench's lit shader (identical light math so
// numbers stay comparable) + a SelectedPick highlight: the runtime
// binds `PickId` per draw for pick-composed effects, and the scene
// publishes the hovered id as `SelectedPick`; equality lights the part
// up without touching any per-part data.

import { effect, fragment, vertex } from "@aardworx/wombat.shader";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { max } from "@aardworx/wombat.shader/types";
import type { u32 } from "@aardworx/wombat.shader/types";
import { V3f, V4f } from "@aardworx/wombat.base";
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
// NOTE the inter-stage color rides as `VtxColor` — the pick-chain
// finals type the `Colors` port as V4f (the FS output), so a V3f
// varying under the same name would clash in the fused effect.
const cityVS = vertex((v: { Positions: V4f; Normals: V3f; Colors: V3f }) => ({
  gl_Position: uniform.ViewProjTrafo.mul(v.Positions),
  Normals: v.Normals,
  VtxColor: v.Colors,
}));

// Renderbench lit FS: fixed light dir (1,2,3), d = 0.25 + 0.75·max(0, n̂·l).
const cityFS = fragment((v: { Normals: V3f; VtxColor: V3f }) => {
  const n = v.Normals.normalize();
  const l = new V3f(0.26726, 0.53452, 0.80178); // normalize(1,2,3)
  const d = 0.25 + 0.75 * max(0.0, n.dot(l));
  const hot = (uniform.PickId as number) === uniform.SelectedPick && uniform.SelectedPick > 0.0 ? 1.0 : 0.0;
  const r = v.VtxColor.x * d * (1.0 - hot) + (v.VtxColor.x * 0.4 + 0.6) * hot;
  const g = v.VtxColor.y * d * (1.0 - hot) + (v.VtxColor.y * 0.4 + 0.45) * hot;
  const b = v.VtxColor.z * d * (1.0 - hot) + (v.VtxColor.z * 0.4 + 0.15) * hot;
  return { Colors: new V4f(r, g, b, 1.0) };
});

export const citySurface = effect(cityVS, cityFS);
