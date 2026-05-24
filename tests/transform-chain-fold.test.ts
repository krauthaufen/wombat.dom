// GPU transform propagation — constant-run folding in the Model chain.
// Consecutive constant <Sg Trafo> scopes pre-multiply into one constant chain
// link (one constituent slot + one GPU multiply); a dynamic scope breaks the
// run. The folded chain must still equal the eager composeModel product.

import { describe, expect, it } from "vitest";
import { cval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, type M44d } from "@aardworx/wombat.base";
import { TraversalState } from "../src/scene/index.js";

const tr = (x: number, y = 0, z = 0): Trafo3d => Trafo3d.translation(new V3d(x, y, z));
const rot = (rad: number): Trafo3d => Trafo3d.rotation(new V3d(0, 0, 1), rad);

function expectMatEq(a: M44d, b: M44d): void {
  const x = a.toArray(), y = b.toArray();
  for (let i = 0; i < 16; i++) expect(x[i]!).toBeCloseTo(y[i]!, 10);
}

describe("modelChain constant-run folding", () => {
  it("folds N consecutive constant scopes into one chain link", () => {
    const s = TraversalState.empty
      .pushTrafo(tr(1, 0, 0))   // root  (const)
      .pushTrafo(rot(0.5))      //       (const)
      .pushTrafo(tr(0, 2, 0));  // leaf  (const)
    expect(s.modelChain.length).toBe(1);
    // The single folded link equals the eager composite (leaf·…·root).
    expectMatEq(s.modelChain[0]!.force().forward, s.model.force().forward);
  });

  it("a dynamic scope breaks the run (constants either side stay separate)", () => {
    const dyn = cval(tr(9, 9, 9));
    const s = TraversalState.empty
      .pushTrafo(tr(1, 0, 0))   // const root
      .pushTrafo(dyn)           // dynamic — not foldable
      .pushTrafo(tr(0, 2, 0));  // const leaf
    expect(s.modelChain.length).toBe(3);
    // Still equals the eager composite.
    expectMatEq(
      s.modelChain[0]!.force().forward
        .mul(s.modelChain[1]!.force().forward)
        .mul(s.modelChain[2]!.force().forward),
      s.model.force().forward,
    );
  });

  it("folds an all-constant Trafo array scope to one link, then into the run", () => {
    const s = TraversalState.empty
      .pushTrafo(tr(1, 0, 0))            // const root
      .pushTrafo([rot(0.3), tr(0, 1, 0)]); // all-const array → one const link → folds in
    expect(s.modelChain.length).toBe(1);
    expectMatEq(s.modelChain[0]!.force().forward, s.model.force().forward);
  });

  it("a constant scope after a dynamic does not fold backwards", () => {
    const dyn = cval(tr(0, 0, 1));
    const s = TraversalState.empty
      .pushTrafo(dyn)           // dynamic root
      .pushTrafo(tr(2, 0, 0))   // const
      .pushTrafo(tr(0, 0, 3));  // const — folds with the one above it, not the dyn
    expect(s.modelChain.length).toBe(2); // [const·const, dyn]
  });
});
