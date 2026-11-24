// A constant aval is an aval to the bindings: an attribute bound to one shows its value. It used to be taken for a
// plain value and written as its text ("constval(…)"), which also left a boolean attribute such as `disabled` set.
import { describe, expect, it } from "vitest";
import { AVal, cval } from "@aardworx/wombat.adaptive";
import { isAVal } from "../src/guards.js";
import { bindAValSink } from "../src/bindings/attr.js";

describe("constant avals", () => {
  it("are avals, and plain values are not", () => {
    expect(isAVal(AVal.constant(1))).toBe(true);
    expect(isAVal(cval(1))).toBe(true);
    expect(isAVal(cval(1).map((v) => v + 1))).toBe(true);
    for (const plain of [1, "a", null, undefined, {}, [], { isConstant: true }]) expect(isAVal(plain)).toBe(false);
  });

  it("are followed as any aval: the sink gets the value once", () => {
    const seen: number[] = [];
    const scope = { add(_d: unknown): void {} };
    bindAValSink(scope as never, AVal.constant(7) as never, (value: number) => { seen.push(value); });
    expect(seen).toEqual([7]);
  });
});
