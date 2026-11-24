// isAVal — a constant aval is an aval: an attribute bound to one shows its value, not its text.
import { describe, expect, it } from "vitest";
import { AVal, cval } from "@aardworx/wombat.adaptive";
import { isAVal } from "../src/guards.js";

describe("isAVal", () => {
  it("takes constants, cells and mapped values, and no plain value", () => {
    expect(isAVal(AVal.constant(1))).toBe(true);
    expect(isAVal(cval(1))).toBe(true);
    expect(isAVal(AVal.constant(1).map((v) => v + 1))).toBe(true);
    for (const plain of [1, "a", null, undefined, {}, [], () => 1, { getValue: () => 1 }]) expect(isAVal(plain)).toBe(false);
  });
});
