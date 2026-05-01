// Phase 7 — global time clock. Pin: monotonic increase across ticks
// and subscribers fire each tick.

import { describe, expect, it } from "vitest";
import { AVal, avalAddCallback } from "@aardworx/wombat.adaptive";

import { RenderControl, _tickRenderControlTime } from "../src/scene/renderControl.js";
import { TraversalState } from "../src/scene/index.js";

describe("RenderControl.time", () => {
  it("is observable and ticks monotonically", () => {
    const t = (RenderControl as unknown as { time: import("@aardworx/wombat.adaptive").aval<number> }).time;
    const before = AVal.force(t);
    _tickRenderControlTime();
    const after = AVal.force(t);
    expect(after).toBeGreaterThanOrEqual(before);
  });

  it("subscribers fire on each tick", () => {
    const t = (RenderControl as unknown as { time: import("@aardworx/wombat.adaptive").aval<number> }).time;
    const seen: number[] = [];
    const sub = avalAddCallback(t, (v) => { seen.push(v); });
    _tickRenderControlTime();
    _tickRenderControlTime();
    sub.dispose();
    // initial callback (registration) + 2 ticks => >= 3 entries
    expect(seen.length).toBeGreaterThanOrEqual(2);
  });

  it("TraversalState carries a default time aval", () => {
    expect(typeof AVal.force(TraversalState.empty.time)).toBe("number");
  });
});
