// FreeFlyController — gamepad polling integration.
//
// We mock navigator.getGamepads; on each time-aval tick, the controller
// polls the API and translates axes/buttons into MoveVec/TurnVec/etc.

import { describe, expect, it } from "vitest";
import { cval, transact } from "@aardworx/wombat.adaptive";
import { V3d } from "@aardworx/wombat.base";

import { FreeFlyController } from "../src/scene/index.js";

function makeTarget(): HTMLElement {
  const el = document.createElement("div");
  (el as unknown as { setPointerCapture?: () => void }).setPointerCapture = () => {};
  (el as unknown as { releasePointerCapture?: () => void }).releasePointerCapture = () => {};
  return el;
}

interface FakeGamepad {
  index: number;
  axes: number[];
  buttons: { pressed: boolean; value: number }[];
}

function gp(index: number, axes: number[], pressed: number[]): FakeGamepad {
  const buttons = Array.from({ length: 17 }, (_, i) => ({
    pressed: pressed.includes(i), value: pressed.includes(i) ? 1 : 0,
  }));
  return { index, axes, buttons };
}

describe("FreeFlyController — gamepad", () => {
  it("LeftStick X→MoveVec X; RightStick X→TurnVec X; tick advances state", () => {
    let pad: FakeGamepad | null = gp(0, [0.8, 0, 0.6, 0], []);
    (navigator as { getGamepads?: () => (FakeGamepad | null)[] }).getGamepads =
      () => [pad];

    const ctl = FreeFlyController.create();
    const target = makeTarget();
    const time = cval(0);
    const detach = ctl.attach(target, time);

    // Tick the time aval — triggers pollGamepads().
    transact(() => { time.value = 16; });

    // LeftStick deflection 0.8 along X → applyExp passes a positive value.
    let total = V3d.zero;
    for (const v of ctl.state.value.MoveVectors.values()) total = total.add(v);
    expect(total.x).toBeGreaterThan(0);

    // RightStick X = 0.6 → TurnVec X negative (sign-flipped per F#).
    let totalT = 0;
    for (const v of ctl.state.value.TurnVectors.values()) totalT += v.x;
    expect(totalT).toBeLessThan(0);

    // Press DPad-Up (idx 12) → AddMoveVec(0,0,1)
    pad = gp(0, [0.8, 0, 0.6, 0], [12]);
    transact(() => { time.value = 32; });
    let zSum = 0;
    for (const v of ctl.state.value.MoveVectors.values()) zSum += v.z;
    expect(zSum).toBeGreaterThan(0);

    detach();
  });

  it("RB (button 5) rising edge calls AdjustMoveSpeed(1.5)", () => {
    let pad: FakeGamepad | null = gp(0, [0, 0, 0, 0], []);
    (navigator as { getGamepads?: () => (FakeGamepad | null)[] }).getGamepads =
      () => [pad];

    const ctl = FreeFlyController.create();
    const target = makeTarget();
    const time = cval(0);
    const detach = ctl.attach(target, time);
    const baseSpeed = ctl.state.value.Config.MoveSpeed;

    transact(() => { time.value = 16; });          // first poll, no edge
    pad = gp(0, [0, 0, 0, 0], [5]);                 // RB pressed
    transact(() => { time.value = 32; });          // edge → adjust *1.5
    expect(ctl.state.value.Config.MoveSpeed).toBeCloseTo(baseSpeed * 1.5, 6);

    detach();
  });
});
