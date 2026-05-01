// Camera controllers — pointer + keyboard-driven view-trafo
// builders. Two flavours, both producing an `aval<Trafo3d>` you can
// hand to `<RenderControl view={…}>` or `<Sg View={…}>`.
//
//   freeFlyController   — WASD + mouse-drag look. Eye-anchored.
//   orbitController     — drag to rotate, wheel to zoom, middle-
//                         click to pan. Target-anchored.
//
// The controllers integrate motion in their own `requestAnimation
// Frame` loop so movement is frame-rate-independent. Each call to
// `attach(canvas)` registers DOM listeners; `dispose()` tears them
// down and stops the rAF loop. State is held in `cval`s so views
// derived from it (e.g. `aval<Trafo3d>`) participate in the
// adaptive graph and trigger re-renders on motion.
//
// Simplified vs Aardvark.Dom's full version: no momentum/damping,
// no virtual touch sticks, no sprint factor. Layer those on top
// later when the use case appears.

import {
  AVal, cval, transact,
  type aval, type ChangeableValue,
} from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d } from "@aardworx/wombat.base";
import { lookAt } from "./camera.js";

// ---------------------------------------------------------------------------
// Shared handle
// ---------------------------------------------------------------------------

export interface CameraControllerHandle {
  /** Reactive view trafo — re-derives whenever the controller's state changes. */
  readonly view: aval<Trafo3d>;
  /** Eye/target/up exposed individually for advanced wiring. */
  readonly eye: aval<V3d>;
  readonly target: aval<V3d>;
  readonly up: aval<V3d>;
  /**
   * Wire pointer + keyboard listeners to a DOM target. Returns a
   * disposer that removes the listeners (the controller stays
   * usable after; you can `attach` again).
   */
  attach(target: HTMLElement): { dispose(): void };
  /**
   * Manually advance the controller by `dt` seconds. The rAF loop
   * calls this internally; tests inject a fixed `dt` to step
   * deterministically.
   */
  tick(dtSeconds: number): void;
  /** Stop the rAF loop and detach all listeners. */
  dispose(): void;
}

// ---------------------------------------------------------------------------
// Free-fly
// ---------------------------------------------------------------------------

export interface FreeFlyOptions {
  readonly eye?: V3d;
  /** Initial forward direction (does NOT need to be normalised). Default `-Z`. */
  readonly forward?: V3d;
  /** World-up. Default `+Y`. */
  readonly up?: V3d;
  /** Movement speed (units/s). Default 5. */
  readonly moveSpeed?: number;
  /** Mouse rotation sensitivity (rad/pixel). Default 0.005. */
  readonly turnSensitivity?: number;
  /**
   * Frame loop. Set to `"manual"` to disable the auto-rAF loop and
   * advance via `tick(dt)` (used in tests). Default `"auto"`.
   */
  readonly loop?: "auto" | "manual";
}

/**
 * WASD + mouse-drag free-fly controller.
 *
 *   W / S        forward / back along view direction
 *   A / D        strafe left / right
 *   Q / E        down / up along world-up
 *   Drag         look around (yaw + pitch)
 */
export function freeFlyController(opts: FreeFlyOptions = {}): CameraControllerHandle {
  const eye0 = opts.eye ?? new V3d(0, 0, 5);
  const fwd0 = (opts.forward ?? new V3d(0, 0, -1)).normalize();
  const up0  = (opts.up ?? new V3d(0, 1, 0)).normalize();

  const eyeC: ChangeableValue<V3d> = cval(eye0);
  const fwdC: ChangeableValue<V3d> = cval(fwd0);
  const upC:  ChangeableValue<V3d> = cval(up0);

  // Held-key set drives a per-frame velocity vector in the view
  // basis. Mouse drag deltas accumulate into yaw/pitch and are
  // applied on the same tick.
  const keys = new Set<string>();
  let pendingYaw = 0;     // around world-up (right-handed: positive turns left)
  let pendingPitch = 0;   // around local-right
  let dragging = false;
  const turnSensitivity = opts.turnSensitivity ?? 0.005;
  const moveSpeed = opts.moveSpeed ?? 5;

  const tick = (dt: number): void => {
    let dx = 0, dy = 0, dz = 0;
    if (keys.has("w")) dz -= 1;
    if (keys.has("s")) dz += 1;
    if (keys.has("a")) dx -= 1;
    if (keys.has("d")) dx += 1;
    if (keys.has("q")) dy -= 1;
    if (keys.has("e")) dy += 1;
    const moveLocal = dx !== 0 || dy !== 0 || dz !== 0;
    const turning = pendingYaw !== 0 || pendingPitch !== 0;
    if (!moveLocal && !turning) return;

    const fwd = AVal.force(fwdC);
    const up  = AVal.force(upC);
    const right = fwd.cross(up).normalize();

    let nextEye = AVal.force(eyeC);
    let nextFwd = fwd;

    if (moveLocal) {
      // dz=-1 means forward. Translate eye along (right, up, -fwd).
      const step = moveSpeed * dt;
      nextEye = nextEye
        .add(right.mul(dx * step))
        .add(up.mul(dy * step))
        .add(fwd.mul(-dz * step));
    }
    if (turning) {
      // Rotate forward by yaw around world-up, then by pitch around right.
      const yaw = -pendingYaw;
      const pitch = -pendingPitch;
      nextFwd = rotateAxis(nextFwd, up, yaw);
      const newRight = nextFwd.cross(up).normalize();
      nextFwd = rotateAxis(nextFwd, newRight, pitch);
      nextFwd = nextFwd.normalize();
      pendingYaw = 0;
      pendingPitch = 0;
    }

    transact(() => {
      eyeC.value = nextEye;
      fwdC.value = nextFwd;
    });
  };

  // ---- Event wiring ----
  const onKeyDown = (e: KeyboardEvent): void => {
    keys.add(e.key.toLowerCase());
  };
  const onKeyUp = (e: KeyboardEvent): void => {
    keys.delete(e.key.toLowerCase());
  };
  const onPointerDown = (e: PointerEvent): void => {
    if (e.button !== 0) return;
    dragging = true;
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent): void => {
    if (!dragging) return;
    pendingYaw   += e.movementX * turnSensitivity;
    pendingPitch += e.movementY * turnSensitivity;
  };
  const onPointerUp = (e: PointerEvent): void => {
    dragging = false;
    (e.target as HTMLElement).releasePointerCapture?.(e.pointerId);
  };

  return makeHandle({
    eyeC, fwdC, upC,
    target: AVal.zip(eyeC, fwdC).map((e, f) => e.add(f)),
    tick,
    attach(target) {
      target.addEventListener("pointerdown", onPointerDown);
      target.addEventListener("pointermove", onPointerMove);
      target.addEventListener("pointerup", onPointerUp);
      target.addEventListener("pointercancel", onPointerUp);
      const onWindowKeyDown = onKeyDown;
      const onWindowKeyUp = onKeyUp;
      window.addEventListener("keydown", onWindowKeyDown);
      window.addEventListener("keyup", onWindowKeyUp);
      return {
        dispose: () => {
          target.removeEventListener("pointerdown", onPointerDown);
          target.removeEventListener("pointermove", onPointerMove);
          target.removeEventListener("pointerup", onPointerUp);
          target.removeEventListener("pointercancel", onPointerUp);
          window.removeEventListener("keydown", onWindowKeyDown);
          window.removeEventListener("keyup", onWindowKeyUp);
          keys.clear();
        },
      };
    },
    loop: opts.loop ?? "auto",
  });
}

// ---------------------------------------------------------------------------
// Orbit
// ---------------------------------------------------------------------------

export interface OrbitOptions {
  readonly target?: V3d;
  /** Initial distance from target to eye. Default 5. */
  readonly distance?: number;
  /** Initial yaw (rad), 0 = +X axis. Default 0. */
  readonly yaw?: number;
  /** Initial pitch (rad), 0 = horizon. Default 0.3 (slightly above). */
  readonly pitch?: number;
  /** World-up. Default `+Y`. */
  readonly up?: V3d;
  /** Drag sensitivity (rad/pixel). Default 0.005. */
  readonly turnSensitivity?: number;
  /** Wheel zoom factor per step (multiplicative). Default 1.1. */
  readonly zoomFactor?: number;
  /** Frame loop. Tests use `"manual"`; production uses `"auto"`. */
  readonly loop?: "auto" | "manual";
}

/**
 * Orbit-around-target controller.
 *
 * Mouse:
 *   Left-drag    rotate (yaw + pitch)
 *   Middle-drag  pan target along the screen plane
 *   Wheel        zoom (multiplicative on distance)
 *
 * Touch:
 *   1 finger     rotate
 *   2 fingers    pinch zoom + drag pan (translation of the
 *                centroid pans, change in distance zooms)
 */
export function orbitController(opts: OrbitOptions = {}): CameraControllerHandle {
  const targetC: ChangeableValue<V3d> = cval(opts.target ?? V3d.zero);
  const upC: ChangeableValue<V3d>     = cval((opts.up ?? new V3d(0, 1, 0)).normalize());
  const yawC: ChangeableValue<number> = cval(opts.yaw ?? 0);
  const pitchC: ChangeableValue<number> = cval(opts.pitch ?? 0.3);
  const distanceC: ChangeableValue<number> = cval(opts.distance ?? 5);

  const turnSensitivity = opts.turnSensitivity ?? 0.005;
  const zoomFactor = opts.zoomFactor ?? 1.1;

  let pendingYaw = 0;
  let pendingPitch = 0;
  let pendingPan = new V3d(0, 0, 0);
  let pendingZoom = 0;

  // Active pointers, tracked by pointerId so we can support
  // multi-touch (and a stuck-button mouse gesture). For each
  // pointer, the last-seen position in CSS pixels.
  interface ActivePointer { x: number; y: number; type: string; button: number }
  const active = new Map<number, ActivePointer>();
  // Last-known centroid + distance from a 2-pointer gesture, used
  // to compute deltas between successive moves of either finger.
  let twoPointer: { cx: number; cy: number; dist: number } | undefined;

  const eye = AVal.zip(targetC, upC, yawC, pitchC, distanceC).map((tgt, up, yaw, pitch, d) =>
    eyeFromOrbit(tgt, up, yaw, pitch, d),
  );

  const tick = (dt: number): void => {
    void dt;  // orbit changes are immediate (per-event), not time-integrated
    if (
      pendingYaw === 0 && pendingPitch === 0
      && pendingZoom === 0
      && pendingPan.x === 0 && pendingPan.y === 0 && pendingPan.z === 0
    ) return;

    transact(() => {
      if (pendingYaw !== 0)   yawC.value = AVal.force(yawC) - pendingYaw;
      if (pendingPitch !== 0) {
        // Clamp pitch to (-π/2 + ε, π/2 - ε) to avoid gimbal flip.
        const eps = 1e-3;
        const lim = Math.PI / 2 - eps;
        let p = AVal.force(pitchC) - pendingPitch;
        if (p > lim) p = lim;
        if (p < -lim) p = -lim;
        pitchC.value = p;
      }
      if (pendingZoom !== 0) {
        const f = Math.pow(zoomFactor, pendingZoom);
        distanceC.value = AVal.force(distanceC) * f;
      }
      if (pendingPan.x !== 0 || pendingPan.y !== 0 || pendingPan.z !== 0) {
        targetC.value = AVal.force(targetC).add(pendingPan);
      }
    });
    pendingYaw = 0;
    pendingPitch = 0;
    pendingZoom = 0;
    pendingPan = new V3d(0, 0, 0);
  };

  // Build the right / screen-up basis vectors at the current
  // camera pose. Used for screen-plane panning.
  const screenBasis = (): { right: V3d; upScreen: V3d; d: number } => {
    const tgt = AVal.force(targetC);
    const up = AVal.force(upC);
    const yaw = AVal.force(yawC);
    const pitch = AVal.force(pitchC);
    const d = AVal.force(distanceC);
    const eyeNow = eyeFromOrbit(tgt, up, yaw, pitch, d);
    const fwd = tgt.sub(eyeNow).normalize();
    const right = fwd.cross(up).normalize();
    const upScreen = right.cross(fwd).normalize();
    return { right, upScreen, d };
  };

  // ---- Event wiring ----
  const onPointerDown = (e: PointerEvent): void => {
    // Mouse: only react to left + middle. Touch / pen: any
    // pointer counts as an active gesture finger (button is 0
    // for primary touch by spec).
    const isMouse = e.pointerType === "mouse";
    if (isMouse && e.button !== 0 && e.button !== 1) return;
    active.set(e.pointerId, { x: e.clientX, y: e.clientY, type: e.pointerType, button: e.button });
    twoPointer = undefined;  // recompute centroid on next move
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    e.preventDefault();
  };
  const onPointerMove = (e: PointerEvent): void => {
    const prev = active.get(e.pointerId);
    if (prev === undefined) return;
    const dx = e.clientX - prev.x;
    const dy = e.clientY - prev.y;
    prev.x = e.clientX;
    prev.y = e.clientY;

    if (active.size === 1) {
      // 1-pointer rotate (mouse left-drag, single touch, single
      // pen). Middle-mouse drag is the pan path below.
      if (prev.type === "mouse" && prev.button === 1) {
        const { right, upScreen, d } = screenBasis();
        const k = d * 0.001;
        pendingPan = pendingPan
          .add(right.mul(-dx * k))
          .add(upScreen.mul(dy * k));
      } else {
        pendingYaw   += dx * turnSensitivity;
        pendingPitch += dy * turnSensitivity;
      }
      return;
    }

    if (active.size === 2) {
      // 2-pointer pinch + pan. Compute centroid + pairwise
      // distance, compare to last-known.
      const pts = Array.from(active.values());
      const a = pts[0]!, b = pts[1]!;
      const cx = (a.x + b.x) * 0.5;
      const cy = (a.y + b.y) * 0.5;
      const ddx = a.x - b.x, ddy = a.y - b.y;
      const dist = Math.sqrt(ddx * ddx + ddy * ddy);
      if (twoPointer !== undefined) {
        const dCx = cx - twoPointer.cx;
        const dCy = cy - twoPointer.cy;
        const { right, upScreen, d } = screenBasis();
        const k = d * 0.001;
        pendingPan = pendingPan
          .add(right.mul(-dCx * k))
          .add(upScreen.mul(dCy * k));
        // Pinch out (dist grows) → zoom in (negative steps).
        // Each `pendingZoom` step multiplies distance by zoomFactor;
        // tune the divisor so a typical pinch covers a few steps.
        if (twoPointer.dist > 0) {
          const ratio = twoPointer.dist / dist;
          pendingZoom += Math.log(ratio) / Math.log(zoomFactor);
        }
      }
      twoPointer = { cx, cy, dist };
    }
  };
  const onPointerUp = (e: PointerEvent): void => {
    active.delete(e.pointerId);
    if (active.size < 2) twoPointer = undefined;
    (e.target as HTMLElement).releasePointerCapture?.(e.pointerId);
  };
  const onWheel = (e: WheelEvent): void => {
    pendingZoom += e.deltaY > 0 ? 1 : -1;
    e.preventDefault();
  };

  return makeHandle({
    eyeC: eye,            // derived
    fwdC: AVal.zip(eye, targetC).map((e, t) => t.sub(e).normalize()),
    upC,
    target: targetC,
    tick,
    attach(target) {
      target.addEventListener("pointerdown", onPointerDown);
      target.addEventListener("pointermove", onPointerMove);
      target.addEventListener("pointerup", onPointerUp);
      target.addEventListener("pointercancel", onPointerUp);
      target.addEventListener("wheel", onWheel, { passive: false });
      return {
        dispose: () => {
          target.removeEventListener("pointerdown", onPointerDown);
          target.removeEventListener("pointermove", onPointerMove);
          target.removeEventListener("pointerup", onPointerUp);
          target.removeEventListener("pointercancel", onPointerUp);
          target.removeEventListener("wheel", onWheel);
        },
      };
    },
    loop: opts.loop ?? "auto",
  });
}

// ---------------------------------------------------------------------------
// Shared handle construction
// ---------------------------------------------------------------------------

interface MakeHandleSpec {
  readonly eyeC: aval<V3d>;
  readonly fwdC: aval<V3d>;
  readonly upC:  aval<V3d>;
  /** Some lookAt computations want a target instead of a forward; we expose both. */
  readonly target: aval<V3d>;
  tick(dt: number): void;
  attach(target: HTMLElement): { dispose(): void };
  loop: "auto" | "manual";
}

function makeHandle(spec: MakeHandleSpec): CameraControllerHandle {
  const view = lookAt({ eye: spec.eyeC, target: spec.target, up: spec.upC });

  // rAF loop — runs unless disabled.
  let lastTime: number | undefined;
  let rafId: number | undefined;
  let stopped = false;
  const loopCb = (now: number): void => {
    if (stopped) return;
    const dt = lastTime === undefined ? 0 : (now - lastTime) / 1000;
    lastTime = now;
    spec.tick(dt);
    rafId = typeof requestAnimationFrame !== "undefined"
      ? requestAnimationFrame(loopCb)
      : undefined;
  };
  if (spec.loop === "auto" && typeof requestAnimationFrame !== "undefined") {
    rafId = requestAnimationFrame(loopCb);
  }

  const attached: Array<{ dispose(): void }> = [];

  return {
    view,
    eye: spec.eyeC,
    target: spec.target,
    up: spec.upC,
    tick: spec.tick,
    attach(target) {
      const a = spec.attach(target);
      attached.push(a);
      return a;
    },
    dispose() {
      stopped = true;
      if (rafId !== undefined && typeof cancelAnimationFrame !== "undefined") {
        cancelAnimationFrame(rafId);
      }
      for (const a of attached) a.dispose();
      attached.length = 0;
    },
  };
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/** Rotate `v` around `axis` (unit) by `angle` radians. Rodrigues' formula. */
function rotateAxis(v: V3d, axis: V3d, angle: number): V3d {
  if (angle === 0) return v;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const dot = axis.dot(v);
  return v.mul(c).add(axis.cross(v).mul(s)).add(axis.mul(dot * (1 - c)));
}

function eyeFromOrbit(target: V3d, up: V3d, yaw: number, pitch: number, distance: number): V3d {
  // Build a right-handed basis from `up`. Pick an arbitrary axis
  // not parallel to `up` for the initial right-direction reference.
  const ref = Math.abs(up.dot(new V3d(1, 0, 0))) < 0.9
    ? new V3d(1, 0, 0)
    : new V3d(0, 0, 1);
  const right0 = up.cross(ref).normalize();
  const fwd0 = right0.cross(up).normalize();
  // Yaw = rotate forward around up; pitch = tilt up/down.
  const fwd = rotateAxis(fwd0, up, yaw);
  const right = up.cross(fwd).normalize();
  const lookDir = rotateAxis(fwd, right, pitch).normalize();
  // Eye is target offset along -lookDir by distance.
  return target.sub(lookDir.mul(distance));
}
