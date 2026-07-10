// OrbitController — port of Aardvark.Dom.Utilities.OrbitController.
//
// Spherical-coordinate camera (phi / theta / radius around `center`,
// plus an orthogonal screen-space `shift`). Targets are animated by
// damped interpolation in the per-frame integration; centerAnimation
// / locationAnimation / panAnimation use easing kinds (Linear,
// QuadInOut, Cubic, Exp, Tanh, Spring, Pow, Parametric).
//
// Multi-touch:
//   1 finger  → rotate (phi / theta).
//   2 fingers → pan (centroid → shift target) + pinch (radius).

import {
  AVal, cval, transact, avalAddCallback,
  type aval, type ChangeableValue, type IDisposable,
} from "@aardworx/wombat.adaptive";
import { Trafo3d, V2d, V3d, V4d } from "@aardworx/wombat.base";

// ---------------------------------------------------------------------------
// Animation kinds + easings
// ---------------------------------------------------------------------------

export type AnimationKind =
  | { kind: "Linear" }
  | { kind: "QuadInOut" }
  | { kind: "Parametric" }
  | { kind: "Cubic" }
  | { kind: "Exp" }
  | { kind: "Tanh" }
  | { kind: "Spring"; frequency: number }
  | { kind: "Pow"; exp: number };

export const Anim = {
  Linear: { kind: "Linear" } as AnimationKind,
  QuadInOut: { kind: "QuadInOut" } as AnimationKind,
  Parametric: { kind: "Parametric" } as AnimationKind,
  Cubic: { kind: "Cubic" } as AnimationKind,
  Exp: { kind: "Exp" } as AnimationKind,
  Tanh: { kind: "Tanh" } as AnimationKind,
  Spring: (frequency: number): AnimationKind => ({ kind: "Spring", frequency }),
  Pow: (exp: number): AnimationKind => ({ kind: "Pow", exp }),
};

const TWO_PI = Math.PI * 2;

export function getParameter(k: AnimationKind, t: number): number {
  switch (k.kind) {
    case "Linear":     return t;
    case "Parametric": { const sq = t * t; return sq / (2 * (sq - t) + 1); }
    case "QuadInOut":
      if (t < 0.5) return 2 * t * t;
      return -1 - 2 * (t - 2) * t;
    case "Cubic":      { const t2 = t * t; return -2 * t * t2 + 3 * t2; }
    case "Exp":        return 1 - Math.exp(-8 * t);
    case "Tanh":       return Math.tanh(t * 7 - 3.5) * 0.5 + 0.5;
    case "Spring":     return 1 - Math.cos(t * TWO_PI * k.frequency) * Math.exp(-6 * t);
    case "Pow":        return Math.pow(t, k.exp);
  }
}

export interface Animation<T> {
  readonly kind: AnimationKind;
  readonly startTimeMs: number;
  readonly stopTimeMs: number;
  readonly startValue: T;
  readonly stopValue: T;
}

function lerpScalar(a: number, b: number, t: number): number { return a + (b - a) * t; }
function lerpV2(a: V2d, b: V2d, t: number): V2d { return new V2d(lerpScalar(a.x, b.x, t), lerpScalar(a.y, b.y, t)); }
function lerpV3(a: V3d, b: V3d, t: number): V3d {
  return new V3d(lerpScalar(a.x, b.x, t), lerpScalar(a.y, b.y, t), lerpScalar(a.z, b.z, t));
}

function interpRaw<T>(now: number, a: Animation<T>, lerp: (a: T, b: T, t: number) => T): T {
  const span = a.stopTimeMs - a.startTimeMs;
  if (span <= 0) return a.stopValue;
  const t = (now - a.startTimeMs) / span;
  if (t < 0) return a.startValue;
  if (t > 1) return a.stopValue;
  return lerp(a.startValue, a.stopValue, getParameter(a.kind, t));
}

export const interpolateV3 = (now: number, a: Animation<V3d>): V3d => interpRaw(now, a, lerpV3);
export const interpolateV2 = (now: number, a: Animation<V2d>): V2d => interpRaw(now, a, lerpV2);

// ---------------------------------------------------------------------------
// Config + state
// ---------------------------------------------------------------------------

/**
 * Per-axis spring "stiffness" used by the integrator instead of the
 * single-knob `speed`. Each value is the gain that drives `target − cur`
 * → `cur` per `0.05 s` of `dt`. Defaults are picked so that with the
 * default `speed = 0.3` the per-axis behaviour matches the historical
 * `speed * dt / 0.05` law.
 */
export interface OrbitSpringConstants {
  readonly phi: number;
  readonly theta: number;
  readonly radius: number;
  readonly center: number;
}

export interface OrbitConfig {
  readonly radiusRange: V2d;          // (min, max)
  readonly thetaRange: V2d;           // (min, max), elevation in radians
  readonly moveSensitivity: number;
  readonly zoomSensitivity: number;
  readonly speed: number;
  readonly rotateButton: number;      // mouse button id: 0=L, 1=M, 2=R
  readonly panButton: number;
  /**
   * When true (default) MMB pan is free in screen X+Y. When false, pan
   * only moves along the camera's screen-X axis (vertical mouse motion
   * is ignored). Mirrors F# `freeMovePan`.
   */
  readonly freeMovePan: boolean;
  /**
   * Wheel behaviour: `"dolly"` (default) moves the rig forward along
   * the view direction; `"radius"` zooms the camera toward the FIXED
   * orbit center (classic orbit zoom — the center never moves).
   */
  readonly wheelZoom: "dolly" | "radius";
  /**
   * Per-axis spring constants for the integrator. When undefined, a
   * uniform `speed` is used for every axis (back-compat with pre-port
   * behaviour).
   */
  readonly springConstants?: OrbitSpringConstants;
}

export const OrbitConfigDefault: OrbitConfig = {
  wheelZoom: "dolly",
  radiusRange: new V2d(0.1, 40_000_000),
  thetaRange: new V2d(-Math.PI / 2 + 1e-4, Math.PI / 2 - 1e-4),
  moveSensitivity: 0.5,
  zoomSensitivity: 1,
  speed: 0.3,
  rotateButton: 0,
  panButton: 1,
  freeMovePan: true,
};

export interface DragStart {
  readonly pos: { x: number; y: number };
  readonly button: number;
}

export interface OrbitState {
  readonly sky: V3d;
  readonly center: V3d;
  readonly phi: number;
  readonly theta: number;
  readonly radius: number;
  readonly shift: V2d;

  readonly targetPhi: number;
  readonly targetTheta: number;
  readonly targetRadius: number;

  readonly centerAnimation: Animation<V3d> | undefined;
  readonly locationAnimation: Animation<V3d> | undefined;
  readonly panAnimation: Animation<V2d> | undefined;

  readonly dragStarts: ReadonlyMap<number, DragStart>;
  readonly lastRenderMs: number | undefined;
  readonly config: OrbitConfig;
  readonly userModifiedAngles: boolean;
  readonly userModifiedCenter: boolean;
  readonly userModifiedRadius: boolean;
  readonly lockedToScene: boolean;
  readonly isOrtho: boolean;
}

export interface OrbitInitial {
  center?: V3d;
  phi?: number;
  theta?: number;
  radius?: number;
  sky?: V3d;
  config?: Partial<OrbitConfig>;
}

const clamp = (lo: number, hi: number, v: number): number => v < lo ? lo : v > hi ? hi : v;

export function defaultOrbitState(init: OrbitInitial = {}): OrbitState {
  const config: OrbitConfig = { ...OrbitConfigDefault, ...(init.config ?? {}) };
  const radius = clamp(config.radiusRange.x, config.radiusRange.y, init.radius ?? 5);
  const theta = clamp(config.thetaRange.x, config.thetaRange.y, init.theta ?? 0.4);
  const phi = ((init.phi ?? 0) % TWO_PI);
  return {
    sky: (init.sky ?? new V3d(0, 0, 1)).normalize(),
    center: init.center ?? V3d.zero,
    phi, theta, radius,
    shift: new V2d(0, 0),
    targetPhi: phi, targetTheta: theta, targetRadius: radius,
    centerAnimation: undefined,
    locationAnimation: undefined,
    panAnimation: undefined,
    dragStarts: new Map(),
    lastRenderMs: undefined,
    config,
    userModifiedAngles: false,
    userModifiedCenter: false,
    userModifiedRadius: false,
    lockedToScene: true,
    isOrtho: false,
  };
}

// ---------------------------------------------------------------------------
// View math (mirrors F# OrbitState.withView)
// ---------------------------------------------------------------------------

export interface OrbitView { eye: V3d; forward: V3d; right: V3d; up: V3d; sky: V3d }

export function deriveView(s: OrbitState): OrbitView {
  const ct = Math.cos(s.theta);
  const dir = new V3d(Math.cos(s.phi) * ct, Math.sin(s.phi) * ct, Math.sin(s.theta));
  const eyeRaw =
    s.radius <= 1.02 * s.config.radiusRange.x
      ? s.center
      : dir.mul(s.radius).add(s.center);
  const r = s.sky.cross(dir).normalize();
  const up = dir.cross(r).normalize();
  const forward = dir.mul(-1);  // looking back at center
  // Apply screen-space shift:
  const eye = eyeRaw
    .add(r.mul(s.shift.x * -0.001))
    .add(up.mul(s.shift.y * 0.001));
  return { eye, forward, right: r, up, sky: s.sky };
}

// ---------------------------------------------------------------------------
// Per-frame integration
// ---------------------------------------------------------------------------

function step(s: OrbitState, nowMs: number): OrbitState {
  // Wrap dphi to (-π, π].
  let dphi = ((s.targetPhi - s.phi) % TWO_PI);
  if (dphi < -Math.PI) dphi += TWO_PI;
  else if (dphi > Math.PI) dphi -= TWO_PI;
  const dtheta = s.targetTheta - s.theta;
  const dradius = s.targetRadius - s.radius;

  const dt = s.lastRenderMs === undefined ? 0 : (nowMs - s.lastRenderMs) / 1000;
  const sk = s.config.springConstants;
  const partOf = (gain: number): number => dt > 0 ? clamp(0, 1, gain * dt / 0.05) : 0;
  const partPhi = partOf(sk?.phi ?? s.config.speed);
  const partTheta = partOf(sk?.theta ?? s.config.speed);
  const partRadius = partOf(sk?.radius ?? s.config.speed);
  const partCenter = partOf(sk?.center ?? s.config.speed);
  let next: OrbitState = { ...s, lastRenderMs: nowMs };

  if (Math.abs(dphi) > 0) {
    next = Math.abs(dphi) < 1e-4
      ? { ...next, phi: next.targetPhi }
      : { ...next, phi: (next.phi + partPhi * dphi) % TWO_PI };
  }
  if (Math.abs(dtheta) > 0) {
    next = Math.abs(dtheta) < 1e-4
      ? { ...next, theta: next.targetTheta }
      : { ...next, theta: next.theta + partTheta * dtheta };
  }
  if (Math.abs(dradius) > 0) {
    next = Math.abs(dradius) < 1e-4
      ? { ...next, radius: next.targetRadius }
      : { ...next, radius: next.radius + partRadius * dradius };
  }

  // Center / location animation (requires deriving current view location).
  if (next.centerAnimation && !next.panAnimation) {
    const ca = next.centerAnimation;
    const la = next.locationAnimation;
    const v = deriveView(next);
    if (la) {
      const dLoc = la.stopValue.sub(v.eye);
      const dCenter = ca.stopValue.sub(next.center);
      const setLocation = (location: V3d, center: V3d, st: OrbitState): OrbitState => {
        const diff = location.sub(center);
        const r = diff.length();
        const phi = Math.atan2(diff.y, diff.x);
        const theta = r > 0 ? Math.asin(diff.z / r) : st.theta;
        return {
          ...st,
          center, radius: r, targetRadius: r,
          phi, targetPhi: phi,
          theta, targetTheta: theta,
        };
      };
      if (ca.kind.kind === "Exp") {
        if (nowMs >= ca.stopTimeMs) {
          next = setLocation(la.stopValue, ca.stopValue, { ...next, centerAnimation: undefined, locationAnimation: undefined });
        } else {
          next = setLocation(v.eye.add(dLoc.mul(partCenter)), next.center.add(dCenter.mul(partCenter)), next);
        }
      } else {
        if (nowMs < ca.stopTimeMs) {
          const pos = interpolateV3(nowMs, ca);
          const loc = interpolateV3(nowMs, la);
          next = setLocation(loc, pos, next);
        } else {
          next = setLocation(la.stopValue, ca.stopValue, { ...next, centerAnimation: undefined, locationAnimation: undefined });
        }
      }
    } else {
      const dCenter = ca.stopValue.sub(next.center);
      const len = dCenter.length();
      if (ca.kind.kind === "Exp") {
        if (len < 1e-4) next = { ...next, center: ca.stopValue, centerAnimation: undefined };
        else next = { ...next, center: next.center.add(dCenter.mul(partCenter)) };
      } else {
        if (nowMs < ca.stopTimeMs) {
          next = { ...next, center: interpolateV3(nowMs, ca) };
        } else {
          next = { ...next, center: ca.stopValue, centerAnimation: undefined };
        }
      }
    }
  }

  if (next.panAnimation) {
    const pa = next.panAnimation;
    const dCenter = pa.stopValue.sub(next.shift);
    const len = Math.hypot(dCenter.x, dCenter.y);
    if (pa.kind.kind === "Exp") {
      if (len < 1e-4) next = { ...next, shift: pa.stopValue, panAnimation: undefined };
      else next = { ...next, shift: next.shift.add(dCenter.mul(partCenter)) };
    } else {
      if (nowMs < pa.stopTimeMs) {
        next = { ...next, shift: interpolateV2(nowMs, pa) };
      } else {
        next = { ...next, shift: pa.stopValue, panAnimation: undefined };
      }
    }
  }

  return next;
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

export interface OrbitAttachOptions {
  /** Optional pick callback for depth-aware pan. Not currently used by the controller core, kept for API parity. */
  pickDepth?: (clientX: number, clientY: number) => number | undefined;
  /**
   * Optional world-space picker — enables pick-anchored navigation:
   *   - starting a ROTATE drag re-centres the orbit on the world hit
   *     under the cursor (camera stays put; background keeps the
   *     current center), so rotation pivots around what you grabbed;
   *   - starting a PAN drag anchors the hit point to the cursor —
   *     with `proj` provided the pan is pixel-perfect (the grabbed
   *     point follows the pointer exactly; background pans at the
   *     last known pick distance).
   */
  picker?: (clientX: number, clientY: number) => Promise<V3d | undefined>;
  /**
   * Projection trafo of the viewport this controller drives. Required
   * for pixel-perfect pan (unprojecting the cursor at the anchor's
   * depth); without it pan falls back to the radius-scaled heuristic.
   */
  proj?: aval<Trafo3d>;
}

export class OrbitController {
  readonly state: ChangeableValue<OrbitState>;
  readonly view: aval<Trafo3d>;
  readonly eye: aval<V3d>;
  readonly forward: aval<V3d>;

  constructor(initial?: OrbitState | OrbitInitial) {
    // `dragStarts` only exists on a full OrbitState — an OrbitInitial
    // with a `config` override must NOT be misclassified as one.
    const s0: OrbitState =
      initial && "dragStarts" in initial
        ? (initial as OrbitState)
        : defaultOrbitState(initial as OrbitInitial | undefined);
    this.state = cval(s0);
    this.view = this.state.map(s => {
      const v = deriveView(s);
      return Trafo3d.viewTrafoRH(v.eye, v.up, v.forward);
    });
    this.eye = this.state.map(s => deriveView(s).eye);
    this.forward = this.state.map(s => deriveView(s).forward);
  }

  static create(initial?: OrbitState | OrbitInitial): OrbitController {
    return new OrbitController(initial);
  }

  // ---------- mutation API ----------

  setPhi(phi: number): void { this.update(s => ({ ...s, phi: phi % TWO_PI, targetPhi: phi % TWO_PI })); }
  setTheta(theta: number): void {
    this.update(s => {
      const t = clamp(s.config.thetaRange.x, s.config.thetaRange.y, theta);
      return { ...s, theta: t, targetTheta: t };
    });
  }
  setRadius(r: number): void {
    this.update(s => {
      const rc = clamp(s.config.radiusRange.x, s.config.radiusRange.y, r);
      return { ...s, radius: rc, targetRadius: rc };
    });
  }
  setCenter(c: V3d): void { this.update(s => ({ ...s, center: c, centerAnimation: undefined })); }

  setTargetPhi(phi: number, user = true): void {
    this.update(s => ({ ...s, targetPhi: phi % TWO_PI, userModifiedAngles: user, locationAnimation: undefined, lastRenderMs: undefined }));
  }
  setTargetTheta(theta: number, user = true): void {
    this.update(s => ({ ...s, targetTheta: clamp(s.config.thetaRange.x, s.config.thetaRange.y, theta), userModifiedAngles: user, locationAnimation: undefined, lastRenderMs: undefined }));
  }
  setTargetRadius(r: number, user = true): void {
    this.update(s => ({ ...s, targetRadius: clamp(s.config.radiusRange.x, s.config.radiusRange.y, r), userModifiedRadius: user, locationAnimation: undefined, lastRenderMs: undefined }));
  }

  set(center: V3d, radius: number, phi: number, theta: number): void {
    this.update(s => {
      const r = clamp(s.config.radiusRange.x, s.config.radiusRange.y, radius);
      const t = clamp(s.config.thetaRange.x, s.config.thetaRange.y, theta);
      const p = phi % TWO_PI;
      return {
        ...s, center, centerAnimation: undefined, panAnimation: undefined, locationAnimation: undefined,
        shift: new V2d(0, 0),
        radius: r, targetRadius: r,
        phi: p, targetPhi: p,
        theta: t, targetTheta: t,
        lastRenderMs: undefined,
      };
    });
  }

  /**
   * Animate to a new center (and optionally radius) using the given
   * easing. Default `Tanh` over 600ms, matching the F# orbit
   * controller's "snap" feel.
   */
  flyTo(
    center: V3d,
    kind: AnimationKind = Anim.Tanh,
    durationMs = 600,
  ): void {
    this.update(s => {
      const now = performance.now();
      const centerAnim: Animation<V3d> = {
        kind, startTimeMs: now, stopTimeMs: now + durationMs,
        startValue: s.center, stopValue: center,
      };
      return {
        ...s, userModifiedCenter: true, panAnimation: undefined,
        centerAnimation: centerAnim, locationAnimation: undefined, lastRenderMs: undefined,
      };
    });
  }

  /** Manually advance integration. Tests pass synthetic `nowMs`. */
  tick(nowMs?: number): void {
    const t = nowMs ?? performance.now();
    this.update(s => step(s, t));
  }

  private update(f: (s: OrbitState) => OrbitState): void {
    transact(() => { this.state.value = f(this.state.value); });
  }

  // ---------- DOM wiring ----------

  attach(target: HTMLElement, time: aval<number>, opts: OrbitAttachOptions = {}): () => void {

    // Pointer tracking — distinct from F# state because mouse + touch
    // both flow through PointerEvents in the browser. We still feed the
    // F# `dragStarts` map equivalent via the state's MMB/L button hooks.
    interface PtrInfo { x: number; y: number; type: string; button: number; anchor?: V3d }
    const pointers = new Map<number, PtrInfo>();
    // Last known pick distance — background pans reuse it (falls back
    // to the orbit radius before the first anchored pan).
    let lastPanDepth = Math.max(this.state.value.radius, 0.3);

    // Rodrigues rotation of `v` about unit axis `a` by `ang`.
    const rotAxis = (v: V3d, a: V3d, ang: number): V3d => {
      const c = Math.cos(ang), si = Math.sin(ang);
      const cross = a.cross(v);
      const dot = a.dot(v);
      return v.mul(c).add(cross.mul(si)).add(a.mul(dot * (1 - c)));
    };

    // World point on the cursor ray at view-depth `zA` (< 0), using the
    // CURRENT camera. Pixel-perfect pan moves the rig by
    // `at(prev) − at(cur)` so the anchored point follows the pointer.
    const worldAtDepth = (s: OrbitState, cssX: number, cssY: number, zA: number): V3d | undefined => {
      if (opts.proj === undefined) return undefined;
      const rect = target.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return undefined;
      const v = deriveView(s);
      const vt = Trafo3d.viewTrafoRH(v.eye, v.up, v.forward);
      const projInv = AVal.force(opts.proj).backward;
      const ndcX = 2 * (cssX - rect.left) / rect.width - 1;
      const ndcY = 1 - 2 * (cssY - rect.top) / rect.height;
      const v4 = projInv.transform(new V4d(ndcX, ndcY, 0.5, 1));
      if (v4.w === 0) return undefined;
      const pv = new V3d(v4.x / v4.w, v4.y / v4.w, v4.z / v4.w);
      if (pv.z >= -1e-9) return undefined;
      const t = zA / pv.z;
      return vt.backward.transformPos(pv.mul(t));
    };

    const onPointerDown = (e: PointerEvent): void => {
      const isMouse = e.pointerType === "mouse";
      // For mouse, only react to L/M/R; touch/pen always count.
      if (isMouse && e.button !== 0 && e.button !== 1 && e.button !== 2) return;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, type: e.pointerType, button: e.button });
      this.update(s => ({
        ...s,
        dragStarts: new Map(s.dragStarts).set(e.pointerId, { pos: { x: e.clientX, y: e.clientY }, button: isMouse ? e.button : -1 }),
        lastRenderMs: undefined,
      }));
      (target as Element).setPointerCapture?.(e.pointerId);
      e.preventDefault();

      // Pick-anchored navigation: resolve the world point under the
      // cursor and stash it as the drag's anchor. Rotate drags pivot
      // the whole rig around it (no snap — orientation preserved);
      // pan drags keep it glued to the pointer. Background grabs
      // leave the anchor unset → classic center-based behaviour.
      if (opts.picker !== undefined && pointers.size === 1) {
        const id = e.pointerId;
        const isPan = isMouse && e.button === this.state.value.config.panButton;
        void opts.picker(e.clientX, e.clientY).then((hit) => {
          const info = pointers.get(id);
          if (info === undefined || hit === undefined) return;
          info.anchor = hit;
          if (isPan) {
            const v = deriveView(this.state.value);
            const vt = Trafo3d.viewTrafoRH(v.eye, v.up, v.forward);
            const zA = vt.forward.transformPos(hit).z;
            if (zA < -1e-6) lastPanDepth = -zA;
          }
        });
      }
    };

    const onPointerUp = (e: PointerEvent): void => {
      const wasMouse = pointers.get(e.pointerId)?.type === "mouse";
      const start = this.state.value.dragStarts.get(e.pointerId);
      const wasPan = wasMouse && start !== undefined && start.button === this.state.value.config.panButton;
      pointers.delete(e.pointerId);
      this.update(s => {
        const m = new Map(s.dragStarts);
        m.delete(e.pointerId);
        return { ...s, dragStarts: m, lastRenderMs: undefined };
      });
      (target as Element).releasePointerCapture?.(e.pointerId);
      void wasPan; // pick anchoring happens at pointer-DOWN now
    };

    const onPointerMove = (e: PointerEvent): void => {
      const prev = pointers.get(e.pointerId);
      if (!prev) return;
      const cur = { x: e.clientX, y: e.clientY };
      pointers.set(e.pointerId, { ...prev, ...cur });

      const n = pointers.size;
      if (n === 1) {
        this.update(s => {
          const start = s.dragStarts.get(e.pointerId);
          if (!start) return s;
          const dx = cur.x - start.pos.x;
          const dy = cur.y - start.pos.y;
          const isTouch = prev.type !== "mouse";
          const isRotate = isTouch || start.button === s.config.rotateButton;
          const isPan = !isTouch && start.button === s.config.panButton;
          if (isRotate) {
            const dphi = dx * -0.01 * s.config.moveSensitivity;
            const dtheta = dy * 0.01 * s.config.moveSensitivity;
            if (Math.abs(dphi) < 1e-8 && Math.abs(dtheta) < 1e-8) return s;
            const newStarts = new Map(s.dragStarts).set(e.pointerId, { pos: cur, button: start.button });
            // Pick-anchored rotate: rigidly rotate the WHOLE rig (eye,
            // orientation, center) around the grabbed world point — the
            // center is not re-seated, it just swings along. Applying
            // the same dphi/dtheta to the angles and rotating `center`
            // around the pivot by Rz(dphi)·R_right(−dtheta) keeps
            // eye = center + dir(phi,theta)·radius exact, so there is
            // no visual snap at any step. (Direct, unsprung — a drag
            // tracks the pointer.)
            const pivot = pointers.get(e.pointerId)?.anchor;
            if (pivot !== undefined) {
              const newTheta = clamp(s.config.thetaRange.x, s.config.thetaRange.y, s.theta + dtheta);
              const dthetaEff = newTheta - s.theta;
              const newPhi = (s.phi + dphi) % TWO_PI;
              const a = new V3d(-Math.sin(s.phi), Math.cos(s.phi), 0); // right axis
              const v0 = s.center.sub(pivot);
              const v1 = rotAxis(v0, a, -dthetaEff);
              const v2 = new V3d(
                v1.x * Math.cos(dphi) - v1.y * Math.sin(dphi),
                v1.x * Math.sin(dphi) + v1.y * Math.cos(dphi),
                v1.z,
              );
              return {
                ...s,
                dragStarts: newStarts,
                userModifiedAngles: true,
                userModifiedCenter: true,
                center: pivot.add(v2),
                phi: newPhi, targetPhi: newPhi,
                theta: newTheta, targetTheta: newTheta,
                centerAnimation: undefined,
                panAnimation: undefined,
                locationAnimation: undefined,
              };
            }
            return {
              ...s,
              dragStarts: newStarts,
              userModifiedAngles: true,
              targetPhi: (s.targetPhi + dphi) % TWO_PI,
              targetTheta: clamp(s.config.thetaRange.x, s.config.thetaRange.y, s.targetTheta + dtheta),
            };
          } else if (isPan) {
            // Pixel-perfect pan: keep the anchored world point (or the
            // last known pick distance for background grabs) glued to
            // the pointer. Falls back to the radius-scaled heuristic
            // when no `proj` was provided.
            if (opts.proj !== undefined) {
              const info = pointers.get(e.pointerId);
              let zA = -lastPanDepth;
              if (info?.anchor !== undefined) {
                const v = deriveView(s);
                const vt = Trafo3d.viewTrafoRH(v.eye, v.up, v.forward);
                const az = vt.forward.transformPos(info.anchor).z;
                if (az < -1e-6) { zA = az; lastPanDepth = -az; }
              }
              const wPrev = worldAtDepth(s, start.pos.x, start.pos.y, zA);
              const wCur = worldAtDepth(s, cur.x, cur.y, zA);
              if (wPrev !== undefined && wCur !== undefined) {
                return {
                  ...s,
                  dragStarts: new Map(s.dragStarts).set(e.pointerId, { pos: cur, button: start.button }),
                  userModifiedCenter: false,
                  centerAnimation: undefined,
                  center: s.center.add(wPrev.sub(wCur)),
                  lockedToScene: false,
                };
              }
            }
            const v = deriveView(s);
            const r = Math.max(s.radius, 0.3);
            const dyEff = s.config.freeMovePan ? dy : 0;
            const newCenter = s.center
              .add(v.right.mul(dx * -0.001 * r))
              .add(v.up.mul(dyEff * 0.001 * r));
            return {
              ...s,
              dragStarts: new Map(s.dragStarts).set(e.pointerId, { pos: cur, button: start.button }),
              userModifiedCenter: false,
              centerAnimation: undefined,
              center: newCenter,
              lockedToScene: false,
            };
          }
          return s;
        });
      } else if (n === 2) {
        // 2-pointer pinch + rotate (touch).
        this.update(s => {
          const start = s.dragStarts.get(e.pointerId);
          if (!start) return s;
          const otherEntry = Array.from(s.dragStarts.entries()).find(([k]) => k !== e.pointerId);
          if (!otherEntry) return s;
          const otherPos = otherEntry[1].pos;
          const op = start.pos;
          const np = cur;
          const oldD = Math.hypot(op.x - otherPos.x, op.y - otherPos.y);
          const newD = Math.hypot(np.x - otherPos.x, np.y - otherPos.y);
          const scale = oldD > 0 ? newD / oldD : 1;
          const r = clamp(s.config.radiusRange.x, s.config.radiusRange.y, s.targetRadius / scale);
          const dx = 0.5 * (np.x - op.x);
          const dy = 0.5 * (np.y - op.y);
          const dphi = dx * -0.01 * s.config.moveSensitivity;
          const dtheta = dy * 0.01 * s.config.moveSensitivity;
          return {
            ...s,
            userModifiedAngles: true,
            userModifiedRadius: true,
            dragStarts: new Map(s.dragStarts).set(e.pointerId, { pos: cur, button: start.button }),
            targetPhi: (s.targetPhi + dphi) % TWO_PI,
            targetTheta: clamp(s.config.thetaRange.x, s.config.thetaRange.y, s.targetTheta + dtheta),
            targetRadius: r,
          };
        });
      }
    };

    const onWheel = (e: WheelEvent): void => {
      const delta = e.deltaY / 120;
      const shift = e.shiftKey;
      this.update(s => {
        if (shift || s.lockedToScene || s.isOrtho || s.config.wheelZoom === "radius") {
          const factor = Math.pow(1.1, delta * s.config.zoomSensitivity);
          return {
            ...s,
            userModifiedRadius: true,
            targetRadius: clamp(s.config.radiusRange.x, s.config.radiusRange.y, s.targetRadius * factor),
          };
        } else {
          const v = deriveView(s);
          // Dolly the rig forward proportional to the current radius —
          // an absolute step is invisible in large scenes (a city at
          // radius 3000 moved 0.5 units per notch = "zoom stopped
          // working") and violent in small ones.
          const stepLen = Math.max(s.radius, 0.3) * 0.15;
          const newCenter = s.center.add(v.forward.mul(-delta * stepLen * s.config.zoomSensitivity));
          const now = performance.now();
          const anim: Animation<V3d> = {
            kind: Anim.Exp,
            startTimeMs: now,
            stopTimeMs: now + 120,
            startValue: s.center,
            stopValue: newCenter,
          };
          return { ...s, userModifiedCenter: true, centerAnimation: anim, lastRenderMs: undefined };
        }
      });
      e.preventDefault();
    };

    const onContextMenu = (e: Event): void => e.preventDefault();

    // ---- Frame integration ----
    const cb: IDisposable = avalAddCallback(time, (now: number) => {
      // step uses lastRenderMs to compute dt; on first tick lastRenderMs is undefined so part=0.
      this.update(s => step(s, now));
    });

    target.addEventListener("pointerdown", onPointerDown as EventListener);
    target.addEventListener("pointermove", onPointerMove as EventListener);
    target.addEventListener("pointerup", onPointerUp as EventListener);
    target.addEventListener("pointercancel", onPointerUp as EventListener);
    target.addEventListener("wheel", onWheel, { passive: false });
    target.addEventListener("contextmenu", onContextMenu);

    return () => {
      cb.dispose();
      target.removeEventListener("pointerdown", onPointerDown as EventListener);
      target.removeEventListener("pointermove", onPointerMove as EventListener);
      target.removeEventListener("pointerup", onPointerUp as EventListener);
      target.removeEventListener("pointercancel", onPointerUp as EventListener);
      target.removeEventListener("wheel", onWheel);
      target.removeEventListener("contextmenu", onContextMenu);
    };
  }
}

void AVal;
