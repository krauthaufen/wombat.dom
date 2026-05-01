// FreeFlyController — port of Aardvark.Dom.Utilities.FreeFlyController.
//
// Owns a `cval<FreeFlyState>` and integrates motion once per frame
// against an externally-supplied time aval (typically `state.time`
// from the scene scope, or RenderControl's per-frame clock). The
// per-frame Rendered branch from the F# update is reproduced in
// `step()`; input is wired to DOM events on a canvas via
// `attach()`.
//
// Multi-touch: replaces the F# virtual-touch-sticks.js. We track
// active pointers in a Map<pointerId, { x, y }> and route gestures
// based on the active count:
//   1 finger  → rotate (TurnVec)
//   2 fingers → pan (XY of centroid → AddTargetMove) + pinch (Z)
//   3 fingers → forward/back (Y of centroid → AddTargetMove on Z)

import {
  AVal, cval, transact, avalAddCallback,
  type aval, type ChangeableValue, type IDisposable,
} from "@aardworx/wombat.adaptive";
import { Rot3d, Trafo3d, V2d, V3d } from "@aardworx/wombat.base";

// ---------------------------------------------------------------------------
// Config + state
// ---------------------------------------------------------------------------

export interface FreeFlyConfig {
  readonly MoveSpeed: number;
  readonly Damping: number;
  readonly TurnSpeed: number;
  readonly TurnFactor: number;
  readonly MoveFactor: number;
  readonly WheelSensitivity: number;
  readonly TurnSensitivity: number;
  readonly PanSensitivity: number;
  readonly StickExponent: number;
}

export const FreeFlyConfigDefault: FreeFlyConfig = {
  MoveSpeed: 1.5,
  Damping: 20,
  TurnSpeed: 1.5,
  TurnFactor: 30,
  MoveFactor: 10,
  WheelSensitivity: 1,
  TurnSensitivity: 1,
  PanSensitivity: 1,
  StickExponent: 2,
};

export interface FreeFlyState {
  readonly Enabled: boolean;
  readonly Position: V3d;
  readonly Sky: V3d;
  readonly Forward: V3d;
  readonly Config: FreeFlyConfig;
  readonly MoveVectors: ReadonlyMap<string, V3d>;
  readonly TurnVectors: ReadonlyMap<string, V2d>;
  readonly PanMove: boolean;
  readonly SprintFactor: number;
  readonly Momentum: V3d;
  readonly TargetMoveLocal: V3d;
  readonly TargetMoveGlobal: V3d;
  readonly TargetTurn: V2d;
  readonly Camera: { eye: V3d; forward: V3d; sky: V3d };
}

const TINY = 1e-4;
const isTinyV3 = (v: V3d): boolean => Math.abs(v.x) < TINY && Math.abs(v.y) < TINY && Math.abs(v.z) < TINY;
const isTinyV2 = (v: V2d): boolean => Math.abs(v.x) < TINY && Math.abs(v.y) < TINY;

function moveVecOf(s: FreeFlyState): V3d {
  let x = 0, y = 0, z = 0;
  for (const v of s.MoveVectors.values()) { x += v.x; y += v.y; z += v.z; }
  // PanMove swap: F# uses .XZY  → (x, z, y)
  if (s.PanMove) return new V3d(x, z, y);
  return new V3d(x, y, z);
}

function turnVecOf(s: FreeFlyState): V2d {
  let x = 0, y = 0;
  for (const v of s.TurnVectors.values()) { x += v.x; y += v.y; }
  return new V2d(x, y);
}

export function isAnimating(s: FreeFlyState): boolean {
  return !isTinyV3(s.Momentum)
      || !isTinyV3(moveVecOf(s))
      || !isTinyV2(s.TargetTurn)
      || !isTinyV2(turnVecOf(s))
      || !isTinyV3(s.TargetMoveLocal)
      || !isTinyV3(s.TargetMoveGlobal);
}

function withCameraDerived(s: FreeFlyState): FreeFlyState {
  return { ...s, Camera: { eye: s.Position, forward: s.Forward, sky: s.Sky } };
}

export interface FreeFlyInitial {
  position?: V3d;
  forward?: V3d;
  sky?: V3d;
  config?: FreeFlyConfig;
}

export function defaultFreeFlyState(init: FreeFlyInitial = {}): FreeFlyState {
  const Position = init.position ?? new V3d(0, 0, 5);
  const Forward = (init.forward ?? new V3d(0, 0, -1)).normalize();
  const Sky = (init.sky ?? new V3d(0, 1, 0)).normalize();
  const s: FreeFlyState = {
    Enabled: true,
    Position, Sky, Forward,
    Config: init.config ?? FreeFlyConfigDefault,
    MoveVectors: new Map(),
    TurnVectors: new Map(),
    PanMove: false,
    SprintFactor: 1,
    Momentum: V3d.zero,
    TargetMoveLocal: V3d.zero,
    TargetMoveGlobal: V3d.zero,
    TargetTurn: new V2d(0, 0),
    Camera: { eye: Position, forward: Forward, sky: Sky },
  };
  return s;
}

// ---------------------------------------------------------------------------
// Per-frame integration (Rendered branch)
// ---------------------------------------------------------------------------

function step(s: FreeFlyState, dt: number): FreeFlyState {
  if (!isAnimating(s) || dt <= 0) return s;

  const right = s.Forward.cross(s.Sky).normalize();
  const up = right.cross(s.Forward).normalize();

  const dstScale = Math.min(1, s.Config.MoveFactor * dt);
  const dstMove = s.TargetMoveLocal.mul(dstScale);
  const dstMoveGlobal = s.TargetMoveGlobal.mul(dstScale);

  const targetMove = s.Momentum.mul(s.Config.MoveSpeed * dt);
  const mv = moveVecOf(s);
  const moveLocal = mv.mul(s.Config.MoveSpeed * s.SprintFactor * dt);

  const moveVec = targetMove.add(moveLocal).add(dstMove);
  const move =
    s.Forward.mul(moveVec.z).add(
      up.mul(moveVec.y)).add(
      right.mul(moveVec.x)).add(
      dstMoveGlobal);

  const turnFactor = Math.min(1, s.Config.TurnFactor * dt);
  const rotSkyAngle = s.TargetTurn.x * turnFactor;
  const rotRightAngle = s.TargetTurn.y * turnFactor;

  const turn = turnVecOf(s).mul(s.Config.TurnSpeed * dt);

  const r1 = Rot3d.rotation(s.Sky, rotSkyAngle + turn.x);
  const r2 = Rot3d.rotation(right, rotRightAngle + turn.y);
  // Equivalent of F# `r1 * r2` applied to forward: first r2 then r1.
  const forwardAfterR2 = r2.transform(s.Forward);
  const newForward = r1.transform(forwardAfterR2).normalize();

  const damp = Math.pow(0.5, s.Config.Damping * dt);
  const newMomentum = s.Momentum.mul(damp);

  return withCameraDerived({
    ...s,
    TargetMoveLocal: s.TargetMoveLocal.sub(dstMove),
    TargetMoveGlobal: s.TargetMoveGlobal.sub(dstMoveGlobal),
    TargetTurn: new V2d(s.TargetTurn.x - rotSkyAngle, s.TargetTurn.y - rotRightAngle),
    Position: s.Position.add(move),
    Forward: newForward,
    Momentum: newMomentum,
  });
}

// ---------------------------------------------------------------------------
// State mutators (mirror F# update branches that aren't `Rendered`)
// ---------------------------------------------------------------------------

function alterAdd<K, V>(map: ReadonlyMap<K, V>, key: K, add: V, addOp: (a: V, b: V) => V, isTinyOp: (v: V) => boolean): ReadonlyMap<K, V> {
  const m = new Map(map);
  const old = m.get(key);
  const next = old === undefined ? add : addOp(old, add);
  if (isTinyOp(next)) m.delete(key);
  else m.set(key, next);
  return m;
}

function setOrRemove<K, V>(map: ReadonlyMap<K, V>, key: K, value: V, isTinyOp: (v: V) => boolean): ReadonlyMap<K, V> {
  const m = new Map(map);
  if (isTinyOp(value)) m.delete(key);
  else m.set(key, value);
  return m;
}

const addV3 = (a: V3d, b: V3d): V3d => a.add(b);
const addV2 = (a: V2d, b: V2d): V2d => a.add(b);

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

export interface FreeFlyAttachOptions {
  /** When provided, MMB pan-speed scales with the picked depth at the click point. */
  pickDepth?: (clientX: number, clientY: number) => number | undefined;
  /** If true, attach key listeners to `window`; otherwise to `target`. Default: window. */
  keysOnWindow?: boolean;
}

export class FreeFlyController {
  readonly state: ChangeableValue<FreeFlyState>;
  readonly view: aval<Trafo3d>;
  readonly eye: aval<V3d>;
  readonly forward: aval<V3d>;
  readonly sky: aval<V3d>;

  constructor(initial?: FreeFlyState | FreeFlyInitial) {
    const s0: FreeFlyState =
      initial && "Position" in initial && "Forward" in initial
        ? (initial as FreeFlyState)
        : defaultFreeFlyState(initial as FreeFlyInitial | undefined);
    this.state = cval(s0);
    this.view = this.state.map(s =>
      Trafo3d.viewTrafoRH(s.Position, s.Sky, s.Forward));
    this.eye = this.state.map(s => s.Position);
    this.forward = this.state.map(s => s.Forward);
    this.sky = this.state.map(s => s.Sky);
  }

  static create(initial?: FreeFlyState | FreeFlyInitial): FreeFlyController {
    return new FreeFlyController(initial);
  }

  // ---------- mutation helpers (used by attach() and tests) ----------

  setEnabled(e: boolean): void { this.update(s => ({ ...s, Enabled: e })); }

  setPanMove(p: boolean): void { this.guarded(s => ({ ...s, PanMove: p })); }

  setMoveVec(source: string, v: V3d): void {
    this.guarded(s => ({ ...s, MoveVectors: setOrRemove(s.MoveVectors, source, v, isTinyV3) }));
  }

  addMoveVec(source: string, v: V3d): void {
    this.guarded(s => ({ ...s, MoveVectors: alterAdd(s.MoveVectors, source, v, addV3, isTinyV3) }));
  }

  setTurnVec(source: string, v: V2d): void {
    this.guarded(s => ({ ...s, TurnVectors: setOrRemove(s.TurnVectors, source, v, isTinyV2) }));
  }

  addTurnVec(source: string, v: V2d): void {
    this.guarded(s => ({ ...s, TurnVectors: alterAdd(s.TurnVectors, source, v, addV2, isTinyV2) }));
  }

  addTargetMove(d: V3d): void { this.guarded(s => ({ ...s, TargetMoveLocal: s.TargetMoveLocal.add(d) })); }
  addTargetMoveGlobal(d: V3d): void { this.guarded(s => ({ ...s, TargetMoveGlobal: s.TargetMoveGlobal.add(d) })); }
  addMomentum(d: V3d): void { this.guarded(s => ({ ...s, Momentum: s.Momentum.add(d) })); }
  addTargetTurn(t: V2d): void { this.guarded(s => ({ ...s, TargetTurn: s.TargetTurn.add(t) })); }
  setSprintFactor(f: number): void { this.guarded(s => ({ ...s, SprintFactor: f })); }
  adjustMoveSpeed(f: number): void {
    this.guarded(s => ({ ...s, Config: { ...s.Config, MoveSpeed: s.Config.MoveSpeed * f } }));
  }
  updateConfig(cfg: FreeFlyConfig): void { this.guarded(s => ({ ...s, Config: cfg })); }

  /** Manually advance integration by `dt` seconds. Used by tests; production drives via `attach()`. */
  tick(dt: number): void {
    this.update(s => step(s, dt));
  }

  private update(f: (s: FreeFlyState) => FreeFlyState): void {
    transact(() => { this.state.value = f(this.state.value); });
  }

  private guarded(f: (s: FreeFlyState) => FreeFlyState): void {
    this.update(s => s.Enabled ? f(s) : s);
  }

  // ---------- DOM wiring ----------

  attach(target: HTMLElement, time: aval<number>, opts: FreeFlyAttachOptions = {}): () => void {
    const cfg = (): FreeFlyConfig => this.state.value.Config;

    // ---- Mouse + wheel ----
    let rotDown = false, panDown = false, zoomDown = false;
    let panStartDepth: number | undefined;
    // Track last clientX/Y per mouse pointer so we can compute deltas
    // when `movementX` is unavailable (e.g. synthetic happy-dom events).
    const mouseLast = new Map<number, { x: number; y: number }>();

    const onMouseDown = (e: MouseEvent | PointerEvent): void => {
      // Skip non-mouse pointer events here; touch path handles them.
      if ("pointerType" in e && e.pointerType !== "mouse") return;
      mouseLast.set("pointerId" in e ? e.pointerId : 0, { x: e.clientX, y: e.clientY });
      if (e.button === 0) rotDown = true;
      else if (e.button === 1) {
        panStartDepth = opts.pickDepth?.(e.clientX, e.clientY);
        panDown = true;
      } else if (e.button === 2) zoomDown = true;
      (target as Element).setPointerCapture?.(("pointerId" in e ? e.pointerId : 0) as number);
      e.preventDefault();
    };
    const onMouseUp = (e: MouseEvent | PointerEvent): void => {
      if ("pointerType" in e && e.pointerType !== "mouse") return;
      if (e.button === 0) rotDown = false;
      else if (e.button === 1) { panDown = false; panStartDepth = undefined; }
      else if (e.button === 2) zoomDown = false;
      mouseLast.delete("pointerId" in e ? e.pointerId : 0);
    };
    const onMouseMove = (e: MouseEvent | PointerEvent): void => {
      if ("pointerType" in e && e.pointerType !== "mouse") return;
      const id = "pointerId" in e ? e.pointerId : 0;
      const prev = mouseLast.get(id);
      const dx = (e as PointerEvent).movementX ?? (prev ? e.clientX - prev.x : 0);
      const dy = (e as PointerEvent).movementY ?? (prev ? e.clientY - prev.y : 0);
      mouseLast.set(id, { x: e.clientX, y: e.clientY });
      const c = cfg();
      if (rotDown) {
        this.addTargetTurn(new V2d(-0.007 * c.TurnSensitivity * dx, -0.007 * c.TurnSensitivity * dy));
      }
      if (panDown) {
        const speed = panStartDepth !== undefined
          ? 0.005 * panStartDepth * c.PanSensitivity
          : 0.025 * c.PanSensitivity;
        this.addTargetMove(new V3d(speed * dx, -speed * dy, 0));
      }
      if (zoomDown) {
        this.addTargetMove(new V3d(0, 0, -0.05 * dy));
      }
    };
    const onWheel = (e: WheelEvent): void => {
      const c = cfg();
      this.addMomentum(new V3d(0, 0, -0.1 * c.WheelSensitivity * e.deltaY));
      e.preventDefault();
    };
    const onContextMenu = (e: Event): void => e.preventDefault();

    // ---- Keys ----
    let shiftDown = false;
    const onKeyDown = (e: KeyboardEvent): void => {
      if (e.shiftKey !== shiftDown) {
        this.setPanMove(e.shiftKey);
        shiftDown = e.shiftKey;
      }
      const k = e.key;
      if (k === "Shift") this.setPanMove(true);
      else if (k === "w" || k === "W") this.setMoveVec("W", new V3d(0, 0, 1));
      else if (k === "s" || k === "S") this.setMoveVec("S", new V3d(0, 0, -1));
      else if (k === "a" || k === "A") this.setMoveVec("A", new V3d(-1, 0, 0));
      else if (k === "d" || k === "D") this.setMoveVec("D", new V3d(1, 0, 0));
      else if (!e.repeat) {
        if (k === "ArrowUp" && e.shiftKey) this.adjustMoveSpeed(1.5);
        else if (k === "ArrowDown" && e.shiftKey) this.adjustMoveSpeed(1 / 1.5);
        else if (k === "PageUp") this.adjustMoveSpeed(1.5);
        else if (k === "PageDown") this.adjustMoveSpeed(1 / 1.5);
      }
    };
    const onKeyUp = (e: KeyboardEvent): void => {
      const k = e.key;
      if (k === "Shift") { this.setPanMove(false); shiftDown = false; }
      else if (k === "w" || k === "W") this.setMoveVec("W", V3d.zero);
      else if (k === "s" || k === "S") this.setMoveVec("S", V3d.zero);
      else if (k === "a" || k === "A") this.setMoveVec("A", V3d.zero);
      else if (k === "d" || k === "D") this.setMoveVec("D", V3d.zero);
    };
    const onBlur = (): void => {
      this.setMoveVec("W", V3d.zero);
      this.setMoveVec("S", V3d.zero);
      this.setMoveVec("A", V3d.zero);
      this.setMoveVec("D", V3d.zero);
      this.setPanMove(false);
      shiftDown = false;
    };

    // ---- Touch (multi-pointer) ----
    interface TouchPt { x: number; y: number }
    const touches = new Map<number, TouchPt>();
    let lastCentroid: { x: number; y: number } | undefined;
    let lastPinch: number | undefined;

    const recomputeRefs = (): void => {
      if (touches.size === 0) { lastCentroid = undefined; lastPinch = undefined; return; }
      let cx = 0, cy = 0;
      for (const p of touches.values()) { cx += p.x; cy += p.y; }
      cx /= touches.size; cy /= touches.size;
      lastCentroid = { x: cx, y: cy };
      if (touches.size >= 2) {
        const arr = Array.from(touches.values());
        const a = arr[0]!, b = arr[1]!;
        lastPinch = Math.hypot(a.x - b.x, a.y - b.y);
      } else {
        lastPinch = undefined;
      }
    };

    const onTouchDown = (e: PointerEvent): void => {
      if (e.pointerType === "mouse") return;
      touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
      recomputeRefs();
      (target as Element).setPointerCapture?.(e.pointerId);
      e.preventDefault();
    };
    const onTouchMove = (e: PointerEvent): void => {
      if (e.pointerType === "mouse") return;
      const prev = touches.get(e.pointerId);
      if (!prev) return;
      touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
      const c = cfg();
      const n = touches.size;
      if (n === 1) {
        const dx = e.clientX - prev.x;
        const dy = e.clientY - prev.y;
        this.addTargetTurn(new V2d(-0.007 * c.TurnSensitivity * dx, -0.007 * c.TurnSensitivity * dy));
        lastCentroid = { x: e.clientX, y: e.clientY };
      } else if (n === 2) {
        // Two-finger: pan via centroid delta + pinch zoom on Z.
        let cx = 0, cy = 0;
        for (const p of touches.values()) { cx += p.x; cy += p.y; }
        cx /= n; cy /= n;
        if (lastCentroid) {
          const dCx = cx - lastCentroid.x;
          const dCy = cy - lastCentroid.y;
          const speed = 0.025 * c.PanSensitivity;
          this.addTargetMove(new V3d(speed * dCx, -speed * dCy, 0));
        }
        const arr = Array.from(touches.values());
        const a = arr[0]!, b = arr[1]!;
        const pinch = Math.hypot(a.x - b.x, a.y - b.y);
        if (lastPinch !== undefined) {
          const dPinch = pinch - lastPinch;
          this.addTargetMove(new V3d(0, 0, dPinch * 0.01 * c.WheelSensitivity));
        }
        lastCentroid = { x: cx, y: cy };
        lastPinch = pinch;
      } else if (n >= 3) {
        // Three-finger drag → forward/back along Z.
        let cx = 0, cy = 0;
        for (const p of touches.values()) { cx += p.x; cy += p.y; }
        cx /= n; cy /= n;
        if (lastCentroid) {
          const dCy = cy - lastCentroid.y;
          this.addTargetMove(new V3d(0, 0, -dCy * 0.025 * c.PanSensitivity));
        }
        lastCentroid = { x: cx, y: cy };
      }
    };
    const onTouchUp = (e: PointerEvent): void => {
      if (e.pointerType === "mouse") return;
      touches.delete(e.pointerId);
      recomputeRefs();
      (target as Element).releasePointerCapture?.(e.pointerId);
    };

    // ---- Frame integration ----
    let lastT: number | undefined;
    const cb: IDisposable = avalAddCallback(time, (now: number) => {
      if (lastT === undefined) { lastT = now; return; }
      const dt = (now - lastT) / 1000;
      lastT = now;
      if (dt > 0) this.tick(dt);
    });

    // ---- Wire listeners ----
    const onPointerDown = (e: PointerEvent): void => {
      if (e.pointerType === "mouse") onMouseDown(e); else onTouchDown(e);
    };
    const onPointerMove = (e: PointerEvent): void => {
      if (e.pointerType === "mouse") onMouseMove(e); else onTouchMove(e);
    };
    const onPointerUp = (e: PointerEvent): void => {
      if (e.pointerType === "mouse") onMouseUp(e); else onTouchUp(e);
    };
    target.addEventListener("pointerdown", onPointerDown as EventListener);
    target.addEventListener("pointermove", onPointerMove as EventListener);
    target.addEventListener("pointerup", onPointerUp as EventListener);
    target.addEventListener("pointercancel", onPointerUp as EventListener);
    target.addEventListener("wheel", onWheel, { passive: false });
    target.addEventListener("contextmenu", onContextMenu);
    target.addEventListener("blur", onBlur);
    const keyTarget: EventTarget = (opts.keysOnWindow ?? true) ? window : target;
    keyTarget.addEventListener("keydown", onKeyDown as EventListener);
    keyTarget.addEventListener("keyup", onKeyUp as EventListener);

    return () => {
      cb.dispose();
      target.removeEventListener("pointerdown", onPointerDown as EventListener);
      target.removeEventListener("pointermove", onPointerMove as EventListener);
      target.removeEventListener("pointerup", onPointerUp as EventListener);
      target.removeEventListener("pointercancel", onPointerUp as EventListener);
      target.removeEventListener("wheel", onWheel);
      target.removeEventListener("contextmenu", onContextMenu);
      target.removeEventListener("blur", onBlur);
      keyTarget.removeEventListener("keydown", onKeyDown as EventListener);
      keyTarget.removeEventListener("keyup", onKeyUp as EventListener);
    };
  }
}

// Re-export utility for tests.
export { isAnimating as freeFlyIsAnimating };
// Workaround unused-import marker if AVal becomes unreferenced.
void AVal;
