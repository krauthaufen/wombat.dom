// Camera helpers — view + projection trafo builders that take
// reactive parameters (`number | aval<number>`, `V3d | aval<V3d>`)
// and produce `aval<Trafo3d>` ready to feed into `<Sg View=…>`,
// `<Sg Proj=…>`, or `<RenderControl view proj>`.
//
// Conventions:
//   - Right-handed coordinate system. Camera looks down `-Z` in
//     view space (same as wombat.base's `viewTrafoRH`).
//   - WebGPU NDC: z ∈ [0, 1]. We use the `RH` variants throughout
//     — never the `GL` ones (which produce z ∈ [-1, 1]).
//   - FOV is horizontal in radians (matches Aardvark.Base /
//     wombat.base). Convert from vertical with
//     `2 * atan(tan(vfov/2) * aspect)`.

import { AVal, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d } from "@aardworx/wombat.base";

// ---------------------------------------------------------------------------
// Camera record
// ---------------------------------------------------------------------------

export interface Camera {
  readonly view: aval<Trafo3d>;
  readonly proj: aval<Trafo3d>;
}

// ---------------------------------------------------------------------------
// Helpers — accept plain or aval, lift into avals
// ---------------------------------------------------------------------------

const isAval = <T>(v: T | aval<T>): v is aval<T> =>
  typeof v === "object" && v !== null && typeof (v as { getValue?: unknown }).getValue === "function";

const liftA = <T>(v: T | aval<T>): aval<T> => (isAval(v) ? v : AVal.constant(v));

// ---------------------------------------------------------------------------
// View — lookAt
// ---------------------------------------------------------------------------

export interface LookAtOptions {
  readonly eye: V3d | aval<V3d>;
  readonly target: V3d | aval<V3d>;
  readonly up: V3d | aval<V3d>;
}

/**
 * Build a view trafo from `{ eye, target, up }`. The forward
 * direction is `target - eye` (normalised). All inputs may be
 * plain or `aval`.
 */
export function lookAt(opts: LookAtOptions): aval<Trafo3d> {
  return AVal.zip(liftA(opts.eye), liftA(opts.target), liftA(opts.up)).map((eye, target, up) => {
    const forward = target.sub(eye).normalize();
    return Trafo3d.viewTrafoRH(eye, up, forward);
  });
}

// ---------------------------------------------------------------------------
// Projection — perspective
// ---------------------------------------------------------------------------

export interface PerspectiveOptions {
  /** Horizontal field of view in radians. */
  readonly fovInRadians: number | aval<number>;
  /** Width / height of the viewport. */
  readonly aspect: number | aval<number>;
  /** Near plane distance (positive). */
  readonly near: number | aval<number>;
  /** Far plane distance (positive). `Infinity` is supported. */
  readonly far?: number | aval<number>;
}

/**
 * Right-handed perspective projection for WebGPU NDC (z in `[0, 1]`).
 *
 * `fovInRadians` is the horizontal FOV. Convert from vertical via
 * `hfov = 2 * atan(tan(vfov / 2) * aspect)`.
 */
export function perspective(opts: PerspectiveOptions): aval<Trafo3d> {
  const far = opts.far ?? Infinity;
  return AVal.zip(
    liftA(opts.fovInRadians),
    liftA(opts.aspect),
    liftA(opts.near),
    liftA(far),
  ).map((fov, aspect, n, f) =>
    Trafo3d.perspectiveProjectionRHFov(fov, aspect, n, f),
  );
}

// ---------------------------------------------------------------------------
// Projection — orthographic
// ---------------------------------------------------------------------------

export interface OrthographicOptions {
  readonly left:   number | aval<number>;
  readonly right:  number | aval<number>;
  readonly bottom: number | aval<number>;
  readonly top:    number | aval<number>;
  readonly near:   number | aval<number>;
  readonly far:    number | aval<number>;
}

/** Right-handed orthographic projection for WebGPU NDC (z in `[0, 1]`). */
export function orthographic(opts: OrthographicOptions): aval<Trafo3d> {
  return AVal.zip(
    liftA(opts.left), liftA(opts.right),
    liftA(opts.bottom), liftA(opts.top),
    liftA(opts.near), liftA(opts.far),
  ).map((l, r, b, t, n, f) => Trafo3d.orthoProjectionRH(l, r, b, t, n, f));
}

// ---------------------------------------------------------------------------
// Convenience — perspective camera with a viewport-driven aspect
// ---------------------------------------------------------------------------

export interface PerspectiveCameraOptions {
  readonly view: aval<Trafo3d>;
  readonly viewport: aval<{ width: number; height: number }>;
  readonly fovInRadians: number | aval<number>;
  readonly near: number | aval<number>;
  readonly far?: number | aval<number>;
}

/**
 * Common perspective-camera setup: the projection auto-tracks the
 * viewport's aspect ratio. Plug `<RenderControl>`'s viewport aval
 * (from `onReady.viewport` or via `useViewport()`) straight in.
 */
export function perspectiveCamera(opts: PerspectiveCameraOptions): Camera {
  const aspect = opts.viewport.map(v => v.width / Math.max(1, v.height));
  const proj = perspective({
    fovInRadians: opts.fovInRadians,
    aspect,
    near: opts.near,
    ...(opts.far !== undefined ? { far: opts.far } : {}),
  });
  return { view: opts.view, proj };
}
