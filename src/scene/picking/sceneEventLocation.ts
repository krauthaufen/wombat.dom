// SceneEventLocation — the rich spatial context for a SceneEvent.
//
// Mirrors Aardvark.Dom's `SceneEventLocation` (F#:
// Aardvark.Dom/SceneGraph/SceneEvent.fs). Carries the eight raw
// inputs (modelTrafo, local2World, viewTrafo, projTrafo, pixel,
// viewportSize, viewPos, viewNormal, partIndex) and computes the
// derived world / model / local positions and pick rays on demand.
//
// Lazy fields are memoised — handlers reading `worldPosition` followed
// by `worldPickRay` should not recompute the matrix product on each
// access. Lazy storage uses private mutable slots (no closure-based
// caches per access).
//
// `transformed(t)` returns a copy with `local2World` pre-multiplied by
// `t`. The dispatcher can use this during capture/bubble to push the
// event into a child scope's local frame (see `dispatcher.ts` —
// currently identity for v1, see docs/FUTURE.md).

import { Ray3d, Trafo3d, V2d, V2i, V3d, V4d } from "@aardworx/wombat.base";
import { AVal, type aval } from "@aardworx/wombat.adaptive";

/** A V3d with `(x,y,z) = projTrafo.forward.transformPos(viewPos)` (NDC), used for `depth` and pick rays. */
function ndcOf(viewPos: V3d, projTrafo: Trafo3d): V3d {
  // projTrafo.forward.transformPos divides by w (homogeneous projective).
  return projTrafo.forward.transformPos(viewPos);
}

function ndc2dOf(pixel: V2d, viewportSize: V2i): V2d {
  const w = viewportSize.x !== 0 ? viewportSize.x : 1;
  const h = viewportSize.y !== 0 ? viewportSize.y : 1;
  const tx = pixel.x / w;
  const ty = pixel.y / h;
  return new V2d(2 * tx - 1, 1 - 2 * ty);
}

export class SceneEventLocation {
  // Raw inputs (ctor params).
  readonly modelTrafo: aval<Trafo3d>;
  readonly local2World: Trafo3d;
  readonly viewTrafo: Trafo3d;
  readonly projTrafo: Trafo3d;
  readonly pixel: V2d;
  readonly viewportSize: V2i;
  readonly viewPos: V3d;
  readonly viewNormal: V3d;
  readonly partIndex: number;

  // Memoisation slots. `undefined` means "not yet computed".
  #ndc: V3d | undefined;
  #ndc2d: V2d | undefined;
  #viewProjTrafo: Trafo3d | undefined;
  #worldPosition: V3d | undefined;
  #worldNormal: V3d | undefined;
  #modelPosition: V3d | undefined;
  #modelNormal: V3d | undefined;
  #position: V3d | undefined;
  #normal: V3d | undefined;
  #viewPickRay: Ray3d | undefined;
  #worldPickRay: Ray3d | undefined;
  #modelPickRay: Ray3d | undefined;
  #pickRay: Ray3d | undefined;

  constructor(
    modelTrafo: aval<Trafo3d>,
    local2World: Trafo3d,
    viewTrafo: Trafo3d,
    projTrafo: Trafo3d,
    pixel: V2d,
    viewportSize: V2i,
    viewPos: V3d,
    viewNormal: V3d,
    partIndex: number,
  ) {
    this.modelTrafo = modelTrafo;
    this.local2World = local2World;
    this.viewTrafo = viewTrafo;
    this.projTrafo = projTrafo;
    this.pixel = pixel;
    this.viewportSize = viewportSize;
    this.viewPos = viewPos;
    this.viewNormal = viewNormal;
    this.partIndex = partIndex;
  }

  // ---- Derived (lazy) -----------------------------------------------------

  /** view * proj — combined view→clip trafo. */
  get viewProjTrafo(): Trafo3d {
    if (this.#viewProjTrafo === undefined) {
      // Aardvark convention: a.mul(b) means "do a first, then b".
      // viewProj = view → proj.
      this.#viewProjTrafo = this.viewTrafo.mul(this.projTrafo);
    }
    return this.#viewProjTrafo;
  }

  /** NDC position of the hit (viewPos transformed by projTrafo, w-divided). */
  private get ndc(): V3d {
    if (this.#ndc === undefined) this.#ndc = ndcOf(this.viewPos, this.projTrafo);
    return this.#ndc;
  }

  /** NDC xy from the device pixel (independent of the hit's depth slot). */
  private get ndc2d(): V2d {
    if (this.#ndc2d === undefined) this.#ndc2d = ndc2dOf(this.pixel, this.viewportSize);
    return this.#ndc2d;
  }

  /** NDC z of the hit. NaN/0-ish for synthetic non-pick locations. */
  get depth(): number { return this.ndc.z; }

  /** Hit point in world space. */
  get worldPosition(): V3d {
    if (this.#worldPosition === undefined) {
      this.#worldPosition = this.viewTrafo.backward.transformPos(this.viewPos);
    }
    return this.#worldPosition;
  }

  /** Hit normal in world space. (`view.forward.transposed.transformDir(viewNormal).normalize`.) */
  get worldNormal(): V3d {
    if (this.#worldNormal === undefined) {
      const n = this.viewTrafo.forward.transpose().transformDir(this.viewNormal);
      const len = Math.hypot(n.x, n.y, n.z);
      this.#worldNormal = len > 0 ? new V3d(n.x / len, n.y / len, n.z / len) : n;
    }
    return this.#worldNormal;
  }

  /** Hit position in the leaf's accumulated model frame. */
  get modelPosition(): V3d {
    if (this.#modelPosition === undefined) {
      const m = AVal.force(this.modelTrafo);
      this.#modelPosition = m.backward.transformPos(this.worldPosition);
    }
    return this.#modelPosition;
  }

  /** Hit normal in model frame. */
  get modelNormal(): V3d {
    if (this.#modelNormal === undefined) {
      const m = AVal.force(this.modelTrafo);
      const n = m.forward.transpose().transformDir(this.worldNormal);
      const len = Math.hypot(n.x, n.y, n.z);
      this.#modelNormal = len > 0 ? new V3d(n.x / len, n.y / len, n.z / len) : n;
    }
    return this.#modelNormal;
  }

  /** Hit position in the handler's local frame (`local2World.backward(worldPosition)`). */
  get position(): V3d {
    if (this.#position === undefined) {
      this.#position = this.local2World.backward.transformPos(this.worldPosition);
    }
    return this.#position;
  }

  /** Hit normal in the handler's local frame. */
  get normal(): V3d {
    if (this.#normal === undefined) {
      const n = this.local2World.forward.transpose().transformDir(this.worldNormal);
      const len = Math.hypot(n.x, n.y, n.z);
      this.#normal = len > 0 ? new V3d(n.x / len, n.y / len, n.z / len) : n;
    }
    return this.#normal;
  }

  /** Cursor pick ray expressed in view space. */
  get viewPickRay(): Ray3d {
    if (this.#viewPickRay === undefined) {
      const n = this.ndc2d;
      const near = this.projTrafo.backward.transformPos(new V3d(n.x, n.y, -1));
      const far  = this.projTrafo.backward.transformPos(new V3d(n.x, n.y,  1));
      this.#viewPickRay = Ray3d.fromPoints(near, far);
    }
    return this.#viewPickRay;
  }

  /** Cursor pick ray expressed in world space. */
  get worldPickRay(): Ray3d {
    if (this.#worldPickRay === undefined) {
      const vp = this.viewProjTrafo;
      const n = this.ndc2d;
      const near = vp.backward.transformPos(new V3d(n.x, n.y, -1));
      const far  = vp.backward.transformPos(new V3d(n.x, n.y,  1));
      this.#worldPickRay = Ray3d.fromPoints(near, far);
    }
    return this.#worldPickRay;
  }

  /** Cursor pick ray expressed in the leaf's model frame. */
  get modelPickRay(): Ray3d {
    if (this.#modelPickRay === undefined) {
      const m = AVal.force(this.modelTrafo);
      // model→world→clip combined: mvp = model · view · proj.
      const mvp = m.mul(this.viewProjTrafo);
      const n = this.ndc2d;
      const near = mvp.backward.transformPos(new V3d(n.x, n.y, -1));
      const far  = mvp.backward.transformPos(new V3d(n.x, n.y,  1));
      this.#modelPickRay = Ray3d.fromPoints(near, far);
    }
    return this.#modelPickRay;
  }

  /** Cursor pick ray expressed in the handler's local frame (worldPickRay through local2World⁻¹). */
  get pickRay(): Ray3d {
    if (this.#pickRay === undefined) {
      const vp = this.viewProjTrafo;
      const n = this.ndc2d;
      const wNear = vp.backward.transformPos(new V3d(n.x, n.y, -1));
      const wFar  = vp.backward.transformPos(new V3d(n.x, n.y,  1));
      const near = this.local2World.backward.transformPos(wNear);
      const far  = this.local2World.backward.transformPos(wFar);
      this.#pickRay = Ray3d.fromPoints(near, far);
    }
    return this.#pickRay;
  }

  /**
   * Returns a copy with `local2World` pre-multiplied by `trafo` —
   * use during capture/bubble to push the event into a child scope's
   * local frame. F# parity: `trafo * local2World` (do trafo first,
   * then existing local2World).
   */
  transformed(trafo: Trafo3d): SceneEventLocation {
    return new SceneEventLocation(
      this.modelTrafo,
      trafo.mul(this.local2World),
      this.viewTrafo,
      this.projTrafo,
      this.pixel,
      this.viewportSize,
      this.viewPos,
      this.viewNormal,
      this.partIndex,
    );
  }
}

// Suppress unused-import warnings for V4d (re-exported by callers via
// the dispatcher; kept in case future ray transforms need it inline).
void V4d;
