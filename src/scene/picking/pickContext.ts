// Pick-context interfaces for offscreen ("portal") picking.
//
// Mirrors Aardvark.Dom's `IPickSubContext` / `IRenderPickContext`
// (RenderTo.fs): the minimal recursion handle an offscreen render
// exposes so a HOST scene can forward picks into it. The host
// composites the offscreen color texture onto arbitrary geometry and
// mounts the context on that node via the `PickContext` scene
// attribute; the portal pick-final records the sampled uv per pixel,
// and the resolver recurses `pickAt` into the inner scene.
//
// Declared in its own module (not renderTo.ts) so `traversalState.ts`
// can reference the type without importing the compile pipeline —
// same compile-order motivation as the F# original.

import type { aval } from "@aardworx/wombat.adaptive";
import type { Trafo3d } from "@aardworx/wombat.base";

import type { PickRegistry } from "./registry.js";
import type { ResolvedHit } from "./spiralHitTest.js";

/**
 * A fully-resolved pick inside (possibly nested) offscreen scenes.
 * `hit` is the INNERMOST winner; its scope carries the view / proj /
 * model of the scene it lives in (captured at lowering), so event
 * locations built from it are automatically in the inner frame.
 */
export interface PortalPickHit {
  /** The innermost resolved hit (scope + view-space pos/normal/part). */
  readonly hit: ResolvedHit;
  /** The registry the hit's pickId belongs to. Ids are per-producer —
   *  never decode them against any other registry. */
  readonly registry: PickRegistry;
}

/**
 * Minimal recursion handle: what the OUTER resolver needs to forward
 * a pick into an offscreen scene. `size` maps the portal uv to inner
 * pixels; `pickAt` resolves (recursively — the inner scene may itself
 * contain portals).
 */
export interface IPickSubContext {
  /** Inner render-target size (device pixels). */
  readonly size: aval<{ readonly width: number; readonly height: number }>;
  /**
   * Resolve the pick at inner device coords. Asynchronous by nature on
   * WebGPU (argmin readback is `mapAsync`); one await per nesting
   * level. `undefined` = inner miss — the caller falls through to the
   * portal geometry itself.
   */
  pickAt(x: number, y: number): Promise<PortalPickHit | undefined>;
}

/**
 * The pick half of a pickable offscreen render (`renderToPickable`).
 * Extends the recursion handle with the live camera avals and the
 * producer's registry, for hosts that want to drive focus / queries
 * against the inner scene directly.
 */
export interface IRenderPickContext extends IPickSubContext {
  readonly view: aval<Trafo3d>;
  readonly proj: aval<Trafo3d>;
  readonly registry: PickRegistry;
}
