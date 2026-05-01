// SgNode — the immutable scene-graph data model.
//
// A scene is a tree of `SgNode`s. Each variant either:
//   - applies an attribute to its sub-tree (Trafo/Shader/Uniform/...),
//   - composes sub-trees (Group/Unordered/Adaptive),
//   - or is a leaf (Leaf, which produces draw calls).
//
// Nodes are plain data — pure tagged-union records, never mutated.
// Constructors live in the `Sg` namespace at the bottom of this file
// (and in `attributes.ts` for the trafo helpers).
//
// Composition semantics — applied by the traversal in `visit.ts`:
//   - Trafo:       composes (matrix multiply, see traversalState.ts)
//   - Uniform:     composes (per-key map merge, inner-wins)
//   - On / events: composes (chain — outer + inner both fire)
//   - Shader:      overrides (innermost wins)
//   - BlendMode:   overrides
//   - Cursor:      overrides
//   - PickThrough: overrides
//   - Intersectable: overrides — innermost wins. Carries the geometry
//       used for BVH ray fall-through when a pixel hit lands on a
//       pickThrough scope (mirrors Aardvark.Dom's
//       `Aardvark.Dom/SceneGraph/TraversalState.fs`
//       `Intersectable of aval<IIntersectable>`).
//   - PixelSnapRadius: overrides — innermost wins
//       Why: matches Aardvark.Dom's `PixelSnapRadius` attribute,
//       which is a `Set` (override) on TraversalState, not an
//       AND/AVG composition. Pickers walking inside-out shouldn't
//       see a tighter outer radius mask a more permissive inner.
//   - Active:      AND across nesting (any false ⇒ inactive)

import type {
  alist, aset, aval, HashMap,
} from "@aardworx/wombat.adaptive";
import type { IIntersectable, Trafo3d } from "@aardworx/wombat.base";
// Type-only imports — wombat.shader / wombat.rendering are optional
// peer-deps. Importers of /scene must have them installed.
import type { Effect } from "@aardworx/wombat.shader";
import type {
  BufferView, BlendState, DrawCall,
} from "@aardworx/wombat.rendering/core";

// ---------------------------------------------------------------------------
// SgNode union
// ---------------------------------------------------------------------------

export type SgNode =
  | SgEmpty
  | SgGroup
  | SgUnorderedGroup
  | SgAdaptiveGroup
  | SgLeaf
  | SgTrafo
  | SgShader
  | SgUniform
  | SgBlendMode
  | SgCursor
  | SgPickThrough
  | SgIntersectable
  | SgPixelSnapRadius
  | SgOn
  | SgActive
  | SgView
  | SgProj
  | SgDelay;

export interface SgEmpty {
  readonly kind: "Empty";
}

/** Ordered children — `Ordered` / `OrderedFromList` walker downstream. */
export interface SgGroup {
  readonly kind: "Group";
  readonly children: alist<SgNode>;
}

/** Unordered (state-sortable) children. */
export interface SgUnorderedGroup {
  readonly kind: "UnorderedGroup";
  readonly children: aset<SgNode>;
}

/** Single-slot subtree swap on aval change. */
export interface SgAdaptiveGroup {
  readonly kind: "AdaptiveGroup";
  readonly child: aval<SgNode>;
}

/**
 * A renderable leaf: just enough to drive a `RenderObject`. Effect,
 * uniforms, transforms, render state all flow in through the
 * `TraversalState` accumulated by parent attribute scopes.
 *
 * Vertex / instance attributes and indices ARE leaf-intrinsic
 * because they describe *this* draw — they don't logically scope
 * down to siblings. (Aardvark.Dom's `DirectDrawNode` shape.)
 */
export interface SgLeaf {
  readonly kind: "Leaf";
  readonly vertexAttributes: HashMap<string, aval<BufferView>>;
  readonly instanceAttributes?: HashMap<string, aval<BufferView>>;
  readonly indices?: aval<BufferView>;
  readonly drawCall: aval<DrawCall>;
}

/**
 * Trafo scope. The value is a `Trafo3d`, an aval of one, or an
 * **ordered array** of either — leftmost is outermost (applied
 * last to a point). See `traversalState.ts` for the multiply rule.
 */
export type TrafoValue =
  | Trafo3d
  | aval<Trafo3d>
  | ReadonlyArray<Trafo3d | aval<Trafo3d>>;

export interface SgTrafo {
  readonly kind: "Trafo";
  readonly value: TrafoValue;
  readonly child: SgNode;
}

export interface SgShader {
  readonly kind: "Shader";
  readonly effect: Effect;
  readonly child: SgNode;
}

/**
 * Uniform values flowing into descendants. Either a static record
 * (each value can itself be `aval<T>`) or an `amap` for fully
 * incremental key-set changes.
 *
 * Composition across nested Uniform scopes: per-key map merge,
 * inner-wins on key conflict.
 */
export type UniformBag =
  | { readonly kind: "Static";  readonly entries: HashMap<string, aval<unknown>> }
  | { readonly kind: "Dynamic"; readonly entries: import("@aardworx/wombat.adaptive").amap<string, aval<unknown>> };

export interface SgUniform {
  readonly kind: "Uniform";
  readonly bag: UniformBag;
  readonly child: SgNode;
}

export interface SgBlendMode {
  readonly kind: "BlendMode";
  readonly mode: BlendState;
  readonly child: SgNode;
}

export interface SgCursor {
  readonly kind: "Cursor";
  readonly cursor: string | aval<string>;
  readonly child: SgNode;
}

export interface SgPickThrough {
  readonly kind: "PickThrough";
  readonly value: boolean;
  readonly child: SgNode;
}

/**
 * Intersectable scope — attaches a per-scope `IIntersectable` (a box,
 * sphere, mesh, transformed wrapper, ...) used by the dispatcher to
 * build a BVH and ray-fall-through past `pickThrough` scopes when the
 * pixel hit-test landed on a "transparent" hit. Override semantics:
 * the innermost `<Sg Intersectable=...>` scope wins.
 */
export interface SgIntersectable {
  readonly kind: "Intersectable";
  readonly intersectable: aval<IIntersectable>;
  readonly child: SgNode;
}

/**
 * Pixel-snap-radius scope. Sets the radius (in device pixels) of
 * the disc the pick dispatcher searches around the cursor. Inner
 * scope wins on conflict. Hard-capped at `SNAP_RADIUS_MAX = 16`
 * inside the dispatcher (matches Aardvark.Dom's `PickSnap.radius`).
 */
export interface SgPixelSnapRadius {
  readonly kind: "PixelSnapRadius";
  readonly radius: aval<number>;
  readonly child: SgNode;
}

/**
 * Event-handler scope. Handlers are appended to the chain at this
 * scope; outer + inner handlers all participate in capture/bubble
 * dispatch (see `picking/dispatcher.ts`).
 *
 * Handlers may return `false` to stop propagation (matches F#'s
 * `List.forall`). Anything else (including `void`) lets dispatch
 * continue. `SceneEvent.stopPropagation()` is the explicit form.
 */
export type SceneEventHandler = (e: import("./picking/sceneEvent.js").SceneEvent) => boolean | void;

/**
 * Per-scope handler bag, keyed by phase. Each phase is a partial
 * map kind → handler. Why two phases instead of a flat record:
 * mirrors Aardvark.Dom's `TraversalState.handleEvent` capture/bubble
 * walk — outer-first capture, inner-first bubble.
 */
export interface EventHandlers {
  readonly capture?: Partial<Record<import("./picking/sceneEvent.js").SceneEventKind, SceneEventHandler>>;
  readonly bubble?: Partial<Record<import("./picking/sceneEvent.js").SceneEventKind, SceneEventHandler>>;
}

export interface SgOn {
  readonly kind: "On";
  readonly handlers: EventHandlers;
  readonly child: SgNode;
}

/** Toggle the entire sub-tree on/off. AND-composed across nesting. */
export interface SgActive {
  readonly kind: "Active";
  readonly active: aval<boolean>;
  readonly child: SgNode;
}

/**
 * View-trafo scope (world → view). Innermost wins. The
 * `<RenderControl>` component sniffs the outermost View/Proj
 * scopes off the scene root for its picking infrastructure (see
 * Aardvark.Dom's `RenderControlBuilderState.Build`).
 */
export interface SgView {
  readonly kind: "View";
  readonly view: aval<Trafo3d>;
  readonly child: SgNode;
}

/** Projection-trafo scope (view → clip). Innermost wins. */
export interface SgProj {
  readonly kind: "Proj";
  readonly proj: aval<Trafo3d>;
  readonly child: SgNode;
}

/**
 * Escape hatch — produce a sub-tree from the **fully accumulated**
 * `TraversalState` at this point in the walk. Used for things that
 * need the live camera / viewport to build their geometry:
 * screen-space-sized gizmos, view-aligned billboards, scene
 * adapters keyed on model trafo, etc. Mirrors Aardvark.Dom's
 * `Sg.Delay : TraversalState -> ISceneNode`.
 *
 * Caveats:
 *   - The creator function runs **once per traversal**, not per
 *     aval delta. The returned `SgNode` should embed its own
 *     reactive plumbing (e.g. `Sg.adaptive`) for runtime updates.
 *   - Avoid expensive computation inside; cache externally if
 *     needed.
 *   - Forwards declared imports — keep this comment block: the
 *     `TraversalState` ref must come from a circular type that
 *     `traversalState.ts` defines, so we type the param via an
 *     interface alias to break the cycle.
 */
export interface SgDelay {
  readonly kind: "Delay";
  readonly create: (state: import("./traversalState.js").TraversalState) => SgNode;
}
