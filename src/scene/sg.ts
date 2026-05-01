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
//   - Active:      AND across nesting (any false ⇒ inactive)

import type {
  alist, aset, aval, HashMap,
} from "@aardworx/wombat.adaptive";
import type { Trafo3d } from "@aardworx/wombat.base";
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
  | SgOn
  | SgActive;

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
 * Event-handler scope. Handlers are appended to the chain at this
 * scope; outer + inner handlers all fire during dispatch.
 *
 * `SceneEvent` proper lands with the pick infra (M7-M8); for now
 * `events` is a record of opaque handlers keyed by event-kind name.
 */
export type EventHandlers = Readonly<Record<string, (e: unknown) => void>>;

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
