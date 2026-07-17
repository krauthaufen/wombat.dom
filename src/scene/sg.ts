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
  BufferView, BlendState, DrawCall, IBuffer,
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
  | SgPickPriority
  | SgPickTag
  | SgOn
  | SgActive
  | SgView
  | SgProj
  | SgDelay
  // Phase 1 — render-state scopes (override semantics)
  | SgDepthTest
  | SgDepthMask
  | SgDepthBias
  | SgDepthClamp
  | SgCullMode
  | SgFrontFace
  | SgFillMode
  | SgBlendConstant
  | SgColorMask
  | SgStencilMode
  | SgPass
  // Phase 2 — geometry-attribute scopes
  | SgVertexAttributes
  | SgInstanceAttributes
  | SgIndex
  | SgMode
  // Phase 3 — misc scopes
  | SgPickContext
  | SgNoEvents
  | SgForcePixelPicking
  | SgCanFocus
  // Auto-instancing scope (see docs/auto-instancing.md).
  | SgInstanced;

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
  /**
   * Vertex-attribute set: the key set is fixed structurally; only the
   * per-attribute BufferView avals are reactive. The compile path
   * threads this through to `RenderObject.vertexAttributes` without
   * forcing.
   */
  readonly vertexAttributes: HashMap<string, BufferView>;
  readonly instanceAttributes?: HashMap<string, BufferView>;
  readonly indices?: BufferView | undefined;
  readonly drawCall: aval<DrawCall>;
  /** Storage buffers bound to the shader (read-only or read-write).
   *  Keyed by name, matched against the shader's declared storage
   *  buffer bindings. */
  readonly storageBuffers?: HashMap<string, aval<IBuffer>>;
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
 * App-chosen pick tag scope — an OPAQUE value (row/item key) carried
 * onto the leaf's pick scope and surfaced as `SceneEvent.pickTag`.
 * Innermost wins. Enables collection-level handlers: one `On` scope
 * above a group resolves WHICH item was hit via the tag instead of a
 * closure per item.
 */
export interface SgPickTag {
  readonly kind: "PickTag";
  readonly value: unknown;
  readonly child: SgNode;
}

export interface SgPickPriority {
  readonly kind: "PickPriority";
  readonly value: aval<number>;
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

// ---------------------------------------------------------------------------
// Phase 1 — render-state scopes
// ---------------------------------------------------------------------------

/** Depth-compare function. Mirrors WebGPU `GPUCompareFunction`. */
export type DepthCompare =
  | "never" | "less" | "equal" | "less-equal"
  | "greater" | "not-equal" | "greater-equal" | "always";

export interface SgDepthTest {
  readonly kind: "DepthTest";
  /** Either an aval of a fixed `DepthCompare`, or a derived-mode rule
   *  evaluating depthCompare as a function of per-RO uniforms. */
  readonly mode:
    | aval<DepthCompare>
    | import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"depthCompare">;
  readonly child: SgNode;
}

export interface SgDepthMask {
  readonly kind: "DepthMask";
  /** Either an aval of a fixed boolean, or a derived-mode rule
   *  evaluating depthWrite as a function of per-RO uniforms. */
  readonly write:
    | aval<boolean>
    | import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"depthWrite">;
  readonly child: SgNode;
}

export interface DepthBiasValue {
  readonly constant: number;
  readonly slopeScale: number;
  readonly clamp: number;
}

export interface SgDepthBias {
  readonly kind: "DepthBias";
  readonly bias: aval<DepthBiasValue>;
  readonly child: SgNode;
}

export interface SgDepthClamp {
  readonly kind: "DepthClamp";
  readonly clamp: aval<boolean>;
  readonly child: SgNode;
}

export type CullValue = "none" | "front" | "back";

export interface SgCullMode {
  readonly kind: "CullMode";
  /**
   * Either an aval of a fixed `CullValue` (the existing path) or a
   * `DerivedModeRule<"cull">` that evaluates the cull mode as a
   * function of per-RO uniforms (see `derivedMode(...)` and
   * `gpuFlipCullByDeterminant(...)` in `@aardworx/wombat.rendering/runtime`).
   * The rule overrides the PipelineState's `cullMode` at descriptor-
   * snapshot time on a per-RO basis.
   */
  readonly mode: aval<CullValue> | import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"cull">;
  readonly child: SgNode;
}

export type FrontFaceValue = "ccw" | "cw";

export interface SgFrontFace {
  readonly kind: "FrontFace";
  /** Either an aval of a fixed `FrontFaceValue`, or a derived-mode rule
   *  evaluating frontFace as a function of per-RO uniforms. */
  readonly mode:
    | aval<FrontFaceValue>
    | import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"frontFace">;
  readonly child: SgNode;
}

export type FillModeValue = "fill" | "line" | "point";

export interface SgFillMode {
  readonly kind: "FillMode";
  readonly mode: aval<FillModeValue>;
  readonly child: SgNode;
}

export interface BlendConstantValue {
  readonly r: number;
  readonly g: number;
  readonly b: number;
  readonly a: number;
}

export interface SgBlendConstant {
  readonly kind: "BlendConstant";
  readonly value: aval<BlendConstantValue>;
  readonly child: SgNode;
}

/** Per-channel write mask. */
export interface ColorMaskValue {
  readonly r: boolean;
  readonly g: boolean;
  readonly b: boolean;
  readonly a: boolean;
}

export interface SgColorMask {
  readonly kind: "ColorMask";
  /** Per-attachment-name; default-key "color" matches the single-attachment path. */
  readonly mask: aval<HashMap<string, ColorMaskValue>>;
  readonly child: SgNode;
}

export type StencilOpValue =
  | "keep" | "zero" | "replace" | "invert"
  | "increment-clamp" | "decrement-clamp"
  | "increment-wrap" | "decrement-wrap";

export interface StencilFace {
  readonly compare: DepthCompare;
  readonly fail: StencilOpValue;
  readonly depthFail: StencilOpValue;
  readonly pass: StencilOpValue;
}

export interface StencilModeValue {
  readonly enabled: boolean;
  readonly reference: number;
  readonly readMask: number;
  readonly writeMask: number;
  readonly front: StencilFace;
  readonly back: StencilFace;
}

export interface SgStencilMode {
  readonly kind: "StencilMode";
  readonly mode: aval<StencilModeValue>;
  readonly child: SgNode;
}

/**
 * Render-pass ordinal. Lower passes are drawn before higher passes;
 * within a pass, scene-graph order is preserved.
 */
export const RenderPass = {
  main: 0,
  transparent: 1000,
  overlay: 2000,
} as const;

export interface SgPass {
  readonly kind: "Pass";
  readonly pass: number;
  readonly child: SgNode;
}

// ---------------------------------------------------------------------------
// Phase 2 — geometry-attribute scopes
// ---------------------------------------------------------------------------

export interface SgVertexAttributes {
  readonly kind: "VertexAttributes";
  readonly attributes: HashMap<string, BufferView>;
  readonly child: SgNode;
}

export interface SgInstanceAttributes {
  readonly kind: "InstanceAttributes";
  readonly attributes: HashMap<string, BufferView>;
  readonly child: SgNode;
}

export interface SgIndex {
  readonly kind: "Index";
  readonly index: BufferView | undefined;
  readonly child: SgNode;
}

export type ModeValue =
  | "triangle-list" | "triangle-strip"
  | "line-list" | "line-strip"
  | "point-list";

export interface SgMode {
  readonly kind: "Mode";
  /** Either an aval of a fixed `ModeValue` (topology), or a derived-mode
   *  rule evaluating topology as a function of per-RO uniforms. */
  readonly mode:
    | aval<ModeValue>
    | import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"topology">;
  readonly child: SgNode;
}

// ---------------------------------------------------------------------------
// Phase 3 — misc scopes
// ---------------------------------------------------------------------------

export interface SgPickContext {
  readonly kind: "PickContext";
  /** The offscreen render's pick recursion handle
   *  (`renderToPickable(...).pick`). Plain value — the handle is a
   *  stable object for the offscreen render's lifetime. */
  readonly value: import("./picking/pickContext.js").IPickSubContext;
  readonly child: SgNode;
}

export interface SgNoEvents {
  readonly kind: "NoEvents";
  readonly value: aval<boolean>;
  readonly child: SgNode;
}

export interface SgForcePixelPicking {
  readonly kind: "ForcePixelPicking";
  readonly value: aval<boolean>;
  readonly child: SgNode;
}

export interface SgCanFocus {
  readonly kind: "CanFocus";
  readonly value: aval<boolean>;
  readonly child: SgNode;
}

// ---------------------------------------------------------------------------
// Auto-instancing
// ---------------------------------------------------------------------------

/**
 * Wraps `child` in an instanced-draw scope: each leaf reachable from
 * `child` runs `count` instances, with per-instance attribute streams
 * supplied by `attributes`. Names in `attributes` correspond to
 * uniforms the shader currently reads — at compile-scene time those
 * reads get rewritten via `instanceUniforms` to come from per-instance
 * vertex attributes instead. Special handling for `ModelTrafo` and
 * its trafo aliases (`ModelViewTrafo`, `ModelTrafoInv`, …,
 * `NormalMatrix`); see `docs/auto-instancing.md`.
 *
 * Subtree precondition (validated at scene-compile time): no nested
 * `SgInstanced`, no leaf with `drawCall.instanceCount > 1`, no
 * indirect-draw leaves. Friendly error otherwise.
 */
export interface SgInstanced {
  readonly kind: "Instanced";
  readonly count: aval<number>;
  /**
   * Per-instance trafos, when the convenience trafo case is used.
   * The runtime pre-multiplies the scope-accumulated `ModelTrafo`
   * into each entry before upload, replaces the leaf's `ModelTrafo`
   * uniform with identity, and adds split-into-4-cols attributes
   * `_InstanceTrafo_col0..3` + `_InstanceTrafoInv_col0..3` to the
   * leaf. The shader-side rewrite (`instanceUniforms`) keys off
   * `"ModelTrafo"` being in the attribute set.
   */
  readonly trafos?: aval<ReadonlyArray<import("@aardworx/wombat.base").Trafo3d>>;
  /**
   * Generic per-instance vertex-attribute streams keyed by uniform
   * name. Use for non-trafo data (e.g. per-instance Color). The
   * shader rewrite replaces `ReadInput("Uniform", X)` with
   * `ReadInput("Input", X)` for each name in this map.
   */
  readonly attributes: HashMap<string, BufferView>;
  readonly child: SgNode;
}
