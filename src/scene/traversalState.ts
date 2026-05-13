// TraversalState — the immutable per-scope state walked from root
// to leaves during scene-graph evaluation.
//
// Every `with*` method returns a new `TraversalState`. The default
// values come from `TraversalState.empty`. The composition rules
// below mirror the per-attribute semantics declared in `sg.ts`.
//
// Why a class with `with*` methods rather than a record + free
// functions: with `readonly` fields + private constructor we can't
// accidentally mutate; static helpers stay easy to grep for; method
// chaining reads naturally on the call site.
//
// ---------------------------------------------------------------
// Composition rules (recap)
// ---------------------------------------------------------------
//
//   model:       composes  (Trafo3d.mul — see §matrix order below)
//   uniforms:    composes  (HashMap merge, inner-wins)
//   handlers:    composes  (append to chain — outer + inner fire)
//   shader:      overrides
//   blendMode:   overrides
//   cursor:      overrides
//   pickThrough: overrides
//   intersectable: overrides
//   pixelSnapRadius: overrides
//   active:      AND-composes
//
// ---------------------------------------------------------------
// Matrix-order convention
// ---------------------------------------------------------------
//
// `wombat.base`'s `Trafo3d.mul` reads `a.mul(b).transform(v) ===
// b.transform(a.transform(v))` — i.e. `a.mul(b)` "applies a, then
// b" to a point. This is the OPPOSITE of F# Aardvark.Base's `*`
// operator. Be careful when porting expressions.
//
// When entering a `<Sg Trafo={T}>` scope, the leaf's local point
// should be transformed first by T, then by the parent's
// accumulated model. So the new model is `T.mul(parentModel)`.
// (`pushTrafo`.)
//
// For an array `[A, B, C]` at one scope: index 0 (A) is outermost
// — applied last. Composed scope-trafo applied to a point: first
// C, then B, then A. Compute by forward iteration:
//
//   acc = identity;
//   for i in 0..n-1: acc = arr[i].mul(acc);
//
// (See `composeTrafoArray`.)

import {
  AVal,
  HashMap,
  type aval,
} from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, V3f, type IIntersectable } from "@aardworx/wombat.base";
import type { Effect } from "@aardworx/wombat.shader";
import type { BlendState, BufferView, PipelineState, IUniformProvider } from "@aardworx/wombat.rendering/core";
import type {
  BlendConstantValue,
  ColorMaskValue,
  CullValue,
  DepthBiasValue,
  DepthCompare,
  EventHandlers,
  FillModeValue,
  FrontFaceValue,
  ModeValue,
  StencilModeValue,
  TrafoValue,
} from "./sg.js";
import type { LeafPickEntry } from "./picking/registry.js";

// ---------------------------------------------------------------------------
// Helpers — value-or-aval normalisation
// ---------------------------------------------------------------------------

const isAval = <T>(v: T | aval<T>): v is aval<T> =>
  typeof v === "object" && v !== null && "getValue" in (v as object);

const asAval = <T>(v: T | aval<T>): aval<T> =>
  isAval(v) ? v : AVal.constant(v);

// ---------------------------------------------------------------------------
// Compose a Trafo array into a single aval<Trafo3d>
// ---------------------------------------------------------------------------

/**
 * Reduce a {@link TrafoValue} (single, aval, or array) into one
 * `aval<Trafo3d>` honouring the array-composition rule. For a
 * static value this is a constant aval. Mixing static and aval
 * elements yields an aval that depends on the dynamic ones only.
 */
export function composeTrafoValue(value: TrafoValue): aval<Trafo3d> {
  if (Array.isArray(value)) {
    return composeTrafoArray(value);
  }
  return asAval(value as Trafo3d | aval<Trafo3d>);
}

function composeTrafoArray(arr: ReadonlyArray<Trafo3d | aval<Trafo3d>>): aval<Trafo3d> {
  if (arr.length === 0) return AVal.constant(Trafo3d.identity);
  if (arr.length === 1) return asAval(arr[0]!);
  // Lift each element to an aval, then zip and reduce by mul.
  const lifted = arr.map(asAval);
  return AVal.zip(...lifted).map((...vs: Trafo3d[]) => {
    let acc: Trafo3d = Trafo3d.identity;
    // Forward iteration: acc = arr[i].mul(acc) — see file header.
    for (let i = 0; i < vs.length; i++) {
      acc = vs[i]!.mul(acc);
    }
    return acc;
  });
}

/** Compose two `aval<Trafo3d>` per scope-descent rule: child applies first, parent last. */
export function composeModel(parent: aval<Trafo3d>, child: aval<Trafo3d>): aval<Trafo3d> {
  // child.mul(parent) === "first child, then parent" — see header.
  return AVal.zip(child, parent).map((c, p) => c.mul(p));
}

// ---------------------------------------------------------------------------
// TraversalState
// ---------------------------------------------------------------------------

/**
 * Names the auto-injected uniform resolver can produce — the standard
 * transforms + their composites + inverses + the normal matrix +
 * camera/light location + viewport. Listed so `names()` works for
 * diagnostics; each value's aval chain is built lazily on `tryGet`, so
 * a leaf only ever materialises the ~2-4 a shader actually references.
 * (Mirrors Aardvark.Rendering's `tryGetDerivedUniform` table.)
 */
export const AUTO_UNIFORM_NAMES: readonly string[] = [
  "ModelTrafo", "ViewTrafo", "ProjTrafo",
  "ModelViewTrafo", "ViewProjTrafo", "ModelViewProjTrafo",
  "ModelTrafoInv", "ViewTrafoInv", "ProjTrafoInv",
  "ModelViewTrafoInv", "ViewProjTrafoInv", "ModelViewProjTrafoInv",
  "NormalMatrix", "CameraLocation", "LightLocation", "ViewportSize",
];

/**
 * Per-scope state accumulated while walking the scene graph. All
 * fields are readonly; mutators return new instances. Avals are
 * persistent values themselves — composing into the state never
 * disturbs unrelated graph dependencies.
 *
 * `TraversalState` is itself an {@link IUniformProvider}: `tryGet`
 * resolves `<Sg Uniform={…}>` scope entries and the auto-injected
 * derived trafos (computed on the fly from `model`/`view`/`proj` —
 * `compose(view, proj)` etc. are memoised by the adaptive plugin on
 * (a,b) identity, so recomputing per call is idempotent and no
 * per-leaf cache is needed). `buildRenderObject` overlays only the
 * leaf-local `PickId` on top — so the per-leaf uniform plumbing is a
 * single small overlay map instead of a merge + texture-split + lazy-
 * provider closure + Map allocated for every leaf.
 *
 * @remarks `view`, `proj`, and `viewport` are populated by the
 * `<Sg.RenderControl>` shell that owns the canvas (M4) and by an
 * `<Sg.Camera>` attribute scope (M5). They start out as
 * `Trafo3d.identity` / `{ width: 1, height: 1 }` so that a leaf
 * accidentally evaluated outside a render control still gets a
 * sane (if useless) state to consume.
 */
export class TraversalState implements IUniformProvider {
  /** Local-to-world model trafo. Starts at identity. */
  readonly model: aval<Trafo3d>;
  /** World-to-view trafo (camera). Starts at identity. */
  readonly view: aval<Trafo3d>;
  /** View-to-clip projection. Starts at identity. */
  readonly proj: aval<Trafo3d>;
  /** Pixel viewport size. Starts at (1, 1) — replaced by `<RenderControl>`. */
  readonly viewport: aval<{ width: number; height: number }>;
  /** Innermost shader-scope effect, or `undefined` if no Shader scope was entered. */
  readonly shader: Effect | undefined;
  /** Per-key uniform map; inner-wins on conflict (achieved by HashMap.add). */
  readonly uniforms: HashMap<string, aval<unknown>>;
  /** Innermost blend mode, or `undefined`. */
  readonly blendMode: BlendState | undefined;
  /** Innermost cursor; an aval is fine because Cursor is a CSS prop, not a pipeline state. */
  readonly cursor: aval<string> | undefined;
  /** Innermost pick-through value; defaults to `false`. */
  readonly pickThrough: boolean;
  /**
   * Innermost intersectable scope, or `undefined` if no
   * `<Sg Intersectable=...>` scope was entered. Consumed by the pick
   * dispatcher's BVH ray fall-through.
   */
  readonly intersectable: aval<IIntersectable> | undefined;
  /**
   * Innermost pixel-snap-radius (device pixels). Defaults to
   * `AVal.constant(1)`. The dispatcher clamps to `[0, 16]`.
   */
  readonly pixelSnapRadius: aval<number>;
  /** Active conjunction across all enclosing Active scopes. Defaults to `true`. */
  readonly active: aval<boolean>;
  /**
   * Event-handler chain, in capture order (outermost first). Empty
   * by default. Append-only — each new scope appends an entry; the
   * walker dispatches in order for capture phase, reversed for
   * bubble phase.
   *
   * Each entry carries both the scope's `EventHandlers` (the
   * identity carrier — same reference across sibling leaves under
   * the same On scope, used by the dispatcher's prefix diff) and a
   * snapshot of `state.model` taken when the scope was pushed (the
   * trafo accumulated UP TO AND INCLUDING this scope). The
   * dispatcher applies that local2World during capture/bubble so
   * each level's handlers see `e.position` etc. in their own local
   * frame. F# parity: `event.Transformed(model)` —
   * `Aardvark.Dom/SceneGraph/TraversalState.fs runCapture/runBubble`.
   */
  readonly handlers: ReadonlyArray<LeafPickEntry>;

  // ---------------- Phase 1 — render-state -----------------------------

  /** Innermost depth-test mode. Defaults to `"less"`. */
  readonly depthTest: aval<DepthCompare>;
  /** Innermost depth-write mask. Defaults to `true`. */
  readonly depthMask: aval<boolean>;
  /** Innermost depth-bias triple, or `undefined` for no bias. */
  readonly depthBias: aval<DepthBiasValue> | undefined;
  /** Unclipped depth (override). Defaults to `false`. */
  readonly depthClamp: aval<boolean>;
  /** Cull mode (override). Defaults to `"none"`. */
  readonly cullMode: aval<CullValue>;
  /**
   * Optional derived-mode rule for cull. When non-undefined, the
   * leaf's RO carries this rule on `RO.modeRules.cull` and it
   * overrides the static `cullMode` value at descriptor-snapshot
   * time. `<Sg CullMode={rule}>` populates this; an aval-shaped
   * `<Sg CullMode={cval}>` clears it.
   */
  readonly cullModeRule: import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"cull"> | undefined;
  /** Optional derived-mode rule for depthCompare. */
  readonly depthTestRule: import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"depthCompare"> | undefined;
  /** Optional derived-mode rule for depthWrite. */
  readonly depthMaskRule: import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"depthWrite"> | undefined;
  /** Front-face winding (override). Defaults to `"ccw"`. */
  readonly frontFace: aval<FrontFaceValue>;
  /** Optional derived-mode rule for frontFace. */
  readonly frontFaceRule: import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"frontFace"> | undefined;
  /** Fill mode (override). Defaults to `"fill"`. */
  readonly fillMode: aval<FillModeValue>;
  /** Per-pass blend constant, or `undefined` if unset. */
  readonly blendConstant: aval<BlendConstantValue> | undefined;
  /** Per-attachment color-write masks (innermost wins). */
  readonly colorMask: aval<HashMap<string, ColorMaskValue>>;
  /** Stencil mode, or `undefined` if disabled. */
  readonly stencilMode: aval<StencilModeValue> | undefined;
  /** Render-pass ordinal (innermost wins). Defaults to `RenderPass.main` (0). */
  readonly renderPass: number;

  // ---------------- Phase 2 — geometry attribute scopes ----------------

  readonly vertexAttributes: HashMap<string, BufferView>;
  readonly instanceAttributes: HashMap<string, BufferView>;
  readonly index: BufferView | undefined;
  readonly mode: aval<ModeValue>;
  /** Optional derived-mode rule for topology. */
  readonly modeRule: import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"topology"> | undefined;

  // ---------------- Phase 3 — misc scopes ------------------------------

  /** When true, descendants skip pick-chain registration. */
  readonly noEvents: aval<boolean>;
  /** When true, descendants opt out of BVH ray fall-through. */
  readonly forcePixelPicking: aval<boolean>;
  /** When true, this scope is a focus target. */
  readonly canFocus: aval<boolean>;
  /** Per-frame time clock (Phase 7). Defaults to a constant initial-load timestamp. */
  readonly time: aval<number>;

  // ---------------- Auto-instancing scope ------------------------------

  /**
   * Innermost `SgInstanced` scope, or `undefined` if none. Set by
   * `pushInstancing`. The leaf-lowering path inspects this to decide
   * whether to apply the `instanceUniforms` IR rewrite + bind the
   * per-instance attribute buffers. See `docs/auto-instancing.md`.
   */
  readonly instancing: import("./sg.js").SgInstanced | undefined;

  /**
   * Cumulative `ModelTrafo` from the scope OUTSIDE the innermost
   * `SgInstanced`. Mirrors Aardvark's `applyTrafos` pattern: the
   * parent scope's accumulated trafo becomes the leaf's `ModelTrafo`
   * uniform; trafos inside the instancing subtree get pre-merged
   * into each per-instance trafo. `pushInstancing` stashes
   * `this.model` here and resets the outer `model` to identity so
   * the in-subtree trafos accumulate cleanly.
   */
  readonly instancingParentModel: aval<Trafo3d> | undefined;

  /**
   * The `PipelineState` derived from the render-state scopes accumulated
   * so far (blend/depth/cull/fill/colour-mask/stencil/pass + the base
   * rasterizer). Computed **once per render-state scope** (and once for
   * the root), then inherited unchanged by descendants — `<Sg Trafo>`,
   * `<Sg Shader>`, geometry scopes etc. don't touch it. Leaves read it
   * directly instead of each rebuilding an identical `PipelineState`;
   * sharing the object also lets the heap path bucket sibling draws
   * together (it keys buckets by `PipelineState` content, cached per
   * object). `undefined` only before any pipeline-state context is set
   * (e.g. `TraversalState.empty` used directly in tests) — in that case
   * the leaf-lowering path derives one on the spot.
   */
  readonly pipelineState: PipelineState | undefined;

  private constructor(spec: TraversalSpec) {
    this.model = spec.model;
    this.view = spec.view;
    this.proj = spec.proj;
    this.viewport = spec.viewport;
    this.shader = spec.shader;
    this.uniforms = spec.uniforms;
    this.blendMode = spec.blendMode;
    this.cursor = spec.cursor;
    this.pickThrough = spec.pickThrough;
    this.intersectable = spec.intersectable;
    this.pixelSnapRadius = spec.pixelSnapRadius;
    this.active = spec.active;
    this.handlers = spec.handlers;
    this.depthTest = spec.depthTest;
    this.depthMask = spec.depthMask;
    this.depthBias = spec.depthBias;
    this.depthClamp = spec.depthClamp;
    this.cullMode = spec.cullMode;
    this.cullModeRule = spec.cullModeRule;
    this.depthTestRule = spec.depthTestRule;
    this.depthMaskRule = spec.depthMaskRule;
    this.frontFace = spec.frontFace;
    this.frontFaceRule = spec.frontFaceRule;
    this.fillMode = spec.fillMode;
    this.blendConstant = spec.blendConstant;
    this.colorMask = spec.colorMask;
    this.stencilMode = spec.stencilMode;
    this.renderPass = spec.renderPass;
    this.vertexAttributes = spec.vertexAttributes;
    this.instanceAttributes = spec.instanceAttributes;
    this.index = spec.index;
    this.mode = spec.mode;
    this.modeRule = spec.modeRule;
    this.noEvents = spec.noEvents;
    this.forcePixelPicking = spec.forcePixelPicking;
    this.canFocus = spec.canFocus;
    this.time = spec.time;
    this.instancing = spec.instancing;
    this.instancingParentModel = spec.instancingParentModel;
    this.pipelineState = spec.pipelineState;
  }

  /** Empty initial state — identity transforms, no shader, no uniforms, active=true. */
  static readonly empty: TraversalState = new TraversalState({
    model: AVal.constant(Trafo3d.identity),
    view: AVal.constant(Trafo3d.identity),
    proj: AVal.constant(Trafo3d.identity),
    viewport: AVal.constant({ width: 1, height: 1 }),
    shader: undefined,
    uniforms: HashMap.empty<string, aval<unknown>>(),
    blendMode: undefined,
    cursor: undefined,
    pickThrough: false,
    intersectable: undefined,
    pixelSnapRadius: AVal.constant(1),
    active: AVal.constant(true),
    handlers: [],
    depthTest: AVal.constant<DepthCompare>("less"),
    depthMask: AVal.constant(true),
    depthBias: undefined,
    depthClamp: AVal.constant(false),
    cullMode: AVal.constant<CullValue>("none"),
    cullModeRule: undefined,
    depthTestRule: undefined,
    depthMaskRule: undefined,
    frontFace: AVal.constant<FrontFaceValue>("ccw"),
    frontFaceRule: undefined,
    fillMode: AVal.constant<FillModeValue>("fill"),
    blendConstant: undefined,
    colorMask: AVal.constant(HashMap.empty<string, ColorMaskValue>()),
    stencilMode: undefined,
    renderPass: 0,
    vertexAttributes: HashMap.empty<string, BufferView>(),
    instanceAttributes: HashMap.empty<string, BufferView>(),
    index: undefined,
    mode: AVal.constant<ModeValue>("triangle-list"),
    modeRule: undefined,
    noEvents: AVal.constant(false),
    forcePixelPicking: AVal.constant(false),
    canFocus: AVal.constant(false),
    time: AVal.constant(0),
    instancing: undefined,
    instancingParentModel: undefined,
    pipelineState: undefined,
  });

  // ---------------------------------------------------------------------------
  // Scope-descent operations — each returns a new state.
  // ---------------------------------------------------------------------------

  /** `<Sg Trafo={t}>` scope: child point first goes through `t`, then parent.model. */
  pushTrafo(t: TrafoValue): TraversalState {
    const composed = composeTrafoValue(t);
    return this.with({ model: composeModel(this.model, composed) });
  }

  /** `<Sg Shader={e}>` scope: override. */
  pushShader(effect: Effect): TraversalState {
    return this.with({ shader: effect });
  }

  /**
   * `<Sg Uniform={…}>` scope: per-key merge with inner-wins. The
   * supplied `entries` are the new scope's entries — for each key,
   * the inner value replaces any outer one in the result map.
   */
  pushUniforms(entries: HashMap<string, aval<unknown>>): TraversalState {
    let merged = this.uniforms;
    for (const [k, v] of entries) merged = merged.add(k, v);
    return this.with({ uniforms: merged });
  }

  /** `<Sg BlendMode={m}>`: override. Clears `pipelineState` (changes a
   *  field `derivePipelineState` reads). */
  pushBlendMode(mode: BlendState): TraversalState {
    return this.with({ blendMode: mode, pipelineState: undefined });
  }

  /** `<Sg Cursor={c}>`: override. */
  pushCursor(cursor: string | aval<string>): TraversalState {
    return this.with({ cursor: asAval(cursor) });
  }

  /** `<Sg PickThrough={b}>`: override. */
  pushPickThrough(value: boolean): TraversalState {
    return this.with({ pickThrough: value });
  }

  /** `<Sg Intersectable={i}>`: override. */
  pushIntersectable(intersectable: aval<IIntersectable>): TraversalState {
    return this.with({ intersectable });
  }

  /** `<Sg PixelSnapRadius={r}>`: override. */
  pushPixelSnapRadius(radius: aval<number>): TraversalState {
    return this.with({ pixelSnapRadius: radius });
  }

  /** `<Sg.Active value={a}>`: AND-compose across nesting. */
  pushActive(active: aval<boolean>): TraversalState {
    return this.with({
      active: AVal.zip(this.active, active).map((p, c) => p && c),
    });
  }

  /** `<Sg On={...}>`: append to chain. Snapshot the current model
   * trafo as the scope's local2World (model accumulated UP TO AND
   * INCLUDING this scope) — the dispatcher uses it to push each
   * handler-level into its own local frame. */
  pushHandlers(handlers: EventHandlers): TraversalState {
    const entry: LeafPickEntry = { handlers, local2World: this.model };
    return this.with({ handlers: [...this.handlers, entry] });
  }

  /** `<Sg.Camera>` (M5): set view+proj at the same time. */
  withCamera(view: aval<Trafo3d>, proj: aval<Trafo3d>): TraversalState {
    return this.with({ view, proj });
  }

  /** `<Sg.RenderControl>` (M4): set viewport size; usually called once at root. */
  withViewport(viewport: aval<{ width: number; height: number }>): TraversalState {
    return this.with({ viewport });
  }

  // ---------------- Phase 1 — render-state pushers ---------------------
  //
  // Each clears `pipelineState` (it changes a field `derivePipelineState`
  // reads) so the leaf-lowering path re-derives one for the subtree; the
  // non-render-state pushers (`pushTrafo`, `pushShader`, …) leave it
  // intact so a whole subtree without a render-state scope shares one.

  pushDepthTest(
    mode: aval<DepthCompare> | import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"depthCompare">,
  ): TraversalState {
    const isRule = typeof mode === "object" && mode !== null
                && (mode as { __derivedModeRule?: unknown }).__derivedModeRule === true;
    if (isRule) {
      return this.with({
        depthTestRule: mode as import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"depthCompare">,
        pipelineState: undefined,
      });
    }
    return this.with({
      depthTest: mode as aval<DepthCompare>,
      depthTestRule: undefined,
      pipelineState: undefined,
    });
  }
  pushDepthMask(
    write: aval<boolean> | import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"depthWrite">,
  ): TraversalState {
    const isRule = typeof write === "object" && write !== null
                && (write as { __derivedModeRule?: unknown }).__derivedModeRule === true;
    if (isRule) {
      return this.with({
        depthMaskRule: write as import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"depthWrite">,
        pipelineState: undefined,
      });
    }
    return this.with({
      depthMask: write as aval<boolean>,
      depthMaskRule: undefined,
      pipelineState: undefined,
    });
  }
  pushDepthBias(bias: aval<DepthBiasValue>): TraversalState { return this.with({ depthBias: bias, pipelineState: undefined }); }
  pushDepthClamp(clamp: aval<boolean>): TraversalState { return this.with({ depthClamp: clamp, pipelineState: undefined }); }
  pushCullMode(
    mode: aval<CullValue> | import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"cull">,
  ): TraversalState {
    // Distinguish rule vs aval: rules carry the brand check
    // `__derivedModeRule === true`. A rule wraps an inner cullMode
    // value (its `gpu.declared`); the static aval cullMode is set
    // to that declared so PipelineState reflects the fallback /
    // bake-time value.
    const isRule = typeof mode === "object" && mode !== null
                && (mode as { __derivedModeRule?: unknown }).__derivedModeRule === true;
    if (isRule) {
      const rule = mode as import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"cull">;
      const declaredSpec = rule.gpu?.declared;
      let cullModeAval: aval<CullValue>;
      if (declaredSpec === undefined) {
        cullModeAval = AVal.constant<CullValue>("back");
      } else if (typeof declaredSpec === "string") {
        cullModeAval = AVal.constant<CullValue>(declaredSpec);
      } else {
        // Already an aval — pass through. The static-PS path forces
        // this each frame so PipelineState reflects current value.
        cullModeAval = declaredSpec as aval<CullValue>;
      }
      return this.with({
        cullMode: cullModeAval,
        cullModeRule: rule,
        pipelineState: undefined,
      });
    }
    return this.with({
      cullMode: mode as aval<CullValue>,
      cullModeRule: undefined,
      pipelineState: undefined,
    });
  }
  pushFrontFace(
    mode: aval<FrontFaceValue> | import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"frontFace">,
  ): TraversalState {
    const isRule = typeof mode === "object" && mode !== null
                && (mode as { __derivedModeRule?: unknown }).__derivedModeRule === true;
    if (isRule) {
      return this.with({
        frontFaceRule: mode as import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"frontFace">,
        pipelineState: undefined,
      });
    }
    return this.with({
      frontFace: mode as aval<FrontFaceValue>,
      frontFaceRule: undefined,
      pipelineState: undefined,
    });
  }
  pushFillMode(mode: aval<FillModeValue>): TraversalState { return this.with({ fillMode: mode, pipelineState: undefined }); }
  pushBlendConstant(value: aval<BlendConstantValue>): TraversalState { return this.with({ blendConstant: value, pipelineState: undefined }); }
  pushColorMask(mask: aval<HashMap<string, ColorMaskValue>>): TraversalState { return this.with({ colorMask: mask, pipelineState: undefined }); }
  pushStencilMode(mode: aval<StencilModeValue>): TraversalState { return this.with({ stencilMode: mode, pipelineState: undefined }); }
  pushRenderPass(pass: number): TraversalState { return this.with({ renderPass: pass, pipelineState: undefined }); }

  // ---------------- Phase 2 — geometry attribute pushers ---------------

  /** `<Sg VertexAttributes={…}>`: COMPOSE per-key (inner-wins on conflict). */
  pushVertexAttributes(attrs: HashMap<string, BufferView>): TraversalState {
    let merged = this.vertexAttributes;
    for (const [k, v] of attrs) merged = merged.add(k, v);
    return this.with({ vertexAttributes: merged });
  }

  pushInstanceAttributes(attrs: HashMap<string, BufferView>): TraversalState {
    let merged = this.instanceAttributes;
    for (const [k, v] of attrs) merged = merged.add(k, v);
    return this.with({ instanceAttributes: merged });
  }

  pushIndex(index: BufferView | undefined): TraversalState { return this.with({ index }); }
  /** Clears `pipelineState` — `mode` feeds the topology in `derivePipelineState`. */
  pushMode(
    mode: aval<ModeValue> | import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"topology">,
  ): TraversalState {
    const isRule = typeof mode === "object" && mode !== null
                && (mode as { __derivedModeRule?: unknown }).__derivedModeRule === true;
    if (isRule) {
      return this.with({
        modeRule: mode as import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"topology">,
        pipelineState: undefined,
      });
    }
    return this.with({
      mode: mode as aval<ModeValue>,
      modeRule: undefined,
      pipelineState: undefined,
    });
  }

  // ---------------- Phase 3 — misc scope pushers -----------------------

  pushNoEvents(value: aval<boolean>): TraversalState { return this.with({ noEvents: value }); }
  pushForcePixelPicking(value: aval<boolean>): TraversalState { return this.with({ forcePixelPicking: value }); }
  pushCanFocus(value: aval<boolean>): TraversalState { return this.with({ canFocus: value }); }
  /** Push an `Sg.Instanced` scope. Validation is the caller's job —
   *  `compile.ts` runs `validateInstancingSubtree` before pushing.
   *  Stashes the current `model` as `instancingParentModel` and
   *  resets `model` to identity inside the subtree (Aardvark
   *  semantics: trafos inside the subtree pre-merge into each
   *  per-instance trafo; the stashed parent becomes the leaf's
   *  `ModelTrafo` uniform). */
  pushInstancing(node: import("./sg.js").SgInstanced): TraversalState {
    return this.with({
      instancing: node,
      instancingParentModel: this.model,
      model: AVal.constant(Trafo3d.identity),
    });
  }
  /** `<RenderControl>` populates this with the global frame clock. */
  withTime(time: aval<number>): TraversalState { return this.with({ time }); }

  // ---------------------------------------------------------------------------
  // Internal helper — copy with patched fields.
  // ---------------------------------------------------------------------------

  private with(patch: Partial<TraversalSpec>): TraversalState {
    return new TraversalState({
      model: patch.model ?? this.model,
      view: patch.view ?? this.view,
      proj: patch.proj ?? this.proj,
      viewport: patch.viewport ?? this.viewport,
      shader: "shader" in patch ? patch.shader : this.shader,
      uniforms: patch.uniforms ?? this.uniforms,
      blendMode: "blendMode" in patch ? patch.blendMode : this.blendMode,
      cursor: "cursor" in patch ? patch.cursor : this.cursor,
      pickThrough: patch.pickThrough ?? this.pickThrough,
      intersectable: "intersectable" in patch ? patch.intersectable : this.intersectable,
      pixelSnapRadius: patch.pixelSnapRadius ?? this.pixelSnapRadius,
      active: patch.active ?? this.active,
      handlers: patch.handlers ?? this.handlers,
      depthTest: patch.depthTest ?? this.depthTest,
      depthMask: patch.depthMask ?? this.depthMask,
      depthBias: "depthBias" in patch ? patch.depthBias : this.depthBias,
      depthClamp: patch.depthClamp ?? this.depthClamp,
      cullMode: patch.cullMode ?? this.cullMode,
      cullModeRule: patch.cullModeRule !== undefined
        ? patch.cullModeRule
        : (Object.prototype.hasOwnProperty.call(patch, "cullModeRule") ? undefined : this.cullModeRule),
      depthTestRule: patch.depthTestRule !== undefined
        ? patch.depthTestRule
        : (Object.prototype.hasOwnProperty.call(patch, "depthTestRule") ? undefined : this.depthTestRule),
      depthMaskRule: patch.depthMaskRule !== undefined
        ? patch.depthMaskRule
        : (Object.prototype.hasOwnProperty.call(patch, "depthMaskRule") ? undefined : this.depthMaskRule),
      frontFace: patch.frontFace ?? this.frontFace,
      frontFaceRule: patch.frontFaceRule !== undefined
        ? patch.frontFaceRule
        : (Object.prototype.hasOwnProperty.call(patch, "frontFaceRule") ? undefined : this.frontFaceRule),
      fillMode: patch.fillMode ?? this.fillMode,
      blendConstant: "blendConstant" in patch ? patch.blendConstant : this.blendConstant,
      colorMask: patch.colorMask ?? this.colorMask,
      stencilMode: "stencilMode" in patch ? patch.stencilMode : this.stencilMode,
      renderPass: patch.renderPass ?? this.renderPass,
      vertexAttributes: patch.vertexAttributes ?? this.vertexAttributes,
      instanceAttributes: patch.instanceAttributes ?? this.instanceAttributes,
      index: patch.index ?? this.index,
      mode: patch.mode ?? this.mode,
      modeRule: patch.modeRule !== undefined
        ? patch.modeRule
        : (Object.prototype.hasOwnProperty.call(patch, "modeRule") ? undefined : this.modeRule),
      noEvents: patch.noEvents ?? this.noEvents,
      forcePixelPicking: patch.forcePixelPicking ?? this.forcePixelPicking,
      canFocus: patch.canFocus ?? this.canFocus,
      time: patch.time ?? this.time,
      instancing: "instancing" in patch ? patch.instancing : this.instancing,
      instancingParentModel: "instancingParentModel" in patch ? patch.instancingParentModel : this.instancingParentModel,
      pipelineState: "pipelineState" in patch ? patch.pipelineState : this.pipelineState,
    });
  }

  /** Attach a derived `PipelineState` — inherited unchanged by
   *  descendants until the next render-state scope re-derives it. */
  withPipelineState(ps: PipelineState | undefined): TraversalState {
    return this.with({ pipelineState: ps });
  }

  // -------------------------------------------------------------------------
  // IUniformProvider — `<Sg Uniform={…}>` scope entries + auto-injected
  // derived trafos. The render backend `tryGet`s only names declared by an
  // effect's program interface, so the over-broad auto set costs nothing
  // for shaders that don't reference it. `<Sg.Uniform>` entries take
  // precedence (user-wins). Texture-valued `<Sg.Uniform>` entries also
  // appear here as raw avals, but the backend never `tryGet`s them via the
  // uniform path (textures/samplers are separate interface bindings, split
  // out into `RenderObject.textures`/`samplers` by `splitTexturesFromUniforms`).
  // -------------------------------------------------------------------------

  tryGet(name: string): aval<unknown> | undefined {
    const u = this.uniforms.tryFind(name);
    if (u !== undefined) return u;
    return this.tryGetAutoUniform(name);
  }

  // Deliberately does NOT enumerate the `<Sg.Uniform>` scope entries: a
  // scope can hold arbitrarily many uniforms, almost none of which a
  // given shader reads, and nothing should ever "bind everything `names()`
  // returns" — the render backend pulls exactly what an effect's program
  // interface declares, via `tryGet`. So `names()` here is just the
  // (small, fixed) auto-injected set, for diagnostics. Scope uniforms
  // remain resolvable on demand through `tryGet`.
  names(): Iterable<string> {
    return AUTO_UNIFORM_NAMES;
  }

  /**
   * Resolve an auto-injected derived uniform. Composition follows
   * Aardvark's `Trafo3d` `<*>`: `a <*> b` = "apply a first, then b"
   * (forward = `b.fw · a.fw`, column-vector math; `Trafo3d.mul` matches).
   * `AVal.{zip,map}` namespace form so the adaptive-memo plugin
   * recognises the combinators — leaves sharing `view`/`proj` collapse
   * to one `viewProj`; the per-leaf `model` gets its own `modelView`,
   * but only if a shader reads it.
   */
  private tryGetAutoUniform(name: string): aval<unknown> | undefined {
    const model = this.model, view = this.view, proj = this.proj;
    const compose = (a: aval<Trafo3d>, b: aval<Trafo3d>): aval<Trafo3d> =>
      AVal.zip(a, b).map((aT, bT) => aT.mul(bT));
    const inv = (t: aval<Trafo3d>): aval<Trafo3d> =>
      AVal.map(t, (x: Trafo3d) => x.inverse());
    const modelView     = (): aval<Trafo3d> => compose(model, view);
    const viewProj      = (): aval<Trafo3d> => compose(view, proj);
    const modelViewProj = (): aval<Trafo3d> => compose(model, viewProj());
    switch (name) {
      case "ModelTrafo": return model;
      case "ViewTrafo":  return view;
      case "ProjTrafo":  return proj;
      case "ModelViewTrafo":     return modelView();
      case "ViewProjTrafo":      return viewProj();
      case "ModelViewProjTrafo": return modelViewProj();
      case "ModelTrafoInv":         return inv(model);
      case "ViewTrafoInv":          return inv(view);
      case "ProjTrafoInv":          return inv(proj);
      case "ModelViewTrafoInv":     return inv(modelView());
      case "ViewProjTrafoInv":      return inv(viewProj());
      case "ModelViewProjTrafoInv": return inv(modelViewProj());
      // NormalMatrix = ModelTrafo.Backward.Transposed (inverse-transpose;
      // non-uniform-scale-correct normals). Full M44; shaders pad V3f
      // normals to V4f(n, 0) so the translation column is ignored.
      case "NormalMatrix":
        return AVal.map(model, (t: Trafo3d) => {
          const iT = t.backward.transpose();
          return Trafo3d.fromMatrices(iT, iT.transpose());
        });
      // CameraLocation = ViewTrafoInv · origin (view.backward's
      // translation). LightLocation defaults to it (headlight).
      case "CameraLocation":
      case "LightLocation":
        return AVal.map(view, (v: Trafo3d) => {
          const o = v.backward.transformPos(V3d.zero);
          return new V3f(o.x, o.y, o.z);
        });
      case "ViewportSize": return this.viewport;
      default: return undefined;
    }
  }
}

interface TraversalSpec {
  instancing: import("./sg.js").SgInstanced | undefined;
  instancingParentModel: aval<Trafo3d> | undefined;
  pipelineState: PipelineState | undefined;
  model: aval<Trafo3d>;
  view: aval<Trafo3d>;
  proj: aval<Trafo3d>;
  viewport: aval<{ width: number; height: number }>;
  shader: Effect | undefined;
  uniforms: HashMap<string, aval<unknown>>;
  blendMode: BlendState | undefined;
  cursor: aval<string> | undefined;
  pickThrough: boolean;
  intersectable: aval<IIntersectable> | undefined;
  pixelSnapRadius: aval<number>;
  active: aval<boolean>;
  handlers: ReadonlyArray<LeafPickEntry>;
  depthTest: aval<DepthCompare>;
  depthMask: aval<boolean>;
  depthBias: aval<DepthBiasValue> | undefined;
  depthClamp: aval<boolean>;
  cullMode: aval<CullValue>;
  cullModeRule: import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"cull"> | undefined;
  depthTestRule: import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"depthCompare"> | undefined;
  depthMaskRule: import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"depthWrite"> | undefined;
  frontFace: aval<FrontFaceValue>;
  frontFaceRule: import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"frontFace"> | undefined;
  fillMode: aval<FillModeValue>;
  blendConstant: aval<BlendConstantValue> | undefined;
  colorMask: aval<HashMap<string, ColorMaskValue>>;
  stencilMode: aval<StencilModeValue> | undefined;
  renderPass: number;
  vertexAttributes: HashMap<string, BufferView>;
  instanceAttributes: HashMap<string, BufferView>;
  index: BufferView | undefined;
  mode: aval<ModeValue>;
  modeRule: import("@aardworx/wombat.rendering/runtime").DerivedModeRule<"topology"> | undefined;
  noEvents: aval<boolean>;
  forcePixelPicking: aval<boolean>;
  canFocus: aval<boolean>;
  time: aval<number>;
}
