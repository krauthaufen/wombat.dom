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
import { Trafo3d } from "@aardworx/wombat.base";
import type { Effect } from "@aardworx/wombat.shader";
import type { BlendState } from "@aardworx/wombat.rendering/core";
import type { EventHandlers, TrafoValue } from "./sg.js";

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
 * Per-scope state accumulated while walking the scene graph. All
 * fields are readonly; mutators return new instances. Avals are
 * persistent values themselves — composing into the state never
 * disturbs unrelated graph dependencies.
 *
 * @remarks `view`, `proj`, and `viewport` are populated by the
 * `<Sg.RenderControl>` shell that owns the canvas (M4) and by an
 * `<Sg.Camera>` attribute scope (M5). They start out as
 * `Trafo3d.identity` / `{ width: 1, height: 1 }` so that a leaf
 * accidentally evaluated outside a render control still gets a
 * sane (if useless) state to consume.
 */
export class TraversalState {
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
   */
  readonly handlers: ReadonlyArray<EventHandlers>;

  private constructor(spec: {
    model: aval<Trafo3d>;
    view: aval<Trafo3d>;
    proj: aval<Trafo3d>;
    viewport: aval<{ width: number; height: number }>;
    shader: Effect | undefined;
    uniforms: HashMap<string, aval<unknown>>;
    blendMode: BlendState | undefined;
    cursor: aval<string> | undefined;
    pickThrough: boolean;
    pixelSnapRadius: aval<number>;
    active: aval<boolean>;
    handlers: ReadonlyArray<EventHandlers>;
  }) {
    this.model = spec.model;
    this.view = spec.view;
    this.proj = spec.proj;
    this.viewport = spec.viewport;
    this.shader = spec.shader;
    this.uniforms = spec.uniforms;
    this.blendMode = spec.blendMode;
    this.cursor = spec.cursor;
    this.pickThrough = spec.pickThrough;
    this.pixelSnapRadius = spec.pixelSnapRadius;
    this.active = spec.active;
    this.handlers = spec.handlers;
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
    pixelSnapRadius: AVal.constant(1),
    active: AVal.constant(true),
    handlers: [],
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

  /** `<Sg BlendMode={m}>`: override. */
  pushBlendMode(mode: BlendState): TraversalState {
    return this.with({ blendMode: mode });
  }

  /** `<Sg Cursor={c}>`: override. */
  pushCursor(cursor: string | aval<string>): TraversalState {
    return this.with({ cursor: asAval(cursor) });
  }

  /** `<Sg PickThrough={b}>`: override. */
  pushPickThrough(value: boolean): TraversalState {
    return this.with({ pickThrough: value });
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

  /** `<Sg On={...}>`: append to chain. */
  pushHandlers(handlers: EventHandlers): TraversalState {
    return this.with({ handlers: [...this.handlers, handlers] });
  }

  /** `<Sg.Camera>` (M5): set view+proj at the same time. */
  withCamera(view: aval<Trafo3d>, proj: aval<Trafo3d>): TraversalState {
    return this.with({ view, proj });
  }

  /** `<Sg.RenderControl>` (M4): set viewport size; usually called once at root. */
  withViewport(viewport: aval<{ width: number; height: number }>): TraversalState {
    return this.with({ viewport });
  }

  // ---------------------------------------------------------------------------
  // Internal helper — copy with patched fields.
  // ---------------------------------------------------------------------------

  private with(patch: Partial<{
    model: aval<Trafo3d>;
    view: aval<Trafo3d>;
    proj: aval<Trafo3d>;
    viewport: aval<{ width: number; height: number }>;
    shader: Effect | undefined;
    uniforms: HashMap<string, aval<unknown>>;
    blendMode: BlendState | undefined;
    cursor: aval<string> | undefined;
    pickThrough: boolean;
    pixelSnapRadius: aval<number>;
    active: aval<boolean>;
    handlers: ReadonlyArray<EventHandlers>;
  }>): TraversalState {
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
      pixelSnapRadius: patch.pixelSnapRadius ?? this.pixelSnapRadius,
      active: patch.active ?? this.active,
      handlers: patch.handlers ?? this.handlers,
    });
  }
}
