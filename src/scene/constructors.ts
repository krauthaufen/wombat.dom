// `Sg` — unified scene-graph surface. Three things on one
// namespace:
//
//   1. JSX component:
//        <Sg Trafo={t} Shader={s}>...</Sg>
//      Wraps children with the supplied scope attributes. Returns
//      a tagged-Fragment VNode that <RenderControl> recognises.
//
//   2. JSX sub-components:
//        <Sg.Unordered>     <Sg.Adaptive value={...}>
//        <Sg.Box/>          <Sg.Quad/>
//
//   3. Imperative builders + math helpers, for code-generated
//      trees and `Trafo` array elements:
//        Sg.empty / Sg.group / Sg.shader / Sg.uniform / Sg.trafo
//        Sg.view / Sg.proj / Sg.camera / Sg.delay / Sg.on / ...
//        Sg.translate / Sg.scale / Sg.rotate / Sg.identity
//
// Locking the JSX surface to one identifier (`Sg`) keeps the user-
// facing DSL terse (`<Sg Trafo={[Sg.translate(v), Sg.scale(2)]}>`)
// while still exposing the full imperative API for advanced uses.

import {
  AList, ASet, AVal,
  HashMap, type alist, type amap, type aset, type aval,
} from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, Rot3d, Scale3d, Shift3d } from "@aardworx/wombat.base";
import type { Effect } from "@aardworx/wombat.shader";
import type {
  BlendState, BufferView, DrawCall,
} from "@aardworx/wombat.rendering/core";

import type { Child, VNode } from "../vnode.js";
import { isVNode } from "../vnode.js";
import { isAList, isASet, isAVal } from "../guards.js";
import type {
  EventHandlers,
  SgNode,
  SgLeaf,
  TrafoValue,
  UniformBag,
} from "./sg.js";
import { box as boxLeaf, quad as quadLeaf, type BoxOptions, type QuadOptions } from "./primitives.js";
import {
  sgVNode, isSgVNode, extractSgNode, SG_KINDS,
} from "./sgVNode.js";

// ---------------------------------------------------------------------------
// Imperative builders
// ---------------------------------------------------------------------------

const empty: SgNode = { kind: "Empty" };

function leaf(spec: {
  vertexAttributes: HashMap<string, aval<BufferView>>;
  instanceAttributes?: HashMap<string, aval<BufferView>>;
  indices?: aval<BufferView>;
  drawCall: aval<DrawCall>;
}): SgLeaf {
  return {
    kind: "Leaf",
    vertexAttributes: spec.vertexAttributes,
    ...(spec.instanceAttributes !== undefined ? { instanceAttributes: spec.instanceAttributes } : {}),
    ...(spec.indices !== undefined ? { indices: spec.indices } : {}),
    drawCall: spec.drawCall,
  };
}

function group(children: alist<SgNode> | ReadonlyArray<SgNode> | SgNode): SgNode {
  if (Array.isArray(children)) {
    return { kind: "Group", children: AList.ofList(children as SgNode[]) };
  }
  if ((children as alist<SgNode>).getReader !== undefined) {
    return { kind: "Group", children: children as alist<SgNode> };
  }
  return { kind: "Group", children: AList.ofList([children as SgNode]) };
}

function unordered(children: aset<SgNode> | ReadonlyArray<SgNode>): SgNode {
  if (Array.isArray(children)) {
    return { kind: "UnorderedGroup", children: ASet.ofList(children as SgNode[]) };
  }
  return { kind: "UnorderedGroup", children: children as aset<SgNode> };
}

function adaptive(child: aval<SgNode>): SgNode {
  return { kind: "AdaptiveGroup", child };
}

function trafo(value: TrafoValue, child: SgNode): SgNode {
  return { kind: "Trafo", value, child };
}

function shader(effect: Effect, child: SgNode): SgNode {
  return { kind: "Shader", effect, child };
}

function uniformBag(entries: Record<string, unknown | aval<unknown>>): UniformBag {
  let map = HashMap.empty<string, aval<unknown>>();
  for (const [k, v] of Object.entries(entries)) {
    map = map.add(k, isAValRuntime(v) ? v as aval<unknown> : AVal.constant(v));
  }
  return { kind: "Static", entries: map };
}

function uniform(entries: Record<string, unknown | aval<unknown>>, child: SgNode): SgNode;
function uniform(entries: amap<string, aval<unknown>>, child: SgNode): SgNode;
function uniform(entries: Record<string, unknown> | amap<string, aval<unknown>>, child: SgNode): SgNode {
  if ((entries as amap<string, aval<unknown>>).content !== undefined) {
    return {
      kind: "Uniform",
      bag: { kind: "Dynamic", entries: entries as amap<string, aval<unknown>> },
      child,
    };
  }
  return { kind: "Uniform", bag: uniformBag(entries as Record<string, unknown>), child };
}

function blendMode(mode: BlendState, child: SgNode): SgNode {
  return { kind: "BlendMode", mode, child };
}

function cursor(value: string | aval<string>, child: SgNode): SgNode {
  return { kind: "Cursor", cursor: value, child };
}

function pickThrough(value: boolean, child: SgNode): SgNode {
  return { kind: "PickThrough", value, child };
}

function on(handlers: EventHandlers, child: SgNode): SgNode {
  return { kind: "On", handlers, child };
}

function active(value: aval<boolean>, child: SgNode): SgNode {
  return { kind: "Active", active: value, child };
}

function viewScope(view: aval<Trafo3d>, child: SgNode): SgNode {
  return { kind: "View", view, child };
}

function projScope(proj: aval<Trafo3d>, child: SgNode): SgNode {
  return { kind: "Proj", proj, child };
}

function camera(view: aval<Trafo3d>, proj: aval<Trafo3d>, child: SgNode): SgNode {
  return viewScope(view, projScope(proj, child));
}

function delay(
  create: (state: import("./traversalState.js").TraversalState) => SgNode,
): SgNode {
  return { kind: "Delay", create };
}

// ---------------------------------------------------------------------------
// Trafo helpers — return Trafo3d (or aval<Trafo3d>) for use in arrays
// ---------------------------------------------------------------------------

const identity: Trafo3d = Trafo3d.identity;

function translate(v: V3d): Trafo3d;
function translate(v: aval<V3d>): aval<Trafo3d>;
function translate(v: V3d | aval<V3d>): Trafo3d | aval<Trafo3d> {
  return isAValRuntime(v)
    ? (v as aval<V3d>).map((vv) => Trafo3d.translation(vv as V3d))
    : Trafo3d.translation(v as V3d);
}

function scale(s: number | V3d | Scale3d): Trafo3d;
function scale(s: aval<number | V3d | Scale3d>): aval<Trafo3d>;
function scale(s: number | V3d | Scale3d | aval<number | V3d | Scale3d>): Trafo3d | aval<Trafo3d> {
  if (isAValRuntime(s)) return (s as aval<number | V3d | Scale3d>).map((v) => Trafo3d.scaling(v as number));
  return typeof s === "number" ? Trafo3d.scaling(s) : Trafo3d.scaling(s as V3d);
}

function rotate(a: Rot3d | { axis: V3d; rad: number }): Trafo3d;
function rotate(a: aval<Rot3d | { axis: V3d; rad: number }>): aval<Trafo3d>;
function rotate(a: unknown): Trafo3d | aval<Trafo3d> {
  const lift = (v: unknown): Trafo3d => {
    if ((v as Rot3d) instanceof Rot3d) return Trafo3d.rotation(v as Rot3d);
    const ar = v as { axis: V3d; rad: number };
    return Trafo3d.rotation(ar.axis, ar.rad);
  };
  return isAValRuntime(a) ? (a as aval<unknown>).map(lift) : lift(a);
}

function trafoOf(t: Trafo3d): Trafo3d { return t; }

function isAValRuntime(v: unknown): boolean {
  return typeof v === "object" && v !== null && typeof (v as { getValue?: unknown }).getValue === "function";
}

// ---------------------------------------------------------------------------
// JSX scope component — `<Sg Trafo Shader Uniform ...>...</Sg>`
// ---------------------------------------------------------------------------

export interface SgScopeProps {
  // Composing across nesting
  Trafo?:   TrafoValue;
  Uniform?: Record<string, unknown | aval<unknown>>;

  // Overriding (innermost wins)
  Shader?:      Effect;
  BlendMode?:   BlendState;
  Cursor?:      string | aval<string>;
  PickThrough?: boolean;

  // Camera scopes — also sniffable from outside the scene.
  View?: aval<Trafo3d>;
  Proj?: aval<Trafo3d>;

  // Active gating (AND across nesting).
  Active?: aval<boolean>;

  // Event handlers — appended to the chain at this scope.
  // Wired into the pick dispatcher in M7-M8.
  OnClick?:        (e: unknown) => void;
  OnPointerEnter?: (e: unknown) => void;
  OnPointerLeave?: (e: unknown) => void;
  OnPointerDown?:  (e: unknown) => void;
  OnPointerUp?:    (e: unknown) => void;
  OnWheel?:        (e: unknown) => void;

  /**
   * Children: ordinary JSX (Sg components, fragments, arrays) plus
   * raw `SgNode` values for code-generated trees and the
   * `Sg.delay` escape hatch. `collectSgChildren` flattens all of
   * the above.
   */
  children?: Child | Child[] | SgNode | ReadonlyArray<Child | SgNode>;
}

function applyScopeAttrs(node: SgNode, props: SgScopeProps): SgNode {
  // Order: events innermost, Trafo outermost. Wrapping inside-out
  // so the call site reads top-to-bottom. The JSX rule: leftmost
  // (= outermost wrapper) is applied last to a point.
  let n = node;
  const events = collectEventHandlers(props);
  if (Object.keys(events).length > 0) n = on(events, n);
  if (props.Proj !== undefined)        n = projScope(props.Proj, n);
  if (props.View !== undefined)        n = viewScope(props.View, n);
  if (props.Active !== undefined)      n = active(props.Active, n);
  if (props.PickThrough !== undefined) n = pickThrough(props.PickThrough, n);
  if (props.Cursor !== undefined)      n = cursor(props.Cursor, n);
  if (props.BlendMode !== undefined)   n = blendMode(props.BlendMode, n);
  if (props.Shader !== undefined)      n = shader(props.Shader, n);
  if (props.Uniform !== undefined)     n = uniform(props.Uniform, n);
  if (props.Trafo !== undefined)       n = trafo(props.Trafo, n);
  return n;
}

function collectEventHandlers(props: SgScopeProps): EventHandlers {
  const out: Record<string, (e: unknown) => void> = {};
  if (props.OnClick)        out.click = props.OnClick;
  if (props.OnPointerEnter) out.pointerenter = props.OnPointerEnter;
  if (props.OnPointerLeave) out.pointerleave = props.OnPointerLeave;
  if (props.OnPointerDown)  out.pointerdown = props.OnPointerDown;
  if (props.OnPointerUp)    out.pointerup = props.OnPointerUp;
  if (props.OnWheel)        out.wheel = props.OnWheel;
  return out;
}

function SgScope(props: SgScopeProps): VNode {
  const childNode = collectSgChildren(props.children);
  return sgVNode(applyScopeAttrs(childNode, props));
}

function SgUnordered(props: SgScopeProps): VNode {
  const inner = collectSgChildren(props.children);
  const unorderedChildren: SgNode =
    inner.kind === "Empty" ? inner : unordered([inner]);
  return sgVNode(applyScopeAttrs(unorderedChildren, props));
}

function SgAdaptive(props: { value: aval<SgNode> }): VNode {
  return sgVNode(adaptive(props.value));
}

function SgBox(props: BoxOptions = {}): VNode {
  return sgVNode(boxLeaf(props));
}

function SgQuad(props: QuadOptions = {}): VNode {
  return sgVNode(quadLeaf(props));
}

// ---------------------------------------------------------------------------
// Children-walker — extract SgNodes from JSX children
// ---------------------------------------------------------------------------

/**
 * Walk a JSX-children value and produce a single `SgNode`.
 *
 *   - Tagged Fragment carriers (built by `sgVNode`) → unwrap.
 *   - Component VNodes → call type fn and recurse.
 *   - Plain Fragment VNodes → recurse on `props.children`.
 *   - Arrays → flatten + group.
 *   - aval<SgNode>  → SgAdaptiveGroup.
 *   - alist<SgNode> → SgGroup.
 *   - aset<SgNode>  → SgUnorderedGroup.
 *   - Raw `SgNode` (object with a known `kind`) → pass through.
 *   - Everything else → ignored.
 *
 * Multi-segment children (mixing leaves + alists + asets) preserve
 * each segment's container shape — see CLAUDE.md "heterogeneous
 * JSX children" rule.
 */
export function collectSgChildren(value: unknown): SgNode {
  const segments = collectImpl(value);
  if (segments.length === 0) return empty;
  if (segments.length === 1) return segments[0]!;
  return group(segments);
}

function collectImpl(value: unknown): SgNode[] {
  if (value === null || value === undefined || value === false || value === true) return [];
  if (Array.isArray(value)) {
    const out: SgNode[] = [];
    for (const v of value) out.push(...collectImpl(v));
    return out;
  }
  if (isVNode(value)) {
    if (isSgVNode(value)) return [extractSgNode(value)];
    const v: { _tag: string; type?: (p: unknown) => unknown; props: { children?: unknown } } = value as never;
    if (v._tag === "component") return collectImpl(v.type!(v.props));
    if (v._tag === "fragment")  return collectImpl(v.props.children);
    return []; // element VNode — HTML inside RenderControl has no surface
  }
  if (isRawSgNode(value)) return [value];
  if (isAVal(value))  return [adaptive(value as aval<SgNode>)];
  if (isAList(value)) return [group(value as alist<SgNode>)];
  if (isASet(value))  return [unordered(value as aset<SgNode>)];
  return [];
}

function isRawSgNode(value: unknown): value is SgNode {
  if (value === null || typeof value !== "object") return false;
  const kind = (value as { kind?: unknown }).kind;
  return typeof kind === "string" && SG_KINDS.has(kind);
}

// ---------------------------------------------------------------------------
// Unified `Sg` namespace — JSX function + imperative + helpers + sub-components
// ---------------------------------------------------------------------------

interface SgNamespace {
  // JSX call signature
  (props: SgScopeProps): VNode;

  // Sub-components (JSX)
  Unordered: typeof SgUnordered;
  Adaptive:  typeof SgAdaptive;
  Box:       typeof SgBox;
  Quad:      typeof SgQuad;

  // Imperative builders — return SgNode / SgLeaf for code-
  // generated trees and inside `Sg.delay`.
  empty:       SgNode;
  leaf:        typeof leaf;
  group:       typeof group;
  unordered:   typeof unordered;
  adaptive:    typeof adaptive;
  trafo:       typeof trafo;
  shader:      typeof shader;
  uniform:     typeof uniform;
  blendMode:   typeof blendMode;
  cursor:      typeof cursor;
  pickThrough: typeof pickThrough;
  on:          typeof on;
  active:      typeof active;
  view:        typeof viewScope;
  proj:        typeof projScope;
  camera:      typeof camera;
  delay:       typeof delay;
  // Imperative primitives (lowercase) — same draws as `<Sg.Box/>`/
  // `<Sg.Quad/>` but as plain SgLeaf values, usable inside `delay`.
  box:         typeof boxLeaf;
  quad:        typeof quadLeaf;

  // Math helpers (return Trafo3d, plain or aval)
  identity:  Trafo3d;
  translate: typeof translate;
  scale:     typeof scale;
  rotate:    typeof rotate;
  trafoOf:   typeof trafoOf;
}

export const Sg: SgNamespace = (() => {
  const fn = SgScope as SgNamespace;
  fn.Unordered = SgUnordered;
  fn.Adaptive  = SgAdaptive;
  fn.Box       = SgBox;
  fn.Quad      = SgQuad;
  fn.empty       = empty;
  fn.leaf        = leaf;
  fn.group       = group;
  fn.unordered   = unordered;
  fn.adaptive    = adaptive;
  fn.trafo       = trafo;
  fn.shader      = shader;
  fn.uniform     = uniform;
  fn.blendMode   = blendMode;
  fn.cursor      = cursor;
  fn.pickThrough = pickThrough;
  fn.on          = on;
  fn.active      = active;
  fn.view        = viewScope;
  fn.proj        = projScope;
  fn.camera      = camera;
  fn.delay       = delay;
  fn.box         = boxLeaf;
  fn.quad        = quadLeaf;
  fn.identity    = identity;
  fn.translate   = translate;
  fn.scale       = scale;
  fn.rotate      = rotate;
  fn.trafoOf     = trafoOf;
  return fn;
})();

void Shift3d;
