// `Sg.*` — pure constructors for the scene-graph data model. Used by
// the rendering layer's compileScene (M3) and by the JSX wrappers
// (`<Sg>` / `<Sg.Box/>` etc., M2 follow-up). Every function here is
// a leaf-level fact about scene-graph shapes; nothing pulls in the
// renderer.

import {
  AList, ASet, AVal,
  HashMap, type alist, type amap, type aset, type aval,
} from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, Rot3d, Scale3d, Shift3d } from "@aardworx/wombat.base";
import type { Effect } from "@aardworx/wombat.shader";
import type {
  BlendState, BufferView, DrawCall,
} from "@aardworx/wombat.rendering/core";
import type {
  EventHandlers,
  SgNode,
  SgLeaf,
  TrafoValue,
  UniformBag,
} from "./sg.js";

// ---------------------------------------------------------------------------
// Leaves and groups
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

/** Ordered group. Accepts an alist or anything liftable to one. */
function group(children: alist<SgNode> | ReadonlyArray<SgNode> | SgNode): SgNode {
  if (Array.isArray(children)) {
    return { kind: "Group", children: AList.ofList(children as SgNode[]) };
  }
  if ((children as alist<SgNode>).getReader !== undefined) {
    return { kind: "Group", children: children as alist<SgNode> };
  }
  return { kind: "Group", children: AList.ofList([children as SgNode]) };
}

/** Unordered (state-sortable) group. */
function unordered(children: aset<SgNode> | ReadonlyArray<SgNode>): SgNode {
  if (Array.isArray(children)) {
    return { kind: "UnorderedGroup", children: ASet.ofList(children as SgNode[]) };
  }
  return { kind: "UnorderedGroup", children: children as aset<SgNode> };
}

/** Single-slot adaptive subtree swap. */
function adaptive(child: aval<SgNode>): SgNode {
  return { kind: "AdaptiveGroup", child };
}

// ---------------------------------------------------------------------------
// Attribute scopes
// ---------------------------------------------------------------------------

function trafo(value: TrafoValue, child: SgNode): SgNode {
  return { kind: "Trafo", value, child };
}

function shader(effect: Effect, child: SgNode): SgNode {
  return { kind: "Shader", effect, child };
}

function uniformBag(entries: Record<string, unknown | aval<unknown>>): UniformBag {
  let map = HashMap.empty<string, aval<unknown>>();
  for (const [k, v] of Object.entries(entries)) {
    map = map.add(k, isAval(v) ? v as aval<unknown> : AVal.constant(v));
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

/** Set both view and proj at once — Sg.view + Sg.proj wrapped. */
function camera(view: aval<Trafo3d>, proj: aval<Trafo3d>, child: SgNode): SgNode {
  return viewScope(view, projScope(proj, child));
}

/**
 * Escape hatch — build a sub-tree from the fully accumulated
 * `TraversalState` at this point in the walk. The creator runs
 * once per traversal (not per delta); embed `Sg.adaptive` /
 * `aval`-driven children inside the returned node for runtime
 * updates.
 */
function delay(
  create: (state: import("./traversalState.js").TraversalState) => SgNode,
): SgNode {
  return { kind: "Delay", create };
}

// ---------------------------------------------------------------------------
// Trafo helpers — return Trafo3d (or aval<Trafo3d>) for use in arrays.
// Composition-by-array semantics is the consumer's job (see `composeTrafoValue`).
// ---------------------------------------------------------------------------

const identity: Trafo3d = Trafo3d.identity;

function translate(v: V3d): Trafo3d;
function translate(v: aval<V3d>): aval<Trafo3d>;
function translate(v: V3d | aval<V3d>): Trafo3d | aval<Trafo3d> {
  return isAval(v)
    ? (v as aval<V3d>).map((vv) => Trafo3d.translation(vv as V3d))
    : Trafo3d.translation(v as V3d);
}

function scale(s: number | V3d | Scale3d): Trafo3d;
function scale(s: aval<number | V3d | Scale3d>): aval<Trafo3d>;
function scale(s: number | V3d | Scale3d | aval<number | V3d | Scale3d>): Trafo3d | aval<Trafo3d> {
  if (isAval(s)) return (s as aval<number | V3d | Scale3d>).map((v) => Trafo3d.scaling(v as number));
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
  return isAval(a) ? (a as aval<unknown>).map(lift) : lift(a);
}

function trafoOf(t: Trafo3d): Trafo3d { return t; }

// ---------------------------------------------------------------------------
// Identity-aval helper (kept private)
// ---------------------------------------------------------------------------

function isAval(v: unknown): boolean {
  return typeof v === "object" && v !== null && typeof (v as { getValue?: unknown }).getValue === "function";
}

// ---------------------------------------------------------------------------
// Public namespace
// ---------------------------------------------------------------------------

/**
 * Pure data constructors for `SgNode` plus the trafo-helper family.
 * The JSX wrappers (`<Sg>` / `<Sg.Box/>`) call into these.
 */
export const Sg = {
  // nodes
  empty,
  leaf,
  group,
  unordered,
  adaptive,
  // attribute scopes
  trafo,
  shader,
  uniform,
  blendMode,
  cursor,
  pickThrough,
  on,
  active,
  view: viewScope,
  proj: projScope,
  camera,
  delay,
  // trafo helpers
  identity,
  translate,
  scale,
  rotate,
  trafoOf,
} as const;

void Shift3d; // currently unused but kept for completeness when adding Shift overloads
