// JSX runtime entry — referenced by `tsconfig.json`'s
// `"jsxImportSource": "@aardworx/wombat.dom"`. The compiler lowers
// `<h1 class={c}>{x}</h1>` to `jsx("h1", { class: c, children: x })`.
//
// HTML elements / fragments / user components package into a plain
// `VNode` (mounted later by `mount(...)`). Scene constructors —
// `<Sg ...>`, `<Sg.Box/>`, … — are function components flagged with
// `__isSg`; for those we call the component eagerly and return its
// `SgNode` as-is. No carrier-VNode bridge, and an `<Sg.Box/>` subtree
// is a plain value, so it can be referenced in several places (the
// scene graph is a DAG, not a tree).

import {
  FRAGMENT,
  type ComponentFn,
  type ElementVNode,
  type ComponentVNode,
  type FragmentVNode,
  type Props,
  type VNode,
} from "./vnode.js";
import type { SgNode } from "./scene/sg.js";

export { FRAGMENT as Fragment } from "./vnode.js";
export type { JSX } from "./jsx-types.js";

/** Anything a JSX expression can evaluate to in this runtime. */
export type JsxResult = VNode | SgNode;

// `(props: never) => unknown` accepts any function: `never` is the
// bottom type so it's contravariantly a supertype of any concrete
// props type, and `unknown` covariantly accepts any return — so both
// a plain `ComponentFn` (→ VNode|Child) and an `__isSg` scene
// constructor (`(props: SgScopeProps) => SgNode`) are assignable.
type JsxType = string | ((props: never) => unknown) | typeof FRAGMENT;

function isSgComponent(t: JsxType): t is (props: Props) => SgNode {
  return typeof t === "function" && (t as { __isSg?: unknown }).__isSg === true;
}

function makeNode(type: JsxType, props: Props): JsxResult {
  if (isSgComponent(type)) return type(props);
  if (type === FRAGMENT) {
    return { _tag: "fragment", props } satisfies FragmentVNode;
  }
  if (typeof type === "string") {
    return { _tag: "element", tag: type, props } satisfies ElementVNode;
  }
  return { _tag: "component", type: type as ComponentFn, props } satisfies ComponentVNode;
}

export function jsx(type: JsxType, props: Props, _key?: string): JsxResult {
  return makeNode(type, props);
}

export function jsxs(type: JsxType, props: Props, _key?: string): JsxResult {
  return makeNode(type, props);
}

export function jsxDEV(
  type: JsxType,
  props: Props,
  _key?: string,
  _isStaticChildren?: boolean,
  _source?: unknown,
  _self?: unknown,
): JsxResult {
  return makeNode(type, props);
}
