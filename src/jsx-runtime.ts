// JSX runtime entry — referenced by `tsconfig.json`'s
// `"jsxImportSource": "@aardworx/wombat.dom"`. The compiler lowers
// `<h1 class={c}>{x}</h1>` to `jsx("h1", { class: c, children: x })`.
// We just package the call into a plain VNode object; mounting is
// done later by `mount(...)`.

import {
  FRAGMENT,
  type ComponentFn,
  type ElementVNode,
  type ComponentVNode,
  type FragmentVNode,
  type Props,
  type VNode,
} from "./vnode.js";

export { FRAGMENT as Fragment } from "./vnode.js";
export type { JSX } from "./jsx-types.js";

type JsxType = string | ComponentFn | typeof FRAGMENT;

function makeNode(type: JsxType, props: Props): VNode {
  if (type === FRAGMENT) {
    return { _tag: "fragment", props } satisfies FragmentVNode;
  }
  if (typeof type === "string") {
    return { _tag: "element", tag: type, props } satisfies ElementVNode;
  }
  return { _tag: "component", type, props } satisfies ComponentVNode;
}

export function jsx(type: JsxType, props: Props, _key?: string): VNode {
  return makeNode(type, props);
}

export function jsxs(type: JsxType, props: Props, _key?: string): VNode {
  return makeNode(type, props);
}

export function jsxDEV(
  type: JsxType,
  props: Props,
  _key?: string,
  _isStaticChildren?: boolean,
  _source?: unknown,
  _self?: unknown,
): VNode {
  return makeNode(type, props);
}
