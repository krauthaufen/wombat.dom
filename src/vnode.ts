import type { aval, alist, amap } from "@aardworx/adaptive";

export const FRAGMENT = Symbol.for("adaptive-ui.Fragment");
export type FragmentTag = typeof FRAGMENT;

export type AttrValue =
  | string
  | number
  | boolean
  | null
  | undefined
  | EventListenerOrEventListenerObject
  | Record<string, string | number | null | undefined>;

export type AdaptiveAttr = AttrValue | aval<AttrValue>;

export type Child =
  | VNode
  | string
  | number
  | bigint
  | boolean
  | null
  | undefined
  | aval<Child>
  | alist<Child>
  | Iterable<Child>;

export type Props = {
  children?: Child | Child[];
  [key: string]: unknown;
};

export type ComponentFn<P extends Props = Props> = (props: P) => VNode | Child;

export interface ElementVNode {
  readonly _tag: "element";
  readonly tag: string;
  readonly props: Props;
}

export interface ComponentVNode {
  readonly _tag: "component";
  readonly type: ComponentFn;
  readonly props: Props;
}

export interface FragmentVNode {
  readonly _tag: "fragment";
  readonly props: Props;
}

export type VNode = ElementVNode | ComponentVNode | FragmentVNode;

export const isVNode = (x: unknown): x is VNode => {
  if (x === null || typeof x !== "object") return false;
  const t = (x as { _tag?: unknown })._tag;
  return t === "element" || t === "component" || t === "fragment";
};

/**
 * For props on elements: an "attribute spread" can be either a plain
 * record, an aval of one, or an amap<string, AttrValue> for keyed
 * incremental updates of attributes.
 */
export type AttrSpread =
  | Record<string, AttrValue>
  | aval<Record<string, AttrValue>>
  | amap<string, AttrValue>;
