// JSX namespace — TypeScript reads this when
// `jsxImportSource: "@aardworx/wombat.dom"` is set. We keep it
// permissive: every intrinsic element accepts any prop, but the prop
// values can be either plain or `aval<plain>`. Children accept plain
// values, avals, alists, or arrays/iterables.

import type { aval, alist, amap } from "@aardworx/wombat.adaptive";
import type { VNode } from "./vnode.js";

// `aval`/`alist` here are `any`-parameterized intentionally: the JSX
// position doesn't constrain the element type (the runtime dispatches
// on the delta shape), and TS treats `alist<T>` invariantly through
// its trace machinery — so anything narrower (e.g. `alist<VNode>`)
// would fail to assign to `alist<unknown>`.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ChildLike =
  | VNode
  | string
  | number
  | bigint
  | boolean
  | null
  | undefined
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | aval<any>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | alist<any>
  | Iterable<unknown>;

type EventName = `on${string}`;

type AdaptivePrimitive = string | number | boolean | null | undefined;

type AdaptiveOr<T> = T | aval<T>;

interface CommonAttrs {
  id?: AdaptiveOr<string>;
  class?: AdaptiveOr<string>;
  className?: AdaptiveOr<string>;
  style?: AdaptiveOr<string | Record<string, AdaptivePrimitive>>;
  title?: AdaptiveOr<string>;
  hidden?: AdaptiveOr<boolean>;
  tabIndex?: AdaptiveOr<number>;
  role?: AdaptiveOr<string>;
  "aria-label"?: AdaptiveOr<string>;
  ref?: (el: Element) => void;
  /** Spread of attributes via amap<string, value>. */
  attrs?: amap<string, AdaptivePrimitive | EventListenerOrEventListenerObject>;
  children?: ChildLike | ChildLike[];
  [key: `data-${string}`]: AdaptiveOr<AdaptivePrimitive>;
  [key: EventName]: EventListenerOrEventListenerObject | undefined;
}

interface InputAttrs extends CommonAttrs {
  type?: AdaptiveOr<string>;
  value?: AdaptiveOr<string | number>;
  checked?: AdaptiveOr<boolean>;
  placeholder?: AdaptiveOr<string>;
  disabled?: AdaptiveOr<boolean>;
  readOnly?: AdaptiveOr<boolean>;
  name?: AdaptiveOr<string>;
  required?: AdaptiveOr<boolean>;
  min?: AdaptiveOr<number | string>;
  max?: AdaptiveOr<number | string>;
  step?: AdaptiveOr<number | string>;
  pattern?: AdaptiveOr<string>;
  autocomplete?: AdaptiveOr<string>;
}

interface ButtonAttrs extends CommonAttrs {
  type?: AdaptiveOr<"button" | "submit" | "reset">;
  disabled?: AdaptiveOr<boolean>;
  name?: AdaptiveOr<string>;
  value?: AdaptiveOr<string>;
}

interface SelectAttrs extends CommonAttrs {
  value?: AdaptiveOr<string>;
  disabled?: AdaptiveOr<boolean>;
  name?: AdaptiveOr<string>;
  multiple?: AdaptiveOr<boolean>;
}

interface OptionAttrs extends CommonAttrs {
  value?: AdaptiveOr<string>;
  selected?: AdaptiveOr<boolean>;
  disabled?: AdaptiveOr<boolean>;
}

interface FormAttrs extends CommonAttrs {
  action?: AdaptiveOr<string>;
  method?: AdaptiveOr<string>;
  noValidate?: AdaptiveOr<boolean>;
}

interface AnchorAttrs extends CommonAttrs {
  href?: AdaptiveOr<string>;
  target?: AdaptiveOr<string>;
  rel?: AdaptiveOr<string>;
  download?: AdaptiveOr<string>;
}

interface LabelAttrs extends CommonAttrs {
  for?: AdaptiveOr<string>;
  htmlFor?: AdaptiveOr<string>;
}

export namespace JSX {
  export type Element = VNode;
  export interface ElementChildrenAttribute {
    children: object;
  }

  export interface IntrinsicElements {
    // structural / generic
    div: CommonAttrs;
    span: CommonAttrs;
    section: CommonAttrs;
    article: CommonAttrs;
    header: CommonAttrs;
    footer: CommonAttrs;
    main: CommonAttrs;
    nav: CommonAttrs;
    aside: CommonAttrs;
    p: CommonAttrs;
    h1: CommonAttrs;
    h2: CommonAttrs;
    h3: CommonAttrs;
    h4: CommonAttrs;
    h5: CommonAttrs;
    h6: CommonAttrs;
    hr: CommonAttrs;
    br: CommonAttrs;
    pre: CommonAttrs;
    code: CommonAttrs;
    em: CommonAttrs;
    strong: CommonAttrs;
    small: CommonAttrs;
    b: CommonAttrs;
    i: CommonAttrs;
    u: CommonAttrs;
    // lists
    ul: CommonAttrs;
    ol: CommonAttrs;
    li: CommonAttrs;
    dl: CommonAttrs;
    dt: CommonAttrs;
    dd: CommonAttrs;
    // form / interactive
    form: FormAttrs;
    input: InputAttrs;
    button: ButtonAttrs;
    select: SelectAttrs;
    option: OptionAttrs;
    optgroup: CommonAttrs;
    textarea: InputAttrs;
    label: LabelAttrs;
    fieldset: CommonAttrs;
    legend: CommonAttrs;
    // anchors / media
    a: AnchorAttrs;
    img: CommonAttrs & {
      src?: AdaptiveOr<string>;
      alt?: AdaptiveOr<string>;
      width?: AdaptiveOr<number | string>;
      height?: AdaptiveOr<number | string>;
    };
    canvas: CommonAttrs & {
      width?: AdaptiveOr<number | string>;
      height?: AdaptiveOr<number | string>;
    };
    // table
    table: CommonAttrs;
    thead: CommonAttrs;
    tbody: CommonAttrs;
    tfoot: CommonAttrs;
    tr: CommonAttrs;
    td: CommonAttrs;
    th: CommonAttrs;
    caption: CommonAttrs;
    colgroup: CommonAttrs;
    col: CommonAttrs;
  }
}
