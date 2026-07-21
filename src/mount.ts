// mount(rootEl, vnode) — top-level entry. Walks the VNode tree,
// creates DOM, attaches adaptive bindings under a fresh root scope.
// Returns a Disposable that tears the whole subtree down.

import { installTapEvents } from "./tap.js";
import { Scope, pushScope, popScope } from "./scope.js";
import { RegionRouter } from "./eventRouter.js";
import { UIScheduler, defaultScheduler } from "./scheduler.js";
import {
  isVNode,
  type VNode,
  type ElementVNode,
  type ComponentVNode,
  type FragmentVNode,
  type Props,
} from "./vnode.js";
import { bindAttr } from "./bindings/attr.js";
import { bindChildren } from "./bindings/children.js";
import type { JsxResult } from "./jsx-runtime.js";

export interface MountResult {
  scope: Scope;
  dispose(): void;
}

export interface MountOptions {
  scheduler?: UIScheduler;
}

/**
 * Mount a VNode tree into `root`. The new content is appended; any
 * existing children are left in place. Returns a Disposable; calling
 * `dispose()` removes everything this mount added and unsubscribes
 * all adaptive listeners it created.
 */
export function mount(
  root: Node,
  vnode: VNode | JsxResult,
  opts: MountOptions = {},
): MountResult {
  // Global tap/dbltap synthesis (Aardvark.Dom parity) — idempotent, so every
  // mount can just ask for it and `onTap` works anywhere in the tree.
  installTapEvents();
  // `vnode: VNode | JsxResult` — the union widens to include scene
  // `SgNode`s only because `JSX.Element` does (one element type for
  // every JSX expression). A scene tree isn't mountable on its own;
  // reject it with a clear message rather than producing garbage.
  if (!("_tag" in vnode)) {
    throw new Error(
      "mount: expected a JSX VNode (HTML), got a scene SgNode -- " +
      "scene trees must be a child of <RenderControl>, not mounted directly",
    );
  }
  const scheduler = opts.scheduler ?? defaultScheduler;
  const scope = new Scope();
  // Unified event region: wombat owns propagation inside the subtree it
  // generates. `attr.ts` routes `on*` through this instead of native
  // listeners; a `<RenderControl>` canvas registers as the async scene
  // leaf. Needs an Element root to host the capture listeners — a
  // non-Element root (rare) falls back to native listeners.
  if (root instanceof Element) {
    const router = new RegionRouter(root);
    scope.region = router;
    scope.onDispose(() => router.dispose());
  }
  mountInto(root, null, vnode, scope, scheduler);
  return {
    scope,
    dispose: () => scope.dispose(),
  };
}

/**
 * Mount `vnode` into `parent` immediately before `before` (or
 * appended if `before` is null). All cleanups attach to `scope`.
 */
export function mountInto(
  parent: Node,
  before: Node | null,
  vnode: VNode,
  scope: Scope,
  scheduler: UIScheduler,
): void {
  if (vnode._tag === "fragment") {
    mountFragment(parent, before, vnode, scope, scheduler);
    return;
  }
  if (vnode._tag === "component") {
    mountComponent(parent, before, vnode, scope, scheduler);
    return;
  }
  mountElement(parent, before, vnode, scope, scheduler);
}

function mountElement(
  parent: Node,
  before: Node | null,
  vnode: ElementVNode,
  scope: Scope,
  scheduler: UIScheduler,
): void {
  const el = document.createElement(vnode.tag);
  bindElementProps(el, vnode.props, scope, scheduler);
  if (before === null) parent.appendChild(el);
  else parent.insertBefore(el, before);
  scope.onDispose(() => {
    if (el.parentNode === parent) parent.removeChild(el);
  });
  // children mount as direct children of the new element
  bindChildren(el, null, vnode.props.children, scope, scheduler);
}

function mountFragment(
  parent: Node,
  before: Node | null,
  vnode: FragmentVNode,
  scope: Scope,
  scheduler: UIScheduler,
): void {
  bindChildren(parent, before, vnode.props.children, scope, scheduler);
}

function mountComponent(
  parent: Node,
  before: Node | null,
  vnode: ComponentVNode,
  scope: Scope,
  scheduler: UIScheduler,
): void {
  // Run the component function with `scope` exposed via the
  // module-level stack so `useScope()` works inside the body.
  pushScope(scope);
  let result: unknown;
  try {
    result = vnode.type(vnode.props);
  } finally {
    popScope();
  }
  if (result === null || result === undefined || result === false || result === true) {
    return;
  }
  if (isVNode(result)) {
    mountInto(parent, before, result, scope, scheduler);
    return;
  }
  // Component returned a non-VNode (string/number/aval/alist/array).
  // Funnel through children dispatch.
  bindChildren(parent, before, result, scope, scheduler);
}

function bindElementProps(
  el: Element,
  props: Props,
  scope: Scope,
  scheduler: UIScheduler,
): void {
  for (const [name, value] of Object.entries(props)) {
    if (name === "children") continue;
    if (name === "ref" && typeof value === "function") {
      (value as (e: Element) => void)(el);
      continue;
    }
    bindAttr(el, name, value, scope, scheduler);
  }
}
