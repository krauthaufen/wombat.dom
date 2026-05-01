// Text-or-element child slot. When the child is an aval whose value
// is sometimes a string and sometimes an element, we maintain a
// single placeholder position and swap the contents in place.

import {
  AdaptiveToken,
  type IDisposable,
  type aval,
} from "@aardworx/wombat.adaptive";
import type { Scope } from "../scope.js";
import type { UIScheduler, Binding } from "../scheduler.js";
import { mountInto } from "../mount.js";
import type { VNode } from "../vnode.js";
import { isVNode } from "../vnode.js";

function asTextNode(value: unknown): Text | null {
  if (value === null || value === undefined || value === false || value === true) {
    return document.createTextNode("");
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "bigint") {
    return document.createTextNode(String(value));
  }
  return null;
}

/**
 * Insert a node-or-tree representing `value` into `parent` just
 * before `before`. All resulting subscriptions and cleanups are
 * registered on `scope` — disposing the scope removes the content
 * from the DOM.
 */
export function mountChildValue(
  parent: Node,
  before: Node | null,
  value: unknown,
  scope: Scope,
  scheduler: UIScheduler,
): void {
  if (isVNode(value)) {
    mountInto(parent, before, value as VNode, scope, scheduler);
    return;
  }
  const txt = asTextNode(value) ?? document.createTextNode(String(value));
  if (before === null) parent.appendChild(txt);
  else parent.insertBefore(txt, before);
  scope.onDispose(() => {
    if (txt.parentNode === parent) parent.removeChild(txt);
  });
}

/**
 * An adaptive child slot driven by an aval. Represented in the DOM
 * as a single anchor comment node; on each value change the previous
 * subtree is disposed and a fresh one mounted just before the
 * anchor.
 */
export function bindAValChild(
  parent: Node,
  before: Node | null,
  v: aval<unknown>,
  scope: Scope,
  scheduler: UIScheduler,
): void {
  const anchor = document.createComment("aval");
  if (before === null) parent.appendChild(anchor);
  else parent.insertBefore(anchor, before);
  scope.onDispose(() => {
    if (anchor.parentNode === parent) parent.removeChild(anchor);
  });

  let current: Scope | null = null;

  const apply = (value: unknown): void => {
    if (current !== null) {
      current.dispose();
      current = null;
    }
    const child = scope.child();
    mountChildValue(parent, anchor, value, child, scheduler);
    current = child;
  };

  const adaptive = v as unknown as {
    getValue(t: AdaptiveToken): unknown;
    addMarkingCallback(cb: () => void): IDisposable;
  };

  // initial sync render
  apply(adaptive.getValue(AdaptiveToken.top));

  const binding: Binding = {
    flush(t) {
      apply(adaptive.getValue(t));
    },
  };
  const disp = adaptive.addMarkingCallback(() => scheduler.notify(binding));
  scope.add({
    dispose: () => {
      scheduler.forget(binding);
      disp.dispose();
    },
  });
}
