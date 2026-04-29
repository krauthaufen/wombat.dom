// Single-attribute binding. Dispatches on prop name and value type:
//   - on*       => event listener (replaceable)
//   - style     => string OR object of CSS subkeys
//   - class     => alias of className; just sets the attribute
//   - boolean   => toggle attribute
//   - null/undef => remove attribute
//   - other     => setAttribute / property assignment as appropriate

import {
  AdaptiveToken,
  type IDisposable,
  type aval,
} from "@aardworx/adaptive";
import { isAVal } from "../guards.js";
import type { Scope } from "../scope.js";
import type { UIScheduler } from "../scheduler.js";
import type { Binding } from "../scheduler.js";

type EventLike = EventListenerOrEventListenerObject;

const PROPERTY_ATTRS = new Set([
  "value",
  "checked",
  "selected",
  "disabled",
  "readOnly",
  "indeterminate",
]);

function setOne(el: Element, name: string, value: unknown): void {
  if (value === null || value === undefined || value === false) {
    el.removeAttribute(name);
    if (name in el) {
      try {
        (el as unknown as Record<string, unknown>)[name] = false;
      } catch {
        /* readonly property — ignore */
      }
    }
    return;
  }
  if (value === true) {
    el.setAttribute(name, "");
    if (name in el) {
      try {
        (el as unknown as Record<string, unknown>)[name] = true;
      } catch {
        /* readonly property — ignore */
      }
    }
    return;
  }
  if (name === "style") {
    if (typeof value === "string") {
      (el as HTMLElement).style.cssText = value;
    } else if (typeof value === "object") {
      const s = (el as HTMLElement).style;
      for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
        s.setProperty(k, v == null ? "" : String(v));
      }
    }
    return;
  }
  if (name === "className" || name === "class") {
    el.setAttribute("class", String(value));
    return;
  }
  if (name === "htmlFor") {
    el.setAttribute("for", String(value));
    return;
  }
  if (PROPERTY_ATTRS.has(name)) {
    try {
      (el as unknown as Record<string, unknown>)[name] = value;
    } catch {
      el.setAttribute(name, String(value));
    }
    return;
  }
  el.setAttribute(name, String(value));
}

export function bindAttr(
  el: Element,
  name: string,
  value: unknown,
  scope: Scope,
  scheduler: UIScheduler,
): void {
  // Event listener — replaceable. Re-binding (when the value is an
  // aval) swaps the previous listener.
  if (name.length > 2 && name.startsWith("on") && name[2]! >= "A" && name[2]! <= "Z") {
    bindEvent(el, name, value, scope);
    return;
  }
  if (isAVal(value)) {
    bindAValAttr(el, name, value as aval<unknown>, scope, scheduler);
    return;
  }
  setOne(el, name, value);
}

function bindEvent(
  el: Element,
  name: string,
  value: unknown,
  scope: Scope,
): void {
  const evt = name.slice(2).toLowerCase();
  let cur: EventLike | undefined;
  const listener: EventListener = (e) => {
    if (cur === undefined) return;
    if (typeof cur === "function") cur(e);
    else cur.handleEvent(e);
  };
  el.addEventListener(evt, listener);
  scope.onDispose(() => el.removeEventListener(evt, listener));

  if (isAVal(value)) {
    const v = value as aval<unknown>;
    bindAValSink(scope, v, (latest) => {
      cur = latest as EventLike | undefined;
    });
  } else {
    cur = value as EventLike | undefined;
  }
}

function bindAValAttr(
  el: Element,
  name: string,
  v: aval<unknown>,
  scope: Scope,
  scheduler: UIScheduler,
): void {
  const binding: Binding = {
    flush(token: AdaptiveToken) {
      const cur = (v as unknown as { getValue(t: AdaptiveToken): unknown }).getValue(token);
      setOne(el, name, cur);
    },
  };
  // initial set
  binding.flush(AdaptiveToken.top);
  // subscribe via marking callback → scheduler.notify
  const disp = (v as unknown as { addMarkingCallback(cb: () => void): IDisposable }).addMarkingCallback(
    () => scheduler.notify(binding),
  );
  scope.add({
    dispose: () => {
      scheduler.forget(binding);
      disp.dispose();
    },
  });
}

/**
 * Subscribe `sink` to an aval, with the same rAF-batched semantics as
 * attr bindings. Used for events and other "set the latest into a
 * mutable slot" cases.
 */
export function bindAValSink<T>(
  scope: Scope,
  v: aval<T>,
  sink: (value: T) => void,
): void {
  const adaptive = v as unknown as {
    getValue(t: AdaptiveToken): T;
    addMarkingCallback(cb: () => void): IDisposable;
  };
  // initial value
  sink(adaptive.getValue(AdaptiveToken.top));
  // re-fetch on every mark; events don't need rAF batching since
  // they're set-once mutable slots.
  const disp = adaptive.addMarkingCallback(() => {
    sink(adaptive.getValue(AdaptiveToken.top));
  });
  scope.add(disp);
}
