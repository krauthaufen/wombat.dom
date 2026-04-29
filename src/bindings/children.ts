// Children-dispatch: takes whatever ended up in `props.children`
// and mounts each piece appropriately. JSX always passes children
// as either a single value or an array.

import type { Scope } from "../scope.js";
import type { UIScheduler } from "../scheduler.js";
import { isAList, isAVal } from "../guards.js";
import { mountChildValue, bindAValChild } from "./text.js";
import { bindAlistChildren } from "./alist.js";

export function bindChildren(
  parent: Node,
  before: Node | null,
  children: unknown,
  scope: Scope,
  scheduler: UIScheduler,
): void {
  if (children === undefined || children === null) return;
  if (Array.isArray(children)) {
    for (const c of children) bindChildren(parent, before, c, scope, scheduler);
    return;
  }
  if (isAList(children)) {
    bindAlistChildren(parent, before, children, scope, scheduler);
    return;
  }
  if (isAVal(children)) {
    bindAValChild(parent, before, children, scope, scheduler);
    return;
  }
  // Iterable that isn't an array/aval/alist (e.g. a Set, generator)
  // — expand once.
  if (
    typeof children === "object" &&
    children !== null &&
    Symbol.iterator in (children as object)
  ) {
    for (const c of children as Iterable<unknown>) {
      bindChildren(parent, before, c, scope, scheduler);
    }
    return;
  }
  // primitive / VNode / coerce
  mountChildValue(parent, before, children, scope, scheduler);
}
