// Adaptive list children. Each row is keyed by its `Index`. We mount
// the row's value into a per-row scope and place its DOM nodes in
// the parent. A per-row marker comment lets us locate (and later
// detach) the row's segment in O(row-size).
//
// Insertion at a fresh Index is O(log N): we ask the row map for the
// next-greater Index's marker via MapExt.neighbours and insertBefore
// the new marker there.
//
// Removal is O(row-size) — we walk siblings from this row's marker
// until we hit the next row's marker (or the end-anchor).

import {
  AdaptiveToken,
  IndexOps,
  MapExt,
  type IIndexListReader,
  type IDisposable,
  type Index,
  type alist,
} from "@aardworx/wombat.adaptive";
import { Scope } from "../scope.js";
import type { UIScheduler, Binding } from "../scheduler.js";
import { mountChildValue } from "./text.js";

interface Row {
  marker: Comment;
  scope: Scope;
}

const indexCmp = (a: Index, b: Index): number => a.compareTo(b);

export function bindAlistChildren(
  parent: Node,
  before: Node | null,
  list: alist<unknown>,
  scope: Scope,
  scheduler: UIScheduler,
): void {
  const endAnchor = document.createComment("alist/end");
  if (before === null) parent.appendChild(endAnchor);
  else parent.insertBefore(endAnchor, before);
  scope.onDispose(() => {
    if (endAnchor.parentNode === parent) parent.removeChild(endAnchor);
  });

  let rows = MapExt.empty<Index, Row>(indexCmp);
  const reader = (list as alist<unknown>).getReader() as IIndexListReader<unknown>;

  const removeRowAt = (idx: Index): void => {
    const row = rows.tryFind(idx);
    if (row === undefined) return;
    const nb = rows.neighbours(idx);
    const stop: Node = nb.right !== undefined ? nb.right[1].marker : endAnchor;
    let n: Node | null = row.marker;
    while (n !== null && n !== stop) {
      const next: Node | null = n.nextSibling;
      parent.removeChild(n);
      n = next;
    }
    row.scope.dispose();
    rows = rows.remove(idx);
  };

  const insertRowAt = (idx: Index, value: unknown): void => {
    const nb = rows.neighbours(idx);
    const insertBeforeNode: Node =
      nb.right !== undefined ? nb.right[1].marker : endAnchor;
    const marker = document.createComment("alist/row");
    parent.insertBefore(marker, insertBeforeNode);
    const rowScope = scope.child();
    mountChildValue(parent, insertBeforeNode, value, rowScope, scheduler);
    rows = rows.add(idx, { marker, scope: rowScope });
  };

  const apply = (token: AdaptiveToken): void => {
    const delta = reader.getChanges(token);
    for (const [idx, op] of delta) {
      if (op.tag === "Set") {
        // Replace if present, else insert.
        if (rows.tryFind(idx) !== undefined) removeRowAt(idx);
        insertRowAt(idx, op.value);
      } else {
        removeRowAt(idx);
      }
    }
  };

  // initial pull
  apply(AdaptiveToken.top);

  const adaptive = reader as unknown as {
    addMarkingCallback(cb: () => void): IDisposable;
  };
  const binding: Binding = { flush: (t) => apply(t) };
  const disp = adaptive.addMarkingCallback(() => scheduler.notify(binding));
  scope.add({
    dispose: () => {
      scheduler.forget(binding);
      disp.dispose();
      // dispose all per-row scopes (will also remove their content;
      // but the parent endAnchor cleanup handled the boundary)
      for (const [, r] of rows) r.scope.dispose();
      rows = MapExt.empty(indexCmp);
    },
  });
}

// Avoid unused import warnings in strict mode if IndexOps isn't used
// here — re-export to keep the side-effect import live for tree-
// shaking analysis.
void IndexOps;
