// A Scope owns a set of Disposables (subscriptions, child scopes,
// mounted DOM ops). Disposing a scope cascades to its children and
// runs cleanups in LIFO order.

import type { EventRegion } from "./eventRouter.js";

export interface Disposable {
  dispose(): void;
}

export class Scope implements Disposable {
  private _disposables: Disposable[] = [];
  private _disposed = false;

  /**
   * The unified event region (RegionRouter) this scope belongs to, set on
   * the mount root and inherited by every child scope. `attr.ts` reads it
   * to route `on*` handlers through the region walk instead of native
   * `addEventListener`. Undefined for scopes created outside a mount
   * (e.g. bare unit tests) — those fall back to native listeners.
   */
  region?: EventRegion;

  /** Register a cleanup to run on dispose. */
  add(d: Disposable): void {
    if (this._disposed) {
      d.dispose();
      return;
    }
    this._disposables.push(d);
  }

  /** Convenience: register a cleanup function. */
  onDispose(fn: () => void): void {
    this.add({ dispose: fn });
  }

  /** Open a child scope. Disposing this scope disposes the child. The
   *  event `region` is inherited so handlers bound in dynamically-added
   *  subtrees (alist rows, aval child swaps) still join the walk. */
  child(): Scope {
    const c = new Scope();
    if (this.region !== undefined) c.region = this.region;
    this.add(c);
    return c;
  }

  get isDisposed(): boolean {
    return this._disposed;
  }

  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    const ds = this._disposables;
    this._disposables = [];
    for (let i = ds.length - 1; i >= 0; i--) {
      try {
        ds[i]!.dispose();
      } catch (err) {
        // never throw out of dispose — log and continue
        console.error("[wombat.dom] scope cleanup threw", err);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Component-scope stack
//
// `mountComponent` pushes the active scope before calling the
// component function and pops after. Component bodies can call
// `useScope()` to grab the scope for cleanup registration —
// matches the React-hook calling convention but without React's
// render-pass machinery (we just have one initial run).
// ---------------------------------------------------------------------------

const scopeStack: Scope[] = [];

/** @internal — used by `mount.ts`'s `mountComponent`. */
export function pushScope(scope: Scope): void {
  scopeStack.push(scope);
}

/** @internal — used by `mount.ts`'s `mountComponent`. */
export function popScope(): void {
  scopeStack.pop();
}

/**
 * Inside a component body: returns the scope this mount belongs
 * to. Throws when called outside a component (e.g. from a top-
 * level module init). Use `scope.onDispose(...)` to register
 * cleanup.
 */
export function useScope(): Scope {
  const top = scopeStack[scopeStack.length - 1];
  if (top === undefined) {
    throw new Error("useScope() called outside a component body");
  }
  return top;
}
