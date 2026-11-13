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
  private _parent: Scope | undefined;
  private _depth = 0;

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
    c._depth = this._depth + 1;
    if (this.region !== undefined) c.region = this.region;
    this.add(c);
    if (!this._disposed) c._parent = this;
    return c;
  }

  /** How many scopes enclose this one (a mount root: 0). A subtree that
   *  appears and goes (a list row, a conditional's branch) is a child
   *  scope of the binding that adds and removes it: deeper than it. */
  get depth(): number {
    return this._depth;
  }

  get isDisposed(): boolean {
    return this._disposed;
  }

  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    // A child disposed on its own (an alist row, a swapped arm) leaves its
    // parent's list, or long-lived parents keep every dead child.
    const p = this._parent;
    if (p !== undefined) {
      this._parent = undefined;
      if (!p._disposed) {
        const i = p._disposables.lastIndexOf(this);
        if (i >= 0) p._disposables.splice(i, 1);
      }
    }
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

// The active-scope stack is process-global, NOT module-local. A bundler
// (notably Vite's optimizeDeps) can pre-bundle `@aardworx/wombat.dom` and
// the `/scene` subpath into SEPARATE chunks, each with its own copy of
// this module — so `mountComponent`'s `pushScope` (main chunk) and a
// scene component's `useScope` (scene chunk) would otherwise touch
// different stacks, and `useScope()` throws "outside a component body"
// even though a scope is active. Anchoring the stack on `globalThis`
// makes every duplicated copy share one stack. (The same applies to any
// component reading `scope.region` — the unified event region only
// reaches scene components because they now find the pushed scope.)
const SCOPE_STACK_KEY = "__wombat_dom_scope_stack__";
const scopeStack: Scope[] =
  ((globalThis as Record<string, unknown>)[SCOPE_STACK_KEY] as Scope[] | undefined) ??
  ((globalThis as Record<string, unknown>)[SCOPE_STACK_KEY] = [] as Scope[]);

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
