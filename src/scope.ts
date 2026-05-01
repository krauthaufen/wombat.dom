// A Scope owns a set of Disposables (subscriptions, child scopes,
// mounted DOM ops). Disposing a scope cascades to its children and
// runs cleanups in LIFO order.

export interface Disposable {
  dispose(): void;
}

export class Scope implements Disposable {
  private _disposables: Disposable[] = [];
  private _disposed = false;

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

  /** Open a child scope. Disposing this scope disposes the child. */
  child(): Scope {
    const c = new Scope();
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
