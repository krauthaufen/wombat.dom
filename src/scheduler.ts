// UIScheduler — coalesces marking notifications from many adaptive
// inputs into a single requestAnimationFrame flush. Each Binding
// registers itself once; when its input becomes outdated, the
// binding's onMark calls scheduler.notify(this), which adds it to a
// dirty Set and (if not already pending) schedules a frame. On the
// frame, the scheduler iterates only the dirty set and asks each
// binding to flush — pulling current values / deltas under a fresh
// AdaptiveToken.

import { AdaptiveToken, type aval } from "@aardworx/wombat.adaptive";

export interface Binding {
  /** Pull the latest from this binding's input and apply to the DOM. */
  flush(token: AdaptiveToken): void;
  /** The depth of the scope it binds in (`Scope.depth`; absent: 0). A
   *  flush runs outer bindings before inner ones: the binding that removes
   *  a subtree (a conditional turned false, a row removed) runs before the
   *  bindings inside it, which it disposes, so they never pull a value
   *  that only held while the subtree was shown. */
  readonly depth?: number;
}

/** `bindings` outer before inner (by `depth`), in their order otherwise;
 *  the same array when it already is. */
export function outerFirst(bindings: Binding[]): Binding[] {
  let prev = -Infinity;
  let ordered = true;
  for (const b of bindings) {
    const d = b.depth ?? 0;
    if (d < prev) { ordered = false; break; }
    prev = d;
  }
  if (ordered) return bindings;
  // a stable sort: same depth, same order
  return bindings
    .map((b, i) => ({ b, i, d: b.depth ?? 0 }))
    .sort((x, y) => x.d - y.d || x.i - y.i)
    .map((x) => x.b);
}

/**
 * How long one binding may keep building rows in a frame before it leaves
 * the rest to the next one (`bindAlistChildren`): a large insertion then
 * shows its first rows at once and the page stays responsive while the
 * others follow. Below this nothing changes: an update that fits is applied
 * whole, as always.
 */
export const SLICE_BUDGET_MS = 50;

export class UIScheduler {
  private _dirty = new Set<Binding>();
  /** The batch currently being flushed (null outside flush). `forget`
   *  must also delete from THIS set: a binding flushed earlier in the
   *  batch can dispose a subtree whose bindings are queued later in the
   *  same batch — flushing those would apply DOM ops against detached
   *  anchors (NotFoundError: insertBefore). Deleting from a Set during
   *  for..of skips not-yet-visited entries, which is exactly right. */
  private _flushing: Set<Binding> | null = null;
  private _frame: number | null = null;
  private _disposed = false;
  private _synchronous = false;
  private readonly _scheduleFrame: (cb: () => void) => number;
  private readonly _cancelFrame: (handle: number) => void;

  constructor(opts?: {
    schedule?: (cb: () => void) => number;
    cancel?: (handle: number) => void;
  }) {
    this._scheduleFrame =
      opts?.schedule ??
      ((cb) => {
        if (typeof requestAnimationFrame !== "undefined") {
          return requestAnimationFrame(cb);
        }
        // SSR / non-browser fallback
        return setTimeout(cb, 16) as unknown as number;
      });
    this._cancelFrame =
      opts?.cancel ??
      ((h) => {
        if (typeof cancelAnimationFrame !== "undefined") cancelAnimationFrame(h);
        else clearTimeout(h);
      });
  }

  /** Mark a binding dirty. Schedules a frame flush if none is pending. */
  notify(b: Binding): void {
    if (this._disposed) return;
    this._dirty.add(b);
    if (this._frame === null) {
      this._frame = this._scheduleFrame(() => this.flush());
    }
  }

  /** Forget a binding (e.g. on dispose). Effective even mid-flush:
   *  the binding is removed from the in-flight batch too, so a
   *  disposed binding never flushes. */
  forget(b: Binding): void {
    this._dirty.delete(b);
    this._flushing?.delete(b);
  }

  /** Run any pending flush immediately and synchronously. Bindings apply
   *  everything they have (`synchronous`): nothing is left to a later
   *  frame. */
  flushNow(): void {
    if (this._frame !== null) {
      this._cancelFrame(this._frame);
      this._frame = null;
    }
    this._synchronous = true;
    try { this.flush(); } finally { this._synchronous = false; }
  }

  /** True during `flushNow`: a binding that spreads large work over
   *  frames must finish it in this flush. */
  get synchronous(): boolean { return this._synchronous; }

  /** The clock bindings measure their frame budget with (ms). */
  now(): number {
    return typeof performance !== "undefined" ? performance.now() : Date.now();
  }

  private flush(): void {
    this._frame = null;
    if (this._disposed) return;
    const work = this._dirty;
    this._dirty = new Set();
    this._flushing = work;
    const tok = AdaptiveToken.top;
    try {
      // outer before inner (`Binding.depth`); a binding forgotten meanwhile
      // (its subtree disposed by one flushed earlier) is left out
      const order = work.size > 1 ? outerFirst([...work]) : [...work];
      for (const b of order) {
        if (!work.has(b)) continue;
        try {
          b.flush(tok);
        } catch (err) {
          console.error("[wombat.dom] binding flush threw", err);
        }
      }
    } finally {
      this._flushing = null;
    }
  }

  dispose(): void {
    this._disposed = true;
    if (this._frame !== null) {
      this._cancelFrame(this._frame);
      this._frame = null;
    }
    this._dirty.clear();
  }
}

/** A default singleton scheduler shared across mounts. */
export const defaultScheduler = new UIScheduler();

/**
 * The scheduler of a subtree that may be hidden (`deferUpdates` on an
 * element): while hidden, its bindings' notifications are parked instead of
 * scheduled; when shown, the parked bindings flush once — each pulls its
 * net change since it last ran (docs/deferred-updates.md). While shown it
 * only forwards to its parent: a visible subtree costs what it did.
 */
export class DeferringScheduler extends UIScheduler {
  private _parked = new Set<Binding>();
  /** The parked bindings being flushed by `_set` (see `UIScheduler._flushing`). */
  private _releasing: Set<Binding> | null = null;
  private _hidden: boolean;

  constructor(
    private readonly _parent: UIScheduler,
    hidden: boolean | aval<boolean>,
    owner: { add(d: { dispose(): void }): void },
  ) {
    super();
    if (typeof hidden === "boolean") {
      this._hidden = hidden;
      return;
    }
    if (hidden.isConstant) {
      // a constant switch: decided once (no edge to watch)
      this._hidden = hidden.getValue(AdaptiveToken.top);
      return;
    }
    this._hidden = hidden.getValue(AdaptiveToken.top);
    // the switch is watched on the parent: it must flush while this hides
    const watch: Binding = { flush: (t) => this._set(hidden.getValue(t)) };
    const cb = (hidden as unknown as { addMarkingCallback(f: () => void): { dispose(): void } })
      .addMarkingCallback(() => _parent.notify(watch));
    owner.add({ dispose: () => { _parent.forget(watch); cb.dispose(); this._parked.clear(); } });
  }

  override notify(b: Binding): void {
    if (this._hidden) this._parked.add(b);
    else this._parent.notify(b);
  }

  override forget(b: Binding): void {
    this._parked.delete(b);
    this._releasing?.delete(b);
    this._parent.forget(b);
  }

  override flushNow(): void {
    this._parent.flushNow();
  }

  override get synchronous(): boolean { return this._parent.synchronous; }

  override now(): number { return this._parent.now(); }

  private _set(hidden: boolean): void {
    this._hidden = hidden;
    if (hidden || this._parked.size === 0) return;
    const releasing = this._parked;
    this._parked = new Set();
    this._releasing = releasing;
    const tok = AdaptiveToken.top;
    try {
      // as `UIScheduler.flush`: outer before inner, the forgotten left out
      const order = releasing.size > 1 ? outerFirst([...releasing]) : [...releasing];
      for (const b of order) {
        if (!releasing.has(b)) continue;
        try {
          b.flush(tok);
        } catch (err) {
          console.error("[wombat.dom] binding flush threw", err);
        }
      }
    } finally {
      this._releasing = null;
    }
  }
}
