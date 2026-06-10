// UIScheduler — coalesces marking notifications from many adaptive
// inputs into a single requestAnimationFrame flush. Each Binding
// registers itself once; when its input becomes outdated, the
// binding's onMark calls scheduler.notify(this), which adds it to a
// dirty Set and (if not already pending) schedules a frame. On the
// frame, the scheduler iterates only the dirty set and asks each
// binding to flush — pulling current values / deltas under a fresh
// AdaptiveToken.

import { AdaptiveToken } from "@aardworx/wombat.adaptive";

export interface Binding {
  /** Pull the latest from this binding's input and apply to the DOM. */
  flush(token: AdaptiveToken): void;
}

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

  /** Run any pending flush immediately and synchronously. */
  flushNow(): void {
    if (this._frame !== null) {
      this._cancelFrame(this._frame);
      this._frame = null;
    }
    this.flush();
  }

  private flush(): void {
    this._frame = null;
    if (this._disposed) return;
    const work = this._dirty;
    this._dirty = new Set();
    this._flushing = work;
    const tok = AdaptiveToken.top;
    try {
      for (const b of work) {
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
