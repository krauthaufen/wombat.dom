// Idle registration of eager `Sg.active` subtrees.
//
// An eager subtree (`Sg.active(a, child, { eager })`) is valid while off and
// is built and handed to the renderer (its draws inactive) before its first
// show, so that show is a flag flip. Doing that for every one at load makes
// the load pay for all of them; instead each is released in idle time after
// it, a few milliseconds at a time, in the order they were lowered: its
// subtree is lowered in the slice, and the renderer registers it with the
// next frame. One shown before its turn is lowered at once, as a subtree
// without `eager` is: correct, just not a flag flip.

import { AVal, transact, cval, type aval } from "@aardworx/wombat.adaptive";

/** One eager subtree waiting for its turn. */
export interface IdleTicket {
  /** True once released (its subtree may be built). */
  readonly ready: cval<boolean>;
  /** Its subtree: read right after its release, in the slice, which
   *  builds it there (outside the render path). */
  target: aval<unknown> | undefined;
  /** Dropped: its lowering left the scene. */
  cancelled: boolean;
}

interface Deadline { timeRemaining(): number }

const queue: IdleTicket[] = [];
let head = 0;
let scheduled = false;
let released = 0;
let doneAt: number | undefined;
const doneListeners: (() => void)[] = [];

/** Settings and state of idle registration (process-wide). */
export const idleRegistration = {
  /** Release eager subtrees in idle time; false: built at once (lowered
   *  with the scene, as before). */
  enabled: true,
  /** Upper bound of one slice, in ms (an idle period's own deadline may
   *  be shorter). */
  budgetMs: 4,
  /** Waiting subtrees. */
  get pending(): number {
    let n = 0;
    for (let i = head; i < queue.length; i++) if (!queue[i]!.cancelled) n++;
    return n;
  },
  /** Subtrees released so far. */
  get released(): number { return released; },
  /** When the queue last ran empty after releasing (`performance.now()`). */
  get doneAt(): number | undefined { return doneAt; },
  /** Resolves once nothing waits (at once if nothing does). */
  whenDone(): Promise<void> {
    if (head >= queue.length) return Promise.resolve();
    return new Promise((r) => doneListeners.push(r));
  },
  /** Releases waiting subtrees now, all or the first `n` (tests). */
  runNow(n = Infinity): void { run(n, Infinity); },
};

// (reachable from a page for measuring: pending, released, doneAt)
(globalThis as { __wombatIdleRegistration?: unknown }).__wombatIdleRegistration = idleRegistration;

/** A ticket for an eager subtree, released in turn. */
export function enqueueIdle(): IdleTicket {
  const ticket: IdleTicket = { ready: cval(false), target: undefined, cancelled: false };
  queue.push(ticket);
  schedule();
  return ticket;
}

const requestIdle: (cb: (d: Deadline) => void) => void =
  typeof (globalThis as { requestIdleCallback?: unknown }).requestIdleCallback === "function"
    ? (cb) => (globalThis as unknown as { requestIdleCallback: (cb: (d: Deadline) => void) => void }).requestIdleCallback(cb)
    : (cb) => { setTimeout(() => cb({ timeRemaining: () => idleRegistration.budgetMs }), 1); };

function schedule(): void {
  if (scheduled) return;
  scheduled = true;
  requestIdle((d) => {
    scheduled = false;
    run(Infinity, Math.min(idleRegistration.budgetMs, Math.max(1, d.timeRemaining())));
  });
}

function run(count: number, budgetMs: number): void {
  const now = (): number => (typeof performance !== "undefined" ? performance.now() : Date.now());
  const end = now() + budgetMs;
  let n = 0;
  while (head < queue.length && n < count) {
    const t = queue[head]!;
    queue[head++] = undefined as unknown as IdleTicket;
    if (t.cancelled) continue;
    released++;
    n++;
    transact(() => { t.ready.value = true; });
    // a subtree that raises while off is left to its first show (the
    // renderer's guard reads it again then)
    try { if (t.target !== undefined) AVal.force(t.target); } catch { /* left until shown */ }
    if (now() >= end) break;
  }
  if (head >= queue.length) {
    if (n > 0) doneAt = now();
    queue.length = 0;
    head = 0;
    const ls = doneListeners.splice(0);
    for (const l of ls) l();
  } else {
    if (head > 4096 && head * 2 > queue.length) { queue.splice(0, head); head = 0; }
    if (count === Infinity && budgetMs !== Infinity) schedule();
  }
}
