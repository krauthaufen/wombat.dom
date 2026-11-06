// Lifetimes of lowered scene subtrees.
//
// Lowering acquires things that must be given back when the lowered subtree
// leaves the render tree: pick ids (PickRegistry) and the scopes handed to
// `Sg.delay` creators. Both go through one mechanism. Lowering is
// synchronous, so a single stack works as the sink: whatever is acquired
// while an owner `collect`s lands above the owner's mark and becomes the
// owner's to release. Owners are the places a lowered subtree can leave:
//
//   - compileScene's root (the whole compiled scene, see
//     `CompileSceneOptions.lifetime`),
//   - each child of an ordered or unordered group (removal),
//   - the current subtree of an adaptive group or a dynamic uniform scope
//     (swap).
//
// An owner that is itself released releases everything it still holds, so
// disposing a compiled scene ends every scope below it.

import type { PickRegistry } from "./picking/registry.js";
import type { SgDelay, SgNode } from "./sg.js";
import type { TraversalState } from "./traversalState.js";

/** What an `Sg.delay` creator gets with the traversal state: the lifetime
 *  of the subtree it builds. It ends when that lowered subtree leaves the
 *  render tree (removed from its group, swapped out, or the compiled scene
 *  disposed). */
export interface SceneScope {
  /** Runs `fn` when the scope ends; at once if it already has. */
  onDispose(fn: () => void): void;
  readonly isDisposed: boolean;
}

export class LoweringScope implements SceneScope {
  private _fns: (() => void)[] | undefined = [];

  onDispose(fn: () => void): void {
    if (this._fns === undefined) { runSafely(fn); return; }
    this._fns.push(fn);
  }

  get isDisposed(): boolean { return this._fns === undefined; }

  dispose(): void {
    const fns = this._fns;
    if (fns === undefined) return;
    this._fns = undefined;
    for (let i = fns.length - 1; i >= 0; i--) runSafely(fns[i]!);
  }
}

function runSafely(fn: () => void): void {
  try { fn(); } catch (err) { console.error("[wombat.dom] scene cleanup threw", err); }
}

/** A release: a pick id (kept as a number, cheap at heap scale) or a
 *  callback. */
type Release = number | (() => void);

const _stack: Release[] = [];

/** Hands a release to the innermost collecting owner. */
export function own(r: Release): void {
  _stack.push(r);
}

export const NOOP_DISPOSE = (): void => {};

/**
 * Runs `f` and returns its result with a `dispose` that gives back
 * everything acquired meanwhile (pick ids through `registry`). `dispose`
 * runs once; it is `NOOP_DISPOSE` when nothing was acquired. If `f` throws,
 * what it acquired so far is released before rethrowing.
 */
export function collect<T>(registry: PickRegistry | undefined, f: () => T): { value: T; dispose: () => void } {
  const mark = _stack.length;
  let value: T;
  try {
    value = f();
  } catch (e) {
    release(registry, _stack.splice(mark));
    throw e;
  }
  if (_stack.length === mark) return { value, dispose: NOOP_DISPOSE };
  const mine = _stack.splice(mark);
  let done = false;
  return {
    value,
    dispose: () => {
      if (done) return;
      done = true;
      release(registry, mine);
    },
  };
}

function release(registry: PickRegistry | undefined, rs: Release[]): void {
  for (let i = rs.length - 1; i >= 0; i--) {
    const r = rs[i]!;
    if (typeof r === "number") registry?.release(r);
    else runSafely(r);
  }
}

/** The disposers of the children an owner currently holds; releasing the
 *  owner releases them all. */
export class LiveSet {
  private readonly _live = new Set<() => void>();
  add(dispose: () => void): void { if (dispose !== NOOP_DISPOSE) this._live.add(dispose); }
  remove(dispose: () => void): void { this._live.delete(dispose); }
  /** Releases one child. */
  release(dispose: () => void): void { this._live.delete(dispose); dispose(); }
  /** Releases every child still held. */
  releaseAll(): void {
    const all = [...this._live];
    this._live.clear();
    for (const d of all) d();
  }
}

/**
 * Runs a Delay's creator as a PROBE: for a scan that only looks at the
 * returned subtree and never renders it (pass discovery, instancing
 * validation, leaf visits, the render control's camera sniffing). Its
 * scope has ended when the creator returns, so anything registered on it
 * runs at once.
 */
export function probeDelay(node: SgDelay, state: TraversalState): SgNode {
  const scope = new LoweringScope();
  try { return node.create(state, scope); }
  finally { scope.dispose(); }
}
