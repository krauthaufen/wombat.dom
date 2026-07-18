// The build-time M3 front half, runtime side.
//
// The Fable plugin (`SgRowFactoryAttribute` in wombat.fable) rewrites
// the mapping-function argument of collection combinators
// (`AMap.map`, `ASet.map`, …) into `rowWrap("File.fs:line", f)`. The
// wrapper is fully VALUE-DRIVEN — the F# type system can't gate the
// rewrite (`ISceneNode = obj` in the bindings), so every mapped value
// passes through and only actual scene nodes are considered:
//
//   - non-node results pass through verbatim (cost: one kind check);
//   - nodes whose template can't fast-row pass through verbatim — the
//     classic path keeps them exactly as expensive as today, no more;
//   - fast-row-capable nodes are STAGED ONCE, the tree is DROPPED,
//     and a thin `SgRow {staged, build}` takes its place. The
//     collection caches upstream (AMap.map / ASet.map readers) now
//     retain ~3 fields instead of the whole SgNode subtree, and
//     row lowering consumes `staged` directly — no per-pass restage.
//
// Fallback correctness: `build` re-runs the user's mapping function,
// so every consumer that needs a real tree can materialize one
// (`materializeRowSg`, cached). A rebuilt tree is re-staged by its
// consumer; the construction-time `staged` is only ever paired with
// the fast path, which never materializes.

import type { SgNode, SgRow } from "./sg.js";
import { SG_KINDS } from "./sgVNode.js";
import { stageNode, withSgSource } from "./template.js";

/** Structural scene-node check — a `kind` from the registry. A user
 *  object that happens to collide is handled downstream: staging is
 *  try/caught and declines to the verbatim value. */
function isSgNodeLike(v: unknown): v is SgNode {
  return v !== null && typeof v === "object"
    && typeof (v as { kind?: unknown }).kind === "string"
    && SG_KINDS.has((v as { kind: string }).kind);
}

let _rowNodesConstructed = 0;
/** Stat: values `rowWrap` replaced with thin `SgRow` nodes. */
export function rowNodesConstructed(): number { return _rowNodesConstructed; }
/** Test hook. */
export function resetRowNodesConstructed(): void { _rowNodesConstructed = 0; }

/**
 * Wrap a collection mapping function (the plugin's target). `arity`
 * is the DECLARED argument count of the decorated member's mapping
 * function (`AMap.map` → 2, `ASet.map` → 1, …) — the plugin knows it
 * statically, which makes the wrapper deterministic against every
 * function shape Fable can emit at the boundary: uncurried
 * `(k, v) => node`, curried `k => v => node`, and curry-adapters
 * invoking the wrapper one argument at a time. Mapping RESULTS that
 * are themselves functions are never misread as partial application —
 * only fewer-than-`arity` invocations continue collecting.
 */
export function rowWrap(
  loc: string,
  arity: number,
  f: (...args: unknown[]) => unknown,
): (...args: unknown[]) => unknown {
  const call = (args: unknown[]): unknown => {
    let r: unknown = f;
    let i = 0;
    while (i < args.length) {
      const g = r as (...xs: unknown[]) => unknown;
      const n = typeof g === "function" && g.length > 0
        ? Math.min(g.length, args.length - i)
        : args.length - i;
      r = g(...args.slice(i, i + n));
      i += n;
    }
    return r;
  };
  const wrapper = (...args: unknown[]): unknown => {
    if (args.length < arity) {
      // curried invocation by a Fable adapter — keep collecting
      return (...more: unknown[]) => wrapper(...args, ...more);
    }
    const r = call(args);
    if (!isSgNodeLike(r) || r.kind === "Row") return r;
    return maybeRowNode(loc, r, () => call(args) as SgNode);
  };
  return wrapper;
}

/**
 * Replace `node` with a thin pre-staged `SgRow` when its template is
 * fast-row-capable; otherwise return it verbatim. Never throws —
 * staging failures decline to the verbatim node.
 */
export function maybeRowNode(
  loc: string | undefined,
  node: SgNode,
  build: () => SgNode,
): SgNode {
  try {
    const staged = stageNode(node);
    if (staged.template.fastRow === undefined || staged.template.hasDynamicUniforms) {
      return node;
    }
    _rowNodesConstructed++;
    const row: SgRow = { kind: "Row", staged, build };
    return loc !== undefined ? withSgSource(row, loc) : row;
  } catch {
    return node;
  }
}
