import {
  AbstractVal,
  ChangeableValue,
  AbstractAset,
  AbstractAlist,
  AbstractAmap,
  type aval,
  type aset,
  type alist,
  type amap,
} from "@aardworx/wombat.adaptive";

/** A constant aval (`AVal.constant`): a class of its own, neither an `AbstractVal` nor exported, known by what every
 *  adaptive object carries (`isConstant`) and by what it does. The bindings follow it as any aval: its marking
 *  callbacks never run (wombat.adaptive 0.3.7), and before they were missing, so an attribute given a constant was
 *  written as its text ("constval(…)"). */
function isConstantVal(x: unknown): boolean {
  const candidate = x as { isConstant?: unknown; getValue?: unknown; getValueUntyped?: unknown; getCached?: unknown } | null;
  return typeof candidate === "object" && candidate !== null && candidate.isConstant === true
    && typeof candidate.getValue === "function" && typeof candidate.getValueUntyped === "function" && typeof candidate.getCached === "function";
}

export function isAVal<T = unknown>(x: unknown): x is aval<T> {
  return x instanceof AbstractVal || x instanceof ChangeableValue || isConstantVal(x);
}

export function isAList<T = unknown>(x: unknown): x is alist<T> {
  return x instanceof AbstractAlist;
}

export function isASet<T = unknown>(x: unknown): x is aset<T> {
  return x instanceof AbstractAset;
}

export function isAMap<K = unknown, V = unknown>(x: unknown): x is amap<K, V> {
  return x instanceof AbstractAmap;
}
