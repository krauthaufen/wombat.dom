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

export function isAVal<T = unknown>(x: unknown): x is aval<T> {
  if (x instanceof AbstractVal || x instanceof ChangeableValue) return true;
  // A constant (`AVal.constant`) is a class of its own that wombat.adaptive does not export: it is an aval by what
  // it does. Taken for a plain value, an attribute bound to one was written as its text ("constval(…)").
  const candidate = x as { getValue?: unknown; getValueUntyped?: unknown; force?: unknown; addCallback?: unknown } | null;
  return typeof candidate === "object" && candidate !== null
    && typeof candidate.getValue === "function" && typeof candidate.getValueUntyped === "function"
    && typeof candidate.force === "function" && typeof candidate.addCallback === "function";
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
