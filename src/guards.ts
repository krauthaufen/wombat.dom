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
} from "@aardworx/adaptive";

export function isAVal<T = unknown>(x: unknown): x is aval<T> {
  return x instanceof AbstractVal || x instanceof ChangeableValue;
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
