// Public barrel for `@aardworx/adaptive-ui`.

export { mount, mountInto, type MountResult, type MountOptions } from "./mount.js";
export { Scope, type Disposable } from "./scope.js";
export {
  UIScheduler,
  defaultScheduler,
  type Binding,
} from "./scheduler.js";
export { Fragment, jsx, jsxs, jsxDEV } from "./jsx-runtime.js";
export type {
  VNode,
  ElementVNode,
  ComponentVNode,
  FragmentVNode,
  Props,
  Child,
  AttrValue,
  AdaptiveAttr,
  AttrSpread,
  ComponentFn,
} from "./vnode.js";
export { isAVal, isAList, isASet, isAMap } from "./guards.js";
