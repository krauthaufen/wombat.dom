// Public barrel for `@aardworx/wombat.dom`.

export { mount, mountInto, type MountResult, type MountOptions } from "./mount.js";
export { Scope, useScope, type Disposable } from "./scope.js";
export {
  installTapEvents,
  TAP_MAX_DURATION_MS, TAP_MAX_MOVE_PX,
  DOUBLE_TAP_GAP_MS, DOUBLE_TAP_MOVE_PX,
  type TapEvent, type TapEventExtras,
} from "./tap.js";
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
