// An effect of a value on an element that is not an attribute: a fetch
// whose url is the value, a dialog opened or closed, an event sent. `fn`
// runs with the value now and after each change, as a binding of the
// scheduler: in the frame flush, outer bindings first (`Binding.depth`), so
// when the change also removes the element (a conditional turned false) the
// removal disposes the effect before it runs, and it never runs with a value
// that only held while the element was shown. It ends with `scope`.

import { AdaptiveToken, type IDisposable, type aval } from "@aardworx/wombat.adaptive";
import type { Scope } from "../scope.js";
import type { Binding, UIScheduler } from "../scheduler.js";

export function bindEffect<T>(
  value: aval<T>,
  fn: (value: T) => void,
  scope: Scope,
  scheduler: UIScheduler,
): void {
  const v = value as unknown as {
    getValue(t: AdaptiveToken): T;
    addMarkingCallback(cb: () => void): IDisposable;
    isConstant?: boolean;
  };
  fn(v.getValue(AdaptiveToken.top));
  if (v.isConstant === true) return;
  const binding: Binding = { depth: scope.depth, flush: (t) => fn(v.getValue(t)) };
  const disp = v.addMarkingCallback(() => scheduler.notify(binding));
  scope.add({
    dispose: () => {
      scheduler.forget(binding);
      disp.dispose();
    },
  });
}
