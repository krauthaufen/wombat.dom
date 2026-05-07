// colorAval — turn an `aval<V4f>` into a `BufferView` that
// broadcasts the four r/g/b/a floats to every vertex. Mirrors
// Aardvark.Dom's `SingleValueBuffer` path; under the new
// BufferView shape this is exactly `BufferView.ofValue(color, "v4f")`.
//
// The view's `singleValue` field carries the original `aval<V4f>`,
// letting the backend lower it to a uniform binding when supported.
// As a fallback the view's `buffer` aval re-packs the V4f bytes on
// each tick (16 bytes; negligible).

import type { aval } from "@aardworx/wombat.adaptive";
import type { V4f } from "@aardworx/wombat.base";
import { BufferView, ElementType} from "@aardworx/wombat.rendering/core";

export function colorAval(color: aval<V4f> | V4f): BufferView {
  return BufferView.ofValue(color, ElementType.V4f);
}
