// colorAval — turn an `aval<V4f>` into an `aval<BufferView>` that
// binds the four r/g/b/a floats as a single-element vertex buffer
// with `stride: 0` (read by every vertex). Mirrors Aardvark.Dom's
// `SingleValueBuffer` path.
//
// The wombat.rendering layer's `prepareAdaptiveBuffer` re-uploads
// when the IBuffer aval's host data changes — we expose a derived
// `aval<BufferView>` whose `buffer` is rebuilt from a fresh
// `Float32Array` each time the upstream colour ticks (16 bytes,
// negligible). The stride: 0 in the BufferView is the load-bearing
// field consumed by `prepareRenderObject` to set
// `GPUVertexBufferLayout.arrayStride = 0`.

import {
  AVal, type aval,
} from "@aardworx/wombat.adaptive";
import type { V4f } from "@aardworx/wombat.base";
import { IBuffer, type BufferView } from "@aardworx/wombat.rendering/core";

export function colorAval(color: aval<V4f> | V4f): aval<BufferView> {
  const src: aval<V4f> = isAVal(color) ? color : AVal.constant(color);
  return src.map((c): BufferView => {
    const arr = new Float32Array(4);
    arr[0] = c.x; arr[1] = c.y; arr[2] = c.z; arr[3] = c.w;
    return {
      buffer: IBuffer.fromHost(arr.buffer),
      offset: 0,
      count: 1,
      stride: 0,
      format: "float32x4",
    };
  });
}

function isAVal<T>(v: T | aval<T>): v is aval<T> {
  return typeof v === "object" && v !== null && typeof (v as { getValue?: unknown }).getValue === "function";
}
