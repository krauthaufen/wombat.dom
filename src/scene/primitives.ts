// Imperative leaf builders for the bundled primitives. The actual
// geometry lives in `./primitives/shared.ts` (one allocation per
// primitive type, shared across instances). Per-instance colour is
// fed via `colorAval(...)` (a stride-0 single-value V4f buffer).
//
// `box()` / `quad()` here remain for backwards compatibility with
// callers that imported them off the scene barrel; they delegate to
// the shared-geometry path.

import {
  AVal, HashMap, type aval,
} from "@aardworx/wombat.adaptive";
import { V3d, V4f } from "@aardworx/wombat.base";
import {
  IBuffer,
  type BufferView, type DrawCall,
} from "@aardworx/wombat.rendering/core";
import type { SgLeaf } from "./sg.js";
import { getBoxGeometry } from "./primitives/shared.js";
import { colorAval } from "./primitives/colorBuffer.js";

const WHITE = new V4f(1, 1, 1, 1);

// ---------------------------------------------------------------------------
// Box (single-colour via stride-0 a_color)
// ---------------------------------------------------------------------------

export interface BoxOptions {
  /** Half-extents along each axis. Default `(1, 1, 1)` → box from -1 to +1. */
  readonly size?: V3d;
  /** Single colour for the whole box. Default white. */
  readonly color?: V4f | aval<V4f>;
}

/**
 * Single-colour cube. 36 verts × 6 faces with face-normals; colour
 * is fed via a stride-0 vertex buffer that all 36 vertices read.
 *
 * Vertex attributes:
 *   - `a_position : V3f` (shared with all other Box instances)
 *   - `a_normal   : V3f`
 *   - `a_color    : V4f` (stride-0 broadcast)
 *
 * `size` scales the unit cube spanning `[0,1]³` to span `[-size, +size]`.
 * Identity (default `1,1,1`) yields the `[-1,1]³` cube the old
 * vertex-coloured `box()` used.
 */
export function box(opts: BoxOptions = {}): SgLeaf {
  const s = opts.size ?? new V3d(1, 1, 1);
  const handle = getBoxGeometry();

  // Old `box()` returned positions in `[-size, +size]`; the new
  // shared geometry is `[0,1]`. Stretch + recentre per-instance by
  // building a bespoke positions BufferView. (Hot-path callers
  // shouldn't pass size at all, hitting the shared-buffer fast-path
  // below; explicit-size users get a fresh per-instance buffer.)
  let vertexAttrs = handle.vertexAttrs;
  if (s.x !== 1 || s.y !== 1 || s.z !== 1) {
    const sharedPos = AVal.force(handle.vertexAttrs.tryFind("a_position")!);
    const baseHost = sharedPos.buffer;
    if (baseHost.kind !== "host") throw new Error("box: shared positions buffer is not host-backed");
    const baseFloat = new Float32Array(baseHost.data instanceof ArrayBuffer
      ? baseHost.data
      : (baseHost.data.buffer as ArrayBuffer),
      baseHost.data instanceof ArrayBuffer ? 0 : baseHost.data.byteOffset,
      baseHost.data instanceof ArrayBuffer ? baseHost.data.byteLength / 4 : baseHost.data.byteLength / 4,
    );
    const scaled = new Float32Array(baseFloat.length);
    for (let i = 0; i < baseFloat.length; i += 3) {
      scaled[i]     = (baseFloat[i]!     * 2 - 1) * s.x;
      scaled[i + 1] = (baseFloat[i + 1]! * 2 - 1) * s.y;
      scaled[i + 2] = (baseFloat[i + 2]! * 2 - 1) * s.z;
    }
    const posView: BufferView = {
      buffer: IBuffer.fromHost(scaled.buffer),
      offset: 0, count: baseFloat.length / 3, stride: 12, format: "float32x3",
    };
    vertexAttrs = handle.vertexAttrs.add("a_position", AVal.constant(posView));
  }

  const colorView = colorAval(opts.color ?? WHITE);
  vertexAttrs = vertexAttrs.add("a_color", colorView);

  return {
    kind: "Leaf",
    vertexAttributes: vertexAttrs,
    indices: handle.indices,
    drawCall: handle.drawCall,
  };
}

// ---------------------------------------------------------------------------
// Quad (XY plane, z=0) — single-colour via stride-0 a_color
// ---------------------------------------------------------------------------

export interface QuadOptions {
  /** Half-extents along x and y. Default `(1, 1)`. */
  readonly width?: number;
  readonly height?: number;
  /** Single colour. Default white. */
  readonly color?: V4f | aval<V4f>;
}

/** Single-colour quad in the XY plane, z=0. 4 verts, 6 indices. */
export function quad(opts: QuadOptions = {}): SgLeaf {
  const w = opts.width ?? 1;
  const h = opts.height ?? 1;
  const positions = new Float32Array([
    -w, -h, 0,
     w, -h, 0,
     w,  h, 0,
    -w,  h, 0,
  ]);
  const normals = new Float32Array([
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
  ]);
  const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
  const colorView = colorAval(opts.color ?? WHITE);
  return {
    kind: "Leaf",
    vertexAttributes: HashMap.empty<string, aval<BufferView>>()
      .add("a_position", AVal.constant({
        buffer: IBuffer.fromHost(positions.buffer), offset: 0, count: 4, stride: 12, format: "float32x3",
      } satisfies BufferView))
      .add("a_normal", AVal.constant({
        buffer: IBuffer.fromHost(normals.buffer), offset: 0, count: 4, stride: 12, format: "float32x3",
      } satisfies BufferView))
      .add("a_color", colorView),
    indices: AVal.constant({
      buffer: IBuffer.fromHost(indices.buffer), offset: 0, count: 6, stride: 4, format: "uint32",
    } satisfies BufferView),
    drawCall: AVal.constant({
      kind: "indexed",
      indexCount: 6, instanceCount: 1, firstIndex: 0, baseVertex: 0, firstInstance: 0,
    } satisfies DrawCall),
  };
}
