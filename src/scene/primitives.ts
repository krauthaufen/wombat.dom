// Primitive `SgLeaf` builders — minimal geometry helpers that
// produce vertex-attributed meshes ready to feed
// `DefaultSurfaces.basic`.
//
// Initial set is intentionally small: enough to build a hello-cube
// demo. More primitives (sphere, cone, torus, custom) and per-
// vertex normals/uvs come later as needs arise.

import {
  AVal, HashMap,
  type aval,
} from "@aardworx/wombat.adaptive";
import { V3d } from "@aardworx/wombat.base";
import {
  IBuffer,
  type BufferView, type DrawCall,
} from "@aardworx/wombat.rendering/core";
import type { SgLeaf } from "./sg.js";

// ---------------------------------------------------------------------------
// Box (vertex-coloured)
// ---------------------------------------------------------------------------

export interface BoxOptions {
  /** Half-extents along each axis. Default `(1, 1, 1)` → box from -1 to +1. */
  readonly size?: V3d;
}

/**
 * Vertex-coloured cube. 8 unique vertices at the corners with
 * per-corner RGB colours; 36 triangle-list indices wrapping the
 * faces in CCW (front-facing for `cullMode: "back"`).
 *
 * Vertex attributes:
 *   - `a_position : V3f`  — corner positions, scaled by `size`.
 *   - `a_color    : V3f`  — corner colours (rainbow).
 *
 * Plug into `DefaultSurfaces.basic` (or any effect that takes
 * those attribute names) inside an `<Sg Shader=…>` scope.
 */
export function box(opts: BoxOptions = {}): SgLeaf {
  const s = opts.size ?? new V3d(1, 1, 1);
  const sx = s.x, sy = s.y, sz = s.z;

  // 8 corner positions × 3 floats each.
  const positions = new Float32Array([
    -sx, -sy, -sz,   // 0
     sx, -sy, -sz,   // 1
     sx,  sy, -sz,   // 2
    -sx,  sy, -sz,   // 3
    -sx, -sy,  sz,   // 4
     sx, -sy,  sz,   // 5
     sx,  sy,  sz,   // 6
    -sx,  sy,  sz,   // 7
  ]);
  // Rainbow palette per corner.
  const colors = new Float32Array([
    0, 0, 0,    1, 0, 0,    1, 1, 0,    0, 1, 0,
    0, 0, 1,    1, 0, 1,    1, 1, 1,    0, 1, 1,
  ]);
  // 36 indices (12 triangles × 3). CCW winding.
  const indices = new Uint32Array([
    // -Z face (0, 1, 2, 3)
    0, 2, 1,   0, 3, 2,
    // +Z face (4, 5, 6, 7)
    4, 5, 6,   4, 6, 7,
    // -Y face (0, 1, 5, 4)
    0, 1, 5,   0, 5, 4,
    // +Y face (3, 7, 6, 2)
    3, 6, 2,   3, 7, 6,
    // -X face (0, 4, 7, 3)
    0, 7, 3,   0, 4, 7,
    // +X face (1, 2, 6, 5)
    1, 2, 6,   1, 6, 5,
  ]);

  return {
    kind: "Leaf",
    vertexAttributes: HashMap.empty<string, aval<BufferView>>()
      .add("a_position", AVal.constant({
        buffer: IBuffer.fromHost(positions),
        offset: 0, count: 8, stride: 12, format: "float32x3",
      } satisfies BufferView))
      .add("a_color", AVal.constant({
        buffer: IBuffer.fromHost(colors),
        offset: 0, count: 8, stride: 12, format: "float32x3",
      } satisfies BufferView)),
    indices: AVal.constant({
      buffer: IBuffer.fromHost(indices),
      offset: 0, count: 36, stride: 4, format: "uint32",
    } satisfies BufferView),
    drawCall: AVal.constant({
      kind: "indexed",
      indexCount: 36, instanceCount: 1, firstIndex: 0, baseVertex: 0, firstInstance: 0,
    } satisfies DrawCall),
  };
}

// ---------------------------------------------------------------------------
// Quad (2D, in XY plane at z=0)
// ---------------------------------------------------------------------------

export interface QuadOptions {
  /** Half-extents along x and y. Default `(1, 1)`. */
  readonly width?: number;
  readonly height?: number;
}

/** Vertex-coloured quad in the XY plane, z=0. 4 verts, 6 indices. */
export function quad(opts: QuadOptions = {}): SgLeaf {
  const w = opts.width ?? 1;
  const h = opts.height ?? 1;
  const positions = new Float32Array([
    -w, -h, 0,
     w, -h, 0,
     w,  h, 0,
    -w,  h, 0,
  ]);
  const colors = new Float32Array([
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
    1, 1, 0,
  ]);
  const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
  return {
    kind: "Leaf",
    vertexAttributes: HashMap.empty<string, aval<BufferView>>()
      .add("a_position", AVal.constant({
        buffer: IBuffer.fromHost(positions),
        offset: 0, count: 4, stride: 12, format: "float32x3",
      } satisfies BufferView))
      .add("a_color", AVal.constant({
        buffer: IBuffer.fromHost(colors),
        offset: 0, count: 4, stride: 12, format: "float32x3",
      } satisfies BufferView)),
    indices: AVal.constant({
      buffer: IBuffer.fromHost(indices),
      offset: 0, count: 6, stride: 4, format: "uint32",
    } satisfies BufferView),
    drawCall: AVal.constant({
      kind: "indexed",
      indexCount: 6, instanceCount: 1, firstIndex: 0, baseVertex: 0, firstInstance: 0,
    } satisfies DrawCall),
  };
}
