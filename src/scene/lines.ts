// `<Sg.Lines lines={[a0, b0, a1, b1, …]} width={2} Color={…}/>` and
// `<Sg.Points points={[p0, p1, …]} size={8} Color={…}/>`
//
// Lines and points of a size in PIXELS, drawn as real geometry: a line is
// a cylinder between its two ends, a point a sphere, and the vertex shader
// gives each the radius that makes it `width` / `size` pixels wide where it
// stands. So they have true depth, occlude and are occluded like any
// surface, and read as round from every side — no screen-space quads.
//
//   - One draw call each: the unit cylinder (or sphere) is instanced, the
//     ends (the centres) are the instance attributes.
//
//   - A line is CLIPPED TO THE VIEW VOLUME in the vertex shader before its
//     frame and radius are computed. An end behind the eye has no screen
//     position, and one far outside the frustum makes the projected numbers
//     large enough to wobble; the part inside is all that is drawn anyway.
//     The volume is taken a little wider than the frustum, so a line's
//     width is not cut at the frame's edge.
//
//   - Positions are relative to the first point (a translation applied as
//     the model trafo), so planet-scale coordinates keep their precision
//     in the f32 instance buffers.

import { AVal, HashMap, type aval } from "@aardworx/wombat.adaptive";
import { V3d, V3f, V4f, Trafo3d } from "@aardworx/wombat.base";
import { IBuffer, type BufferView, type DrawCall, ElementType } from "@aardworx/wombat.rendering/core";
import type { Effect } from "@aardworx/wombat.shader";
import { effect, vertex, fragment } from "@aardworx/wombat.shader";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { abs, max, min } from "@aardworx/wombat.shader/types";

import type { VNode } from "../vnode.js";
import { Sg } from "./constructors.js";
import type { SgNode } from "./sg.js";
import type { SgScopeProps } from "./constructors.js";

declare module "@aardworx/wombat.shader/uniforms" {
  interface UniformScope {
    /** `Sg.Lines` / `Sg.Points`: the width (diameter) in framebuffer pixels. */
    readonly ScreenSizePx: number;
    /** `Sg.Lines` / `Sg.Points`: the colour. */
    readonly ScreenColor: V4f;
  }
}

declare module "./constructors.js" {
  interface SgNamespace {
    Lines: (props: SgLinesProps & SgScopeProps) => VNode;
    Points: (props: SgPointsProps & SgScopeProps) => VNode;
  }
}

export interface SgLinesProps {
  /** The segments' ends in pairs: `[a0, b0, a1, b1, …]`. */
  lines: ReadonlyArray<V3d> | aval<ReadonlyArray<V3d>>;
  /** Width in framebuffer pixels. Default 2. */
  width?: number | aval<number>;
  /** Colour, vec4 (rgba). Default: opaque white. */
  Color?: V4f | aval<V4f>;
  /** Sides of the cylinder. Default 8. */
  tessellation?: number;
}

export interface SgPointsProps {
  points: ReadonlyArray<V3d> | aval<ReadonlyArray<V3d>>;
  /** Diameter in framebuffer pixels. Default 8. */
  size?: number | aval<number>;
  Color?: V4f | aval<V4f>;
  /** Segments around the sphere. Default 12. */
  tessellation?: number;
}

// The view volume's margin: 1.25 keeps a line's width whole at the frame's edge.
const lineVertex = vertex((input: { a_circle: V3f; a_lineA: V3f; a_lineB: V3f }) => {
  const va = uniform.ModelViewTrafo.mul(new V4f(input.a_lineA.x, input.a_lineA.y, input.a_lineA.z, 1.0));
  const vb = uniform.ModelViewTrafo.mul(new V4f(input.a_lineB.x, input.a_lineB.y, input.a_lineB.z, 1.0));
  const ca = uniform.ProjTrafo.mul(va);
  const cb = uniform.ProjTrafo.mul(vb);
  // Clip the segment a + t (b - a), t in [t0, t1], against the planes
  // f >= 0 of the (widened) view volume, f linear along the segment.
  const eps = 1.0e-5;
  const m = 1.25;
  // near: w > eps
  const na = ca.w - eps;
  const nb = cb.w - eps;
  let t0 = na < 0.0 ? na / (na - nb) : 0.0;
  let t1 = nb < 0.0 ? na / (na - nb) : 1.0;
  // left: m w + x
  const la = m * ca.w + ca.x;
  const lb = m * cb.w + cb.x;
  t0 = max(t0, la < 0.0 ? la / (la - lb) : 0.0);
  t1 = min(t1, lb < 0.0 ? la / (la - lb) : 1.0);
  // right: m w - x
  const ra = m * ca.w - ca.x;
  const rb = m * cb.w - cb.x;
  t0 = max(t0, ra < 0.0 ? ra / (ra - rb) : 0.0);
  t1 = min(t1, rb < 0.0 ? ra / (ra - rb) : 1.0);
  // bottom: m w + y
  const ba = m * ca.w + ca.y;
  const bb = m * cb.w + cb.y;
  t0 = max(t0, ba < 0.0 ? ba / (ba - bb) : 0.0);
  t1 = min(t1, bb < 0.0 ? ba / (ba - bb) : 1.0);
  // top: m w - y
  const ua = m * ca.w - ca.y;
  const ub = m * cb.w - cb.y;
  t0 = max(t0, ua < 0.0 ? ua / (ua - ub) : 0.0);
  t1 = min(t1, ub < 0.0 ? ua / (ua - ub) : 1.0);
  // Wholly outside (both ends beyond one plane leave t0 > t1): no triangle.
  const inside = t1 > t0 ? 1.0 : 0.0;
  const dv = new V3f(vb.x - va.x, vb.y - va.y, vb.z - va.z);
  const t = t0 + (t1 - t0) * input.a_circle.z;
  const p = new V3f(va.x + dv.x * t, va.y + dv.y * t, va.z + dv.z * t);
  // A frame around the axis, in view space. The helper is the view axis
  // the line is least along, so the cross product never vanishes.
  const dl = max(dv.length(), 1.0e-20);
  const d = new V3f(dv.x / dl, dv.y / dl, dv.z / dl);
  const h = abs(d.z) < 0.9 ? new V3f(0.0, 0.0, 1.0) : new V3f(1.0, 0.0, 0.0);
  const u = d.cross(h).normalize();
  const v = d.cross(u);
  // What one view-space unit is on screen at p: project p and p + up.
  const c0 = uniform.ProjTrafo.mul(new V4f(p.x, p.y, p.z, 1.0));
  const c1 = uniform.ProjTrafo.mul(new V4f(p.x, p.y + 1.0, p.z, 1.0));
  const perUnit = abs(c1.y / max(c1.w, eps) - c0.y / max(c0.w, eps)) * uniform.ViewportSize.y * 0.5;
  const r = uniform.ScreenSizePx * 0.5 / max(perUnit, 1.0e-20);
  const q = new V3f(
    p.x + (u.x * input.a_circle.x + v.x * input.a_circle.y) * r,
    p.y + (u.y * input.a_circle.x + v.y * input.a_circle.y) * r,
    p.z + (u.z * input.a_circle.x + v.z * input.a_circle.y) * r);
  const clip = uniform.ProjTrafo.mul(new V4f(q.x, q.y, q.z, 1.0));
  return { gl_Position: new V4f(clip.x * inside, clip.y * inside, clip.z * inside, inside > 0.5 ? clip.w : -1.0) };
});

const pointVertex = vertex((input: { a_sphere: V3f; a_center: V3f }) => {
  const c = uniform.ModelViewTrafo.mul(new V4f(input.a_center.x, input.a_center.y, input.a_center.z, 1.0));
  const eps = 1.0e-5;
  const c0 = uniform.ProjTrafo.mul(c);
  const c1 = uniform.ProjTrafo.mul(new V4f(c.x, c.y + 1.0, c.z, 1.0));
  const perUnit = abs(c1.y / max(c1.w, eps) - c0.y / max(c0.w, eps)) * uniform.ViewportSize.y * 0.5;
  const r = uniform.ScreenSizePx * 0.5 / max(perUnit, 1.0e-20);
  const clip = uniform.ProjTrafo.mul(new V4f(c.x + input.a_sphere.x * r, c.y + input.a_sphere.y * r, c.z + input.a_sphere.z * r, 1.0));
  // A centre behind the eye has no size on screen.
  const inside = c0.w > eps ? 1.0 : 0.0;
  return { gl_Position: new V4f(clip.x * inside, clip.y * inside, clip.z * inside, inside > 0.5 ? clip.w : -1.0) };
});

const colorFragment = fragment((_input: {}) => {
  return { Colors: uniform.ScreenColor };
});

let lineEffect: Effect | undefined;
let pointEffect: Effect | undefined;

/** The unit cylinder along z: (x, y) on the unit circle, z = 0 or 1, closed by a cap at each end. */
export function cylinderGeometry(sides: number): { positions: Float32Array; indices: Uint32Array } {
  const n = Math.max(3, sides | 0);
  const positions = new Float32Array((2 * n + 2) * 3);
  for (let i = 0; i < n; i++) {
    const a = (i / n) * 2 * Math.PI, x = Math.cos(a), y = Math.sin(a);
    positions.set([x, y, 0], i * 3);
    positions.set([x, y, 1], (n + i) * 3);
  }
  positions.set([0, 0, 0], 2 * n * 3);
  positions.set([0, 0, 1], (2 * n + 1) * 3);
  const indices: number[] = [];
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    indices.push(i, j, n + i, j, n + j, n + i);     // side
    indices.push(2 * n, j, i);                      // cap at 0
    indices.push(2 * n + 1, n + i, n + j);          // cap at 1
  }
  return { positions, indices: new Uint32Array(indices) };
}

/** The unit sphere, as rings of latitude. */
export function sphereGeometry(segments: number): { positions: Float32Array; indices: Uint32Array } {
  const n = Math.max(4, segments | 0), rings = Math.max(2, n >> 1);
  const positions: number[] = [];
  for (let r = 0; r <= rings; r++) {
    const phi = (r / rings) * Math.PI, z = Math.cos(phi), s = Math.sin(phi);
    for (let i = 0; i < n; i++) {
      const a = (i / n) * 2 * Math.PI;
      positions.push(Math.cos(a) * s, Math.sin(a) * s, z);
    }
  }
  const indices: number[] = [];
  for (let r = 0; r < rings; r++) {
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n, a = r * n + i, b = r * n + j, c = (r + 1) * n + i, d = (r + 1) * n + j;
      indices.push(a, c, b, b, c, d);
    }
  }
  return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}

const asAVal = <T>(value: T | aval<T>, isValue: (v: unknown) => boolean): aval<T> =>
  isValue(value) ? AVal.constant(value as T) : (value as aval<T>);

/** The points relative to the first, as f32 triples, and the first. */
function relative(points: ReadonlyArray<V3d>): { origin: V3d; data: Float32Array } {
  const origin = points.length > 0 ? points[0]! : new V3d(0, 0, 0);
  const data = new Float32Array(points.length * 3);
  for (let i = 0; i < points.length; i++) {
    const p = points[i]!;
    data[i * 3] = p.x - origin.x; data[i * 3 + 1] = p.y - origin.y; data[i * 3 + 2] = p.z - origin.z;
  }
  return { origin, data };
}

function build(
  geometry: { positions: Float32Array; indices: Uint32Array },
  shape: string,
  points: aval<ReadonlyArray<V3d>>,
  perInstance: number,
  names: string[],
  shader: Effect,
  size: aval<number>,
  color: aval<V4f>,
  scope: SgScopeProps,
): SgNode {
  const placed = points.map(relative);
  const vertexAttrs = HashMap.empty<string, BufferView>()
    .add(shape, { buffer: AVal.constant(IBuffer.fromHost(geometry.positions)), elementType: ElementType.V3f });
  // The ends of a line are two attributes over ONE buffer of six floats a
  // segment; a point's centre is one of three.
  const stride = perInstance * 12;
  const buffer = placed.map((p) => IBuffer.fromHost(p.data));
  let instanceAttrs = HashMap.empty<string, BufferView>();
  names.forEach((name, i) => {
    instanceAttrs = instanceAttrs.add(name, { buffer, elementType: ElementType.V3f, offset: i * 12, stride } as BufferView);
  });
  const drawCall = placed.map<DrawCall>((p) => ({
    kind: "indexed",
    indexCount: geometry.indices.length,
    instanceCount: Math.floor(p.data.length / (perInstance * 3)),
    firstIndex: 0, baseVertex: 0, firstInstance: 0,
  }));
  const leaf = Sg.leaf({
    vertexAttributes: vertexAttrs,
    instanceAttributes: instanceAttrs,
    indices: { buffer: AVal.constant(IBuffer.fromHost(geometry.indices)), elementType: ElementType.U32 },
    drawCall,
  });
  const tree = Sg({
    Shader: shader,
    Uniform: { ScreenSizePx: size, ScreenColor: color },
    CullMode: "none",
    Trafo: placed.map((p) => Trafo3d.translation(p.origin)),
    children: [leaf],
  } as never);
  return Sg({ ...(scope as object), children: [tree] } as never);
}

const isNumber = (v: unknown) => typeof v === "number";
const isColor = (v: unknown) => v instanceof V4f;

/** Lines of a width in pixels: cylinders with a constant screen width. */
export function SgLines(props: SgLinesProps & SgScopeProps): SgNode {
  const { lines, width = 2, Color = new V4f(1, 1, 1, 1), tessellation = 8, ...scope } = props;
  lineEffect ??= effect(lineVertex, colorFragment);
  return build(cylinderGeometry(tessellation), "a_circle", asAVal(lines, Array.isArray), 2, ["a_lineA", "a_lineB"],
               lineEffect, asAVal(width, isNumber), asAVal(Color, isColor), scope as SgScopeProps);
}

/** Points of a size in pixels: spheres with a constant screen size. */
export function SgPoints(props: SgPointsProps & SgScopeProps): SgNode {
  const { points, size = 8, Color = new V4f(1, 1, 1, 1), tessellation = 12, ...scope } = props;
  pointEffect ??= effect(pointVertex, colorFragment);
  return build(sphereGeometry(tessellation), "a_sphere", asAVal(points, Array.isArray), 1, ["a_center"],
               pointEffect, asAVal(size, isNumber), asAVal(Color, isColor), scope as SgScopeProps);
}
