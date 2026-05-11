// shared.ts — module-level lazy caches for primitive geometry.
//
// Each primitive's positions / normals / indices live in a single
// `IBuffer` allocated ONCE and shared by every instance of that
// primitive. The returned `vertexAttrs` / `indices` / `drawCall`
// avals are constants over those buffers; consumers append a
// (per-instance) `Colors` BufferView via colourBuffer.ts.
//
// Sphere / Cylinder / Cone caches are keyed by tessellation; Box /
// Tetrahedron / Octahedron / FullscreenQuad / ScreenQuad have a
// single fixed entry.

import {
  AVal, HashMap, type aval,
} from "@aardworx/wombat.adaptive";
import {
  IBuffer,
  BufferView,
  type DrawCall, ElementType} from "@aardworx/wombat.rendering/core";
import {
  buildBox, buildWireBox,
  buildTetrahedron, buildWireTetrahedron,
  buildOctahedron, buildWireOctahedron,
  buildSphere, buildWireSphere,
  buildCylinder, buildWireCylinder,
  buildCone, buildWireCone,
  buildFullscreenQuad, buildScreenQuad,
  type GeometryData,
} from "./geometry.js";

export interface GeometryHandle {
  readonly vertexAttrs: HashMap<string, BufferView>;
  readonly indices: BufferView;
  readonly drawCall: aval<DrawCall>;
  readonly mode: "triangle-list" | "line-list";
}

function toHandle(g: GeometryData): GeometryHandle {
  const positions: BufferView = {
    buffer: AVal.constant(IBuffer.fromHost(g.positions)),
    elementType: ElementType.V3f,
  };
  const normals: BufferView = {
    buffer: AVal.constant(IBuffer.fromHost(g.normals)),
    elementType: ElementType.V3f,
  };
  const indices: BufferView = {
    buffer: AVal.constant(IBuffer.fromHost(g.indices)),
    elementType: ElementType.U32,
  };
  const drawCall: DrawCall = {
    kind: "indexed",
    indexCount: g.indices.length,
    instanceCount: 1,
    firstIndex: 0,
    baseVertex: 0,
    firstInstance: 0,
  };
  let vertexAttrs = HashMap.empty<string, BufferView>()
    .add("Positions", positions)
    .add("Normals",   normals);
  // Solid builders emit per-vertex UVs (`DiffuseColorCoordinates`).
  // Wire variants leave `g.uvs` undefined; they don't carry a
  // surface for texturing.
  if (g.uvs !== undefined) {
    const uvs: BufferView = {
      buffer: AVal.constant(IBuffer.fromHost(g.uvs)),
      elementType: ElementType.V2f,
    };
    vertexAttrs = vertexAttrs.add("DiffuseColorCoordinates", uvs);
  }
  return {
    vertexAttrs,
    indices,
    drawCall: AVal.constant(drawCall),
    mode: g.mode,
  };
}

// ---------- single-entry caches ----------

let boxCache: GeometryHandle | undefined;
let wireBoxCache: GeometryHandle | undefined;
let tetraCache: GeometryHandle | undefined;
let wireTetraCache: GeometryHandle | undefined;
let octCache: GeometryHandle | undefined;
let wireOctCache: GeometryHandle | undefined;
let fsQuadCache: GeometryHandle | undefined;
const screenQuadCache = new Map<number, GeometryHandle>();

export function getBoxGeometry(): GeometryHandle {
  return (boxCache ??= toHandle(buildBox()));
}
export function getWireBoxGeometry(): GeometryHandle {
  return (wireBoxCache ??= toHandle(buildWireBox()));
}
export function getTetrahedronGeometry(): GeometryHandle {
  return (tetraCache ??= toHandle(buildTetrahedron()));
}
export function getWireTetrahedronGeometry(): GeometryHandle {
  return (wireTetraCache ??= toHandle(buildWireTetrahedron()));
}
export function getOctahedronGeometry(): GeometryHandle {
  return (octCache ??= toHandle(buildOctahedron()));
}
export function getWireOctahedronGeometry(): GeometryHandle {
  return (wireOctCache ??= toHandle(buildWireOctahedron()));
}
export function getFullscreenQuadGeometry(): GeometryHandle {
  return (fsQuadCache ??= toHandle(buildFullscreenQuad()));
}
export function getScreenQuadGeometry(z: number): GeometryHandle {
  let h = screenQuadCache.get(z);
  if (h === undefined) { h = toHandle(buildScreenQuad(z)); screenQuadCache.set(z, h); }
  return h;
}

// ---------- tessellation-keyed caches ----------

const sphereCache = new Map<number, GeometryHandle>();
const wireSphereCache = new Map<number, GeometryHandle>();
const cylinderCache = new Map<number, GeometryHandle>();
const wireCylinderCache = new Map<number, GeometryHandle>();
const coneCache = new Map<number, GeometryHandle>();
const wireConeCache = new Map<number, GeometryHandle>();

function memoTess(cache: Map<number, GeometryHandle>, tess: number, build: (t: number) => GeometryData): GeometryHandle {
  let h = cache.get(tess);
  if (h === undefined) { h = toHandle(build(tess)); cache.set(tess, h); }
  return h;
}

export function getSphereGeometry(tessellation: number): GeometryHandle {
  return memoTess(sphereCache, tessellation, buildSphere);
}
export function getWireSphereGeometry(tessellation: number): GeometryHandle {
  return memoTess(wireSphereCache, tessellation, buildWireSphere);
}
export function getCylinderGeometry(tessellation: number): GeometryHandle {
  return memoTess(cylinderCache, tessellation, buildCylinder);
}
export function getWireCylinderGeometry(tessellation: number): GeometryHandle {
  return memoTess(wireCylinderCache, tessellation, buildWireCylinder);
}
export function getConeGeometry(tessellation: number): GeometryHandle {
  return memoTess(coneCache, tessellation, buildCone);
}
export function getWireConeGeometry(tessellation: number): GeometryHandle {
  return memoTess(wireConeCache, tessellation, buildWireCone);
}
