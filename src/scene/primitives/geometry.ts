// Pure procedural geometry builders — ports of Aardvark.Dom's
// `Primitives.{tetrahedron,octahedron,box,sphere,cylinder,cone}Scene`
// (and `FullscreenQuad`/`ScreenQuad`). Each builder emits raw
// Float32Array positions+normals plus a Uint32Array index list and
// the GPU primitive topology.
//
// Reference: aardvark.dom/src/Aardvark.Dom/SceneGraph/SceneFrontend.fs
// (lines ~911-1734 in the F# source).

export interface GeometryData {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly indices: Uint32Array;
  readonly mode: "triangle-list" | "line-list";
}

const PI = Math.PI;
const TWO_PI = 2 * Math.PI;
const SQRT2 = Math.SQRT2;
const SQRT2_HALF = Math.SQRT2 / 2;

// ---------------------------------------------------------------------------
// Tetrahedron — F# tetrahedronScene
// ---------------------------------------------------------------------------

interface TetraCorners {
  readonly p0: [number, number, number];
  readonly p1: [number, number, number];
  readonly p2: [number, number, number];
  readonly p3: [number, number, number];
}

function tetrahedronCorners(): TetraCorners {
  const a = Math.sqrt(3.0 / 4.0);
  const b = Math.sqrt(1.0 / 12.0);
  const c = Math.sqrt(2.0 / 3.0);
  const s = Math.sqrt(1.5);
  let p0: [number, number, number] = [s * 0.0, s * 0.0, s * 0.0];
  let p1: [number, number, number] = [s * 1.0, s * 0.0, s * 0.0];
  let p2: [number, number, number] = [s * 0.5, s * a,   s * 0.0];
  let p3: [number, number, number] = [s * 0.5, s * b,   s * c  ];
  // Re-centre XY around (0,0).
  const cx = (p0[0] + p1[0] + p2[0] + p3[0]) / 4.0;
  const cy = (p0[1] + p1[1] + p2[1] + p3[1]) / 4.0;
  p0 = [p0[0] - cx, p0[1] - cy, p0[2]];
  p1 = [p1[0] - cx, p1[1] - cy, p1[2]];
  p2 = [p2[0] - cx, p2[1] - cy, p2[2]];
  p3 = [p3[0] - cx, p3[1] - cy, p3[2]];
  return { p0, p1, p2, p3 };
}

function sub(a: readonly [number, number, number], b: readonly [number, number, number]): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function cross(a: readonly [number, number, number], b: readonly [number, number, number]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalize(v: [number, number, number]): [number, number, number] {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

function pushV3(out: Float32Array, oi: number, v: readonly [number, number, number]): void {
  out[oi] = v[0]; out[oi + 1] = v[1]; out[oi + 2] = v[2];
}

export function buildTetrahedron(): GeometryData {
  const { p0, p1, p2, p3 } = tetrahedronCorners();
  const n012 = normalize(cross(sub(p2, p0), sub(p1, p0)));
  const n013 = normalize(cross(sub(p1, p0), sub(p3, p0)));
  const n023 = normalize(cross(sub(p3, p0), sub(p2, p0)));
  const n123 = normalize(cross(sub(p2, p1), sub(p3, p1)));

  const verts: ReadonlyArray<[number, number, number]> = [
    p0, p2, p1,
    p0, p1, p3,
    p0, p3, p2,
    p1, p2, p3,
  ];
  const norms: ReadonlyArray<[number, number, number]> = [
    n012, n012, n012,
    n013, n013, n013,
    n023, n023, n023,
    n123, n123, n123,
  ];
  const positions = new Float32Array(verts.length * 3);
  const normals = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) { pushV3(positions, i * 3, verts[i]!); pushV3(normals, i * 3, norms[i]!); }
  const indices = new Uint32Array(verts.length);
  for (let i = 0; i < indices.length; i++) indices[i] = i;
  return { positions, normals, indices, mode: "triangle-list" };
}

export function buildWireTetrahedron(): GeometryData {
  const { p0, p1, p2, p3 } = tetrahedronCorners();
  const verts = [
    p0, p2, p2, p1, p1, p0,
    p0, p3, p1, p3, p2, p3,
  ];
  const positions = new Float32Array(verts.length * 3);
  const normals = new Float32Array(verts.length * 3); // zeroed; line lists don't shade
  for (let i = 0; i < verts.length; i++) pushV3(positions, i * 3, verts[i]!);
  const indices = new Uint32Array(verts.length);
  for (let i = 0; i < indices.length; i++) indices[i] = i;
  return { positions, normals, indices, mode: "line-list" };
}

// Tetrahedron corner positions (used for Intersectable.tetrahedron).
export function tetrahedronCornersV3(): {
  p0: [number, number, number]; p1: [number, number, number];
  p2: [number, number, number]; p3: [number, number, number];
} {
  return tetrahedronCorners();
}

// Octahedron corner positions (used for Intersectable.octahedron).
export function octahedronCornersV3(): {
  p00: [number, number, number]; p10: [number, number, number];
  p11: [number, number, number]; p01: [number, number, number];
  top: [number, number, number]; bottom: [number, number, number];
} {
  return octahedronCorners();
}

// ---------------------------------------------------------------------------
// Octahedron — F# octahedronScene
// ---------------------------------------------------------------------------

function octahedronCorners(): {
  p00: [number, number, number]; p10: [number, number, number];
  p01: [number, number, number]; p11: [number, number, number];
  top: [number, number, number]; bottom: [number, number, number];
} {
  const s = SQRT2_HALF;
  return {
    p00: [s * -0.5, s * -0.5, s * SQRT2_HALF],
    p10: [s *  0.5, s * -0.5, s * SQRT2_HALF],
    p01: [s * -0.5, s *  0.5, s * SQRT2_HALF],
    p11: [s *  0.5, s *  0.5, s * SQRT2_HALF],
    top: [0, 0, s * SQRT2],
    bottom: [0, 0, 0],
  };
}

export function buildOctahedron(): GeometryData {
  const { p00, p10, p01, p11, top, bottom } = octahedronCorners();
  const h = SQRT2_HALF;
  const verts: ReadonlyArray<[number, number, number]> = [
    p00, p10, top, p00, bottom, p10,
    p10, p11, top, p10, bottom, p11,
    p11, p01, top, p11, bottom, p01,
    p01, p00, top, p01, bottom, p00,
  ];
  const norms: ReadonlyArray<[number, number, number]> = [
    [0, -h, h], [0, -h, h], [0, -h, h],
    [0, -h, -h], [0, -h, -h], [0, -h, -h],
    [h, 0, h], [h, 0, h], [h, 0, h],
    [h, 0, -h], [h, 0, -h], [h, 0, -h],
    [0, h, h], [0, h, h], [0, h, h],
    [0, h, -h], [0, h, -h], [0, h, -h],
    [-h, 0, h], [-h, 0, h], [-h, 0, h],
    [-h, 0, -h], [-h, 0, -h], [-h, 0, -h],
  ];
  const positions = new Float32Array(verts.length * 3);
  const normals = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) { pushV3(positions, i * 3, verts[i]!); pushV3(normals, i * 3, norms[i]!); }
  const indices = new Uint32Array(verts.length);
  for (let i = 0; i < indices.length; i++) indices[i] = i;
  return { positions, normals, indices, mode: "triangle-list" };
}

export function buildWireOctahedron(): GeometryData {
  const { p00, p10, p01, p11, top, bottom } = octahedronCorners();
  const verts = [
    p00, p10, p10, p11, p11, p01, p01, p00,
    p00, top, p01, top, p10, top, p11, top,
    p00, bottom, p01, bottom, p10, bottom, p11, bottom,
  ];
  const positions = new Float32Array(verts.length * 3);
  const normals = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) pushV3(positions, i * 3, verts[i]!);
  const indices = new Uint32Array(verts.length);
  for (let i = 0; i < indices.length; i++) indices[i] = i;
  return { positions, normals, indices, mode: "line-list" };
}

// ---------------------------------------------------------------------------
// Box — F# boxScene (unit cube spanning [0,1]³ — caller scales/translates)
// ---------------------------------------------------------------------------

export function buildBox(): GeometryData {
  // Vertex order matches F# boxScene exactly. 36 verts, no indexing
  // savings (per-face normals demand split corners).
  const O = 0, I = 1;
  const v = (x: number, y: number, z: number): [number, number, number] => [x, y, z];
  const verts: ReadonlyArray<[number, number, number]> = [
    // MinZ
    v(O,O,O), v(O,I,O), v(I,O,O), v(I,O,O), v(O,I,O), v(I,I,O),
    // MaxZ
    v(O,O,I), v(I,O,I), v(O,I,I), v(O,I,I), v(I,O,I), v(I,I,I),
    // MinY
    v(O,O,O), v(I,O,O), v(O,O,I), v(O,O,I), v(I,O,O), v(I,O,I),
    // MaxY
    v(O,I,O), v(O,I,I), v(I,I,O), v(I,I,O), v(O,I,I), v(I,I,I),
    // MinX
    v(O,O,O), v(O,O,I), v(O,I,O), v(O,I,O), v(O,O,I), v(O,I,I),
    // MaxX
    v(I,O,O), v(I,I,O), v(I,O,I), v(I,O,I), v(I,I,O), v(I,I,I),
  ];
  const n: ReadonlyArray<[number, number, number]> = [
    [0,0,-1], [0,0,-1], [0,0,-1], [0,0,-1], [0,0,-1], [0,0,-1],
    [0,0, 1], [0,0, 1], [0,0, 1], [0,0, 1], [0,0, 1], [0,0, 1],
    [0,-1,0], [0,-1,0], [0,-1,0], [0,-1,0], [0,-1,0], [0,-1,0],
    [0, 1,0], [0, 1,0], [0, 1,0], [0, 1,0], [0, 1,0], [0, 1,0],
    [-1,0,0], [-1,0,0], [-1,0,0], [-1,0,0], [-1,0,0], [-1,0,0],
    [ 1,0,0], [ 1,0,0], [ 1,0,0], [ 1,0,0], [ 1,0,0], [ 1,0,0],
  ];
  const positions = new Float32Array(verts.length * 3);
  const normals = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) { pushV3(positions, i * 3, verts[i]!); pushV3(normals, i * 3, n[i]!); }
  const indices = new Uint32Array(verts.length);
  for (let i = 0; i < indices.length; i++) indices[i] = i;
  return { positions, normals, indices, mode: "triangle-list" };
}

export function buildWireBox(): GeometryData {
  const O = 0, I = 1;
  const v = (x: number, y: number, z: number): [number, number, number] => [x, y, z];
  const verts: ReadonlyArray<[number, number, number]> = [
    v(O,O,O), v(I,O,O), v(I,O,O), v(I,I,O), v(I,I,O), v(O,I,O), v(O,I,O), v(O,O,O),
    v(O,O,I), v(I,O,I), v(I,O,I), v(I,I,I), v(I,I,I), v(O,I,I), v(O,I,I), v(O,O,I),
    v(O,O,O), v(O,O,I), v(O,I,O), v(O,I,I), v(I,O,O), v(I,O,I), v(I,I,O), v(I,I,I),
  ];
  const positions = new Float32Array(verts.length * 3);
  const normals = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) pushV3(positions, i * 3, verts[i]!);
  const indices = new Uint32Array(verts.length);
  for (let i = 0; i < indices.length; i++) indices[i] = i;
  return { positions, normals, indices, mode: "line-list" };
}

// ---------------------------------------------------------------------------
// Cone — F# getCone (unit cone: base radius 1 at z=1, apex at z=0)
// ---------------------------------------------------------------------------

export function buildCone(tessellation: number): GeometryData {
  const fvc = tessellation * 6;
  const positions = new Float32Array(fvc * 3);
  const normals = new Float32Array(fvc * 3);
  const step = TWO_PI / tessellation;
  let phi = 0;
  let oi = 0;
  for (let i = 0; i < tessellation; i++) {
    const cp = Math.cos(phi), sp = Math.sin(phi);
    const cp1 = Math.cos(phi + step), sp1 = Math.sin(phi + step);
    const p0: [number, number, number] = [cp, sp, 1];
    const p1: [number, number, number] = [cp1, sp1, 1];
    const p2: [number, number, number] = [0, 0, 0];

    const n0: [number, number, number] = [cp * SQRT2_HALF, sp * SQRT2_HALF, -SQRT2_HALF];
    const n1: [number, number, number] = [cp1 * SQRT2_HALF, sp1 * SQRT2_HALF, -SQRT2_HALF];
    const n2 = normalize([n0[0] + n1[0], n0[1] + n1[1], n0[2] + n1[2]]);

    pushV3(positions, oi * 3, p0); pushV3(normals, oi * 3, n0); oi++;
    pushV3(positions, oi * 3, p2); pushV3(normals, oi * 3, n2); oi++;
    pushV3(positions, oi * 3, p1); pushV3(normals, oi * 3, n1); oi++;

    pushV3(positions, oi * 3, p0); pushV3(normals, oi * 3, [0, 0, 1]); oi++;
    pushV3(positions, oi * 3, p1); pushV3(normals, oi * 3, [0, 0, 1]); oi++;
    pushV3(positions, oi * 3, [0, 0, 1]); pushV3(normals, oi * 3, [0, 0, 1]); oi++;

    phi += step;
  }
  const indices = new Uint32Array(fvc);
  for (let i = 0; i < fvc; i++) indices[i] = i;
  return { positions, normals, indices, mode: "triangle-list" };
}

export function buildWireCone(tessellation: number): GeometryData {
  const fvc = tessellation * 4;
  const positions = new Float32Array(fvc * 3);
  const normals = new Float32Array(fvc * 3);
  const step = TWO_PI / tessellation;
  let phi = 0;
  let oi = 0;
  for (let i = 0; i < tessellation; i++) {
    const cp = Math.cos(phi), sp = Math.sin(phi);
    const cp1 = Math.cos(phi + step), sp1 = Math.sin(phi + step);
    const p0: [number, number, number] = [cp, sp, 1];
    const p1: [number, number, number] = [cp1, sp1, 1];
    const p2: [number, number, number] = [0, 0, 0];
    pushV3(positions, oi * 3, p0); oi++;
    pushV3(positions, oi * 3, p1); oi++;
    pushV3(positions, oi * 3, p1); oi++;
    pushV3(positions, oi * 3, p2); oi++;
    phi += step;
  }
  const indices = new Uint32Array(fvc);
  for (let i = 0; i < fvc; i++) indices[i] = i;
  return { positions, normals, indices, mode: "line-list" };
}

// ---------------------------------------------------------------------------
// Cylinder — F# getCylinder (unit cylinder: radius 1, z in [0,1])
// ---------------------------------------------------------------------------

export function buildCylinder(tessellation: number): GeometryData {
  const vertexCount = 4 * (tessellation + 1) + 2;
  const indexCount = tessellation * 12;
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);

  const step = TWO_PI / tessellation;
  let oi = 0;
  let phi = 0;
  for (let i = 0; i <= tessellation; i++) {
    const c = Math.cos(phi), s = Math.sin(phi);
    const p0: [number, number, number] = [c, s, 0];
    const p1: [number, number, number] = [c, s, 1];
    const n0: [number, number, number] = [0, 0, -1];
    const n1: [number, number, number] = [0, 0, 1];
    const n: [number, number, number] = [c, s, 0];
    pushV3(positions, oi * 3, p0); pushV3(normals, oi * 3, n0); oi++;
    pushV3(positions, oi * 3, p0); pushV3(normals, oi * 3, n);  oi++;
    pushV3(positions, oi * 3, p1); pushV3(normals, oi * 3, n1); oi++;
    pushV3(positions, oi * 3, p1); pushV3(normals, oi * 3, n);  oi++;
    phi += step;
  }
  const bottom = oi;
  pushV3(positions, oi * 3, [0, 0, 0]); pushV3(normals, oi * 3, [0, 0, -1]); oi++;
  const top = oi;
  pushV3(positions, oi * 3, [0, 0, 1]); pushV3(normals, oi * 3, [0, 0, 1]);  oi++;

  const indices = new Uint32Array(indexCount);
  let ii = 0;
  for (let i = 0; i < tessellation; i++) {
    const getIndex = (k: number, side: boolean, t: boolean): number =>
      4 * k + (t ? 2 : 0) + (side ? 1 : 0);

    const i00 = getIndex(i,     true, false);
    const i10 = getIndex(i + 1, true, false);
    const i01 = getIndex(i,     true, true);
    const i11 = getIndex(i + 1, true, true);
    indices[ii++] = i00; indices[ii++] = i10; indices[ii++] = i01;
    indices[ii++] = i01; indices[ii++] = i10; indices[ii++] = i11;

    const j00 = getIndex(i,     false, false);
    const j10 = getIndex(i + 1, false, false);
    const j01 = getIndex(i,     false, true);
    const j11 = getIndex(i + 1, false, true);
    indices[ii++] = j00; indices[ii++] = bottom; indices[ii++] = j10;
    indices[ii++] = j01; indices[ii++] = j11; indices[ii++] = top;
  }
  return { positions, normals, indices, mode: "triangle-list" };
}

export function buildWireCylinder(tessellation: number): GeometryData {
  const vertexCount = 2 * (tessellation + 1);
  const indexCount = tessellation * 6;
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const step = TWO_PI / tessellation;
  let oi = 0;
  let phi = 0;
  for (let i = 0; i <= tessellation; i++) {
    const c = Math.cos(phi), s = Math.sin(phi);
    pushV3(positions, oi * 3, [c, s, 0]); pushV3(normals, oi * 3, [c, s, 0]); oi++;
    pushV3(positions, oi * 3, [c, s, 1]); pushV3(normals, oi * 3, [c, s, 0]); oi++;
    phi += step;
  }
  const indices = new Uint32Array(indexCount);
  let ii = 0;
  for (let i = 0; i < tessellation; i++) {
    const getIndex = (k: number, t: boolean): number => 2 * k + (t ? 1 : 0);
    const i00 = getIndex(i, false);
    const i10 = getIndex(i + 1, false);
    const i01 = getIndex(i, true);
    const i11 = getIndex(i + 1, true);
    indices[ii++] = i00; indices[ii++] = i10;
    indices[ii++] = i10; indices[ii++] = i11;
    indices[ii++] = i11; indices[ii++] = i01;
  }
  return { positions, normals, indices, mode: "line-list" };
}

// ---------------------------------------------------------------------------
// Sphere — F# getSphere (UV sphere, radius 1)
// ---------------------------------------------------------------------------

export function buildSphere(tessellation: number): GeometryData {
  const h = Math.floor(tessellation / 2);
  const dPhi = TWO_PI / tessellation;
  const dTheta = PI / h;

  const vertCount = (tessellation + 1) * (h - 1) + 2 * tessellation;
  const positions = new Float32Array(vertCount * 3);
  const normals = new Float32Array(vertCount * 3);
  let oi = 0;
  let theta = -PI / 2 + dTheta;
  for (let y = 1; y < h; y++) {
    let phi = 0;
    const ct = Math.cos(theta), st = Math.sin(theta);
    for (let x = 0; x <= tessellation; x++) {
      const v: [number, number, number] = [Math.cos(phi) * ct, Math.sin(phi) * ct, st];
      pushV3(positions, oi * 3, v);
      pushV3(normals, oi * 3, v);
      oi++;
      phi += dPhi;
    }
    theta += dTheta;
  }
  const n = oi;
  let phi2 = dPhi / 2;
  for (let i = 0; i < tessellation; i++) {
    pushV3(positions, oi * 3, [0, 0, 1]); pushV3(normals, oi * 3, [0, 0, 1]); oi++;
    phi2 += dPhi;
  }
  const sIdx = oi;
  phi2 = dPhi / 2;
  for (let i = 0; i < tessellation; i++) {
    pushV3(positions, oi * 3, [0, 0, -1]); pushV3(normals, oi * 3, [0, 0, -1]); oi++;
    phi2 += dPhi;
  }

  const faces = 3 * (2 * tessellation + 2 * (h - 2) * tessellation);
  const indices = new Uint32Array(faces);
  let ii = 0;
  for (let x = 0; x < tessellation; x++) {
    indices[ii++] = x;
    indices[ii++] = sIdx + x;
    indices[ii++] = x + 1;
  }
  const o = (h - 2) * (tessellation + 1);
  for (let x = 0; x < tessellation; x++) {
    const i = o + x;
    indices[ii++] = n + x;
    indices[ii++] = i;
    indices[ii++] = i + 1;
  }
  for (let y = 1; y < h - 1; y++) {
    const o0 = (y - 1) * (tessellation + 1);
    const o1 = y * (tessellation + 1);
    for (let x = 0; x < tessellation; x++) {
      const i00 = o0 + x + 1;
      const i01 = o1 + x + 1;
      const i10 = o0 + x;
      const i11 = o1 + x;
      indices[ii++] = i10; indices[ii++] = i00; indices[ii++] = i01;
      indices[ii++] = i10; indices[ii++] = i01; indices[ii++] = i11;
    }
  }
  return { positions, normals, indices, mode: "triangle-list" };
}

export function buildWireSphere(tessellation: number): GeometryData {
  const h = Math.floor(tessellation / 2);
  const dPhi = TWO_PI / tessellation;
  const dTheta = PI / h;

  const vertCount = (tessellation + 1) * (h - 1) + 2 * tessellation;
  const positions = new Float32Array(vertCount * 3);
  const normals = new Float32Array(vertCount * 3);
  let oi = 0;
  let theta = -PI / 2 + dTheta;
  for (let y = 1; y < h; y++) {
    let phi = 0;
    const ct = Math.cos(theta), st = Math.sin(theta);
    for (let x = 0; x <= tessellation; x++) {
      const v: [number, number, number] = [Math.cos(phi) * ct, Math.sin(phi) * ct, st];
      pushV3(positions, oi * 3, v);
      pushV3(normals, oi * 3, v);
      oi++;
      phi += dPhi;
    }
    theta += dTheta;
  }
  const n = oi;
  for (let i = 0; i < tessellation; i++) {
    pushV3(positions, oi * 3, [0, 0, 1]); pushV3(normals, oi * 3, [0, 0, 1]); oi++;
  }
  const sIdx = oi;
  for (let i = 0; i < tessellation; i++) {
    pushV3(positions, oi * 3, [0, 0, -1]); pushV3(normals, oi * 3, [0, 0, -1]); oi++;
  }

  const lines = tessellation * (6 + 4 * (h - 2));
  const indices = new Uint32Array(lines);
  let ii = 0;
  for (let x = 0; x < tessellation; x++) {
    const i = x;
    indices[ii++] = i;     indices[ii++] = i + 1;
    indices[ii++] = i + 1; indices[ii++] = sIdx + x;
  }
  const o = (h - 2) * (tessellation + 1);
  for (let x = 0; x < tessellation; x++) {
    const i = o + x;
    indices[ii++] = i + 1; indices[ii++] = n + x;
  }
  for (let y = 1; y < h - 1; y++) {
    const o0 = (y - 1) * (tessellation + 1);
    const o1 = y * (tessellation + 1);
    for (let x = 0; x < tessellation; x++) {
      const i00 = o0 + x + 1;
      const i01 = o1 + x + 1;
      const i11 = o1 + x;
      indices[ii++] = (o0 + x); indices[ii++] = i11;
      indices[ii++] = i11;     indices[ii++] = i01;
      void i00;
    }
  }
  return { positions, normals, indices, mode: "line-list" };
}

// ---------------------------------------------------------------------------
// Quad — F# FullscreenQuad / ScreenQuad
// ---------------------------------------------------------------------------

export function buildFullscreenQuad(): GeometryData {
  // F# uses Z = -1 (NDC near plane in F#'s convention). For WebGPU
  // the near plane is z=0; pick z=0 as a safe centre — the actual
  // FullscreenQuad shaders typically rewrite gl_Position anyway.
  const positions = new Float32Array([
    -1, -1, 0,
     1, -1, 0,
     1,  1, 0,
    -1,  1, 0,
  ]);
  const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
  const indices = new Uint32Array([0, 1, 3, 3, 1, 2]);
  return { positions, normals, indices, mode: "triangle-list" };
}

export function buildScreenQuad(z: number): GeometryData {
  const positions = new Float32Array([
    -1, -1, z,
     1, -1, z,
    -1,  1, z,
     1,  1, z,
    -1,  1, z,
     1, -1, z,
  ]);
  const normals = new Float32Array(6 * 3);
  for (let i = 0; i < 6; i++) { normals[i * 3 + 2] = 1; }
  const indices = new Uint32Array([0, 1, 2, 3, 4, 5]);
  return { positions, normals, indices, mode: "triangle-list" };
}
