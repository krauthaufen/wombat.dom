// 3D-tiles → SG demo.
//
// Loads a 3D Tiles dataset via NASA AMMOS's `3d-tiles-renderer`
// (the THREE.js variant — TilesRenderer drives fetch/LOD/cull from a
// camera + viewport size). For every tile model that lands, we walk
// its THREE scene graph, extract each `Mesh`'s `BufferGeometry` data,
// and push an SG leaf into a `cset<VNode>` under the heap renderer's
// scene. Tiles disposed by LOD culling have their leaves removed.
//
// Goals:
//   - Exercise the heap renderer with many RenderObjects coming from
//     a real dataset (each tile = a draw call, hundreds-to-thousands
//     of tiles over the dataset's BVH).
//   - Show that the §3 chunked arena + chunk-aware DerivedUniformsScene
//     handle organic-load patterns from streaming.
//
// URL params:
//   ?url=…         custom tileset.json (default: NASA AMMOS dragon)
//   ?chunk=N       per-arena-chunk byte cap in MB (default: adapter-max)
//   ?errorTarget=N TilesRenderer pixel error budget (default 6)

import { mount, type VNode } from "@aardworx/wombat.dom";
import {
  OrbitController,
  RenderControl,
  Sg,
  aspectFromViewport,
  perspective,
  DefaultSurfaces,
  type SceneEvent,
} from "@aardworx/wombat.dom/scene";
import { AVal, AdaptiveToken, HashMap, cset, transact } from "@aardworx/wombat.adaptive";
import { V3d, V4f, M44d, Trafo3d } from "@aardworx/wombat.base";
import { effect } from "@aardworx/wombat.shader";
import {
  type BufferView, ElementType, IBuffer, type ClearValues, type DrawCall,
} from "@aardworx/wombat.rendering/core";
import type { SgLeaf } from "@aardworx/wombat.dom/scene";
import { TilesRenderer } from "3d-tiles-renderer";
import * as THREE from "three";

// ─── Page chrome ───────────────────────────────────────────────────────

const root = document.getElementById("app")!;
const status = document.getElementById("status")!;
status.textContent = "starting…";

const setStatus = (m: string, err = false): void => {
  status.textContent = m;
  status.style.color = err ? "#ff7777" : "#888";
};
window.addEventListener("error", (e) => setStatus("error: " + (e.error?.message ?? e.message), true));
window.addEventListener("unhandledrejection", (e) => setStatus("rejected: " + (e.reason?.message ?? String(e.reason)), true));

// ─── Demo knobs ────────────────────────────────────────────────────────

const params = new URLSearchParams(location.search);
const TILESET_URL = params.get("url")
  ?? "https://raw.githubusercontent.com/NASA-AMMOS/3DTilesRendererJS/master/example/public/data/tileset.json";
const chunkMB = (() => {
  const v = params.get("chunk");
  if (v === null) return undefined;
  const n = parseInt(v, 10) | 0;
  return n > 0 ? n : undefined;
})();
const ERROR_TARGET = (() => {
  const v = params.get("errorTarget");
  return v !== null ? Math.max(1, parseFloat(v)) : 6;
})();

// ─── SG composition ────────────────────────────────────────────────────

const ctl = OrbitController.create({ radius: 4, phi: Math.PI / 4, theta: 0.5 });

// One leaf per tile mesh primitive. `cset` keeps the live set; tiles
// disposed by the LOD walker get their leaves removed.
const tileLeaves = cset<VNode>();

// Built-in lambertian shader — Positions + Normals + Colors in,
// shaded fragments out.
const surface = effect(DefaultSurfaces.trafo(), DefaultSurfaces.simpleLighting());

// Per-mesh colour palette so adjacent tiles read as distinct shapes.
const palette: V4f[] = [
  new V4f(0.92, 0.72, 0.45, 1),
  new V4f(0.55, 0.85, 0.95, 1),
  new V4f(0.95, 0.55, 0.45, 1),
  new V4f(0.45, 0.95, 0.65, 1),
  new V4f(0.85, 0.65, 0.95, 1),
  new V4f(0.75, 0.95, 0.45, 1),
];
let paletteCursor = 0;

// ─── BufferGeometry → SgLeaf ───────────────────────────────────────────

function copyOf(typedArr: Float32Array | Uint32Array | Uint16Array | ArrayLike<number>): ArrayBuffer {
  // Make a standalone ArrayBuffer copy so the SG/heap path can hold it
  // without aliasing THREE's internal buffer (which may be mutated by
  // loader instancing optimizations). The .slice() return type widens
  // to ArrayBufferLike under WebGPU's `SharedArrayBuffer` types; cast
  // back to plain ArrayBuffer since we always pass a typed view.
  if (typedArr instanceof Float32Array || typedArr instanceof Uint32Array || typedArr instanceof Uint16Array) {
    const out = typedArr.buffer.slice(typedArr.byteOffset, typedArr.byteOffset + typedArr.byteLength);
    return out as ArrayBuffer;
  }
  const f = new Float32Array(typedArr.length);
  for (let i = 0; i < typedArr.length; i++) f[i] = typedArr[i]!;
  return f.buffer as ArrayBuffer;
}

function tileNodeFromMesh(mesh: THREE.Mesh, color: V4f): VNode | undefined {
  const g = mesh.geometry as THREE.BufferGeometry;
  const posAttr = g.attributes["position"] as THREE.BufferAttribute | undefined;
  if (posAttr === undefined || posAttr.itemSize !== 3) return undefined;

  // Positions — copy into a standalone Float32Array (de-interleave if needed).
  let positions: Float32Array;
  if (posAttr.array instanceof Float32Array
    && posAttr.array.byteOffset === 0
    && posAttr.array.length === posAttr.count * 3) {
    positions = new Float32Array(posAttr.array);  // copy
  } else {
    positions = new Float32Array(posAttr.count * 3);
    for (let i = 0; i < posAttr.count; i++) {
      positions[i * 3 + 0] = posAttr.getX(i);
      positions[i * 3 + 1] = posAttr.getY(i);
      positions[i * 3 + 2] = posAttr.getZ(i);
    }
  }

  // Normals — same de-interleave dance; synthesize if absent.
  const normAttr = g.attributes["normal"] as THREE.BufferAttribute | undefined;
  let normals: Float32Array;
  if (normAttr !== undefined && normAttr.itemSize === 3) {
    if (normAttr.array instanceof Float32Array
      && normAttr.array.byteOffset === 0
      && normAttr.array.length === normAttr.count * 3) {
      normals = new Float32Array(normAttr.array);
    } else {
      normals = new Float32Array(normAttr.count * 3);
      for (let i = 0; i < normAttr.count; i++) {
        normals[i * 3 + 0] = normAttr.getX(i);
        normals[i * 3 + 1] = normAttr.getY(i);
        normals[i * 3 + 2] = normAttr.getZ(i);
      }
    }
  } else {
    normals = new Float32Array(posAttr.count * 3);
    for (let i = 0; i < normals.length; i += 3) {
      normals[i] = 0; normals[i + 1] = 1; normals[i + 2] = 0;
    }
  }

  // Indices — promote everything to u32 (heap renderer requires u32).
  const ix = g.index;
  let indices: Uint32Array;
  if (ix === null || ix === undefined) {
    indices = new Uint32Array(posAttr.count);
    for (let i = 0; i < indices.length; i++) indices[i] = i;
  } else if (ix.array instanceof Uint32Array) {
    indices = new Uint32Array(ix.array);
  } else {
    indices = new Uint32Array(ix.array.length);
    for (let i = 0; i < indices.length; i++) indices[i] = (ix.array as ArrayLike<number>)[i]!;
  }

  // Per-tile world transform. THREE matrixWorld is column-major; that
  // matches M44d.fromArray's expected flat-16 layout.
  mesh.updateMatrixWorld(true);
  const m = M44d.fromArray(mesh.matrixWorld.elements);
  const trafo = Trafo3d.fromMatrix(m);

  // SG leaf with raw vertex attrs.
  const colorBuf = new Float32Array([color.x, color.y, color.z, color.w]);
  const dc: DrawCall = {
    kind: "indexed",
    indexCount: indices.length, instanceCount: 1,
    firstIndex: 0, baseVertex: 0, firstInstance: 0,
  };
  const leaf: SgLeaf = {
    kind: "Leaf",
    vertexAttributes: HashMap.empty<string, BufferView>()
      .add("Positions", {
        buffer: AVal.constant(IBuffer.fromHost(copyOf(positions))),
        elementType: ElementType.V3f,
      })
      .add("Normals", {
        buffer: AVal.constant(IBuffer.fromHost(copyOf(normals))),
        elementType: ElementType.V3f,
      })
      .add("Colors", {
        buffer: AVal.constant(IBuffer.fromHost(colorBuf.buffer as ArrayBuffer)),
        stride: 0,                            // broadcast — all verts read the same V4f
        elementType: ElementType.V4f,
      }),
    indices: {
      buffer: AVal.constant(IBuffer.fromHost(indices.buffer as ArrayBuffer)),
      elementType: ElementType.U32,
    },
    drawCall: AVal.constant(dc),
  };

  return (<Sg Shader={surface} Trafo={trafo}>{leaf}</Sg>) as unknown as VNode;
}

// ─── 3d-tiles-renderer wiring ──────────────────────────────────────────

const threeCam = new THREE.PerspectiveCamera(60, 1, 0.01, 1e5);
threeCam.position.set(0, 0, 4);

const tiles = new TilesRenderer(TILESET_URL);
tiles.setCamera(threeCam);
tiles.setResolution(threeCam, new THREE.Vector2(800, 600));
tiles.errorTarget = ERROR_TARGET;

// Map tile → list of pushed leaves so dispose can remove them.
const leavesByTile = new WeakMap<object, VNode[]>();

let loadedTiles = 0;
let activeMeshes = 0;

tiles.addEventListener("load-model", ((ev: unknown) => {
  const e = ev as { scene: THREE.Object3D; tile: object };
  const leaves: VNode[] = [];
  e.scene.traverse((obj) => {
    if (!(obj as THREE.Mesh).isMesh) return;
    const m = obj as THREE.Mesh;
    const color = palette[paletteCursor++ % palette.length]!;
    const leaf = tileNodeFromMesh(m, color);
    if (leaf !== undefined) leaves.push(leaf);
  });
  if (leaves.length === 0) return;
  leavesByTile.set(e.tile, leaves);
  transact(() => { for (const l of leaves) tileLeaves.add(l); });
  loadedTiles++;
  activeMeshes += leaves.length;
  setStatus(`loaded ${loadedTiles} tiles · ${activeMeshes} meshes live`);
}) as EventListener);

tiles.addEventListener("dispose-model", ((ev: unknown) => {
  const e = ev as { tile: object };
  const leaves = leavesByTile.get(e.tile);
  if (leaves === undefined) return;
  transact(() => { for (const l of leaves) tileLeaves.remove(l); });
  leavesByTile.delete(e.tile);
  activeMeshes -= leaves.length;
  setStatus(`loaded ${loadedTiles} tiles · ${activeMeshes} meshes live`);
}) as EventListener);

let pendingAutoFrame: { center: V3d; radius: number } | undefined;
let renderReady = false;
function tryAutoFrame(): void {
  if (!renderReady || pendingAutoFrame === undefined) return;
  const f = pendingAutoFrame;
  pendingAutoFrame = undefined;
  ctl.setCenter(f.center);
  ctl.setRadius(f.radius);
}

tiles.addEventListener("load-tileset", (() => {
  setStatus(`tileset loaded — ${TILESET_URL}`);
  const box = new THREE.Box3();
  if (tiles.getBoundingBox(box)) {
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z) * 1.5;
    pendingAutoFrame = {
      center: new V3d(center.x, center.y, center.z),
      radius,
    };
    tryAutoFrame();
  }
}) as EventListener);

// ─── Per-frame tick: mirror orbit cam into THREE + update tiles ───────

let lastViewportW = 800, lastViewportH = 600;

let tickRunning = false;
function startTick(): void {
  if (tickRunning) return;
  tickRunning = true;
  requestAnimationFrame(tick);
}
function tick(timeMs: number): void {
  void timeMs;
  const sz = RenderControl.viewport.getValue(AdaptiveToken.top);
  if (sz !== undefined) {
    const w = sz.width | 0, h = sz.height | 0;
    if (w !== lastViewportW || h !== lastViewportH) {
      lastViewportW = w; lastViewportH = h;
      threeCam.aspect = Math.max(0.001, w / Math.max(1, h));
      threeCam.updateProjectionMatrix();
      tiles.setResolution(threeCam, new THREE.Vector2(w, h));
    }
  }
  // Mirror our orbit cam into the THREE camera. wombat.base's M44d
  // stores row-major; THREE.Matrix4.set takes its 16 args in
  // ROW-MAJOR reading order (despite the matrix's internal column-
  // major storage). So passing the flat row-major array straight
  // through is the identity-mapping — no transpose.
  const v = ctl.view.getValue(AdaptiveToken.top);
  if (v !== undefined) {
    const a = v.backward.toArray();
    threeCam.matrixWorld.set(
      a[0]!,  a[1]!,  a[2]!,  a[3]!,
      a[4]!,  a[5]!,  a[6]!,  a[7]!,
      a[8]!,  a[9]!,  a[10]!, a[11]!,
      a[12]!, a[13]!, a[14]!, a[15]!,
    );
    threeCam.matrixWorldInverse.copy(threeCam.matrixWorld).invert();
    threeCam.matrixWorldNeedsUpdate = false;
    threeCam.position.setFromMatrixPosition(threeCam.matrixWorld);
    threeCam.quaternion.setFromRotationMatrix(threeCam.matrixWorld);
  }
  tiles.update();
  requestAnimationFrame(tick);
}

// ─── Mount ─────────────────────────────────────────────────────────────

const clear: ClearValues = {
  colors: HashMap.empty<string, V4f>().add("Colors", new V4f(0.07, 0.07, 0.08, 1)),
  depth: 1.0,
};

mount(root, (
  <RenderControl
    clear={clear}
    {...(chunkMB !== undefined ? { maxChunkBytes: chunkMB * 1024 * 1024 } : {})}
    onReady={({ canvas, time: rcTime }) => {
      ctl.attach(canvas, rcTime);
      renderReady = true;
      tryAutoFrame();
      startTick();
      setStatus(`fetching tileset… ${TILESET_URL}`);
    }}
  >
    <Sg
      View={ctl.view}
      Proj={perspective({
        fovInRadians: Math.PI / 3,
        aspect: aspectFromViewport(RenderControl.viewport),
        near: 0.01,
        far: 1e5,
      })}
      OnDoubleTap={(e: SceneEvent) => ctl.flyTo(e.worldPos)}
    >
      {tileLeaves}
    </Sg>
  </RenderControl>
));
