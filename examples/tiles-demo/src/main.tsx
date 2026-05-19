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
  type SceneEvent,
  type CullValue,
} from "@aardworx/wombat.dom/scene";
import { AVal, AdaptiveToken, HashMap, cset, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { V3d, V2f, V3f, V4f, M44d, Trafo3d } from "@aardworx/wombat.base";
import { effect, vertex, fragment } from "@aardworx/wombat.shader";
import { uniform } from "@aardworx/wombat.shader/uniforms";
import { abs, type Sampler2D, texture } from "@aardworx/wombat.shader/types";
import {
  type BufferView, ElementType, IBuffer, ITexture, type ClearValues, type DrawCall,
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
// Local default: Sonnenburghof_25 (a real photogrammetry dataset
// served out of `public/sonnenburghof` via a symlink to TileRenderer's
// wwwroot). Pass `?url=…` for any other tileset.json — e.g.
// `?url=https://raw.githubusercontent.com/NASA-AMMOS/3DTilesRendererJS/master/example/public/data/tileset.json`
// for the NASA-AMMOS sample dragon.
const TILESET_URL = params.get("url")
  ?? "/sonnenburghof/3d_tiles/tileset.json";
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
// DEBUG: ?heap=0 forces the legacy (non-heap) RO path by setting
// `firstInstance != 0` on each draw call — isHeapEligible rejects
// it and the runtime routes through the per-RO pipeline. Use to
// isolate "is this artefact a heap-path bug or something else?".
const HEAP_DEBUG_FORCE_LEGACY = params.get("heap") === "0";

// ─── SG composition ────────────────────────────────────────────────────

const ctl = OrbitController.create({ radius: 4, phi: Math.PI / 4, theta: 0.5 });
// Expose for headless repro scripts.
(window as unknown as { __ctl: typeof ctl }).__ctl = ctl;

// ─── Camera recording ───────────────────────────────────────────────────
// Drive a flythrough manually, then dump the path as JSON to share for
// a deterministic repro. Keystroke API:
//   'R' — toggle recording (logs to a buffer, ~60Hz)
//   'P' — dump the buffer to console as JSON + copy to clipboard
//   'C' — clear the buffer
//
// Each sample is { t, phi, theta, radius, cx, cy, cz } where (cx,cy,cz)
// are the orbit centre's components. A replay script can call
// __ctl.setPhi/setTheta/setRadius + setCenter(new V3d(cx,cy,cz)) per
// sample with a real-time `setTimeout` chain.
type CamSample = { t: number; phi: number; theta: number; radius: number; cx: number; cy: number; cz: number };
const camPath: CamSample[] = [];
let camRecording = false;
let camStartMs = 0;
(window as unknown as { __cam: object }).__cam = {
  start(): void { camPath.length = 0; camStartMs = performance.now(); camRecording = true; console.log('[cam] recording started'); },
  stop(): void { camRecording = false; console.log(`[cam] recording stopped (${camPath.length} samples, ${((performance.now()-camStartMs)/1000).toFixed(1)}s)`); },
  clear(): void { camPath.length = 0; console.log('[cam] cleared'); },
  setPose(s: { phi: number; theta: number; radius: number; cx: number; cy: number; cz: number }): void {
    ctl.set(new V3d(s.cx, s.cy, s.cz), s.radius, s.phi, s.theta);
  },
  get path(): CamSample[] { return camPath.slice(); },
  dump(): void {
    const json = JSON.stringify(camPath);
    console.log(`[cam] dumping ${camPath.length} samples`);
    try { void navigator.clipboard.writeText(json); console.log('[cam] copied to clipboard'); }
    catch { /* clipboard write requires user gesture in some contexts */ }
    console.log(json);
  },
};
function camTick(): void {
  if (camRecording) {
    const s = ctl.state.value;
    const c = s.center as unknown as { x?: number; y?: number; z?: number; X?: number; Y?: number; Z?: number };
    camPath.push({
      t: performance.now() - camStartMs,
      phi: s.phi, theta: s.theta, radius: s.radius,
      cx: c.x ?? c.X ?? 0, cy: c.y ?? c.Y ?? 0, cz: c.z ?? c.Z ?? 0,
    });
  }
}
window.addEventListener('keydown', (e: KeyboardEvent) => {
  if (e.key === 'R' || e.key === 'r') {
    const w = window as unknown as { __cam: { start(): void; stop(): void } };
    if (camRecording) w.__cam.stop(); else w.__cam.start();
  } else if (e.key === 'P' || e.key === 'p') {
    (window as unknown as { __cam: { dump(): void } }).__cam.dump();
  } else if (e.key === 'C' || e.key === 'c') {
    (window as unknown as { __cam: { clear(): void } }).__cam.clear();
  }
});

// ─── Touch-accessible recording controls ────────────────────────────────
// Floating button strip in the top-right. Tap-friendly (44px hit target).
// Recording button toggles red while live; "Dump" copies to clipboard and
// also shows a textarea overlay so you can long-press → Copy on iOS.
function mountCamUi(): void {
  const bar = document.createElement('div');
  bar.style.cssText = [
    'position:fixed','top:50px','right:8px','z-index:1000',
    'display:flex','flex-direction:column','gap:6px',
    'font:13px system-ui,sans-serif','user-select:none',
  ].join(';');
  const mkBtn = (label: string, onTap: () => void): HTMLButtonElement => {
    const b = document.createElement('button');
    b.textContent = label;
    b.style.cssText = [
      'min-width:64px','min-height:44px','padding:8px 12px',
      'border:1px solid #555','background:#222','color:#ddd',
      'border-radius:6px','touch-action:manipulation',
      'cursor:pointer','box-shadow:0 1px 2px rgba(0,0,0,.4)',
    ].join(';');
    // Use pointerup so single-tap works without click-delay on mobile.
    b.addEventListener('pointerup', e => { e.preventDefault(); onTap(); });
    b.addEventListener('click', e => e.preventDefault());
    return b;
  };
  const status = document.createElement('div');
  status.style.cssText = 'min-width:120px;padding:4px 8px;color:#999;font-size:11px;text-align:right';
  status.textContent = 'cam: idle';
  const updateStatus = (): void => {
    status.textContent = camRecording
      ? `cam: REC (${camPath.length} pts)`
      : `cam: ${camPath.length} pts`;
  };
  setInterval(updateStatus, 250);
  const recBtn = mkBtn('● REC', () => {
    const w = window as unknown as { __cam: { start(): void; stop(): void } };
    if (camRecording) w.__cam.stop(); else w.__cam.start();
    recBtn.style.background = camRecording ? '#c33' : '#222';
    recBtn.textContent = camRecording ? '■ STOP' : '● REC';
    updateStatus();
  });
  const dumpBtn = mkBtn('SAVE', () => {
    const json = JSON.stringify(camPath);
    dumpBtn.textContent = '…';
    dumpBtn.style.background = '#444';
    fetch('/__cam-save', { method: 'POST', headers: { 'content-type': 'application/json' }, body: json })
      .then(r => r.json())
      .then((r: { ok: boolean; file?: string }) => {
        dumpBtn.textContent = r.ok ? '✓ SAVED' : '✗ SAVE';
        dumpBtn.style.background = r.ok ? '#363' : '#633';
        if (r.ok && r.file) console.log(`[cam] saved to server: ${r.file}`);
        setTimeout(() => { dumpBtn.textContent = 'SAVE'; dumpBtn.style.background = '#222'; }, 2500);
      })
      .catch((e: Error) => {
        dumpBtn.textContent = '✗ ERR';
        dumpBtn.style.background = '#633';
        console.error('[cam] save failed:', e);
        setTimeout(() => { dumpBtn.textContent = 'SAVE'; dumpBtn.style.background = '#222'; }, 2500);
      });
  });
  const clearBtn = mkBtn('CLR', () => {
    (window as unknown as { __cam: { clear(): void } }).__cam.clear();
    updateStatus();
  });
  bar.appendChild(recBtn);
  bar.appendChild(dumpBtn);
  bar.appendChild(clearBtn);
  bar.appendChild(status);
  document.body.appendChild(bar);
}
mountCamUi();

// One leaf per tile mesh primitive. `cset` keeps the live set; tiles
// disposed by the LOD walker get their leaves removed.
const tileLeaves = cset<VNode>();
(window as unknown as { __tileLeaves: typeof tileLeaves }).__tileLeaves = tileLeaves;

// Sampler bound by name to the per-leaf `Uniform={{ DiffuseTex: ... }}`.
// The wombat.shader-vite plugin rewrites `texture(DiffuseTex, uv)` into
// the corresponding IR Sampler op.
const DiffuseTex: Sampler2D = null as unknown as Sampler2D;

// Vertex stage. Photogrammetry tiles ship Positions + (UV via
// glTF "TEXCOORD_0" → THREE attribute "uv") — but generally no
// per-vertex normals. We compute flat-ish per-vertex normals
// (face-normal accumulated to each corner, then normalized) on
// the JS side before building the leaf, so the shader can read
// `Normals` unconditionally.
const tilesVS = vertex((v: { Positions: V4f; Normals: V3f; DiffuseColorCoordinates: V2f }) => {
  const wp = uniform.ModelTrafo.mul(v.Positions);
  const n4 = new V4f(v.Normals.xyz, 0.0).mul(uniform.ModelTrafoInv);
  return {
    gl_Position:    uniform.ViewProjTrafo.mul(wp),
    WorldPositions: wp,
    Normals:        n4.xyz,
    Uv:             v.DiffuseColorCoordinates,
  };
});

// FS: sample diffuse texture, modulate by Lambert against the
// default `LightLocation` point-light uniform.
const tilesFS = fragment((v: { Normals: V3f; WorldPositions: V4f; Uv: V2f }) => {
  const n = v.Normals.normalize();
  const l = uniform.LightLocation.sub(v.WorldPositions.xyz).normalize();
  const ambient = 0.35;
  const diffuse = abs(l.dot(n));
  const k = ambient + (1.0 - ambient) * diffuse;
  const tex = texture(DiffuseTex, v.Uv);
  return { Colors: new V4f(tex.xyz.mul(k), 1.0) };
});
const surface = effect(tilesVS, tilesFS);

// 1×1 white fallback texture for meshes without a material map.
function makeWhitePixel(): ITexture {
  const cv = document.createElement("canvas");
  cv.width = 1; cv.height = 1;
  const cx = cv.getContext("2d")!;
  cx.fillStyle = "#ffffff";
  cx.fillRect(0, 0, 1, 1);
  return ITexture.fromExternal(cv);
}
const whiteTex = makeWhitePixel();


// ─── BufferGeometry → SgLeaf ───────────────────────────────────────────

function computeFlatNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
  const n = new Float32Array(positions.length);
  for (let t = 0; t < indices.length; t += 3) {
    const ia = indices[t]! * 3, ib = indices[t + 1]! * 3, ic = indices[t + 2]! * 3;
    const ax = positions[ia]!,     ay = positions[ia + 1]!, az = positions[ia + 2]!;
    const bx = positions[ib]!,     by = positions[ib + 1]!, bz = positions[ib + 2]!;
    const cx = positions[ic]!,     cy = positions[ic + 1]!, cz = positions[ic + 2]!;
    const ex = bx - ax, ey = by - ay, ez = bz - az;
    const fx = cx - ax, fy = cy - ay, fz = cz - az;
    const nx = ey * fz - ez * fy;
    const ny = ez * fx - ex * fz;
    const nz = ex * fy - ey * fx;
    n[ia    ] = n[ia    ]! + nx; n[ia + 1] = n[ia + 1]! + ny; n[ia + 2] = n[ia + 2]! + nz;
    n[ib    ] = n[ib    ]! + nx; n[ib + 1] = n[ib + 1]! + ny; n[ib + 2] = n[ib + 2]! + nz;
    n[ic    ] = n[ic    ]! + nx; n[ic + 1] = n[ic + 1]! + ny; n[ic + 2] = n[ic + 2]! + nz;
  }
  for (let i = 0; i < n.length; i += 3) {
    const x = n[i]!, y = n[i + 1]!, z = n[i + 2]!;
    const len = Math.sqrt(x * x + y * y + z * z);
    if (len > 1e-12) { n[i] = x / len; n[i + 1] = y / len; n[i + 2] = z / len; }
    else            { n[i] = 0;        n[i + 1] = 1;        n[i + 2] = 0;        }
  }
  return n;
}

function extractMaterialTexture(mesh: THREE.Mesh): ITexture | undefined {
  // glTF photogrammetry typically uses a single MeshStandardMaterial
  // with .map → THREE.Texture whose .image is an ImageBitmap /
  // HTMLImageElement / HTMLCanvasElement — all directly acceptable
  // to ITexture.fromExternal.
  const m = mesh.material as THREE.MeshStandardMaterial | THREE.MeshStandardMaterial[] | undefined;
  const mat = Array.isArray(m) ? m[0] : m;
  const map = mat?.map;
  const img = map?.image as
    | ImageBitmap | HTMLImageElement | HTMLCanvasElement | OffscreenCanvas | ImageData | undefined;
  if (img === undefined) return undefined;
  return ITexture.fromExternal(img, { generateMips: true });
}

// Per-frame mirror so leaf trafos see the *current* matrixWorld even
// after TilesRenderer mutates it post-load-model. Filled by
// `tileNodeFromMesh`, walked from `tick()`.
const liveMeshTrafos: { mesh: THREE.Mesh; trafo: ReturnType<typeof cval<Trafo3d>> }[] = [];

function meshWorldTrafo(mesh: THREE.Mesh): Trafo3d {
  mesh.updateMatrixWorld(true);
  const c = mesh.matrixWorld.elements;
  const r = [
    c[0]!,  c[4]!,  c[8]!,  c[12]!,
    c[1]!,  c[5]!,  c[9]!,  c[13]!,
    c[2]!,  c[6]!,  c[10]!, c[14]!,
    c[3]!,  c[7]!,  c[11]!, c[15]!,
  ];
  return Trafo3d.fromMatrix(M44d.fromArray(r));
}

function tileNodeFromMesh(mesh: THREE.Mesh, active: aval<boolean>): VNode | undefined {
  const g = mesh.geometry as THREE.BufferGeometry;
  const posAttr = g.attributes["position"] as THREE.BufferAttribute | undefined;
  if (posAttr === undefined || posAttr.itemSize !== 3) return undefined;

  // Positions — copy into a standalone Float32Array (de-interleave if needed).
  let positions: Float32Array;
  if (posAttr.array instanceof Float32Array
    && posAttr.array.byteOffset === 0
    && posAttr.array.length === posAttr.count * 3) {
    positions = new Float32Array(posAttr.array);
  } else {
    positions = new Float32Array(posAttr.count * 3);
    for (let i = 0; i < posAttr.count; i++) {
      positions[i * 3 + 0] = posAttr.getX(i);
      positions[i * 3 + 1] = posAttr.getY(i);
      positions[i * 3 + 2] = posAttr.getZ(i);
    }
  }

  // Indices — promote to u32 (heap renderer requires u32), and HONOUR
  // `geometry.drawRange`. glTF loaders emit one BufferGeometry per
  // primitive but multiple primitives may share a buffer, with each
  // pointing at a different `drawRange.{start, count}` window. Using
  // the whole index array draws every primitive's triangles over
  // every other's vertices → long stretched-ribbon artefacts.
  const ix = g.index;
  const drawStart = g.drawRange.start | 0;
  let indices: Uint32Array;
  if (ix === null || ix === undefined) {
    const total = Math.min(g.drawRange.count, posAttr.count - drawStart);
    indices = new Uint32Array(total);
    for (let i = 0; i < total; i++) indices[i] = drawStart + i;
  } else {
    const total = Math.min(g.drawRange.count, ix.count - drawStart);
    indices = new Uint32Array(total);
    if (ix.array instanceof Uint32Array) {
      for (let i = 0; i < total; i++) indices[i] = ix.array[drawStart + i]!;
    } else {
      const src = ix.array as ArrayLike<number>;
      for (let i = 0; i < total; i++) indices[i] = src[drawStart + i]!;
    }
  }

  // Normals — use the glTF-supplied set if present, otherwise compute
  // smooth-ish flat normals from triangle faces. Photogrammetry tiles
  // typically ship without normals.
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
    normals = computeFlatNormals(positions, indices);
  }

  // UVs (glTF TEXCOORD_0 → THREE attribute "uv").
  const uvAttr = g.attributes["uv"] as THREE.BufferAttribute | undefined;
  let uvs: Float32Array;
  if (uvAttr !== undefined && uvAttr.itemSize === 2) {
    if (uvAttr.array instanceof Float32Array
      && uvAttr.array.byteOffset === 0
      && uvAttr.array.length === uvAttr.count * 2) {
      uvs = new Float32Array(uvAttr.array);
    } else {
      uvs = new Float32Array(uvAttr.count * 2);
      for (let i = 0; i < uvAttr.count; i++) {
        uvs[i * 2 + 0] = uvAttr.getX(i);
        uvs[i * 2 + 1] = uvAttr.getY(i);
      }
    }
  } else {
    uvs = new Float32Array(posAttr.count * 2);  // all zeros — fallback white tex shows uniform colour
  }

  const diffuse = extractMaterialTexture(mesh) ?? whiteTex;

  // Per-tile world transform. matrixWorld is often STALE at
  // load-model dispatch — TilesRenderer sets the group/tile matrices
  // shortly after, during `tiles.update()`. We snapshot now and
  // refresh in `tick()` so the leaf trafo tracks the real value.
  const trafo = cval<Trafo3d>(meshWorldTrafo(mesh));
  liveMeshTrafos.push({ mesh, trafo });

  // DEBUG: set baseVertex=1 then back to 0 in a way that forces
  // the legacy (non-heap) path. We set firstIndex=1 and bake the
  // offset into indices? No — simpler: just set baseVertex=0
  // (heap-eligible) but conditional on a global debug flag.
  const dc: DrawCall = HEAP_DEBUG_FORCE_LEGACY
    ? {
        kind: "indexed",
        indexCount: indices.length, instanceCount: 1,
        firstIndex: 0, baseVertex: 0, firstInstance: 1,  // firstInstance≠0 → legacy
      }
    : {
        kind: "indexed",
        indexCount: indices.length, instanceCount: 1,
        firstIndex: 0, baseVertex: 0, firstInstance: 0,
      };
  // Pass TYPED ARRAYS to IBuffer.fromHost (matches what the standard
  // primitives in wombat.dom do). Passing the underlying ArrayBuffer
  // works for vertex attrs (the adapter wraps it), but for INDICES
  // the heap adapter takes a different path that only works on a
  // typed-array view — see asUint32 in heapAdapter.
  const leaf: SgLeaf = {
    kind: "Leaf",
    vertexAttributes: HashMap.empty<string, BufferView>()
      .add("Positions", {
        buffer: AVal.constant(IBuffer.fromHost(positions)),
        elementType: ElementType.V3f,
      })
      .add("Normals", {
        buffer: AVal.constant(IBuffer.fromHost(normals)),
        elementType: ElementType.V3f,
      })
      .add("DiffuseColorCoordinates", {
        buffer: AVal.constant(IBuffer.fromHost(uvs)),
        elementType: ElementType.V2f,
      }),
    indices: {
      buffer: AVal.constant(IBuffer.fromHost(indices.buffer as ArrayBuffer)),
      elementType: ElementType.U32,
    },
    drawCall: AVal.constant(dc),
  };

  return (
    <Sg Shader={surface} Trafo={trafo} Active={active} Uniform={{ DiffuseTex: AVal.constant(diffuse) }}>
      {leaf}
    </Sg>
  ) as unknown as VNode;
}

// ─── 3d-tiles-renderer wiring ──────────────────────────────────────────

const threeCam = new THREE.PerspectiveCamera(60, 1, 0.01, 1e5);
threeCam.position.set(0, 0, 4);

const tiles = new TilesRenderer(TILESET_URL);
// Expose for debugging from headed Chromium tests.
(window as unknown as { __tiles: TilesRenderer }).__tiles = tiles;
tiles.setCamera(threeCam);
tiles.setResolution(threeCam, new THREE.Vector2(800, 600));
tiles.errorTarget = ERROR_TARGET;
// LRU sized to the atlas: every loaded tile pins an atlas slot until
// it's disposed (active=false only gates draws, doesn't release the
// pool ref). 8 pages × ~32 tiles/page ≈ 256 tile capacity, so cap
// maxSize there. Bigger caps just make the atlas explode.
const tilesAnyCache = (tiles as unknown as { lruCache?: { minBytesSize: number; maxBytesSize: number; minSize: number; maxSize: number } }).lruCache;
if (tilesAnyCache !== undefined) {
  tilesAnyCache.minBytesSize = 128 * 1024 * 1024;
  tilesAnyCache.maxBytesSize = 256 * 1024 * 1024;
  tilesAnyCache.minSize = 64;
  tilesAnyCache.maxSize = 256;
}

// 3D Tiles refinement strategy: a tile may be REPLACE'd by its
// children (= "show the children instead of me") or ADD'd to
// (= "show me AND my children"). The lib reports this via the
// `tile-visibility-change` event — `visible=true` means "include
// this tile's content in the current frame", `false` means
// "exclude it". `load-model` only means "content is in memory",
// NOT "should be drawn". Adding every loaded tile blindly violates
// REPLACE semantics: parent + child both render, and you see big
// fan-out triangles from the parent's coarse mesh crossing the
// children's fine triangles.
//
// Visibility is wired through `<Sg Active={cval<boolean>}>` on each
// tile-leaf, so the LOD walker's many-times-per-second flipping
// reaches the heap renderer as drawTable-record indexCount toggles
// — no pool churn, no arena re-allocation. Tiles only leave the
// scene when `dispose-model` fires (the LRU cache evicting their
// content from memory).
type TileEntry = {
  readonly leaves: VNode[];
  readonly meshes: Set<THREE.Mesh>;
  readonly active: ReturnType<typeof cval<boolean>>;
  shown: boolean;
};
const loadedTilesByTile = new WeakMap<object, TileEntry>();
(window as unknown as { __loadedTiles: typeof loadedTilesByTile }).__loadedTiles = loadedTilesByTile;

let loadedTiles = 0;
let visibleMeshes = 0;

// Inner type access: the lib exposes `visibleTiles` as a public
// Set<Tile> on TilesRendererBase. Use it to resolve the "is this
// tile currently in the visible set?" race at load time.
const tilesAny = tiles as unknown as { visibleTiles: Set<object> };

function showTile(entry: TileEntry): void {
  if (entry.active.value === true) return;
  transact(() => { entry.active.value = true; });
  visibleMeshes += entry.leaves.length;
}
function hideTile(entry: TileEntry): void {
  if (entry.active.value === false) return;
  transact(() => { entry.active.value = false; });
  visibleMeshes -= entry.leaves.length;
}

tiles.addEventListener("load-model", ((ev: unknown) => {
  const e = ev as { scene: THREE.Object3D; tile: object };
  // Each tile owns ONE active cval that gates every leaf in the tile
  // through the `<Sg Active={...}>` scope. Default to `false`; the
  // visibility-change listener flips it on if/when the LOD walker
  // says this tile is in the camera's view set.
  const initiallyVisible = tilesAny.visibleTiles.has(e.tile);
  const active = cval<boolean>(initiallyVisible);
  const leaves: VNode[] = [];
  const meshSet = new Set<THREE.Mesh>();
  e.scene.traverse((obj) => {
    if (!(obj as THREE.Mesh).isMesh) return;
    const m = obj as THREE.Mesh;
    meshSet.add(m);
    const leaf = tileNodeFromMesh(m, active);
    if (leaf !== undefined) leaves.push(leaf);
  });
  if (leaves.length === 0) return;
  const entry: TileEntry = { leaves, meshes: meshSet, active, shown: initiallyVisible };
  loadedTilesByTile.set(e.tile, entry);
  transact(() => { for (const l of leaves) tileLeaves.add(l); });
  loadedTiles++;
  if (initiallyVisible) visibleMeshes += leaves.length;
  setStatus(`loaded ${loadedTiles} tiles · ${visibleMeshes} visible`);
  if (!refLocked) autoFrameFromCurrentTiles();
}) as EventListener);

// Walk the tile tree counting (parent-shown, any-descendant-also-shown)
// pairs. A non-zero count under REPLACE refinement means the demo's
// active gating is letting parents through alongside their children —
// the cubist-shatter symptom. Surface it in the status bar so a touch-
// only user can see whether the bug is live without typing commands.
function countParentChildOverlaps(): number {
  const root = (tiles as unknown as { rootTileSet?: { root?: TileNode } }).rootTileSet?.root;
  if (root === undefined) return 0;
  let n = 0;
  const visit = (t: TileNode | undefined): void => {
    if (t === undefined) return;
    const e = loadedTilesByTile.get(t);
    if (e !== undefined && e.active.value === true) {
      const walk = (node: TileNode): void => {
        for (const c of node.children ?? []) {
          const ce = loadedTilesByTile.get(c);
          if (ce !== undefined && ce.active.value === true) n++;
          walk(c);
        }
      };
      walk(t);
    }
    for (const c of t.children ?? []) visit(c);
  };
  visit(root);
  return n;
}
type TileNode = { children?: TileNode[]; refine?: string };

tiles.addEventListener("tile-visibility-change", ((ev: unknown) => {
  const e = ev as { tile: object; visible: boolean };
  const entry = loadedTilesByTile.get(e.tile);
  if (entry === undefined) return;  // pre-load; load-model will reconcile via visibleTiles.has
  if (e.visible) showTile(entry); else hideTile(entry);
  const overlaps = countParentChildOverlaps();
  setStatus(`loaded ${loadedTiles} · ${visibleMeshes} visible · overlap:${overlaps}`);
}) as EventListener);

tiles.addEventListener("dispose-model", ((ev: unknown) => {
  const e = ev as { tile: object };
  const entry = loadedTilesByTile.get(e.tile);
  if (entry === undefined) return;
  // Memory eviction by the LRU cache: tile content is truly gone, so
  // tear the leaves out of the SG (pool/freelist activity is correct
  // here — the data is no longer kept).
  hideTile(entry);
  transact(() => { for (const l of entry.leaves) tileLeaves.remove(l); });
  loadedTilesByTile.delete(e.tile);
  loadedTiles--;
  for (let i = liveMeshTrafos.length - 1; i >= 0; i--) {
    if (entry.meshes.has(liveMeshTrafos[i]!.mesh)) liveMeshTrafos.splice(i, 1);
  }
  setStatus(`loaded ${loadedTiles} tiles · ${visibleMeshes} visible`);
}) as EventListener);

let pendingAutoFrame: { center: V3d; radius: number } | undefined;
let renderReady = false;
function tryAutoFrame(): void {
  if (!renderReady || pendingAutoFrame === undefined) return;
  const f = pendingAutoFrame;
  pendingAutoFrame = undefined;
  // Immediate (skip the orbit's spring animation). Offscreen /
  // headless renders may have a paused time aval; the camera
  // wouldn't reach the target otherwise.
  ctl.set(f.center, f.radius, Math.PI / 4, 0.5);
}

// Scene-wide reference shift. Photogrammetry tilesets carry tile
// positions at projected/geodesic scale (~80km in our Sonnenburghof
// example) — f32 precision at that magnitude is too coarse for cm-
// scale rendering. We snapshot the tileset's center at load time
// and emit a SG-parent `Trafo3d.translation(-refPoint)` so every
// descendant's ModelTrafo composes as `shift × tileTrafo`. Tile
// leaves themselves keep AMMOS's world-space matrixWorld
// unchanged; only the SG parent is reactive. The camera orbits
// ref-relative; we add refPoint back when feeding TilesRenderer's
// world-space camera.
let refPoint: V3d = new V3d(0, 0, 0);
let refLocked = false;
const refShiftTrafo = cval<Trafo3d>(Trafo3d.identity);

// Inspect a few loaded meshes once: if their matrixWorld translations
// are huge (>1e4) the renderer baked geodesic coords into the meshes
// themselves, and we'll want a parent SG shift. Otherwise the meshes
// are already near-origin (TilesRenderer applied an inverse on its
// root group) and no shift is needed.
function meshScale(m: THREE.Mesh): number {
  m.updateMatrixWorld(true);
  const c = m.matrixWorld.elements;
  return Math.max(Math.abs(c[12]!), Math.abs(c[13]!), Math.abs(c[14]!));
}

function autoFrameFromCurrentTiles(): void {
  // Probe one mesh to decide whether we need to apply a parent
  // SG shift (some datasets bake geodesic coords into per-mesh
  // matrixWorld, others let TilesRenderer's root-group inverse-
  // translate everything near origin). One mesh is enough — every
  // tile in a given dataset uses the same coord-system convention.
  tiles.group.updateMatrixWorld(true);
  let probe = 0;
  let probeCenter: THREE.Vector3 | undefined;
  tiles.group.traverse((o) => {
    if (probeCenter !== undefined) return;
    if ((o as THREE.Mesh).isMesh) {
      const m = o as THREE.Mesh;
      probe = meshScale(m);
      m.geometry.computeBoundingBox();
      if (m.geometry.boundingBox) {
        probeCenter = m.geometry.boundingBox.getCenter(new THREE.Vector3()).applyMatrix4(m.matrixWorld);
      }
    }
  });
  if (probeCenter === undefined) return;

  // Tileset bbox in *its* world frame — this is the authoritative
  // full-dataset extent we want to frame, not just the meshes that
  // happen to be loaded right now. Falls back to mesh-driven if
  // the tileset can't supply an AABB.
  const box = new THREE.Box3();
  let bboxCenter: THREE.Vector3;
  let bboxSize:   THREE.Vector3;
  if (tiles.getBoundingBox(box)) {
    bboxCenter = box.getCenter(new THREE.Vector3());
    bboxSize   = box.getSize(new THREE.Vector3());
  } else {
    const obb = new THREE.Matrix4();
    const tbox = new THREE.Box3();
    if (tiles.getOrientedBoundingBox(tbox, obb)) {
      const wb = tbox.clone().applyMatrix4(obb);
      bboxCenter = wb.getCenter(new THREE.Vector3());
      bboxSize   = wb.getSize(new THREE.Vector3());
    } else {
      bboxCenter = probeCenter;
      bboxSize   = new THREE.Vector3(100, 100, 100);
    }
  }

  if (!refLocked) {
    refLocked = true;
    // Apply the SG-parent shift iff the data is at geodesic scale.
    if (probe > 1e4) {
      refPoint = new V3d(bboxCenter.x, bboxCenter.y, bboxCenter.z);
      transact(() => { refShiftTrafo.value = Trafo3d.translation(refPoint.neg()); });
    }
    console.log(`[tiles-demo] tileset bbox center=(${bboxCenter.x.toFixed(1)},${bboxCenter.y.toFixed(1)},${bboxCenter.z.toFixed(1)}) size=(${bboxSize.x.toFixed(1)},${bboxSize.y.toFixed(1)},${bboxSize.z.toFixed(1)}) probe=${probe.toFixed(1)} → ref=(${refPoint.x.toFixed(1)},${refPoint.y.toFixed(1)},${refPoint.z.toFixed(1)})`);
  }
  // Sphere fit: half-diagonal × 1.2 — frames any bbox shape (use
  // OnDoubleTap to fly into interior scans like the Sonnenburghof
  // tunnel; can't auto-detect "inside vs outside" from the bbox
  // alone since photogrammetry tilesets often report a coarse
  // partition-root extent, not the actual data extent).
  const half = Math.sqrt(bboxSize.x * bboxSize.x + bboxSize.y * bboxSize.y + bboxSize.z * bboxSize.z) * 0.5;
  const radius = Math.max(half * 1.2, 1);
  const camCenter = new V3d(bboxCenter.x - refPoint.x, bboxCenter.y - refPoint.y, bboxCenter.z - refPoint.z);
  pendingAutoFrame = { center: camCenter, radius };
  tryAutoFrame();
}

let didInitialFrame = false;
tiles.addEventListener("load-tileset", (() => {
  setStatus(`tileset loaded — ${TILESET_URL}`);
  // 3D Tiles supports nested tilesets — `load-tileset` fires for
  // every sub-tileset as the LOD walker descends. We only want to
  // frame the camera once, off the root metadata. Re-framing on
  // every sub-tileset causes a feedback loop: camera jump →
  // tiles unload → more sub-tilesets get walked → more events.
  if (didInitialFrame) return;
  didInitialFrame = true;
  // Initial camera frame from the tileset's metadata bbox — this
  // is in TilesRenderer's centered world (the lib applies an
  // inverse-translation on `tiles.group` for geodesic datasets,
  // so the bbox here is small even for 80km-coord inputs). This
  // gives TilesRenderer enough camera info to start streaming
  // tiles. Once real meshes arrive we refine via the
  // `autoFrameFromCurrentTiles()` walk on `load-model`.
  const box = new THREE.Box3();
  if (tiles.getBoundingBox(box)) {
    const c = box.getCenter(new THREE.Vector3());
    const s = box.getSize(new THREE.Vector3());
    const radius = Math.max(s.x, s.y, s.z, 1) * 1.5;
    pendingAutoFrame = { center: new V3d(c.x, c.y, c.z), radius };
    tryAutoFrame();
    console.log(`[tiles-demo] initial frame center=(${c.x.toFixed(1)},${c.y.toFixed(1)},${c.z.toFixed(1)}) size=(${s.x.toFixed(1)},${s.y.toFixed(1)},${s.z.toFixed(1)})`);
  } else {
    const tbox = new THREE.Box3();
    const obb = new THREE.Matrix4();
    if (tiles.getOrientedBoundingBox(tbox, obb)) {
      const wb = tbox.clone().applyMatrix4(obb);
      const c = wb.getCenter(new THREE.Vector3());
      const s = wb.getSize(new THREE.Vector3());
      const radius = Math.max(s.x, s.y, s.z, 1) * 1.5;
      pendingAutoFrame = { center: new V3d(c.x, c.y, c.z), radius };
      tryAutoFrame();
      console.log(`[tiles-demo] initial frame (obb) center=(${c.x.toFixed(1)},${c.y.toFixed(1)},${c.z.toFixed(1)}) size=(${s.x.toFixed(1)},${s.y.toFixed(1)},${s.z.toFixed(1)})`);
    }
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
  camTick();
  const sz = RenderControl.viewport.getValue(AdaptiveToken.top);
  if (sz !== undefined) {
    const w = sz.width | 0, h = sz.height | 0;
    if (w !== lastViewportW || h !== lastViewportH) {
      lastViewportW = w; lastViewportH = h;
      const aspect = Math.max(0.001, w / Math.max(1, h));
      // wombat's `perspective({ fovInRadians: PI/3 })` is *horizontal*
      // FOV; THREE.PerspectiveCamera takes *vertical* FOV in degrees.
      // Convert: vFov = 2 * atan(tan(hFov/2) / aspect). Get this wrong
      // and TilesRenderer's screen-space-error LOD math underestimates
      // tile coverage (especially in portrait phone aspect), refusing
      // to descend → lots of coarse tiles instead of crisp ones.
      const hFovRad = Math.PI / 3;
      const vFovRad = 2 * Math.atan(Math.tan(hFovRad / 2) / aspect);
      threeCam.aspect = aspect;
      threeCam.fov = (vFovRad * 180) / Math.PI;
      threeCam.updateProjectionMatrix();
      // setResolution should see *device pixels* — that's what our
      // WebGPU canvas actually renders at, and TilesRenderer's
      // screen-space-error formula compares tile-projected size in
      // pixels against the `errorTarget`. Passing CSS pixels on a
      // 3x-DPR phone makes the renderer think the canvas is 1/3 as
      // wide and stops descending → blurry tiles.
      const dpr = window.devicePixelRatio || 1;
      tiles.setResolution(threeCam, new THREE.Vector2(w * dpr, h * dpr));
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
    // M44d is row-major (M[i][j] at i*4+j); THREE.Matrix4.set args
    // are row-major reading order — pass straight through.
    // Translation (n14, n24, n34) = indices 3, 7, 11.
    // The SG orbits ref-relative; TilesRenderer needs the world
    // camera, so add refPoint back into the translation column.
    threeCam.matrixWorld.set(
      a[0]!,  a[1]!,  a[2]!,  a[3]!  + refPoint.x,
      a[4]!,  a[5]!,  a[6]!,  a[7]!  + refPoint.y,
      a[8]!,  a[9]!,  a[10]!, a[11]! + refPoint.z,
      a[12]!, a[13]!, a[14]!, a[15]!,
    );
    threeCam.matrixWorldInverse.copy(threeCam.matrixWorld).invert();
    threeCam.matrixWorldNeedsUpdate = false;
    threeCam.position.setFromMatrixPosition(threeCam.matrixWorld);
    threeCam.quaternion.setFromRotationMatrix(threeCam.matrixWorld);
  }
  tiles.update();
  // After tiles.update() the renderer has settled every tile's
  // local matrix; refresh each leaf's reactive trafo so the SG
  // sees the up-to-date world transform. Most frames this is a
  // no-op (cval skips if .value is === identical, but we set a
  // fresh Trafo3d each frame — wombat.adaptive's structural-
  // equality short-circuit handles the actual no-change case).
  if (liveMeshTrafos.length > 0) {
    transact(() => {
      for (const e of liveMeshTrafos) {
        e.trafo.value = meshWorldTrafo(e.mesh);
      }
    });
  }
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
    onReady={({ canvas, time: rcTime, runtime }) => {
      ctl.attach(canvas, rcTime);
      // Expose for test diagnostics: walk runtime._tasks to find the
      // hybrid scene's task and surface its validate / coherence hooks.
      (window as unknown as { __runtime: typeof runtime }).__runtime = runtime;
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
      CullMode={AVal.constant<CullValue>("none")}
      ForcePixelPicking={AVal.constant(true)}
      OnDoubleTap={(e: SceneEvent) => ctl.flyTo(e.worldPos)}
    >
      <Sg Trafo={refShiftTrafo}>
        {tileLeaves}
      </Sg>
    </Sg>
  </RenderControl>
));
