// Atlas multi-page stress test.
//
// Renders an N×N grid of quads at varying screen-distances so that
// the FS samples MULTIPLE mip levels per cell. Each cell gets a
// uniquely-coloured texture of a varying (often non-power-of-two) size.
// Patterns:
//
//   ?grid=N          number of cells per axis (default 12)
//   ?camMul=R        camera distance multiplier; larger ⇒ tinier
//                    cells on-screen ⇒ HIGHER mip sampled (default 3)
//   ?churn=N         random replacements per second once filled
//                    (default 40)
//   ?duration=ms     time to churn before verifying (default 15000)
//
// Texture sizes are mixed: a small lookup table of POT and NPOT sizes
// rotates per cell so the atlas packer sees heterogeneous bounding
// rects (this matches the 3D-tiles workload).
//
// Verification samples each cell's centre from the screenshot and
// reports cells whose colour is BLACK or doesn't match expected.

import { mount, type VNode } from "@aardworx/wombat.dom";
import { RenderControl, Sg, aspectFromViewport, perspective } from "@aardworx/wombat.dom/scene";
import { AVal, HashMap, cset, transact } from "@aardworx/wombat.adaptive";
import { V3d, V4f, Trafo3d } from "@aardworx/wombat.base";
import { ITexture } from "@aardworx/wombat.rendering/core";
import type { ClearValues } from "@aardworx/wombat.rendering/core";
import { texturedSurface } from "../../heap-demo-sg/src/effects.js";

const params = new URLSearchParams(window.location.search);
const GRID = Math.max(2, parseInt(params.get("grid") ?? "12", 10));
const CAM_MUL = Math.max(1, parseFloat(params.get("camMul") ?? "3"));
const CHURN_HZ = Math.max(0, parseInt(params.get("churn") ?? "40", 10));
const DURATION_MS = Math.max(1000, parseInt(params.get("duration") ?? "15000", 10));

const root = document.getElementById("app")!;

const status = document.createElement("div");
status.style.cssText =
  "position:fixed;bottom:0;left:0;right:0;padding:6px 10px;background:rgba(0,0,0,0.6);color:#cdd;font:13px monospace;z-index:1000;pointer-events:none";
status.textContent = `atlas-stress: ${GRID}×${GRID}, camMul=${CAM_MUL}, churn=${CHURN_HZ}/s — running`;
document.body.appendChild(status);

const clear: ClearValues = {
  colors: HashMap.empty<string, V4f>().add("Colors", new V4f(0, 0, 0, 1)),
  depth: 1.0,
};

// Pool of texture sizes — mixes POT, NPOT, small, large. Each cell
// rotates through these so the atlas packer must handle heterogeneity.
const SIZES = [256, 333, 512, 700, 137, 400, 1000, 600];

function makeColorTexture(size: number, r: number, g: number, b: number, label: string): ITexture {
  const cv = document.createElement("canvas");
  cv.width = size; cv.height = size;
  const ctx = cv.getContext("2d")!;
  ctx.fillStyle = `rgb(${r},${g},${b})`;
  ctx.fillRect(0, 0, size, size);
  // Add a thin border in a SECOND distinguishable colour so any mip
  // sampling at the edge picks up the border — helps see "I'm
  // sampling the right sub-rect" vs "I'm sampling neighbouring atlas
  // content".
  ctx.fillStyle = `rgb(${255-r},${255-g},${255-b})`;
  const bw = Math.max(2, Math.floor(size * 0.03));
  ctx.fillRect(0, 0, size, bw);
  ctx.fillRect(0, size - bw, size, bw);
  ctx.fillRect(0, 0, bw, size);
  ctx.fillRect(size - bw, 0, bw, size);
  // Cell-id text — too small to actually read at high mips, but
  // gives the mip-0 a visually distinct pattern per cell.
  ctx.fillStyle = "#000";
  ctx.font = `${Math.floor(size * 0.3)}px monospace`;
  ctx.fillText(label, size * 0.1, size * 0.5);
  return ITexture.fromExternal(cv, { generateMips: true });
}

function colorFor(col: number, row: number): { r: number; g: number; b: number } {
  const step = Math.floor(220 / Math.max(1, GRID - 1));
  return { r: 30 + col * step, g: 30 + row * step, b: 128 };
}

const STEP = 1.0;
const QUAD_SIZE = STEP * 0.92;
const halfSpan = (GRID - 1) / 2 * STEP;

let rebuildSeq = 0;
function buildLeaf(col: number, row: number): VNode {
  const c = colorFor(col, row);
  const idx = row * GRID + col;
  const sz = SIZES[(idx + rebuildSeq) % SIZES.length]!;
  const tex = makeColorTexture(sz, c.r, c.g, c.b, `${col},${row}`);
  const wx = (col - (GRID - 1) / 2) * STEP;
  const wy = (row - (GRID - 1) / 2) * STEP;
  const trafo = Trafo3d.translation(new V3d(wx, wy, 0));
  return (
    <Sg Shader={texturedSurface} Trafo={trafo} Uniform={{ DiffuseTex: AVal.constant(tex) }}>
      <Sg.Box Size={new V3d(QUAD_SIZE, QUAD_SIZE, 0.02)} Color={new V4f(1, 1, 1, 1)} />
    </Sg> as unknown as VNode
  );
}

const expected: { col: number; row: number; r: number; g: number; b: number }[] = [];
for (let row = 0; row < GRID; row++) {
  for (let col = 0; col < GRID; col++) {
    expected.push({ col, row, ...colorFor(col, row) });
  }
}

const liveLeaves = cset<VNode>();
const cellEntries = new Map<number, VNode>();

function addCell(col: number, row: number): void {
  const idx = row * GRID + col;
  const v = buildLeaf(col, row);
  cellEntries.set(idx, v);
  transact(() => { liveLeaves.add(v); });
}
function replaceCell(col: number, row: number): void {
  const idx = row * GRID + col;
  const old = cellEntries.get(idx);
  rebuildSeq++;
  if (old) transact(() => { liveLeaves.remove(old); });
  const v = buildLeaf(col, row);
  cellEntries.set(idx, v);
  transact(() => { liveLeaves.add(v); });
}

let addIdx = 0;
function addOneMore(): void {
  if (addIdx >= GRID * GRID) return;
  addCell(addIdx % GRID, Math.floor(addIdx / GRID));
  addIdx++;
}
addOneMore();
const addTimer = setInterval(() => {
  addOneMore();
  if (addIdx >= GRID * GRID) {
    clearInterval(addTimer);
    // Once full, churn at the requested rate.
    if (CHURN_HZ > 0) {
      setInterval(() => {
        const i = Math.floor(Math.random() * GRID * GRID);
        replaceCell(i % GRID, Math.floor(i / GRID));
      }, Math.max(5, Math.floor(1000 / CHURN_HZ)));
    }
  }
}, 15);

// Camera: pull WAY back so each cell is tiny on screen → FS LOD picks
// a high mip → sampling mip-N actually exercises that part of the
// atlas page.
const fovRad = Math.PI / 3;
const halfExtent = halfSpan + STEP * 0.5;
const camDist = (halfExtent / Math.tan(fovRad / 2)) * CAM_MUL;
const view = AVal.constant(Trafo3d.viewTrafoRH(
  new V3d(0, 0, camDist),
  new V3d(0, 1, 0),
  new V3d(0, 0, -1),
));
const proj = perspective({
  fovInRadians: fovRad,
  aspect: aspectFromViewport(RenderControl.viewport),
  near: 0.1,
  far: camDist * 4,
});

mount(root, (
  <RenderControl
    style={{ width: "100vw", height: "100vh", display: "block" }}
    clear={clear}
    onReady={({ runtime }) => {
      (window as unknown as { __runtime: unknown }).__runtime = runtime;
      // Wait for initial fill + full churn period before verifying.
      setTimeout(() => {
        status.textContent = `atlas-stress: verify…`;
      }, DURATION_MS - 500);
    }}
  >
    <Sg View={view} Proj={proj}>
      {liveLeaves}
    </Sg>
  </RenderControl>
) as unknown as VNode);

// Expose for the puppeteer-side verifier (in-page 2d-canvas drawImage
// of a WebGPU canvas doesn't reliably capture pixels; use
// page.screenshot instead).
(window as unknown as { __ats: { grid: number; expected: typeof expected; camMul: number; duration: number } }).__ats =
  { grid: GRID, expected, camMul: CAM_MUL, duration: DURATION_MS };
