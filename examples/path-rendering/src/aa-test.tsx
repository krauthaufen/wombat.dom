// One-shot AA correctness test. Renders Lato 'B' (rotated) to an
// offscreen FBO with COPY_SRC usage, downloads the colour bytes via
// copyTextureToBuffer + mapAsync, and exposes them on
// `window.__aaTest` for the Playwright harness to pull.
//
// No RenderControl, no mount, no animation loop — direct path
// through `runtime.compile(compileScene(sg, fboAval))` then
// `task.encode(encoder, token)` + a single `queue.submit`.

import { Sg, compileScene, setAmbient } from "@aardworx/wombat.dom/scene";
import {
  HashMap, AVal, AdaptiveToken,
} from "@aardworx/wombat.adaptive";
import { V3d, V4f, Trafo3d } from "@aardworx/wombat.base";
import { Font } from "@aardworx/wombat.base/font";
import { extractSgNode, isSgVNode } from "@aardworx/wombat.dom/scene";
import { orthographic } from "@aardworx/wombat.dom/scene";
import { Runtime } from "@aardworx/wombat.rendering/runtime";
import { RenderContext, type FramebufferSignature } from "@aardworx/wombat.rendering/core";
import { allocateFramebuffer } from "@aardworx/wombat.rendering/resources";

import latoUrl from "./lato.ttf?url";

declare global {
  interface Window {
    __aaTest?: {
      ready: boolean;
      width: number;
      height: number;
      angleDeg: number;
      aaWidthPx: number;
      // Ground-truth boundary pixels (canvas-pixel space, top-left
      // origin, y-down). Computed from Lato 'B' segments evaluated
      // at fine t and pushed through the same rotation/projection
      // the renderer uses.
      boundary: ReadonlyArray<{ x: number; y: number }>;
      // RGBA bytes of the rendered FBO (length = width*height*4).
      pixels: Uint8Array | null;
      error?: string;
    };
  }
}

const CANVAS_PX = 800;            // logical canvas size
const ANGLE_DEG = 17;             // non-axis-aligned to break pixel grid
const AA_WIDTH_PX = 2.0;          // width of the AA ramp under test

// Orthographic frame: world x, y ∈ [-1, +1] → NDC [-1, +1].
// world (x, y) → canvas px ((x+1)/2 · W, (1-y)/2 · H).
const WORLD = 1.0;
const worldToCanvasX = (x: number): number => ((x + WORLD) / (2 * WORLD)) * CANVAS_PX;
const worldToCanvasY = (y: number): number => ((WORLD - y) / (2 * WORLD)) * CANVAS_PX;

const angleRad = (ANGLE_DEG * Math.PI) / 180;
const cs = Math.cos(angleRad);
const sn = Math.sin(angleRad);

const GLYPH_SCALE = 1.4;   // 'B' fills ~70% of [-1,+1] world.

window.__aaTest = {
  ready: false,
  width: CANVAS_PX,
  height: CANVAS_PX,
  angleDeg: ANGLE_DEG,
  aaWidthPx: AA_WIDTH_PX,
  boundary: [],
  pixels: null,
};

async function main(): Promise<void> {
  const lato = await Font.load(latoUrl);

  // ── Ground-truth boundary in canvas px ──────────────────────────
  const segs = lato.charToSegments("B");
  const upem = lato.unitsPerEm || 1;
  const halfAdv = (lato.advanceWidth("B") / upem) * 0.5;
  const SAMPLES_PER_SEG = 200;
  const boundary: { x: number; y: number }[] = [];
  for (const seg of segs) {
    for (let i = 0; i <= SAMPLES_PER_SEG; i++) {
      const t = i / SAMPLES_PER_SEG;
      const p = seg.eval(t);
      // Sg.Text centres around x=0 but leaves y on the baseline.
      // Match exactly: x shift by -halfAdv, y unchanged.
      let x = (p.x / upem) - halfAdv;
      let y = (p.y / upem);
      x *= GLYPH_SCALE; y *= GLYPH_SCALE;
      const rx = cs * x - sn * y;
      const ry = sn * x + cs * y;
      boundary.push({ x: worldToCanvasX(rx), y: worldToCanvasY(ry) });
    }
  }
  window.__aaTest!.boundary = boundary;

  // ── Device + Runtime ────────────────────────────────────────────
  if (!("gpu" in navigator)) throw new Error("WebGPU not available");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no GPU adapter");
  const device = await adapter.requestDevice();
  const runtime = new Runtime({ device });

  // ── Offscreen FBO with COPY_SRC ─────────────────────────────────
  const colorFormat: GPUTextureFormat = "bgra8unorm";
  const sig: FramebufferSignature = {
    colors: HashMap.empty<string, GPUTextureFormat>().add("outColor", colorFormat),
    colorNames: ["outColor"],
    sampleCount: 1,
    depthStencil: {
      format: "depth24plus",
      hasDepth: true,
      hasStencil: false,
    },
  };
  const sizeAval = AVal.constant({ width: CANVAS_PX, height: CANVAS_PX });
  const fbo = allocateFramebuffer(device, sig, sizeAval, {
    extraUsage: GPUTextureUsage.COPY_SRC,
    labelPrefix: "aa-test",
  });
  fbo.acquire();

  // Set ambient viewport so Sg.Text's `Viewport` uniform reflects
  // our render-target size (otherwise it defaults to (1, 1) and the
  // ribbon-expansion math goes wildly out of scale).
  const sizePx = AVal.constant({ width: CANVAS_PX, height: CANVAS_PX });
  setAmbient({
    viewport: sizePx,
    view: AVal.constant(Trafo3d.identity),
    proj: AVal.constant(Trafo3d.identity),
    time: AVal.constant(0),
    registry: undefined as never,
  });

  // ── Build the scene (call Sg/Sg.Text directly so we get the
  // SgVNode immediately — JSX would wrap it in a deferred-component
  // VNode whose function only runs at mount time). ─────────────────
  const textVNode = Sg.Text({
    font: lato,
    text: "B",
    align: "center",
    aa: "alpha-blending",
    aaWidthPx: AA_WIDTH_PX,
    Color: new V4f(1, 1, 1, 1),
    Trafo: [Sg.scale(GLYPH_SCALE), Sg.rotate({ axis: new V3d(0, 0, 1), rad: angleRad })],
  });
  const sceneVNode = Sg({
    View: AVal.constant(Trafo3d.identity),
    Proj: orthographic({
      left: -1, right: 1, bottom: -1, top: 1, near: -1, far: 1,
    }),
    children: [textVNode],
  });
  if (!isSgVNode(sceneVNode)) throw new Error("scene root not an SgVNode");
  const sg = extractSgNode(sceneVNode);

  // ── Compile + run ───────────────────────────────────────────────
  const fbAval = fbo.map(f => f);   // expose IFramebuffer through aval
  const commands = compileScene(sg, fbAval, {
    clear: {
      colors: HashMap.empty<string, V4f>().add("outColor", new V4f(0, 0, 0, 1)),
      depth: 1.0,
    },
  });
  const task = runtime.compile(commands);

  const enc = device.createCommandEncoder({ label: "aa-test.encoder" });
  // task.encode wraps the work in a RenderContext.withEncoder so the
  // command-encoding callsites can pull the active encoder.
  RenderContext.withEncoder(enc, () => {
    task.encode(enc, AdaptiveToken.top);
  });
  // Read back: copy the colour texture into a packed buffer.
  // bytesPerRow must be a multiple of 256; for 4-byte BGRA8 at width
  // 800 it's 800*4 = 3200 (already aligned).
  const bytesPerRow = Math.ceil(CANVAS_PX * 4 / 256) * 256;
  const readbackSize = bytesPerRow * CANVAS_PX;
  const readback = device.createBuffer({
    size: readbackSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: "aa-test.readback",
  });
  const colorTex = AVal.force(fbAval).colorTextures!.tryFind("outColor")!;
  enc.copyTextureToBuffer(
    { texture: colorTex },
    { buffer: readback, bytesPerRow, rowsPerImage: CANVAS_PX },
    { width: CANVAS_PX, height: CANVAS_PX, depthOrArrayLayers: 1 },
  );
  device.queue.submit([enc.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(readback.getMappedRange()).slice();
  readback.unmap();

  // Repack into RGBA tightly (strip the bytesPerRow padding) and
  // BGRA → RGBA byte order.
  const rgba = new Uint8Array(CANVAS_PX * CANVAS_PX * 4);
  for (let y = 0; y < CANVAS_PX; y++) {
    for (let x = 0; x < CANVAS_PX; x++) {
      const si = y * bytesPerRow + x * 4;
      const di = (y * CANVAS_PX + x) * 4;
      rgba[di + 0] = mapped[si + 2]!;   // B → R
      rgba[di + 1] = mapped[si + 1]!;   // G → G
      rgba[di + 2] = mapped[si + 0]!;   // R → B
      rgba[di + 3] = mapped[si + 3]!;   // A → A
    }
  }
  window.__aaTest!.pixels = rgba;
  window.__aaTest!.ready = true;
  console.log(`[aa-test] ready, pixels ${rgba.length}`);
}

main().catch(e => {
  const msg = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
  console.error("[aa-test] failed:", msg);
  window.__aaTest!.error = msg;
  window.__aaTest!.ready = true;
});
