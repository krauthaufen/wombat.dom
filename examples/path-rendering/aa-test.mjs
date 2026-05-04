// AA correctness harness. Loads the rotated-B test page in a
// headless Chromium with WebGPU, reads the WebGPU canvas as a PNG,
// pulls the ground-truth boundary points the page exposed, and
// bilinear-samples α at each. Reports the histogram so we can
// quantify how close to the ideal `α ≈ 0.5 on the curve` invariant
// the current encoding is.
//
// Pre-requisite: `npm run dev` must be running on https://localhost:8443.
//
// Usage:
//   node aa-test.mjs            # one run, dump histogram + worst offenders
//   node aa-test.mjs --png /tmp/aa.png    # also save the rendered canvas

import { chromium } from "playwright";
import { PNG } from "pngjs";
import * as fs from "node:fs";

const argv = process.argv.slice(2);
const argv_get = (flag, dflt) => {
  const i = argv.indexOf(flag);
  return i < 0 ? dflt : argv[i + 1];
};
const PNG_OUT = argv_get("--png", null);

const browser = await chromium.launch({
  executablePath: "/usr/bin/chromium",
  headless: true,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,UseSkiaRenderer",
    "--use-vulkan=native",
    "--ignore-gpu-blocklist",
    "--enable-webgpu-developer-features",
    "--use-angle=vulkan",
    "--ignore-certificate-errors",
  ],
});
const ctx = await browser.newContext({
  viewport: { width: 1024, height: 1024 },
  ignoreHTTPSErrors: true,
});
const page = await ctx.newPage();
const log = [];
page.on("console", m => log.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", e => log.push(`[pageerror] ${e.message}`));
page.on("requestfailed", r => log.push(`[reqfail] ${r.url()} ${r.failure()?.errorText ?? ""}`));
page.on("response", r => { if (r.status() >= 400) log.push(`[${r.status()}] ${r.url()}`); });

await page.goto("https://localhost:8443/aa-test.html", { waitUntil: "domcontentloaded", timeout: 15_000 });
try {
  await page.waitForFunction(() => window.__aaTest?.ready === true, { timeout: 30_000 });
} catch (e) {
  console.error("timeout waiting for __aaTest.ready");
  console.error("page log:\n" + log.join("\n"));
  await browser.close();
  process.exit(1);
}
const err = await page.evaluate(() => window.__aaTest?.error ?? null);
if (err) {
  console.error("aa-test page error:\n" + err);
  console.error(log.join("\n"));
  await browser.close();
  process.exit(1);
}

const meta = await page.evaluate(() => {
  if (!window.__aaTest) return null;
  return {
    width: window.__aaTest.width,
    height: window.__aaTest.height,
    angleDeg: window.__aaTest.angleDeg,
    aaWidthPx: window.__aaTest.aaWidthPx,
    boundary: window.__aaTest.boundary,
  };
});
if (!meta) {
  console.error("__aaTest not exposed; did the page load?");
  console.error(log.join("\n"));
  await browser.close();
  process.exit(1);
}

// One-shot render path: page does device + runtime.compile +
// copyTextureToBuffer + mapAsync; pixels land on window.__aaTest.
// Pull as a regular array (Playwright serialises Uint8Array as
// array of numbers).
const pixels = await page.evaluate(() => {
  const px = window.__aaTest?.pixels;
  return px ? Array.from(px) : null;
});
if (!pixels) {
  console.error("no pixels exposed");
  console.error(log.join("\n"));
  await browser.close();
  process.exit(1);
}
const png = {
  width: meta.width,
  height: meta.height,
  data: Buffer.from(pixels),
};
if (PNG_OUT) {
  // Wrap in a PNG and overlay every CPU boundary point in red so
  // we can visually verify the harness is sampling the actual
  // rendered boundary.
  const out = new PNG({ width: meta.width, height: meta.height });
  out.data = Buffer.from(pixels);
  for (const { x, y } of meta.boundary) {
    const xi = Math.round(x), yi = Math.round(y);
    if (xi < 0 || yi < 0 || xi >= meta.width || yi >= meta.height) continue;
    const idx = (yi * meta.width + xi) * 4;
    out.data[idx + 0] = 255; // R
    out.data[idx + 1] = 0;
    out.data[idx + 2] = 0;
    out.data[idx + 3] = 255;
  }
  fs.writeFileSync(PNG_OUT, PNG.sync.write(out));
}
const sx = 1, sy = 1;

// Bilinear sample α (RGBA, premultiplied). Out-of-bounds → α=0.
function sampleAlpha(x, y) {
  if (x < 0 || y < 0 || x > png.width - 1 || y > png.height - 1) return 0;
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, png.width - 1);
  const y1 = Math.min(y0 + 1, png.height - 1);
  const fx = x - x0, fy = y - y0;
  // Black BG + white glyph + premultiplied output ⇒ rendered R/G/B
  // channel = coverage. We sample R because it's identical to α
  // under those conditions but survives the screenshot compositor.
  const at = (xi, yi) => {
    const idx = (yi * png.width + xi) * 4;
    return png.data[idx + 0] / 255;
  };
  const a00 = at(x0, y0), a10 = at(x1, y0);
  const a01 = at(x0, y1), a11 = at(x1, y1);
  return a00 * (1 - fx) * (1 - fy) + a10 * fx * (1 - fy)
       + a01 * (1 - fx) * fy       + a11 * fx * fy;
}

const samples = meta.boundary.map(({ x, y }) => sampleAlpha(x * sx, y * sy));
const n = samples.length;
const sorted = [...samples].sort((a, b) => a - b);
const mean = samples.reduce((s, v) => s + v, 0) / n;
const median = sorted[n >> 1];
const stddev = Math.sqrt(samples.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
const inBand = (lo, hi) => samples.filter(v => v >= lo && v <= hi).length / n;

console.log(`=== AA correctness — rotated B (angle ${meta.angleDeg}°, AaWidthPx ${meta.aaWidthPx}) ===`);
console.log(`canvas      : ${meta.width}×${meta.height}    PNG: ${png.width}×${png.height}    scale: ${sx.toFixed(2)}×${sy.toFixed(2)}`);
console.log(`boundary px : ${n} samples`);
console.log(`α mean      : ${mean.toFixed(4)}    (target 0.5)`);
console.log(`α median    : ${median.toFixed(4)}`);
console.log(`α stddev    : ${stddev.toFixed(4)}`);
console.log(`α in [.45,.55]: ${(inBand(0.45, 0.55) * 100).toFixed(1)}%`);
console.log(`α in [.40,.60]: ${(inBand(0.40, 0.60) * 100).toFixed(1)}%`);
console.log(`α in [.30,.70]: ${(inBand(0.30, 0.70) * 100).toFixed(1)}%`);
console.log(`α = 0  count : ${samples.filter(v => v === 0).length}`);
console.log(`α = 1  count : ${samples.filter(v => v === 1).length}`);
// Decile breakdown.
const deciles = new Array(10).fill(0);
for (const v of samples) deciles[Math.min(9, Math.floor(v * 10))]++;
console.log("decile dist :", deciles.map(c => (c / n * 100).toFixed(1) + "%").join(" "));

// Worst offenders (largest |α − 0.5|), top 10 with their boundary px.
const worst = meta.boundary.map((p, i) => ({ p, alpha: samples[i] }))
  .sort((a, b) => Math.abs(b.alpha - 0.5) - Math.abs(a.alpha - 0.5))
  .slice(0, 10);
console.log("=== worst offenders (px, α) ===");
for (const w of worst) {
  console.log(`  (${w.p.x.toFixed(1)}, ${w.p.y.toFixed(1)})  α=${w.alpha.toFixed(3)}`);
}

if (PNG_OUT) console.log(`\nrendered canvas → ${PNG_OUT}`);

if (log.length > 0) {
  console.log("\n=== page log ===");
  for (const l of log) console.log(l);
}

await browser.close();
