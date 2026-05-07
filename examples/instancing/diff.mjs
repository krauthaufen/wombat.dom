// Same-session image diff: capture canvas in instanced mode, toggle,
// capture again. Both shots come from the same OrbitController
// instance with no user input, so the camera state is identical.

import { chromium } from "playwright";
import fs from "node:fs";
import zlib from "node:zlib";

const browser = await chromium.launch({
  executablePath: "/usr/bin/chromium",
  headless: false,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan",
    "--use-vulkan=native",
    "--ignore-gpu-blocklist",
    "--ignore-certificate-errors",
  ],
});
const ctx = await browser.newContext({
  viewport: { width: 800, height: 600 },
  ignoreHTTPSErrors: true,
});
const page = await ctx.newPage();
const log = [];
page.on("console", m => log.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", e => log.push(`[pageerror] ${e.message}`));

// Hook createShaderModule to capture WGSL produced
await page.addInitScript(() => {
  globalThis.__wgsl = [];
  const orig = GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule = function(desc) {
    globalThis.__wgsl.push(desc.code);
    return orig.call(this, desc);
  };
});

await page.goto("https://localhost:8446/", { waitUntil: "networkidle" });
await page.waitForTimeout(3000);
// Force initial canvas resize via a toggle round-trip (toggle ×2 lands on starting mode).
await page.click("button"); await page.waitForTimeout(1000);
await page.click("button"); await page.waitForTimeout(3000);

async function snap() {
  // page.screenshot of just the canvas element. The compositor capture
  // includes the actual rendered WebGPU contents (toDataURL on a
  // WebGPU canvas can return blank because the swap-chain image is
  // already presented and not preserveDrawingBuffer-style readable).
  const buf = await page.locator("canvas").screenshot({ type: "png" });
  return "data:image/png;base64," + buf.toString("base64");
}

// Toggle then snap with a tight delay between shots so the
// OrbitController's time integration doesn't drift between captures.
const a = await snap();         // instanced
const wgsl = await page.evaluate(() => globalThis.__wgsl.slice());
fs.writeFileSync("wgsl-inst.txt", wgsl.join("\n=====\n"));
await page.click("button");
await page.waitForTimeout(150);
const b = await snap();         // non-instanced

const decode = (url) => Buffer.from(url.replace(/^data:image\/png;base64,/, ""), "base64");
fs.writeFileSync("diff-instanced.png", decode(a));
fs.writeFileSync("diff-noninstanced.png", decode(b));

// Compare raw pixels — decode PNGs via Sharp would be easier but
// avoid the dep; do pixel diff in the page instead.
const stats = await page.evaluate(async ([dataA, dataB]) => {
  const load = (src) => new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = src; });
  const ia = await load(dataA);
  const ib = await load(dataB);
  const c = document.createElement("canvas");
  c.width = ia.width; c.height = ia.height;
  const g = c.getContext("2d");
  g.drawImage(ia, 0, 0);
  const pa = g.getImageData(0, 0, c.width, c.height).data;
  g.drawImage(ib, 0, 0);
  const pb = g.getImageData(0, 0, c.width, c.height).data;
  let diff = 0, maxD = 0, sum = 0;
  for (let i = 0; i < pa.length; i += 4) {
    const dr = Math.abs(pa[i]   - pb[i]);
    const dg = Math.abs(pa[i+1] - pb[i+1]);
    const db = Math.abs(pa[i+2] - pb[i+2]);
    const d = dr + dg + db;
    if (d > 0) diff++;
    if (d > maxD) maxD = d;
    sum += d;
  }
  const totalPx = pa.length / 4;
  return { totalPx, diffPx: diff, pctDiff: 100 * diff / totalPx, maxChannelDelta: maxD, meanDelta: sum / totalPx };
}, [a, b]);
console.log(stats);

// Save a side-by-side + diff visualization
const sideBySide = await page.evaluate(async ([dataA, dataB]) => {
  const load = (src) => new Promise(r => { const i = new Image(); i.onload = () => r(i); i.src = src; });
  const ia = await load(dataA);
  const ib = await load(dataB);
  const W = ia.width, H = ia.height;
  const c = document.createElement("canvas");
  c.width = W * 3; c.height = H;
  const g = c.getContext("2d");
  g.drawImage(ia, 0, 0);
  g.drawImage(ib, W, 0);
  const tmp = document.createElement("canvas");
  tmp.width = W; tmp.height = H;
  const tg = tmp.getContext("2d");
  tg.drawImage(ia, 0, 0);
  const pa = tg.getImageData(0, 0, W, H).data;
  tg.drawImage(ib, 0, 0);
  const pb = tg.getImageData(0, 0, W, H).data;
  const diff = tg.createImageData(W, H);
  for (let i = 0; i < pa.length; i += 4) {
    const d = Math.abs(pa[i] - pb[i]) + Math.abs(pa[i+1] - pb[i+1]) + Math.abs(pa[i+2] - pb[i+2]);
    const v = Math.min(255, d * 4);
    diff.data[i] = v; diff.data[i+1] = v; diff.data[i+2] = v; diff.data[i+3] = 255;
  }
  tg.putImageData(diff, 0, 0);
  g.drawImage(tmp, W * 2, 0);
  return c.toDataURL("image/png");
}, [a, b]);
fs.writeFileSync("diff-sidebyside.png", decode(sideBySide));
console.log("--- log ---");
for (const l of log.slice(-20)) console.log(l);
await browser.close();
