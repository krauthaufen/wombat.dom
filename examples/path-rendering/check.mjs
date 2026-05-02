// Headless verification of the path-rendering demo. Launches system
// Chromium with WebGPU enabled, points it at the local HTTPS dev
// server (self-signed cert), runs a few frames, then dumps:
//   - the #status text
//   - all console messages + page errors
//   - a PNG screenshot of the canvas

import { chromium } from "playwright";
import * as fs from "node:fs";

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
  viewport: { width: 1024, height: 768 },
  ignoreHTTPSErrors: true,
});
const page = await ctx.newPage();
const log = [];
page.on("console", m => log.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", e => log.push(`[pageerror] ${e.message}\n${e.stack ?? ""}`));

try {
  await page.goto("https://localhost:8443/", { waitUntil: "networkidle", timeout: 15_000 });
} catch (err) {
  console.error("goto failed:", err.message);
}
await page.waitForTimeout(2500);

const status = await page.evaluate(() => document.getElementById("status")?.textContent ?? "<no status>");
const canvasPng = await page.evaluate(() => {
  const c = document.querySelector("canvas");
  return c?.toDataURL("image/png") ?? null;
});

console.log("=== STATUS ===");
console.log(status);
console.log("=== CONSOLE / ERRORS ===");
for (const l of log) console.log(l);
if (canvasPng) {
  const buf = Buffer.from(canvasPng.replace(/^data:image\/png;base64,/, ""), "base64");
  fs.writeFileSync("/tmp/path-rendering.png", buf);
  console.log("=== SCREENSHOT ===");
  console.log(`/tmp/path-rendering.png (${buf.length} bytes)`);
}

await browser.close();
