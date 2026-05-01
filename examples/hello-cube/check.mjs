// Headless verification: open the dev server in Playwright Chromium
// with WebGPU enabled, render a few frames, save a screenshot,
// dump status + console logs.
//
// Run with the dev server already up:
//   (cd examples/hello-cube && npx vite &)
//   node examples/hello-cube/check.mjs

import { chromium } from "playwright";

const browser = await chromium.launch({
  // System Chromium (full build, not Playwright's headless_shell)
  // — needs the real compositor to present WebGPU to canvas.
  executablePath: "/usr/bin/chromium",
  headless: true,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,UseSkiaRenderer",
    "--use-vulkan=native",
    "--ignore-gpu-blocklist",
    "--enable-webgpu-developer-features",
    "--use-angle=vulkan",
  ],
});
const ctx = await browser.newContext({ viewport: { width: 800, height: 600 } });
const page = await ctx.newPage();
const consoleLog = [];
page.on("console", m => consoleLog.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", e => consoleLog.push(`[pageerror] ${e.message}`));

await page.goto("http://localhost:5175/", { waitUntil: "networkidle" });
await page.waitForTimeout(2000);  // let a few rAF ticks happen

// Save the canvas bytes (bypasses the compositor / screenshot path).
const canvasPng = await page.evaluate(() => {
  const c = document.querySelector("canvas");
  return c?.toDataURL("image/png") ?? null;
});
const fs = await import("node:fs");
if (canvasPng) {
  const b64 = canvasPng.replace(/^data:image\/png;base64,/, "");
  fs.writeFileSync("canvas.png", Buffer.from(b64, "base64"));
}
await page.screenshot({ path: "screenshot.png" });

const status = await page.locator("#status").textContent();
console.log("status:", status);
console.log("--- logs ---");
for (const l of consoleLog) console.log(l);

await browser.close();
