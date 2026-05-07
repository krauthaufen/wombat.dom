import { chromium } from "playwright";

const browser = await chromium.launch({
  executablePath: "/usr/bin/chromium",
  headless: false,
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
  viewport: { width: 800, height: 600 },
  ignoreHTTPSErrors: true,
});
const page = await ctx.newPage();
const consoleLog = [];
page.on("console", m => consoleLog.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", e => consoleLog.push(`[pageerror] ${e.message}`));
page.on("requestfailed", r => consoleLog.push(`[reqfail] ${r.url()} :: ${r.failure()?.errorText}`));
page.on("response", r => { if (r.status() >= 400) consoleLog.push(`[${r.status()}] ${r.url()}`); });

const url = process.env.URL ?? "https://localhost:8446/";
await page.goto(url, { waitUntil: "networkidle" });
await page.waitForTimeout(5000);

// Click the toggle button to flip mode (and to force canvas resize).
// Two clicks lands us back on starting mode (instanced) but with proper size.
const toggles = process.argv[2] === "noninstanced" ? 1 : 2;
for (let i = 0; i < toggles; i++) {
  await page.click("button");
  await page.waitForTimeout(800);
}
await page.waitForTimeout(2000);

const canvasInfo = await page.evaluate(() => {
  const c = document.querySelector("canvas");
  if (!c) return { found: false };
  return {
    found: true,
    width: c.width, height: c.height,
    cssWidth: c.clientWidth, cssHeight: c.clientHeight,
  };
});
console.log("canvas:", canvasInfo);

const canvasPng = await page.evaluate(() => {
  const c = document.querySelector("canvas");
  return c?.toDataURL("image/png") ?? null;
});
const fs = await import("node:fs");
const tag = process.argv[2] ?? "instanced";
if (canvasPng) {
  const b64 = canvasPng.replace(/^data:image\/png;base64,/, "");
  fs.writeFileSync(`canvas-${tag}.png`, Buffer.from(b64, "base64"));
}
await page.screenshot({ path: `screenshot-${tag}.png` });

const status = await page.locator("#status").textContent().catch(() => null);
console.log("status:", status);
console.log("--- logs ---");
for (const l of consoleLog) console.log(l);

await browser.close();
