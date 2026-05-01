import { chromium } from "playwright";
const browser = await chromium.launch({
  executablePath: "/usr/bin/chromium", headless: true,
  args: ["--enable-unsafe-webgpu","--enable-features=Vulkan","--use-vulkan=native","--ignore-gpu-blocklist","--enable-webgpu-developer-features","--use-angle=vulkan"],
});
const ctx = await browser.newContext({ viewport: { width: 800, height: 600 }, hasTouch: true });
const page = await ctx.newPage();
const logs = [];
page.on("console", m => logs.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", e => logs.push(`[err] ${e.message}`));

await page.goto("http://localhost:5175/", { waitUntil: "networkidle" });
await page.waitForTimeout(1500);

// Inject probe to log dispatcher events
await page.evaluate(() => {
  const c = document.querySelector("canvas");
  c.addEventListener("pointerdown", e => console.log(`pd ${e.clientX},${e.clientY}`));
  c.addEventListener("pointerup", e => console.log(`pu ${e.clientX},${e.clientY}`));
});

// Two quick taps in the center
const c = await page.locator("canvas");
const box = await c.boundingBox();
const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
await page.mouse.move(cx, cy);
await page.mouse.down();
await page.waitForTimeout(50);
await page.mouse.up();
await page.waitForTimeout(2000);  // longer wait for async readback

// Probe the pick FB's view: check whether canvas has a 2nd attachment
const probe = await page.evaluate(async () => {
  const c = document.querySelector("canvas");
  const ctx = c.getContext("webgpu");
  return { hasCanvas: !!c, w: c.width, h: c.height, hasGPU: !!ctx };
});
console.log("probe:", JSON.stringify(probe));

console.log("status:", await page.locator("#status").textContent());
for (const l of logs.slice(-30)) console.log(l);
await browser.close();
