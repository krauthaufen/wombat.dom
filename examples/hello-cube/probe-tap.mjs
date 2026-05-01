import { chromium } from "playwright";
const browser = await chromium.launch({ executablePath: "/usr/bin/chromium", headless: true,
  args: ["--enable-unsafe-webgpu","--enable-features=Vulkan","--use-vulkan=native","--ignore-gpu-blocklist","--enable-webgpu-developer-features","--use-angle=vulkan"] });
const ctx = await browser.newContext({ viewport: { width: 800, height: 600 } });
const page = await ctx.newPage();
const logs = [];
page.on("console", m => logs.push(`[${m.type()}] ${m.text()}`));
await page.goto("http://localhost:5175/", { waitUntil: "networkidle" });
await page.waitForTimeout(1500);
await page.evaluate(() => {
  const c = document.querySelector("canvas");
  c.addEventListener("pointerup", e => {
    // Mock: just log click
    console.log(`click ${e.clientX},${e.clientY}`);
  });
});
const c = await page.locator("canvas");
const box = await c.boundingBox();
const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
await page.mouse.move(cx, cy);
await page.mouse.down();
await page.mouse.up();
await page.waitForTimeout(800);
for (const l of logs.slice(-12)) console.log(l);
await browser.close();
