import { chromium } from "playwright";
const browser = await chromium.launch({ executablePath: "/usr/bin/chromium", headless: true,
  args: ["--enable-unsafe-webgpu","--enable-features=Vulkan","--use-vulkan=native","--ignore-gpu-blocklist","--enable-webgpu-developer-features","--use-angle=vulkan"] });
const ctx = await browser.newContext({ viewport: { width: 800, height: 600 } });
const page = await ctx.newPage();
page.on("console", m => console.log(`[${m.type()}] ${m.text()}`));
await page.goto("http://localhost:5175/", { waitUntil: "networkidle" });
await page.waitForTimeout(1500);
await page.evaluate(() => {
  // Inject a tap handler at top level via Sg... not easy. Use a global hook.
  // Just listen for the controller orbit camera to read its center after click.
});
const c = await page.locator("canvas");
const box = await c.boundingBox();
const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
// Two quick clicks for double-tap
await page.mouse.click(cx, cy);
await page.waitForTimeout(80);
await page.mouse.click(cx, cy);
await page.waitForTimeout(2000);  // wait for animation
await browser.close();
