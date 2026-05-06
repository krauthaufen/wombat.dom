import { chromium } from "playwright";

const browser = await chromium.launch({
  args: ["--ignore-certificate-errors", "--enable-unsafe-webgpu"],
});
const ctx = await browser.newContext({ ignoreHTTPSErrors: true });
const page = await ctx.newPage();

const lines = [];
page.on("console", (msg) => {
  lines.push(`[${msg.type()}] ${msg.text()}`);
});
page.on("pageerror", (err) => {
  lines.push(`[pageerror] ${err.message}\n${err.stack ?? ""}`);
});

await page.goto("https://localhost:8443/", { waitUntil: "load", timeout: 30000 });
await page.waitForTimeout(4000);

for (const l of lines) console.log(l);

await page.screenshot({ path: "/tmp/render.png", fullPage: false });
console.log("[screenshot] /tmp/render.png");

await browser.close();
