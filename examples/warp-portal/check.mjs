// warp-portal acceptance driver — runs against the persistent
// headed Chromium (CDP :9222, real GPU). Starts its own vite on :5176,
// probes the portal with real mouse input, and asserts:
//
//   A (amp=0):  clicks at analytically-predicted box centers hit the
//               RIGHT inner box through the portal (uv chain + Y-flip
//               correct end to end); hover enter fires; tap deletes —
//               a second click at the same spot falls through to the
//               portal background.
//   B (warped): hover scan across the warped image still resolves ≥3
//               distinct inner boxes; a click deletes the box under
//               the warped cursor.
//
// Usage: node check.mjs   (from examples/warp-portal)

import { spawn, execSync } from "node:child_process";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { browser } = require("/home/schorsch/.headed-chrome");

const PORT = 5176;
const URL_BASE = `http://localhost:${PORT}/`;

let failures = 0;
const check = (cond, msg) => {
  if (cond) console.log(`  ok   ${msg}`);
  else { failures++; console.error(`  FAIL ${msg}`); }
};

// ─── vite ───────────────────────────────────────────────────────────────

try { execSync(`fuser -k ${PORT}/tcp 2>/dev/null`); } catch { /* none */ }
const vite = spawn("npx", ["vite", "--port", String(PORT), "--strictPort"], {
  cwd: new URL(".", import.meta.url).pathname,
  stdio: "ignore",
  detached: true,
});
const stopVite = () => { try { process.kill(-vite.pid, "SIGKILL"); } catch { /* gone */ } };
process.on("exit", stopVite);

const waitPort = async () => {
  for (let i = 0; i < 60; i++) {
    try { const r = await fetch(URL_BASE); if (r.ok) return; } catch { /* not yet */ }
    await new Promise(r => setTimeout(r, 250));
  }
  throw new Error("vite did not come up");
};
await waitPort();

// ─── browser helpers ────────────────────────────────────────────────────

const b = await browser();

async function openPage(query) {
  const page = await b.newPage();
  await page.setViewport({ width: 800, height: 600 });
  page.on("console", m => { if (m.type() === "error") console.error("  [page]", m.text()); });
  page.on("pageerror", e => console.error("  [pageerror]", e.message));
  await page.goto(URL_BASE + query, { waitUntil: "networkidle0" });
  await page.waitForFunction("window.__ready === true", { timeout: 30000 });
  await sleep(600); // first frames + pick texture
  return page;
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const logLen = (page) => page.evaluate(() => window.__log.length);
const logSince = (page, n) => page.evaluate((k) => window.__log.slice(k), n);

async function clickAt(page, x, y) {
  await page.mouse.move(x, y);
  await sleep(250);
  await page.mouse.down();
  await sleep(60);
  await page.mouse.up();
  await sleep(350);
}

// ─── phase A: amp = 0 (identity mapping) ───────────────────────────────

console.log("phase A — amp=0, analytic click targets");
{
  const page = await openPage("?amp=0&freeze=1");
  const positions = await page.evaluate(() => window.__boxScreen());
  const names = Object.keys(positions);
  check(names.length === 9, `9 live boxes reported (${names.length})`);

  // Park the mouse off any box first so the first move produces a
  // clean enter.
  await page.mouse.move(5, 5);
  await sleep(300);

  let hits = 0, enters = 0;
  for (const name of ["box-0", "box-2", "box-4", "box-6", "box-8"]) {
    const [x, y] = positions[name];
    const n0 = await logLen(page);
    await clickAt(page, x, y);
    const entries = await logSince(page, n0);
    const enterOk = entries.some(e => e.kind === "enter" && e.target === name);
    const tapOk = entries.some(e => e.kind === "tap" && e.target === name);
    if (enterOk) enters++;
    if (tapOk) hits++;
    else console.error(`  [debug] ${name} @ (${x.toFixed(0)},${y.toFixed(0)}):`, JSON.stringify(entries));
  }
  check(hits === 5, `5/5 predicted box centers tapped the right inner box (${hits})`);
  check(enters >= 4, `hover enter fired on the way (${enters}/5)`);

  // Tap deletes: box-4 is gone — the same screen point now falls
  // through to the portal background.
  {
    const [x, y] = positions["box-4"];
    const n0 = await logLen(page);
    await clickAt(page, x, y);
    const entries = await logSince(page, n0);
    check(entries.some(e => e.kind === "tap" && e.target === "portal-bg"),
      "click on deleted box falls through to portal-bg");
    const live = await page.evaluate(() => window.__live());
    check(!live.includes("box-4") && live.length === 4, `deleted boxes stay gone (live: ${live.length})`);
  }

  // A gap between boxes is portal background from the start.
  {
    const [x0, y0] = positions["box-0"];
    const [x1, y1] = positions["box-1"];
    const gx = (x0 + x1) / 2, gy = (y0 + y1) / 2;
    const n0 = await logLen(page);
    await clickAt(page, gx, gy);
    const entries = await logSince(page, n0);
    check(entries.some(e => e.kind === "tap" && e.target === "portal-bg"),
      "click in the grid gap hits portal-bg");
  }

  await page.close();
}

// ─── phase B: warped ────────────────────────────────────────────────────

console.log("phase B — amp=0.05, frozen warp");
{
  const page = await openPage("?amp=0.05&freeze=1&t0=0.7");

  // Hover scan: sweep a coarse grid, collect which inner boxes light up.
  const seen = new Set();
  let lastBoxPos;
  for (let gy = 0; gy < 5; gy++) {
    for (let gx = 0; gx < 5; gx++) {
      const x = 80 + gx * 160, y = 60 + gy * 120;
      const n0 = await logLen(page);
      await page.mouse.move(x, y);
      await sleep(180);
      const entries = await logSince(page, n0);
      for (const e of entries) {
        if (e.kind === "enter" && e.target.startsWith("box-")) {
          seen.add(e.target);
          lastBoxPos = [x, y, e.target];
        }
      }
    }
  }
  check(seen.size >= 3, `hover through the warp resolves distinct boxes (${seen.size} seen: ${[...seen].join(",")})`);

  // Click the last hovered box position — it must tap THAT box and
  // delete it.
  if (lastBoxPos !== undefined) {
    const [x, y, name] = lastBoxPos;
    const n0 = await logLen(page);
    await clickAt(page, x, y);
    const entries = await logSince(page, n0);
    check(entries.some(e => e.kind === "tap" && e.target === name),
      `warped click taps the hovered box (${name})`);
    const live = await page.evaluate(() => window.__live());
    check(!live.includes(name), `warped tap deleted ${name}`);
  } else {
    check(false, "no box hovered during warp scan");
  }

  await page.screenshot({ path: "/tmp/warp-portal.png" });
  console.log("  screenshot: /tmp/warp-portal.png");
  await page.close();
}

b.disconnect();
stopVite();
console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
