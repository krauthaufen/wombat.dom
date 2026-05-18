import { defineConfig, type Plugin } from "vite";
import { readFileSync, existsSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { boperators } from "@boperators/plugin-vite";
import { wombatShader } from "@aardworx/wombat.shader-vite";
import { adaptiveMemoPlugin } from "@aardworx/wombat.adaptive/plugin";

// POST /__cam-save → writes body to /tmp/cam-paths/cam-<timestamp>.json
// Used by the demo's "DUMP" button to persist a recorded camera path
// without copy/paste round-trips through a chat client.
const camSavePlugin = (): Plugin => ({
  name: "cam-save",
  configureServer(server) {
    server.middlewares.use("/__cam-save", (req, res) => {
      if (req.method !== "POST") {
        res.statusCode = 405; res.end("POST only"); return;
      }
      const chunks: Buffer[] = [];
      req.on("data", c => chunks.push(c));
      req.on("end", () => {
        try {
          const body = Buffer.concat(chunks).toString("utf8");
          const dir = "/tmp/cam-paths";
          mkdirSync(dir, { recursive: true });
          const file = join(dir, `cam-${Date.now()}.json`);
          writeFileSync(file, body);
          res.setHeader("content-type", "application/json");
          res.end(JSON.stringify({ ok: true, file }));
          console.log(`[cam-save] wrote ${file} (${body.length} bytes, ${(JSON.parse(body) as unknown[]).length} samples)`);
        } catch (e) {
          res.statusCode = 500;
          res.end(String(e));
        }
      });
    });
  },
});

const here = fileURLToPath(new URL(".", import.meta.url));

// Optional HTTPS — drop in `.certs/server.crt` + `server.key` to enable.
// Without certs vite serves plain HTTP on localhost (WebGPU works on
// localhost over HTTP in Chrome).
const certPath = `${here}.certs/server.crt`;
const keyPath  = `${here}.certs/server.key`;
const httpsCfg = existsSync(certPath) && existsSync(keyPath)
  ? { cert: readFileSync(certPath), key: readFileSync(keyPath) }
  : undefined;

export default defineConfig({
  plugins: [camSavePlugin(), adaptiveMemoPlugin(), boperators(), wombatShader({ rootDir: here })],
  server: {
    host: "0.0.0.0",
    port: 8446,
    strictPort: true,
    // Allow vite to serve files reached via `public/` symlinks that
    // resolve outside the project root (e.g. the local Sonnenburghof
    // dataset under ~/projects/TileRenderer/.../wwwroot/data,
    // symlinked into public/sonnenburghof).
    fs: { strict: false },
    ...(httpsCfg !== undefined ? { https: httpsCfg } : {}),
    allowedHosts: [".ts.net", ".loca.lt", "localhost"],
  },
  esbuild: {
    jsx: "automatic",
    jsxImportSource: "@aardworx/wombat.dom",
    // Keep native class fields (`x = v`) instead of esbuild's
    // `__publicField(this, "x", v)` → `Object.defineProperty` downlevel.
    // The wombat packages ship dist with native ES2022 class fields;
    // without this the dev-server transpile / dep-prebundle re-downlevels
    // them and `__defNormalProp` shows up as ~10% of cold-boot CPU
    // (thousands of aval/SgNode constructions, one defineProperty per
    // field). All target browsers (and the headed Chromium we test in)
    // have had native class fields for years.
    target: "es2022",
  },
  build: { target: "es2022" },
  optimizeDeps: { esbuildOptions: { target: "es2022" } },
  define: { global: "globalThis" },
});
