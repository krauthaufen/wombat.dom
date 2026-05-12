import { defineConfig } from "vite";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { boperators } from "@boperators/plugin-vite";
import { wombatShader } from "@aardworx/wombat.shader-vite";
import { adaptiveMemoPlugin } from "@aardworx/wombat.adaptive/plugin";

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
  plugins: [adaptiveMemoPlugin(), boperators(), wombatShader({ rootDir: here })],
  server: {
    host: "0.0.0.0",
    port: 8445,
    strictPort: true,
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
