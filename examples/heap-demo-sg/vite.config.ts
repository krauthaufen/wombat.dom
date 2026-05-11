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
  },
  define: { global: "globalThis" },
});
