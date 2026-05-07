import { defineConfig } from "vite";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { boperators } from "@boperators/plugin-vite";
import { wombatShader } from "@aardworx/wombat.shader-vite";

const here = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  plugins: [boperators(), wombatShader({ rootDir: here })],
  server: {
    host: "0.0.0.0",
    port: 8446,
    strictPort: true,
    https: {
      cert: readFileSync(`${here}.certs/server.crt`),
      key: readFileSync(`${here}.certs/server.key`),
    },
    allowedHosts: [".ts.net", ".loca.lt", "localhost"],
  },
  esbuild: {
    jsx: "automatic",
    jsxImportSource: "@aardworx/wombat.dom",
  },
  // `poly2tri` (transitive dep via wombat.base path triangulator)
  // is a CJS package that references Node's `global`. Vite's
  // browser bundle has no `global` — alias it to `globalThis` so
  // the optimizeDeps prebundle resolves cleanly.
  define: {
    global: "globalThis",
  },
});
