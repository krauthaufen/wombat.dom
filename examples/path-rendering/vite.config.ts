import { defineConfig } from "vite";
import { wombatShader } from "@aardworx/wombat.shader-vite";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { boperators } from "@boperators/plugin-vite";

const here = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  server: {
    host: "0.0.0.0",
    port: 8443,
    strictPort: true,
    https: {
      cert: readFileSync(`${here}.certs/server.crt`),
      key: readFileSync(`${here}.certs/server.key`),
    },
    allowedHosts: [".ts.net", ".loca.lt", "localhost"],
  },
  plugins: [
    boperators(),
    // Inlines `effect(vertex(...), fragment(...))` markers at
    // build time. Ambient `declare const` uniforms become real
    // Uniform ValueDefs in the compiled IR.
    wombatShader({ rootDir: here }),
  ],
  esbuild: {
    jsx: "automatic",
    jsxImportSource: "@aardworx/wombat.dom",
  },
  define: {
    // poly2tri's UMD entry references node's `global`. Polyfill so
    // it initialises in the browser.
    global: "globalThis",
  },
});
