import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { wombatShader } from "@aardworx/wombat.shader-vite";
import { adaptiveMemoPlugin } from "@aardworx/wombat.adaptive/plugin";

// This example runs against the WORKSPACE wombat.dom sources (it
// exercises portal picking, which is newer than the published
// package). The shader plugin's rootDir is the repo root so inline
// markers in BOTH the example and ../../src get lifted.
const repo = fileURLToPath(new URL("../..", import.meta.url));

export default defineConfig({
  plugins: [adaptiveMemoPlugin(), wombatShader({ rootDir: repo })],
  resolve: {
    alias: [
      // Marker-intrinsic shim — see shader-types-shim.js.
      { find: "@aardworx/wombat.shader/types", replacement: fileURLToPath(new URL("./shader-types-shim.js", import.meta.url)) },
      { find: "@aardworx/wombat.dom/jsx-runtime", replacement: repo + "src/jsx-runtime.ts" },
      { find: "@aardworx/wombat.dom/jsx-dev-runtime", replacement: repo + "src/jsx-dev-runtime.ts" },
      { find: "@aardworx/wombat.dom/scene", replacement: repo + "src/scene/index.ts" },
      { find: "@aardworx/wombat.dom", replacement: repo + "src/index.ts" },
    ],
  },
  server: {
    port: 5176,
    allowedHosts: [".ts.net", "localhost"],
  },
  optimizeDeps: {
    // Marker-only helpers (`abs`, `sin`, `texture`, …) are erased by
    // the wombatShader transform but their import statements remain;
    // a PREBUNDLED @aardworx/wombat.shader would hard-fail the named-
    // import link. Keep it un-optimized so vite's lenient module
    // rewriting applies.
    exclude: [
      "@aardworx/wombat.shader",
      "@aardworx/wombat.adaptive",
      "@aardworx/wombat.base",
      "@aardworx/wombat.rendering",
    ],
    // …but its CJS dep still needs prebundle interop (`import ts from
    // "typescript"` has no ESM default otherwise).
    include: [
      "@aardworx/wombat.shader > typescript",
      "@aardworx/wombat.base > poly2tri",
      "@aardworx/wombat.base > libtess",
      "@aardworx/wombat.base > clipper-lib",
      "@aardworx/wombat.base > opentype.js",
    ],
    esbuildOptions: { target: "es2022", define: { global: "globalThis" } },
  },
  // poly2tri's UMD entry references node's `global`.
  define: { global: "globalThis" },
  esbuild: {
    jsx: "automatic",
    jsxImportSource: "@aardworx/wombat.dom",
  },
});
