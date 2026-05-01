import { defineConfig } from "vite";

// `DefaultSurfaces.basic` ships pre-built via `parseShader + stage`
// (raw shader source compiled at first use), so the wombat.shader-
// vite marker plugin isn't strictly necessary for this demo.
// Include it once we add inline `vertex(…)` / `fragment(…)` calls
// in the app code.

export default defineConfig({
  server: { port: 5175 },
  esbuild: {
    jsx: "automatic",
    jsxImportSource: "@aardworx/wombat.dom",
  },
});
