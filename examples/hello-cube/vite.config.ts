import { defineConfig } from "vite";
import { boperators } from "@boperators/plugin-vite";

// `DefaultSurfaces.basic` ships pre-built via `parseShader + stage`
// (raw shader source compiled at first use), so the wombat.shader-
// vite marker plugin isn't strictly necessary for this demo.
// Include it once we add inline `vertex(…)` / `fragment(…)` calls
// in the app code.

export default defineConfig({
  plugins: [boperators()],
  server: {
    port: 5175,
    // Permit Tailscale-issued hostnames + LAN access. Vite's
    // default host-allowlist refuses requests that come in
    // through a reverse proxy under a different name.
    allowedHosts: [".ts.net", ".loca.lt", "localhost"],
  },
  esbuild: {
    jsx: "automatic",
    jsxImportSource: "@aardworx/wombat.dom",
  },
});
