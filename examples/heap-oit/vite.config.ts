import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { wombatShader } from "@aardworx/wombat.shader-vite";

const here = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  plugins: [wombatShader({ rootDir: here })],
  server: {
    port: 5176,
    // Permit Tailscale-issued hostnames + LAN access. Vite's
    // default host-allowlist refuses requests that come in
    // through a reverse proxy under a different name.
    allowedHosts: [".ts.net", ".loca.lt", "localhost"],
  },
  esbuild: {
    jsx: "automatic",
    jsxImportSource: "@aardworx/wombat.dom",
  },
  define: {
    // poly2tri's UMD entry references node's `global`.
    global: "globalThis",
  },
  optimizeDeps: {
    esbuildOptions: {
      define: { global: "globalThis" },
    },
  },
});
