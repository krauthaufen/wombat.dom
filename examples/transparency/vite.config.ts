import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { wombatShader } from "@aardworx/wombat.shader-vite";

const here = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  plugins: [wombatShader({ rootDir: here })],
  server: {
    port: 5176,
    allowedHosts: [".ts.net", ".loca.lt", "localhost"],
  },
  define: { global: "globalThis" },
  optimizeDeps: { esbuildOptions: { define: { global: "globalThis" } } },
});
