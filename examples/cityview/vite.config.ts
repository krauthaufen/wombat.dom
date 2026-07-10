import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { wombatShader } from "@aardworx/wombat.shader-vite";
import { adaptiveMemoPlugin } from "@aardworx/wombat.adaptive/plugin";

// Runs against the WORKSPACE wombat.dom sources (portal picking,
// pick producer — newer than the published package). Same setup as
// examples/warp-portal; see that config for the why of each knob.
const repo = fileURLToPath(new URL("../..", import.meta.url));

export default defineConfig({
  // Relative base so the built bundle works under any URL prefix
  // (e.g. https://georg.haaser.net/vienna_demo/).
  base: "./",
  // `NO_PUBLIC=1 vite build` skips copying public/ (the vienna symlink
  // points at ~2 GB of districts — data is deployed separately).
  ...(process.env.NO_PUBLIC ? { publicDir: false as const } : {}),
  plugins: [adaptiveMemoPlugin(), wombatShader({ rootDir: repo })],
  resolve: {
    alias: [
      { find: "@aardworx/wombat.shader/types", replacement: fileURLToPath(new URL("./shader-types-shim.js", import.meta.url)) },
      { find: "@aardworx/wombat.dom/jsx-runtime", replacement: repo + "src/jsx-runtime.ts" },
      { find: "@aardworx/wombat.dom/jsx-dev-runtime", replacement: repo + "src/jsx-dev-runtime.ts" },
      { find: "@aardworx/wombat.dom/scene", replacement: repo + "src/scene/index.ts" },
      { find: "@aardworx/wombat.dom", replacement: repo + "src/index.ts" },
    ],
  },
  server: {
    port: 5178,
    host: true,
    allowedHosts: [".ts.net", "localhost"],
    // Tailscale cert (same provisioning as heap-demo/renderbench) —
    // WebGPU needs a secure context on non-localhost origins.
    ...((): { https?: { cert: Buffer; key: Buffer } } => {
      const dir = join(homedir(), ".local/share/heap-demo-cert");
      const cert = join(dir, "airtop.crt"), key = join(dir, "airtop.key");
      return existsSync(cert) && existsSync(key)
        ? { https: { cert: readFileSync(cert), key: readFileSync(key) } }
        : {};
    })(),
  },
  optimizeDeps: {
    exclude: [
      "@aardworx/wombat.shader",
      "@aardworx/wombat.adaptive",
      "@aardworx/wombat.base",
      "@aardworx/wombat.rendering",
    ],
    include: [
      "@aardworx/wombat.shader > typescript",
      "@aardworx/wombat.base > poly2tri",
      "@aardworx/wombat.base > libtess",
      "@aardworx/wombat.base > clipper-lib",
      "@aardworx/wombat.base > opentype.js",
    ],
    esbuildOptions: { target: "es2022", define: { global: "globalThis" } },
  },
  define: { global: "globalThis" },
});
