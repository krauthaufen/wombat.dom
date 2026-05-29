// Browser-mode vitest config — runs WebGPU tests in real Chromium via
// Playwright. Used for tests that must compile + execute actual WGSL on
// a GPU (the node/happy-dom suite can't: it has no `navigator.gpu`, so
// a hand-written JS reference is the most it can check — and a JS
// reference never catches a shader COMPILE error, as the pick-argmin
// `meta` reserved-keyword bug showed).
//
// Linux + NVIDIA notes (mirrors wombat.rendering):
//   - Use the system Chromium (full build), not Playwright's bundled
//     headless shell, so adapter selection can pick a real GPU via
//     Vulkan. With the WebGPU/Vulkan flags below it picks the discrete
//     GPU where the ICD is discoverable; otherwise it falls back to a
//     conformant CPU impl (SwiftShader) — still valid for CORRECTNESS
//     regression (the shader still compiles + runs).

import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";
import { wombatShader } from "@aardworx/wombat.shader-vite";

const here = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  plugins: [wombatShader({ rootDir: here })],
  define: { global: "globalThis" },
  esbuild: {
    jsx: "automatic",
    jsxImportSource: "@aardworx/wombat.dom",
  },
  resolve: {
    alias: {
      "@aardworx/wombat.dom/jsx-runtime": new URL("./src/jsx-runtime.ts", import.meta.url).pathname,
      "@aardworx/wombat.dom/jsx-dev-runtime": new URL("./src/jsx-dev-runtime.ts", import.meta.url).pathname,
      "@aardworx/wombat.dom": new URL("./src/index.ts", import.meta.url).pathname,
    },
  },
  test: {
    include: ["tests-browser/**/*.test.ts"],
    browser: {
      enabled: true,
      provider: "playwright",
      name: "chromium",
      headless: true,
      providerOptions: {
        launch: {
          executablePath: "/usr/bin/chromium",
          args: [
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan,UseSkiaRenderer",
            "--use-vulkan=native",
            "--ignore-gpu-blocklist",
            "--enable-webgpu-developer-features",
            "--use-angle=vulkan",
          ],
        },
      },
    },
  },
});
