import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";
import { wombatShader } from "@aardworx/wombat.shader-vite";

export default defineConfig({
  plugins: [
    wombatShader({ rootDir: fileURLToPath(new URL(".", import.meta.url)) }),
  ],
  test: {
    environment: "happy-dom",
    include: ["tests/**/*.test.ts", "tests/**/*.test.tsx"],
  },
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
});
