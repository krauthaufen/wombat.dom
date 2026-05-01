import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "happy-dom",
    include: ["tests/**/*.test.ts", "tests/**/*.test.tsx"],
  },
  esbuild: {
    jsx: "automatic",
    jsxImportSource: "@aardworx/wombat.adaptive-ui",
  },
  resolve: {
    alias: {
      "@aardworx/wombat.adaptive-ui/jsx-runtime": new URL("./src/jsx-runtime.ts", import.meta.url).pathname,
      "@aardworx/wombat.adaptive-ui/jsx-dev-runtime": new URL("./src/jsx-dev-runtime.ts", import.meta.url).pathname,
      "@aardworx/wombat.adaptive-ui": new URL("./src/index.ts", import.meta.url).pathname,
    },
  },
});
