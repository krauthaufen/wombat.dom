import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "happy-dom",
    include: ["tests/**/*.test.ts", "tests/**/*.test.tsx"],
  },
  esbuild: {
    jsx: "automatic",
    jsxImportSource: "@aardworx/adaptive-ui",
  },
  resolve: {
    alias: {
      "@aardworx/adaptive-ui/jsx-runtime": new URL("./src/jsx-runtime.ts", import.meta.url).pathname,
      "@aardworx/adaptive-ui/jsx-dev-runtime": new URL("./src/jsx-dev-runtime.ts", import.meta.url).pathname,
      "@aardworx/adaptive-ui": new URL("./src/index.ts", import.meta.url).pathname,
    },
  },
});
