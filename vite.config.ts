// Vite library-mode build for `@aardworx/wombat.dom`.
//
// Why Vite instead of plain `tsc`: we need the
// `@aardworx/wombat.shader-vite` plugin to transform inline shader
// markers (`vertex(...)` / `fragment(...)` / `compute(...)`) into
// build-time `__wombat_stage(...)` literals before publish. Without
// this the markers ship as runtime stubs that throw, so consumer
// apps would have to wire the same plugin in their own Vite config
// to use any of our default surfaces / picking shaders / text-sdf
// effect.
//
// `tsc` still emits the .d.ts declarations alongside (via
// `vite-plugin-dts`); the source path -> dist path mapping is
// preserved by `preserveModules: true` so subpath exports
// (`./scene`, `./jsx-runtime`, `./jsx-dev-runtime`) keep working.

import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { readdirSync, statSync } from "node:fs";
import { resolve, relative } from "node:path";
import dts from "vite-plugin-dts";
import { wombatShader } from "@aardworx/wombat.shader-vite";
import { adaptiveMemoPlugin } from "@aardworx/wombat.adaptive/plugin";

const here = fileURLToPath(new URL(".", import.meta.url));
const srcDir = resolve(here, "src");

/** Walk `src/` for every .ts / .tsx file (excluding tests). Used as
 *  the `entry` map so Vite emits one .js per source file with the
 *  same relative path under `dist/`. */
function collectEntries(): Record<string, string> {
  const entries: Record<string, string> = {};
  const walk = (dir: string): void => {
    for (const e of readdirSync(dir)) {
      const full = resolve(dir, e);
      if (statSync(full).isDirectory()) { walk(full); continue; }
      if (!/\.(ts|tsx)$/.test(e)) continue;
      if (e.endsWith(".test.ts") || e.endsWith(".test.tsx")) continue;
      const rel = relative(srcDir, full).replace(/\.(tsx?)$/, "");
      entries[rel] = full;
    }
  };
  walk(srcDir);
  return entries;
}

export default defineConfig({
  plugins: [
    // Memoize aval/aset/alist/amap combinator call-sites so identical
    // (source, fn-closure-deps) tuples collapse to a single shared
    // adaptive node at runtime. Without this, helpers like
    // `autoInjectedUniforms` allocate a fresh `compose(view, proj)`
    // aval per leaf, blowing up the reactive graph proportional to
    // scene-leaf count instead of unique-input count.
    adaptiveMemoPlugin(),
    wombatShader({ rootDir: here }),
    dts({
      entryRoot: "src",
      outDir: "dist",
      include: ["src/**/*.ts", "src/**/*.tsx"],
      exclude: ["**/*.test.ts", "**/*.test.tsx"],
      // Keep declarationMap on so consumers' tooling can jump from
      // .d.ts back into our .ts sources (matches the old tsc setup).
      tsconfigPath: "tsconfig.build.json",
    }),
  ],
  esbuild: {
    // Self-package automatic-runtime JSX. The single .tsx in
    // wombat.dom (`scene/renderControl.tsx`) needs jsx lowering; the
    // runtime resolves through the consumer-side `node_modules/
    // @aardworx/wombat.dom/jsx-runtime` (= the previous published
    // version during bootstrap), which is API-stable across our
    // versions. If this ever drifts, swap the JSX literal for a
    // manual jsx-runtime call.
    jsx: "automatic",
    jsxImportSource: "@aardworx/wombat.dom",
  },
  build: {
    target: "es2022",
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
    minify: false, // library output should stay readable
    lib: {
      entry: collectEntries(),
      formats: ["es"],
    },
    rollupOptions: {
      // Don't bundle deps — every `@aardworx/...` and node-builtin
      // imports stay external. Consumers re-resolve them.
      external: (id) => {
        if (id.startsWith("node:")) return true;
        if (id.startsWith("@aardworx/")) return true;
        if (id === "happy-dom") return true;
        // Same-package relative imports are NOT external — they get
        // emitted as separate chunks via preserveModules.
        return false;
      },
      output: {
        preserveModules: true,
        preserveModulesRoot: "src",
        entryFileNames: "[name].js",
      },
    },
  },
});
