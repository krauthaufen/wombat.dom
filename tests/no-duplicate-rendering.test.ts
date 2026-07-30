// Packaging regression: wombat.rendering must NOT be a hard `dependency`.
//
// It was, pinned `^0.21.3`. An app on wombat.rendering 0.22.x therefore got
// TWO copies installed — its own 0.22.x at the top level, plus a NESTED
// 0.21.x under node_modules/@aardworx/wombat.dom/ (npm nests when the
// declared range excludes the hoisted version). The bundler then resolved
// this package's `import ... from "@aardworx/wombat.rendering"` to the OLD
// nested copy, so the scene layer ran a different rendering build than the
// host app.
//
// Two production failures came out of that split:
//   - device-loss recovery never rendered again (RenderControl rebuilt a
//     Runtime from the stale copy while the app's render objects belonged
//     to the current one), and
//   - the old copy's `isHeapServableTexture` had never heard of
//     `kind: "cube"`, so it fell through to `t.source.kind` on a cube
//     ITexture and threw inside the frame callback.
//
// `wombat.rendering` is a PEER dependency (already declared, optional via
// peerDependenciesMeta) — the host app owns the single instance. The range
// must also stay permissive enough that a newer rendering still satisfies
// it, or npm starts nesting again.

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

// The suite runs under happy-dom, where `import.meta.url` is an http URL —
// `fileURLToPath` rejects it. vitest sets cwd to the project root.
const pkg = JSON.parse(
  readFileSync(resolve(process.cwd(), "package.json"), "utf-8"),
) as {
  dependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
  peerDependenciesMeta?: Record<string, { optional?: boolean }>;
};

const RENDERING = "@aardworx/wombat.rendering";

describe("packaging: single wombat.rendering instance", () => {
  it("does not declare wombat.rendering as a hard dependency", () => {
    expect(pkg.dependencies?.[RENDERING]).toBeUndefined();
  });

  it("declares wombat.rendering as an optional peer dependency", () => {
    expect(pkg.peerDependencies?.[RENDERING]).toBeDefined();
    expect(pkg.peerDependenciesMeta?.[RENDERING]?.optional).toBe(true);
  });

  it("uses a peer range that a newer rendering still satisfies", () => {
    // A caret range pins the minor for 0.x (^0.21.3 excludes 0.22.0), which is
    // exactly what caused the nested install. Require an open-ended range.
    const range = pkg.peerDependencies?.[RENDERING] ?? "";
    expect(range.startsWith("^"), `peer range ${range} pins the 0.x minor`).toBe(false);
    expect(range).toMatch(/^>=/);
  });
});
