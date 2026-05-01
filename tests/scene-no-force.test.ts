// Guard: derivePipelineState in src/scene/compile.ts must not call
// `.force()` on render-state avals. The pipeline-state aval shape
// flows through to wombat.rendering's PreparedRenderObject without
// a single AVal.force on the render path.

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const compileTs = resolve(here, "..", "src", "scene", "compile.ts");

describe("compile.ts — no force on render-state avals", () => {
  it("derivePipelineState does not call AVal.force / .force()", () => {
    const file = readFileSync(compileTs, "utf8");
    // Locate the function and slice out its body (between the
    // function-open brace and the first matching close at column 0).
    const startMatch = /^function derivePipelineState\b/m.exec(file);
    expect(startMatch, "derivePipelineState not found").not.toBeNull();
    const start = startMatch!.index;
    // Walk braces.
    let depth = 0;
    let end = -1;
    let i = start;
    while (i < file.length) {
      const c = file[i]!;
      if (c === "{") depth++;
      else if (c === "}") { depth--; if (depth === 0) { end = i + 1; break; } }
      i++;
    }
    expect(end, "could not bracket derivePipelineState body").toBeGreaterThan(start);
    const body = file.slice(start, end);
    expect(body).not.toMatch(/AVal\.force\b/);
    expect(body).not.toMatch(/\.force\(/);
  });
});
