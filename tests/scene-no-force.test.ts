// Audit: `src/scene/compile.ts` MUST NOT call `.force()` on the live
// render path. The few remaining forces are construction-boundary
// (compile-scene time) or operate on constants — they are explicitly
// allowlisted by line-prefix here. Each entry is paired with a
// "Why force here:" comment in the source.
//
// The rule: a frame's render walk (resolve / record) must never
// reach a `.force()`. The forces below run ONCE at `compileScene`
// time when the SG tree shape is being lowered to a `RenderTree`.
//
// `dispatcher.ts` legitimately forces inside pointer event handlers
// (the dispatcher runs OUTSIDE adaptive context — pointer events are
// not adaptive computations). It is NOT scanned here.

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const compileTs = resolve(here, "..", "src", "scene", "compile.ts");

/**
 * Allowlisted force sites in compile.ts. Each entry is a substring of
 * the source line that contains the `.force(`. Keep tight — adding
 * to this list demands a "Why force here:" justification in the
 * source comment immediately preceding the line.
 */
const ALLOWLIST: readonly string[] = [
  // sceneUsesPassStatic — STATIC structural pre-pass, runs once at
  // compileScene time to decide whether to bucket-by-pass.
  "for (const c of node.children.content.force()) if (sceneUsesPassStatic(c)) return true;",
  "case \"AdaptiveGroup\": return sceneUsesPassStatic(node.child.force());",
  // collectByPass — STATIC pass-bucketing fallback (see lowerByPass
  // doc for tradeoff). Bucket contents are reactive via lowerLeaf.
  "for (const c of node.children.content.force()) collectByPass(c, state, opts, buckets);",
  "collectByPass(node.child.force(), state, opts, buckets);",
  // Uniform key-set static snapshot at the bucketing boundary.
  "const entries = node.bag.kind === \"Static\" ? node.bag.entries : node.bag.entries.content.force();",
  // state.noEvents.isConstant fast path — force on a constant aval has
  // no upstream dependency to lose; collapses to skipRegister at compile.
  "const noEventsNow = constantNoEvents ? state.noEvents.force() : false;",
  // state.active.isConstant fast path — force on a constant aval has
  // no upstream dependency to lose; collapses RenderTree at compile.
  "return state.active.force() ? baseTree : RenderTree.empty;",
  // splitTexturesFromUniforms — structural classification at compile-
  // scene time. Reads each merged uniform's current value once to
  // decide texture vs scalar; runs only during leaf lowering.
  "const current = v.force();",
  // Pick-path selection — `forcePixelPicking` / `noEvents` constants
  // are resolved once at register-time so the picker can decide
  // pixel-vs-BVH path. Reactive forms fall through to the "bvh"
  // default. Constant-aval one-shot reads at compile-scene time.
  "const fppConst = state.forcePixelPicking.isConstant && state.forcePixelPicking.force();",
  "const noEventsConst = state.noEvents.isConstant && state.noEvents.force();",
];

describe("compile.ts — zero force on the live render path", () => {
  it("every `.force(` call site is on the documented allowlist", () => {
    const file = readFileSync(compileTs, "utf8");
    const offenders: string[] = [];
    const lines = file.split("\n");
    for (let i = 0; i < lines.length; i++) {
      const ln = lines[i]!;
      if (!/\.force\(|AVal\.force/.test(ln)) continue;
      // Skip comments — line begins with `//` or `*` after whitespace.
      const trimmed = ln.trimStart();
      if (trimmed.startsWith("//") || trimmed.startsWith("*")) continue;
      const trimmedFull = ln.trim();
      if (ALLOWLIST.some(allowed => trimmedFull.includes(allowed))) continue;
      offenders.push(`${i + 1}: ${trimmedFull}`);
    }
    expect(
      offenders,
      `unexpected .force() in compile.ts — add "Why force here:" comment and append to ALLOWLIST:\n${offenders.join("\n")}`,
    ).toEqual([]);
  });

  it("no AVal.force(...) call sites in compile.ts (only `.force()` method calls remain in the allowlist; comments mentioning AVal.force are fine)", () => {
    const file = readFileSync(compileTs, "utf8");
    const offenders: string[] = [];
    const lines = file.split("\n");
    for (let i = 0; i < lines.length; i++) {
      const ln = lines[i]!;
      if (!/AVal\.force\b/.test(ln)) continue;
      const trimmed = ln.trimStart();
      if (trimmed.startsWith("//") || trimmed.startsWith("*")) continue;
      offenders.push(`${i + 1}: ${ln.trim()}`);
    }
    expect(offenders).toEqual([]);
  });
});
