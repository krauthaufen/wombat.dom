// Scene templates — the staging compiler's front half (M0/M1 of
// docs/scene-templates.md).
//
// `stageNode` walks an SgNode spine and splits it into:
//   - a structural KEY covering everything static: node kinds, uniform
//     names, effect ids, handler kinds, constant scope values, leaf
//     attribute name sets, nesting;
//   - a HOLES array holding everything per-instance: avals, closures,
//     BufferViews, dynamic children.
//
// Identical source locations produce identical keys, so the registry
// hash-conses them into shared `SceneTemplate`s. The sharing ratio
// (instances / templates) is the signal that drives M2 (template-based
// lowering); `templateStats()` exposes it.
//
// M1: `validateTemplateEffect` checks — once per (template, effect) —
// that every uniform the effect's IR declares is resolvable from the
// template's uniform scopes, the traversal's auto-injected names, or a
// caller-supplied allowlist, and `console.warn`s the difference. Today
// a misspelled uniform silently renders wrong; this names it.
//
// Constant folding rule: an aval with `isConstant` whose value is a
// primitive is folded into the KEY (so `PickPriority 1` scopes share);
// everything else stays a hole. Identity, not value, is what keys
// non-constant avals — two distinct cvals never merge (same rule as
// the heap dedup pools).

import type {
  SgNode, UniformBag, EventHandlers,
} from "./sg.js";
import type { aval } from "@aardworx/wombat.adaptive";
import type { Effect } from "@aardworx/wombat.shader";
import type { BufferView } from "@aardworx/wombat.rendering";
import { AVal, HashMap } from "@aardworx/wombat.adaptive";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SceneTemplate {
  /** Registry-assigned id, stable within a session. */
  readonly id: number;
  /** Structural key this template was interned under. */
  readonly key: string;
  /** Number of holes an instance of this template carries. */
  readonly holeCount: number;
  /** Uniform names provided by Uniform scopes on the spine (static bags). */
  readonly providedUniforms: ReadonlySet<string>;
  /** name → hole index of the providing (innermost) Uniform-scope value.
   *  The plan's slot table (instance-tables M2) resolves through this. */
  readonly uniformHoles: ReadonlyMap<string, number>;
  /** True when the spine carries a Dynamic (amap) uniform bag — such
   *  subtrees are not row-lowerable (their key set is reactive). */
  readonly hasDynamicUniforms: boolean;
  /** Hole indices of Trafo scope values BEFORE any Instanced scope
   *  (outer→inner order) — compose over the parent's model. */
  readonly preTrafoHoles: ReadonlyArray<number>;
  /** Hole indices of Trafo scope values AFTER the innermost Instanced
   *  scope — compose over identity (pushInstancing resets model). */
  readonly postTrafoHoles: ReadonlyArray<number>;
  /** True when the spine contains an Instanced scope. */
  readonly hasInstancing: boolean;
  /** effect.id of the innermost Shader scope ON the spine (per-item
   *  effect application — usually hoistable), or undefined. */
  readonly spineEffectId: string | undefined;
  /** Instances staged against this template (stats). */
  instances: number;
}

export interface StagedNode {
  readonly template: SceneTemplate;
  readonly holes: ReadonlyArray<unknown>;
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

const registry = new Map<string, SceneTemplate>();
let nextId = 0;

export interface TemplateStats {
  readonly templates: number;
  readonly instances: number;
  /** instances / templates — the sharing ratio. */
  readonly ratio: number;
  readonly top: ReadonlyArray<{ id: number; instances: number; holes: number; key: string }>;
}

export function templateStats(topN: number = 10): TemplateStats {
  let instances = 0;
  const all: SceneTemplate[] = [];
  for (const t of registry.values()) { instances += t.instances; all.push(t); }
  all.sort((a, b) => b.instances - a.instances);
  return {
    templates: registry.size,
    instances,
    ratio: registry.size === 0 ? 0 : instances / registry.size,
    top: all.slice(0, topN).map((t) => ({
      id: t.id, instances: t.instances, holes: t.holeCount,
      key: t.key.length > 160 ? t.key.slice(0, 157) + "..." : t.key,
    })),
  };
}

/** Test hook — forget every template and validation memo. */
export function resetTemplates(): void {
  registry.clear();
  validated.clear();
  nextId = 0;
}

// ---------------------------------------------------------------------------
// Key building
// ---------------------------------------------------------------------------

function isAvalLike(v: unknown): v is aval<unknown> {
  return typeof v === "object" && v !== null
    && typeof (v as { getValue?: unknown }).getValue === "function"
    && "isConstant" in (v as object);
}

/**
 * Encode a scope VALUE into the key, or record it as a hole.
 * Constant-primitive avals and bare primitives fold into the key;
 * plain JSON-able objects (BlendState-style config records) fold via
 * a stable stringify; anything else (non-constant avals, functions,
 * class instances) becomes a hole.
 */
function encodeValue(v: unknown, holes: unknown[], parts: string[]): void {
  if (isAvalLike(v)) {
    if (v.isConstant) {
      // AVal.force OK: isConstant — immutable by definition.
      const c = AVal.force(v);
      if (typeof c === "number" || typeof c === "boolean" || typeof c === "string") {
        parts.push(`c:${String(c)}`);
        return;
      }
    }
    parts.push("?");
    holes.push(v);
    return;
  }
  switch (typeof v) {
    case "number": case "boolean": case "string":
      parts.push(`v:${String(v)}`);
      return;
    case "undefined":
      parts.push("u");
      return;
    case "object": {
      if (v === null) { parts.push("n"); return; }
      const s = stableStringify(v);
      if (s !== undefined) { parts.push(`j:${s}`); return; }
      parts.push("?");
      holes.push(v);
      return;
    }
    default:
      parts.push("?");
      holes.push(v);
      return;
  }
}

/** JSON-stringify with sorted keys; `undefined` when the value holds
 *  anything non-plain (function, aval, class instance, cycle). */
function stableStringify(v: unknown): string | undefined {
  try {
    const seen = new Set<object>();
    const enc = (x: unknown): string => {
      if (x === null) return "null";
      switch (typeof x) {
        case "number": case "boolean": return String(x);
        case "string": return JSON.stringify(x);
        case "undefined": return "undefined";
        case "object": break;
        default: throw new Error("non-plain");
      }
      const o = x as object;
      if (seen.has(o)) throw new Error("cycle");
      if (isAvalLike(o)) throw new Error("non-plain");
      const proto = Object.getPrototypeOf(o);
      if (Array.isArray(o)) {
        seen.add(o);
        const r = `[${o.map(enc).join(",")}]`;
        seen.delete(o);
        return r;
      }
      if (proto !== Object.prototype && proto !== null) throw new Error("non-plain");
      seen.add(o);
      const keys = Object.keys(o).sort();
      const r = `{${keys.map((k) => `${k}:${enc((o as Record<string, unknown>)[k])}`).join(",")}}`;
      seen.delete(o);
      return r;
    };
    return enc(v);
  } catch {
    return undefined;
  }
}

function sortedNames<V>(m: HashMap<string, V>): string[] {
  const names: string[] = [];
  for (const [k] of m) names.push(k);
  names.sort();
  return names;
}

function encodeHandlerBag(h: EventHandlers, holes: unknown[], parts: string[]): void {
  for (const phase of ["capture", "bubble"] as const) {
    const bag = h[phase];
    if (bag === undefined) continue;
    const kinds = Object.keys(bag).sort();
    for (const k of kinds) {
      parts.push(`${phase}:${k}?`);
      holes.push((bag as Record<string, unknown>)[k]);
    }
  }
}

function encodeUniformBag(out: WalkOut, bag: UniformBag, parts: string[]): void {
  const { holes, provided, uniformHoles } = out;
  if (bag.kind === "Static") {
    parts.push("Us(");
    for (const name of sortedNames(bag.entries)) {
      provided.add(name);
      parts.push(name);
      // NB: value avals hole by IDENTITY even when constant — a folded
      // uniform VALUE would merge leaves that legitimately differ.
      parts.push("?");
      // walk is outer→inner, so a later (deeper) scope overwrites —
      // matching the traversal's inner-wins shadowing.
      uniformHoles.set(name, holes.length);
      holes.push(bag.entries.tryFind(name));
    }
    parts.push(")");
  } else {
    parts.push("Ud?");
    out.hasDynamicUniforms = true;
    holes.push(bag.entries);
  }
}

// ---------------------------------------------------------------------------
// The walker
// ---------------------------------------------------------------------------

interface WalkOut {
  readonly parts: string[];
  readonly holes: unknown[];
  readonly provided: Set<string>;
  readonly uniformHoles: Map<string, number>;
  hasDynamicUniforms: boolean;
  readonly preTrafoHoles: number[];
  readonly postTrafoHoles: number[];
  hasInstancing: boolean;
  spineEffectId: string | undefined;
}

function walk(node: SgNode, out: WalkOut): void {
  const { parts, holes } = out;
  switch (node.kind) {
    case "Empty":
      parts.push("E");
      return;
    case "Leaf": {
      parts.push("L(va:");
      for (const n of sortedNames(node.vertexAttributes)) {
        parts.push(n + "?");
        holes.push(node.vertexAttributes.tryFind(n) as BufferView);
      }
      if (node.instanceAttributes !== undefined) {
        parts.push(";ia:");
        for (const n of sortedNames(node.instanceAttributes)) {
          parts.push(n + "?");
          holes.push(node.instanceAttributes.tryFind(n));
        }
      }
      if (node.indices !== undefined) { parts.push(";ix?"); holes.push(node.indices); }
      if (node.storageBuffers !== undefined) {
        parts.push(";sb:");
        for (const n of sortedNames(node.storageBuffers)) {
          parts.push(n + "?");
          holes.push(node.storageBuffers.tryFind(n));
        }
      }
      parts.push(";dc");
      encodeValue(node.drawCall, holes, parts);
      parts.push(")");
      return;
    }
    case "Group": case "UnorderedGroup": {
      const children = node.children;
      // Constant collections inline their children into the key — the
      // common `sg { op; op; child }` output is a constant group.
      if (children.isConstant) {
        // AVal.force OK: isConstant collection content.
        const content = AVal.force(children.content as aval<Iterable<SgNode>>);
        const arr = [...content];
        parts.push(`${node.kind === "Group" ? "G" : "UG"}${arr.length}(`);
        for (const c of arr) walk(c, out);
        parts.push(")");
      } else {
        parts.push(node.kind === "Group" ? "G?" : "UG?");
        holes.push(children);
      }
      return;
    }
    case "AdaptiveGroup":
      parts.push("A?");
      holes.push(node.child);
      return;
    case "Delay":
      parts.push("D?");
      holes.push(node.create);
      return;
    case "Shader":
      // effect.id is a build-time stable template hash — static.
      parts.push(`S(${node.effect.id})`);
      out.spineEffectId = node.effect.id;
      walk(node.child, out);
      return;
    case "Uniform":
      encodeUniformBag(out, node.bag, parts);
      walk(node.child, out);
      return;
    case "On":
      parts.push("On(");
      encodeHandlerBag(node.handlers, holes, parts);
      parts.push(")");
      walk(node.child, out);
      return;
    case "Trafo": {
      parts.push("T");
      const v = node.value;
      const bucket = out.hasInstancing ? out.postTrafoHoles : out.preTrafoHoles;
      if (Array.isArray(v)) {
        parts.push(`[${v.length}]`);
        for (const e of v) { parts.push("?"); bucket.push(holes.length); holes.push(e); }
      } else {
        // Trafo3d instances and avals are per-instance values.
        parts.push("?");
        bucket.push(holes.length);
        holes.push(v);
      }
      walk(node.child, out);
      return;
    }
    case "PickTag":
      // ALWAYS a hole — tags are per-row keys (often small ints);
      // folding them into the key would explode one template per row.
      parts.push("PT?");
      out.holes.push((node as { value: unknown }).value);
      walk((node as { child: SgNode }).child, out);
      return;
    case "Instanced": {
      out.hasInstancing = true;
      parts.push("I(");
      encodeValue(node.count, holes, parts);
      if (node.trafos !== undefined) { parts.push(";tr?"); holes.push(node.trafos); }
      parts.push(";at:");
      for (const n of sortedNames(node.attributes)) {
        parts.push(n + "?");
        holes.push(node.attributes.tryFind(n));
      }
      parts.push(")");
      walk(node.child, out);
      return;
    }
    default: {
      // Every remaining kind is a single-value scope over `child`:
      // Active, View, Proj, Cursor, PickThrough, Intersectable,
      // PixelSnapRadius, PickPriority, render-state scopes, pick
      // scopes, Index, Mode, VertexAttributes, InstanceAttributes, …
      // Encode kind + each non-child own property generically.
      const rec = node as unknown as Record<string, unknown>;
      parts.push(`${node.kind}(`);
      const keys = Object.keys(rec).filter((k) => k !== "kind" && k !== "child").sort();
      for (const k of keys) {
        if (k === "attributes" || k === "index") {
          // VertexAttributes / InstanceAttributes scopes & Index override.
          const v = rec[k];
          if (v instanceof HashMap) {
            parts.push(`${k}:`);
            for (const n of sortedNames(v as HashMap<string, unknown>)) {
              parts.push(n + "?");
              holes.push((v as HashMap<string, unknown>).tryFind(n));
            }
            continue;
          }
        }
        parts.push(k + "=");
        encodeValue(rec[k], holes, parts);
      }
      parts.push(")");
      const child = (node as unknown as { child?: SgNode }).child;
      if (child !== undefined) walk(child, out);
      return;
    }
  }
}

/**
 * Stage a node: split static structure from holes and intern the
 * structure. Two nodes produced by the same source location (same
 * shape, same static values) share one `SceneTemplate`.
 */
export function stageNode(node: SgNode): StagedNode {
  const out: WalkOut = {
    parts: [], holes: [], provided: new Set(), uniformHoles: new Map(),
    hasDynamicUniforms: false, preTrafoHoles: [], postTrafoHoles: [],
    hasInstancing: false, spineEffectId: undefined,
  };
  walk(node, out);
  const key = out.parts.join(" ");
  let t = registry.get(key);
  if (t === undefined) {
    t = {
      id: nextId++,
      key,
      holeCount: out.holes.length,
      providedUniforms: out.provided,
      uniformHoles: out.uniformHoles,
      hasDynamicUniforms: out.hasDynamicUniforms,
      preTrafoHoles: out.preTrafoHoles,
      postTrafoHoles: out.postTrafoHoles,
      hasInstancing: out.hasInstancing,
      spineEffectId: out.spineEffectId,
      instances: 0,
    };
    registry.set(key, t);
  }
  t.instances++;
  return { template: t, holes: out.holes };
}

// ---------------------------------------------------------------------------
// M1 — template × effect validation
// ---------------------------------------------------------------------------

/** Names `TraversalState.tryFindUniform` derives without any scope. */
const AUTO_UNIFORMS: ReadonlySet<string> = new Set([
  "ModelTrafo", "ViewTrafo", "ProjTrafo",
  "ModelViewTrafo", "ViewProjTrafo", "ModelViewProjTrafo",
  "ModelTrafoInv", "ViewTrafoInv", "ProjTrafoInv",
  "ModelViewTrafoInv", "ViewProjTrafoInv", "ModelViewProjTrafoInv",
  "NormalMatrix", "CameraLocation", "LightLocation", "ViewportSize",
  // First-class / injected by the render control & picking infra.
  "PickId", "ViewportSizePx",
]);

/** Uniform names an Effect's stage IR declares (pre-compile). */
export function effectUniformNames(effect: Effect): ReadonlySet<string> {
  const names = new Set<string>();
  // Defensive: substituted/synthetic effects (tests, wrappers) may not
  // carry full stage IR — no names means no warnings, never a throw.
  if (!Array.isArray(effect.stages)) return names;
  // Collect uniforms actually READ (`ReadInput` with scope "Uniform")
  // rather than the stage's declaration list — module-level shader
  // files declare every uniform their file mentions, so decls
  // over-approximate what a given entry uses. Generic structural walk:
  // robust to IR node shapes without enumerating them.
  const scan = (x: unknown): void => {
    if (x === null || typeof x !== "object") return;
    if (Array.isArray(x)) { for (const e of x) scan(e); return; }
    const o = x as Record<string, unknown>;
    if (o["kind"] === "ReadInput" && o["scope"] === "Uniform" && typeof o["name"] === "string") {
      names.add(o["name"] as string);
    }
    for (const k of Object.keys(o)) scan(o[k]);
  };
  for (const st of effect.stages) {
    const values = st?.template?.values;
    if (!Array.isArray(values)) continue;
    for (const v of values) {
      if (v.kind === "Function" || v.kind === "Entry" || v.kind === "Constant") scan(v);
    }
  }
  // Uniforms bound ON the effect (aval holes) are self-provided —
  // they never need a scope provider.
  for (const st of effect.stages) {
    const holes = (st as { avalHoles?: Record<string, unknown> }).avalHoles;
    if (holes !== undefined) for (const n of Object.keys(holes)) names.delete(n);
  }
  return names;
}

const validated = new Set<string>();

const effectNamesMemo = new Map<string, ReadonlySet<string>>();

function effectUniformNamesMemo(effect: Effect): ReadonlySet<string> {
  let names = effectNamesMemo.get(effect.id);
  if (names === undefined) {
    names = effectUniformNames(effect);
    effectNamesMemo.set(effect.id, names);
  }
  return names;
}

/**
 * Leaf-lowering hook (M1 wiring): warn about effect uniforms that
 * neither the accumulated uniform scopes, the auto-injected set, nor
 * `extraProvided` resolve. Deduped on (effect, missing-set) so a
 * 10k-leaf subtree warns once. Cost per leaf: one memoized name-set
 * lookup + a few Set probes.
 */
export function warnUnresolvedUniforms(
  effect: Effect,
  scopeUniforms: { tryFind(name: string): unknown },
  extraProvided?: ReadonlySet<string>,
  /** Per-leaf instance attributes — the instancing rewrite feeds these
   *  names from vertex inputs, so their uniform decls are satisfied. */
  instanceProvided?: { tryFind(name: string): unknown },
): void {
  let missing: string[] | undefined;
  for (const name of effectUniformNamesMemo(effect)) {
    if (AUTO_UNIFORMS.has(name)) continue;
    if (scopeUniforms.tryFind(name) !== undefined) continue;
    if (extraProvided !== undefined && extraProvided.has(name)) continue;
    if (instanceProvided !== undefined && instanceProvided.tryFind(name) !== undefined) continue;
    (missing ??= []).push(name);
  }
  if (missing === undefined) return;
  const key = `${effect.id}|${missing.join(",")}`;
  if (validated.has(key)) return;
  validated.add(key);
  const entries = Array.isArray(effect.stages)
    ? effect.stages.flatMap((st) => {
        const vs = (st as { template?: { values?: readonly { kind: string; entry?: { name?: string } }[] } }).template?.values;
        return Array.isArray(vs) ? vs.filter((v) => v.kind === "Entry").map((v) => v.entry?.name ?? "?") : [];
      })
    : [];
  console.warn(
    `[wombat.dom] effect ${effect.id} [${entries.join("+")}]: uniforms with no provider in scope: ` +
    `${missing.join(", ")} — they will read as zero/defaults at draw time.`,
  );
}

/**
 * Warn — once per (template, effect) — about uniforms the effect
 * declares that neither the template's Uniform scopes, the traversal's
 * auto-injected set, nor `extraProvided` can resolve. Purely
 * diagnostic; returns the missing names for tests.
 */
export function validateTemplateEffect(
  staged: StagedNode,
  effect: Effect,
  extraProvided?: Iterable<string>,
): string[] {
  const memoKey = `${staged.template.id}|${effect.id}`;
  if (validated.has(memoKey)) return [];
  validated.add(memoKey);
  const extra = extraProvided !== undefined ? new Set(extraProvided) : undefined;
  const missing: string[] = [];
  for (const name of effectUniformNames(effect)) {
    if (staged.template.providedUniforms.has(name)) continue;
    if (AUTO_UNIFORMS.has(name)) continue;
    if (extra !== undefined && extra.has(name)) continue;
    missing.push(name);
  }
  if (missing.length > 0) {
    console.warn(
      `[wombat.dom] scene template #${staged.template.id}: effect ${effect.id} ` +
      `declares uniforms with no provider on this subtree: ${missing.join(", ")}. ` +
      `They will read as zero/defaults at draw time.`,
    );
  }
  return missing;
}

// ---------------------------------------------------------------------------
// Efficiency accounting (quantified — see the design rule: a warning
// without counts is noise; only aggregate numbers above thresholds are
// ever reported, and at most once per burst).
// ---------------------------------------------------------------------------

export type RowBailReason =
  | "dynamic-uniform-bag"
  | "injected-uniforms-pass"
  | "auto-uniforms-off"
  | "multi-leaf-subtree"
  | "per-leaf-effect-scope"
  | "staging-failed";

interface BailStat { count: number; sameEffectId: string | undefined; mixed: boolean }

const _rowBails = new Map<RowBailReason, BailStat>();
let _rowsLowered = 0;

/** Measured on TileRenderer @5k (2026-07-17): per-row overhead of the
 *  classic path vs row lowering, bytes. Used ONLY for impact estimates
 *  in reports — never for behavior. */
const CLASSIC_OVERHEAD_BYTES = 700;

export function recordRowLowered(): void {
  _rowsLowered++;
}

export function recordRowBail(reason: RowBailReason, effectId?: string): void {
  let s = _rowBails.get(reason);
  if (s === undefined) {
    s = { count: 0, sameEffectId: effectId, mixed: false };
    _rowBails.set(reason, s);
  }
  s.count++;
  if (effectId !== undefined && s.sameEffectId !== effectId) s.mixed = true;
  scheduleEfficiencyReport();
}

export interface EfficiencyReport {
  readonly rowsLowered: number;
  readonly bails: ReadonlyArray<{
    reason: RowBailReason;
    count: number;
    /** Estimated avoidable bytes (count × measured classic overhead). */
    estBytes: number;
    hint: string;
  }>;
  /** Template-explosion suspects: templates with ≥2 instances are
   *  healthy; a burst of single-instance templates is the smell. */
  readonly singleInstanceTemplates: number;
  readonly templates: number;
  readonly instances: number;
}

const HINTS: Record<RowBailReason, string> = {
  "dynamic-uniform-bag": "an amap-backed Sg.Uniform prevents row lowering — use per-key avals in a static bag if the key set is fixed",
  "injected-uniforms-pass": "pass-level injected uniforms (OIT build) route rows through the classic path for that pass only",
  "auto-uniforms-off": "autoUniforms:false disables the derivations rows rely on",
  "multi-leaf-subtree": "children lowering to several leaves aren't rows — split the collection so each child is one leaf",
  "per-leaf-effect-scope": "Sg.Effect applied per child — apply it ONCE above the collection",
  "staging-failed": "staging threw (internal — please report)",
};

export function sceneEfficiency(): EfficiencyReport {
  let single = 0, templates = 0, instances = 0;
  for (const t of registry.values()) {
    templates++;
    instances += t.instances;
    if (t.instances === 1) single++;
  }
  const bails = [..._rowBails.entries()]
    .map(([reason, s]) => ({
      reason,
      count: s.count,
      estBytes: s.count * CLASSIC_OVERHEAD_BYTES,
      hint: reason === "per-leaf-effect-scope" && !s.mixed && s.sameEffectId !== undefined
        ? `${HINTS[reason]} (all ${s.count} use the SAME effect ${s.sameEffectId})`
        : HINTS[reason],
    }))
    .sort((a, b) => b.count - a.count);
  return { rowsLowered: _rowsLowered, bails, singleInstanceTemplates: single, templates, instances };
}

/** Test hook. */
export function resetEfficiency(): void {
  _rowBails.clear();
  _rowsLowered = 0;
}

// Report policy: ONE console line per quiet-period burst, only when a
// single reason crosses BOTH thresholds (instances AND estimated MB) —
// three inefficient objects stay silent forever; five thousand speak
// once, with numbers.
const REPORT_MIN_COUNT = 500;
const REPORT_MIN_BYTES = 1 << 20;
let _reportTimer: ReturnType<typeof setTimeout> | undefined;
const _reported = new Set<string>();

function scheduleEfficiencyReport(): void {
  if (_reportTimer !== undefined) return;
  _reportTimer = setTimeout(() => {
    _reportTimer = undefined;
    const r = sceneEfficiency();
    for (const b of r.bails) {
      if (b.count < REPORT_MIN_COUNT || b.estBytes < REPORT_MIN_BYTES) continue;
      const key = `${b.reason}|${Math.log2(b.count) | 0}`;
      if (_reported.has(key)) continue;
      _reported.add(key);
      console.warn(
        `[wombat.dom] ${b.count} scene items lowered on the classic path ` +
        `(~${(b.estBytes / 1048576).toFixed(1)} MB avoidable): ${b.hint}`,
      );
    }
  }, 1000);
}
