# Scene templates — staging compiler for the scene graph

Status: **M0 + M1 shipped** (2026-07-17, `src/scene/template.ts`);
M1 wired into `buildRenderObject` (validates uniforms actually READ in
the stage IR — decl lists over-approximate; aval-hole-bound and
instance-attribute names count as provided). Verified end-to-end on
TileRenderer: a misspelled `LineColour` is named once per effect
variant, zero false positives on the full app. M2/M3 open.

## Problem

A compiled Sg leaf retains ~10 KB of scene machinery (measured on
TileRenderer `?anno=5000`, see the wombat-adaptive diet notes): its own
TraversalState with materialized uniform HashMap merges, per-leaf lazy
avals that are mostly constant-folds or duplicates of sibling values,
copied handler arrays, and many near-empty collections. None of that is
required by the sg API's semantics — the tree is immutable and
construction is deterministic, so identical structure is shareable.
Leaf *count* is currently the only memory knob users have, which forces
batching decisions (e.g. instanced segments) that the heap renderer's
O(buckets) megacall was designed to make unnecessary.

## Idea

Treat `sg { ... }` / `<Sg ...>` as a staged program:

- The **static spine** — scope kinds, uniform names, effect identity,
  pick flags, nesting — is the same for every instance produced by the
  same source location. Hash-cons it into a `SceneTemplate`, compiled
  ONCE: per-template resolution plans, validation, handler chains.
- The **holes** — avals, handler closures, buffer views, dynamic
  children — are the only per-instance data. A compiled leaf retains
  `(template, holes)` plus its draw/pick records; the TraversalState
  becomes a *transient* value that exists only during lowering.

Because the compiler lives in wombat.dom (JS), **both TypeScript and
Fable consumers benefit** — Fable emits plain JS modules, so the
library is the common denominator. A Fable compiler plugin can later
act as a thin front-end: assign template identities at build time
(skipping structural hashing) and turn template diagnostics into
compile errors. Same IR, two entry points.

## Milestones

### M0 — shape extraction + interning (`src/scene/template.ts`)
- `stageNode(node) : { template, holes }` — walk an SgNode spine,
  split static structure from holes, hash-cons via a structural key.
- Boundaries: `Adaptive`, `Group`/`UnorderedGroup` children, `Leaf`
  geometry buffers, any aval/function/BufferView → hole.
- `templateStats()` — distinct templates vs staged instances (the
  sharing ratio is the go/no-go signal for M2).

### M1 — template-level validation
- Per (template, effect): compare the effect schema's declared
  uniforms/attributes against what the template's scopes + the
  standard auto-injected uniforms can resolve. Warn ONCE per pair on
  missing/unresolvable names (today these fail silently at draw time).
- Dev-mode only (`console.warn`), no behavior change.

### M2 — template-based lowering (the memory collapse)
- Per (template, parent-context) compute a **resolution plan**: for
  each name the effect needs, where it comes from — hole index, shared
  scope value, auto-injected derived, or constant.
- Leaf lowering applies the plan to the holes array → flat resolved
  inputs. The RenderObject stops retaining a per-leaf TraversalState;
  derived avals (ModelViewTrafo & co) are computed per distinct
  (template, parent) — shared by all sibling instances.
- Handler chains: template-level ancestor array + per-instance own
  handlers (shared tail instead of per-leaf copies).
- Target: ≤ 1 KB per leaf residual (user closures + cvals + draw/pick
  records are the irreducible floor).

### M3 — Fable plugin front-end (wombat.fable)
- Collapse the sg CE at build time: emit module-level template tables
  + `templateInstance(id, holes)` calls; no builder records, no
  runtime hashing.
- Static diagnostics: unknown uniform names (FShade effects are
  compiled in the same build, schemas available), the bare-op-beside-
  yield trap, control flow in DOM CEs.

## Non-goals / constraints

- Hash-consing keys on aval IDENTITY, never value — two leaves with
  distinct cvals holding equal values must not merge (reactivity stays
  exact; same rule as the heap dedup pools).
- No API change. Every existing test must stay green after each
  milestone; M2 ships behind a flag until parity is proven.
- Dynamic children (`adaptive`, aset/alist groups) stay runtime — each
  element stages to a template instance again, so collections of
  similar things share one template.
