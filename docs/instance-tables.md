# Instance tables — M2 proper (leaf = row)

Status: **design, implementation-ready** (2026-07-17). Prerequisites
all shipped: template staging (M0), uniform-scope chains, applicator
hoists, adaptive fast paths, plain-array double types, small-map
HashMap. Companion: `scene-templates.md`.

## Why (measured, TileRenderer `?anno=5000` editable)

Per-leaf JS heap after all representation fixes: ~11.7 KB. Breakdown:

| component                                   | ≈/leaf | removable by tables? |
|---------------------------------------------|--------|----------------------|
| app data (`Pts` V3d[], epochs, record)      | 2.0 KB | no (user's)          |
| app avals (Pts→seg→fst/snd, color, count)   | 1.0 KB | no (user allocates); collection idioms remove |
| retained SgNode tree (held by the aset)     | 1.5 KB | partially (see §Retention) |
| TraversalState + derived-uniform residue    | 0.6 KB | yes                  |
| RO + providers + BufferViews + drawCall     | 1.5 KB | yes → row            |
| pick scope + handler arrays                 | 0.4 KB | yes → row            |
| reader/subscription records (collection)    | 1.6 KB | yes → one reader per group |
| trie nodes of the big maps (cmap, caches)   | 1.3 KB | yes → row keyed on int |
| misc (closures, WeakRefs, contexts)         | ~1.8 KB| partially            |

Target: ≤ 1 KB/leaf with the current frontend; ≤ 500 B with the
collection idioms. North star: `?anno=200000` on an iPhone.

## Design

### Where it hooks

`compileScene`, `UnorderedGroup` lowering. For each child node:
`stageNode(child)` → `(template, holes)`. Children are grouped by
`template.id`; each group lowers into ONE `TemplateGroup` instead of N
independent RenderTree leaves. Children whose template is unsuitable
(dynamic bags, Delay, nested collections) fall back to today's path —
correctness NEVER depends on staging. Fallback is per child, so mixed
groups work.

### The plan (one per template × parent context)

Computed on first instance, cached on the template keyed by parent
TraversalState identity:

- composed effect (pick chain + OIT compose, existing caches),
- pipeline state (existing sharing),
- uniform slot table: effect-needed name → `Hole(i)` | `Scope(aval)` |
  `Auto(derived)` — resolved ONCE against the parent chain,
- attribute layout (vertex from the shared leaf, instance from holes),
- handler-chain prefix (ancestors) — per-row only the own handler hole,
- validation (M1) runs once per plan.

### The row

Per instance, retained:

```
row: { holes: unknown[], pickId, drawHandle, rowIndex }
```

- RenderObject fields materialize FROM the plan at registration and
  hand off to wombat.rendering; the row does not keep a state, node
  tree, or provider graph. Uniform provider for the draw = tiny
  `RowProvider { plan, holes }` resolving by slot table.
- Model: `Trafo` holes ride `RO.modelChain` (GPU compose) — no CPU
  ModelView derivation per row on the heap path.
- Pick registry entry references plan (shared scope data) + row.

### Reactivity

- Hole avals are subscribed ONCE each by the draw registration exactly
  as today (heap pools by aval identity) — user avals stay the user's.
- Structural deltas (aset add/remove) map to row add/remove; ONE group
  reader instead of per-child machinery.
- No per-row wrapper avals are created by the library — anything the
  plan needs derived from a hole is computed inside the pool update
  callback, not as an aval node.

### Retention

The staged child releases its SgNode tree: the group stores
`(template, holes)`. NOTE: an `aset<ISceneNode>` still retains the
nodes in its content HashSet — the library-side fix is a weak
staging cache (node → staged) so the nodes' own maps/bags at least
are not duplicated; the full fix is the Fable-plugin front-end (M3)
where construction emits template instances directly.

### Idioms (available, never required)

- Collection-level `OnTap` on the group: handler receives the row id
  (pick resolves hit → row). Removes per-row closures.
- Derived-uniform selection: one `SelectedId` cval + heap derived-mode
  rule. Removes per-row selection cvals + color avals.

## Acceptance

- All existing suites green, plus a template-group parity suite:
  same scene lowered with tables on/off renders identical images and
  identical pick results (headed-chrome test).
- `?anno=5000` delta ≤ 5 MB library share (app data excluded).
- `?anno=200000` loads and picks on iPhone (final gate, with idioms).

## Execution order

1. `templatePlan.ts`: plan computation + RowProvider + slot table
   (pure, unit-testable against buildRenderObject outputs).
2. UnorderedGroup interception behind internal constant
   `TEMPLATE_GROUPS` (default on once parity proves out; no user flag).
3. Parity suite; 5k measurement.
4. Pick-registry row entries; group reader consolidation.
5. Idiom support (collection OnTap row id; SelectedId rule example).
6. M3 plugin front-end (wombat.fable) — construction bypass.
