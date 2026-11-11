# Split rows: multi-leaf children of a collection become rows

Status: implemented (2026-11-11), wombat.dom 0.25.16.

## Why (measured, pro3d-web on wombat.dom 0.25.15)

A collection (`UnorderedGroup`) lowers its children to thin rows of ONE row
set when each child lowers to exactly one Leaf with one plan; any other child
is lowered on the classic path (a full `RenderObject` walk per leaf, the
"classic path" warning). In pro3d-web every annotation is a child of the
annotations collection, and none is a single leaf:

    polyline (Shaler 2,157, Beaver Falls 2,824):
      Active[guarded](Trafo(U(U(U~[S(Mode(Leaf))]))))
    dip and strike (316 / 666):
      Active[guarded](Trafo(U(U(U~[S(Mode(Leaf)), PickThrough(G[4 × U?(S(Mode(Leaf)))]), S(Mode(Leaf))]))))

(`Active[guarded]`: the app's prebuilt on/off switch; `U~`: the annotation's
parts, a dynamic collection; `G[…]`: the dip-and-strike glyph.) The rows
therefore formed inside each annotation's own part collection: a row set per
annotation, plans per annotation (a parent state each), nothing shared.

What it costs (alternating runs, same build): load, heap and frame time do
not move with row lowering on or off; wombat.dom's lowering of these children
is about 0.5 s of Beaver Falls' ~5 s load-to-settled CPU; the classic
`RenderObject`s about 700 B each (~2–3 MB). The renderer side gains little
either way: the hybrid renderer consumes row sets directly only on a static
spine (`Ordered`/`Unordered` arrays from the root); row sets under reactive
containers are materialized there into one `RenderObject` per row, as
classic leaves are. So the gain of this change is the lowering (rows built
from (template, holes) on the fast path, no classic walk), memory, and the
warning; making row sets under reactive containers first-class in the hybrid
renderer is a separate, later step (`wombat.rendering` hybridScene).

## Rule

A child of a collection whose subtree is **row-safe scopes over groups of
leaves** contributes **each leaf as a row of the collection**, in a row set
per plan. The child's other parts (anything not row-safe) are lowered
classically, as today. A child that already is a single-leaf row takes the
unchanged path.

### Which spines are row-safe (decided once, at lowering, per child)

Walking from the child's root:

- **Scopes** that staging encodes and the classic lowering applies as state:
  `Trafo`, `Shader`, `Uniform`, `PickThrough`, `Mode`, render-state and
  pick-flag scopes, `On`, … — everything the template walk treats as a
  single-value scope. They stay in each leaf's chain (so they are part of its
  template and holes, and a per-child `Sg.Shader` with the same effect gives
  the same plan).
- **Groups** — `Group` and `UnorderedGroup`, constant or dynamic — are split:
  each element continues the walk with the same prefix. Splitting an ordered
  group is sound because draw order inside a Render command is not
  load-bearing (`flattenTree`'s contract; the hybrid renderer draws unordered).
- **`Active`** (non-constant) is taken out of the chain and becomes a **gate**
  of every leaf below it (see "Active as the row's flag").
- A pre-staged `Row` (a collection mapping wrapped by `rowWrap` inside the
  child) is walked through: its materialized subtree under the prefix (its
  staging held no prefix). A `Row` child itself is a single fast row already
  and is never split.
- Anything else — `AdaptiveGroup`, `Delay`, `Instanced`, `Pass` — ends the walk: the prefix chain plus that node is one classic
  part, lowered as today (with its original `Active` nodes).

A child is split only when its spine reaches a group or a non-constant
`Active` before a leaf; otherwise it takes today's path unchanged.

### A leaf's chain

Each leaf yields a chain: its prefix scopes (copied with the next scope as
child) ending in the leaf. The chain is lowered, staged and admitted exactly
as a single-leaf child is today (classic lowering of the first row of a plan
and its pick capture, `constructFastRow` for the next ones), so every
existing proof (heap assertion, fast-row evidence, pick context) applies
unchanged. One that is not admitted (not heap-asserted, dynamic uniforms, …)
is a classic part.

### One row set per plan

The group keeps an anchor per plan (template × parent × effect) instead of
one. The first plan's set stays where the group's single set is today (on the
static spine when the group is); further sets are `Rows` trees in the group's
classic set. "mixed-plan-group" no longer bails: a second shape gets its own
set. Each anchor captures its own pick context.

### Adaptive membership

A split child is a set of entries: `collect` over its groups (a dynamic group
maps its elements with `mapUse`, so an element leaving releases its rows and
pick ids), and each leaf's entry is gated by its membership (below) with
`chooseA`. Single-leaf children stay one entry each; the group unions both.

### `Active` as the row's flag

Gates are not staged (a per-row `active` aval would make the template unsafe
for fast rows, and pushing it on the state would give every child its own
plan again). Instead the row carries `active` = AND of its gates (and of the
state's own active), as a classic leaf under the same scopes would. A gate's
eager mode becomes the row's **membership**, mirroring `Sg.active`'s lowering:

- constant `active`, or `eager` with idle registration off: walked and
  built at once (registered hidden);
- `eager: true` / `"guarded"` with idle registration: nothing below is read
  (not even the item's part collection) until the switch is shown or its
  idle ticket — one per switch, as before — is released; the slice walks and
  builds it;
- no `eager`: nothing below is read until the first show;
- `"guarded"`: each leaf's row is read under the renderer's guard
  (`heapValuesRaise`, exported by wombat.rendering 0.25.12) when it is
  lowered (in the slice): one whose values do not raise is taken in at once,
  a constant member with no edge on its switch (so a show re-evaluates
  nothing per leaf); one that raises (or whose lowering raises) waits for
  its switches, is left out while hidden and tried again when one changes;
  once shown it is an ordinary row (a raise is a real error, as
  `RenderTree.guarded` treats a subtree).

An entry, once built, is kept (as `Sg.active` keeps a lowered subtree).

## Tests (wombat.dom)

(tests/split-rows.test.ts) Multi-leaf child → rows of the outer set; nested group; mixed plans in one
collection (a set each); dynamic inner group membership (add, remove,
release); `Active` in the spine (row flag, idle release, non-eager first
show); a guarded branch that raises while hidden (left out, taken in when
shown); nothing below an idle-time switch read before its release; a
single-leaf child unchanged; picking through `PickThrough` rows. The build's
memo transform rewrites adaptive combinator calls: keep type arguments out of
them (`ASet.empty<T>()` inside an `ASet.bind` became a runtime reference to
the type) — the unit tests run the sources, so check the built package in
an app.

## Fast rows for data that changes

A row is admitted only when heap-safe, and built on the fast path only when
its draw call and buffers are provably host / pinned. A mesh whose data
changes has adaptive buffers and draw call; it carries construction proofs:
`markHostBufferAVal` on each buffer and `markHeapSafeDrawAVal(drawCall,
kind)` (kind pinned, first vertex/instance 0) — the producer's mapping
builds them so. A per-row value on a pipeline scope (`Sg.mode` with an aval)
makes a template unsafe for fast rows, rightly (a row set shares its
pipeline): such a value belongs to the shape. (fablecube-toy's `meshA`
does both.)

## Result (pro3d-web, alternating runs, same server, cache off; the machine
was loaded by other jobs, load average 10–23)

No classic-path warning on Shaler, Beaver Falls, DQ2_DQ3; renders match
(screenshots, Beaver Falls also after showing all). Shaler over 16 rounds
(min / median): ready 2.94 / 4.02 → 2.84 / 3.32 s, CPU to ready 2.69 / 3.78 →
2.62 / 3.08 s, settled CPU 3.89 / 4.77 → 3.84 / 4.49 s. Beaver Falls over 8:
ready 2.08 / 2.24 → 1.84 / 2.07 s, CPU to ready 1.78 / 1.90 → 1.62 / 1.78 s,
settled CPU 4.88 / 5.72 → 4.55 / 4.94 s. Live heap Shaler 216 → 205 MB,
Beaver Falls 422 → 407 MB (every run). Within wombat.dom, Shaler's lowering
before ready ~350 → ~190 ms. Show, clicks and orbit within noise.
