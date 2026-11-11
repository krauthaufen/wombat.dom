# Split rows: multi-leaf children of a collection become rows

Status: design (2026-11-11), implementing.

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
- Anything else — `AdaptiveGroup`, `Delay`, `Instanced`, `Pass`, `View`/`Proj`,
  a `Row` — ends the walk: the prefix chain plus that node is one classic
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

- constant `active`, or `eager: true` with idle registration off: a member at once;
- `eager: true` / `"guarded"` with idle registration: a member once shown or
  once its idle ticket is released (one ticket per leaf, built in the slice,
  `target` = its entry), cancelled when it leaves;
- no `eager`: a member from its first show (`shownOnce`), kept;
- `"guarded"`: while hidden, the row's heap values are read under the
  renderer's guard (`heapValuesRaise`, exported by wombat.rendering): a row
  whose values raise is left out and tried again when `active` changes; once
  shown a raise is a real error (as `RenderTree.guarded` does for a subtree).

An entry, once built, is kept (as `Sg.active` keeps a lowered subtree).

## Tests (wombat.dom)

Multi-leaf child → rows of the outer set; nested group; mixed plans in one
collection (a set each); dynamic inner group membership (add, remove,
release); `Active` in the spine (row flag, idle release, non-eager first
show); a guarded branch that raises while hidden (left out, taken in when
shown); a single-leaf child unchanged; picking through `PickThrough` rows.
