# Deferred updates: a hidden subtree catches up when shown

Status: design (2026-11-12).

## Why (measured, pro3d-web state-ref, 714 annotations)

A scene-state swap that expands the annotation groups (Bookmark_0) takes
~2.3 s busy, ~1 s of it building 714 list rows (47,094 DOM nodes) in the
Annotations tab — a tab that is not shown. Inactive tab panes are live: every
binding in them flushes on the next frame like a visible one.

## What it is

An element may declare `deferUpdates` (a boolean or an `aval<boolean>`): while
it is true, the bindings of the element's **children** (alist rows inserted and
removed, attribute and text bindings, nested components) do not flush; when it
turns false they flush once. The element's **own** attributes stay live — a tab
pane's own `active` class must change while its content waits.

Who declares it: the component that hides the subtree (the toy's `Tabs` for
its inactive panes). The app does nothing.

## How

The DOM scheduler is passed down the mount; an element with `deferUpdates`
mounts its children with a `DeferringScheduler` over its parent's:

- `notify(b)` while hidden parks `b` (a set; a binding marked many times parks
  once — marking callbacks fire once per out-of-date transition anyway);
  while shown it forwards to the parent scheduler (no other cost: a visible
  subtree, or one without the prop, is exactly as before).
- `forget(b)` (a disposed binding) drops it from the parked set and forwards.
- The `deferUpdates` aval itself is watched by a binding on the **parent**
  scheduler; when it reads false, the parked bindings flush in that same
  frame.

**Catch-up = net change, not a replay.** A binding's flush pulls its reader's
delta since its last pull: a row inserted and removed while hidden is no op,
a text changed ten times is written once. Nothing is recorded per change.

**DOM state is kept.** The hidden subtree's elements stay mounted (scroll
position, focus, input text survive); rows that did not change keep their
nodes. Only what changed is touched on show.

**Nesting.** A deferring scheduler's parent may itself defer: the inner
subtree's watch and bindings park in the outer one until it shows.

**Idle work.** Unrelated: scene lowering and idle registration live under a
`<RenderControl>`, whose own adaptive pulls are not DOM bindings. A hidden
pane holding a render control defers its DOM bindings only.

**Initial mount** is unchanged: a subtree hidden at mount is still built once
(the first pull is synchronous). Showing it later with nothing changed costs
nothing — no parked bindings.

**Slicing the catch-up.** On show, the parked bindings flush in one frame. For
the 714-row case that is the same work the swap did before, moved to the
moment the user opens the tab. A per-row sliced catch-up would need the alist
binding to apply a delta partially across frames; not done unless the
measured show stalls (it is one frame of ~the old swap cost, only after a swap
that changed the hidden list, and only when the tab is opened).

## Tests (tests/deferred-updates.test.ts)

Rows inserted/removed while hidden are applied on show (net change); a text
and an attribute catch up; the element's own attributes stay live; a visible
subtree flushes as before; a binding disposed while parked never flushes;
nested deferral; the subtree's nodes are the same objects after show.
