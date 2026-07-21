# Unified DOM ↔ Scene Event Propagation

*Status: IMPLEMENTED in wombat.dom — the full router model (whole mount
subtree unified; RenderControl canvas is the async scene-leaf) plus the
camera wiring. Keyboard/focus unification and the app migration
(TileRenderer markers/Drag module) are the remaining steps — see "What is
built" at the bottom.*

## The principle

The scene graph (`<Sg>`) is meant to be a **natural extension of the DOM
into 3D**. Event handling is where that promise is kept or broken. The
principle:

> There is **one** event-propagation model. A pointer/keyboard event
> does a single **capture** pass down the DOM ancestor chain, dives into
> the **scene** at the canvas (scene capture then scene bubble, resolved
> by picking), then **bubbles** back out up the DOM chain. Every callback
> along that path — on a `<div>`, on the canvas, or on a 3D leaf —
> participates in the same walk, and **any** of them can
> `stopPropagation()` with the same meaning.

Everything else people want falls out of this principle as a
*consequence*, with no bespoke code:

- **"Stop the camera during a drag."** The camera is an ordinary DOM
  handler on the render-control element (an ancestor of the canvas). A
  3D marker's `OnDragStart` is *inner* to it. The marker stops
  propagation → the outer camera handler never runs. No special case.
- **Focus, hit-testing, capture, hover enter/leave** — all one model
  across the boundary.
- **Overlay HTML UI over the canvas** composes with scene events under
  the same capture/bubble rules.

If any of these needs a bespoke mechanism (a "controller registry", a
CPU screen-space hit-test in app code, a capture-phase DOM listener that
front-runs the camera), that is a smell: it means the unification is
incomplete and we are patching a symptom. The correct fix is always to
complete the walk, not to special-case the consequence.

## Why the current split is the bug

Today wombat.dom has **two** event systems:

1. **Scene events** — `PickDispatcher` (`src/scene/picking/dispatcher.ts`)
   listens on the canvas, resolves the pick (async GPU argmin readback,
   fused with a sync BVH ray), and runs its **own** capture/bubble walk
   over the scene scope path (`runCaptureBubble`). This half already
   works and matches Aardvark.Dom.
2. **DOM element handlers** — `src/bindings/attr.ts` binds `on*` props as
   raw `el.addEventListener(...)`. These fire through **native browser
   propagation**, synchronously, entirely outside the scene walk.

The camera controller (portable `FreeFlyController`) is case 2: a native
`Dom.OnPointerDown`/`Move`/`Up` on the render-control element. So:

- Its handler fires **synchronously**, *before* the scene pick resolves
  (the pick is a microtask later — WebGPU has **no** synchronous buffer
  map; `mapAsync` is the only API, so this can never be made sync).
- A scene handler therefore **cannot** stop it — there is no shared walk
  to stop within.

The async readback is *not* the blocker (and can't be removed). The
blocker is the **split**: DOM handlers don't live in the same walk as
scene handlers.

## How Aardvark.Dom solves it (the reference)

Aardvark.Dom fully owns propagation. The canonical pieces:

- **`TraversalState.handleEvent`** (`Aardvark.Dom/SceneGraph/TraversalState.fs`):
  `runCapture` root→leaf, then `runBubble` leaf→root; handlers return
  `bool`; `List.forall` short-circuits → returning `false` stops.
- **`SceneEventHandler = { Capture: (SceneEvent -> bool) list; Bubble:
  (SceneEvent -> bool) list }`** (`SceneEvent.fs`). Composition across
  nesting concatenates the lists.
- **`seenByAardvark`** (`Aardvark.Dom.Remote/AbstractRemoteHtmlBackend.fs`,
  ~line 424): the single installed listener, on **first sight** of a
  bubbling event, synchronously does
  ```js
  if (e.bubbles) { if (e.seenByAardvark) return; e.seenByAardvark = true; }
  if (flags.preventDefault) e.preventDefault();
  if (flags.pointerCapture) aardvark.setPointerCapture(el, e.pointerId, true/false);
  aardvark.trigger(el, ...);   // hand off to Aardvark's OWN walk
  ```
  Native propagation is thereby **replaced** by Aardvark's walk. DOM
  element handlers are registered with Aardvark (`aardvark.setListener` /
  `getListenerFlags`), not as independent native listeners, so Aardvark
  controls their firing order and can stop them. **This is done for DOM
  handlers too, not just scene** — that is the whole point.

The camera stays *DOM-world* (a handler on the render-control element).
It is not a scene node. It simply participates in the unified walk as a
DOM bubble handler that is an ancestor of the canvas.

## Target design for wombat.dom

One walk per event:

```
   DOM capture   :  outermost ancestor → … → render-control → canvas
                    (fire each element's CAPTURE handlers, in order)
        │
        ▼
   SCENE          :  pick-resolve at the canvas, then
                    scene capture (root scope → leaf) then
                    scene bubble (leaf → root scope)      ← existing runCaptureBubble
        │
        ▼
   DOM bubble    :  canvas → render-control → … → outermost ancestor
                    (fire each element's BUBBLE handlers, in order)
```

- Any handler returning `false` / calling `stopPropagation()` halts the
  rest of the walk (capture-stop also skips the scene + bubble, matching
  F# `runCapture` short-circuit).
- **`seenByAardvark`-style takeover:** a single synchronous interceptor
  marks the event seen, `preventDefault`s, sets DOM pointer capture, and
  suppresses native propagation so no element handler double-fires.
- **DOM handlers route through the walk:** `attr.ts` registers `on*`
  handlers with the dispatcher (capture/bubble lists per element), rather
  than `el.addEventListener`. The dispatcher knows the DOM ancestor chain
  from the canvas up to the render-control root and walks it.
- **Camera orbit stays cheap:** the camera claims the pointer via DOM
  pointer capture on its down; while captured, moves route straight to it
  and **skip the scene pick** — no GPU readback per orbit-move (the
  concern in `App.fs`'s "no per-move scene handler" note is thereby
  respected).

## Semantics to pin (decide before coding)

1. **Async boundary.** The scene pick is async. Does the *whole* unified
   walk defer to the microtask after the pick (so DOM capture handlers
   also fire late), or do DOM capture handlers fire synchronously and
   only the scene + DOM-bubble portion defer? Aardvark's remote model
   serializes and defers; a local WebGPU model should decide explicitly.
   Recommendation: take over synchronously (`seenByAardvark` +
   `preventDefault` + pointer-capture set), run the *rest* of the walk
   after the pick resolves. Camera (a bubble handler) is post-pick
   anyway, so no regression; capture-phase DOM handlers that need sync
   behavior are rare and can be flagged.
2. **Pointer capture ownership.** DOM pointer capture (camera orbit) vs
   the dispatcher's scene-scope capture. A pointer can be captured by a
   DOM element OR a scene scope, never both. Define precedence: a scene
   scope capture (active drag on a marker) wins; else a DOM element
   capture (camera orbit).
3. **`stopPropagation` across the boundary.** A scene handler stopping
   propagation must prevent the *DOM bubble* (outer camera). A DOM
   capture handler stopping propagation must prevent the *scene* from
   ever dispatching. Both directions must hold.
4. **Which DOM events unify.** Pointer + wheel + keyboard/focus at least.
   Non-interactive events (load, etc.) need not route through the walk.
5. **`seenByAardvark` dedup key.** Per-event flag (`e.__seenByWombat`) so
   nested render controls / multiple registered elements intercept once.

## Implementation order (task #19 → #21)

1. **wombat.dom** — the unified walk (this doc). Gate: `npm test`
   (mock + picking) + `npm run typecheck` + `npm run test:browser`.
   Additive-first where possible; the `attr.ts` routing is the invasive
   part and needs the browser suite.
2. **Portable `FreeFlyController`** — no code change *needed* once the
   walk lands: its DOM handlers already participate as bubble handlers.
   Verify: camera orbits over empty sky + the (pixel-only) tunnel; a
   marker `OnDragStart` returning false suppresses the camera; orbit
   moves skip the scene pick via DOM pointer capture.
3. **App (TileRenderer)** — markers get `Sg.OnDragStart`/`OnDrag`/
   `OnDragEnd` (return false to stop the camera) + `Sg.OnTap` for vertex
   click; **delete** the Drag module's CPU screen-space projection and
   its capture-phase DOM listener hack. Gate: app e2e (apply/ignore drag
   policy — camera untouched) + full suite; deploy.

## What must NOT happen

- The BVH must **never** hold large streaming geometry (e.g. 20k
  overlay-line segments). Line-body/marker picking uses the **pixel**
  path (`Sg.OnTap` + `PixelSnapRadius`), which needs no BVH. (Already the
  case: line-body selection shipped on the pixel path.)
- No app-side re-implementation of hit-testing or camera suppression.
  Those are consequences of the unified walk, not features to add.

## What is built (wombat.dom)

The full router model landed: wombat owns propagation over the entire
mount subtree, with the RenderControl canvas as the async scene-leaf. The
app-side migration is the only remaining step.

**The region router — `RegionRouter` (`src/eventRouter.ts`).** One
capture-phase listener per unified event (pointer/mouse/wheel) on the
**mount root**. On first sight (`__wombatSeen` dedup) it runs wombat's own
walk over the DOM ancestor chain: capture root→target, then — if the
target is a registered scene-leaf — the scene sub-walk, then bubble
target→root. `attr.ts` auto-registers every `on*` (and `onFooCapture` for
the capture phase) with the enclosing region instead of `addEventListener`;
`Scope.region` is set on the mount root and inherited by `child()` scopes,
so dynamically-added rows join the walk. `mount()` installs the router.
Any handler stops the walk (`return false` or `stopPropagation()`), one
meaning across the subtree. **Boundary:** an unstopped event continues
natively and bubbles out of the root (the host page sees the subtree as
one opaque node); a stopped one is halted at the root. Foreign native
listeners inside the subtree still fire when nothing stops. Only BUBBLING
pointer/mouse/wheel events unify; keyboard/focus and non-bubbling events
stay native (deliberate follow-up).

**The canvas as scene-leaf.** `RenderControl` registers its canvas as the
region's `SceneLeaf`; `PickDispatcher.attach(canvas, resolvePixel,
{ region })` then does NOT install native pointer/wheel listeners — the
router drives it via the leaf's `dispatch(name, ev, prop)`, which runs the
pick + scene (+ camera participant) walk, returns the scene walk's promise
so the router bubbles DOM ancestors *after* the pick, and threads the
shared `prop` so a scene stop suppresses the ancestor bubble. The router
takes canvas events over synchronously (`ev.stopPropagation()` up front,
so they don't also reach the host) since the async pick can't stop native
propagation retroactively. Keyboard/focus/beforeinput/pointerleave stay
native on the canvas in both modes. Standalone (no region) keeps the
original native-listener path — so the pick unit tests are unchanged.

The camera stays a `PickDispatcher` DOM participant (below), now driven
through the leaf — nothing about its wiring changed.

**Mechanism — `PickDispatcher` (`src/scene/picking/dispatcher.ts`).**
- `registerDomParticipant(p: DomParticipant): DomParticipantHandle`. A
  participant has `capture` / `bubble` handler maps keyed by raw DOM
  event name (`pointerdown/up/move/cancel`, `wheel`, `click`,
  `dblclick`). Participants are kept OUTER→INNER in registration order:
  capture fires front→back, bubble back→front, so the first-registered
  (the camera) is the outermost bubble handler and runs LAST.
- One walk per raw event, post pick-resolve (`runUnified`): DOM capture →
  scene `runCaptureBubble` → DOM bubble, all sharing one stop-flag
  (`WalkProp`, threaded into the MAIN scene event). A scene handler's
  `stopPropagation()` — or an equivalent `return false`, now unified in
  `runCaptureBubble` — halts the DOM bubble, so an inner marker suppresses
  the outer camera. A scene scope holding pointer capture (an active
  drag) owns the pointer outright: the DOM side stays suppressed for the
  whole gesture.
- DOM pointer capture (`handle.capturePointer` / `releasePointer`): while
  a participant holds a pointer, MOVES route straight to it and skip the
  scene pick (orbit fast path); the dispatcher holds the browser capture
  underneath so off-canvas moves keep flowing. A no-move press still
  delivers its `pointerup` through the full walk, so a click/tap on scene
  geometry survives an orbit-claim.
- `seenByWombat`-style takeover: when participants are present, the
  dispatcher `preventDefault`s pointerdown/wheel synchronously (the
  participant handler itself runs a microtask later, too late to stop
  page scroll). With NO participant registered the path is byte-identical
  to before — the whole feature is gated on registration.
- Surfaced through `RenderControl` `onReady` as `info.input`
  (`DomParticipantHost`).

**Camera — `FreeFlyController.attach(target, time, { input })`
(`src/scene/controllers/freefly.ts`).** Passing `input` (the onReady
host) makes the camera a DOM-world **bubble** participant instead of
installing native pointer/wheel listeners; it claims the pointer on press
(via the dispatcher's DOM capture) so orbit moves skip the pick, and its
handlers return void so a scene stop suppresses it. Keyboard, gamepad and
per-frame integration stay native in both modes (they never conflict with
picking).

**Tests.** `tests/event-router.test.tsx` (subtree capture/bubble order,
stop, root-exit boundary, foreign-listener survival, dynamic rows);
`tests/scene-leaf-router.test.ts` (ancestor DOM ↔ scene interleave both
directions, ancestor-capture stops before the pick, camera-stop in region
mode, overlay stays disjoint); `tests/pick-dom-participant.test.ts` (walk
order, scene-stops-camera via both `stopPropagation()` and `return false`,
DOM-capture suppressing the scene, capture-skips-pick, click-preserved-
through-orbit, wheel + sync preventDefault, no-participant parity);
`tests/controller-unified-input.test.ts` (camera orbits via the walk; a
scene stop suppresses it). Full mock suite + typecheck + build green.

**Remaining (app migration — task #21, TileRenderer-wombat repo, NOT
wombat.dom).** Wire the app camera through `info.input`; move markers to
`Sg.OnDragStart/OnDrag/OnDragEnd` (return false to stop the camera) +
`Sg.OnTap`; delete the Drag module's CPU screen-space projection and its
capture-phase DOM listener hack. End-to-end orbit-vs-scene-click and
marker-drag-stops-camera want a real-browser check (the mock suite proves
the routing, not the pixels). Gate: app e2e + deploy.
