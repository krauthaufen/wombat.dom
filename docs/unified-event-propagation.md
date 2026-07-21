# Unified DOM ↔ Scene Event Propagation

*Status: DESIGN (not yet implemented). This pins the semantics before
touching the shared event core.*

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
