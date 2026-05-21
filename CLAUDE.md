# wombat.dom

Adaptive DOM + 3D scene-graph DSL for the Wombat TypeScript stack.
JSX is the surface; underneath is a direct-DOM renderer driven by
`@aardworx/wombat.adaptive`. The 3D scene-graph layer (subpath
`/scene`) wraps `@aardworx/wombat.rendering` + `@aardworx/wombat.shader`.

This repo previously lived at `krauthaufen/adaptive-ui`. Renamed to
`krauthaufen/wombat.dom` for naming-consistency with the rest of the
Wombat stack. The npm package was previously published as
`@aardworx/wombat.adaptive-ui`; renamed to `@aardworx/wombat.dom`
within the npm 72h unpublish window.

## Repository layout

```
src/
├── core             scope, scheduler, vnode, guards
├── bindings/        attr, children, alist, text — adaptive binding strategies
├── jsx-runtime.ts   JSX factory (referenced by tsconfig jsxImportSource)
├── jsx-types.ts     JSX intrinsic-element types
├── mount.ts         top-level mount(root, vnode)
└── scene/           subpath @aardworx/wombat.dom/scene
                     (3D scene graph; uses wombat.rendering + wombat.shader)
```

Scene-layer status is listed in `README.md §Scene roadmap` and
`TODO.md`. Code lives in `src/scene/`. Cross-repo status:
`~/claude/wombat-todo.md`.

## Tooling

- `npm test` — vitest with `happy-dom`, smoke + binding tests.
- `npm run typecheck` — `tsc --noEmit`.
- `npm run build` — clean + emit to `dist/`.
- `npm run prepublishOnly` chains typecheck + test + build.

## Architecture

DOM side (today):

```
JSX  ──►  jsx-runtime  ──►  VNode  ──►  mount  ──►  bindings/{attr,children,alist,text}
                                            │
                                            └──►  Scope (disposal cascade)
                                            └──►  UIScheduler (rAF batched flushes)
```

- **JSX runtime** lowers `<el a={x}>{c}</el>` into a `VNode` (plain
  object, not a React VDOM). Mounting walks the VNode and creates
  real DOM elements. There is no virtual DOM or diffing pass — each
  binding subscribes directly to its adaptive source.
- **`Scope`** is the disposal hierarchy. Every mount creates a root
  scope; nested mounts (alist rows, aval child swaps) create child
  scopes. `dispose()` cascades.
- **`UIScheduler`** batches binding flushes onto rAF. A binding
  subscribes via `addMarkingCallback`; on mark, it calls
  `scheduler.notify(binding)`; the scheduler de-dupes and flushes
  in the next frame.
- **bindings/**:
  - `attr.ts` — sets attribute / DOM property, including `style`,
    `class`, event listeners. Routes by attribute kind.
  - `text.ts` — text-node binding. Plain string → text node;
    aval → text node updated in-place.
  - `alist.ts` — incremental children. Per-row scope; index keyed
    via `MapExt<Index, Row>`. Insertion at fresh Index is
    O(log n + 1 DOM op).
  - `children.ts` — dispatch table for whatever ended up in the
    JSX `children` slot.

Scene side (shipped — see `TODO.md` for remaining gaps):

```
SgNode tree  ──►  TraversalState walk  ──►  alist<Command>  ──►  wombat.rendering Runtime
```

- `<RenderControl>` JSX component owns a canvas, attaches via
  `wombat.rendering`'s `attachCanvas`, runs the rAF render loop.
- The Sg DSL is JSX-native: `<Sg Trafo Shader Uniform OnClick>`
  with scoped attributes flowing down to children, plus leaf
  components like `<Sg.Box/>`, `<Sg.Sphere/>`, `<Sg.Cylinder/>`,
  `<Sg.Cone/>`, `<Sg.Quad/>` (with DefaultSurfaces). See `README.md`
  for the surface.
- Auto-instancing fuses by Effect with per-swap re-validation of
  instanced subtrees (`docs/auto-instancing.md`).
- Picking: pick-framebuffer pixel pick plus reactive BVH ray queries,
  with pixel↔BVH fusion (`src/scene/picking/`). SceneEvent
  capture/bubble plus focus/key routing.
- FreeFly + Orbit controllers (`src/scene/controllers/`).
- Symmetric SDF text AA (`src/scene/text-sdf.ts`).

## Design decisions (locked-in)

These reflect the `<Sg>` surface in `src/scene/`. Re-litigating
them costs more than it's worth; carry the rationale forward.

- **JSX-native DSL.** No CE-style mixing of attributes and
  children. Attributes are JSX props; children are JSX children;
  scoped scene attributes go on `<Sg>`'s prop bag.
- **Children default to ordered.** `<Sg>` builds an
  `Ordered`/`OrderedFromList` `RenderTree`. Opt into unordered
  via `<Sg.Unordered>`. F# Aardvark.Dom defaults to `aset` for
  scene children — we deliberately diverge for JSX-feeling
  authoring; perf opt-in stays one wrapper away.
- **Heterogeneous children.** A single `<Sg>` can mix JSX leaves,
  alists, asets, and avals as separate child segments. Each
  segment gets the right `RenderTree` walker. No
  same-container-type rule.
- **Compose vs override is per-attribute, not per-syntax.** Two
  nested `<Sg Trafo>` scopes compose by matrix multiply; two
  nested `<Sg Shader>` scopes override (innermost wins). Within
  a single `<Sg>` each prop appears once — composition only
  shows up via nesting.
- **Shader / PickThrough / BlendMode are NOT adaptive.** Pipeline
  caching depends on these being stable per scope. To switch
  reactively, swap the whole subtree via
  `<Sg.Adaptive value={aval<SgNode>}>`.
- **Trafo array order = nested-Sg order, flattened.** Index 0 is
  outermost. `Trafo={[A, B, C]}` is equivalent to
  `<Sg Trafo={A}><Sg Trafo={B}><Sg Trafo={C}>`. A point is
  transformed by C, then B, then A. `Trafo3d` multiplication is
  `(t1 * t2).Forward = t1.Forward * t2.Forward`, applied to a
  column vector — t2 hits first, then t1. Same convention as
  Aardvark.Base.
- **Convenience trafos** (`Sg.translate`, `Sg.scale`, `Sg.rotate`)
  are constructors returning `Trafo3d` (or `aval<Trafo3d>`) for
  use inside the `Trafo` array. They don't carry composition
  semantics on their own — composition is the array's job.
- **Uniform is a map prop.** Object literal or `amap` are both
  accepted. Composition across nesting is per-key map merge with
  inner-wins on key conflict.
- **Event handlers append, not override.** Multiple nested `<Sg
  OnClick>` scopes all see capture/bubble dispatch; nearest
  handler doesn't shadow outer ones. (DOM convention.)

## What this package does NOT do

- **No SSR / hydration.** DOM is built imperatively at mount time;
  there's no server-render-then-resume story.
- **No virtual DOM.** Bindings subscribe directly. Don't introduce
  diff passes.
- **No React adapter** in this package. A separate
  `wombat.dom-react` could wrap a `<RenderControl>` for use inside
  a React tree later if needed.

## Don'ts

- Don't introduce a virtual-DOM or diffing layer. The whole point
  is direct subscriptions.
- Don't make the `Shader` / `PickThrough` / `BlendMode` props
  adaptive. Pipeline cache invariants depend on them being stable
  per scope.
- Don't bake a canonical Trafo order (T·R·S) into the Trafo
  composer. Source order = composition order; that's the contract.
- Don't `npm publish` from a dirty tree. CI runs
  `prepublishOnly` (typecheck + tests + build).
- Don't put scene-layer code outside `src/scene/`. The DOM core
  must remain importable without pulling in WebGPU.

## Open questions (track here)

- **Children default order on the DOM side.** Currently
  arrays/alists. Whether to add `aset` support to DOM children
  (with arbitrary ordering by hash) is open; F# DomNode uses
  `alist` only.
- **`<Sg.Adaptive>` unmount cleanup timing** (rAF vs sync). The
  rendering layer's resource refcount should make either fine,
  but pin the convention before the first scene swap test.
