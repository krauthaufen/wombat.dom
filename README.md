# @aardworx/wombat.dom

Adaptive DOM + 3D scene-graph DSL for the
[Wombat](https://github.com/krauthaufen/wombat.adaptive) TypeScript
stack. JSX is the surface; the runtime is a direct-DOM renderer over
[`@aardworx/wombat.adaptive`](https://www.npmjs.com/package/@aardworx/wombat.adaptive),
so plain values, `aval<T>`, `alist<T>`, `amap<K, V>` and `aset<T>`
are all accepted in the same JSX positions. Only the parts of the
DOM that actually depend on a changed adaptive input get touched,
and updates are batched onto the next animation frame.

The 3D scene graph is delivered as a separate subpath
(`@aardworx/wombat.dom/scene`) — you only pay for WebGPU when you
import it.

Part of the Wombat TypeScript port of the Aardvark stack:

1. [`@aardworx/wombat.adaptive`](https://github.com/krauthaufen/wombat.adaptive) — incremental adaptive computations.
2. [`@aardworx/wombat.base`](https://github.com/krauthaufen/wombat.base) — math/geometry primitives.
3. [`@aardworx/wombat.shader`](https://github.com/krauthaufen/wombat.shader) — TS-as-shader DSL.
4. [`@aardworx/wombat.rendering`](https://github.com/krauthaufen/wombat.rendering) — WebGPU rendering layer.
5. **`@aardworx/wombat.dom`** — this repo: adaptive DOM + 3D scene-graph DSL.

## Install

```bash
npm install @aardworx/wombat.adaptive @aardworx/wombat.dom
```

For the 3D scene layer:

```bash
npm install @aardworx/wombat.base @aardworx/wombat.shader @aardworx/wombat.rendering
```

In your `tsconfig.json`:

```jsonc
{
  "compilerOptions": {
    "jsx": "react-jsx",
    "jsxImportSource": "@aardworx/wombat.dom"
  }
}
```

In Vite:

```ts
// vite.config.ts
export default defineConfig({
  esbuild: { jsx: "automatic", jsxImportSource: "@aardworx/wombat.dom" }
});
```

## Hello world

```tsx
import { cval, transact } from "@aardworx/wombat.adaptive";
import { mount } from "@aardworx/wombat.dom";

const name = cval("world");

mount(document.getElementById("app")!, (
  <h1>hello, {name}</h1>
));

setTimeout(() => transact(() => name.value = "adaptive"), 1000);
```

## What you can pass where

| JSX position | Plain | `aval<T>` | `alist<T>` | `aset<T>` |
| --- | --- | --- | --- | --- |
| Attribute / property | yes | yes — value updates in place | — | — |
| Event listener (`onClick`) | yes | yes (latest replaces previous) | — | — |
| Text child | yes | yes — text node updates in place | — | — |
| Element child | yes | yes — subtree swap with disposal | — | — |
| Children list | yes | yes (single child) | yes — incremental insert/remove | yes (in scene-graph; planned for DOM) |

Plain values are set once with no subscription overhead. The runtime
checks `instanceof` against the abstract base classes from
`@aardworx/wombat.adaptive`, so wrapping a static value in
`AVal.constant(...)` is never required.

## Incremental list rendering

```tsx
const items = clist<Item>([…]);

mount(root, (
  <ul>{items.map(it => <li>{it.name}</li>)}</ul>
));
```

Each item gets one DOM mount. Adding an item touches the DOM once
(insert one `<li>`); removing an item touches it once. An aval
inside a row updates only that row.

## Disposal

```ts
const handle = mount(root, <App/>);
// later:
handle.dispose();
```

`dispose()` cascades through every nested binding, unsubscribes all
adaptive readers, and removes everything this mount put into the
DOM.

## Scene roadmap

The `@aardworx/wombat.dom/scene` subpath ships in milestones. Tracked
in [`TODO.md`](TODO.md) and [`CLAUDE.md`](CLAUDE.md).

| M | Goal |
| --- | --- |
| M1 | rename + scaffold (this release: 0.2.0) |
| M2 | `Sg` core: tagged-union scene nodes, traversal state, attribute composition rules |
| M3 | scene → `RenderObject` lowering (uses `wombat.rendering`) |
| M4 | `<RenderControl>` JSX component |
| M5 | camera + view/proj uniforms |
| M6 | free-fly + orbit controllers |
| M7 | pick framebuffer + pick effect (Mode A) |
| M8 | pick read + `SceneEvent` capture/bubble |
| M9 | default surfaces + primitives (`<Sg.Box/>`, …) |
| M10 | BVH picking + fusion (optional) |

After M8 the package can render a hello-cube demo with click-to-log
on a real GPU.

## Status

Pre-1.0. Stable shape on the DOM side, expect breaking renames as
the scene layer fills in. No SSR.

## License

MIT.
