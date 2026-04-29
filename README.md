# @aardworx/adaptive-ui

A direct-DOM JSX runtime over [`@aardworx/adaptive`](https://www.npmjs.com/package/@aardworx/adaptive).
Plain values, `aval<T>`, `alist<T>`, `amap<K,V>` are all accepted in
the same JSX positions; only the parts of the DOM that actually
depend on a changed adaptive input get touched, and updates are
batched onto the next animation frame.

## Install

```bash
npm install @aardworx/adaptive @aardworx/adaptive-ui
```

In your `tsconfig.json`:

```jsonc
{
  "compilerOptions": {
    "jsx": "react-jsx",
    "jsxImportSource": "@aardworx/adaptive-ui"
  }
}
```

In Vite:

```ts
// vite.config.ts
export default defineConfig({
  esbuild: { jsx: "automatic", jsxImportSource: "@aardworx/adaptive-ui" }
});
```

## Hello world

```tsx
import { cval, transact } from "@aardworx/adaptive";
import { mount } from "@aardworx/adaptive-ui";

const name = cval("world");

mount(document.getElementById("app")!, (
  <h1>hello, {name}</h1>
));

setTimeout(() => transact(() => name.value = "adaptive"), 1000);
```

## What you can pass where

| JSX position | Plain | `aval<T>` | `alist<T>` |
| --- | --- | --- | --- |
| Attribute / property | yes | yes — value updates in place | — |
| Event listener (`onClick`) | yes | yes (latest replaces previous) | — |
| Text child | yes | yes — text node updates in place | — |
| Element child | yes | yes — subtree swap with disposal | — |
| Children list | yes | yes (single child) | yes — incremental insert/remove keyed by Index |

Plain values are set once with no subscription overhead. The runtime
checks `instanceof` against the abstract base classes from
`@aardworx/adaptive`, so wrapping a static value in
`AVal.constant(...)` is never required.

## Incremental list rendering

```tsx
const items = clist<Item>([…]);

mount(root, (
  <ul>{items.map(it => <li>{it.name}</li>)}</ul>
));
```

Each item gets one DOM mount. Adding an item touches the DOM once
(insert one `<li>`); removing an item touches it once. An aval inside
a row updates only that row.

## Disposal

```ts
const handle = mount(root, <App/>);
// later:
handle.dispose();
```

`dispose()` cascades through every nested `<For>`/`<aval>` slot,
unsubscribes every binding, and removes everything this mount put
into the DOM.

## Status

Pre-1.0. Stable shape, expect breaking renames. No SSR, no React
adapter (planned: `@aardworx/adaptive-react` for hooks-based use
inside an existing React tree).

## Source

- [adaptive-ts](https://github.com/krauthaufen/adaptive-ts) — the underlying adaptive engine
- [adaptive-ts-demo](https://github.com/krauthaufen/adaptive-ts-demo) — live cart built on this library
