// URL → ITexture resolution for the Sg compile path.
//
// `ITexture.fromUrl(url)` is an authoring shorthand that the rendering
// runtime can't bind directly (no synchronous fetch). Sg wraps any
// `aval<ITexture>` whose current value is `kind: "url"` with an aval
// that:
//
//   - hands the runtime a singleton 8×8 checker placeholder so the
//     pipeline can prepare immediately, and
//
//   - kicks off `fetch + createImageBitmap`, swapping in the loaded
//     bitmap (via a per-URL `cval<ITexture>`) once ready.
//
// Per-URL `cval`s are cached so multiple ROs sharing one URL resolve
// to the same `aval<ITexture>` — atlas dedup + ref-count work as
// usual, and one swap-in updates every consumer in a single transact.
//
// If the source `aval<ITexture>` itself ticks (e.g. user swaps the
// `ITexture.fromUrl(...)` for a different URL or a real `kind:"gpu"`
// texture), the wrapping aval re-evaluates: a non-url value flows
// through untouched, a different url installs a different cached
// `cval`.

import { AVal, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { ITexture } from "@aardworx/wombat.rendering/core";

// 8×8 checker, 4 bytes per pixel — magenta / dark grey to be obviously
// a placeholder (not silently matching a real surface). Built lazily.
let _checker: ITexture | undefined;
function checkerTexture(): ITexture {
  if (_checker !== undefined) return _checker;
  const W = 8;
  const data = new Uint8Array(W * W * 4);
  for (let y = 0; y < W; y++) {
    for (let x = 0; x < W; x++) {
      const on = ((x ^ y) & 1) === 0;
      const i = (y * W + x) * 4;
      if (on) { data[i] = 0xff; data[i+1] = 0x00; data[i+2] = 0xff; }
      else    { data[i] = 0x22; data[i+1] = 0x22; data[i+2] = 0x22; }
      data[i+3] = 0xff;
    }
  }
  _checker = ITexture.fromRaw({ data, width: W, height: W, format: "rgba8unorm" });
  return _checker;
}

// `cval<ITexture>` per URL. Starts at the checker; swaps to the real
// bitmap once `fetch + createImageBitmap` resolves. Failures keep the
// checker (and log once).
const urlCache = new Map<string, aval<ITexture>>();
const loadFailed = new Set<string>();

function resolveUrl(spec: Extract<ITexture, { kind: "url" }>): aval<ITexture> {
  const cached = urlCache.get(spec.url);
  if (cached !== undefined) return cached;
  const cell = cval<ITexture>(checkerTexture());
  urlCache.set(spec.url, cell);

  // Browser-only async load. In SSR or test harnesses without `fetch`
  // we just keep the checker.
  if (typeof fetch !== "function" || typeof createImageBitmap !== "function") {
    return cell;
  }

  void (async () => {
    try {
      const resp = await fetch(spec.url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const blob = await resp.blob();
      const bitmap = await createImageBitmap(blob, {
        // ImageBitmap is uploaded via copyExternalImageToTexture; the
        // runtime defaults to rgba8unorm. Generate mips when the
        // caller asked for them (lets atlas honour `wantsMips`).
        ...(spec.generateMips === true ? { /* hint only — actual mips are runtime-driven */ } : {}),
      });
      const tex = ITexture.fromExternal(bitmap, {
        ...(spec.format !== undefined ? { format: spec.format } : {}),
        ...(spec.generateMips !== undefined ? { generateMips: spec.generateMips } : {}),
      });
      transact(() => { cell.value = tex; });
    } catch (err) {
      if (!loadFailed.has(spec.url)) {
        loadFailed.add(spec.url);
        console.warn(`[wombat.dom] failed to load texture URL "${spec.url}":`, err);
      }
    }
  })();

  return cell;
}

/**
 * Wrap a user-supplied `aval<ITexture>` so URL-deferred textures
 * resolve through a per-URL placeholder. Non-url values pass through
 * untouched.
 *
 * Crucially, the result must be *shared* across all leaves referencing
 * the same texture, or the atlas pool — which keys on the resolved
 * aval's identity/value — allocates one sub-rect per leaf. With 1000s
 * of textured leaves cycling through ~12 URLs that exhausts the atlas
 * (`ATLAS_MAX_PAGES_PER_FORMAT`). So:
 *
 *   - constant source, url value  → return `resolveUrl(t)` directly:
 *     that's the *per-URL* shared `cval` (placeholder → loaded), keyed
 *     by url string. N leaves → 1 atlas entry per url.
 *   - constant source, non-url    → return `src` unchanged: the atlas
 *     pool's content-keyed `entriesByAval` HashTable already collapses
 *     two `AVal.constant(sameTexture)` to one entry.
 *   - reactive source             → wrap in `AVal.custom` (the value
 *     can flip url ↔ non-url), cached per source-aval identity.
 */
const resolveCache = new WeakMap<aval<ITexture>, aval<ITexture>>();
export function resolveTextureAval(src: aval<ITexture>): aval<ITexture> {
  if (src.isConstant) {
    // AVal.force OK: isConstant guard — constant avals are force-safe
    // and never tick. Same category as splitTexturesFromUniforms.
    const t = src.force();
    if (isITexture(t)) {
      return t.kind === "url" ? resolveUrl(t) : src;
    }
    return src;
  }
  const cached = resolveCache.get(src);
  if (cached !== undefined) return cached;
  const wrapped = AVal.custom<ITexture>(token => {
    const t = src.getValue(token);
    if (t.kind !== "url") return t;
    return resolveUrl(t).getValue(token);
  });
  resolveCache.set(src, wrapped);
  return wrapped;
}

/**
 * `true` iff a value is one of `ITexture`'s shape variants. Used by
 * the Sg compile layer to split texture-typed uniforms out of the
 * uniform map so the runtime sees them on `RenderObject.textures`.
 */
export function isITexture(v: unknown): v is ITexture {
  if (v === null || typeof v !== "object") return false;
  const k = (v as { kind?: unknown }).kind;
  return k === "gpu" || k === "host" || k === "url";
}
