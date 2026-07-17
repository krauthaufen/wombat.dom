// Structural host-buffer proof for adaptive `aval<IBuffer>`s.
//
// Rule (locked-in): the only two ways to know something about an
// aval's future values are its runtime `isConstant` property or a
// construction-level proof. A one-shot force of a CHANGEABLE aval is
// never evidence — the next value may differ.
//
// `markHostBufferAVal` records exactly such a construction-level
// proof: the call site's mapping function constructs
// `IBuffer.fromHost(...)` for EVERY input, so every value the aval
// can ever produce is host-kind. The VALUES stay fully reactive
// (lengths, contents, identities change freely) — only the `kind` is
// pinned by the code that builds them. Row lowering uses the marker
// to assert heap eligibility without touching the live aval.

/** Brand an aval whose construction provably yields `kind: "host"`
 *  buffers for all time. Returns the same aval (marked). Call ONLY at
 *  a site where the mapping function itself constructs the host
 *  buffer — never because a sampled value happened to be host. */
export function markHostBufferAVal<T>(av: T): T {
  (av as { __sgHostBuffer?: true }).__sgHostBuffer = true;
  return av;
}

/** True when `av` carries the construction-level host proof. */
export function isMarkedHostBufferAVal(av: unknown): boolean {
  return av !== null && typeof av === "object" &&
    (av as { __sgHostBuffer?: true }).__sgHostBuffer === true;
}
