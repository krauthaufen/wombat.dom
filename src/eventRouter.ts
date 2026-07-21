// RegionRouter — the unified DOM ↔ scene event walk for a wombat mount
// subtree. See docs/unified-event-propagation.md.
//
// wombat OWNS event propagation inside the subtree it generated. One
// capture-phase listener per unified event type sits on the mount root;
// on first sight of an event it runs wombat's OWN walk over the DOM
// ancestor chain (root → target for capture, target → root for bubble),
// firing handlers that `attr.ts` registered here instead of via native
// `addEventListener`. Any handler can stop the walk (return `false` or
// call `stopPropagation()`), with one meaning across the whole subtree.
//
// To the HOST page (native DOM outside the mount root) the entire subtree
// is one opaque node: an unstopped event continues its native journey and
// bubbles out of the root, so outer handlers still see it; a stopped one
// is halted at the root, so it does not. This is the same "child is an
// opaque leaf" composition the scene graph gives wombat — one level up.
//
// A `<RenderControl>` canvas registers as a SCENE LEAF: when an event
// targets it, the DOM walk hands off to the (async) scene pick between
// the capture and bubble phases. That half is wired by RenderControl —
// this module only knows a leaf as an opaque sink. The rest of the
// subtree (HTML overlays, HUD) is a purely SYNCHRONOUS walk: no pick, no
// GPU, no deferral.

/**
 * Unified interactive events. Deliberately only BUBBLING pointer / mouse
 * / wheel events — the ones where DOM and scene handlers actually
 * collide. Everything else (`input`, `change`, `focus`, `blur`,
 * `pointerenter`, and — for now — `keydown`/`keyup`) stays native:
 * routing non-bubbling events through a root walk would invent a bubble
 * phase they don't have, and keyboard/focus unification (keys route to
 * the focused scope, not the pointer path) is a deliberate follow-up.
 */
export const UNIFIED_EVENTS = [
  "pointerdown", "pointerup", "pointermove", "pointercancel",
  "wheel", "click", "dblclick", "contextmenu",
] as const;

export type UnifiedEventName = (typeof UNIFIED_EVENTS)[number];

const UNIFIED_SET: ReadonlySet<string> = new Set(UNIFIED_EVENTS);

/** True for event names that route through the unified walk. */
export function isUnifiedEvent(name: string): name is UnifiedEventName {
  return UNIFIED_SET.has(name);
}

/**
 * A registered DOM handler. Returning `false` stops the walk (Aardvark
 * convention); calling `ev.stopPropagation()` does the same. `ev` is the
 * raw DOM event.
 */
export type RegionHandler = (ev: Event) => boolean | void;

export type WalkPhase = "capture" | "bubble";

/**
 * The async scene sub-walk for an event whose target is this leaf's
 * canvas. Runs between the DOM capture and DOM bubble phases; the router
 * awaits it, then resumes bubbling. `prop.stopped` reflects a scene
 * handler's `stopPropagation()` so the outer DOM bubble is suppressed.
 */
export interface SceneLeaf {
  dispatch(name: UnifiedEventName, ev: Event, prop: WalkProp): void | Promise<void>;
}

/** Shared stop flag threaded through one walk. */
export interface WalkProp { stopped: boolean; }

/** The registration surface a `Scope` carries as its `region`. */
export interface EventRegion {
  registerHandler(el: Element, phase: WalkPhase, name: UnifiedEventName, fn: RegionHandler): () => void;
  registerSceneLeaf(canvas: Element, leaf: SceneLeaf): () => void;
}

interface ElementReg {
  capture: Map<UnifiedEventName, Set<RegionHandler>>;
  bubble: Map<UnifiedEventName, Set<RegionHandler>>;
}

interface SeenFlag { __wombatSeen?: RegionRouter; }

export class RegionRouter implements EventRegion {
  private readonly handlers = new Map<Element, ElementReg>();
  private readonly sceneLeaves = new Map<Element, SceneLeaf>();
  private readonly listeners: Array<[UnifiedEventName, EventListener]> = [];
  private disposed = false;

  constructor(private readonly root: Element) {
    for (const name of UNIFIED_EVENTS) {
      const l: EventListener = (ev) => this.onRootEvent(name, ev);
      // Capture phase on the root: we see the event before it reaches any
      // element in the subtree, so wombat's walk fully replaces native
      // propagation for the events it owns.
      root.addEventListener(name, l, true);
      this.listeners.push([name, l]);
    }
  }

  registerHandler(el: Element, phase: WalkPhase, name: UnifiedEventName, fn: RegionHandler): () => void {
    let reg = this.handlers.get(el);
    if (reg === undefined) {
      reg = { capture: new Map(), bubble: new Map() };
      this.handlers.set(el, reg);
    }
    const byName = reg[phase];
    let set = byName.get(name);
    if (set === undefined) { set = new Set(); byName.set(name, set); }
    set.add(fn);
    return () => {
      const r = this.handlers.get(el);
      if (r === undefined) return;
      const s = r[phase].get(name);
      s?.delete(fn);
      if (s !== undefined && s.size === 0) r[phase].delete(name);
      if (r.capture.size === 0 && r.bubble.size === 0) this.handlers.delete(el);
    };
  }

  registerSceneLeaf(canvas: Element, leaf: SceneLeaf): () => void {
    this.sceneLeaves.set(canvas, leaf);
    return () => { if (this.sceneLeaves.get(canvas) === leaf) this.sceneLeaves.delete(canvas); };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const [name, l] of this.listeners) this.root.removeEventListener(name, l, true);
    this.listeners.length = 0;
    this.handlers.clear();
    this.sceneLeaves.clear();
  }

  // --- the walk -------------------------------------------------------------

  private onRootEvent(name: UnifiedEventName, ev: Event): void {
    if (this.disposed) return;
    // Dedup / re-entrancy: only the first (outermost) router that sees an
    // event owns it. A nested mount's inner router bails.
    const seen = ev as unknown as SeenFlag;
    if (seen.__wombatSeen !== undefined) return;
    seen.__wombatSeen = this;

    const path = this.pathTo(ev.target);
    if (path.length === 0) return;

    // Is there a scene leaf on the path? (The canvas is the target of a
    // scene interaction, so it is the last element on the path.)
    let leaf: SceneLeaf | undefined;
    for (let i = path.length - 1; i >= 0; i--) {
      const l = this.sceneLeaves.get(path[i]!);
      if (l !== undefined) { leaf = l; break; }
    }

    const prop: WalkProp = { stopped: false };

    // Capture: root → target.
    this.runPhase(path, "capture", name, ev, prop, /*reverse*/ false);

    if (leaf === undefined) {
      // Pure DOM (overlay / HUD): fully synchronous.
      if (!prop.stopped) this.runPhase(path, "bubble", name, ev, prop, /*reverse*/ true);
      this.finalize(ev, prop);
      return;
    }

    // Scene leaf on the path: the canvas region owns this pointer/wheel
    // event, so take it over synchronously (stop it from also reaching
    // the host page) — the pick + bubble resume a microtask later, too
    // late to stop native propagation retroactively.
    ev.stopPropagation();
    if (prop.stopped) return;
    const res = leaf.dispatch(name, ev, prop);
    if (res === undefined) {
      if (!prop.stopped) this.runPhase(path, "bubble", name, ev, prop, true);
      return;
    }
    void res.then(() => {
      if (this.disposed) return;
      if (!prop.stopped) this.runPhase(path, "bubble", name, ev, prop, true);
    });
  }

  /** Fire one phase's handlers. Returns nothing; sets `prop.stopped`. */
  private runPhase(
    path: ReadonlyArray<Element>,
    phase: WalkPhase,
    name: UnifiedEventName,
    ev: Event,
    prop: WalkProp,
    reverse: boolean,
  ): void {
    const n = path.length;
    for (let k = 0; k < n; k++) {
      const el = path[reverse ? n - 1 - k : k]!;
      const set = this.handlers.get(el)?.[phase].get(name);
      if (set === undefined) continue;
      // Snapshot: a handler may register/unregister during dispatch.
      for (const fn of [...set]) {
        let r: boolean | void;
        try { r = fn(ev); } catch (err) { console.error(`[wombat.dom] ${phase} ${name} handler threw:`, err); continue; }
        if (r === false || ev.cancelBubble) prop.stopped = true;
        if (prop.stopped) return;
      }
    }
  }

  /** Root → target element path (inclusive), or [] if target is outside. */
  private pathTo(target: EventTarget | null): Element[] {
    let n: Node | null =
      target instanceof Element ? target
      : target instanceof Node ? (target.parentElement as Node | null)
      : null;
    const path: Element[] = [];
    while (n instanceof Element) {
      path.push(n);
      if (n === this.root) { path.reverse(); return path; }
      n = n.parentNode;
    }
    return []; // target not under this root
  }

  /** After an unstopped walk the event continues natively and bubbles out
   *  of the root (host page sees it). A stopped walk halts it here. */
  private finalize(ev: Event, prop: WalkProp): void {
    if (prop.stopped) ev.stopPropagation();
  }
}
