// SceneEvent — what pick dispatch hands to user-supplied handlers.
//
// Carries CSS-pixel cursor coordinates (canvas-local), the decoded
// pickId from the rgba32f pick attachment, the mode-A/mode-B flag,
// and (when available) a view-space hit position. The original
// PointerEvent stays accessible via `raw` for handlers that need
// pressure / pointerType / button-mask details we haven't surfaced.
//
// Mirrors Aardvark.Dom's `ScenePointerEvent`: handlers can call
// `stopPropagation()` to halt the capture/bubble walk, and there's
// a `defaultPrevented` flag that handlers can set via
// `preventDefault()`. PointerCapture is exposed as instance methods
// that delegate to the dispatcher (see `dispatcher.ts`).

import { V3d } from "@aardworx/wombat.base";

import type { LeafPickScope } from "./registry.js";

export type SceneEventKind =
  | "OnClick"
  | "OnPointerDown"
  | "OnPointerUp"
  | "OnPointerMove"
  | "OnPointerEnter"
  | "OnPointerLeave"
  | "OnTap"
  | "OnDoubleTap"
  | "OnLongPress"
  // Phase 4
  | "OnWheel"
  // Phase 5
  | "OnFocus"
  | "OnBlur"
  | "OnKeyDown"
  | "OnKeyUp"
  | "OnKeyPress"
  // Phase 6
  | "OnDragStart"
  | "OnDrag"
  | "OnDragEnd";

/**
 * Dispatcher hook the SceneEvent uses to delegate pointer-capture
 * calls. The dispatcher implements this interface; we keep it as a
 * forward-only interface so sceneEvent.ts doesn't import the
 * dispatcher (would create a cycle).
 */
export interface SceneEventDispatch {
  setPointerCapture(scope: LeafPickScope, pointerId: number): void;
  releasePointerCapture(scope: LeafPickScope, pointerId: number): void;
  hasPointerCapture(scope: LeafPickScope, pointerId: number): boolean;
}

export interface SceneEventInit {
  readonly kind: SceneEventKind;
  readonly clientX: number;
  readonly clientY: number;
  readonly pickId: number;
  readonly modeB: boolean;
  readonly button?: number;
  readonly buttons?: number;
  readonly viewPos?: V3d;
  readonly viewNormal?: V3d;
  readonly pointerId: number;
  readonly pointerType: string;
  readonly raw: PointerEvent | WheelEvent | KeyboardEvent | FocusEvent;
  readonly scope?: LeafPickScope;
  readonly dispatch?: SceneEventDispatch;
  // Phase 4 — wheel
  readonly deltaX?: number;
  readonly deltaY?: number;
  readonly deltaZ?: number;
  readonly deltaMode?: 0 | 1 | 2;
  // Phase 5 — keyboard / focus
  readonly key?: string;
  readonly code?: string;
  readonly repeat?: boolean;
  readonly ctrlKey?: boolean;
  readonly shiftKey?: boolean;
  readonly altKey?: boolean;
  readonly metaKey?: boolean;
  readonly relatedTarget?: number;  // PickId of the previously / next focused scope.
  // Phase 6 — drag
  readonly dragStartX?: number;
  readonly dragStartY?: number;
}

export class SceneEvent {
  readonly kind: SceneEventKind;
  /** CSS px relative to the canvas's bounding rect (top-left origin). */
  readonly clientX: number;
  readonly clientY: number;
  /** 0 ⇒ no hit. */
  readonly pickId: number;
  /** True iff the pick fragment was a Mode-B (negative slot 0) write. */
  readonly modeB: boolean;
  /** Click / pointerdown / pointerup carry a button index. */
  readonly button?: number;
  /** Bitmask of buttons currently held (mirrors PointerEvent.buttons). */
  readonly buttons?: number;
  /**
   * Decoded view-space hit position when available. Mode-B writes pvp
   * directly into the pick attachment, so we always have one. Mode-A
   * requires unprojection from NDC depth using the hit scope's view /
   * proj — implemented in the dispatcher.
   */
  readonly viewPos?: V3d;
  /**
   * Optional view-space surface normal at the hit. Populated by the
   * BVH ray fall-through path (transformed from world-space using the
   * scope's view trafo); pixel-pick hits leave this undefined.
   */
  readonly viewNormal?: V3d;
  /** Forwarded from PointerEvent.pointerId. */
  readonly pointerId: number;
  /** Forwarded from PointerEvent.pointerType. */
  readonly pointerType: string;
  /** Escape hatch for handlers that need the unprocessed DOM event. */
  readonly raw: PointerEvent | WheelEvent | KeyboardEvent | FocusEvent;
  // Phase 4 — wheel deltas (only set on OnWheel events).
  readonly deltaX?: number;
  readonly deltaY?: number;
  readonly deltaZ?: number;
  readonly deltaMode?: 0 | 1 | 2;
  // Phase 5 — keyboard / focus state.
  readonly key?: string;
  readonly code?: string;
  readonly repeat?: boolean;
  readonly ctrlKey?: boolean;
  readonly shiftKey?: boolean;
  readonly altKey?: boolean;
  readonly metaKey?: boolean;
  /** PickId of the previously focused (Blur) / newly focused (Focus) scope. */
  readonly relatedTarget?: number;
  // Phase 6 — drag origin
  readonly dragStartX?: number;
  readonly dragStartY?: number;

  // Why mutable internals despite readonly fields above: handlers
  // mutate these via the public methods. The fields stay readonly to
  // catch accidental writes; the flags are stored as private slots.
  private _propagationStopped = false;
  private _defaultPrevented = false;

  private readonly _scope?: LeafPickScope;
  private readonly _dispatch?: SceneEventDispatch;

  constructor(init: SceneEventInit) {
    this.kind = init.kind;
    this.clientX = init.clientX;
    this.clientY = init.clientY;
    this.pickId = init.pickId;
    this.modeB = init.modeB;
    if (init.button !== undefined) this.button = init.button;
    if (init.buttons !== undefined) this.buttons = init.buttons;
    if (init.viewPos !== undefined) this.viewPos = init.viewPos;
    if (init.viewNormal !== undefined) this.viewNormal = init.viewNormal;
    this.pointerId = init.pointerId;
    this.pointerType = init.pointerType;
    this.raw = init.raw;
    if (init.deltaX !== undefined) this.deltaX = init.deltaX;
    if (init.deltaY !== undefined) this.deltaY = init.deltaY;
    if (init.deltaZ !== undefined) this.deltaZ = init.deltaZ;
    if (init.deltaMode !== undefined) this.deltaMode = init.deltaMode;
    if (init.key !== undefined) this.key = init.key;
    if (init.code !== undefined) this.code = init.code;
    if (init.repeat !== undefined) this.repeat = init.repeat;
    if (init.ctrlKey !== undefined) this.ctrlKey = init.ctrlKey;
    if (init.shiftKey !== undefined) this.shiftKey = init.shiftKey;
    if (init.altKey !== undefined) this.altKey = init.altKey;
    if (init.metaKey !== undefined) this.metaKey = init.metaKey;
    if (init.relatedTarget !== undefined) this.relatedTarget = init.relatedTarget;
    if (init.dragStartX !== undefined) this.dragStartX = init.dragStartX;
    if (init.dragStartY !== undefined) this.dragStartY = init.dragStartY;
    if (init.scope !== undefined) this._scope = init.scope;
    if (init.dispatch !== undefined) this._dispatch = init.dispatch;
  }

  /** True once a handler has called `stopPropagation()`. Capture/bubble loops poll this after every handler. */
  get propagationStopped(): boolean { return this._propagationStopped; }

  /**
   * Halts further capture/bubble dispatch on the SceneEvent and ALSO
   * stops the underlying DOM PointerEvent from propagating. The DOM-
   * side stop only matters when the SceneEvent handler runs
   * synchronously inside the originating DOM listener (which is
   * always the case for raw pointer events — we dispatch inline
   * inside the canvas listener — and also for Tap / DoubleTap /
   * LongPress, which we synthesise + dispatch inline before the
   * originating listener returns).
   */
  stopPropagation(): void {
    this._propagationStopped = true;
    this.raw.stopPropagation();
    this.raw.stopImmediatePropagation();
  }

  get defaultPrevented(): boolean { return this._defaultPrevented; }

  /**
   * Sets `defaultPrevented` AND calls `preventDefault()` on the
   * underlying PointerEvent. Same synchronous-listener caveat as
   * `stopPropagation`: the DOM-side preventDefault is best-effort
   * for synthesised events, but works for all of our current call
   * sites (we synthesise + dispatch inline at pointerup /
   * pointerdown / inside the long-press timer).
   */
  preventDefault(): void {
    this._defaultPrevented = true;
    this.raw.preventDefault();
  }

  // ---- PointerCapture --------------------------------------------------

  /**
   * Capture this pointer to the event's target scope. Subsequent
   * pointer events for `this.pointerId` will route to the captured
   * scope until released. Mirrors `SceneHandler.SetPointerCapture`.
   */
  setPointerCapture(): void {
    if (this._dispatch !== undefined && this._scope !== undefined) {
      this._dispatch.setPointerCapture(this._scope, this.pointerId);
    }
  }

  /** Release a prior capture on this pointer/scope. Triggers a synthetic move to re-establish hover. */
  releasePointerCapture(): void {
    if (this._dispatch !== undefined && this._scope !== undefined) {
      this._dispatch.releasePointerCapture(this._scope, this.pointerId);
    }
  }

  /** True iff this scope currently holds a capture for `this.pointerId`. */
  hasPointerCapture(): boolean {
    if (this._dispatch !== undefined && this._scope !== undefined) {
      return this._dispatch.hasPointerCapture(this._scope, this.pointerId);
    }
    return false;
  }
}
