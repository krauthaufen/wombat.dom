// SceneEvent — what pick dispatch hands to user-supplied handlers.
//
// Carries a rich `SceneEventLocation` (canvas pixel, viewport,
// view/proj/model trafos, view-space hit + normal, and lazy
// derivations: world / model / local positions, normals, and
// pick rays). Pass-through getters surface the most common fields
// directly on the event so handlers don't have to reach through
// `e.location.*`.
//
// The original DOM event stays accessible via `raw` for handlers that
// need pressure / pointerType / button-mask details we haven't
// surfaced. Handlers can call `stopPropagation()` to halt the
// capture/bubble walk; `preventDefault()` flips a flag and forwards
// to the underlying DOM event. PointerCapture is exposed as instance
// methods that delegate to the dispatcher.

import { Box2d, Ray3d, Trafo3d, V2d, V2i, V3d } from "@aardworx/wombat.base";
import type { aval } from "@aardworx/wombat.adaptive";

import type { LeafPickScope, PickId } from "./registry.js";
import { SceneEventLocation } from "./sceneEventLocation.js";

export type SceneEventKind =
  | "OnClick"
  | "OnDoubleClick"
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
  | "OnKeyInput"
  // Phase 6
  | "OnDragStart"
  | "OnDrag"
  | "OnDragEnd"
  // Multi-touch gestures
  | "OnPinch"
  | "OnTwoFingerPan"
  | "OnTwoFingerRotate"
  // Hover after dwell
  | "OnHover";

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

/**
 * Aardvark `IEventHandler` counterpart: the live event-handler context —
 * the canvas size and the adaptive cursor the dispatcher maintains.
 */
export interface SceneEventContext {
  readonly size: V2i;
  readonly cursor: aval<string | undefined>;
}

export interface SceneEventInit {
  readonly kind: SceneEventKind;
  readonly location: SceneEventLocation;
  readonly raw: PointerEvent | MouseEvent | WheelEvent | KeyboardEvent | FocusEvent | InputEvent;
  readonly pickId: PickId;
  /** True iff the pick fragment was a Mode-B (negative slot 0) write. */
  readonly modeB?: boolean;
  // Pointer-derived events
  readonly pointerId?: number;
  readonly pointerType?: string;
  readonly button?: number;
  readonly buttons?: number;
  // Modifier keys (also surfaced on wheel / key events).
  readonly ctrl?: boolean;
  readonly shift?: boolean;
  readonly alt?: boolean;
  readonly meta?: boolean;
  // Wheel
  readonly deltaX?: number;
  readonly deltaY?: number;
  readonly deltaZ?: number;
  readonly deltaMode?: 0 | 1 | 2;
  // Keyboard
  readonly key?: string;
  readonly code?: string;
  readonly repeat?: boolean;
  // Focus
  readonly relatedTarget?: PickId;
  // Tap / double-tap (Aardvark parity: down->up duration and net movement)
  readonly deltaTime?: number;
  readonly movementX?: number;
  readonly movementY?: number;
  // Text input (OnKeyInput; from the DOM InputEvent on the focused canvas)
  readonly data?: string;
  readonly inputType?: string;
  // Canvas bounding rect at dispatch time (Aardvark SceneEvent.ClientRect)
  readonly clientRect?: Box2d;
  // Event-handler context (Aardvark SceneEvent.Context)
  readonly context?: SceneEventContext;
  /** Handler-path entry currently being invoked (Aardvark SceneEvent.This);
   * set per level by the capture/bubble walk. */
  readonly self?: unknown;
  // Drag
  readonly dragStartX?: number;
  readonly dragStartY?: number;
  // Multi-touch gestures
  readonly pinchScale?: number;
  readonly pinchCenter?: V2d;
  readonly panDeltaX?: number;
  readonly panDeltaY?: number;
  readonly rotateRadians?: number;

  // Dispatcher hooks (for setPointerCapture).
  readonly scope?: LeafPickScope;
  readonly dispatch?: SceneEventDispatch;
}

/**
 * The event passed to scene-graph handlers. All spatial info lives on
 * `location`; the most common fields have pass-through getters on the
 * event itself (`viewPos`, `worldPos`, `pickRay`, etc.).
 */
export class SceneEvent {
  readonly kind: SceneEventKind;
  readonly location: SceneEventLocation;
  readonly raw: PointerEvent | MouseEvent | WheelEvent | KeyboardEvent | FocusEvent | InputEvent;
  readonly pickId: PickId;
  /** True iff the pick fragment was a Mode-B (negative slot 0) write. Defaults to `false`. */
  readonly modeB: boolean;

  // Pointer-derived
  readonly pointerId?: number;
  readonly pointerType?: string;
  readonly button?: number;
  readonly buttons?: number;

  // Modifiers
  readonly ctrl?: boolean;
  readonly shift?: boolean;
  readonly alt?: boolean;
  readonly meta?: boolean;

  // Wheel
  readonly deltaX?: number;
  readonly deltaY?: number;
  readonly deltaZ?: number;
  readonly deltaMode?: 0 | 1 | 2;

  // Keyboard
  readonly key?: string;
  readonly code?: string;
  readonly repeat?: boolean;

  // Focus
  readonly relatedTarget?: PickId;

  // Tap / double-tap. `deltaTime` is the press duration (tap) or the
  // tap-to-tap gap (double-tap); `movementX/Y` the matching net delta.
  readonly deltaTime?: number;
  readonly movementX?: number;
  readonly movementY?: number;

  // Text input (OnKeyInput)
  readonly data?: string;
  readonly inputType?: string;

  // Canvas bounding rect at dispatch time (Aardvark SceneEvent.ClientRect).
  readonly clientRect?: Box2d;

  // Event-handler context (Aardvark SceneEvent.Context).
  readonly context?: SceneEventContext;

  /** Handler-path entry currently being invoked (Aardvark SceneEvent.This). */
  readonly self?: unknown;

  /** The dispatch-target scope (Aardvark SceneEvent.Target). */
  get target(): unknown { return this._scope; }

  // Drag
  readonly dragStartX?: number;
  readonly dragStartY?: number;

  // Gesture
  readonly pinchScale?: number;
  readonly pinchCenter?: V2d;
  readonly panDeltaX?: number;
  readonly panDeltaY?: number;
  readonly rotateRadians?: number;

  // Mutable propagation state. Shared across `transformed()` copies
  // via a holder object so a handler at any depth that calls
  // `stopPropagation()` halts dispatch on the originating event the
  // capture/bubble loop is polling.
  private readonly _prop: { stopped: boolean; prevented: boolean };

  private readonly _scope?: LeafPickScope;
  private readonly _dispatch?: SceneEventDispatch;
  private readonly _init: SceneEventInit;

  constructor(init: SceneEventInit, sharedProp?: { stopped: boolean; prevented: boolean }) {
    this.kind = init.kind;
    this.location = init.location;
    this.raw = init.raw;
    this.pickId = init.pickId;
    this.modeB = init.modeB ?? false;
    if (init.pointerId !== undefined) this.pointerId = init.pointerId;
    if (init.pointerType !== undefined) this.pointerType = init.pointerType;
    if (init.button !== undefined) this.button = init.button;
    if (init.buttons !== undefined) this.buttons = init.buttons;
    if (init.ctrl !== undefined) this.ctrl = init.ctrl;
    if (init.shift !== undefined) this.shift = init.shift;
    if (init.alt !== undefined) this.alt = init.alt;
    if (init.meta !== undefined) this.meta = init.meta;
    if (init.deltaX !== undefined) this.deltaX = init.deltaX;
    if (init.deltaY !== undefined) this.deltaY = init.deltaY;
    if (init.deltaZ !== undefined) this.deltaZ = init.deltaZ;
    if (init.deltaMode !== undefined) this.deltaMode = init.deltaMode;
    if (init.key !== undefined) this.key = init.key;
    if (init.code !== undefined) this.code = init.code;
    if (init.repeat !== undefined) this.repeat = init.repeat;
    if (init.relatedTarget !== undefined) this.relatedTarget = init.relatedTarget;
    if (init.deltaTime !== undefined) this.deltaTime = init.deltaTime;
    if (init.movementX !== undefined) this.movementX = init.movementX;
    if (init.movementY !== undefined) this.movementY = init.movementY;
    if (init.data !== undefined) this.data = init.data;
    if (init.inputType !== undefined) this.inputType = init.inputType;
    if (init.clientRect !== undefined) this.clientRect = init.clientRect;
    if (init.context !== undefined) this.context = init.context;
    if (init.self !== undefined) this.self = init.self;
    if (init.dragStartX !== undefined) this.dragStartX = init.dragStartX;
    if (init.dragStartY !== undefined) this.dragStartY = init.dragStartY;
    if (init.pinchScale !== undefined) this.pinchScale = init.pinchScale;
    if (init.pinchCenter !== undefined) this.pinchCenter = init.pinchCenter;
    if (init.panDeltaX !== undefined) this.panDeltaX = init.panDeltaX;
    if (init.panDeltaY !== undefined) this.panDeltaY = init.panDeltaY;
    if (init.rotateRadians !== undefined) this.rotateRadians = init.rotateRadians;
    if (init.scope !== undefined) this._scope = init.scope;
    if (init.dispatch !== undefined) this._dispatch = init.dispatch;
    this._init = init;
    this._prop = sharedProp ?? { stopped: false, prevented: false };
  }

  // ---- Pass-through location getters --------------------------------------

  /** Device-pixel cursor position (canvas-local). Same as `location.pixel`. */
  get pixel(): V2d { return this.location.pixel; }
  get viewportSize(): V2i { return this.location.viewportSize; }
  get viewPos(): V3d { return this.location.viewPos; }
  /** Hit position in world space. */
  get worldPos(): V3d { return this.location.worldPosition; }
  /** Hit position in the leaf's accumulated model frame. */
  get modelPos(): V3d { return this.location.modelPosition; }
  /** Hit position in the handler's local frame. */
  get position(): V3d { return this.location.position; }
  get viewNormal(): V3d { return this.location.viewNormal; }
  get worldNormal(): V3d { return this.location.worldNormal; }
  get modelNormal(): V3d { return this.location.modelNormal; }
  get normal(): V3d { return this.location.normal; }
  get depth(): number { return this.location.depth; }
  get partIndex(): number { return this.location.partIndex; }
  get pickRay(): Ray3d { return this.location.pickRay; }
  get viewPickRay(): Ray3d { return this.location.viewPickRay; }
  get worldPickRay(): Ray3d { return this.location.worldPickRay; }
  get modelPickRay(): Ray3d { return this.location.modelPickRay; }

  // ---- Propagation --------------------------------------------------------

  /** True once a handler has called `stopPropagation()`. Capture/bubble loops poll this after every handler. */
  get propagationStopped(): boolean { return this._prop.stopped; }

  /**
   * Halts further capture/bubble dispatch on the SceneEvent and ALSO
   * stops the underlying DOM event from propagating. The DOM-side
   * stop only matters when the SceneEvent handler runs synchronously
   * inside the originating DOM listener.
   */
  stopPropagation(): void {
    this._prop.stopped = true;
    this.raw.stopPropagation();
    this.raw.stopImmediatePropagation();
  }

  get defaultPrevented(): boolean { return this._prop.prevented; }

  /**
   * Sets `defaultPrevented` AND calls `preventDefault()` on the
   * underlying DOM event.
   */
  preventDefault(): void {
    this._prop.prevented = true;
    this.raw.preventDefault();
  }

  // ---- PointerCapture -----------------------------------------------------

  /** Capture this pointer to the event's target scope. Mirrors `SceneHandler.SetPointerCapture`. */
  setPointerCapture(): void {
    if (this._dispatch !== undefined && this._scope !== undefined && this.pointerId !== undefined) {
      this._dispatch.setPointerCapture(this._scope, this.pointerId);
    }
  }

  /** Release a prior capture. Triggers a synthetic move to re-establish hover. */
  releasePointerCapture(): void {
    if (this._dispatch !== undefined && this._scope !== undefined && this.pointerId !== undefined) {
      this._dispatch.releasePointerCapture(this._scope, this.pointerId);
    }
  }

  /** True iff this scope currently holds a capture for `this.pointerId`. */
  hasPointerCapture(): boolean {
    if (this._dispatch !== undefined && this._scope !== undefined && this.pointerId !== undefined) {
      return this._dispatch.hasPointerCapture(this._scope, this.pointerId);
    }
    return false;
  }

  // ---- transform -----------------------------------------------------------

  /**
   * Returns a copy with the event's `location` transformed (i.e.
   * `location.transformed(trafo)`). Used during capture/bubble to push
   * the event into a child scope's local frame.
   */
  transformed(trafo: Trafo3d): SceneEvent {
    return new SceneEvent(
      { ...this._init, location: this.location.transformed(trafo) },
      this._prop,
    );
  }

  /**
   * `transformed` + sets `self` to the handler-path entry about to be
   * invoked — used by the capture/bubble walk so handlers see Aardvark's
   * `This` (their own scope level) alongside `Target` (the hit scope).
   */
  transformedAt(trafo: Trafo3d, self: unknown): SceneEvent {
    return new SceneEvent(
      { ...this._init, location: this.location.transformed(trafo), self },
      this._prop,
    );
  }
}
