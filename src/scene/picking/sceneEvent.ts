// SceneEvent — what pick dispatch hands to user-supplied handlers.
//
// Carries CSS-pixel cursor coordinates (canvas-local), the decoded
// pickId from the rgba32f pick attachment, the mode-A/mode-B flag,
// and (when available) a view-space hit position. The original
// PointerEvent stays accessible via `raw` for handlers that need
// pressure / pointerType / button-mask details we haven't surfaced.

export type SceneEventKind =
  | "OnClick"
  | "OnPointerDown"
  | "OnPointerUp"
  | "OnPointerMove"
  | "OnPointerEnter"
  | "OnPointerLeave";

export interface SceneEventViewPos {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

export interface SceneEvent {
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
  readonly viewPos?: SceneEventViewPos;
  /** Escape hatch for handlers that need the unprocessed PointerEvent. */
  readonly raw: PointerEvent;
}
