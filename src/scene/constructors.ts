// `Sg` — unified scene-graph surface. Three things on one
// namespace:
//
//   1. JSX component:
//        <Sg Trafo={t} Shader={s}>...</Sg>
//      Wraps children with the supplied scope attributes. Returns
//      a tagged-Fragment VNode that <RenderControl> recognises.
//
//   2. JSX sub-components:
//        <Sg.Unordered>     <Sg.Adaptive value={...}>
//        <Sg.Box/>          <Sg.Quad/>
//
//   3. Imperative builders + math helpers, for code-generated
//      trees and `Trafo` array elements:
//        Sg.empty / Sg.group / Sg.shader / Sg.uniform / Sg.trafo
//        Sg.view / Sg.proj / Sg.camera / Sg.delay / Sg.on / ...
//        Sg.translate / Sg.scale / Sg.rotate / Sg.identity
//
// Locking the JSX surface to one identifier (`Sg`) keeps the user-
// facing DSL terse (`<Sg Trafo={[Sg.translate(v), Sg.scale(2)]}>`)
// while still exposing the full imperative API for advanced uses.

import {
  AList, ASet, AVal,
  HashMap, type alist, type amap, type aset, type aval,
} from "@aardworx/wombat.adaptive";
import { Trafo3d, V3d, Rot3d, Scale3d, Shift3d, type IIntersectable } from "@aardworx/wombat.base";
import type { Effect } from "@aardworx/wombat.shader";
import type {
  BlendState, BufferView, DrawCall,
} from "@aardworx/wombat.rendering/core";

import type { Child, VNode } from "../vnode.js";
import { isVNode } from "../vnode.js";
import { isAList, isASet, isAVal } from "../guards.js";
import type {
  BlendConstantValue,
  ColorMaskValue,
  CullValue,
  DepthBiasValue,
  DepthCompare,
  EventHandlers,
  FillModeValue,
  FrontFaceValue,
  ModeValue,
  SceneEventHandler,
  SgNode,
  SgLeaf,
  StencilModeValue,
  TrafoValue,
  UniformBag,
} from "./sg.js";
import { RenderPass } from "./sg.js";
import type { SceneEventKind } from "./picking/sceneEvent.js";
import { box as boxLeaf, quad as quadLeaf, type BoxOptions, type QuadOptions } from "./primitives.js";
import {
  sgVNode, isSgVNode, extractSgNode, SG_KINDS,
} from "./sgVNode.js";

// ---------------------------------------------------------------------------
// Imperative builders
// ---------------------------------------------------------------------------

const empty: SgNode = { kind: "Empty" };

function leaf(spec: {
  vertexAttributes: HashMap<string, aval<BufferView>>;
  instanceAttributes?: HashMap<string, aval<BufferView>>;
  indices?: aval<BufferView> | aval<BufferView | undefined>;
  drawCall: aval<DrawCall>;
}): SgLeaf {
  return {
    kind: "Leaf",
    vertexAttributes: spec.vertexAttributes,
    ...(spec.instanceAttributes !== undefined ? { instanceAttributes: spec.instanceAttributes } : {}),
    ...(spec.indices !== undefined ? { indices: spec.indices as aval<BufferView | undefined> } : {}),
    drawCall: spec.drawCall,
  };
}

function group(children: alist<SgNode> | ReadonlyArray<SgNode> | SgNode): SgNode {
  if (Array.isArray(children)) {
    return { kind: "Group", children: AList.ofList(children as SgNode[]) };
  }
  if ((children as alist<SgNode>).getReader !== undefined) {
    return { kind: "Group", children: children as alist<SgNode> };
  }
  return { kind: "Group", children: AList.ofList([children as SgNode]) };
}

function unordered(children: aset<SgNode> | ReadonlyArray<SgNode>): SgNode {
  if (Array.isArray(children)) {
    return { kind: "UnorderedGroup", children: ASet.ofList(children as SgNode[]) };
  }
  return { kind: "UnorderedGroup", children: children as aset<SgNode> };
}

function adaptive(child: aval<SgNode>): SgNode {
  return { kind: "AdaptiveGroup", child };
}

function trafo(value: TrafoValue, child: SgNode): SgNode {
  return { kind: "Trafo", value, child };
}

function shader(effect: Effect, child: SgNode): SgNode {
  return { kind: "Shader", effect, child };
}

function uniformBag(entries: Record<string, unknown | aval<unknown>>): UniformBag {
  let map = HashMap.empty<string, aval<unknown>>();
  for (const [k, v] of Object.entries(entries)) {
    map = map.add(k, isAValRuntime(v) ? v as aval<unknown> : AVal.constant(v));
  }
  return { kind: "Static", entries: map };
}

function uniform(entries: Record<string, unknown | aval<unknown>>, child: SgNode): SgNode;
function uniform(entries: amap<string, aval<unknown>>, child: SgNode): SgNode;
function uniform(entries: Record<string, unknown> | amap<string, aval<unknown>>, child: SgNode): SgNode {
  if ((entries as amap<string, aval<unknown>>).content !== undefined) {
    return {
      kind: "Uniform",
      bag: { kind: "Dynamic", entries: entries as amap<string, aval<unknown>> },
      child,
    };
  }
  return { kind: "Uniform", bag: uniformBag(entries as Record<string, unknown>), child };
}

function blendMode(mode: BlendState, child: SgNode): SgNode {
  return { kind: "BlendMode", mode, child };
}

function cursor(value: string | aval<string>, child: SgNode): SgNode {
  return { kind: "Cursor", cursor: value, child };
}

function pickThrough(value: boolean, child: SgNode): SgNode {
  return { kind: "PickThrough", value, child };
}

function intersectable(value: IIntersectable | aval<IIntersectable>): (child: SgNode) => SgNode {
  const i: aval<IIntersectable> = isAValRuntime(value)
    ? value as aval<IIntersectable>
    : AVal.constant(value as IIntersectable);
  return (child: SgNode): SgNode => ({ kind: "Intersectable", intersectable: i, child });
}

function pixelSnapRadius(radius: number | aval<number>): (child: SgNode) => SgNode {
  const r: aval<number> = isAValRuntime(radius)
    ? radius as aval<number>
    : AVal.constant(radius as number);
  return (child: SgNode): SgNode => ({ kind: "PixelSnapRadius", radius: r, child });
}

function on(handlers: EventHandlers, child: SgNode): SgNode {
  return { kind: "On", handlers, child };
}

/** Curried convenience: bubble-only single-event SgOn. */
function onEvent(kind: SceneEventKind, fn: SceneEventHandler): (child: SgNode) => SgNode {
  return (child) => on({ bubble: { [kind]: fn } }, child);
}

const onClick      = (fn: SceneEventHandler): ((child: SgNode) => SgNode) => onEvent("OnClick", fn);
const onPointerDown  = (fn: SceneEventHandler): ((child: SgNode) => SgNode) => onEvent("OnPointerDown", fn);
const onPointerUp    = (fn: SceneEventHandler): ((child: SgNode) => SgNode) => onEvent("OnPointerUp", fn);
const onPointerMove  = (fn: SceneEventHandler): ((child: SgNode) => SgNode) => onEvent("OnPointerMove", fn);
const onPointerEnter = (fn: SceneEventHandler): ((child: SgNode) => SgNode) => onEvent("OnPointerEnter", fn);
const onPointerLeave = (fn: SceneEventHandler): ((child: SgNode) => SgNode) => onEvent("OnPointerLeave", fn);

function active(value: aval<boolean>, child: SgNode): SgNode {
  return { kind: "Active", active: value, child };
}

function viewScope(view: aval<Trafo3d>, child: SgNode): SgNode {
  return { kind: "View", view, child };
}

function projScope(proj: aval<Trafo3d>, child: SgNode): SgNode {
  return { kind: "Proj", proj, child };
}

function camera(view: aval<Trafo3d>, proj: aval<Trafo3d>, child: SgNode): SgNode {
  return viewScope(view, projScope(proj, child));
}

function delay(
  create: (state: import("./traversalState.js").TraversalState) => SgNode,
): SgNode {
  return { kind: "Delay", create };
}

// ---------------------------------------------------------------------------
// Phase 1 — render-state scope builders
// ---------------------------------------------------------------------------

function liftAval<T>(v: T | aval<T>): aval<T> {
  return isAValRuntime(v) ? (v as aval<T>) : AVal.constant(v as T);
}

function depthTest(mode: DepthCompare | aval<DepthCompare>): (child: SgNode) => SgNode {
  const m = liftAval(mode);
  return (child: SgNode): SgNode => ({ kind: "DepthTest", mode: m, child });
}
function depthMask(write: boolean | aval<boolean>): (child: SgNode) => SgNode {
  const w = liftAval(write);
  return (child: SgNode): SgNode => ({ kind: "DepthMask", write: w, child });
}
function depthBias(bias: DepthBiasValue | aval<DepthBiasValue>): (child: SgNode) => SgNode {
  const b = liftAval(bias);
  return (child: SgNode): SgNode => ({ kind: "DepthBias", bias: b, child });
}
function depthClamp(clamp: boolean | aval<boolean>): (child: SgNode) => SgNode {
  const c = liftAval(clamp);
  return (child: SgNode): SgNode => ({ kind: "DepthClamp", clamp: c, child });
}
function cullMode(mode: CullValue | aval<CullValue>): (child: SgNode) => SgNode {
  const m = liftAval(mode);
  return (child: SgNode): SgNode => ({ kind: "CullMode", mode: m, child });
}
function frontFace(mode: FrontFaceValue | aval<FrontFaceValue>): (child: SgNode) => SgNode {
  const m = liftAval(mode);
  return (child: SgNode): SgNode => ({ kind: "FrontFace", mode: m, child });
}
function fillMode(mode: FillModeValue | aval<FillModeValue>): (child: SgNode) => SgNode {
  const m = liftAval(mode);
  return (child: SgNode): SgNode => ({ kind: "FillMode", mode: m, child });
}
function multisample(enabled: boolean | aval<boolean>): (child: SgNode) => SgNode {
  const e = liftAval(enabled);
  return (child: SgNode): SgNode => ({ kind: "Multisample", enabled: e, child });
}
function blendConstant(value: BlendConstantValue | aval<BlendConstantValue>): (child: SgNode) => SgNode {
  const v = liftAval(value);
  return (child: SgNode): SgNode => ({ kind: "BlendConstant", value: v, child });
}
function colorMask(
  mask: ColorMaskValue | HashMap<string, ColorMaskValue> | aval<HashMap<string, ColorMaskValue>>,
): (child: SgNode) => SgNode {
  let m: aval<HashMap<string, ColorMaskValue>>;
  if (isAValRuntime(mask)) {
    m = mask as aval<HashMap<string, ColorMaskValue>>;
  } else if ((mask as HashMap<string, ColorMaskValue>).count !== undefined) {
    m = AVal.constant(mask as HashMap<string, ColorMaskValue>);
  } else {
    const single = mask as ColorMaskValue;
    m = AVal.constant(HashMap.empty<string, ColorMaskValue>().add("color", single));
  }
  return (child: SgNode): SgNode => ({ kind: "ColorMask", mask: m, child });
}
function stencilMode(mode: StencilModeValue | aval<StencilModeValue>): (child: SgNode) => SgNode {
  const m = liftAval(mode);
  return (child: SgNode): SgNode => ({ kind: "StencilMode", mode: m, child });
}
function pass(value: number, child: SgNode): SgNode {
  return { kind: "Pass", pass: value, child };
}

// ---------------------------------------------------------------------------
// Phase 2 — geometry-attribute scopes
// ---------------------------------------------------------------------------

function vertexAttributes(
  attrs: HashMap<string, aval<BufferView>>,
): (child: SgNode) => SgNode {
  return (child: SgNode): SgNode => ({ kind: "VertexAttributes", attributes: attrs, child });
}
function instanceAttributes(
  attrs: HashMap<string, aval<BufferView>>,
): (child: SgNode) => SgNode {
  return (child: SgNode): SgNode => ({ kind: "InstanceAttributes", attributes: attrs, child });
}
function index(idx: BufferView | undefined | aval<BufferView | undefined>): (child: SgNode) => SgNode {
  const i: aval<BufferView | undefined> = isAValRuntime(idx)
    ? (idx as aval<BufferView | undefined>)
    : AVal.constant(idx as BufferView | undefined);
  return (child: SgNode): SgNode => ({ kind: "Index", index: i, child });
}
function mode(m: ModeValue | aval<ModeValue>): (child: SgNode) => SgNode {
  const v = liftAval(m);
  return (child: SgNode): SgNode => ({ kind: "Mode", mode: v, child });
}

// ---------------------------------------------------------------------------
// Phase 3 — misc scopes
// ---------------------------------------------------------------------------

function noEvents(value: boolean | aval<boolean>, child: SgNode): SgNode {
  return { kind: "NoEvents", value: liftAval(value), child };
}
function forcePixelPicking(value: boolean | aval<boolean>, child: SgNode): SgNode {
  return { kind: "ForcePixelPicking", value: liftAval(value), child };
}
function canFocus(value: boolean | aval<boolean>, child: SgNode): SgNode {
  return { kind: "CanFocus", value: liftAval(value), child };
}

// ---------------------------------------------------------------------------
// Trafo helpers — return Trafo3d (or aval<Trafo3d>) for use in arrays
// ---------------------------------------------------------------------------

const identity: Trafo3d = Trafo3d.identity;

function translate(v: V3d): Trafo3d;
function translate(v: aval<V3d>): aval<Trafo3d>;
function translate(v: V3d | aval<V3d>): Trafo3d | aval<Trafo3d> {
  return isAValRuntime(v)
    ? (v as aval<V3d>).map((vv) => Trafo3d.translation(vv as V3d))
    : Trafo3d.translation(v as V3d);
}

function scale(s: number | V3d | Scale3d): Trafo3d;
function scale(s: aval<number | V3d | Scale3d>): aval<Trafo3d>;
function scale(s: number | V3d | Scale3d | aval<number | V3d | Scale3d>): Trafo3d | aval<Trafo3d> {
  if (isAValRuntime(s)) return (s as aval<number | V3d | Scale3d>).map((v) => Trafo3d.scaling(v as number));
  return typeof s === "number" ? Trafo3d.scaling(s) : Trafo3d.scaling(s as V3d);
}

function rotate(a: Rot3d | { axis: V3d; rad: number }): Trafo3d;
function rotate(a: aval<Rot3d | { axis: V3d; rad: number }>): aval<Trafo3d>;
function rotate(a: unknown): Trafo3d | aval<Trafo3d> {
  const lift = (v: unknown): Trafo3d => {
    if ((v as Rot3d) instanceof Rot3d) return Trafo3d.rotation(v as Rot3d);
    const ar = v as { axis: V3d; rad: number };
    return Trafo3d.rotation(ar.axis, ar.rad);
  };
  return isAValRuntime(a) ? (a as aval<unknown>).map(lift) : lift(a);
}

function trafoOf(t: Trafo3d): Trafo3d { return t; }

function isAValRuntime(v: unknown): boolean {
  return typeof v === "object" && v !== null && typeof (v as { getValue?: unknown }).getValue === "function";
}

// ---------------------------------------------------------------------------
// JSX scope component — `<Sg Trafo Shader Uniform ...>...</Sg>`
// ---------------------------------------------------------------------------

export interface SgScopeProps {
  // Composing across nesting
  Trafo?:   TrafoValue;
  Uniform?: Record<string, unknown | aval<unknown>>;

  // Overriding (innermost wins)
  Shader?:      Effect;
  BlendMode?:   BlendState;
  Cursor?:      string | aval<string>;
  PickThrough?: boolean;
  Intersectable?: IIntersectable | aval<IIntersectable>;
  PixelSnapRadius?: number | aval<number>;

  // Camera scopes — also sniffable from outside the scene.
  View?: aval<Trafo3d>;
  Proj?: aval<Trafo3d>;

  // Active gating (AND across nesting).
  Active?: aval<boolean>;

  // Phase 1 — render-state scopes (override semantics)
  DepthTest?: DepthCompare | aval<DepthCompare>;
  DepthMask?: boolean | aval<boolean>;
  DepthBias?: DepthBiasValue | aval<DepthBiasValue>;
  DepthClamp?: boolean | aval<boolean>;
  CullMode?: CullValue | aval<CullValue>;
  FrontFace?: FrontFaceValue | aval<FrontFaceValue>;
  FillMode?: FillModeValue | aval<FillModeValue>;
  Multisample?: boolean | aval<boolean>;
  BlendConstant?: BlendConstantValue | aval<BlendConstantValue>;
  ColorMask?:
    | ColorMaskValue
    | HashMap<string, ColorMaskValue>
    | aval<HashMap<string, ColorMaskValue>>;
  StencilMode?: StencilModeValue | aval<StencilModeValue>;
  Pass?: number;

  // Phase 2 — geometry attribute scopes
  VertexAttributes?: HashMap<string, aval<BufferView>>;
  InstanceAttributes?: HashMap<string, aval<BufferView>>;
  Index?: BufferView | undefined | aval<BufferView | undefined>;
  Mode?: ModeValue | aval<ModeValue>;

  // Phase 3 — misc scopes
  NoEvents?: boolean | aval<boolean>;
  ForcePixelPicking?: boolean | aval<boolean>;
  CanFocus?: boolean | aval<boolean>;

  // Event handlers — appended to the chain at this scope. Each
  // pair is `On<Kind>` (bubble) and `OnCapture<Kind>` (capture);
  // the dispatcher walks capture outer-first then bubble inner-
  // first across the chain.
  OnClick?:           SceneEventHandler;
  OnCaptureClick?:    SceneEventHandler;
  OnPointerEnter?:    SceneEventHandler;
  OnCapturePointerEnter?: SceneEventHandler;
  OnPointerLeave?:    SceneEventHandler;
  OnCapturePointerLeave?: SceneEventHandler;
  OnPointerDown?:     SceneEventHandler;
  OnCapturePointerDown?: SceneEventHandler;
  OnPointerUp?:       SceneEventHandler;
  OnCapturePointerUp?: SceneEventHandler;
  OnPointerMove?:     SceneEventHandler;
  OnCapturePointerMove?: SceneEventHandler;
  OnTap?:             SceneEventHandler;
  OnCaptureTap?:      SceneEventHandler;
  OnDoubleTap?:       SceneEventHandler;
  OnCaptureDoubleTap?: SceneEventHandler;
  OnLongPress?:       SceneEventHandler;
  OnCaptureLongPress?: SceneEventHandler;
  // Phase 4
  OnWheel?:           SceneEventHandler;
  OnCaptureWheel?:    SceneEventHandler;
  // Phase 5 — focus/blur and keyboard
  OnFocus?:           SceneEventHandler;
  OnCaptureFocus?:    SceneEventHandler;
  OnBlur?:            SceneEventHandler;
  OnCaptureBlur?:     SceneEventHandler;
  OnKeyDown?:         SceneEventHandler;
  OnCaptureKeyDown?:  SceneEventHandler;
  OnKeyUp?:           SceneEventHandler;
  OnCaptureKeyUp?:    SceneEventHandler;
  OnKeyPress?:        SceneEventHandler;
  OnCaptureKeyPress?: SceneEventHandler;
  // Phase 6 — drag
  OnDragStart?:       SceneEventHandler;
  OnCaptureDragStart?: SceneEventHandler;
  OnDrag?:            SceneEventHandler;
  OnCaptureDrag?:     SceneEventHandler;
  OnDragEnd?:         SceneEventHandler;
  OnCaptureDragEnd?:  SceneEventHandler;

  /**
   * Children: ordinary JSX (Sg components, fragments, arrays) plus
   * raw `SgNode` values for code-generated trees and the
   * `Sg.delay` escape hatch. `collectSgChildren` flattens all of
   * the above.
   */
  children?: Child | Child[] | SgNode | ReadonlyArray<Child | SgNode>;
}

function applyScopeAttrs(node: SgNode, props: SgScopeProps): SgNode {
  // Order: events innermost, Trafo outermost. Wrapping inside-out
  // so the call site reads top-to-bottom. The JSX rule: leftmost
  // (= outermost wrapper) is applied last to a point.
  let n = node;
  const events = collectEventHandlers(props);
  if (events !== undefined) n = on(events, n);
  if (props.Proj !== undefined)        n = projScope(props.Proj, n);
  if (props.View !== undefined)        n = viewScope(props.View, n);
  if (props.Active !== undefined)      n = active(props.Active, n);
  if (props.PickThrough !== undefined) n = pickThrough(props.PickThrough, n);
  if (props.Intersectable !== undefined) n = intersectable(props.Intersectable)(n);
  if (props.PixelSnapRadius !== undefined) n = pixelSnapRadius(props.PixelSnapRadius)(n);
  if (props.Cursor !== undefined)      n = cursor(props.Cursor, n);
  if (props.BlendMode !== undefined)   n = blendMode(props.BlendMode, n);
  if (props.Shader !== undefined)      n = shader(props.Shader, n);
  if (props.Uniform !== undefined)     n = uniform(props.Uniform, n);
  // Phase 3 — innermost (close to leaves)
  if (props.NoEvents !== undefined)        n = noEvents(props.NoEvents, n);
  if (props.ForcePixelPicking !== undefined) n = forcePixelPicking(props.ForcePixelPicking, n);
  if (props.CanFocus !== undefined)        n = canFocus(props.CanFocus, n);
  // Phase 2 — geometry
  if (props.VertexAttributes !== undefined)   n = vertexAttributes(props.VertexAttributes)(n);
  if (props.InstanceAttributes !== undefined) n = instanceAttributes(props.InstanceAttributes)(n);
  if (props.Index !== undefined)              n = index(props.Index)(n);
  if (props.Mode !== undefined)               n = mode(props.Mode)(n);
  // Phase 1 — render state
  if (props.DepthTest !== undefined)   n = depthTest(props.DepthTest)(n);
  if (props.DepthMask !== undefined)   n = depthMask(props.DepthMask)(n);
  if (props.DepthBias !== undefined)   n = depthBias(props.DepthBias)(n);
  if (props.DepthClamp !== undefined)  n = depthClamp(props.DepthClamp)(n);
  if (props.CullMode !== undefined)    n = cullMode(props.CullMode)(n);
  if (props.FrontFace !== undefined)   n = frontFace(props.FrontFace)(n);
  if (props.FillMode !== undefined)    n = fillMode(props.FillMode)(n);
  if (props.Multisample !== undefined) n = multisample(props.Multisample)(n);
  if (props.BlendConstant !== undefined) n = blendConstant(props.BlendConstant)(n);
  if (props.ColorMask !== undefined)   n = colorMask(props.ColorMask)(n);
  if (props.StencilMode !== undefined) n = stencilMode(props.StencilMode)(n);
  if (props.Pass !== undefined)        n = pass(props.Pass, n);
  if (props.Trafo !== undefined)       n = trafo(props.Trafo, n);
  return n;
}

function collectEventHandlers(props: SgScopeProps): EventHandlers | undefined {
  const bubble: Partial<Record<SceneEventKind, SceneEventHandler>> = {};
  const capture: Partial<Record<SceneEventKind, SceneEventHandler>> = {};
  let any = false;
  if (props.OnClick)        { bubble.OnClick = props.OnClick; any = true; }
  if (props.OnPointerEnter) { bubble.OnPointerEnter = props.OnPointerEnter; any = true; }
  if (props.OnPointerLeave) { bubble.OnPointerLeave = props.OnPointerLeave; any = true; }
  if (props.OnPointerDown)  { bubble.OnPointerDown = props.OnPointerDown; any = true; }
  if (props.OnPointerUp)    { bubble.OnPointerUp = props.OnPointerUp; any = true; }
  if (props.OnPointerMove)  { bubble.OnPointerMove = props.OnPointerMove; any = true; }
  if (props.OnTap)          { bubble.OnTap = props.OnTap; any = true; }
  if (props.OnDoubleTap)    { bubble.OnDoubleTap = props.OnDoubleTap; any = true; }
  if (props.OnLongPress)    { bubble.OnLongPress = props.OnLongPress; any = true; }
  if (props.OnCaptureClick)        { capture.OnClick = props.OnCaptureClick; any = true; }
  if (props.OnCapturePointerEnter) { capture.OnPointerEnter = props.OnCapturePointerEnter; any = true; }
  if (props.OnCapturePointerLeave) { capture.OnPointerLeave = props.OnCapturePointerLeave; any = true; }
  if (props.OnCapturePointerDown)  { capture.OnPointerDown = props.OnCapturePointerDown; any = true; }
  if (props.OnCapturePointerUp)    { capture.OnPointerUp = props.OnCapturePointerUp; any = true; }
  if (props.OnCapturePointerMove)  { capture.OnPointerMove = props.OnCapturePointerMove; any = true; }
  if (props.OnCaptureTap)          { capture.OnTap = props.OnCaptureTap; any = true; }
  if (props.OnCaptureDoubleTap)    { capture.OnDoubleTap = props.OnCaptureDoubleTap; any = true; }
  if (props.OnCaptureLongPress)    { capture.OnLongPress = props.OnCaptureLongPress; any = true; }
  if (props.OnWheel)        { bubble.OnWheel = props.OnWheel; any = true; }
  if (props.OnCaptureWheel) { capture.OnWheel = props.OnCaptureWheel; any = true; }
  if (props.OnFocus)        { bubble.OnFocus = props.OnFocus; any = true; }
  if (props.OnCaptureFocus) { capture.OnFocus = props.OnCaptureFocus; any = true; }
  if (props.OnBlur)         { bubble.OnBlur = props.OnBlur; any = true; }
  if (props.OnCaptureBlur)  { capture.OnBlur = props.OnCaptureBlur; any = true; }
  if (props.OnKeyDown)      { bubble.OnKeyDown = props.OnKeyDown; any = true; }
  if (props.OnCaptureKeyDown) { capture.OnKeyDown = props.OnCaptureKeyDown; any = true; }
  if (props.OnKeyUp)        { bubble.OnKeyUp = props.OnKeyUp; any = true; }
  if (props.OnCaptureKeyUp) { capture.OnKeyUp = props.OnCaptureKeyUp; any = true; }
  if (props.OnKeyPress)     { bubble.OnKeyPress = props.OnKeyPress; any = true; }
  if (props.OnCaptureKeyPress) { capture.OnKeyPress = props.OnCaptureKeyPress; any = true; }
  if (props.OnDragStart)    { bubble.OnDragStart = props.OnDragStart; any = true; }
  if (props.OnCaptureDragStart) { capture.OnDragStart = props.OnCaptureDragStart; any = true; }
  if (props.OnDrag)         { bubble.OnDrag = props.OnDrag; any = true; }
  if (props.OnCaptureDrag)  { capture.OnDrag = props.OnCaptureDrag; any = true; }
  if (props.OnDragEnd)      { bubble.OnDragEnd = props.OnDragEnd; any = true; }
  if (props.OnCaptureDragEnd) { capture.OnDragEnd = props.OnCaptureDragEnd; any = true; }
  if (!any) return undefined;
  const out: { capture?: typeof capture; bubble?: typeof bubble } = {};
  if (Object.keys(capture).length > 0) out.capture = capture;
  if (Object.keys(bubble).length > 0) out.bubble = bubble;
  return out;
}

function SgScope(props: SgScopeProps): VNode {
  const childNode = collectSgChildren(props.children);
  return sgVNode(applyScopeAttrs(childNode, props));
}

function SgUnordered(props: SgScopeProps): VNode {
  const inner = collectSgChildren(props.children);
  const unorderedChildren: SgNode =
    inner.kind === "Empty" ? inner : unordered([inner]);
  return sgVNode(applyScopeAttrs(unorderedChildren, props));
}

function SgAdaptive(props: { value: aval<SgNode> }): VNode {
  return sgVNode(adaptive(props.value));
}

function SgBox(props: BoxOptions = {}): VNode {
  return sgVNode(boxLeaf(props));
}

function SgQuad(props: QuadOptions = {}): VNode {
  return sgVNode(quadLeaf(props));
}

// ---------------------------------------------------------------------------
// Children-walker — extract SgNodes from JSX children
// ---------------------------------------------------------------------------

/**
 * Walk a JSX-children value and produce a single `SgNode`.
 *
 *   - Tagged Fragment carriers (built by `sgVNode`) → unwrap.
 *   - Component VNodes → call type fn and recurse.
 *   - Plain Fragment VNodes → recurse on `props.children`.
 *   - Arrays → flatten + group.
 *   - aval<SgNode>  → SgAdaptiveGroup.
 *   - alist<SgNode> → SgGroup.
 *   - aset<SgNode>  → SgUnorderedGroup.
 *   - Raw `SgNode` (object with a known `kind`) → pass through.
 *   - Everything else → ignored.
 *
 * Multi-segment children (mixing leaves + alists + asets) preserve
 * each segment's container shape — see CLAUDE.md "heterogeneous
 * JSX children" rule.
 */
export function collectSgChildren(value: unknown): SgNode {
  const segments = collectImpl(value);
  if (segments.length === 0) return empty;
  if (segments.length === 1) return segments[0]!;
  return group(segments);
}

function collectImpl(value: unknown): SgNode[] {
  if (value === null || value === undefined || value === false || value === true) return [];
  if (Array.isArray(value)) {
    const out: SgNode[] = [];
    for (const v of value) out.push(...collectImpl(v));
    return out;
  }
  if (isVNode(value)) {
    if (isSgVNode(value)) return [extractSgNode(value)];
    const v: { _tag: string; type?: (p: unknown) => unknown; props: { children?: unknown } } = value as never;
    if (v._tag === "component") return collectImpl(v.type!(v.props));
    if (v._tag === "fragment")  return collectImpl(v.props.children);
    return []; // element VNode — HTML inside RenderControl has no surface
  }
  if (isRawSgNode(value)) return [value];
  if (isAVal(value))  return [adaptive(value as aval<SgNode>)];
  if (isAList(value)) return [group(value as alist<SgNode>)];
  if (isASet(value))  return [unordered(value as aset<SgNode>)];
  return [];
}

function isRawSgNode(value: unknown): value is SgNode {
  if (value === null || typeof value !== "object") return false;
  const kind = (value as { kind?: unknown }).kind;
  return typeof kind === "string" && SG_KINDS.has(kind);
}

// ---------------------------------------------------------------------------
// Unified `Sg` namespace — JSX function + imperative + helpers + sub-components
// ---------------------------------------------------------------------------

interface SgNamespace {
  // JSX call signature
  (props: SgScopeProps): VNode;

  // Sub-components (JSX)
  Unordered: typeof SgUnordered;
  Adaptive:  typeof SgAdaptive;
  Box:       typeof SgBox;
  Quad:      typeof SgQuad;

  // Imperative builders — return SgNode / SgLeaf for code-
  // generated trees and inside `Sg.delay`.
  empty:       SgNode;
  leaf:        typeof leaf;
  group:       typeof group;
  unordered:   typeof unordered;
  adaptive:    typeof adaptive;
  trafo:       typeof trafo;
  shader:      typeof shader;
  uniform:     typeof uniform;
  blendMode:   typeof blendMode;
  cursor:      typeof cursor;
  pickThrough: typeof pickThrough;
  intersectable: typeof intersectable;
  pixelSnapRadius: typeof pixelSnapRadius;
  on:          typeof on;
  onClick:        typeof onClick;
  onPointerDown:  typeof onPointerDown;
  onPointerUp:    typeof onPointerUp;
  onPointerMove:  typeof onPointerMove;
  onPointerEnter: typeof onPointerEnter;
  onPointerLeave: typeof onPointerLeave;
  active:      typeof active;
  view:        typeof viewScope;
  proj:        typeof projScope;
  camera:      typeof camera;
  delay:       typeof delay;
  // Imperative primitives (lowercase) — same draws as `<Sg.Box/>`/
  // `<Sg.Quad/>` but as plain SgLeaf values, usable inside `delay`.
  box:         typeof boxLeaf;
  quad:        typeof quadLeaf;

  // Math helpers (return Trafo3d, plain or aval)
  identity:  Trafo3d;
  translate: typeof translate;
  scale:     typeof scale;
  rotate:    typeof rotate;
  trafoOf:   typeof trafoOf;

  // Phase 1 builders
  depthTest:    typeof depthTest;
  depthMask:    typeof depthMask;
  depthBias:    typeof depthBias;
  depthClamp:   typeof depthClamp;
  cullMode:     typeof cullMode;
  frontFace:    typeof frontFace;
  fillMode:     typeof fillMode;
  multisample:  typeof multisample;
  blendConstant: typeof blendConstant;
  colorMask:    typeof colorMask;
  stencilMode:  typeof stencilMode;
  pass:         typeof pass;
  RenderPass:   typeof RenderPass;

  // Phase 2 builders
  vertexAttributes:   typeof vertexAttributes;
  instanceAttributes: typeof instanceAttributes;
  index:              typeof index;
  mode:               typeof mode;

  // Phase 3 builders
  noEvents:          typeof noEvents;
  forcePixelPicking: typeof forcePixelPicking;
  canFocus:          typeof canFocus;
}

export const Sg: SgNamespace = (() => {
  const fn = SgScope as SgNamespace;
  fn.Unordered = SgUnordered;
  fn.Adaptive  = SgAdaptive;
  fn.Box       = SgBox;
  fn.Quad      = SgQuad;
  fn.empty       = empty;
  fn.leaf        = leaf;
  fn.group       = group;
  fn.unordered   = unordered;
  fn.adaptive    = adaptive;
  fn.trafo       = trafo;
  fn.shader      = shader;
  fn.uniform     = uniform;
  fn.blendMode   = blendMode;
  fn.cursor      = cursor;
  fn.pickThrough = pickThrough;
  fn.intersectable = intersectable;
  fn.pixelSnapRadius = pixelSnapRadius;
  fn.on          = on;
  fn.onClick        = onClick;
  fn.onPointerDown  = onPointerDown;
  fn.onPointerUp    = onPointerUp;
  fn.onPointerMove  = onPointerMove;
  fn.onPointerEnter = onPointerEnter;
  fn.onPointerLeave = onPointerLeave;
  fn.active      = active;
  fn.view        = viewScope;
  fn.proj        = projScope;
  fn.camera      = camera;
  fn.delay       = delay;
  fn.box         = boxLeaf;
  fn.quad        = quadLeaf;
  fn.identity    = identity;
  fn.translate   = translate;
  fn.scale       = scale;
  fn.rotate      = rotate;
  fn.trafoOf     = trafoOf;
  fn.depthTest    = depthTest;
  fn.depthMask    = depthMask;
  fn.depthBias    = depthBias;
  fn.depthClamp   = depthClamp;
  fn.cullMode     = cullMode;
  fn.frontFace    = frontFace;
  fn.fillMode     = fillMode;
  fn.multisample  = multisample;
  fn.blendConstant = blendConstant;
  fn.colorMask    = colorMask;
  fn.stencilMode  = stencilMode;
  fn.pass         = pass;
  fn.RenderPass   = RenderPass;
  fn.vertexAttributes   = vertexAttributes;
  fn.instanceAttributes = instanceAttributes;
  fn.index              = index;
  fn.mode               = mode;
  fn.noEvents          = noEvents;
  fn.forcePixelPicking = forcePixelPicking;
  fn.canFocus          = canFocus;
  return fn;
})();

void Shift3d;
