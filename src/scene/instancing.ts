// Auto-instancing — scene-compile-time validator + leaf rewrite.
//
// `Sg.Instanced` wraps a subtree with a `count` + per-uniform-name
// attribute streams (or a `trafos` aval for the trafo convenience).
// At compile time we walk the subtree once to enforce the
// precondition (mirroring Aardvark's
// `Aardvark.SceneGraph.Semantics.Instancing.applyTrafos`):
//
//   * No nested `SgInstanced` — composition isn't sound, and
//     Aardvark errors here too. Use multiple top-level scopes if
//     you need disjoint instance sets.
//   * No leaf whose `drawCall.instanceCount > 1` — those are
//     already-instanced and would conflict with our `count`.
//   * No indirect-draw leaves (none exist in wombat.dom yet — the
//     check is a placeholder for the future).
//
// At leaf-lowering time `applyInstancing` rewrites the effect via
// `instanceEffect`, splits matrix attributes into their 4 vec4-column
// streams, runs the trafo CPU pre-merge, and overrides the leaf's
// `instanceCount`. See `docs/auto-instancing.md` for design.

import { AVal, HashMap, type aval } from "@aardworx/wombat.adaptive";
import { Trafo3d } from "@aardworx/wombat.base";
import { IBuffer, type BufferView, type DrawCall,
  ElementType,
} from "@aardworx/wombat.rendering/core";
import type { Effect } from "@aardworx/wombat.shader";
import { instanceEffect } from "@aardworx/wombat.shader";
import type { SgInstanced, SgLeaf, SgNode } from "./sg.js";
import { markHostBufferAVal } from "./hostBuffers.js";

/**
 * Walk the subtree once and surface a friendly error for any
 * precondition violation. Throws on the first violation found.
 *
 * Called twice: once at scene-compile time (eager check, by
 * `compile.ts:lower`'s `Instanced` case), and once per swap when an
 * adaptive container *inside* the instancing scope changes content
 * (so a violator that wasn't present at compile time doesn't render
 * silently-wrong pixels). The compile-time call surfaces the error
 * directly to the user; the swap-time call is wrapped in a try/catch
 * by the caller so a violator just renders `RenderTree.empty` plus a
 * console.error rather than crashing the rAF tick.
 */
export function validateInstancingSubtree(child: SgNode): void {
  walk(child);

  function walk(node: SgNode): void {
    switch (node.kind) {
      case "Empty": return;
      case "Leaf": {
        const dc = AVal.force(node.drawCall);
        if (dc.instanceCount !== undefined && dc.instanceCount > 1) {
          throw new Error(
            `Sg.Instanced: nested leaf has drawCall.instanceCount=${dc.instanceCount}; ` +
            `cannot wrap an already-instanced draw. Move the leaf out of this scope.`,
          );
        }
        return;
      }
      case "Instanced":
        throw new Error(
          "Sg.Instanced: nested `Sg.Instanced` is not supported. " +
          "Compose at most one instancing scope per subtree.",
        );
      case "Group":
        for (const c of AVal.force(node.children.content)) walk(c); return;
      case "UnorderedGroup":
        for (const c of AVal.force(node.children.content)) walk(c); return;
      case "AdaptiveGroup":
        walk(AVal.force(node.child)); return;
      case "Delay":
        // Delay's creator may produce arbitrary subtrees; force once
        // for the validation pass. If that throws, surface as-is.
        try { walk(node.create(undefined as never)); } catch { /* nothing to validate */ }
        return;
      // Pass-through scopes — recurse into the child with no state
      // change as far as instancing validation cares.
      case "Trafo": case "Shader": case "Uniform": case "BlendMode":
      case "Cursor": case "PickThrough": case "Intersectable":
      case "PixelSnapRadius": case "On": case "Active":
      case "View": case "Proj":
      case "DepthTest": case "DepthMask": case "DepthBias": case "DepthClamp":
      case "CullMode": case "FrontFace": case "FillMode":
      case "BlendConstant": case "ColorMask": case "StencilMode": case "Pass":
      case "VertexAttributes": case "InstanceAttributes":
      case "Index": case "Mode":
      case "NoEvents": case "ForcePixelPicking": case "CanFocus":
        walk(node.child); return;
    }
  }
}

/** Re-export so callers can name the type without reaching into `sg.js`. */
export type { SgInstanced };

// ---------------------------------------------------------------------------
// Leaf rewrite — produces a (rewritten Effect, instance-attribute
// HashMap, identity-trafo uniform overrides, drawCall override).
// ---------------------------------------------------------------------------

export interface AppliedInstancing {
  readonly effect: Effect;
  /** Combined instance-attribute streams (existing leaf + synthesised). */
  readonly instanceAttributes: HashMap<string, BufferView>;
  /** Uniform overrides — used to bind ModelTrafo (etc.) to identity. */
  readonly uniformOverrides: HashMap<string, aval<unknown>>;
  /** drawCall override — `instanceCount` set from the scope's `count`. */
  readonly drawCall: aval<DrawCall>;
}

export function applyInstancing(
  inst: SgInstanced,
  /** Cumulative ModelTrafo accumulated INSIDE the instancing subtree
   *  (i.e. the leaf's `state.model` after the scope reset). LAZY —
   *  only the trafos branch resolves it, so non-trafo instanced rows
   *  never build a model composite. */
  innerModel: () => aval<Trafo3d>,
  /** Cumulative ModelTrafo from OUTSIDE the instancing scope. Becomes
   *  the leaf's `ModelTrafo` uniform after the rewrite — the shader
   *  reads `uniform.ModelTrafo * inst.InstanceTrafo`, and we want
   *  `uniform.ModelTrafo` to carry the parent-scope trafo so the
   *  full transform comes out as `parentTrafo * (instance · inner)`. */
  parentModel: () => aval<Trafo3d>,
  /** Scope view/proj — needed to recompute the composite ModelView*
   *  uniforms with the parent (rather than inner) model. */
  view: aval<Trafo3d>,
  proj: aval<Trafo3d>,
  /** The user-supplied (or pick-composed) effect for the leaf. */
  innerEffect: Effect,
  leaf: SgLeaf,
): AppliedInstancing {
  const attrNames = collectAttrNames(inst);
  const effect = instanceEffect(innerEffect, attrNames);

  let instAttrs = leaf.instanceAttributes ?? HashMap.empty<string, BufferView>();
  let uniformOverrides = HashMap.empty<string, aval<unknown>>();

  // Trafo case — pre-merge inner trafo with each per-instance trafo,
  // then split the resulting M44 buffers into 4 vec4 columns each.
  // The default `trafo()` VS computes world-space normals as
  // `vec.mul(uniform.ModelTrafoInv)` (row-vec form), which under
  // wombat.shader's row/col-major convention is the inv-transpose
  // normal transform — no separate NormalMatrix uniform needed.
  // `ModelTrafoInv`'s per-instance rewrite (in `instanceUniforms`'s
  // INVERSE_TRAFOS table) already composes the parent uniform with
  // `InstanceTrafoInv`, so the per-instance normal transform falls
  // out for free.
  if (inst.trafos !== undefined) {
    const innerModelR = innerModel();
    const parentModelR = parentModel();
    const merged = mergeTrafosCached(innerModelR, inst.trafos);
    for (let i = 0; i < 4; i++) {
      instAttrs = instAttrs.add(`_InstanceTrafo_col${i}`,    merged.fwCols[i]!);
      instAttrs = instAttrs.add(`_InstanceTrafoInv_col${i}`, merged.bwCols[i]!);
    }
    // The auto-injected trafo uniforms at the leaf are derived from
    // `state.model` — but inside an `Sg.Instanced` scope the leaf's
    // `state.model` is `innerModel` (the trafos accumulated AFTER the
    // pushInstancing reset), NOT the outer/parent model. The
    // instanceUniforms IR rules expect the standard Aardvark
    // composition: `uniform.X * InstanceTrafo` for forwards,
    // `InstanceTrafoInv * uniform.X` for inverses — where `uniform.X`
    // is the OUTER (parent-of-scope) trafo so the product yields the
    // full chain. We have to override every auto-injected trafo at
    // the leaf so the IR rules see the parent value, not innerModel.
    //
    // (`ModelTrafo` only would have worked if the user only ever
    // multiplied `uniform.ModelTrafo · vertex` — but `trafo()` also
    // reads `ModelTrafoInv` for the normal transform, and the picking
    // VS reads `ModelViewTrafoInv`. Without the matching overrides,
    // the per-instance normal transform comes out as
    // `(innerModel.backward · t_i.backward) · innerModel.backward · v`
    // — innerModel applied twice — which is exactly the symptom of
    // the cylinder/cone normals not picking up the per-instance
    // rotation.)
    const inv = (t: aval<Trafo3d>): aval<Trafo3d> => t.map(x => x.inverse());
    const compose = (a: aval<Trafo3d>, b: aval<Trafo3d>): aval<Trafo3d> =>
      AVal.zip(a, b).map((aT, bT) => aT.mul(bT));
    const parentModelView    = compose(parentModelR, view);
    const viewProj           = compose(view, proj);
    const parentModelViewProj = compose(parentModelR, viewProj);
    // Match the auto-injected NormalMatrix shape (M44 = inv-T padded
    // for direction-vec multiplies). For the parent scope.
    const parentNormalMatrix = parentModelR.map(t => {
      const invT = t.backward.transpose();
      return Trafo3d.fromMatrices(invT, invT.transpose());
    });
    uniformOverrides = uniformOverrides
      .add("ModelTrafo",            parentModelR as aval<unknown>)
      .add("ModelTrafoInv",         inv(parentModelR) as aval<unknown>)
      .add("ModelViewTrafo",        parentModelView as aval<unknown>)
      .add("ModelViewTrafoInv",     inv(parentModelView) as aval<unknown>)
      .add("ModelViewProjTrafo",    parentModelViewProj as aval<unknown>)
      .add("ModelViewProjTrafoInv", inv(parentModelViewProj) as aval<unknown>)
      .add("NormalMatrix",          parentNormalMatrix as aval<unknown>);
  }

  // Generic non-trafo attributes — passed through verbatim. Matrix-
  // typed plain attributes aren't supported in this phase.
  if (!inst.attributes.isEmpty) {
    for (const [name, view] of inst.attributes) {
      instAttrs = instAttrs.add(name, view);
    }
  }

  // Override the leaf's drawCall to set instanceCount = scope.count.
  // (The validator already rejected leaves with instanceCount>1, so
  // this is always a clean overwrite.) Constant-fold when both inputs
  // are constant; a constant leaf drawCall with an adaptive count
  // (the collection-editing shape: per-item geometry template + count
  // aval) needs only ONE map node — the zip pair is reserved for the
  // rare fully-adaptive leaf drawCall.
  let drawCall: aval<DrawCall>;
  if (leaf.drawCall.isConstant) {
    // AVal.force OK: isConstant — immutable by definition.
    const dc0 = AVal.force(leaf.drawCall);
    drawCall = inst.count.isConstant
      ? AVal.constant<DrawCall>({ ...dc0, instanceCount: AVal.force(inst.count) })
      : inst.count.map((n) => ({ ...dc0, instanceCount: n }));
    // Construction-level proof: every value this aval can ever
    // produce is `{ ...dc0, instanceCount: n }` — dc0 is CONSTANT, so
    // the shape (kind, firstVertex/firstInstance, geometry counts) is
    // pinned by this code for all time; only instanceCount varies
    // (and the heap draws n=0 as nothing). Record the proof so row
    // lowering can assert heap eligibility without ever touching the
    // live aval. This is NOT a snapshot of a changeable value.
    if (
      dc0.firstInstance === 0 &&
      (dc0.kind === "indexed" || dc0.firstVertex === 0)
    ) {
      (drawCall as { __sgHeapSafeDraw?: { kind: DrawCall["kind"] } })
        .__sgHeapSafeDraw = { kind: dc0.kind };
    }
  } else {
    drawCall = AVal.zip(leaf.drawCall, inst.count).map((dc, n) => ({
      ...dc, instanceCount: n,
    }));
  }

  return { effect, instanceAttributes: instAttrs, uniformOverrides, drawCall };
}

function collectAttrNames(inst: SgInstanced): Set<string> {
  const out = new Set<string>();
  if (inst.trafos !== undefined) out.add("ModelTrafo");
  for (const [name] of inst.attributes) out.add(name);
  return out;
}

// ---------------------------------------------------------------------------
// CPU trafo pre-merge — `parent.mul(inst[i])` for each i, packed into
// two Float32 buffers (forward + backward), each split into 4 vec4
// column BufferViews keyed `_InstanceTrafo_col0..3` etc.
//
// Cached by (parentModel, instances) aval-pair identity. Both inputs
// are `aval`; the cache is keyed on the JS object identity of the two
// avals so two leaves under the same `Sg.InstancedTrafos` (same parent
// scope, same trafo array) share the merged buffers.
// ---------------------------------------------------------------------------

interface MergedTrafos {
  readonly fwCols: readonly BufferView[]; // length 4 — InstanceTrafo
  readonly bwCols: readonly BufferView[]; // length 4 — InstanceTrafoInv
}

const mergeCache = new WeakMap<aval<Trafo3d>, WeakMap<aval<ReadonlyArray<Trafo3d>>, MergedTrafos>>();

function mergeTrafosCached(
  parent: aval<Trafo3d>,
  instances: aval<ReadonlyArray<Trafo3d>>,
): MergedTrafos {
  let inner = mergeCache.get(parent);
  if (!inner) { inner = new WeakMap(); mergeCache.set(parent, inner); }
  const cached = inner.get(instances);
  if (cached) return cached;

  // One Float32Array per direction. Re-emitted whenever either input
  // changes; downstream `prepareAdaptiveBuffer` re-uploads on aval
  // change.
  const fwBytes = AVal.zip(parent, instances).map(packMergedFW);
  const bwBytes = AVal.zip(parent, instances).map(packMergedBW);

  // Each column is one BufferView pointing at the same backing buffer
  // at offset N*16, format float32x4, stride 64 bytes.
  const fwCols = colsFromPackedM44(fwBytes);
  const bwCols = colsFromPackedM44(bwBytes);

  const merged: MergedTrafos = { fwCols, bwCols };
  inner.set(instances, merged);
  return merged;
}

function packMergedFW(parent: Trafo3d, instances: ReadonlyArray<Trafo3d>): Float32Array {
  const out = new Float32Array(instances.length * 16);
  for (let i = 0; i < instances.length; i++) {
    const merged = parent.mul(instances[i]!);
    const src = (merged.forward as unknown as { _data: Float64Array })._data;
    const dst = i * 16;
    for (let j = 0; j < 16; j++) out[dst + j] = src[j]!;
  }
  return out;
}

function packMergedBW(parent: Trafo3d, instances: ReadonlyArray<Trafo3d>): Float32Array {
  const out = new Float32Array(instances.length * 16);
  for (let i = 0; i < instances.length; i++) {
    const merged = parent.mul(instances[i]!);
    const src = (merged.backward as unknown as { _data: Float64Array })._data;
    const dst = i * 16;
    for (let j = 0; j < 16; j++) out[dst + j] = src[j]!;
  }
  return out;
}


function colsFromPackedM44(packed: aval<Float32Array>): BufferView[] {
  // Four column views, each pointing at the SAME aval<IBuffer> at the
  // appropriate byte offset within each 64-byte matrix entry. Sharing
  // the aval reference is essential for the runtime's buffer-grouping
  // pass (`prepareRenderObject` packs multiple attributes into one
  // GPU vertex slot when they share an aval<IBuffer>) — without it
  // we'd hit `maxVertexBuffers` (8) the moment a leaf has more than
  // two matrix instance attributes.
  const sharedBuf: aval<import("@aardworx/wombat.rendering/core").IBuffer> =
    markHostBufferAVal(packed.map((arr) => IBuffer.fromHost(arr)));
  const out: BufferView[] = [];
  for (let col = 0; col < 4; col++) {
    const offset = col * 16;
    out.push({
      buffer: sharedBuf,
      elementType: ElementType.V4f,
      offset,
      stride: 64,
    });
  }
  return out;
}

