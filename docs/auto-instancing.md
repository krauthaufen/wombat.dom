# Auto-instancing — design + execution log

Status: **in progress** (last updated 2026-05-06).

User-driven scene-graph instancing modeled on
`Aardvark.SceneGraph.Instancing` (see
`aardvark.rendering/src/Aardvark.SceneGraph/HighLevelSceneGraph/Instancing.fs`
for the F# original we're porting).

The user wraps a subtree with `Sg.Instanced`, supplies per-instance
attribute streams keyed by uniform name, and gets one instanced draw
per leaf in that subtree. The subtree must not itself contain another
`Sg.Instanced`, must not use indirect draws, and must not have leaves
with `drawCall.instanceCount > 1`. We validate at compile-scene time
and surface a friendly error otherwise.

## Public surface

```ts
// Generic.
<Sg.Instanced count={n} attributes={{ ModelTrafo: trafos, Color: colors }}>
  <Sg.Box />
</Sg.Instanced>

// Convenience for the common case.
<Sg.InstancedTrafos trafos={trafos}>
  <Sg.Box />
</Sg.InstancedTrafos>
```

`trafos: aval<Trafo3d[]>` — bookkept as a `BufferView`; the runtime
uploads + binds with `stepMode: "instance"`.

## Effect rewrite (the IR pass)

`inlineInstancing(module, attrNames: ReadonlySet<string>): Module`
in `wombat.shader/packages/shader/src/passes/instanceUniforms.ts`.

For each name `X` in `attrNames`:

| name pattern | rewrite |
|---|---|
| `ModelTrafo`, `ModelViewTrafo`, `ModelViewProjTrafo` | `ReadUniform(X) * ReadInput("InstanceTrafo")` |
| `ModelTrafoInv`, `ModelViewTrafoInv`, `ModelViewProjTrafoInv` | `ReadInput("InstanceTrafoInv") * ReadUniform(X)` |
| `NormalMatrix` | `ReadUniform("NormalMatrix") * transpose(m33(ReadInput("InstanceTrafoInv")))` |
| anything else | `ReadInput(X)` (plain attribute substitution) |

VS gets new entry inputs: `InstanceTrafo: M44f`,
`InstanceTrafoInv: M44f`, and one input per non-trafo attribute name.
FS reads of any rewritten name get piped through a flat-interpolated
varying synthesised in the VS.

Cached by `(effect.id, sorted(attrNames))`.

## Trafo pre-merge

CPU-side, before upload:
- Each per-instance trafo `Tᵢ` becomes `ModelScopeForward · Tᵢ` for the
  forward buffer, `Tᵢ⁻¹ · ModelScopeBackward` for the inverse buffer.
- The leaf's `ModelTrafo` uniform is then forced to identity (the
  scope-accumulated trafo lives in `InstanceTrafo[i]`).

Cached by `BinaryCache(parentModel, instances)` so multiple leaves
sharing the same parent + buffer share the merged buffers.

## Phases / execution log

### Phase 1 — IR pass (`instanceUniforms`)

- [x] Re-read Aardvark's `Effect.inlineTrafo`.
- [x] Pass implementation in `wombat.shader/packages/shader/src/passes/instanceUniforms.ts`.
- [x] Plain uniform → instance attribute rewrite (`Color: V4f` etc.).
- [x] Matrix attributes split into 4 vec4 columns (WebGPU forbids
      mat-typed vertex attributes).
- [x] `ModelTrafo` (+ `ModelViewTrafo`, `ModelViewProjTrafo`)
      post-multiply with `InstanceTrafo`.
- [x] `ModelTrafoInv` (+ inverses) pre-multiply with `InstanceTrafoInv`.
- [x] `NormalMatrix` rebuild from `m33(InstanceTrafoInv).transpose`,
      with M33→M44 padding when the source uniform was declared M44
      (wombat's `UniformScope.NormalMatrix` is M44 padded; Aardvark's
      is bare M33).
- [x] FS reads of rewritten uniforms get flat varyings synthesised in
      the VS. Matrix-typed varyings split into 4 vec4 columns same as
      vertex attributes; the FS reconstructs.
- [x] Auto-assigned VS-output Locations for synth varyings (`max +1`
      from existing outputs, mirroring `linkCrossStage`'s position
      pass-through).
- [x] Dawn-validated tests in `tests/instance-rewrite.test.ts`
      (7 cases).

Status: shipped. 7/7 phase-1 tests pass under Dawn; 293 total tests
across the shader suite (was 286 before this phase).

Notable WGSL-quirks handled:
- `mat3x3<f32>(mat4x4<f32>(...))` is invalid → build M33 directly
  from `vec4.xyz` swizzles or `MatrixCol(m44, i).xyz`.
- `@location(...)` can't be applied to `mat4x4<f32>` (vertex
  attributes or interstage varyings) → split into vec4 columns,
  reconstruct via `MatrixFromCols`.

### Phase 2 — `SgInstanced` node + subtree validation

- [x] `SgInstanced` variant added to the `SgNode` union
      (`wombat.dom/src/scene/sg.ts`). Carries `count: aval<number>`,
      `attributes: HashMap<string, aval<BufferView>>`, and `child`.
- [x] All three exhaustive `kind` switches in `compile.ts`
      (`sceneUsesPassStatic`, `collectByPass`, `lower`) plus
      `forEachLeaf` in `visit.ts` extended with the `"Instanced"`
      case.
- [x] `validateInstancingSubtree(child)` in
      `wombat.dom/src/scene/instancing.ts`. Walks once; throws on
      nested `SgInstanced` and on leaves where
      `drawCall.instanceCount > 1`. Recurses through Adaptive /
      Unordered / Ordered groups (forces avals once).
- [x] `TraversalState.pushInstancing(node)` + `instancing` field
      threaded through `with()` and `empty`. The `lower` case for
      `Instanced` calls `validateInstancingSubtree` then pushes the
      scope.
- [x] 6 unit tests in `tests/scene-instancing.test.ts`. Full
      wombat.dom suite stays at 235/235 (was 229 + 6 new).

The indirect-draw clause is a placeholder — wombat.dom has no
`SgIndirect` / indirect-draw construct yet, so there's nothing to
reject. When it lands the validator gets a clause.

### Phase 3 — `compileScene` integration

- [x] `instanceEffect(inner, attrNames)` wrapper added to
      `wombat.shader/runtime/instanceEffect.ts`. Returns a new Effect
      whose `compile()` threads `instanceAttributes` through
      `CompileOptions`; the runtime's pipeline applies the
      `instanceUniforms` IR pass after `liftReturns`.
- [x] `applyInstancing(scope, scopeModel, effect, leaf)` in
      `wombat.dom/scene/instancing.ts`. Returns:
  - rewritten Effect (via `instanceEffect`).
  - patched `instanceAttributes` (8 vec4-column streams for
    InstanceTrafo + InstanceTrafoInv; plain attrs pass through).
  - `uniformOverrides` (forces `ModelTrafo` to identity in the
    trafo case).
  - `drawCall` aval with `instanceCount = scope.count`.
- [x] Trafo CPU pre-merge cached per `(parentModelAval, instancesAval)`
      via nested WeakMaps; produces `Float32Array` buffers
      (forward + backward), each split into 4 vec4-column BufferViews
      at offsets 0/16/32/48, stride 64.
- [x] `lowerLeaf` consults `state.instancing` and merges
      `applyInstancing`'s output into the RenderObject before
      `buildRenderObject`. Active gating + Effect composition with
      the picking chain stay unchanged downstream.
- [x] 8/8 unit tests in `tests/scene-instancing.test.ts`. wombat.dom
      suite stays green at 237/237.

Out of scope for this phase (deferred to a future polish pass):
- Generic matrix-typed plain attributes (only `ModelTrafo` via the
  trafo convenience is supported). Plain non-matrix attributes work.
- Re-validation when an `SgAdaptiveGroup` inside an instanced subtree
  swaps in a leaf with `instanceCount > 1` mid-frame. Validation is
  one-shot at scene-compile time today.

### Phase 4 — public constructors + showcase demo

- [x] `Sg.Instanced` and `Sg.InstancedTrafos` constructors (`wombat.dom/src/scene/constructors.ts`).
- [x] Module augmentation in `text.ts`-style declaration so `<Sg.Instanced/>` JSX works.
- [x] Showcase: 1024 coordinate-cross gizmos (cylinder + cone arrows + centre sphere) at deterministic random trafos with a toggle to a non-instanced control variant. `wombat.dom/examples/instancing/`.
- [x] Manual smoke check: open in Chromium, verify single instanced draw via DevTools or PIX-equivalent.
- [x] Geometry equivalence: with `effect(trafo, vertexColor)` the
  toggle is **pixel-perfect** (`pctDiff: 0, maxChannelDelta: 0`).
- [x] Lit equivalence: with `effect(trafo, simpleLighting)` and a
  non-trivial inner trafo chain (cylinder/cone non-uniform scales +
  arrow `orient` rotation + per-instance random rotation+translation),
  the toggle is **pixel-perfect** (`pctDiff: 0, maxChannelDelta: 0`)
  with the GPU-side normal transform — no per-instance NormalMatrix
  buffer, no CPU pre-merge.

#### Root cause: `applyInstancing` was leaking inner-scope uniforms

The bug was in **`wombat.dom/src/scene/instancing.ts`'s
`applyInstancing`**, not in the shader passes. The IR rewrite for
`uniform.ModelTrafoInv` is

```
uniform.ModelTrafoInv → MulMatMat(m44FromCols(InstanceTrafoInv),
                                  ReadUniform("ModelTrafoInv"))
```

— Aardvark's `InstanceTrafoInv · uniform.MTI` form. The composition
relies on `uniform.MTI` carrying the **outer (parent-of-instancing-
scope)** model-trafo inverse, so the product `InstanceTrafoInv · MTI`
yields the full inverse chain.

But `pushInstancing` resets `state.model` to `Trafo3d.identity` for
the inside-scope descent, and the leaf-time auto-injection in
`compile.ts:autoInjectedUniforms` derives every trafo uniform from
`state.model`. Inside the scope `state.model` accumulates the
*inner* trafos (`cylinderScale · orient` for our cylinder leaf), so
`uniform.ModelTrafoInv` at the leaf was bound to
`inverse(innerModel)` — **inner-model inverse, not parent-model
inverse**. The shader's normal computation then evaluated to
`inverse(innerModel) · M_inst_inv · n` — innerModel applied twice —
which lost the per-instance rotation contribution and produced the
classic "all gizmos look the same shade" symptom.

Fix: `applyInstancing` overrides every auto-injected trafo uniform
at the leaf with the parent-derived value. Was

```ts
uniformOverrides = uniformOverrides.add("ModelTrafo", parentModel);
```

Now

```ts
uniformOverrides = uniformOverrides
  .add("ModelTrafo",            parentModel)
  .add("ModelTrafoInv",         inv(parentModel))
  .add("ModelViewTrafo",        compose(parentModel, view))
  .add("ModelViewTrafoInv",     inv(compose(parentModel, view)))
  .add("ModelViewProjTrafo",    compose(parentModel, viewProj))
  .add("ModelViewProjTrafoInv", inv(compose(parentModel, viewProj)))
  .add("NormalMatrix",          parentNormalMatrix);
```

That's the entire root-cause fix. Without it, the shader-side
infrastructure (Transpose-vanish removal, simplifyTranspose,
chain-lowering, MulMatVec absorb) was producing algebraically
correct WGSL applied to the *wrong input matrix*.

#### Shader-side infrastructure that landed alongside

Even with the parent-override fix in place, the IR pipeline needed
several improvements to emit clean WGSL for the rebuild:

- **`reverseMatrixOps`: `Transpose` no longer vanishes.** The
  pre-existing rule `Transpose(M) → M.value` was a leaky shortcut —
  fine at a leaf (`uniform.M.transpose()` ≈ uniform var as-is under
  the upload trick), but wrong as soon as the result composed with
  anything (`M.transpose().mul(v)` came out as `M_cpu · v` instead
  of `transpose(M_cpu) · v`). FShade matches: no IR rewrite for
  transpose, just an emit. See `passes/reverseMatrixOps.ts`.
- **`reverseMatrixOps`: `MulMatVec(Transpose(M), v)` /
  `MulVecMat(v, Transpose(M))` peephole-absorb the operand-flip
  into the column-vec form** — `transpose(M).mul(v)` emits as the
  clean column-vec WGSL `M * v` (no `transpose(...)` call).
- **`reverseMatrixOps`: chain-lowering**
  `MulMatVec(MulMatMat(M1, M2), v) → MulMatVec(M1, MulMatVec(M2, v))`
  emits `M1 * (M2 * v)` instead of `(M1 * M2) * v`. 8 dot-products
  instead of 20. CSE has already bound multi-use mat-mats to `Var`s,
  so a literal `MulMatMat` here is single-use; lowering is always
  a win. The post-reverse-form transformation is
  `MulVecMat(MulVecMat(v, M1'), M2')` (= `(v * M1') * M2'`).
- **`passes/simplifyTranspose.ts` (new pass).** Runs immediately
  after `instanceUniforms` (before CSE binds intermediates).
  Handles `T(T(X)) → X`,
  `T(MatrixFromCols([cs])) → MatrixFromRows([cs])`, and the
  symmetric rows-side. Does **not** distribute through `MulMatMat`
  on purpose: `T(A · B)` is a no-op around the eventual `mul(v)`
  thanks to the absorb + chain-lowering above, and distribution
  would force two leaf `transpose(...)` calls in WGSL where none
  are needed.
- **`m44FromCols(InstanceTrafo)` uses `MatrixFromRows` IR** (one of
  two earlier traps that came out in the wash). Each vec4
  attribute holds one *row* of the CPU matrix (offsets 0/16/32/48,
  stride 64). Building the matrix from `MatrixFromRows` survives
  `reverseMatrixOps` as `MatrixFromCols` and emits `mat4x4(rowAttrs)`
  cleanly — the GPU then sees `transpose(M_cpu)`, matching the
  invariant every uniform matrix follows.

#### Final WGSL emit for the per-instance normal transform

`trafo()` reads the world-space normal as
`vec.mul(uniform.ModelTrafoInv)` (row-vec form — eliminates the
redundant `NormalMatrix` uniform; the upload trick on the inverse
matrix gives the inverse-transpose for free). For the instanced
pipeline this becomes

```wgsl
let n4 = (_w_uniform.ModelTrafoInv *
          (mat4x4(InstanceTrafoInv_col0..3) *
           vec4(out.Normals.xyz, 0.0)));
```

Eight scalar dot-products. No `transpose(...)` calls. Right-
associated mat-vec chain. Identical math to the non-instanced
control's `(_w_uniform.ModelTrafoInv * vec(...))`.

## Risks

- Sub-graph dynamism: if an `SgAdaptiveGroup` inside an `SgInstanced` later swaps in a leaf with `instanceCount > 1`, validation runs at scene-compile time only. Document the limitation or re-validate on every delta (TBD).
- Inline-marker effects: their captured uniforms lower to
  `ReadInput("Uniform", name)` just like the parseShader path. The
  rewrite is at the IR level, so the marker / source-string distinction
  doesn't matter. Validated by the phase-1 test corpus.
- Bind-group layout: removing a uniform from the `Uniform` ValueDef
  shrinks the layout. The interface builder + emitter already share a
  slot counter; should compose cleanly. Watch in the phase-1 tests.

## Out of scope (for now)

- Auto-fusing leaves *without* an explicit `SgInstanced` wrapper.
- MDI (different geometries in one call) — text uses MDI but builds it
  manually; auto-MDI is a separate ticket.
- Storage-buffer-backed instance data (would unlock `@builtin(instance_index)`-
  driven random access, but standard instance-step attributes cover the
  90% case).
