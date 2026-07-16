// Host-maintained per-id metadata buffer for the argmin pick kernel.
//
// Layout: 2 floats per pickId — `[effectiveRadius, modeSign]` — indexed
// by absolute pickId (slot 0 unused; pickIds start at 1).
//   effectiveRadius < 0  ⇒ scope inactive / noEvents / released / unknown
//                          ⇒ the kernel never lets it win.
//   modeSign = +1 (Mode-A, slot0 = +id) | -1 (Mode-B, slot0 = -id).
//
// The host keeps the CPU mirror current as scopes register / deregister
// and uploads only what changed:
//   - register/deregister mark the id dirty directly.
//   - the three inputs (active / noEvents / pixelSnapRadius) are avals;
//     CONSTANT ones are snapshotted once at register (no ongoing cost),
//     NON-constant ones are tracked and re-checked at flush (a handful).
//   - flush coalesces dirty ids into contiguous runs; once the dirty
//     count crosses `fullUploadFraction · size`, it uploads the whole
//     buffer in one shot instead (cheaper than many small writes).

import { AVal, type aval } from "@aardworx/wombat.adaptive";

import type { LeafPickScope, PickMode } from "./registry.js";
import { PICK_SNAP_RADIUS } from "./pickArgminCompute.js";

export const METADATA_FLOATS_PER_ID = 4; // [effR, modeSign, priority, pad]

/** Clamped pick priority; default 0 when the scope never set one. */
export function effectivePriority(p: number | undefined): number {
  if (p === undefined || !isFinite(p)) return 0;
  return Math.max(-8, Math.min(7, Math.floor(p)));
}

/** Effective snap radius, folding active/noEvents into the radius test. */
export function effectiveRadius(active: boolean, noEvents: boolean, radius: number): number {
  if (!active || noEvents) return -1;
  if (radius < 0) return 0;
  return radius > PICK_SNAP_RADIUS ? PICK_SNAP_RADIUS : radius;
}

/**
 * Plan the writeBuffer calls for a set of dirty ids. Returns contiguous
 * `[startId, countIds)` runs, or a single full-buffer run once the dirty
 * fraction crosses `fullUploadFraction`. Pure — unit-tested directly.
 */
export function planMetadataUploads(
  dirtyIds: ReadonlyArray<number>,
  size: number,
  fullUploadFraction: number,
): Array<{ startId: number; countIds: number }> {
  if (dirtyIds.length === 0) return [];
  if (dirtyIds.length >= fullUploadFraction * size) {
    return [{ startId: 0, countIds: size }];
  }
  const sorted = [...dirtyIds].sort((a, b) => a - b);
  const runs: Array<{ startId: number; countIds: number }> = [];
  let runStart = sorted[0]!;
  let prev = sorted[0]!;
  for (let i = 1; i < sorted.length; i++) {
    const id = sorted[i]!;
    if (id === prev + 1) {
      prev = id;
    } else {
      runs.push({ startId: runStart, countIds: prev - runStart + 1 });
      runStart = id;
      prev = id;
    }
  }
  runs.push({ startId: runStart, countIds: prev - runStart + 1 });
  return runs;
}

interface Entry {
  readonly scope: LeafPickScope;
  readonly modeSign: number;
  /** ids whose active/noEvents/pixelSnapRadius is non-constant get re-checked. */
  readonly dynamic: boolean;
}

export class PickMetadata {
  private readonly device: GPUDevice;
  private readonly fullUploadFraction: number;
  private data: Float32Array;
  private capacityIds: number; // number of id slots (== data.length / 2)
  private gpu: GPUBuffer;
  private readonly entries = new Map<number, Entry>();
  private readonly dynamicIds = new Set<number>();
  private readonly dirty = new Set<number>();
  private maxId = 0; // highest id ever registered; the kernel reads [0, maxId]

  constructor(device: GPUDevice, opts?: { initialIds?: number; fullUploadFraction?: number }) {
    this.device = device;
    this.fullUploadFraction = opts?.fullUploadFraction ?? 0.5;
    this.capacityIds = Math.max(64, opts?.initialIds ?? 256);
    this.data = new Float32Array(this.capacityIds * METADATA_FLOATS_PER_ID);
    // Slot 0 and all unused slots default to "invalid" so a stray id
    // never matches before it's been registered.
    this.data.fill(-1);
    this.gpu = this.allocBuffer(this.capacityIds);
    this.dirty.add(0);
  }

  /** The storage buffer the kernel binds at group(0) binding(2). */
  get buffer(): GPUBuffer { return this.gpu; }
  /** Highest registered id; sizing hint for callers. */
  get highId(): number { return this.maxId; }

  private allocBuffer(capacityIds: number): GPUBuffer {
    return this.device.createBuffer({
      size: capacityIds * METADATA_FLOATS_PER_ID * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "pick.metadata",
    });
  }

  private ensureCapacity(id: number): void {
    if (id < this.capacityIds) return;
    let next = this.capacityIds;
    while (id >= next) next *= 2;
    const grown = new Float32Array(next * METADATA_FLOATS_PER_ID);
    grown.fill(-1);
    grown.set(this.data, 0);
    this.data = grown;
    this.capacityIds = next;
    try { this.gpu.destroy(); } catch { /* gone */ }
    this.gpu = this.allocBuffer(next);
    // Whole buffer is fresh GPU memory → re-upload everything next flush.
    this.dirty.clear();
    for (let i = 0; i <= this.maxId; i++) this.dirty.add(i);
  }

  private writeEntry(id: number, effR: number, modeSign: number, priority: number): void {
    const base = id * METADATA_FLOATS_PER_ID;
    this.data[base] = effR;
    this.data[base + 1] = modeSign;
    this.data[base + 2] = priority;
    this.dirty.add(id);
  }

  private isConst(a: aval<unknown> | undefined): boolean {
    return a === undefined || a.isConstant;
  }

  private computeEffR(scope: LeafPickScope): number {
    const active = AVal.force(scope.active);
    const noEvents = scope.noEvents !== undefined && AVal.force(scope.noEvents);
    const radius = AVal.force(scope.pixelSnapRadius);
    return effectiveRadius(active, noEvents, radius);
  }

  private computePrio(scope: LeafPickScope): number {
    return effectivePriority(scope.pickPriority !== undefined ? AVal.force(scope.pickPriority) : 0);
  }

  /** Register (or re-register) a scope's metadata. */
  register(scope: LeafPickScope, mode: PickMode): void {
    const id = scope.pickId;
    this.ensureCapacity(id);
    if (id > this.maxId) this.maxId = id;
    const modeSign = mode === "A" ? 1 : -1;
    const dynamic =
      !this.isConst(scope.active) ||
      !this.isConst(scope.noEvents) ||
      !this.isConst(scope.pixelSnapRadius) ||
      !this.isConst(scope.pickPriority);
    this.entries.set(id, { scope, modeSign, dynamic });
    if (dynamic) this.dynamicIds.add(id); else this.dynamicIds.delete(id);
    this.writeEntry(id, this.computeEffR(scope), modeSign, this.computePrio(scope));
  }

  /** Drop a scope — its id becomes permanently invalid until re-registered. */
  deregister(id: number): void {
    if (!this.entries.has(id)) return;
    this.entries.delete(id);
    this.dynamicIds.delete(id);
    this.writeEntry(id, -1, this.data[id * METADATA_FLOATS_PER_ID + 1] ?? 1, 0);
  }

  clear(): void {
    this.entries.clear();
    this.dynamicIds.clear();
    this.data.fill(-1);
    this.dirty.clear();
    for (let i = 0; i <= this.maxId; i++) this.dirty.add(i);
    this.maxId = 0;
  }

  /** Re-check the non-constant ids; mark any whose effective radius moved. */
  private refreshDynamic(): void {
    for (const id of this.dynamicIds) {
      const e = this.entries.get(id);
      if (e === undefined) continue;
      const effR = this.computeEffR(e.scope);
      const prio = this.computePrio(e.scope);
      if (this.data[id * METADATA_FLOATS_PER_ID] !== effR
          || this.data[id * METADATA_FLOATS_PER_ID + 2] !== prio) {
        this.writeEntry(id, effR, e.modeSign, prio);
      }
    }
  }

  /** Upload pending changes. Call once before encoding the argmin dispatch. */
  flush(): void {
    this.refreshDynamic();
    if (this.dirty.size === 0) return;
    const runs = planMetadataUploads([...this.dirty], this.maxId + 1, this.fullUploadFraction);
    for (const { startId, countIds } of runs) {
      const byteOffset = startId * METADATA_FLOATS_PER_ID * 4;
      const floatOffset = startId * METADATA_FLOATS_PER_ID;
      const floatCount = countIds * METADATA_FLOATS_PER_ID;
      this.device.queue.writeBuffer(
        this.gpu, byteOffset, this.data.buffer, this.data.byteOffset + floatOffset * 4, floatCount * 4,
      );
    }
    this.dirty.clear();
  }

  dispose(): void {
    try { this.gpu.destroy(); } catch { /* gone */ }
  }
}
