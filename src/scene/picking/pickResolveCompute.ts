// Custom MSAA pickId-resolve compute pass.
//
// WebGPU's render-pass `resolveTarget` does a per-sample average,
// which is correct for color but wrong for the rgba32f pickId we
// pack (slot0 is `f32(int pickId)` — averaging 7.0 and 13.0 yields
// 10.0, which decodes to a meaningless id). This compute pass runs
// a per-pixel MAJORITY VOTE on slot-0 and writes the FULL vec4 of
// the winning sample to a single-sample storage texture.
//
// Tie-break: when two values share the maximum count, the one whose
// FIRST occurring sample index is lower wins. This is deterministic
// and matches "lowest sample index" intuition (sample 0 is the
// pixel center for most rasterisers). The sample we copy to the
// output is also that first-occurring sample.
//
// SampleCount is BAKED into the WGSL via string substitution so the
// per-sample loops are bounded constants the compiler can unroll.
// One pipeline is cached per sampleCount.
//
// Workgroup size: 8x8 — small enough to be friendly to GPUs without
// large register files, big enough to hide the K=sampleCount inner
// loops. Pick-resolve cost is O(width*height*sampleCount^2) but
// sampleCount is tiny (2/4/8) and pickId images are at most a few
// MP, so this is dwarfed by the actual draw.

export interface PickResolveCompute {
  /** Encodes a dispatch into the given encoder. */
  resolve(
    encoder: GPUCommandEncoder,
    srcMS: GPUTextureView,
    dst: GPUTextureView,
    width: number,
    height: number,
  ): void;
  dispose(): void;
}

export const PICK_RESOLVE_WORKGROUP_X = 8;
export const PICK_RESOLVE_WORKGROUP_Y = 8;

/** Build the WGSL source for a given sampleCount. Exported for tests. */
export function buildPickResolveWgsl(sampleCount: number): string {
  if (!Number.isInteger(sampleCount) || sampleCount < 2) {
    throw new Error(`buildPickResolveWgsl: sampleCount must be an integer >= 2, got ${sampleCount}`);
  }
  const N = sampleCount;
  return `// Auto-generated MSAA pickId resolve. sampleCount=${N}.
@group(0) @binding(0) var srcMS: texture_multisampled_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba32float, write>;

struct Params {
  width: u32,
  height: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

const SAMPLE_COUNT: u32 = ${N}u;

@compute @workgroup_size(${PICK_RESOLVE_WORKGROUP_X}, ${PICK_RESOLVE_WORKGROUP_Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let p = vec2<i32>(i32(gid.x), i32(gid.y));

  // Load slot-0 (the signed pickId) for every sample.
  var slot0: array<f32, ${N}>;
  for (var i: u32 = 0u; i < SAMPLE_COUNT; i = i + 1u) {
    slot0[i] = textureLoad(srcMS, p, i32(i)).x;
  }

  // Majority vote with first-occurrence tie-break.
  // counts[i] holds the number of later samples whose bitcast<u32>
  // matches slot0[i]; the winner has the highest count, ties broken
  // by lowest sample index.
  var counts: array<u32, ${N}>;
  for (var i: u32 = 0u; i < SAMPLE_COUNT; i = i + 1u) {
    counts[i] = 0u;
  }
  for (var i: u32 = 0u; i < SAMPLE_COUNT; i = i + 1u) {
    let bi = bitcast<u32>(slot0[i]);
    for (var j: u32 = i; j < SAMPLE_COUNT; j = j + 1u) {
      let bj = bitcast<u32>(slot0[j]);
      if (bi == bj) { counts[i] = counts[i] + 1u; }
    }
  }
  // Pick the sample whose value-bucket has the highest tally; on ties
  // take the lowest sample index. We only consider a sample as a
  // candidate if it is the FIRST occurrence of its value (so each
  // distinct value is voted for exactly once), which falls out
  // naturally because counts[i] for a non-first occurrence is < the
  // first occurrence's count by construction (j starts at i).
  // Actually counts[i] above counts only j>=i; the FIRST occurrence
  // therefore has the highest tally for its value. So plain argmax
  // works.
  var winner: u32 = 0u;
  var best: u32 = counts[0];
  for (var i: u32 = 1u; i < SAMPLE_COUNT; i = i + 1u) {
    if (counts[i] > best) { best = counts[i]; winner = i; }
  }

  let outv = textureLoad(srcMS, p, i32(winner));
  textureStore(dst, p, outv);
}
`;
}

/** JS reference impl of the majority vote, used by tests. */
export function majorityVoteReference(
  samples: Float32Array,
  sampleCount: number,
): { winnerIdx: number; winnerValue: number } {
  if (samples.length < sampleCount) {
    throw new Error("majorityVoteReference: samples shorter than sampleCount");
  }
  // Bitcast comparator via Uint32Array view.
  const u32 = new Uint32Array(samples.buffer, samples.byteOffset, sampleCount);
  const counts = new Uint32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) {
    for (let j = i; j < sampleCount; j++) {
      if (u32[i] === u32[j]) counts[i]!++;
    }
  }
  let winner = 0;
  let best = counts[0]!;
  for (let i = 1; i < sampleCount; i++) {
    if (counts[i]! > best) { best = counts[i]!; winner = i; }
  }
  return { winnerIdx: winner, winnerValue: samples[winner]! };
}

interface CacheEntry {
  module: GPUShaderModule;
  pipeline: GPUComputePipeline;
  layout: GPUBindGroupLayout;
}

const moduleCache: WeakMap<GPUDevice, Map<number, CacheEntry>> = new WeakMap();

function getOrBuildEntry(device: GPUDevice, sampleCount: number): CacheEntry {
  let perDevice = moduleCache.get(device);
  if (perDevice === undefined) {
    perDevice = new Map();
    moduleCache.set(device, perDevice);
  }
  const cached = perDevice.get(sampleCount);
  if (cached !== undefined) return cached;

  const code = buildPickResolveWgsl(sampleCount);
  const module = device.createShaderModule({ code, label: `pick.resolve.ms${sampleCount}` });
  const layout = device.createBindGroupLayout({
    label: `pick.resolve.bgl.ms${sampleCount}`,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d", multisampled: true } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "2d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipeline = device.createComputePipeline({
    label: `pick.resolve.pipeline.ms${sampleCount}`,
    layout: pipelineLayout,
    compute: { module, entryPoint: "main" },
  });
  const entry: CacheEntry = { module, pipeline, layout };
  perDevice.set(sampleCount, entry);
  return entry;
}

export function createPickResolveCompute(
  device: GPUDevice,
  sampleCount: number,
): PickResolveCompute {
  const { pipeline, layout } = getOrBuildEntry(device, sampleCount);

  // Per-instance: a small uniform buffer holding (width, height).
  const uniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: "pick.resolve.params",
  });

  return {
    resolve(encoder: GPUCommandEncoder, srcMS: GPUTextureView, dst: GPUTextureView, width: number, height: number): void {
      const w = Math.max(1, width | 0);
      const h = Math.max(1, height | 0);
      device.queue.writeBuffer(uniformBuffer, 0, new Uint32Array([w, h, 0, 0]));
      const bg = device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: srcMS },
          { binding: 1, resource: dst },
          { binding: 2, resource: { buffer: uniformBuffer } },
        ],
      });
      const pass = encoder.beginComputePass({ label: "pick.resolve.pass" });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      const gx = Math.ceil(w / PICK_RESOLVE_WORKGROUP_X);
      const gy = Math.ceil(h / PICK_RESOLVE_WORKGROUP_Y);
      pass.dispatchWorkgroups(gx, gy, 1);
      pass.end();
    },
    dispose(): void {
      try { uniformBuffer.destroy(); } catch { /* already gone */ }
    },
  };
}
