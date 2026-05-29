// Real-WebGPU helper for browser-mode tests.

export async function requestRealDevice(): Promise<GPUDevice> {
  if (!("gpu" in navigator)) {
    throw new Error("navigator.gpu unavailable — WebGPU not enabled in this browser");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no GPUAdapter available");
  // Surface the picked adapter so CI logs show real-GPU vs SwiftShader.
  // eslint-disable-next-line no-console
  console.log("[webgpu] adapter:", JSON.stringify({
    vendor: adapter.info?.vendor,
    architecture: adapter.info?.architecture,
    device: adapter.info?.device,
    description: adapter.info?.description,
  }));
  return adapter.requestDevice();
}
