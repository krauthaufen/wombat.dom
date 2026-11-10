// The note over the canvas for GPU device errors (deviceErrorNote): an
// uncaptured device error shows it (kind, first line, count), a recovered one
// (wombat.rendering's atlas out of memory) too, as a warning; dispose removes it.

import { describe, expect, it, vi } from "vitest";
import { reportDeviceError, watchDeviceErrors } from "@aardworx/wombat.rendering";
import { deviceErrorNote } from "../src/scene/deviceErrorNote.js";

function fakeDevice(): GPUDevice & EventTarget {
  return new EventTarget() as unknown as GPUDevice & EventTarget;
}

describe("deviceErrorNote", () => {
  it("shows device errors over the canvas, counts them, goes on dispose", () => {
    const errors = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      const host = document.createElement("div");
      const canvas = document.createElement("canvas");
      host.appendChild(canvas);
      document.body.appendChild(host);
      const device = fakeDevice();
      watchDeviceErrors(device);
      const note = deviceErrorNote(device, canvas);
      expect(host.querySelector("[data-wombat-device-error]")).toBeNull();

      reportDeviceError(device, { kind: "out-of-memory", message: "texture atlas rgba8unorm: could not allocate\nmore", recovered: true });
      const el = host.querySelector("[data-wombat-device-error]") as HTMLElement;
      expect(el.textContent).toBe("GPU warning: out-of-memory, recovered: texture atlas rgba8unorm: could not allocate");

      const ev = new Event("uncapturederror");
      (ev as unknown as { error: unknown }).error = { message: "Invalid BindGroup" };
      device.dispatchEvent(ev);
      // the first error stays the one shown (later ones are often its consequences); red now
      expect(el.textContent).toBe("GPU error (2): out-of-memory, recovered: texture atlas rgba8unorm: could not allocate");
      expect(el.getAttribute("data-count")).toBe("2");
      expect(errors).toHaveBeenCalledTimes(2);
      expect(errors).toHaveBeenLastCalledWith("[wombat] GPU device error (unknown): Invalid BindGroup");

      note.dispose();
      expect(host.querySelector("[data-wombat-device-error]")).toBeNull();
      host.remove();
    } finally { errors.mockRestore(); }
  });
});
