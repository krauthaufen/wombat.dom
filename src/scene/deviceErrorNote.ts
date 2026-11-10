// A note over the canvas when the GPU device reports an error.
//
// wombat.rendering logs each device error (`console.error`) and makes them
// observable (`onDeviceError`); a page whose console nobody sees (a phone, a
// laptop of a user reporting "it renders nothing") shows nothing at all. The
// render control therefore puts a small note in the canvas's corner on the
// first error: its kind and first line, how many there were, whether the
// renderer recovered. Click to dismiss (it comes back with the next error).

import { onDeviceError, type DeviceError } from "@aardworx/wombat.rendering";

export function deviceErrorNote(device: GPUDevice, canvas: HTMLCanvasElement): { dispose(): void } {
  let note: HTMLDivElement | undefined;
  let count = 0;
  let first: DeviceError | undefined;
  let lastFatal: DeviceError | undefined;
  const show = (e: DeviceError): void => {
    count++;
    first ??= e;
    if (!e.recovered) lastFatal = e;
    const parent = canvas.parentElement;
    if (parent === null) return;
    if (note === undefined || note.parentElement !== parent) {
      note = document.createElement("div");
      note.setAttribute("data-wombat-device-error", "");
      note.title = "GPU device error (click to hide); details in the console";
      Object.assign(note.style, {
        position: "absolute", zIndex: "1000", maxWidth: "60%",
        padding: "4px 8px", borderRadius: "4px",
        font: "12px/1.35 system-ui, sans-serif", color: "#fff",
        pointerEvents: "auto", cursor: "pointer", whiteSpace: "pre-wrap",
      } satisfies Partial<CSSStyleDeclaration>);
      note.addEventListener("click", () => { note?.remove(); });
      parent.appendChild(note);
    }
    // the canvas's bottom-left corner, in the coordinates of its offset parent
    note.style.left = `${canvas.offsetLeft + 8}px`;
    note.style.top = `${canvas.offsetTop + Math.max(0, canvas.offsetHeight - 8)}px`;
    note.style.transform = "translateY(-100%)";
    const shown = lastFatal ?? first!;
    note.style.background = lastFatal !== undefined ? "rgba(160, 20, 20, 0.9)" : "rgba(150, 95, 0, 0.9)";
    note.textContent =
      `GPU ${lastFatal !== undefined ? "error" : "warning"}${count > 1 ? ` (${count})` : ""}: ` +
      `${shown.kind}${shown.recovered ? ", recovered" : ""}: ${shown.message.split("\n")[0]!.slice(0, 240)}`;
    note.setAttribute("data-count", String(count));
  };
  const sub = onDeviceError(device, show);
  return { dispose: () => { sub.dispose(); note?.remove(); } };
}
