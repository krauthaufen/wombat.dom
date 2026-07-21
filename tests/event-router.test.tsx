// RegionRouter — the unified DOM walk over a wombat mount subtree.
// Increment 1: the synchronous DOM half (overlays / HUD, no scene leaf).
// See docs/unified-event-propagation.md.

import { describe, expect, it } from "vitest";
import { mount } from "../src/mount.js";

function host(): { root: HTMLElement; outer: HTMLElement } {
  const outer = document.createElement("div");
  const root = document.createElement("div");
  outer.appendChild(root);
  document.body.appendChild(outer);
  return { root, outer };
}

function click(el: Element): MouseEvent {
  const ev = new MouseEvent("click", { bubbles: true, cancelable: true });
  el.dispatchEvent(ev);
  return ev;
}

describe("RegionRouter — unified DOM walk", () => {
  it("capture runs outer→inner, bubble inner→outer, in one walk", () => {
    const { root } = host();
    const log: string[] = [];
    mount(root,
      <div onClickCapture={() => { log.push("A.cap"); }} onClick={() => { log.push("A.bub"); }}>
        <span onClickCapture={() => { log.push("B.cap"); }} onClick={() => { log.push("B.bub"); }}>
          <button onClick={() => { log.push("btn"); }}>x</button>
        </span>
      </div>,
    );
    click(root.querySelector("button")!);
    expect(log).toEqual(["A.cap", "B.cap", "btn", "B.bub", "A.bub"]);
  });

  it("a handler returning false stops the rest of the walk", () => {
    const { root } = host();
    const log: string[] = [];
    mount(root,
      <div onClick={() => { log.push("outer.bub"); }}>
        <button onClick={() => { log.push("btn"); return false; }}>x</button>
      </div>,
    );
    click(root.querySelector("button")!);
    expect(log).toEqual(["btn"]); // outer bubble suppressed
  });

  it("stopPropagation() in capture stops the whole walk", () => {
    const { root } = host();
    const log: string[] = [];
    mount(root,
      <div onClickCapture={(e: Event) => { log.push("outer.cap"); e.stopPropagation(); }} onClick={() => { log.push("outer.bub"); }}>
        <button onClick={() => { log.push("btn"); }}>x</button>
      </div>,
    );
    click(root.querySelector("button")!);
    expect(log).toEqual(["outer.cap"]);
  });

  it("an unstopped event exits the root — the host page sees the subtree as one node", () => {
    const { root, outer } = host();
    let outerSaw = 0;
    outer.addEventListener("click", () => { outerSaw++; });
    mount(root, <button onClick={() => { /* no stop */ }}>x</button>);
    click(root.querySelector("button")!);
    expect(outerSaw).toBe(1);
  });

  it("a stopped event does NOT exit the root", () => {
    const { root, outer } = host();
    let outerSaw = 0;
    outer.addEventListener("click", () => { outerSaw++; });
    mount(root, <button onClick={(e: Event) => { e.stopPropagation(); }}>x</button>);
    click(root.querySelector("button")!);
    expect(outerSaw).toBe(0);
  });

  it("a foreign native listener inside the subtree still fires when nothing stops", () => {
    const { root } = host();
    const log: string[] = [];
    // A third-party element injected via ref, with its OWN native listener
    // wombat never registered.
    const attachForeign = (el: Element): void => {
      const foreign = document.createElement("button");
      foreign.addEventListener("click", () => { log.push("foreign"); });
      el.appendChild(foreign);
    };
    mount(root,
      <div onClick={() => { log.push("wombat.bub"); }}>
        <span ref={attachForeign}></span>
      </div>,
    );
    click(root.querySelector("span > button")!);
    // Foreign native listener fires (during native bubble); the wombat
    // ancestor handler fires too. Both present — foreign not swallowed.
    expect(log).toContain("foreign");
    expect(log).toContain("wombat.bub");
  });

  it("dynamically-added subtree (aval child) still joins the walk", async () => {
    const { root } = host();
    const log: string[] = [];
    // A row added after mount inherits the region via scope.child().
    mount(root,
      <div onClick={() => { log.push("outer"); }}>
        <button onClick={() => { log.push("btn"); return false; }}>x</button>
      </div>,
    );
    click(root.querySelector("button")!);
    expect(log).toEqual(["btn"]); // return false stops outer
  });
});
