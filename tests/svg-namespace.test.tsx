// SVG elements: an <svg> and what is inside it are made in the SVG
// namespace (an HTML element named "path" draws nothing); the HTML content
// of a <foreignObject> is HTML again. Attributes keep their case (viewBox).

import { describe, it, expect, beforeEach } from "vitest";
import { cval, clist, transact, AList } from "@aardworx/wombat.adaptive";
import { mount, UIScheduler } from "@aardworx/wombat.dom";

const SVG = "http://www.w3.org/2000/svg";
const HTML = "http://www.w3.org/1999/xhtml";

describe("SVG namespace", () => {
  let root: HTMLDivElement;
  let scheduler: UIScheduler;
  const flush = () => new Promise((r) => setTimeout(r, 5));

  beforeEach(() => {
    document.body.innerHTML = "";
    root = document.createElement("div");
    document.body.appendChild(root);
    scheduler = new UIScheduler({
      schedule: (cb) => setTimeout(cb, 0) as unknown as number,
      cancel: (h) => clearTimeout(h),
    });
  });

  it("an svg and its descendants are SVG elements; HTML around them stays HTML", () => {
    mount(root, (
      <div>
        <svg viewBox="0 0 10 10" width="100">
          <g class="c"><path d="M 0 0 L 10 10" stroke="red" /><circle cx="5" cy="5" r="2" /></g>
          <foreignObject width="10" height="10"><span>t</span></foreignObject>
        </svg>
        <p>after</p>
      </div>
    ), { scheduler });
    const ns = (sel: string) => root.querySelector(sel)!.namespaceURI;
    expect(ns("div")).toBe(HTML);
    expect(ns("svg")).toBe(SVG);
    expect(ns("g")).toBe(SVG);
    expect(ns("path")).toBe(SVG);
    expect(ns("circle")).toBe(SVG);
    expect(ns("foreignObject")).toBe(SVG);
    expect(ns("span")).toBe(HTML);
    expect(ns("p")).toBe(HTML);
    expect(root.querySelector("svg")!.getAttribute("viewBox")).toBe("0 0 10 10");
    expect(root.querySelector("g")!.getAttribute("class")).toBe("c");
    expect(root.querySelector("path")!.getAttribute("d")).toBe("M 0 0 L 10 10");
  });

  it("children added later (alist) and changing attributes (aval) inside an svg", async () => {
    const pts = clist([1, 2]);
    const r = cval(3);
    mount(root, (
      <svg>{AList.map((x: number) => <circle cx={String(x)} r={r} />, pts)}</svg>
    ), { scheduler });
    transact(() => { pts.add(5); r.value = 4; });
    await flush();
    const cs = Array.from(root.querySelectorAll("circle"));
    expect(cs.length).toBe(3);
    for (const c of cs) { expect(c.namespaceURI).toBe(SVG); expect(c.getAttribute("r")).toBe("4"); }
  });
});
