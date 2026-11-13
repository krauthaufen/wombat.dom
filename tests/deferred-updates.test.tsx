// deferUpdates: a hidden subtree's bindings wait and catch up once when shown
// (docs/deferred-updates.md).

import { describe, it, expect, beforeEach } from "vitest";
import { cval, clist, transact, AList, AVal } from "@aardworx/wombat.adaptive";
import { mount, UIScheduler } from "@aardworx/wombat.dom";

describe("deferUpdates", () => {
  let root: HTMLDivElement;
  let scheduler: UIScheduler;

  beforeEach(() => {
    document.body.innerHTML = "";
    root = document.createElement("div");
    document.body.appendChild(root);
    scheduler = new UIScheduler({
      schedule: (cb) => setTimeout(cb, 0) as unknown as number,
      cancel: (h) => clearTimeout(h),
    });
  });

  const items = (): string[] => Array.from(root.querySelectorAll("li")).map((e) => e.textContent ?? "");

  it("rows inserted and removed while hidden are applied on show, as the net change", () => {
    const hidden = cval(true);
    const list = clist(["a", "b"]);
    mount(root, <ul deferUpdates={hidden}>{AList.map((x: string) => <li>{x}</li>, list)}</ul>, { scheduler });
    expect(items()).toEqual(["a", "b"]);
    transact(() => { list.add("c"); list.removeAt(0); list.add("x"); });
    transact(() => { list.removeAt(2); }); // "x" came and went while hidden
    scheduler.flushNow();
    expect(items()).toEqual(["a", "b"]); // waiting
    transact(() => { hidden.value = false; });
    scheduler.flushNow();
    expect(items()).toEqual(["b", "c"]);
  });

  it("a text and an attribute catch up; the element's own attributes stay live", () => {
    const hidden = cval(true);
    const text = cval("one");
    const title = cval("t1");
    const cls = cval("pane");
    mount(root, <div class={cls} deferUpdates={hidden}><span title={title}>{text}</span></div>, { scheduler });
    transact(() => { text.value = "two"; title.value = "t2"; cls.value = "pane active"; });
    scheduler.flushNow();
    const div = root.querySelector("div")!, span = root.querySelector("span")!;
    expect(div.className).toBe("pane active"); // own attribute: live
    expect(span.textContent).toBe("one");
    expect(span.getAttribute("title")).toBe("t1");
    transact(() => { hidden.value = false; });
    scheduler.flushNow();
    expect(span.textContent).toBe("two");
    expect(span.getAttribute("title")).toBe("t2");
    expect(root.querySelector("span")).toBe(span); // the same node
    expect(div.hasAttribute("deferUpdates") || div.hasAttribute("deferupdates")).toBe(false);
  });

  it("a shown subtree updates as before", () => {
    const text = cval("one");
    mount(root, <div deferUpdates={AVal.constant(false)}><span>{text}</span></div>, { scheduler });
    transact(() => { text.value = "two"; });
    scheduler.flushNow();
    expect(root.querySelector("span")!.textContent).toBe("two");
  });

  it("rows that did not change keep their nodes across a catch-up", () => {
    const hidden = cval(true);
    const list = clist(["a", "b", "c"]);
    mount(root, <ul deferUpdates={hidden}>{AList.map((x: string) => <li>{x}</li>, list)}</ul>, { scheduler });
    const [a, , c] = Array.from(root.querySelectorAll("li"));
    transact(() => { list.removeAt(1); });
    transact(() => { hidden.value = false; });
    scheduler.flushNow();
    const after = Array.from(root.querySelectorAll("li"));
    expect(after.map((e) => e.textContent)).toEqual(["a", "c"]);
    expect(after[0]).toBe(a);
    expect(after[1]).toBe(c);
  });

  it("a binding disposed while waiting never flushes", () => {
    const hidden = cval(true);
    const text = cval("one");
    const present = cval(true);
    mount(root, <div>{present.map((p) => (p ? <section deferUpdates={hidden}><span>{text}</span></section> : <i>gone</i>))}</div>, { scheduler });
    transact(() => { text.value = "two"; });
    scheduler.flushNow();
    transact(() => { present.value = false; });
    scheduler.flushNow();
    transact(() => { hidden.value = false; });
    expect(() => scheduler.flushNow()).not.toThrow();
    expect(root.querySelector("span")).toBe(null);
    expect(root.querySelector("i")!.textContent).toBe("gone");
  });

  it("nested: the inner subtree waits for the outer one too", () => {
    const outer = cval(true);
    const inner = cval(true);
    const text = cval("one");
    mount(root, <div deferUpdates={outer}><p deferUpdates={inner}><span>{text}</span></p></div>, { scheduler });
    transact(() => { text.value = "two"; });
    transact(() => { inner.value = false; });
    scheduler.flushNow();
    expect(root.querySelector("span")!.textContent).toBe("one"); // outer still hidden
    transact(() => { outer.value = false; });
    scheduler.flushNow();
    expect(root.querySelector("span")!.textContent).toBe("two");
  });
});

// A large insertion is built over several frames once a flush has used its
// budget; removals never wait; `flushNow` and the first mount finish at once.
describe("alist rows: a large insertion is spread over frames", () => {
  let root: HTMLDivElement;
  const frames: (() => void)[] = [];
  let clock = 0;
  class Sched extends UIScheduler {
    constructor() { super({ schedule: (cb) => frames.push(cb), cancel: () => { frames.length = 0; } }); }
    // every look at the clock costs a millisecond: 8 rows per look
    override now(): number { return clock++; }
  }
  const frame = (): void => { const f = frames.splice(0); for (const cb of f) cb(); };
  const items = (): string[] => Array.from(root.querySelectorAll("li")).map((e) => e.textContent ?? "");
  const names = (n: number, from = 0): string[] => Array.from({ length: n }, (_, i) => `r${from + i}`);

  beforeEach(() => {
    document.body.innerHTML = "";
    root = document.createElement("div");
    document.body.appendChild(root);
    frames.length = 0; clock = 0;
  });

  it("the first mount is whole; a small update is applied whole", () => {
    const scheduler = new Sched();
    const list = clist(names(1000));
    mount(root, <ul>{AList.map((x: string) => <li>{x}</li>, list)}</ul>, { scheduler });
    expect(items().length).toBe(1000);
    transact(() => { list.add("x"); list.removeAt(0); });
    frame();
    expect(items().length).toBe(1000);
    expect(items()[999]).toBe("x");
    expect(frames.length).toBe(0);
  });

  it("many new rows come in list order over several frames, the first at once", () => {
    const scheduler = new Sched();
    const list = clist<string>(["first"]);
    mount(root, <ul>{AList.map((x: string) => <li>{x}</li>, list)}</ul>, { scheduler });
    transact(() => { for (const n of names(1000)) list.add(n); });
    frame();
    const after1 = items();
    expect(after1.length).toBeGreaterThan(8);
    expect(after1.length).toBeLessThan(1001);
    expect(after1).toEqual(["first", ...names(after1.length - 1)]);   // a prefix, in order
    let n = 0;
    while (frames.length > 0 && n++ < 100) frame();
    expect(n).toBeGreaterThan(1);
    expect(items()).toEqual(["first", ...names(1000)]);
  });

  it("removals and later changes while rows wait give the list's final state", () => {
    const scheduler = new Sched();
    const list = clist<string>([]);
    mount(root, <ul>{AList.map((x: string) => <li>{x}</li>, list)}</ul>, { scheduler });
    transact(() => { for (const n of names(1000)) list.add(n); });
    frame();
    expect(items().length).toBeLessThan(1000);
    // a built row and a waiting one go, one is added in front, one at the end
    transact(() => { list.removeAt(999); list.removeAt(0); list.insertAt(0, "front"); list.add("end"); });
    frame();
    expect(items()[0]).not.toBe("r0");       // the removal did not wait
    while (frames.length > 0) frame();
    expect(items()).toEqual(["front", ...names(998, 1), "end"]);
  });

  it("flushNow leaves nothing waiting", () => {
    const scheduler = new Sched();
    const list = clist<string>([]);
    mount(root, <ul>{AList.map((x: string) => <li>{x}</li>, list)}</ul>, { scheduler });
    transact(() => { for (const n of names(1000)) list.add(n); });
    scheduler.flushNow();
    expect(items()).toEqual(names(1000));
    expect(frames.length).toBe(0);
  });

  it("a hidden list catches up over frames when shown", () => {
    const scheduler = new Sched();
    const hidden = cval(true);
    const list = clist<string>([]);
    mount(root, <ul deferUpdates={hidden}>{AList.map((x: string) => <li>{x}</li>, list)}</ul>, { scheduler });
    transact(() => { for (const n of names(1000)) list.add(n); });
    frame();
    expect(items().length).toBe(0);
    transact(() => { hidden.value = false; });
    frame();
    expect(items().length).toBeGreaterThan(8);
    expect(items().length).toBeLessThan(1000);
    while (frames.length > 0) frame();
    expect(items()).toEqual(names(1000));
  });
});
