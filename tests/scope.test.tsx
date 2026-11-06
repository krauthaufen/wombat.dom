import { describe, it, expect } from "vitest";
import { cval, transact } from "@aardworx/wombat.adaptive";
import { mount, UIScheduler, Scope } from "@aardworx/wombat.dom";

describe("scopes", () => {
  it("a ref gets a scope that ends with its element", () => {
    document.body.innerHTML = "";
    const root = document.createElement("div");
    document.body.appendChild(root);
    const scheduler = new UIScheduler({ schedule: (cb) => setTimeout(cb, 0) as unknown as number, cancel: (h) => clearTimeout(h) });
    let ended = 0;
    const ref = (_el: Element, s: Scope) => s.onDispose(() => ended++);
    const show = cval(true);
    const m = mount(root, <div>{show.map((b: boolean) => (b ? <span ref={ref}>a</span> : <b>b</b>))}</div>, { scheduler });
    expect(ended).toBe(0);
    transact(() => (show.value = false));
    scheduler.flushNow();
    expect(root.querySelector("b")).not.toBeNull();
    expect(ended).toBe(1);
    m.dispose();
    expect(ended).toBe(1);
  });

  it("a child disposed on its own leaves its parent", () => {
    const p = new Scope();
    const kids = Array.from({ length: 5 }, () => p.child());
    let ended = 0;
    kids.forEach((k) => k.onDispose(() => ended++));
    kids[1]!.dispose();
    kids[3]!.dispose();
    expect(ended).toBe(2);
    expect((p as unknown as { _disposables: unknown[] })._disposables.length).toBe(3);
    p.dispose();
    expect(ended).toBe(5);
  });
});
