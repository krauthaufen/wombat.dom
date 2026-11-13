// A flush runs outer bindings before inner ones (`Binding.depth`): the binding
// that removes a subtree (a conditional turned false, a removed row) runs
// before the bindings inside it, which it disposes. Otherwise an inner one
// pulls its value once more under the condition that no longer holds
// (FableCube dogfood/poetry.md, "What is inside a conditional is worked out
// once more when the condition turns false": a text `l.Head` threw on the
// emptied list, a fetch went out with the emptied url).

import { describe, it, expect, beforeEach } from "vitest";
import { AdaptiveToken, cval, transact, type aval } from "@aardworx/wombat.adaptive";
import { bindEffect, mount, UIScheduler, type Binding, type Scope } from "@aardworx/wombat.dom";

describe("flush order: outer bindings before inner ones", () => {
  let root: HTMLDivElement;
  let scheduler: UIScheduler;
  let errors: unknown[];
  const origError = console.error;

  beforeEach(() => {
    document.body.innerHTML = "";
    root = document.createElement("div");
    document.body.appendChild(root);
    scheduler = new UIScheduler({ schedule: (cb) => setTimeout(cb, 0) as unknown as number, cancel: (h) => clearTimeout(h) });
    errors = [];
    console.error = (...a: unknown[]) => { errors.push(a); };
    return () => { console.error = origError; };
  });

  it("the scheduler: depth order, stable, and a binding forgotten by an earlier one does not run", () => {
    const ran: string[] = [];
    const inner: Binding = { depth: 2, flush: () => ran.push("inner") };
    const sibling: Binding = { depth: 2, flush: () => ran.push("sibling") };
    const outer: Binding = { depth: 1, flush: () => { ran.push("outer"); scheduler.forget(inner); } };
    // notified inner first
    scheduler.notify(inner);
    scheduler.notify(sibling);
    scheduler.notify(outer);
    scheduler.flushNow();
    expect(ran).toEqual(["outer", "sibling"]);
  });

  it("a binding notified during a flush runs in a later one", () => {
    const ran: string[] = [];
    let again = true;
    const b: Binding = { depth: 0, flush: () => { ran.push("b"); if (again) { again = false; scheduler.notify(b); } } };
    scheduler.notify(b);
    scheduler.flushNow();
    expect(ran).toEqual(["b"]);
    scheduler.flushNow();
    expect(ran).toEqual(["b", "b"]);
  });

  /** A text valid only while the list is not empty; the inner cell subscribed
   *  before the outer one, so it is notified first. */
  function conditional(deferred?: aval<boolean>) {
    const items = cval<string[]>(["21.4M"]);
    const pulls: string[] = [];
    const text = items.map((l) => {
      if (l.length === 0) throw new Error("the text pulled with the list empty");
      pulls.push(l[0]!);
      return l[0] + " DPS";
    });
    const url = items.map((l) => { pulls.push("url " + l.join(",")); return "q?w=" + l.join(","); });
    text.getValue(AdaptiveToken.top);
    url.getValue(AdaptiveToken.top);
    const shown = items.map((l) => l.length > 0);
    const body = shown.map((s) => (s ? <span data-url={url}>{text}</span> : null));
    mount(root, deferred === undefined ? <div>{body}</div> : <div deferUpdates={deferred}><p deferUpdates={deferred}>{body}</p></div>, { scheduler });
    return { items, pulls };
  }

  it("a conditional turned false: nothing inside it is pulled again", () => {
    const { items, pulls } = conditional();
    expect(root.querySelector("span")?.textContent).toBe("21.4M DPS");
    transact(() => { items.value = ["9.1M"]; });
    scheduler.flushNow();
    expect(root.querySelector("span")?.textContent).toBe("9.1M DPS");
    pulls.length = 0;
    transact(() => { items.value = []; });
    scheduler.flushNow();
    expect(root.querySelector("span")).toBeNull();
    expect(pulls).toEqual([]);
    expect(errors).toEqual([]);
  });

  it("the same released from hidden (deferUpdates) subtrees, one inside the other", () => {
    const hidden = cval(false);
    const { items, pulls } = conditional(hidden);
    transact(() => { hidden.value = true; });
    scheduler.flushNow();
    pulls.length = 0;
    transact(() => { items.value = []; });
    scheduler.flushNow();
    transact(() => { hidden.value = false; });
    scheduler.flushNow();
    expect(root.querySelector("span")).toBeNull();
    expect(pulls).toEqual([]);
    expect(errors).toEqual([]);
  });

  it("an effect bound through a ref (bindEffect) does not run when its element goes", () => {
    const w = cval("x");
    const runs: string[] = [];
    const url = w.map((v) => "q?w=" + v);
    url.getValue(AdaptiveToken.top);   // subscribed before the conditional: notified first
    const shown = w.map((v) => v !== "");
    const ref = (_: Element, scope: Scope, sch: UIScheduler) => bindEffect(url, (u) => { runs.push(u); }, scope, sch);
    mount(root, <div>{shown.map((s) => (s ? <span ref={ref} /> : null))}</div>, { scheduler });
    expect(runs).toEqual(["q?w=x"]);
    transact(() => { w.value = "y"; });
    scheduler.flushNow();
    expect(runs).toEqual(["q?w=x", "q?w=y"]);
    transact(() => { w.value = ""; });
    scheduler.flushNow();
    expect(root.querySelector("span")).toBeNull();
    expect(runs).toEqual(["q?w=x", "q?w=y"]);
  });
});
