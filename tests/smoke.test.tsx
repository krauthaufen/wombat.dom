import { describe, it, expect, beforeEach } from "vitest";
import { cval, cset, clist, transact, AList } from "@aardworx/wombat.adaptive";
import { mount, UIScheduler } from "@aardworx/wombat.adaptive-ui";

describe("adaptive-ui smoke", () => {
  let root: HTMLDivElement;
  let scheduler: UIScheduler;

  beforeEach(() => {
    document.body.innerHTML = "";
    root = document.createElement("div");
    document.body.appendChild(root);
    // synchronous scheduling — flushNow() not needed
    scheduler = new UIScheduler({
      schedule: (cb) => {
        const h = setTimeout(cb, 0);
        return h as unknown as number;
      },
      cancel: (h) => clearTimeout(h),
    });
  });

  it("renders static elements with text", () => {
    mount(root, <h1>hello</h1>, { scheduler });
    expect(root.innerHTML).toBe("<h1>hello</h1>");
  });

  it("binds aval text content and updates on change", () => {
    const name = cval("alice");
    const m = mount(root, <h1>{name}</h1>, { scheduler });
    expect(root.querySelector("h1")?.textContent).toBe("alice");
    transact(() => (name.value = "bob"));
    scheduler.flushNow();
    expect(root.querySelector("h1")?.textContent).toBe("bob");
    m.dispose();
    expect(root.innerHTML).toBe("");
  });

  it("binds aval attribute (class)", () => {
    const cls = cval("a");
    mount(root, <div class={cls}>x</div>, { scheduler });
    expect(root.querySelector("div")?.className).toBe("a");
    transact(() => (cls.value = "b c"));
    scheduler.flushNow();
    expect(root.querySelector("div")?.className).toBe("b c");
  });

  it("binds boolean attribute as toggle", () => {
    const dis = cval(false);
    mount(root, <button disabled={dis}>x</button>, { scheduler });
    const btn = root.querySelector("button")!;
    expect(btn.hasAttribute("disabled")).toBe(false);
    transact(() => (dis.value = true));
    scheduler.flushNow();
    expect(btn.hasAttribute("disabled")).toBe(true);
  });

  it("event listeners fire", () => {
    let clicks = 0;
    mount(root, <button onClick={() => clicks++}>x</button>, { scheduler });
    root.querySelector("button")!.click();
    root.querySelector("button")!.click();
    expect(clicks).toBe(2);
  });

  it("alist children: incremental insert/remove", () => {
    const items = clist(["a", "b", "c"]);
    mount(
      root,
      <ul>{items.map((s) => (<li>{s}</li>) as unknown as never) as unknown as never}</ul>,
      { scheduler },
    );
    const ul = root.querySelector("ul")!;
    const lis = () => Array.from(ul.querySelectorAll("li")).map((li) => li.textContent);
    expect(lis()).toEqual(["a", "b", "c"]);

    transact(() => items.add("d"));
    scheduler.flushNow();
    expect(lis()).toEqual(["a", "b", "c", "d"]);

    transact(() => items.removeAt(0));
    scheduler.flushNow();
    expect(lis()).toEqual(["b", "c", "d"]);

    transact(() => items.prepend("z"));
    scheduler.flushNow();
    expect(lis()).toEqual(["z", "b", "c", "d"]);
  });

  it("nested aval inside alist row updates only that row", () => {
    interface Row { id: number; name: ReturnType<typeof cval<string>> }
    const r1: Row = { id: 1, name: cval("one") };
    const r2: Row = { id: 2, name: cval("two") };
    const items = clist<Row>([r1, r2]);
    mount(
      root,
      <ul>{items.map((it) => (<li data-id={String(it.id)}>{it.name}</li>) as unknown as never) as unknown as never}</ul>,
      { scheduler },
    );
    const find = (id: number) => root.querySelector(`li[data-id="${id}"]`);
    expect(find(1)?.textContent).toBe("one");
    transact(() => (r1.name.value = "ONE"));
    scheduler.flushNow();
    expect(find(1)?.textContent).toBe("ONE");
    expect(find(2)?.textContent).toBe("two");
  });
});
