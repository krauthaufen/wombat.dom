// `<RenderControl>` smoke tests under happy-dom.
//
// happy-dom has no WebGPU. The component renders a canvas element
// synchronously; GPU init is async and gated on `navigator.gpu`,
// which is undefined here. We verify:
//   - the canvas DOM element is mounted with the expected attributes;
//   - mount disposal removes the canvas without throwing;
//   - sniffViewProj behaviour is exercised at the data layer.
//
// Real-GPU coverage lives downstream in a browser-mode test once
// the example app is set up — same pattern as wombat.rendering's
// tests-browser/.

import { describe, expect, it } from "vitest";
import { effect } from "@aardworx/wombat.shader";
import { mount, useScope } from "../src/index.js";
import {
  collectSgChildren, DefaultSurfaces, RenderControl, Sg,
} from "../src/scene/index.js";

describe("RenderControl — DOM", () => {
  it("mounts a canvas element with passed-through HTML props", () => {
    const root = document.createElement("div");
    document.body.appendChild(root);

    const handle = mount(
      root,
      <RenderControl
        scene={Sg.empty}
        id="rc"
        class="canvas-thing"
        width={320}
        height={240}
      />,
    );

    const c = root.querySelector("canvas")!;
    expect(c).not.toBeNull();
    expect(c.id).toBe("rc");
    expect(c.getAttribute("class")).toBe("canvas-thing");
    expect(c.getAttribute("width")).toBe("320");
    expect(c.getAttribute("height")).toBe("240");

    handle.dispose();
    expect(root.querySelector("canvas")).toBeNull();
  });

  it("dispose is safe when GPU init never starts (no navigator.gpu)", () => {
    const root = document.createElement("div");
    document.body.appendChild(root);
    expect(() => {
      const handle = mount(root, <RenderControl scene={Sg.empty}/>);
      handle.dispose();
    }).not.toThrow();
  });

  it("useScope() returns the active component scope", () => {
    let captured: ReturnType<typeof useScope> | undefined;
    function Probe(): null {
      captured = useScope();
      return null;
    }
    const root = document.createElement("div");
    const handle = mount(root, <Probe/>);
    expect(captured).toBeDefined();
    expect(captured!.isDisposed).toBe(false);
    handle.dispose();
    expect(captured!.isDisposed).toBe(true);
  });

  it("useScope() throws when called outside a component body", () => {
    expect(() => useScope()).toThrow(/outside a component/);
  });
});

describe("Sg JSX wrappers", () => {
  it("<Sg.Box/> wraps the box leaf in a tagged Fragment", () => {
    const got = collectSgChildren(<Sg.Box/>);
    // Sg.Box auto-wires an Intersectable scope around the leaf.
    expect(got.kind).toBe("Intersectable");
    if (got.kind === "Intersectable") {
      expect(got.child.kind).toBe("Leaf");
    }
  });

  it("<Sg Trafo Shader>...</Sg> wraps children with attribute scopes (innermost-out)", async () => {
    const { V3d, Trafo3d } = await import("@aardworx/wombat.base");
    const node = collectSgChildren(
      <Sg
        Trafo={[Trafo3d.translation(new V3d(1, 0, 0))]}
        Shader={effect(DefaultSurfaces.trafo(), DefaultSurfaces.vertexColor())}
      >
        <Sg.Box/>
      </Sg>,
    );
    // Outermost wrapper is Trafo (per JSX rule: leftmost outermost,
    // applied last to a point).
    expect(node.kind).toBe("Trafo");
    if (node.kind === "Trafo") {
      expect(node.child.kind).toBe("Shader");
      if (node.child.kind === "Shader") {
        // Sg.Box auto-wires an Intersectable scope around the leaf.
        expect(node.child.child.kind).toBe("Intersectable");
        if (node.child.child.kind === "Intersectable") {
          expect(node.child.child.child.kind).toBe("Leaf");
        }
      }
    }
  });

  it("multiple JSX children become an Sg.group", () => {
    const node = collectSgChildren(
      <>
        <Sg.Box/>
        <Sg.Quad/>
      </>,
    );
    expect(node.kind).toBe("Group");
  });

  it("an array of children flattens into a Group", () => {
    const arr = [<Sg.Box/>, <Sg.Quad/>];
    const node = collectSgChildren(arr);
    expect(node.kind).toBe("Group");
  });

  it("RenderControl accepts JSX children scenes", () => {
    const root = document.createElement("div");
    const handle = mount(
      root,
      <RenderControl>
        <Sg Shader={effect(DefaultSurfaces.trafo(), DefaultSurfaces.vertexColor())}>
          <Sg.Box/>
        </Sg>
      </RenderControl>,
    );
    expect(root.querySelector("canvas")).not.toBeNull();
    handle.dispose();
  });
});
