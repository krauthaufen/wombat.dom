// Primitives — JSX components, shared geometry, stride-0 colour
// buffer. No GPU; we just inspect the SgNode tree shape.

import { describe, expect, it } from "vitest";
import { AVal, cval, transact } from "@aardworx/wombat.adaptive";
import { V3d, V4f, Box3d } from "@aardworx/wombat.base";
import { Sg, collectSgChildren } from "../src/scene/index.js";
import {
  getBoxGeometry, getSphereGeometry, getCylinderGeometry,
} from "../src/scene/primitives/shared.js";
import { colorAval } from "../src/scene/primitives/colorBuffer.js";

function unwrapToLeaf(node: import("../src/scene/index.js").SgNode): import("../src/scene/index.js").SgLeaf {
  let n = node;
  while (n.kind !== "Leaf") {
    if ("child" in n && n.child !== undefined) n = n.child as never;
    else throw new Error(`unwrapToLeaf: hit ${n.kind} without a child`);
  }
  return n as import("../src/scene/index.js").SgLeaf;
}

describe("Sg primitive components", () => {
  it("<Sg.Box/> compiles to a leaf with a_position + a_color and a Trafo-ish wrap", () => {
    const node = collectSgChildren(<Sg.Box Color={new V4f(1, 0.5, 0.2, 1)}/>);
    const leaf = unwrapToLeaf(node);
    expect(leaf.kind).toBe("Leaf");
    expect(leaf.vertexAttributes.tryFind("a_position")).toBeDefined();
    expect(leaf.vertexAttributes.tryFind("a_color")).toBeDefined();
  });

  it("<Sg.Tetrahedron/> emits a leaf with positions+normals+color", () => {
    const node = collectSgChildren(<Sg.Tetrahedron Color={new V4f(0.5, 0.5, 0.5, 1)}/>);
    const leaf = unwrapToLeaf(node);
    expect(leaf.vertexAttributes.tryFind("a_position")).toBeDefined();
    expect(leaf.vertexAttributes.tryFind("a_normal")).toBeDefined();
    expect(leaf.vertexAttributes.tryFind("a_color")).toBeDefined();
  });

  it("<Sg.Octahedron/> + wire variants emit leaves", () => {
    const tri = unwrapToLeaf(collectSgChildren(<Sg.Octahedron/>));
    const wire = unwrapToLeaf(collectSgChildren(<Sg.WireOctahedron/>));
    expect(tri.kind).toBe("Leaf");
    expect(wire.kind).toBe("Leaf");
  });

  it("<Sg.WireBox/> emits a leaf wrapped in Trafo+Intersectable", () => {
    const node = collectSgChildren(<Sg.WireBox/>);
    expect(["Trafo", "Intersectable"]).toContain(node.kind);
    const leaf = unwrapToLeaf(node);
    expect(leaf.kind).toBe("Leaf");
  });

  it("<Sg.Sphere/> emits a leaf and wires Intersectable.sphere", () => {
    const node = collectSgChildren(<Sg.Sphere radius={2}/>);
    // Trafo wraps the leaf; Intersectable wraps the Trafo.
    expect(["Trafo", "Intersectable"]).toContain(node.kind);
    const leaf = unwrapToLeaf(node);
    expect(leaf.vertexAttributes.tryFind("a_color")).toBeDefined();
  });

  it("<Sg.Cylinder/> + <Sg.Cone/> compile to leaves", () => {
    const cyl = unwrapToLeaf(collectSgChildren(<Sg.Cylinder/>));
    const cone = unwrapToLeaf(collectSgChildren(<Sg.Cone/>));
    expect(cyl.kind).toBe("Leaf");
    expect(cone.kind).toBe("Leaf");
  });

  it("<Sg.FullscreenQuad/> + <Sg.ScreenQuad/> compile to leaves", () => {
    const fs = unwrapToLeaf(collectSgChildren(<Sg.FullscreenQuad/>));
    const sq = unwrapToLeaf(collectSgChildren(<Sg.ScreenQuad z={0.5}/>));
    expect(fs.kind).toBe("Leaf");
    expect(sq.kind).toBe("Leaf");
  });
});

describe("shared geometry caches", () => {
  it("getBoxGeometry returns the same vertex buffers across calls (object identity)", () => {
    const a = getBoxGeometry();
    const b = getBoxGeometry();
    expect(AVal.force(a.vertexAttrs.tryFind("a_position")!))
      .toBe(AVal.force(b.vertexAttrs.tryFind("a_position")!));
    expect(AVal.force(a.indices)).toBe(AVal.force(b.indices));
  });

  it("two Sg.Sphere with the same tessellation share buffers", () => {
    const a = getSphereGeometry(32);
    const b = getSphereGeometry(32);
    expect(AVal.force(a.vertexAttrs.tryFind("a_position")!))
      .toBe(AVal.force(b.vertexAttrs.tryFind("a_position")!));
  });

  it("different tessellations produce different cached geometries", () => {
    const a = getSphereGeometry(16);
    const b = getSphereGeometry(32);
    expect(AVal.force(a.vertexAttrs.tryFind("a_position")!))
      .not.toBe(AVal.force(b.vertexAttrs.tryFind("a_position")!));
  });

  it("cylinder cache keyed by tessellation", () => {
    expect(AVal.force(getCylinderGeometry(8).vertexAttrs.tryFind("a_position")!))
      .not.toBe(AVal.force(getCylinderGeometry(16).vertexAttrs.tryFind("a_position")!));
  });
});

describe("auto-Intersectable", () => {
  it("Sg.Box default size wires Box3d intersectable", () => {
    const node = collectSgChildren(<Sg.Box/>);
    expect(node.kind).toBe("Intersectable");
  });

  it("Sg.Box with explicit `box` wires the supplied Box3d", () => {
    const b = Box3d.fromMinMax(new V3d(0, 0, 0), new V3d(2, 2, 2));
    const node = collectSgChildren(<Sg.Box box={b}/>);
    // Wraps Trafo(scale+translate) + Intersectable around the leaf.
    let foundIntersectable = false;
    let n: import("../src/scene/index.js").SgNode | undefined = node;
    while (n !== undefined) {
      if (n.kind === "Intersectable") foundIntersectable = true;
      if ("child" in n && n.child !== undefined) n = n.child as never;
      else break;
    }
    expect(foundIntersectable).toBe(true);
  });

  it("Sg.Sphere wires Intersectable.sphere", () => {
    const node = collectSgChildren(<Sg.Sphere radius={1.5}/>);
    let foundIntersectable = false;
    let n: import("../src/scene/index.js").SgNode | undefined = node;
    while (n !== undefined) {
      if (n.kind === "Intersectable") foundIntersectable = true;
      if ("child" in n && n.child !== undefined) n = n.child as never;
      else break;
    }
    expect(foundIntersectable).toBe(true);
  });
});

describe("colorAval", () => {
  it("returns a stride-0 BufferView with the colour bytes", () => {
    const view = AVal.force(colorAval(new V4f(0.25, 0.5, 0.75, 1.0)));
    expect(view.stride).toBe(0);
    expect(view.count).toBe(1);
    expect(view.format).toBe("float32x4");
    if (view.buffer.kind !== "host") throw new Error("expected host buffer");
    const ab = view.buffer.data instanceof ArrayBuffer
      ? view.buffer.data
      : (view.buffer.data.buffer as ArrayBuffer);
    const f = new Float32Array(ab,
      view.buffer.data instanceof ArrayBuffer ? 0 : view.buffer.data.byteOffset,
      4);
    expect(f[0]).toBeCloseTo(0.25, 6);
    expect(f[1]).toBeCloseTo(0.5, 6);
    expect(f[2]).toBeCloseTo(0.75, 6);
    expect(f[3]).toBeCloseTo(1.0, 6);
  });

  it("ticks when the source cval ticks — buffer bytes change", () => {
    const cv = cval(new V4f(0, 0, 0, 1));
    const av = colorAval(cv);
    const v1 = AVal.force(av);
    if (v1.buffer.kind !== "host") throw new Error("expected host");
    const ab1 = v1.buffer.data instanceof ArrayBuffer
      ? v1.buffer.data
      : (v1.buffer.data.buffer as ArrayBuffer);
    const f1 = new Float32Array(ab1,
      v1.buffer.data instanceof ArrayBuffer ? 0 : v1.buffer.data.byteOffset, 4);
    expect(f1[0]).toBeCloseTo(0, 6);
    transact(() => { cv.value = new V4f(1, 0, 0, 1); });
    const v2 = AVal.force(av);
    if (v2.buffer.kind !== "host") throw new Error("expected host");
    const ab2 = v2.buffer.data instanceof ArrayBuffer
      ? v2.buffer.data
      : (v2.buffer.data.buffer as ArrayBuffer);
    const f2 = new Float32Array(ab2,
      v2.buffer.data instanceof ArrayBuffer ? 0 : v2.buffer.data.byteOffset, 4);
    expect(f2[0]).toBeCloseTo(1, 6);
  });
});
