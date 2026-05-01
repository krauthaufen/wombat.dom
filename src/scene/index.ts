// @aardworx/wombat.dom/scene — adaptive 3D scene-graph layer.
//
// Imports from this subpath bring in @aardworx/wombat.rendering
// + @aardworx/wombat.shader (declared as optional peerDependencies
// at the package level). The DOM-only side stays free of WebGPU
// concerns when this subpath isn't imported.
//
// Roadmap (see README §Scene roadmap):
//   M2  Sg core: SgNode tagged union, TraversalState, attribute
//       composition rules. No GPU.
//   M3  scene-to-RenderObject lowering. Hooks into wombat.rendering.
//   M4  <RenderControl> JSX component.
//   M5  Camera + view/proj uniforms.
//   M6  Free-fly + Orbit controllers.
//   M7  Pick framebuffer + pick effect.
//   M8  Pick-read + SceneEvent dispatch.
//   M9  Default surfaces + primitives.
//   M10 BVH picking + fusion (optional).

export const SCENE_LAYER_VERSION = "0.0.0";
