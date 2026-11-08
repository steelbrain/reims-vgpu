// The vertex function of the pipeline that ends a pass over a mutable rasterization rate map
// (gpu/rate_map.rs, `no_draw`), which build.rs compiles into no_draw.metallib: it returns nothing,
// and the pipeline does not rasterize.
#include <metal_stdlib>
using namespace metal;

vertex void reims_vgpu_no_draw() {}
