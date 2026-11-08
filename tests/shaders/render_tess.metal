// Render suite: a post-tessellation vertex function for triangle patches
// whose three control points are float2 positions in buffer 0, indexed by
// patch_id; the colour is fixed. It surfaces drawPatches and its variants.
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float4 color [[user(color)]]; };
[[patch(triangle, 3)]]
vertex VOut render_tess(uint pid [[patch_id]], float3 b [[position_in_patch]], const device float2 *cp [[buffer(0)]]) {
    float2 p = cp[pid * 3] * b.x + cp[pid * 3 + 1] * b.y + cp[pid * 3 + 2] * b.z;
    VOut o;
    o.pos = float4(p, 0.0, 1.0);
    o.color = float4(0.0, 0.0, 1.0, 1.0);
    return o;
}
