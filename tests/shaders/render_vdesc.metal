// Render suite: a vertex function fed through a vertex descriptor
// (attribute 0: float2 position, attribute 1: uchar4 normalized colour).
#include <metal_stdlib>
using namespace metal;
struct VIn { float2 pos [[attribute(0)]]; float4 color [[attribute(1)]]; };
struct VOut { float4 pos [[position]]; float4 color [[user(color)]]; };
vertex VOut render_vdesc(VIn in [[stage_in]]) {
    VOut o;
    o.pos = float4(in.pos, 0.0, 1.0);
    o.color = in.color;
    return o;
}
