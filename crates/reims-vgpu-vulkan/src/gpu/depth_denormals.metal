// The depth-denormal test's render functions (`depth_denormals.rs`): a triangle over the whole
// target at depth `z` (vertex buffer 0), and a fragment of `colour` (fragment buffer 0).
#include <metal_stdlib>
using namespace metal;

constant float2 corners[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};

struct Covered {
    float4 position [[position]];
};

vertex Covered cover(uint vid [[vertex_id]], constant float &z [[buffer(0)]]) {
    return Covered{float4(corners[vid], z, 1)};
}

fragment float4 paint(constant float4 &colour [[buffer(0)]]) {
    return colour;
}
