// Render suite: a full-target triangle at the depth vertex buffer 0's first
// float gives, coloured by buffer 0's second float4, for depth and stencil
// tests, depth bias and clipping.
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float4 color [[user(color)]]; };
struct Params { float4 z; float4 color; };
vertex VOut render_depth(uint vid [[vertex_id]], const device Params *p [[buffer(0)]]) {
    const float2 v[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut o;
    o.pos = float4(v[vid], p->z.x, 1.0);
    o.color = p->color;
    return o;
}
