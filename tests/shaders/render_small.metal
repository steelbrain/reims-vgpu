// Render suite: the guest bench's small triangle (draws.10k's v_small, in
// guest/bundle/tools/bench/bench.m): a triangle at d.xy, d.zw in size, in
// clip space, its d read from constant buffer 0, which the bench sets with
// setVertexBytes before each draw.
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; };
vertex VOut render_small(uint vid [[vertex_id]], constant float4 &d [[buffer(0)]]) {
    float2 p = float2((vid << 1) & 2, vid & 2) * 0.5;
    VOut o;
    o.pos = float4(d.xy + p * d.zw, 0.0, 1.0);
    return o;
}
