// The tests' functions that call a visible function they do not define (`states::tests`): the
// kernel and the vertex function each call `linked_add_one`, which only `linked_callee.metal`'s
// library defines, so a pipeline of either links it (`MTLLinkedFunctions`).
#include <metal_stdlib>
using namespace metal;

[[visible]] int linked_add_one(int x);

kernel void linked_kernel(device int *out [[buffer(0)]]) {
    out[0] = linked_add_one(out[0]);
}

vertex float4 linked_vertex(uint vid [[vertex_id]]) {
    return float4(float(linked_add_one(int(vid))), 0.0, 0.0, 1.0);
}

fragment float4 linked_fragment() {
    return float4(1.0, 0.0, 0.0, 1.0);
}

// RenderBox's custom effect, in small: a fragment function of a function constant of its own
// that calls `custom_fn`, which its library only declares, and which a pipeline links by making
// `linked_callee.metal`'s `linked_gradient` under that specialized name. Each fragment of the
// 4x4 target its triangle covers writes what it returns to `out`, by its pixel.
constant float linked_effect_alpha [[function_constant(0)]];

[[visible]] float4 custom_fn(float2 p);

vertex float4 linked_effect_vertex(uint vid [[vertex_id]]) {
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    return float4(p * 2.0 - 1.0, 0.0, 1.0);
}

fragment float4 linked_effect_fragment(float4 at [[position]], device float4 *out [[buffer(0)]]) {
    uint2 xy = uint2(at.xy);
    float4 c = custom_fn(at.xy);
    c.a = linked_effect_alpha;
    out[xy.y * 4 + xy.x] = c;
    return c;
}
