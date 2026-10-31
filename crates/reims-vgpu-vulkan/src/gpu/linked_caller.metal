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
