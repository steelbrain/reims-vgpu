// A vertex function whose colour is flat-interpolated: position from buffer
// 0 and colour from buffer 1, both by vertex_id, so the provoking vertex
// decides a triangle's colour (scenarios/mutgap2_rdec.py, 0x90).
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float4 color [[user(color), flat]]; };
vertex VOut rdec_flat_v(uint vid [[vertex_id]], const device float4 *pos [[buffer(0)]],
                        const device float4 *col [[buffer(1)]]) {
    VOut o;
    o.pos = pos[vid];
    o.color = col[vid];
    return o;
}
