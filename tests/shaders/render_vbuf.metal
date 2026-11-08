// Render suite: a vertex function whose positions come from vertex buffer 0
// (indexed by vertex_id) and whose colour comes from buffer 1 (indexed by
// instance_id); each instance is shifted right by half the target. It
// surfaces vertex buffers and their offsets, vertex start and count,
// instances, base instance and base vertex.
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float4 color [[user(color)]]; float psize [[point_size]]; };
vertex VOut render_vbuf(uint vid [[vertex_id]], uint iid [[instance_id]],
                        const device float4 *pos [[buffer(0)]], const device float4 *col [[buffer(1)]]) {
    VOut o;
    o.pos = pos[vid];
    o.pos.x += float(iid) * 0.5;
    o.color = col[iid];
    o.psize = 1.0;
    return o;
}
