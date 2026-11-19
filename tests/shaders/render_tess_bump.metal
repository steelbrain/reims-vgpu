// Render suite: post-tessellation vertex functions for triangle patches that
// push the edge between control points 0 and 1 (position_in_patch.z == 0)
// down the target by a bump of the position t along it (position_in_patch.y
// there), 0 at t = 0, 1/2 and 1, so the target shows where the tessellator put
// that edge's vertices. render_tess_bump reads the patch's three float2
// control points from buffer 0 by patch_id; render_tess_bump_cp reads them
// through [[stage_in]] (attribute 0, a float2). render_tess_facing colours a
// triangle that faces front blue and one that faces back red.
#include <metal_stdlib>
using namespace metal;
struct BumpOut { float4 pos [[position]]; };
static BumpOut bump(float2 c0, float2 c1, float2 c2, float3 b) {
    float2 p = c0 * b.x + c1 * b.y + c2 * b.z;
    if (b.z < 0.0001) {
        float t = b.y;
        float g = t * (1.0 - t) * (1.0 - 2.0 * t) * 10.0;
        p.y -= 0.95 * g * g;
    }
    BumpOut o;
    o.pos = float4(p, 0.0, 1.0);
    return o;
}
[[patch(triangle, 3)]]
vertex BumpOut render_tess_bump(uint pid [[patch_id]], float3 b [[position_in_patch]], const device float2 *cp [[buffer(0)]]) {
    return bump(cp[pid * 3], cp[pid * 3 + 1], cp[pid * 3 + 2], b);
}
struct ControlPoint { float2 pos [[attribute(0)]]; };
[[patch(triangle, 3)]]
vertex BumpOut render_tess_bump_cp(patch_control_point<ControlPoint> cp [[stage_in]], float3 b [[position_in_patch]]) {
    return bump(cp[0].pos, cp[1].pos, cp[2].pos, b);
}
fragment float4 render_tess_facing(bool front [[front_facing]]) {
    return front ? float4(0.0, 0.0, 1.0, 1.0) : float4(1.0, 0.0, 0.0, 1.0);
}
