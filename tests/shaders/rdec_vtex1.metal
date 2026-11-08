// render_vtex at level 1: the vertex stage samples texture 0 at (0.75, 0.25),
// explicit level 1, through sampler 0, and every vertex passes the texel on
// (scenarios/mutgap2_rdec.py: a sampler's LOD clamps move the level).
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float4 color [[user(color)]]; };
vertex VOut rdec_vtex1(uint vid [[vertex_id]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {
    const float2 p[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut o;
    o.pos = float4(p[vid], 0.0, 1.0);
    o.color = t.sample(s, float2(0.75, 0.25), level(1.0));
    return o;
}
