// Render suite: a full-target triangle whose colour the vertex stage reads
// from vertex texture 0 through vertex sampler 0 at (0.75, 0.25).
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float4 color [[user(color)]]; };
vertex VOut render_vtex(uint vid [[vertex_id]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {
    const float2 p[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut o;
    o.pos = float4(p[vid], 0.0, 1.0);
    o.color = t.sample(s, float2(0.75, 0.25), level(0.0));
    return o;
}
