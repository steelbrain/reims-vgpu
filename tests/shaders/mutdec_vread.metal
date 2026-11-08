// For scenarios/mutgap_decoders.py: a vertex function that places a full-target
// triangle only when vertex buffer 0's first word is 1, and outside the target
// otherwise (see mutdec_fwrite.metal).
#include <metal_stdlib>
using namespace metal;
vertex float4 mutdec_vread(uint vid [[vertex_id]], const device uint *flag [[buffer(0)]]) {
    const float2 v[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    return float4(v[vid] + (flag[0] == 1 ? float2(0.0) : float2(8.0)), 0.5, 1.0);
}
