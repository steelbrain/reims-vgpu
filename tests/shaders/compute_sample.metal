#include <metal_stdlib>
using namespace metal;
// Samples texture 0 with sampler 0 at x = (gid + 0.5) / 4 - 1, left of the
// texture, so the sampler's address mode decides the texel.
kernel void sample_k(texture2d<float, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
                     device uint *out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    float2 uv = float2((float(gid) + 0.5) / 4.0 - 1.0, 0.5);
    out[gid] = pack_float_to_unorm4x8(t.sample(s, uv));
}
