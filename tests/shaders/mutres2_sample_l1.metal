#include <metal_stdlib>
using namespace metal;
// Samples texture 0 with sampler 0 at an explicit level 1 (the sampler's
// lodMaxClamp can hold it at level 0): one texel per thread, packed.
kernel void sample_l1(texture2d<float, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
                      device uint *out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    float2 uv = float2((float(gid) + 0.5) / 4.0, 0.5);
    out[gid] = pack_float_to_unorm4x8(t.sample(s, uv, level(1.0)));
}
