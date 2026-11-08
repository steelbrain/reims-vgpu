#include <metal_stdlib>
using namespace metal;
// Samples texture 0 with sampler 0 at texel centres of row 0.5 with explicit
// gradients: dPdx (1, 0), dPdy (0, 1/16), a footprint 16 times longer in x
// than in y. An isotropic sampler takes its LOD from the major axis (level
// 2, clamped to the last level); an anisotropic one of 16 from the minor.
kernel void sample_grad(texture2d<float, access::sample> t [[texture(0)]], sampler s [[sampler(0)]],
                        device uint *out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    float2 uv = float2((float(gid) + 0.5) / 4.0, 0.5);
    out[gid] = pack_float_to_unorm4x8(t.sample(s, uv, gradient2d(float2(1.0, 0.0), float2(0.0, 1.0 / 16.0))));
}
