// Render suite: a fragment function that samples fragment texture 0 with
// fragment sampler 0 at a coordinate that runs from -0.5 to 1.5 across the
// target, so the sampler's address modes and filters show in the pixels.
#include <metal_stdlib>
using namespace metal;
fragment float4 render_ftex(float4 p [[position]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {
    float2 uv = p.xy / float2(8.0, 8.0) * 2.0 - 0.5;
    return t.sample(s, uv);
}
