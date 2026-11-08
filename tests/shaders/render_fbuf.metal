// Render suite: a fragment function that returns fragment buffer 0's first
// float4, for fragment buffers, their offsets and blending.
#include <metal_stdlib>
using namespace metal;
fragment float4 render_fbuf(const device float4 *c [[buffer(0)]]) { return c[0]; }
