#include <metal_stdlib>
using namespace metal;
// A fragment function that reaches a texture beyond its slots: it writes
// (1, 0, 1, 1) to texel (0, 0) of the texture whose resource ID the argument
// buffer at fragment buffer 0 holds, and returns opaque green.
struct Written {
    texture2d<float, access::write> t [[id(0)]];
};
fragment float4 render_argwrite(constant Written &w [[buffer(0)]]) {
    w.t.write(float4(1.0, 0.0, 1.0, 1.0), uint2(0, 0));
    return float4(0.0, 1.0, 0.0, 1.0);
}
