#include <metal_stdlib>
using namespace metal;
// Writes texel (x, y) of texture 0 as {x, y, 0x80, 0xff} / 255.
kernel void tex_write(texture2d<float, access::write> t [[texture(0)]], uint2 g [[thread_position_in_grid]]) {
    t.write(float4(float(g.x) / 255.0, float(g.y) / 255.0, 128.0 / 255.0, 1.0), g);
}
