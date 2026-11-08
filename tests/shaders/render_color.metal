// Render suite: a fragment function that returns the interpolated colour
// render_vbuf.metal and render_vdesc.metal output.
#include <metal_stdlib>
using namespace metal;
struct FIn { float4 pos [[position]]; float4 color [[user(color)]]; };
fragment float4 render_color(FIn in [[stage_in]]) { return in.color; }
