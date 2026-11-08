#include <metal_stdlib>
using namespace metal;
kernel void double_gid(device uint *out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = gid * 2;
}
