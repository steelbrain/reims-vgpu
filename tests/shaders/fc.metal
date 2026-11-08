#include <metal_stdlib>
using namespace metal;
constant uint k_mul [[function_constant(0)]];
kernel void fc_kernel(device uint *out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    out[gid] = gid * k_mul;
}
