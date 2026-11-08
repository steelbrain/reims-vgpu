#include <metal_stdlib>
using namespace metal;
// out[gid] = in[gid] + 1: a kernel whose result depends on what ran before it.
kernel void add_one(device uint *out [[buffer(0)]], device const uint *in [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    out[gid] = in[gid] + 1;
}
