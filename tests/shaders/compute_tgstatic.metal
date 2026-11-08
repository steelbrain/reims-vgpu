#include <metal_stdlib>
using namespace metal;
// 64 words of static threadgroup memory, which a pipeline reports as its
// staticThreadgroupMemoryLength.
kernel void tg_static(device uint *out [[buffer(0)]], uint lid [[thread_position_in_threadgroup]],
                      uint gid [[thread_position_in_grid]]) {
    threadgroup uint tg[64];
    tg[lid & 63] = gid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[gid] = tg[(lid + 1) & 63];
}
