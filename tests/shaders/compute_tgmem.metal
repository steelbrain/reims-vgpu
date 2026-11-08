#include <metal_stdlib>
using namespace metal;
// Reverses each threadgroup through threadgroup memory 0, whose length the
// encoder sets.
kernel void tg_reverse(threadgroup uint *tg [[threadgroup(0)]], device uint *out [[buffer(0)]],
                       uint lid [[thread_position_in_threadgroup]], uint gid [[thread_position_in_grid]],
                       uint n [[threads_per_threadgroup]]) {
    tg[lid] = gid * 3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[gid] = tg[n - 1 - lid];
}
