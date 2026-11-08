#include <metal_stdlib>
using namespace metal;
// Relations between host-assigned 64-bit values the guest cannot lock by
// value: for pair i (in[2i], in[2i+1]), out[i] = bit 0 equal, bit 1 first
// nonzero, bit 2 second nonzero.
kernel void rel(device const ulong *in [[buffer(0)]], device uint *out [[buffer(1)]],
                uint gid [[thread_position_in_grid]]) {
    ulong a = in[2 * gid], b = in[2 * gid + 1];
    out[gid] = (a == b ? 1u : 0u) | (a != 0 ? 2u : 0u) | (b != 0 ? 4u : 0u);
}
