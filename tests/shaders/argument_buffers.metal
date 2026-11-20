#include <metal_stdlib>
using namespace metal;
// Argument buffers whose words the guest's encoder writes at the offsets the
// host's layout gives, read as guests' programs read them.

// sum_args: two pointers, at offsets 0 and 8 (16 bytes, aligned to 8).
struct Args {
    device const uint *a [[id(0)]];
    device const uint *b [[id(1)]];
};
kernel void sum_args(constant Args &args [[buffer(0)]], device uint *out [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    out[i] = args.a[i] + args.b[i];
}

// mixed_args: an array of two pointers, a constant, a texture it reads, and a
// pointer to a nested argument buffer of a pointer and a constant.
struct Inner {
    device const uint *p [[id(0)]];
    uint k [[id(1)]];
};
struct Mixed {
    array<device const uint *, 2> ptrs [[id(0)]];
    uint scale [[id(2)]];
    texture2d<uint, access::read> tex [[id(3)]];
    device Inner *inner [[id(4)]];
};
kernel void mixed_args(constant Mixed &m [[buffer(0)]], device uint *out [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    out[i] = m.ptrs[0][i] + m.ptrs[1][i] * m.scale + m.tex.read(uint2(i, 0)).x + m.inner->p[i] +
             m.inner->k;
}
