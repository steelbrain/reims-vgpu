#include <metal_stdlib>
using namespace metal;
// One thread: c[0] += 1, and c[c[0]] = c[0] * 0x11 for c[0] < 16, so a loop's
// iterations each leave a mark.
kernel void inc_k(device uint *c [[buffer(0)]]) {
    uint n = c[0] + 1;
    c[0] = n;
    if (n < 16) c[n] = n * 0x11;
}
