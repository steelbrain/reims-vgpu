// The tests' kernel that reaches a buffer through an address it loads from another, naming
// nothing of it: it reads the word there into `words[0]`, and writes `words[1]` to the word after
// it (`src/gpu/resource/tests/metal_tests.rs` makes its compute pipeline state).
#include <metal_stdlib>
using namespace metal;

kernel void reach(device const ulong *addresses [[buffer(0)]], device uint *words [[buffer(1)]]) {
    device uint *at = (device uint *)addresses[0];
    words[0] = at[0];
    at[1] = words[1];
}
