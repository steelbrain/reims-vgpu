// Render suite: the guest bench's flat colour (draws.10k's f_flat, in
// guest/bundle/tools/bench/bench.m): constant buffer 0's float4, which the
// bench sets with setFragmentBytes once a pass.
#include <metal_stdlib>
using namespace metal;
fragment half4 render_flat(constant float4 &c [[buffer(0)]]) { return half4(c); }
