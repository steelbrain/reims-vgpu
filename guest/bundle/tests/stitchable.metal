// The shell tests' and ci_probe's stitchable functions (Makefile), and the host tests' (the
// Vulkan crate's build.rs): a stitched library's graph adds two of its inputs and multiplies the
// sum by the third, as `fused`, which stitch_caller calls from a library that links it; and one
// adds its first two, as `pair`, which stitch_caller_extra calls with three arguments, one more
// than the graph takes, as MPS's reduce_stitched calls the write_fn it stitches of two inputs.
// Metal links such a call, and drops the extra argument (measured: macOS 26.7.1, AMD Radeon Pro
// 5300M and Intel UHD Graphics 630).
#include <metal_stdlib>
using namespace metal;

[[stitchable]] float stitch_add(float a, float b) { return a + b; }

[[stitchable]] float stitch_mul(float a, float b) { return a * b; }

[[visible]] float fused(float a, float b, float c);

kernel void stitch_caller(device float *out [[buffer(0)]], constant float *in [[buffer(1)]]) {
    out[0] = fused(in[0], in[1], in[2]);
}

[[visible]] float pair(float a, float b, float c);

kernel void stitch_caller_extra(device float *out [[buffer(0)]], constant float *in [[buffer(1)]]) {
    out[0] = pair(in[0], in[1], in[2]);
}
