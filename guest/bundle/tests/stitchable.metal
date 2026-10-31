// The shell tests' stitchable functions (Makefile): a stitched library's graph adds two of its
// inputs and multiplies the sum by the third.
#include <metal_stdlib>
using namespace metal;

[[stitchable]] float stitch_add(float a, float b) { return a + b; }

[[stitchable]] float stitch_mul(float a, float b) { return a * b; }
