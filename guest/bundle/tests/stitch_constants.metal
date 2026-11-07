// The shell tests' stitchable functions specialised before their stitch (Makefile), as MPS
// specialises its uber shaders' visible functions: `scaled` and `bias` read a function constant
// each, `add1` none.
#include <metal_stdlib>
using namespace metal;

constant float k [[function_constant(0)]];
constant float b [[function_constant(1)]];

[[stitchable]] float scaled(float x) { return x * k; }

[[stitchable]] float bias(float x) { return x + b; }

[[stitchable]] float add1(float x) { return x + 1.0f; }
