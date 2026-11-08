#include <metal_stdlib>
using namespace metal;
kernel void a(device uint *o [[buffer(0)]]) { o[0] = 1; }
kernel void b(device uint *o [[buffer(0)]]) { o[0] = 2; }
