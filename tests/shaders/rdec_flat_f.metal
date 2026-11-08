// The fragment half of rdec_flat_v: returns the flat colour.
#include <metal_stdlib>
using namespace metal;
struct FIn { float4 pos [[position]]; float4 color [[user(color), flat]]; };
fragment float4 rdec_flat_f(FIn in [[stage_in]]) { return in.color; }
