// The stage input attributes of tests/shell_tests.mm's test_function_stage_attributes_as_metal,
// specialised beside the oracle probe's (tools/oracle/oracle_fnreflect.metal): attributes under a
// predicate that is an expression of function constants, one under a constant of its own, one read
// only under a constant, at scattered indices, and a kernel's stage input under a constant. Our own
// source; the Makefile builds it, and nothing of it is committed but this.
#include <metal_stdlib>
using namespace metal;

constant bool kA [[function_constant(0)]];
constant bool kB [[function_constant(1)]];
constant bool kAB = kA && !kB;

struct In {
	float4 p [[attribute(0)]];
	float2 uv [[attribute(1), function_constant(kAB)]];
	float4 c [[attribute(5)]];
	half4 h [[attribute(7), function_constant(kB)]];
};

vertex float4 v_expr(In in [[stage_in]])
{
	float4 r = in.p;
	if (kAB)
		r.xy += in.uv;
	if (kB)
		r += float4(in.h) + in.c;
	return r;
}

struct KIn {
	float2 a [[attribute(0)]];
	uint b [[attribute(2), function_constant(kB)]];
	char4 d [[attribute(3)]];
};

kernel void k_expr(KIn in [[stage_in]], device float *out [[buffer(0)]], uint g [[thread_position_in_grid]])
{
	out[g] = in.a.x + (kB ? float(in.b) : 0) + (kA ? float(in.d.x) : 0);
}
