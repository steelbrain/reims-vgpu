// oracle_bundle's functions (oracle_bundle.m): one of each stage the library answers a function type
// for, and a kernel with function constants.
#include <metal_stdlib>
using namespace metal;

constant bool div_flag [[function_constant(0)]];
constant float div_scale [[function_constant(1)]];

struct DivOut {
	float4 position [[position]];
};

vertex DivOut div_vertex(uint vid [[vertex_id]])
{
	DivOut o;
	o.position = float4(float(vid), 0.0f, 0.0f, 1.0f);
	return o;
}

fragment float4 div_fragment(DivOut in [[stage_in]])
{
	return in.position;
}

kernel void div_kernel(device float *out [[buffer(0)]], uint i [[thread_position_in_grid]])
{
	out[i] = div_flag ? div_scale : 1.0f;
}

[[visible]] float div_visible(float x)
{
	return x * 2.0f;
}

[[stitchable]] float div_stitchable(float x)
{
	return x + 1.0f;
}
