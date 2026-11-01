// samplepos_probe's shaders (samplepos_probe.m): one triangle whose right edge is the vertical
// line x = 1.5 pixels of a 4 x 4 target, so pixel column 1's coverage is the share of its samples
// left of the pixel's centre, and a flat red.
#include <metal_stdlib>
using namespace metal;

struct V {
	float4 p [[position]];
};

vertex V samplepos_vertex(uint i [[vertex_id]])
{
	const float2 at[3] = {float2(-3.0, 0.0), float2(-0.25, -3.0), float2(-0.25, 3.0)};
	V o;
	o.p = float4(at[i], 0.5, 1.0);
	return o;
}

fragment float4 samplepos_fragment()
{
	return float4(1.0, 0.0, 0.0, 1.0);
}
