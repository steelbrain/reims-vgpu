// lock_probe's shaders (lock_probe.m): a kernel that writes a surface's texture and one that only
// reads it, each running a loop of `n` steps, and a full-screen triangle whose fragment writes it.
// `k` is 0 at run time, so the colour written is fixed while the loop's cost stays.
#include <metal_stdlib>
using namespace metal;

kernel void spin(texture2d<float, access::write> t [[texture(0)]], constant uint &n [[buffer(0)]],
                 constant float &k [[buffer(1)]], uint2 g [[thread_position_in_grid]])
{
	float v = float(g.x & 255) / 255.0;
	for (uint i = 0; i < n; i++) {
		v = fract(v * 1.0001 + 0.37);
	}
	t.write(float4(64.0 / 255.0 + k * v, 128.0 / 255.0, 192.0 / 255.0, 1.0), g);
}

kernel void reader(texture2d<float, access::read> t [[texture(0)]], constant uint &n [[buffer(0)]],
                   constant float &k [[buffer(1)]], device float *o [[buffer(2)]],
                   uint2 g [[thread_position_in_grid]])
{
	float v = t.read(g).x;
	for (uint i = 0; i < n; i++) {
		v = fract(v * 1.0001 + 0.37);
	}
	if (k > 0.5) {
		o[0] = v;
	}
}

struct VO {
	float4 p [[position]];
};

vertex VO vs(uint vid [[vertex_id]])
{
	float2 q[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
	VO o;
	o.p = float4(q[vid], 0, 1);
	return o;
}

fragment float4 fs(VO in [[stage_in]], constant uint &n [[buffer(0)]],
                   constant float &k [[buffer(1)]])
{
	float v = fract(in.p.x * 0.001);
	for (uint i = 0; i < n; i++) {
		v = fract(v * 1.0001 + 0.37);
	}
	return float4(64.0 / 255.0 + k * v, 128.0 / 255.0, 192.0 / 255.0, 1.0);
}
