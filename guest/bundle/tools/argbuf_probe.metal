// argbuf_probe's kernel (argbuf_probe.m): two argument buffers of the same layout, each a texture,
// a sampler, a buffer, a constant and a nested argument buffer of a texture and a sampler.
#include <metal_stdlib>
using namespace metal;

struct Inner {
	texture2d<float> t [[id(0)]];
	sampler s [[id(1)]];
};

struct Args {
	texture2d<float> tex [[id(0)]];
	sampler smp [[id(1)]];
	device const float4 *data [[id(2)]];
	float4 k [[id(3)]];
	constant Inner *inner [[id(4)]];
};

kernel void argbuf_probe(constant Args *args [[buffer(0)]], device float4 *out [[buffer(1)]],
                         uint i [[thread_position_in_grid]])
{
	constant Args &a = args[i / 4];
	switch (i % 4) {
	case 0: out[i] = a.tex.sample(a.smp, float2(0.6, 0.25), level(0)); break;
	case 1: out[i] = a.tex.sample(a.smp, float2(1.25, 0.25), level(0)); break;
	case 2: out[i] = a.data[1] + a.k; break;
	default: out[i] = a.inner->t.sample(a.inner->s, float2(0.25, 0.75), level(0)); break;
	}
}
