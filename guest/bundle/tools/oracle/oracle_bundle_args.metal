// oracle_bundle's argument buffer function (oracle_bundle.m, cases argbuf.*): a kernel whose
// buffer 0 is an argument buffer of a texture, a buffer, a sampler, a constant and an array of
// textures, and whose buffer 1 is a plain buffer.
#include <metal_stdlib>
using namespace metal;

struct DivArgs {
	texture2d<float> tex [[id(0)]];
	device float *buf [[id(1)]];
	sampler smp [[id(2)]];
	float scale [[id(3)]];
	array<texture2d<float>, 3> texs [[id(4)]];
};

kernel void div_argbuf(constant DivArgs &args [[buffer(0)]], device float *out [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
	out[i] = args.tex.sample(args.smp, float2(0.5f)).x * args.scale + args.buf[i] +
	         args.texs[2].read(uint2(0)).y;
}
