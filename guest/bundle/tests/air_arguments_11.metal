// Shaders of air_arguments_test.cpp and shell_tests.mm built for macOS 11 (-std=macos-metal2.3
// -mmacosx-version-min=11.0), whose AIR names no air.address_space on a buffer argument, as the
// compiler of macOS 11 and 12 writes it: a kernel of a device buffer, a constant struct, threadgroup
// memory, a texture and a sampler; a vertex function of stage input and a constant matrix; a
// fragment function of a texture, a sampler and a constant colour. Built into a .metallib by Apple's
// Metal compiler at test time; never committed.
#include <metal_stdlib>
using namespace metal;

struct Params11 {
	float4 scale;
	int counts[3];
};

kernel void k_11(device float *out [[buffer(0)]], constant Params11 &params [[buffer(1)]],
                 threadgroup float *tg [[threadgroup(0)]], texture2d<float> tex [[texture(0)]],
                 sampler smp [[sampler(0)]], uint2 gid [[thread_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]])
{
	tg[tid] = tex.sample(smp, float2(gid)).x;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	out[gid.x] = params.scale.x * tg[0] + float(params.counts[1]);
}

struct In11 {
	float4 pos [[attribute(0)]];
	float2 uv [[attribute(1)]];
};

struct Out11 {
	float4 pos [[position]];
	float2 uv;
};

vertex Out11 v_11(In11 in [[stage_in]], constant float4x4 &mvp [[buffer(2)]])
{
	Out11 o;
	o.pos = mvp * in.pos;
	o.uv = in.uv;
	return o;
}

fragment float4 f_11(Out11 in [[stage_in]], texture2d<float> color [[texture(0)]], sampler smp [[sampler(0)]],
                     constant float4 &tint [[buffer(0)]])
{
	return color.sample(smp, in.uv) * tint;
}
