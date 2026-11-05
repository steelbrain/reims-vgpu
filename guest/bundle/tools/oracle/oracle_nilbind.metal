// oracle_nilbind.metal: the shaders of oracle_nilbind. Each writes a marker beside what it read,
// so a draw or dispatch that ran shows apart from one that did not.
#include <metal_stdlib>
using namespace metal;

struct NilVOut {
	float4 position [[position]];
};

// A triangle that covers the whole target.
vertex NilVOut nil_vs(uint vid [[vertex_id]])
{
	const float2 p = float2((vid << 1) & 2, vid & 2);
	NilVOut out;
	out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
	return out;
}

// What texture 0 reads at the fragment, halved, plus red one half: a texture reading zero draws
// (0.5, 0, 0, 1).
fragment float4 nil_tex_fs(NilVOut in [[stage_in]], texture2d<float, access::read> t [[texture(0)]])
{
	const float4 v = t.read(uint2(0, 0));
	return float4(v.rgb * 0.5 + float3(0.5, 0.0, 0.0), 1.0);
}

// What texture 0 samples with sampler 0, as nil_tex_fs.
fragment float4 nil_sample_fs(NilVOut in [[stage_in]], texture2d<float> t [[texture(0)]],
                              sampler s [[sampler(0)]])
{
	const float4 v = t.sample(s, float2(0.5, 0.5));
	return float4(v.rgb * 0.5 + float3(0.5, 0.0, 0.0), 1.0);
}

// What buffer 0's first float4 holds, as nil_tex_fs.
fragment float4 nil_buf_fs(NilVOut in [[stage_in]], device const float4 *b [[buffer(0)]])
{
	const float4 v = b[0];
	return float4(v.rgb * 0.5 + float3(0.5, 0.0, 0.0), 1.0);
}

// A plain colour: for the passes' cases.
fragment float4 nil_solid_fs(NilVOut in [[stage_in]])
{
	return float4(0.5, 0.0, 0.0, 1.0);
}

// out[0..3]: what texture 0 reads at (0, 0); out[4]: 42, the marker.
kernel void nil_tex_k(texture2d<float, access::read> t [[texture(0)]],
                      device float *out [[buffer(1)]], uint gid [[thread_position_in_grid]])
{
	if (gid != 0)
		return;
	const float4 v = t.read(uint2(0, 0));
	out[0] = v.r;
	out[1] = v.g;
	out[2] = v.b;
	out[3] = v.a;
	out[4] = 42.0;
}

// out[0..3]: buffer 0's first float4; out[4]: 42.
kernel void nil_buf_k(device const float4 *b [[buffer(0)]], device float *out [[buffer(1)]],
                      uint gid [[thread_position_in_grid]])
{
	if (gid != 0)
		return;
	const float4 v = b[0];
	out[0] = v.r;
	out[1] = v.g;
	out[2] = v.b;
	out[3] = v.a;
	out[4] = 42.0;
}

// Writes texture 0 at (0, 0); out[4]: 42.
kernel void nil_texwrite_k(texture2d<float, access::write> t [[texture(0)]],
                           device float *out [[buffer(1)]], uint gid [[thread_position_in_grid]])
{
	if (gid != 0)
		return;
	t.write(float4(1.0), uint2(0, 0));
	out[4] = 42.0;
}
