// The shaders of tools/pipelines/pipelines.m: one source the tool compiles at run time
// (newLibraryWithSource:, with a `#define RGFX_NONCE` line before it) and `make pipelines`
// compiles into its metallibs (newLibraryWithData:, one a nonce). Every function adds RGFX_NONCE
// whole to a value it computes at run time (nonced()), so no compiler folds it away and each
// nonce is a library, and a set of functions, no cache has seen.
//   v_quad, f_tex   a textured quad of a UI: a vertex descriptor, one texture, a tint and an
//                   optional fade (function constants 0 and 1).
//   f_pbr           a heavier fragment: four textures and a loop over eight lights.
//   k_blur          a 5x5 blur through threadgroup memory, its weights a function constant (2).
//   k_fill          the smallest kernel: a pattern into a buffer.
#include <metal_stdlib>
using namespace metal;

#ifndef RGFX_NONCE
#define RGFX_NONCE 0u
#endif

// A tiny value of `x` and the nonce: the nonce stays an operand of an add the compiler cannot
// fold, since `x` is known only at run time.
static float nonced(uint x)
{
	return float((x * 2654435761u + RGFX_NONCE) >> 31) * 1e-9f;
}

constant float kTint [[function_constant(0)]];
constant bool kFade [[function_constant(1)]];
constant float kBlurSharp [[function_constant(2)]];
constant float kTintOrOne = is_function_constant_defined(kTint) ? kTint : 1.0f;
constant bool kFadeOrNot = is_function_constant_defined(kFade) ? kFade : false;
constant float kSharpOrOne = is_function_constant_defined(kBlurSharp) ? kBlurSharp : 1.0f;

struct QuadIn {
	float2 pos [[attribute(0)]];
	float2 uv [[attribute(1)]];
};

struct QuadOut {
	float4 pos [[position]];
	float2 uv;
	float3 world;
	float3 normal;
};

struct Frame {
	float4x4 mvp;
	float4 color;
	float time;
	uint lights;
};

vertex QuadOut v_quad(QuadIn in [[stage_in]], constant Frame &f [[buffer(1)]],
                      uint vid [[vertex_id]])
{
	QuadOut o;
	o.pos = f.mvp * float4(in.pos, 0.0f, 1.0f);
	o.uv = in.uv + nonced(vid);
	o.world = float3(in.pos, sin(f.time + float(vid)));
	o.normal = normalize(float3(in.uv - 0.5f, 1.0f));
	return o;
}

fragment half4 f_tex(QuadOut in [[stage_in]], texture2d<half> tex [[texture(0)]],
                     sampler s [[sampler(0)]], constant Frame &f [[buffer(0)]])
{
	half4 c = tex.sample(s, in.uv) * half4(f.color) * half(kTintOrOne);
	if (kFadeOrNot)
		c.a *= half(saturate(1.0f - length(in.uv - 0.5f) * 1.4f));
	return c + half(nonced(uint(in.pos.x)));
}

struct Light {
	float4 pos;
	float4 color;
};

static float ggx(float nh, float rough)
{
	const float a = rough * rough;
	const float d = nh * nh * (a * a - 1.0f) + 1.0f;
	return a * a / (M_PI_F * d * d + 1e-6f);
}

static float smith(float nv, float nl, float rough)
{
	const float k = (rough + 1.0f) * (rough + 1.0f) / 8.0f;
	return (nv / (nv * (1.0f - k) + k)) * (nl / (nl * (1.0f - k) + k));
}

static float3 fresnel(float vh, float3 f0)
{
	return f0 + (1.0f - f0) * pow(1.0f - vh, 5.0f);
}

fragment float4 f_pbr(QuadOut in [[stage_in]], texture2d<float> albedo [[texture(0)]],
                      texture2d<float> normals [[texture(1)]], texture2d<float> material [[texture(2)]],
                      texturecube<float> env [[texture(3)]], sampler s [[sampler(0)]],
                      constant Frame &f [[buffer(0)]], constant Light *lights [[buffer(1)]])
{
	const float3 base = albedo.sample(s, in.uv).rgb * f.color.rgb;
	const float3 tn = normals.sample(s, in.uv).xyz * 2.0f - 1.0f;
	const float3 n = normalize(in.normal + tn * 0.5f);
	const float4 m = material.sample(s, in.uv);
	const float rough = clamp(m.g, 0.04f, 1.0f), metal = m.b;
	const float3 v = normalize(float3(0.0f, 0.0f, 2.0f) - in.world);
	const float3 f0 = mix(float3(0.04f), base, metal);
	float3 acc = 0.0f;
	for (uint i = 0; i < min(f.lights, 8u); i++) {
		const float3 d = lights[i].pos.xyz - in.world;
		const float3 l = normalize(d);
		const float3 h = normalize(l + v);
		const float nl = saturate(dot(n, l)), nv = saturate(dot(n, v)) + 1e-4f;
		const float nh = saturate(dot(n, h)), vh = saturate(dot(v, h));
		const float3 F = fresnel(vh, f0);
		const float3 spec = F * ggx(nh, rough) * smith(nv, nl, rough) / (4.0f * nv * nl + 1e-4f);
		const float3 diff = (1.0f - F) * (1.0f - metal) * base / M_PI_F;
		acc += (diff + spec) * lights[i].color.rgb * nl / (1.0f + dot(d, d) * lights[i].pos.w);
	}
	const float3 r = reflect(-v, n);
	acc += env.sample(s, r, level(rough * 6.0f)).rgb * fresnel(saturate(dot(n, v)), f0) * m.r;
	acc = acc / (acc + 1.0f);
	return float4(pow(acc, float3(1.0f / 2.2f)) + nonced(uint(in.pos.y)), 1.0f);
}

kernel void k_blur(texture2d<float, access::sample> src [[texture(0)]],
                   texture2d<float, access::write> dst [[texture(1)]],
                   uint2 gid [[thread_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
                   uint2 tg [[threadgroup_position_in_grid]])
{
	threadgroup float4 tile[20][20];
	constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
	const int2 origin = int2(tg * 16u) - 2;
	for (uint y = lid.y; y < 20u; y += 16u)
		for (uint x = lid.x; x < 20u; x += 16u)
			tile[y][x] = src.sample(s, float2(origin + int2(x, y)) + 0.5f);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	float4 acc = 0.0f;
	float total = 0.0f;
	for (int dy = -2; dy <= 2; dy++)
		for (int dx = -2; dx <= 2; dx++) {
			const float w = exp(-float(dx * dx + dy * dy) * 0.5f * kSharpOrOne);
			acc += tile[lid.y + 2 + dy][lid.x + 2 + dx] * w;
			total += w;
		}
	if (gid.x < dst.get_width() && gid.y < dst.get_height())
		dst.write(acc / total + nonced(gid.x), gid);
}

kernel void k_fill(device uint *out [[buffer(0)]], uint gid [[thread_position_in_grid]])
{
	out[gid] = gid ^ RGFX_NONCE;
}
