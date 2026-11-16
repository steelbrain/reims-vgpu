// The shaders of oracle_fnreflect.m: functions whose own reflection the probe prints (a vertex
// function's vertexAttributes, a kernel's stageInputAttributes, the constants a function declares,
// the library's reflection of each function), and whose pipelines' own answers it prints: stage
// inputs of every scalar and vector type an attribute takes, at scattered indices and out of
// their order, one unread, attributes under function constants and one read only under one, a
// vertex function of no stage
// input, a fragment function's stage input, a kernel's stage input, and a kernel of threadgroup
// memory. build.sh builds it twice: for macOS 26 (oracle_fnreflect.metallib) and for macOS 11
// (oracle_fnreflect_11.metallib), as Metal's reflection of a library's function depends on the
// release the library was built for.
#include <metal_stdlib>
using namespace metal;

struct Out {
	float4 position [[position]];
	float4 color;
};

// Attributes of the 32-bit and 16-bit float and integer types, 1 to 4 wide, and one unread.
struct TypesA {
	float f [[attribute(0)]];
	float2 f2 [[attribute(1)]];
	float3 f3 [[attribute(2)]];
	float4 f4 [[attribute(3)]];
	half h [[attribute(4)]];
	half2 h2 [[attribute(5)]];
	half3 h3 [[attribute(6)]];
	half4 h4 [[attribute(7)]];
	int i [[attribute(8)]];
	int2 i2 [[attribute(9)]];
	int3 i3 [[attribute(10)]];
	int4 i4 [[attribute(11)]];
	uint u [[attribute(12)]];
	uint2 u2 [[attribute(13)]];
	uint3 u3 [[attribute(14)]];
	uint4 u4 [[attribute(15)]];
	float4 unread [[attribute(30)]];
};

vertex Out v_types_a(TypesA in [[stage_in]])
{
	float4 s = float4(in.f) + float4(in.f2, 0, 0) + float4(in.f3, 0) + in.f4;
	s += float4(float(in.h)) + float4(float2(in.h2), 0, 0) + float4(float3(in.h3), 0) + float4(in.h4);
	s += float4(float(in.i)) + float4(float2(in.i2), 0, 0) + float4(float3(in.i3), 0) + float4(in.i4);
	s += float4(float(in.u)) + float4(float2(in.u2), 0, 0) + float4(float3(in.u3), 0) + float4(in.u4);
	Out o;
	o.position = s;
	o.color = float4(1);
	return o;
}

// Attributes of the 16-bit and 8-bit integer types, 1 to 4 wide.
struct TypesB {
	short s [[attribute(0)]];
	short2 s2 [[attribute(1)]];
	short3 s3 [[attribute(2)]];
	short4 s4 [[attribute(3)]];
	ushort us [[attribute(4)]];
	ushort2 us2 [[attribute(5)]];
	ushort3 us3 [[attribute(6)]];
	ushort4 us4 [[attribute(7)]];
	char c [[attribute(8)]];
	char2 c2 [[attribute(9)]];
	char3 c3 [[attribute(10)]];
	char4 c4 [[attribute(11)]];
	uchar uc [[attribute(12)]];
	uchar2 uc2 [[attribute(13)]];
	uchar3 uc3 [[attribute(14)]];
	uchar4 uc4 [[attribute(15)]];
};

vertex Out v_types_b(TypesB in [[stage_in]])
{
	float4 s = float4(float(in.s)) + float4(float2(in.s2), 0, 0) + float4(float3(in.s3), 0) + float4(in.s4);
	s += float4(float(in.us)) + float4(float2(in.us2), 0, 0) + float4(float3(in.us3), 0) + float4(in.us4);
	s += float4(float(in.c)) + float4(float2(in.c2), 0, 0) + float4(float3(in.c3), 0) + float4(in.c4);
	s += float4(float(in.uc)) + float4(float2(in.uc2), 0, 0) + float4(float3(in.uc3), 0) + float4(in.uc4);
	Out o;
	o.position = s;
	o.color = float4(0);
	return o;
}

// Attributes declared out of their indices' order, one read only through another's value.
struct Order {
	float4 b [[attribute(7)]];
	float4 a [[attribute(2)]];
	float2 c [[attribute(29)]];
	float d [[attribute(16)]];
};

vertex Out v_order(Order in [[stage_in]])
{
	Out o;
	o.position = in.b + in.a;
	o.color = float4(in.c, in.d, 1);
	return o;
}

// Attributes under function constants, beside one under none.
constant bool kOffset [[function_constant(0)]];
constant bool kColor [[function_constant(1)]];

struct Gated {
	float4 position [[attribute(0)]];
	float4 offset [[attribute(1), function_constant(kOffset)]];
	float4 color [[attribute(2), function_constant(kColor)]];
};

vertex Out v_gated(Gated in [[stage_in]])
{
	Out o;
	o.position = in.position;
	if (kOffset)
		o.position += in.offset;
	o.color = float4(1);
	if (kColor)
		o.color = in.color;
	return o;
}

// An attribute under no function constant whose only read is under one.
struct DeadRead {
	float4 position [[attribute(0)]];
	float4 extra [[attribute(4)]];
};

vertex Out v_deadread(DeadRead in [[stage_in]])
{
	Out o;
	o.position = in.position;
	if (kColor)
		o.position += in.extra;
	o.color = float4(1);
	return o;
}

// A stage input beside built-ins and buffers.
struct Mixed {
	float3 position [[attribute(0)]];
	uchar4 color [[attribute(3)]];
};

vertex Out v_mixed(Mixed in [[stage_in]], uint vid [[vertex_id]], uint iid [[instance_id]],
                   constant float4x4 &mvp [[buffer(1)]], device const float4 *offsets [[buffer(2)]])
{
	Out o;
	o.position = mvp * float4(in.position, 1) + offsets[vid + iid];
	o.color = float4(in.color) / 255.0;
	return o;
}

// No stage input.
vertex Out v_plain(uint vid [[vertex_id]], device const float4 *positions [[buffer(0)]])
{
	Out o;
	o.position = positions[vid];
	o.color = float4(1);
	return o;
}

// A fragment function's stage input.
fragment float4 f_in(Out in [[stage_in]], bool front [[front_facing]])
{
	return front ? in.color : in.color * 0.5;
}

// A kernel's stage input, one attribute unread.
struct KIn {
	int2 cell [[attribute(0)]];
	uchar4 rgba [[attribute(4)]];
	float weight [[attribute(9)]];
	half4 unread [[attribute(12)]];
};

kernel void k_stage(KIn in [[stage_in]], device float4 *out [[buffer(0)]], uint gid [[thread_position_in_grid]])
{
	out[gid] = float4(float2(in.cell), 0, 0) + float4(in.rgba) * in.weight;
}

// Threadgroup memory, declared in the kernel and as an argument.
kernel void k_tg(device float *out [[buffer(0)]], threadgroup float *dynamic [[threadgroup(0)]],
                 uint lid [[thread_position_in_threadgroup]], uint gid [[thread_position_in_grid]])
{
	threadgroup float shared[64];
	shared[lid % 64] = float(gid);
	dynamic[lid] = float(lid);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	out[gid] = shared[(lid + 1) % 64] + dynamic[lid];
}
