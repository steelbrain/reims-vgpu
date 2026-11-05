// The shaders of oracle_reflection.m: render and compute entry points whose pipelines' reflection
// the probe prints, with bindings of every kind the guests' pipelines bind (buffers of scalars,
// vectors, matrices, structs, atomics and argument buffers; textures of each type and access,
// arrays of them; samplers; threadgroup memory), arguments the function never references,
// arguments under function constants, a vertex function's and a kernel's stage input, and an
// object and a mesh function that share a payload; and the kinds tools/pipeline_reflection_check.mm
// adds (nested argument buffers and their arrays, function and intersection tables, acceleration
// structures, imageblocks, framebuffer fetch, and functions function constants specialise).
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

struct Inner {
	float a;
	int2 b;
};

struct Params {
	float scale;
	float2 offset;
	float3 color;
	packed_float3 packed;
	half h;
	uchar4 bytes;
	int counts[3];
	Inner inner;
	float4x4 m;
	bool flag;
	uint2 extent;
	Inner inners[2];
};

kernel void k_basic(device float *out [[buffer(0)]],
                    constant Params &params [[buffer(1)]],
                    device const float4 *in [[buffer(2)]],
                    texture2d<float, access::read> r [[texture(0)]],
                    texture2d<float> s [[texture(1)]],
                    texture2d<float, access::write> w [[texture(2)]],
                    texture2d<half, access::read_write> rw [[texture(3)]],
                    sampler smp [[sampler(0)]],
                    threadgroup float *tg [[threadgroup(0)]],
                    uint2 gid [[thread_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]])
{
	float4 v = r.read(gid) + s.sample(smp, float2(params.offset)) + in[gid.x] + float4(rw.read(gid));
	tg[tid] = v.x * params.scale + params.color.y + params.packed.z + float(params.h) +
	          float(params.bytes.w) + float(params.counts[2]) + params.inner.a + params.m[1][2] +
	          (params.flag ? 1.0f : 0.0f) + float(params.extent.y) + float(params.inners[1].b.y);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	w.write(v, gid);
	rw.write(half4(v), gid);
	out[gid.x] = tg[tid ^ 1];
}

kernel void k_unused(device float *used [[buffer(0)]],
                     device float *unused [[buffer(1)]],
                     constant float4 &unusedConstant [[buffer(2)]],
                     texture2d<float> unusedTexture [[texture(0)]],
                     sampler unusedSampler [[sampler(0)]],
                     threadgroup float *unusedMemory [[threadgroup(0)]],
                     uint gid [[thread_position_in_grid]])
{
	used[gid] = 1;
}

kernel void k_types(device atomic_uint *counter [[buffer(0)]],
                    device float4x4 *matrices [[buffer(1)]],
                    device half3 *halves [[buffer(2)]],
                    device char *chars [[buffer(3)]],
                    device Inner *inners [[buffer(4)]],
                    device ulong *longs [[buffer(5)]],
                    device bool *bools [[buffer(6)]],
                    device packed_float3 *packed [[buffer(7)]],
                    device float2x3 *narrow [[buffer(8)]],
                    constant ushort4 &shorts [[buffer(9)]],
                    array<texture2d<float>, 3> textures [[texture(0)]],
                    texture2d_array<float> layered [[texture(3)]],
                    depth2d<float> depth [[texture(4)]],
                    texturecube<half> cube [[texture(5)]],
                    texture3d<uint, access::read> volume [[texture(6)]],
                    texture1d<int, access::write> line [[texture(7)]],
                    texture_buffer<float, access::read> texels [[texture(8)]],
                    texture2d_ms<float, access::read> multisampled [[texture(9)]],
                    depth2d_array<float> depths [[texture(10)]],
                    texturecube_array<float> cubes [[texture(11)]],
                    texture1d_array<float> lines [[texture(12)]],
                    array<sampler, 2> samplers [[sampler(0)]],
                    uint gid [[thread_position_in_grid]])
{
	float v = textures[2].sample(samplers[1], float2(0)).x + layered.sample(samplers[0], float2(0), 1).x +
	          depth.sample(samplers[0], float2(0)) + float(cube.sample(samplers[0], float3(1)).x) +
	          float(volume.read(uint3(gid)).x) + texels.read(gid).x + multisampled.read(uint2(gid), 0).x +
	          depths.sample(samplers[0], float2(0), 0) + cubes.sample(samplers[0], float3(1), 0).x +
	          lines.sample(samplers[0], 0.5f, 0).x;
	atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
	matrices[gid][0][0] = v;
	halves[gid] = half3(v);
	chars[gid] = char(v);
	inners[gid].a = v;
	longs[gid] = ulong(v);
	bools[gid] = v > 0;
	packed[gid] = packed_float3(v);
	narrow[gid][1][2] = v;
	line.write(int4(int(v) + int(shorts.y)), gid);
}

struct Shared {
	float4 v;
	uint n;
};

kernel void k_more(threadgroup Shared *shared [[threadgroup(0)]],
                   threadgroup float4 *vectors [[threadgroup(1)]],
                   depthcube<float> depthCube [[texture(0)]],
                   depth2d_ms<float> depthMultisampled [[texture(1)]],
                   texture2d_ms_array<float, access::read> multisampledLayers [[texture(2)]],
                   depthcube_array<float> depthCubes [[texture(3)]],
                   constant char &oneChar [[buffer(1)]],
                   constant half &oneHalf [[buffer(2)]],
                   constant half3 &threeHalves [[buffer(3)]],
                   device atomic_int *signedCounter [[buffer(4)]],
                   constant Inner *inners [[buffer(5)]],
                   sampler smp [[sampler(3)]],
                   uint3 group [[threadgroup_position_in_grid]],
                   uint3 perGroup [[threads_per_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint simdgroup [[simdgroup_index_in_threadgroup]],
                   uint3 grid [[threads_per_grid]],
                   uint t [[thread_index_in_threadgroup]])
{
	shared[t].v = vectors[t] + float4(depthCube.sample(smp, float3(1))) + depthMultisampled.read(uint2(0), 0) +
	              multisampledLayers.read(uint2(0), 0, 0) + depthCubes.sample(smp, float3(1), 0);
	shared[t].n = uint(oneChar) + uint(oneHalf) + uint(threeHalves.y) + group.x + perGroup.y + lane + simdgroup +
	              grid.x + uint(inners[1].b.x);
	atomic_fetch_add_explicit(signedCounter, 1, memory_order_relaxed);
}

kernel void k_builtins_only(uint g [[thread_position_in_grid]])
{
}

kernel void k_nothing()
{
}

struct Members {
	texture2d<float> texture [[id(0)]];
	device float *data [[id(1)]];
	sampler smp [[id(2)]];
	float value [[id(3)]];
	array<texture2d<float>, 2> textures [[id(4)]];
	constant Inner *inner [[id(6)]];
	texture2d<float, access::write> written [[id(7)]];
};

kernel void k_argument_buffer(device Members &members [[buffer(0)]],
                              constant Members *more [[buffer(1)]],
                              device float *out [[buffer(2)]],
                              uint gid [[thread_position_in_grid]])
{
	float4 c = members.texture.sample(members.smp, float2(0)) + members.textures[1].sample(members.smp, float2(0));
	members.written.write(c, uint2(gid));
	out[gid] = c.x + members.data[gid] + members.value + members.inner->a + more[1].value;
}

constant bool kOn [[function_constant(0)]];
constant bool kOff [[function_constant(1)]];
constant int kIndex [[function_constant(2)]];

kernel void k_constants(device float *out [[buffer(0)]],
                        device float *on [[buffer(1), function_constant(kOn)]],
                        device float *off [[buffer(2), function_constant(kOff)]],
                        texture2d<float> offTexture [[texture(0), function_constant(kOff)]],
                        texture2d<float> onTexture [[texture(1), function_constant(kOn)]],
                        device float *placed [[buffer(kIndex)]],
                        device float *readWhereOff [[buffer(3)]],
                        uint gid [[thread_position_in_grid]])
{
	float v = placed[gid];
	if (kOn)
		v += on[gid] + onTexture.read(uint2(gid)).x;
	if (kOff)
		v += off[gid] + offTexture.read(uint2(gid)).x + readWhereOff[gid];
	out[gid] = v;
}

struct StageIn {
	float4 position [[attribute(0)]];
	float2 uv [[attribute(1)]];
	float4 unread [[attribute(2)]];
};

kernel void k_stage_in(StageIn in [[stage_in]], device float4 *out [[buffer(0)]],
                       uint gid [[thread_position_in_grid]])
{
	out[gid] = in.position + float4(in.uv, 0, 0);
}

struct VertexOut {
	float4 position [[position]];
	float2 uv;
};

struct Uniforms {
	float4x4 mvp;
	float4 tint;
};

vertex VertexOut v_buffers(uint vid [[vertex_id]], uint iid [[instance_id]],
                           constant Uniforms &uniforms [[buffer(0)]],
                           device const float4 *positions [[buffer(1)]],
                           device const float2 *unusedUV [[buffer(2)]],
                           texture2d<float> heights [[texture(0)]])
{
	VertexOut o;
	o.position = uniforms.mvp * positions[vid + iid] + heights.read(uint2(vid, 0));
	o.uv = uniforms.tint.xy;
	return o;
}

fragment float4 f_textures(VertexOut in [[stage_in]],
                           texture2d<float> color [[texture(0)]],
                           texture2d<float> unusedTexture [[texture(1)]],
                           depth2d<float> shadow [[texture(2)]],
                           sampler smp [[sampler(0)]],
                           sampler unusedSampler [[sampler(1)]],
                           constant float4 &tint [[buffer(0)]],
                           device float *unusedBuffer [[buffer(1)]])
{
	return color.sample(smp, in.uv) * tint + shadow.sample(smp, in.uv);
}

struct VertexIn {
	float4 position [[attribute(0)]];
	float2 uv [[attribute(1)]];
	float4 unread [[attribute(2)]];
};

vertex VertexOut v_stage_in(VertexIn in [[stage_in]], constant float4x4 &mvp [[buffer(3)]])
{
	VertexOut o;
	o.position = mvp * in.position;
	o.uv = in.uv;
	return o;
}

fragment float4 f_plain(VertexOut in [[stage_in]])
{
	return float4(in.uv, 0, 1);
}

vertex void v_only(device float *out [[buffer(0)]], uint vid [[vertex_id]])
{
	out[vid] = 1;
}

struct Targets {
	float4 color [[color(0)]];
	half4 extra [[color(1)]];
};

fragment Targets f_targets(VertexOut in [[stage_in]], constant Params &params [[buffer(0)]])
{
	Targets t;
	t.color = float4(params.scale);
	t.extra = half4(params.h);
	return t;
}

struct Payload {
	float scale;
	uint count;
};

[[object]] void o_payload(object_data Payload &payload [[payload]], mesh_grid_properties grid,
                          constant float &scale [[buffer(0)]], device float *unused [[buffer(1)]],
                          uint tid [[thread_index_in_threadgroup]])
{
	if (tid == 0) {
		payload.scale = scale;
		payload.count = 1;
		grid.set_threadgroups_per_grid(uint3(1, 1, 1));
	}
}

struct MeshVertex {
	float4 position [[position]];
	float2 uv;
};

using TriangleMesh = metal::mesh<MeshVertex, void, 3, 1, metal::topology::triangle>;

[[mesh]] void m_payload(TriangleMesh output, const object_data Payload &payload [[payload]],
                        device const float4 *positions [[buffer(1)]], texture2d<float> unused [[texture(0)]],
                        uint tid [[thread_index_in_threadgroup]])
{
	if (tid == 0) {
		output.set_primitive_count(payload.count);
		output.set_index(0, 0);
		output.set_index(1, 1);
		output.set_index(2, 2);
	}
	if (tid < 3) {
		MeshVertex v;
		v.position = positions[tid] * payload.scale;
		v.uv = float2(0);
		output.set_vertex(tid, v);
	}
}

fragment float4 f_mesh(MeshVertex in [[stage_in]], texture2d<float> color [[texture(0)]], sampler smp [[sampler(0)]])
{
	return color.sample(smp, in.uv);
}

// The cases tools/pipeline_reflection_check.mm adds to the probe's.
struct Arrays {
	float floats[2];
	float4 vectors[2];
	float4x4 matrices[2];
	uint2 pairs[3];
	Inner inners[2];
	short s;
	packed_half3 ph;
};

kernel void k_arrays(constant Arrays &arrays [[buffer(0)]], device float *out [[buffer(1)]],
                     uint gid [[thread_position_in_grid]])
{
	out[gid] = arrays.floats[1] + arrays.vectors[1].x + arrays.matrices[1][2][3] + float(arrays.pairs[2].y) +
	           arrays.inners[1].a + float(arrays.s) + float(arrays.ph.y);
}

kernel void k_predicates(device float *out [[buffer(0)]], device float *on [[buffer(1), function_constant(kOn)]],
                         device float *off [[buffer(2), function_constant(kOff)]],
                         texture2d<float> offTexture [[texture(0), function_constant(kOff)]],
                         uint gid [[thread_position_in_grid]])
{
	float v = 0;
	if (kOn)
		v += on[gid];
	if (kOff)
		v += off[gid] + offTexture.read(uint2(gid)).x;
	out[gid] = v;
}

kernel void k_builtins(device uint *out [[buffer(0)]], uint3 inGroup [[thread_position_in_threadgroup]],
                       uint3 groups [[threadgroups_per_grid]], uint width [[threads_per_simdgroup]],
                       uint simdgroups [[simdgroups_per_threadgroup]], uint quad [[quadgroup_index_in_threadgroup]],
                       uint inQuad [[thread_index_in_quadgroup]], uint quads [[quadgroups_per_threadgroup]],
                       uint3 dispatch [[dispatch_threads_per_threadgroup]], uint3 gid [[thread_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]])
{
	out[tid] = inGroup.x + groups.y + width + simdgroups + quad + inQuad + quads + dispatch.z + gid.y;
}

struct MoreOut {
	float4 position [[position]];
	float3 normal;
	half2 h;
	int i [[flat]];
	uint4 u [[flat]];
	float size [[point_size]];
};

vertex MoreOut v_more(VertexIn in [[stage_in]], uint vid [[vertex_id]], uint base [[base_vertex]],
                      uint baseInstance [[base_instance]], constant float4x4 &mvp [[buffer(3)]])
{
	MoreOut o;
	o.position = mvp * in.position + float4(float(vid + base + baseInstance));
	o.normal = float3(in.uv, 0);
	o.h = half2(in.uv);
	o.i = int(vid);
	o.u = uint4(base);
	o.size = 1;
	return o;
}

fragment float4 f_more(MoreOut in [[stage_in]], bool front [[front_facing]], uint sid [[sample_id]])
{
	return float4(in.normal, float(in.h.x)) + float4(float(in.i + int(in.u.x) + int(front) + int(sid)));
}

// An argument buffer of members of every other kind: a struct of members, an array of them, arrays
// of samplers, buffers, constants and textures, and a pointer to an argument buffer of its own.
struct Leaf {
	texture2d<float> t [[id(0)]];
	sampler s [[id(1)]];
};

struct Child {
	device float *data [[id(0)]];
	float v [[id(1)]];
};

struct Nested {
	Leaf leaf [[id(0)]];
	device float *data [[id(2)]];
	Leaf leaves [[id(3)]] [2];
	array<sampler, 2> samplers [[id(7)]];
	device float *buffers [[id(9)]] [2];
	float4 values [[id(11)]] [2];
	device Child *child [[id(13)]];
	constant Inner *inners [[id(14)]] [2];
	texture2d<float> textures [[id(16)]] [2];
};

kernel void k_nested(device Nested &n [[buffer(0)]], device float *out [[buffer(1)]],
                     uint gid [[thread_position_in_grid]])
{
	float4 c = n.leaf.t.sample(n.leaf.s, float2(0)) + n.leaves[1].t.sample(n.samplers[1], float2(0)) +
	           n.textures[1].sample(n.leaves[0].s, float2(0));
	out[gid] = c.x + n.data[gid] + n.buffers[1][gid] + n.values[1].y + n.child->data[gid] + n.child->v +
	           n.inners[1]->a;
}

[[visible]] float twice(float x)
{
	return 2 * x;
}

kernel void k_tables(visible_function_table<float(float)> table [[buffer(0)]],
                     intersection_function_table<triangle_data> intersections [[buffer(1)]],
                     primitive_acceleration_structure primitives [[buffer(2)]],
                     instance_acceleration_structure instances [[buffer(3)]],
                     device float *out [[buffer(4)]], uint gid [[thread_position_in_grid]])
{
	ray r(float3(0), float3(0, 0, 1));
	intersector<triangle_data> i;
	auto hit = i.intersect(r, primitives, intersections);
	intersector<instancing, triangle_data> j;
	auto hit2 = j.intersect(r, instances, 1);
	out[gid] = table[0](float(gid)) + hit.distance + hit2.distance;
}

kernel void k_more_textures(texture2d<ushort> shorts [[texture(0)]],
                            depth2d_ms_array<float> depths [[texture(1)]],
                            texture_buffer<int, access::read_write> texels [[texture(2)]],
                            texture1d_array<uint, access::read> lines [[texture(3)]],
                            device float *out [[buffer(0)]], uint gid [[thread_position_in_grid]])
{
	out[gid] = float(shorts.read(uint2(gid)).x) + depths.read(uint2(0), 0, 0) + float(texels.read(gid).x) +
	           float(lines.read(gid, 0).x);
}

struct Block {
	float4 c [[color(0)]];
};

kernel void k_imageblock(imageblock<Block, imageblock_layout_implicit> block, device float *out [[buffer(0)]],
                         ushort2 lid [[thread_position_in_threadgroup]])
{
	out[0] = block.read(lid).c.x;
}

struct Fetched {
	float4 color [[color(0)]];
	half4 extra [[color(1)]];
};

fragment Fetched f_fetch(VertexOut in [[stage_in]], float4 last [[color(0)]], half4 other [[color(1)]])
{
	Fetched f;
	f.color = last * 0.5 + float4(other);
	f.extra = other;
	return f;
}

fragment float4 f_tables(VertexOut in [[stage_in]], primitive_acceleration_structure primitives [[buffer(0)]],
                         visible_function_table<float(float)> table [[buffer(1)]])
{
	ray r(float3(in.uv, 0), float3(0, 0, 1));
	intersector<triangle_data> i;
	auto hit = i.intersect(r, primitives);
	return float4(hit.distance + table[0](in.uv.x));
}

// Functions function constants specialise: a binding index and a predicate that are expressions of
// constants, a switch and a select on one, and an optional constant.
constant int kPlus = kIndex + 1;
constant bool kEither = kOn || kOff;
constant int kMode [[function_constant(3)]];
constant bool kOpt [[function_constant(4)]];
constant bool kOptDefined = is_function_constant_defined(kOpt);

kernel void k_specialised(device float *out [[buffer(0)]], device float *plus [[buffer(kPlus)]],
                          device float *either [[buffer(9), function_constant(kEither)]],
                          device float *modeOne [[buffer(10)]], device float *optional [[buffer(11)]],
                          texture2d<float> modeTwo [[texture(0)]], sampler smp [[sampler(0)]],
                          uint gid [[thread_position_in_grid]])
{
	float v = plus[gid];
	if (kEither)
		v += either[gid];
	switch (kMode) {
	case 1:
		v += modeOne[gid];
		break;
	default:
		break;
	}
	if (kOptDefined && kOpt)
		v += optional[gid];
	float4 x = modeTwo.sample(smp, float2(0));
	v = kMode == 2 ? v + x.x : v;
	out[gid] = v;
}

struct SpecialisedOut {
	float4 position [[position]];
	float2 uv;
};

vertex SpecialisedOut v_specialised(VertexIn in [[stage_in]], constant float4x4 &mvp [[buffer(3)]],
                                    device const float2 *uvs [[buffer(4)]], uint vid [[vertex_id]])
{
	SpecialisedOut o;
	o.position = mvp * in.position;
	o.uv = kMode == 1 ? in.uv : (kMode == 2 ? uvs[vid] : float2(0));
	return o;
}
