// The shaders of air_arguments_test.cpp: entry points of every stage the reader answers for,
// with binding arguments of each kind and access, arrays of them, argument buffers with and
// without a written texture, a function constant, arguments declared under function constants,
// binding indices function constants set, a vertex function's and a kernel's stage input, a
// function table and an acceleration structure, threadgroup memory, and a function of no texture.
// Built into a .metallib by Apple's Metal compiler at test time; never committed.
#include <metal_raytracing>
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

struct WritesOne {
	texture2d<float, access::write> out [[id(0)]];
	device float *values [[id(1)]];
};

struct ReadsOnly {
	texture2d<float> in [[id(0)]];
	constant float *values [[id(1)]];
};

constant bool kScale [[function_constant(0)]];

kernel void every_access(texture2d<float, access::read> r [[texture(0)]],
                         texture2d<float> s [[texture(1)]],
                         texture2d<float, access::read_write> rw [[texture(2)]],
                         texture2d<float, access::write> w [[texture(3)]],
                         array<texture2d<float>, 4> sampled [[texture(4)]],
                         array<texture2d<float, access::write>, 2> written [[texture(9)]],
                         device float *buf [[buffer(0)]],
                         constant float &k [[buffer(1)]],
                         device WritesOne &writesOne [[buffer(2)]],
                         constant ReadsOnly &readsOnly [[buffer(3)]],
                         sampler smp [[sampler(0)]],
                         uint2 g [[thread_position_in_grid]])
{
	float4 v = r.read(g) + s.sample(smp, float2(0)) + rw.read(g) + sampled[1].sample(smp, float2(0)) +
	           readsOnly.in.sample(smp, float2(0)) + readsOnly.values[0];
	rw.write(v, g);
	w.write(v, g);
	written[0].write(v, g);
	writesOne.out.write(v, g);
	writesOne.values[0] = v.x;
	buf[0] = kScale ? v.x * k : v.x;
}

kernel void no_texture(device float *b [[buffer(3)]], uint g [[thread_position_in_grid]])
{
	b[g] = 1;
}

struct VertexOut {
	float4 p [[position]];
};

vertex VertexOut reads_in_vertex(uint vid [[vertex_id]], texture2d<float> t [[texture(7)]])
{
	VertexOut o;
	o.p = t.read(uint2(vid, 0));
	return o;
}

fragment float4 writes_in_fragment(VertexOut in [[stage_in]],
                                   texture2d<float, access::write> w [[texture(5)]],
                                   texture2d<float> s [[texture(0)]], sampler smp [[sampler(0)]])
{
	float4 c = s.sample(smp, float2(0));
	w.write(c, uint2(in.p.xy));
	return c;
}

constant bool kHasLuma [[function_constant(1)]];
constant bool kHasChroma [[function_constant(2)]];
constant bool kHasPacked [[function_constant(3)]];

// A conversion kernel whose planes and output are declared under function constants, among
// four read buffers: each is its argument whatever the constants' values.
kernel void conditional_planes(device const uint *rows [[buffer(0)]],
                               constant uint &width [[buffer(1)]],
                               constant float4 &matrix [[buffer(2)]],
                               device const float *offsets [[buffer(3)]],
                               texture2d<half, access::read> luma [[texture(0), function_constant(kHasLuma)]],
                               texture2d<half, access::read> chroma
                                   [[texture(1), function_constant(kHasChroma)]],
                               texture2d<uint, access::write> packed
                                   [[texture(2), function_constant(kHasPacked)]],
                               device float *spill [[buffer(4), function_constant(kHasPacked)]],
                               uint2 g [[thread_position_in_grid]])
{
	float v = float(rows[g.y] + width) * matrix.x + offsets[0];
	if (kHasLuma)
		v += float(luma.read(g).x);
	if (kHasChroma)
		v += float(chroma.read(g / 2).x);
	if (kHasPacked) {
		packed.write(uint4(uint(v)), g);
		spill[g.x] = v;
	}
}

constant int kPlane [[function_constant(4)]];
constant int kOut [[function_constant(5)]];

// A plane read and an output written at binding indices function constants set, the plane's
// under a constant too: either may be at any index.
kernel void planes_where_constants_say(device const uint *rows [[buffer(0)]],
                                       texture2d<half, access::read> luma
                                           [[texture(kPlane), function_constant(kHasLuma)]],
                                       texture2d<half, access::write> out [[texture(kOut + 1)]],
                                       uint2 g [[thread_position_in_grid]])
{
	half v = half(rows[g.y]);
	if (kHasLuma)
		v += luma.read(g).x;
	out.write(half4(v), g);
}

// A plane read at an index a function constant sets, its output written at 3.
kernel void reads_where_constants_say(texture2d<half, access::read> luma [[texture(kPlane)]],
                                      texture2d<half, access::write> out [[texture(3)]],
                                      uint2 g [[thread_position_in_grid]])
{
	out.write(luma.read(g), g);
}

// Threadgroup memory at 0 beside a device buffer at 0 the kernel only reads: no buffer is bound
// at the memory's index.
kernel void shares_threadgroup_memory(threadgroup float *shared [[threadgroup(0)]],
                                      device const float *in [[buffer(0)]],
                                      device float *out [[buffer(1)]],
                                      uint t [[thread_position_in_threadgroup]])
{
	shared[t] = in[t];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	out[t] = shared[t ^ 1];
}

[[visible]] float doubled(float x)
{
	return x * 2;
}

struct Attributes {
	float4 position [[attribute(0)]];
};

// Bindings of no texture besides buffers and samplers: a stage input's attribute, a function
// table and an acceleration structure.
vertex VertexOut reads_attributes(Attributes in [[stage_in]], constant float4 &scale [[buffer(1)]])
{
	VertexOut o;
	o.p = in.position * scale;
	return o;
}

// A kernel's stage input, its attributes fetched from buffers: a binding of no texture.
kernel void reads_stage_in(Attributes in [[stage_in]], device float4 *out [[buffer(1)]],
                           texture2d<float, access::write> w
                               [[texture(0), function_constant(kHasPacked)]],
                           uint g [[thread_position_in_grid]])
{
	out[g] = in.position;
	if (kHasPacked)
		w.write(in.position, uint2(g, 0));
}

kernel void calls_through_table(device float *out [[buffer(0)]],
                                visible_function_table<float(float)> table [[buffer(1)]],
                                instance_acceleration_structure scene [[buffer(2)]],
                                texture2d<float, access::write> w [[texture(3)]],
                                uint g [[thread_position_in_grid]])
{
	(void)scene;
	out[g] = table[0](out[g]);
	w.write(float4(out[g]), uint2(g, 0));
}
