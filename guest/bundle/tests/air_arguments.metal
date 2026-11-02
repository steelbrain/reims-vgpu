// The shaders of air_arguments_test.cpp: entry points of every stage the reader answers for,
// with binding arguments of each kind and access, arrays of them, argument buffers with and
// without a written texture, a function constant and a function of no texture. Built into a
// .metallib by Apple's Metal compiler at test time; never committed.
#include <metal_stdlib>
using namespace metal;

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
