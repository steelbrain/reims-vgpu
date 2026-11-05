// The older shaders of air_arguments_test.cpp, built for macOS 10.12 (-std=macos-metal1.2), whose
// AIR modules hold LLVM 3.5's metadata nodes (operands of type and value pairs): entry points of
// each stage with binding arguments of each access Metal 1.2 has, a function constant and one
// tested for. Built into a .metallib by Apple's Metal compiler at test time; never committed.
#include <metal_stdlib>
using namespace metal;

constant bool kOldScale [[function_constant(0)]];
constant int kOldOptional [[function_constant(1)]];
constant bool kOldHasOptional = is_function_constant_defined(kOldOptional);

kernel void old_every_access(texture2d<float, access::read> r [[texture(0)]],
                             texture2d<float> s [[texture(1)]],
                             texture2d<float, access::write> w [[texture(2)]],
                             device float *buf [[buffer(0)]],
                             constant float &k [[buffer(1)]],
                             sampler smp [[sampler(0)]],
                             uint2 g [[thread_position_in_grid]])
{
	float4 v = r.read(g) + s.sample(smp, float2(0));
	if (kOldHasOptional)
		v += float(kOldOptional);
	w.write(v, g);
	buf[0] = kOldScale ? v.x * k : v.x;
}

struct OldOut {
	float4 p [[position]];
};

vertex OldOut old_vertex(uint vid [[vertex_id]], texture2d<float> t [[texture(4)]])
{
	OldOut o;
	o.p = t.read(uint2(vid, 0));
	return o;
}

fragment float4 old_fragment(OldOut in [[stage_in]], texture2d<float, access::write> w [[texture(5)]],
                             texture2d<float> s [[texture(0)]], sampler smp [[sampler(0)]])
{
	float4 c = s.sample(smp, float2(0));
	w.write(c, uint2(in.p.xy));
	return c;
}
