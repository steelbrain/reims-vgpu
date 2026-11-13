// oracle_overlap's functions: arguments of one argument table at locations that overlap, each
// present under function constants, by kind (buffers, constant and argument buffers, threadgroup
// memory, textures and arrays of them, samplers), predicate (a constant, an expression of
// constants), use (used, unused, dead under the values) and index (literal, a constant's), and
// a vertex and a fragment function's. Our own code.
#include <metal_stdlib>
using namespace metal;
constant bool useA [[function_constant(0)]];
constant bool useB [[function_constant(1)]];
constant short kW [[function_constant(2)]];
constant uint kIdx [[function_constant(3)]];
constant bool both = useA && useB;
constant bool notA = !useA;
kernel void k_overlap(device float *a [[buffer(0), function_constant(useA)]],
                      device float *b [[buffer(0), function_constant(useB)]],
                      device float *c [[buffer(1)]],
                      uint i [[thread_position_in_grid]])
{
	if (useA)
		a[i] = 1.0f + kW;
	if (useB)
		b[i] = 2.0f;
	c[i] = 3.0f;
}
kernel void k_tex(texture2d<float, access::write> s [[texture(0), function_constant(useA)]],
                  texture2d<float, access::write> t [[texture(0), function_constant(useB)]],
                  uint2 i [[thread_position_in_grid]])
{
	if (useA)
		s.write(float4(1), i);
	if (useB)
		t.write(float4(2), i);
}
kernel void k_texarr(array<texture2d<float, access::write>, 2> s [[texture(0), function_constant(useA)]],
                     texture2d<float, access::write> t [[texture(1), function_constant(useB)]],
                     uint2 i [[thread_position_in_grid]])
{
	if (useA)
		s[1].write(float4(1), i);
	if (useB)
		t.write(float4(2), i);
}
kernel void k_smp(texture2d<float> src [[texture(0)]],
                  sampler s [[sampler(0), function_constant(useA)]],
                  sampler t [[sampler(0), function_constant(useB)]],
                  device float4 *out [[buffer(0)]],
                  uint i [[thread_position_in_grid]])
{
	float4 v = 0;
	if (useA)
		v += src.sample(s, float2(0.5));
	if (useB)
		v += src.sample(t, float2(0.25));
	out[i] = v;
}
kernel void k_tg(threadgroup float *a [[threadgroup(0), function_constant(useA)]],
                 threadgroup float *b [[threadgroup(0), function_constant(useB)]],
                 device float *c [[buffer(0)]],
                 uint i [[thread_position_in_grid]])
{
	float v = 0;
	if (useA) {
		a[i] = 1;
		v += a[0];
	}
	if (useB) {
		b[i] = 2;
		v += b[0];
	}
	c[i] = v;
}
kernel void k_derived(device float *a [[buffer(0), function_constant(both)]],
                      device float *b [[buffer(0), function_constant(notA)]],
                      uint i [[thread_position_in_grid]])
{
	if (both)
		a[i] = 1.0f;
	if (notA)
		b[i] = 2.0f;
}
kernel void k_idx(device float *a [[buffer(0)]],
                  device float *b [[buffer(kIdx)]],
                  uint i [[thread_position_in_grid]])
{
	a[i] = 1.0f;
	b[i] = 2.0f;
}
kernel void k_bufconst(device float *a [[buffer(0), function_constant(useA)]],
                       constant float4 &k [[buffer(0), function_constant(useB)]],
                       device float *c [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
	float v = 0;
	if (useA)
		a[i] = 1;
	if (useB)
		v = k.x;
	c[i] = v;
}
// b present but never referenced (air.arg_unused).
kernel void k_unused(device float *a [[buffer(0), function_constant(useA)]],
                     device float *b [[buffer(0), function_constant(useB)]],
                     uint i [[thread_position_in_grid]])
{
	if (useA)
		a[i] = 1.0f;
}
// b present, read only where kW is 7: dead under kW 3.
kernel void k_dead(device float *a [[buffer(0), function_constant(useA)]],
                   device float *b [[buffer(0), function_constant(useB)]],
                   uint i [[thread_position_in_grid]])
{
	if (useA)
		a[i] = 1.0f;
	if (useB && kW == 7)
		b[i] = 2.0f;
}
struct Args {
	texture2d<float, access::write> t [[id(0)]];
	device float *p [[id(1)]];
};
kernel void k_argbuf(device float *a [[buffer(0), function_constant(useA)]],
                     device Args &b [[buffer(0), function_constant(useB)]],
                     uint i [[thread_position_in_grid]])
{
	if (useA)
		a[i] = 1.0f;
	if (useB)
		b.p[i] = 2.0f;
}
// Three at buffer(0): which one Metal names.
kernel void k_three(device float *a [[buffer(0)]],
                    device float *b [[buffer(1), function_constant(useA)]],
                    device float *c [[buffer(0), function_constant(useB)]],
                    device float *d [[buffer(1), function_constant(useB)]],
                    uint i [[thread_position_in_grid]])
{
	a[i] = 0;
	if (useA)
		b[i] = 1.0f;
	if (useB) {
		c[i] = 2.0f;
		d[i] = 3.0f;
	}
}
struct VOut {
	float4 position [[position]];
};
vertex VOut v_overlap(device const float4 *a [[buffer(0), function_constant(useA)]],
                      device const float4 *b [[buffer(0), function_constant(useB)]],
                      uint vid [[vertex_id]])
{
	VOut o;
	o.position = float4(0);
	if (useA)
		o.position += a[vid];
	if (useB)
		o.position += b[vid];
	return o;
}
fragment float4 f_overlap(texture2d<float> s [[texture(0), function_constant(useA)]],
                          texture2d<float> t [[texture(0), function_constant(useB)]],
                          VOut in [[stage_in]])
{
	float4 v = 0;
	if (useA)
		v += s.read(uint2(0));
	if (useB)
		v += t.read(uint2(0));
	return v;
}
