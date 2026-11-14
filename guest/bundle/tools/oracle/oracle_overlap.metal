// oracle_overlap's functions: arguments of one argument table at locations that overlap, each
// present under function constants, by kind (buffers, constant and argument buffers, threadgroup
// memory, textures and arrays of them, samplers, function tables, acceleration structures),
// predicate (a constant, an expression of constants, one of a half constant), use (used, unused,
// dead under the values) and index (literal, a constant's, one of a half constant, one through
// conversions to a float and back), and a vertex and a fragment function's. Our own code.
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;
constant bool useA [[function_constant(0)]];
constant bool useB [[function_constant(1)]];
constant short kW [[function_constant(2)]];
constant uint kIdx [[function_constant(3)]];
constant half kH [[function_constant(4)]];
constant bool hBig = kH > 1.0h;
constant uint hIdx = uint(kH * 0.5h);
constant uint fIdx = uint(float(kIdx + 1u) * 0.75f);
constant uint sIdx = uint(int(float(kW) * -0.5f) + 2 - int(kIdx));
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
// A predicate and an index of a half constant: present where kH > 1; at kH / 2 truncated.
kernel void k_halfpred(device float *a [[buffer(0), function_constant(useA)]],
                       device float *b [[buffer(0), function_constant(hBig)]],
                       uint i [[thread_position_in_grid]])
{
	if (useA)
		a[i] = 1.0f;
	if (hBig)
		b[i] = 2.0f;
}
kernel void k_halfidx(device float *a [[buffer(0), function_constant(useA)]],
                      device float *b [[buffer(hIdx)]],
                      uint i [[thread_position_in_grid]])
{
	if (useA)
		a[i] = 1.0f;
	b[i] = 2.0f;
}
// Indices through conversions of an integer to a float and back, truncated toward zero: kIdx 0 at
// 0 (0.75), 1 at 1 (1.5); kW 3, kIdx 0 at 1 (-1.5 to -1, + 2), 1 at 0.
kernel void k_fidx(device float *a [[buffer(0), function_constant(useA)]],
                   device float *b [[buffer(fIdx)]],
                   uint i [[thread_position_in_grid]])
{
	if (useA)
		a[i] = 1.0f;
	b[i] = 2.0f;
}
kernel void k_sidx(device float *a [[buffer(0), function_constant(useA)]],
                   device float *b [[buffer(sIdx)]],
                   uint i [[thread_position_in_grid]])
{
	if (useA)
		a[i] = 1.0f;
	b[i] = 2.0f;
}
// Function tables and acceleration structures at a buffer's location.
kernel void k_vft(device float *a [[buffer(0), function_constant(useA)]],
                  visible_function_table<float(float)> t [[buffer(0), function_constant(useB)]],
                  uint i [[thread_position_in_grid]])
{
	if (useA)
		a[i] = 1.0f;
	if (useB)
		a[i] = t[0](1.0f);
}
kernel void k_ift(device float *a [[buffer(0), function_constant(useA)]],
                  intersection_function_table<triangle_data> t [[buffer(0), function_constant(useB)]],
                  uint i [[thread_position_in_grid]])
{
	if (useA)
		a[i] = 1.0f;
}
kernel void k_pas(device float *a [[buffer(0), function_constant(useA)]],
                  primitive_acceleration_structure s [[buffer(0), function_constant(useB)]],
                  device float *o [[buffer(1)]],
                  uint i [[thread_position_in_grid]])
{
	if (useA)
		a[i] = 1.0f;
	if (useB) {
		intersector<triangle_data> x;
		ray r(float3(0), float3(0, 0, 1));
		o[i] = x.intersect(r, s).distance;
	}
}
kernel void k_ias(device float *a [[buffer(0), function_constant(useA)]],
                  instance_acceleration_structure s [[buffer(0), function_constant(useB)]],
                  device float *o [[buffer(1)]],
                  uint i [[thread_position_in_grid]])
{
	if (useA)
		a[i] = 1.0f;
	if (useB) {
		intersector<triangle_data, instancing> x;
		ray r(float3(0), float3(0, 0, 1));
		o[i] = x.intersect(r, s).distance;
	}
}
