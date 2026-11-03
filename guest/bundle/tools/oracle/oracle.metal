// oracle.metal: the shaders the oracle probes run, compiled into oracle.metallib (README.md).
#include <metal_stdlib>
using namespace metal;

// Each thread spins `iters` steps of an LCG, then writes `value + (x & mask)` to out[i]; the
// host passes mask 0, so every word ends `value`, written only after the loop.
kernel void slow_write(device uint *out [[buffer(0)]],
                       constant uint &iters [[buffer(1)]],
                       constant uint &value [[buffer(2)]],
                       constant uint &mask [[buffer(3)]],
                       uint i [[thread_position_in_grid]])
{
	uint x = i;
	for (uint n = 0; n < iters; n++)
		x = x * 1664525u + 1013904223u;
	out[i] = value + (x & mask);
}

kernel void copy_u32(device const uint *in [[buffer(0)]],
                     device uint *out [[buffer(1)]],
                     uint i [[thread_position_in_grid]])
{
	out[i] = in[i];
}

// out[i] = in[i] + 1, so a chain of these shows which write each read saw.
kernel void add_one(device const uint *in [[buffer(0)]],
                    device uint *out [[buffer(1)]],
                    uint i [[thread_position_in_grid]])
{
	out[i] = in[i] + 1;
}

// One thread: polls flag[0] (atomic, relaxed) until it is nonzero or `max_iters` polls passed;
// writes the poll count to result[0] and the flag it saw to result[1].
kernel void spin_atomic(device atomic_uint *flag [[buffer(0)]],
                        device uint *result [[buffer(1)]],
                        constant uint &max_iters [[buffer(2)]],
                        uint i [[thread_position_in_grid]])
{
	if (i != 0)
		return;
	uint n = 0, seen = 0;
	for (; n < max_iters; n++) {
		seen = atomic_load_explicit(flag, memory_order_relaxed);
		if (seen != 0)
			break;
	}
	result[0] = n;
	result[1] = seen;
}

// The same with plain volatile loads.
kernel void spin_plain(device volatile uint *flag [[buffer(0)]],
                       device uint *result [[buffer(1)]],
                       constant uint &max_iters [[buffer(2)]],
                       uint i [[thread_position_in_grid]])
{
	if (i != 0)
		return;
	uint n = 0, seen = 0;
	for (; n < max_iters; n++) {
		seen = flag[0];
		if (seen != 0)
			break;
	}
	result[0] = n;
	result[1] = seen;
}

// One thread: stores 1..steps into counter[0] (atomic, relaxed), `spin` LCG steps apart.
kernel void progress(device atomic_uint *counter [[buffer(0)]],
                     constant uint &steps [[buffer(1)]],
                     constant uint &spin [[buffer(2)]],
                     device uint *sink [[buffer(3)]],
                     uint i [[thread_position_in_grid]])
{
	if (i != 0)
		return;
	uint x = 1;
	for (uint s = 1; s <= steps; s++) {
		for (uint n = 0; n < spin; n++)
			x = x * 1664525u + 1013904223u;
		atomic_store_explicit(counter, s, memory_order_relaxed);
	}
	sink[0] = x;
}

struct VOut {
	float4 pos [[position]];
};

// A rectangle (x0, y0, x1, y1 in NDC) as a 4-vertex triangle strip, at depth `rect_z`.
vertex VOut quad_vs(uint vid [[vertex_id]], constant float4 &rect [[buffer(0)]],
                    constant float &z [[buffer(1)]])
{
	float2 p = float2((vid & 1) ? rect.z : rect.x, (vid & 2) ? rect.w : rect.y);
	VOut o;
	o.pos = float4(p, z, 1);
	return o;
}

fragment float4 solid_fs(constant float4 &color [[buffer(0)]])
{
	return color;
}

// Framebuffer fetch: reads the attachment's current value.
fragment float4 fetch_fs(float4 dst [[color(0)]], constant float4 &color [[buffer(0)]])
{
	return dst * 0.5 + color;
}

// Raster order groups over a read-write texture.
fragment float4 rog_fs(VOut in [[stage_in]],
                       texture2d<float, access::read_write> t [[texture(0), raster_order_group(0)]],
                       constant float4 &color [[buffer(0)]])
{
	uint2 p = uint2(in.pos.xy);
	t.write(t.read(p) + color, p);
	return color;
}

// A tile function.
kernel void tile_k(ushort2 tid [[thread_position_in_threadgroup]])
{
}

// oracle_ycbcr: the texel read() reads at the middle of each 32 x 32 quadrant of a 64 x 64
// texture, top left, top right, bottom left and bottom right, into out[0..3].
kernel void ycbcr_read(texture2d<float, access::read> t [[texture(0)]],
                       device float4 *out [[buffer(0)]],
                       uint i [[thread_position_in_grid]])
{
	uint2 at = uint2((i & 1) ? 48 : 16, (i & 2) ? 48 : 16);
	out[i] = t.read(at);
}

// oracle_sampler: params[i].x picks the LOD's source (0: level(params[i].y); 1: gradient2d whose
// LOD is params[i].y), params[i].zw is the coordinate; out[i] is the sample.
kernel void sampler_probe(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                          constant float4 *params [[buffer(0)]], device float4 *out [[buffer(1)]],
                          uint i [[thread_position_in_grid]])
{
	float4 p = params[i];
	float g = exp2(p.y) / float(t.get_width());
	if (p.x == 0.0)
		out[i] = t.sample(s, p.zw, level(p.y));
	else
		out[i] = t.sample(s, p.zw, gradient2d(float2(g, 0.0), float2(0.0, g)));
}

struct SamplerVOut {
	float4 pos [[position]];
};

// A triangle covering the target.
vertex SamplerVOut sampler_vs(uint v [[vertex_id]])
{
	float2 c = float2((v << 1) & 2, v & 2);
	SamplerVOut o;
	o.pos = float4(c * 2.0 - 1.0, 0.0, 1.0);
	return o;
}

// Implicit LOD: the coordinate steps exp2(lod) / width a pixel in x and in y.
fragment float4 sampler_fs(SamplerVOut in [[stage_in]], texture2d<float> t [[texture(0)]],
                           sampler s [[sampler(0)]], constant float &lod [[buffer(0)]])
{
	float k = exp2(lod) / float(t.get_width());
	return t.sample(s, in.pos.xy * k);
}
