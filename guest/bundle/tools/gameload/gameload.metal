// gameload.metal: the shaders of gameload (gameload.m), compiled from source at run time on the
// device it runs on. Our own code.
#include <metal_stdlib>
using namespace metal;

// One frame's constants (gameload.m's GlFrame).
struct Frame {
	float4x4 view_proj;
	float4x4 light_view_proj;
	float4 light_dir;
	float time;
	uint objects;
	uint icb_objects;
	uint cicb_commands;
};

// One object (gameload.m's GlObject): 64 bytes.
struct Object {
	float4 pos_scale; // xyz position, w scale
	float4 color;
	float4 rot; // xyz axis, w angular speed
	uint material;
	uint pad0, pad1, pad2;
};

struct VIn {
	float3 pos [[attribute(0)]];
	float3 nrm [[attribute(1)]];
	float2 uv [[attribute(2)]];
};

struct VOut {
	float4 pos [[position]];
	float3 nrm;
	float2 uv;
	float4 shadow_pos;
	float4 color;
};

// One material in the materials' argument buffer.
struct Material {
	texture2d<float> albedo [[id(0)]];
	texture2d<float> detail [[id(1)]];
	constant float4 *params [[id(2)]];
};

// The scene pipelines' variants (alpha test, doubled detail, emissive), one per value.
constant int variant [[function_constant(0)]];

static float3 rotate(float3 p, float3 axis, float angle)
{
	float c = cos(angle), s = sin(angle);
	return p * c + cross(axis, p) * s + axis * dot(axis, p) * (1 - c);
}

static float4 place(float3 p, float3 n, Object o, float t, thread float3 &nw)
{
	float3 axis = normalize(o.rot.xyz + float3(1e-4));
	float a = t * o.rot.w;
	nw = rotate(n, axis, a);
	return float4(rotate(p * o.pos_scale.w, axis, a) + o.pos_scale.xyz, 1);
}

// A CPU draw selects its object by the objects buffer's offset (instance 0); a draw of the
// indirect command buffer by its base instance (the buffer at offset 0).
vertex VOut scene_vs(VIn v [[stage_in]], constant Frame &f [[buffer(1)]],
                     device const Object *objs [[buffer(2)]], uint iid [[instance_id]])
{
	Object o = objs[iid];
	float3 n;
	float4 w = place(v.pos, v.nrm, o, f.time, n);
	VOut out;
	out.pos = f.view_proj * w;
	out.nrm = n;
	out.uv = v.uv;
	out.shadow_pos = f.light_view_proj * w;
	out.color = o.color;
	return out;
}

fragment float4 scene_fs(VOut in [[stage_in]], device const Material &mat [[buffer(0)]],
                         constant Frame &f [[buffer(1)]], depth2d<float> shadow [[texture(0)]],
                         sampler smp [[sampler(0)]], sampler cmp [[sampler(1)]])
{
	float4 a = mat.albedo.sample(smp, in.uv);
	float4 d = mat.detail.sample(smp, in.uv * (variant == 2 ? 8.0 : 4.0));
	if (variant == 1 && a.a * d.a < 0.02)
		discard_fragment();
	float3 sp = in.shadow_pos.xyz / in.shadow_pos.w;
	float2 suv = sp.xy * float2(0.5, -0.5) + 0.5;
	float lit = shadow.sample_compare(cmp, suv, sp.z - 0.002);
	float ndl = saturate(dot(normalize(in.nrm), -f.light_dir.xyz));
	float3 c = a.rgb * d.rgb * in.color.rgb * mat.params[0].rgb * (0.15 + 0.85 * ndl * lit);
	if (variant == 3)
		c += a.rgb * mat.params[0].a;
	return float4(c * 4.0, 1);
}

// The GPU-driven pass's fragment: an indirect command buffer's pipeline takes textures only
// through argument buffers, so this one samples the material's alone, with a sampler of its own.
fragment float4 icb_fs(VOut in [[stage_in]], device const Material &mat [[buffer(0)]],
                       constant Frame &f [[buffer(1)]])
{
	constexpr sampler s(filter::linear, mip_filter::linear, address::repeat);
	float4 a = mat.albedo.sample(s, in.uv);
	float4 d = mat.detail.sample(s, in.uv * 4.0);
	float ndl = saturate(dot(normalize(in.nrm), -f.light_dir.xyz));
	return float4(a.rgb * d.rgb * in.color.rgb * mat.params[0].rgb * (0.15 + 0.85 * ndl) * 4.0, 1);
}

vertex float4 shadow_vs(VIn v [[stage_in]], constant Frame &f [[buffer(1)]],
                        device const Object *objs [[buffer(2)]], uint iid [[instance_id]])
{
	float3 n;
	return f.light_view_proj * place(v.pos, v.nrm, objs[iid], f.time, n);
}

// The GPU-driven pass: one thread an object, which encodes its draw into the indirect command
// buffer where it lies in the view, and resets its command otherwise.
struct IcbArgs {
	command_buffer cmds [[id(0)]];
};

kernel void icb_encode(device IcbArgs &args [[buffer(0)]], device const Object *objs [[buffer(1)]],
                       constant Frame &f [[buffer(2)]], device ushort *indices [[buffer(3)]],
                       uint i [[thread_position_in_grid]])
{
	if (i >= f.icb_objects)
		return;
	render_command cmd(args.cmds, i);
	Object o = objs[i];
	float4 c = f.view_proj * float4(o.pos_scale.xyz, 1);
	float r = o.pos_scale.w * 2.0;
	bool seen = c.w > 0.1 && all(abs(c.xy) < float2(c.w + r));
	if (seen)
		cmd.draw_indexed_primitives(primitive_type::triangle, 36, indices, 1, 0, i);
	else
		cmd.reset();
}

// The same, for an indirect command buffer that inherits no buffers (gameload -b): each drawn
// command sets the vertex and fragment buffers the pipeline reads.
kernel void icb_encode_bound(device IcbArgs &args [[buffer(0)]], device const Object *objs [[buffer(1)]],
                             constant Frame &f [[buffer(2)]], device ushort *indices [[buffer(3)]],
                             device const float *mesh [[buffer(4)]], device const uchar *materials [[buffer(5)]],
                             uint i [[thread_position_in_grid]])
{
	if (i >= f.icb_objects)
		return;
	render_command cmd(args.cmds, i);
	Object o = objs[i];
	float4 c = f.view_proj * float4(o.pos_scale.xyz, 1);
	float r = o.pos_scale.w * 2.0;
	bool seen = c.w > 0.1 && all(abs(c.xy) < float2(c.w + r));
	if (seen) {
		cmd.set_vertex_buffer(mesh, 0);
		cmd.set_vertex_buffer(&f, 1);
		cmd.set_vertex_buffer(objs, 2);
		cmd.set_fragment_buffer(materials, 0);
		cmd.set_fragment_buffer(&f, 1);
		cmd.draw_indexed_primitives(primitive_type::triangle, 36, indices, 1, 0, i);
	} else {
		cmd.reset();
	}
}

// The GPU-driven compute pass (cicb): a kernel encodes CICB_COMMANDS concurrent dispatches into
// a compute indirect command buffer, which a later compute encoder executes. Command i dispatches
// 1 + i % 4 threadgroups of 64 threads, and every fifth command (i % 5 == 4) is reset, so the sum
// the dispatches make (cicb_work) holds each command's grid and the encoder's bindings, which
// gameload.m checks against its own.
struct CicbArgs {
	command_buffer cmds [[id(0)]];
};

kernel void cicb_encode(device CicbArgs &args [[buffer(0)]], constant Frame &f [[buffer(1)]],
                        uint i [[thread_position_in_grid]])
{
	if (i >= f.cicb_commands)
		return;
	compute_command cmd(args.cmds, i);
	if (i % 5 == 4)
		cmd.reset();
	else
		cmd.concurrent_dispatch_threadgroups(uint3(1 + i % 4, 1, 1), uint3(64, 1, 1));
}

// The same, for a compute indirect command buffer that inherits no buffers (gameload -b): each
// command sets the three buffers cicb_work reads, its input 64 words further on than the last's.
kernel void cicb_encode_bound(device CicbArgs &args [[buffer(0)]], constant Frame &f [[buffer(1)]],
                              device const uint *in [[buffer(2)]], device uint *sum [[buffer(3)]],
                              device const uint *mul [[buffer(4)]], uint i [[thread_position_in_grid]])
{
	if (i >= f.cicb_commands)
		return;
	compute_command cmd(args.cmds, i);
	if (i % 5 == 4) {
		cmd.reset();
		return;
	}
	cmd.set_kernel_buffer(in + i * 64, 0);
	cmd.set_kernel_buffer(sum, 1);
	cmd.set_kernel_buffer(mul, 2);
	cmd.concurrent_dispatch_threadgroups(uint3(1 + i % 4, 1, 1), uint3(64, 1, 1));
}

// One dispatch of the compute indirect command buffer: each thread's input word times mul, summed
// over its SIMD group and added into the sum.
kernel void cicb_work(device const uint *in [[buffer(0)]], device atomic_uint *sum [[buffer(1)]],
                      constant uint &mul [[buffer(2)]], uint g [[thread_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]])
{
	uint v = simd_sum(in[g] * mul);
	if (lane == 0)
		atomic_fetch_add_explicit(sum, v, memory_order_relaxed);
}

// Post-processing: a luminance histogram, a bloom chain down and up, and a tone map.
kernel void hist_clear(device atomic_uint *hist [[buffer(0)]], uint i [[thread_position_in_grid]])
{
	if (i < 64)
		atomic_store_explicit(&hist[i], 0, memory_order_relaxed);
}

kernel void luminance(texture2d<float, access::read> src [[texture(0)]],
                      device atomic_uint *hist [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]], uint lid [[thread_index_in_threadgroup]])
{
	threadgroup atomic_uint local[64];
	if (lid < 64)
		atomic_store_explicit(&local[lid], 0, memory_order_relaxed);
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (all(gid < uint2(src.get_width(), src.get_height()))) {
		float3 c = src.read(gid).rgb;
		float l = dot(c, float3(0.2126, 0.7152, 0.0722));
		uint bin = uint(clamp(log2(l + 1e-4) * 4.0 + 32.0, 0.0, 63.0));
		atomic_fetch_add_explicit(&local[bin], 1, memory_order_relaxed);
	}
	threadgroup_barrier(mem_flags::mem_threadgroup);
	if (lid < 64)
		atomic_fetch_add_explicit(&hist[lid], atomic_load_explicit(&local[lid], memory_order_relaxed),
		                          memory_order_relaxed);
}

kernel void bloom_down(texture2d<float, access::sample> src [[texture(0)]],
                       texture2d<float, access::write> dst [[texture(1)]], sampler smp [[sampler(0)]],
                       constant float &threshold [[buffer(0)]], uint2 gid [[thread_position_in_grid]])
{
	uint2 size = uint2(dst.get_width(), dst.get_height());
	if (any(gid >= size))
		return;
	float2 uv = (float2(gid) + 0.5) / float2(size);
	float2 px = 1.0 / float2(src.get_width(), src.get_height());
	float4 c = src.sample(smp, uv + px * float2(-1, -1)) + src.sample(smp, uv + px * float2(1, -1)) +
	           src.sample(smp, uv + px * float2(-1, 1)) + src.sample(smp, uv + px * float2(1, 1));
	dst.write(max(c * 0.25 - threshold, 0.0), gid);
}

kernel void bloom_up(texture2d<float, access::sample> lo [[texture(0)]],
                     texture2d<float, access::sample> hi [[texture(1)]],
                     texture2d<float, access::write> dst [[texture(2)]], sampler smp [[sampler(0)]],
                     uint2 gid [[thread_position_in_grid]])
{
	uint2 size = uint2(dst.get_width(), dst.get_height());
	if (any(gid >= size))
		return;
	float2 uv = (float2(gid) + 0.5) / float2(size);
	dst.write(lo.sample(smp, uv) * 0.6 + hi.sample(smp, uv), gid);
}

kernel void tonemap(texture2d<float, access::read> hdr [[texture(0)]],
                    texture2d<float, access::sample> bloom [[texture(1)]],
                    texture2d<float, access::write> out [[texture(2)]], sampler smp [[sampler(0)]],
                    device const uint *hist [[buffer(0)]], uint2 gid [[thread_position_in_grid]])
{
	uint2 size = uint2(out.get_width(), out.get_height());
	if (any(gid >= size))
		return;
	uint n = 0, s = 0;
	for (uint i = 0; i < 64; i++) {
		n += hist[i];
		s += hist[i] * i;
	}
	float mean_bin = n ? float(s) / float(n) : 32.0;
	float exposure = 0.5 / exp2((mean_bin - 32.0) / 4.0);
	float2 uv = (float2(gid) + 0.5) / float2(size);
	float3 c = hdr.read(gid).rgb + bloom.sample(smp, uv).rgb * 0.3;
	c *= exposure;
	out.write(float4(c / (1.0 + c), 1), gid);
}

// The UI's quads and the final composite, both a full-screen-triangle shape.
struct QOut {
	float4 pos [[position]];
	float2 uv;
};

vertex QOut ui_vs(uint vid [[vertex_id]], uint iid [[instance_id]], constant float4 *rects [[buffer(0)]])
{
	float4 r = rects[iid];
	float2 t = float2(vid & 1, (vid >> 1) & 1);
	QOut o;
	o.pos = float4(mix(r.xy, r.zw, t), 0, 1);
	o.uv = t;
	return o;
}

fragment float4 ui_fs(QOut in [[stage_in]])
{
	return float4(in.uv, 0.5, 0.6);
}

vertex QOut fullscreen_vs(uint vid [[vertex_id]])
{
	float2 p = float2((vid << 1) & 2, vid & 2);
	QOut o;
	o.pos = float4(p * 2 - 1, 0, 1);
	o.uv = float2(p.x, 1 - p.y);
	return o;
}

fragment float4 composite_fs(QOut in [[stage_in]], texture2d<float> scene [[texture(0)]],
                             texture2d<float> ui [[texture(1)]], texture2d<float> overlay [[texture(2)]],
                             sampler smp [[sampler(0)]])
{
	float4 s = scene.sample(smp, in.uv);
	float4 u = ui.sample(smp, in.uv);
	float4 o = overlay.sample(smp, in.uv * 4);
	return float4(mix(s.rgb, u.rgb, u.a) + o.rgb * 0.02, 1);
}
