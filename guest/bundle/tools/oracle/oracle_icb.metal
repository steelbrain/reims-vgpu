// oracle_icb.metal: the shaders oracle_icb runs. Our own code.
#include <metal_stdlib>
using namespace metal;

struct IcbVOut {
	float4 pos [[position]];
};

// A rectangle (x0, y0, x1, y1 in NDC) as a 4-vertex triangle strip: rects[instance + vertex / 4],
// so an indexed draw's base vertex of 4 moves it one rectangle on.
vertex IcbVOut icb_vs(uint vid [[vertex_id]], uint iid [[instance_id]],
                      device const float4 *rects [[buffer(0)]])
{
	float4 r = rects[iid + (vid >> 2)];
	float2 p = float2((vid & 1) ? r.z : r.x, (vid & 2) ? r.w : r.y);
	IcbVOut o;
	o.pos = float4(p, 0.5, 1);
	return o;
}

fragment float4 icb_fs(device const float4 &color [[buffer(0)]])
{
	return color;
}

// A quad patch over rects[patch + instance].
[[patch(quad, 4)]] vertex IcbVOut icb_patch_vs(float2 uv [[position_in_patch]],
                                               uint pid [[patch_id]], uint iid [[instance_id]],
                                               device const float4 *rects [[buffer(0)]])
{
	float4 r = rects[pid + iid];
	IcbVOut o;
	o.pos = float4(mix(r.xy, r.zw, uv), 0.5, 1);
	return o;
}

kernel void icb_add(device uint *out [[buffer(0)]], device const uint &val [[buffer(1)]],
                    uint tid [[thread_position_in_grid]])
{
	out[tid] += val;
}

kernel void icb_mul(device uint *out [[buffer(0)]], device const uint &val [[buffer(1)]],
                    uint tid [[thread_position_in_grid]])
{
	out[tid] *= val;
}

// out[tid] = the grid position of the mirror thread in the threadgroup, through threadgroup memory.
kernel void icb_tg(device uint *out [[buffer(0)]], threadgroup uint *tg [[threadgroup(0)]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint tid [[thread_position_in_grid]], uint tsize [[threads_per_threadgroup]])
{
	tg[lid] = tid;
	threadgroup_barrier(mem_flags::mem_threadgroup);
	out[tid] = tg[tsize - 1 - lid] + 1000;
}

// Spins about n iterations, so a command buffer runs a while.
kernel void icb_spin(device atomic_uint *c [[buffer(0)]], device const uint &n [[buffer(1)]])
{
	for (uint i = 0; i < n; i++)
		atomic_fetch_add_explicit(c, 1, memory_order_relaxed);
}

struct IcbRenderArgs {
	command_buffer cmds [[id(0)]];
	render_pipeline_state pso [[id(1)]];
};

// Encodes render command i by modes[i]: 0 a draw of rects[i] in colors[i], 1 the same as an
// indexed draw (16-bit indices), 2 a reset, 3 nothing; +16 also sets the pipeline state from the
// argument buffer.
kernel void icb_encode_draw(device IcbRenderArgs &args [[buffer(0)]],
                            device const float4 *rects [[buffer(1)]],
                            device const float4 *colors [[buffer(2)]],
                            device const uint *modes [[buffer(3)]],
                            device const ushort *indices [[buffer(4)]],
                            uint i [[thread_position_in_grid]])
{
	uint m = modes[i];
	if ((m & 15) == 3)
		return;
	render_command cmd(args.cmds, i);
	if ((m & 15) == 2) {
		cmd.reset();
		return;
	}
	if (m & 16)
		cmd.set_render_pipeline_state(args.pso);
	cmd.set_vertex_buffer(rects + i, 0);
	cmd.set_fragment_buffer(colors + i, 0);
	if ((m & 15) == 1)
		cmd.draw_indexed_primitives(primitive_type::triangle_strip, 4, indices, 1, 0, 0);
	else
		cmd.draw_primitives(primitive_type::triangle_strip, 0, 4, 1, 0);
}

struct IcbComputeArgs {
	command_buffer cmds [[id(0)]];
	compute_pipeline_state pso [[id(1)]];
};

// Encodes compute command i: icb_add of vals[i] over out[i * 64 ..], one threadgroup of 64; or a
// reset where vals[i] is 0.
kernel void icb_encode_dispatch(device IcbComputeArgs &args [[buffer(0)]],
                                device uint *out [[buffer(1)]], device uint *vals [[buffer(2)]],
                                uint i [[thread_position_in_grid]])
{
	compute_command cmd(args.cmds, i);
	if (vals[i] == 0) {
		cmd.reset();
		return;
	}
	cmd.set_compute_pipeline_state(args.pso);
	cmd.set_kernel_buffer(out + i * 64, 0);
	cmd.set_kernel_buffer(vals + i, 1);
	cmd.concurrent_dispatch_threadgroups(uint3(1, 1, 1), uint3(64, 1, 1));
}
