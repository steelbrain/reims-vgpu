// The shell tests' post-tessellation vertex functions (Makefile, RGFX_PATCH): one [[patch]] of
// each type with and without a control point count, a counted one up to Metal's most (32), and
// functions of every other stage (vertex, fragment, kernel, visible, mesh, object and
// intersection), which state no patch.
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;

struct ControlPoint {
	float4 position [[attribute(0)]];
};

[[patch(quad, 4)]] vertex float4 patch_quad4(
	patch_control_point<ControlPoint> cp [[stage_in]],
	float2 uv [[position_in_patch]])
{
	return cp[0].position + float4(uv, 0, 0);
}

[[patch(quad)]] vertex float4 patch_quad_any(
	patch_control_point<ControlPoint> cp [[stage_in]],
	float2 uv [[position_in_patch]])
{
	return cp[0].position + float4(uv, 0, 0);
}

[[patch(quad, 32)]] vertex float4 patch_quad32(
	patch_control_point<ControlPoint> cp [[stage_in]],
	float2 uv [[position_in_patch]])
{
	return cp[0].position + float4(uv, 0, 0);
}

[[patch(triangle)]] vertex float4 patch_triangle_any(
	patch_control_point<ControlPoint> cp [[stage_in]],
	float3 uv [[position_in_patch]])
{
	return cp[0].position + float4(uv, 0);
}

[[patch(triangle, 3)]] vertex float4 patch_triangle3(
	patch_control_point<ControlPoint> cp [[stage_in]],
	float3 uv [[position_in_patch]])
{
	return cp[0].position + float4(uv, 0);
}

[[patch(quad)]] vertex float4 patch_quad_no_points(float2 uv [[position_in_patch]])
{
	return float4(uv, 0, 1);
}

vertex float4 plain_vertex(uint v [[vertex_id]])
{
	return float4(float(v));
}

fragment float4 plain_fragment()
{
	return float4(1);
}

kernel void plain_kernel(device float *out [[buffer(0)]])
{
	out[0] = 1;
}

[[visible]] float plain_visible(float x)
{
	return x * 2.0f;
}

struct MeshVertex {
	float4 position [[position]];
};

using TriangleMesh = metal::mesh<MeshVertex, void, 3, 1, metal::topology::triangle>;

[[mesh]] void plain_mesh(TriangleMesh output, uint tid [[thread_index_in_threadgroup]])
{
	if (tid == 0) {
		output.set_primitive_count(1);
		output.set_index(0, 0);
		output.set_index(1, 1);
		output.set_index(2, 2);
	}
	if (tid < 3) {
		MeshVertex v;
		v.position = float4(float(tid), 0.0f, 0.0f, 1.0f);
		output.set_vertex(tid, v);
	}
}

[[object]] void plain_object(mesh_grid_properties props)
{
	props.set_threadgroups_per_grid(uint3(1, 1, 1));
}

using namespace metal::raytracing;

[[intersection(triangle, triangle_data)]] bool plain_intersection(
	float2 bary [[barycentric_coord]])
{
	return bary.x > 0.5f;
}
