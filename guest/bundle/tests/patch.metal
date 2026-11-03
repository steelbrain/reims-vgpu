// The shell tests' post-tessellation vertex functions (Makefile, RGFX_PATCH): one [[patch]] of
// each type with and without a control point count, a counted one up to Metal's most (32), and
// functions of every other stage, which state no patch.
#include <metal_stdlib>
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
