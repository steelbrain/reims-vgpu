// oracle_ratemap's pass: one triangle over the whole target, whose fragment writes its position
// and the map's decode of it to the screen, and counts itself.
#include <metal_stdlib>
using namespace metal;

struct RatemapOut {
	float4 position [[position]];
};

vertex RatemapOut ratemap_vs(uint vid [[vertex_id]])
{
	const float2 corners[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
	RatemapOut out;
	out.position = float4(corners[vid], 0, 1);
	return out;
}

fragment float4 ratemap_fs(RatemapOut in [[stage_in]],
                           constant rasterization_rate_map_data &data [[buffer(0)]],
                           device atomic_uint *count [[buffer(1)]])
{
	rasterization_rate_map_decoder map(data);
	const float2 screen = map.map_physical_to_screen_coordinates(in.position.xy);
	atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
	return float4(in.position.xy, screen);
}
