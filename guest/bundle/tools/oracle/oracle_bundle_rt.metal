// oracle_bundle's intersection function (oracle_bundle.m, case functypes).
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

[[intersection(triangle, triangle_data)]] bool div_intersection(float2 bary [[barycentric_coord]])
{
	return bary.x > 0.5f;
}
