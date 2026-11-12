// oracle_fnptr's functions: visible functions a kernel and a fragment function call through a
// visible function table, bound at a buffer index or held in an argument buffer.
#include <metal_stdlib>
using namespace metal;

[[visible]] int fp_add1(int x)
{
	return x + 1;
}

[[visible]] int fp_mul2(int x)
{
	return x * 2;
}

// Never linked: its handle is asked of a pipeline that does not link it.
[[visible]] int fp_neg(int x)
{
	return -x;
}

// out[i] = table[which[i]](i + 10)
kernel void fp_kernel(visible_function_table<int(int)> table [[buffer(0)]],
                      device const uint *which [[buffer(1)]],
                      device int *out [[buffer(2)]],
                      uint i [[thread_position_in_grid]])
{
	out[i] = table[which[i]](int(i) + 10);
}

struct FpArgs {
	visible_function_table<int(int)> table;
};

// The same through a table an argument buffer holds.
kernel void fp_arg_kernel(constant FpArgs &args [[buffer(0)]],
                          device const uint *which [[buffer(1)]],
                          device int *out [[buffer(2)]],
                          uint i [[thread_position_in_grid]])
{
	out[i] = args.table[which[i]](int(i) + 10);
}

// The table's size, as the shader reads it.
kernel void fp_size_kernel(visible_function_table<int(int)> table [[buffer(0)]],
                           device uint *out [[buffer(2)]],
                           uint i [[thread_position_in_grid]])
{
	out[i] = table.size();
}

[[visible]] float4 fp_red(float2 p)
{
	return float4(1, 0, 0, 1);
}

[[visible]] float4 fp_green(float2 p)
{
	return float4(0, 1, 0, 1);
}

struct FpOut {
	float4 position [[position]];
};

vertex FpOut fp_vs(uint vid [[vertex_id]])
{
	const float2 corners[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
	FpOut out;
	out.position = float4(corners[vid], 0, 1);
	return out;
}

fragment float4 fp_fs(FpOut in [[stage_in]],
                      visible_function_table<float4(float2)> table [[buffer(0)]],
                      constant uint &which [[buffer(1)]])
{
	return table[which](in.position.xy);
}
