// The shader of pipeline_reflection_check's c_atomic_texture case: a texture of 64-bit atomics,
// which the language takes from Metal 3.1 on, so it is a library of its own.
#include <metal_stdlib>
using namespace metal;

kernel void k_atomic_texture(texture2d<ulong, access::read_write> atomics [[texture(0)]],
                             device float *out [[buffer(0)]], uint gid [[thread_position_in_grid]])
{
	out[gid] = float(atomics.read(uint2(gid)).x);
}
