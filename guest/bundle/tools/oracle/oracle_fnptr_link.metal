// oracle_fnptr's kernel linked against its dynamic library (-loracle_fnptr_dyn):
// out[i] = fpd_triple(i + 10), which the library exports.
#include <metal_stdlib>
using namespace metal;

extern int fpd_triple(int x);

kernel void fpd_kernel(device int *out [[buffer(0)]], uint i [[thread_position_in_grid]])
{
	out[i] = fpd_triple(int(i) + 10);
}
