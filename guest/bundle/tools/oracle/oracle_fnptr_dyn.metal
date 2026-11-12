// oracle_fnptr's dynamic library (built with -dynamiclib, install name
// @executable_path/liboracle_fnptr_dyn.metallib): the function oracle_fnptr_link.metal calls.
#include <metal_stdlib>
using namespace metal;

int fpd_triple(int x)
{
	return x * 3;
}
