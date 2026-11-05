// oracle_bundle's dynamic library (oracle_bundle.m, case libtype), built with -dynamiclib.
#include <metal_stdlib>
using namespace metal;

float div_dyn(float x)
{
	return x * 3.0f;
}
