// The [[stage_in]] kernels of tests/gl_tests.mm's stage input oracle: each stores its stage input's
// one attribute, as this Mac's Metal fetches it under the pipeline's stage input descriptor, at
// out + 16 * (x + y * width). Our own source; the Makefile builds it, and nothing of it is
// committed but this.
#include <metal_stdlib>
using namespace metal;

#define ORACLE(name, T)                                                                            \
	struct name##_in {                                                                             \
		T a [[attribute(0)]];                                                                      \
	};                                                                                             \
	kernel void name(name##_in in [[stage_in]], device char *out [[buffer(1)]],                    \
	                 uint2 i [[thread_position_in_grid]], uint2 g [[threads_per_grid]])            \
	{                                                                                              \
		*(device T *)(out + 16 * (i.x + i.y * g.x)) = in.a;                                        \
	}

ORACLE(f1, float)
ORACLE(f2, float2)
ORACLE(f3, float3)
ORACLE(f4, float4)
ORACLE(i1, int)
ORACLE(i4, int4)
ORACLE(u4, uint4)
