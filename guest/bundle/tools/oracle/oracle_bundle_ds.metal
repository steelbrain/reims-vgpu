// oracle_bundle's depth/stencil argument buffer (oracle_bundle.m, case argbuf.ds.function): a
// kernel whose argument buffer holds a depth/stencil state, where the language has the type.
#include <metal_stdlib>
using namespace metal;

struct DivDsArgs {
	depth_stencil_state ds [[id(0)]];
	device float *buf [[id(1)]];
};

kernel void div_ds(constant DivDsArgs &args [[buffer(0)]], device float *out [[buffer(1)]],
                   uint i [[thread_position_in_grid]])
{
	out[i] = args.buf[i];
}
