#include <metal_stdlib>
using namespace metal;
// A stage-in kernel: attribute 0 comes from the pipeline's stage-input
// descriptor (buffer 1), so the stage-in region decides what each thread reads.
struct StageIn { uint v [[attribute(0)]]; };
kernel void stage_in_k(StageIn in [[stage_in]], device uint *out [[buffer(0)]],
                       uint gid [[thread_position_in_grid]]) {
    out[gid] = in.v + 0x100;
}
