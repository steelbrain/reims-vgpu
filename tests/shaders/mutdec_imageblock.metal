// For scenarios/mutgap_decoders.py: a compute kernel with an explicit
// imageblock, whose size the compute encoder sets (0xd8,
// setImageblockWidth:height:); thread 0 writes the imageblock's width and
// height to buffer 0.
#include <metal_stdlib>
using namespace metal;
struct Px { float v; };
kernel void mutdec_imageblock(imageblock<Px, imageblock_layout_explicit> ib, device uint *out [[buffer(0)]],
                              ushort2 lid [[thread_position_in_threadgroup]]) {
    threadgroup_imageblock Px *p = ib.data(lid);
    p->v = float(lid.x + lid.y);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid.x == 0 && lid.y == 0) {
        out[0] = ib.get_width();
        out[1] = ib.get_height();
    }
}
