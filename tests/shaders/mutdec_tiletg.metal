// For scenarios/mutgap_decoders.py: a tile function that goes through
// threadgroup memory 0, whose length the render encoder sets (0x9c): each
// thread of a 16x16 tile stores blue at its own index, and after a barrier
// writes the half4 its mirror thread stored into colour 0.
#include <metal_stdlib>
using namespace metal;
struct TileOut { half4 c [[color(0)]]; };
kernel void mutdec_tiletg(imageblock<TileOut, imageblock_layout_implicit> ib, threadgroup half4 *tg [[threadgroup(0)]],
                          ushort2 lid [[thread_position_in_threadgroup]]) {
    uint i = lid.y * 16 + lid.x;
    tg[i] = half4(0.0h, 0.0h, 1.0h, 1.0h);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TileOut t = ib.read(lid); t.c = tg[255 - i]; ib.write(t, lid);
}
