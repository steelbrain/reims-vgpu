// For scenarios/mutgap_decoders.py: a tile function that stores blue into its
// 256 half4s of threadgroup memory 0 (one per thread of a 16x16 tile) and
// leaves the imageblock alone. With mutdec_tgread.metal, it shows where 0x9c
// placed each dispatch's threadgroup memory in the pass's.
#include <metal_stdlib>
using namespace metal;
struct TileOut { half4 c [[color(0)]]; };
kernel void mutdec_tgwrite_blue(imageblock<TileOut, imageblock_layout_implicit> ib,
                             threadgroup half4 *tg [[threadgroup(0)]],
                             ushort2 lid [[thread_position_in_threadgroup]]) {
    tg[lid.y * 16 + lid.x] = half4(0.0h, 0.0h, 1.0h, 1.0h);
}
