// For scenarios/mutgap_decoders.py: a tile function that writes the half4 its
// thread finds in threadgroup memory 0 into colour 0 (see
// mutdec_tgwrite_blue.metal).
#include <metal_stdlib>
using namespace metal;
struct TileOut { half4 c [[color(0)]]; };
kernel void mutdec_tgread(imageblock<TileOut, imageblock_layout_implicit> ib, threadgroup half4 *tg [[threadgroup(0)]],
                          ushort2 lid [[thread_position_in_threadgroup]]) {
    TileOut o = ib.read(lid); o.c = tg[lid.y * 16 + lid.x]; ib.write(o, lid);
}
