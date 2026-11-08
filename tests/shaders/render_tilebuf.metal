// Render suite: a tile function that writes tile buffer 0's first half4
// into colour 0 of every pixel of the tile.
#include <metal_stdlib>
using namespace metal;
struct TileOut { half4 c [[color(0)]]; };
kernel void render_tilebuf(imageblock<TileOut, imageblock_layout_implicit> ib, constant half4 *c [[buffer(0)]],
                           ushort2 lid [[thread_position_in_threadgroup]]) {
    TileOut t = ib.read(lid); t.c = c[0]; ib.write(t, lid);
}
