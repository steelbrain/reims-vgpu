#include <metal_stdlib>
using namespace metal;
struct TileOut { half4 c [[color(0)]]; };
kernel void tile_k(imageblock<TileOut, imageblock_layout_implicit> ib, ushort2 lid [[thread_position_in_threadgroup]]) {
    TileOut t = ib.read(lid); t.c = half4(0.0h, 1.0h, 0.0h, 1.0h); ib.write(t, lid);
}
