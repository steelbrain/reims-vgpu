// For scenarios/mutgap_decoders.py: a tile function that samples tile texture 0
// through tile sampler 1 at level 0 and writes the texel into colour 0, so the
// sampler's LOD clamps (0xa0) decide the mip level.
#include <metal_stdlib>
using namespace metal;
struct TileOut { half4 c [[color(0)]]; };
kernel void mutdec_tilesample(imageblock<TileOut, imageblock_layout_implicit> ib, texture2d<half> t [[texture(0)]],
                              sampler s [[sampler(1)]], ushort2 lid [[thread_position_in_threadgroup]]) {
    TileOut o = ib.read(lid);
    o.c = t.sample(s, float2(0.5, 0.5), level(0.0));
    ib.write(o, lid);
}
