// For scenarios/mutgap_decoders.py: a fragment function that stores 1 into
// fragment buffer 0 and returns green. With mutdec_vread.metal: between two
// draws of one pass, only a fragment-to-vertex memory barrier (0x17) makes the
// second draw's vertex stage see the first draw's fragment store.
#include <metal_stdlib>
using namespace metal;
fragment float4 mutdec_fwrite(device uint *flag [[buffer(0)]]) {
    flag[0] = 1;
    return float4(0.0, 1.0, 0.0, 1.0);
}
