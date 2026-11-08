// The tests' kernel whose argument buffer the runtime lays out as Metal does
// (`argument_layouts_are_metals`): a texture, a
// pointer to an argument buffer of a texture and a sampler, a sampler, a float4 and a pointer to
// plain floats, which nests no layout.
#include <metal_stdlib>
using namespace metal;

struct Inner {
    texture2d<float> t [[id(0)]];
    sampler s [[id(1)]];
};

struct Args {
    texture2d<float> tex [[id(0)]];
    device Inner *inner [[id(1)]];
    sampler smp [[id(2)]];
    float4 c [[id(3)]];
    device float *plain [[id(4)]];
};

kernel void reims_vgpu_argument_probe(device Args &args [[buffer(0)]],
                                      device float *out [[buffer(1)]]) {
    out[0] = args.c.x + args.tex.read(uint2(0)).x + args.inner->t.read(uint2(0)).x + args.plain[0];
}
