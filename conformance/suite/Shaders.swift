import Metal
import Foundation
import IOSurface

// ---------------------------------------------------------------------------
// Shaders. Compiled from source at run time on purpose: that exercises the same
// shader path the guest's own apps take, rather than a pre-built archive.
// ---------------------------------------------------------------------------

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// Exact texel fetch. `read` bypasses the sampler, so a mismatch here is about
// the texture's memory interpretation and nothing else.
kernel void read_texels(texture2d<float, access::read> tex [[texture(0)]],
                        device uint *out [[buffer(0)]],
                        constant uint &width [[buffer(1)]],
                        constant uint2 &extent [[buffer(3)]],
                        device uint *ran [[buffer(4)]],
                        uint2 gid [[thread_position_in_grid]]) {
    // Before the grid guard, so this says the kernel was reached
    // rather than that some thread was in range. Every thread
    // writes the same value; the race is benign and the point is
    // that a dispatch nothing refused cannot leave it zero.
    ran[0] = 1u;
    if (gid.x >= extent.x || gid.y >= extent.y) { return; }
    float4 v = tex.read(gid);
    uint r = uint(round(v.r * 255.0));
    uint g = uint(round(v.g * 255.0));
    uint b = uint(round(v.b * 255.0));
    uint a = uint(round(v.a * 255.0));
    out[gid.y * width + gid.x] = (a << 24) | (b << 16) | (g << 8) | r;
}

// The sampler path, with nearest filtering and unnormalized coordinates, so the
// result is still exactly one texel and any difference from `read_texels` is
// the sampler/view rather than the memory.
kernel void sample_texels(texture2d<float, access::sample> tex [[texture(0)]],
                          device uint *out [[buffer(0)]],
                          constant uint &width [[buffer(1)]],
                          constant uint2 &extent [[buffer(3)]],
                          device uint *ran [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    // Before the grid guard, so this says the kernel was reached
    // rather than that some thread was in range. Every thread
    // writes the same value; the race is benign and the point is
    // that a dispatch nothing refused cannot leave it zero.
    ran[0] = 1u;
    if (gid.x >= extent.x || gid.y >= extent.y) { return; }
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_edge);
    float4 v = tex.sample(s, float2(gid.x + 0.5, gid.y + 0.5));
    uint r = uint(round(v.r * 255.0));
    uint g = uint(round(v.g * 255.0));
    uint b = uint(round(v.b * 255.0));
    uint a = uint(round(v.a * 255.0));
    out[gid.y * width + gid.x] = (a << 24) | (b << 16) | (g << 8) | r;
}

// Read one explicit mip level.
// `coord::pixel` may not be combined with a mip filter, and `level` is the name
// of the LOD constructor, so the uniform is `lod` and the coordinates are
// normalized against this level's own dimensions.
kernel void read_level(texture2d<float, access::sample> tex [[texture(0)]],
                       device uint *out [[buffer(0)]],
                       constant uint &width [[buffer(1)]],
                       constant uint &lod [[buffer(2)]],
                       constant uint2 &extent [[buffer(3)]],
                       device uint *ran [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    // Before the grid guard, so this says the kernel was reached
    // rather than that some thread was in range. Every thread
    // writes the same value; the race is benign and the point is
    // that a dispatch nothing refused cannot leave it zero.
    ran[0] = 1u;
    if (gid.x >= extent.x || gid.y >= extent.y) { return; }
    constexpr sampler s(filter::nearest, mip_filter::nearest, address::clamp_to_edge);
    float2 dim = float2(max(1u, tex.get_width() >> lod), max(1u, tex.get_height() >> lod));
    float2 uv = (float2(gid) + 0.5f) / dim;
    float4 v = tex.sample(s, uv, level(float(lod)));
    uint r = uint(round(v.r * 255.0));
    uint g = uint(round(v.g * 255.0));
    uint b = uint(round(v.b * 255.0));
    uint a = uint(round(v.a * 255.0));
    out[gid.y * width + gid.x] = (a << 24) | (b << 16) | (g << 8) | r;
}

// The same level, fetched rather than sampled. `read(coord, lod)` names the
// level in the fetch itself, with no sampler and no LOD computation, so a
// device that returns level 0 here has not got the level's *bytes*, while one
// that passes here and fails `read_level` has the bytes and is losing the
// explicit LOD on the sampling path. That is the whole difference between a
// residency bug and a translation bug and nothing else separates them.
kernel void fetch_level(texture2d<float, access::read> tex [[texture(0)]],
                        device uint *out [[buffer(0)]],
                        constant uint &width [[buffer(1)]],
                        constant uint &lod [[buffer(2)]],
                        constant uint2 &extent [[buffer(3)]],
                        device uint *ran [[buffer(4)]],
                        uint2 gid [[thread_position_in_grid]]) {
    // Before the grid guard, so this says the kernel was reached
    // rather than that some thread was in range. Every thread
    // writes the same value; the race is benign and the point is
    // that a dispatch nothing refused cannot leave it zero.
    ran[0] = 1u;
    if (gid.x >= extent.x || gid.y >= extent.y) { return; }
    float4 v = tex.read(gid, lod);
    uint r = uint(round(v.r * 255.0));
    uint g = uint(round(v.g * 255.0));
    uint b = uint(round(v.b * 255.0));
    uint a = uint(round(v.a * 255.0));
    out[gid.y * width + gid.x] = (a << 24) | (b << 16) | (g << 8) | r;
}

// Every thread writes to a slot of its own, in a grid padded out to the
// threadgroup size. `dispatchThreads` promises the grid it is given and no
// more, so a slot outside that grid holding a marker is a thread Metal
// promised would not run.
kernel void grid_bounds(device uint *out [[buffer(0)]],
                        constant uint &stride [[buffer(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
    out[gid.y * stride + gid.x] = 1u + gid.x;
}

struct VOut { float4 pos [[position]]; float2 uv; };

// A full-target triangle strip driven from a vertex buffer the test owns, so a
// wrong vertex-buffer read shows up as geometry rather than as colour.
vertex VOut quad_vs(uint vid [[vertex_id]],
                    device const float4 *verts [[buffer(0)]]) {
    VOut o;
    float4 v = verts[vid];
    o.pos = float4(v.xy, 0.0, 1.0);
    o.uv = v.zw;
    return o;
}

fragment float4 solid_fs(VOut in [[stage_in]],
                         constant float4 &colour [[buffer(0)]]) {
    return colour;
}

// The same flat colour, made expensive on purpose.
//
// A race is only a test if the arm under test loses it. A solid fill of a
// window-sized target finishes before a host-side reader can decode a copy and
// memcpy three megabytes, so a device that reads those pixels unordered still
// happens to read the right ones and the case passes for a reason that has
// nothing to do with correctness. This shader gives the GPU real per-pixel work
// so the render is still running when the copy behind it is serviced.
//
// The accumulator has to survive the optimizer: `acc` is compared against a
// bound it cannot reach, so the compiler cannot fold the loop away, and the
// colour returned is exactly `colour` on every pixel.
fragment float4 heavy_fs(VOut in [[stage_in]],
                         constant float4 &colour [[buffer(0)]]) {
    float acc = 0.0;
    for (int i = 0; i < 2048; ++i) {
        acc += fract(sin(float(i) * 12.9898 + in.pos.x * 0.017 + in.pos.y * 0.031) * 43758.5453);
    }
    float poison = (acc > 1.0e30) ? 1.0 : 0.0;
    return float4(colour.rgb + poison, colour.a);
}

fragment float4 tex_fs(VOut in [[stage_in]],
                       texture2d<float, access::sample> tex [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    return tex.sample(s, in.uv);
}
// Coverage type. The atlas carries a single channel, exactly as CoreText
// rasterizes glyphs and exactly the `R8Unorm` a driven Maps boot is observed
// binding, and the colour arrives as a constant. Premultiplied out, so the
// pipeline below can blend it with `one / oneMinusSourceAlpha` the way a text
// layer is composited.
fragment float4 glyph_fs(VOut in [[stage_in]],
                         texture2d<float, access::sample> atlas [[texture(0)]],
                         constant float4 &colour [[buffer(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float cov = atlas.sample(s, in.uv).r;
    return float4(colour.rgb * cov, cov);
}
"""
