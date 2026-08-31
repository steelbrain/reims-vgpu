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

// Preserve each sample separately. Reading through a single-sample texture
// would make an implicit resolve indistinguishable from a correct multisample
// store, so the sample index is part of the output address.
kernel void read_ms_texels(texture2d_ms<float, access::read> tex [[texture(0)]],
                           device uint *out [[buffer(0)]],
                           constant uint &width [[buffer(1)]],
                           constant uint &samples [[buffer(2)]],
                           constant uint2 &extent [[buffer(3)]],
                           device uint *ran [[buffer(4)]],
                           uint2 gid [[thread_position_in_grid]]) {
    ran[0] = 1u;
    if (gid.x >= extent.x || gid.y >= extent.y) { return; }
    uint count = min(samples, tex.get_num_samples());
    for (uint sample = 0; sample < count; ++sample) {
        float4 v = tex.read(gid, sample);
        uint r = uint(round(v.r * 255.0));
        uint g = uint(round(v.g * 255.0));
        uint b = uint(round(v.b * 255.0));
        uint a = uint(round(v.a * 255.0));
        out[(gid.y * width + gid.x) * samples + sample] =
            (a << 24) | (b << 16) | (g << 8) | r;
    }
}

// The same readback with the sample count supplied by the host instead of
// queried from the texture.
//
// `read_ms_texels` above is a witness for two independent obligations at once:
// that a `texture2d_ms` bind is served at all, and that `get_num_samples()`
// reports the texture's own count. A device that serves the bind correctly but
// runs against a translator that folds the query to a constant produces the
// same result line as a device that refused the bind outright, and the two are
// opposite defects. This kernel drops the query so its result speaks only to
// the first obligation; the caller already knows the count, because it created
// the texture.
kernel void read_ms_texels_host_count(texture2d_ms<float, access::read> tex [[texture(0)]],
                                      device uint *out [[buffer(0)]],
                                      constant uint &width [[buffer(1)]],
                                      constant uint &samples [[buffer(2)]],
                                      constant uint2 &extent [[buffer(3)]],
                                      device uint *ran [[buffer(4)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    ran[0] = 1u;
    if (gid.x >= extent.x || gid.y >= extent.y) { return; }
    for (uint sample = 0; sample < samples; ++sample) {
        float4 v = tex.read(gid, sample);
        uint r = uint(round(v.r * 255.0));
        uint g = uint(round(v.g * 255.0));
        uint b = uint(round(v.b * 255.0));
        uint a = uint(round(v.a * 255.0));
        out[(gid.y * width + gid.x) * samples + sample] =
            (a << 24) | (b << 16) | (g << 8) | r;
    }
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

// Preserve the sampled float values themselves. Values outside [0,1] make a
// lossy unorm fallback observable instead of letting it produce the same
// colour by coincidence.
kernel void sample_float4(texture2d<float, access::sample> tex [[texture(0)]],
                          device float4 *out [[buffer(0)]],
                          constant uint &width [[buffer(1)]],
                          device uint *ran [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
    ran[0] = 1u;
    if (gid >= width) { return; }
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_edge);
    out[gid] = tex.sample(s, float2(float(gid) + 0.5f, 0.5f));
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

// Two dispatches in a concurrent compute encoder have no implicit dependency.
// Each thread owns one word, so the explicit barrier is the only ordering term
// and neither kernel relies on atomics or a cross-thread race.
static uint compute_barrier_value(uint token, uint gid) {
    uint value = token ^ gid;
    for (uint i = 0; i < 512; ++i) {
        value = value * 1664525u + 1013904223u;
        value ^= value >> ((i & 7u) + 1u);
    }
    return value;
}

kernel void compute_barrier_write(device uint *words [[buffer(0)]],
                                  constant uint &token [[buffer(1)]],
                                  uint gid [[thread_position_in_grid]]) {
    words[gid] = compute_barrier_value(token, gid);
}

kernel void compute_barrier_read(device const uint *words [[buffer(0)]],
                                 device uint *verdict [[buffer(1)]],
                                 constant uint &token [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    verdict[gid] = words[gid] == compute_barrier_value(token, gid) ? 1u : 0u;
}

kernel void sync_resource_write(device uint *words [[buffer(0)]],
                                constant uint &token [[buffer(1)]],
                                uint gid [[thread_position_in_grid]]) {
    words[gid] = token ^ gid;
}

// The indirect arguments carry threadgroup counts, while the shader-visible
// grid is their product with the encoded threads-per-threadgroup. Recording
// both the grid and one marker per thread distinguishes those two quantities.
kernel void indirect_threadgroups_write(device uint *out [[buffer(0)]],
                                         device uint *observed_grid [[buffer(1)]],
                                         constant uint &capacity [[buffer(2)]],
                                         device uint *ran [[buffer(4)]],
                                         uint3 gid [[thread_position_in_grid]],
                                         uint3 grid [[threads_per_grid]]) {
    ran[0] = 1u;
    observed_grid[0] = grid.x;
    observed_grid[1] = grid.y;
    observed_grid[2] = grid.z;
    uint linear = gid.x + grid.x * (gid.y + grid.y * gid.z);
    if (linear < capacity) {
        out[linear] = 0xa5000000u | linear;
    }
}

// One thread per indirect command. Distinct output and value-buffer offsets
// make reset and copy observable per command slot rather than only as a total
// dispatch count.
kernel void indirect_command_write(device uint *out [[buffer(0)]],
                                   device const uint *value [[buffer(1)]]) {
    out[0] = value[0];
}

struct VOut { float4 pos [[position]]; float2 uv; };

struct PointOut {
    float4 pos [[position]];
    float point_size [[point_size]];
};

vertex VOut topology_vs(uint vid [[vertex_id]],
                        device const float2 *positions [[buffer(0)]]) {
    VOut o;
    o.pos = float4(positions[vid], 0.0, 1.0);
    o.uv = float2(0.0);
    return o;
}

// The texture is vertex data by API contract, not a colour. Sampling rather
// than reading also exercises the linear-filter capability required by this
// format on the native upload rail.
vertex VOut float_positions_vs(uint vid [[vertex_id]],
                               texture2d<float, access::sample> positions [[texture(0)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float u = (float(vid) + 0.5f) / 4.0f;
    float4 p = positions.sample(s, float2(u, 0.5f));
    VOut o;
    o.pos = float4(p.xy, 0.0, 1.0);
    o.uv = float2(0.0);
    return o;
}

// One shader for both the non-indexed control and all indexed arms. Indexed
// semantics therefore change only which `vid`/`iid` Metal supplies, not the
// translation unit or vertex fetch mechanism.
vertex VOut indexed_vs(uint vid [[vertex_id]],
                       uint iid [[instance_id]],
                       device const float2 *positions [[buffer(0)]]) {
    float2 p = positions[vid];
    if (iid >= 5u) {
        p.x += (float(iid) - 5.5f) * 0.9f;
    }
    VOut o;
    o.pos = float4(p, 0.0, 1.0);
    o.uv = float2(0.0);
    return o;
}

// Topology cases provide clip-space positions directly. Point size is explicit
// so the point oracle observes a small island rather than relying on a
// device-dependent default raster size.
vertex PointOut point_vs(uint vid [[vertex_id]],
                         device const float2 *positions [[buffer(0)]]) {
    PointOut o;
    o.pos = float4(positions[vid], 0.0, 1.0);
    o.point_size = 3.0;
    return o;
}

fragment float4 point_fs(PointOut in [[stage_in]]) {
    return float4(0.0, 1.0, 0.0, 1.0);
}

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

// Every covered pixel receives four distinct, exact unorm samples. A backend
// that resolves the attachment instead of storing it turns these into one
// repeated average and cannot pass the per-sample oracle.
fragment float4 sample_id_fs(VOut in [[stage_in]], uint sample [[sample_id]]) {
    switch (sample) {
        case 0: return float4(1.0, 0.0, 0.0, 1.0);
        case 1: return float4(0.0, 1.0, 0.0, 1.0);
        case 2: return float4(0.0, 0.0, 1.0, 1.0);
        default: return float4(1.0, 1.0, 1.0, 1.0);
    }
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

// A render-stage resource barrier must make every producer write visible to
// the following fragment draw in the same encoder. Each fragment owns one
// word, avoiding atomics and data races while making the oracle cover the
// entire raster rather than one scheduling-sensitive fragment.
fragment float4 render_barrier_write_fs(VOut in [[stage_in]],
                                        device uint *words [[buffer(0)]],
                                        constant uint2 &params [[buffer(1)]]) {
    float acc = 0.0;
    for (int i = 0; i < 512; ++i) {
        acc += fract(sin(float(i) * 12.9898 + in.pos.x * 0.017 + in.pos.y * 0.031) * 43758.5453);
    }
    uint2 p = uint2(in.pos.xy);
    words[p.y * params.x + p.x] = params.y + uint(acc > 1.0e30);
    return float4(0.0, 0.0, 0.0, 1.0);
}

fragment float4 render_barrier_read_fs(VOut in [[stage_in]],
                                       device const uint *words [[buffer(0)]],
                                       constant uint2 &params [[buffer(1)]]) {
    uint2 p = uint2(in.pos.xy);
    bool current = words[p.y * params.x + p.x] == params.y;
    return current ? float4(0.0, 1.0, 0.0, 1.0)
                   : float4(1.0, 0.0, 0.0, 1.0);
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

kernel void heap_alias_fill(texture2d<float, access::write> output [[texture(0)]],
                            constant float4 &value [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x < output.get_width() && gid.y < output.get_height()) {
        output.write(value, gid);
    }
}
"""
