// The tests' functions that write a sweep of values through a texture (`encode_render.rs`'s tests):
// value `i` of 65536, at texel `(i % 256, i / 256)`, in every component, from a sweep `kind`
// (buffer 0) chooses. `sweep_kernel` writes it with `access::write`; `sweep_vertex` and
// `sweep_fragment` draw it into colour attachment 0 over a 256 by 256 target, and `sweep_vertex`
// and `write_fragment` write it into texture 0 with `access::write` from the fragment stage.
#include <metal_stdlib>
using namespace metal;

static float value(uint i, uint kind) {
    float f = float(i);
    switch (kind) {
    case 0: return -0.25 + f / 65536.0 * 1.5;
    case 1: return 1.0 + f * exp2(-23.0);
    case 2: return -1.0 - f * exp2(-23.0);
    case 3: return 65504.0 + f * 0.5;
    default: return f / 65536.0 * 4.0 - 2.0;
    }
}

kernel void sweep_kernel(texture2d<float, access::write> t [[texture(0)]],
                         constant uint &kind [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    t.write(float4(value(gid.y * 256 + gid.x, kind)), gid);
}

struct Sweep {
    float4 position [[position]];
};

vertex Sweep sweep_vertex(uint vid [[vertex_id]]) {
    const float2 p[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return Sweep{float4(p[vid], 0, 1)};
}

fragment float4 sweep_fragment(Sweep in [[stage_in]], constant uint &kind [[buffer(0)]]) {
    uint2 g = uint2(in.position.xy);
    return float4(value(g.y * 256 + g.x, kind));
}

fragment void write_fragment(Sweep in [[stage_in]], constant uint &kind [[buffer(0)]],
                             texture2d<float, access::write> t [[texture(0)]]) {
    uint2 g = uint2(in.position.xy);
    t.write(float4(value(g.y * 256 + g.x, kind)), g);
}

// `write_args_fragment` writes `write_fragment`'s value into the texture an argument buffer at
// buffer 1 holds (`encode_render.rs`'s tests).
struct WrittenArgs {
    texture2d<float, access::write> t [[id(0)]];
};

fragment void write_args_fragment(Sweep in [[stage_in]], constant uint &kind [[buffer(0)]],
                                  constant WrittenArgs &args [[buffer(1)]]) {
    uint2 g = uint2(in.position.xy);
    args.t.write(float4(value(g.y * 256 + g.x, kind)), g);
}

// `read_vertex` and `read_fragment` read buffer 1 through a `device` pointer that is not `const`,
// which Metal's reflection answers read-write, and write no memory (`client_direction.rs`).
vertex Sweep read_vertex(uint vid [[vertex_id]], device float *scale [[buffer(1)]]) {
    const float2 p[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return Sweep{float4(p[vid] * scale[0], 0, 1)};
}

fragment float4 read_fragment(Sweep in [[stage_in]], device float4 *color [[buffer(1)]]) {
    return color[0];
}

// `store_words`, `exchange_words`, `store_vertex` and `store_fragment` store 0x11223344 into
// buffer 1, a word a thread, vertex or fragment (`exchange_words` by an atomic exchange): the
// fragment at `(x, y)` into word `y * width + x`, `width` from buffer 2 (`client_direction.rs`).
kernel void store_words(device uint *dst [[buffer(1)]], uint g [[thread_position_in_grid]]) {
    dst[g] = 0x11223344;
}

kernel void exchange_words(device atomic_uint *dst [[buffer(1)]],
                           uint g [[thread_position_in_grid]]) {
    atomic_exchange_explicit(&dst[g], 0x11223344u, memory_order_relaxed);
}

vertex Sweep store_vertex(uint vid [[vertex_id]], device uint *dst [[buffer(1)]]) {
    dst[vid] = 0x11223344;
    return Sweep{float4(0, 0, 0, 1)};
}

fragment float4 store_fragment(Sweep in [[stage_in]], device uint *dst [[buffer(1)]],
                               constant uint &width [[buffer(2)]]) {
    uint2 g = uint2(in.position.xy);
    dst[g.y * width + g.x] = 0x11223344;
    return float4(0);
}

// `sweep_four_fragment` draws `sweep_fragment`'s value into colour attachments 0 to 3
// (`encode_render.rs`'s tests).
struct Four {
    float4 c0 [[color(0)]];
    float4 c1 [[color(1)]];
    float4 c2 [[color(2)]];
    float4 c3 [[color(3)]];
};

fragment Four sweep_four_fragment(Sweep in [[stage_in]], constant uint &kind [[buffer(0)]]) {
    uint2 g = uint2(in.position.xy);
    float4 v = float4(value(g.y * 256 + g.x, kind));
    return Four{v, v, v, v};
}

// `sweep_half_fragment` draws `sweep_fragment`'s value as a `half4`, which the function's
// conversion rounds, into colour attachment 0 (`encode_render.rs`'s tests).
fragment half4 sweep_half_fragment(Sweep in [[stage_in]], constant uint &kind [[buffer(0)]]) {
    uint2 g = uint2(in.position.xy);
    return half4(value(g.y * 256 + g.x, kind));
}

// `fetch_half_fragment` and `fetch_float_fragment` read colour attachment 0 by framebuffer fetch
// and draw half of it as a `half4` or a `float4` (`encode_render.rs`, the blends of an
// extended-range attachment the runtime blends in the shader).
fragment half4 fetch_half_fragment(Sweep in [[stage_in]], half4 d [[color(0)]]) {
    return d * 0.5h;
}

fragment float4 fetch_float_fragment(Sweep in [[stage_in]], float4 d [[color(0)]]) {
    return d * 0.5f;
}

// `fetch_seen_fragment` reads colour attachment 0 by framebuffer fetch and draws what it read,
// unchanged, into colour attachments 0 and 1 (`encode_render.rs`, what a memoryless attachment
// loaded `DontCare` reads).
struct Seen {
    float4 kept [[color(0)]];
    float4 seen [[color(1)]];
};

fragment Seen fetch_seen_fragment(Sweep in [[stage_in]], float4 d [[color(0)]]) {
    return Seen{d, d};
}

// `xr_float_fragment` draws a float4 of exact values a hash of the pixel and buffer 0's seed
// chooses: colour lanes in [-0.9375, 1.5), past an extended-range format's both ends, and alpha
// in [-0.125, 1.125), each a multiple of 2^-11 (`encode_render.rs`, the float4 blends into
// extended-range attachments). `xr_half_fragment` draws a half4 alike, each lane a multiple of
// 2^-10 a half holds exactly: colour lanes in [-0.9375, 1.5), alpha in [-0.125, 1.125).
// `xr_read` reads a 64 by 64 texture into buffer 0, a float4 a texel.
static uint xr_hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static float xr_lane(uint h, uint span) {
    return float(int(h % span) - 1920) / 2048.0;
}

fragment float4 xr_float_fragment(Sweep in [[stage_in]], constant uint &seed [[buffer(0)]]) {
    uint2 g = uint2(in.position.xy);
    uint h = xr_hash(g.y * 64 + g.x + seed * 0x10000u);
    uint k = xr_hash(h ^ 0x9e3779b9u);
    return float4(xr_lane(h, 4992u), xr_lane(h >> 13, 4992u), xr_lane(k, 4992u),
                  float(int((k >> 13) % 2560u) - 256) / 2048.0);
}

static half xr_half_lane(uint h) {
    return half(float(int(h % 2496u) - 960) / 1024.0);
}

fragment half4 xr_half_fragment(Sweep in [[stage_in]], constant uint &seed [[buffer(0)]]) {
    uint2 g = uint2(in.position.xy);
    uint h = xr_hash(g.y * 64 + g.x + seed * 0x10000u);
    uint k = xr_hash(h ^ 0x9e3779b9u);
    return half4(xr_half_lane(h), xr_half_lane(h >> 13), xr_half_lane(k),
                 half(float(int((k >> 13) % 1280u) - 128) / 1024.0));
}

// `xr_dual_half_fragment` and `xr_dual_float_fragment` return two outputs into colour 0, the
// first (index 0) as `xr_half_fragment` and `xr_float_fragment` draw it, the second (index 1),
// which a blend of the second source reads, alike of other hashes of the pixel and seed.
struct XrDualHalf {
    half4 first [[color(0), index(0)]];
    half4 second [[color(0), index(1)]];
};

struct XrDualFloat {
    float4 first [[color(0), index(0)]];
    float4 second [[color(0), index(1)]];
};

static half4 xr_half4(uint h, uint k) {
    return half4(xr_half_lane(h), xr_half_lane(h >> 13), xr_half_lane(k),
                 half(float(int((k >> 13) % 1280u) - 128) / 1024.0));
}

static float4 xr_float4(uint h, uint k) {
    return float4(xr_lane(h, 4992u), xr_lane(h >> 13, 4992u), xr_lane(k, 4992u),
                  float(int((k >> 13) % 2560u) - 256) / 2048.0);
}

fragment XrDualHalf xr_dual_half_fragment(Sweep in [[stage_in]],
                                          constant uint &seed [[buffer(0)]]) {
    uint2 g = uint2(in.position.xy);
    uint h = xr_hash(g.y * 64 + g.x + seed * 0x10000u);
    uint k = xr_hash(h ^ 0x9e3779b9u);
    uint m = xr_hash(k ^ 0x85ebca6bu);
    uint n = xr_hash(m ^ 0xc2b2ae35u);
    return {xr_half4(h, k), xr_half4(m, n)};
}

fragment XrDualFloat xr_dual_float_fragment(Sweep in [[stage_in]],
                                            constant uint &seed [[buffer(0)]]) {
    uint2 g = uint2(in.position.xy);
    uint h = xr_hash(g.y * 64 + g.x + seed * 0x10000u);
    uint k = xr_hash(h ^ 0x9e3779b9u);
    uint m = xr_hash(k ^ 0x85ebca6bu);
    uint n = xr_hash(m ^ 0xc2b2ae35u);
    return {xr_float4(h, k), xr_float4(m, n)};
}

// `xr_loaded_fragment` returns buffer 1's float4 at the pixel, 4096 of them a draw, the draw of
// buffer 0's seed (1 first): a float4 output each of whose lanes is a load.
fragment float4 xr_loaded_fragment(Sweep in [[stage_in]], constant uint &seed [[buffer(0)]],
                                   device const float4 *values [[buffer(1)]]) {
    uint2 g = uint2(in.position.xy);
    return values[(seed - 1) * 4096 + g.y * 64 + g.x];
}

// `reach_vertex` places buffer 0's float4s, one a vertex, and `xr_constant_fragment` writes one
// colour an extended-range format holds exactly (`encode_render.rs`, the tiles a load pass
// rewrites).
vertex float4 reach_vertex(uint vid [[vertex_id]], const device float4 *positions [[buffer(0)]]) {
    return positions[vid];
}

fragment half4 xr_constant_fragment() {
    return half4(0.5h, 0.25h, 0.75h, 1.0h);
}

kernel void xr_read(texture2d<float, access::read> t [[texture(0)]],
                    device float4 *out [[buffer(0)]],
                    uint2 g [[thread_position_in_grid]]) {
    out[g.y * 64 + g.x] = t.read(g);
}
