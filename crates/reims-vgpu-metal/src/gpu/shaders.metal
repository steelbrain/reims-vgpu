// The display's present shaders and the XR10 paging kernels, compiled at run time once per Metal
// device (gpu/shaders.rs). Core binds their arguments: display/render.rs for the present pass,
// resources/texture/compute.rs for the XR10 dispatches.

#include <metal_stdlib>
using namespace metal;

// ---- The display ----

// A vertex of the full-screen strip: its clip-space position and the frame's texture coordinate.
struct PresentVertex {
    float4 position [[position]];
    float2 tex;
};

// Vertex `vid` of the strip: the display's position buffer (float4 each) at 0 and its
// texture-coordinate buffer (float2 each) at 1.
vertex PresentVertex displayPresentVertex(constant float4 *positions [[buffer(0)]],
                                          constant float2 *tex_coords [[buffer(1)]],
                                          uint vid [[vertex_id]]) {
    PresentVertex out;
    out.position = positions[vid];
    out.tex = tex_coords[vid];
    return out;
}

// The frame, filtered where the host's region scales it.
constexpr sampler frame_sampler(coord::normalized, address::clamp_to_edge, filter::linear);

// Without a gamma table: the frame as it is, alpha included. The colour is neither
// premultiplied nor blended, and the render target's format converts it.
fragment float4 displayPresentFragment(PresentVertex in [[stage_in]],
                                       texture2d<float> frame [[texture(0)]]) {
    return frame.sample(frame_sampler, in.tex);
}

// The display's fragment property buffer (96 bytes, written by display/render.rs): the colour
// matrix as four columns of three floats, each column padded to 16 bytes, then the gamma table's
// slopes at 0 and at 1, three packed floats each.
struct DisplayFragmentProperties {
    float4x3 color_matrix;
    packed_float3 gamma_slope_at_zero;
    packed_float3 gamma_slope_at_one;
};

static_assert(sizeof(DisplayFragmentProperties) == 96, "the property buffer's length");

// The gamma table's entries, one slice per channel: the entry whose span holds the value, not
// an interpolation between entries (a two-entry ramp is a threshold at 0.5).
constexpr sampler gamma_sampler(coord::normalized, address::clamp_to_edge, filter::nearest);

// Channel `channel` of `v` through its table. Inside [0, 1] the table's entry; outside it the
// table's end entry, extended by the table's slope at that end, since nothing clamps a surface's
// components (an 'RGhA' frame holds values below 0 and above 1).
static float gamma_lookup(texture1d_array<float> table, uint channel, float v, float slope_at_zero,
                          float slope_at_one) {
    float inside = table.sample(gamma_sampler, clamp(v, 0.0f, 1.0f), channel).r;
    float at_zero = table.sample(gamma_sampler, 0.0f, channel).r;
    float at_one = table.sample(gamma_sampler, 1.0f, channel).r;
    if (v < 0.0f) {
        return at_zero + v * slope_at_zero;
    }
    if (v > 1.0f) {
        return at_one + (v - 1.0f) * slope_at_one;
    }
    return inside;
}

// With a gamma table: each channel through its table, then row c of the colour matrix gives
// output c = m[c][0] r + m[c][1] g + m[c][2] b + m[c][3]. Alpha is 1, whatever the frame holds.
fragment float4 displayPresentFragmentWithGammaTableAndColorMatrix(
    PresentVertex in [[stage_in]], texture2d<float> frame [[texture(0)]],
    texture1d_array<float> gamma_table [[texture(1)]],
    constant DisplayFragmentProperties &properties [[buffer(0)]]) {
    float3 color = frame.sample(frame_sampler, in.tex).rgb;
    float3 slope_at_zero = properties.gamma_slope_at_zero;
    float3 slope_at_one = properties.gamma_slope_at_one;
    float3 looked_up;
    for (uint c = 0; c < 3; c++) {
        looked_up[c] = gamma_lookup(gamma_table, c, color[c], slope_at_zero[c], slope_at_one[c]);
    }
    return float4(properties.color_matrix * float4(looked_up, 1.0f), 1.0f);
}

// ---- The graphics device's display pipe colour ----

// The display pipe's colour (reims-vgpu's graphics/color.rs, written by graphics/sink.rs): the
// matrix stage's three rows as the guest's transaction carries them, then each stage's entry
// count, 0 where the stage is off.
struct DisplayPipeColor {
    float4 matrix[3];
    uint plane_count;
    uint pre_count;
    uint matrix_count;
    uint post_count;
};

static_assert(sizeof(DisplayPipeColor) == 64, "the colour buffer's length");

// `v` through a gamma stage of `n` entries, each channel by its own: entry k is the output at
// input k / (n - 1), linear between entries, the input clamped to [0, 1]. A stage of no entries
// passes `v` through; one of one entry is that entry everywhere.
static float3 display_pipe_gamma(device const float4 *table, uint n, float3 v) {
    if (n == 0) {
        return v;
    }
    if (n == 1) {
        return table[0].rgb;
    }
    float3 x = clamp(v, 0.0f, 1.0f) * float(n - 1);
    float3 out;
    for (uint c = 0; c < 3; c++) {
        uint i = min(uint(x[c]), n - 2);
        out[c] = mix(table[i][c], table[i + 1][c], x[c] - float(i));
    }
    return out;
}

// `v` through the display pipe's stages, in order: plane 0's gamma, the pre-gamma, the colour
// matrix (output c = row c's first three . the colour, plus its fourth), the post-gamma.
static float3 display_pipe_color(float3 v, constant DisplayPipeColor &color,
                                 device const float4 *plane_gamma, device const float4 *pre_gamma,
                                 device const float4 *post_gamma) {
    v = display_pipe_gamma(plane_gamma, color.plane_count, v);
    v = display_pipe_gamma(pre_gamma, color.pre_count, v);
    if (color.matrix_count == 3) {
        v = float3(dot(color.matrix[0].xyz, v) + color.matrix[0].w,
                   dot(color.matrix[1].xyz, v) + color.matrix[1].w,
                   dot(color.matrix[2].xyz, v) + color.matrix[2].w);
    }
    return display_pipe_gamma(post_gamma, color.post_count, v);
}

// A presented frame through the display pipe's colour. Alpha is 1, whatever the frame holds.
fragment float4 displayPipeColorFragment(PresentVertex in [[stage_in]],
                                         texture2d<float> frame [[texture(0)]],
                                         constant DisplayPipeColor &color [[buffer(0)]],
                                         device const float4 *plane_gamma [[buffer(1)]],
                                         device const float4 *pre_gamma [[buffer(2)]],
                                         device const float4 *post_gamma [[buffer(3)]]) {
    float3 v = frame.sample(frame_sampler, in.tex).rgb;
    return float4(display_pipe_color(v, color, plane_gamma, pre_gamma, post_gamma), 1.0f);
}

// Where the guest's scanout lies (graphics/sink.rs): the drawable pixel its top left is drawn
// at, its bytes per row, and where its first pixel is in the buffer, both in 4-byte words.
struct DisplayPipeScanout {
    uint2 origin;
    uint stride;
    uint first;
};

static_assert(sizeof(DisplayPipeScanout) == 16, "the scanout's buffer length");

// The guest's scanout, read in place from its buffer (BGRA8, one pixel of the drawable per pixel
// of the scanout), through the display pipe's colour. Alpha is 1, whatever the scanout holds.
fragment float4 displayPipeColorScanoutFragment(PresentVertex in [[stage_in]],
                                                constant DisplayPipeColor &color [[buffer(0)]],
                                                device const float4 *plane_gamma [[buffer(1)]],
                                                device const float4 *pre_gamma [[buffer(2)]],
                                                device const float4 *post_gamma [[buffer(3)]],
                                                device const uchar4 *scanout [[buffer(4)]],
                                                constant DisplayPipeScanout &at [[buffer(5)]]) {
    uint2 p = uint2(in.position.xy) - at.origin;
    uchar4 bgra = scanout[at.first + p.y * at.stride + p.x];
    float3 v = float3(bgra.z, bgra.y, bgra.x) / 255.0f;
    return float4(display_pipe_color(v, color, plane_gamma, pre_gamma, post_gamma), 1.0f);
}

// ---- XR10 two-plane textures (pixel formats 550 and 551) ----
//
// The guest's texture is two planes: colour, viewed as BGR10_XR (or its sRGB form), and alpha,
// viewed in a four-byte format whose first component is the byte after the texel's first. The
// kernels move texels between the planes and the host's texture, one thread per texel.

// Paging in: colour from plane 0 and alpha from plane 1, into the texture.
kernel void BlitInRGB_2P_XR10_A8(texture2d<half, access::read> src_color [[texture(0)]],
                                 texture2d<half, access::read> src_alpha [[texture(1)]],
                                 texture2d<half, access::write> dst [[texture(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    dst.write(half4(src_color.read(gid).rgb, src_alpha.read(gid).r), gid);
}

// Paging out: the texture's colour into plane 0 and its alpha into every component of plane 1.
kernel void BlitOutRGB_2P_XR10_A8(texture2d<half, access::read> src [[texture(0)]],
                                  texture2d<half, access::write> dst_color [[texture(1)]],
                                  texture2d<half, access::write> dst_alpha [[texture(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    half4 texel = src.read(gid);
    dst_color.write(half4(texel.rgb, 1.0h), gid);
    dst_alpha.write(half4(texel.a), gid);
}
