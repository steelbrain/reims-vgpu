// The M2 scaler device's transform (106b:1a0b), in Metal Shading Language: kernels that read a
// guest surface's bytes where they are and write another's, as core's CPU reference does a plan
// (reims-vgpu-video-core's `scaler/reference.rs`, `run`). The CPU works out from the plan where
// each destination pixel samples and with what weights (the reference's `taps`), and every number
// below that a format, a matrix or a window gives; these apply them.
//
// A transform is dispatches of seven kernels, which the CPU picks (reims-vgpu's
// `scaler/gpu/params.rs`, `work`), each summing in the reference's order.
// - `scaler_transform`: one dispatch per plane of the destination. A thread writes one 32-bit word
//   of the plane, so that no two threads write one word: the bytes of it the destination window
//   covers (core's `scaler/coverage.rs`), each worked out as the reference works it out, and the
//   rest of the word as it was. Each plane's offset and row pitch are multiples of 4, which the
//   CPU checks before it asks for a dispatch. `scaler_transform_adaptive` is the same for a
//   transform whose passes the adaptive filter moves, a kernel of its own so that the others run,
//   and are made, without its code; `scaler_transform_both_planes` the same for both planes of a
//   4:2:0 window whose every edge is even, which the filter does not move, a kernel of its own
//   so that its unrolled code is not made into the others.
// - `scaler_quads`, in its place for the RGB plane of a copy of a Y'CbCr source: a thread decodes
//   two by two pixels, which share their chroma samples, into threadgroup memory, and the
//   threadgroup then writes the plane's words a row at a time, side by side.
// - Where a destination pixel takes more than one pixel of the turned source each way, the
//   transform is separable, down then across. A dispatch of `scaler_lines` first works out, for
//   each row of the window and each column of the turned source the window's taps take, the down
//   taps' sum of that column (a line: four floats, or three of a Y'CbCr source, whose pixels are
//   opaque) into the lines buffer, a threadgroup decoding each source pixel its taps take once,
//   into threadgroup memory. Then a dispatch per plane sums the lines across: `scaler_across`, a
//   thread a word, or, for a reduction across of a plane whose words hold more than one pixel,
//   `scaler_across_shared`, whose threads work out the words' pixels side by side into threadgroup
//   memory before they write the words. So a source pixel is decoded once a line rather than once
//   a destination pixel: down taps plus across taps an output, not their product. A tall window is
//   a band of rows at a time, each its lines' dispatch then its planes', so the lines buffer stays
//   small.
//
// The reference's steps, and where each is here:
// - decode (`decode_turned`): `decoded`. The source turned by the plan's orientation, pixel by
//   pixel; a Y'CbCr source's chroma plane turned as it is, each sample read on the first pixel
//   it covers and its codes interpolated between, from the samples the source window covers
//   alone, the mean rounded down, in Y'CbCr; an RGB source's into linear light (`light`,
//   `srgb_to_linear`), through a table of its codes' values that a threadgroup works out first
//   (`work_out_light`).
// - resample (`resample_taps`): `line_at` and `working`, or `scaler_lines` and `across_lines`. A
//   destination pixel is the taps' weighted sum of the turned source's pixels, the weights
//   already divided by their sum and some negative, down each column (`line_at`, or a line) then
//   across, a Y'CbCr source's chroma with its own taps where it has them, each pass's sums moved
//   toward the bilinear sample at their point by the six pixels about it (`adapt`, `adapted`);
//   then a Y'CbCr source's through RGB, clamped, where the destination is not Y'CbCr
//   (`in_space`), and back out of linear light where it went into it (`finished`).
// - encode (`encode`): the kernel's planes. Luma, or RGB, per pixel; a chroma sample the mean of
//   the window's pixels its siting names, or of all the window's pixels it covers where it names
//   none of them.
//
// Every value is a float; the CPU's are doubles, so a code a result puts within a few millionths
// of a half may round the other way.

#include <metal_stdlib>
using namespace metal;

// One pixel of the turned source along one axis, and its weight.
struct ScalerTap {
    uint index;
    float weight;
};

// What the kernel is told, laid out as `Params` in reims-vgpu's `scaler/gpu/params.rs`.
struct ScalerParams {
    // The source.
    uint source_packing;         // 0: luma, then interleaved chroma at half each way; 1: 4:2:2
                                 // pairs in one plane; 2: four bytes a pixel
    uint source_sample_bytes;    // 1, or 2 for 10 bits in the top of 16
    uint source_order[4];        // 4:2:2: the bytes of Y0, Cb, Y1, Cr; RGB: of R, G, B, A
    uint source_luma_offset;     // the luma plane (or the one plane), in the source buffer
    uint source_luma_pitch;
    uint source_chroma_offset;   // the chroma plane (or the one plane)
    uint source_chroma_pitch;
    float source_luma_low;       // a luma code's value: (code - low) * scale
    float source_luma_scale;
    float source_chroma_middle;  // a chroma code's: (code - middle) * scale, the code the
    float source_chroma_scale;   // interpolation of whole codes rounded down
    int luma_map[6];             // the turned source's pixel (x, y) is the source's
                                 // (m0 x + m1 y + m2, m3 x + m4 y + m5)
    uint turned_last_x;          // the turned source's last pixel across and down
    uint turned_last_y;
    uint source_subsample_x;     // the source's pixels one of its chroma samples covers
    uint source_subsample_y;
    uint chroma_step_x;          // the turned pixels one of the turned window's chroma samples
    uint chroma_step_y;          // covers
    uint chroma_origin_x;        // the turned pixel its first chroma sample sits on: the source
    uint chroma_origin_y;        // window's first
    uint chroma_last_x;          // its last chroma sample across and down, counted from the
    uint chroma_last_y;          // first: chroma is interpolated from these alone, the last
                                 // repeated past them

    // The working space.
    uint rgb;                    // 1: through RGB; 0: in Y'CbCr, both formats being Y'CbCr
    uint linear;                 // 1: resampled in linear light, the source being RGB
    float kr;                    // the matrix's luma weights of red, green and blue
    float kg;
    float kb;
    float cb_scale;              // 1 / (2 (1 - kb))
    float cr_scale;              // 1 / (2 (1 - kr))
    float b_of_cb;               // 2 (1 - kb)
    float r_of_cr;               // 2 (1 - kr)
    float g_scale;               // 1 / (1 - kr - kb)

    // The taps, in the taps buffer: each column's, then each row's.
    uint across_first;
    uint across_count;           // taps a column
    uint down_first;
    uint down_count;             // taps a row

    // The destination window, in pixels, and where its chroma sits.
    uint window_x;
    uint window_y;
    uint window_width;
    uint window_height;
    uint site_x;                 // 0: the first pixel; 1: between the two; 2: the second
    uint site_y;
    uint subsample_x;            // the destination's luma pixels one chroma sample covers
    uint subsample_y;

    // This dispatch's plane.
    uint plane_kind;             // KIND_*, below
    uint plane_order[4];         // 4:2:2: the bytes of Y0, Cb, Y1, Cr; RGB: of R, G, B, A
    uint plane_offset;           // in the destination buffer: a multiple of 4
    uint plane_pitch;            // a multiple of 4
    uint first_row;              // the rows of the plane the window covers
    uint rows;
    uint first_byte;             // the bytes of each such row it covers
    uint end_byte;
    uint words;                  // the words of a row a thread each, from first_byte's
    float luma_low;              // a luma code: low + span * value
    float luma_span;
    float chroma_middle;         // a chroma code: middle + span * value
    float chroma_span;
    float code_max;              // the largest code

    // The lines, for a separable transform: a band of the window's rows.
    uint lines_first;            // the first column of the turned source the buffer holds
    uint lines_pitch;            // its lines a row: the columns the window's across taps take,
                                 // `scaler_lines`'s threads across
    uint lines_row;              // the band's first row, in the window
    uint lines_rows;             // its rows: `scaler_lines`'s threads down

    // A 4:2:0 destination's chroma plane, where one dispatch writes both its planes.
    uint chroma_plane_offset;    // in the destination buffer: a multiple of 4
    uint chroma_plane_pitch;     // a multiple of 4

    // A Y'CbCr source's chroma's taps, where they are not its luma's (the luma's first and count
    // where they are): along an axis that keeps pixel centres and that the source subsamples.
    uint chroma_across_first;
    uint chroma_across_count;
    uint chroma_down_first;
    uint chroma_down_count;
    uint ycbcr_to_rgb;           // 1: a Y'CbCr source into RGB, converted after the scale

    // The adaptive filter (the reference's `adaptive_blend`): each pass's sums moved toward the
    // bilinear sample at their point, by the six pixels about it.
    uint across_points_first;    // in the taps buffer: each column's point, then each row's, as
    uint down_points_first;      // the pixel it lies in (an int) and its fraction past it
    uint across_span_first;      // the turned source's pixels each axis's taps are clamped to
    uint across_span_last;
    uint down_span_first;
    uint down_span_last;
    uint across_reduces;         // 1: the axis takes more than one source pixel a pixel
    uint down_reduces;
    uint adapt_across;           // 1: the pass across moves its sums
    uint adapt_down;             // 1: the pass down moves its sums
    uint adaptive_channels;      // the channels it moves, from the first: luma, or R, G and B
    float adaptive_scale;        // a step's fraction of the full range a unit of the value
};

// What a dispatch's plane holds.
enum : uint {
    KIND_LUMA8 = 0,     // 8-bit luma: four samples a word
    KIND_LUMA16 = 1,    // 10-bit luma in 16: two a word
    KIND_CHROMA8 = 2,   // 8-bit Cb and Cr pairs: two pairs a word
    KIND_CHROMA16 = 3,  // 10-bit Cb and Cr in 16 each: one pair a word
    KIND_PACKED422 = 4, // one 4:2:2 pair a word
    KIND_RGB32 = 5,     // one pixel a word
    KIND_BIPLANAR8 = 6, // a 4:2:0 window of even edges, both planes: a chroma word of 8-bit
                        // pairs and the luma words of the same pixels in the two rows it covers
    KIND_BIPLANAR16 = 7, // the same in 10 bits
};

// How the source's samples lie.
enum : uint {
    PACKING_BIPLANAR = 0,
    PACKING_PACKED422 = 1,
    PACKING_RGB = 2,
};

// The byte at `at` of `buffer`.
static uint byte_at(device const uint *buffer, uint at) {
    return (buffer[at >> 2] >> ((at & 3u) * 8u)) & 0xffu;
}

// The sample at `at`: a byte, or the top 10 bits of a little-endian 16.
static float sample_at(device const uint *buffer, uint at, uint bytes) {
    if (bytes == 1u) {
        return float(byte_at(buffer, at));
    }
    uint v = byte_at(buffer, at) | (byte_at(buffer, at + 1u) << 8u);
    return float(v >> 6u);
}

static int2 mapped(constant int *m, uint x, uint y) {
    int ix = int(x);
    int iy = int(y);
    return int2(m[0] * ix + m[1] * iy + m[2], m[3] * ix + m[4] * iy + m[5]);
}

// sRGB's decoding of a value, 0 to 1, to linear light (the reference's `srgb_to_linear`).
static float srgb_to_linear(float v) {
    return v <= 0.04045f ? v / 12.92f : pow((v + 0.055f) / 1.055f, 2.4f);
}

// The value in linear light of each code of an RGB source's channels, as `srgb_to_linear` decodes
// it: a table in threadgroup memory, which each threadgroup of a kernel that decodes the source
// works out first, a code a thread, where the transform goes into linear light. A pixel's
// decoding is then three loads (`rgb_pixel`), not three powers at each tap, which safe math keeps
// precise and slow. Every thread of the threadgroup reaches the barrier: the dispatches are of
// whole threadgroups.
constant uint LIGHT_CODES = 256u;

static void work_out_light(constant ScalerParams &p, threadgroup float *light, uint lane,
                           uint lanes) {
    if (p.linear == 0u) {
        return;
    }
    for (uint c = lane; c < LIGHT_CODES; c += lanes) {
        light[c] = srgb_to_linear(float(c) / 255.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Its inverse, of light clamped to black and white first (the reference's `linear_to_srgb`).
static float3 linear_to_srgb(float3 l) {
    float3 out;
    for (uint k = 0; k < 3u; ++k) {
        float v = clamp(l[k], 0.0f, 1.0f);
        out[k] = v <= 0.0031308f ? v * 12.92f : 1.055f * pow(v, 1.0f / 2.4f) - 0.055f;
    }
    return out;
}

// Y'CbCr to RGB, each clamped to 0 to 1 (the reference's `to_rgb`).
static float4 to_rgb(constant ScalerParams &p, float4 ycc) {
    float r = ycc.x + p.r_of_cr * ycc.z;
    float b = ycc.x + p.b_of_cb * ycc.y;
    float g = (ycc.x - p.kr * r - p.kb * b) * p.g_scale;
    return float4(clamp(r, 0.0f, 1.0f), clamp(g, 0.0f, 1.0f), clamp(b, 0.0f, 1.0f), ycc.w);
}

// RGB to Y'CbCr (the reference's `to_ycbcr`).
static float4 to_ycbcr(constant ScalerParams &p, float4 rgb) {
    float y = p.kr * rgb.x + p.kg * rgb.y + p.kb * rgb.z;
    return float4(y, (rgb.z - y) * p.cb_scale, (rgb.x - y) * p.cr_scale, rgb.w);
}

// `v / step` and its remainder, for a step of 1 or 2, as every format's subsampling is: a shift,
// with no division, so that the many places that decode a pixel stay small.
static uint2 step_of(uint step, uint v) {
    uint shift = step >> 1u;
    return uint2(v >> shift, v & shift);
}

// The step between neighbouring pixels, and the smallest that keeps a reduction from moving, of
// the full range (the reference's `ADAPTIVE_STEP` and `ADAPTIVE_SMALL`).
constant float ADAPTIVE_STEP = 0.45f;
constant float ADAPTIVE_SMALL = 14.5f / 255.0f;

// How far a sample moves from its taps' sum toward the bilinear sample at its point, by the five
// steps between the six pixels about it (the reference's `adaptive_blend`).
static float adaptive_blend(thread const float *steps, bool reduces) {
    float outer = max(steps[1], steps[3]);
    if (!reduces) {
        if (outer >= ADAPTIVE_STEP) {
            return 0.6f;
        }
        float largest = 0.0f;
        for (uint k = 0; k < 5u; ++k) {
            largest = max(largest, steps[k]);
        }
        return clamp((255.0f * largest - 32.0f) / 500.0f, 0.0f, 0.25f);
    }
    if (outer < ADAPTIVE_STEP) {
        return 0.0f;
    }
    uint big = 0u;
    uint small = 0u;
    for (uint k = 0; k < 5u; ++k) {
        if (steps[k] >= ADAPTIVE_STEP) {
            ++big;
        } else if (steps[k] >= ADAPTIVE_SMALL) {
            ++small;
        }
    }
    if (small != 0u) {
        return 0.0f;
    }
    return big == 1u ? 0.6f : (big == 2u ? 0.45f : 0.0f);
}

// One channel of a pass's sum, `value`, moved as the reference's `adapt` moves it, by that
// channel's values of the six pixels about the point, `s2` the one it lies in, `f` past it.
static float adapted_channel(float value, float s0, float s1, float s2, float s3, float s4,
                             float s5, float f, float scale, bool reduces) {
    float steps[5] = {abs(s1 - s0) * scale, abs(s2 - s1) * scale, abs(s3 - s2) * scale,
                      abs(s4 - s3) * scale, abs(s5 - s4) * scale};
    float beta = adaptive_blend(steps, reduces);
    float bilinear = (1.0f - f) * s2 + f * s3;
    return value + beta * (bilinear - value);
}

// `value`, a pass's sum at a point a fraction `f` past the pixel it lies in, moved as the
// reference's `adapt` moves it, by the six pixels about the point, `six[2]` the one it lies in:
// its luma, or each of R, G and B. Each channel by name, so that the six stay in registers.
static float4 adapted(constant ScalerParams &p, float4 value, thread const float4 *six, float f,
                      bool reduces) {
    float scale = p.adaptive_scale;
    value.x = adapted_channel(value.x, six[0].x, six[1].x, six[2].x, six[3].x, six[4].x, six[5].x,
                              f, scale, reduces);
    if (p.adaptive_channels > 1u) {
        value.y = adapted_channel(value.y, six[0].y, six[1].y, six[2].y, six[3].y, six[4].y,
                                  six[5].y, f, scale, reduces);
        value.z = adapted_channel(value.z, six[0].z, six[1].z, six[2].z, six[3].z, six[4].z,
                                  six[5].z, f, scale, reduces);
    }
    return value;
}

// Pixel `k` of the six about a point in pixel `b`, `b + k - 2`, clamped to the axis's span.
static uint six_at(int b, uint k, uint first, uint last) {
    return uint(clamp(b - 2 + int(k), int(first), int(last)));
}

// Each of the six about a point in pixel `b` that is pixel `at` takes `value`: the six by name, so
// that they stay in registers.
static void six_put(thread float4 *six, int b, uint first, uint last, uint at, float4 value) {
    for (uint k = 0; k < 6u; ++k) {
        if (six_at(b, k, first, last) == at) {
            six[k] = value;
        }
    }
}

// Of the six about a point in pixel `b`, the one that is pixel `at`, which lies among them.
static float4 six_of(thread const float4 *six, int b, uint at) {
    int k = int(at) - (b - 2);
    float4 value = six[0];
    for (int i = 1; i < 6; ++i) {
        if (i == k) {
            value = six[i];
        }
    }
    return value;
}

// An RGB source's pixel (sx, sy), in the working space: its channels' codes through `light`
// where the transform goes into linear light.
static float4 rgb_pixel(constant ScalerParams &p, device const uint *source,
                        threadgroup const float *light, uint sx, uint sy) {
    uint at = p.source_luma_offset + sy * p.source_luma_pitch + sx * 4u;
    uint4 bytes;
    if ((at & 3u) == 0u) {
        // The pixel's word, read once.
        uint word = source[at >> 2];
        bytes = uint4(word >> (8u * p.source_order[0]), word >> (8u * p.source_order[1]),
                      word >> (8u * p.source_order[2]), word >> (8u * p.source_order[3])) &
                0xffu;
    } else {
        bytes = uint4(byte_at(source, at + p.source_order[0]),
                      byte_at(source, at + p.source_order[1]),
                      byte_at(source, at + p.source_order[2]),
                      byte_at(source, at + p.source_order[3]));
    }
    if (p.linear != 0u) {
        return float4(light[bytes.x], light[bytes.y], light[bytes.z], float(bytes.w) / 255.0f);
    }
    return float4(bytes) / 255.0f;
}

// How far turned pixel `v` lies past the source window's first, `origin`: none before it, whose
// chroma is the first sample's as the reference interpolates it.
static uint from_origin(uint v, uint origin) {
    return v > origin ? v - origin : 0u;
}

// The Cb and Cr codes of chroma sample (cx, cy) of the turned source window, clamped to its
// samples: the source's sample under the turned pixel it sits on (the reference's
// `decode_turned`), as the window is turned as a picture of its own.
static float2 chroma_codes(constant ScalerParams &p, device const uint *source, uint cx,
                           uint cy) {
    uint tx = min(p.chroma_origin_x + p.chroma_step_x * min(cx, p.chroma_last_x), p.turned_last_x);
    uint ty = min(p.chroma_origin_y + p.chroma_step_y * min(cy, p.chroma_last_y), p.turned_last_y);
    int2 s = mapped(p.luma_map, tx, ty);
    uint2 c = uint2(step_of(p.source_subsample_x, uint(s.x)).x,
                    step_of(p.source_subsample_y, uint(s.y)).x);
    uint at = p.source_chroma_offset + c.y * p.source_chroma_pitch;
    float cb;
    float cr;
    if (p.source_packing == PACKING_BIPLANAR) {
        at += c.x * 2u * p.source_sample_bytes;
        uint pair = at & (p.source_sample_bytes * 2u - 1u);
        if (pair == 0u || (p.source_sample_bytes == 1u && (at & 1u) == 0u)) {
            // Cb and Cr in one word, read once.
            uint word = source[at >> 2];
            uint shift = (at & 3u) * 8u;
            if (p.source_sample_bytes == 1u) {
                cb = float((word >> shift) & 0xffu);
                cr = float((word >> (shift + 8u)) & 0xffu);
            } else {
                cb = float((word & 0xffffu) >> 6u);
                cr = float(word >> 22u);
            }
        } else {
            cb = sample_at(source, at, p.source_sample_bytes);
            cr = sample_at(source, at + p.source_sample_bytes, p.source_sample_bytes);
        }
    } else {
        at += c.x * 4u;
        cb = float(byte_at(source, at + p.source_order[1]));
        cr = float(byte_at(source, at + p.source_order[3]));
    }
    return float2(cb, cr);
}

// The chroma samples a pixel's chroma is interpolated from: read from the source as they are
// asked for (`Loaded`), or four read once, from (bx, by), that a quad of pixels shares (`Grid`).
struct Loaded {
    device const uint *source;

    float2 at(constant ScalerParams &p, uint cx, uint cy) const {
        return chroma_codes(p, source, cx, cy);
    }
};

struct Grid {
    float2 s00;
    float2 s10;
    float2 s01;
    float2 s11;
    uint bx;
    uint by;

    float2 at(constant ScalerParams &p, uint cx, uint cy) const {
        bool right = cx != bx;
        return cy != by ? (right ? s11 : s01) : (right ? s10 : s00);
    }
};

// Pixel (x, y) of a Y'CbCr source turned, in Y'CbCr: its luma, and each chroma sample on the
// first of the turned pixels it covers, its codes interpolated between and the sum rounded down.
template <typename Samples>
static float4 ycbcr_pixel(constant ScalerParams &p, device const uint *source,
                          thread const Samples &samples, uint x, uint y) {
    int2 s = mapped(p.luma_map, x, y);
    uint sx = uint(s.x);
    uint row = p.source_luma_offset + uint(s.y) * p.source_luma_pitch;
    uint luma_at = p.source_packing == PACKING_BIPLANAR
                       ? row + sx * p.source_sample_bytes
                       : row + (sx / 2u) * 4u + p.source_order[(sx & 1u) * 2u];
    float luma = (sample_at(source, luma_at, p.source_sample_bytes) - p.source_luma_low) *
                 p.source_luma_scale;
    uint2 across = step_of(p.chroma_step_x, from_origin(x, p.chroma_origin_x));
    uint2 down = step_of(p.chroma_step_y, from_origin(y, p.chroma_origin_y));
    // `remainder / step`: a half or none, the step being 1 or 2.
    float ax = 0.5f * float(across.y);
    float ay = 0.5f * float(down.y);
    float2 codes = 0.0f;
    for (uint j = 0; j < 2u; ++j) {
        for (uint i = 0; i < 2u; ++i) {
            float weight = (i == 1u ? ax : 1.0f - ax) * (j == 1u ? ay : 1.0f - ay);
            if (weight == 0.0f) {
                continue;
            }
            codes += weight * samples.at(p, across.x + i, down.x + j);
        }
    }
    float2 chroma = (floor(codes) - p.source_chroma_middle) * p.source_chroma_scale;
    return float4(luma, chroma.x, chroma.y, 1.0f);
}

// Pixel (x, y) of the turned source, as it is resampled: an RGB source's in the working space, a
// Y'CbCr source's in Y'CbCr.
static float4 decoded(constant ScalerParams &p, device const uint *source,
                      threadgroup const float *light, uint x, uint y) {
    if (p.source_packing == PACKING_RGB) {
        int2 s = mapped(p.luma_map, x, y);
        return rgb_pixel(p, source, light, uint(s.x), uint(s.y));
    }
    Loaded samples = {source};
    return ycbcr_pixel(p, source, samples, x, y);
}

// Pixels (x, y), (x + 1, y), (x, y + 1) and (x + 1, y + 1) of a Y'CbCr source turned, in
// Y'CbCr, into `out` in that order: those of them `right` and `below` say the source holds. Their
// chroma takes at most two samples each way, from the first pixel's first, which the four read
// once. The values are `ycbcr_pixel`'s.
static void ycbcr_quad(constant ScalerParams &p, device const uint *source, uint x, uint y,
                       bool right, bool below, thread float4 *out) {
    uint bx = step_of(p.chroma_step_x, from_origin(x, p.chroma_origin_x)).x;
    uint by = step_of(p.chroma_step_y, from_origin(y, p.chroma_origin_y)).x;
    float2 samples[4];
    for (uint k = 0; k < 4u; ++k) {
        samples[k] = chroma_codes(p, source, bx + (k & 1u), by + (k >> 1u));
    }
    Grid grid = {samples[0], samples[1], samples[2], samples[3], bx, by};
    for (uint k = 0; k < 4u; ++k) {
        if (((k & 1u) != 0u && !right) || ((k >> 1u) != 0u && !below)) {
            continue;
        }
        out[k] = ycbcr_pixel(p, source, grid, x + (k & 1u), y + (k >> 1u));
    }
}

// The same four pixels of the turned source, in the working space: `decoded`'s values.
static void decoded_quad(constant ScalerParams &p, device const uint *source,
                         threadgroup const float *light, uint x, uint y, bool right, bool below,
                         thread float4 *out) {
    if (p.source_packing != PACKING_RGB) {
        ycbcr_quad(p, source, x, y, right, below, out);
        return;
    }
    for (uint k = 0; k < 4u; ++k) {
        if (((k & 1u) != 0u && !right) || ((k >> 1u) != 0u && !below)) {
            continue;
        }
        int2 s = mapped(p.luma_map, x + (k & 1u), y + (k >> 1u));
        out[k] = rgb_pixel(p, source, light, uint(s.x), uint(s.y));
    }
}

// Tap `k` of column `dx` of the destination window, and of row `dy`. The taps buffer holds an
// axis's taps a tap at a time, that tap of every pixel of the axis in turn, so that the threads of
// neighbouring pixels read neighbouring taps.
static ScalerTap across_tap(constant ScalerParams &p, device const ScalerTap *taps, uint k,
                            uint dx) {
    return taps[p.across_first + k * p.window_width + dx];
}

static ScalerTap down_tap(constant ScalerParams &p, device const ScalerTap *taps, uint k,
                          uint dy) {
    return taps[p.down_first + k * p.window_height + dy];
}

// Whether a Y'CbCr source's chroma takes taps of its own across, and down.
static bool chroma_apart_across(constant ScalerParams &p) {
    return p.chroma_across_first != p.across_first;
}

static bool chroma_apart_down(constant ScalerParams &p) {
    return p.chroma_down_first != p.down_first;
}

// Tap `k` of the chroma of column `dx`, and of row `dy`: the luma's where it takes the luma's.
static ScalerTap chroma_across_tap(constant ScalerParams &p, device const ScalerTap *taps, uint k,
                                   uint dx) {
    return taps[p.chroma_across_first + k * p.window_width + dx];
}

static ScalerTap chroma_down_tap(constant ScalerParams &p, device const ScalerTap *taps, uint k,
                                 uint dy) {
    return taps[p.chroma_down_first + k * p.window_height + dy];
}

// Column x of the turned source taken down for row dy of the destination window by the `count`
// taps of a row from `first`: their weighted sum of its pixels.
static float4 column_sum(constant ScalerParams &p, device const uint *source,
                         threadgroup const float *light, device const ScalerTap *taps, uint first,
                         uint count, uint x, uint dy) {
    float4 sum = 0.0f;
    for (uint j = 0; j < count; ++j) {
        ScalerTap row = taps[first + j * p.window_height + dy];
        if (row.weight == 0.0f) {
            continue;
        }
        sum += row.weight * decoded(p, source, light, x, row.index);
    }
    return sum;
}

// Column x taken down for row dy: the down taps' weighted sum of its pixels; with `MOVES`, moved
// by the adaptive filter where the pass down moves its sums. Then one loop decodes each of the six
// pixels about the point once, and each tap's pixel, a tap among the six taking it from there: so
// the code decodes a pixel at one place, which keeps `scaler_transform_adaptive` small enough for
// Vulkan drivers to make within a transform's wait (docs/video/scaler.md, "The kernel").
template <bool MOVES>
static float4 line_at(constant ScalerParams &p, device const uint *source,
                      threadgroup const float *light, device const ScalerTap *taps, uint x,
                      uint dy) {
    if (!MOVES) {
        return column_sum(p, source, light, taps, p.down_first, p.down_count, x, dy);
    }
    bool adapt = p.adapt_down != 0u;
    ScalerTap point = {0u, 0.0f};
    if (adapt) {
        point = taps[p.down_points_first + dy];
    }
    int b = as_type<int>(point.index);
    // The six's first and last rows, and how many there are: none where the sums stay.
    uint lo = six_at(b, 0u, p.down_span_first, p.down_span_last);
    uint hi = six_at(b, 5u, p.down_span_first, p.down_span_last);
    uint rows = adapt ? hi - lo + 1u : 0u;
    float4 six[6];
    float4 sum = 0.0f;
    for (uint i = 0; i < rows + p.down_count; ++i) {
        bool tap = i >= rows;
        ScalerTap row = {lo + i, 0.0f};
        if (tap) {
            row = down_tap(p, taps, i - rows, dy);
            if (row.weight == 0.0f) {
                continue;
            }
        }
        float4 pixel = tap && rows != 0u && row.index >= lo && row.index <= hi
                           ? six_of(six, b, row.index)
                           : decoded(p, source, light, x, row.index);
        if (tap) {
            sum += row.weight * pixel;
        } else {
            six_put(six, b, p.down_span_first, p.down_span_last, row.index, pixel);
        }
    }
    if (adapt) {
        sum = adapted(p, sum, six, point.weight, p.down_reduces != 0u);
    }
    return sum;
}

static float2 chroma_line_at(constant ScalerParams &p, device const uint *source,
                             threadgroup const float *light, device const ScalerTap *taps, uint x,
                             uint dy) {
    return column_sum(p, source, light, taps, p.chroma_down_first, p.chroma_down_count, x, dy).yz;
}

// A sum of the across taps as the destination takes it: a Y'CbCr source's through RGB where the
// destination is RGB (the reference's `in_space`), and back out of linear light where it went
// into it.
static float4 finished(constant ScalerParams &p, float4 sum) {
    if (p.ycbcr_to_rgb != 0u) {
        return to_rgb(p, sum);
    }
    return p.linear != 0u ? float4(linear_to_srgb(sum.xyz), sum.w) : sum;
}

// Pixel (dx, dy)'s across taps' weighted sum of its columns taken down, moved by the adaptive
// filter where the pass across moves its sums: the six columns about the point and each tap's
// taken down in one loop, as `line_at` takes rows.
static float4 moved_working(constant ScalerParams &p, device const uint *source,
                            threadgroup const float *light, device const ScalerTap *taps, uint dx,
                            uint dy) {
    bool adapt = p.adapt_across != 0u;
    ScalerTap point = {0u, 0.0f};
    if (adapt) {
        point = taps[p.across_points_first + dx];
    }
    int b = as_type<int>(point.index);
    uint lo = six_at(b, 0u, p.across_span_first, p.across_span_last);
    uint hi = six_at(b, 5u, p.across_span_first, p.across_span_last);
    uint columns = adapt ? hi - lo + 1u : 0u;
    float4 six[6];
    float4 sum = 0.0f;
    for (uint i = 0; i < columns + p.across_count; ++i) {
        bool tap = i >= columns;
        ScalerTap column = {lo + i, 0.0f};
        if (tap) {
            column = across_tap(p, taps, i - columns, dx);
            if (column.weight == 0.0f) {
                continue;
            }
        }
        float4 line = tap && columns != 0u && column.index >= lo && column.index <= hi
                          ? six_of(six, b, column.index)
                          : line_at<true>(p, source, light, taps, column.index, dy);
        if (tap) {
            sum += column.weight * line;
        } else {
            six_put(six, b, p.across_span_first, p.across_span_last, column.index, line);
        }
    }
    if (adapt) {
        sum = adapted(p, sum, six, point.weight, p.across_reduces != 0u);
    }
    return sum;
}

// Pixel (dx, dy) of the destination window, in the working space, of a transform that is not
// separable: the across taps' weighted sum of its columns taken down, and a Y'CbCr source's
// chroma by its own taps where it has them.
template <bool MOVES>
static float4 working(constant ScalerParams &p, device const uint *source,
                      threadgroup const float *light, device const ScalerTap *taps, uint dx,
                      uint dy) {
    bool apart = chroma_apart_across(p) || chroma_apart_down(p);
    float4 sum = 0.0f;
    if (!MOVES) {
        for (uint i = 0; i < p.across_count; ++i) {
            ScalerTap column = across_tap(p, taps, i, dx);
            if (column.weight == 0.0f) {
                continue;
            }
            sum += column.weight * line_at<false>(p, source, light, taps, column.index, dy);
        }
    } else {
        sum = moved_working(p, source, light, taps, dx, dy);
    }
    if (apart) {
        float2 chroma = 0.0f;
        for (uint i = 0; i < p.chroma_across_count; ++i) {
            ScalerTap column = chroma_across_tap(p, taps, i, dx);
            if (column.weight == 0.0f) {
                continue;
            }
            chroma += column.weight * chroma_line_at(p, source, light, taps, column.index, dy);
        }
        sum.yz = chroma;
    }
    return finished(p, sum);
}

static uint code(constant ScalerParams &p, float v) {
    return uint(clamp(round(v), 0.0f, p.code_max));
}

static uint luma_code(constant ScalerParams &p, float y) {
    return code(p, p.luma_low + p.luma_span * y);
}

static uint chroma_code(constant ScalerParams &p, float c) {
    return code(p, p.chroma_middle + p.chroma_span * c);
}

// The pixels `[from, to)` along one axis of the block of `size` at `first` that the siting names,
// within the window's `[low, high)`.
static uint2 named(uint site, uint size, uint first, uint low, uint high) {
    uint from = first;
    uint to = first + size;
    if (size > 1u) {
        if (site == 0u) {
            to = first + 1u;
        } else if (site == 2u) {
            from = first + 1u;
        }
    }
    return uint2(max(from, low), min(to, high));
}

// The most pixels of the turned source a threadgroup of `scaler_lines` decodes once each, into
// threadgroup memory: 12 KiB of float4s, 48 rows of its 16 columns, which 16 rows of the window
// take at a reduction of up to 2.5 with 9 taps, and 8 at one of up to 4.
constant uint SHARED_PIXELS = 768u;

// Each column of the turned source the window's across taps take, taken down for each row of a
// band of the window: a thread a line. A threadgroup's rows of the window take a run of the
// source's rows, from the first its first row's taps take to the last its last row's take: where
// the run's pixels in the threadgroup's columns fit its memory, its threads decode each of them
// once, and every tap reads its pixel there; a tap outside the run, and every tap of a run too
// long, decodes its pixel itself. The values, and the sums' order, are the same either way.
kernel void scaler_lines(device const uint *source [[buffer(0)]],
                         device const ScalerTap *taps [[buffer(2)]],
                         constant ScalerParams &p [[buffer(3)]],
                         device float4 *lines [[buffer(4)]],
                         device float *lines3 [[buffer(5)]],
                         uint2 at [[thread_position_in_grid]],
                         uint2 in_group [[thread_position_in_threadgroup]],
                         uint2 group_size [[threads_per_threadgroup]]) {
    threadgroup float4 pixels[SHARED_PIXELS];
    threadgroup float light[LIGHT_CODES];
    uint columns = group_size.x;
    uint lane = in_group.y * columns + in_group.x;
    work_out_light(p, light, lane, columns * group_size.y);
    // The threadgroup's first column, and its first and last rows of the window.
    uint x0 = at.x - in_group.x;
    uint top = p.lines_row + at.y - in_group.y;
    uint bottom = p.lines_row + min(at.y - in_group.y + group_size.y, p.lines_rows) - 1u;
    uint first = 0xffffffffu;
    uint last = 0u;
    for (uint j = 0; j < p.down_count; ++j) {
        ScalerTap head = down_tap(p, taps, j, top);
        ScalerTap tail = down_tap(p, taps, j, bottom);
        if (head.weight != 0.0f) {
            first = min(first, head.index);
        }
        if (tail.weight != 0.0f) {
            last = max(last, tail.index);
        }
    }
    bool apart = chroma_apart_down(p);
    for (uint j = 0; apart && j < p.chroma_down_count; ++j) {
        ScalerTap head = chroma_down_tap(p, taps, j, top);
        ScalerTap tail = chroma_down_tap(p, taps, j, bottom);
        if (head.weight != 0.0f) {
            first = min(first, head.index);
        }
        if (tail.weight != 0.0f) {
            last = max(last, tail.index);
        }
    }
    bool adapt = p.adapt_down != 0u;
    if (adapt) {
        // The six rows about the first row's point and the last's.
        int b_top = as_type<int>(taps[p.down_points_first + top].index);
        int b_bottom = as_type<int>(taps[p.down_points_first + bottom].index);
        first = min(first, six_at(b_top, 0u, p.down_span_first, p.down_span_last));
        last = max(last, six_at(b_bottom, 5u, p.down_span_first, p.down_span_last));
    }
    bool shared = first <= last && (last - first + 1u) * columns <= SHARED_PIXELS;
    if (shared) {
        // A thread a quad of two by two pixels, which share their chroma samples.
        uint rows = last - first + 1u;
        uint quads = (rows + 1u) / 2u * (columns / 2u);
        for (uint k = lane; k < quads; k += columns * group_size.y) {
            uint qx = (k % (columns / 2u)) * 2u;
            uint qy = (k / (columns / 2u)) * 2u;
            uint column = x0 + qx;
            if (column >= p.lines_pitch) {
                continue;
            }
            bool right = column + 1u < p.lines_pitch;
            bool below = qy + 1u < rows;
            float4 quad[4];
            decoded_quad(p, source, light, p.lines_first + column, first + qy, right, below,
                         quad);
            uint at = qy * columns + qx;
            pixels[at] = quad[0];
            if (right) {
                pixels[at + 1u] = quad[1];
            }
            if (below) {
                pixels[at + columns] = quad[2];
                if (right) {
                    pixels[at + columns + 1u] = quad[3];
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (at.x >= p.lines_pitch || at.y >= p.lines_rows) {
        return;
    }
    uint x = p.lines_first + at.x;
    uint dy = p.lines_row + at.y;
    float4 sum = 0.0f;
    for (uint j = 0; j < p.down_count; ++j) {
        ScalerTap row = down_tap(p, taps, j, dy);
        if (row.weight == 0.0f) {
            continue;
        }
        uint k = row.index - first;
        float4 pixel = shared && k <= last - first
                           ? pixels[k * columns + in_group.x]
                           : decoded(p, source, light, x, row.index);
        sum += row.weight * pixel;
    }
    if (apart) {
        float2 chroma = 0.0f;
        for (uint j = 0; j < p.chroma_down_count; ++j) {
            ScalerTap row = chroma_down_tap(p, taps, j, dy);
            if (row.weight == 0.0f) {
                continue;
            }
            uint k = row.index - first;
            float4 pixel = shared && k <= last - first
                               ? pixels[k * columns + in_group.x]
                               : decoded(p, source, light, x, row.index);
            chroma += row.weight * pixel.yz;
        }
        sum.yz = chroma;
    }
    if (adapt) {
        ScalerTap point = taps[p.down_points_first + dy];
        int b = as_type<int>(point.index);
        float4 six[6];
        if (shared) {
            // The run holds the six of every row: a row's point lies at or past the first row's
            // and at or before the last row's.
            for (uint k = 0; k < 6u; ++k) {
                uint row = six_at(b, k, p.down_span_first, p.down_span_last);
                six[k] = pixels[(row - first) * columns + in_group.x];
            }
        } else {
            // Each decoded once, in turn.
            uint lo = six_at(b, 0u, p.down_span_first, p.down_span_last);
            uint hi = six_at(b, 5u, p.down_span_first, p.down_span_last);
            for (uint row = lo; row <= hi; ++row) {
                six_put(six, b, p.down_span_first, p.down_span_last, row,
                        decoded(p, source, light, x, row));
            }
        }
        sum = adapted(p, sum, six, point.weight, p.down_reduces != 0u);
    }
    uint line = at.y * p.lines_pitch + at.x;
    if (p.source_packing == PACKING_RGB) {
        lines[line] = sum;
    } else {
        // A Y'CbCr source's pixels are opaque: a line's three values alone.
        lines3[3u * line] = sum.x;
        lines3[3u * line + 1u] = sum.y;
        lines3[3u * line + 2u] = sum.z;
    }
}

// Where a plane's pixels come from: worked out from the source, for a transform that is not
// separable.
template <bool MOVES>
struct Direct {
    device const uint *source;
    device const ScalerTap *taps;
    threadgroup const float *light;

    float4 working_at(constant ScalerParams &p, uint dx, uint dy) const {
        return working<MOVES>(p, source, light, taps, dx, dy);
    }
};

// The RGB plane of a copy of a Y'CbCr source (one tap of weight 1 a pixel, turned or not), a
// threadgroup of 16 by 4 threads `QUAD_WORDS` words by `QUAD_ROWS` rows of it. First a thread a
// quad of two by two pixels, which share the chroma samples they are interpolated from where their
// source pixels lie two by two as well, into threadgroup memory; then the threads write the words
// a row at a time, side by side. A discrete GPU writes guest memory in system memory across the
// bus, which takes stores a word apart at its speed and stores with gaps between them far slower
// (measured: reims-vgpu's `scaler/gpu.rs`, "Speed"). The dispatch is of whole threadgroups, so
// that every thread of one reaches its barrier.
constant uint QUAD_WORDS = 32u;
constant uint QUAD_ROWS = 8u;

kernel void scaler_quads(device const uint *source [[buffer(0)]],
                         device uint *destination [[buffer(1)]],
                         device const ScalerTap *taps [[buffer(2)]],
                         constant ScalerParams &p [[buffer(3)]],
                         uint2 at [[thread_position_in_grid]],
                         uint2 in_group [[thread_position_in_threadgroup]]) {
    threadgroup uint words[QUAD_WORDS * QUAD_ROWS];
    uint x_first = p.first_byte / 4u;
    uint x_end = p.end_byte / 4u;
    uint y_end = p.first_row + p.rows;
    uint x0 = x_first + 2u * at.x;
    uint y0 = p.first_row + 2u * at.y;
    if (x0 < x_end && y0 < y_end) {
        bool right = x0 + 1u < x_end;
        bool below = y0 + 1u < y_end;
        uint dx = x0 - p.window_x;
        uint dy = y0 - p.window_y;
        uint a0 = across_tap(p, taps, 0u, dx).index;
        uint d0 = down_tap(p, taps, 0u, dy).index;
        uint a1 = right ? across_tap(p, taps, 0u, dx + 1u).index : a0;
        uint d1 = below ? down_tap(p, taps, 0u, dy + 1u).index : d0;
        float4 v[4];
        if ((!right || a1 == a0 + 1u) && (!below || d1 == d0 + 1u)) {
            ycbcr_quad(p, source, a0, d0, right, below, v);
        } else {
            for (uint k = 0; k < 4u; ++k) {
                if (((k & 1u) != 0u && !right) || ((k >> 1u) != 0u && !below)) {
                    continue;
                }
                Loaded samples = {source};
                v[k] = ycbcr_pixel(p, source, samples, (k & 1u) != 0u ? a1 : a0,
                                   (k >> 1u) != 0u ? d1 : d0);
            }
        }
        for (uint k = 0; k < 4u; ++k) {
            if (((k & 1u) != 0u && !right) || ((k >> 1u) != 0u && !below)) {
                continue;
            }
            float4 rgb = finished(p, v[k]);
            uint value = 0u;
            for (uint c = 0; c < 4u; ++c) {
                value |= code(p, rgb[c] * 255.0f) << (8u * p.plane_order[c]);
            }
            words[(2u * in_group.y + (k >> 1u)) * QUAD_WORDS + 2u * in_group.x + (k & 1u)] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint lane = in_group.y * (QUAD_WORDS / 2u) + in_group.x;
    uint group_x = x_first + 2u * (at.x - in_group.x);
    uint group_y = p.first_row + 2u * (at.y - in_group.y);
    for (uint k = lane; k < QUAD_WORDS * QUAD_ROWS; k += QUAD_WORDS * QUAD_ROWS / 4u) {
        uint x = group_x + k % QUAD_WORDS;
        uint y = group_y + k / QUAD_WORDS;
        if (x < x_end && y < y_end) {
            destination[(p.plane_offset + y * p.plane_pitch + x * 4u) >> 2] = words[k];
        }
    }
}

// Where a plane's pixels come from for a separable transform: summed across the lines
// `scaler_lines` wrote.
struct Lines {
    device const ScalerTap *taps;
    device const float4 *lines;
    device const float *lines3;

    float4 working_at(constant ScalerParams &p, uint dx, uint dy) const;
};

// The pixels of the destination a word of a plane covers each way, as the plane's words lie in
// its rows and its rows in the destination's: a word of luma or RGB covers its pixels; one of
// chroma the pixels of its samples; one of both planes of a 4:2:0 window the pixels of its
// chroma samples, in their two rows.
static uint2 covered(constant ScalerParams &p) {
    switch (p.plane_kind) {
    case KIND_LUMA8:
        return uint2(4u, 1u);
    case KIND_LUMA16:
    case KIND_PACKED422:
        return uint2(2u, 1u);
    case KIND_CHROMA8:
        return uint2(2u * p.subsample_x, p.subsample_y);
    case KIND_CHROMA16:
        return uint2(p.subsample_x, p.subsample_y);
    case KIND_BIPLANAR8:
        return uint2(4u, 2u);
    case KIND_BIPLANAR16:
        return uint2(2u, 2u);
    default:
        return uint2(1u, 1u);
    }
}

// The pixels chroma sample (bx, by) of the destination is the mean of: the window's pixels its
// siting names, or all the window's pixels it covers where the window holds none of those. Its
// columns, from the first to past the last, then its rows.
static uint4 chroma_pixels(constant ScalerParams &p, uint bx, uint by) {
    uint x_low = p.window_x;
    uint x_high = p.window_x + p.window_width;
    uint y_low = p.window_y;
    uint y_high = p.window_y + p.window_height;
    uint2 xs = named(p.site_x, p.subsample_x, bx * p.subsample_x, x_low, x_high);
    uint2 ys = named(p.site_y, p.subsample_y, by * p.subsample_y, y_low, y_high);
    if (xs.x >= xs.y || ys.x >= ys.y) {
        xs = named(1u, p.subsample_x, bx * p.subsample_x, x_low, x_high);
        ys = named(1u, p.subsample_y, by * p.subsample_y, y_low, y_high);
    }
    return uint4(xs, ys);
}

// Writes the bytes of `value` that `mask` names into word `word` of `destination`, and leaves the
// rest of it as it was.
static void store(device uint *destination, uint word, uint value, uint mask) {
    if (mask == 0u) {
        return;
    }
    if (mask != 0xffffffffu) {
        value = (destination[word] & ~mask) | (value & mask);
    }
    destination[word] = value;
}

// Pixel (dx, dy) of the window, in Y'CbCr.
template <typename From>
static float4 ycbcr_at(constant ScalerParams &p, thread const From &from, uint dx, uint dy) {
    float4 v = from.working_at(p, dx, dy);
    return p.rgb != 0u ? to_ycbcr(p, v) : v;
}

// Both planes of a 4:2:0 window whose every edge is even, the first byte of whose word `first` of
// chroma row `row` the window covers: that word of chroma, and the words of luma of the same
// pixels in the two rows it covers, each pixel worked out once. Every chroma sample's two by two
// pixels lie in the window, so its siting names pixels of its own (`chroma_pixels` without the
// window's edges), and it is their mean. Its loop over the two rows has a constant count, which
// the driver unrolls into two copies of a pixel's work, worked out side by side: faster than
// `write_word`'s loop, and more code, which only `scaler_transform_both_planes` holds
// (docs/video/scaler.md, "Making the kernels").
template <typename From>
static void both_planes(constant ScalerParams &p, thread const From &from,
                        device uint *destination, uint row, uint first) {
    uint bytes = p.plane_kind == KIND_BIPLANAR8 ? 1u : 2u;
    uint shift = bytes == 1u ? 0u : 6u;
    uint x0 = first / bytes;
    float4 v[2][4];
    uint luma[2] = {0u, 0u};
    uint luma_mask = 0u;
    for (uint k = 0; k < 4u / bytes; ++k) {
        uint b = first + k * bytes;
        if (b < p.first_byte || b >= p.end_byte) {
            continue;
        }
        for (uint r = 0; r < 2u; ++r) {
            v[r][k] = ycbcr_at(p, from, x0 + k - p.window_x, 2u * row + r - p.window_y);
            luma[r] |= (luma_code(p, v[r][k].x) << shift) << (8u * bytes * k);
        }
        luma_mask |= (bytes == 1u ? 0xffu : 0xffffu) << (8u * bytes * k);
    }
    uint chroma = 0u;
    uint chroma_mask = 0u;
    uint2 xs = named(p.site_x, 2u, 0u, 0u, 2u);
    uint2 ys = named(p.site_y, 2u, 0u, 0u, 2u);
    for (uint q = 0; q < 2u / bytes; ++q) {
        uint b = first + 2u * bytes * q;
        if (b < p.first_byte || b >= p.end_byte) {
            continue;
        }
        float2 sum = 0.0f;
        float n = 0.0f;
        for (uint py = ys.x; py < ys.y; ++py) {
            for (uint px = xs.x; px < xs.y; ++px) {
                sum += v[py][2u * q + px].yz;
                n += 1.0f;
            }
        }
        float2 c = sum / n;
        uint cb = chroma_code(p, c.x) << shift;
        uint cr = chroma_code(p, c.y) << shift;
        uint pair = cb | (cr << (8u * bytes));
        chroma |= pair << (16u * q);
        chroma_mask |= (bytes == 1u ? 0xffffu : 0xffffffffu) << (16u * q);
    }
    for (uint r = 0; r < 2u; ++r) {
        store(destination, (p.plane_offset + (2u * row + r) * p.plane_pitch + first) >> 2, luma[r],
              luma_mask);
    }
    store(destination, (p.chroma_plane_offset + row * p.chroma_plane_pitch + first) >> 2, chroma,
          chroma_mask);
}

// Word `first / 4` of row `row` of the dispatch's plane: the bytes of it the window covers, each
// worked out as the reference works it out, and the rest of it as it was; of both planes of a
// 4:2:0 window whose every edge is even, that word of chroma and the words of luma of the same
// pixels in the two rows it covers. The word's pixels are a block of the destination, `covered`
// each way. One loop takes them a row at a time and works out once each the word takes (its
// luma or RGB, or its chroma sample's mean takes it), at the one place a kernel works a pixel
// out, so that the kernel holds one copy of that code, not one for each kind of plane (the
// driver's making of a kernel follows the size of its code once every call is inlined,
// docs/video/scaler.md, "Making the kernels"). A chroma sample is the mean of its pixels
// (`chroma_pixels`), which the loop reaches in rows, as the reference takes them.
template <typename From>
static void write_word(constant ScalerParams &p, thread const From &from,
                       device uint *destination, uint row, uint first) {
    uint kind = p.plane_kind;
    bool rgb32 = kind == KIND_RGB32;
    bool both = kind == KIND_BIPLANAR8 || kind == KIND_BIPLANAR16;
    bool packed = kind == KIND_PACKED422;
    bool has_chroma = kind == KIND_CHROMA8 || kind == KIND_CHROMA16 || packed || both;
    bool has_luma = kind == KIND_LUMA8 || kind == KIND_LUMA16 || both;
    // A sample's bytes, a 10-bit code's place in its 16 bits, and a chroma sample's bytes.
    uint bytes = kind == KIND_LUMA16 || kind == KIND_CHROMA16 || kind == KIND_BIPLANAR16 ? 2u : 1u;
    uint shift = bytes == 1u ? 0u : 6u;
    uint pair_bytes = packed ? 4u : 2u * bytes;
    // Whether byte `b` of the row lies in what the window covers.
    #define COVERED(b) ((b) >= p.first_byte && (b) < p.end_byte)
    if (rgb32) {
        // A word of RGB is one pixel, worked out on its own: the loop's tests for the other
        // kinds' words cost a cheap transform's pixels more than their own work.
        if (COVERED(first)) {
            float4 v = from.working_at(p, (first >> 2) - p.window_x, row - p.window_y);
            uint value = 0u;
            for (uint k = 0; k < 4u; ++k) {
                value |= code(p, v[k] * 255.0f) << (8u * p.plane_order[k]);
            }
            store(destination, (p.plane_offset + row * p.plane_pitch + first) >> 2, value,
                  0xffffffffu);
        }
        return;
    }
    // The word's first chroma sample, and the block's first pixel and size, each way a power of
    // two, so that the steps are shifts and loops, not divisions.
    uint bx = first >> (pair_bytes == 4u ? 2u : 1u);
    uint x0 = has_chroma ? bx * p.subsample_x : first >> (rgb32 ? 2u : bytes - 1u);
    uint y0 = has_chroma ? row * p.subsample_y : row;
    uint2 size = covered(p);
    // Each chroma sample's pixels, and whether the word holds it and the window covers its bytes:
    // a word holds a second where its pixels are two samples wide.
    uint4 pixels0 = uint4(0u);
    uint4 pixels1 = uint4(0u);
    if (both) {
        // Every edge of the window is even, so each sample's two by two pixels lie in it, and
        // its siting names pixels of its own (`chroma_pixels` without the window's edges).
        uint2 xs = named(p.site_x, 2u, x0, x0, x0 + 4u);
        uint2 ys = named(p.site_y, 2u, y0, y0, y0 + 2u);
        pixels0 = uint4(xs, ys);
        pixels1 = uint4(xs + 2u, ys);
    } else if (has_chroma) {
        pixels0 = chroma_pixels(p, bx, row);
        pixels1 = chroma_pixels(p, bx + 1u, row);
    }
    bool sample0 = has_chroma && COVERED(first);
    bool sample1 = has_chroma && size.x > p.subsample_x && COVERED(first + pair_bytes);
    uint luma0 = 0u;  // the luma or RGB word, or a 4:2:0 window's first row's luma
    uint luma1 = 0u;  // its second row's
    uint luma_mask = 0u;
    float2 sum0 = 0.0f;
    float2 sum1 = 0.0f;
    float n0 = 0.0f;
    float n1 = 0.0f;
    for (uint py = 0; py < size.y; ++py) {
        for (uint px = 0; px < size.x; ++px) {
            uint x = x0 + px;
            uint y = y0 + py;
            // Where its luma goes in its word, and whether the window covers it.
            uint at = packed ? 8u * p.plane_order[2u * px] : 8u * bytes * px;
            bool luma = packed ? COVERED(first) && x >= p.window_x &&
                                     x < p.window_x + p.window_width
                               : (has_luma && COVERED(first + bytes * px)) ||
                                     (rgb32 && COVERED(first));
            // Whether it is one of its chroma sample's pixels.
            bool second = has_chroma && px >= p.subsample_x;
            uint4 c = second ? pixels1 : pixels0;
            bool sampled = (second ? sample1 : sample0) && x >= c.x && x < c.y && y >= c.z &&
                           y < c.w;
            if (!luma && !sampled) {
                continue;
            }
            float4 v = from.working_at(p, x - p.window_x, y - p.window_y);
            if (rgb32) {
                for (uint k = 0; k < 4u; ++k) {
                    luma0 |= code(p, v[k] * 255.0f) << (8u * p.plane_order[k]);
                }
                luma_mask = 0xffffffffu;
                continue;
            }
            if (p.rgb != 0u) {
                v = to_ycbcr(p, v);
            }
            if (luma) {
                uint bits = (luma_code(p, v.x) << shift) << at;
                luma0 |= py == 0u ? bits : 0u;
                luma1 |= py == 0u ? 0u : bits;
                luma_mask |= (bytes == 1u ? 0xffu : 0xffffu) << at;
            }
            if (sampled && !second) {
                sum0 += v.yz;
                n0 += 1.0f;
            } else if (sampled) {
                sum1 += v.yz;
                n1 += 1.0f;
            }
        }
    }
    // The chroma samples' codes.
    uint chroma = 0u;
    uint chroma_mask = 0u;
    for (uint q = 0; q < 2u; ++q) {
        if (!(q == 0u ? sample0 : sample1)) {
            continue;
        }
        float2 c = q == 0u ? sum0 / n0 : sum1 / n1;
        uint cb = chroma_code(p, c.x);
        uint cr = chroma_code(p, c.y);
        if (packed) {
            chroma = (cb << (8u * p.plane_order[1])) | (cr << (8u * p.plane_order[3]));
            chroma_mask = (0xffu << (8u * p.plane_order[1])) | (0xffu << (8u * p.plane_order[3]));
        } else {
            chroma |= ((cb << shift) | ((cr << shift) << (8u * bytes))) << (16u * q);
            chroma_mask |= (bytes == 1u ? 0xffffu : 0xffffffffu) << (16u * q);
        }
    }
    #undef COVERED
    if (both) {
        for (uint r = 0; r < 2u; ++r) {
            store(destination, (p.plane_offset + (2u * row + r) * p.plane_pitch + first) >> 2,
                  r == 0u ? luma0 : luma1, luma_mask);
        }
        store(destination, (p.chroma_plane_offset + row * p.chroma_plane_pitch + first) >> 2,
              chroma, chroma_mask);
        return;
    }
    store(destination, (p.plane_offset + row * p.plane_pitch + first) >> 2, luma0 | chroma,
          luma_mask | chroma_mask);
}

// A plane of the destination, of a transform that is not separable: a thread a word of it.
kernel void scaler_transform(device const uint *source [[buffer(0)]],
                             device uint *destination [[buffer(1)]],
                             device const ScalerTap *taps [[buffer(2)]],
                             constant ScalerParams &p [[buffer(3)]],
                             uint2 at [[thread_position_in_grid]],
                             uint2 in_group [[thread_position_in_threadgroup]],
                             uint2 group_size [[threads_per_threadgroup]]) {
    threadgroup float light[LIGHT_CODES];
    work_out_light(p, light, in_group.y * group_size.x + in_group.x, group_size.x * group_size.y);
    if (at.x >= p.words || at.y >= p.rows) {
        return;
    }
    Direct<false> from = {source, taps, light};
    write_word(p, from, destination, p.first_row + at.y, (p.first_byte & ~3u) + at.x * 4u);
}

// The same, for both planes of a 4:2:0 window whose every edge is even, of a transform whose
// passes the adaptive filter does not move (`both_planes`).
kernel void scaler_transform_both_planes(device const uint *source [[buffer(0)]],
                                         device uint *destination [[buffer(1)]],
                                         device const ScalerTap *taps [[buffer(2)]],
                                         constant ScalerParams &p [[buffer(3)]],
                                         uint2 at [[thread_position_in_grid]],
                                         uint2 in_group [[thread_position_in_threadgroup]],
                                         uint2 group_size [[threads_per_threadgroup]]) {
    threadgroup float light[LIGHT_CODES];
    work_out_light(p, light, in_group.y * group_size.x + in_group.x, group_size.x * group_size.y);
    if (at.x >= p.words || at.y >= p.rows) {
        return;
    }
    Direct<false> from = {source, taps, light};
    both_planes(p, from, destination, p.first_row + at.y, (p.first_byte & ~3u) + at.x * 4u);
}

// The same, for a transform whose passes the adaptive filter moves: a kernel of its own, so that
// the transforms it does not move run, and are made, without its code.
kernel void scaler_transform_adaptive(device const uint *source [[buffer(0)]],
                                      device uint *destination [[buffer(1)]],
                                      device const ScalerTap *taps [[buffer(2)]],
                                      constant ScalerParams &p [[buffer(3)]],
                                      uint2 at [[thread_position_in_grid]],
                                      uint2 in_group [[thread_position_in_threadgroup]],
                                      uint2 group_size [[threads_per_threadgroup]]) {
    threadgroup float light[LIGHT_CODES];
    work_out_light(p, light, in_group.y * group_size.x + in_group.x, group_size.x * group_size.y);
    if (at.x >= p.words || at.y >= p.rows) {
        return;
    }
    Direct<true> from = {source, taps, light};
    write_word(p, from, destination, p.first_row + at.y, (p.first_byte & ~3u) + at.x * 4u);
}

// The threads of a threadgroup of `scaler_across`, as the CPU dispatches it: 16 words of a
// plane's rows by 4 of its rows.
constant uint ACROSS_WORDS = 16u;
constant uint ACROSS_ROWS = 4u;

// The most pixels a threadgroup of `scaler_across_shared` works out: its 16 words of a 4:2:0
// window's both planes, of four pixels each, by its 4 rows of two.
constant uint ACROSS_PIXELS = 512u;

// Line `line` of the lines buffer: four floats, or a Y'CbCr source's three, whose pixels are
// opaque.
static float4 line_value(constant ScalerParams &p, device const float4 *lines,
                         device const float *lines3, uint line) {
    return p.source_packing == PACKING_RGB
               ? lines[line]
               : float4(lines3[3u * line], lines3[3u * line + 1u], lines3[3u * line + 2u], 1.0f);
}

// Pixel (dx, dy) of the destination window, in the working space, of a separable transform: the
// across taps' weighted sum of the lines of its row, moved by the adaptive filter where the pass
// across moves its sums.
static float4 across_lines(constant ScalerParams &p, device const ScalerTap *taps,
                           device const float4 *lines, device const float *lines3, uint dx,
                           uint dy) {
    float4 sum = 0.0f;
    uint row = (dy - p.lines_row) * p.lines_pitch - p.lines_first;
    for (uint i = 0; i < p.across_count; ++i) {
        ScalerTap column = across_tap(p, taps, i, dx);
        if (column.weight == 0.0f) {
            continue;
        }
        sum += column.weight * line_value(p, lines, lines3, row + column.index);
    }
    if (chroma_apart_across(p)) {
        // A Y'CbCr source's chroma, by its own taps.
        float2 chroma = 0.0f;
        for (uint i = 0; i < p.chroma_across_count; ++i) {
            ScalerTap column = chroma_across_tap(p, taps, i, dx);
            if (column.weight == 0.0f) {
                continue;
            }
            uint line = row + column.index;
            chroma += column.weight * float2(lines3[3u * line + 1u], lines3[3u * line + 2u]);
        }
        sum.yz = chroma;
    }
    if (p.adapt_across != 0u) {
        ScalerTap point = taps[p.across_points_first + dx];
        int b = as_type<int>(point.index);
        float4 six[6];
        for (uint k = 0; k < 6u; ++k) {
            // A Y'CbCr source's luma alone, the one channel the filter reads.
            uint line = row + six_at(b, k, p.across_span_first, p.across_span_last);
            six[k] = p.source_packing == PACKING_RGB ? lines[line]
                                                     : float4(lines3[3u * line], 0.0f, 0.0f, 0.0f);
        }
        sum = adapted(p, sum, six, point.weight, p.across_reduces != 0u);
    }
    // A Y'CbCr source's pixels are opaque.
    if (p.source_packing != PACKING_RGB) {
        sum.w = 1.0f;
    }
    return finished(p, sum);
}

float4 Lines::working_at(constant ScalerParams &p, uint dx, uint dy) const {
    return across_lines(p, taps, lines, lines3, dx, dy);
}

// A plane of the destination, of a separable transform, over a band of the window's rows: a
// thread a word of it, its pixels each the across taps' sum of the lines `scaler_lines` wrote.
kernel void scaler_across(device uint *destination [[buffer(1)]],
                          device const ScalerTap *taps [[buffer(2)]],
                          constant ScalerParams &p [[buffer(3)]],
                          device const float4 *lines [[buffer(4)]],
                          device const float *lines3 [[buffer(5)]],
                          uint2 at [[thread_position_in_grid]]) {
    if (at.x >= p.words || at.y >= p.rows) {
        return;
    }
    Lines from = {taps, lines, lines3};
    write_word(p, from, destination, p.first_row + at.y, (p.first_byte & ~3u) + at.x * 4u);
}

// A threadgroup's pixels of a plane of `scaler_across_shared`, worked out: from `first`, `width` a
// row, each its code (a luma code, or an RGB pixel's word) and its Cb and Cr.
struct Block {
    threadgroup const uint *codes;
    threadgroup const float2 *chroma;
    uint2 first;
    uint width;

    uint at(uint x, uint y) const {
        return (y - first.y) * width + x - first.x;
    }
};

// Chroma sample (bx, by)'s Cb and Cr, from the block: the mean of its pixels (`chroma_pixels`), in
// rows, as `write_word` takes it.
static float2 block_chroma(constant ScalerParams &p, thread const Block &b, uint bx, uint by) {
    uint4 c = chroma_pixels(p, bx, by);
    float2 sum = 0.0f;
    float n = 0.0f;
    for (uint py = c.z; py < c.w; ++py) {
        for (uint px = c.x; px < c.y; ++px) {
            sum += b.chroma[b.at(px, py)];
            n += 1.0f;
        }
    }
    return sum / n;
}

// A plane of the destination, of a separable transform, over a band of the window's rows, from
// the lines `scaler_lines` wrote. A threadgroup's threads first work out its words' pixels, a
// thread a pixel at a time and neighbouring threads neighbouring pixels, so that they read
// neighbouring lines and taps: each pixel's code and Cb and Cr, into threadgroup memory. Then
// each writes its word from there, as `write_word` writes it. The values, and the sums' order,
// are those `scaler_transform` would take.
kernel void scaler_across_shared(device uint *destination [[buffer(1)]],
                                 device const ScalerTap *taps [[buffer(2)]],
                                 constant ScalerParams &p [[buffer(3)]],
                                 device const float4 *lines [[buffer(4)]],
                                 device const float *lines3 [[buffer(5)]],
                                 uint2 at [[thread_position_in_grid]],
                                 uint2 in_group [[thread_position_in_threadgroup]]) {
    threadgroup uint codes[ACROSS_PIXELS];
    threadgroup float2 chroma[ACROSS_PIXELS];
    uint lane = in_group.y * ACROSS_WORDS + in_group.x;
    uint2 size = covered(p);
    // The threadgroup's pixels: from its first word's first, a power of two a row.
    uint2 group = uint2((p.first_byte >> 2) + at.x - in_group.x, p.first_row + at.y - in_group.y);
    uint2 first = group * size;
    uint width = ACROSS_WORDS * size.x;
    uint count = width * ACROSS_ROWS * size.y;
    uint x_low = p.window_x;
    uint x_high = p.window_x + p.window_width;
    uint y_low = p.window_y + p.lines_row;
    uint y_high = y_low + p.lines_rows;
    for (uint k = lane; k < count; k += ACROSS_WORDS * ACROSS_ROWS) {
        uint x = first.x + (k & (width - 1u));
        uint y = first.y + k / width;
        if (x < x_low || x >= x_high || y < y_low || y >= y_high) {
            continue;
        }
        float4 v = across_lines(p, taps, lines, lines3, x - p.window_x, y - p.window_y);
        if (p.plane_kind == KIND_RGB32) {
            uint word = 0u;
            for (uint c = 0; c < 4u; ++c) {
                word |= code(p, v[c] * 255.0f) << (8u * p.plane_order[c]);
            }
            codes[k] = word;
        } else {
            float4 ycc = p.rgb != 0u ? to_ycbcr(p, v) : v;
            codes[k] = luma_code(p, ycc.x);
            chroma[k] = ycc.yz;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (at.x >= p.words || at.y >= p.rows) {
        return;
    }
    Block b = {codes, chroma, first, width};
    uint row = p.first_row + at.y;
    uint word_first = (p.first_byte & ~3u) + at.x * 4u;
    uint value = 0u;
    uint mask = 0u;
    // Whether byte `b` of the row lies in what the window covers.
    #define COVERED(byte) ((byte) >= p.first_byte && (byte) < p.end_byte)
    switch (p.plane_kind) {
    case KIND_LUMA8:
        for (uint k = 0; k < 4u; ++k) {
            if (COVERED(word_first + k)) {
                value |= b.codes[b.at(word_first + k, row)] << (8u * k);
                mask |= 0xffu << (8u * k);
            }
        }
        break;
    case KIND_LUMA16:
        for (uint k = 0; k < 2u; ++k) {
            uint byte = word_first + 2u * k;
            if (COVERED(byte)) {
                value |= (b.codes[b.at(byte / 2u, row)] << 6u) << (16u * k);
                mask |= 0xffffu << (16u * k);
            }
        }
        break;
    case KIND_CHROMA8:
        for (uint k = 0; k < 2u; ++k) {
            uint byte = word_first + 2u * k;
            if (COVERED(byte)) {
                float2 c = block_chroma(p, b, byte / 2u, row);
                value |= (chroma_code(p, c.x) | (chroma_code(p, c.y) << 8u)) << (16u * k);
                mask |= 0xffffu << (16u * k);
            }
        }
        break;
    case KIND_CHROMA16:
        if (COVERED(word_first)) {
            float2 c = block_chroma(p, b, word_first / 4u, row);
            value = (chroma_code(p, c.x) << 6u) | ((chroma_code(p, c.y) << 6u) << 16u);
            mask = 0xffffffffu;
        }
        break;
    case KIND_PACKED422:
        if (COVERED(word_first)) {
            uint pair = word_first / 4u;
            float2 c = block_chroma(p, b, pair, row);
            value = (chroma_code(p, c.x) << (8u * p.plane_order[1])) |
                    (chroma_code(p, c.y) << (8u * p.plane_order[3]));
            mask = (0xffu << (8u * p.plane_order[1])) | (0xffu << (8u * p.plane_order[3]));
            for (uint k = 0; k < 2u; ++k) {
                uint x = pair * 2u + k;
                if (x >= p.window_x && x < p.window_x + p.window_width) {
                    uint shift = 8u * p.plane_order[2u * k];
                    value |= b.codes[b.at(x, row)] << shift;
                    mask |= 0xffu << shift;
                }
            }
        }
        break;
    case KIND_RGB32:
        if (COVERED(word_first)) {
            value = b.codes[b.at(word_first / 4u, row)];
            mask = 0xffffffffu;
        }
        break;
    case KIND_BIPLANAR8:
    case KIND_BIPLANAR16: {
        uint bytes = p.plane_kind == KIND_BIPLANAR8 ? 1u : 2u;
        uint shift = bytes == 1u ? 0u : 6u;
        uint x0 = word_first / bytes;
        uint luma[2] = {0u, 0u};
        uint luma_mask = 0u;
        for (uint k = 0; k < 4u / bytes; ++k) {
            if (!COVERED(word_first + k * bytes)) {
                continue;
            }
            for (uint r = 0; r < 2u; ++r) {
                luma[r] |= (b.codes[b.at(x0 + k, 2u * row + r)] << shift) << (8u * bytes * k);
            }
            luma_mask |= (bytes == 1u ? 0xffu : 0xffffu) << (8u * bytes * k);
        }
        uint2 xs = named(p.site_x, 2u, 0u, 0u, 2u);
        uint2 ys = named(p.site_y, 2u, 0u, 0u, 2u);
        for (uint q = 0; q < 2u / bytes; ++q) {
            if (!COVERED(word_first + 2u * bytes * q)) {
                continue;
            }
            float2 sum = 0.0f;
            float n = 0.0f;
            for (uint py = ys.x; py < ys.y; ++py) {
                for (uint px = xs.x; px < xs.y; ++px) {
                    sum += b.chroma[b.at(x0 + 2u * q + px, 2u * row + py)];
                    n += 1.0f;
                }
            }
            float2 c = sum / n;
            uint pair = (chroma_code(p, c.x) << shift) |
                        ((chroma_code(p, c.y) << shift) << (8u * bytes));
            value |= pair << (16u * q);
            mask |= (bytes == 1u ? 0xffffu : 0xffffffffu) << (16u * q);
        }
        for (uint r = 0; r < 2u; ++r) {
            store(destination, (p.plane_offset + (2u * row + r) * p.plane_pitch + word_first) >> 2,
                  luma[r], luma_mask);
        }
        store(destination,
              (p.chroma_plane_offset + row * p.chroma_plane_pitch + word_first) >> 2, value, mask);
        return;
    }
    default:
        break;
    }
    #undef COVERED
    store(destination, (p.plane_offset + row * p.plane_pitch + word_first) >> 2, value, mask);
}
