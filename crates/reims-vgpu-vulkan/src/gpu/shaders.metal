// The display's present shaders, compiled into the Vulkan platform's shaders.metallib (build.rs),
// which gpu/shaders.rs loads once per device. Core's display/pipeline.rs and reims-vgpu's
// graphics/sink.rs make their pipelines and bind their arguments; reims-vgpu's
// graphics/vaapi/fill.rs makes the video encoder's input fill's.

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

// The display pipe's colour as a table of every input value (graphics/sink.rs): for a colour with
// no matrix, each output channel depends only on the same channel's input, and a frame whose
// channels take `n` values (an 8-bit or 10-bit unorm surface, or the scanout's bytes) has `n`
// inputs per channel. Entry k of `lut` (n × 1, in the drawable's format) is what the colour's
// function above drew for a pixel of value k in every channel, so each channel is looked up at
// its own value, and a pixel comes out as the colour's function would draw it. Alpha is 1, as
// there.

// A presented frame, sampled as `displayPipeColorFragment` samples it: its value in each
// channel is the nearest of the table's n.
fragment float4 displayPipeLutFragment(PresentVertex in [[stage_in]],
                                       texture2d<float> frame [[texture(0)]],
                                       texture2d<float, access::read> lut [[texture(1)]]) {
    float3 v = frame.sample(frame_sampler, in.tex).rgb;
    uint3 k = uint3(rint(clamp(v, 0.0f, 1.0f) * float(lut.get_width() - 1)));
    return float4(lut.read(uint2(k.r, 0)).r, lut.read(uint2(k.g, 0)).g, lut.read(uint2(k.b, 0)).b,
                  1.0f);
}

// The guest's scanout, read in place as `displayPipeColorScanoutFragment` reads it: each byte is
// its channel's entry.
fragment float4 displayPipeLutScanoutFragment(PresentVertex in [[stage_in]],
                                              device const uchar4 *scanout [[buffer(4)]],
                                              constant DisplayPipeScanout &at [[buffer(5)]],
                                              texture2d<float, access::read> lut [[texture(1)]]) {
    uint2 p = uint2(in.position.xy) - at.origin;
    uchar4 bgra = scanout[at.first + p.y * at.stride + p.x];
    return float4(lut.read(uint2(bgra.z, 0)).r, lut.read(uint2(bgra.y, 0)).g,
                  lut.read(uint2(bgra.x, 0)).b, 1.0f);
}

// ---- The window's mirror on the compositor's GPU ----

// The window's mirror (reims-vgpu's graphics/mirror.rs): a presented frame's bytes, `rows` rows
// `words_per_row` 32-bit words apart, whose first `row_words` words are its pixels, in 64-row
// tiles `tile_words` words wide. `all` is 1 where every tile goes.
struct DisplayMirrorFrame {
    uint words_per_row;
    uint row_words;
    uint rows;
    uint tile_words;
    uint all;
};

// The threads of a mirror tile's threadgroup.
constant uint display_mirror_threads = 256;

// Whether tile `tile` of the presented frame `now` differs in any of its words from the frame the
// mirror holds (`sent`, the frame its last sync copied), or every tile goes, as each thread of
// the tile's threadgroup (`index` among them) learns; `count` and `width` are the tile's words and
// its words a row.
static bool display_mirror_tile_changed(device const uint *now,
                                        device const uint *sent,
                                        constant DisplayMirrorFrame &frame,
                                        uint x0, uint y0, uint width, uint count, uint index,
                                        threadgroup uint *differs,
                                        threadgroup uint &changed) {
    uint mine = 0;
    for (uint i = index; i < count; i += display_mirror_threads) {
        uint at = (y0 + i / width) * frame.words_per_row + x0 + i % width;
        mine |= now[at] ^ sent[at];
    }
    differs[index] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (index == 0) {
        uint any = frame.all;
        for (uint t = 0; t < display_mirror_threads; t++) {
            any |= differs[t];
        }
        changed = any;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return changed != 0;
}

// Tile `tile` of the presented frame `now`: where it changed (`display_mirror_tile_changed`), all
// of its words are copied into the mirror (`mirror`, in host memory) and `tiles` counts it. A tile
// that changed nothing writes nothing.
kernel void displayMirrorTile(device const uint *now [[buffer(0)]],
                              device const uint *sent [[buffer(1)]],
                              device uint *mirror [[buffer(2)]],
                              constant DisplayMirrorFrame &frame [[buffer(3)]],
                              device atomic_uint *tiles [[buffer(4)]],
                              uint2 tile [[threadgroup_position_in_grid]],
                              uint index [[thread_index_in_threadgroup]]) {
    threadgroup uint differs[display_mirror_threads];
    threadgroup uint changed;
    uint x0 = tile.x * frame.tile_words;
    uint y0 = tile.y * 64;
    uint width = min(x0 + frame.tile_words, frame.row_words) - x0;
    uint count = width * (min(y0 + 64, frame.rows) - y0);
    if (!display_mirror_tile_changed(now, sent, frame, x0, y0, width, count, index, differs,
                                     changed)) {
        return;
    }
    if (index == 0) {
        atomic_fetch_add_explicit(tiles, 1u, memory_order_relaxed);
    }
    for (uint i = index; i < count; i += display_mirror_threads) {
        uint at = (y0 + i / width) * frame.words_per_row + x0 + i % width;
        mirror[at] = now[at];
    }
}

// Tile `tile` of the presented frame `now`, where the mirror's copies run on a copy engine: where
// it changed (`display_mirror_tile_changed`), its number (`tile.y` times the tiles a row, plus
// `tile.x`) is added to `list`, whose first word counts the numbers after it, in no order. It
// writes nothing else: the copy engine copies the listed tiles out of `now` into the mirror.
kernel void displayMirrorTileList(device const uint *now [[buffer(0)]],
                                  device const uint *sent [[buffer(1)]],
                                  device atomic_uint *list [[buffer(2)]],
                                  constant DisplayMirrorFrame &frame [[buffer(3)]],
                                  uint2 tile [[threadgroup_position_in_grid]],
                                  uint index [[thread_index_in_threadgroup]]) {
    threadgroup uint differs[display_mirror_threads];
    threadgroup uint changed;
    uint x0 = tile.x * frame.tile_words;
    uint y0 = tile.y * 64;
    uint width = min(x0 + frame.tile_words, frame.row_words) - x0;
    uint count = width * (min(y0 + 64, frame.rows) - y0);
    if (!display_mirror_tile_changed(now, sent, frame, x0, y0, width, count, index, differs,
                                     changed) ||
        index != 0) {
        return;
    }
    uint across = (frame.row_words + frame.tile_words - 1) / frame.tile_words;
    uint at = atomic_fetch_add_explicit(&list[0], 1u, memory_order_relaxed);
    atomic_store_explicit(&list[1 + at], tile.y * across + tile.x, memory_order_relaxed);
}

// ---- The host's video encoder's input ----

// One plane of a frame that videoEncodeFill writes into a video encoder's input (reims-vgpu's
// graphics/vaapi/fill.rs, `FillPlane`, 36 bytes): where the frame's plane lies in the source
// buffer as bound, its first byte and the bytes from one row to the next; the bytes of a row of the
// frame's samples, its rows, and the bytes a row's margin repeats (a sample, or a Cb/Cr pair), or
// 0 for a row with no margin, whose last word keeps the input's bytes past the row (a decode's
// copy into its output's planes, graphics/vaapi/gpu_copy.rs); the input's words a row, its rows
// and its words from one row to the next; and how the input lays its words out: 0 linear, 1 in
// Intel's Y tiles.
struct VideoFillPlane {
    uint first;
    uint source_pitch;
    uint row_bytes;
    uint rows;
    uint unit;
    uint coded_words;
    uint coded_rows;
    uint pitch_words;
    uint tiling;
};

// Where word `x` of row `y` of a plane of the input lies, in words from the plane's first: rows
// `pitch_words` apart where linear; in Intel's Y tiles, 4 KiB tiles of 128 bytes by 32 rows,
// `pitch_words / 32` of them to a row of tiles, each 8 columns of 16 bytes by 32 rows one after
// another, a column's rows 16 bytes apart (the VA crate's `TargetTiling::byte_at`).
static uint video_fill_word_at(uint x, uint y, uint pitch_words, uint tiling) {
    if (tiling == 0) {
        return y * pitch_words + x;
    }
    uint tile = (y >> 5) * (pitch_words >> 5) + (x >> 5);
    return tile * 1024 + ((x & 31) >> 2) * 128 + (y & 31) * 4 + (x & 3);
}

// The source's byte `at`, of the words the buffer holds from its bound offset (little-endian).
static uint video_fill_byte(device const uint *source, uint at) {
    return (source[at >> 2] >> ((at & 3) * 8)) & 0xffu;
}

// Word `at.x` of row `at.y` of a plane of the encoder's input: the frame's row there (its last
// for a row past them), and past the row's bytes its last `unit` bytes repeated, as the VA crate's
// upload fills the margin to the coded size. 32-bit loads and stores alone; a word within the
// row's bytes is one load where it starts on a word, two where it does not. With no margin
// (`unit` 0), a row's last word that holds 1 to 3 of its bytes takes those from the source and
// keeps the input's others, by a 32-bit atomic AND that clears the row's bytes and an atomic OR
// that sets them: no thread writes a byte past the row, as the CPU's copy of a row writes its
// bytes alone, with no 8-bit store, which not every device makes.
kernel void videoEncodeFill(device const uint *source [[buffer(0)]],
                            device uint *input [[buffer(1)]],
                            constant VideoFillPlane &plane [[buffer(2)]],
                            uint2 at [[thread_position_in_grid]]) {
    if (at.x >= plane.coded_words || at.y >= plane.coded_rows) {
        return;
    }
    uint row = plane.first + min(at.y, plane.rows - 1) * plane.source_pitch;
    uint x = at.x * 4;
    uint word;
    if (x + 4 <= plane.row_bytes) {
        uint from = row + x;
        uint shift = (from & 3) * 8;
        word = source[from >> 2];
        if (shift != 0) {
            word = (word >> shift) | (source[(from >> 2) + 1] << (32 - shift));
        }
    } else if (plane.unit == 0) {
        uint bytes = plane.row_bytes - x;
        word = 0;
        for (uint i = 0; i < bytes; i++) {
            word |= video_fill_byte(source, row + x + i) << (i * 8);
        }
        uint kept = 0xffffffffu << (bytes * 8);
        device atomic_uint *to = reinterpret_cast<device atomic_uint *>(
            input + video_fill_word_at(at.x, at.y, plane.pitch_words, plane.tiling));
        atomic_fetch_and_explicit(to, kept, memory_order_relaxed);
        atomic_fetch_or_explicit(to, word, memory_order_relaxed);
        return;
    } else {
        uint last = plane.row_bytes - plane.unit;
        word = 0;
        for (uint i = 0; i < 4; i++) {
            uint byte_x = x + i;
            uint from = byte_x < plane.row_bytes ? byte_x
                                                 : last + (byte_x - plane.row_bytes) % plane.unit;
            word |= video_fill_byte(source, row + from) << (i * 8);
        }
    }
    input[video_fill_word_at(at.x, at.y, plane.pitch_words, plane.tiling)] = word;
}

// ---- A 4:4:4 picture into a surface's two planes ----

// One plane of a surface that videoGather444 writes from a 4:4:4 picture (reims-vgpu's
// graphics/gather.rs, `GatherPlane`, 40 bytes): in the source as bound, the byte of the first
// sample of each component the plane takes in the picture's first row, `first` and `second`
// (luma's twice, or Cb's and Cr's), the bytes from one of the picture's rows to the next, and
// from one sample of a component to the next (1 in a plane of its own, 4 in XYUV), and the
// components a pixel of the plane holds (1 for luma, 2 for a Cb/Cr pair); then the plane's bytes
// written a row, its rows written, its words a row written, its first word in the destination as
// bound, and its words from one row to the next.
struct VideoGatherPlane {
    uint first;
    uint second;
    uint source_pitch;
    uint step;
    uint components;
    uint row_bytes;
    uint rows;
    uint row_words;
    uint out_first;
    uint pitch_words;
};

// Word `at.x` of row `at.y` of a surface's plane: each of its bytes within the row the sample of
// its pixel and component, read from the picture's row by 32-bit loads, as videoEncodeFill reads.
// A row's last word that holds 1 to 3 of its bytes keeps the plane's other bytes, read and written
// back by the one thread that writes the word: the plane's rows are whole words apart, so no other
// thread of the pass writes it.
kernel void videoGather444(device const uint *source [[buffer(0)]],
                           device uint *plane_out [[buffer(1)]],
                           constant VideoGatherPlane &plane [[buffer(2)]],
                           uint2 at [[thread_position_in_grid]]) {
    if (at.x >= plane.row_words || at.y >= plane.rows) {
        return;
    }
    uint row = at.y * plane.source_pitch;
    uint x = at.x * 4;
    uint bytes = min(4u, plane.row_bytes - x);
    uint word = 0;
    for (uint i = 0; i < bytes; i++) {
        uint byte_x = x + i;
        uint first = byte_x % plane.components == 0 ? plane.first : plane.second;
        uint from = first + row + (byte_x / plane.components) * plane.step;
        word |= video_fill_byte(source, from) << (i * 8);
    }
    uint to = plane.out_first + at.y * plane.pitch_words + at.x;
    if (bytes < 4) {
        word |= plane_out[to] & (0xffffffffu << (bytes * 8));
    }
    plane_out[to] = word;
}
