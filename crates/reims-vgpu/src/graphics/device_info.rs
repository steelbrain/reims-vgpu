//! The stage-3 device-info caps blob: the host fills [`RgfxDeviceInfo`] from the
//! platform's [`DeviceCaps`] and [`IdentityCaps`], so the guest bundle's `MTLDevice`
//! answers `supportsFamily:` of every family, `supportsFeatureSet:` of every
//! feature set, its identity (the host GPU's real name among it), the limits and
//! the feature queries as the host device does on the Vulkan platform. A query the platform cannot answer (a runtime gap, an `Err`
//! or `None` in the caps) is reported as unsupported here, never guessed yes.
//! Where the lane's own device can do less than the host GPU, its answer is the
//! lane's ([`lane`]'s table).
//!
//! The decoder does not touch this: [`super::GraphicsDevice`] writes it once at
//! the device's init into the region at
//! [`RGFX_DEVICE_INFO_OFFSET`](reims_vgpu_graphics::wire::RGFX_DEVICE_INFO_OFFSET),
//! and the guest bundle reads it there, not a guest command.

use reims_vgpu_graphics::wire::{
    RGFX_DEVICE_ARCHITECTURE_NAME_LEN, RGFX_DEVICE_INFO_COUNTER_POINTS, RGFX_DEVICE_INFO_FAMILIES,
    RGFX_DEVICE_INFO_FEATURE_SETS, RGFX_DEVICE_INFO_FORMATS, RGFX_DEVICE_INFO_IDENTITY_FLAGS,
    RGFX_DEVICE_INFO_SIZES, RGFX_DEVICE_INFO_SPARSE_TILES, RGFX_DEVICE_INFO_VERSION,
    RGFX_DEVICE_NAME_LEN, RgfxDeviceInfo, RgfxFormatAnswers,
};
use reims_vgpu_vulkan::core::platform::caps::{
    COUNTER_SAMPLING_POINTS, DeviceCaps, FAMILIES, FEATURE_SETS, Flags, FormatAnswers,
    IdentityCaps, Limits, SDK_FAMILIES, SDK_PIXEL_FORMATS, SPARSE_PAGE_SIZES, SPARSE_SAMPLE_COUNTS,
    feature_set_family,
};
use reims_vgpu_vulkan::core::raise::RaiseResult;

use crate::GuestArch;

/// `MTLGPUFamilyMetal3`.
const MTL_GPU_FAMILY_METAL3: i64 = 5001;
/// `MTLGPUFamilyMetal4`.
const MTL_GPU_FAMILY_METAL4: i64 = 5002;

/// Whether the device makes mesh render pipelines and draws them (`RGFX_FEATURE_MESH_PIPELINES`):
/// where the platform answers `supportsFamily:` of Metal3, whose rows of Apple's Metal Feature Set
/// Tables hold mesh shading, yes. Every Mac GPU that answers Metal3 makes them, the x86 Macs' AMD
/// and Intel GPUs too (measured: `guest/bundle/tools/oracle/oracle_mesh` on the AMD Radeon Pro
/// 5300M and the Intel UHD Graphics 630, macOS 26.7.1, and the M3 Max, macOS 27.0.1).
pub(crate) fn makes_mesh_pipelines(identity: &IdentityCaps) -> bool {
    identity.families[at(MTL_GPU_FAMILY_METAL3)] == Some(true)
}

/// 1 when the query answered a present `true`; 0 for a present `false` or a host
/// gap (an `Err`). A gap is never reported as support.
fn bit(answer: &RaiseResult<bool>) -> u8 {
    u8::from(matches!(answer, Ok(true)))
}

/// 1 for a present `true`; 0 for a `false` or a query the platform has no answer of.
fn known(answer: Option<bool>) -> u8 {
    u8::from(answer == Some(true))
}

/// `text` in `out`, cut at `out`'s length on a character's boundary; its length.
fn put_text(out: &mut [u8], text: &str) -> u32 {
    let mut len = text.len().min(out.len());
    while !text.is_char_boundary(len) {
        len -= 1;
    }
    out[..len].copy_from_slice(&text.as_bytes()[..len]);
    // At most 256 bytes: lossless.
    len as u32
}

/// `answers` as the blob holds them; a value the blob's field cannot hold (no such answer is
/// known) is 0, no answer.
fn format_answers(answers: &FormatAnswers) -> RgfxFormatAnswers {
    let mut out = RgfxFormatAnswers {
        format: u32::try_from(answers.format).unwrap_or(0),
        texture_buffer_alignment: answers
            .texture_buffer_alignment
            .and_then(|a| u32::try_from(a).ok())
            .unwrap_or(0),
        ..RgfxFormatAnswers::default()
    };
    if let Some(tiles) = answers.sparse_tiles {
        for (out, tile) in out.sparse_tiles.iter_mut().zip(tiles) {
            if let (Ok(w), Ok(h), Ok(d)) = (
                u16::try_from(tile[0]),
                u16::try_from(tile[1]),
                u16::try_from(tile[2]),
            ) {
                *out = [w, h, d];
            }
        }
    }
    out
}

/// The index of `family` in [`SDK_FAMILIES`].
fn at(family: i64) -> usize {
    SDK_FAMILIES
        .iter()
        .position(|&f| f == family)
        .unwrap_or_else(|| unreachable!("{family} is an SDK family"))
}

/// A feature that the rows of Apple's Metal Feature Set Tables require of an Apple GPU family.
/// [`WIRED`] lists those the lane's wire, decoder and bundle carry; the bundle answers each other
/// one's members as Metal answers a device without the feature, and marks them
/// `RGFX_UNSUPPORTED`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Piece {
    /// `MTLArgumentEncoder`, from the device, a function and an encoder (argument buffers tier 1).
    ArgumentEncoders,
    /// Placement in an `Automatic` heap, and `heapTextureSizeAndAlignWithDescriptor:` (resource
    /// heaps).
    AutomaticHeaps,
    /// A pass's sample positions and `setDepthStoreActionOptions:` (programmable sample
    /// positions).
    SamplePositions,
    /// `MTLIOCommandQueue` and its file handles (fast resource loading).
    IoCommandQueues,
    /// Texture views, the stencil texture views among them.
    TextureViews,
    /// Indirect command buffers, render and compute, and their executions.
    IndirectCommandBuffers,
    /// Memory barriers in a compute indirect command buffer.
    ComputeIndirectBarriers,
    /// `MTLBinaryArchive`.
    BinaryArchives,
    /// Tile pipelines and a pass's tile size (tile shaders).
    TilePipelines,
    /// `imageblockMemoryLengthForDimensions:` (imageblocks).
    Imageblocks,
    /// A tile dispatch of a nonsquare tile.
    NonsquareTileDispatch,
    /// Sparse textures, and the resource state encoder that maps them.
    SparseTextures,
    /// `MTLResidencySet`.
    ResidencySets,
    /// `MTLLogState` (shader logging).
    ShaderLogging,
    /// Shader validation chosen per pipeline.
    ShaderValidation,
    /// Acceleration structures and intersection functions (ray tracing).
    RayTracing,
    /// Visible function tables and function handles (function pointers).
    FunctionPointers,
    /// `MTLDynamicLibrary`.
    DynamicLibraries,
    /// Rasterization rate maps (variable rasterization rate).
    RasterizationRateMaps,
    /// Mesh pipelines and their draws (mesh shading).
    MeshShading,
    /// `setColorAttachmentMap:`.
    ColorAttachmentMapping,
    /// `MTLTextureViewPool`.
    TextureViewPools,
    /// The Metal 4 programming model.
    Metal4,
    /// Placement sparse buffers and textures.
    PlacementSparse,
    /// Mesh draws in an indirect command buffer.
    IndirectMeshDraws,
    /// Memory barriers in a render indirect command buffer.
    RenderIndirectBarriers,
    /// `setDepthTestMinBound:maxBound:` (depth bounds testing).
    DepthBounds,
}

/// The pieces each Apple family's rows add to the family before it, Apple1 to Apple11, read from
/// the 2026-10-23 edition of Apple's Metal Feature Set Tables. Its first Apple column is Apple2:
/// Apple1, which left the tables with the A7, answers as Apple2, as Metal answers on an M3 Max and
/// as metal2vulkan's family audit reads it, so Apple1 takes Apple2's pieces. Apple11 has no column
/// in any edition, and adds none.
const APPLE_PIECES: [(i64, &[Piece]); 11] = {
    use Piece as P;
    [
        (
            1001,
            &[
                P::ArgumentEncoders,
                P::AutomaticHeaps,
                P::SamplePositions,
                P::IoCommandQueues,
                P::TextureViews,
            ],
        ),
        (1002, &[]),
        (
            1003,
            &[
                P::IndirectCommandBuffers,
                P::ComputeIndirectBarriers,
                P::BinaryArchives,
            ],
        ),
        (1004, &[P::TilePipelines, P::Imageblocks]),
        (1005, &[P::NonsquareTileDispatch]),
        (
            1006,
            &[
                P::SparseTextures,
                P::ResidencySets,
                P::ShaderLogging,
                P::ShaderValidation,
                P::RayTracing,
                P::FunctionPointers,
                P::DynamicLibraries,
                P::RasterizationRateMaps,
            ],
        ),
        (
            1007,
            &[
                P::MeshShading,
                P::ColorAttachmentMapping,
                P::TextureViewPools,
                P::Metal4,
            ],
        ),
        (1008, &[P::PlacementSparse]),
        (1009, &[P::IndirectMeshDraws, P::RenderIndirectBarriers]),
        (1010, &[P::DepthBounds]),
        (1011, &[]),
    ]
};

/// The pieces the lane carries: argument encoders (the host's layout, `ARGUMENT_LAYOUT`, and the
/// sampler ids `SAMPLER_STATE` answers), texture views (`TEXTURE_VIEW`), placement in an Automatic
/// heap with the heap texture size (`HEAP_TEXTURE_SIZE_AND_ALIGN`), a pass's sample positions
/// (after its colour records, `RGFX_PASS_FLAG_SAMPLE_POSITIONS`) with
/// `SET_DEPTH_STORE_ACTION_OPTIONS`, indirect command buffers, render and compute, with the
/// memory barriers of a compute one's commands (`ICB_CREATE`, `ICB_WRITE` and `ICB_SLOTS`, and
/// their executions' render and compute ops), and two the bundle does in the guest with no wire
/// message: binary archives (`RgfxBinaryArchive`: no compiled code crosses) and IO queues
/// (Metal's own IO queue and file handles over the bundle's device, `RgfxMetalDevice+IO.h`), and
/// mesh shading
/// (`MESH_RENDER_PIPELINE_CREATE` and the object and mesh stages' render ops, where the platform
/// makes mesh pipelines: [`makes_mesh_pipelines`]). Mesh draws in an indirect command buffer are
/// not: no mesh pipeline the bundle makes supports them, as on the x86 Macs.
/// A piece joins this list when its wire messages, its decoding and its bundle members land, and
/// the Apple families it holds back rise with it once every piece up to Apple7's is here
/// ([`cap_apple_families`]).
const WIRED: &[Piece] = &[
    Piece::ArgumentEncoders,
    Piece::AutomaticHeaps,
    Piece::SamplePositions,
    Piece::TextureViews,
    Piece::IndirectCommandBuffers,
    Piece::ComputeIndirectBarriers,
    Piece::BinaryArchives,
    Piece::IoCommandQueues,
    Piece::MeshShading,
];

/// `MTLGPUFamilyApple7`, the M1's: the first Apple family a Mac's GPU answers. A Mac's GPU
/// answers no Apple family (an Intel or AMD GPU) or every one from Apple1 to Apple7 at least (an
/// Apple silicon GPU); none answers Apple1 or Apple2 without Apple7.
const FIRST_MAC_APPLE_FAMILY: i64 = 1007;

/// `identity` with each Apple family answered yes only where the platform answers yes and `wired`
/// holds every piece of its rows and of each Apple family's before it ([`APPLE_PIECES`]) and of
/// Apple7's and each family's before it ([`FIRST_MAC_APPLE_FAMILY`]), and each iOS feature set of
/// a family so lowered answered no with it. Short of Apple7, the device answers no Apple family,
/// as an Intel or AMD Mac's GPU does:
/// the guest's frameworks choose their paths by the families a Mac's GPU answers together, and
/// Apple1 and Apple2 without Apple3 is no Mac's. On those answers RenderBox makes its passes'
/// extra attachments memoryless and uses them in a way Metal fails: every command buffer it
/// renders an app's icon in failed with `MTLCommandBufferErrorInvalidResource` and the icons
/// were blank (measured, an x86-64 macOS 26 guest); with no Apple family it renders them.
fn cap_apple_families(identity: &mut IdentityCaps, wired: &[Piece]) {
    let holds = |pieces: &[Piece]| pieces.iter().all(|piece| wired.contains(piece));
    let mut carried = APPLE_PIECES
        .iter()
        .filter(|(family, _)| *family <= FIRST_MAC_APPLE_FAMILY)
        .all(|(_, pieces)| holds(pieces));
    let mut lowered = [false; SDK_FAMILIES.len()];
    for (family, pieces) in APPLE_PIECES {
        carried &= holds(pieces);
        if !carried {
            identity.families[at(family)] = Some(false);
            lowered[at(family)] = true;
        }
    }
    for (answer, &set) in identity.feature_sets.iter_mut().zip(&FEATURE_SETS) {
        if feature_set_family(set).is_some_and(|family| lowered[at(family)]) {
            *answer = Some(false);
        }
    }
}

/// The guest's device's answers where they are not the platform's: what the lane does not
/// carry yet. The guest's `MTLDevice` is the lane's own, which can do only what the wire, the
/// bundle and the decoder carry, so it answers no to a feature they lack whatever the host GPU
/// answers, and an app takes its fallback instead of sending a selector the bundle does not
/// implement. Each row names what it waits on, and goes when that lands:
///
/// | answer | the lane's | waits on |
/// |---|---|---|
/// | `supportsFamily:` of Apple1 to Apple11, and `supportsFeatureSet:` of the iOS sets that go with them | the platform's, but no from the first Apple family whose rows need a piece the lane lacks, and no for every one while the lane lacks a piece of Apple7's rows or of a family's before it ([`cap_apple_families`]): today no for every one, as an Intel or AMD Mac's GPU answers, though the lane carries Apple1's to Apple3's pieces | each piece in [`APPLE_PIECES`] up to Apple7's the lane lacks, from Apple4's tile pipelines and imageblocks on, then each later family's |
/// | `supportsFamily:` of Mac1, Mac2, Common1 to Common3 and Metal3 | the platform's | nothing [`APPLE_PIECES`] lists: it lists the Apple families' pieces alone, and the pieces of Apple1 and Apple2 these rows were held to (argument encoders, automatic heaps, sample positions and IO command queues) the lane carries |
/// | `supportsFamily:` of Metal4 | no | the `MTL4` API: its command queues and buffers, argument tables, compiler and allocators |
/// | `supportsRaytracing`, `supportsRaytracingFromRender`, `supportsPrimitiveMotionBlur` | no | acceleration structures, their builds, and intersection function tables |
/// | `supportsFunctionPointers`, `supportsFunctionPointersFromRender` | no | visible function tables and function handles |
/// | `supportsDynamicLibraries`, `supportsRenderDynamicLibraries` | no | `newDynamicLibrary:` and linked functions |
/// | `supportsPlacementSparse` | no | placement sparse buffers and textures |
/// | `supportsCounterSampling:` | no, at every point | counter sample buffers |
/// | `supportsRasterizationRateMapWithLayerCount:` | no, of every count | rasterization rate maps |
/// | `maximumConcurrentCompilationTaskCount` | the device's compile threads, which run its pipeline compiles | nothing: the device's own answer |
/// | `maxBufferLength` | the platform's, but at most three quarters of `recommendedMaxWorkingSetSize`, rounded down to 16 KiB ([`lane_max_buffer_length`]) | nothing: a choice |
/// | `minimumLinearTextureAlignmentForPixelFormat:` and the SPI `linearTextureAlignmentBytes` and `deviceLinearTextureAlignmentBytes` | 256 bytes, or the platform's where larger ([`LANE_LINEAR_TEXTURE_ALIGNMENT`]) | nothing: a choice |
/// | `isFramebufferReadSupported` (SPI) | no for an x86-64 guest; the platform's for an arm64 guest ([`lane_flags`]) | nothing: a choice |
/// | `supportsTileShaders`, `supportsNonSquareTileShaders`, `supportsImageBlocks` (SPI) | no ([`lane_flags`]) | tile pipelines, their non-square dispatch and imageblocks, Apple4's and Apple5's pieces |
///
/// The rows marked "a choice" are answers real Macs give, picked from those the lane can keep for
/// what they cost the host (docs/graphics-kext/dgpu-oracle.md, "Reading the oracles").
///
/// `hasUnifiedMemory` is the platform's: the decoder keeps `Managed` resources in one copy or two
/// by the same answer (docs/graphics-kext/design.md, Decided 4).
fn lane(mut identity: IdentityCaps, compile_threads: u64) -> IdentityCaps {
    cap_apple_families(&mut identity, WIRED);
    identity.families[at(MTL_GPU_FAMILY_METAL4)] = Some(false);
    let f = &mut identity.flags;
    for no in [
        &mut f.supports_raytracing,
        &mut f.supports_raytracing_from_render,
        &mut f.supports_primitive_motion_blur,
        &mut f.supports_function_pointers,
        &mut f.supports_function_pointers_from_render,
        &mut f.supports_dynamic_libraries,
        &mut f.supports_render_dynamic_libraries,
        &mut f.supports_placement_sparse,
    ] {
        *no = Some(false);
    }
    identity.counter_sampling = [Some(false); 5];
    identity.sizes.largest_rasterization_rate_map_layer_count = Some(0);
    identity.sizes.maximum_concurrent_compilation_task_count = Some(compile_threads);
    let z = &mut identity.sizes;
    z.max_buffer_length =
        lane_max_buffer_length(z.max_buffer_length, z.recommended_max_working_set_size);
    identity
}

/// `maxBufferLength`: the platform's, but at most three quarters of `recommended_max_working_set`,
/// rounded down to 16 KiB; the platform's alone where it gives no working set. That is the rule
/// both Macs measured with GPU memory of their own answer by: an AMD Radeon Pro 5300M (macOS
/// 26.7.1) 3208642560 of 4278190080 bytes, three quarters exactly, and an Apple M3 Max (macOS
/// 27.0.1) 41747087360 of 55662788608, three quarters rounded down to its 16 KiB pages. It keeps the
/// answer to a buffer the host's memory can hold: the Vulkan platform answers the driver's
/// `maxBufferSize` (a terabyte on an RTX 5080), which no allocation reaches, and its working set is
/// the GPU's device-local memory. Never above the platform's answer.
fn lane_max_buffer_length(platform: Option<u64>, working_set: Option<u64>) -> Option<u64> {
    const PAGE: u64 = 16 << 10;
    match (platform, working_set) {
        (Some(length), Some(set)) => Some(length.min(set / 4 * 3 / PAGE * PAGE)),
        (length, _) => length,
    }
}

/// `minimumLinearTextureAlignmentForPixelFormat:` of every format a buffer can back, and the SPI
/// `linearTextureAlignmentBytes` and `deviceLinearTextureAlignmentBytes`: 256 bytes, as an AMD
/// Radeon Pro 5300M and an Intel UHD 630 answer (macOS 26.7.1; an Apple M3 Max answers 16), or
/// the platform's where it needs more. A guest lays a texture's rows out in its buffer at a
/// multiple of this alignment. The Vulkan platform on RADV takes a texture over a buffer's
/// dma-buf pages in place only with its rows a multiple of 256 bytes apart (metal2vulkan's
/// `dma_buf.rs`, measured on Mesa 26.0.8 and 26.2.4), and shadows one at any other pitch; how many
/// of a guest's textures this moves in place is unmeasured. Any 256-byte layout is one the
/// platform's 16 takes too. An IOSurface's rows follow two other answers. A surface made with no
/// pitch named (`IOSurfaceCreate`, and CoreVideo's buffers and pools) takes rows at IOSurface's own
/// alignment, `IOSurfaceGetPropertyAlignment(kIOSurfaceBytesPerRow)`: 64 bytes on the 5300M's Mac
/// (macOS 26.7.1), and here the device info's `linear_row_alignment` where the guest's kext raises
/// it to that (256 bytes in macOS 11.7.11 and 14.8.9 guests on a Radeon 780M). CoreAnimation names
/// the rows of its backing stores' surfaces itself, rounded to the device's
/// `iosurfaceTextureAlignmentBytes` and to no less than 64 bytes, and IOSurface keeps them. The
/// guest's device answers Metal's 16 for `iosurfaceTextureAlignmentBytes` and
/// `iosurfaceReadOnlyTextureAlignmentBytes` (the 5300M, the 630 and the M3 Max answer the first
/// 16), not the platform's answers this info carries: CoreAnimation makes no texture of a surface whose rows
/// are not at a multiple of the answer, and other code's surfaces keep IOSurface's own 64-byte
/// rows (the bundle's `kRgfxIOSurfaceTextureAlignmentBytes`; macOS 26.7.2 guests answering an
/// RTX 3090's and an RTX 2080's 128 drew Safari's pages blank). So on RADV a texture over a
/// backing store at a pitch that is not a multiple of 256 is the runtime's shadow (its refused
/// pitch): over a cold boot and a 15 s Safari scroll of a macOS 11.7.11 guest on a Radeon 780M,
/// 1399 textures were shadows for their pitch answering 16 (A8 rows of 64 and 128 bytes, a BGRA8
/// surface 463 pixels wide at 1856), and 11 with the guest's device answering 512 there instead.
/// metal2vulkan measured a macOS 26.7.2 guest's surfaces on RADV at 4032 bytes for 1001 `BGRA`
/// pixels, 7360 for 1831 and 14656 for 3662, and in two cold boots of that guest on a Radeon 780M
/// 583 and 639 textures over the guest's pages were shadows for their pitch, against 532 and 503
/// in place.
pub const LANE_LINEAR_TEXTURE_ALIGNMENT: u64 = 256;

/// The device's `NSUInteger` SPI answers where the lane's differ from the platform's: the linear
/// texture alignments, at least [`LANE_LINEAR_TEXTURE_ALIGNMENT`].
fn lane_limits(mut limits: Limits) -> Limits {
    for align in [
        &mut limits.linear_texture_alignment_bytes,
        &mut limits.device_linear_texture_alignment_bytes,
    ] {
        *align = (*align).max(LANE_LINEAR_TEXTURE_ALIGNMENT);
    }
    limits
}

/// The device's `BOOL` SPI answers where the lane's differ from the platform's, for a guest of
/// `guest`'s architecture:
/// - `isFramebufferReadSupported`, for an x86-64 guest no, as an AMD Radeon Pro 5300M answers,
///   where an Intel UHD 630 answers yes (macOS 26.7.1, measured again on 2026-11-09): of the two
///   answers the x86 Macs give, the one that steers the guest's frameworks off reading their
///   attachments in a fragment function, which costs the Vulkan platform most (an input
///   attachment, or the pixel interlock), onto the paths they take on an AMD Mac. For an arm64
///   guest the platform's, as an Apple silicon Mac's GPU answers yes (an Apple M3 Max, macOS
///   27.0.1): CoreAnimation in an arm64 guest compiles framebuffer fetch into its generic shader
///   by this answer, and that shader blends against the fetched destination whatever the answer,
///   so with no it blended against NaN and drew the menu bar black in 14.6.1, 15.6.1, 26.6.2 and
///   27.0.1 guests (Metal on the M3 Max writes the same NaN for that pipeline and its buffers).
///   The platform answers yes where
///   its compile layer fetches a colour attachment at all (MoltenVK, RADV, ANV and NVIDIA); a
///   pipeline that fetches an attachment of a format the runtime does not fetch on the host's GPU
///   (the one- and two-channel 8-bit sRGB formats on ANV and NVIDIA, and the packed 16-bit ones
///   and `RGB9E5Float` on NVIDIA) is a gap its creator answers nil for, relayed
///   (`reims_vgpu_vulkan::answer::relay_gap`);
/// - `supportsTileShaders`, `supportsNonSquareTileShaders` and `supportsImageBlocks` no, as the
///   5300M and the UHD 630 answer: the lane carries neither tile pipelines, nor their non-square
///   dispatch, nor imageblocks ([`Piece::TilePipelines`], [`Piece::NonsquareTileDispatch`],
///   [`Piece::Imageblocks`]).
fn lane_flags(mut flags: Flags, guest: GuestArch) -> Flags {
    if guest == GuestArch::X86_64 {
        flags.is_framebuffer_read_supported = false;
    }
    flags.supports_tile_shaders = false;
    flags.supports_non_square_tile_shaders = false;
    flags.supports_image_blocks = false;
    flags
}

/// The device-info caps blob for `caps` and `identity`, of a device that runs `compile_threads`
/// pipeline compiles at once for a guest of `guest`'s architecture: every value the guest bundle
/// reads once at init, the host device's answers but where the lane's own differ ([`lane`],
/// [`lane_flags`]).
pub fn device_info(
    caps: &DeviceCaps,
    identity: IdentityCaps,
    compile_threads: u64,
    guest: GuestArch,
) -> RgfxDeviceInfo {
    let identity = lane(identity, compile_threads);
    let mut info = RgfxDeviceInfo {
        version: RGFX_DEVICE_INFO_VERSION,
        registry_id: caps.registry_id,
        gpu_core_count: caps.gpu_core_count.unwrap_or(0),
        max_vertex_amplification_count: caps.max_vertex_amplification_count,
        heap_buffer_page_private: match &caps.heap_buffer_page_private {
            Ok(sa) => [sa.size, sa.align],
            Err(_) => [0, 0],
        },
        min_linear_alignment_bgra8: *caps.min_linear_alignment_bgra8.as_ref().unwrap_or(&0),
        ..RgfxDeviceInfo::default()
    };
    info.name_len = put_text(&mut info.name, caps.name.as_deref().unwrap_or(""));
    info.architecture_name_len = put_text(
        &mut info.architecture_name,
        identity.architecture_name.as_deref().unwrap_or(""),
    );
    let t = caps.max_threads_per_threadgroup;
    info.max_threads_per_threadgroup = [t.width, t.height, t.depth];

    let l = &lane_limits(caps.limits);
    // In the host `Limits` field order; the bundle reads them in this order.
    info.limits = [
        l.device_linear_texture_alignment_bytes,
        l.iosurface_read_only_texture_alignment_bytes,
        l.iosurface_texture_alignment_bytes,
        l.linear_texture_alignment_bytes,
        l.max_compute_buffers,
        l.max_compute_local_memory_sizes,
        l.max_compute_samplers,
        l.max_compute_textures,
        l.max_compute_threadgroup_memory,
        l.max_compute_threadgroup_memory_alignment_bytes,
        l.max_fragment_buffers,
        l.max_fragment_samplers,
        l.max_fragment_textures,
        l.max_predicated_nesting_depth,
        l.max_rasterization_rate_layer_count,
        l.max_texture_height_2d,
        l.max_texture_layers,
        l.max_texture_width_2d,
        l.max_threadgroup_memory_length,
        l.max_tile_buffers,
        l.max_tile_samplers,
        l.max_tile_textures,
        l.max_total_compute_threads_per_threadgroup,
        l.max_vertex_buffers,
        l.max_vertex_samplers,
        l.max_vertex_textures,
        l.max_viewport_count,
    ];
    let z = &identity.sizes;
    // In the host `IdentitySizes` field order; the bundle reads them in this order.
    let sizes: [Option<u64>; RGFX_DEVICE_INFO_SIZES] = [
        z.recommended_max_working_set_size,
        z.location,
        z.location_number,
        z.max_transfer_rate,
        z.peer_group_id,
        z.peer_index,
        z.peer_count,
        z.max_buffer_length,
        z.max_argument_buffer_sampler_count,
        z.argument_buffers_support,
        z.read_write_texture_support,
        z.sparse_tile_size_in_bytes,
        z.maximum_concurrent_compilation_task_count,
        z.largest_vertex_amplification_count,
        z.largest_rasterization_rate_map_layer_count,
        z.feature_profile,
        z.half_fp_config,
        z.single_fp_config,
        z.double_fp_config,
    ];
    info.sizes = sizes.map(|size| size.unwrap_or(0));

    // Every family of the SDK, then -1009, which core's caps answer last.
    let families = SDK_FAMILIES
        .iter()
        .zip(identity.families)
        .map(|(&family, answer)| (family, known(answer)))
        .chain(std::iter::once((FAMILIES[6], bit(&caps.families[6]))));
    let mut family_count = 0;
    for (k, (family, answer)) in families.enumerate() {
        info.family_values[k] = family;
        info.family_answers[k] = answer;
        family_count = k + 1;
    }
    // At most RGFX_DEVICE_INFO_FAMILIES: lossless.
    info.family_count = family_count as u32;
    for (k, (&set, answer)) in FEATURE_SETS.iter().zip(identity.feature_sets).enumerate() {
        info.feature_set_values[k] = set;
        info.feature_set_answers[k] = known(answer);
    }
    info.feature_set_count = FEATURE_SETS.len() as u32;

    info.sample_counts = caps.sample_counts.each_ref().map(bit);
    info.primitive_types = caps.primitive_types.each_ref().map(bit);
    info.write_rounding_modes = caps.write_rounding_modes.each_ref().map(bit);
    let f = &lane_flags(caps.flags, guest);
    // In the host `Flags` field order; the bundle reads them in this order.
    info.flags = [
        f.are_raster_order_groups_supported,
        f.is_depth24_stencil8_pixel_format_supported,
        f.is_framebuffer_read_supported,
        f.is_large_mrt_supported,
        f.is_rgb10a2_gamma_supported,
        f.supports_buffer_with_address_ranges,
        f.supports_buffer_with_iosurface,
        f.supports_command_buffer_jump,
        f.supports_compute_compressed_texture_write,
        f.supports_conditional_load_store,
        f.supports_dynamic_attribute_stride,
        f.supports_extended_xr10_formats,
        f.supports_float16_bcubic_filtering,
        f.supports_heap_with_address_ranges,
        f.supports_image_blocks,
        f.supports_memory_order_atomics,
        f.supports_native_hardware_fp16,
        f.supports_non_square_tile_shaders,
        f.supports_programmable_sample_positions,
        f.supports_resource_detach_backing,
        f.supports_shared_texture_handles,
        f.supports_simd_reduction,
        f.supports_simd_shuffle_and_fill,
        f.supports_texture_2d_multisample_array,
        f.supports_tile_shaders,
    ]
    .map(u8::from);
    let g = &identity.flags;
    // In the host `IdentityFlags` field order; the bundle reads them in this order.
    let identity_flags: [Option<bool>; RGFX_DEVICE_INFO_IDENTITY_FLAGS] = [
        g.is_low_power,
        g.is_headless,
        g.is_removable,
        g.has_unified_memory,
        g.supports_32bit_float_filtering,
        g.supports_32bit_msaa,
        g.supports_query_texture_lod,
        g.supports_bc_texture_compression,
        g.supports_pull_model_interpolation,
        g.are_barycentric_coords_supported,
        g.supports_shader_barycentric_coordinates,
        g.supports_dynamic_libraries,
        g.supports_render_dynamic_libraries,
        g.supports_raytracing,
        g.supports_function_pointers,
        g.supports_function_pointers_from_render,
        g.supports_raytracing_from_render,
        g.supports_primitive_motion_blur,
        g.supports_placement_sparse,
    ];
    info.identity_flags = identity_flags.map(known);
    info.counter_sampling = identity.counter_sampling.map(known);
    let mut format_count = 0;
    for (out, answers) in info.formats.iter_mut().zip(&identity.formats) {
        *out = format_answers(answers);
        format_count += 1;
    }
    info.format_count = format_count;
    info
}

// The blob holds every answer the host asks.
const _: () = assert!(SDK_FAMILIES.len() < RGFX_DEVICE_INFO_FAMILIES);
const _: () = assert!(FEATURE_SETS.len() <= RGFX_DEVICE_INFO_FEATURE_SETS);
const _: () = assert!(COUNTER_SAMPLING_POINTS.len() == RGFX_DEVICE_INFO_COUNTER_POINTS);
const _: () = assert!(RGFX_DEVICE_NAME_LEN <= 256 && RGFX_DEVICE_ARCHITECTURE_NAME_LEN <= 256);
const _: () = assert!(FAMILIES[6] == -1009);
const _: () = assert!(SDK_PIXEL_FORMATS.len() <= RGFX_DEVICE_INFO_FORMATS);
const _: () =
    assert!(SPARSE_SAMPLE_COUNTS.len() * SPARSE_PAGE_SIZES.len() == RGFX_DEVICE_INFO_SPARSE_TILES);

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_vulkan::core::platform::device::GpuDevice;
    use reims_vgpu_vulkan::core::raise::Raise;
    use reims_vgpu_vulkan::core::testkit::FakePlatform;
    use zerocopy::IntoBytes;

    use super::*;

    /// The answer the blob gives of `family`, as the bundle looks it up: `None` where it lists
    /// no such family.
    fn family(info: &RgfxDeviceInfo, family: i64) -> Option<u8> {
        let n = info.family_count as usize;
        info.family_values[..n]
            .iter()
            .position(|&f| f == family)
            .map(|k| info.family_answers[k])
    }

    /// The answer the blob gives of feature set `set`.
    fn feature_set(info: &RgfxDeviceInfo, set: u64) -> Option<u8> {
        let n = info.feature_set_count as usize;
        info.feature_set_values[..n]
            .iter()
            .position(|&f| f == set)
            .map(|k| info.feature_set_answers[k])
    }

    /// The blob mirrors the device's caps: the name, a limit and a flag by
    /// position, and the version.
    #[test]
    fn mirrors_the_device_caps() {
        let p = Arc::new(FakePlatform::new());
        let caps = p.device().caps();
        let info = device_info(&caps, p.device().identity_caps(), 4, GuestArch::X86_64);

        assert_eq!(info.version, RGFX_DEVICE_INFO_VERSION);
        assert_eq!(info.registry_id, caps.registry_id);
        assert_eq!(
            &info.name[..info.name_len as usize],
            caps.name.as_deref().unwrap().as_bytes()
        );
        assert_eq!(info.gpu_core_count, caps.gpu_core_count.unwrap());
        // The last limit by position is max_viewport_count.
        assert_eq!(info.limits[26], caps.limits.max_viewport_count);
        // The first flag by position is are_raster_order_groups_supported.
        assert_eq!(
            info.flags[0],
            u8::from(caps.flags.are_raster_order_groups_supported)
        );
    }

    /// Every family of the SDK and -1009 is answered, each the platform's (the fake answers as
    /// an M3 Max: Apple1 to Apple9, Mac1 and Mac2, the Common families and Metal3), but the
    /// Apple families, none of which the lane claims short of Apple7's pieces, and Metal4; a value
    /// that names no family is listed nowhere
    #[test]
    fn every_family_is_the_platforms_or_the_lanes() {
        let p = Arc::new(FakePlatform::new());
        let mut caps = (*p.device().caps()).clone();
        caps.families[6] = Ok(false);
        let info = device_info(&caps, p.device().identity_caps(), 4, GuestArch::X86_64);
        assert_eq!(info.family_count, 21);
        for yes in [2001, 2002, 3001, 3002, 3003, 5001] {
            assert_eq!(family(&info, yes), Some(1), "family {yes}");
        }
        for no in [
            1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 4001, 4002, 5002,
            -1009,
        ] {
            assert_eq!(family(&info, no), Some(0), "family {no}");
        }
        assert_eq!(family(&info, 7), None);
    }

    /// The device makes mesh pipelines where the platform answers Metal3 (the fake, as an M3
    /// Max), and not where it answers no, or has no answer
    #[test]
    fn mesh_pipelines_follow_metal3() {
        let p = Arc::new(FakePlatform::new());
        let mut identity = p.device().identity_caps();
        assert!(makes_mesh_pipelines(&identity));
        identity.families[at(MTL_GPU_FAMILY_METAL3)] = Some(false);
        assert!(!makes_mesh_pipelines(&identity));
        identity.families[at(MTL_GPU_FAMILY_METAL3)] = None;
        assert!(!makes_mesh_pipelines(&identity));
    }

    /// Every feature set is answered as the platform answers it, the fake's (an M3 Max's) every
    /// iOS one and macOS' but 10002 and no tvOS one, but the iOS sets, whose families the lane
    /// answers no; the macOS sets stand with Mac1 and Mac2
    #[test]
    fn every_feature_set_is_the_platforms_or_its_familys() {
        let p = Arc::new(FakePlatform::new());
        let info = device_info(
            &p.device().caps(),
            p.device().identity_caps(),
            4,
            GuestArch::X86_64,
        );
        assert_eq!(info.feature_set_count as usize, FEATURE_SETS.len());
        for set in [10000, 10001, 10003, 10004, 10005] {
            assert_eq!(feature_set(&info, set), Some(1), "feature set {set}");
        }
        for set in [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 10002, 30000, 30005,
        ] {
            assert_eq!(feature_set(&info, set), Some(0), "feature set {set}");
        }
    }

    /// An Apple family keeps the platform's yes only while the lane carries every piece of its
    /// rows and the rows before it, and every piece up to Apple7's: with every piece the fake's
    /// Apple1 to Apple9 and their iOS sets stand; without placement sparse, an Apple8 piece,
    /// Apple1 to Apple7 and their sets stand and Apple8 and Apple9 do not; without a piece of
    /// Apple7's rows or a family's before it (Apple7's Metal 4, Apple4's imageblocks, Apple1's
    /// argument encoders) no Apple family and no iOS set stands, while Mac2 and the macOS sets
    /// do; the lane's own pieces, Apple1's, Apple2's and binary archives, claim none, nor do
    /// Apple1 to Apple3's, while Apple1 to Apple7's claim those seven
    #[test]
    fn an_apple_family_drops_where_the_lane_lacks_a_piece() {
        let p = Arc::new(FakePlatform::new());
        let platform = p.device().identity_caps();
        let capped = |wired: &[Piece]| -> IdentityCaps {
            let mut identity = platform.clone();
            cap_apple_families(&mut identity, wired);
            identity
        };
        let up_to = |families: usize| -> Vec<Piece> {
            APPLE_PIECES[..families]
                .iter()
                .flat_map(|(_, pieces)| pieces.iter().copied())
                .collect()
        };
        let every = up_to(APPLE_PIECES.len());
        let without = |missing: Piece| -> IdentityCaps {
            let wired: Vec<Piece> = every.iter().copied().filter(|&p| p != missing).collect();
            capped(&wired)
        };
        let answer = |identity: &IdentityCaps, family: i64| identity.families[at(family)];
        let set = |identity: &IdentityCaps, set: u64| {
            identity.feature_sets[FEATURE_SETS.iter().position(|&s| s == set).unwrap()]
        };

        let whole = capped(&every);
        assert_eq!(whole.families, platform.families);
        assert_eq!(whole.feature_sets, platform.feature_sets);

        // Past Apple7, a family falls with its own pieces and the later ones with it.
        let no_placement = without(Piece::PlacementSparse);
        for family in [1001, 1002, 1003, 1004, 1005, 1006, 1007, 2001, 2002, 5001] {
            assert_eq!(answer(&no_placement, family), Some(true), "family {family}");
        }
        for family in [1008, 1009] {
            assert_eq!(
                answer(&no_placement, family),
                Some(false),
                "family {family}"
            );
        }
        // iOS_GPUFamily1_v1 (0) and iOS_GPUFamily5_v1 (16) stand.
        assert_eq!(set(&no_placement, 0), Some(true));
        assert_eq!(set(&no_placement, 16), Some(true));

        // Short of Apple7, no Apple family stands, whichever piece is missing.
        for missing in [Piece::Metal4, Piece::Imageblocks, Piece::ArgumentEncoders] {
            let short = without(missing);
            for family in 1001..=1011 {
                assert_eq!(
                    answer(&short, family),
                    Some(false),
                    "{missing:?} family {family}"
                );
            }
            assert_eq!(answer(&short, 2002), Some(true), "{missing:?}");
            // iOS_GPUFamily1_v1 (0), 2_v1 (1), 3_v4 (14) and 5_v1 (16) fall;
            // macOS_GPUFamily2_v1 (10005) stands.
            for s in [0, 1, 14, 16] {
                assert_eq!(set(&short, s), Some(false), "{missing:?} set {s}");
            }
            assert_eq!(set(&short, 10005), Some(true), "{missing:?}");
        }

        // The lane's own pieces: argument encoders, texture views, Automatic heaps, sample
        // positions and IO queues, Apple1's and Apple2's, and indirect command buffers, compute
        // indirect barriers and binary archives, Apple3's. They fall short of Apple7, so no Apple
        // family stands; Mac2 does.
        for piece in [
            Piece::ArgumentEncoders,
            Piece::TextureViews,
            Piece::AutomaticHeaps,
            Piece::SamplePositions,
            Piece::IoCommandQueues,
            Piece::IndirectCommandBuffers,
            Piece::ComputeIndirectBarriers,
            Piece::BinaryArchives,
        ] {
            assert!(WIRED.contains(&piece), "{piece:?}");
        }
        let lane = capped(WIRED);
        for family in 1001..=1011 {
            assert_eq!(answer(&lane, family), Some(false), "family {family}");
        }
        assert_eq!(answer(&lane, 2002), Some(true));
        assert_eq!(set(&lane, 0), Some(false));
        assert_eq!(set(&lane, 10005), Some(true));

        // Every piece of Apple1 to Apple3's claims none; every piece of Apple1 to Apple7's claims
        // those seven, and not Apple8.
        let apple3 = capped(&up_to(3));
        for family in 1001..=1011 {
            assert_eq!(answer(&apple3, family), Some(false), "family {family}");
        }
        let apple7 = capped(&up_to(7));
        for family in 1001..=1007 {
            assert_eq!(answer(&apple7, family), Some(true), "family {family}");
        }
        assert_eq!(answer(&apple7, 1008), Some(false));
    }

    /// The identity is the platform's, a name longer than v1's 32 bytes whole and the memory as
    /// the platform shares it, but where the lane's table says otherwise: what the wire does
    /// not carry answers no
    #[test]
    fn the_identity_is_the_platforms_but_the_lanes_table() {
        let p = Arc::new(FakePlatform::new());
        let mut caps = (*p.device().caps()).clone();
        caps.name = Some(String::from("NVIDIA GeForce RTX 5080 Laptop GPU"));
        let mut identity = p.device().identity_caps();
        identity.flags.has_unified_memory = Some(false);
        identity.flags.is_low_power = Some(true);
        let info = device_info(&caps, identity.clone(), 4, GuestArch::X86_64);
        assert_eq!(
            &info.name[..info.name_len as usize],
            b"NVIDIA GeForce RTX 5080 Laptop GPU"
        );
        assert_eq!(
            &info.architecture_name[..info.architecture_name_len as usize],
            b"fake_gpu"
        );
        // is_low_power, then is_headless, is_removable, has_unified_memory.
        assert_eq!(info.identity_flags[..4], [1, 0, 0, 0]);
        // supports_bc_texture_compression is the platform's yes; raytracing (13) the lane's no.
        assert_eq!(info.identity_flags[7], 1);
        assert_eq!(info.identity_flags[13], 0);
        assert_eq!(info.identity_flags[18], 0);
        assert_eq!(info.counter_sampling, [0; 5]);
        // recommended_max_working_set_size first, the rate maps' layers at 14 (the lane's 0), the
        // driver's feature profile and FP configs last, the platform's.
        assert_eq!(
            info.sizes[0],
            identity.sizes.recommended_max_working_set_size.unwrap()
        );
        assert_eq!(
            info.sizes[reims_vgpu_graphics::wire::RGFX_DEVICE_INFO_MAX_BUFFER_LENGTH],
            identity.sizes.max_buffer_length.unwrap()
        );
        // maximum_concurrent_compilation_task_count is the device's compile threads.
        assert_eq!(info.sizes[12], 4);
        assert_eq!(info.sizes[13], 8);
        assert_eq!(info.sizes[14], 0);
        assert_eq!(info.sizes[15..], [10, 0xe, 0x54, 0]);
        // Every SDK format, each the platform's: BGRA8Unorm (80) 16-byte texture buffers and
        // 64x64 tiles; BC1_RGBA (130) no texture buffers.
        assert_eq!(info.format_count as usize, SDK_PIXEL_FORMATS.len());
        let at = |f: u32| info.formats.iter().find(|a| a.format == f).unwrap();
        assert_eq!(at(80).texture_buffer_alignment, 16);
        assert_eq!(at(80).sparse_tiles[0], [64, 64, 1]);
        assert_eq!(at(80).sparse_tiles[8], [128, 128, 1]);
        assert_eq!(at(130).texture_buffer_alignment, 0);
    }

    /// `maxBufferLength` is three quarters of the working set rounded down to 16 KiB where that is
    /// below the platform's answer, as the 5300M's and the M3 Max's answers are of theirs, and the
    /// platform's otherwise; the linear texture alignments are 256 bytes, or the platform's where
    /// larger; for an x86-64 guest the framebuffer read, tile shader and imageblock answers are
    /// no, whatever the platform's, and the other SPI flags the platform's.
    #[test]
    fn the_lanes_buffer_length_alignments_and_spi_flags() {
        // An RTX 5080 on the Vulkan platform: a terabyte against 17094934528 bytes of VRAM.
        assert_eq!(
            lane_max_buffer_length(Some(1 << 40), Some(17_094_934_528)),
            Some(12_821_200_896)
        );
        // The 5300M's and the M3 Max's own answers, from their working sets.
        assert_eq!(
            lane_max_buffer_length(Some(u64::MAX), Some(4_278_190_080)),
            Some(3_208_642_560)
        );
        assert_eq!(
            lane_max_buffer_length(Some(u64::MAX), Some(55_662_788_608)),
            Some(41_747_087_360)
        );
        // Never above the platform's; the platform's where it gives no working set.
        assert_eq!(
            lane_max_buffer_length(Some(4096), Some(1 << 34)),
            Some(4096)
        );
        assert_eq!(lane_max_buffer_length(Some(4096), None), Some(4096));
        assert_eq!(lane_max_buffer_length(None, Some(1 << 34)), None);

        let p = Arc::new(FakePlatform::new());
        let mut caps = (*p.device().caps()).clone();
        let mut identity = p.device().identity_caps();
        identity.sizes.max_buffer_length = Some(1 << 40);
        identity.sizes.recommended_max_working_set_size = Some(17_094_934_528);
        // The fake answers 16 for both alignments, and yes for every SPI flag.
        let info = device_info(&caps, identity.clone(), 4, GuestArch::X86_64);
        assert_eq!(info.sizes[7], 12_821_200_896);
        // device_linear_texture_alignment_bytes first, linear_texture_alignment_bytes fourth.
        assert_eq!(info.limits[0], 256);
        assert_eq!(info.limits[3], 256);
        // is_framebuffer_read_supported (2), supports_image_blocks (14),
        // supports_non_square_tile_shaders (17) and supports_tile_shaders (24) no; the others
        // the platform's, all yes on the fake but is_depth24_stencil8_pixel_format_supported (1).
        for (k, &flag) in info.flags.iter().enumerate() {
            let want = !matches!(k, 1 | 2 | 14 | 17 | 24);
            assert_eq!(flag, u8::from(want), "flag {k}");
        }
        // A platform that needs more than 256 bytes keeps its own.
        caps.limits.linear_texture_alignment_bytes = 512;
        caps.limits.device_linear_texture_alignment_bytes = 1024;
        let info = device_info(&caps, identity, 4, GuestArch::X86_64);
        assert_eq!(info.limits[0], 1024);
        assert_eq!(info.limits[3], 512);
    }

    /// `isFramebufferReadSupported` follows the guest: an x86-64 guest's device answers no, an
    /// arm64 guest's the platform's (the fake's yes, a platform's no as no), and the arm64 guest's
    /// blob differs from the x86-64 guest's in that answer's byte alone, so an x86-64 guest is
    /// told what it was told before guests had an architecture; a config that names none serves
    /// an x86-64 guest
    #[test]
    fn framebuffer_read_follows_the_guest() {
        // is_framebuffer_read_supported, third of the flags.
        const FRAMEBUFFER_READ: usize = 2;
        let p = Arc::new(FakePlatform::new());
        let mut caps = (*p.device().caps()).clone();
        let identity = p.device().identity_caps();
        assert!(caps.flags.is_framebuffer_read_supported);
        let x86 = device_info(&caps, identity.clone(), 4, GuestArch::X86_64);
        let arm64 = device_info(&caps, identity.clone(), 4, GuestArch::Arm64);
        assert_eq!(x86.flags[FRAMEBUFFER_READ], 0);
        assert_eq!(arm64.flags[FRAMEBUFFER_READ], 1);
        let at = std::mem::offset_of!(RgfxDeviceInfo, flags) + FRAMEBUFFER_READ;
        let differ: Vec<usize> = (x86.as_bytes().iter().zip(arm64.as_bytes()))
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(k, _)| k)
            .collect();
        assert_eq!(differ, [at]);
        // The tile and imageblock answers stay no for both.
        for k in [14, 17, 24] {
            assert_eq!(arm64.flags[k], 0, "flag {k}");
        }

        caps.flags.is_framebuffer_read_supported = false;
        let arm64 = device_info(&caps, identity, 4, GuestArch::Arm64);
        assert_eq!(arm64.flags[FRAMEBUFFER_READ], 0);

        assert_eq!(GuestArch::default(), GuestArch::X86_64);
    }

    /// A present no reads 0, a present yes reads 1, and a platform gap (an `Err` or `None`)
    /// reads 0 — never a spurious yes; a name longer than the blob's is cut on a character
    #[test]
    fn a_platform_gap_is_reported_as_unsupported() {
        let p = Arc::new(FakePlatform::new());
        let mut caps = (*p.device().caps()).clone();
        let mut identity = p.device().identity_caps();
        // The fake answers sample count 8 (index 3) with a present no.
        assert_eq!(
            device_info(&caps, identity.clone(), 4, GuestArch::X86_64).sample_counts[3],
            0
        );
        // A family the platform could not answer at all (a gap) also reads 0.
        caps.families[6] = Err(Raise::foreign("device_info_test_gap"));
        identity.families[8] = None;
        identity.feature_sets[0] = None;
        identity.flags.is_low_power = None;
        identity.sizes.location = None;
        let info = device_info(&caps, identity.clone(), 4, GuestArch::X86_64);
        assert_eq!(family(&info, -1009), Some(0));
        assert_eq!(family(&info, 1009), Some(0));
        assert_eq!(feature_set(&info, 0), Some(0));
        assert_eq!(info.identity_flags[0], 0);
        assert_eq!(info.sizes[1], 0);
        // A present yes still reads 1.
        caps.families[6] = Ok(true);
        assert_eq!(
            family(
                &device_info(&caps, identity.clone(), 4, GuestArch::X86_64),
                -1009
            ),
            Some(1)
        );
        // 255 ASCII bytes and a two-byte character: the character does not fit, and is left out.
        caps.name = Some(format!("{}\u{e9}", "a".repeat(255)));
        let info = device_info(&caps, identity, 4, GuestArch::X86_64);
        assert_eq!(info.name_len, 255);
    }
}
