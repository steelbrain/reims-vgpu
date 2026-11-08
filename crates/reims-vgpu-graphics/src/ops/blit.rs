//! The blit-encoder ops (`tag >> 12 == `[`RGFX_OP_GROUP_BLIT`](super::RGFX_OP_GROUP_BLIT)):
//! one tag and one body struct per `MTLBlitCommandEncoder` selector the guest
//! serialises (docs/graphics-kext/design.md, "Stage 3 wire protocol"). The host's
//! blit decoder (`reims-vgpu`'s `graphics::decode::blit`) reads the body, resolves
//! its object references and sends the matching `BlitCmd`.
//!
//! A body struct is the op's bytes **after** its [`RgfxOpHeader`](super::RgfxOpHeader),
//! not the whole op (see the [module doc](super)). Object references are `u64`
//! `obj_id` (0 = nil). An `MTLSize`/`MTLOrigin`/`MTLRegion` travels as its `u64`
//! fields, and floating-point arguments (an `MTLClearColor`) as the bits of their
//! `f64`, so the schema stays all-integer and comparable, as the render ops and the
//! sampler's LOD clamps do.
//!
//! The blit encoder is opened by the framing `ENCODER_BEGIN_BLIT` op (no body) and
//! closed by `ENCODER_END` (`endEncoding`), so the `BlitCmd::EndEncoding` variant
//! has no op here; every other `MTLBlitCommandEncoder` selector has one. The tags
//! stay in the `0x3xxx` range.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Op `COPY_BUFFER_TO_BUFFER` ([`RgfxBlitOpCopyBufferToBuffer`]):
/// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`.
pub const RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER: u16 = 0x3000;
/// Op `COPY_BUFFER_TO_TEXTURE` ([`RgfxBlitOpCopyBufferToTexture`]):
/// `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:`.
pub const RGFX_OP_BLIT_COPY_BUFFER_TO_TEXTURE: u16 = 0x3001;
/// Op `COPY_BUFFER_TO_TEXTURE_OPTIONS` ([`RgfxBlitOpCopyBufferToTextureOptions`]): the
/// `:options:` form of the above.
pub const RGFX_OP_BLIT_COPY_BUFFER_TO_TEXTURE_OPTIONS: u16 = 0x3002;
/// Op `COPY_TEXTURE_TO_BUFFER` ([`RgfxBlitOpCopyTextureToBuffer`]):
/// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:`.
pub const RGFX_OP_BLIT_COPY_TEXTURE_TO_BUFFER: u16 = 0x3003;
/// Op `COPY_TEXTURE_TO_BUFFER_OPTIONS` ([`RgfxBlitOpCopyTextureToBufferOptions`]): the
/// `:options:` form of the above.
pub const RGFX_OP_BLIT_COPY_TEXTURE_TO_BUFFER_OPTIONS: u16 = 0x3004;
/// Op `COPY_TEXTURE_REGION` ([`RgfxBlitOpCopyTextureRegion`]):
/// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:`.
pub const RGFX_OP_BLIT_COPY_TEXTURE_REGION: u16 = 0x3005;
/// Op `COPY_TEXTURE_REGION_OPTIONS` ([`RgfxBlitOpCopyTextureRegionOptions`]): the
/// `:options:` form of the above (SPI).
pub const RGFX_OP_BLIT_COPY_TEXTURE_REGION_OPTIONS: u16 = 0x3006;
/// Op `COPY_TEXTURE_SLICES` ([`RgfxBlitOpCopyTextureSlices`]):
/// `copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:`.
pub const RGFX_OP_BLIT_COPY_TEXTURE_SLICES: u16 = 0x3007;
/// Op `COPY_INDIRECT_COMMAND_BUFFER` ([`RgfxBlitOpCopyIndirectCommandBuffer`]):
/// `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:`.
pub const RGFX_OP_BLIT_COPY_INDIRECT_COMMAND_BUFFER: u16 = 0x3008;
/// Op `FILL_BUFFER` ([`RgfxBlitOpFillBuffer`]): `fillBuffer:range:value:`.
pub const RGFX_OP_BLIT_FILL_BUFFER: u16 = 0x3009;
/// Op `FILL_BUFFER_PATTERN4` ([`RgfxBlitOpFillBufferPattern4`]): `fillBuffer:range:pattern4:` (SPI).
pub const RGFX_OP_BLIT_FILL_BUFFER_PATTERN4: u16 = 0x300a;
/// Op `FILL_TEXTURE` ([`RgfxBlitOpFillTexture`]): `fillTexture:level:slice:region:bytes:length:` (SPI).
pub const RGFX_OP_BLIT_FILL_TEXTURE: u16 = 0x300b;
/// Op `FILL_TEXTURE_COLOR` ([`RgfxBlitOpFillTextureColor`]):
/// `fillTexture:level:slice:region:color:pixelFormat:` (SPI).
pub const RGFX_OP_BLIT_FILL_TEXTURE_COLOR: u16 = 0x300c;
/// Op `GENERATE_MIPMAPS` ([`RgfxBlitOpGenerateMipmaps`]): `generateMipmapsForTexture:`.
pub const RGFX_OP_BLIT_GENERATE_MIPMAPS: u16 = 0x300d;
/// Op `INVALIDATE_COMPRESSED_TEXTURE` ([`RgfxBlitOpInvalidateCompressedTexture`]):
/// `invalidateCompressedTexture:` (SPI).
pub const RGFX_OP_BLIT_INVALIDATE_COMPRESSED_TEXTURE: u16 = 0x300e;
/// Op `INVALIDATE_COMPRESSED_TEXTURE_SLICE` ([`RgfxBlitOpInvalidateCompressedTextureSlice`]):
/// `invalidateCompressedTexture:slice:level:` (SPI).
pub const RGFX_OP_BLIT_INVALIDATE_COMPRESSED_TEXTURE_SLICE: u16 = 0x300f;
/// Op `OPTIMIZE_FOR_CPU` ([`RgfxBlitOpOptimizeForCpu`]): `optimizeContentsForCPUAccess:`.
pub const RGFX_OP_BLIT_OPTIMIZE_FOR_CPU: u16 = 0x3010;
/// Op `OPTIMIZE_FOR_CPU_SLICE` ([`RgfxBlitOpOptimizeForCpuSlice`]):
/// `optimizeContentsForCPUAccess:slice:level:`.
pub const RGFX_OP_BLIT_OPTIMIZE_FOR_CPU_SLICE: u16 = 0x3011;
/// Op `OPTIMIZE_FOR_GPU` ([`RgfxBlitOpOptimizeForGpu`]): `optimizeContentsForGPUAccess:`.
pub const RGFX_OP_BLIT_OPTIMIZE_FOR_GPU: u16 = 0x3012;
/// Op `OPTIMIZE_FOR_GPU_SLICE` ([`RgfxBlitOpOptimizeForGpuSlice`]):
/// `optimizeContentsForGPUAccess:slice:level:`.
pub const RGFX_OP_BLIT_OPTIMIZE_FOR_GPU_SLICE: u16 = 0x3013;
/// Op `OPTIMIZE_INDIRECT_COMMAND_BUFFER` ([`RgfxBlitOpOptimizeIndirectCommandBuffer`]):
/// `optimizeIndirectCommandBuffer:withRange:`.
pub const RGFX_OP_BLIT_OPTIMIZE_INDIRECT_COMMAND_BUFFER: u16 = 0x3014;
/// Op `RESET_COMMANDS_IN_BUFFER` ([`RgfxBlitOpResetCommandsInBuffer`]):
/// `resetCommandsInBuffer:withRange:`.
pub const RGFX_OP_BLIT_RESET_COMMANDS_IN_BUFFER: u16 = 0x3015;
/// Op `SYNCHRONIZE_RESOURCE` ([`RgfxBlitOpSynchronizeResource`]): `synchronizeResource:`.
pub const RGFX_OP_BLIT_SYNCHRONIZE_RESOURCE: u16 = 0x3016;
/// Op `SYNCHRONIZE_TEXTURE` ([`RgfxBlitOpSynchronizeTexture`]): `synchronizeTexture:slice:level:`.
pub const RGFX_OP_BLIT_SYNCHRONIZE_TEXTURE: u16 = 0x3017;
/// Op `UPDATE_FENCE` ([`RgfxBlitOpUpdateFence`]): `updateFence:`.
pub const RGFX_OP_BLIT_UPDATE_FENCE: u16 = 0x3018;
/// Op `WAIT_FOR_FENCE` ([`RgfxBlitOpWaitForFence`]): `waitForFence:`.
pub const RGFX_OP_BLIT_WAIT_FOR_FENCE: u16 = 0x3019;

/// The kind of resource a [`RgfxBlitOpSynchronizeResource`] names, which picks the
/// resolver the host turns its `obj_id` into (`synchronizeResource:` takes any
/// `id<MTLResource>`: a buffer, a texture or an indirect command buffer).
/// A nil resource (`resource_kind` [`RGFX_BLIT_RESOURCE_NIL`]).
pub const RGFX_BLIT_RESOURCE_NIL: u32 = 0;
/// An `id<MTLBuffer>` resource.
pub const RGFX_BLIT_RESOURCE_BUFFER: u32 = 1;
/// An `id<MTLTexture>` resource.
pub const RGFX_BLIT_RESOURCE_TEXTURE: u32 = 2;
/// An `id<MTLIndirectCommandBuffer>` resource.
pub const RGFX_BLIT_RESOURCE_ICB: u32 = 3;

/// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`. Either
/// buffer may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpCopyBufferToBuffer {
    /// The source buffer, or 0 for nil.
    pub source_buffer_obj_id: u64,
    /// The destination buffer, or 0 for nil.
    pub destination_buffer_obj_id: u64,
    /// The offset into the source buffer.
    pub source_offset: u64,
    /// The offset into the destination buffer.
    pub destination_offset: u64,
    /// The bytes to copy.
    pub size: u64,
}

/// `copyFromBuffer:...:toTexture:...`. Both the buffer and the texture must name
/// an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpCopyBufferToTexture {
    /// The source buffer (must not be nil).
    pub source_buffer_obj_id: u64,
    /// The destination texture (must not be nil).
    pub destination_texture_obj_id: u64,
    /// The offset into the source buffer.
    pub source_offset: u64,
    /// `sourceBytesPerRow`.
    pub source_bytes_per_row: u64,
    /// `sourceBytesPerImage`.
    pub source_bytes_per_image: u64,
    /// `sourceSize.width`.
    pub source_size_width: u64,
    /// `sourceSize.height`.
    pub source_size_height: u64,
    /// `sourceSize.depth`.
    pub source_size_depth: u64,
    /// `destinationSlice`.
    pub destination_slice: u64,
    /// `destinationLevel`.
    pub destination_level: u64,
    /// `destinationOrigin.x`.
    pub destination_origin_x: u64,
    /// `destinationOrigin.y`.
    pub destination_origin_y: u64,
    /// `destinationOrigin.z`.
    pub destination_origin_z: u64,
}

/// The `:options:` form of [`RgfxBlitOpCopyBufferToTexture`]; either object may be
/// nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpCopyBufferToTextureOptions {
    /// The source buffer, or 0 for nil.
    pub source_buffer_obj_id: u64,
    /// The destination texture, or 0 for nil.
    pub destination_texture_obj_id: u64,
    /// The offset into the source buffer.
    pub source_offset: u64,
    /// `sourceBytesPerRow`.
    pub source_bytes_per_row: u64,
    /// `sourceBytesPerImage`.
    pub source_bytes_per_image: u64,
    /// `sourceSize.width`.
    pub source_size_width: u64,
    /// `sourceSize.height`.
    pub source_size_height: u64,
    /// `sourceSize.depth`.
    pub source_size_depth: u64,
    /// `destinationSlice`.
    pub destination_slice: u64,
    /// `destinationLevel`.
    pub destination_level: u64,
    /// `destinationOrigin.x`.
    pub destination_origin_x: u64,
    /// `destinationOrigin.y`.
    pub destination_origin_y: u64,
    /// `destinationOrigin.z`.
    pub destination_origin_z: u64,
    /// `MTLBlitOption` set.
    pub options: u64,
}

/// `copyFromTexture:...:toBuffer:...`. Both the texture and the buffer must name
/// an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpCopyTextureToBuffer {
    /// The source texture (must not be nil).
    pub source_texture_obj_id: u64,
    /// The destination buffer (must not be nil).
    pub destination_buffer_obj_id: u64,
    /// `sourceSlice`.
    pub source_slice: u64,
    /// `sourceLevel`.
    pub source_level: u64,
    /// `sourceOrigin.x`.
    pub source_origin_x: u64,
    /// `sourceOrigin.y`.
    pub source_origin_y: u64,
    /// `sourceOrigin.z`.
    pub source_origin_z: u64,
    /// `sourceSize.width`.
    pub source_size_width: u64,
    /// `sourceSize.height`.
    pub source_size_height: u64,
    /// `sourceSize.depth`.
    pub source_size_depth: u64,
    /// `destinationOffset`.
    pub destination_offset: u64,
    /// `destinationBytesPerRow`.
    pub destination_bytes_per_row: u64,
    /// `destinationBytesPerImage`.
    pub destination_bytes_per_image: u64,
}

/// The `:options:` form of [`RgfxBlitOpCopyTextureToBuffer`]; either object may be
/// nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpCopyTextureToBufferOptions {
    /// The source texture, or 0 for nil.
    pub source_texture_obj_id: u64,
    /// The destination buffer, or 0 for nil.
    pub destination_buffer_obj_id: u64,
    /// `sourceSlice`.
    pub source_slice: u64,
    /// `sourceLevel`.
    pub source_level: u64,
    /// `sourceOrigin.x`.
    pub source_origin_x: u64,
    /// `sourceOrigin.y`.
    pub source_origin_y: u64,
    /// `sourceOrigin.z`.
    pub source_origin_z: u64,
    /// `sourceSize.width`.
    pub source_size_width: u64,
    /// `sourceSize.height`.
    pub source_size_height: u64,
    /// `sourceSize.depth`.
    pub source_size_depth: u64,
    /// `destinationOffset`.
    pub destination_offset: u64,
    /// `destinationBytesPerRow`.
    pub destination_bytes_per_row: u64,
    /// `destinationBytesPerImage`.
    pub destination_bytes_per_image: u64,
    /// `MTLBlitOption` set.
    pub options: u64,
}

/// `copyFromTexture:...:toTexture:...` (the region form). Both textures must name
/// an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpCopyTextureRegion {
    /// The source texture (must not be nil).
    pub source_texture_obj_id: u64,
    /// The destination texture (must not be nil).
    pub destination_texture_obj_id: u64,
    /// `sourceSlice`.
    pub source_slice: u64,
    /// `sourceLevel`.
    pub source_level: u64,
    /// `sourceOrigin.x`.
    pub source_origin_x: u64,
    /// `sourceOrigin.y`.
    pub source_origin_y: u64,
    /// `sourceOrigin.z`.
    pub source_origin_z: u64,
    /// `sourceSize.width`.
    pub source_size_width: u64,
    /// `sourceSize.height`.
    pub source_size_height: u64,
    /// `sourceSize.depth`.
    pub source_size_depth: u64,
    /// `destinationSlice`.
    pub destination_slice: u64,
    /// `destinationLevel`.
    pub destination_level: u64,
    /// `destinationOrigin.x`.
    pub destination_origin_x: u64,
    /// `destinationOrigin.y`.
    pub destination_origin_y: u64,
    /// `destinationOrigin.z`.
    pub destination_origin_z: u64,
}

/// The `:options:` form of [`RgfxBlitOpCopyTextureRegion`] (SPI). Both textures must
/// name an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpCopyTextureRegionOptions {
    /// The source texture (must not be nil).
    pub source_texture_obj_id: u64,
    /// The destination texture (must not be nil).
    pub destination_texture_obj_id: u64,
    /// `sourceSlice`.
    pub source_slice: u64,
    /// `sourceLevel`.
    pub source_level: u64,
    /// `sourceOrigin.x`.
    pub source_origin_x: u64,
    /// `sourceOrigin.y`.
    pub source_origin_y: u64,
    /// `sourceOrigin.z`.
    pub source_origin_z: u64,
    /// `sourceSize.width`.
    pub source_size_width: u64,
    /// `sourceSize.height`.
    pub source_size_height: u64,
    /// `sourceSize.depth`.
    pub source_size_depth: u64,
    /// `destinationSlice`.
    pub destination_slice: u64,
    /// `destinationLevel`.
    pub destination_level: u64,
    /// `destinationOrigin.x`.
    pub destination_origin_x: u64,
    /// `destinationOrigin.y`.
    pub destination_origin_y: u64,
    /// `destinationOrigin.z`.
    pub destination_origin_z: u64,
    /// `MTLBlitOption` set.
    pub options: u64,
}

/// `copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:`.
/// Both textures must name an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpCopyTextureSlices {
    /// The source texture (must not be nil).
    pub source_texture_obj_id: u64,
    /// The destination texture (must not be nil).
    pub destination_texture_obj_id: u64,
    /// `sourceSlice`.
    pub source_slice: u64,
    /// `sourceLevel`.
    pub source_level: u64,
    /// `destinationSlice`.
    pub destination_slice: u64,
    /// `destinationLevel`.
    pub destination_level: u64,
    /// `sliceCount`.
    pub slice_count: u64,
    /// `levelCount`.
    pub level_count: u64,
}

/// `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:`. Both
/// indirect command buffers must name an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpCopyIndirectCommandBuffer {
    /// The source indirect command buffer (must not be nil).
    pub source_obj_id: u64,
    /// The destination indirect command buffer (must not be nil).
    pub destination_obj_id: u64,
    /// `sourceRange.location`.
    pub source_range_location: u64,
    /// `sourceRange.length`.
    pub source_range_length: u64,
    /// `destinationIndex`.
    pub destination_index: u64,
}

/// `fillBuffer:range:value:`. The buffer must name an object (nil faults the
/// stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpFillBuffer {
    /// The buffer (must not be nil).
    pub buffer_obj_id: u64,
    /// `range.location`.
    pub range_location: u64,
    /// `range.length`.
    pub range_length: u64,
    /// The byte to fill with.
    pub value: u8,
    /// Pads to a multiple of 8; 0.
    pub reserved_19: [u8; 7],
}

/// `fillBuffer:range:pattern4:` (SPI). The buffer must name an object (nil faults
/// the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpFillBufferPattern4 {
    /// The buffer (must not be nil).
    pub buffer_obj_id: u64,
    /// `range.location`.
    pub range_location: u64,
    /// `range.length`.
    pub range_length: u64,
    /// The 4-byte pattern.
    pub pattern: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_1c: u32,
}

/// The fixed head of a `fillTexture:level:slice:region:bytes:length:` op (SPI).
/// The `length` bytes of fill data follow it at [`RGFX_OP_BLIT_FILL_TEXTURE_BYTES_OFFSET`].
/// The texture must name an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpFillTexture {
    /// The texture (must not be nil).
    pub texture_obj_id: u64,
    /// `level`.
    pub level: u64,
    /// `slice`.
    pub slice: u64,
    /// `region.origin.x`.
    pub region_origin_x: u64,
    /// `region.origin.y`.
    pub region_origin_y: u64,
    /// `region.origin.z`.
    pub region_origin_z: u64,
    /// `region.size.width`.
    pub region_size_width: u64,
    /// `region.size.height`.
    pub region_size_height: u64,
    /// `region.size.depth`.
    pub region_size_depth: u64,
    /// `length`: the bytes of fill data that follow this head.
    pub length: u64,
}

/// Where a `FILL_TEXTURE` op's fill bytes start in its body: right after the fixed
/// [`RgfxBlitOpFillTexture`] head.
pub const RGFX_OP_BLIT_FILL_TEXTURE_BYTES_OFFSET: u32 = 80;

/// `fillTexture:level:slice:region:color:pixelFormat:` (SPI). The texture must
/// name an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpFillTextureColor {
    /// The texture (must not be nil).
    pub texture_obj_id: u64,
    /// `level`.
    pub level: u64,
    /// `slice`.
    pub slice: u64,
    /// `region.origin.x`.
    pub region_origin_x: u64,
    /// `region.origin.y`.
    pub region_origin_y: u64,
    /// `region.origin.z`.
    pub region_origin_z: u64,
    /// `region.size.width`.
    pub region_size_width: u64,
    /// `region.size.height`.
    pub region_size_height: u64,
    /// `region.size.depth`.
    pub region_size_depth: u64,
    /// `MTLClearColor` as the bits of its four `f64` (red, green, blue, alpha).
    pub color_bits: [u64; 4],
    /// `MTLPixelFormat`.
    pub pixel_format: u64,
}

/// `generateMipmapsForTexture:`. The texture may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpGenerateMipmaps {
    /// The texture, or 0 for nil.
    pub texture_obj_id: u64,
}

/// `invalidateCompressedTexture:` (SPI). The texture must name an object (nil
/// faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpInvalidateCompressedTexture {
    /// The texture (must not be nil).
    pub texture_obj_id: u64,
}

/// `invalidateCompressedTexture:slice:level:` (SPI). The texture must name an
/// object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpInvalidateCompressedTextureSlice {
    /// The texture (must not be nil).
    pub texture_obj_id: u64,
    /// `slice`.
    pub slice: u64,
    /// `level`.
    pub level: u64,
}

/// `optimizeContentsForCPUAccess:`. The texture must name an object (nil faults
/// the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpOptimizeForCpu {
    /// The texture (must not be nil).
    pub texture_obj_id: u64,
}

/// `optimizeContentsForCPUAccess:slice:level:`. The texture must name an object
/// (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpOptimizeForCpuSlice {
    /// The texture (must not be nil).
    pub texture_obj_id: u64,
    /// `slice`.
    pub slice: u64,
    /// `level`.
    pub level: u64,
}

/// `optimizeContentsForGPUAccess:`. The texture must name an object (nil faults
/// the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpOptimizeForGpu {
    /// The texture (must not be nil).
    pub texture_obj_id: u64,
}

/// `optimizeContentsForGPUAccess:slice:level:`. The texture must name an object
/// (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpOptimizeForGpuSlice {
    /// The texture (must not be nil).
    pub texture_obj_id: u64,
    /// `slice`.
    pub slice: u64,
    /// `level`.
    pub level: u64,
}

/// `optimizeIndirectCommandBuffer:withRange:`. The indirect command buffer must
/// name an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpOptimizeIndirectCommandBuffer {
    /// The indirect command buffer (must not be nil).
    pub icb_obj_id: u64,
    /// `range.location`.
    pub range_location: u64,
    /// `range.length`.
    pub range_length: u64,
}

/// `resetCommandsInBuffer:withRange:`. The indirect command buffer must name an
/// object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpResetCommandsInBuffer {
    /// The indirect command buffer (must not be nil).
    pub icb_obj_id: u64,
    /// `range.location`.
    pub range_location: u64,
    /// `range.length`.
    pub range_length: u64,
}

/// `synchronizeResource:`. The resource is any `id<MTLResource>`; `resource_kind`
/// (an `RGFX_BLIT_RESOURCE_*`) picks the resolver, and [`RGFX_BLIT_RESOURCE_NIL`]
/// (or a 0 `resource_obj_id`) is nil.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpSynchronizeResource {
    /// The resource, or 0 for nil.
    pub resource_obj_id: u64,
    /// The resource's kind: an `RGFX_BLIT_RESOURCE_*`.
    pub resource_kind: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_0c: u32,
}

/// `synchronizeTexture:slice:level:`. The texture must name an object (nil faults
/// the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpSynchronizeTexture {
    /// The texture (must not be nil).
    pub texture_obj_id: u64,
    /// `slice`.
    pub slice: u64,
    /// `level`.
    pub level: u64,
}

/// `updateFence:`. The fence must name an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpUpdateFence {
    /// The fence (must not be nil).
    pub fence_obj_id: u64,
}

/// `waitForFence:`. The fence must name an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlitOpWaitForFence {
    /// The fence (must not be nil).
    pub fence_obj_id: u64,
}

// The blit ops' layout.
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<RgfxBlitOpCopyBufferToBuffer>() == 40);
    assert!(size_of::<RgfxBlitOpCopyBufferToTexture>() == 104);
    assert!(size_of::<RgfxBlitOpCopyBufferToTextureOptions>() == 112);
    assert!(size_of::<RgfxBlitOpCopyTextureToBuffer>() == 104);
    assert!(size_of::<RgfxBlitOpCopyTextureToBufferOptions>() == 112);
    assert!(size_of::<RgfxBlitOpCopyTextureRegion>() == 120);
    assert!(size_of::<RgfxBlitOpCopyTextureRegionOptions>() == 128);
    assert!(size_of::<RgfxBlitOpCopyTextureSlices>() == 64);
    assert!(size_of::<RgfxBlitOpCopyIndirectCommandBuffer>() == 40);
    assert!(size_of::<RgfxBlitOpFillBuffer>() == 32);
    assert!(size_of::<RgfxBlitOpFillBufferPattern4>() == 32);
    assert!(size_of::<RgfxBlitOpFillTexture>() == 80);
    assert!(RGFX_OP_BLIT_FILL_TEXTURE_BYTES_OFFSET as usize == size_of::<RgfxBlitOpFillTexture>());
    assert!(size_of::<RgfxBlitOpFillTextureColor>() == 112);
    assert!(size_of::<RgfxBlitOpGenerateMipmaps>() == 8);
    assert!(size_of::<RgfxBlitOpInvalidateCompressedTexture>() == 8);
    assert!(size_of::<RgfxBlitOpInvalidateCompressedTextureSlice>() == 24);
    assert!(size_of::<RgfxBlitOpOptimizeForCpu>() == 8);
    assert!(size_of::<RgfxBlitOpOptimizeForCpuSlice>() == 24);
    assert!(size_of::<RgfxBlitOpOptimizeForGpu>() == 8);
    assert!(size_of::<RgfxBlitOpOptimizeForGpuSlice>() == 24);
    assert!(size_of::<RgfxBlitOpOptimizeIndirectCommandBuffer>() == 24);
    assert!(size_of::<RgfxBlitOpResetCommandsInBuffer>() == 24);
    assert!(size_of::<RgfxBlitOpSynchronizeResource>() == 16);
    assert!(size_of::<RgfxBlitOpSynchronizeTexture>() == 24);
    assert!(size_of::<RgfxBlitOpUpdateFence>() == 8);
    assert!(size_of::<RgfxBlitOpWaitForFence>() == 8);
    // The blit tags stay in the blit group.
    assert!(super::op_group(RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER) == super::RGFX_OP_GROUP_BLIT);
    assert!(super::op_group(RGFX_OP_BLIT_WAIT_FOR_FENCE) == super::RGFX_OP_GROUP_BLIT);
};

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::{FromBytes, IntoBytes};

    /// A representative set of blit op bodies round-trips, the `MTLSize`/`MTLOrigin`
    /// fields and the clear colour bits surviving the round trip.
    #[test]
    fn blit_ops_round_trip() {
        let bb = RgfxBlitOpCopyBufferToBuffer {
            source_buffer_obj_id: 2,
            destination_buffer_obj_id: 3,
            source_offset: 16,
            destination_offset: 32,
            size: 4096,
        };
        assert_eq!(
            RgfxBlitOpCopyBufferToBuffer::read_from_bytes(bb.as_bytes()).unwrap(),
            bb
        );

        let bt = RgfxBlitOpCopyBufferToTexture {
            source_buffer_obj_id: 2,
            destination_texture_obj_id: 4,
            source_offset: 0,
            source_bytes_per_row: 256,
            source_bytes_per_image: 256 * 64,
            source_size_width: 64,
            source_size_height: 64,
            source_size_depth: 1,
            destination_slice: 0,
            destination_level: 1,
            destination_origin_x: 8,
            destination_origin_y: 8,
            destination_origin_z: 0,
        };
        assert_eq!(
            RgfxBlitOpCopyBufferToTexture::read_from_bytes(bt.as_bytes()).unwrap(),
            bt
        );

        let region = RgfxBlitOpCopyTextureRegion {
            source_texture_obj_id: 4,
            destination_texture_obj_id: 5,
            source_slice: 1,
            source_level: 2,
            source_origin_x: 1,
            source_origin_y: 2,
            source_origin_z: 3,
            source_size_width: 16,
            source_size_height: 16,
            source_size_depth: 1,
            destination_slice: 0,
            destination_level: 0,
            destination_origin_x: 4,
            destination_origin_y: 5,
            destination_origin_z: 6,
        };
        assert_eq!(
            RgfxBlitOpCopyTextureRegion::read_from_bytes(region.as_bytes()).unwrap(),
            region
        );

        let fill = RgfxBlitOpFillBuffer {
            buffer_obj_id: 2,
            range_location: 0,
            range_length: 1024,
            value: 0xab,
            reserved_19: [0; 7],
        };
        assert_eq!(
            RgfxBlitOpFillBuffer::read_from_bytes(fill.as_bytes()).unwrap(),
            fill
        );

        let color = RgfxBlitOpFillTextureColor {
            texture_obj_id: 4,
            level: 0,
            slice: 0,
            region_origin_x: 0,
            region_origin_y: 0,
            region_origin_z: 0,
            region_size_width: 32,
            region_size_height: 32,
            region_size_depth: 1,
            color_bits: [
                0.0f64.to_bits(),
                0.25f64.to_bits(),
                0.5f64.to_bits(),
                1.0f64.to_bits(),
            ],
            pixel_format: 80,
        };
        assert_eq!(
            RgfxBlitOpFillTextureColor::read_from_bytes(color.as_bytes()).unwrap(),
            color
        );
        assert_eq!(f64::from_bits(color.color_bits[3]), 1.0);

        let sync = RgfxBlitOpSynchronizeResource {
            resource_obj_id: 4,
            resource_kind: RGFX_BLIT_RESOURCE_TEXTURE,
            reserved_0c: 0,
        };
        assert_eq!(
            RgfxBlitOpSynchronizeResource::read_from_bytes(sync.as_bytes()).unwrap(),
            sync
        );

        let fence = RgfxBlitOpWaitForFence { fence_obj_id: 9 };
        assert_eq!(
            RgfxBlitOpWaitForFence::read_from_bytes(fence.as_bytes()).unwrap(),
            fence
        );
    }
}
