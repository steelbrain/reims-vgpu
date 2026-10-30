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

/// Op `COPY_BUFFER_TO_BUFFER` ([`RgfxOpCopyBufferToBuffer`]):
/// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`.
pub const RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER: u16 = 0x3000;
/// Op `COPY_BUFFER_TO_TEXTURE` ([`RgfxOpCopyBufferToTexture`]):
/// `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:`.
pub const RGFX_OP_BLIT_COPY_BUFFER_TO_TEXTURE: u16 = 0x3001;
/// Op `COPY_BUFFER_TO_TEXTURE_OPTIONS` ([`RgfxOpCopyBufferToTextureOptions`]): the
/// `:options:` form of the above.
pub const RGFX_OP_BLIT_COPY_BUFFER_TO_TEXTURE_OPTIONS: u16 = 0x3002;
/// Op `COPY_TEXTURE_TO_BUFFER` ([`RgfxOpCopyTextureToBuffer`]):
/// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:`.
pub const RGFX_OP_BLIT_COPY_TEXTURE_TO_BUFFER: u16 = 0x3003;
/// Op `COPY_TEXTURE_TO_BUFFER_OPTIONS` ([`RgfxOpCopyTextureToBufferOptions`]): the
/// `:options:` form of the above.
pub const RGFX_OP_BLIT_COPY_TEXTURE_TO_BUFFER_OPTIONS: u16 = 0x3004;
/// Op `COPY_TEXTURE_REGION` ([`RgfxOpCopyTextureRegion`]):
/// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:`.
pub const RGFX_OP_BLIT_COPY_TEXTURE_REGION: u16 = 0x3005;
/// Op `COPY_TEXTURE_REGION_OPTIONS` ([`RgfxOpCopyTextureRegionOptions`]): the
/// `:options:` form of the above (SPI).
pub const RGFX_OP_BLIT_COPY_TEXTURE_REGION_OPTIONS: u16 = 0x3006;
/// Op `COPY_TEXTURE_SLICES` ([`RgfxOpCopyTextureSlices`]):
/// `copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:`.
pub const RGFX_OP_BLIT_COPY_TEXTURE_SLICES: u16 = 0x3007;
/// Op `COPY_INDIRECT_COMMAND_BUFFER` ([`RgfxOpCopyIndirectCommandBuffer`]):
/// `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:`.
pub const RGFX_OP_BLIT_COPY_INDIRECT_COMMAND_BUFFER: u16 = 0x3008;
/// Op `FILL_BUFFER` ([`RgfxOpFillBuffer`]): `fillBuffer:range:value:`.
pub const RGFX_OP_BLIT_FILL_BUFFER: u16 = 0x3009;
/// Op `FILL_BUFFER_PATTERN4` ([`RgfxOpFillBufferPattern4`]): `fillBuffer:range:pattern4:` (SPI).
pub const RGFX_OP_BLIT_FILL_BUFFER_PATTERN4: u16 = 0x300a;
/// Op `FILL_TEXTURE` ([`RgfxOpFillTexture`]): `fillTexture:level:slice:region:bytes:length:` (SPI).
pub const RGFX_OP_BLIT_FILL_TEXTURE: u16 = 0x300b;
/// Op `FILL_TEXTURE_COLOR` ([`RgfxOpFillTextureColor`]):
/// `fillTexture:level:slice:region:color:pixelFormat:` (SPI).
pub const RGFX_OP_BLIT_FILL_TEXTURE_COLOR: u16 = 0x300c;
/// Op `GENERATE_MIPMAPS` ([`RgfxOpGenerateMipmaps`]): `generateMipmapsForTexture:`.
pub const RGFX_OP_BLIT_GENERATE_MIPMAPS: u16 = 0x300d;
/// Op `INVALIDATE_COMPRESSED_TEXTURE` ([`RgfxOpInvalidateCompressedTexture`]):
/// `invalidateCompressedTexture:` (SPI).
pub const RGFX_OP_BLIT_INVALIDATE_COMPRESSED_TEXTURE: u16 = 0x300e;
/// Op `INVALIDATE_COMPRESSED_TEXTURE_SLICE` ([`RgfxOpInvalidateCompressedTextureSlice`]):
/// `invalidateCompressedTexture:slice:level:` (SPI).
pub const RGFX_OP_BLIT_INVALIDATE_COMPRESSED_TEXTURE_SLICE: u16 = 0x300f;
/// Op `OPTIMIZE_FOR_CPU` ([`RgfxOpOptimizeForCpu`]): `optimizeContentsForCPUAccess:`.
pub const RGFX_OP_BLIT_OPTIMIZE_FOR_CPU: u16 = 0x3010;
/// Op `OPTIMIZE_FOR_CPU_SLICE` ([`RgfxOpOptimizeForCpuSlice`]):
/// `optimizeContentsForCPUAccess:slice:level:`.
pub const RGFX_OP_BLIT_OPTIMIZE_FOR_CPU_SLICE: u16 = 0x3011;
/// Op `OPTIMIZE_FOR_GPU` ([`RgfxOpOptimizeForGpu`]): `optimizeContentsForGPUAccess:`.
pub const RGFX_OP_BLIT_OPTIMIZE_FOR_GPU: u16 = 0x3012;
/// Op `OPTIMIZE_FOR_GPU_SLICE` ([`RgfxOpOptimizeForGpuSlice`]):
/// `optimizeContentsForGPUAccess:slice:level:`.
pub const RGFX_OP_BLIT_OPTIMIZE_FOR_GPU_SLICE: u16 = 0x3013;
/// Op `OPTIMIZE_INDIRECT_COMMAND_BUFFER` ([`RgfxOpOptimizeIndirectCommandBuffer`]):
/// `optimizeIndirectCommandBuffer:withRange:`.
pub const RGFX_OP_BLIT_OPTIMIZE_INDIRECT_COMMAND_BUFFER: u16 = 0x3014;
/// Op `RESET_COMMANDS_IN_BUFFER` ([`RgfxOpResetCommandsInBuffer`]):
/// `resetCommandsInBuffer:withRange:`.
pub const RGFX_OP_BLIT_RESET_COMMANDS_IN_BUFFER: u16 = 0x3015;
/// Op `SYNCHRONIZE_RESOURCE` ([`RgfxOpSynchronizeResource`]): `synchronizeResource:`.
pub const RGFX_OP_BLIT_SYNCHRONIZE_RESOURCE: u16 = 0x3016;
/// Op `SYNCHRONIZE_TEXTURE` ([`RgfxOpSynchronizeTexture`]): `synchronizeTexture:slice:level:`.
pub const RGFX_OP_BLIT_SYNCHRONIZE_TEXTURE: u16 = 0x3017;
/// Op `UPDATE_FENCE` ([`RgfxOpUpdateFence`]): `updateFence:`.
pub const RGFX_OP_BLIT_UPDATE_FENCE: u16 = 0x3018;
/// Op `WAIT_FOR_FENCE` ([`RgfxOpWaitForFence`]): `waitForFence:`.
pub const RGFX_OP_BLIT_WAIT_FOR_FENCE: u16 = 0x3019;

/// The kind of resource a [`RgfxOpSynchronizeResource`] names, which picks the
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

/// The standard derives every blit op body carries.
macro_rules! op_body {
    ($(#[$m:meta])* $v:vis struct $name:ident { $($(#[$fm:meta])* $fv:vis $f:ident : $t:ty),* $(,)? }) => {
        $(#[$m])*
        #[repr(C)]
        #[derive(
            Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
        )]
        $v struct $name { $($(#[$fm])* $fv $f : $t),* }
    };
}

op_body! {
    /// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`. Either
    /// buffer may be nil (`0`).
    pub struct RgfxOpCopyBufferToBuffer {
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
}

op_body! {
    /// `copyFromBuffer:...:toTexture:...`. Both the buffer and the texture must name
    /// an object (nil faults the stream).
    pub struct RgfxOpCopyBufferToTexture {
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
}

op_body! {
    /// The `:options:` form of [`RgfxOpCopyBufferToTexture`]; either object may be
    /// nil (`0`).
    pub struct RgfxOpCopyBufferToTextureOptions {
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
}

op_body! {
    /// `copyFromTexture:...:toBuffer:...`. Both the texture and the buffer must name
    /// an object (nil faults the stream).
    pub struct RgfxOpCopyTextureToBuffer {
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
}

op_body! {
    /// The `:options:` form of [`RgfxOpCopyTextureToBuffer`]; either object may be
    /// nil (`0`).
    pub struct RgfxOpCopyTextureToBufferOptions {
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
}

op_body! {
    /// `copyFromTexture:...:toTexture:...` (the region form). Both textures must name
    /// an object (nil faults the stream).
    pub struct RgfxOpCopyTextureRegion {
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
}

op_body! {
    /// The `:options:` form of [`RgfxOpCopyTextureRegion`] (SPI). Both textures must
    /// name an object (nil faults the stream).
    pub struct RgfxOpCopyTextureRegionOptions {
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
}

op_body! {
    /// `copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:`.
    /// Both textures must name an object (nil faults the stream).
    pub struct RgfxOpCopyTextureSlices {
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
}

op_body! {
    /// `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:`. Both
    /// indirect command buffers must name an object (nil faults the stream).
    pub struct RgfxOpCopyIndirectCommandBuffer {
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
}

op_body! {
    /// `fillBuffer:range:value:`. The buffer must name an object (nil faults the
    /// stream).
    pub struct RgfxOpFillBuffer {
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
}

op_body! {
    /// `fillBuffer:range:pattern4:` (SPI). The buffer must name an object (nil faults
    /// the stream).
    pub struct RgfxOpFillBufferPattern4 {
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
}

op_body! {
    /// The fixed head of a `fillTexture:level:slice:region:bytes:length:` op (SPI).
    /// The `length` bytes of fill data follow it at [`RGFX_OP_BLIT_FILL_TEXTURE_BYTES_OFFSET`].
    /// The texture must name an object (nil faults the stream).
    pub struct RgfxOpFillTexture {
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
}

/// Where a `FILL_TEXTURE` op's fill bytes start in its body: right after the fixed
/// [`RgfxOpFillTexture`] head.
pub const RGFX_OP_BLIT_FILL_TEXTURE_BYTES_OFFSET: u32 = 80;

op_body! {
    /// `fillTexture:level:slice:region:color:pixelFormat:` (SPI). The texture must
    /// name an object (nil faults the stream).
    pub struct RgfxOpFillTextureColor {
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
}

op_body! {
    /// `generateMipmapsForTexture:`. The texture may be nil (`0`).
    pub struct RgfxOpGenerateMipmaps {
        /// The texture, or 0 for nil.
        pub texture_obj_id: u64,
    }
}

op_body! {
    /// `invalidateCompressedTexture:` (SPI). The texture must name an object (nil
    /// faults the stream).
    pub struct RgfxOpInvalidateCompressedTexture {
        /// The texture (must not be nil).
        pub texture_obj_id: u64,
    }
}

op_body! {
    /// `invalidateCompressedTexture:slice:level:` (SPI). The texture must name an
    /// object (nil faults the stream).
    pub struct RgfxOpInvalidateCompressedTextureSlice {
        /// The texture (must not be nil).
        pub texture_obj_id: u64,
        /// `slice`.
        pub slice: u64,
        /// `level`.
        pub level: u64,
    }
}

op_body! {
    /// `optimizeContentsForCPUAccess:`. The texture must name an object (nil faults
    /// the stream).
    pub struct RgfxOpOptimizeForCpu {
        /// The texture (must not be nil).
        pub texture_obj_id: u64,
    }
}

op_body! {
    /// `optimizeContentsForCPUAccess:slice:level:`. The texture must name an object
    /// (nil faults the stream).
    pub struct RgfxOpOptimizeForCpuSlice {
        /// The texture (must not be nil).
        pub texture_obj_id: u64,
        /// `slice`.
        pub slice: u64,
        /// `level`.
        pub level: u64,
    }
}

op_body! {
    /// `optimizeContentsForGPUAccess:`. The texture must name an object (nil faults
    /// the stream).
    pub struct RgfxOpOptimizeForGpu {
        /// The texture (must not be nil).
        pub texture_obj_id: u64,
    }
}

op_body! {
    /// `optimizeContentsForGPUAccess:slice:level:`. The texture must name an object
    /// (nil faults the stream).
    pub struct RgfxOpOptimizeForGpuSlice {
        /// The texture (must not be nil).
        pub texture_obj_id: u64,
        /// `slice`.
        pub slice: u64,
        /// `level`.
        pub level: u64,
    }
}

op_body! {
    /// `optimizeIndirectCommandBuffer:withRange:`. The indirect command buffer must
    /// name an object (nil faults the stream).
    pub struct RgfxOpOptimizeIndirectCommandBuffer {
        /// The indirect command buffer (must not be nil).
        pub icb_obj_id: u64,
        /// `range.location`.
        pub range_location: u64,
        /// `range.length`.
        pub range_length: u64,
    }
}

op_body! {
    /// `resetCommandsInBuffer:withRange:`. The indirect command buffer must name an
    /// object (nil faults the stream).
    pub struct RgfxOpResetCommandsInBuffer {
        /// The indirect command buffer (must not be nil).
        pub icb_obj_id: u64,
        /// `range.location`.
        pub range_location: u64,
        /// `range.length`.
        pub range_length: u64,
    }
}

op_body! {
    /// `synchronizeResource:`. The resource is any `id<MTLResource>`; `resource_kind`
    /// (an `RGFX_BLIT_RESOURCE_*`) picks the resolver, and [`RGFX_BLIT_RESOURCE_NIL`]
    /// (or a 0 `resource_obj_id`) is nil.
    pub struct RgfxOpSynchronizeResource {
        /// The resource, or 0 for nil.
        pub resource_obj_id: u64,
        /// The resource's kind: an `RGFX_BLIT_RESOURCE_*`.
        pub resource_kind: u32,
        /// Pads to a multiple of 8; 0.
        pub reserved_0c: u32,
    }
}

op_body! {
    /// `synchronizeTexture:slice:level:`. The texture must name an object (nil faults
    /// the stream).
    pub struct RgfxOpSynchronizeTexture {
        /// The texture (must not be nil).
        pub texture_obj_id: u64,
        /// `slice`.
        pub slice: u64,
        /// `level`.
        pub level: u64,
    }
}

op_body! {
    /// `updateFence:`. The fence must name an object (nil faults the stream).
    pub struct RgfxOpUpdateFence {
        /// The fence (must not be nil).
        pub fence_obj_id: u64,
    }
}

op_body! {
    /// `waitForFence:`. The fence must name an object (nil faults the stream).
    pub struct RgfxOpWaitForFence {
        /// The fence (must not be nil).
        pub fence_obj_id: u64,
    }
}

// The blit ops' layout.
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<RgfxOpCopyBufferToBuffer>() == 40);
    assert!(size_of::<RgfxOpCopyBufferToTexture>() == 104);
    assert!(size_of::<RgfxOpCopyBufferToTextureOptions>() == 112);
    assert!(size_of::<RgfxOpCopyTextureToBuffer>() == 104);
    assert!(size_of::<RgfxOpCopyTextureToBufferOptions>() == 112);
    assert!(size_of::<RgfxOpCopyTextureRegion>() == 120);
    assert!(size_of::<RgfxOpCopyTextureRegionOptions>() == 128);
    assert!(size_of::<RgfxOpCopyTextureSlices>() == 64);
    assert!(size_of::<RgfxOpCopyIndirectCommandBuffer>() == 40);
    assert!(size_of::<RgfxOpFillBuffer>() == 32);
    assert!(size_of::<RgfxOpFillBufferPattern4>() == 32);
    assert!(size_of::<RgfxOpFillTexture>() == 80);
    assert!(RGFX_OP_BLIT_FILL_TEXTURE_BYTES_OFFSET as usize == size_of::<RgfxOpFillTexture>());
    assert!(size_of::<RgfxOpFillTextureColor>() == 112);
    assert!(size_of::<RgfxOpGenerateMipmaps>() == 8);
    assert!(size_of::<RgfxOpInvalidateCompressedTexture>() == 8);
    assert!(size_of::<RgfxOpInvalidateCompressedTextureSlice>() == 24);
    assert!(size_of::<RgfxOpOptimizeForCpu>() == 8);
    assert!(size_of::<RgfxOpOptimizeForCpuSlice>() == 24);
    assert!(size_of::<RgfxOpOptimizeForGpu>() == 8);
    assert!(size_of::<RgfxOpOptimizeForGpuSlice>() == 24);
    assert!(size_of::<RgfxOpOptimizeIndirectCommandBuffer>() == 24);
    assert!(size_of::<RgfxOpResetCommandsInBuffer>() == 24);
    assert!(size_of::<RgfxOpSynchronizeResource>() == 16);
    assert!(size_of::<RgfxOpSynchronizeTexture>() == 24);
    assert!(size_of::<RgfxOpUpdateFence>() == 8);
    assert!(size_of::<RgfxOpWaitForFence>() == 8);
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
        let bb = RgfxOpCopyBufferToBuffer {
            source_buffer_obj_id: 2,
            destination_buffer_obj_id: 3,
            source_offset: 16,
            destination_offset: 32,
            size: 4096,
        };
        assert_eq!(
            RgfxOpCopyBufferToBuffer::read_from_bytes(bb.as_bytes()).unwrap(),
            bb
        );

        let bt = RgfxOpCopyBufferToTexture {
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
            RgfxOpCopyBufferToTexture::read_from_bytes(bt.as_bytes()).unwrap(),
            bt
        );

        let region = RgfxOpCopyTextureRegion {
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
            RgfxOpCopyTextureRegion::read_from_bytes(region.as_bytes()).unwrap(),
            region
        );

        let fill = RgfxOpFillBuffer {
            buffer_obj_id: 2,
            range_location: 0,
            range_length: 1024,
            value: 0xab,
            reserved_19: [0; 7],
        };
        assert_eq!(
            RgfxOpFillBuffer::read_from_bytes(fill.as_bytes()).unwrap(),
            fill
        );

        let color = RgfxOpFillTextureColor {
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
            RgfxOpFillTextureColor::read_from_bytes(color.as_bytes()).unwrap(),
            color
        );
        assert_eq!(f64::from_bits(color.color_bits[3]), 1.0);

        let sync = RgfxOpSynchronizeResource {
            resource_obj_id: 4,
            resource_kind: RGFX_BLIT_RESOURCE_TEXTURE,
            reserved_0c: 0,
        };
        assert_eq!(
            RgfxOpSynchronizeResource::read_from_bytes(sync.as_bytes()).unwrap(),
            sync
        );

        let fence = RgfxOpWaitForFence { fence_obj_id: 9 };
        assert_eq!(
            RgfxOpWaitForFence::read_from_bytes(fence.as_bytes()).unwrap(),
            fence
        );
    }
}
