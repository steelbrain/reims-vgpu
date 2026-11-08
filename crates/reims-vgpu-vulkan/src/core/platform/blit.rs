//! Platform trait `BlitEncoder`: every `MTLBlitCommandEncoder` selector core sends (D4), and
//! `ResourceRef`, the `id<MTLResource>` argument the three encoder traits take.
//!
//! `BlitCmd::selector` names each command's selector.
//!
//! Conventions (architecture 4.2): every method takes arguments or encodes, so every one returns
//! `RaiseResult<T>`; an object argument is `Option<&P::X>` unless every caller is shown never to
//! pass nil, and the `nil:` line of such a method says why.

// One method per selector and one argument per keyword (D4): Metal's copy selectors take up
// to ten.
#![allow(clippy::too_many_arguments)]

use std::ops::Range;

use super::host_range::HostRange;
use super::types::{ClearColor, MtlOrigin, MtlRegion, MtlSize, NsRange};
use super::{Handle, Platform};
use crate::core::raise::{OnFault, RaiseResult};

/// An `id<MTLResource>` core passes to `synchronizeResource:`, `useResources:count:usage:[stages:]`
/// and `memoryBarrierWithResources:count:`.
///
/// Core gets it from `-[PGResourceManagerDelegate getResourceForReference:]`, which returns
/// `-[PGResource getResource]`: a buffer (`PGSharedBufferResource`, `PGPrivateBufferResource`,
/// `PGDetachBufferResource`, `PGHeapBufferResource`, `PGMapperRefBufferResource`), a texture (the
/// texture resources) or an indirect command buffer (`PGIndirectCommandBufferResource`). Every
/// other resource kind raises in `-[PGResource getResource]` before the send, so these three are
/// every object core can pass. Nil is `None` around this type.
///
/// A borrowed reference, not a handle: the send retains nothing (D6).
#[derive(Debug)]
pub enum ResourceRef<'a, P: Platform> {
    /// An `id<MTLBuffer>`.
    Buffer(&'a P::Buffer),
    /// An `id<MTLTexture>`.
    Texture(&'a P::Texture),
    /// An `id<MTLIndirectCommandBuffer>`.
    Icb(&'a P::Icb),
}

// Written out: a derive would require `P: Clone`/`P: Copy`.
impl<P: Platform> Clone for ResourceRef<'_, P> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<P: Platform> Copy for ResourceRef<'_, P> {}

/// One `MTLBlitCommandEncoder` command.
///
/// One variant per selector, whose fields are the selector's arguments in order;
/// [`BlitEncoder::encode`] sends one.
#[derive(Debug)]
#[allow(missing_docs)] // the fields are the selector's arguments
pub enum BlitCmd<'a, P: Platform> {
    /// `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:`
    /// nil: `source_buffer`, `destination_texture` —
    /// never nil: sends `_transferBuffer` after `-ensurePhysical` threw on nil, and `_textures[i]`
    /// (an NSArray element).
    CopyBufferToTexture {
        source_buffer: &'a P::Buffer,
        source_offset: u64,
        source_bytes_per_row: u64,
        source_bytes_per_image: u64,
        source_size: MtlSize,
        destination_texture: &'a P::Texture,
        destination_slice: u64,
        destination_level: u64,
        destination_origin: MtlOrigin,
    },
    /// `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:`
    CopyBufferToTextureOptions {
        source_buffer: Option<&'a P::Buffer>,
        source_offset: u64,
        source_bytes_per_row: u64,
        source_bytes_per_image: u64,
        source_size: MtlSize,
        destination_texture: Option<&'a P::Texture>,
        destination_slice: u64,
        destination_level: u64,
        destination_origin: MtlOrigin,
        options: u64,
    },
    /// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`
    CopyBufferToBuffer {
        source_buffer: Option<&'a P::Buffer>,
        source_offset: u64,
        destination_buffer: Option<&'a P::Buffer>,
        destination_offset: u64,
        size: u64,
    },
    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:`
    /// nil: `source_texture`, `destination_buffer` —
    /// never nil: sends `_textures[i]` (an NSArray element) and `_transferBuffer` after
    /// `-ensurePhysical` threw on nil.
    CopyTextureToBuffer {
        source_texture: &'a P::Texture,
        source_slice: u64,
        source_level: u64,
        source_origin: MtlOrigin,
        source_size: MtlSize,
        destination_buffer: &'a P::Buffer,
        destination_offset: u64,
        destination_bytes_per_row: u64,
        destination_bytes_per_image: u64,
    },
    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:`
    CopyTextureToBufferOptions {
        source_texture: Option<&'a P::Texture>,
        source_slice: u64,
        source_level: u64,
        source_origin: MtlOrigin,
        source_size: MtlSize,
        destination_buffer: Option<&'a P::Buffer>,
        destination_offset: u64,
        destination_bytes_per_row: u64,
        destination_bytes_per_image: u64,
        options: u64,
    },
    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:`
    /// nil: `source_texture`, `destination_texture` — never nil: both come from
    /// `getTextureForReferenceNonNull:`, which throws on nil.
    CopyTextureRegion {
        source_texture: &'a P::Texture,
        source_slice: u64,
        source_level: u64,
        source_origin: MtlOrigin,
        source_size: MtlSize,
        destination_texture: &'a P::Texture,
        destination_slice: u64,
        destination_level: u64,
        destination_origin: MtlOrigin,
    },
    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:` (SPI)
    /// nil: `source_texture`, `destination_texture` — never nil: both come from
    /// `getTextureForReferenceNonNull:`, which throws on nil.
    CopyTextureRegionOptions {
        source_texture: &'a P::Texture,
        source_slice: u64,
        source_level: u64,
        source_origin: MtlOrigin,
        source_size: MtlSize,
        destination_texture: &'a P::Texture,
        destination_slice: u64,
        destination_level: u64,
        destination_origin: MtlOrigin,
        options: u64,
    },
    /// `copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:`
    /// nil: `source_texture`, `destination_texture` — never nil: both come from
    /// `getTextureForReferenceNonNull:`, which throws on nil.
    CopyTextureSlices {
        source_texture: &'a P::Texture,
        source_slice: u64,
        source_level: u64,
        destination_texture: &'a P::Texture,
        destination_slice: u64,
        destination_level: u64,
        slice_count: u64,
        level_count: u64,
    },
    /// `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:`
    /// nil: `source`, `destination` — never nil: both
    /// come from the decoder's `getIndirectCommandBufferForReferenceNonNull:`, which throws on nil.
    CopyIndirectCommandBuffer {
        source: &'a P::Icb,
        source_range: NsRange,
        destination: &'a P::Icb,
        destination_index: u64,
    },
    /// `endEncoding`
    EndEncoding,
    /// `fillBuffer:range:pattern4:` (SPI)
    /// nil: `buffer` — never nil: it comes from `getBufferForReferenceNonNull:`, which
    /// throws on nil.
    FillBufferPattern4 {
        buffer: &'a P::Buffer,
        range: NsRange,
        pattern: u32,
    },
    /// `fillBuffer:range:value:`
    /// nil: `buffer` — never nil: it comes from `getBufferForReferenceNonNull:`, which
    /// throws on nil.
    FillBuffer {
        buffer: &'a P::Buffer,
        range: NsRange,
        value: u8,
    },
    /// `fillTexture:level:slice:region:bytes:length:` (SPI)
    /// nil: `texture` — never nil: it comes from `getTextureForReferenceNonNull:`, which
    /// throws on nil. `bytes` is the pointer and the length: a shared buffer's `contents +
    /// offset`, after the caller checked `offset + length <= buffer.length` without overflow and
    /// the buffer's storage mode; the buffer outlives the send.
    FillTexture {
        texture: &'a P::Texture,
        level: u64,
        slice: u64,
        region: MtlRegion,
        bytes: HostRange<'a>,
    },
    /// `fillTexture:level:slice:region:color:pixelFormat:` (SPI)
    /// nil: `texture` — never nil: it comes from `getTextureForReferenceNonNull:`, which
    /// throws on nil.
    FillTextureColor {
        texture: &'a P::Texture,
        level: u64,
        slice: u64,
        region: MtlRegion,
        color: ClearColor,
        pixel_format: u64,
    },
    /// `generateMipmapsForTexture:`
    GenerateMipmaps { texture: Option<&'a P::Texture> },
    /// `invalidateCompressedTexture:` (SPI)
    /// nil: `texture` — never nil: it comes from `getTextureForReferenceNonNull:`, which
    /// throws on nil.
    InvalidateCompressedTexture { texture: &'a P::Texture },
    /// `invalidateCompressedTexture:slice:level:` (SPI)
    /// nil: `texture` — never nil: it comes from `getTextureForReferenceNonNull:`, which
    /// throws on nil.
    InvalidateCompressedTextureSlice {
        texture: &'a P::Texture,
        slice: u64,
        level: u64,
    },
    /// `optimizeContentsForCPUAccess:`
    /// nil: `texture` — never nil: it comes from `getTextureForReferenceNonNull:`, which
    /// throws on nil.
    OptimizeForCpu { texture: &'a P::Texture },
    /// `optimizeContentsForCPUAccess:slice:level:`
    /// nil: `texture` — never nil: it comes from `getTextureForReferenceNonNull:`, which
    /// throws on nil.
    OptimizeForCpuSlice {
        texture: &'a P::Texture,
        slice: u64,
        level: u64,
    },
    /// `optimizeContentsForGPUAccess:`
    /// nil: `texture` — never nil: it comes from `getTextureForReferenceNonNull:`, which
    /// throws on nil.
    OptimizeForGpu { texture: &'a P::Texture },
    /// `optimizeContentsForGPUAccess:slice:level:`
    /// nil: `texture` — never nil: it comes from `getTextureForReferenceNonNull:`, which
    /// throws on nil.
    OptimizeForGpuSlice {
        texture: &'a P::Texture,
        slice: u64,
        level: u64,
    },
    /// `optimizeIndirectCommandBuffer:withRange:`
    /// nil: `indirect_command_buffer` — never nil: it comes from the decoder's
    /// `getIndirectCommandBufferForReferenceNonNull:`, which throws on nil.
    OptimizeIndirectCommandBuffer {
        indirect_command_buffer: &'a P::Icb,
        range: NsRange,
    },
    /// `resetCommandsInBuffer:withRange:`
    /// nil: `buffer` — never nil: it comes from the decoder's
    /// `getIndirectCommandBufferForReferenceNonNull:`, which throws on nil.
    ResetCommandsInBuffer { buffer: &'a P::Icb, range: NsRange },
    /// `synchronizeResource:`
    /// nil: `resource` comes from `getResourceForReference:`, nil for reference 0 or a missing
    /// object
    SynchronizeResource {
        resource: Option<ResourceRef<'a, P>>,
    },
    /// `synchronizeTexture:slice:level:`
    /// nil: `texture` — never nil: it comes from `getTextureForReferenceNonNull:`, which
    /// throws on nil.
    SynchronizeTexture {
        texture: &'a P::Texture,
        slice: u64,
        level: u64,
    },
    /// `updateFence:`
    /// nil: `fence` — never nil: it comes from `getFenceForReferenceNonNull:`, which
    /// throws on nil.
    UpdateFence { fence: &'a P::Fence },
    /// `waitForFence:`
    /// nil: `fence` — never nil: it comes from `getFenceForReferenceNonNull:`, which
    /// throws on nil.
    WaitForFence { fence: &'a P::Fence },
}

impl<P: Platform> BlitCmd<'_, P> {
    /// The selector this sends.
    pub fn selector(&self) -> &'static str {
        match self {
            Self::CopyBufferToTexture { .. } => {
                "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:"
            }
            Self::CopyBufferToTextureOptions { .. } => {
                "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:"
            }
            Self::CopyBufferToBuffer { .. } => {
                "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"
            }
            Self::CopyTextureToBuffer { .. } => {
                "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:"
            }
            Self::CopyTextureToBufferOptions { .. } => {
                "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:"
            }
            Self::CopyTextureRegion { .. } => {
                "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:"
            }
            Self::CopyTextureRegionOptions { .. } => {
                "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:"
            }
            Self::CopyTextureSlices { .. } => {
                "copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:"
            }
            Self::CopyIndirectCommandBuffer { .. } => {
                "copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:"
            }
            Self::EndEncoding => "endEncoding",
            Self::FillBufferPattern4 { .. } => "fillBuffer:range:pattern4:",
            Self::FillBuffer { .. } => "fillBuffer:range:value:",
            Self::FillTexture { .. } => "fillTexture:level:slice:region:bytes:length:",
            Self::FillTextureColor { .. } => "fillTexture:level:slice:region:color:pixelFormat:",
            Self::GenerateMipmaps { .. } => "generateMipmapsForTexture:",
            Self::InvalidateCompressedTexture { .. } => "invalidateCompressedTexture:",
            Self::InvalidateCompressedTextureSlice { .. } => {
                "invalidateCompressedTexture:slice:level:"
            }
            Self::OptimizeForCpu { .. } => "optimizeContentsForCPUAccess:",
            Self::OptimizeForCpuSlice { .. } => "optimizeContentsForCPUAccess:slice:level:",
            Self::OptimizeForGpu { .. } => "optimizeContentsForGPUAccess:",
            Self::OptimizeForGpuSlice { .. } => "optimizeContentsForGPUAccess:slice:level:",
            Self::OptimizeIndirectCommandBuffer { .. } => {
                "optimizeIndirectCommandBuffer:withRange:"
            }
            Self::ResetCommandsInBuffer { .. } => "resetCommandsInBuffer:withRange:",
            Self::SynchronizeResource { .. } => "synchronizeResource:",
            Self::SynchronizeTexture { .. } => "synchronizeTexture:slice:level:",
            Self::UpdateFence { .. } => "updateFence:",
            Self::WaitForFence { .. } => "waitForFence:",
        }
    }
}
/// The blit encoder core gets from `-[MTLCommandBuffer blitCommandEncoder]`
/// (`CommandBuffer::blit_encoder`). Core never checks that for nil, so its callers hold an
/// `Option<P::BlitEncoder>` and send through `nil_send!`.
pub trait BlitEncoder<P: Platform>: Handle {
    /// Sends one [`BlitCmd`]. `on` says what a raise in a command reached both under a handler
    /// and with none (`EndEncoding`) does; the other commands are reached one way only.
    fn encode(&self, on: OnFault, cmd: BlitCmd<'_, P>) -> RaiseResult<()>;
    /// No Metal counterpart: copies the guest's RAM at `runs`, host addresses in the host's
    /// mapping of the guest's RAM (the memory map's), in the bytes' order, into `destination`
    /// from `destination_offset`, ordered against the encoder's other commands as a
    /// `CopyBufferToBuffer` is; so a copy of guest pages needs no buffer over a view of them.
    /// False, with nothing encoded, where the platform does not copy them: a device that copies
    /// none of the guest's RAM ([`GpuDevice::copies_guest_ram`]), runs outside the RAM it was
    /// handed, or a refusal of the runtime's. `on` is as for [`Self::encode`].
    ///
    /// [`GpuDevice::copies_guest_ram`]: super::device::GpuDevice::copies_guest_ram
    fn copy_from_guest_ram(
        &self,
        on: OnFault,
        runs: &[Range<u64>],
        destination: &P::Buffer,
        destination_offset: u64,
    ) -> RaiseResult<bool>;
    /// No Metal counterpart: copies `source` into the guest's RAM at `runs`, each a byte of
    /// `source` and a run, a host address range as [`Self::copy_from_guest_ram`] takes, that
    /// takes the bytes from that byte on, in the order given: so a buffer that keeps a surface's
    /// layout copies its rows out in one call. Ordered as [`Self::copy_from_guest_ram`] is, the
    /// other way; false, with nothing encoded, where it does not copy.
    fn copy_to_guest_ram(
        &self,
        on: OnFault,
        source: &P::Buffer,
        runs: &[(u64, Range<u64>)],
    ) -> RaiseResult<bool>;
}
