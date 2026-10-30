//! Platform traits for Metal resources and state objects: `GpuResource`, `Buffer`, `Texture`,
//! `Heap`, `SamplerState` (architecture 4.1's `StateObject`), `DepthStencilState`,
//! `ComputePipeline`, `RenderPipeline`, `Library`.
//!
//! Conventions (architecture 4.2): one method per selector core sends; each doc line is
//! `sel: <selector> [SPI]`. Integer widths
//! follow the runtime's encodings (D41), as `method_getTypeEncoding` reads them on the AGX
//! classes (macOS 26.7), where a declaration in the headers differs. Methods with arguments,
//! creators and actions return `RaiseResult<T>`; zero-argument property getters and descriptor
//! property setters do not (the metal crate wraps them in `send_get`). A handle argument core can
//! pass as nil is `Option<&P::X>`; one it never passes as nil is `&P::X`.

use super::host_range::HostRange;
use super::types::{MtlRegion, MtlSize, NsRange, Swizzle};
use super::{Handle, Platform};
use crate::host::HostPtr;
use crate::raise::{OnFault, RaiseResult};

/// Architecture 4.1's name for the sampler state's bound (`type SamplerState: StateObject`).
pub use self::SamplerState as StateObject;

/// `MTLResource`: the selectors core sends to buffers and textures through the protocol they share.
pub trait GpuResource<P: Platform>: Handle {
    /// sel: device
    fn device(&self) -> Option<P::Device>;
    /// sel: resourceOptions
    fn resource_options(&self) -> u64;
    /// sel: setLabel:
    fn set_label(&self, label: &str) -> RaiseResult<()>;
    /// sel: setPurgeableState:
    /// Returns the PREVIOUS state; a previous `Empty` on devolatilize means the contents were lost
    /// (guest-visible).
    fn set_purgeable_state(&self, state: u64) -> RaiseResult<u64>;
    /// sel: storageMode
    fn storage_mode(&self) -> u64;
}

/// `MTLBuffer` (with `MTLBufferSPI`).
pub trait Buffer<P: Platform>: GpuResource<P> {
    /// sel: contents
    /// The buffer's CPU address; core reads and writes through it (never NULL-checked).
    fn contents(&self) -> HostPtr;
    /// sel: detachBacking SPI
    /// The runtime returns `BOOL` (`B16@0:8`); core never reads it.
    fn detach_backing(&self) -> RaiseResult<()>;
    /// sel: didModifyRange:
    /// Reached both under a handler and with none: `on` says which this call is.
    fn did_modify_range(&self, on: OnFault, range: NsRange) -> RaiseResult<()>;
    /// sel: gpuAddress SPI
    fn gpu_address(&self) -> u64;
    /// sel: iosurface SPI
    /// Returns +1: the handle is the retain core makes right after the getter (`[(id)surface
    /// retain]`); callers do not `.clone_ref()` it. `None` when the buffer has no IOSurface.
    fn iosurface(&self) -> Option<P::Surface>;
    /// sel: length
    fn length(&self) -> u64;
    /// sel: newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage: SPI
    fn new_linear_texture(
        &self,
        descriptor: &P::TextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> RaiseResult<Option<P::Texture>>;
    /// sel: newTextureWithDescriptor:offset:bytesPerRow:
    /// Reached both under a handler and with none: `on` says which this call is.
    fn new_texture(
        &self,
        on: OnFault,
        descriptor: &P::TextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
    ) -> RaiseResult<Option<P::Texture>>;
    /// sel: replaceBackingWithRanges:readOnly: SPI
    /// `false` makes core throw "Failed to replace buffer backing".
    fn replace_backing(&self, ranges: &P::AddressRangeArray, read_only: bool) -> RaiseResult<bool>;
}

/// `MTLTexture` (with `MTLTextureSPI`).
///
/// `didModifyData` (SPI) is not declared: only the tiled EFI path, which nothing reaches, would
/// send it (D33).
pub trait Texture<P: Platform>: GpuResource<P> {
    /// sel: arrayLength
    fn array_length(&self) -> u64;
    /// sel: depth
    fn depth(&self) -> u64;
    /// sel: gpuResourceID
    /// `MTLResourceID._impl` (runtime `{MTLResourceID=Q}`).
    fn gpu_resource_id(&self) -> u64;
    /// sel: height
    fn height(&self) -> u64;
    /// sel: iosurface SPI
    /// Returns +1: the handle is the retain core makes right after the getter (`[(id)surface
    /// retain]`); callers do not `.clone_ref()` it. `None` when the texture has no IOSurface.
    fn iosurface(&self) -> Option<P::Surface>;
    /// sel: isShareable
    fn is_shareable(&self) -> bool;
    /// sel: mipmapLevelCount
    fn mipmap_level_count(&self) -> u64;
    /// sel: newSharedTextureHandle
    fn new_shared_texture_handle(&self) -> RaiseResult<Option<P::SharedTextureHandle>>;
    /// sel: newTextureViewWithPixelFormat:
    fn new_texture_view(&self, pixel_format: u64) -> RaiseResult<Option<P::Texture>>;
    /// sel: newTextureViewWithPixelFormat:resourceIndex: SPI
    fn new_texture_view_at_index(
        &self,
        pixel_format: u64,
        resource_index: u64,
    ) -> RaiseResult<Option<P::Texture>>;
    /// sel: newTextureViewWithPixelFormat:textureType:levels:slices:
    fn new_texture_subview(
        &self,
        pixel_format: u64,
        texture_type: u64,
        level_range: NsRange,
        slice_range: NsRange,
    ) -> RaiseResult<Option<P::Texture>>;
    /// sel: newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:
    fn new_swizzled_texture_subview(
        &self,
        pixel_format: u64,
        texture_type: u64,
        level_range: NsRange,
        slice_range: NsRange,
        swizzle: Swizzle,
    ) -> RaiseResult<Option<P::Texture>>;
    /// sel: newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex: SPI
    fn new_swizzled_texture_subview_at_index(
        &self,
        pixel_format: u64,
        texture_type: u64,
        levels: NsRange,
        slices: NsRange,
        swizzle: Swizzle,
        resource_index: u64,
    ) -> RaiseResult<Option<P::Texture>>;
    /// sel: newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex: SPI
    fn new_texture_subview_at_index(
        &self,
        pixel_format: u64,
        texture_type: u64,
        levels: NsRange,
        slices: NsRange,
        resource_index: u64,
    ) -> RaiseResult<Option<P::Texture>>;
    /// sel: pixelFormat
    fn pixel_format(&self) -> u64;
    /// sel: protectionOptions SPI
    /// Nonzero means the texture is never paged.
    fn protection_options(&self) -> u64;
    /// sel: replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:
    /// `pixel_bytes` covers the bytes Metal copies out of the region (`bytes_per_row` and
    /// `bytes_per_image` as the caller computed them); only its pointer is sent. Metal copies
    /// synchronously and keeps no pointer.
    fn replace_region(
        &self,
        region: MtlRegion,
        level: u64,
        slice: u64,
        pixel_bytes: HostRange<'_>,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> RaiseResult<()>;
    /// sel: resourceIndex SPI
    fn resource_index(&self) -> u64;
    /// sel: sampleCount
    fn sample_count(&self) -> u64;
    /// sel: textureType
    fn texture_type(&self) -> u64;
    /// sel: uniqueIdentifier SPI
    fn unique_identifier(&self) -> u64;
    /// sel: usage
    fn usage(&self) -> u64;
    /// sel: width
    fn width(&self) -> u64;
}

/// `MTLHeap` (with `MTLHeapSPI`). Not an `MTLResource`: its purgeable state, storage mode and
/// options are its own selectors.
pub trait Heap<P: Platform>: Handle {
    /// sel: detachBacking SPI
    /// The runtime returns `BOOL` (`B16@0:8`); core never reads it.
    fn detach_backing(&self) -> RaiseResult<()>;
    /// sel: gpuAddress SPI
    fn gpu_address(&self) -> u64;
    /// sel: newBufferWithLength:options:offset:
    fn new_buffer_of_length(
        &self,
        length: u64,
        options: u64,
        offset: u64,
    ) -> RaiseResult<Option<P::Buffer>>;
    /// sel: newTextureWithDescriptor:offset:
    fn new_texture(
        &self,
        descriptor: &P::TextureDescriptor,
        offset: u64,
    ) -> RaiseResult<Option<P::Texture>>;
    /// sel: replaceBackingWithRanges:readOnly: SPI
    /// `false` makes core throw "Failed to replace heap backing".
    fn replace_backing(&self, ranges: &P::AddressRangeArray, read_only: bool) -> RaiseResult<bool>;
    /// sel: resourceOptions
    fn resource_options(&self) -> u64;
    /// sel: setPurgeableState:
    fn set_purgeable_state(&self, state: u64) -> RaiseResult<u64>;
    /// sel: storageMode
    fn storage_mode(&self) -> u64;
}

/// `MTLSamplerState` (with `MTLSamplerStateSPI`).
pub trait SamplerState: Handle {
    /// sel: gpuResourceID
    /// `MTLResourceID._impl` (runtime `{MTLResourceID=Q}`).
    fn gpu_resource_id(&self) -> u64;
    /// sel: resourceIndex SPI
    fn resource_index(&self) -> u64;
}

/// `MTLDepthStencilState`.
pub trait DepthStencilState: Handle {
    /// sel: gpuResourceID
    /// `MTLResourceID._impl` (runtime `{MTLResourceID=Q}`). `RaiseResult`, unlike the other
    /// zero-argument getters (4.2 item 3): the host's AGX class implements it as a stub that raises
    /// `NSInvalidArgumentException`, which reaches the info decoder's `@catch`.
    fn gpu_resource_id(&self) -> RaiseResult<u64>;
    /// sel: resourceIndex SPI
    /// Undeclared in the headers: core sends it to an `id`, which clang binds to a `- (uint64_t)`
    /// declaration of another protocol. The runtime encoding is `Q16@0:8` (measured on the AGX
    /// class, macOS 26.7). `RaiseResult` as `gpu_resource_id`: the AGX stub raises.
    fn resource_index(&self) -> RaiseResult<u64>;
}

/// `MTLComputePipelineState` (with `MTLComputePipelineStateSPI`).
pub trait ComputePipeline: Handle {
    /// sel: gpuResourceID
    /// `MTLResourceID._impl` (runtime `{MTLResourceID=Q}`).
    fn gpu_resource_id(&self) -> u64;
    /// sel: imageblockMemoryLengthForDimensions:
    fn imageblock_memory_length(&self, imageblock_dimensions: MtlSize) -> RaiseResult<u64>;
    /// sel: maxTotalThreadsPerThreadgroup
    fn max_total_threads_per_threadgroup(&self) -> u64;
    /// sel: resourceIndex SPI
    fn resource_index(&self) -> u64;
    /// sel: staticThreadgroupMemoryLength
    fn static_threadgroup_memory_length(&self) -> u64;
    /// sel: supportIndirectCommandBuffers
    fn support_indirect_command_buffers(&self) -> bool;
    /// sel: threadExecutionWidth
    fn thread_execution_width(&self) -> u64;
    /// sel: uniqueIdentifier SPI
    fn unique_identifier(&self) -> u64;
    /// sel: uniqueIdentifier SPI, as the guest reads it (info command 460). Core registers the
    /// state under [`unique_identifier`](Self::unique_identifier) as it is made; a platform that
    /// cannot answer Metal's identifier of a state dies here, where the guest reads it.
    fn guest_unique_identifier(&self) -> u64 {
        self.unique_identifier()
    }
}

/// `MTLRenderPipelineState` (with `MTLRenderPipelineStateSPI`).
pub trait RenderPipeline: Handle {
    /// sel: gpuResourceID
    /// `MTLResourceID._impl` (runtime `{MTLResourceID=Q}`).
    fn gpu_resource_id(&self) -> u64;
    /// sel: imageblockMemoryLengthForDimensions:
    fn imageblock_memory_length(&self, imageblock_dimensions: MtlSize) -> RaiseResult<u64>;
    /// sel: imageblockSampleLength
    fn imageblock_sample_length(&self) -> u64;
    /// sel: maxTotalThreadsPerThreadgroup
    fn max_total_threads_per_threadgroup(&self) -> u64;
    /// sel: resourceIndex SPI
    fn resource_index(&self) -> u64;
    /// sel: supportIndirectCommandBuffers
    fn support_indirect_command_buffers(&self) -> bool;
    /// sel: threadgroupSizeMatchesTileSize
    fn threadgroup_size_matches_tile_size(&self) -> bool;
    /// sel: uniqueIdentifier SPI
    fn unique_identifier(&self) -> u64;
    /// sel: uniqueIdentifier SPI, as the guest reads it (info command 460). Core registers the
    /// state under [`unique_identifier`](Self::unique_identifier) as it is made; a platform that
    /// cannot answer Metal's identifier of a state dies here, where the guest reads it.
    fn guest_unique_identifier(&self) -> u64 {
        self.unique_identifier()
    }
}

/// One function-constant value, keyed by its index (`setConstantValue:type:atIndex:`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunctionConstant {
    /// The constant's index.
    pub index: u32,
    /// `MTLDataType` of the value.
    pub data_type: u32,
    /// The value's raw little-endian bits, zero-extended to eight bytes.
    pub value: u64,
}

impl FunctionConstant {
    /// The value's bytes, as many as `data_type` holds: `None` for a type wider than eight
    /// bytes, or not a scalar or small vector of one.
    pub fn bytes(&self) -> Option<Vec<u8>> {
        // `MTLDataType` raw values (Metal's MTLDataType.h).
        let len = match self.data_type {
            // Bool, Char, UChar
            53 | 45 | 49 => 1,
            // Bool2, Char2, UChar2, Short, UShort, Half
            54 | 46 | 50 | 37 | 41 | 16 => 2,
            // Bool3, Char3, UChar3
            55 | 47 | 51 => 3,
            // Bool4, Char4, UChar4, Short2, UShort2, Half2, Int, UInt, Float
            56 | 48 | 52 | 38 | 42 | 17 | 29 | 33 | 3 => 4,
            // Short3, UShort3, Half3
            39 | 43 | 18 => 6,
            // Short4, UShort4, Half4, Int2, UInt2, Float2, Long, ULong
            40 | 44 | 19 | 30 | 34 | 4 | 81 | 85 => 8,
            _ => return None,
        };
        Some(self.value.to_le_bytes()[..len].to_vec())
    }
}

/// `MTLLibrary`.
pub trait Library<P: Platform>: Handle {
    /// sel: functionNames
    fn function_names(&self) -> Vec<String>;
    /// sel: newFunctionWithName:
    /// Reached both under a handler and with none: `on` says which this call is.
    fn new_function(&self, on: OnFault, function_name: &str) -> RaiseResult<Option<P::Function>>;
    /// sel: newFunctionWithName:constantValues:error:
    /// The function named `function_name`, specialised by `constants`, each keyed by its
    /// index. Nil for a name the library does not hold, or a specialisation the platform
    /// refuses. Reached with no handler.
    fn new_function_with_constants(
        &self,
        on: OnFault,
        function_name: &str,
        constants: &[FunctionConstant],
    ) -> RaiseResult<Option<P::Function>>;
}
