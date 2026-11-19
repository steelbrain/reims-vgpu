//! Platform traits for Metal resources and state objects: `GpuResource`, `Buffer`, `Texture`,
//! `Heap`, `SamplerState` (architecture 4.1's `StateObject`), `DepthStencilState`,
//! `ComputePipeline`, `RenderPipeline`, `Library`. The tables a compute pipeline state makes are
//! `function_table`'s.
//!
//! Conventions (architecture 4.2): one method per selector core sends; each doc line is
//! `sel: <selector> [SPI]`. Integer widths
//! follow the runtime's encodings (D41), as `method_getTypeEncoding` reads them on the AGX
//! classes (macOS 26.7), where a declaration in the headers differs. Methods with arguments,
//! creators and actions return `RaiseResult<T>`; zero-argument property getters and descriptor
//! property setters do not (a platform dies at their send on a raise). A handle argument core can
//! pass as nil is `Option<&P::X>`; one it never passes as nil is `&P::X`.

use super::host_range::HostRange;
use super::types::{MtlRegion, MtlSize, NsRange, Swizzle};
use super::values::SurfacePlane;
use super::{Handle, Platform};
use crate::core::host::HostPtr;
use crate::core::raise::{OnFault, RaiseResult};

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
    /// sel: didModifyRange:
    /// Reached both under a handler and with none: `on` says which this call is.
    fn did_modify_range(&self, on: OnFault, range: NsRange) -> RaiseResult<()>;
    /// sel: gpuAddress SPI
    fn gpu_address(&self) -> u64;
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
    /// sel: newTiledTextureWithDescriptor:offset:bytesPerRow: SPI
    /// As `newTextureWithDescriptor:offset:bytesPerRow:`, and also a 2D texture of more than one
    /// level: a copy of the buffer's bytes from `offset`, rows `bytes_per_row` apart, its other
    /// levels made from them, taken at its first use.
    fn new_tiled_texture(
        &self,
        descriptor: &P::TextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
    ) -> RaiseResult<Option<P::Texture>>;
    /// sel: newTextureWithDescriptor:iosurface:plane: over the buffer — Metal's IOSurface
    /// creator, where the buffer stands for the surface: its first byte is the surface's base
    /// address, `plane` says where the asked plane lies in it, and the facts, where given, what
    /// the texture reads of the surface besides (plane 1 of a two-plane YCbCr or XR10 texture,
    /// the surface's range and matrix). The texture lies over the buffer's bytes, as one of
    /// `newTextureWithDescriptor:offset:bytesPerRow:` does. A platform that makes no such
    /// texture over a buffer answers as that creator answers.
    /// Reached both under a handler and with none: `on` says which this call is.
    fn new_texture_over_surface_plane(
        &self,
        on: OnFault,
        descriptor: &P::TextureDescriptor,
        plane: SurfacePlane,
    ) -> RaiseResult<Option<P::Texture>>;
    /// No Metal counterpart: the buffer's memory as a new file descriptor another API imports,
    /// with what the import names ([`ExportedMemory`]), where the buffer was made exportable
    /// ([`GpuDevice::new_exportable_buffer_of_length`](super::device::GpuDevice::new_exportable_buffer_of_length)).
    /// `Err` says why not: the default, and a buffer made as any other.
    #[cfg(unix)]
    fn export_memory(&self) -> Result<ExportedMemory, String> {
        Err("the platform exports no buffer's memory".to_owned())
    }
    /// No Metal counterpart: the buffer's memory as a new dma-buf of its allocation, with what an
    /// import names ([`ExportedMemory`], its `fd` a dma-buf), where the buffer was made so
    /// ([`GpuDevice::new_dma_buf_buffer_of_length_padding_unobserved`](super::device::GpuDevice::new_dma_buf_buffer_of_length_padding_unobserved)).
    /// `Err` says why not: the default, and a buffer made as any other.
    #[cfg(unix)]
    fn export_dma_buf(&self) -> Result<ExportedMemory, String> {
        Err("the platform exports no buffer's memory as a dma-buf".to_owned())
    }
    /// No Metal counterpart: says that another API wrote the buffer's exported memory
    /// ([`Buffer::export_memory`], [`Buffer::export_dma_buf`]) and that the caller waited for
    /// those writes (a CUDA stream's synchronize, VA-API's `vaSyncSurface`), so that the device's
    /// later reads of the buffer see them. Called once after each such write, while no command
    /// buffer that uses the buffer is in flight. `Ok` by default, for a platform whose devices
    /// need no acquire of memory another API wrote; `Err` says why the platform's device will not
    /// read the writes.
    #[cfg(unix)]
    fn external_write_done(&self) -> Result<(), String> {
        Ok(())
    }
    /// No Metal counterpart (Metal's own is `replaceBackingWithRanges:`, which no guest message
    /// asks): the no-copy buffer lies over `bytes` from now on, the same length, its object,
    /// GPU address and options as they were: the graphics lane moves a buffer over a view of
    /// guest pages into a view of more pages so, the bytes the same guest pages, so that one
    /// import holds many such buffers (`decode::surface_pages`, "Regroups"). `deallocator` runs
    /// once the platform let go of `bytes`; the buffer's deallocator before it runs once it let
    /// go of the old bytes. `Err` says why not, the buffer then over its old bytes and
    /// `deallocator` dropped unrun: the default, a buffer that is not a no-copy buffer, one a
    /// command buffer not yet committed has used, and one a committed command buffer still holds
    /// (one whose encoder the platform cannot encode again, until its submission; one with a
    /// dispatch left to the host's step, a GPU-written indirect one among them, from the
    /// dispatch's call until it completed). Commands that use the buffer may be encoded and
    /// committed on other threads meanwhile, and none of them fails for the move; no other call
    /// moves it or lets go of its bytes, and no texture is made over its bytes, while this runs.
    fn rebind_no_copy(
        &self,
        bytes: HostRange<'_>,
        deallocator: Option<super::device::NoCopyDeallocator>,
    ) -> Result<(), String> {
        let _ = (bytes, deallocator);
        Err("the platform rebinds no buffer".to_owned())
    }
}

/// A buffer's memory, exported for another API to import: from [`Buffer::export_memory`], an
/// opaque file descriptor of its allocation (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`),
/// which CUDA's `cuImportExternalMemory` takes with the allocation's size and whether it is
/// dedicated, and the buffer's place in it, which `cuExternalMemoryGetMappedBuffer` takes; from
/// [`Buffer::export_dma_buf`], a dma-buf of the allocation, which VA-API imports with the planes'
/// offsets in it (the buffer's own offset added). The buffer must outlive what the importer reads
/// or writes through it.
#[cfg(unix)]
#[derive(Debug)]
pub struct ExportedMemory {
    /// A new descriptor, the caller's: an import that succeeds takes it, else dropping it closes
    /// it.
    pub fd: std::os::fd::OwnedFd,
    /// The allocation's bytes.
    pub allocation_size: u64,
    /// The buffer's first byte in the allocation.
    pub offset: u64,
    /// The buffer's bytes.
    pub length: u64,
    /// Whether the allocation is the buffer's dedicated one, which an import names so.
    pub dedicated: bool,
}

/// No Metal counterpart: the kind of handle an exported surface's memory is shared through
/// ([`GpuDevice::new_exported_surface`](super::device::GpuDevice::new_exported_surface)).
#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceHandle {
    /// An opaque fd, which CUDA imports (NVDEC's landing).
    OpaqueFd,
    /// A dma-buf, which VA-API imports.
    DmaBuf,
}

/// No Metal counterpart: one plane of an exported surface: a 2D texture of one level, slice and
/// sample, of `pixel_format` (a colour format of one texel an element), `width` by `height`, of
/// Metal's `usage`, with `rows_below` rows of room below its own before the next plane's start.
#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfacePlaneDesc {
    /// The plane's `MTLPixelFormat`.
    pub pixel_format: u64,
    /// Its width in texels.
    pub width: u64,
    /// Its height in texels.
    pub height: u64,
    /// Its texture's `MTLTextureUsage`.
    pub usage: u64,
    /// Rows of room below its own, before the next plane's start.
    pub rows_below: u32,
}

/// No Metal counterpart: what
/// [`GpuDevice::new_exported_surface`](super::device::GpuDevice::new_exported_surface) makes: the
/// planes, in their order in memory, shared through `handle`, their one pitch a multiple of
/// `pitch_multiple` and each plane after the first starting on a multiple of `start_multiple`
/// (the other API's rules).
#[cfg(unix)]
#[derive(Clone, Copy, Debug)]
pub struct SurfaceDesc<'a> {
    /// The handle the memory is shared through.
    pub handle: SurfaceHandle,
    /// What the planes' one pitch is a multiple of; at least 1.
    pub pitch_multiple: u64,
    /// What each plane's start after the first is a multiple of; at least 1.
    pub start_multiple: u64,
    /// The planes, in their order in memory.
    pub planes: &'a [SurfacePlaneDesc],
}

/// No Metal counterpart: where a plane of an exported surface lies in its memory, as the other
/// API imports it: its first byte, and the bytes from one of its rows to the next, the surface's
/// one pitch. Plane 0 lies at 0.
#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfacePlaneLayout {
    /// The plane's first byte in the memory.
    pub offset: u64,
    /// Bytes from one of its rows to the next.
    pub row_pitch: u64,
}

/// No Metal counterpart: an exported surface's memory, which its planes' textures lie in
/// ([`ExportedSurface`]), and what another API does with it.
#[cfg(unix)]
pub trait SurfaceMemory: Send + Sync {
    /// The memory as a new descriptor of the surface's handle, with what an import names: the
    /// whole allocation from its first byte ([`ExportedMemory::offset`] 0), and whether it is a
    /// dedicated allocation. `Err` says why not.
    fn export(&self) -> Result<ExportedMemory, String>;
    /// Says that another API wrote the memory and that the caller waited for those writes, as
    /// [`Buffer::external_write_done`] says of a buffer's: the device's later reads of the planes
    /// see them. `Err` says why the device will not.
    fn external_writes_done(&self) -> Result<(), String>;
    /// The kind of handle the memory is shared through.
    fn handle(&self) -> SurfaceHandle;
    /// Whether the planes' textures lie in the shared memory itself; else each is the
    /// platform's copy, which it moves the bytes into and back.
    fn in_place(&self) -> bool;
    /// Each plane's place in the memory, in the description's order.
    fn layout(&self) -> &[SurfacePlaneLayout];
}

/// No Metal counterpart: an exported surface
/// ([`GpuDevice::new_exported_surface`](super::device::GpuDevice::new_exported_surface)): each
/// plane's texture, in the description's order, made with no resource index of the guest's, and
/// the memory they lie in. The textures and the memory go together: a texture made as a view of
/// a plane's holds the memory.
#[cfg(unix)]
pub struct ExportedSurface<P: Platform> {
    /// Each plane's texture, in the description's order.
    pub planes: Vec<P::Texture>,
    /// The memory they lie in.
    pub memory: std::sync::Arc<dyn SurfaceMemory>,
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
    /// sel: didModifyData SPI
    /// The bytes a texture over a buffer lies over changed; side effect only, raising nothing.
    fn did_modify_data(&self);
    /// sel: gpuResourceID
    /// `MTLResourceID._impl` (runtime `{MTLResourceID=Q}`).
    fn gpu_resource_id(&self) -> u64;
    /// sel: height
    fn height(&self) -> u64;
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
pub trait ComputePipeline<P: Platform>: Handle {
    /// sel: gpuResourceID
    /// `MTLResourceID._impl` (runtime `{MTLResourceID=Q}`).
    fn gpu_resource_id(&self) -> u64;
    /// sel: functionHandleWithFunction:
    /// The handle of `function` among the functions the state's kernel links
    /// (`linkedFunctions.functions` and `binaryFunctions`), or `None` where Metal answers nil: a
    /// function it does not link, a private one, the kernel itself.
    fn function_handle(&self, function: &P::Function) -> RaiseResult<Option<P::FunctionHandle>>;
    /// sel: newVisibleFunctionTableWithDescriptor:
    /// A table of `function_count` null entries (the descriptor's `functionCount`), which takes the
    /// state's handles; `None` where the platform makes none.
    fn new_visible_function_table(
        &self,
        function_count: u64,
    ) -> RaiseResult<Option<P::VisibleFunctionTable>>;
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
    /// No Metal counterpart: whether the kernel may write the texture bound at texture slot
    /// `index` (`setTexture:atIndex:`), from the access its compiled interface states for the
    /// slot (it writes it, or reads and writes it), not the argument's `access::` qualifier:
    /// `Some(false)` where it only samples, reads or queries the slot's texture, or uses no such
    /// slot. `None` where the platform knows no slot's access, the default: the caller then goes
    /// by the texture's usage.
    fn texture_slot_writes(&self, _index: u64) -> Option<bool> {
        None
    }
}

/// The function of a render pipeline a texture slot is bound for, as the render encoder's
/// `set{Vertex,Fragment,Tile,Object,Mesh}Texture:atIndex:` names it
/// ([`RenderPipeline::texture_slot_writes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderFunction {
    /// The vertex function: `setVertexTexture:atIndex:`.
    Vertex,
    /// The fragment function: `setFragmentTexture:atIndex:`.
    Fragment,
    /// The tile function: `setTileTexture:atIndex:`.
    Tile,
    /// The object function: `setObjectTexture:atIndex:`.
    Object,
    /// The mesh function: `setMeshTexture:atIndex:`.
    Mesh,
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
    /// sel: maxTotalThreadsPerObjectThreadgroup — 0 for a pipeline with no object stage, as
    /// Metal answers.
    fn max_total_threads_per_object_threadgroup(&self) -> u64;
    /// sel: maxTotalThreadsPerMeshThreadgroup — 0 for a pipeline with no mesh stage.
    fn max_total_threads_per_mesh_threadgroup(&self) -> u64;
    /// sel: objectThreadExecutionWidth — 0 for a pipeline with no object stage.
    fn object_thread_execution_width(&self) -> u64;
    /// sel: meshThreadExecutionWidth — 0 for a pipeline with no mesh stage.
    fn mesh_thread_execution_width(&self) -> u64;
    /// sel: maxTotalThreadgroupsPerMeshGrid — 0 for a pipeline with no mesh stage.
    fn max_total_threadgroups_per_mesh_grid(&self) -> u64;
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
    /// No Metal counterpart: whether `function`'s stage of the pipeline may write the texture
    /// bound at its texture slot `index`, as
    /// [`ComputePipeline::texture_slot_writes`] answers a kernel's: `Some(false)` where it only
    /// samples, reads or queries the slot's texture, or uses no such slot; `None` where the
    /// platform knows no slot's access, the default.
    fn texture_slot_writes(&self, _function: RenderFunction, _index: u64) -> Option<bool> {
        None
    }
}

/// One function-constant value, keyed by its index (`setConstantValue:type:atIndex:`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunctionConstant {
    /// The constant's index.
    pub index: u32,
    /// `MTLDataType` of the value.
    pub data_type: u32,
    /// The value's bytes, packed as its type's lanes are, from the first; the rest are zero.
    pub value: [u8; FunctionConstant::MAX_BYTES],
}

impl FunctionConstant {
    /// The widest value's bytes: a `long4`'s or `ulong4`'s.
    pub const MAX_BYTES: usize = 32;

    /// The packed size of a function constant of `MTLDataType` `data_type` (a `float3` is 12
    /// bytes): `None` for a type no function constant takes, which is a scalar or vector of
    /// `bool`, `char`, `uchar`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `half` or
    /// `float` (Metal's compiler refuses the rest, `bfloat` among them).
    pub fn size_of(data_type: u32) -> Option<usize> {
        Self::lanes(data_type).map(|(lane, lanes)| lane.size() * lanes)
    }

    /// The lanes of a function constant of `MTLDataType` `data_type`: their scalar type, and how
    /// many (a `float3` has 3); `None` for a type no function constant takes ([`Self::size_of`]).
    pub fn lanes(data_type: u32) -> Option<(ConstantLane, usize)> {
        // `MTLDataType` raw values (Metal's MTLDataType.h): each family's scalar, then its two-,
        // three- and four-lane vectors, numbered in a row. The lanes' type, and the scalar's value.
        let (lane, scalar) = match data_type {
            3..=6 => (ConstantLane::Float, 3),
            16..=19 => (ConstantLane::Half, 16),
            29..=32 => (ConstantLane::Int, 29),
            33..=36 => (ConstantLane::UInt, 33),
            37..=40 => (ConstantLane::Short, 37),
            41..=44 => (ConstantLane::UShort, 41),
            45..=48 => (ConstantLane::Char, 45),
            49..=52 => (ConstantLane::UChar, 49),
            53..=56 => (ConstantLane::Bool, 53),
            81..=84 => (ConstantLane::Long, 81),
            85..=88 => (ConstantLane::ULong, 85),
            _ => return None,
        };
        Some((lane, usize::try_from(data_type - scalar + 1).ok()?))
    }

    /// The value's bytes, as many as `data_type` packs: `None` for a type no function constant
    /// takes ([`Self::size_of`]).
    pub fn bytes(&self) -> Option<&[u8]> {
        Self::size_of(self.data_type).and_then(|n| self.value.get(..n))
    }
}

/// The scalar type of a function constant's lanes ([`FunctionConstant::lanes`]), each named as
/// its `MTLDataType` is (`MTLDataTypeUInt`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstantLane {
    /// `float`, 32 bits.
    Float,
    /// `half`, 16 bits.
    Half,
    /// `int`, 32 bits.
    Int,
    /// `uint`, 32 bits.
    UInt,
    /// `short`, 16 bits.
    Short,
    /// `ushort`, 16 bits.
    UShort,
    /// `char`, 8 bits.
    Char,
    /// `uchar`, 8 bits.
    UChar,
    /// `bool`, a byte.
    Bool,
    /// `long`, 64 bits.
    Long,
    /// `ulong`, 64 bits.
    ULong,
}

impl ConstantLane {
    /// A lane's size in bytes.
    pub fn size(self) -> usize {
        match self {
            ConstantLane::Char | ConstantLane::UChar | ConstantLane::Bool => 1,
            ConstantLane::Half | ConstantLane::Short | ConstantLane::UShort => 2,
            ConstantLane::Float | ConstantLane::Int | ConstantLane::UInt => 4,
            ConstantLane::Long | ConstantLane::ULong => 8,
        }
    }

    /// The name of its `MTLDataType` but the prefix (`UInt`); a vector's adds its lanes (`UInt2`).
    pub fn name(self) -> &'static str {
        match self {
            ConstantLane::Float => "Float",
            ConstantLane::Half => "Half",
            ConstantLane::Int => "Int",
            ConstantLane::UInt => "UInt",
            ConstantLane::Short => "Short",
            ConstantLane::UShort => "UShort",
            ConstantLane::Char => "Char",
            ConstantLane::UChar => "UChar",
            ConstantLane::Bool => "Bool",
            ConstantLane::Long => "Long",
            ConstantLane::ULong => "ULong",
        }
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
    /// sel: newFunctionWithDescriptor:error:
    /// The function named `function_name` made with `options` (`MTLFunctionOptions`, as
    /// `MTLFunctionDescriptor.options`) and, where given, `specialized_name`
    /// (`MTLFunctionDescriptor.specializedName`: the name the function answers, and a pipeline
    /// that links it resolves references to it by), specialised by `constants`, each keyed by its
    /// index; with none the descriptor's `constantValues` is nil. Nil for a name the library does
    /// not hold, or a function the platform refuses. Reached with no handler.
    fn new_function_with_options(
        &self,
        on: OnFault,
        function_name: &str,
        constants: &[FunctionConstant],
        options: u64,
        specialized_name: Option<&str>,
    ) -> RaiseResult<Option<P::Function>>;
}

#[cfg(test)]
mod tests {
    use super::FunctionConstant;

    /// A function constant's packed size follows its type's lanes, a `long4` the widest, and a
    /// type no function constant takes has none (the sizes Apple's compiler gives the `CNST` table
    /// of a library of such constants)
    #[test]
    fn a_function_constant_packs_its_lanes() {
        let sizes = [
            (53, 1),  // Bool
            (55, 3),  // Bool3
            (16, 2),  // Half
            (18, 6),  // Half3
            (3, 4),   // Float
            (5, 12),  // Float3
            (6, 16),  // Float4
            (32, 16), // Int4
            (85, 8),  // ULong
            (84, 32), // Long4
        ];
        for (data_type, size) in sizes {
            assert_eq!(
                FunctionConstant::size_of(data_type),
                Some(size),
                "{data_type}"
            );
        }
        for data_type in [0, 1, 7, 57, 60, 121] {
            assert_eq!(FunctionConstant::size_of(data_type), None, "{data_type}");
        }
        let mut c = FunctionConstant {
            index: 1,
            data_type: 5,
            value: [7; FunctionConstant::MAX_BYTES],
        };
        assert_eq!(c.bytes(), Some(&[7u8; 12][..]));
        c.data_type = 121; // BFloat
        assert_eq!(c.bytes(), None);
    }
}
