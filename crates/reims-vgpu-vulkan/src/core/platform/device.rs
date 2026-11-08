//! Platform trait: `GpuDevice`: MTLDevice selectors, SPI included, and the two Metal C SPI
//! functions that take a device.
//!
//! Conventions (architecture 4.2): zero-argument getters return `T` (a platform dies at the
//! send on a raise); every method with arguments and every creator returns `RaiseResult<T>`;
//! creators return `Option` because Metal can return nil whether or not core checks.
//!
//! Creators and encoders hand Metal much of what the guest sends as it is, so Metal's own
//! validation, or the driver, ends the process at the op whose input breaks Metal's rules: a
//! pipeline made from a nil or invalid function or with an unspecialized function constant, a
//! texture descriptor Metal refuses, a second encoder over an open one, an encoder open at the
//! commit, a nil indirect buffer. A method whose input can do so
//! says it on an `aborts:` line.

use std::sync::Arc;

use super::caps::{DeviceCaps, IdentityCaps};
use super::host_range::HostRange;
use super::types::{
    ArgumentLayout, NsErrorText, PixelFormatInfo, SizeAndAlign, TextureLevelInfo, TextureLevelQuery,
};
use super::values::{
    ArgumentDesc, BufferDesc, DepthStencilDesc, FullComputePipelineDesc,
    FullMeshRenderPipelineDesc, FullRenderPipelineDesc, HeapDesc, HeapMade, IcbDesc, PagesInPlace,
    PagesPlane, RenderPipelineDesc, SamplerDesc,
};
use super::{Handle, Platform};
use crate::core::raise::{OnFault, RaiseResult};

/// A deallocator Metal calls once when a no-copy buffer is freed (the
/// `newBufferWithBytesNoCopy:…deallocator:` block). `None` is a nil block.
pub type NoCopyDeallocator = Box<dyn FnOnce() + Send>;

thread_local! {
    /// Whether the platform relayed a gap at a creator this thread called inside the innermost
    /// [`relaying_gaps`] ([`note_relayed_gap`]).
    static RELAYED_GAP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Notes that the creator running on this thread answers nil, or an `NSError` where it has an
/// `error:` out-parameter, for a gap of the platform's own or of its device's, where Metal makes
/// the object: a platform over a translation runtime relays such a refusal and goes on (the
/// Vulkan platform's `answer::relay_gap`). The platform calls it inside the creator, on the
/// creator's thread; Metal's never does. Core reads it around the creators whose nil Metal ends
/// the process at, so it can fault the work instead ([`relaying_gaps`]).
pub fn note_relayed_gap() {
    RELAYED_GAP.with(|r| r.set(true));
}

/// Runs `create`, a call to a platform creator on this thread, and says whether the platform
/// relayed a gap inside it ([`note_relayed_gap`]). A gap relayed inside it is also one of the
/// enclosing call's, where calls nest.
pub fn relaying_gaps<R>(create: impl FnOnce() -> R) -> (R, bool) {
    let outer = RELAYED_GAP.with(|r| r.replace(false));
    let made = create();
    let relayed = RELAYED_GAP.with(|r| r.replace(outer | r.get()));
    (made, relayed)
}

/// `id<MTLDevice>` (`id<MTLDeviceSPI>`).
pub trait GpuDevice<P: Platform>: Handle {
    /// sel: allowLibrariesFromOtherPlatforms SPI
    /// Side effect only; sent once per device, where no handler is live: a raise inside it dies
    /// at the send.
    fn allow_foreign_libraries(&self);
    /// sel: newArgumentEncoderWithArguments: — the layout of the encoder Metal makes of
    /// `arguments`, in their order: its `encodedLength`, its `alignment` and where each argument
    /// lands. Nothing is kept. `None` where Metal makes no encoder (nil, or a raise a handler
    /// catches: no arguments, a data type no argument has).
    fn argument_layout(&self, arguments: &[ArgumentDesc]) -> RaiseResult<Option<ArgumentLayout>>;
    /// sel: -[MTLFunction newArgumentEncoderWithBufferIndex:] — the layout of the encoder Metal
    /// makes of the argument buffer `function` takes at `buffer_index`, its nested argument
    /// buffers' layouts with it (`newArgumentEncoderForBufferAtIndex:`). Nothing is kept. `None`
    /// where Metal makes none (no argument buffer at the index).
    fn function_argument_layout(
        &self,
        function: &P::Function,
        buffer_index: u64,
    ) -> RaiseResult<Option<ArgumentLayout>>;
    /// sel: heapTextureSizeAndAlignWithDescriptor:
    /// The result reaches the guest (opcode 64, info 451/469).
    fn heap_texture_size_and_align(&self, desc: &P::TextureDescriptor)
    -> RaiseResult<SizeAndAlign>;
    /// sel: newBufferWithBytesNoCopy:length:options:deallocator: — (nil
    /// deallocator),. Wraps host memory with no copy: `bytes` is the pointer and the length, and
    /// its constructor's contract keeps it mapped until Metal runs `deallocator`, or, with no
    /// deallocator, for as long as the buffer lives.
    /// Reached both under a handler and with none: `on` says which this call is.
    fn new_buffer_over_host_memory(
        &self,
        on: OnFault,
        bytes: HostRange<'_>,
        options: u64,
        deallocator: Option<NoCopyDeallocator>,
    ) -> RaiseResult<Option<P::Buffer>>;
    /// sel: newBufferWithIOSurface: SPI — The buffer retains
    /// the surface (a guest-visible count).
    fn new_buffer_on_surface(
        &self,
        iosurface: Option<&P::Surface>,
    ) -> RaiseResult<Option<P::Buffer>>;
    /// sel: newBufferWithLength:options:
    /// nil means the resource is refused.
    /// Reached both under a handler and with none: `on` says which this call is.
    fn new_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
    ) -> RaiseResult<Option<P::Buffer>>;
    /// No Metal counterpart: a buffer as [`new_buffer_of_length`](GpuDevice::new_buffer_of_length)'s,
    /// in an allocation of its own whose memory another API imports
    /// ([`Buffer::export_memory`](super::resource::Buffer::export_memory)): the host's video
    /// writes its pictures into the IOSurface device caches so, on its own GPU work. `Ok(None)`
    /// where the platform makes none (the default, and a device that exports no buffer's memory),
    /// and the caller makes the buffer as any other. Reached both under a handler and with none:
    /// `on` says which this call is.
    #[cfg(unix)]
    fn new_exportable_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
    ) -> RaiseResult<Option<P::Buffer>> {
        let _ = (on, length, options);
        Ok(None)
    }
    /// No Metal counterpart: a buffer as [`new_buffer_of_length`](GpuDevice::new_buffer_of_length)'s,
    /// that declares the bytes between the rows (and slices) of a texture over it unobserved: a
    /// texture whose rows lie with bytes of the buffer between them may hold those bytes in its
    /// memory, and change them, and so lies over the buffer's memory where the platform would
    /// otherwise make it a copy of its own. The graphics lane makes a device cache so where the
    /// kext's ask gave its surface's layout, as its copies into the pages take the surface's
    /// texels alone (`decode::surface_cache`). Where `aligned_byte` is given, that byte of the
    /// buffer lies on a multiple of 256 bytes of its memory, so that a texture whose texels start
    /// there, or a multiple of 256 bytes past it, lies in its memory as one over its first byte
    /// does: for a cache whose surface's planes start off such a multiple, as CoreVideo's start
    /// 32 bytes into their surface. Its length and the answers of its own are
    /// `new_buffer_of_length`'s. The default, for a platform with neither declaration, makes the
    /// buffer as `new_buffer_of_length` does. Reached both under a handler and with none: `on`
    /// says which this call is.
    fn new_buffer_of_length_padding_unobserved(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
        aligned_byte: Option<u64>,
    ) -> RaiseResult<Option<P::Buffer>> {
        let _ = aligned_byte;
        self.new_buffer_of_length(on, length, options)
    }
    /// No Metal counterpart: a buffer as
    /// [`new_buffer_of_length_padding_unobserved`](GpuDevice::new_buffer_of_length_padding_unobserved)'s,
    /// whose memory the platform also exports as a dma-buf
    /// ([`Buffer::export_dma_buf`](super::resource::Buffer::export_dma_buf)): the host's video
    /// decodes its pictures into the IOSurface device caches so, its hardware writing the cache's
    /// memory. `Ok(None)` where the platform makes none (the default, and a device that exports no
    /// dma-buf of a buffer's memory), and the caller makes the buffer as before. Reached both under
    /// a handler and with none: `on` says which this call is.
    #[cfg(unix)]
    fn new_dma_buf_buffer_of_length_padding_unobserved(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
        aligned_byte: Option<u64>,
    ) -> RaiseResult<Option<P::Buffer>> {
        let _ = (on, length, options, aligned_byte);
        Ok(None)
    }
    /// sel: newCommandQueue
    /// Never nil-checked. No path reaches it under a handler: a raise inside it dies at the send.
    fn new_command_queue(&self) -> Option<P::CommandQueue>;
    /// No Metal counterpart: a queue as [`new_command_queue`](GpuDevice::new_command_queue)'s,
    /// which shares the GPU queue it submits to with no other command queue for its life, where
    /// the platform would otherwise share one between command queues: the queues on a present's
    /// path are made so, lest another queue's long command buffer hold theirs back: WindowServer's
    /// in the graphics lane (`RGFX_QUEUE_UNSHARED`), so that no other process's work queues on
    /// the GPU ahead of its frames, and the host's own on that path (the device's blit and exec
    /// queues, the process's frame queue, a graphics display's scanout draw and the IOSurface
    /// device caches' transfer queue). A platform that shares no GPU queue between command queues
    /// answers `new_command_queue`'s queue. Never nil-checked; a raise inside it dies at the
    /// call, as one inside `newCommandQueue` does.
    fn new_command_queue_unshared(&self) -> Option<P::CommandQueue>;
    /// No Metal counterpart: whether the device copies between its buffers and the guest's RAM
    /// by the RAM's host addresses ([`BlitEncoder::copy_from_guest_ram`],
    /// [`BlitEncoder::copy_to_guest_ram`]), so a copy of guest pages needs no view of them.
    /// False where the platform has no such copy; where true, a copy may still be refused, and
    /// the caller copies another way.
    ///
    /// [`BlitEncoder::copy_from_guest_ram`]: super::blit::BlitEncoder::copy_from_guest_ram
    /// [`BlitEncoder::copy_to_guest_ram`]: super::blit::BlitEncoder::copy_to_guest_ram
    fn copies_guest_ram(&self) -> bool;
    /// No Metal counterpart: whether textures over an IOSurface's own pages, which `pages`, a
    /// no-copy buffer of this device over all of them from the surface's first byte, lies over,
    /// its planes laid out as `planes` gives, lie in place in those pages for every access a
    /// texture over the surface may take (sampled or read, rendered, written by a shader), as the
    /// platform makes a texture over such a buffer; where one would be the platform's copy of
    /// them instead (a shadow), why. Asked only of a device whose GPU shares the CPU's memory,
    /// where the graphics lane gives a surface a device cache only where its textures cannot lie
    /// in place. The default is yes.
    fn surface_pages_in_place(&self, pages: &P::Buffer, planes: &[PagesPlane]) -> PagesInPlace {
        let _ = (pages, planes);
        PagesInPlace::Yes
    }
    /// No Metal counterpart: why textures over an IOSurface's own pages, laid out as `planes`
    /// from a page's first byte, would lie in place over no buffer of them this device makes,
    /// asked before any is made: whatever the platform imports them as,
    /// [`Self::surface_pages_in_place`] would answer no over a buffer over a view of them.
    /// `pages` are the surface's pages as host addresses in the host's mapping of the guest's RAM,
    /// in the surface's order, as runs: what a view of them would map. `None` where that answer
    /// may be other than no, and so is asked over the view. Asked only of a device whose GPU
    /// shares the CPU's memory, as that is. The default is `None`.
    fn surface_pages_never_in_place(
        &self,
        planes: &[PagesPlane],
        pages: &[std::ops::Range<u64>],
    ) -> Option<String> {
        let _ = (planes, pages);
        None
    }
    /// No Metal counterpart: the bytes the device aligns the rows of a linear BGRA8 image to, to
    /// make one over host pages in place there, as the platform answers; 0 where it does not. On
    /// a device whose GPU shares the CPU's memory the graphics lane has the guest align its
    /// surfaces' rows to it. The default is 0.
    fn linear_row_alignment(&self) -> u64 {
        0
    }
    /// sel: newComputePipelineStateWithFunction:error:
    /// The function may be nil (a library with no such function); core asserts on nil in the
    /// service. No path reaches it under a handler: a raise inside it dies at the send.
    fn new_compute_pipeline_state_for_function(
        &self,
        compute_function: Option<&P::Function>,
    ) -> Result<Option<P::ComputePipeline>, NsErrorText>;
    /// sel: newLibraryWithSource:options:error:
    /// The library of our own shaders: the display's present functions and the XR10 blit
    /// kernels, from the Metal Shading Language the platform embeds. It is compiled once per
    /// Metal device, and each call answers a new reference to it; `Err` is the compiler's
    /// `NSError`. No path reaches it under a handler: a raise inside it dies at the send.
    fn new_shader_library(&self) -> Result<Option<P::Library>, NsErrorText>;
    /// sel: newEvent
    /// Reached both under a handler and with none: `on` says which this call is.
    fn new_event(&self, on: OnFault) -> RaiseResult<Option<P::Event>>;
    /// sel: newFence
    fn new_fence(&self) -> RaiseResult<Option<P::Fence>>;
    /// sel: newLibraryWithData:error: — `data` becomes a
    /// `dispatch_data_t`; `Err` is the `NSError`, whose description core logs.
    fn new_library(&self, data: &[u8]) -> RaiseResult<Result<Option<P::Library>, NsErrorText>>;
    /// sel: newRasterizationRateMapWithDescriptor:
    /// aborts: the third layer on the suite's host (render_ratemap_three_layers).
    fn new_rate_map(&self, descriptor: &P::RateMapDescriptor) -> RaiseResult<Option<P::RateMap>>;
    /// sel: newSharedEvent — No path reaches it under a handler: a raise inside it dies at the
    /// send.
    fn new_shared_event(&self) -> Option<P::SharedEvent>;
    /// sel: newSharedTextureWithDescriptor:
    fn new_shared_texture(
        &self,
        descriptor: &P::TextureDescriptor,
    ) -> RaiseResult<Option<P::Texture>>;
    /// sel: newSharedTextureWithHandle:withResourceIndex: SPI
    /// The index is the guest's placement word, unverified.
    fn open_shared_texture(
        &self,
        handle: Option<&P::SharedTextureHandle>,
        resource_index: u64,
    ) -> RaiseResult<Option<P::Texture>>;
    /// sel: newTextureWithDescriptor:
    /// Reached both under a handler and with none: `on` says which this call is.
    fn new_texture(
        &self,
        on: OnFault,
        descriptor: &P::TextureDescriptor,
    ) -> RaiseResult<Option<P::Texture>>;
    /// sel: newTextureWithDescriptor:iosurface:plane:
    /// The texture retains the surface (guest-visible count). aborts: Metal's validation on some
    /// descriptors (measured).
    fn new_texture_on_surface(
        &self,
        descriptor: &P::TextureDescriptor,
        iosurface: Option<&P::Surface>,
        plane: u64,
    ) -> RaiseResult<Option<P::Texture>>;
    /// sel: reserveResourceIndicesForResourceType:indices:indexCount: SPI
    /// (types 0, 1 and 5). `indexCount` is `indices.len()`.
    fn reserve_resource_indices(&self, resource_type: u64, indices: &[u64]) -> RaiseResult<()>;
    /// C SPI `MTLPixelFormatGetInfoForDevice(device, format)`
    /// 0x38 bytes, copied out whole.
    fn pixel_format_info(&self, format: u64) -> PixelFormatInfo;
    /// C SPI `MTLGetTextureLevelInfoForDeviceWithOptions(device, …)`
    /// 0x68 bytes.
    fn texture_level_info(&self, q: &TextureLevelQuery) -> TextureLevelInfo;

    // Capabilities and creators from values.
    /// The device's capabilities: read in full on the first call for this device, then shared.
    fn caps(&self) -> Arc<DeviceCaps>;
    /// The device's identity and the capability answers beyond [`caps`](GpuDevice::caps): every
    /// family and feature set of the SDK, and the identity and capability getters. Read on each
    /// call; only the graphics lane asks it, once per device.
    fn identity_caps(&self) -> IdentityCaps;
    /// Whether the device makes render pipelines with the private state Apple's OpenGL renderer
    /// sets ([`FullRenderPipelineDesc::private_state`]): on the Vulkan platform, always. The
    /// graphics lane offers `PRIVATE_RENDER_STATE` where it does.
    fn takes_private_render_state(&self) -> bool;
    /// The bytes of the GPU's own memory the process may hold, as the platform answers at the
    /// call: Metal's `recommendedMaxWorkingSetSize`; on the Vulkan platform the budget of the
    /// device-local heaps (`VK_EXT_memory_budget`) where the device has the extension, else their
    /// size. The graphics lane's deferred surfaces take a quarter of it, read once at its device's
    /// start.
    fn own_memory_budget(&self) -> u64;
    /// Whether the device makes compute pipelines of OpenCL mode
    /// ([`FullComputePipelineDesc::open_cl_mode`]): on the Vulkan platform, whose runtime links
    /// such a kernel as Metal does. The graphics lane offers `OPENCL_MODE` where it does.
    fn makes_opencl_mode_pipelines(&self) -> bool;
    /// sel: newBufferWithDescriptor: SPI — from a private `MTLBufferDescriptor` made from
    /// `desc`.
    fn new_buffer(&self, desc: BufferDesc<'_>) -> RaiseResult<Option<P::Buffer>>;
    /// sel: newHeapWithDescriptor: — from a heap descriptor made from `desc`, and that
    /// descriptor's `resourceOptions` read after the creation.
    fn new_heap(&self, desc: &HeapDesc<'_, P>) -> RaiseResult<HeapMade<P>>;
    /// sel: newHeapWithDescriptor: over `bytes` in place — a Shared heap of `desc`, whose
    /// `address_ranges` the platform ignores, with `bytes` as its memory, zero-copy, as Metal
    /// makes one over one address range (`setAddressRanges:` SPI): a resource placed at an offset
    /// lies in `bytes` from that offset, and the CPU's and the GPU's writes there are each
    /// other's. Unlike a heap over address ranges it never detaches. `bytes` starts on a page and
    /// covers the heap's size rounded up to Metal's 16 KiB page, and its constructor's contract
    /// keeps it mapped for as long as the heap lives. Nil where the platform makes no heap over
    /// it, as for `new_heap`.
    fn new_heap_over_host_memory(
        &self,
        bytes: HostRange<'_>,
        desc: &HeapDesc<'_, P>,
    ) -> RaiseResult<HeapMade<P>>;
    /// sel: newSamplerStateWithDescriptor: — from a sampler descriptor made from `desc`.
    fn new_sampler_state(&self, desc: &SamplerDesc) -> RaiseResult<Option<P::SamplerState>>;
    /// sel: newDepthStencilStateWithDescriptor: — from a depth-stencil descriptor made from
    /// `desc`.
    fn new_depth_stencil_state(
        &self,
        desc: &DepthStencilDesc,
    ) -> RaiseResult<Option<P::DepthStencilState>>;
    /// sel: newIndirectCommandBufferWithDescriptor:maxCommandCount:options: — from an ICB
    /// descriptor made from `desc`.
    fn new_indirect_command_buffer(
        &self,
        desc: &IcbDesc,
        max_count: u64,
        options: u64,
    ) -> RaiseResult<Option<P::Icb>>;
    /// sel: newRenderPipelineStateWithDescriptor:error: — from a render pipeline descriptor
    /// made from `desc` (a raise from its colour attachment lookup comes first). No path reaches
    /// it under a handler: either raise dies at its send.
    fn new_render_pipeline_state(
        &self,
        desc: &RenderPipelineDesc<'_, P>,
    ) -> Result<Option<P::RenderPipeline>, NsErrorText>;
    /// sel: newRenderPipelineStateWithDescriptor:error: — from a render pipeline descriptor made
    /// from the whole value `desc`, the graphics lane's. A raise inside it, and a refusal Metal
    /// answers with no `NSError`, die at the send: the lane calls it inside its refusal scope.
    fn new_render_pipeline_state_full(
        &self,
        desc: &FullRenderPipelineDesc<'_, P>,
    ) -> Result<Option<P::RenderPipeline>, NsErrorText>;
    /// sel: newRenderPipelineStateWithMeshDescriptor:options:reflection:error: — from a mesh
    /// render pipeline descriptor made from the whole value `desc`, the graphics lane's, with no
    /// options and a nil reflection. A raise inside it, and a refusal Metal answers with no
    /// `NSError`, die at the send: the lane calls it inside its refusal scope. A platform whose
    /// device makes no mesh pipeline answers the `NSError` Metal would.
    fn new_mesh_render_pipeline_state_full(
        &self,
        desc: &FullMeshRenderPipelineDesc<'_, P>,
    ) -> Result<Option<P::RenderPipeline>, NsErrorText>;
    /// sel: newComputePipelineStateWithDescriptor:options:reflection:error: — from a compute
    /// pipeline descriptor made from the whole value `desc`, the graphics lane's, with no options
    /// and a nil reflection. A raise inside it, and a refusal Metal answers with no `NSError`, die
    /// at the send: the lane calls it inside its refusal scope.
    fn new_compute_pipeline_state_full(
        &self,
        desc: &FullComputePipelineDesc<'_, P>,
    ) -> Result<Option<P::ComputePipeline>, NsErrorText>;
}
