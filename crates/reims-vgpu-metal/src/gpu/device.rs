//! `id<MTLDevice>` (with its SPI): core's `GpuDevice` in typed calls.
//!
//! - **The capabilities are read once per device** ([`GpuDevice::caps`]): every query is sent
//!   once, in no order core observes, and the answers are kept by registry ID. `gpuCoreCount` is
//!   called only after `respondsToSelector:` answers YES.
//! - **A creator returns what Metal made**, at `+1` (`new…`), nil as `None`, and an `error:`
//!   out-parameter's `NSError` as `Err`. A raise inside a creator is the caller's to catch; its
//!   send's policy says where core sends it: `Caught` for one sent only under a handler (the
//!   resource manager's creators, the capability queries), `Fatal` for one sent only with
//!   none (the device's and a FIFO's setup, the GPU task's handshake and jobs, a display's
//!   pipelines), and `On` for one sent both ways, whose caller says by its `OnFault` which way
//!   each call is. A getter's raise ends the process at the call.
//! - What core's trait hides is made here: the `dispatch_data_t`s and their destructors, the
//!   no-copy deallocator block, the deserialization context.
//! - **Our shader library** is the one creator whose result is kept: its source is compiled once
//!   per device ([`super::shaders`]).

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::{Arc, LazyLock};

use block2::{Block, RcBlock};
use dispatch2::{DispatchData, DispatchRetained, dispatch_block_t};
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObjectProtocol, ProtocolObject, Sel};
use objc2::{AnyThread as _, Message, sel};
use objc2_foundation::{NSData, NSError};
use objc2_metal::{MTLDevice, MTLGPUFamily, MTLPixelFormat, MTLResourceOptions};
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::HostRange;
use reims_vgpu_core::platform::caps::{
    COUNTER_SAMPLING_POINTS, DeviceCaps, FAMILIES, FEATURE_SETS, Flags, FormatAnswers,
    HEAP_QUERY_OPTIONS, IdentityCaps, IdentityFlags, IdentitySizes, LARGEST_COUNT_ASKED,
    LINEAR_ALIGNMENT_FORMAT, Limits, SAMPLE_COUNTS, SDK_FAMILIES, SDK_PIXEL_FORMATS,
    SPARSE_PAGE_SIZES, SPARSE_SAMPLE_COUNTS, TEXTURE_BUFFER_FORMATS, WRITE_ROUNDING_MODES,
};
use reims_vgpu_core::platform::device::{FunctionResolver, GpuDevice, NoCopyDeallocator};
use reims_vgpu_core::platform::types::{
    ArgumentLayout, MtlSize, NsErrorText, PixelFormatInfo, ResourceAddressRange, SizeAndAlign,
    TextureLevelInfo, TextureLevelQuery,
};
use reims_vgpu_core::platform::values::{
    ArgumentDesc, BufferDesc, DepthStencilDesc, FullComputePipelineDesc, FullRenderPipelineDesc,
    HeapDesc, HeapMade, IcbDesc, RenderPipelineDesc, SamplerDesc,
};
use reims_vgpu_core::raise::death::DeathKind;
use reims_vgpu_core::raise::{Kept, keep_forever};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::rt::Hold;
use reims_vgpu_core::sync::Mutex;

use super::rate_map::{MetalRateMap, MetalRateMapDescriptor};
use super::resource::{
    MetalBuffer, MetalComputePipeline, MetalDepthStencilState, MetalFence, MetalFunction,
    MetalHeap, MetalLibrary, MetalRenderPipeline, MetalSamplerState, MetalSharedTextureHandle,
    MetalTexture, created,
};
use crate::gpu::command::MetalCommandQueue;
use crate::gpu::desc::{
    self, MetalBufferDescriptor, MetalComputePipelineDescriptor, MetalDepthStencilDescriptor,
    MetalHeapDescriptor, MetalIcbDescriptor, MetalRenderPipelineDescriptor, MetalSamplerDescriptor,
    MetalTextureDescriptor, MetalTileRenderPipelineDescriptor,
};
use crate::gpu::event::{MetalEvent, MetalSharedEvent, MetalSharedEventHandle};
use crate::gpu::icb::MetalIcb;
use crate::msg::encode::FromMtl;
use crate::msg::{self, Policy};
use crate::os::iosurface::MetalSurface;
use crate::platform::{MetalPlatform, Obj, ns_error_text};
use crate::spi::classes::DeserializationContext;
use crate::spi::functions;
use crate::spi::methods::{Deallocator, DeviceSpi};

/// `id<MTLDevice>`.
pub type MetalDevice = Obj<ProtocolObject<dyn MTLDevice>>;

unsafe extern "C" {
    /// `dispatch_data_t dispatch_data_create(const void *buffer, size_t size, dispatch_queue_t
    /// queue, dispatch_block_t destructor)`: +1. A block destructor with a NULL `queue` is
    /// submitted with `dispatch_async` to the default-QoS root queue when the data's last
    /// reference goes (`_dispatch_data_destroy_buffer`); an empty buffer returns
    /// `dispatch_data_empty` and submits the destructor at once.
    fn dispatch_data_create(
        buffer: *const c_void,
        size: usize,
        queue: *const c_void,
        destructor: dispatch_block_t,
    ) -> *mut DispatchData;
}

/// `dispatch_data_create(buffer, size, nullptr, ^{ … })`: the data holds `buffer` as it is (no
/// copy), and libdispatch copies `destructor` (`Block_copy`) and runs it on a
/// `com.apple.root.default-qos` GCD worker, at the releaser's QoS, after the data's last release
/// has returned.
///
/// # Safety
///
/// `buffer` is `size` readable bytes that outlive the data (until `destructor` runs); NULL only
/// with `size` 0.
unsafe fn data_with_block_destructor(
    buffer: *const c_void,
    size: usize,
    destructor: &Block<dyn Fn()>,
) -> DispatchRetained<DispatchData> {
    let destructor: dispatch_block_t = std::ptr::from_ref(destructor).cast_mut();
    // SAFETY: the caller's buffer; a NULL queue is the default root queue; libdispatch copies
    // the block.
    let data = unsafe { dispatch_data_create(buffer, size, std::ptr::null(), destructor) };
    match std::ptr::NonNull::new(data) {
        // SAFETY: `dispatch_data_create` returned +1 (or `dispatch_data_empty`, a global object
        // whose retain and release do nothing).
        Some(data) => unsafe { DispatchRetained::from_raw(data) },
        None => {
            reims_vgpu_core::raise::death::fatal(reims_vgpu_core::raise::death::DeathId::Platform {
                kind: DeathKind::Abort,
                call: "dispatch_data_create",
            })
        }
    }
}

/// A `malloc`ed block of `size` bytes (at least `bytes.len()`) starting with `bytes`; the caller
/// owns it.
fn malloc_copy(bytes: &[u8], size: usize) -> std::ptr::NonNull<c_void> {
    // SAFETY: `malloc` of any size; macOS returns a unique pointer for 0.
    let buffer = unsafe { libc::malloc(size.max(bytes.len())) };
    let Some(buffer) = std::ptr::NonNull::new(buffer) else {
        reims_vgpu_core::raise::death::fatal(reims_vgpu_core::raise::death::DeathId::Platform {
            kind: DeathKind::Abort,
            call: "malloc",
        })
    };
    // SAFETY: `buffer` holds at least `bytes.len()` bytes, fresh, so the ranges do not overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.as_ptr().cast::<u8>(), bytes.len());
    }
    buffer
}

/// The library's IR: a `malloc`ed copy of the bytes in an `NSData` made with `freeWhenDone:YES`,
/// whose end frees it. One reference, the caller's, and none in a pool, so the destructor block's
/// release (see [`ir_dispatch_data`]) is always the last.
fn ir_ns_data(ir: &[u8]) -> Retained<NSData> {
    let buffer = malloc_copy(ir, ir.len());
    let data = msg::value("alloc", NSData::alloc);
    // SAFETY: a fresh `malloc` block of `ir.len()` bytes, which the data owns and frees.
    msg::value("initWithBytesNoCopy:length:freeWhenDone:", || unsafe {
        NSData::initWithBytesNoCopy_length_freeWhenDone(data, buffer, ir.len(), true)
    })
}

/// A `dispatch_data_t` over the `NSData`'s own bytes, whose destructor block holds the `NSData`:
/// the IR is freed (the `NSData`'s end) on the default-QoS root worker that runs the block, when
/// Metal lets go of the data ([`data_with_block_destructor`]).
fn ir_dispatch_data(data: Retained<NSData>) -> DispatchRetained<DispatchData> {
    // SAFETY: `-bytes` and `-length` of our own NSData, never mutated.
    let region = msg::value("bytes", || unsafe { data.as_bytes_unchecked() });
    let (bytes, length) = (region.as_ptr().cast::<c_void>(), region.len());
    // The block's copy of `data` goes with the block.
    let destructor = RcBlock::new(move || {
        let _ = &data;
    });
    // SAFETY: the bytes are the NSData's, which the block keeps alive until libdispatch runs
    // and releases it, after the data's last release.
    unsafe { data_with_block_destructor(bytes, length, &destructor) }
}

block2::global_block! {
    /// The tile form's destructor, `^{}`: a global block, which libdispatch's `Block_copy` returns
    /// as it is.
    static TILE_DESTRUCTOR = || {}
}

/// The tile descriptor at the start of a `malloc`ed buffer of `buffer_length` bytes (the
/// cursor's remaining bytes), in a `dispatch_data_t` spanning the descriptor alone whose
/// destructor is an empty block: nothing ever frees the buffer.
fn tile_descriptor_data(descriptor: &[u8], buffer_length: u64) -> DispatchRetained<DispatchData> {
    let size = usize::try_from(buffer_length).unwrap_or(usize::MAX);
    let buffer = malloc_copy(descriptor, size);
    // SAFETY: the buffer is never freed, so it outlives the data.
    unsafe { data_with_block_destructor(buffer.as_ptr(), descriptor.len(), &TILE_DESTRUCTOR) }
}

/// A creator with an `error:` out-parameter, which `call` passes where the method takes it:
/// `Ok(Some)` for an object (whatever the error), `Err(NsErrorText)` for nil with an error,
/// `Ok(None)` for nil with a nil error. `policy` is the creator's own: each creator sent through
/// it is reached only under a handler ([`Policy::Caught`]); one no path reaches under a handler
/// goes through [`create_or_die`].
pub(super) fn create_or_error<T: Message>(
    policy: Policy,
    sel: &'static str,
    call: impl FnOnce(*mut *mut NSError) -> Option<Retained<T>>,
) -> RaiseResult<Result<Option<Obj<T>>, NsErrorText>> {
    let mut error: *mut NSError = null_mut();
    let out = &raw mut error;
    let made = msg::send(sel, policy, || call(out))?;
    // SAFETY: Metal wrote nil or an autoreleased NSError to the out-parameter.
    Ok(unsafe { made_or_error(made, error) })
}

/// [`create_or_error`] for a creator no path reaches under a handler: a raise inside it dies at
/// the send ([`msg::get`]).
pub(super) fn create_or_die<T: Message>(
    sel: &'static str,
    call: impl FnOnce(*mut *mut NSError) -> Option<Retained<T>>,
) -> Result<Option<Obj<T>>, NsErrorText> {
    let mut error: *mut NSError = null_mut();
    let out = &raw mut error;
    let made = msg::get(sel, || call(out));
    // SAFETY: Metal wrote nil or an autoreleased NSError to the out-parameter.
    unsafe { made_or_error(made, error) }
}

/// A creator's answer: the object it made, else the `NSError` it wrote, else nil.
///
/// # Safety
///
/// `error` is nil or an autoreleased `NSError` the creator wrote.
unsafe fn made_or_error<T: Message>(
    made: Option<Retained<T>>,
    error: *mut NSError,
) -> Result<Option<Obj<T>>, NsErrorText> {
    match made {
        Some(o) => Ok(Some(Obj::from_retained(o))),
        // SAFETY: the caller's.
        None => match unsafe { ns_error_text(error) } {
            Some(e) => Err(e),
            None => Ok(None),
        },
    }
}

/// The three `new…PipelineDescriptorWithSerializedData:deserializationContext:` SPI calls.
///
/// The caller makes the data: for the compute and render forms a copy that libdispatch frees when
/// the data goes, and for the tile form [`tile_descriptor_data`]. A fresh context holds `resolver`
/// for the call; Metal calls it synchronously, on this thread, under `hold`, and its raise comes
/// back as `Err`. A raise inside the call is the resolver's own throw when it made one (carried
/// through Metal to the caller unchanged, and thrown again once Metal has unwound), otherwise
/// Metal's.
///
/// Metal lets the resolver's throw through to the call: measured on the M3 Max, macOS 27.0.1
/// (26A434), with a throw at each resolver call in turn of nine descriptor shapes (compute,
/// render and tile; with linked, private, binary and grouped linked functions, which Metal does
/// not resolve through the context), 5 runs, every throw reached the call, and Metal made
/// nothing. Were Metal to catch one itself and answer, the throw would be lost, as the exception
/// would be to Apple's framework: the call answers what Metal made, and what the raise carries
/// lands in `hold`'s holder, where a catch would land it.
///
/// A death or a stop inside the resolver, in an isolated task's code, is a Rust unwind to the
/// task's run, which never crosses Metal's frames: the context keeps it and raises through Metal
/// in its place, and once Metal has returned the unwind goes on from here, so the task ends at
/// the same op as it would with no Metal frame between.
///
/// Every caller is a pipeline creation, which runs only under a handler (its creator's own sends
/// are `Caught`): either raise is `Err`, the caller's to catch.
fn deserialize<T: Message>(
    hold: Hold<'_>,
    d: &MetalDevice,
    sel: &'static str,
    data: DispatchRetained<DispatchData>,
    resolver: FunctionResolver<'_, MetalPlatform>,
    call: impl FnOnce(&ProtocolObject<dyn MTLDevice>, &DispatchData, &AnyObject) -> Option<Retained<T>>,
) -> RaiseResult<Option<Obj<T>>> {
    let context = msg::value("init", DeserializationContext::new);
    let (device, data, any) = (d.get(), &*data, context.as_any());
    // Metal's own assertions on a malformed blob end the process inside the call.
    let made = context.with_resolver(hold, resolver, || {
        msg::catch(sel, || call(device, data, any))
    });
    if let Some(unwound) = context.take_unwound() {
        std::panic::resume_unwind(unwound)
    }
    match made {
        Ok(made) => {
            if let Some(lost) = context.take_thrown() {
                hold.land(lost);
            }
            Ok(created(made))
        }
        Err(caught) => match context.take_thrown() {
            Some(r) => Err(r),
            None => caught.decide(Policy::Caught),
        },
    }
}

impl GpuDevice<MetalPlatform> for MetalDevice {
    fn allow_foreign_libraries(&self) {
        let d = self.get();
        msg::get("allowLibrariesFromOtherPlatforms", || {
            d.allowLibrariesFromOtherPlatforms()
        });
    }

    /// Read off the encoder Metal makes ([`super::argument`]).
    fn argument_layout(&self, arguments: &[ArgumentDesc]) -> RaiseResult<Option<ArgumentLayout>> {
        super::argument::of_descriptors(self.get(), arguments)
    }

    /// Read off the encoder Metal makes, and its reflection ([`super::argument`]).
    fn function_argument_layout(
        &self,
        function: &MetalFunction,
        buffer_index: u64,
    ) -> RaiseResult<Option<ArgumentLayout>> {
        super::argument::of_function(function, buffer_index)
    }

    fn heap_texture_size_and_align(
        &self,
        desc: &MetalTextureDescriptor,
    ) -> RaiseResult<SizeAndAlign> {
        let d = self.get();
        msg::send(
            "heapTextureSizeAndAlignWithDescriptor:",
            Policy::Caught,
            || d.heapTextureSizeAndAlignWithDescriptor(desc),
        )
        .map(SizeAndAlign::from_mtl)
    }

    /// The deallocator becomes a block that calls it at most once, when Metal frees the buffer;
    /// `None` passes a nil block.
    ///
    /// When Metal refuses the buffer (nil, or a raise a handler catches) it never calls the block,
    /// and the deallocator is kept for good with its captures (the caller's mapping stays mapped:
    /// its address space stays taken).
    fn new_buffer_over_host_memory(
        &self,
        on: OnFault,
        bytes: HostRange<'_>,
        options: u64,
        deallocator: Option<NoCopyDeallocator>,
    ) -> RaiseResult<Option<MetalBuffer>> {
        let slot = deallocator.map(|d| Arc::new(Mutex::new(Some(d))));
        let block = slot.clone().map(|slot| {
            RcBlock::new(move |_pointer: *mut c_void, _length: u64| {
                // A foreign callout: Metal runs it on whichever thread lets the buffer go last,
                // its completion thread among them. Marked ours, so a panic in it aborts before
                // it unwinds into Metal (6.5); the thread's own mark comes back after.
                let _ours = reims_vgpu_core::rt::enter_ours();
                let d = slot.lock().take();
                if let Some(d) = d {
                    d();
                }
            })
        });
        let block_ref: Option<&Deallocator> = block.as_deref();
        let (d, pointer, length) = (self.get(), bytes.ptr().as_ptr::<c_void>(), bytes.len());
        // SAFETY: `HostRange::new`'s contract keeps the memory mapped until the deallocator runs
        // (or for the buffer's life with none); Metal copies the block.
        let result = msg::send(
            "newBufferWithBytesNoCopy:length:options:deallocator:",
            Policy::On(on),
            || unsafe {
                DeviceSpi::newBufferWithBytesNoCopy_length_options_deallocator(
                    d, pointer, length, options, block_ref,
                )
            },
        )
        .map(created);
        if !matches!(result, Ok(Some(_)))
            && let Some(d) = slot.and_then(|slot| slot.lock().take())
        {
            keep_forever(d, Kept::AddressSpace);
        }
        result
    }

    /// The buffer retains the surface.
    fn new_buffer_on_surface(
        &self,
        iosurface: Option<&MetalSurface>,
    ) -> RaiseResult<Option<MetalBuffer>> {
        let (d, surface) = (self.get(), iosurface.map(MetalSurface::as_iosurface_ref));
        msg::send("newBufferWithIOSurface:", Policy::Caught, || {
            d.newBufferWithIOSurface(surface)
        })
        .map(created)
    }

    fn new_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
    ) -> RaiseResult<Option<MetalBuffer>> {
        let d = self.get();
        let (length, options) = (length.to_usize(), MTLResourceOptions(options.to_usize()));
        msg::send("newBufferWithLength:options:", Policy::On(on), || {
            d.newBufferWithLength_options(length, options)
        })
        .map(created)
    }

    fn new_command_queue(&self) -> Option<MetalCommandQueue> {
        let d = self.get();
        created(msg::get("newCommandQueue", || d.newCommandQueue()))
    }

    /// `newCommandQueue`'s queue: Metal has no call that keeps a queue from sharing the GPU's
    /// queues, so the platform makes the queue any create makes.
    fn new_command_queue_unshared(&self) -> Option<MetalCommandQueue> {
        self.new_command_queue()
    }

    fn deserialize_compute_pipeline_descriptor(
        &self,
        hold: Hold<'_>,
        data: &[u8],
        context: FunctionResolver<'_, MetalPlatform>,
    ) -> RaiseResult<Option<MetalComputePipelineDescriptor>> {
        let sel = "newComputePipelineDescriptorWithSerializedData:deserializationContext:";
        deserialize(
            hold,
            self,
            sel,
            DispatchData::from_bytes(data),
            context,
            |d, data, context| {
                d.newComputePipelineDescriptorWithSerializedData_deserializationContext(
                    data, context,
                )
            },
        )
    }

    /// Reflection is nil; a nil descriptor is passed as it is.
    fn new_compute_pipeline_state(
        &self,
        descriptor: Option<&MetalComputePipelineDescriptor>,
        options: u64,
    ) -> RaiseResult<Result<Option<MetalComputePipeline>, NsErrorText>> {
        let (d, descriptor) = (self.get(), descriptor.map(Obj::get));
        // SAFETY: `create_or_error`'s out-parameter.
        create_or_error(
            Policy::Caught,
            "newComputePipelineStateWithDescriptor:options:reflection:error:",
            |error| unsafe {
                DeviceSpi::newComputePipelineStateWithDescriptor_options_reflection_error(
                    d, descriptor, options, error,
                )
            },
        )
    }

    fn new_compute_pipeline_state_for_function(
        &self,
        compute_function: Option<&MetalFunction>,
    ) -> Result<Option<MetalComputePipeline>, NsErrorText> {
        let (d, function) = (self.get(), compute_function.map(Obj::get));
        // SAFETY: `create_or_die`'s out-parameter.
        create_or_die(
            "newComputePipelineStateWithFunction:error:",
            |error| unsafe {
                DeviceSpi::newComputePipelineStateWithFunction_error(d, function, error)
            },
        )
    }

    /// The one creator whose result is kept: the source is compiled once per device, and a
    /// later call answers the same library ([`super::shaders`]).
    fn new_shader_library(&self) -> Result<Option<MetalLibrary>, NsErrorText> {
        super::shaders::library(self)
    }

    fn new_event(&self, on: OnFault) -> RaiseResult<Option<MetalEvent>> {
        let d = self.get();
        msg::send("newEvent", Policy::On(on), || d.newEvent()).map(created)
    }

    fn new_fence(&self) -> RaiseResult<Option<MetalFence>> {
        let d = self.get();
        msg::send("newFence", Policy::Caught, || d.newFence()).map(created)
    }

    /// The IR is a `dispatch_data_t` over an `NSData` holding a copy of `data`, released by the
    /// data's destructor block on a default-QoS root worker when Metal lets go of it
    /// ([`ir_ns_data`], [`ir_dispatch_data`]). The data's own reference is autoreleased into the
    /// caller's pool.
    fn new_library(&self, data: &[u8]) -> RaiseResult<Result<Option<MetalLibrary>, NsErrorText>> {
        let ir = DispatchRetained::into_raw(ir_dispatch_data(ir_ns_data(data)));
        // SAFETY: a dispatch object is an Objective-C object; its +1 goes to the current pool.
        unsafe { objc2::ffi::objc_autorelease(ir.as_ptr().cast::<AnyObject>()) };
        // SAFETY: the pool keeps the data alive past this call, which is inside its scope.
        let ir: &DispatchData = unsafe { ir.as_ref() };
        let d = self.get();
        // SAFETY: `create_or_error`'s out-parameter.
        create_or_error(
            Policy::Caught,
            "newLibraryWithData:error:",
            |error| unsafe { DeviceSpi::newLibraryWithData_error(d, ir, error) },
        )
    }

    /// Metal's own assertion ends the process for a third layer on the reference host.
    fn new_rate_map(
        &self,
        descriptor: &MetalRateMapDescriptor,
    ) -> RaiseResult<Option<MetalRateMap>> {
        let d = self.get();
        msg::send(
            "newRasterizationRateMapWithDescriptor:",
            Policy::Caught,
            || d.newRasterizationRateMapWithDescriptor(descriptor),
        )
        .map(created)
    }

    fn deserialize_render_pipeline_descriptor(
        &self,
        hold: Hold<'_>,
        data: &[u8],
        context: FunctionResolver<'_, MetalPlatform>,
    ) -> RaiseResult<Option<MetalRenderPipelineDescriptor>> {
        let sel = "newRenderPipelineDescriptorWithSerializedData:deserializationContext:";
        deserialize(
            hold,
            self,
            sel,
            DispatchData::from_bytes(data),
            context,
            |d, data, context| {
                d.newRenderPipelineDescriptorWithSerializedData_deserializationContext(
                    data, context,
                )
            },
        )
    }

    /// Reflection is nil; a nil descriptor is passed as it is.
    fn new_render_pipeline_state_from_descriptor(
        &self,
        descriptor: Option<&MetalRenderPipelineDescriptor>,
        options: u64,
    ) -> RaiseResult<Result<Option<MetalRenderPipeline>, NsErrorText>> {
        let (d, descriptor) = (self.get(), descriptor.map(Obj::get));
        // SAFETY: `create_or_error`'s out-parameter.
        create_or_error(
            Policy::Caught,
            "newRenderPipelineStateWithDescriptor:options:reflection:error:",
            |error| unsafe {
                DeviceSpi::newRenderPipelineStateWithDescriptor_options_reflection_error(
                    d, descriptor, options, error,
                )
            },
        )
    }

    /// Reflection is nil; a nil descriptor is passed as it is.
    fn new_tile_render_pipeline_state(
        &self,
        descriptor: Option<&MetalTileRenderPipelineDescriptor>,
        options: u64,
    ) -> RaiseResult<Result<Option<MetalRenderPipeline>, NsErrorText>> {
        let (d, descriptor) = (self.get(), descriptor.map(Obj::get));
        // SAFETY: `create_or_error`'s out-parameter.
        create_or_error(
            Policy::Caught,
            "newRenderPipelineStateWithTileDescriptor:options:reflection:error:",
            |error| unsafe {
                DeviceSpi::newRenderPipelineStateWithTileDescriptor_options_reflection_error(
                    d, descriptor, options, error,
                )
            },
        )
    }

    fn new_shared_event(&self) -> Option<MetalSharedEvent> {
        let d = self.get();
        created(msg::get("newSharedEvent", || d.newSharedEvent()))
    }

    /// A nil handle is passed as it is.
    fn open_shared_event(
        &self,
        shared_event_handle: Option<&MetalSharedEventHandle>,
    ) -> Option<MetalSharedEvent> {
        let (d, handle) = (self.get(), shared_event_handle.map(Obj::get));
        created(msg::get("newSharedEventWithHandle:", || {
            DeviceSpi::newSharedEventWithHandle(d, handle)
        }))
    }

    fn new_shared_texture(
        &self,
        descriptor: &MetalTextureDescriptor,
    ) -> RaiseResult<Option<MetalTexture>> {
        let d = self.get();
        msg::send("newSharedTextureWithDescriptor:", Policy::Caught, || {
            d.newSharedTextureWithDescriptor(descriptor)
        })
        .map(created)
    }

    /// The resource index is the caller's, unverified.
    fn open_shared_texture(
        &self,
        handle: Option<&MetalSharedTextureHandle>,
        resource_index: u64,
    ) -> RaiseResult<Option<MetalTexture>> {
        let (d, handle) = (self.get(), handle.map(Obj::get));
        msg::send(
            "newSharedTextureWithHandle:withResourceIndex:",
            Policy::Caught,
            || d.newSharedTextureWithHandle_withResourceIndex(handle, resource_index),
        )
        .map(created)
    }

    fn new_texture(
        &self,
        on: OnFault,
        descriptor: &MetalTextureDescriptor,
    ) -> RaiseResult<Option<MetalTexture>> {
        let d = self.get();
        msg::send("newTextureWithDescriptor:", Policy::On(on), || {
            d.newTextureWithDescriptor(descriptor)
        })
        .map(created)
    }

    /// The texture retains the surface. Metal's own validation ends the process for some
    /// descriptors (a nil surface among them).
    fn new_texture_on_surface(
        &self,
        descriptor: &MetalTextureDescriptor,
        iosurface: Option<&MetalSurface>,
        plane: u64,
    ) -> RaiseResult<Option<MetalTexture>> {
        let (d, surface) = (self.get(), iosurface.map(MetalSurface::as_iosurface_ref));
        msg::send(
            "newTextureWithDescriptor:iosurface:plane:",
            Policy::Caught,
            || DeviceSpi::newTextureWithDescriptor_iosurface_plane(d, descriptor, surface, plane),
        )
        .map(created)
    }

    fn deserialize_tile_render_pipeline_descriptor(
        &self,
        hold: Hold<'_>,
        data: &[u8],
        buffer_length: u64,
        context: FunctionResolver<'_, MetalPlatform>,
    ) -> RaiseResult<Option<MetalTileRenderPipelineDescriptor>> {
        let sel = "newTileRenderPipelineDescriptorWithSerializedData:deserializationContext:";
        let data = tile_descriptor_data(data, buffer_length);
        deserialize(hold, self, sel, data, context, |d, data, context| {
            d.newTileRenderPipelineDescriptorWithSerializedData_deserializationContext(
                data, context,
            )
        })
    }

    fn reserve_resource_indices(&self, resource_type: u64, indices: &[u64]) -> RaiseResult<()> {
        let (d, pointer, count) = (self.get(), indices.as_ptr(), indices.len().to_u64());
        // SAFETY: the pointer is valid for `count` indices during the call.
        msg::send(
            "reserveResourceIndicesForResourceType:indices:indexCount:",
            Policy::Caught,
            || unsafe {
                d.reserveResourceIndicesForResourceType_indices_indexCount(
                    resource_type,
                    pointer,
                    count,
                )
            },
        )
    }

    /// `MTLPixelFormatGetInfoForDevice(device, format)`: the 0x38 bytes, copied out whole.
    fn pixel_format_info(&self, format: u64) -> PixelFormatInfo {
        // SAFETY: a C function of a live device and an integer; it returns a struct by value.
        msg::get("MTLPixelFormatGetInfoForDevice()", || unsafe {
            functions::MTLPixelFormatGetInfoForDevice(self, format)
        })
        .to_info()
    }

    /// `MTLGetTextureLevelInfoForDeviceWithOptions(device, …)`: the 0x68 bytes.
    fn texture_level_info(&self, q: &TextureLevelQuery) -> TextureLevelInfo {
        let q = *q;
        // SAFETY: as `pixel_format_info`.
        msg::get("MTLGetTextureLevelInfoForDeviceWithOptions()", || unsafe {
            functions::MTLGetTextureLevelInfoForDeviceWithOptions(
                self,
                q.pixel_format,
                q.width,
                q.height,
                q.depth,
                q.array_length,
                q.options,
                q.flags,
            )
        })
    }

    // Capabilities and creators from values.

    /// Read in full on the first call for a registry ID (`read_caps`), then shared: a handle to
    /// the same device can also come from a `device` getter.
    fn caps(&self) -> Arc<DeviceCaps> {
        static CAPS: LazyLock<Mutex<HashMap<u64, Arc<DeviceCaps>>>> = LazyLock::new(Mutex::default);
        let d = self.get();
        let id = msg::get("registryID", || d.registryID());
        if let Some(c) = CAPS.lock().get(&id) {
            return c.clone();
        }
        // SAFETY: `vm_page_size` is a libsystem_kernel global, set before `main` and never
        // written after.
        let page_size = (unsafe { mach2::vm_page_size::vm_page_size }).to_u64();
        let read = Arc::new(read_caps(self, page_size));
        // Another thread's read of the same device may have come first: its caps are kept.
        CAPS.lock().entry(id).or_insert(read).clone()
    }

    /// Every getter's answer, Metal's own (`read_identity_caps`).
    fn identity_caps(&self) -> IdentityCaps {
        read_identity_caps(self)
    }

    // Each creator from a value makes the descriptor with its class's `new`, makes it hold the
    // value (`desc`'s `apply_*`), then sends the descriptor's creator.

    fn new_buffer(&self, desc: BufferDesc<'_>) -> RaiseResult<Option<MetalBuffer>> {
        let d = crate::platform::class_new(crate::spi::classes::MTLBufferDescriptor::new);
        desc::apply_buffer(desc, &d);
        buffer_with_descriptor(self, &d)
    }

    /// The descriptor's `resourceOptions` are read after the creation.
    fn new_heap(&self, desc: &HeapDesc<'_, MetalPlatform>) -> RaiseResult<HeapMade<MetalPlatform>> {
        let d = crate::platform::class_new(objc2_metal::MTLHeapDescriptor::new);
        desc::apply_heap(desc, &d);
        let heap = heap_with_descriptor(self, &d)?;
        Ok(HeapMade {
            heap,
            descriptor_resource_options: desc::heap_descriptor_resource_options(&d),
        })
    }

    /// A heap over one address range, `bytes`, which Metal makes over the range's pages in place.
    fn new_heap_over_host_memory(
        &self,
        bytes: HostRange<'_>,
        desc: &HeapDesc<'_, MetalPlatform>,
    ) -> RaiseResult<HeapMade<MetalPlatform>> {
        let range = ResourceAddressRange {
            address: bytes.ptr().addr().to_u64(),
            length: bytes.len(),
        };
        let ranges = crate::platform::address_range_array(&[range])?;
        self.new_heap(&HeapDesc {
            size: desc.size,
            heap_type: desc.heap_type,
            options: desc.options.clone(),
            address_ranges: Some(&ranges),
            pinned_gpu_address: desc.pinned_gpu_address,
        })
    }

    fn new_sampler_state(&self, desc: &SamplerDesc) -> RaiseResult<Option<MetalSamplerState>> {
        let d = crate::platform::class_new(objc2_metal::MTLSamplerDescriptor::new);
        desc::apply_sampler(desc, &d);
        sampler_state_with_descriptor(self, &d)
    }

    fn new_depth_stencil_state(
        &self,
        desc: &DepthStencilDesc,
    ) -> RaiseResult<Option<MetalDepthStencilState>> {
        let d = crate::platform::class_new(objc2_metal::MTLDepthStencilDescriptor::new);
        desc::apply_depth_stencil(desc, &d);
        depth_stencil_state_with_descriptor(self, &d)
    }

    fn new_indirect_command_buffer(
        &self,
        desc: &IcbDesc,
        max_count: u64,
        options: u64,
    ) -> RaiseResult<Option<MetalIcb>> {
        let d = crate::platform::class_new(objc2_metal::MTLIndirectCommandBufferDescriptor::new);
        desc::apply_icb(desc, &d);
        icb_with_descriptor(self, &d, max_count, options)
    }

    /// A raise from the colour attachment's lookup comes before the creation.
    fn new_render_pipeline_state(
        &self,
        desc: &RenderPipelineDesc<'_, MetalPlatform>,
    ) -> Result<Option<MetalRenderPipeline>, NsErrorText> {
        let d = crate::platform::class_new(objc2_metal::MTLRenderPipelineDescriptor::new);
        desc::apply_render_pipeline(desc, &d);
        render_pipeline_state_with_descriptor(self, &d)
    }

    fn new_render_pipeline_state_full(
        &self,
        desc: &FullRenderPipelineDesc<'_, MetalPlatform>,
    ) -> Result<Option<MetalRenderPipeline>, NsErrorText> {
        let d = crate::platform::class_new(objc2_metal::MTLRenderPipelineDescriptor::new);
        desc::apply_full_render_pipeline(desc, &d);
        render_pipeline_state_with_descriptor(self, &d)
    }

    fn new_compute_pipeline_state_full(
        &self,
        desc: &FullComputePipelineDesc<'_, MetalPlatform>,
    ) -> Result<Option<MetalComputePipeline>, NsErrorText> {
        let d = crate::platform::class_new(objc2_metal::MTLComputePipelineDescriptor::new);
        desc::apply_full_compute_pipeline(desc, &d);
        let (device, descriptor) = (self.get(), d.get());
        // SAFETY: `create_or_die`'s out-parameter.
        create_or_die(
            "newComputePipelineStateWithDescriptor:options:reflection:error:",
            |error| unsafe {
                DeviceSpi::newComputePipelineStateWithDescriptor_options_reflection_error(
                    device,
                    Some(descriptor),
                    0,
                    error,
                )
            },
        )
    }
}

// ---- The creators that take a descriptor.

/// `newBufferWithDescriptor:` (SPI).
fn buffer_with_descriptor(
    device: &MetalDevice,
    descriptor: &MetalBufferDescriptor,
) -> RaiseResult<Option<MetalBuffer>> {
    let d = device.get();
    msg::send("newBufferWithDescriptor:", Policy::Caught, || {
        d.newBufferWithDescriptor(descriptor)
    })
    .map(created)
}

/// `newDepthStencilStateWithDescriptor:`.
fn depth_stencil_state_with_descriptor(
    device: &MetalDevice,
    descriptor: &MetalDepthStencilDescriptor,
) -> RaiseResult<Option<MetalDepthStencilState>> {
    let d = device.get();
    msg::send(
        "newDepthStencilStateWithDescriptor:",
        Policy::Caught,
        || d.newDepthStencilStateWithDescriptor(descriptor),
    )
    .map(created)
}

/// `newHeapWithDescriptor:`.
fn heap_with_descriptor(
    device: &MetalDevice,
    descriptor: &MetalHeapDescriptor,
) -> RaiseResult<Option<MetalHeap>> {
    let d = device.get();
    msg::send("newHeapWithDescriptor:", Policy::Caught, || {
        d.newHeapWithDescriptor(descriptor)
    })
    .map(created)
}

/// `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:`.
fn icb_with_descriptor(
    device: &MetalDevice,
    descriptor: &MetalIcbDescriptor,
    max_count: u64,
    options: u64,
) -> RaiseResult<Option<MetalIcb>> {
    let d = device.get();
    let (max_count, options) = (max_count.to_usize(), MTLResourceOptions(options.to_usize()));
    // SAFETY: raw guest values go to Metal unvalidated; Metal checks them itself.
    msg::send(
        "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",
        Policy::Caught,
        || unsafe {
            d.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                descriptor, max_count, options,
            )
        },
    )
    .map(created)
}

/// `newRenderPipelineStateWithDescriptor:error:`.
fn render_pipeline_state_with_descriptor(
    device: &MetalDevice,
    descriptor: &MetalRenderPipelineDescriptor,
) -> Result<Option<MetalRenderPipeline>, NsErrorText> {
    let d = device.get();
    // SAFETY: `create_or_die`'s out-parameter.
    create_or_die(
        "newRenderPipelineStateWithDescriptor:error:",
        |error| unsafe {
            DeviceSpi::newRenderPipelineStateWithDescriptor_error(d, descriptor, error)
        },
    )
}

/// `newSamplerStateWithDescriptor:`.
fn sampler_state_with_descriptor(
    device: &MetalDevice,
    descriptor: &MetalSamplerDescriptor,
) -> RaiseResult<Option<MetalSamplerState>> {
    let d = device.get();
    msg::send("newSamplerStateWithDescriptor:", Policy::Caught, || {
        d.newSamplerStateWithDescriptor(descriptor)
    })
    .map(created)
}

// ---- Capabilities.

/// `name`, copied out; `None` when the device answers nil (a host's device may).
fn name(d: &ProtocolObject<dyn MTLDevice>) -> Option<String> {
    msg::get("name", || DeviceSpi::name_or_nil(d)).map(|name| msg::string_text(&name))
}

/// `gpuCoreCount`, or `None` when the device does not respond to it, and then only
/// `respondsToSelector:` is called. The runtime answers `unsigned int` (`I@:`).
fn gpu_core_count(d: &ProtocolObject<dyn MTLDevice>) -> Option<u64> {
    if !msg::get("respondsToSelector:", || {
        d.respondsToSelector(sel!(gpuCoreCount))
    }) {
        return None;
    }
    let n = msg::get("gpuCoreCount", || d.gpuCoreCount());
    Some(u64::from(n))
}

/// Whether the device responds to `sel`, an `MTLDeviceSPI` getter sent only where it does.
fn responds(d: &ProtocolObject<dyn MTLDevice>, sel: Sel) -> bool {
    msg::get("respondsToSelector:", || d.respondsToSelector(sel))
}

/// `heapBufferSizeAndAlignWithLength:options:`.
fn heap_buffer_size_and_align(
    d: &ProtocolObject<dyn MTLDevice>,
    length: u64,
    options: u64,
) -> RaiseResult<SizeAndAlign> {
    let (length, options) = (length.to_usize(), MTLResourceOptions(options.to_usize()));
    msg::send(
        "heapBufferSizeAndAlignWithLength:options:",
        Policy::Caught,
        || d.heapBufferSizeAndAlignWithLength_options(length, options),
    )
    .map(SizeAndAlign::from_mtl)
}

/// `minimumLinearTextureAlignmentForPixelFormat:`.
fn minimum_linear_alignment(d: &ProtocolObject<dyn MTLDevice>, format: u64) -> RaiseResult<u64> {
    let format = MTLPixelFormat(format.to_usize());
    msg::send(
        "minimumLinearTextureAlignmentForPixelFormat:",
        Policy::Caught,
        || d.minimumLinearTextureAlignmentForPixelFormat(format),
    )
    .map(ToU64::to_u64)
}

/// Every capability of `device`, each query sent once, in no order core observes. A query that
/// raises (a `Caught` one) is asked under a handler of its own, whatever handler the asker has
/// (the answers are shared by every asker), and its raise is kept as its answer.
/// `page_size` is the length of the heap-buffer query (`vm_page_size`).
fn read_caps(device: &MetalDevice, page_size: u64) -> DeviceCaps {
    let d = device.get();
    /// A `Fatal` getter of the device: one call, whose selector is the getter's name.
    macro_rules! get {
        ($getter:ident) => {
            msg::get(stringify!($getter), || d.$getter())
        };
    }
    DeviceCaps {
        name: name(d),
        registry_id: get!(registryID),
        gpu_core_count: gpu_core_count(d),
        max_threads_per_threadgroup: MtlSize::from_mtl(get!(maxThreadsPerThreadgroup)),
        max_vertex_amplification_count: get!(maxVertexAmplificationCount),
        limits: Limits {
            device_linear_texture_alignment_bytes: get!(deviceLinearTextureAlignmentBytes),
            iosurface_read_only_texture_alignment_bytes: get!(
                iosurfaceReadOnlyTextureAlignmentBytes
            ),
            iosurface_texture_alignment_bytes: get!(iosurfaceTextureAlignmentBytes),
            linear_texture_alignment_bytes: get!(linearTextureAlignmentBytes),
            max_compute_buffers: get!(maxComputeBuffers),
            max_compute_local_memory_sizes: get!(maxComputeLocalMemorySizes),
            max_compute_samplers: get!(maxComputeSamplers),
            max_compute_textures: get!(maxComputeTextures),
            max_compute_threadgroup_memory: get!(maxComputeThreadgroupMemory),
            max_compute_threadgroup_memory_alignment_bytes: get!(
                maxComputeThreadgroupMemoryAlignmentBytes
            ),
            max_fragment_buffers: get!(maxFragmentBuffers),
            max_fragment_samplers: get!(maxFragmentSamplers),
            max_fragment_textures: get!(maxFragmentTextures),
            max_predicated_nesting_depth: get!(maxPredicatedNestingDepth),
            max_rasterization_rate_layer_count: get!(maxRasterizationRateLayerCount),
            max_texture_height_2d: get!(maxTextureHeight2D),
            max_texture_layers: get!(maxTextureLayers),
            max_texture_width_2d: get!(maxTextureWidth2D),
            max_threadgroup_memory_length: get!(maxThreadgroupMemoryLength).to_u64(),
            max_tile_buffers: get!(maxTileBuffers),
            max_tile_samplers: get!(maxTileSamplers),
            max_tile_textures: get!(maxTileTextures),
            max_total_compute_threads_per_threadgroup: get!(maxTotalComputeThreadsPerThreadgroup),
            max_vertex_buffers: get!(maxVertexBuffers),
            max_vertex_samplers: get!(maxVertexSamplers),
            max_vertex_textures: get!(maxVertexTextures),
            max_viewport_count: get!(maxViewportCount),
        },
        flags: Flags {
            are_raster_order_groups_supported: get!(areRasterOrderGroupsSupported),
            is_depth24_stencil8_pixel_format_supported: get!(isDepth24Stencil8PixelFormatSupported),
            is_framebuffer_read_supported: get!(isFramebufferReadSupported),
            is_large_mrt_supported: get!(isLargeMRTSupported),
            is_rgb10a2_gamma_supported: get!(isRGB10A2GammaSupported),
            supports_buffer_with_address_ranges: get!(supportsBufferWithAddressRanges),
            supports_buffer_with_iosurface: get!(supportsBufferWithIOSurface),
            supports_command_buffer_jump: get!(supportsCommandBufferJump),
            supports_compute_compressed_texture_write: get!(supportsComputeCompressedTextureWrite),
            supports_conditional_load_store: get!(supportsConditionalLoadStore),
            supports_dynamic_attribute_stride: get!(supportsDynamicAttributeStride),
            supports_extended_xr10_formats: get!(supportsExtendedXR10Formats),
            supports_float16_bcubic_filtering: get!(supportsFloat16BCubicFiltering),
            supports_heap_with_address_ranges: get!(supportsHeapWithAddressRanges),
            supports_image_blocks: get!(supportsImageBlocks),
            supports_memory_order_atomics: get!(supportsMemoryOrderAtomics),
            supports_native_hardware_fp16: get!(supportsNativeHardwareFP16),
            supports_non_square_tile_shaders: get!(supportsNonSquareTileShaders),
            supports_programmable_sample_positions: get!(supportsProgrammableSamplePositions),
            supports_resource_detach_backing: get!(supportsResourceDetachBacking),
            supports_shared_texture_handles: get!(supportsSharedTextureHandles),
            supports_simd_reduction: get!(supportsSIMDReduction),
            supports_simd_shuffle_and_fill: get!(supportsSIMDShuffleAndFill),
            supports_texture_2d_multisample_array: get!(supportsTexture2DMultisampleArray),
            supports_tile_shaders: get!(supportsTileShaders),
        },
        sample_counts: SAMPLE_COUNTS.map(|count| {
            let count = count.to_usize();
            msg::send("supportsTextureSampleCount:", Policy::Caught, || {
                d.supportsTextureSampleCount(count)
            })
        }),
        primitive_types: std::array::from_fn(|t| {
            let primitive_type = t.to_u64();
            msg::send("supportsPrimitiveType:", Policy::Caught, || {
                d.supportsPrimitiveType(primitive_type)
            })
        }),
        families: FAMILIES.map(|family| {
            // `NSInteger` is `i64` on arm64: lossless.
            let family = MTLGPUFamily(family as isize);
            msg::send("supportsFamily:", Policy::Caught, || {
                d.supportsFamily(family)
            })
        }),
        write_rounding_modes: WRITE_ROUNDING_MODES.map(|mode| {
            msg::send("supportsTextureWriteRoundingMode:", Policy::Caught, || {
                d.supportsTextureWriteRoundingMode(mode)
            })
        }),
        heap_buffer_page_private: heap_buffer_size_and_align(d, page_size, HEAP_QUERY_OPTIONS),
        min_linear_alignment_bgra8: minimum_linear_alignment(d, LINEAR_ALIGNMENT_FORMAT),
    }
}

/// The largest count in `1..=LARGEST_COUNT_ASKED` that `accepts` takes, counting up from 1 until
/// it refuses one; 0 where it refuses 1.
fn largest(accepts: impl Fn(u64) -> bool) -> u64 {
    (1..=LARGEST_COUNT_ASKED)
        .take_while(|&count| accepts(count))
        .last()
        .unwrap_or(0)
}

/// The answers of `format` [`FormatAnswers`] holds: its texture buffers' alignment where Metal
/// answers it ([`TEXTURE_BUFFER_FORMATS`]; it asserts on every other format), and its 2D sparse
/// tiles.
fn format_answers(d: &ProtocolObject<dyn MTLDevice>, format: u64) -> FormatAnswers {
    let pf = MTLPixelFormat(format.to_usize());
    let texture_buffer_alignment = TEXTURE_BUFFER_FORMATS.contains(&format).then(|| {
        msg::get("minimumTextureBufferAlignmentForPixelFormat:", || {
            d.minimumTextureBufferAlignmentForPixelFormat(pf)
        })
        .to_u64()
    });
    let mut tiles = [[0; 3]; 9];
    let pairs = SPARSE_SAMPLE_COUNTS
        .iter()
        .flat_map(|&samples| SPARSE_PAGE_SIZES.iter().map(move |&page| (samples, page)));
    for (tile, (samples, page)) in tiles.iter_mut().zip(pairs) {
        // `NSInteger` is `i64` on arm64: lossless.
        let page = objc2_metal::MTLSparsePageSize(page as isize);
        let samples = samples.to_usize();
        // SAFETY: any format, count and page; Metal answers a tile for each of the SDK's.
        let size = msg::get(
            "sparseTileSizeWithTextureType:pixelFormat:sampleCount:sparsePageSize:",
            || unsafe {
                d.sparseTileSizeWithTextureType_pixelFormat_sampleCount_sparsePageSize(
                    objc2_metal::MTLTextureType::Type2D,
                    pf,
                    samples,
                    page,
                )
            },
        );
        *tile = [size.width, size.height, size.depth].map(ToU64::to_u64);
    }
    FormatAnswers {
        format,
        texture_buffer_alignment,
        sparse_tiles: Some(tiles),
    }
}

/// Every identity and capability answer of `device` [`IdentityCaps`] holds, each Metal's getter
/// or query, sent once. Metal answers each of them, so none is `None`, but for the
/// `MTLDeviceSPI` getters a device may not respond to.
#[allow(deprecated)] // `supportsFeatureSet:`, `location` and the peer getters, deprecated in the SDK
fn read_identity_caps(device: &MetalDevice) -> IdentityCaps {
    let d = device.get();
    /// A `Fatal` getter of the device: one call, whose selector is the getter's name.
    macro_rules! get {
        ($getter:ident) => {
            msg::get(stringify!($getter), || d.$getter())
        };
    }
    let arch = msg::get("architecture", || d.architecture());
    let architecture_name = msg::get("name", || arch.name());
    // `supportsPlacementSparse` is the macOS 26 SDK's; a device of an older OS lacks it.
    let placement_sparse = msg::get("respondsToSelector:", || {
        d.respondsToSelector(sel!(supportsPlacementSparse))
    })
    .then(|| msg::get("supportsPlacementSparse", || d.supportsPlacementSparse()));
    IdentityCaps {
        families: SDK_FAMILIES.map(|family| {
            // `NSInteger` is `i64` on arm64: lossless.
            let family = MTLGPUFamily(family as isize);
            Some(msg::get("supportsFamily:", || d.supportsFamily(family)))
        }),
        feature_sets: FEATURE_SETS.map(|set| {
            let set = objc2_metal::MTLFeatureSet(set.to_usize());
            Some(msg::get("supportsFeatureSet:", || {
                d.supportsFeatureSet(set)
            }))
        }),
        counter_sampling: COUNTER_SAMPLING_POINTS.map(|point| {
            let point = objc2_metal::MTLCounterSamplingPoint(point.to_usize());
            Some(msg::get("supportsCounterSampling:", || {
                d.supportsCounterSampling(point)
            }))
        }),
        architecture_name: Some(msg::string_text(&architecture_name)),
        formats: SDK_PIXEL_FORMATS
            .iter()
            .map(|&format| format_answers(d, format))
            .collect(),
        sizes: IdentitySizes {
            recommended_max_working_set_size: Some(get!(recommendedMaxWorkingSetSize)),
            location: Some(get!(location).0.to_u64()),
            location_number: Some(get!(locationNumber).to_u64()),
            max_transfer_rate: Some(get!(maxTransferRate)),
            peer_group_id: Some(get!(peerGroupID)),
            peer_index: Some(u64::from(get!(peerIndex))),
            peer_count: Some(u64::from(get!(peerCount))),
            max_buffer_length: Some(get!(maxBufferLength).to_u64()),
            max_argument_buffer_sampler_count: Some(get!(maxArgumentBufferSamplerCount).to_u64()),
            argument_buffers_support: Some(get!(argumentBuffersSupport).0.to_u64()),
            read_write_texture_support: Some(get!(readWriteTextureSupport).0.to_u64()),
            sparse_tile_size_in_bytes: Some(get!(sparseTileSizeInBytes).to_u64()),
            maximum_concurrent_compilation_task_count: Some(
                get!(maximumConcurrentCompilationTaskCount).to_u64(),
            ),
            largest_vertex_amplification_count: Some(largest(|count| {
                let count = count.to_usize();
                msg::get("supportsVertexAmplificationCount:", || {
                    d.supportsVertexAmplificationCount(count)
                })
            })),
            largest_rasterization_rate_map_layer_count: Some(largest(|count| {
                let count = count.to_usize();
                // SAFETY: a count is any `NSUInteger`; Metal answers no for one it does not take.
                msg::get("supportsRasterizationRateMapWithLayerCount:", || unsafe {
                    d.supportsRasterizationRateMapWithLayerCount(count)
                })
            })),
            // `Q@:` getters, `None` where the device does not respond to one, and then only
            // `respondsToSelector:` is sent.
            feature_profile: responds(d, sel!(featureProfile))
                .then(|| msg::get("featureProfile", || d.featureProfile())),
            half_fp_config: responds(d, sel!(halfFPConfig))
                .then(|| msg::get("halfFPConfig", || d.halfFPConfig())),
            single_fp_config: responds(d, sel!(singleFPConfig))
                .then(|| msg::get("singleFPConfig", || d.singleFPConfig())),
            double_fp_config: responds(d, sel!(doubleFPConfig))
                .then(|| msg::get("doubleFPConfig", || d.doubleFPConfig())),
        },
        flags: IdentityFlags {
            is_low_power: Some(get!(isLowPower)),
            is_headless: Some(get!(isHeadless)),
            is_removable: Some(get!(isRemovable)),
            has_unified_memory: Some(get!(hasUnifiedMemory)),
            supports_32bit_float_filtering: Some(get!(supports32BitFloatFiltering)),
            supports_32bit_msaa: Some(get!(supports32BitMSAA)),
            supports_query_texture_lod: Some(get!(supportsQueryTextureLOD)),
            supports_bc_texture_compression: Some(get!(supportsBCTextureCompression)),
            supports_pull_model_interpolation: Some(get!(supportsPullModelInterpolation)),
            are_barycentric_coords_supported: Some(get!(areBarycentricCoordsSupported)),
            supports_shader_barycentric_coordinates: Some(get!(
                supportsShaderBarycentricCoordinates
            )),
            supports_dynamic_libraries: Some(get!(supportsDynamicLibraries)),
            supports_render_dynamic_libraries: Some(get!(supportsRenderDynamicLibraries)),
            supports_raytracing: Some(get!(supportsRaytracing)),
            supports_function_pointers: Some(get!(supportsFunctionPointers)),
            supports_function_pointers_from_render: Some(get!(supportsFunctionPointersFromRender)),
            supports_raytracing_from_render: Some(get!(supportsRaytracingFromRender)),
            supports_primitive_motion_blur: Some(get!(supportsPrimitiveMotionBlur)),
            supports_placement_sparse: placement_sparse,
        },
    }
}

#[cfg(test)]
pub(crate) mod tests {

    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use objc2::msg_send;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer as _, MTLCommandQueue as _, MTLComputePipelineDescriptor, MTLFunction as _,
        MTLGPUFamily, MTLHeap as _, MTLIndirectCommandBuffer as _, MTLLibrary as _, MTLPixelFormat,
        MTLRasterizationRateLayerDescriptor, MTLRasterizationRateMapDescriptor,
        MTLRenderPipelineDescriptor, MTLSharedEvent as _, MTLSize as MtlSizeObjc, MTLStorageMode,
        MTLTexture as _, MTLTextureDescriptor, MTLTileRenderPipelineDescriptor,
    };
    use reims_vgpu_core::platform::Handle;

    use super::*;
    use reims_vgpu_core::host::HostPtr;
    use reims_vgpu_core::num::ToUsize;

    pub(crate) fn dev() -> MetalDevice {
        MetalPlatform::system_default_device().expect("a Metal device")
    }

    /// Our shader library, whose functions stand in for any.
    pub(crate) fn library(d: &MetalDevice) -> MetalLibrary {
        d.new_shader_library().unwrap().unwrap()
    }

    pub(crate) fn function(lib: &MetalLibrary, name: &str) -> MetalFunction {
        Obj::from_retained(lib.newFunctionWithName(&NSString::from_str(name)).unwrap())
    }

    fn texture_descriptor(w: u64, h: u64, mode: MTLStorageMode) -> MetalTextureDescriptor {
        let td = MTLTextureDescriptor::new();
        // SAFETY: plain property sets on a fresh descriptor.
        unsafe {
            td.setWidth(w.to_usize());
            td.setHeight(h.to_usize());
        }
        td.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        td.setStorageMode(mode);
        Obj::from_retained(td)
    }

    /// The caps' public getters agree with objc2-metal's view of the same selectors (-name,
    /// -registryID, -maxThreadsPerThreadgroup, -maxThreadgroupMemoryLength,
    /// -areRasterOrderGroupsSupported, -isDepth24Stencil8PixelFormatSupported, -supportsFamily:,
    /// -maxVertexAmplificationCount)
    #[test]
    fn public_getters_match_objc2_metal() {
        let d = dev();
        let o = d.get();
        let c = d.caps();
        assert_eq!(c.name, Some(o.name().to_string()));
        assert_eq!(c.registry_id, o.registryID());
        let s = o.maxThreadsPerThreadgroup();
        assert_eq!(
            c.max_threads_per_threadgroup,
            MtlSize::new(s.width.to_u64(), s.height.to_u64(), s.depth.to_u64())
        );
        assert_eq!(
            c.limits.max_threadgroup_memory_length,
            o.maxThreadgroupMemoryLength().to_u64()
        );
        assert_eq!(
            c.flags.are_raster_order_groups_supported,
            o.areRasterOrderGroupsSupported()
        );
        assert_eq!(
            c.flags.is_depth24_stencil8_pixel_format_supported,
            o.isDepth24Stencil8PixelFormatSupported()
        );
        // Apple7 is the third family asked.
        assert_eq!(c.families[2], Ok(o.supportsFamily(MTLGPUFamily::Apple7)));
        assert_eq!(
            c.flags.supports_programmable_sample_positions,
            o.areProgrammableSamplePositionsSupported()
        );
        assert_eq!(
            c.max_vertex_amplification_count,
            o.maxVertexAmplificationCount()
        );
    }

    /// The SPI getters answer on the reference host: linearTextureAlignmentBytes 16,
    /// maxComputeBuffers 31, and YES from the five capability BOOLs and isFramebufferReadSupported
    #[test]
    fn spi_getters_answer() {
        let c = dev().caps();
        let l = &c.limits;
        assert_eq!(l.linear_texture_alignment_bytes, 16);
        assert_eq!(l.max_compute_buffers, 31);
        let f = &c.flags;
        assert!(f.supports_shared_texture_handles);
        assert!(f.supports_buffer_with_iosurface);
        assert!(f.supports_buffer_with_address_ranges);
        assert!(f.supports_heap_with_address_ranges);
        assert!(f.supports_resource_detach_backing);
        assert!(f.is_framebuffer_read_supported);
        for (sel, v) in [
            (
                "deviceLinearTextureAlignmentBytes",
                l.device_linear_texture_alignment_bytes,
            ),
            (
                "iosurfaceReadOnlyTextureAlignmentBytes",
                l.iosurface_read_only_texture_alignment_bytes,
            ),
            (
                "iosurfaceTextureAlignmentBytes",
                l.iosurface_texture_alignment_bytes,
            ),
            (
                "maxComputeLocalMemorySizes",
                l.max_compute_local_memory_sizes,
            ),
            ("maxComputeSamplers", l.max_compute_samplers),
            ("maxComputeTextures", l.max_compute_textures),
            (
                "maxComputeThreadgroupMemory",
                l.max_compute_threadgroup_memory,
            ),
            (
                "maxComputeThreadgroupMemoryAlignmentBytes",
                l.max_compute_threadgroup_memory_alignment_bytes,
            ),
            ("maxFragmentBuffers", l.max_fragment_buffers),
            ("maxFragmentSamplers", l.max_fragment_samplers),
            ("maxFragmentTextures", l.max_fragment_textures),
            ("maxPredicatedNestingDepth", l.max_predicated_nesting_depth),
            (
                "maxRasterizationRateLayerCount",
                l.max_rasterization_rate_layer_count,
            ),
            ("maxTextureHeight2D", l.max_texture_height_2d),
            ("maxTextureLayers", l.max_texture_layers),
            ("maxTextureWidth2D", l.max_texture_width_2d),
            ("maxTileBuffers", l.max_tile_buffers),
            ("maxTileSamplers", l.max_tile_samplers),
            ("maxTileTextures", l.max_tile_textures),
            (
                "maxTotalComputeThreadsPerThreadgroup",
                l.max_total_compute_threads_per_threadgroup,
            ),
            ("maxVertexBuffers", l.max_vertex_buffers),
            ("maxVertexSamplers", l.max_vertex_samplers),
            ("maxVertexTextures", l.max_vertex_textures),
            ("maxViewportCount", l.max_viewport_count),
        ] {
            assert!(v > 0, "{sel} = {v}");
        }
        // gpuCoreCount is sent only after respondsToSelector:, and answers here.
        assert!(c.gpu_core_count.is_some_and(|n| n > 0));
    }

    /// The argument-taking queries (supportsTextureSampleCount: 1, 2, 4, 8,
    /// supportsPrimitiveType: 0..9 raw, supportsFamily: -1009, supportsTextureWriteRoundingMode:
    /// 1 and 2, minimumLinearTextureAlignmentForPixelFormat: BGRA8Unorm)
    #[test]
    fn argument_queries() {
        let c = dev().caps();
        assert_eq!(c.sample_counts[0], Ok(true));
        assert!(c.primitive_types.iter().all(Result::is_ok));
        assert_eq!(c.families[6], Ok(false));
        assert!(c.write_rounding_modes.iter().all(Result::is_ok));
        assert!(c.min_linear_alignment_bgra8.clone().unwrap() > 0);
    }

    /// heapBufferSizeAndAlignWithLength:options: (the caps' page-sized private buffer) and
    /// heapTextureSizeAndAlignWithDescriptor
    #[test]
    fn heap_size_and_align() {
        let d = dev();
        let b = d.caps().heap_buffer_page_private.clone().unwrap();
        assert!(b.size >= 16384 && b.align.is_power_of_two(), "{b:?}");
        let td = texture_descriptor(64, 64, MTLStorageMode::Private);
        let t = d.heap_texture_size_and_align(&td).unwrap();
        assert!(t.size >= 64 * 64 * 4 && t.align.is_power_of_two(), "{t:?}");
    }

    /// The argument-free creators (newCommandQueue, newEvent, newFence, newSharedEvent)
    #[test]
    fn plain_creators() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        assert!(q.get().device().registryID() == d.caps().registry_id);
        assert!(d.new_event(OnFault::Die).unwrap().is_some());
        assert!(d.new_fence().unwrap().is_some());
        let e = d.new_shared_event().unwrap();
        assert_eq!(e.get().signaledValue(), 0);
    }

    /// An object; nil for a length Metal refuses
    #[test]
    fn new_buffer_of_length() {
        let d = dev();
        let b = d
            .new_buffer_of_length(OnFault::Die, 256, 0)
            .unwrap()
            .unwrap();
        assert_eq!(b.get().length(), 256);
        assert!(
            d.new_buffer_of_length(OnFault::Die, u64::MAX, 0)
                .unwrap()
                .is_none()
        );
    }

    /// newBufferWithBytesNoCopy:length:options:deallocator: wraps the memory, and the deallocator
    /// runs exactly once when Metal frees the buffer; a nil deallocator is sent nil
    #[test]
    fn new_buffer_over_host_memory() {
        let d = dev();
        let layout = std::alloc::Layout::from_size_align(16384, 16384).unwrap();
        // SAFETY: a non-zero layout.
        let mem = unsafe { std::alloc::alloc_zeroed(layout) };
        let calls = Arc::new(AtomicU32::new(0));
        let c = calls.clone();
        let mem_addr = mem.addr();
        let deallocator: NoCopyDeallocator = Box::new(move || {
            c.fetch_add(1, Ordering::SeqCst);
            // SAFETY: allocated above with `layout`, freed once, here.
            unsafe { std::alloc::dealloc(std::ptr::with_exposed_provenance_mut(mem_addr), layout) };
        });
        // SAFETY: the memory stays allocated until the deallocator runs.
        let bytes = unsafe { HostRange::new(HostPtr::from_ptr(mem), 16384) };
        let b = d
            .new_buffer_over_host_memory(OnFault::Die, bytes, 0, Some(deallocator))
            .unwrap()
            .unwrap();
        assert_eq!(b.get().contents().as_ptr().addr(), mem_addr);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        drop(b);
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // SAFETY: as above.
        let mem = unsafe { std::alloc::alloc_zeroed(layout) };
        // SAFETY: the memory outlives the buffer (freed after the drop below).
        let bytes = unsafe { HostRange::new(HostPtr::from_ptr(mem), 16384) };
        let b = d
            .new_buffer_over_host_memory(OnFault::Die, bytes, 0, None)
            .unwrap()
            .unwrap();
        assert_eq!(b.get().length(), 16384);
        drop(b);
        // SAFETY: the buffer is gone; allocated above with `layout`.
        unsafe { std::alloc::dealloc(mem, layout) };
    }

    /// A heap over host memory is Metal's over the one address range in place: a buffer placed at
    /// an offset is the bytes there, at the heap's address plus the offset, and the CPU's writes
    /// and the GPU's fill of the buffer land in them
    #[test]
    fn a_heap_over_host_memory_lies_in_its_bytes() {
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
        use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
        use reims_vgpu_core::platform::resource::{Buffer, Heap};
        use reims_vgpu_core::platform::types::NsRange;
        use reims_vgpu_core::platform::values::OptionSet;
        let d = dev();
        let layout = std::alloc::Layout::from_size_align(0x2_0000, 0x4000).unwrap();
        // SAFETY: a non-zero layout; the memory is kept for the process.
        let mem = unsafe { std::alloc::alloc_zeroed(layout) };
        let at = |offset: usize| {
            // SAFETY: inside the allocation.
            unsafe { mem.add(offset) }
        };
        // SAFETY: the memory is kept for the process.
        let bytes = unsafe { HostRange::new(HostPtr::from_ptr(mem), 0x1_4000) };
        let desc = HeapDesc {
            size: 0x1_0001,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(0)],
            address_ranges: None,
            pinned_gpu_address: None,
        };
        let heap = d
            .new_heap_over_host_memory(bytes, &desc)
            .unwrap()
            .heap
            .unwrap();
        let buffer = heap
            .new_buffer_of_length(0x1000, 0, 0x4000)
            .unwrap()
            .unwrap();
        assert_eq!(buffer.contents().addr(), at(0x4000).addr());
        assert_eq!(buffer.gpu_address(), heap.gpu_address() + 0x4000);
        // SAFETY: the buffer's bytes are the allocation's, which nothing else touches.
        unsafe { at(0x4000).write_bytes(0xa1, 0x800) };
        let queue = d.new_command_queue().unwrap();
        let cb = queue.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for command in [
            BlitCmd::FillBuffer {
                buffer: &buffer,
                range: NsRange::new(0x800, 0x800),
                value: 0x5c,
            },
            BlitCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, command).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!(cb.status().0, 4, "completed");
        // SAFETY: as above, the command buffer done.
        unsafe {
            assert_eq!(at(0x47ff).read(), 0xa1, "the CPU's bytes stay");
            assert_eq!(
                at(0x4800).read(),
                0x5c,
                "the GPU's fill lands in the heap's bytes"
            );
            assert_eq!(at(0x4fff).read(), 0x5c);
            assert_eq!(at(0x5000).read(), 0, "past the buffer");
        }
    }

    /// Metal runs a no-copy buffer's deallocator on the thread that lets the buffer go, which
    /// need not be the library's: the deallocator runs marked as ours there (a panic in it aborts
    /// before it unwinds into Metal), and the thread's own mark comes back after it
    #[test]
    fn a_no_copy_deallocator_runs_marked_as_ours() {
        use reims_vgpu_core::rt::is_our_thread;
        let d = dev();
        let layout = std::alloc::Layout::from_size_align(16384, 16384).unwrap();
        // SAFETY: a non-zero layout.
        let mem = unsafe { std::alloc::alloc_zeroed(layout) };
        let mem_addr = mem.expose_provenance();
        let ours = Arc::new(Mutex::new(None));
        let o = ours.clone();
        let deallocator: NoCopyDeallocator = Box::new(move || {
            *o.lock() = Some(is_our_thread());
            // SAFETY: allocated above with `layout`, freed once, here.
            unsafe { std::alloc::dealloc(std::ptr::with_exposed_provenance_mut(mem_addr), layout) };
        });
        // SAFETY: the memory stays allocated until the deallocator runs.
        let bytes = unsafe { HostRange::new(HostPtr::from_ptr(mem), 16384) };
        let b = d
            .new_buffer_over_host_memory(OnFault::Die, bytes, 0, Some(deallocator))
            .unwrap()
            .unwrap();
        assert!(!is_our_thread());
        drop(b);
        assert_eq!(*ours.lock(), Some(true));
        assert!(!is_our_thread());
    }

    /// A refused no-copy buffer keeps its deallocator's captures for good, and with them the
    /// mapping the block would release (Metal never calls the block for nil)
    #[test]
    fn no_copy_nil_leaks_the_deallocator() {
        struct Guard(Arc<AtomicU32>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let d = dev();
        let layout = std::alloc::Layout::from_size_align(16384, 16384).unwrap();
        // SAFETY: a non-zero layout.
        let mem = unsafe { std::alloc::alloc_zeroed(layout) };
        for length in [0u64, 1 << 46] {
            let called = Arc::new(AtomicU32::new(0));
            let dropped = Arc::new(AtomicU32::new(0));
            let (c, g) = (called.clone(), Guard(dropped.clone()));
            let deallocator: NoCopyDeallocator = Box::new(move || {
                let _g = &g;
                c.fetch_add(1, Ordering::SeqCst);
            });
            // SAFETY: Metal refuses these lengths and never touches the memory.
            let bytes = unsafe { HostRange::new(HostPtr::from_ptr(mem), length) };
            let b = d
                .new_buffer_over_host_memory(OnFault::Die, bytes, 0, Some(deallocator))
                .unwrap();
            assert!(b.is_none());
            assert_eq!(called.load(Ordering::SeqCst), 0);
            // The captured mapping is never released.
            assert_eq!(dropped.load(Ordering::SeqCst), 0);
        }
        // SAFETY: no buffer wraps the memory; allocated above with `layout`.
        unsafe { std::alloc::dealloc(mem, layout) };
    }

    /// newBufferWithIOSurface:, newSharedTextureWithHandle:withResourceIndex: and
    /// newSharedEventWithHandle: send nil through, and Metal answers nil
    #[test]
    fn nil_arguments_reach_metal() {
        let d = dev();
        assert!(d.new_buffer_on_surface(None).unwrap().is_none());
        assert!(d.open_shared_texture(None, 0).unwrap().is_none());
        assert!(d.open_shared_event(None).is_none());
    }

    /// From a real handle
    #[test]
    fn texture_creators() {
        let d = dev();
        let td = texture_descriptor(4, 4, MTLStorageMode::Private);
        let t = d.new_texture(OnFault::Die, &td).unwrap().unwrap();
        assert_eq!(t.get().width(), 4);
        let s = d.new_shared_texture(&td).unwrap().unwrap();
        let h: MetalSharedTextureHandle =
            Obj::from_retained(s.get().newSharedTextureHandle().unwrap());
        let s2 = d.open_shared_texture(Some(&h), 0).unwrap().unwrap();
        assert_eq!(s2.get().height(), 4);
    }

    /// newSharedEventWithHandle: from a real handle opens the same event
    #[test]
    fn shared_event_with_handle() {
        let d = dev();
        let e = d.new_shared_event().unwrap();
        e.get().setSignaledValue(7);
        let h: MetalSharedEventHandle = Obj::from_retained(e.get().newSharedEventHandle());
        let e2 = d.open_shared_event(Some(&h)).unwrap();
        assert_eq!(e2.get().signaledValue(), 7);
    }

    /// newRasterizationRateMapWithDescriptor: from a descriptor with one layer
    #[test]
    fn rate_map_creator() {
        let d = dev();
        // SAFETY: a plain initializer.
        let layer = unsafe {
            MTLRasterizationRateLayerDescriptor::initWithSampleCount(
                MTLRasterizationRateLayerDescriptor::alloc(),
                MtlSizeObjc {
                    width: 2,
                    height: 2,
                    depth: 0,
                },
            )
        };
        let rd = MTLRasterizationRateMapDescriptor::new();
        // SAFETY: a plain property set and one layer.
        unsafe {
            rd.setScreenSize(MtlSizeObjc {
                width: 64,
                height: 64,
                depth: 0,
            });
            rd.setLayer_atIndex(Some(&layer), 0);
        }
        let rd = Obj::from_retained(rd);
        assert!(d.new_rate_map(&rd).unwrap().is_some());
    }

    /// Each creator from a value makes its object: the buffer's length and options, the heap's
    /// options as its descriptor reports them, and the states, the ICB and the pipeline
    #[test]
    fn creators_from_values() {
        use reims_vgpu_core::platform::resource::{GpuResource, Heap as _};
        use reims_vgpu_core::platform::values::{OptionSet, StencilDesc};
        let d = dev();
        let b = d
            .new_buffer(BufferDesc {
                length: 4096,
                options: vec![OptionSet::StorageMode(0), OptionSet::ResourceOptionsOr(0)],
                pinned_gpu_address: None,
                contents: None,
                no_copy: false,
                deallocator: None,
            })
            .unwrap()
            .unwrap();
        assert_eq!(b.get().length(), 4096);
        assert_eq!(GpuResource::storage_mode(&b), 0);
        let made = d
            .new_heap(&HeapDesc::<MetalPlatform> {
                size: 65536,
                heap_type: 0,
                options: vec![OptionSet::StorageMode(2), OptionSet::HazardTrackingMode(2)],
                address_ranges: None,
                pinned_gpu_address: None,
            })
            .unwrap();
        assert_eq!(made.descriptor_resource_options & 0xf0, 0x20);
        let heap = made.heap.unwrap();
        assert_eq!(heap.storage_mode(), 2);
        assert!(heap.get().size() >= 65536);
        assert!(
            d.new_sampler_state(&SamplerDesc {
                min_filter: Some(1),
                mag_filter: Some(1),
                ..SamplerDesc::default()
            })
            .unwrap()
            .is_some()
        );
        assert!(
            d.new_depth_stencil_state(&DepthStencilDesc {
                depth_compare_function: Some(1),
                depth_write_enabled: Some(true),
                front: Some(StencilDesc {
                    read_mask: 0xff,
                    write_mask: 0xff,
                    ..StencilDesc::default()
                }),
                back: None,
            })
            .unwrap()
            .is_some()
        );
        let icb = d
            .new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    max_vertex_buffer_bind_count: 1,
                    max_fragment_buffer_bind_count: 1,
                    max_scissor_rect_count: 1,
                    max_viewport_count: 1,
                    ..IcbDesc::default()
                },
                4,
                0,
            )
            .unwrap()
            .unwrap();
        assert_eq!(icb.get().size(), 4);
        let lib = library(&d);
        let (v, f) = (
            function(&lib, "displayPresentVertex"),
            function(&lib, "displayPresentFragment"),
        );
        let pipeline = d
            .new_render_pipeline_state(&RenderPipelineDesc::<MetalPlatform> {
                label: Some("values"),
                vertex: &v,
                fragment: &f,
                color0_pixel_format: MTLPixelFormat::BGRA8Unorm.0.to_u64(),
                blend_over: false,
                resource_index: None,
            })
            .unwrap();
        assert!(pipeline.is_some());
    }

    /// The caps are read once per device and shared
    #[test]
    fn caps_are_read_once() {
        let d = dev();
        let (a, b) = (d.caps(), d.caps());
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.registry_id, d.get().registryID());
        assert_eq!(a.name, Some(d.get().name().to_string()));
        assert_eq!(a.sample_counts[0], Ok(true));
        assert!(a.heap_buffer_page_private.is_ok());
    }

    /// The identity is Metal's: every family and feature set answered, Apple5 to Apple10 as the
    /// caps answer them, and the getters the device's own
    #[test]
    fn identity_caps_are_metals() {
        let d = dev();
        let caps = d.caps();
        let identity = d.identity_caps();
        assert!(identity.families.iter().all(Option::is_some));
        assert!(identity.feature_sets.iter().all(Option::is_some));
        // SDK_FAMILIES[4..10] are Apple5 to Apple10, core's FAMILIES[..6].
        for (k, answer) in caps.families.iter().take(6).enumerate() {
            assert_eq!(identity.families[4 + k], answer.clone().ok());
        }
        assert_eq!(
            identity.flags.has_unified_memory,
            Some(d.get().hasUnifiedMemory())
        );
        assert_eq!(
            identity.sizes.max_buffer_length,
            Some(d.get().maxBufferLength().to_u64())
        );
        assert_eq!(
            identity.architecture_name,
            Some(d.get().architecture().name().to_string())
        );
        // BGRA8Unorm (80): 16-byte texture buffers, and a 4-byte texel's 64x64 tile in 16 KiB.
        let bgra8 = identity.formats.iter().find(|f| f.format == 80).unwrap();
        assert_eq!(bgra8.texture_buffer_alignment, Some(16));
        assert_eq!(bgra8.sparse_tiles.unwrap()[0], [64, 64, 1]);
        // BC1_RGBA (130), which Metal asserts on, is not asked.
        let bc1 = identity.formats.iter().find(|f| f.format == 130).unwrap();
        assert_eq!(bc1.texture_buffer_alignment, None);
        // The driver's feature profile and FP configs, as the device answers them.
        assert_eq!(
            identity.sizes.feature_profile,
            Some(d.get().featureProfile())
        );
        assert!(identity.sizes.feature_profile.is_some_and(|p| p >= 8));
        assert_eq!(identity.sizes.half_fp_config, Some(d.get().halfFPConfig()));
        assert_eq!(
            identity.sizes.single_fp_config,
            Some(d.get().singleFPConfig())
        );
        assert_eq!(identity.sizes.double_fp_config, Some(0));
    }

    /// Garbage is the NSError; a compiled metallib (the suite's one-kernel fixture) is a library
    #[test]
    fn new_library() {
        let d = dev();
        let e = d.new_library(b"garbage").unwrap().unwrap_err();
        assert_eq!(e.domain, "MTLLibraryErrorDomain");
        assert_eq!(e.code, 1);
        assert!(
            e.description
                .starts_with("Error Domain=MTLLibraryErrorDomain"),
            "{e:?}"
        );
        let bytes = include_bytes!("../../../../tests/suite/fixtures/shaders/kernel.metallib");
        let lib = d.new_library(bytes).unwrap().unwrap().unwrap();
        assert_eq!(lib.get().functionNames().count(), 1);
    }

    /// newComputePipelineStateWithFunction:error: and
    /// newComputePipelineStateWithDescriptor:options:reflection:error
    #[test]
    fn compute_pipelines() {
        let d = dev();
        let lib = library(&d);
        let f = function(&lib, "BlitInRGB_2P_XR10_A8");
        assert!(
            d.new_compute_pipeline_state_for_function(Some(&f))
                .unwrap()
                .is_some()
        );
        let cd = MTLComputePipelineDescriptor::new();
        cd.setComputeFunction(Some(f.get()));
        let cd = Obj::from_retained(cd);
        assert!(
            d.new_compute_pipeline_state(Some(&cd), 0)
                .unwrap()
                .unwrap()
                .is_some()
        );
    }

    fn render_descriptor(lib: &MetalLibrary) -> MetalRenderPipelineDescriptor {
        let rd = MTLRenderPipelineDescriptor::new();
        rd.setVertexFunction(Some(function(lib, "displayPresentVertex").get()));
        rd.setFragmentFunction(Some(function(lib, "displayPresentFragment").get()));
        // SAFETY: index 0 exists on every render pipeline descriptor.
        unsafe { rd.colorAttachments().objectAtIndexedSubscript(0) }
            .setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        Obj::from_retained(rd)
    }

    /// newRenderPipelineStateWithDescriptor:options:reflection:error
    #[test]
    fn render_pipeline_with_reflection() {
        let d = dev();
        let lib = library(&d);
        let rd = render_descriptor(&lib);
        assert!(
            d.new_render_pipeline_state_from_descriptor(Some(&rd), 0)
                .unwrap()
                .unwrap()
                .is_some()
        );
    }

    /// The bytes of `-[_MTLDevice serialize…PipelineDescriptor:]` (test-only SPI).
    fn serialize(d: &MetalDevice, sel: objc2::runtime::Sel, desc: &AnyObject) -> Vec<u8> {
        // SAFETY: the device implements the selector (`@24@0:8@16`, measured); it returns a
        // dispatch_data_t, retained for the read.
        let p: *mut AnyObject =
            unsafe { msg_send![d.get(), performSelector: sel, withObject: desc] };
        assert!(!p.is_null());
        // SAFETY: a dispatch_data_t is a dispatch object; `DispatchData` is its Rust type.
        let data: &DispatchData = unsafe { &*p.cast::<DispatchData>() };
        data.to_vec()
    }

    /// newComputePipelineDescriptorWithSerializedData:deserializationContext: calls the resolver
    /// synchronously and rebuilds the descriptor with the function it answers
    #[test]
    fn compute_descriptor_from_serialized_data() {
        let d = dev();
        let lib = library(&d);
        let f = function(&lib, "BlitInRGB_2P_XR10_A8");
        let cd = MTLComputePipelineDescriptor::new();
        cd.setComputeFunction(Some(f.get()));
        let bytes = serialize(&d, objc2::sel!(serializeComputePipelineDescriptor:), &cd);
        let mut refs = Vec::new();
        let mut resolver = |_: Hold<'_>, r: u64| {
            refs.push(r);
            Ok(Some(f.clone_ref()))
        };
        let out = d
            .deserialize_compute_pipeline_descriptor(Hold::NONE, &bytes, &mut resolver)
            .unwrap()
            .unwrap();
        assert_eq!(refs, vec![0]);
        assert_eq!(
            out.get().computeFunction().unwrap().name().to_string(),
            "BlitInRGB_2P_XR10_A8"
        );
    }

    /// A throw inside the resolver leaves functionWithFunctionRef: as an Objective-C exception;
    /// Metal lets it through, and the caller's scope gets the resolver's own throw back
    #[test]
    fn a_resolver_throw_reaches_the_caller_unchanged() {
        let d = dev();
        let lib = library(&d);
        let f = function(&lib, "BlitInRGB_2P_XR10_A8");
        let cd = MTLComputePipelineDescriptor::new();
        cd.setComputeFunction(Some(f.get()));
        let bytes = serialize(&d, objc2::sel!(serializeComputePipelineDescriptor:), &cd);
        let mut resolver = |_: Hold<'_>, _r: u64| -> RaiseResult<Option<MetalFunction>> {
            Err(reims_vgpu_core::testkit::capture::raise(
                OnFault::Return,
                "resolver",
                "boom",
            ))
        };
        let hold = Hold::NONE;
        let t = d
            .deserialize_compute_pipeline_descriptor(hold, &bytes, &mut resolver)
            .err()
            .unwrap();
        assert!(matches!(
            t.origin(),
            reims_vgpu_core::raise::Origin::Core(_)
        ));
        assert_eq!(t.detail(), Some("boom"));
        assert_eq!((t.site().file, t.site().line), ("resolver", 0));
    }

    /// The resolver runs under a handler of its own, so its throw crosses Metal; once Metal has
    /// unwound it is the caller's raise, which names the throw's own site
    #[test]
    fn a_resolver_throw_is_the_callers_after_metal() {
        let d = dev();
        let lib = library(&d);
        let f = function(&lib, "BlitInRGB_2P_XR10_A8");
        let cd = MTLComputePipelineDescriptor::new();
        cd.setComputeFunction(Some(f.get()));
        let bytes = serialize(&d, objc2::sel!(serializeComputePipelineDescriptor:), &cd);
        let mut calls = 0u32;
        let mut resolver = |_: Hold<'_>, _r: u64| -> RaiseResult<Option<MetalFunction>> {
            calls += 1;
            Err(reims_vgpu_core::testkit::capture::raise(
                OnFault::Return,
                "resolver",
                "boom",
            ))
        };
        let site = d
            .deserialize_compute_pipeline_descriptor(Hold::NONE, &bytes, &mut resolver)
            .err()
            .unwrap()
            .site();
        assert_eq!(calls, 1);
        assert_eq!(
            (site.kind, site.file, site.line),
            (DeathKind::UnhandledRaise, "resolver", 0)
        );
    }

    /// Were Metal to catch the resolver's throw itself and answer, the throw is lost, as the
    /// exception would be to Apple's framework: the call answers what Metal made, and what the
    /// raise carries lands in the hold's holder, not when the context goes
    #[test]
    fn a_throw_metal_catches_lands_in_the_hold() {
        use std::panic::AssertUnwindSafe;
        use std::sync::atomic::AtomicBool;

        use objc2::runtime::NSObject;
        use reims_vgpu_core::rt::Held;

        struct Probe(Arc<AtomicBool>);
        impl Drop for Probe {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let d = dev();
        let released = Arc::new(AtomicBool::new(false));
        let held = Held::new();
        let carried = Probe(Arc::clone(&released));
        let mut carried = Some(carried);
        let mut resolver = |_: Hold<'_>, _r: u64| -> RaiseResult<Option<MetalFunction>> {
            let raise =
                reims_vgpu_core::testkit::capture::raise(OnFault::Return, "resolver", "boom");
            Err(raise.defer(carried.take().unwrap()))
        };
        // Metal's stand-in: one resolver call, whose throw it catches itself, then nil.
        let made = deserialize(
            Hold::of(&held),
            &d,
            "newComputePipelineDescriptorWithSerializedData:deserializationContext:",
            DispatchData::from_bytes(&[]),
            &mut resolver,
            |_, _, context| {
                let caught = objc2::exception::catch(AssertUnwindSafe(|| {
                    // SAFETY: the context answers `functionWithFunctionRef:` (`@24@0:8Q16`).
                    let f: *mut AnyObject =
                        unsafe { msg_send![context, functionWithFunctionRef: 1u64] };
                    f
                }));
                assert!(caught.is_err(), "the resolver threw");
                None::<Retained<NSObject>>
            },
        );
        assert!(matches!(made, Ok(None)), "{made:?}");
        assert!(!released.load(Ordering::SeqCst));
        drop(held);
        assert!(released.load(Ordering::SeqCst));
    }

    /// A death in an isolated task's resolver while Metal deserializes, and a stop there once the
    /// task has ended, each end the task once Metal has returned: nothing after the
    /// deserialization runs, the resolver is called once, and Metal deserializes again after
    #[test]
    fn a_death_or_stop_in_the_resolver_ends_the_task_after_metal() {
        use reims_vgpu_core::raise::death::DeathSite;
        use reims_vgpu_core::tasks::isolation::TaskLife;
        use reims_vgpu_core::tasks::isolation::life::stop_if_task_ended;

        let d = dev();
        let lib = library(&d);
        let v = function(&lib, "displayPresentVertex");
        let rd = render_descriptor(&lib);
        let bytes = serialize(
            &d,
            objc2::sel!(serializeRenderPipelineDescriptor:),
            rd.get(),
        );
        for stop in [false, true] {
            let life = TaskLife::new();
            let mut calls = 0u32;
            let mut ended = |_: Hold<'_>, _r: u64| -> RaiseResult<Option<MetalFunction>> {
                calls += 1;
                if stop {
                    assert!(TaskLife::current().unwrap().exit());
                    stop_if_task_ended();
                } else {
                    reims_vgpu_core::fatal!(DeathSite {
                        kind: DeathKind::Abort,
                        file: "resolver.test",
                        line: 0,
                    });
                }
                Ok(None)
            };
            let mut after = false;
            let r = life.run(|| {
                let _ = d.deserialize_render_pipeline_descriptor(Hold::NONE, &bytes, &mut ended);
                after = true;
            });
            assert_eq!((r, after, calls), (None, false, 1), "stop: {stop}");
            assert_eq!(life.death().is_some(), !stop);
        }
        let mut resolver = |_: Hold<'_>, _r: u64| Ok(Some(v.clone_ref()));
        assert!(
            d.deserialize_render_pipeline_descriptor(Hold::NONE, &bytes, &mut resolver)
                .unwrap()
                .is_some()
        );
    }

    /// newRenderPipelineDescriptorWithSerializedData:deserializationContext: (a resolver call per
    /// function) and newTileRenderPipelineDescriptorWithSerializedData:… (none)
    #[test]
    fn render_and_tile_descriptors_from_serialized_data() {
        let d = dev();
        let lib = library(&d);
        let rd = render_descriptor(&lib);
        let bytes = serialize(
            &d,
            objc2::sel!(serializeRenderPipelineDescriptor:),
            rd.get(),
        );
        let v = function(&lib, "displayPresentVertex");
        let mut calls = 0u32;
        let mut resolver = |_: Hold<'_>, _r: u64| {
            calls += 1;
            Ok(Some(v.clone_ref()))
        };
        assert!(
            d.deserialize_render_pipeline_descriptor(Hold::NONE, &bytes, &mut resolver)
                .unwrap()
                .is_some()
        );
        assert_eq!(calls, 2);
        let td = MTLTileRenderPipelineDescriptor::new();
        let bytes = serialize(&d, objc2::sel!(serializeTileRenderPipelineDescriptor:), &td);
        let mut none = |_: Hold<'_>, _r: u64| -> RaiseResult<Option<MetalFunction>> {
            panic!("no function to resolve")
        };
        let tile = d
            .deserialize_tile_render_pipeline_descriptor(
                Hold::NONE,
                &bytes,
                bytes.len() as u64,
                &mut none,
            )
            .unwrap()
            .unwrap();
        // A tile descriptor with no function: Metal's own assertion would abort the state
        // creation, so only the descriptor's class is checked here.
        let obj: &AnyObject = tile.get();
        assert!(
            obj.class()
                .name()
                .to_bytes()
                .starts_with(b"MTLTileRenderPipelineDescriptor")
        );
    }

    /// allowLibrariesFromOtherPlatforms and reserveResourceIndicesForResourceType:… (no raise; the
    /// indices pointer and count are passed)
    #[test]
    fn side_effect_sends() {
        let d = dev();
        d.allow_foreign_libraries();
        assert_eq!(d.reserve_resource_indices(1, &[]), Ok(()));
        assert_eq!(d.reserve_resource_indices(1, &[4000]), Ok(()));
    }

    /// BGRA8Unorm: blockBytes 4, flags 0x703019f; Invalid: all zero, named
    #[test]
    fn pixel_format_info() {
        let d = dev();
        let i = d.pixel_format_info(80);
        assert_eq!(i.name.as_deref(), Some("MTLPixelFormatBGRA8Unorm"));
        assert_eq!((i.block_bytes, i.flags), (4, 0x703019f));
        let z = d.pixel_format_info(0);
        assert_eq!(z.name.as_deref(), Some("MTLPixelFormatInvalid"));
        assert_eq!(z.flags, 0);
    }

    /// BGRA8 16x16: 4 bytes per pixel, aligned height 16, a row of at least 64 bytes
    #[test]
    fn texture_level_info() {
        let d = dev();
        let i = d.texture_level_info(&TextureLevelQuery {
            pixel_format: 80,
            width: 16,
            height: 16,
            depth: 1,
            array_length: 1,
            options: 0,
            flags: 0,
        });
        assert_eq!(i.bytes_per_pixel, 4);
        assert_eq!(i.aligned_height, 16);
        assert!(i.bytes_per_row >= 64, "{i:?}");
    }

    /// Runs `body` in a child copy of this test binary and returns how it ended: Metal's own
    /// assertions end the process, which only a child can show.
    pub(crate) fn in_child(test: &str, body: impl FnOnce()) -> Option<std::process::ExitStatus> {
        if std::env::var("RVG_CHILD").as_deref() == Ok(test) {
            body();
            std::process::exit(0);
        }
        Some(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", test, "--nocapture", "--test-threads=1"])
                .env("RVG_CHILD", test)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .unwrap(),
        )
    }

    /// newTextureWithDescriptor:iosurface:plane: with a nil surface is Metal's own assertion (a
    /// process death: "IOSurface argument must not be NULL")
    #[test]
    fn texture_with_nil_iosurface_aborts() {
        use std::os::unix::process::ExitStatusExt as _;
        let st = in_child(
            "gpu::device::tests::texture_with_nil_iosurface_aborts",
            || {
                let d = dev();
                let td = texture_descriptor(4, 4, MTLStorageMode::Shared);
                let _ = d.new_texture_on_surface(&td, None, 0);
            },
        );
        assert_eq!(st.and_then(|s| s.signal()), Some(6));
    }

    /// The `dispatch_data_t`s made with a block destructor and a NULL queue: the IR's, over its
    /// `NSData`, and the tile descriptor's `^{}`
    mod dispatch_data_block_destructor {
        use super::*;
        use std::sync::mpsc;
        use std::thread::ThreadId;
        use std::time::Duration;

        const WAIT: Duration = Duration::from_secs(10);
        const QOS_CLASS_USER_INITIATED: u32 = 0x19;

        unsafe extern "C" {
            /// `dispatch_queue_get_label(DISPATCH_CURRENT_QUEUE_LABEL)` (`NULL`).
            fn dispatch_queue_get_label(queue: *const c_void) -> *const libc::c_char;
            /// `qos_class_t qos_class_self(void)`, as its raw value.
            fn qos_class_self() -> u32;
            /// `pthread_set_qos_class_self_np(qos_class_t, int)`, the class as its raw value.
            fn pthread_set_qos_class_self_np(
                class: u32,
                relative_priority: libc::c_int,
            ) -> libc::c_int;
            /// `size_t malloc_size(const void *)`: 0 for a pointer malloc does not own.
            fn malloc_size(ptr: *const c_void) -> usize;
        }

        /// What the running thread sees: its id, the current queue's label, its QoS.
        fn seen() -> (ThreadId, String, u32) {
            // SAFETY: libdispatch returns a NUL-terminated label (possibly empty).
            let label = unsafe { std::ffi::CStr::from_ptr(dispatch_queue_get_label(null_mut())) }
                .to_string_lossy()
                .into_owned();
            // SAFETY: a query of the running thread.
            (std::thread::current().id(), label, unsafe {
                qos_class_self()
            })
        }

        /// Reports [`seen`] when dropped, on the thread whose release drops it.
        struct Probe(mpsc::Sender<(ThreadId, String, u32)>);

        impl Drop for Probe {
            fn drop(&mut self) {
                let _ = self.0.send(seen());
            }
        }

        /// Associates `probe` with `object`: the runtime releases it when the object ends, on the
        /// thread that runs it.
        fn attach(object: &NSData, probe: Probe) {
            static KEY: u8 = 0;
            let holder = RcBlock::new(move || {
                let _ = &probe;
            });
            // SAFETY: a live object, a static key, and a block (an Objective-C object) the
            // association retains.
            unsafe {
                objc2::ffi::objc_setAssociatedObject(
                    std::ptr::from_ref(object).cast_mut().cast(),
                    (&raw const KEY).cast(),
                    RcBlock::as_ptr(&holder).cast(),
                    objc2::ffi::OBJC_ASSOCIATION_RETAIN_NONATOMIC,
                );
            }
        }

        /// A data reference moved to another thread to be released there.
        struct SendData(DispatchRetained<DispatchData>);

        // SAFETY: a dispatch data object is immutable and its retain and release are
        // thread-safe; the reference is only released on the other thread.
        unsafe impl Send for SendData {}

        /// The data's one contiguous region.
        fn region(d: &DispatchData) -> (*const c_void, usize) {
            let (mut p, mut n) = (std::ptr::null(), 0);
            // SAFETY: valid out-pointers; the map is released at once.
            drop(unsafe { d.map(&mut p, &mut n) });
            (p, n)
        }

        /// new_library's IR: an `NSData` over a `malloc` copy of the bytes, the data
        /// spanning its bytes (no second copy); after the data's last release, a
        /// `com.apple.root.default-qos` GCD worker, not the releasing thread, runs the destructor
        /// block, whose release of the `NSData` frees the IR there, at the releaser's QoS
        #[test]
        fn the_ir_is_freed_on_a_default_qos_root_worker() {
            let (tx, rx) = mpsc::channel();
            let ns = ir_ns_data(b"metallib IR");
            // SAFETY: `-bytes` of our own NSData, never mutated.
            let bytes: *const c_void = unsafe { ns.as_bytes_unchecked() }.as_ptr().cast();
            // SAFETY: the data's 11 bytes.
            let copied = unsafe { std::slice::from_raw_parts(bytes.cast::<u8>(), 11) };
            assert_eq!(copied, b"metallib IR");
            // SAFETY: a query of a live pointer.
            assert!(unsafe { malloc_size(bytes) } >= 11, "a malloc copy");
            attach(&ns, Probe(tx));
            let data = SendData(ir_dispatch_data(ns));
            assert_eq!(
                region(&data.0),
                (bytes, 11),
                "the data spans the NSData's bytes"
            );
            let (done, released) = mpsc::channel();
            #[allow(clippy::disallowed_methods)] // test helper thread: no Objective-C runs here
            let _ = std::thread::Builder::new()
                .name("releaser".into())
                .spawn(move || {
                    // SAFETY: a valid QoS class, relative priority 0.
                    let rc = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0) };
                    assert_eq!(rc, 0);
                    let me = std::thread::current().id();
                    // The data's last release (Metal's, when the library goes).
                    drop(data);
                    done.send(me).unwrap();
                })
                .unwrap();
            let releaser = released.recv_timeout(WAIT).unwrap();
            let (worker, label, qos) = rx.recv_timeout(WAIT).unwrap();
            assert_ne!(worker, releaser);
            assert_ne!(worker, std::thread::current().id());
            assert_eq!(label, "com.apple.root.default-qos");
            assert_eq!(qos, QOS_CLASS_USER_INITIATED);
        }

        /// A block destructor with a NULL queue, as both forms pass it, is run by a
        /// `com.apple.root.default-qos` GCD worker after the data's last release, not by the
        /// releasing thread
        #[test]
        fn a_block_destructor_runs_on_a_default_qos_root_worker() {
            static BYTES: [u8; 4] = *b"tile";
            let (tx, rx) = mpsc::channel();
            let block = RcBlock::new(move || {
                let _ = tx.send(seen());
            });
            // SAFETY: static bytes outlive the data.
            let data =
                unsafe { data_with_block_destructor(BYTES.as_ptr().cast(), BYTES.len(), &block) };
            drop(block);
            assert_eq!(region(&data), (BYTES.as_ptr().cast(), 4), "no copy");
            drop(data);
            let (worker, label, _) = rx.recv_timeout(WAIT).unwrap();
            assert_ne!(worker, std::thread::current().id());
            assert_eq!(label, "com.apple.root.default-qos");
        }

        /// deserialize_tile_render_pipeline_descriptor's data spans the descriptor's
        /// bytes at the start of its own `malloc(length)` buffer of the cursor's remaining bytes
        /// (no second copy), its destructor is `^{}`, a global block (`Block_copy` allocates
        /// nothing), and the whole buffer is still allocated after the data's last release: nothing
        /// frees it
        #[test]
        fn the_tile_descriptor_leaks_its_malloc_copy() {
            let data = tile_descriptor_data(b"tile descriptor", 0x1000);
            let (p, n) = region(&data);
            assert_eq!(n, 15);
            // SAFETY: a query of a live pointer.
            assert!(
                unsafe { malloc_size(p) } >= 0x1000,
                "malloc(length), not the descriptor's"
            );
            // SAFETY: the data's 15 bytes.
            assert_eq!(
                unsafe { std::slice::from_raw_parts(p.cast::<u8>(), n) },
                b"tile descriptor"
            );
            // SAFETY: a query of a live pointer.
            assert!(unsafe { malloc_size(p) } >= 15, "a malloc copy");
            // SAFETY: a global block is an Objective-C object.
            let class = unsafe {
                std::ffi::CStr::from_ptr(objc2::ffi::object_getClassName(
                    std::ptr::from_ref(&*TILE_DESTRUCTOR).cast(),
                ))
            };
            assert_eq!(class.to_str().unwrap(), "__NSGlobalBlock__");
            // The dispose runs here, synchronously; libdispatch only submits the empty block.
            drop(data);
            // SAFETY: a query of a pointer malloc still owns.
            assert!(unsafe { malloc_size(p) } >= 15, "never freed");
        }
    }
}
