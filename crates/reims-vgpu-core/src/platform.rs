//! The platform traits (architecture 4): `Platform`, `Handle`, `ObjId`, `nil_send!`, and the
//! `ProcessState` re-export.
//!
//! Core is generic over one `P: Platform` (D3). Its associated types name every Metal object kind
//! core touches, so a handle flows into another Metal call with its kind checked at compile time.
//! Two implementations: `reims_vgpu_metal::MetalPlatform` and
//! `core::testkit::platform::FakePlatform`.

pub mod blit;
pub mod caps;
pub mod command;
pub mod compute;
pub mod descriptor;
pub mod device;
pub mod event;
pub mod host_range;
pub mod icb;
pub mod os;
pub mod rate_map;
pub mod render;
pub mod resource;
pub mod types;
pub mod values;

pub use crate::process::ProcessState;
use crate::raise::OnFault;
pub use crate::raise::Raise;
pub use crate::raise::RaiseResult;
pub use event::EventRef;
pub use host_range::HostRange;
pub use types::NilDefault;

use crate::host::HostBlocks;
use os::surface::{IoSurfaces, Surface};
use os::vm::HostVm;
use os::{Clock, Threads};
use types::{MtlSize, ResourceAddressRange};

/// An object's identity: its pointer, for the fake's call log and for core's pointer-equality
/// checks. Name from architecture 4.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ObjId(pub u64);

/// A reference-counted platform object. NOT `Clone`: `.clone_ref()` is the one
/// explicit way to take another counted reference, and `Drop` is the release.
pub trait Handle: Send + Sync + 'static {
    /// One more reference to the same object (`[x retain]`, `CFRetain`).
    fn clone_ref(&self) -> Self
    where
        Self: Sized;
    /// The object's identity.
    fn id(&self) -> ObjId;
}

/// The Metal object families plus the non-handle services (architecture 4.1).
///
/// Associated types without a selector core sends to them (`Event`, `Fence`, `Function`, the two
/// shared handles, the address-range array, the event listener) are bounded by `Handle` alone.
pub trait Platform: Sized + Send + Sync + 'static {
    // Metal objects.
    /// `id<MTLDevice>`.
    type Device: device::GpuDevice<Self>;
    /// `id<MTLCommandQueue>`.
    type CommandQueue: command::CommandQueue<Self>;
    /// `id<MTLCommandBuffer>`.
    type CommandBuffer: command::CommandBuffer<Self>;
    /// `id<MTLBlitCommandEncoder>`.
    type BlitEncoder: blit::BlitEncoder<Self>;
    /// `id<MTLComputeCommandEncoder>`.
    type ComputeEncoder: compute::ComputeEncoder<Self>;
    /// `id<MTLRenderCommandEncoder>`.
    type RenderEncoder: render::RenderEncoder<Self>;
    /// `id<MTLBuffer>`.
    type Buffer: resource::Buffer<Self>;
    /// `id<MTLTexture>`.
    type Texture: resource::Texture<Self>;
    /// `id<MTLHeap>`.
    type Heap: resource::Heap<Self>;
    /// `id<MTLSamplerState>`.
    type SamplerState: resource::SamplerState;
    /// `id<MTLDepthStencilState>`.
    type DepthStencilState: resource::DepthStencilState;
    /// `id<MTLFence>`.
    type Fence: Handle;
    /// `id<MTLEvent>`.
    type Event: Handle;
    /// `id<MTLSharedEvent>`.
    type SharedEvent: event::SharedEvent<Self>;
    /// `MTLSharedEventHandle *`.
    type SharedEventHandle: Handle;
    /// `MTLSharedEventListener *` (the deserializer creates one and never uses it; the graphics
    /// lane's notifications run on one, [`event::SharedEvent::notify_listener`]).
    type SharedEventListener: Handle;
    /// `MTLSharedTextureHandle *`.
    type SharedTextureHandle: Handle;
    /// `id<MTLLibrary>`.
    type Library: resource::Library<Self>;
    /// `id<MTLFunction>`.
    type Function: Handle;
    /// `id<MTLComputePipelineState>`.
    type ComputePipeline: resource::ComputePipeline;
    /// `id<MTLRenderPipelineState>`.
    type RenderPipeline: resource::RenderPipeline;
    /// `id<MTLRasterizationRateMap>`.
    type RateMap: rate_map::RateMap<Self>;
    /// `id<MTLIndirectCommandBuffer>`.
    type Icb: icb::IndirectCommandBuffer<Self>;
    /// `id<MTLIndirectRenderCommand>`.
    type IndirectRenderCommand: icb::IndirectRenderCommand<Self>;
    /// `id<MTLIndirectComputeCommand>`.
    type IndirectComputeCommand: icb::IndirectComputeCommand<Self>;

    // The descriptors core keeps as objects (D5): the others are values (`values`).
    /// `MTLTextureDescriptor *`.
    type TextureDescriptor: descriptor::TextureDescriptor<Self>;
    /// `MTLRenderPassDescriptor *`.
    type RenderPassDescriptor: descriptor::RenderPassDescriptor<Self>;
    /// `MTLRenderPipelineDescriptor *`.
    type RenderPipelineDescriptor: descriptor::RenderPipelineDescriptor<Self>;
    /// `MTLComputePipelineDescriptor *`.
    type ComputePipelineDescriptor: descriptor::ComputePipelineDescriptor<Self>;
    /// `MTLTileRenderPipelineDescriptor *`.
    type TileRenderPipelineDescriptor: descriptor::TileRenderPipelineDescriptor<Self>;
    /// `MTLRasterizationRateMapDescriptor *`.
    type RateMapDescriptor: rate_map::RateMapDescriptor<Self>;
    /// `MTLRasterizationRateLayerDescriptor *`.
    type RateMapLayerDescriptor: rate_map::RateMapLayerDescriptor;
    /// The private `MTLResourceAddressRangeArray *`.
    type AddressRangeArray: Handle;

    // Non-Metal handles.
    /// `IOSurfaceRef` / `IOSurface *`.
    type Surface: Surface;
    /// Time and timed waits (not object safe: its waits are generic).
    type Clock: Clock;

    // Class methods, one per class core instantiates.
    /// sel: new — (`[MTLRenderPassDescriptor new]`, +1).
    fn new_render_pass_descriptor(&self) -> Self::RenderPassDescriptor;
    /// sel: new — (`MTLSharedEventListener`: the deserializer's, created and never used, and the
    /// graphics lane's, which its notifications run on).
    fn new_shared_event_listener(&self) -> Self::SharedEventListener;
    /// sel: renderPassDescriptor — (`+[MTLRenderPassDescriptor
    /// renderPassDescriptor]`, autoreleased).
    fn render_pass_descriptor(&self) -> Self::RenderPassDescriptor;
    /// sel: texture2DDescriptorWithPixelFormat:width:height:mipmapped:
    /// (autoreleased).
    /// Reached both under a handler and with none: `on` says which this call is.
    fn texture_2d_descriptor(
        &self,
        on: OnFault,
        pixel_format: u64,
        width: u64,
        height: u64,
        mipmapped: bool,
    ) -> RaiseResult<Self::TextureDescriptor>;
    /// sel: initWithSampleCount:
    /// (`[[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:]`). The sample storage
    /// is filled in place afterwards, through the layer's own methods. Nil is Metal's answer for
    /// guest-controlled counts it refuses (measured on macOS 26.7: `{0,0,0}` and `{~0,~0,0}` give
    /// nil, `{100000,100000,0}` an object); core checks it at both sites (returns "Failed to create
    /// layer descriptor", throws).
    /// Reached both under a handler and with none: `on` says which this call is.
    fn new_rate_map_layer(
        &self,
        on: OnFault,
        sample_count: MtlSize,
    ) -> RaiseResult<Option<Self::RateMapLayerDescriptor>>;
    /// sel: initWithRanges:count: SPI
    /// (`[[MTLResourceAddressRangeArray alloc] initWithRanges:count:]`). The runtime encodes the
    /// ranges as `^{MTLAddressRange=QQ}`; the array copies them, so `ranges` may live on the stack
    /// (D41; the metal crate's test `new_address_range_array`).
    fn new_address_range_array(
        &self,
        ranges: &[ResourceAddressRange],
    ) -> RaiseResult<Self::AddressRangeArray>;

    // Made descriptors and devices, from values.
    /// A texture descriptor made from `desc`: a new descriptor with the value's setters sent in
    /// [`values::TextureDesc`]'s order (`setPixelFormat:` with the raw value, so a pixel format
    /// Metal refuses aborts here).
    fn texture_descriptor(&self, desc: &values::TextureDesc) -> Self::TextureDescriptor;
    /// A rasterization-rate-map descriptor made from `desc`; its layers are set on it after.
    fn rate_map_descriptor(&self, desc: &values::RateMapDesc) -> Self::RateMapDescriptor;
    /// The device with `registry_id` (+1), or `None`: in one process, the instance the host's own
    /// device is.
    fn open_device(&self, registry_id: u64) -> Option<Self::Device>;

    // Services.
    /// IOSurface creation.
    fn surfaces(&self) -> &dyn IoSurfaces<Self>;
    /// Mach VM.
    fn vm(&self) -> &dyn HostVm;
    /// Time.
    fn clock(&self) -> &Self::Clock;
    /// Every OS thread core creates (D35).
    fn threads(&self) -> &dyn Threads;
    /// The process-wide state (D17).
    fn globals(&self) -> &ProcessState<Self>;
    /// Starts isolated GPU task services: `tasks::isolation::TaskThreads`, each task's service on
    /// threads of its own (the testkit's records what it launched).
    fn launcher(&self) -> &dyn crate::tasks::launcher::TaskLauncher<Self>;

    /// The task and memory blocks a device without a memory map keeps (`createTask`,
    /// `destroyTask`, `mapMemory`, `unmapMemory` and `readMemory`; the others are `None`), asked
    /// once, as the device copies the host's. By default the host's own, unchanged.
    ///
    /// A platform may answer blocks of its own around the host's, to learn of the host's mapping
    /// work. Each calls the host's block once per call, with the same arguments, and answers its
    /// answer; holds no other block, so the host sees the blocks' `destroy` callbacks in the order
    /// it does ([`crate::host::callbacks`]); and a nil block stays nil. The answer's other blocks
    /// are not read.
    fn host_blocks(&self, blocks: HostBlocks) -> HostBlocks {
        blocks
    }
}

/// A send to nil does nothing and returns zero (Objective-C). Used exactly where core sends to a
/// receiver it has not checked: `nil_send!(recv, method(args…))` where `recv: Option<X>`. A nil
/// receiver yields [`NilDefault::nil_default`]: zero, nil, or `Ok(zero)` for a method that returns
/// `RaiseResult<T>` (a send to nil never raises). Objective-C evaluates the arguments even for nil;
/// this macro does not, so pass values already computed, in order.
#[macro_export]
macro_rules! nil_send {
    ($recv:expr, $method:ident ( $($arg:expr),* $(,)? )) => {
        match &$recv {
            Some(r) => r.$method($($arg),*),
            None => $crate::platform::NilDefault::nil_default(),
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct Obj {
        id: u64,
        calls: AtomicU32,
    }

    impl Handle for Obj {
        fn clone_ref(&self) -> Self {
            Obj {
                id: self.id,
                calls: AtomicU32::new(0),
            }
        }
        fn id(&self) -> ObjId {
            ObjId(self.id)
        }
    }

    impl Obj {
        fn length(&self) -> u64 {
            self.calls.fetch_add(1, Ordering::Relaxed);
            7
        }
        fn fill(&self, v: u8) -> RaiseResult<u8> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(v)
        }
        fn child(&self) -> Option<Obj> {
            Some(Obj {
                id: self.id.wrapping_add(1),
                calls: AtomicU32::new(0),
            })
        }
    }

    /// A live receiver gets the send; nil gets zero, nil or Ok(zero) and no send
    #[test]
    fn nil_send_sends_or_returns_zero() {
        let live = Some(Obj {
            id: 1,
            calls: AtomicU32::new(0),
        });
        assert_eq!(nil_send!(live, length()), 7);
        assert_eq!(nil_send!(live, fill(3)), Ok(3));
        assert_eq!(nil_send!(live, child()).map(|c| c.id()), Some(ObjId(2)));
        assert_eq!(
            live.as_ref().map(|o| o.calls.load(Ordering::Relaxed)),
            Some(2)
        );

        let nil: Option<Obj> = None;
        assert_eq!(nil_send!(nil, length()), 0);
        assert_eq!(nil_send!(nil, fill(3)), Ok(0));
        assert!(nil_send!(nil, child()).is_none());
    }

    /// clone_ref() and its old name retained() name the same object
    #[test]
    fn handle_clone_ref_keeps_identity() {
        let o = Obj {
            id: 9,
            calls: AtomicU32::new(0),
        };
        assert_eq!(o.clone_ref().id(), o.id());
        assert_eq!(o.clone_ref().id(), o.id());
        assert_eq!(ObjId::default(), ObjId(0));
    }
}
