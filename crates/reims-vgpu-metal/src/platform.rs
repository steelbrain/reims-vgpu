//! `MetalPlatform`: `impl Platform`, the process singleton, `ProcessState`.
//!
//! Every Metal handle is an [`Obj<T>`]: exactly one `Retained`, not `Clone` (D6), `.clone_ref()` is
//! `-retain` and `Drop` is `-release` (leaked instead when an isolated task that died lets go of
//! it). The associated types are aliases of `Obj` over objc2-metal's protocol objects and classes,
//! or over the private classes in `spi::classes`. One `MetalPlatform` is one process's (D17);
//! `reims-vgpu`'s singleton makes exactly one.

use std::fmt;
use std::mem::ManuallyDrop;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObjectProtocol};
use objc2::{AnyThread as _, Message};
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLDevice as _, MTLPixelFormat, MTLRasterizationRateLayerDescriptor, MTLRenderPassDescriptor,
    MTLSharedEventListener, MTLTextureDescriptor,
};
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::Threads;
use reims_vgpu_core::platform::os::surface::IoSurfaces;
use reims_vgpu_core::platform::os::vm::HostVm;
use reims_vgpu_core::platform::types::{MtlSize, NsErrorText, ResourceAddressRange};
use reims_vgpu_core::platform::values::{RateMapDesc, TextureDesc};
use reims_vgpu_core::platform::{Handle, ObjId, Platform, ProcessState};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::tasks::launcher::TaskLauncher;

use crate::gpu::command::{MetalCommandBuffer, MetalCommandQueue};
use crate::gpu::desc::{
    MetalComputePipelineDescriptor, MetalRenderPassDescriptor, MetalRenderPipelineDescriptor,
    MetalTextureDescriptor, MetalTileRenderPipelineDescriptor,
};
use crate::gpu::device::MetalDevice;
use crate::gpu::encode_blit::MetalBlitEncoder;
use crate::gpu::encode_compute::MetalComputeEncoder;
use crate::gpu::encode_render::MetalRenderEncoder;
use crate::gpu::event::{
    MetalEvent, MetalSharedEvent, MetalSharedEventHandle, MetalSharedEventListener,
};
use crate::gpu::icb::{MetalIcb, MetalIndirectComputeCommand, MetalIndirectRenderCommand};
use crate::gpu::rate_map::{MetalRateMap, MetalRateMapDescriptor, MetalRateMapLayerDescriptor};
use crate::gpu::resource::{
    MetalBuffer, MetalComputePipeline, MetalDepthStencilState, MetalFence, MetalFunction,
    MetalHeap, MetalLibrary, MetalRenderPipeline, MetalSamplerState, MetalSharedTextureHandle,
    MetalTexture,
};
use crate::msg::encode::{Enc, ToMtl};
use crate::msg::{self, Policy};
use crate::os::clock::MetalClock;
use crate::os::iosurface::{MetalSurface, MetalSurfaces};
use crate::os::threads::MetalThreads;
use crate::os::vm::MetalVm;
use crate::spi::classes::MTLResourceAddressRangeArray;
use crate::spi::methods::RateLayerDescriptorSpi;

/// One reference to an Objective-C object: a Metal handle (architecture 4.1).
///
/// Not `Clone` (D6): [`Handle::clone_ref`] is the only retain. `Drop` is the release, except
/// while the thread unwinds or releases an isolated task that died
/// ([`leaks_objects`](reims_vgpu_core::tasks::isolation::life::leaks_objects)): the reference
/// is then leaked, as a crashed process never released it. objc2 answers every result as one
/// reference of the caller's, a `+1` result's (`new…`, `init…`, `copy…`) or a retain of a `+0`
/// one, and [`Obj::from_retained`] takes it over. A `+0` result's own autorelease, where objc2
/// does not elide it (a dev build), stays in the thread's pool until the pool pops.
pub struct Obj<T: Message>(ManuallyDrop<Retained<T>>);

// SAFETY: Metal objects are shared between threads with retain/release alone; the runtime's
// reference counting is atomic, and every send core makes on a shared object is one Metal documents
// as thread-safe or one core serialises itself. `Obj` adds no state of its own.
unsafe impl<T: Message> Send for Obj<T> {}
// SAFETY: as for `Send`.
unsafe impl<T: Message> Sync for Obj<T> {}

impl<T: Message> Obj<T> {
    /// The handle over `r`.
    fn wrap(r: Retained<T>) -> Obj<T> {
        Obj(ManuallyDrop::new(r))
    }

    /// Takes over one reference.
    pub fn from_retained(r: Retained<T>) -> Obj<T> {
        Obj::wrap(r)
    }

    /// Takes over a `+1` pointer (`new…`, `init…`, `copy…`); `None` for nil.
    ///
    /// # Safety
    /// `ptr` is nil or a valid object of type `T` whose reference the caller owns.
    pub unsafe fn from_owned_ptr(ptr: *mut T) -> Option<Obj<T>> {
        // SAFETY: the caller's contract.
        unsafe { Retained::from_raw(ptr) }.map(Obj::wrap)
    }

    /// Retains a `+0` pointer (a getter's autoreleased result, or a borrowed `id` from the C
    /// ABI); `None` for nil.
    ///
    /// # Safety
    /// `ptr` is nil or a valid object of type `T`.
    pub unsafe fn retain_ptr(ptr: *mut T) -> Option<Obj<T>> {
        // SAFETY: the caller's contract.
        unsafe { Retained::retain(ptr) }.map(Obj::wrap)
    }

    /// Gives the reference back.
    pub fn into_retained(self) -> Retained<T> {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so the reference is taken out exactly once, here.
        unsafe { ManuallyDrop::take(&mut this.0) }
    }

    /// The object.
    pub fn get(&self) -> &T {
        &self.0
    }

    /// The `Retained` itself.
    pub fn as_retained(&self) -> &Retained<T> {
        &self.0
    }
}

impl<T: Message> Drop for Obj<T> {
    /// The release, or nothing while the thread unwinds or releases an isolated task that died.
    fn drop(&mut self) {
        if reims_vgpu_core::tasks::isolation::life::leaks_objects() {
            return;
        }
        // SAFETY: the reference is released once, here; `into_retained` never drops the handle.
        unsafe { ManuallyDrop::drop(&mut self.0) }
    }
}

impl<T: Message> std::ops::Deref for Obj<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: Message> fmt::Debug for Obj<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: every `T: Message` is an Objective-C object.
        let obj: &AnyObject = unsafe { &*Retained::as_ptr(&self.0).cast::<AnyObject>() };
        write!(
            f,
            "<{} {:#x}>",
            obj.class().name().to_string_lossy(),
            Retained::as_ptr(&self.0).addr()
        )
    }
}

impl<T: Message + 'static> Handle for Obj<T> {
    fn clone_ref(&self) -> Self {
        Obj(ManuallyDrop::new(Retained::clone(&self.0)))
    }

    fn id(&self) -> ObjId {
        ObjId(Retained::as_ptr(&self.0).cast::<u8>().addr().to_u64())
    }
}

/// `-[NSError description]`, `-domain` and `-code`, copied out.
pub(crate) fn error_text(e: &NSError) -> NsErrorText {
    // `-[NSObject description]` returns an NSString (typed NSObject by objc2's runtime trait).
    let description: Retained<NSString> = msg::value("description", || {
        // SAFETY: `-description` always returns an NSString.
        unsafe { Retained::cast_unchecked(e.description()) }
    });
    let domain = msg::string_text(&msg::value("domain", || e.domain()));
    NsErrorText {
        domain,
        // NSInteger is 64-bit on every target we build (D32), so this never fails.
        code: i64::try_from(msg::value("code", || e.code())).unwrap_or(0),
        description: msg::string_text(&description),
    }
}

/// [`error_text`] of an `error:` out-parameter; `None` for nil.
///
/// # Safety
/// `err` is nil or a valid `NSError` (an autoreleased `+0` out-parameter).
pub(crate) unsafe fn ns_error_text(err: *mut NSError) -> Option<NsErrorText> {
    // SAFETY: the caller's contract.
    unsafe { err.as_ref() }.map(error_text)
}

/// The process's platform: Metal, IOSurface, Mach VM, threads and the task launcher.
pub struct MetalPlatform {
    surfaces: MetalSurfaces,
    vm: MetalVm,
    clock: MetalClock,
    threads: MetalThreads,
    globals: ProcessState<MetalPlatform>,
    launcher: Box<dyn TaskLauncher<MetalPlatform>>,
}

/// `[[MTLResourceAddressRangeArray alloc] initWithRanges:count:]` (`Platform`'s
/// `new_address_range_array`), for the device's heap over host memory too.
pub(crate) fn address_range_array(
    ranges: &[ResourceAddressRange],
) -> RaiseResult<MetalAddressRangeArray> {
    let sel = "initWithRanges:count:";
    let this = msg::value("alloc", MTLResourceAddressRangeArray::alloc);
    // `Enc` is transparent: the pointer is to `ranges.len()` ranges.
    let (ptr, count) = (
        ranges.as_ptr().cast::<Enc<ResourceAddressRange>>(),
        ranges.len().to_u64(),
    );
    // SAFETY: the pointer is valid for `count` ranges during the call.
    let array = msg::send(sel, Policy::Caught, || unsafe {
        MTLResourceAddressRangeArray::initWithRanges_count(this, ptr, count)
    })?;
    Ok(array.map_or_else(|| msg::nil_result(sel), Obj::from_retained))
}

impl fmt::Debug for MetalPlatform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalPlatform").finish_non_exhaustive()
    }
}

impl MetalPlatform {
    /// The process's platform. `launcher` starts isolated GPU task services (each on threads of
    /// its own in the product, `tasks::isolation::TaskThreads`). Make one per process:
    /// `ProcessState` is the process-wide state (D17).
    pub fn new(launcher: Box<dyn TaskLauncher<MetalPlatform>>) -> MetalPlatform {
        crate::os::autorelease::install();
        if std::env::var_os(reims_vgpu_core::trace::ENV_VAR).is_some() {
            reims_vgpu_core::trace::set_clock(crate::os::clock::trace_clock);
        }
        MetalPlatform {
            surfaces: MetalSurfaces::new(),
            vm: MetalVm::new(),
            clock: MetalClock::new(),
            threads: MetalThreads::new(),
            globals: ProcessState::new(),
            launcher,
        }
    }

    /// The system default device (`MTLCreateSystemDefaultDevice`), for tests; a device the public
    /// APIs make uses the Metal device the host's config names instead.
    pub fn system_default_device() -> Option<MetalDevice> {
        objc2_metal::MTLCreateSystemDefaultDevice().map(Obj::from_retained)
    }
}

/// `+new` of a class (a `Fatal` call): its `+1` result. objc2-metal declares `+new` nonnull, and
/// Metal never answers nil here.
pub(crate) fn class_new<T: Message>(new: impl FnOnce() -> Retained<T>) -> Obj<T> {
    Obj::from_retained(msg::get("new", new))
}

impl Platform for MetalPlatform {
    type Device = MetalDevice;
    type CommandQueue = MetalCommandQueue;
    type CommandBuffer = MetalCommandBuffer;
    type BlitEncoder = MetalBlitEncoder;
    type ComputeEncoder = MetalComputeEncoder;
    type RenderEncoder = MetalRenderEncoder;
    type Buffer = MetalBuffer;
    type Texture = MetalTexture;
    type Heap = MetalHeap;
    type SamplerState = MetalSamplerState;
    type DepthStencilState = MetalDepthStencilState;
    type Fence = MetalFence;
    type Event = MetalEvent;
    type SharedEvent = MetalSharedEvent;
    type SharedEventHandle = MetalSharedEventHandle;
    type SharedEventListener = MetalSharedEventListener;
    type SharedTextureHandle = MetalSharedTextureHandle;
    type Library = MetalLibrary;
    type Function = MetalFunction;
    type ComputePipeline = MetalComputePipeline;
    type RenderPipeline = MetalRenderPipeline;
    type RateMap = MetalRateMap;
    type Icb = MetalIcb;
    type IndirectRenderCommand = MetalIndirectRenderCommand;
    type IndirectComputeCommand = MetalIndirectComputeCommand;
    type TextureDescriptor = MetalTextureDescriptor;
    type RenderPassDescriptor = MetalRenderPassDescriptor;
    type RenderPipelineDescriptor = MetalRenderPipelineDescriptor;
    type ComputePipelineDescriptor = MetalComputePipelineDescriptor;
    type TileRenderPipelineDescriptor = MetalTileRenderPipelineDescriptor;
    type RateMapDescriptor = MetalRateMapDescriptor;
    type RateMapLayerDescriptor = MetalRateMapLayerDescriptor;
    type AddressRangeArray = MetalAddressRangeArray;
    type Surface = MetalSurface;
    type Clock = MetalClock;

    /// Sends `[MTLRenderPassDescriptor new]`.
    fn new_render_pass_descriptor(&self) -> MetalRenderPassDescriptor {
        class_new(MTLRenderPassDescriptor::new)
    }

    /// Sends `[MTLSharedEventListener new]`.
    fn new_shared_event_listener(&self) -> MetalSharedEventListener {
        class_new(MTLSharedEventListener::new)
    }

    /// `+[MTLRenderPassDescriptor renderPassDescriptor]`: a `+0` result, retained.
    fn render_pass_descriptor(&self) -> MetalRenderPassDescriptor {
        Obj::from_retained(msg::get(
            "renderPassDescriptor",
            MTLRenderPassDescriptor::renderPassDescriptor,
        ))
    }

    /// `mipmapped:NO` there. Raises for a pixel format Metal refuses.
    fn texture_2d_descriptor(
        &self,
        on: OnFault,
        pixel_format: u64,
        width: u64,
        height: u64,
        mipmapped: bool,
    ) -> RaiseResult<MetalTextureDescriptor> {
        let sel = "texture2DDescriptorWithPixelFormat:width:height:mipmapped:";
        let format = MTLPixelFormat(pixel_format.to_usize());
        let (width, height) = (width.to_usize(), height.to_usize());
        // SAFETY: a class method of plain values; a `+0` result.
        let descriptor = msg::send(sel, Policy::On(on), || unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format, width, height, mipmapped,
            )
        })?;
        Ok(Obj::from_retained(descriptor))
    }

    /// `[[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:]`: nil for counts Metal
    /// refuses; the caller checks at both sites.
    fn new_rate_map_layer(
        &self,
        on: OnFault,
        sample_count: MtlSize,
    ) -> RaiseResult<Option<MetalRateMapLayerDescriptor>> {
        let this = msg::value("alloc", MTLRasterizationRateLayerDescriptor::alloc);
        let count = sample_count.mtl();
        msg::send("initWithSampleCount:", Policy::On(on), || {
            MTLRasterizationRateLayerDescriptor::initWithSampleCount_or_nil(this, count)
        })
        .map(|layer| layer.map(Obj::from_retained))
    }

    /// `[[MTLResourceAddressRangeArray alloc] initWithRanges:count:]`: the array copies the ranges
    /// (the test `new_address_range_array`), so `ranges` may be on the caller's stack.
    fn new_address_range_array(
        &self,
        ranges: &[ResourceAddressRange],
    ) -> RaiseResult<MetalAddressRangeArray> {
        address_range_array(ranges)
    }

    // Made descriptors and devices, from values, sent natively.

    /// `[MTLTextureDescriptor new]` made to hold `desc` ([`crate::gpu::desc::apply_texture`]).
    fn texture_descriptor(&self, desc: &TextureDesc) -> MetalTextureDescriptor {
        crate::gpu::desc::texture_descriptor(desc)
    }

    /// `[MTLRasterizationRateMapDescriptor new]` made to hold `desc`.
    fn rate_map_descriptor(&self, desc: &RateMapDesc) -> MetalRateMapDescriptor {
        crate::gpu::rate_map::rate_map_descriptor(desc)
    }

    /// The device of `MTLCopyAllDevices()` whose `registryID` is `registry_id` (+1), or `None`.
    /// In one process it is the instance the host's own device is.
    fn open_device(&self, registry_id: u64) -> Option<MetalDevice> {
        let all = msg::fatal_send("MTLCopyAllDevices()", objc2_metal::MTLCopyAllDevices);
        let count = msg::value("count", || all.count());
        (0..count)
            .map(|i| msg::value("objectAtIndex:", || all.objectAtIndex(i)))
            .find(|d| msg::get("registryID", || d.registryID()) == registry_id)
            .map(Obj::from_retained)
    }

    fn surfaces(&self) -> &dyn IoSurfaces<MetalPlatform> {
        &self.surfaces
    }

    fn vm(&self) -> &dyn HostVm {
        &self.vm
    }

    fn clock(&self) -> &MetalClock {
        &self.clock
    }

    fn threads(&self) -> &dyn Threads {
        &self.threads
    }

    fn globals(&self) -> &ProcessState<MetalPlatform> {
        &self.globals
    }

    fn launcher(&self) -> &dyn TaskLauncher<MetalPlatform> {
        &*self.launcher
    }
}

/// The private `MTLResourceAddressRangeArray *`.
pub type MetalAddressRangeArray = Obj<MTLResourceAddressRangeArray>;

#[cfg(test)]
pub(crate) mod tests {

    use std::sync::Arc;

    use objc2::runtime::AnyClass;
    use objc2::runtime::AnyObject;
    use objc2::{ClassType as _, msg_send};
    use objc2_foundation::NSString;
    use objc2_metal::MTLRasterizationRateLayerDescriptor as _Layer;
    use reims_vgpu_core::tasks::launcher::{HostSide, ServiceLink};

    use super::*;
    use crate::gpu::device::tests::dev;
    use crate::spi::classes::MTLBufferDescriptor;

    /// A launcher no test launches through.
    struct NoLauncher;

    impl TaskLauncher<MetalPlatform> for NoLauncher {
        fn launch(
            &self,
            _platform: &Arc<MetalPlatform>,
            _host: Arc<dyn HostSide<MetalPlatform>>,
        ) -> Arc<dyn ServiceLink<MetalPlatform>> {
            panic!("no task is launched in these tests")
        }
    }

    pub(crate) fn platform() -> MetalPlatform {
        MetalPlatform::new(Box::new(NoLauncher))
    }

    fn retain_count(o: &AnyObject) -> usize {
        // SAFETY: `-retainCount` on a live object.
        unsafe { msg_send![o, retainCount] }
    }

    fn class_name<T: Message>(o: &Obj<T>) -> String {
        // SAFETY: every `T: Message` is an Objective-C object.
        let a: &AnyObject = unsafe { &*Retained::as_ptr(o.as_retained()).cast::<AnyObject>() };
        a.class().name().to_string_lossy().into_owned()
    }

    /// Obj: clone_ref() is one retain of the same object, Drop one release (D6)
    #[test]
    fn obj_retain_and_release() {
        let s = Obj::from_retained(NSString::from_str("x"));
        let d = dev();
        // The retain count is measured on an object only this test holds: the device is a
        // process-wide singleton that parallel tests retain and release concurrently.
        let o = Obj::from_retained(objc2::runtime::NSObject::new());
        // SAFETY: an NSObject is an Objective-C object.
        let a: &AnyObject = unsafe { &*Retained::as_ptr(o.as_retained()).cast::<AnyObject>() };
        let before = retain_count(a);
        let r = o.clone_ref();
        assert_eq!(r.id(), o.id());
        assert_eq!(retain_count(a), before + 1);
        drop(r);
        assert_eq!(retain_count(a), before);
        assert_eq!(d.clone_ref().id(), d.id());
        assert_ne!(s.id(), d.id());
        assert!(format!("{d:?}").contains("Device"));
        // SAFETY: a +0 pointer to a live object; nil gives None.
        let again = unsafe { Obj::retain_ptr(Retained::as_ptr(d.as_retained()).cast_mut()) };
        assert_eq!(again.map(|o| o.id()), Some(d.id()));
        // SAFETY: nil.
        assert!(unsafe { Obj::<NSString>::from_owned_ptr(std::ptr::null_mut()) }.is_none());
        assert_eq!(s.into_retained().to_string(), "x");
    }

    /// Obj: a handle an isolated task's death unwinds past is leaked, not released; one the
    /// task's work drops is released
    #[test]
    fn a_dead_tasks_unwind_leaks_its_handles() {
        use reims_vgpu_core::raise::death::{DeathKind, DeathSite};
        use reims_vgpu_core::tasks::isolation::TaskLife;

        let o = Obj::from_retained(objc2::runtime::NSObject::new());
        // SAFETY: an NSObject is an Objective-C object.
        let a: &AnyObject = unsafe { &*Retained::as_ptr(o.as_retained()).cast::<AnyObject>() };
        let before = retain_count(a);
        let life = TaskLife::new();
        let _ = life.run(|| drop(o.clone_ref()));
        assert_eq!(retain_count(a), before);
        let r: Option<()> = life.run(|| {
            let _held = o.clone_ref();
            reims_vgpu_core::fatal!(DeathSite {
                kind: DeathKind::Abort,
                file: "platform.test",
                line: 0,
            })
        });
        assert_eq!(r, None);
        assert_eq!(retain_count(a), before + 1);
    }

    /// A thread of an isolated task that has ended sends nothing more: it stops at its next send
    #[test]
    fn an_ended_tasks_thread_stops_at_its_next_send() {
        use reims_vgpu_core::tasks::isolation::TaskLife;

        let d = dev();
        let life = TaskLife::new();
        let mut reached = Vec::new();
        let registry_id = || msg::get("registryID", || d.get().registryID());
        let r = life.run(|| {
            reached.push(registry_id() != 0);
            life.exit();
            reached.push(registry_id() != 0);
        });
        assert_eq!(r, None);
        assert_eq!(reached, vec![true]);
    }

    /// The `+new` constructors each make an object of the class core asks for
    #[test]
    fn class_new_constructors() {
        let p = platform();
        assert_eq!(
            class_name(&Obj::from_retained(MTLBufferDescriptor::new())),
            "MTLBufferDescriptor"
        );
        assert!(
            class_name(&p.rate_map_descriptor(&RateMapDesc::default()))
                .starts_with("MTLRasterizationRateMapDescriptor")
        );
        assert!(class_name(&p.new_render_pass_descriptor()).starts_with("MTLRenderPassDescriptor"));
        assert!(class_name(&p.new_shared_event_listener()).starts_with("MTLSharedEventListener"));
        let texture = || p.texture_descriptor(&TextureDesc::default());
        assert!(class_name(&texture()).starts_with("MTLTextureDescriptor"));
        // A fresh object each time.
        assert_ne!(texture().id(), texture().id());
    }

    /// +renderPassDescriptor and +texture2DDescriptorWithPixelFormat:width:height:mipmapped
    #[test]
    fn autoreleased_class_constructors() {
        let p = platform();
        let rp = p.render_pass_descriptor();
        assert!(class_name(&rp).starts_with("MTLRenderPassDescriptor"));
        let td = p
            .texture_2d_descriptor(OnFault::Die, 80, 12, 34, false)
            .unwrap();
        assert_eq!((td.get().width(), td.get().height()), (12, 34));
        assert_eq!(td.get().mipmapLevelCount(), 1);
        let tm = p
            .texture_2d_descriptor(OnFault::Die, 80, 16, 16, true)
            .unwrap();
        assert_eq!(tm.get().mipmapLevelCount(), 5);
    }

    /// [[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:] (nil for {0,0,0},
    /// measured; an object otherwise)
    #[test]
    fn new_rate_map_layer() {
        let p = platform();
        assert!(
            p.new_rate_map_layer(OnFault::Die, MtlSize::new(0, 0, 0))
                .unwrap()
                .is_none()
        );
        let l = p
            .new_rate_map_layer(OnFault::Die, MtlSize::new(4, 2, 0))
            .unwrap()
            .unwrap();
        let s = _Layer::sampleCount(l.get());
        assert_eq!((s.width, s.height), (4, 2));
    }

    /// [[MTLResourceAddressRangeArray alloc] initWithRanges:count:] copies the ranges
    #[test]
    fn new_address_range_array() {
        let p = platform();
        let ranges = [
            ResourceAddressRange {
                address: 0x1000,
                length: 0x2000,
            },
            ResourceAddressRange {
                address: 0x8000,
                length: 0x4000,
            },
        ];
        let a = p.new_address_range_array(&ranges).unwrap();
        // SAFETY: `-count` (`Q16@0:8`).
        let n: u64 = unsafe { msg_send![a.get(), count] };
        assert_eq!(n, 2);
        // SAFETY: `-ranges` (`^{MTLAddressRange=QQ}16@0:8`): `count` ranges.
        let r: *const Enc<ResourceAddressRange> = unsafe { msg_send![a.get(), ranges] };
        assert_ne!(r.cast::<ResourceAddressRange>(), ranges.as_ptr());
        // SAFETY: two ranges, as counted.
        let copied = unsafe { std::slice::from_raw_parts(r, 2) };
        assert_eq!(copied[1].0, ranges[1]);
        let empty = p.new_address_range_array(&[]).unwrap();
        // SAFETY: as above.
        let n: u64 = unsafe { msg_send![empty.get(), count] };
        assert_eq!(n, 0);
    }

    /// open_device finds the device among MTLCopyAllDevices by its registry ID: in one process
    /// the instance the host's own device is, +1; None for an unknown ID
    #[test]
    fn open_device_finds_the_device() {
        let p = platform();
        let d = dev();
        let id = d.get().registryID();
        let found = p.open_device(id).unwrap();
        assert_eq!(found.id(), d.id());
        let before = retain_count(obj_of(&found));
        let again = p.open_device(id).unwrap();
        assert!(retain_count(obj_of(&found)) > before);
        drop(again);
        assert!(p.open_device(0xdead_beef).is_none());
    }

    fn obj_of<T: Message>(o: &Obj<T>) -> &AnyObject {
        // SAFETY: every `T: Message` is an Objective-C object.
        unsafe { &*Retained::as_ptr(o.as_retained()).cast::<AnyObject>() }
    }

    /// The services are the ones the platform was made with
    #[test]
    fn services() {
        let p = platform();
        let _ = (p.surfaces(), p.vm(), p.clock(), p.threads());
        let _ = (p.globals(), p.launcher());
        assert!(format!("{p:?}").contains("MetalPlatform"));
        assert!(AnyClass::get(c"MTLTextureDescriptor").is_some());
    }

    /// ns_error_text copies domain, code and -description; nil is None
    #[test]
    fn ns_error_text_reads_the_error() {
        // SAFETY: nil.
        assert_eq!(unsafe { ns_error_text(std::ptr::null_mut()) }, None);
        let domain = NSString::from_str("reims-vgpu.test");
        // SAFETY: `+[NSError errorWithDomain:code:userInfo:]`, autoreleased.
        let e: *mut NSError = unsafe {
            msg_send![NSError::class(), errorWithDomain: &*domain, code: 7isize, userInfo: std::ptr::null::<AnyObject>()]
        };
        // SAFETY: a live, autoreleased NSError.
        let t = unsafe { ns_error_text(e) }.unwrap();
        assert_eq!((t.domain.as_str(), t.code), ("reims-vgpu.test", 7));
        assert!(t.description.contains("reims-vgpu.test"), "{t:?}");
    }
}
