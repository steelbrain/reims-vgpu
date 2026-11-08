//! Fake: `FakeObj` handles and their small models (architecture 13.2).
//!
//! Every fake platform handle is an [`Fk<K>`]: a counted reference to one shared [`FakeObj`],
//! typed by a [`Kind`] marker so that `FakePlatform`'s associated types stay distinct (a
//! texture cannot be passed where a buffer is expected, as with the real platform). A handle is
//! not `Clone` (D6): `clone_ref()` adds a reference and `Drop` removes one, and the live count is
//! the object's retain count as far as core can observe it ([`Fk::retain_count`]). For an
//! IOSurface that count is guest-visible, and every model that keeps a surface (a texture or
//! buffer over it) holds a real handle, so it counts as Metal's retain does.
//!
//! The shared [`World`] holds what one fake "process" has: the call log, the script, the virtual
//! clock, the address space, the GPU VA allocator, resource-index spaces, and the list of
//! committed command buffers a test completes.
//!
//! Models (deliberately small, and **not** measured Metal behaviour unless a comment says so):
//! - buffers are real memory (or alias a caller's pointer for no-copy buffers);
//! - GPU VAs come from a bump allocator starting at `0x10000090000` (the first buffer VA the
//!   suite's `textures_placement_written` records);
//! - resource indices are the smallest free index from 1 in a space, honouring forced indices
//!   and reservations, freed when the object dies (Metal reuses a released texture's index);
//! - `setPurgeableState:` returns the previous state;
//! - ICB commands are per-index child objects recording what their commands set;
//! - shared events wait on the virtual clock;
//! - command buffers complete only when a test calls [`World::complete`] (or auto-complete), and
//!   the world holds a committed buffer until then, as Metal does;
//! - each kind keeps a typed model of what its setters set and its getters answer, and a test
//!   changes what a getter answers through the model's setters (on the object, or with
//!   [`World::on_new`] for an object core makes).

use std::any::{Any, TypeId};
use std::borrow::BorrowMut;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use crate::platform::values::{PagesInPlace, PagesPlane};
use crate::sync::{Condvar, Mutex, MutexGuard};

use super::calllog::{Arg, CallLog, Event, ReleaseContext};
use super::clock::VirtualClock;
use super::guest_ram::{PAGE_SIZE, RawRegion};
use super::impls::descriptor::{PassModel, TextureDescriptorModel};
use super::impls::device::DeviceModel;
use super::impls::icb::{IcbCommandModel, IcbModel};
use super::impls::rate_map::{RateMapDescriptorModel, RateMapModel};
use super::impls::resource::PipelineModel;
use super::script::{HeldCall, Script, Step};
use super::vm::FakeVm;
use crate::host::HostPtr;
use crate::num;
use crate::platform::command::{CompletedHandler, CompletedInfo, ScheduledHandler};
use crate::platform::device::NoCopyDeallocator;
use crate::platform::event::Notification;
use crate::platform::os::{Clock, Deadline};
use crate::platform::types::{
    CommandBufferStatus, MtlSize, NsErrorText, ResourceAddressRange, Swizzle,
};
use crate::platform::{Handle, ObjId};

/// A handle kind: the Objective-C class the fake stands in for.
pub trait Kind: Send + Sync + 'static {
    /// The class name the call log records.
    const CLASS: &'static str;
    /// Whether `clone_ref()` and `Drop` are logged (`CFRetain`/`CFRelease`): only for IOSurfaces,
    /// whose count is guest-visible (D6).
    const LOG_RETAIN: bool = false;
}

/// The kind markers, one per associated type of `Platform`.
pub mod kind {
    use super::Kind;

    macro_rules! kinds {
        ($($name:ident => $class:literal),* $(,)?) => {
            $(
                #[doc = concat!("`", $class, "`.")]
                #[derive(Debug)]
                pub struct $name;
                impl Kind for $name {
                    const CLASS: &'static str = $class;
                }
            )*
        };
    }

    kinds! {
        Device => "MTLDevice",
        CommandQueue => "MTLCommandQueue",
        CommandBuffer => "MTLCommandBuffer",
        BlitEncoder => "MTLBlitCommandEncoder",
        ComputeEncoder => "MTLComputeCommandEncoder",
        RenderEncoder => "MTLRenderCommandEncoder",
        Buffer => "MTLBuffer",
        Texture => "MTLTexture",
        Heap => "MTLHeap",
        SamplerState => "MTLSamplerState",
        DepthStencilState => "MTLDepthStencilState",
        Fence => "MTLFence",
        Event => "MTLEvent",
        SharedEvent => "MTLSharedEvent",
        SharedEventListener => "MTLSharedEventListener",
        SharedTextureHandle => "MTLSharedTextureHandle",
        Library => "MTLLibrary",
        Function => "MTLFunction",
        ComputePipeline => "MTLComputePipelineState",
        RenderPipeline => "MTLRenderPipelineState",
        RateMap => "MTLRasterizationRateMap",
        Icb => "MTLIndirectCommandBuffer",
        IndirectRenderCommand => "MTLIndirectRenderCommand",
        IndirectComputeCommand => "MTLIndirectComputeCommand",
        TextureDescriptor => "MTLTextureDescriptor",
        RenderPassDescriptor => "MTLRenderPassDescriptor",
        RateMapDescriptor => "MTLRasterizationRateMapDescriptor",
        RateMapLayerDescriptor => "MTLRasterizationRateLayerDescriptor",
        AddressRangeArray => "MTLResourceAddressRangeArray",
    }

    /// `IOSurfaceRef`: its retains are logged (`CFRetain`/`CFRelease`).
    #[derive(Debug)]
    pub struct Surface;
    impl Kind for Surface {
        const CLASS: &'static str = "IOSurface";
        const LOG_RETAIN: bool = true;
    }
}

/// A resource-index space: the reservation type the platform is given. Textures, samplers and
/// ICBs are the product's reservation types 0, 1 and 5; the other spaces are the fake's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Space(pub u64);

impl Space {
    /// Textures (reservation type 0).
    pub const TEXTURE: Space = Space(0);
    /// Sampler states (reservation type 1).
    pub const SAMPLER: Space = Space(1);
    /// Render pipeline states (the fake's space).
    pub const RENDER_PIPELINE: Space = Space(2);
    /// Compute pipeline states (the fake's space).
    pub const COMPUTE_PIPELINE: Space = Space(3);
    /// Depth-stencil states (the fake's space).
    pub const DEPTH_STENCIL: Space = Space(4);
    /// Indirect command buffers (reservation type 5).
    pub const ICB: Space = Space(5);
}

/// The first GPU VA the fake hands out: the first buffer VA the suite's
/// `textures_placement_written` records.
pub const GPU_VA_BASE: u64 = 0x100_0009_0000;

/// The largest allocation the fake backs with real memory; a larger shared buffer is nil.
pub const FAKE_MAX_BYTES: u64 = 1 << 30;

/// `MTLPurgeableStateKeepCurrent`.
pub const PURGEABLE_KEEP_CURRENT: u64 = 1;
/// `MTLPurgeableStateNonVolatile`, every resource's initial state.
pub const PURGEABLE_NON_VOLATILE: u64 = 2;

/// Where [`World::complete`] runs a command buffer's completed handlers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompleteOn {
    /// On the calling thread, as a foreign thread would run them: owning no queue
    /// (`rt::BlockEnv::foreign`), with no handler live and no holder
    /// ([`Hold::NONE`](crate::rt::Hold::NONE)), as every completion handler runs, so a raise in a
    /// handler dies as it does on Metal's completion thread. The calling thread's locks stay held,
    /// though, which Metal's thread never holds: use it only where the caller holds no core lock (a
    /// test thread, or `commit` under auto-complete from code that holds none); otherwise use
    /// `Thread`.
    Caller,
    /// On a fresh thread (Metal's completion thread), joined before returning.
    Thread,
    /// On a fresh thread, not joined.
    Detached,
}

/// The common part of a resource or state object.
#[derive(Debug, Default)]
pub struct ResourceCommon {
    /// `setPurgeableState:`'s current state.
    pub purgeable: u64,
    /// The object's resource index, freed with the object.
    pub index: Option<IndexGuard>,
    /// `uniqueIdentifier`.
    pub unique_id: u64,
    /// `gpuResourceID` (its `_impl`).
    pub gpu_resource_id: u64,
    /// `setLabel:`.
    pub label: Option<String>,
}

/// A buffer's model.
#[derive(Default)]
pub struct BufferModel {
    /// Memory the fake owns (none for no-copy or private buffers).
    pub mem: Option<Arc<RawRegion>>,
    /// `contents`.
    pub contents: HostPtr,
    /// `length`.
    pub length: u64,
    /// `resourceOptions`.
    pub options: u64,
    /// `gpuAddress`.
    pub gpu_address: u64,
    /// The IOSurface it was made over (a counted reference).
    pub surface: Option<Fk<kind::Surface>>,
    /// The heap it was placed in.
    pub heap: Option<AnyHandle>,
    /// A no-copy buffer's deallocator, run when the buffer dies.
    pub deallocator: Option<NoCopyDeallocator>,
    /// A descriptor-made buffer's deallocator `(contents, length)`, run when the buffer dies.
    pub desc_deallocator: Option<DescDeallocator>,
    /// `detachBacking` was sent.
    pub detached: bool,
    /// The address ranges it was re-backed with (retained).
    pub ranges: Option<AnyHandle>,
    /// Common resource state.
    pub common: ResourceCommon,
}

impl std::fmt::Debug for BufferModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferModel")
            .field("contents", &self.contents)
            .field("length", &self.length)
            .field("options", &self.options)
            .field("gpu_address", &self.gpu_address)
            .finish_non_exhaustive()
    }
}

/// A texture descriptor's values (Metal's defaults where nothing was set).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TexDesc {
    /// `textureType` (default `MTLTextureType2D`).
    pub texture_type: u64,
    /// `pixelFormat` (default `MTLPixelFormatRGBA8Unorm`).
    pub pixel_format: u64,
    /// `width`.
    pub width: u64,
    /// `height`.
    pub height: u64,
    /// `depth`.
    pub depth: u64,
    /// `mipmapLevelCount`.
    pub mipmap_level_count: u64,
    /// `sampleCount`.
    pub sample_count: u64,
    /// `arrayLength`.
    pub array_length: u64,
    /// `resourceOptions` (storage mode in bits 4-7).
    pub resource_options: u64,
    /// `usage` (default `MTLTextureUsageShaderRead`).
    pub usage: u64,
    /// `swizzle` (default identity).
    pub swizzle: Swizzle,
    /// `rotation` (SPI).
    pub rotation: u64,
    /// `protectionOptions` (SPI).
    pub protection_options: u64,
    /// `resourceIndex` (SPI), if set.
    pub resource_index: Option<u64>,
    /// `forceResourceIndex` (SPI).
    pub force_resource_index: bool,
    /// `framebufferOnly` (SPI).
    pub framebuffer_only: bool,
    /// `isDrawable` (SPI).
    pub is_drawable: bool,
    /// `writeSwizzleEnabled` (SPI).
    pub write_swizzle_enabled: bool,
    /// `allowGPUOptimizedContents`.
    pub allow_gpu_optimized_contents: bool,
}

impl Default for TexDesc {
    fn default() -> Self {
        TexDesc {
            texture_type: 2,
            pixel_format: 70,
            width: 1,
            height: 1,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            resource_options: 0,
            usage: 1,
            swizzle: Swizzle {
                red: 2,
                green: 3,
                blue: 4,
                alpha: 5,
            },
            rotation: 0,
            protection_options: 0,
            resource_index: None,
            force_resource_index: false,
            framebuffer_only: false,
            is_drawable: false,
            write_swizzle_enabled: false,
            allow_gpu_optimized_contents: true,
        }
    }
}

impl TexDesc {
    /// `storageMode`: bits 4-7 of the resource options.
    pub fn storage_mode(&self) -> u64 {
        self.resource_options.wrapping_shr(4) & 0xf
    }
}

/// A texture's model.
#[derive(Debug, Default)]
pub struct TextureModel {
    /// The descriptor it was made with (a view's changed fields applied).
    pub desc: TexDesc,
    /// The IOSurface it was made over (a counted reference), and the plane.
    pub surface: Option<Fk<kind::Surface>>,
    /// The plane of `surface`.
    pub plane: u64,
    /// The texture or buffer it is a view of (retained, as Metal's views retain their parent).
    pub parent: Option<AnyHandle>,
    /// Made shareable (`newSharedTexture…`).
    pub shareable: bool,
    /// Common resource state.
    pub common: ResourceCommon,
}

/// A heap's model.
#[derive(Debug, Default)]
pub struct HeapModel {
    /// `size`.
    pub size: u64,
    /// `resourceOptions`.
    pub options: u64,
    /// `gpuAddress`.
    pub gpu_address: u64,
    /// Memory the fake owns for a CPU-visible heap.
    pub mem: Option<Arc<RawRegion>>,
    /// The address ranges it was made over or re-backed with.
    pub ranges: Option<AnyHandle>,
    /// Common resource state.
    pub common: ResourceCommon,
}

/// A command buffer's model.
#[derive(Default)]
pub struct CommandBufferModel {
    /// `status`.
    pub status: CommandBufferStatus,
    /// `error`.
    pub error: Option<NsErrorText>,
    /// The completed handlers, in the order added.
    pub handlers: Vec<CompletedHandler>,
    /// The completion-seen handlers, in the order added: run before the completed handlers.
    pub seen: Vec<CompletedHandler>,
    /// The scheduled handlers, in the order added: run at commit, on the committing thread.
    pub scheduled: Vec<ScheduledHandler>,
    /// The submission handlers, in the order added: run at commit, on the committing thread,
    /// before the scheduled handlers.
    pub submission: Vec<ScheduledHandler>,
    /// `encodeSignalEvent:value:`s, applied at completion.
    pub signal_on_complete: Vec<(AnyHandle, u64)>,
    /// `encodeSignalEventScheduled:value:`s, applied at commit.
    pub signal_on_schedule: Vec<(AnyHandle, u64)>,
    /// The detail of the death the platform's read of the buffer dies with as each completed
    /// handler runs ([`World::complete_dying_read`]); `None` for a read that finds what the model
    /// holds.
    pub read_dies: Option<String>,
    /// `GPUStartTime` and `GPUEndTime` in nanoseconds, which its completed handlers learn
    /// ([`World::set_gpu_times`]); `None`, the default, for none.
    pub gpu_times_ns: Option<(u64, u64)>,
}

impl std::fmt::Debug for CommandBufferModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandBufferModel")
            .field("status", &self.status)
            .field("error", &self.error)
            .field("handlers", &self.handlers.len())
            .field("seen", &self.seen.len())
            .field("scheduled", &self.scheduled.len())
            .field("submission", &self.submission.len())
            .finish_non_exhaustive()
    }
}

/// An IOSurface's model.
#[derive(Debug, Default)]
pub struct SurfaceModel {
    /// The surface's memory (`IOSurfaceGetBaseAddress`).
    pub mem: Option<Arc<RawRegion>>,
    /// `IOSurfaceGetPixelFormat`.
    pub pixel_format: u32,
    /// `IOSurfaceGetWidth`.
    pub width: u64,
    /// `IOSurfaceGetHeight`.
    pub height: u64,
    /// `IOSurfaceGetBytesPerElement`.
    pub bytes_per_element: u64,
    /// `IOSurfaceGetBytesPerRow`.
    pub bytes_per_row: u64,
    /// `IOSurfaceGetPlaneCount`.
    pub plane_count: u64,
    /// `IOSurfaceGetAllocSize`.
    pub alloc_size: u64,
    /// `IOSurfaceGetProtectionOptions`.
    pub protection_options: u64,
    /// `IOSurfaceAllowsPixelSizeCasting`.
    pub allows_pixel_size_casting: bool,
    /// Outstanding `IOSurfaceLock`s.
    pub lock_count: i64,
    /// The creation spec, when made by `IOSurfaceCreate`.
    pub spec: Option<crate::platform::os::surface::SurfaceSpec>,
    /// The creation properties, when made by `initWithProperties:`.
    pub backing_spec: Option<crate::platform::os::surface::BackingSurfaceSpec>,
}

/// A rasterization-rate layer: its sample count and sample storage.
#[derive(Debug, Default)]
pub struct LayerModel {
    /// `sampleCount`.
    pub sample_count: MtlSize,
    /// `horizontalSampleStorage`: `sample_count.width` floats of `storage` when made.
    pub horizontal: HostPtr,
    /// `verticalSampleStorage`: `sample_count.height` floats of `storage` when made.
    pub vertical: HostPtr,
    /// The float storage the fake made for it.
    pub storage: Vec<Arc<RawRegion>>,
}

/// What an object's model holds.
#[derive(Debug, Default)]
pub enum Model {
    /// Nothing (encoders, fences, queues).
    #[default]
    None,
    /// A buffer.
    Buffer(BufferModel),
    /// A texture.
    Texture(TextureModel),
    /// A heap.
    Heap(HeapModel),
    /// A command buffer.
    CommandBuffer(CommandBufferModel),
    /// An `MTLEvent` or `MTLSharedEvent`: its signalled value.
    Event {
        /// The signalled value.
        value: u64,
    },
    /// A sampler or depth-stencil state: its common state.
    State(ResourceCommon),
    /// A compute or render pipeline state.
    Pipeline(PipelineModel),
    /// A rasterization rate map.
    RateMap(RateMapModel),
    /// An indirect command buffer.
    Icb(IcbModel),
    /// An ICB's render or compute command.
    IcbCommand(Box<IcbCommandModel>),
    /// An IOSurface.
    Surface(SurfaceModel),
    /// A rasterization-rate layer.
    Layer(LayerModel),
    /// An address-range array.
    Ranges(Vec<ResourceAddressRange>),
    /// A function's name, or a library's function names.
    Names(Vec<String>),
    /// A handle object's target (shared event handle, shared texture handle).
    Target(AnyHandle),
    /// A device's capabilities.
    Device(DeviceModel),
    /// A texture descriptor.
    TextureDescriptor(TextureDescriptorModel),
    /// A render pass descriptor and its attachments.
    Pass(Box<PassModel>),
    /// A rasterization rate map descriptor.
    RateMapDescriptor(RateMapDescriptorModel),
}

/// A kind's typed model, as a [`Model`] holds it ([`Fk::with_model`], [`Fk::with_model_mut`]).
pub trait ModelOf: Default {
    /// The model `m` holds, if it is one of these.
    fn of(m: &mut Model) -> Option<&mut Self>;
    /// `self` as a [`Model`].
    fn into_model(self) -> Model;
}

/// Implements [`ModelOf`] for each model, held by its variant as is or boxed.
macro_rules! model_of {
    ($($ty:ty => $variant:ident),* $(,)?) => {
        $(
            impl ModelOf for $ty {
                fn of(m: &mut Model) -> Option<&mut Self> {
                    match m {
                        Model::$variant(x) => Some(BorrowMut::<$ty>::borrow_mut(x)),
                        _ => None,
                    }
                }
                fn into_model(self) -> Model {
                    Model::$variant(self.into())
                }
            }
        )*
    };
}

model_of! {
    TextureModel => Texture,
    PipelineModel => Pipeline,
    RateMapModel => RateMap,
    IcbModel => Icb,
    IcbCommandModel => IcbCommand,
    LayerModel => Layer,
    DeviceModel => Device,
    TextureDescriptorModel => TextureDescriptor,
    PassModel => Pass,
    RateMapDescriptorModel => RateMapDescriptor,
}

/// A buffer descriptor's deallocator block `(contents, length)`.
pub struct DescDeallocator(pub Box<dyn FnOnce(HostPtr, u64) + Send>);

impl std::fmt::Debug for DescDeallocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DescDeallocator(..)")
    }
}

/// An object's mutable state.
#[derive(Debug, Default)]
pub struct State {
    /// Child objects (an ICB's commands, a rate-map descriptor's layers), keyed by name and
    /// index.
    pub children: Vec<(&'static str, u64, AnyHandle)>,
    /// The device that made it (its `device` property).
    pub device: Option<AnyHandle>,
    /// The kind-specific model.
    pub model: Model,
}

impl State {
    /// The common resource state, for kinds that have one.
    pub fn common(&mut self) -> Option<&mut ResourceCommon> {
        match &mut self.model {
            Model::Buffer(b) => Some(&mut b.common),
            Model::Texture(t) => Some(&mut t.common),
            Model::Heap(h) => Some(&mut h.common),
            Model::State(c) => Some(c),
            Model::Pipeline(p) => Some(&mut p.common),
            Model::RateMap(r) => Some(&mut r.common),
            Model::Icb(i) => Some(&mut i.common),
            _ => None,
        }
    }

    /// The common resource state, read-only.
    pub fn common_ref(&self) -> Option<&ResourceCommon> {
        match &self.model {
            Model::Buffer(b) => Some(&b.common),
            Model::Texture(t) => Some(&t.common),
            Model::Heap(h) => Some(&h.common),
            Model::State(c) => Some(c),
            Model::Pipeline(p) => Some(&p.common),
            Model::RateMap(r) => Some(&r.common),
            Model::Icb(i) => Some(&i.common),
            _ => None,
        }
    }
}

/// One fake object: identity, class, shared world, live handle count, state.
pub struct FakeObj {
    id: ObjId,
    class: &'static str,
    world: Arc<World>,
    handles: AtomicI64,
    state: Mutex<State>,
    cv: Condvar,
}

impl std::fmt::Debug for FakeObj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeObj")
            .field("id", &self.id)
            .field("class", &self.class)
            .field("handles", &self.handles.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

impl FakeObj {
    /// The identity.
    pub fn id(&self) -> ObjId {
        self.id
    }

    /// The class name.
    pub fn class(&self) -> &'static str {
        self.class
    }

    /// The world it lives in.
    pub fn world(&self) -> &Arc<World> {
        &self.world
    }

    /// Locks the state.
    pub fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock()
    }

    /// The object's condition variable (command buffer and event waits).
    pub fn cv(&self) -> &Condvar {
        &self.cv
    }

    /// The live handle count.
    pub fn handle_count(&self) -> i64 {
        self.handles.load(Ordering::SeqCst)
    }
}

impl Drop for FakeObj {
    fn drop(&mut self) {
        if self.world.log.recording_events() {
            self.world.log.event(Event::Released {
                obj: self.id,
                class: self.class,
                context: ReleaseContext::here(),
            });
        }
        // A buffer's deallocator runs when Metal destroys the buffer.
        let model = std::mem::take(&mut self.state.get_mut().model);
        if let Model::Buffer(mut b) = model {
            if let Some(d) = b.deallocator.take() {
                d();
            }
            if let Some(d) = b.desc_deallocator.take() {
                (d.0)(b.contents, b.length);
            }
        }
    }
}

/// A counted, type-erased reference to a fake object (what models hold).
pub struct AnyHandle(Arc<FakeObj>);

impl std::fmt::Debug for AnyHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}#{}", self.0.class, self.0.id.0)
    }
}

impl AnyHandle {
    fn new(obj: Arc<FakeObj>) -> AnyHandle {
        obj.handles.fetch_add(1, Ordering::SeqCst);
        AnyHandle(obj)
    }

    /// One more reference.
    pub fn clone_ref(&self) -> AnyHandle {
        AnyHandle::new(self.0.clone())
    }

    /// The object.
    pub fn obj(&self) -> &Arc<FakeObj> {
        &self.0
    }

    /// The identity.
    pub fn id(&self) -> ObjId {
        self.0.id
    }

    /// A typed handle (retained), if the object is of class `K`.
    pub fn cast<K: Kind>(&self) -> Option<Fk<K>> {
        (self.0.class == K::CLASS).then(|| Fk::from_obj(self.0.clone()))
    }
}

impl Drop for AnyHandle {
    fn drop(&mut self) {
        self.0.handles.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A typed fake handle: `FakePlatform`'s associated types are all `Fk<kind::…>`.
pub struct Fk<K: Kind> {
    h: AnyHandle,
    _k: PhantomData<fn() -> K>,
}

impl<K: Kind> std::fmt::Debug for Fk<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}#{}", K::CLASS, self.h.id().0)
    }
}

impl<K: Kind> Handle for Fk<K> {
    fn clone_ref(&self) -> Self {
        if K::LOG_RETAIN {
            self.obj()
                .world
                .log
                .push(self.id(), K::CLASS, "CFRetain", vec![]);
        }
        Fk::from_obj(self.h.0.clone())
    }

    fn id(&self) -> ObjId {
        self.h.id()
    }
}

impl<K: Kind> Drop for Fk<K> {
    /// A release: logged where `K` logs its retains, then the death [`World::die_at_release`]
    /// set for the object and the hold [`World::hold_at_release`] set for it, where the thread
    /// is not unwinding already.
    fn drop(&mut self) {
        if K::LOG_RETAIN {
            self.obj()
                .world
                .log
                .push(self.id(), K::CLASS, "CFRelease", vec![]);
        }
        if !std::thread::panicking() && self.obj().world.take_release_death(self.id()) {
            crate::raise::Raise::platform("release")
                .with_detail("the platform died letting go of the object")
                .fatal();
        }
        if !std::thread::panicking()
            && let Some(held) = self.obj().world.take_release_hold(self.id())
        {
            held.hold();
        }
    }
}

impl<K: Kind> Fk<K> {
    pub(crate) fn from_obj(obj: Arc<FakeObj>) -> Fk<K> {
        Fk {
            h: AnyHandle::new(obj),
            _k: PhantomData,
        }
    }

    /// The object.
    pub fn obj(&self) -> &Arc<FakeObj> {
        self.h.obj()
    }

    /// The world the object lives in.
    pub fn world(&self) -> &Arc<World> {
        &self.h.0.world
    }

    /// The object's retain count as core can observe it: the live handles, including those
    /// models hold (a texture over a surface holds one of the surface).
    pub fn retain_count(&self) -> i64 {
        self.h.0.handle_count()
    }

    /// A retained, erased copy (for a model to hold).
    pub fn erase(&self) -> AnyHandle {
        self.h.clone_ref()
    }

    /// Locks the state.
    pub fn state(&self) -> MutexGuard<'_, State> {
        self.h.0.state.lock()
    }

    /// Logs a call on this object and returns its scripted step: a call whose position is
    /// observable (a creation or release, an aliasing setter, a command, a call with a guest
    /// argument, a getter that can raise). A free call (a getter, a query, a setter no other
    /// property aliases) is neither logged nor scripted: it reads or writes the model.
    pub fn call(&self, sel: &'static str, args: Vec<Arg>) -> Step {
        let w = &self.h.0.world;
        w.log.push(self.id(), K::CLASS, sel, args);
        w.script.next_for(self.id(), K::CLASS, sel)
    }

    /// Reads the object's model of type `M` with `f`. An object that holds none (one a test made
    /// with `Model::None`) reads a default one.
    pub fn with_model<M: ModelOf, T>(&self, f: impl FnOnce(&M) -> T) -> T {
        let mut s = self.state();
        match M::of(&mut s.model) {
            Some(m) => f(m),
            None => f(&M::default()),
        }
    }

    /// Changes the object's model of type `M` with `f`. An object that holds none (one a test
    /// made with `Model::None`) gets a default one first.
    pub fn with_model_mut<M: ModelOf, T>(&self, f: impl FnOnce(&mut M) -> T) -> T {
        let mut s = self.state();
        if let Some(m) = M::of(&mut s.model) {
            return f(m);
        }
        let mut m = M::default();
        let r = f(&mut m);
        s.model = m.into_model();
        r
    }

    /// The child object `key[index]`, made on first use and the same object afterwards (an ICB's
    /// command at an index).
    pub fn child<C: Kind>(
        &self,
        key: &'static str,
        index: u64,
        init: impl FnOnce() -> Model,
    ) -> Fk<C> {
        let mut s = self.state();
        if let Some((_, _, h)) = s.children.iter().find(|(k, i, _)| *k == key && *i == index)
            && let Some(c) = h.cast::<C>()
        {
            return c;
        }
        let c: Fk<C> = self.h.0.world.new_obj(init());
        s.children.retain(|(k, i, _)| !(*k == key && *i == index));
        s.children.push((key, index, c.erase()));
        c
    }

    /// Replaces the child `key[index]` with `h` (`setLayer:atIndex:`).
    pub fn set_child(&self, key: &'static str, index: u64, h: AnyHandle) {
        let mut s = self.state();
        s.children.retain(|(k, i, _)| !(*k == key && *i == index));
        s.children.push((key, index, h));
    }

    /// Every child, its key and index with it, in the order they were made.
    pub fn children(&self) -> Vec<(&'static str, u64, AnyHandle)> {
        self.state()
            .children
            .iter()
            .map(|(k, i, h)| (*k, *i, h.clone_ref()))
            .collect()
    }

    /// The child `key[index]`, if set or made.
    pub fn get_child(&self, key: &'static str, index: u64) -> Option<AnyHandle> {
        self.state()
            .children
            .iter()
            .find(|(k, i, _)| *k == key && *i == index)
            .map(|(_, _, h)| h.clone_ref())
    }

    /// The device that made this object, as a typed handle.
    pub fn device_handle(&self) -> Option<Fk<kind::Device>> {
        self.state()
            .device
            .as_ref()
            .and_then(AnyHandle::cast::<kind::Device>)
    }
}

/// Resource indices in use, per space.
#[derive(Debug, Default)]
struct Indices {
    used: Vec<(Space, u64)>,
    // A set: a reservation may hold thousands (the graphics device's texture ids).
    reserved: std::collections::HashSet<(Space, u64)>,
}

/// A resource index held by a live object; freed when the object dies.
#[derive(Debug)]
pub struct IndexGuard {
    world: Weak<World>,
    space: Space,
    index: u64,
}

impl IndexGuard {
    /// The index.
    pub fn index(&self) -> u64 {
        self.index
    }
}

impl Drop for IndexGuard {
    fn drop(&mut self) {
        if let Some(w) = self.world.upgrade() {
            let mut g = w.indices.lock();
            if let Some(p) = g
                .used
                .iter()
                .position(|&(s, i)| s == self.space && i == self.index)
            {
                let _ = g.used.swap_remove(p);
            }
        }
    }
}

/// Everything one fake "process" shares between its objects and services.
pub struct World {
    /// The call log.
    pub log: CallLog,
    /// The script.
    pub script: Script,
    /// Virtual time.
    pub clock: VirtualClock,
    /// The address space.
    pub vm: FakeVm,
    this: Weak<World>,
    next_id: AtomicU64,
    next_unique: AtomicU64,
    gpu_va: Mutex<u64>,
    indices: Mutex<Indices>,
    registry: Mutex<Vec<Weak<FakeObj>>>,
    /// Committed command buffers not yet completed, held strongly: Metal retains a committed
    /// buffer until it completes, whoever else lets it go.
    committed: Mutex<Vec<Arc<FakeObj>>>,
    auto_complete: Mutex<Option<CompleteOn>>,
    library_functions: Mutex<Vec<String>>,
    /// What [`World::on_new`] runs on the next object of a kind, in the order added.
    on_new: Mutex<Vec<(TypeId, Box<dyn Any + Send>)>>,
    /// The objects whose next release dies ([`World::die_at_release`]).
    release_dies: Mutex<Vec<ObjId>>,
    /// The objects whose next release holds its thread ([`World::hold_at_release`]).
    release_holds: Mutex<Vec<(ObjId, HeldCall)>>,
    /// Committed command buffers whose work the platform has not taken, in commit order, while
    /// submissions are held ([`World::hold_submissions`]); `None` while they are not.
    held_submissions: Mutex<Option<Vec<Arc<FakeObj>>>>,
    /// Whether the next commit, on whichever thread, first takes the held work
    /// ([`World::release_submissions_at_next_commit`]).
    release_at_commit: AtomicBool,
    /// The shared events' notifications not yet due, each with its event and value
    /// ([`notify_event`]), in the order asked for.
    notifications: Mutex<Vec<(ObjId, u64, Notification)>>,
    /// Whether a command buffer's `split` commits it as a part and answers the next
    /// ([`World::split_command_buffers`]).
    splits: AtomicBool,
    /// What the devices' [`GpuDevice::copies_guest_ram`] answers ([`World::copy_guest_ram`]).
    ///
    /// [`GpuDevice::copies_guest_ram`]: crate::platform::device::GpuDevice::copies_guest_ram
    guest_ram_copies: AtomicBool,
    /// What the devices' [`GpuDevice::surface_pages_in_place`] answers
    /// ([`World::answer_surface_pages`]), and the planes each call was asked of, in order.
    ///
    /// [`GpuDevice::surface_pages_in_place`]: crate::platform::device::GpuDevice::surface_pages_in_place
    surface_pages: Mutex<(PagesInPlace, Vec<Vec<PagesPlane>>)>,
}

/// A test's model setters for the next object of kind `K` ([`World::on_new`]).
type OnNew<K> = Box<dyn FnOnce(&Fk<K>) + Send>;

impl std::fmt::Debug for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("World").finish_non_exhaustive()
    }
}

impl World {
    /// A fresh world.
    pub fn new() -> Arc<World> {
        let log = CallLog::new();
        let script = Script::new();
        Arc::new_cyclic(|this| World {
            vm: FakeVm::new(log.clone(), script.clone()),
            log,
            script,
            clock: VirtualClock::new(),
            this: this.clone(),
            next_id: AtomicU64::new(1),
            next_unique: AtomicU64::new(1),
            gpu_va: Mutex::new(GPU_VA_BASE),
            indices: Mutex::new(Indices::default()),
            registry: Mutex::new(Vec::new()),
            committed: Mutex::new(Vec::new()),
            auto_complete: Mutex::new(None),
            library_functions: Mutex::new(Vec::new()),
            on_new: Mutex::new(Vec::new()),
            release_dies: Mutex::new(Vec::new()),
            release_holds: Mutex::new(Vec::new()),
            held_submissions: Mutex::new(None),
            release_at_commit: AtomicBool::new(false),
            notifications: Mutex::new(Vec::new()),
            splits: AtomicBool::new(false),
            guest_ram_copies: AtomicBool::new(false),
            surface_pages: Mutex::new((PagesInPlace::Yes, Vec::new())),
        })
    }

    /// Has the devices answer `answer` to [`GpuDevice::surface_pages_in_place`] from now on;
    /// [`PagesInPlace::Yes`] at first.
    ///
    /// [`GpuDevice::surface_pages_in_place`]: crate::platform::device::GpuDevice::surface_pages_in_place
    pub fn answer_surface_pages(&self, answer: PagesInPlace) {
        self.surface_pages.lock().0 = answer;
    }

    /// What the devices answer to [`GpuDevice::surface_pages_in_place`] asked of `planes`, the
    /// call recorded ([`World::surface_pages_asked`]).
    ///
    /// [`GpuDevice::surface_pages_in_place`]: crate::platform::device::GpuDevice::surface_pages_in_place
    pub fn surface_pages_in_place(&self, planes: &[PagesPlane]) -> PagesInPlace {
        let mut state = self.surface_pages.lock();
        state.1.push(planes.to_vec());
        state.0.clone()
    }

    /// The planes each call of [`GpuDevice::surface_pages_in_place`] was asked of, in order.
    ///
    /// [`GpuDevice::surface_pages_in_place`]: crate::platform::device::GpuDevice::surface_pages_in_place
    pub fn surface_pages_asked(&self) -> Vec<Vec<PagesPlane>> {
        self.surface_pages.lock().1.clone()
    }

    /// From now on a command buffer's `split` commits it as a part of the command buffer, as a
    /// commit does, and answers a new command buffer that goes on from it, as a platform that
    /// splits command buffers does (`CommandBuffer::split`); until then it answers `None`, the
    /// default, as Metal's does, and logs nothing.
    pub fn split_command_buffers(&self) {
        self.splits.store(true, Ordering::SeqCst);
    }

    /// Whether a command buffer's `split` splits it ([`World::split_command_buffers`]).
    pub fn splits(&self) -> bool {
        self.splits.load(Ordering::SeqCst)
    }

    /// Has the devices copy the guest's RAM by its runs ([`GpuDevice::copies_guest_ram`]), or
    /// not, as a device of the Vulkan platform handed the guest's RAM whole does; not at first.
    /// The copies answer as scripted ([`GUEST_RAM_TO_BUFFER`], [`BUFFER_TO_GUEST_RAM`]).
    ///
    /// [`GpuDevice::copies_guest_ram`]: crate::platform::device::GpuDevice::copies_guest_ram
    /// [`GUEST_RAM_TO_BUFFER`]: super::impls::blit::GUEST_RAM_TO_BUFFER
    /// [`BUFFER_TO_GUEST_RAM`]: super::impls::blit::BUFFER_TO_GUEST_RAM
    pub fn copy_guest_ram(&self, on: bool) {
        self.guest_ram_copies.store(on, Ordering::SeqCst);
    }

    /// What the devices' [`GpuDevice::copies_guest_ram`] answers.
    ///
    /// [`GpuDevice::copies_guest_ram`]: crate::platform::device::GpuDevice::copies_guest_ram
    pub fn copies_guest_ram(&self) -> bool {
        self.guest_ram_copies.load(Ordering::SeqCst)
    }

    fn arc(&self) -> Arc<World> {
        // `this` is set at construction and `self` is alive, so the upgrade holds.
        match self.this.upgrade() {
            Some(w) => w,
            None => crate::testkit::capture::die(
                crate::raise::death::DeathKind::Abort,
                "fake/objects.rs",
            ),
        }
    }

    /// A new object of kind `K` with `model`, registered for lookup. The first of
    /// [`World::on_new`]'s setters for `K` runs on it before it is returned.
    pub fn new_obj<K: Kind>(&self, model: Model) -> Fk<K> {
        let id = ObjId(self.next_id.fetch_add(1, Ordering::SeqCst));
        let obj = Arc::new(FakeObj {
            id,
            class: K::CLASS,
            world: self.arc(),
            handles: AtomicI64::new(0),
            state: Mutex::new(State {
                model,
                ..State::default()
            }),
            cv: Condvar::new(),
        });
        let mut reg = self.registry.lock();
        reg.retain(|w| w.strong_count() > 0);
        reg.push(Arc::downgrade(&obj));
        drop(reg);
        self.log.event(Event::Created {
            obj: id,
            class: K::CLASS,
        });
        let made = Fk::from_obj(obj);
        let hook = {
            let mut g = self.on_new.lock();
            g.iter()
                .position(|(k, _)| *k == TypeId::of::<K>())
                .map(|i| g.remove(i).1)
        };
        if let Some(Ok(set)) = hook.map(|h| h.downcast::<OnNew<K>>()) {
            set(&made);
        }
        made
    }

    /// Runs `set` on the next object of kind `K` this world makes, as it is made: a test's model
    /// setters for an object core makes, such as a texture's `gpuResourceID` or a rate-map
    /// layer's sample storage. Each call is for one object, in the order added.
    pub fn on_new<K: Kind>(&self, set: impl FnOnce(&Fk<K>) + Send + 'static) {
        let set: OnNew<K> = Box::new(set);
        self.on_new.lock().push((TypeId::of::<K>(), Box::new(set)));
    }

    /// Makes the next release of a handle of object `id` die at `release`, as a platform that
    /// dies letting go of an object does (the Vulkan platform's release of a no-copy buffer whose
    /// deallocator the runtime delays): in a refusal scope it is refused, outside every scope it
    /// ends the process.
    pub fn die_at_release(&self, id: ObjId) {
        self.release_dies.lock().push(id);
    }

    /// Whether this release of object `id` dies ([`World::die_at_release`]): once.
    fn take_release_death(&self, id: ObjId) -> bool {
        let mut dies = self.release_dies.lock();
        match dies.iter().position(|d| *d == id) {
            Some(i) => {
                dies.remove(i);
                true
            }
            None => false,
        }
    }

    /// Holds the next release of a handle of object `id` on its thread until the test lets
    /// `held` go, as a platform holds the thread that lets an object go where its driver takes
    /// long to free the object's memory or an import of the guest's pages.
    pub fn hold_at_release(&self, id: ObjId, held: HeldCall) {
        self.release_holds.lock().push((id, held));
    }

    /// The hold this release of object `id` waits in ([`World::hold_at_release`]): once.
    fn take_release_hold(&self, id: ObjId) -> Option<HeldCall> {
        let mut holds = self.release_holds.lock();
        let at = holds.iter().position(|(held, _)| *held == id)?;
        Some(holds.remove(at).1)
    }

    /// A live object by identity (a retained, erased handle).
    pub fn find(&self, id: ObjId) -> Option<AnyHandle> {
        self.registry
            .lock()
            .iter()
            .filter_map(Weak::upgrade)
            .find(|o| o.id == id)
            .map(AnyHandle::new)
    }

    /// The live objects of `class`, in creation order (leak checks).
    pub fn live(&self, class: &str) -> Vec<ObjId> {
        self.registry
            .lock()
            .iter()
            .filter_map(Weak::upgrade)
            .filter(|o| o.class == class)
            .map(|o| o.id)
            .collect()
    }

    /// A fresh `uniqueIdentifier` / `gpuResourceID`.
    pub fn next_unique(&self) -> u64 {
        self.next_unique.fetch_add(1, Ordering::SeqCst)
    }

    /// A GPU VA range of `len` bytes (page granular; never freed).
    pub fn alloc_gpu_va(&self, len: u64) -> u64 {
        let mut g = self.gpu_va.lock();
        let va = *g;
        let size = num::align_up_wrapping(len.max(1), PAGE_SIZE);
        *g = va.wrapping_add(size);
        va
    }

    /// A resource index in `space`: `forced` as given, else the smallest index from 1 that is
    /// neither in use nor reserved.
    pub fn alloc_index(&self, space: Space, forced: Option<u64>) -> IndexGuard {
        let mut g = self.indices.lock();
        let index = match forced {
            Some(i) => i,
            None => {
                let mut i = 1u64;
                while g.reserved.contains(&(space, i))
                    || g.used.iter().any(|&(s, x)| s == space && x == i)
                {
                    i = i.wrapping_add(1);
                }
                i
            }
        };
        g.used.push((space, index));
        IndexGuard {
            world: self.this.clone(),
            space,
            index,
        }
    }

    /// Marks `indices` of `space` reserved for the life of the process
    /// (`reserveResourceIndicesForResourceType:…`).
    pub fn reserve_indices(&self, space: Space, indices: &[u64]) {
        let mut g = self.indices.lock();
        for &i in indices {
            g.reserved.insert((space, i));
        }
    }

    /// The indices of `space` in use now (tests).
    pub fn indices_in_use(&self, space: Space) -> Vec<u64> {
        self.indices
            .lock()
            .used
            .iter()
            .filter(|(s, _)| *s == space)
            .map(|(_, i)| *i)
            .collect()
    }

    /// Makes every later `commit` complete at once (status Completed), running the handlers as
    /// `on` says; `None` turns it off (the default: buffers complete only on
    /// [`World::complete`]).
    pub fn set_auto_complete(&self, on: Option<CompleteOn>) {
        *self.auto_complete.lock() = on;
    }

    /// The auto-complete mode.
    pub fn auto_complete(&self) -> Option<CompleteOn> {
        *self.auto_complete.lock()
    }

    /// From now until [`World::release_submissions`], a committed command buffer's work is not
    /// taken at its commit: it stays `Committed`, its submission handlers, scheduled signals and
    /// scheduled handlers wait, as a platform holds a buffer whose wait is not yet satisfied (the
    /// Vulkan platform's wait for a shared event's value). Its completion is listed as at any
    /// commit.
    pub fn hold_submissions(&self) {
        let mut held = self.held_submissions.lock();
        if held.is_none() {
            *held = Some(Vec::new());
        }
    }

    /// Takes the work of every command buffer held since [`World::hold_submissions`], in commit
    /// order, as a commit takes it, and takes each later buffer's at its commit again; how many
    /// there were. One completed meanwhile ([`World::complete`]) was failed before the platform
    /// took it, and runs no submission handler.
    pub fn release_submissions(&self) -> usize {
        let held = self.held_submissions.lock().take().unwrap_or_default();
        for cb in &held {
            let completed = matches!(&cb.state.lock().model, Model::CommandBuffer(m)
                if m.status.raw() >= CommandBufferStatus::COMPLETED.raw());
            if !completed {
                submit_obj(cb);
            }
        }
        held.len()
    }

    /// The next command buffer committed, on whichever thread, first takes the work held since
    /// [`World::hold_submissions`], inside its commit and on its thread, as
    /// [`World::release_submissions`] does, then its own: as the Vulkan platform's runtime
    /// submits, inside a commit, the ready batches committed before it, and runs their submission
    /// handlers there.
    pub fn release_submissions_at_next_commit(&self) {
        self.release_at_commit.store(true, Ordering::Release);
    }

    /// Whether `cb`, just committed, is held ([`World::hold_submissions`]), listed if so.
    fn holds_submission(&self, cb: &Arc<FakeObj>) -> bool {
        match self.held_submissions.lock().as_mut() {
            Some(held) => {
                held.push(Arc::clone(cb));
                true
            }
            None => false,
        }
    }

    /// The names `functionNames` returns and the fake's libraries list.
    pub fn set_library_functions(&self, names: Vec<String>) {
        *self.library_functions.lock() = names;
    }

    /// The configured function names.
    pub fn library_functions(&self) -> Vec<String> {
        self.library_functions.lock().clone()
    }

    /// Keeps `cb` alive until it completes, as Metal retains a committed command buffer: a
    /// caller that drops its handle after `commit` (the GPU task does) still has it completed.
    pub(crate) fn note_committed(&self, cb: &Arc<FakeObj>) {
        let mut g = self.committed.lock();
        if !g.iter().any(|o| Arc::ptr_eq(o, cb)) {
            g.push(cb.clone());
        }
    }

    /// Lets a completed buffer go (outside the list's lock: the last release may drop it).
    fn note_completed(&self, cb: &Arc<FakeObj>) {
        let released = {
            let mut g = self.committed.lock();
            g.iter()
                .position(|o| Arc::ptr_eq(o, cb))
                .map(|i| g.remove(i))
        };
        drop(released);
    }

    /// The committed command buffers not yet completed, in commit order.
    pub fn committed(&self) -> Vec<ObjId> {
        self.committed
            .lock()
            .iter()
            .filter(|o| {
                matches!(&o.state.lock().model, Model::CommandBuffer(c)
                    if c.status.raw() < CommandBufferStatus::COMPLETED.raw())
            })
            .map(|o| o.id)
            .collect()
    }

    /// Completes the committed command buffer `id` with `status` and `error`: applies its
    /// `encodeSignalEvent:value:`s, wakes `waitUntilCompleted`, then runs its completed handlers
    /// in order as `on` says. `false` if no such live buffer.
    pub fn complete(
        &self,
        id: ObjId,
        status: CommandBufferStatus,
        error: Option<NsErrorText>,
        on: CompleteOn,
    ) -> bool {
        let Some(h) = self.find(id) else {
            return false;
        };
        complete_obj(h.obj(), status, error, on);
        true
    }

    /// Completes the committed command buffer `id` with status Error, as [`World::complete`]
    /// does, where the platform's read of it, as each completed handler runs, dies with `detail`
    /// (a failure the platform has no answer of Metal's for, which the Vulkan platform dies at on
    /// the runtime's thread): a handler added in a refusal scope (the graphics lane's) is handed
    /// the buffer refused ([`CompletedInfo::refused`]), and one added outside every scope dies
    /// with it. `false` if no such live buffer.
    pub fn complete_dying_read(&self, id: ObjId, detail: &str, on: CompleteOn) -> bool {
        let Some(h) = self.find(id) else {
            return false;
        };
        if let Model::CommandBuffer(m) = &mut h.obj().state.lock().model {
            m.read_dies = Some(detail.to_owned());
        }
        complete_obj(h.obj(), CommandBufferStatus::ERROR, None, on);
        true
    }

    /// Sets the GPU times the command buffer `id`'s completed handlers learn when it completes
    /// ([`CompletedInfo::gpu_times_ns`]). `false` if no such live buffer.
    pub fn set_gpu_times(&self, id: ObjId, start_ns: u64, end_ns: u64) -> bool {
        let Some(h) = self.find(id) else {
            return false;
        };
        if let Model::CommandBuffer(m) = &mut h.obj().state.lock().model {
            m.gpu_times_ns = Some((start_ns, end_ns));
        }
        true
    }

    /// Completes every committed buffer successfully, in commit order.
    pub fn complete_all(&self, on: CompleteOn) -> usize {
        let ids = self.committed();
        for id in &ids {
            let _ = self.complete(*id, CommandBufferStatus::COMPLETED, None, on);
        }
        ids.len()
    }
}

/// Sets an event's value, wakes its waiters, then runs the notifications it reached, on this
/// thread, in the order they were asked for ([`notify_event`]).
pub fn signal_event(ev: &Arc<FakeObj>, v: u64) {
    if let Model::Event { value } = &mut ev.state.lock().model {
        *value = v;
    }
    ev.cv.notify_all();
    run_due(ev);
}

/// Runs `block` once the event `ev` reaches `v`: at once, on this thread, where it has, else on
/// the thread whose signal or set raises it there ([`signal_event`]). One never reached stays
/// with the world.
pub fn notify_event(ev: &Arc<FakeObj>, v: u64, block: Notification) {
    ev.world.notifications.lock().push((ev.id, v, block));
    run_due(ev);
}

/// Runs the notifications of the event `target` whose value it reached, with no lock held.
fn run_due(target: &Arc<FakeObj>) {
    let value = event_value(target);
    let due: Vec<(ObjId, u64, Notification)> = {
        let mut pending = target.world.notifications.lock();
        let (due, kept) = std::mem::take(&mut *pending)
            .into_iter()
            .partition(|&(id, at, _)| id == target.id && at <= value);
        *pending = kept;
        due
    };
    for (_, _, block) in due {
        block();
    }
}

/// An event's value.
pub fn event_value(ev: &Arc<FakeObj>) -> u64 {
    match &ev.state.lock().model {
        Model::Event { value } => *value,
        _ => 0,
    }
}

/// Waits on the virtual clock until `ev` reaches `v` or `ms` virtual milliseconds pass.
pub fn wait_event(ev: &Arc<FakeObj>, v: u64, ms: u64) -> bool {
    let target = ev;
    let clock = &target.world.clock;
    let deadline = Deadline::Mono(
        clock
            .now_mono()
            .saturating_add(std::time::Duration::from_millis(ms)),
    );
    let mut g = target.state.lock();
    loop {
        let cur = match &g.model {
            Model::Event { value } => *value,
            _ => 0,
        };
        if cur >= v {
            return true;
        }
        if clock.wait_until(&target.cv, &mut g, deadline).timed_out() {
            let cur = match &g.model {
                Model::Event { value } => *value,
                _ => 0,
            };
            return cur >= v;
        }
    }
}

thread_local! {
    /// The detail of the death the platform's read of the command buffer whose completed handlers
    /// this thread runs dies with ([`World::complete_dying_read`]).
    static READ_DIES: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Runs `f`, the completed handlers of a command buffer whose read dies with `detail` where it is
/// `Some` ([`completion_read`]).
fn with_read_death<R>(detail: Option<String>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<String>);
    impl Drop for Restore {
        fn drop(&mut self) {
            READ_DIES.with(|d| *d.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(READ_DIES.with(|d| d.replace(detail)));
    f()
}

/// The platform's read of a command buffer as its completed handler runs: `info`, what the model
/// holds, or the death [`World::complete_dying_read`] set for it, at `commit`, the selector the
/// Vulkan platform dies at for a failure it reads there.
pub(crate) fn completion_read(info: CompletedInfo) -> CompletedInfo {
    if let Some(detail) = READ_DIES.with(|d| d.borrow().clone()) {
        crate::raise::Raise::platform("commit")
            .with_detail(detail)
            .fatal();
    }
    info
}

/// [`World::complete`]'s body.
pub fn complete_obj(
    cb: &Arc<FakeObj>,
    status: CommandBufferStatus,
    error: Option<NsErrorText>,
    on: CompleteOn,
) {
    let (handlers, signals, read_dies, gpu_times_ns) = {
        let mut s = cb.state.lock();
        let Model::CommandBuffer(m) = &mut s.model else {
            return;
        };
        m.status = status;
        m.error = error.clone();
        let mut handlers = std::mem::take(&mut m.seen);
        handlers.append(&mut m.handlers);
        (
            handlers,
            std::mem::take(&mut m.signal_on_complete),
            m.read_dies.take(),
            m.gpu_times_ns,
        )
    };
    for (ev, v) in &signals {
        signal_event(ev.obj(), *v);
    }
    cb.cv.notify_all();
    cb.world.note_completed(cb);
    let info = CompletedInfo {
        status,
        error,
        gpu_times_ns,
    };
    let run = move || {
        super::calllog::as_completion_thread(|| {
            with_read_death(read_dies, || {
                for h in handlers {
                    h(info.clone());
                }
            })
        })
    };
    match on {
        CompleteOn::Caller => {
            let _env = crate::rt::BlockEnv::foreign();
            run()
        }
        CompleteOn::Thread | CompleteOn::Detached => {
            let slot = crate::testkit::capture::current_slot();
            let spawned = std::thread::Builder::new()
                .name("com.Metal.CompletionQueueDispatch".into())
                .spawn(move || crate::testkit::capture::with_slot(slot, run));
            if on == CompleteOn::Thread
                && let Ok(j) = spawned
            {
                let _ = j.join();
            }
        }
    }
}

/// `commit`'s model: its submission handlers run, then Scheduled at once, the scheduled signals
/// applied, the buffer listed for completion (or completed at once under auto-complete). While
/// submissions are held ([`World::hold_submissions`]) it is only listed for completion, and the
/// rest waits for [`World::release_submissions`].
pub fn commit_obj(cb: &Arc<FakeObj>) {
    if cb.world.release_at_commit.swap(false, Ordering::AcqRel) {
        cb.world.release_submissions();
    }
    if cb.world.holds_submission(cb) {
        cb.world.note_committed(cb);
        return;
    }
    submit_obj(cb);
    cb.world.note_committed(cb);
    if let Some(on) = cb.world.auto_complete() {
        complete_obj(cb, CommandBufferStatus::COMPLETED, None, on);
    }
}

/// The platform takes a committed buffer's work: its submission handlers run, then Scheduled, the
/// scheduled signals applied and its scheduled handlers run.
fn submit_obj(cb: &Arc<FakeObj>) {
    // Its submission handlers, before it reads `Scheduled`.
    let submission = {
        let mut s = cb.state.lock();
        let Model::CommandBuffer(m) = &mut s.model else {
            return;
        };
        std::mem::take(&mut m.submission)
    };
    for h in submission {
        h();
    }
    let (signals, scheduled) = {
        let mut s = cb.state.lock();
        let Model::CommandBuffer(m) = &mut s.model else {
            return;
        };
        m.status = CommandBufferStatus::SCHEDULED;
        (
            std::mem::take(&mut m.signal_on_schedule),
            std::mem::take(&mut m.scheduled),
        )
    };
    for (ev, v) in &signals {
        signal_event(ev.obj(), *v);
    }
    for h in scheduled {
        h();
    }
    cb.cv.notify_all();
}

/// Blocks (on the virtual clock's untimed wait) until the buffer's status reaches `at_least`.
pub fn wait_status(cb: &Arc<FakeObj>, at_least: CommandBufferStatus) {
    let mut g = cb.state.lock();
    loop {
        let st = match &g.model {
            Model::CommandBuffer(m) => m.status,
            _ => return,
        };
        if st.raw() >= at_least.raw() {
            return;
        }
        cb.world.clock.wait(&cb.cv, &mut g);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::num::ToU64;

    /// Handles count retains; erase and cast keep the object
    #[test]
    fn handle_counting() {
        let w = World::new();
        let b: Fk<kind::Buffer> = w.new_obj(Model::None);
        assert_eq!(b.retain_count(), 1);
        let b2 = b.clone_ref();
        assert_eq!(b.retain_count(), 2);
        assert_eq!(b2.id(), b.id());
        let e = b.erase();
        assert_eq!(b.retain_count(), 3);
        assert!(e.cast::<kind::Texture>().is_none());
        let back = e.cast::<kind::Buffer>().unwrap();
        assert_eq!(back.id(), b.id());
        drop((b2, e, back));
        assert_eq!(b.retain_count(), 1);
        assert_eq!(format!("{b:?}"), format!("MTLBuffer#{}", b.id().0));
    }

    /// Surface retains and releases are logged
    #[test]
    fn surface_retains_logged() {
        let w = World::new();
        let s: Fk<kind::Surface> = w.new_obj(Model::None);
        let s2 = s.clone_ref();
        drop(s2);
        assert_eq!(w.log.sels(), vec!["CFRetain", "CFRelease"]);
    }

    /// Lookup by id and live lists follow object lifetime
    #[test]
    fn registry() {
        let w = World::new();
        let a: Fk<kind::Fence> = w.new_obj(Model::None);
        let id = a.id();
        assert_eq!(w.live("MTLFence"), vec![id]);
        assert_eq!(w.find(id).map(|h| h.id()), Some(id));
        drop(a);
        assert!(w.find(id).is_none());
        assert!(w.live("MTLFence").is_empty());
    }

    /// Resource indices: smallest free from 1, reservations skipped, forced honoured, freed with
    /// the object
    #[test]
    fn resource_indices() {
        let w = World::new();
        w.reserve_indices(Space::TEXTURE, &[2]);
        let a = w.alloc_index(Space::TEXTURE, None);
        let b = w.alloc_index(Space::TEXTURE, None);
        assert_eq!((a.index(), b.index()), (1, 3));
        let f = w.alloc_index(Space::TEXTURE, Some(9));
        assert_eq!(f.index(), 9);
        let s = w.alloc_index(Space::SAMPLER, None);
        assert_eq!(s.index(), 1);
        drop(a);
        assert_eq!(w.alloc_index(Space::TEXTURE, None).index(), 1);
        let used = w.indices_in_use(Space::TEXTURE);
        assert_eq!(used.len(), 2);
        assert!(used.contains(&3) && used.contains(&9));
    }

    /// GPU VAs are page granular from the fake's base
    #[test]
    fn gpu_vas() {
        let w = World::new();
        assert_eq!(w.alloc_gpu_va(1), GPU_VA_BASE);
        assert_eq!(w.alloc_gpu_va(PAGE_SIZE + 1), GPU_VA_BASE + PAGE_SIZE);
        assert_eq!(w.alloc_gpu_va(0), GPU_VA_BASE + 3 * PAGE_SIZE);
        assert!(w.next_unique() < w.next_unique());
    }

    /// Children are made once and replaced by set_child
    #[test]
    fn children() {
        let w = World::new();
        let icb: Fk<kind::Icb> = w.new_obj(Model::None);
        let c1: Fk<kind::IndirectRenderCommand> = icb.child("render", 0, || Model::None);
        let c2: Fk<kind::IndirectRenderCommand> = icb.child("render", 0, || Model::None);
        assert_eq!(c1.id(), c2.id());
        let other: Fk<kind::IndirectRenderCommand> = w.new_obj(Model::None);
        icb.set_child("render", 0, other.erase());
        assert_eq!(icb.get_child("render", 0).map(|h| h.id()), Some(other.id()));
        assert!(icb.get_child("x", 0).is_none());
    }

    /// A typed model reads as its default until the first write installs it; the texture
    /// descriptor's defaults
    #[test]
    fn typed_models() {
        use super::super::impls::descriptor::TextureDescriptorModel;
        let w = World::new();
        let d: Fk<kind::TextureDescriptor> = w.new_obj(Model::None);
        assert_eq!(d.desc(), TexDesc::default());
        assert!(matches!(d.state().model, Model::None));
        d.with_model_mut(|m: &mut TextureDescriptorModel| {
            m.desc.width = 64;
            m.desc.resource_options = 0x20;
            m.desc.resource_index = Some(7);
        });
        assert!(matches!(d.state().model, Model::TextureDescriptor(_)));
        let t = d.desc();
        assert_eq!(
            (t.width, t.storage_mode(), t.resource_index),
            (64, 2, Some(7))
        );
        assert_eq!(t.height, 1);
    }

    /// `on_new` runs once, on the next object of its kind as it is made, in the order added
    #[test]
    fn on_new_sets_the_next_object_of_its_kind() {
        let w = World::new();
        w.on_new::<kind::Texture>(|t| t.set_gpu_resource_id(5));
        w.on_new::<kind::Texture>(|t| t.set_gpu_resource_id(6));
        let texture =
            || -> Fk<kind::Texture> { w.new_obj(Model::Texture(TextureModel::default())) };
        let b: Fk<kind::Buffer> = w.new_obj(Model::Buffer(BufferModel::default()));
        let made = [texture(), texture(), texture()];
        assert_eq!(made.map(|t| t.gpu_resource_id_value()), [5, 6, 0]);
        assert_eq!(b.gpu_resource_id_value(), 0);
    }

    /// Commit, complete with handlers on the caller and on a thread, and the waits
    #[test]
    fn command_buffer_lifecycle() {
        let w = World::new();
        let cb: Fk<kind::CommandBuffer> =
            w.new_obj(Model::CommandBuffer(CommandBufferModel::default()));
        let ev: Fk<kind::SharedEvent> = w.new_obj(Model::Event { value: 0 });
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s2 = seen.clone();
        if let Model::CommandBuffer(m) = &mut cb.state().model {
            m.handlers.push(Box::new(move |i| {
                s2.lock()
                    .push((i.status, std::thread::current().name().map(str::to_owned)))
            }));
            m.signal_on_complete.push((ev.erase(), 5));
            m.signal_on_schedule.push((ev.erase(), 2));
        }
        commit_obj(cb.obj());
        assert_eq!(event_value(ev.obj()), 2);
        wait_status(cb.obj(), CommandBufferStatus::SCHEDULED);
        assert_eq!(w.committed(), vec![cb.id()]);
        assert!(w.complete(
            cb.id(),
            CommandBufferStatus::ERROR,
            None,
            CompleteOn::Thread
        ));
        assert_eq!(event_value(ev.obj()), 5);
        wait_status(cb.obj(), CommandBufferStatus::COMPLETED);
        assert_eq!(
            *seen.lock(),
            vec![(
                CommandBufferStatus::ERROR,
                Some("com.Metal.CompletionQueueDispatch".to_string())
            )]
        );
        assert!(w.committed().is_empty());
        assert!(!w.complete(
            ObjId(999),
            CommandBufferStatus::COMPLETED,
            None,
            CompleteOn::Caller
        ));
    }

    /// A committed buffer stays alive, and completable, after its last handle goes, until it
    /// completes (Metal's retain of an in-flight buffer)
    #[test]
    fn committed_buffers_live_until_completed() {
        let w = World::new();
        let cb: Fk<kind::CommandBuffer> =
            w.new_obj(Model::CommandBuffer(CommandBufferModel::default()));
        let id = cb.id();
        let weak = Arc::downgrade(cb.obj());
        commit_obj(cb.obj());
        drop(cb);
        assert!(weak.upgrade().is_some());
        assert_eq!(w.committed(), vec![id]);
        assert_eq!(w.complete_all(CompleteOn::Caller), 1);
        assert!(w.committed().is_empty());
        assert!(weak.upgrade().is_none());
    }

    /// Caller-mode handlers run owning no queue and marked ours, as on Metal's completion thread;
    /// the caller's mark is restored after
    #[test]
    fn caller_mode_handlers_run_as_foreign_thread() {
        use crate::rt::block_env::current_label;
        use crate::rt::is_our_thread;
        let w = World::new();
        let cb: Fk<kind::CommandBuffer> =
            w.new_obj(Model::CommandBuffer(CommandBufferModel::default()));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s2 = seen.clone();
        if let Model::CommandBuffer(m) = &mut cb.state().model {
            m.handlers.push(Box::new(move |_| {
                s2.lock().push((current_label(), is_our_thread()));
            }));
        }
        commit_obj(cb.obj());
        assert!(w.complete(
            cb.id(),
            CommandBufferStatus::COMPLETED,
            None,
            CompleteOn::Caller
        ));
        assert_eq!(*seen.lock(), vec![(None, true)]);
        assert!(!is_our_thread());
    }

    /// Auto-complete completes at commit
    #[test]
    fn auto_complete() {
        let w = World::new();
        w.set_auto_complete(Some(CompleteOn::Caller));
        assert_eq!(w.auto_complete(), Some(CompleteOn::Caller));
        let cb: Fk<kind::CommandBuffer> =
            w.new_obj(Model::CommandBuffer(CommandBufferModel::default()));
        commit_obj(cb.obj());
        let st = match &cb.state().model {
            Model::CommandBuffer(m) => m.status,
            _ => CommandBufferStatus(0),
        };
        assert_eq!(st, CommandBufferStatus::COMPLETED);
        assert_eq!(w.complete_all(CompleteOn::Caller), 0);
    }

    /// Event waits on virtual time
    #[test]
    fn event_waits() {
        let w = World::new();
        let ev: Fk<kind::SharedEvent> = w.new_obj(Model::Event { value: 0 });
        signal_event(ev.obj(), 3);
        assert_eq!(event_value(ev.obj()), 3);
        assert!(wait_event(ev.obj(), 3, 5000));
        let o = ev.obj().clone();
        let h = std::thread::Builder::new()
            .spawn(move || wait_event(&o, 4, 5000))
            .unwrap();
        assert!(
            w.clock
                .wait_for_waiters(1, std::time::Duration::from_secs(10))
        );
        w.clock.advance(std::time::Duration::from_millis(5000));
        assert!(!h.join().unwrap());
    }

    /// A buffer's deallocators run when it dies
    #[test]
    fn deallocators_run_at_death() {
        let w = World::new();
        let hits = Arc::new(Mutex::new(Vec::new()));
        let (h1, h2) = (hits.clone(), hits.clone());
        let b: Fk<kind::Buffer> = w.new_obj(Model::Buffer(BufferModel {
            contents: HostPtr::from_addr(0x1000),
            length: 16,
            deallocator: Some(Box::new(move || h1.lock().push(0))),
            desc_deallocator: Some(DescDeallocator(Box::new(move |p, n| {
                h2.lock().push(p.addr().to_u64() + n)
            }))),
            ..BufferModel::default()
        }));
        drop(b);
        assert_eq!(*hits.lock(), vec![0, 0x1010]);
    }

    /// The events of `w`'s log: object lifetimes.
    fn lifetimes(w: &World) -> Vec<Event> {
        w.log
            .events()
            .into_iter()
            .map(|r| r.event)
            .filter(|e| matches!(e, Event::Created { .. } | Event::Released { .. }))
            .collect()
    }

    fn released(obj: ObjId, class: &'static str, context: ReleaseContext) -> Event {
        Event::Released {
            obj,
            class,
            context,
        }
    }

    fn ctx(thread: ThreadClass) -> ReleaseContext {
        ReleaseContext { thread }
    }

    use super::super::calllog::ThreadClass;

    /// With events off a world records none; with them on, each object's creation and last release
    /// (not a handle's drop), with the release context of the releasing thread
    #[test]
    fn creation_and_last_release_are_recorded_with_their_context() {
        let w = World::new();
        let quiet: Fk<kind::Buffer> = w.new_obj(Model::None);
        drop(quiet);
        assert!(w.log.events().is_empty());

        w.log.record_events(true);
        let b: Fk<kind::Buffer> = w.new_obj(Model::None);
        let b2 = b.clone_ref();
        drop(b);
        assert_eq!(
            lifetimes(&w),
            vec![Event::Created {
                obj: b2.id(),
                class: "MTLBuffer"
            }]
        );
        let id = b2.id();
        drop(b2);
        assert_eq!(
            lifetimes(&w)[1..],
            [released(id, "MTLBuffer", ctx(ThreadClass::Other))]
        );
        // Recorded after no call: `at` is the next call's seq.
        let last = w
            .log
            .events()
            .into_iter()
            .rfind(|r| matches!(r.event, Event::Released { .. }))
            .unwrap();
        assert_eq!(last.at, w.log.mark());
        assert_eq!(
            last.thread_name,
            std::thread::current().name().map(str::to_owned)
        );
    }

    /// The release context carries the completion thread
    #[test]
    fn release_context_reads_the_thread() {
        let w = World::new();
        w.log.record_events(true);
        let mut want = Vec::new();

        // On the test's thread.
        let t: Fk<kind::Texture> = w.new_obj(Model::None);
        want.push(released(t.id(), "MTLTexture", ctx(ThreadClass::Other)));
        drop(t);

        // On the completion thread.
        let f: Fk<kind::Fence> = w.new_obj(Model::None);
        want.push(released(f.id(), "MTLFence", ctx(ThreadClass::Completion)));
        super::super::calllog::as_completion_thread(|| drop(f));

        // In a sync block of a queue, on the caller's thread.
        let p = super::super::platform::FakePlatform::new();
        let q = crate::rt::queue::SerialQueue::new(
            crate::platform::Platform::threads(&p),
            "test.events.sync",
        );
        let h: Fk<kind::Heap> = w.new_obj(Model::None);
        want.push(released(h.id(), "MTLHeap", ctx(ThreadClass::Other)));
        q.sync(move || drop(h));

        let got: Vec<Event> = lifetimes(&w)
            .into_iter()
            .filter(|e| matches!(e, Event::Released { .. }))
            .collect();
        assert_eq!(got, want);
    }

    /// A release in an async block runs on a dispatch worker; one on a thread core started runs on
    /// a Core thread
    #[test]
    fn release_context_names_worker_and_core_threads() {
        let p = super::super::platform::FakePlatform::new();
        let w = p.world().clone();
        p.record_events(true);
        let threads = crate::platform::Platform::threads(&p);
        let q = crate::rt::queue::SerialQueue::new(threads, "test.events.async");
        let a: Fk<kind::Buffer> = w.new_obj(Model::None);
        let a_id = a.id();
        let (tx, rx) = std::sync::mpsc::channel();
        let tx2 = tx.clone();
        q.exec_async(move || {
            drop(a);
            let _ = tx2.send(());
        });
        rx.recv().unwrap();
        let c: Fk<kind::Buffer> = w.new_obj(Model::None);
        let c_id = c.id();
        crate::rt::thread::spawn_named(
            threads,
            "test.events.thread",
            1 << 20,
            Box::new(()),
            move || {
                drop(c);
                let _ = tx.send(());
            },
        );
        rx.recv().unwrap();
        let recs: Vec<_> = p
            .events()
            .into_iter()
            .filter(|r| matches!(r.event, Event::Released { .. }))
            .collect();
        assert_eq!(recs.len(), 2);
        assert_eq!(
            recs[0].event,
            released(a_id, "MTLBuffer", ctx(ThreadClass::Worker))
        );
        assert_eq!(recs[0].thread_name, None);
        assert_eq!(
            recs[1].event,
            released(c_id, "MTLBuffer", ctx(ThreadClass::Core))
        );
        assert_eq!(recs[1].thread_name.as_deref(), Some("test.events.thread"));
        assert_ne!(recs[0].thread, recs[1].thread);
    }
}
