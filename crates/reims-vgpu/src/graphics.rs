//! [`GraphicsDevice`]: the reims-graphics device (PCI 1234:5247), `reims-graphics`'s host side
//! with its display shown in a window of the library's, made from a [`GraphicsConfig`].
//!
//! The device's region (BAR 2) is made here and shared: the device takes the guest's commands
//! from its rings, and the window's sink imports the region's pages under the scanout as a
//! no-copy buffer on the platform (Metal on macOS, the Vulkan platform's runtime with the
//! `vulkan` feature) and blits the scanout out of it into each drawable, on the GPU, with no copy
//! on the CPU (`sink`). The buffer's deallocator holds the region, so it stays mapped until the
//! GPU has let the buffer go, after the device and the window.
//!
//! **The window** ([`WindowScreen`]) is a display window of `reims-vgpu-window` that keeps its
//! size (`WindowConfig::keeps_its_size`): the guest follows it. It opens sized for the config's
//! mode, else 1920 × 1080, and the display then offers the guest one mode, the window's size in
//! pixels as its render thread last drew into it, at the refresh rate of the host screen it
//! opened on; when that size changes (the person resized the window) the device tells the
//! guest (`DISPLAY_CHANGED`). The guest's scanout shows in the window at once and then whenever
//! the host's dirty log says the guest wrote it, at the screen's rate; the window draws it at
//! the screen's pace. The guest's cursor image is the host's cursor over the image, shown while
//! the guest shows its cursor; the guest's cursor position moves nothing, since the host's
//! pointer moves the guest's (the window never draws the cursor into the image itself).
//!
//! **Without a window** the display offers 1920 × 1080 at 60 Hz and shows nothing.
//!
//! Every public method runs as an entry ([`crate::entry`]).

use std::borrow::Cow;
use std::marker::PhantomData;
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use parking_lot::lock_api::ArcRwLockReadGuard;
use parking_lot::{RawRwLock, RwLock};
use reims_graphics::ops::RgfxCommandBuffer;
use reims_graphics::region::Region;
use reims_graphics::wire::{
    RGFX_BLOCK_RELEASE_NO_CLIENT, RGFX_CMD_ARGUMENT_LAYOUT, RGFX_CMD_BUFFER_OVER_PAGES,
    RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_COMPUTE_PIPELINE_CREATE,
    RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN, RGFX_CMD_ICB_CREATE, RGFX_CMD_ICB_SLOTS,
    RGFX_CMD_ICB_WRITE, RGFX_CMD_LIBRARY_CREATE, RGFX_CMD_LIBRARY_FROM_REGION,
    RGFX_CMD_LIBRARY_LOOKUP, RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_QUEUE_CREATE,
    RGFX_CMD_RENDER_PIPELINE_CREATE, RGFX_CMD_SAMPLER_CREATE, RGFX_CONTEXT_OBJ_ID_SHIFT,
    RGFX_DEVICE_INFO_OFFSET, RGFX_EVT_ICB_STATE, RGFX_EVT_SAMPLER_STATE, RGFX_FEATURE_CB_SUBMITTED,
    RGFX_FEATURE_COMMIT_ORDER, RGFX_FEATURE_HEAP_OVER_PAGES, RGFX_FEATURE_ICB_SLOTS,
    RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS, RGFX_FEATURE_MANAGED_AT_OFFSET,
    RGFX_FEATURE_PAGES_BUFFERS, RGFX_FEATURE_SUBMIT_ROOM, RGFX_FEATURE_SURFACE_CACHE,
    RGFX_FEATURE_SURFACE_DEFERRED, RGFX_FEATURE_SURFACE_ORDER, RGFX_FEATURE_SURFACE_PAGES,
    RGFX_FEATURE_UNSHARED_QUEUES, RGFX_RESOURCE_ID_CHUNK, RGFX_RESOURCE_ID_CHUNKS,
    RGFX_STATUS_INVALID_ARGUMENT, RGFX_STATUS_OK, RGFX_STATUS_REFUSED, RGFX_VIDEO_CAPS_CODECS,
    RGFX_VIDEO_CAPS_OFFSET, RGFX_VIDEO_CAPS_OUTPUT_CACHED, RGFX_VIDEO_CAPS_VERSION, RgfxBlockGrant,
    RgfxBlockRelease, RgfxBufferOverPages, RgfxCbComplete, RgfxCbScheduled, RgfxCbSubmitted,
    RgfxEventSignalled, RgfxIcbState, RgfxLibraryLookup, RgfxMsgHeader, RgfxPageRun,
    RgfxPipelineReflection, RgfxSamplerState, RgfxSurfacePages, RgfxVideoCaps,
};
use reims_graphics::{
    CompleteAnswer, CompletionPoster, CursorImage, FixedScreen, Host, Mode, PresentAnswer, Scanout,
    Screen,
};
use reims_vgpu_core::memory::map::MemoryMap as CoreMemoryMap;
use reims_vgpu_core::num::ToU64;
use reims_vgpu_core::platform::command::CommandBuffer as _;
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::os::clock::Clock as _;
use reims_vgpu_core::platform::resource::Texture;
use reims_vgpu_core::platform::values::TextureDesc;
use reims_vgpu_core::platform::{Handle, Platform};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_window::{FrameSink, Window, WindowConfig};
use zerocopy::{FromBytes, IntoBytes};

use crate::DefaultPlatform;
use crate::device::{GuestMemoryLease, MemoryMap};
use crate::display::no_catch;
use crate::entry;
use crate::error::Error;
use crate::graphics::compile::{CompileKind, CompileOf, CompilePool, Pending};
use crate::graphics::completion::cb_failed;
use crate::graphics::decode::worker::{Unsettled, Worker};
use crate::graphics::decode::{
    Begun, CompletionSink, DecodeError, Decoders, DrainAnswer, FailedContexts, GraphicsDecoder,
    LibraryJob, LibraryStart, MadeLibrary, PipelineJob, ReflectionSink, ScheduledSink, SignalSink,
    Sinks, SubmittedSink, SurfacePresent, compile_refusing, complete, make_library, refused_reply,
    refusing,
};
use crate::graphics::device_info::device_info;
use crate::graphics::present::{Drawing, PresentAnswers, PresentQueue};
use crate::platform::ApiPlatform;
use crate::window::WindowLink;

mod compile;
mod completion;
mod decode;
mod device_info;
#[cfg(target_os = "macos")]
mod metal;
mod present;
mod sink;
#[cfg(feature = "vulkan")]
mod vulkan;

#[cfg(feature = "vulkan")]
pub(crate) use vulkan::graphics_video as vulkan_graphics_video;

pub use decode::{GraphicsVideo, GraphicsVideoSession, VideoDecode, VideoPlane};

/// Raises MSI-X vector `vector`: the device's interrupt.
pub type GraphicsRaiseFn = Arc<dyn Fn(u32) + Send + Sync>;
/// Whether the guest wrote any of the `length` bytes at `offset` in the region since the last
/// ask: the host's dirty log over the scanout, taken and cleared.
pub type ScanoutWrittenFn = Arc<dyn Fn(u64, u64) -> bool + Send + Sync>;

/// A graphics device's callbacks. `scanout_written` is called on the device's refresh thread
/// (`reims-vgpu.graphics-refresh`), and `raise_interrupt` on the thread that posts the event it
/// raises for: the device's control thread (`reims-vgpu.graphics-control`), a lane that carries
/// out a guest process's control messages (`reims-vgpu.graphics-context-<n>`), its presents
/// (`reims-vgpu.graphics-presents`) or a block's pages and downloads
/// (`reims-vgpu.graphics-block-<offset>`), its refresh thread, a command queue's service thread
/// (`reims-vgpu.graphics-queue-<id>`) or one of its lane's threads
/// (`reims-vgpu.graphics-lane-<id>`), a pipeline compile's thread
/// (`reims-vgpu.graphics-compile-<n>`), the surface worker (`reims-vgpu.graphics-surfaces-<i>`),
/// the indirect command buffers' worker (`reims-vgpu.icb-create-<i>`), the worker that lets go of
/// a closed process's objects and answers its close (`reims-vgpu.graphics-closes-<i>`), or the
/// platform's thread that runs a command buffer's scheduled or completed handler; or the host's
/// own, inside [`GraphicsDevice::mmio_write_taking`].
/// Each is called never inside another call of the host's into the device, and with no lock held
/// that a function of the device's waits for, directly or through a thread of the library's, so
/// they may call it at once. `scanout_written` runs with no lock of the library's held.
/// `raise_interrupt` may run under these: a command queue's claim on its next message, on the
/// queue's service thread and inside `mmio_write_taking`; there, at a command buffer's
/// `CB_SUBMITTED` answered on the calling thread, the decoder's contexts' lock, the buffer's
/// context's table read, its queue's answer slot and the reset gate read; the device caches'
/// lock, with the download's answer slot, at a `SURFACE_DOWNLOAD` answered at once; a command
/// buffer's `CB_SUBMITTED` latch, wherever that is posted; and the presents' queue, at a present
/// answered at once with no window. No function of the device takes any of them, and
/// `mmio_write_taking` only tries the claim and the decoder's locks.
#[derive(Clone, Default)]
pub struct GraphicsCallbacks {
    /// Raises the device's interrupt. Without it the guest gets no interrupt.
    pub raise_interrupt: Option<GraphicsRaiseFn>,
    /// The host's dirty log over the scanout, on the refresh thread. Without it the guest's
    /// scanout is taken as written at every refresh.
    pub scanout_written: Option<ScanoutWrittenFn>,
}

/// A graphics device's config.
#[derive(Clone, Default)]
pub struct GraphicsConfig {
    /// Its callbacks.
    pub callbacks: GraphicsCallbacks,
    /// The display's window: its input callbacks and the mode it opens sized for; `None` is no
    /// window. The window keeps its size whatever this says (the module doc).
    pub window: Option<WindowConfig>,
    /// The guest's RAM, with the files behind it, through which the GPU uses an IOSurface's own
    /// pages in place (`RGFX_FEATURE_SURFACE_PAGES`, where the host's page is the guest's 4 KiB);
    /// `None` keeps each surface's GPU view a copy of its pages in the region. The pages stay the
    /// guest's RAM while the device lives: a host that discards guest RAM (a balloon, hot-unplug)
    /// gives none.
    pub memory_map: Option<MemoryMap>,
}

/// The bytes of the host GPU's memory deferred blocks' device caches may hold, which
/// `DEFERRED_BUDGET_LO` and `_HI` read, of a device offering `features` on a GPU whose own memory's
/// budget is `own_memory` ([`GpuDevice::own_memory_budget`]): a quarter of it, as the wire gives
/// it (`RGFX_REG_DEFERRED_BUDGET_LO`); 0 without `SURFACE_DEFERRED`.
fn deferred_budget(features: u64, own_memory: u64) -> u64 {
    if features & u64::from(RGFX_FEATURE_SURFACE_DEFERRED) == 0 {
        return 0;
    }
    own_memory / 4
}

/// The mode a display offers before its window has a size, and without a window.
const FALLBACK_MODE: Mode = Mode {
    width: 1920,
    height: 1080,
    refresh_mhz: 60_000,
};

/// The reims-graphics device.
pub struct GraphicsDevice<P: ApiPlatform = DefaultPlatform> {
    /// The window, which closes when the link drops: before the device.
    window: Option<WindowLink>,
    /// The device.
    device: reims_graphics::Device,
    /// The stage-3 decoder and the device handle it drives, shared with the device's [`Host`],
    /// which feeds it the control ring's stage-3 messages; held here for the tests' look at it.
    #[cfg(any(test, feature = "testkit"))]
    control: Arc<GraphicsControl<P>>,
    /// The platform the decoder runs on.
    platform: PhantomData<fn() -> P>,
}

impl<P: ApiPlatform> std::fmt::Debug for GraphicsDevice<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphicsDevice")
            .field("device", &self.device)
            .field("window", &self.window.is_some())
            .finish()
    }
}

impl GraphicsDevice<DefaultPlatform> {
    /// A device on the process's platform.
    pub fn new(config: GraphicsConfig) -> Result<GraphicsDevice, Error> {
        GraphicsDevice::with_platform(DefaultPlatform::process(), config)
    }
}

impl<P: ApiPlatform> GraphicsDevice<P> {
    /// A device whose window draws on `platform`. [`Error::HostResources`] where the host
    /// refuses the region's memory, a pipe or a thread.
    pub fn with_platform(platform: Arc<P>, config: GraphicsConfig) -> Result<Self, Error> {
        entry::run(|_| {
            let size =
                usize::try_from(reims_graphics::REGION_SIZE).map_err(|_| Error::HostResources)?;
            let region = Arc::new(Region::new(size).ok_or(Error::HostResources)?);
            // The device the decoder drives, and its caps blob, which the guest bundle reads
            // before it makes any queue. A host with no GPU device cannot run the lane.
            let gpu_device = platform.metal_device(None).ok_or(Error::HostResources)?;
            // Before anything else makes a queue or a texture on it.
            reserve_guest_texture_ids(platform.as_ref(), &gpu_device);
            let clock = Arc::clone(&platform);
            let pool = CompilePool::new(Box::new(move || clock.clock().now_mono()));
            let identity = gpu_device.identity_caps();
            // `Managed` resources are one copy where the platform answers that the GPU shares the
            // CPU's memory, else two, which hold Metal's contract on any GPU: the guest's device
            // answers `hasUnifiedMemory` alike, from the blob.
            let unified_memory = identity.flags.has_unified_memory.unwrap_or(false);
            let mut info = device_info(&gpu_device.caps(), identity, pool.threads().to_u64());
            let mut decoder = GraphicsDecoder::new(Arc::clone(&platform), Arc::clone(&region));
            decoder.set_unified_memory(unified_memory);
            // The platform's lease on the guest's RAM, held by the views of the surfaces' pages
            // made through the map, so it outlives the last of them.
            if let Some(map) = &config.memory_map {
                let lease: GuestMemoryLease = Arc::from(platform.guest_memory(map, None));
                let memory = CoreMemoryMap::new(Arc::clone(&platform), &map.to_core())
                    .ok_or(Error::InvalidArgument)?;
                decoder.set_guest_memory(memory, lease);
            }
            // A GPU of its own memory keeps IOSurfaces' device caches where the kext asks, and
            // places its `Managed` buffers' CPU copies in the guest's heaps. Every host answers a
            // command buffer's ask for its submission, orders command buffers of different queues
            // over the IOSurfaces they share where the guest names the order, commits a process's
            // command buffers across its queues in the order it names, makes a queue the
            // kext asks unshared (WindowServer's) on the platform's unshared queue, and tells a
            // guest producer waiting for room on a submission ring that the device took from it
            // (the device carries SUBMIT_ROOM out). Both platforms make indirect command buffers,
            // so every host decodes their messages, slots included.
            // A host that maps the guest's pages maps a process's own pages too, for its no-copy
            // buffers (`BUFFER_OVER_PAGES`) and the heaps of its buffers' slabs
            // (`HEAP_OVER_PAGES`).
            let in_place = u64::from(RGFX_FEATURE_SURFACE_PAGES)
                | u64::from(RGFX_FEATURE_PAGES_BUFFERS)
                | u64::from(RGFX_FEATURE_HEAP_OVER_PAGES);
            let features = u64::from(RGFX_FEATURE_CB_SUBMITTED)
                | u64::from(RGFX_FEATURE_SURFACE_ORDER)
                | u64::from(RGFX_FEATURE_COMMIT_ORDER)
                | u64::from(RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS)
                | u64::from(RGFX_FEATURE_UNSHARED_QUEUES)
                | u64::from(RGFX_FEATURE_SUBMIT_ROOM)
                | u64::from(RGFX_FEATURE_ICB_SLOTS)
                | if unified_memory {
                    0
                } else {
                    u64::from(RGFX_FEATURE_MANAGED_AT_OFFSET)
                }
                | match (decoder.surface_pages_available(), unified_memory) {
                    (false, _) => 0,
                    (true, true) => in_place,
                    // A host that keeps caches keeps fresh surfaces in them alone where the kext
                    // defers their pages.
                    (true, false) => {
                        in_place
                            | u64::from(RGFX_FEATURE_SURFACE_CACHE)
                            | u64::from(RGFX_FEATURE_SURFACE_DEFERRED)
                    }
                };
            let deferred_budget = deferred_budget(features, gpu_device.own_memory_budget());
            decoder.set_deferred_budget(deferred_budget);
            // The processes read the features from the device info, as no register is theirs.
            info.features_lo = u8::try_from(features & 0xff).unwrap_or_default();
            region.write(RGFX_DEVICE_INFO_OFFSET, info.as_bytes());
            // The host's hardware video, whose codecs the video caps after the device info list
            // for the guest's video driver.
            if let Some(video) = platform.graphics_video(&gpu_device) {
                // Pictures land in device caches only where the kext makes them.
                let cached = features & u64::from(RGFX_FEATURE_SURFACE_CACHE) != 0
                    && video.lands_in_device_caches();
                let mut caps = RgfxVideoCaps {
                    version: RGFX_VIDEO_CAPS_VERSION,
                    flags: if cached {
                        RGFX_VIDEO_CAPS_OUTPUT_CACHED
                    } else {
                        0
                    },
                    ..RgfxVideoCaps::default()
                };
                for (slot, codec) in caps.codecs.iter_mut().zip(video.caps()) {
                    *slot = codec;
                    caps.codec_count += 1;
                }
                debug_assert!(caps.codec_count as usize <= RGFX_VIDEO_CAPS_CODECS);
                region.write(RGFX_VIDEO_CAPS_OFFSET, caps.as_bytes());
                decoder.set_video(video);
            }
            let presented = Arc::new(Presented::default());
            let presents = Arc::new(PresentQueue::new(Arc::clone(&presented)));
            // A present that waits for its surface's fill is shown as the fill finishes.
            let waiting = Arc::downgrade(&presents);
            decoder.on_fill_done(Arc::new(move || {
                if let Some(presents) = waiting.upgrade() {
                    presents.filled();
                }
            }));
            let control = Arc::new(GraphicsControl {
                failed: decoder.failed_contexts(),
                decoders: decoder.decoders(),
                device: gpu_device,
                presents,
                presented,
                pool,
                pending: Pending::default(),
                region: Arc::clone(&region),
                icb_worker: Worker::new("reims-vgpu.icb-create"),
                icbs: Arc::default(),
                pages_buffers: Arc::default(),
                reset_gate: Arc::default(),
            });
            let host = Box::new(CallbackHost {
                callbacks: config.callbacks,
                features,
                deferred_budget,
                control: Arc::clone(&control),
            });
            let (window, screen): (Option<WindowLink>, Box<dyn Screen>) = match config.window {
                Some(window_config) => {
                    let (link, screen) =
                        WindowScreen::open(&platform, &region, &control.presented, window_config);
                    (Some(link), Box::new(screen))
                }
                None => (None, Box::new(FixedScreen(FALLBACK_MODE))),
            };
            let device =
                reims_graphics::Device::new(region, host, screen).ok_or(Error::HostResources)?;
            Ok(GraphicsDevice {
                window,
                device,
                #[cfg(any(test, feature = "testkit"))]
                control,
                platform: PhantomData,
            })
        })
    }

    /// The region's first byte: BAR 2's memory, which the host maps into the guest, write-back.
    /// It stays mapped until the device and its window's GPU work have ended.
    pub fn region_base(&self) -> *mut u8 {
        self.device.region_base()
    }

    /// The objects the stage-3 decoder holds: the tests check a create reached it.
    #[cfg(any(test, feature = "testkit"))]
    pub fn decoder_object_count(&self) -> usize {
        self.control
            .decoders
            .sender(None, Sinks::default())
            .object_count()
    }

    /// How many command queues have a per-queue ring and service thread: the tests check a
    /// `QUEUE_CREATE` with valid ring parameters made one.
    #[cfg(any(test, feature = "testkit"))]
    pub fn graphics_queue_count(&self) -> usize {
        self.device.queue_count()
    }

    /// The region's size: BAR 2's, a power of two.
    pub fn region_size(&self) -> u64 {
        self.device.region_size()
    }

    /// A 32-bit read of BAR 0 at `offset`.
    pub fn mmio_read(&self, offset: u64) -> u32 {
        entry::run(|_| self.device.mmio_read(offset))
    }

    /// A 32-bit write of BAR 0 at `offset`. A reset (`STATUS` 0) waits for the command the
    /// device is taking, which waits for nothing of the host's.
    pub fn mmio_write(&self, offset: u64, value: u32) {
        entry::run(|_| self.device.mmio_write(offset, value));
    }

    /// [`Self::mmio_write`], but a `DOORBELL` write of a command queue's ring may answer the
    /// queue's next command buffer on the calling thread, posting its `CB_SUBMITTED` there and
    /// calling `raise_interrupt` inside this call (`reims-graphics`'s `Device::mmio_write_taking`),
    /// allocating nothing; the queue's thread then takes it as it takes any. It waits for no lock
    /// but the queues' table, as `mmio_write`'s doorbell does.
    pub fn mmio_write_taking(&self, offset: u64, value: u32) {
        entry::run(|_| self.device.mmio_write_taking(offset, value));
    }

    /// Takes a copy of `fd`, the host's `ioeventfd` for `DOORBELL` with ring 0. False where `fd`
    /// is not open or the device has one already.
    pub fn set_doorbell(&self, fd: RawFd) -> bool {
        entry::run(|_| self.device.set_doorbell(fd))
    }

    /// The host's reset of the device (the guest's bus reset).
    pub fn reset(&self) {
        entry::run(|_| self.device.reset());
    }
}

impl<P: ApiPlatform> Drop for GraphicsDevice<P> {
    /// The window closes, then the device ends; neither waits (`reims-graphics`'s `device`).
    fn drop(&mut self) {
        entry::run(|_| {
            drop(self.window.take());
        });
    }
}

/// The stage-3 decoder and the device handle it drives, shared by the device's [`Host`], which
/// feeds it ring 0's object lifecycle on each context's lane (`reims-graphics`' device) and the
/// command queues' own rings' messages on their service threads. Each message is carried out on a
/// decoder of its own over the shared state ([`Decoders::sender`]), whose locks are each held for a
/// short section (`decode`'s module doc): no thread holds one across a command buffer's walk, so a
/// walk holds up neither ring 0 nor another queue. A command buffer's completion, which runs on a
/// GPU thread of its own, never touches the decoder: it only posts through the queue's poster.
struct GraphicsControl<P: ApiPlatform> {
    /// The decoder's state, which each message's decoder is made over, and which a doorbell's
    /// thread's answer reads with no decoder made ([`Self::answer_command_buffer`]).
    decoders: Decoders<P>,
    /// The GPU device it drives.
    device: P::Device,
    /// The surface the display pipe shows, shared with the window's sink.
    presented: Arc<Presented<P>>,
    /// The presents waiting for the command buffers that write their surfaces.
    presents: Arc<PresentQueue<P>>,
    /// The threads ring 0's pipeline and library creates compile on (`compile`, `make`).
    pool: CompilePool,
    /// The pipelines being compiled, which a command buffer naming one waits for.
    pending: Pending,
    /// The device's region, where an `ICB_SLOTS` names the slots whose pipelines it waits for.
    region: Arc<Region>,
    /// The thread ring 0's `ICB_CREATE`s are made on, in the order their lanes hand them over, off
    /// the lanes: the platform's make of a large buffer can take seconds (measured: about 24 s
    /// near 11.7 million commands on the Vulkan runtime on an RTX 5080), which would hold the
    /// process's later control messages ([`Self::start_icb`]).
    icb_worker: Worker,
    /// The indirect command buffers whose create was handed to [`Self::icb_worker`] and is not
    /// yet answered, by object id.
    icbs: Arc<Unsettled>,
    /// The no-copy buffers whose `BUFFER_OVER_PAGES` was handed to the surface worker and is not
    /// yet made ([`Self::start_buffer_over_pages`]): by their context ([`pages_context`]), and by
    /// their pages block's offset.
    pages_buffers: Arc<Unsettled>,
    /// The decoder's failed contexts, which the device asks of at a `QUEUE_CREATE`
    /// (`Host::context_refused`).
    failed: Arc<FailedContexts>,
    /// What the device's reset waits for ([`Self::reset`]): held shared by each message carried
    /// out off ring 0's threads, a queue's command buffer, create or destroy, from before it
    /// asks whether its poster is stale to its end, by each compile's and library make's store,
    /// and by each `SURFACE_PAGES` the surface worker carries out ([`Self::start_surface_pages`]);
    /// held alone by the reset. So a reset comes after everything of the guest's that
    /// began before it, as under one lock over the decoder, and whatever begins after it finds
    /// its poster stale (or its compile cancelled), and none of it outlives the reset. Ring 0's
    /// own messages are carried out in order with the reset: the device tells the host of it, on
    /// its control thread, once every message a lane began before it has returned.
    reset_gate: Arc<RwLock<()>>,
}

/// The display's hold of the region block a surface's texture lies over, as a compositor holds
/// the IOSurface it shows: made as the present is taken ([`Presented::hold`]), and held by the
/// present until it is shown, by the display while it shows the texture, and by each window draw
/// of it until the draw has finished on the GPU, so the block's memory stays while any of them
/// may read it. A `BLOCK_RELEASE` of a block that meets the span is answered only once every hold
/// of it has gone ([`Presented::after_shown`]).
pub(crate) struct DisplayHold {
    /// The block the texture lies in, its offset and size; `None` for one in none.
    span: Option<(u64, u64)>,
    /// Given when the hold goes.
    answers: Mutex<Vec<DrainAnswer>>,
}

impl Drop for DisplayHold {
    fn drop(&mut self) {
        for answer in self.answers.get_mut().drain(..) {
            answer();
        }
    }
}

/// A texture over a surface a present shows, with the display's hold of its block.
pub(crate) struct ShownTexture<P: Platform> {
    /// The texture.
    pub(crate) texture: P::Texture,
    /// The hold of the block it lies in.
    hold: Arc<DisplayHold>,
}

/// The IOSurface the display pipe shows in place of the scanout (`PRESENT_SURFACE`): the texture
/// the decoder made over it, which the window's sink draws into the next drawable (`sink`), and
/// the presents' answers, given at the first refresh of the device's display after that draw has
/// finished ([`PresentAnswers`]). The window draws a present when it is made, and again where the
/// window needs a frame (a new layout, the cursor): each present is the guest's own, and none is
/// batched or repeated by the GPU's work. The texture's block is held until the next present, the
/// device's reset or the device's end, and each draw of it until it has finished
/// ([`DisplayHold`]).
pub(crate) struct Presented<P: Platform> {
    /// The texture shown; `None` shows the scanout.
    pub(crate) texture: Mutex<Option<ShownTexture<P>>>,
    /// The holds made and not yet gone, for a release to find those of its block.
    holds: Mutex<Vec<Weak<DisplayHold>>>,
    /// The surface's IOSurface ID, 0 for none.
    surface_id: AtomicU32,
    /// The window, once it has opened.
    window: OnceLock<Arc<Window>>,
    /// A surface of a format the sink cannot draw was logged.
    refused: AtomicBool,
    /// The presents' answers, owed until the device's display shows them.
    answers: PresentAnswers<PresentAnswer>,
    /// The last (surface, drawable) pixel formats logged.
    formats: Mutex<Option<(u64, u64)>>,
}

impl<P: Platform> Default for Presented<P> {
    fn default() -> Self {
        Presented {
            texture: Mutex::new(None),
            holds: Mutex::default(),
            surface_id: AtomicU32::new(0),
            window: OnceLock::new(),
            refused: AtomicBool::new(false),
            formats: Mutex::new(None),
            answers: PresentAnswers::default(),
        }
    }
}

impl<P: Platform> Presented<P> {
    /// `texture`, over a surface whose present was taken, held for the display from now: the
    /// block at `span` it lies in is not given back until the display is done with it
    /// ([`DisplayHold`]).
    fn hold(&self, texture: P::Texture, span: Option<(u64, u64)>) -> ShownTexture<P> {
        let hold = Arc::new(DisplayHold {
            span,
            answers: Mutex::default(),
        });
        if span.is_some() {
            let mut holds = self.holds.lock();
            holds.retain(|h| h.strong_count() > 0);
            holds.push(Arc::downgrade(&hold));
        }
        ShownTexture { texture, hold }
    }

    /// `answer`, the answer to the release of the `size` bytes at region offset `offset`, given
    /// once every hold of a block that meets them has gone too ([`DisplayHold`]): `answer` itself
    /// where the display holds none.
    fn after_shown(&self, offset: u64, size: u64, answer: DrainAnswer) -> DrainAnswer {
        let end = offset.saturating_add(size);
        // Taken out of the list first: a hold whose other holders go meanwhile gives its answers
        // where it is let go of here, with no lock held.
        let live: Vec<Arc<DisplayHold>> =
            self.holds.lock().iter().filter_map(Weak::upgrade).collect();
        let held: Vec<&Arc<DisplayHold>> = (live.iter())
            .filter(|h| {
                h.span
                    .is_some_and(|(at, len)| at < end && offset < at.saturating_add(len))
            })
            .collect();
        if held.is_empty() {
            return answer;
        }
        let join = Arc::new(DisplayHold {
            span: None,
            answers: Mutex::new(vec![answer]),
        });
        for hold in held {
            let join = Arc::clone(&join);
            hold.answers.lock().push(Box::new(move || drop(join)));
        }
        drop(live);
        Box::new(move || drop(join))
    }

    /// Shows `texture` (the surface `id`'s), or the scanout for `None`, and asks the window for a
    /// frame; `answer` is given at the vblank the display first shows it ([`PresentAnswers`]), or
    /// at once without a window. The hold of the texture shown before is returned, for the caller
    /// to let go of with no lock held: a release waiting for it may be answered there.
    #[must_use]
    fn show(
        &self,
        id: u32,
        texture: Option<ShownTexture<P>>,
        answer: PresentAnswer,
    ) -> Option<Arc<DisplayHold>> {
        let previous = self.surface_id.swap(id, Ordering::AcqRel);
        if previous != id {
            log::info!(
                target: "reims_vgpu::graphics",
                "the display shows IOSurface {id} (was {previous}; 0 is the scanout)"
            );
        }
        // The texture shown before may be the last hold of a closed context's: let go of in a
        // refusal scope, so a release the platform dies at is refused, logged, and the present
        // goes on.
        let shown = std::mem::replace(&mut *self.texture.lock(), texture);
        let (shown, hold) = shown.map(|s| (s.texture, s.hold)).unzip();
        if let Err(refusal) = reims_vgpu_core::raise::refusals::refusing(move || drop(shown)) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the texture of IOSurface {previous} was let go of, and its release was refused: \
                 {refusal}"
            );
        }
        match self.window.get() {
            Some(window) => {
                self.answers.shown(answer);
                window.new_frame();
            }
            None => answer.answer(RGFX_STATUS_OK),
        }
        hold
    }

    /// The device's reset: the display shows the scanout (none, after the reset) in place of any
    /// surface, whose texture is let go of in a refusal scope; the answers still owed are given
    /// as they come, and dropped, as each is of a present taken before the reset.
    fn reset(&self) {
        let previous = self.surface_id.swap(0, Ordering::AcqRel);
        let shown = self.texture.lock().take();
        let (shown, hold) = shown.map(|s| (s.texture, s.hold)).unzip();
        if let Err(refusal) = reims_vgpu_core::raise::refusals::refusing(move || drop(shown)) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the texture of IOSurface {previous} was let go of at the device's reset, and its \
                 release was refused: {refusal}"
            );
        }
        drop(hold);
        if let Some(window) = self.window.get() {
            window.new_frame();
        }
    }

    /// The window draws the latest present: the presents shown so far, answered once the draw has
    /// finished, or once it is dropped undone ([`DrawnPresents`]).
    pub(crate) fn drawn(self: &Arc<Self>) -> DrawnPresents<P> {
        DrawnPresents {
            presented: Arc::clone(self),
            drawing: self.answers.drawn(),
            hold: None,
        }
    }

    /// The texture shown, and the hold of its block, for a window draw of it; `None` for the
    /// scanout.
    pub(crate) fn shown(&self) -> Option<(P::Texture, Arc<DisplayHold>)> {
        (self.texture.lock().as_ref()).map(|s| (Handle::clone_ref(&s.texture), Arc::clone(&s.hold)))
    }

    /// A refresh of the device's display: the presents it now shows are answered.
    fn refreshed(&self) {
        self.answers.refreshed();
    }

    /// The sink draws a surface of pixel format `surface` into a drawable of `drawable`: logged
    /// when the pair changes.
    pub(crate) fn seen_format(&self, surface: u64, drawable: u64) {
        let mut formats = self.formats.lock();
        if *formats != Some((surface, drawable)) {
            *formats = Some((surface, drawable));
            log::info!(
                target: "reims_vgpu::graphics",
                "the presented IOSurface's texture has pixel format {surface}, the window's \
                 drawable {drawable}"
            );
        }
    }

    /// The sink cannot draw a surface of pixel format `surface` into a drawable of `drawable`:
    /// logged once.
    pub(crate) fn refuse_format(&self, surface: u64, drawable: u64) {
        if !self.refused.swap(true, Ordering::AcqRel) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the presented IOSurface's texture (pixel format {surface}) cannot be drawn into \
                 the window's drawable (pixel format {drawable}): the scanout shows instead"
            );
        }
    }
}

/// The presents a window's draw shows: answered at the refresh after the draw has finished on the
/// GPU ([`DrawnPresents::after`]), or after the draw is dropped before its commit, once every
/// present shown before them is ([`PresentAnswers::refreshed`]); and the hold of the block of the
/// surface it draws, let go of then too.
pub(crate) struct DrawnPresents<P: Platform> {
    /// Where they are answered.
    presented: Arc<Presented<P>>,
    /// The presents the draw took.
    drawing: Drawing,
    /// The hold of the block of the surface the draw reads, where it reads one.
    hold: Option<Arc<DisplayHold>>,
}

impl<P: Platform> DrawnPresents<P> {
    /// The draw reads the surface whose block `hold` holds: it holds it until it has finished.
    pub(crate) fn reading(&mut self, hold: Arc<DisplayHold>) {
        self.hold = Some(hold);
    }

    /// The presents are answered, and the surface's block let go of, after `command_buffer`, the
    /// draw, has finished: its completed handler holds them. Called before its commit; where the
    /// handler is refused, or the buffer is never committed, they are answered as if it had
    /// finished.
    pub(crate) fn after(self, command_buffer: &P::CommandBuffer) {
        if self.drawing.is_empty() && self.hold.is_none() {
            return;
        }
        no_catch(command_buffer.add_completed_handler(OnFault::Die, Box::new(move |_| drop(self))));
    }
}

impl<P: Platform> Drop for DrawnPresents<P> {
    fn drop(&mut self) {
        self.presented
            .answers
            .done(std::mem::take(&mut self.drawing));
    }
}

impl<P: ApiPlatform> GraphicsControl<P> {
    /// Takes every object of context `context` (its process closed the device or died) out of
    /// the decoder's tables, with the surfaces' entries whose objects went with them, and lets go
    /// of them on the decoder's close worker, so the context's lane is done with the close, and
    /// the jobs of other lanes that wait for it go on, while the platform lets them go. `answer`
    /// is given once they are let go of and every command buffer of the context the decoder
    /// committed has finished: the kext then gives the context's blocks, texture resource ids and
    /// GPU addresses out again.
    fn close_context(&self, context: u32, answer: CompleteAnswer) {
        // After its no-copy buffers being made, so none lands in the table its close empties.
        self.pages_buffers.wait(pages_context(context));
        entry::run(|_| {
            let decoder = self.decoder();
            // Its compiles first, so none stores a pipeline in the table its close empties.
            self.pending.cancel_context(context);
            let (dropped, closed) = decoder.close_tables(context);
            self.pool.context_closed(context);
            self.presents.context_gone(context);
            let drain = decoder.drain_context(
                context,
                true,
                Box::new(move || answer.answer(RGFX_STATUS_OK)),
            );
            decoder.let_go_later(closed, drain);
            log::info!(
                target: "reims_vgpu::graphics",
                "context {context} closed: {dropped} objects dropped"
            );
        });
    }

    /// The device's reset (`Host::device_reset`): every context's objects dropped as at its close
    /// ([`Self::close_context`]), with no answer owed, and everything else the decoder keeps for
    /// the guest's processes ([`GraphicsDecoder::reset`]); the pipelines compiling are not stored,
    /// the presents waiting are dropped, and the display shows the scanout in place of any
    /// surface. The device is then as one just made, to the guest: the ids it picks, the
    /// texture resource ids and samplers' ids it is handed, and what a library lookup finds.
    fn reset(&self) {
        entry::run(|_| {
            let (contexts, dropped) = {
                // After everything of the guest's that began before the reset has ended, its
                // waits for other queues' buffers cut short.
                self.decoders.resetting();
                let _alone = self.reset_gate.write();
                self.pending.reset();
                let decoder = self.decoder();
                // After the closes before it let go of their objects, as the guest's next creates
                // may take the texture resource ids and GPU addresses they held.
                decoder.closes_let_go();
                let (contexts, dropped) = decoder.reset();
                for &context in &contexts {
                    self.pool.context_closed(context);
                }
                self.presents.reset();
                self.decoders.reset_done();
                (contexts, dropped)
            };
            self.presented.reset();
            log::info!(
                target: "reims_vgpu::graphics",
                "the device was reset: {dropped} objects of {} contexts dropped",
                contexts.len()
            );
        });
    }

    /// `BLOCK_RELEASE` (`Host::block_released`): refused at once where an object the decoder
    /// holds lies in the block, else answered once every command buffer of the block's context
    /// the decoder committed has finished, every one an earlier holder's context committed
    /// before its `BLOCK_UNSHARE` ([`Self::unshare_block`]), and once the display no longer shows
    /// or draws a surface's texture over the block ([`DisplayHold`]).
    fn release_block(&self, release: &RgfxBlockRelease, answer: CompleteAnswer) {
        // After the no-copy buffers being made over it, which keep it.
        self.pages_buffers.wait(release.offset);
        let decoder = self.decoder();
        if decoder.span_in_use(release.offset, release.size) {
            drop(decoder);
            log::warn!(
                target: "reims_vgpu::graphics",
                "context {} freed its block of {:#x} bytes at {:#x} while an object lies in it: \
                 the block is kept",
                release.context, release.size, release.offset
            );
            answer.answer(RGFX_STATUS_REFUSED);
            return;
        }
        let answer = self.presented.after_shown(
            release.offset,
            release.size,
            Box::new(move || answer.answer(RGFX_STATUS_OK)),
        );
        let answer = decoder.after_unshares(release.offset, answer);
        let drain = decoder.drain_context(release.context, false, answer);
        // No buffer lies over the block's surface view any more: it goes with the block, and its
        // device cache once the surface's pages hold what was written of it, which the answer
        // waits for. The copy is made once the buffers still walked that may write the cache
        // have finished, on the last one's thread: the context's lane goes on.
        let held = Arc::clone(&drain);
        let unread = release.flags & RGFX_BLOCK_RELEASE_NO_CLIENT != 0;
        let offset = release.offset;
        entry::run(|_| {
            decoder.block_released(
                &self.device,
                offset,
                unread,
                Box::new(move || drop(held)),
                Box::new(move |raise| {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "the block at {offset:#x}: the platform raised at its device cache's \
                         write back ({raise:?})"
                    );
                }),
            );
        });
        drop(decoder);
        drop(drain);
    }

    /// `BLOCK_UNSHARE` (`Host::block_unshared`): the block's release, by its last holder, waits
    /// for the command buffers the decoder committed for `unshare.context` before this, which may
    /// still use the block ([`GraphicsDecoder::block_unshared`]).
    fn unshare_block(&self, unshare: &RgfxBlockGrant) {
        let drain = self
            .decoder()
            .block_unshared(unshare.context, unshare.offset);
        drop(drain);
    }

    /// `SURFACE_PAGES` (`Host::surface_pages`): begun here, in ring order, and carried out on the
    /// surface worker (`decode::surface_pages`), so ring 0 goes on meanwhile, under the reset
    /// gate, held shared recursively, as a queue that holds it may wait for the block. Answered
    /// there once the block's view and device cache are made, before the block settles, so a
    /// message that names the block and waited for it is answered after it.
    fn start_surface_pages(
        self: &Arc<Self>,
        pages: RgfxSurfacePages,
        runs: Vec<RgfxPageRun>,
        flags: u32,
        answer: CompleteAnswer,
    ) {
        // After the no-copy buffers being made over the block it describes again, which see it as
        // before, and whose make on the surface worker would else wait for it, behind them there.
        self.pages_buffers.wait(pages.block_offset);
        let me = Arc::clone(self);
        let job = move |settle| {
            let status = {
                let _carried = me.reset_gate.read_recursive();
                (me.decoder()).carry_out_surface_pages(&me.device, &pages, &runs, flags, &settle)
            };
            // Nothing is owed where the device was reset since.
            if let Some(status) = status {
                answer.answer(status);
            }
            drop(settle);
        };
        entry::run(|_| {
            self.decoder()
                .hand_off_surface_pages(pages.block_offset, job);
        });
    }

    /// `SURFACE_DOWNLOAD` of the block at `block`: answered once the surface's pages hold its
    /// device cache ([`GraphicsDecoder::surface_download`]), which is copied once the buffers
    /// still walked that may write it have finished, on the last one's thread: the block's lane
    /// goes on.
    fn download_surface(&self, block: u64, answer: CompleteAnswer) {
        let answer = Arc::new(Mutex::new(Some(answer)));
        let (give, refuse) = (Arc::clone(&answer), answer);
        entry::run(|_| {
            self.decoder().surface_download(
                &self.device,
                block,
                Box::new(move || {
                    if let Some(answer) = give.lock().take() {
                        answer.answer(RGFX_STATUS_OK);
                    }
                }),
                Box::new(move |raise| {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "the block at {block:#x}: the platform raised at its device cache's \
                         download ({raise:?})"
                    );
                    if let Some(answer) = refuse.lock().take() {
                        answer.answer(RGFX_STATUS_REFUSED);
                    }
                }),
            );
        });
    }

    /// Carries out one stage-3 message under a library entry: from ring 0 an object-lifecycle create
    /// or destroy, from a command queue's own submission ring a `COMMAND_BUFFER` (or the queue's
    /// own create and destroy). The decoder's completion and reflection sinks are pointed at that
    /// ring's completion ring (through `poster`) for the decode, so a buffer's completion handler,
    /// running on a GPU thread, posts its `CB_COMPLETE` on its queue's ring, and a pipeline
    /// create's reflection lands on ring 0's, each raising its ring's MSI-X vector. A command
    /// buffer the host could not carry out owes the guest a completion that no handler will post:
    /// a stand-in on its queue posts an internal-error one in the queue's order, with the values
    /// the buffer signalled its shared events to ahead of it
    /// ([`GraphicsDecoder::stand_in_for_failed`]), else the host posts it onto the queue's ring
    /// here, at once.
    ///
    /// A pipeline or library create is made on the compile pool instead
    /// ([`Self::start_pipeline`], [`Self::start_library`]), and a command buffer that names a
    /// pipeline still compiling waits for it first (`compile`), and is then begun here and walked
    /// and committed on its queue's lane ([`Self::begin_command_buffer`]). An indirect command
    /// buffer's create is made on the ICB worker ([`Self::start_icb`]), and the destroy of one
    /// still being made waits for it. A library lookup is answered here
    /// ([`Self::lookup_library`]).
    ///
    /// `bytes` is the message, borrowed, or the device's own copy of a command buffer's
    /// ([`Host::command_buffer`]), which goes to the buffer's lane as it is.
    fn handle_stage3(self: &Arc<Self>, poster: &CompletionPoster, bytes: Cow<'_, [u8]>) {
        let message: &[u8] = &bytes;
        let r#type = RgfxMsgHeader::read_from_prefix(message).map_or(0, |(h, _)| h.r#type);
        if r#type == RGFX_CMD_BUFFER_OVER_PAGES && self.start_buffer_over_pages(poster, message) {
            return;
        }
        // A message of a context whose no-copy buffer is being made on the surface worker, which
        // it may name, is carried out after it.
        self.pages_buffers
            .wait(pages_context(sender_context(poster, message)));
        match r#type {
            RGFX_CMD_RENDER_PIPELINE_CREATE | RGFX_CMD_COMPUTE_PIPELINE_CREATE => {
                self.start_pipeline(poster, message);
                return;
            }
            RGFX_CMD_LIBRARY_CREATE | RGFX_CMD_LIBRARY_FROM_REGION => {
                self.start_library(poster, message);
                return;
            }
            RGFX_CMD_LIBRARY_LOOKUP => {
                self.lookup_library(poster, message);
                return;
            }
            RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN => {
                self.heap_texture_size(poster, message);
                return;
            }
            RGFX_CMD_ARGUMENT_LAYOUT => {
                self.argument_layout(poster, message);
                return;
            }
            RGFX_CMD_SAMPLER_CREATE => {
                self.create_sampler(poster, message);
                return;
            }
            RGFX_CMD_ICB_CREATE => {
                self.start_icb(poster, message);
                return;
            }
            RGFX_CMD_COMMAND_BUFFER => {
                self.pending.wait_for_command_buffer(message);
                if !foreign_queue(poster, message) {
                    self.begin_command_buffer(poster, bytes.into_owned());
                    return;
                }
            }
            RGFX_CMD_ICB_WRITE => self.pending.wait_for_icb_write(message),
            RGFX_CMD_ICB_SLOTS => self.pending.wait_for_icb_slots(&self.region, message),
            // The destroy of an indirect command buffer still being made (the kext destroys
            // behind a create it stopped waiting for, after a minute, or one made for a context
            // that had closed) waits for its create, so it and everything after it keep their
            // places in ring order.
            RGFX_CMD_OBJECT_DESTROY => self.icbs.wait(message_identity(message).2),
            _ => {}
        }
        entry::run(|_| {
            let sinks = self.sinks(poster);
            let outcome = {
                let _carried = self.reset_gate.read();
                // Of a queue the device's reset ended, which the decoder's reset dropped, or is to
                // drop: nothing of it is carried out, and nothing is owed.
                if poster.stale() {
                    return;
                }
                let mut decoder =
                    (self.decoders).sender(Some(sender_context(poster, message)), sinks);
                if r#type == RGFX_CMD_OBJECT_DESTROY {
                    let (_, _, obj_id) = message_identity(message);
                    self.pending.cancel(obj_id);
                    // A queue's destroy: nothing more of it finishes for a present to wait for.
                    self.presents.queue_gone(obj_id);
                }
                // A queue's create, handed with its own poster: what its ring publishes bounds
                // what a present waits for of it.
                if r#type == RGFX_CMD_QUEUE_CREATE
                    && let Some(queue) = poster.queue_obj_id()
                    && queue == message_identity(message).2
                {
                    self.presents.queue_made(queue, poster.clone());
                    // ... and what a buffer of another queue waits for of it (`surface_order`).
                    let ring = poster.clone();
                    decoder
                        .surface_order_queue_made(queue, Arc::new(move || ring.published_seqno()));
                }
                let foreign = foreign_queue(poster, message);
                // An ICB_WRITE's records and an ICB_SLOTS's slots run on the control ring, ahead
                // of the process's later command buffers: how long they take is logged
                // (diagnostics).
                let icb_start = (r#type == RGFX_CMD_ICB_WRITE || r#type == RGFX_CMD_ICB_SLOTS)
                    .then(|| self.pool.now());
                let outcome = if foreign {
                    Ok(Err(DecodeError::BadReference))
                } else {
                    refusing(|| decoder.decode(&self.device, message))
                };
                if let Some(start) = icb_start {
                    log::debug!(
                        target: "reims_vgpu::graphics::icb",
                        "{} of {} bytes applied in {} us",
                        if r#type == RGFX_CMD_ICB_SLOTS { "ICB_SLOTS" } else { "ICB_WRITE" },
                        message.len(),
                        self.pool.now().saturating_duration_since(start).as_micros()
                    );
                }
                // A command buffer of the process's own queue that was not carried out is stood
                // in for on that queue, which posts its failure in the queue's order, its shared
                // event values ahead of it; else its failure is posted below, at once.
                let stood_in = !matches!(outcome, Ok(Ok(())))
                    && r#type == RGFX_CMD_COMMAND_BUFFER
                    && !foreign
                    && matches!(
                        refusing(|| decoder.stand_in_for_failed(&self.device, message)),
                        Ok(true)
                    );
                (outcome, stood_in)
            };
            let (outcome, stood_in) = outcome;
            self.report(poster, message, outcome, stood_in);
        });
    }

    /// A decoder over the shared state for what is no context's message, posting nothing.
    fn decoder(&self) -> GraphicsDecoder<P> {
        self.decoders.sender(None, Sinks::default())
    }

    /// The sinks of the ring `poster` posts onto, for a message from it ([`Self::handle_stage3`]):
    /// a buffer's completion, with the present waiting for it, its shared events' values, its
    /// `CB_SCHEDULED` and `CB_SUBMITTED`, and a pipeline's reflection.
    fn sinks(&self, poster: &CompletionPoster) -> Sinks {
        let queue_poster = poster.clone();
        let presents = Arc::clone(&self.presents);
        let queue = poster.queue_obj_id();
        let completion: CompletionSink = Arc::new(move |record: RgfxCbComplete| {
            queue_poster.post(record.as_bytes());
            // A present waiting for the buffer is shown once its completion is posted.
            if let Some(queue) = queue {
                presents.finished(queue, record.header.seqno, &queue_poster);
            }
        });
        let signal_poster = poster.clone();
        let signal: SignalSink = Arc::new(move |record: RgfxEventSignalled| {
            signal_poster.post(record.as_bytes());
        });
        let scheduled_poster = poster.clone();
        let scheduled: ScheduledSink = Arc::new(move |record: RgfxCbScheduled| {
            scheduled_poster.post(record.as_bytes());
        });
        let submitted_poster = poster.clone();
        let submitted: SubmittedSink = Arc::new(move |record: RgfxCbSubmitted| {
            submitted_poster.post(record.as_bytes());
        });
        let reflection_poster = poster.clone();
        let reflection: ReflectionSink = Arc::new(move |record: RgfxPipelineReflection| {
            reflection_poster.post(record.as_bytes());
        });
        Sinks {
            completion: Some(completion),
            signal: Some(signal),
            scheduled: Some(scheduled),
            submitted: Some(submitted),
            reflection: Some(reflection),
        }
    }

    /// Logs a stage-3 `message` the decoder did not carry out (`outcome`), and posts its command
    /// buffer's failure onto its ring through `poster` where no stand-in (`stood_in`) posts it.
    fn report(
        &self,
        poster: &CompletionPoster,
        message: &[u8],
        outcome: RaiseResult<Result<(), DecodeError>>,
        stood_in: bool,
    ) {
        match outcome {
            Ok(Ok(())) => {}
            // Its queue ended at the close, so nothing posted onto it would reach the guest.
            Ok(Err(DecodeError::ContextClosed)) => {
                let (ty, seqno, obj_id) = message_identity(message);
                log::debug!(
                    target: "reims_vgpu::graphics",
                    "a message (type {ty:#x}, seqno {seqno}, object {obj_id:#x}) of a context \
                     closed before it came was dropped"
                );
            }
            Ok(Err(e)) => {
                let (ty, seqno, obj_id) = message_identity(message);
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "a message (type {ty:#x}, seqno {seqno}, object {obj_id:#x}) was not \
                     carried out: {e:?}"
                );
                if !stood_in {
                    self.fail_queue_command_buffer(poster, message);
                }
            }
            Err(raise) => {
                let (ty, seqno, obj_id) = message_identity(message);
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "a message (type {ty:#x}, seqno {seqno}, object {obj_id:#x}) faulted: \
                     {raise}"
                );
                if !stood_in {
                    self.fail_queue_command_buffer(poster, message);
                }
            }
        }
    }

    /// A `COMMAND_BUFFER` `message` of the process's own queue, which `poster` posts onto, on the
    /// queue's service thread ([`Self::handle_stage3`]): begun here, its turn in its queue's lane
    /// taken first and its `CB_SUBMITTED` posted where it gets it at its take
    /// ([`GraphicsDecoder::begin_taken`]), then walked and committed on the lane, in its turn
    /// ([`Self::carry_out`]), so the thread goes on to the queue's next buffer meanwhile, `message`
    /// going there as it is.
    fn begin_command_buffer(self: &Arc<Self>, poster: &CompletionPoster, message: Vec<u8>) {
        entry::run(|_| {
            // Held to the end of the buffer's work, on its lane ([`Self::reset_gate`]).
            let carried = self.reset_gate.read_arc();
            // Of a queue the device's reset ended, which the decoder's reset dropped, or is to
            // drop: nothing of it is carried out, and nothing is owed.
            if poster.stale() {
                return;
            }
            let mut decoder =
                (self.decoders).sender(Some(sender_context(poster, &message)), self.sinks(poster));
            let begun =
                refusing(|| decoder.begin_taken(&self.device, &message, poster.queue_obj_id()));
            self.carry_out(decoder, begun, carried, poster.clone(), message);
        });
    }

    /// The rest of the `COMMAND_BUFFER` `message`'s work once it is begun (`begun`, on `decoder`),
    /// on its queue's lane where it took a turn there ([`GraphicsDecoder::lane`]), else here: it
    /// is walked and committed in its turn ([`GraphicsDecoder::finish_command_buffer`]), stood in
    /// for where it was not carried out, and reported through `poster` as
    /// [`Self::handle_stage3`] reports a message, its failure posted in its turn too. The decoder
    /// lets go of what the message holds, and `carried` of the reset gate, before the report,
    /// which may wait for room on the ring.
    ///
    /// A buffer walked in its turn ([`Begun::ordered`], one that holds a video encoder) whose
    /// turn is next is carried out here, on the queue's service thread, rather than on the lane:
    /// it waits there for no buffer of its queue, and a lane's thread would first have to be
    /// woken or made for it (measured on the RTX 5080 host: 80 to 90 µs a picture). Its queue's
    /// next buffer is begun once it is committed; on the video driver's
    /// queues, a session's own each, that one holds a video encoder too and would have waited for
    /// this one's turn before its walk anyway.
    fn carry_out(
        self: &Arc<Self>,
        decoder: GraphicsDecoder<P>,
        begun: RaiseResult<Result<Begun<P>, DecodeError>>,
        carried: ArcRwLockReadGuard<RawRwLock, ()>,
        poster: CompletionPoster,
        message: Vec<u8>,
    ) {
        let lane = decoder.lane();
        let here = matches!(&begun, Ok(Ok(b)) if b.ordered()) && decoder.turn_is_next();
        let control = Arc::clone(self);
        let job = move || {
            entry::run(|_| {
                let mut decoder = decoder;
                let outcome = match begun {
                    Ok(Ok(begun)) => {
                        refusing(|| decoder.finish_command_buffer(&control.device, &message, begun))
                    }
                    Ok(Err(e)) => Ok(Err(e)),
                    Err(raise) => Err(raise),
                };
                // A buffer of a context closed before it came is owed nothing: its queue ended.
                let stood_in = !matches!(outcome, Ok(Ok(()) | Err(DecodeError::ContextClosed)))
                    && matches!(
                        refusing(|| decoder.stand_in_for_failed(&control.device, &message)),
                        Ok(true)
                    );
                // A failure posted below comes after the commits of the buffers before it.
                if !matches!(outcome, Ok(Ok(()))) && !stood_in {
                    decoder.wait_turn();
                }
                let turn = decoder.take_turn();
                drop(decoder);
                drop(carried);
                control.report(&poster, &message, outcome, stood_in);
                drop(turn);
            });
        };
        match lane {
            Some(lane) if !here => lane.run(Box::new(job)),
            _ => job(),
        }
    }

    /// [`Host::answer_command_buffer`]: answers the `COMMAND_BUFFER` `message` at the head of the
    /// ring of the queue `poster` posts onto `CB_SUBMITTED`, on a doorbell's thread, before the
    /// queue's service thread takes it, where it gets that at its take and nothing need be waited
    /// for or allocated (`decode`'s [`Decoders::answer_at_doorbell`]): it runs on that queue, no
    /// reset is under way, every lock it takes is free, each tried, no pipeline it names is still
    /// compiling, and no no-copy buffer of its context is being made on the surface worker
    /// ([`Self::start_buffer_over_pages`]), which its begin waits for. Its `CB_SUBMITTED` is posted through `poster` waiting for nothing, once
    /// its turn in its queue's lane and its ticket in the commit order are taken, which its begin
    /// on the service thread takes in their place ([`Self::begin_command_buffer`]), so the
    /// guest's later buffers are committed after it as they would be had it been answered there.
    fn answer_command_buffer(&self, poster: &CompletionPoster, message: &[u8]) -> bool {
        if foreign_queue(poster, message) || !self.pending.none_named_by(message) {
            return false;
        }
        let context = sender_context(poster, message);
        if (self.pages_buffers).try_any(std::iter::once(pages_context(context))) != Some(false) {
            return false;
        }
        // No reset under way: the decoder's reset lets go of what the answer keeps on the queue.
        let Some(_gate) = self.reset_gate.try_read() else {
            return false;
        };
        if poster.stale() {
            return false;
        }
        (self.decoders).answer_at_doorbell(context, message, |record| {
            poster.try_post(record.as_bytes())
        })
    }

    /// A pipeline create from ring 0: prepared here, on its context's lane, with the decoder's
    /// table (a malformed create or one naming no library is refused here, unanswered, as the
    /// decoder refuses it), marked pending, and compiled on the compile pool, so ring 0 goes on
    /// meanwhile; its reply goes through `poster` when the compile ends ([`Self::compile`]).
    fn start_pipeline(self: &Arc<Self>, poster: &CompletionPoster, message: &[u8]) {
        let prepared = entry::run(|_| {
            let decoder = self
                .decoders
                .sender(Some(sender_context(poster, message)), Sinks::default());
            decoder
                .prepare_pipeline(message)
                .map(|job| (self.pending.begin(job.obj_id), job))
        });
        let (began, job) = match prepared {
            Ok(prepared) => prepared,
            Err(e) => {
                let (ty, seqno, obj_id) = message_identity(message);
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "a pipeline create (type {ty:#x}, seqno {seqno}, object {obj_id:#x}) was not \
                     carried out: {e:?}"
                );
                // Refused at its op, and answered at once: its process waits for the reply.
                if let Some(refused) = refused_reply(message) {
                    poster.post(refused.as_bytes());
                }
                return;
            }
        };
        let of = CompileOf::new(CompileKind::Pipeline, job.obj_id);
        let me = Arc::clone(self);
        let poster = poster.clone();
        if let Err(compile) = self
            .pool
            .submit(of, Box::new(move || me.compile(job, began, &poster)))
        {
            compile();
        }
    }

    /// Compiles a prepared pipeline create, pending since reset count `began`, under a library
    /// entry, with no lock held; stores the pipeline where no destroy, close or reset cancelled it
    /// meanwhile, lets the command buffers that wait for it go on, and posts its reply (its
    /// reflection, or a refusal) through `poster`.
    fn compile(&self, job: PipelineJob<P>, began: u64, poster: &CompletionPoster) {
        entry::run(|_| {
            let (obj_id, seqno) = (job.obj_id, job.seqno);
            let compiled = compile_refusing::<P>(&self.device, job);
            if let Some(reason) = compiled.refusal() {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "a pipeline create (seqno {seqno}, object {obj_id:#x}) was refused: {reason}"
                );
            }
            let reflection = compiled.reflection;
            // Stored where no destroy, close or reset cancelled it, under the lock of its table,
            // which its destroy removes from after cancelling it; a reset waits for the store.
            {
                let _carried = self.reset_gate.read();
                self.decoder()
                    .finish_pipeline_if(compiled, || self.pending.live(obj_id, began));
                self.pending.finish(obj_id, began);
            }
            poster.post(reflection.as_bytes());
        });
    }

    /// A library lookup from ring 0, answered at once: a hit binds the library's new id to the
    /// library the device keeps under its key, and no bytes cross.
    fn lookup_library(&self, poster: &CompletionPoster, message: &[u8]) {
        let (seqno, status) = entry::run(|_| {
            (self.decoders)
                .sender(Some(sender_context(poster, message)), Sinks::default())
                .lookup_library(message)
        });
        let (obj_id, size) = RgfxLibraryLookup::read_from_prefix(message)
            .map_or((0, 0), |(m, _)| (m.obj_id, m.slice_size));
        log::info!(
            target: "reims_vgpu::graphics",
            "library lookup (object {obj_id:#x}, {size} bytes): {}",
            if status == RGFX_STATUS_OK { "hit" } else { "miss" }
        );
        if let Some(seqno) = seqno {
            poster.post(complete(seqno, status).as_bytes());
        }
    }

    /// A heap texture size query from ring 0, answered at once through `poster`: the device's
    /// size and alignment, or a refusal, which the guest answers by a bound of its own.
    fn heap_texture_size(&self, poster: &CompletionPoster, message: &[u8]) {
        let answer = entry::run(|_| {
            (self.decoders)
                .sender(Some(sender_context(poster, message)), Sinks::default())
                .heap_texture_size_and_align(&self.device, message)
        });
        if let Some(answer) = answer {
            poster.post(answer.as_bytes());
        }
    }

    /// A sampler create from ring 0, answered at once through `poster` with the sampler's
    /// `gpuResourceID` (`SAMPLER_STATE`), which the guest's sampler answers and its argument
    /// buffers hold: the id the platform numbers it by, in the one space the host's own samplers
    /// and its pipelines' take theirs from, so none is held twice. The sampler is stored under its
    /// id before the answer. A create the platform raises at is answered refused, as one it
    /// refuses.
    fn create_sampler(&self, poster: &CompletionPoster, message: &[u8]) {
        let (ty, seqno, obj_id) = message_identity(message);
        let decoder =
            (self.decoders).sender(Some(sender_context(poster, message)), Sinks::default());
        let answer = entry::run(|_| refusing(|| decoder.sampler_state(&self.device, message)));
        let answer = answer.unwrap_or_else(|raise| {
            log::warn!(
                target: "reims_vgpu::graphics",
                "a sampler create (type {ty:#x}, seqno {seqno}, object {obj_id:#x}) faulted: \
                 {raise}"
            );
            RgfxMsgHeader::read_from_prefix(message)
                .ok()
                .map(|(header, _)| RgfxSamplerState {
                    header: RgfxMsgHeader {
                        r#type: RGFX_EVT_SAMPLER_STATE,
                        size: size_of::<RgfxSamplerState>() as u32,
                        seqno: header.seqno,
                    },
                    status: RGFX_STATUS_REFUSED,
                    ..RgfxSamplerState::default()
                })
        });
        let Some(answer) = answer else {
            return;
        };
        if answer.status != RGFX_STATUS_OK {
            log::warn!(
                target: "reims_vgpu::graphics",
                "a sampler create (type {ty:#x}, seqno {seqno}, object {obj_id:#x}) was answered \
                 status {}",
                answer.status
            );
        }
        poster.post(answer.as_bytes());
    }

    /// A `BUFFER_OVER_PAGES` from ring 0 that makes its pages block's view (the block's first),
    /// follows its block's `SURFACE_PAGES` still on the surface worker, or follows another of its
    /// context's still being made: begun here, in ring order, and carried out on the surface
    /// worker behind its block's jobs handed to it before (another block's may be made first:
    /// `decode::worker`), so the view's mapping and import hold neither ring 0 nor the views'
    /// lock (`decode::surface_pages`). Until it is made its context and its block are unsettled
    /// ([`Self::pages_buffers`]): every later message of the context waits for it
    /// ([`Self::handle_stage3`]), none of its command buffers is answered at a doorbell
    /// ([`Self::answer_command_buffer`]), and its block's release, a new `SURFACE_PAGES` of its
    /// block and its context's close wait for it, so each sees the buffer as right after its
    /// create. Made under the reset gate, held shared recursively, as the surface
    /// worker's other jobs are; where the device was reset since, nothing is made. Whether it was
    /// handed over; else it is carried out here.
    fn start_buffer_over_pages(
        self: &Arc<Self>,
        poster: &CompletionPoster,
        message: &[u8],
    ) -> bool {
        let Ok((m, _)) = RgfxBufferOverPages::read_from_prefix(message) else {
            return false;
        };
        let context = pages_context(sender_context(poster, message));
        let decoder = self.decoder();
        if !self
            .pages_buffers
            .any([context, m.block_offset].into_iter())
            && !decoder.buffer_over_pages_makes(message)
        {
            return false;
        }
        let begun = self.pool.now();
        let settles = (
            self.pages_buffers.begin(context, 0, begun),
            self.pages_buffers.begin(m.block_offset, 0, begun),
        );
        let (me, poster, message) = (Arc::clone(self), poster.clone(), message.to_vec());
        decoder.on_surface_worker(m.block_offset, Box::new(move || {
            let outcome = {
                let _carried = me.reset_gate.read_recursive();
                if poster.stale() {
                    return;
                }
                let mut decoder = (me.decoders)
                    .sender(Some(sender_context(&poster, &message)), me.sinks(&poster));
                refusing(|| decoder.decode(&me.device, &message))
            };
            me.report(&poster, &message, outcome, false);
            log::debug!(
                target: "reims_vgpu::graphics::surface_pages",
                "BUFFER_OVER_PAGES of the block at {:#x} made on the worker {} us after its begin",
                m.block_offset,
                me.pool.now().saturating_duration_since(begun).as_micros()
            );
            drop(settles);
        }));
        true
    }

    /// An `ICB_CREATE` from ring 0: unsettled here, in ring order, and made on the ICB worker,
    /// behind its context's creates handed to it before ([`Self::create_icb`]), so ring 0 goes on
    /// meanwhile. Its answer goes through `poster` before it settles. Nothing of the guest's
    /// names the buffer before that answer: the bundle waits for it, for the buffer's
    /// `gpuResourceID`, before it hands the buffer out. So only the kext's destroy behind a create
    /// it stopped waiting for can name it while it is made, and that destroy waits for it
    /// ([`Self::handle_stage3`]). Made under the reset gate, held shared (recursively, as for the
    /// surface worker's jobs); where the device was reset since its begin, nothing is made and
    /// nothing answered.
    fn start_icb(self: &Arc<Self>, poster: &CompletionPoster, message: &[u8]) {
        let (_, seqno, obj_id) = message_identity(message);
        // The poster's epoch says whether the device was reset since: the settle's count of
        // resets is not read.
        let settle = self.icbs.begin(obj_id, 0, self.pool.now());
        // Its context opened here, in ring order: a close the worker's make comes after leaves it
        // no state to store into.
        self.decoder().open_for_create(obj_id);
        // Behind its process's creates, beside other processes'.
        let key = u64::from(sender_context(poster, message));
        let (me, poster, message) = (Arc::clone(self), poster.clone(), message.to_vec());
        let job = Box::new(move || {
            {
                let _carried = me.reset_gate.read_recursive();
                if !poster.stale() {
                    me.create_icb(&poster, &message);
                }
            }
            log::debug!(
                target: "reims_vgpu::graphics::icb",
                "ICB_CREATE (seqno {seqno}, object {obj_id:#x}) answered {} us after its begin",
                me.pool
                    .now()
                    .saturating_duration_since(settle.begun)
                    .as_micros()
            );
            drop(settle);
        });
        self.icb_worker.submit(key, job);
    }

    /// An indirect command buffer's create from ring 0, on the ICB worker ([`Self::start_icb`]),
    /// answered through `poster` with its `gpuResourceID` (`ICB_STATE`), which the guest's buffer answers and its argument
    /// buffers hold, so a kernel encoding through one reaches this buffer: the id the platform
    /// numbers it by. The buffer is stored under its id before the answer. A create the platform
    /// raises at is answered refused, as one it refuses.
    fn create_icb(&self, poster: &CompletionPoster, message: &[u8]) {
        let (ty, seqno, obj_id) = message_identity(message);
        let decoder =
            (self.decoders).sender(Some(sender_context(poster, message)), Sinks::default());
        let answer = entry::run(|_| refusing(|| decoder.icb_state(&self.device, message)));
        let answer = answer.unwrap_or_else(|raise| {
            log::warn!(
                target: "reims_vgpu::graphics",
                "an indirect command buffer create (type {ty:#x}, seqno {seqno}, object \
                 {obj_id:#x}) faulted: {raise}"
            );
            RgfxMsgHeader::read_from_prefix(message)
                .ok()
                .map(|(header, _)| RgfxIcbState {
                    header: RgfxMsgHeader {
                        r#type: RGFX_EVT_ICB_STATE,
                        size: size_of::<RgfxIcbState>() as u32,
                        seqno: header.seqno,
                    },
                    status: RGFX_STATUS_REFUSED,
                    ..RgfxIcbState::default()
                })
        });
        if let Some(answer) = answer {
            poster.post(answer.as_bytes());
        }
    }

    /// An argument buffer's layout from ring 0, answered at once: written into the sender's
    /// block, then its `COMPLETE` (`GraphicsDecoder::argument_layout`). The platform's encoder
    /// is made in the refusal scope, so one Metal raises at is refused, as a create is.
    fn argument_layout(&self, poster: &CompletionPoster, message: &[u8]) {
        let (ty, seqno, obj_id) = message_identity(message);
        let answer = entry::run(|_| {
            let decoder =
                (self.decoders).sender(Some(sender_context(poster, message)), Sinks::default());
            refusing(|| decoder.argument_layout(&self.device, message))
        });
        let (seqno, status) = answer.unwrap_or((Some(seqno), RGFX_STATUS_REFUSED));
        if status != RGFX_STATUS_OK {
            log::warn!(
                target: "reims_vgpu::graphics",
                "an argument layout (type {ty:#x}, object {obj_id:#x}) was answered status \
                 {status}"
            );
        }
        if let Some(seqno) = seqno {
            poster.post(complete(seqno, status).as_bytes());
        }
    }

    /// A library create from ring 0 (a `LIBRARY_CREATE`, or a `LIBRARY_FROM_REGION` chunk):
    /// prepared here, on its context's lane, with the decoder's table (a malformed message is
    /// refused here, as the decoder refuses it, and answered where the guest waits), then made on
    /// the compile pool ([`Self::make`]), so ring 0 goes on meanwhile.
    fn start_library(self: &Arc<Self>, poster: &CompletionPoster, message: &[u8]) {
        let start = entry::run(|_| {
            (self.decoders)
                .sender(Some(sender_context(poster, message)), Sinks::default())
                .prepare_library(message)
        });
        let job = match start {
            LibraryStart::Make(job) => job,
            LibraryStart::Chunk => return,
            LibraryStart::Refused { error, answer } => {
                let (ty, seqno, obj_id) = message_identity(message);
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "a library create (type {ty:#x}, seqno {seqno}, object {obj_id:#x}) was not \
                     carried out: {error:?}"
                );
                if let Some(seqno) = answer {
                    poster.post(complete(seqno, RGFX_STATUS_INVALID_ARGUMENT).as_bytes());
                }
                return;
            }
        };
        let of = CompileOf::new(CompileKind::Library, job.obj_id);
        let me = Arc::clone(self);
        let poster = poster.clone();
        if let Err(make) = self
            .pool
            .submit(of, Box::new(move || me.make(job, &poster)))
        {
            make();
        }
    }

    /// Makes a prepared library under a library entry, with no lock held; stores it, hands it to
    /// the pipeline creates waiting on it, and posts the create's `COMPLETE` through `poster`.
    fn make(&self, job: LibraryJob<P>, poster: &CompletionPoster) {
        entry::run(|_| {
            let (obj_id, seqno) = (job.obj_id, job.seqno);
            let made = match refusing(|| make_library::<P>(&self.device, &job)) {
                Ok(made) => made,
                Err(raise) => MadeLibrary::refused(job, raise.to_string()),
            };
            let reason = made.refusal().map(str::to_owned);
            // A reset waits for the store ([`Self::reset_gate`]).
            let status = {
                let _carried = self.reset_gate.read();
                self.decoder().finish_library(made)
            };
            if status != RGFX_STATUS_OK {
                let reason = reason.as_deref().unwrap_or("it was cancelled");
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "a library create (seqno {seqno}, object {obj_id:#x}) was refused: {reason}"
                );
            }
            poster.post(complete(seqno, status).as_bytes());
        });
    }

    /// Shows the IOSurface `iosurface_id` (0: the scanout) once the command buffers committed that
    /// write its copy have finished, after the presents before it ([`PresentQueue`]), answering
    /// once the window shows it; answered invalid at once, with nothing changed, where the decoder
    /// holds no texture over it.
    fn present_surface(&self, iosurface_id: u32, generation: Option<u64>, answer: PresentAnswer) {
        if iosurface_id == 0 {
            self.presents.present(0, None, answer, Vec::new(), None);
            return;
        }
        let decoder = self.decoder();
        // The pages as of the present's generation, where the surface has a device cache: the
        // fill is committed here, and the present waits for it in the present queue.
        let fill = generation.and_then(|generation| {
            let filled =
                entry::run(|_| decoder.fill_presented(&self.device, iosurface_id, generation));
            filled.unwrap_or_else(|raise| {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "IOSurface {iosurface_id}: the platform raised at its device cache's fill for \
                     a present ({raise:?})"
                );
                None
            })
        });
        let found = decoder.surface_present(iosurface_id);
        drop(decoder);
        match found {
            Some(SurfacePresent {
                texture,
                span,
                writers,
                owner,
            }) => {
                // The process that made the texture over the surface the display shows owns the
                // scanout (WindowServer): its compiles go first.
                self.pool.favour(owner);
                // Its block is held from here, before the texture's destroy and the block's
                // release can come: the present shows it later.
                let shown = self.presented.hold(texture, span);
                self.presents
                    .present(iosurface_id, Some(shown), answer, writers, fill);
            }
            None => answer.answer(RGFX_STATUS_INVALID_ARGUMENT),
        }
    }

    /// Posts an internal-error completion onto a queue's own completion ring for a `COMMAND_BUFFER`
    /// its service thread took but the decode failed, so a guest blocked on the buffer unblocks.
    fn fail_queue_command_buffer(&self, poster: &CompletionPoster, message: &[u8]) {
        let Ok((header, _)) = RgfxMsgHeader::read_from_prefix(message) else {
            return;
        };
        if header.r#type != RGFX_CMD_COMMAND_BUFFER {
            return;
        }
        let Ok((cb, _)) = RgfxCommandBuffer::read_from_prefix(message) else {
            return;
        };
        poster.post(cb_failed(cb.cb_obj_id, header.seqno).as_bytes());
        if let Some(queue) = poster.queue_obj_id() {
            self.presents.finished(queue, header.seqno, poster);
        }
    }
}

/// The host's callbacks and its stage-3 decoder as the device's [`Host`].
struct CallbackHost<P: ApiPlatform> {
    /// The host's interrupt and dirty-log callbacks.
    callbacks: GraphicsCallbacks,
    /// The `RGFX_FEATURE_*` the decoder carries out, fixed at the device's make: the guest reads
    /// them on a vCPU, which takes no lock of the decoder's.
    features: u64,
    /// The deferred blocks' budget ([`deferred_budget`]), read once at the device's make, so the
    /// kext's one read holds.
    deferred_budget: u64,
    /// The stage-3 decoder, fed the stage-3 messages of ring 0 and of the queues' rings.
    control: Arc<GraphicsControl<P>>,
}

impl<P: ApiPlatform> Host for CallbackHost<P> {
    fn raise_interrupt(&self, vector: u32) {
        if let Some(f) = &self.callbacks.raise_interrupt {
            f(vector);
        }
    }

    fn scanout_written(&self, offset: u64, length: u64) -> bool {
        self.callbacks
            .scanout_written
            .as_ref()
            .is_none_or(|f| f(offset, length))
    }

    fn present_surface(&self, iosurface_id: u32, generation: Option<u64>, answer: PresentAnswer) {
        self.control
            .present_surface(iosurface_id, generation, answer);
    }

    fn surface_textures(&self, iosurface_id: u32) -> Vec<u64> {
        self.control.decoder().surface_textures(iosurface_id)
    }

    fn refreshed(&self) {
        self.control.presented.refreshed();
    }

    fn context_closed(&self, context: u32, answer: CompleteAnswer) {
        self.control.close_context(context, answer);
    }

    fn context_refused(&self, context: u32) -> bool {
        self.control.failed.has(context)
    }

    fn block_released(&self, release: &RgfxBlockRelease, answer: CompleteAnswer) {
        self.control.release_block(release, answer);
    }

    fn block_unshared(&self, unshare: &RgfxBlockGrant) {
        self.control.unshare_block(unshare);
    }

    fn device_reset(&self) {
        self.control.reset();
    }

    fn features(&self) -> u64 {
        self.features
    }

    fn deferred_budget(&self) -> u64 {
        self.deferred_budget
    }

    fn surface_pages(
        &self,
        pages: &RgfxSurfacePages,
        runs: Vec<RgfxPageRun>,
        flags: u32,
        answer: CompleteAnswer,
    ) {
        self.control
            .start_surface_pages(*pages, runs, flags, answer);
    }

    fn surface_download(&self, block_offset: u64, answer: CompleteAnswer) {
        self.control.download_surface(block_offset, answer);
    }

    fn stage3_message(&self, completion: &CompletionPoster, message: &[u8]) {
        self.control
            .handle_stage3(completion, Cow::Borrowed(message));
    }

    fn command_buffer(&self, completion: &CompletionPoster, message: Vec<u8>) {
        self.control.handle_stage3(completion, Cow::Owned(message));
    }

    fn answer_command_buffer(&self, completion: &CompletionPoster, message: &[u8]) -> bool {
        self.control.answer_command_buffer(completion, message)
    }
}

/// What the window and its sink share: the scanout the guest set, and the size of the drawable
/// the window's render thread last drew into, its size in pixels.
#[derive(Debug, Default)]
pub(crate) struct Shown {
    /// The guest's scanout; `None` shows black.
    pub(crate) scanout: Mutex<Option<Scanout>>,
    /// The window's size in pixels, once drawn into.
    pub(crate) drawable: Mutex<Option<(u64, u64)>>,
}

/// The display's window as the device's [`Screen`] (the module doc).
pub(crate) struct WindowScreen {
    /// The window.
    window: Arc<Window>,
    /// What it shows.
    shown: Arc<Shown>,
    /// The refresh rate of the host screen it opened on, in thousandths of a hertz.
    refresh_mhz: u32,
    /// The guest's cursor, as last told.
    cursor: Mutex<CursorState>,
}

/// The guest's cursor, as the window shows it: over the image while the guest has an image and
/// shows it.
#[derive(Debug)]
struct CursorState {
    /// The guest has an image.
    image: bool,
    /// The guest shows its cursor: until it says otherwise, it does.
    visible: bool,
    /// What the window was last told, once told.
    shown: Option<bool>,
}

impl CursorState {
    /// Tells `window` whether the cursor shows, where that changed.
    fn update(&mut self, window: &Window) {
        let show = self.image && self.visible;
        if self.shown != Some(show) {
            self.shown = Some(show);
            window.cursor_show(show);
        }
    }
}

impl WindowScreen {
    /// The window of a display over `region`, opened with its sink on `platform` where the
    /// platform draws one (no sink: the window never opens), and the screen over it.
    fn open<P: ApiPlatform>(
        platform: &Arc<P>,
        region: &Arc<Region>,
        presented: &Arc<Presented<P>>,
        mut config: WindowConfig,
    ) -> (WindowLink, WindowScreen) {
        config.keeps_its_size = true;
        let first = config.mode.unwrap_or((1920, 1080));
        let window = Window::new(config, 0, reims_vgpu_window::handle::title(0));
        window.offered_mode(first.0, first.1);
        let refresh_mhz = reims_vgpu_window::screens::screen_for_port(0)
            .map_or(FALLBACK_MODE.refresh_mhz, |s| hz_to_mhz(s.refresh_hz));
        let shown = Arc::new(Shown::default());
        if let Some(sink) = scanout_sink(platform, region, &shown, presented) {
            window.attach(sink);
            // Presents wait for the window's draw only where it has a sink to draw them.
            let _ = presented.window.set(Arc::clone(&window));
        }
        let screen = WindowScreen {
            window: Arc::clone(&window),
            shown,
            refresh_mhz,
            cursor: Mutex::new(CursorState {
                image: false,
                visible: true,
                shown: None,
            }),
        };
        (WindowLink::new(window), screen)
    }
}

/// The texture resource ids the kext hands the guest's processes in chunks: 1 to
/// `RGFX_RESOURCE_ID_CHUNKS * RGFX_RESOURCE_ID_CHUNK` (docs/graphics-kext/design.md, "Texture
/// resource ids").
fn guest_texture_ids() -> std::ops::RangeInclusive<u64> {
    1..=u64::from(RGFX_RESOURCE_ID_CHUNKS) * RGFX_RESOURCE_ID_CHUNK
}

/// How many plain textures the reservation's check makes: Metal's texture pool's first range,
/// which it hands out a second time where a reservation follows the device's first queue.
const RESERVATION_CHECKS: usize = 16;

/// Keeps the guest's texture resource ids ([`guest_texture_ids`]) from every texture the process
/// numbers itself on `device`: the window's drawables and cursor, and any other the host makes.
/// The guest places each of its textures at its own id, and Metal hands out a placed id again to
/// plain allocation, so without the reservation two textures share it; a texture placed past
/// the ids the device's table has grown to corrupts Metal's heap. The reservation grows the table
/// to hold them. Measured on an M3 Max, macOS 27; metal2vulkan imitates both.
///
/// Metal takes a texture reservation only before the device's first texture (it ignores a later
/// one, logging so), and one after the device's first command queue keeps the pool's first range
/// beside the ids it leaves, which it hands out twice. So this is the first call on the device,
/// and a few plain textures check it took: one inside the guest's ids is logged as an error.
fn reserve_guest_texture_ids<P: Platform>(platform: &P, device: &P::Device) {
    let ids: Vec<u64> = guest_texture_ids().collect();
    if let Err(raise) = refusing(|| device.reserve_resource_indices(0, &ids)) {
        log::error!(
            "the guest's texture resource ids are not reserved ({raise}): its textures may \
             share ids with the host's"
        );
        return;
    }
    let descriptor = platform.texture_descriptor(&TextureDesc {
        pixel_format: 80,
        width: 1,
        height: 1,
        usage: 1,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    });
    let made: Vec<P::Texture> = (0..RESERVATION_CHECKS)
        .filter_map(|_| refusing(|| device.new_texture(OnFault::Return, &descriptor)).ok()?)
        .collect();
    if let Some(id) = made
        .iter()
        .map(Texture::gpu_resource_id)
        .find(|id| guest_texture_ids().contains(id))
    {
        log::error!(
            "a texture of the host's took resource id {id}, one of the guest's ({} to {}): \
             something made a texture or a command queue on the device before the guest's ids \
             were reserved, and the guest's textures may share ids with the host's",
            guest_texture_ids().start(),
            guest_texture_ids().end()
        );
    }
}

/// `hz` in thousandths of a hertz: 60 Hz where it is not a positive rate.
fn hz_to_mhz(hz: f64) -> u32 {
    if !(hz.is_finite() && hz > 0.0) {
        return FALLBACK_MODE.refresh_mhz;
    }
    #[expect(
        clippy::as_conversions,
        reason = "a rate up to 1000 Hz, rounded to nearest"
    )]
    let mhz = (hz * 1000.0).round() as u32;
    mhz
}

impl Screen for WindowScreen {
    /// The window's size in pixels as last drawn into, else the mode it opened for, at its host
    /// screen's rate.
    fn mode(&self) -> Mode {
        let drawn = *self.shown.drawable.lock();
        let (width, height) = drawn.unwrap_or_else(|| {
            let (w, h) = self.window.offered().unwrap_or((1920, 1080));
            (u64::from(w), u64::from(h))
        });
        let clamp = |v: u64| u32::try_from(v.max(1)).unwrap_or(u32::MAX);
        Mode {
            width: clamp(width),
            height: clamp(height),
            refresh_mhz: self.refresh_mhz,
        }
    }

    /// The window takes the scanout's size as the guest's mode, and draws it.
    fn scanout(&self, scanout: Option<Scanout>) {
        *self.shown.scanout.lock() = scanout;
        if let Some(s) = scanout {
            let mode = |v: u32| u16::try_from(v).unwrap_or(u16::MAX);
            self.window.guest_mode(mode(s.width), mode(s.height));
        }
        self.window.new_frame();
    }

    fn present(&self) {
        self.window.new_frame();
    }

    /// The host's cursor over the image takes the guest's image; with none, nothing shows
    /// there.
    fn cursor_image(&self, image: Option<CursorImage>) {
        let mut cursor = self.cursor.lock();
        cursor.image = image.is_some();
        if let Some(image) = image {
            let hot = |v: u32| u16::try_from(v).unwrap_or(u16::MAX);
            self.window.cursor_glyph(
                image.width,
                image.height,
                (hot(image.hot_spot.0), hot(image.hot_spot.1)),
                &image.rgba,
            );
        }
        cursor.update(&self.window);
    }

    /// The window shows or hides the guest's cursor as the guest does; the position is the
    /// host's pointer's (the module doc).
    fn cursor_moved(&self, _x: i32, _y: i32, visible: bool) {
        let mut cursor = self.cursor.lock();
        cursor.visible = visible;
        cursor.update(&self.window);
    }
}

/// The window's sink over `region` on `platform` ([`sink::ScanoutDraw`]): Metal's on macOS's
/// product platform, the Vulkan platform's swapchain presenter with the `vulkan` feature;
/// `None` on any other platform (the testkit's), or where the platform refuses a command queue.
fn scanout_sink<P: ApiPlatform>(
    platform: &Arc<P>,
    region: &Arc<Region>,
    shown: &Arc<Shown>,
    presented: &Arc<Presented<P>>,
) -> Option<Arc<dyn FrameSink>> {
    let any: Arc<dyn std::any::Any + Send + Sync> = Arc::clone(platform) as _;
    let presented_any: Arc<dyn std::any::Any + Send + Sync> = Arc::clone(presented) as _;
    #[cfg(target_os = "macos")]
    if let (Ok(metal), Ok(presented)) = (
        Arc::clone(&any).downcast::<reims_vgpu_metal::MetalPlatform>(),
        Arc::clone(&presented_any).downcast::<Presented<reims_vgpu_metal::MetalPlatform>>(),
    ) {
        return metal::sink(metal, region, shown, &presented);
    }
    #[cfg(feature = "vulkan")]
    if let (Ok(vulkan), Ok(presented)) = (
        Arc::clone(&any).downcast::<reims_vgpu_vulkan::VulkanPlatform>(),
        Arc::clone(&presented_any).downcast::<Presented<reims_vgpu_vulkan::VulkanPlatform>>(),
    ) {
        return vulkan::sink(vulkan, region, shown, &presented);
    }
    let _ = (any, presented_any, region, shown);
    None
}

#[cfg(test)]
mod tests;

/// The context (the process) `message` came from: for a message off a queue's ring, the queue's;
/// on ring 0, that of the object it creates or destroys, which the kext checked is the sender's.
fn sender_context(poster: &CompletionPoster, message: &[u8]) -> u32 {
    let obj_id = poster
        .queue_obj_id()
        .unwrap_or_else(|| message_identity(message).2);
    (obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32
}

/// The key the no-copy buffers of context `context` being made are unsettled under in
/// `GraphicsControl::pages_buffers`, apart from their blocks' offsets, which lie below 2^63.
fn pages_context(context: u32) -> u64 {
    1 << 63 | u64::from(context)
}

/// Whether `message` is a `COMMAND_BUFFER` off a queue's ring that names another queue to run on:
/// a buffer runs on the queue whose ring carried it.
fn foreign_queue(poster: &CompletionPoster, message: &[u8]) -> bool {
    match (
        poster.queue_obj_id(),
        RgfxCommandBuffer::read_from_prefix(message),
    ) {
        (Some(queue), Ok((cb, _))) => {
            cb.header.r#type == RGFX_CMD_COMMAND_BUFFER && cb.queue_obj_id != queue
        }
        _ => false,
    }
}

/// A message's type and seqno (its header) and the object id that follows the header in every
/// create, destroy and command buffer (0 where the message is shorter), for a refusal's log line.
fn message_identity(message: &[u8]) -> (u32, u64, u64) {
    let word = |at: usize, len: usize| {
        message.get(at..at + len).map_or(0, |b| {
            let mut v = [0u8; 8];
            v[..len].copy_from_slice(b);
            u64::from_le_bytes(v)
        })
    };
    (word(0, 4) as u32, word(8, 8), word(16, 8))
}
