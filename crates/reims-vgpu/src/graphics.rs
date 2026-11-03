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

use std::marker::PhantomData;
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::RwLock;
use reims_graphics::ops::RgfxCommandBuffer;
use reims_graphics::region::Region;
use reims_graphics::wire::{
    RGFX_BLOCK_RELEASE_NO_CLIENT, RGFX_CMD_ARGUMENT_LAYOUT, RGFX_CMD_COMMAND_BUFFER,
    RGFX_CMD_COMPUTE_PIPELINE_CREATE, RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN, RGFX_CMD_ICB_CREATE,
    RGFX_CMD_ICB_WRITE, RGFX_CMD_LIBRARY_CREATE, RGFX_CMD_LIBRARY_FROM_REGION,
    RGFX_CMD_LIBRARY_LOOKUP, RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_QUEUE_CREATE,
    RGFX_CMD_RENDER_PIPELINE_CREATE, RGFX_CMD_SAMPLER_CREATE, RGFX_CONTEXT_OBJ_ID_SHIFT,
    RGFX_DEVICE_INFO_OFFSET, RGFX_EVT_ICB_STATE, RGFX_EVT_SAMPLER_STATE, RGFX_FEATURE_CB_SUBMITTED,
    RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS, RGFX_FEATURE_MANAGED_AT_OFFSET,
    RGFX_FEATURE_SURFACE_CACHE, RGFX_FEATURE_SURFACE_ORDER, RGFX_FEATURE_SURFACE_PAGES,
    RGFX_FEATURE_UNSHARED_QUEUES, RGFX_RESOURCE_ID_CHUNK, RGFX_RESOURCE_ID_CHUNKS,
    RGFX_STATUS_INVALID_ARGUMENT, RGFX_STATUS_OK, RGFX_STATUS_REFUSED, RGFX_VIDEO_CAPS_CODECS,
    RGFX_VIDEO_CAPS_OFFSET, RGFX_VIDEO_CAPS_OUTPUT_CACHED, RGFX_VIDEO_CAPS_VERSION,
    RgfxBlockRelease, RgfxCbComplete, RgfxCbScheduled, RgfxCbSubmitted, RgfxEventSignalled,
    RgfxIcbState, RgfxLibraryLookup, RgfxMsgHeader, RgfxPageRun, RgfxPipelineReflection,
    RgfxSamplerState, RgfxSurfacePages, RgfxVideoCaps,
};
use reims_graphics::{
    CompleteAnswer, CompletionPoster, CursorImage, FixedScreen, Host, Mode, PresentAnswer, Scanout,
    Screen, Taken,
};
use reims_vgpu_core::memory::map::MemoryMap as CoreMemoryMap;
use reims_vgpu_core::num::ToU64;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::command::CommandBuffer as _;
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::os::clock::Clock as _;
use reims_vgpu_core::platform::resource::Texture;
use reims_vgpu_core::platform::values::TextureDesc;
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
    CompletionSink, DecodeError, Decoders, FailedContexts, GraphicsDecoder, LibraryJob,
    LibraryStart, MadeLibrary, PipelineJob, ReflectionSink, ScheduledSink, SignalSink, Sinks,
    SubmittedSink, compile_refusing, complete, make_library, refused_reply, refusing,
};
use crate::graphics::device_info::device_info;
use crate::graphics::present::{PresentAnswers, PresentQueue};
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
/// raises for: the device's control thread (`reims-vgpu.graphics-control`), its refresh thread, a
/// command queue's service thread (`reims-vgpu.graphics-queue-<id>`), a pipeline compile's thread
/// (`reims-vgpu.graphics-compile-<n>`), or the platform's thread that runs a command buffer's
/// scheduled or completed handler; or the host's own, inside [`GraphicsDevice::mmio_write_taking`].
/// Each is called never inside another call of the host's into the device, and with no lock held
/// that a function of the device's waits for, directly or through a thread of the library's, so
/// they may call it at once. `scanout_written` runs with no lock of the library's held.
/// `raise_interrupt` may run under these: a command queue's claim on its next message, on the
/// queue's service thread; the device caches' lock, with the download's answer slot, at a
/// `SURFACE_DOWNLOAD` answered at once; a command buffer's `CB_SUBMITTED` latch, wherever that is
/// posted; and the presents' queue, at a present answered at once with no window. No function of
/// the device takes any of them, and `mmio_write_taking` only tries the claim and the decoder's
/// locks.
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
            // over the IOSurfaces they share where the guest names the order, and makes a queue
            // the kext asks unshared (WindowServer's) on the platform's unshared queue. Both
            // platforms make indirect command buffers, so every host decodes their messages.
            let features = u64::from(RGFX_FEATURE_CB_SUBMITTED)
                | u64::from(RGFX_FEATURE_SURFACE_ORDER)
                | u64::from(RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS)
                | u64::from(RGFX_FEATURE_UNSHARED_QUEUES)
                | if unified_memory {
                    0
                } else {
                    u64::from(RGFX_FEATURE_MANAGED_AT_OFFSET)
                }
                | match (decoder.surface_pages_available(), unified_memory) {
                    (false, _) => 0,
                    (true, true) => u64::from(RGFX_FEATURE_SURFACE_PAGES),
                    (true, false) => {
                        u64::from(RGFX_FEATURE_SURFACE_PAGES)
                            | u64::from(RGFX_FEATURE_SURFACE_CACHE)
                    }
                };
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
            let control = Arc::new(GraphicsControl {
                failed: decoder.failed_contexts(),
                decoders: decoder.decoders(),
                device: gpu_device,
                presents: Arc::new(PresentQueue::new(Arc::clone(&presented))),
                presented,
                pool,
                pending: Pending::default(),
                icb_worker: Worker::new("reims-vgpu.icb-create"),
                icbs: Arc::default(),
                reset_gate: Arc::default(),
            });
            let host = Box::new(CallbackHost {
                callbacks: config.callbacks,
                features,
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

    /// [`Self::mmio_write`], but a `DOORBELL` write of a command queue's ring may take the queue's
    /// next command buffer on the calling thread and post its `CB_SUBMITTED` there, calling
    /// `raise_interrupt` inside this call (`reims-graphics`'s `Device::mmio_write_taking`). It
    /// waits for no lock but the queues' table, as `mmio_write`'s doorbell does.
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
/// feeds it ring 0's object lifecycle on the control thread and the command queues' own rings'
/// messages on their service threads. Each message is carried out on a decoder of its own over the
/// shared state ([`Decoders::sender`]), whose locks are each held for a short section
/// (`decode`'s module doc): no thread holds one across a command buffer's walk, so a walk holds up
/// neither ring 0 nor another queue. A command buffer's completion, which runs on a GPU thread of
/// its own, never touches the decoder: it only posts through the queue's poster.
struct GraphicsControl<P: ApiPlatform> {
    /// The decoder's state, which each message's decoder is made over; a doorbell's thread's take
    /// carries its decoder to the queue's service thread ([`Self::take_command_buffer`]).
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
    /// The thread ring 0's `ICB_CREATE`s are made on, in ring order, off the control thread: the
    /// platform's make of a large buffer can take seconds (measured: about 24 s near 11.7 million
    /// commands on the Vulkan runtime on an RTX 5080), which would hold every process's control
    /// messages ([`Self::start_icb`]).
    icb_worker: Worker,
    /// The indirect command buffers whose create was handed to [`Self::icb_worker`] and is not
    /// yet answered, by object id.
    icbs: Arc<Unsettled>,
    /// The decoder's failed contexts, which the device asks of at a `QUEUE_CREATE`
    /// (`Host::context_refused`).
    failed: Arc<FailedContexts>,
    /// What the device's reset waits for ([`Self::reset`]): held shared by each message carried
    /// out off the control thread, a queue's command buffer, create or destroy, from before it
    /// asks whether its poster is stale to its end, and by each compile's and library make's
    /// store; held alone by the reset. So a reset comes after everything of the guest's that
    /// began before it, as under one lock over the decoder, and whatever begins after it finds
    /// its poster stale (or its compile cancelled), and none of it outlives the reset. The control
    /// thread's own messages are carried out in order with the reset, on its thread.
    reset_gate: Arc<RwLock<()>>,
}

/// The IOSurface the display pipe shows in place of the scanout (`PRESENT_SURFACE`): the texture
/// the decoder made over it, which the window's sink draws into the next drawable (`sink`), and
/// the presents' answers, given at the first refresh of the device's display after that draw has
/// finished ([`PresentAnswers`]). The window draws a present when it is made, never on its own:
/// each present is the guest's own, and none is batched or repeated by the GPU's work.
pub(crate) struct Presented<P: Platform> {
    /// The texture shown; `None` shows the scanout.
    pub(crate) texture: Mutex<Option<P::Texture>>,
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
            surface_id: AtomicU32::new(0),
            window: OnceLock::new(),
            refused: AtomicBool::new(false),
            formats: Mutex::new(None),
            answers: PresentAnswers::default(),
        }
    }
}

impl<P: Platform> Presented<P> {
    /// Shows `texture` (the surface `id`'s), or the scanout for `None`, and asks the window for a
    /// frame; `answer` is given at the vblank the display first shows it ([`PresentAnswers`]), or
    /// at once without a window.
    fn show(&self, id: u32, texture: Option<P::Texture>, answer: PresentAnswer) {
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
    }

    /// The device's reset: the display shows the scanout (none, after the reset) in place of any
    /// surface, whose texture is let go of in a refusal scope; the answers still owed are given
    /// as they come, and dropped, as each is of a present taken before the reset.
    fn reset(&self) {
        let previous = self.surface_id.swap(0, Ordering::AcqRel);
        let shown = self.texture.lock().take();
        if let Err(refusal) = reims_vgpu_core::raise::refusals::refusing(move || drop(shown)) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the texture of IOSurface {previous} was let go of at the device's reset, and its \
                 release was refused: {refusal}"
            );
        }
        if let Some(window) = self.window.get() {
            window.new_frame();
        }
    }

    /// The window draws the latest present: the presents shown so far, answered once the draw has
    /// finished, or once it is dropped undone ([`DrawnPresents`]).
    pub(crate) fn drawn(self: &Arc<Self>) -> DrawnPresents<P> {
        DrawnPresents {
            presented: Arc::clone(self),
            answers: self.answers.drawn(),
        }
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
/// GPU ([`DrawnPresents::after`]), or after the draw is dropped before its commit.
pub(crate) struct DrawnPresents<P: Platform> {
    /// Where they are answered.
    presented: Arc<Presented<P>>,
    /// Their answers.
    answers: Vec<PresentAnswer>,
}

impl<P: Platform> DrawnPresents<P> {
    /// The presents are answered after `command_buffer`, the draw, has finished: its completed
    /// handler holds them. Called before its commit; where the handler is refused, or the buffer
    /// is never committed, they are answered as if it had finished.
    pub(crate) fn after(self, command_buffer: &P::CommandBuffer) {
        if self.answers.is_empty() {
            return;
        }
        no_catch(command_buffer.add_completed_handler(OnFault::Die, Box::new(move |_| drop(self))));
    }
}

impl<P: Platform> Drop for DrawnPresents<P> {
    fn drop(&mut self) {
        self.presented
            .answers
            .done(std::mem::take(&mut self.answers));
    }
}

impl<P: ApiPlatform> GraphicsControl<P> {
    /// Drops every object of context `context` (its process closed the device or died), and the
    /// surfaces' entries whose objects went with them, then gives `answer` once every command
    /// buffer of the context the decoder committed has finished: the kext then gives the
    /// context's blocks out again.
    fn close_context(&self, context: u32, answer: CompleteAnswer) {
        entry::run(|_| {
            let (dropped, drain) = {
                let decoder = self.decoder();
                // Its compiles first, so none stores a pipeline in the table its close empties.
                self.pending.cancel_context(context);
                let dropped = decoder.drop_context(context);
                self.pool.context_closed(context);
                self.presents.context_gone(context);
                let drain = decoder.drain_context(
                    context,
                    true,
                    Box::new(move || answer.answer(RGFX_STATUS_OK)),
                );
                (dropped, drain)
            };
            log::info!(
                target: "reims_vgpu::graphics",
                "context {context} closed: {dropped} objects dropped"
            );
            drop(drain);
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
                // After everything of the guest's that began before the reset has ended.
                let _alone = self.reset_gate.write();
                self.pending.reset();
                let (contexts, dropped) = self.decoder().reset();
                for &context in &contexts {
                    self.pool.context_closed(context);
                }
                self.presents.reset();
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
    /// the decoder committed has finished.
    fn release_block(&self, release: &RgfxBlockRelease, answer: CompleteAnswer) {
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
        let drain = decoder.drain_context(
            release.context,
            false,
            Box::new(move || answer.answer(RGFX_STATUS_OK)),
        );
        // No buffer lies over the block's surface view any more: it goes with the block, and its
        // device cache once the surface's pages hold what was written of it, which the answer
        // waits for.
        let held = Arc::clone(&drain);
        let unread = release.flags & RGFX_BLOCK_RELEASE_NO_CLIENT != 0;
        let released = entry::run(|_| {
            decoder.block_released(
                &self.device,
                release.offset,
                unread,
                Box::new(move || drop(held)),
            )
        });
        if let Err(raise) = released {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the block at {:#x}: the platform raised at its device cache's write back \
                 ({raise:?})",
                release.offset
            );
        }
        drop(decoder);
        drop(drain);
    }

    /// `SURFACE_DOWNLOAD` of the block at `block`: answered once the surface's pages hold its
    /// device cache ([`GraphicsDecoder::surface_download`]).
    fn download_surface(&self, block: u64, answer: CompleteAnswer) {
        let answer = Arc::new(Mutex::new(Some(answer)));
        let give = Arc::clone(&answer);
        let done = entry::run(|_| {
            self.decoder().surface_download(
                &self.device,
                block,
                Box::new(move || {
                    if let Some(answer) = give.lock().take() {
                        answer.answer(RGFX_STATUS_OK);
                    }
                }),
            )
        });
        if let Err(raise) = done {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the block at {block:#x}: the platform raised at its device cache's download \
                 ({raise:?})"
            );
            if let Some(answer) = answer.lock().take() {
                answer.answer(RGFX_STATUS_REFUSED);
            }
        }
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
    /// pipeline still compiling waits for it first (`compile`). An indirect command buffer's
    /// create is made on the ICB thread ([`Self::start_icb`]), and the destroy of one still being
    /// made waits for it. A library lookup is answered here ([`Self::lookup_library`]).
    fn handle_stage3(self: &Arc<Self>, poster: &CompletionPoster, message: &[u8]) {
        let r#type = RgfxMsgHeader::read_from_prefix(message).map_or(0, |(h, _)| h.r#type);
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
            RGFX_CMD_COMMAND_BUFFER => self.pending.wait_for_command_buffer(message),
            RGFX_CMD_ICB_WRITE => self.pending.wait_for_icb_write(message),
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
                // An ICB_WRITE's records run on the control ring, ahead of the process's later
                // command buffers: how long they take is logged (diagnostics).
                let icb_start = (r#type == RGFX_CMD_ICB_WRITE).then(|| self.pool.now());
                let outcome = if foreign {
                    Ok(Err(DecodeError::BadReference))
                } else {
                    refusing(|| decoder.decode(&self.device, message))
                };
                if let Some(start) = icb_start {
                    log::debug!(
                        target: "reims_vgpu::graphics::icb",
                        "ICB_WRITE of {} bytes applied in {} us",
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

    /// [`Host::take_command_buffer`]: takes the `COMMAND_BUFFER` `message` of the queue `poster`
    /// posts onto, on a doorbell's thread, where it gets its `CB_SUBMITTED` at its take
    /// ([`GraphicsDecoder::submits_at_take`]) and nothing need be waited for: it runs on that
    /// queue, every lock of the decoder's it takes is free, each tried, and no pipeline it names is
    /// still compiling. It is begun here, its command buffer made and its sinks pointed at the
    /// queue's ring, as [`Self::handle_stage3`] begins it, and its decoder is carried to the
    /// queue's service thread with the rest of its work, which walks and commits it, then reports
    /// a failure as `handle_stage3` does. Its `CB_SUBMITTED` is posted here, waiting for nothing,
    /// once its ticket in the commit order is issued, which the guest's later buffers are
    /// committed after (`decode`'s `sequencer`); else by the service thread before the walk. A
    /// buffer the platform refused at its make gets none, and its failure on the service thread.
    /// A take that is not made lets go of what it took, under locks each held for a short section
    /// only.
    fn take_command_buffer(
        self: &Arc<Self>,
        poster: &CompletionPoster,
        message: &[u8],
    ) -> Option<Taken> {
        if foreign_queue(poster, message) || !self.pending.none_named_by(message) {
            return None;
        }
        // Held to the end of the buffer's work, on the service thread ([`Self::reset_gate`]).
        let carried = self.reset_gate.try_read_arc()?;
        if poster.stale() {
            return None;
        }
        let mut decoder =
            (self.decoders).sender(Some(sender_context(poster, message)), self.sinks(poster));
        decoder.set_try_only(true);
        // Its drain before its view of the table, as a buffer the service thread takes.
        let drain = decoder.command_buffer_drain()?;
        if !decoder.submits_at_take(message) {
            return None;
        }
        let begun =
            refusing(|| decoder.begin_command_buffer(&self.device, message, Some(drain), false));
        let submitted = match &begun {
            Ok(Ok(begun)) => begun.submitted_at_take().cloned(),
            _ => None,
        };
        // Its answer is posted here only once its ticket is issued.
        let issued = submitted.is_some() && decoder.issue_ticket(true);
        let post = submitted.clone().filter(|_| issued).map(|submitted| {
            let poster = poster.clone();
            Box::new(move || {
                submitted.try_post(|record| poster.try_post(record.as_bytes()));
            }) as Box<dyn FnOnce() + Send>
        });
        let (control, poster, message) = (Arc::clone(self), poster.clone(), message.to_vec());
        let rest = Box::new(move || {
            entry::run(|_| {
                let mut decoder = decoder;
                decoder.set_try_only(false);
                if let Some(submitted) = &submitted {
                    decoder.issue_ticket(false);
                    submitted.post();
                }
                let outcome = match begun {
                    Ok(Ok(begun)) => {
                        refusing(|| decoder.finish_command_buffer(&control.device, &message, begun))
                    }
                    Ok(Err(e)) => Ok(Err(e)),
                    Err(raise) => Err(raise),
                };
                let stood_in = !matches!(outcome, Ok(Ok(())))
                    && matches!(
                        refusing(|| decoder.stand_in_for_failed(&control.device, &message)),
                        Ok(true)
                    );
                drop(decoder);
                drop(carried);
                control.report(&poster, &message, outcome, stood_in);
            });
        });
        Some(Taken { post, rest })
    }

    /// A pipeline create from ring 0: prepared here, on the control thread, with the decoder's
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

    /// An `ICB_CREATE` from ring 0: unsettled here, in ring order, and made on the ICB thread
    /// ([`Self::create_icb`]), so ring 0 goes on meanwhile. Its answer goes through `poster`
    /// before it settles. Nothing of the guest's names the buffer before that answer: the bundle
    /// waits for it, for the buffer's `gpuResourceID`, before it hands the buffer out. So only the
    /// kext's destroy behind a create it stopped waiting for can name it while it is made, and
    /// that destroy waits for it ([`Self::handle_stage3`]). Made under the reset gate, held shared
    /// (recursively, as for the surface worker's jobs); where the device was reset since its
    /// begin, nothing is made and nothing answered.
    fn start_icb(self: &Arc<Self>, poster: &CompletionPoster, message: &[u8]) {
        let (_, seqno, obj_id) = message_identity(message);
        // The poster's epoch says whether the device was reset since: the settle's count of
        // resets is not read.
        let settle = self.icbs.begin(obj_id, 0, self.pool.now());
        let (me, poster, message) = (Arc::clone(self), poster.clone(), message.to_vec());
        self.icb_worker.submit(Box::new(move || {
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
        }));
    }

    /// An indirect command buffer's create from ring 0, on the ICB thread ([`Self::start_icb`]),
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
    /// prepared here, on the control thread, with the decoder's table (a malformed message is
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
            self.presents.present(0, None, answer, Vec::new());
            return;
        }
        let decoder = self.decoder();
        if let Some(generation) = generation {
            // The pages as of the present's generation, where the surface has a device cache.
            let filled =
                entry::run(|_| decoder.fill_presented(&self.device, iosurface_id, generation));
            if let Err(raise) = filled {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "IOSurface {iosurface_id}: the platform raised at its device cache's fill for \
                     a present ({raise:?})"
                );
            }
        }
        let found = decoder.surface_present(iosurface_id);
        drop(decoder);
        match found {
            Some((texture, writers, owner)) => {
                // The process that made the texture over the surface the display shows owns the
                // scanout (WindowServer): its compiles go first.
                self.pool.favour(owner);
                self.presents
                    .present(iosurface_id, Some(texture), answer, writers);
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

    fn device_reset(&self) {
        self.control.reset();
    }

    fn features(&self) -> u64 {
        self.features
    }

    fn surface_pages(&self, pages: &RgfxSurfacePages, runs: Vec<RgfxPageRun>, cached: bool) -> u32 {
        let control = &self.control;
        entry::run(|_| {
            control
                .decoder()
                .surface_pages(&control.device, pages, &runs, cached)
        })
    }

    fn surface_download(&self, block_offset: u64, answer: CompleteAnswer) {
        self.control.download_surface(block_offset, answer);
    }

    fn stage3_message(&self, completion: &CompletionPoster, message: &[u8]) {
        self.control.handle_stage3(completion, message);
    }

    fn take_command_buffer(&self, completion: &CompletionPoster, message: &[u8]) -> Option<Taken> {
        self.control.take_command_buffer(completion, message)
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
