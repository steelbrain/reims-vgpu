//! [`GraphicsDevice`]: the reims-graphics device (PCI 1234:5247), `reims-graphics`'s host side
//! with its display shown in a window of the library's, made from a [`GraphicsConfig`].
//!
//! The device's region (BAR 2) is made here and shared: the device takes the guest's commands
//! from its rings, and the window's sink imports it once as a no-copy buffer on the platform
//! (Metal on macOS, the Vulkan platform's runtime with the `vulkan` feature) and blits the
//! scanout out of it into each drawable, on the GPU, with no copy on the CPU (`sink`). The
//! buffer's deallocator holds the region, so it stays mapped until the GPU has let the buffer
//! go, after the device and the window.
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

use std::os::fd::RawFd;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use reims_graphics::ops::RgfxCommandBuffer;
use reims_graphics::region::Region;
use reims_graphics::wire::{
    RGFX_CMD_COMMAND_BUFFER, RGFX_DEVICE_INFO_OFFSET, RgfxCbComplete, RgfxMsgHeader,
    RgfxPipelineReflection,
};
use reims_graphics::{
    ControlCompletionPoster, CursorImage, FixedScreen, Host, Mode, QueueCompletionPoster, Scanout,
    Screen,
};
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_window::{FrameSink, Window, WindowConfig};
use zerocopy::{FromBytes, IntoBytes};

use crate::DefaultPlatform;
use crate::entry;
use crate::error::Error;
use crate::graphics::completion::cb_failed;
use crate::graphics::decode::{CompletionSink, GraphicsDecoder};
use crate::graphics::device_info::device_info;
use crate::platform::ApiPlatform;
use crate::window::WindowLink;

mod completion;
mod decode;
mod device_info;
#[cfg(target_os = "macos")]
mod metal;
mod sink;
#[cfg(feature = "vulkan")]
mod vulkan;

/// Raises MSI-X vector `vector`: the device's interrupt.
pub type GraphicsRaiseFn = Arc<dyn Fn(u32) + Send + Sync>;
/// Whether the guest wrote any of the `length` bytes at `offset` in the region since the last
/// ask: the host's dirty log over the scanout, taken and cleared.
pub type ScanoutWrittenFn = Arc<dyn Fn(u64, u64) -> bool + Send + Sync>;

/// A graphics device's callbacks. Each is called on one of the device's two threads
/// (`reims-vgpu.graphics-control`, `reims-vgpu.graphics-refresh`), never inside a call of the
/// host's into the device, with no lock of the library's held; the two may call at once.
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
    /// which feeds it the control ring's stage-3 messages.
    control: Arc<GraphicsControl<P>>,
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
            region.write(
                RGFX_DEVICE_INFO_OFFSET,
                device_info(&gpu_device.caps()).as_bytes(),
            );
            let control = Arc::new(GraphicsControl {
                decoder: Mutex::new(GraphicsDecoder::new(
                    Arc::clone(&platform),
                    Arc::clone(&region),
                )),
                device: gpu_device,
                poster: OnceLock::new(),
                control_completion: OnceLock::new(),
                presented: Arc::new(Presented::default()),
            });
            let host = Box::new(CallbackHost {
                callbacks: config.callbacks,
                control: Arc::clone(&control),
            });
            let (window, screen): (Option<WindowLink>, Box<dyn Screen>) = match config.window {
                Some(window_config) => {
                    let (link, screen) = WindowScreen::open(
                        &platform,
                        &region,
                        &control.presented,
                        window_config,
                    );
                    (Some(link), Box::new(screen))
                }
                None => (None, Box::new(FixedScreen(FALLBACK_MODE))),
            };
            let device =
                reims_graphics::Device::new(region, host, screen).ok_or(Error::HostResources)?;
            // The completion ring exists now: let the decoder post its command-buffer completions
            // and pipeline reflections onto it, and keep a poster for the command buffers the host
            // could not carry out at all (which owe the guest a completion too).
            let poster = device.completion_poster();
            let _ = control.poster.set(poster.clone());
            control.set_sinks(&poster);
            Ok(GraphicsDevice {
                window,
                device,
                control,
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
        self.control.decoder.lock().object_count()
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

/// The stage-3 decoder and the device handle it drives, shared by the device's [`Host`] (which
/// feeds it the control ring's stage-3 messages on the control thread) and [`GraphicsDevice`]
/// (which, once the reims-graphics device and so its completion ring exist, gives the decoder the
/// sinks that post onto it). The decoder is behind a [`Mutex`]: the control thread drives it one
/// message at a time, and a command buffer's completion, which runs on a GPU thread of its own,
/// never touches the decoder — it only posts through the poster.
struct GraphicsControl<P: ApiPlatform> {
    /// The decoder.
    decoder: Mutex<GraphicsDecoder<P>>,
    /// The GPU device it drives.
    device: P::Device,
    /// The control completion ring's poster, set once the device exists: the host posts a failed
    /// command buffer's completion through it.
    poster: OnceLock<ControlCompletionPoster>,
    /// The decoder's completion sink onto the control completion ring, set once the device exists.
    /// Each queue's submissions post onto their own ring, so the decoder's one completion sink is
    /// set to the right ring before every decode; this is the control ring's, re-set before every
    /// control-ring message.
    control_completion: OnceLock<CompletionSink>,
    /// The surface the display pipe shows, shared with the window's sink.
    presented: Arc<Presented<P>>,
}

/// The IOSurface the display pipe shows in place of the scanout (`PRESENT_SURFACE`): the texture
/// the decoder made over it, which the window's sink copies into each drawable (`sink`), and
/// what asks the window for a frame when it changes or a command buffer completes.
pub(crate) struct Presented<P: Platform> {
    /// The texture shown; `None` shows the scanout.
    pub(crate) texture: Mutex<Option<P::Texture>>,
    /// The surface's IOSurface ID, 0 for none.
    surface_id: AtomicU32,
    /// The window, once it has opened.
    window: OnceLock<Arc<Window>>,
    /// A surface of a format the sink cannot copy was logged.
    refused: AtomicBool,
    /// A command buffer completed since the refresh last asked: the GPU may have written the
    /// scanout, which the host's dirty log never sees.
    gpu_wrote: AtomicBool,
}

impl<P: Platform> Default for Presented<P> {
    fn default() -> Self {
        Presented {
            texture: Mutex::new(None),
            surface_id: AtomicU32::new(0),
            window: OnceLock::new(),
            refused: AtomicBool::new(false),
            gpu_wrote: AtomicBool::new(false),
        }
    }
}

impl<P: Platform> Presented<P> {
    /// Shows `texture` (the surface `id`'s), or the scanout for `None`, and asks for a frame.
    fn show(&self, id: u32, texture: Option<P::Texture>) {
        let previous = self.surface_id.swap(id, Ordering::AcqRel);
        if previous != id {
            log::info!(
                target: "reims_vgpu::graphics",
                "the display shows IOSurface {id} (was {previous}; 0 is the scanout)"
            );
        }
        *self.texture.lock() = texture;
        self.frame();
    }

    /// A command buffer completed: the GPU may have written what the window shows.
    fn completed(&self) {
        self.gpu_wrote.store(true, Ordering::Release);
        if self.texture.lock().is_some() {
            self.frame();
        }
    }

    /// Whether a command buffer completed since the last ask.
    fn take_gpu_wrote(&self) -> bool {
        self.gpu_wrote.swap(false, Ordering::AcqRel)
    }

    /// Asks the window for a frame.
    fn frame(&self) {
        if let Some(window) = self.window.get() {
            window.new_frame();
        }
    }

    /// The sink met a surface of `format`, which it cannot copy into its drawable: logged once.
    pub(crate) fn refuse_format(&self, format: u64) {
        if !self.refused.swap(true, Ordering::AcqRel) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the presented IOSurface's texture has pixel format {format}, which the window \
                 cannot copy (BGRA8 only): the scanout shows instead"
            );
        }
    }
}

impl<P: ApiPlatform> GraphicsControl<P> {
    /// Gives the decoder its completion and reflection sinks, each posting onto the control
    /// completion ring through `poster`: a finished command buffer's [`RgfxCbComplete`] and a
    /// pipeline create's [`RgfxPipelineReflection`].
    fn set_sinks(&self, poster: &ControlCompletionPoster) {
        let completion = poster.clone();
        let presented = Arc::clone(&self.presented);
        let sink: CompletionSink = Arc::new(move |record: RgfxCbComplete| {
            completion.post(record.as_bytes());
            presented.completed();
        });
        let _ = self.control_completion.set(Arc::clone(&sink));
        let mut decoder = self.decoder.lock();
        decoder.set_completion_sink(sink);
        let reflection = poster.clone();
        decoder.set_reflection_sink(Arc::new(move |record: RgfxPipelineReflection| {
            reflection.post(record.as_bytes());
        }));
    }

    /// Carries out one stage-3 control message against the GPU device, under a library entry (an
    /// autorelease pool and the thread marked as ours). The decoder posts a command buffer's
    /// completion and a pipeline create's reflection through its sinks; a command buffer the host
    /// could not carry out — a malformed stream, a bad reference, or an exec fault before it was
    /// committed — owes the guest a completion that no handler will post, so the host posts an
    /// internal-error one for it here.
    fn handle(&self, message: &[u8]) {
        entry::run(|_| {
            let outcome = {
                let mut decoder = self.decoder.lock();
                // A queue's submissions may have left the decoder's completion sink pointed at a
                // per-queue ring; a control-ring command buffer's completion goes on the control
                // ring, so re-point it before the decode.
                if let Some(sink) = self.control_completion.get() {
                    decoder.set_completion_sink(Arc::clone(sink));
                }
                decoder.decode(&self.device, message)
            };
            match outcome {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "a stage-3 control message was not carried out: {e:?}"
                    );
                    self.fail_command_buffer(message);
                }
                Err(raise) => {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "a stage-3 control message faulted: {raise}"
                    );
                    self.fail_command_buffer(message);
                }
            }
        });
    }

    /// Carries out one `COMMAND_BUFFER` message taken from a command queue's own submission ring,
    /// under a library entry. The decoder's completion sink is pointed at this queue's completion
    /// ring (through `poster`) for the decode, so the buffer's completion handler, running on a GPU
    /// thread, posts its `CB_COMPLETE` there and raises the queue's MSI-X vector. A command buffer
    /// the host could not carry out owes the guest a completion that no handler will post, so the
    /// host posts an internal-error one onto the queue's ring here.
    fn handle_queue(&self, poster: &QueueCompletionPoster, message: &[u8]) {
        entry::run(|_| {
            let queue_poster = poster.clone();
            let presented = Arc::clone(&self.presented);
            let sink: CompletionSink = Arc::new(move |record: RgfxCbComplete| {
                queue_poster.post(record.as_bytes());
                presented.completed();
            });
            let outcome = {
                let mut decoder = self.decoder.lock();
                decoder.set_completion_sink(sink);
                decoder.decode(&self.device, message)
            };
            match outcome {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "a queue command buffer was not carried out: {e:?}"
                    );
                    self.fail_queue_command_buffer(poster, message);
                }
                Err(raise) => {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "a queue command buffer faulted: {raise}"
                    );
                    self.fail_queue_command_buffer(poster, message);
                }
            }
        });
    }

    /// Shows the IOSurface `iosurface_id` (0: the scanout): false, with nothing changed, where the
    /// decoder holds no texture over it.
    fn present_surface(&self, iosurface_id: u32) -> bool {
        if iosurface_id == 0 {
            self.presented.show(0, None);
            return true;
        }
        let texture = self.decoder.lock().surface_texture(iosurface_id);
        match texture {
            Some(texture) => {
                self.presented.show(iosurface_id, Some(texture));
                true
            }
            None => false,
        }
    }

    /// Posts an internal-error completion for `message` where it is a `COMMAND_BUFFER` whose
    /// decode failed, so a guest blocked on the buffer unblocks; other failed messages are
    /// fire-and-forget and owe no completion.
    fn fail_command_buffer(&self, message: &[u8]) {
        let Some(poster) = self.poster.get() else {
            return;
        };
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
    }

    /// Posts an internal-error completion onto a queue's own completion ring for a `COMMAND_BUFFER`
    /// its service thread took but the decode failed, so a guest blocked on the buffer unblocks.
    fn fail_queue_command_buffer(&self, poster: &QueueCompletionPoster, message: &[u8]) {
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
    }
}

/// The host's callbacks and its stage-3 decoder as the device's [`Host`].
struct CallbackHost<P: ApiPlatform> {
    /// The host's interrupt and dirty-log callbacks.
    callbacks: GraphicsCallbacks,
    /// The stage-3 decoder, fed the control ring's stage-3 messages.
    control: Arc<GraphicsControl<P>>,
}

impl<P: ApiPlatform> Host for CallbackHost<P> {
    fn raise_interrupt(&self, vector: u32) {
        if let Some(f) = &self.callbacks.raise_interrupt {
            f(vector);
        }
    }

    /// The host's dirty log, or a command buffer's completion since the last ask: the GPU's
    /// writes to the scanout never reach the dirty log.
    fn scanout_written(&self, offset: u64, length: u64) -> bool {
        let gpu = self.control.presented.take_gpu_wrote();
        let cpu = self
            .callbacks
            .scanout_written
            .as_ref()
            .is_none_or(|f| f(offset, length));
        gpu || cpu
    }

    fn present_surface(&self, iosurface_id: u32) -> bool {
        self.control.present_surface(iosurface_id)
    }

    fn stage3_message(&self, message: &[u8]) {
        self.control.handle(message);
    }

    fn queue_message(&self, completion: &QueueCompletionPoster, message: &[u8]) {
        self.control.handle_queue(completion, message);
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
        }
        let _ = presented.window.set(Arc::clone(&window));
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
/// `None` on any other platform (the testkit's), or where the platform refuses the region's
/// import or a command queue.
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
