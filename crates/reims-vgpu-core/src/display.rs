//! Displays: the host's display (`Display` over `DisplayCore`), its guest display on the guest's
//! side of a port ([`guest_display`]), the EFI display ([`efi`]) and the page they share
//! ([`page`]).
//!
//! A display's parts, by file: the frame it presents and the wait for frames in flight
//! ([`frame`]), the pass that encodes it into the host's command buffer and the fragment
//! properties ([`render`]), the gamma table ([`gamma`]), the calls into the host's handlers on its
//! queue ([`host_calls`]), and the pipelines ([`pipeline`]).
//!
//! **Two tiers.** `Display` is the host's reference: its last drop is hot-unplug (detach from the
//! port, the guest display's disconnect interrupt), on the releasing thread. `DisplayCore` holds
//! the display's state: the device and the guest display reach it unretained
//! ([`DeviceCore::display_core`]), so neither ever makes the last release on its own thread, under
//! its own locks. A block the display dispatches holds a `Display`, so it keeps the display alive
//! and its end may be the last release; a block made while that release runs holds the core alone.
//!
//! **Locks.** `display_mutex` guards [`DisplayLocked`]; `guest_display` is the guest display
//! pointer under its own mutex. What is read without the display mutex (the counters, the
//! responsiveness, the protection options, the colour-space flag, the mode list, the queue, the
//! newFrame source) is an atomic or a leaf cell beside it, so no lock is taken where none is
//! needed. Methods that run with the display mutex held are `*_locked` and take its contents.
//!
//! **Ordering contracts** (testing.md 7):
//! - A present runs under the guest display's mutex, then the display's. `set_mode_list` never
//!   holds the display's mutex while it takes the guest display's, so the two never wait on each
//!   other (`guest_display::tests::set_mode_list_while_a_present_holds_the_guest_display`). A
//!   present waits for frames in flight up to 2 s of wall clock, then the mode change, the gamma
//!   table, the frame and the colour matrix, and the completion last, all under the display mutex.
//! - The frame is held, never copied, until the next present, `set_guest_display`, or the display's
//!   end: a frame over the guest's IOSurface keeps it in use (the guest sees its unmap refused).
//! - An encode counts a frame in flight, even with no frame; its completion counts it out.
//! - The gamma table's single entry dies after the uploads.
//! - The display's end releases, in order: the unplug, the buffers and the name, the handlers,
//!   the mode list, the frame, the gamma texture and the pipelines, the device, the property
//!   buffer and the guest colour space; then the line that says it was destroyed.

pub mod descriptor;
pub mod efi;
mod frame;
mod gamma;
pub mod guest_display;
pub mod host_calls;
pub mod mode;
pub mod page;
pub mod pipeline;
mod render;
#[cfg(test)]
mod tests;

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use parking_lot::{Condvar, Mutex};

pub use frame::{NewFrameSource, PENDING_FRAMES_TIMEOUT};
pub use gamma::{GAMMA_TABLE_LABEL, GammaTableArgs};
pub use render::{
    ColorMatrix, DisplayFragmentShaderProperties, IDENTITY_COLOR_MATRIX, MINIMUM_TEXTURE_USAGE,
    POS_DATA, TEX_DATA,
};

use crate::device::{Device, DeviceCore};
use crate::display::descriptor::DisplayDescriptor;
use crate::display::guest_display::GuestDisplay;
use crate::display::host_calls::SdkProps;
use crate::display::mode::DisplayMode;
use crate::display::pipeline::{DisplayPipelineCache, PipelineKey};
use crate::host::HostQueue;
use crate::platform::{Handle as _, Platform};
use crate::raise::{DeathId, Kept, OnFault, RaiseId, RaiseResult, keep_forever};
use crate::rt::pool;

/// `PGDisplayCoord_t`: `{uint16_t x; uint16_t y;}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct DisplayCoord {
    /// `x`.
    pub x: u16,
    /// `y`.
    pub y: u16,
}

/// `PGDisplayChromaticity`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DisplayChromaticity {
    /// `x`.
    pub x: f32,
    /// `y`.
    pub y: f32,
}

/// `PGDisplayColorSpace`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DisplayColorSpace {
    /// `redPrimary`.
    pub red_primary: DisplayChromaticity,
    /// `greenPrimary`.
    pub green_primary: DisplayChromaticity,
    /// `bluePrimary`.
    pub blue_primary: DisplayChromaticity,
    /// `whitePoint`.
    pub white_point: DisplayChromaticity,
}

/// A display's `name`: the text the SDK call copied. The host's string object is not kept, so its
/// identity and its later mutations do not show.
#[derive(Clone)]
pub struct DisplayName(Arc<str>);

impl DisplayName {
    /// A name.
    pub fn new(name: impl Into<String>) -> DisplayName {
        DisplayName(Arc::from(name.into()))
    }

    /// The name's text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The name's contents.
    pub fn current(&self) -> String {
        String::from(&*self.0)
    }
}

impl From<String> for DisplayName {
    fn from(name: String) -> DisplayName {
        DisplayName::new(name)
    }
}

impl From<&str> for DisplayName {
    fn from(name: &str) -> DisplayName {
        DisplayName::new(name)
    }
}

/// Equal contents.
impl PartialEq for DisplayName {
    fn eq(&self, other: &DisplayName) -> bool {
        self.0 == other.0
    }
}

impl PartialEq<&str> for DisplayName {
    fn eq(&self, other: &&str) -> bool {
        &*self.0 == *other
    }
}

impl fmt::Debug for DisplayName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

/// A display's name in a log line: quoted, or `none`.
fn name_arg(name: Option<&DisplayName>) -> String {
    name.map_or_else(|| "none".to_owned(), |n| format!("{:?}", n.as_str()))
}

/// What a display raises. The FIFO's display commands catch it and fault the command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaiseKind {
    /// The gamma table's mapped length is past the largest local task.
    GammaLength,
    /// The gamma table's sum is not the one the guest sent.
    GammaCorrupt,
}

impl RaiseKind {
    /// The dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            RaiseKind::GammaLength => "display.gamma.length",
            RaiseKind::GammaCorrupt => "display.gamma.corrupt",
        }
    }
}

impl fmt::Display for RaiseKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<RaiseKind> for RaiseId {
    fn from(k: RaiseKind) -> RaiseId {
        RaiseId::Display(k)
    }
}

/// Why a display was refused ([`DisplayCore::new`]), in the order the checks are made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DisplayRefusal {
    /// The port is not below the device's display port count.
    PortOutOfRange,
    /// Metal refused the display's pipeline cache.
    Metal,
    /// Another display of the process holds the serial number.
    SerialInUse,
    /// Another display holds the port.
    PortInUse,
}

/// Where a display ends the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// Metal gave no default library for the pipelines.
    NoLibrary,
    /// The library has no vertex function.
    NoVertexFunction,
    /// The library has no fragment function.
    NoFragmentFunction,
    /// The library has no gamma fragment function.
    NoGammaFragmentFunction,
    /// Metal refused the position buffer.
    PositionBufferRefused,
    /// Metal refused the texture-coordinate buffer.
    TexCoordBufferRefused,
    /// Metal refused the fragment property buffer.
    PropertyBufferRefused,
    /// A buffer's contents are NULL where the display writes them.
    NullContents,
    /// A present to a display with no queue or no newFrame handler.
    PresentWithoutNewFrameHandler,
    /// A mode change with a handler and no queue (`dispatch_async(NULL)`).
    ModeChangeWithoutQueue,
    /// The mode change block found no handler when it ran (a call through nil).
    ModeChangeHandlerGone,
    /// `set_guest_display` with both or neither of the old and new guest display nil.
    GuestDisplaySwapUnbalanced,
    /// A guest present with a frame and no newFrame source (`dispatch_source_merge_data(NULL)`).
    NoNewFrameSource,
    /// A gamma table of no entries.
    GammaNoEntries,
    /// Metal refused the gamma texture.
    GammaTextureRefused,
    /// A gamma table of one entry reads index 0xffffffff, after the uploads.
    GammaOneEntry,
    /// A gamma table read past its end.
    GammaTableRead,
    /// An EFI present, stop or cancel off the EFI display's present queue.
    EfiOffPresentQueue,
}

impl DeathKind {
    /// The dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            DeathKind::NoLibrary => "display.pipeline.no_library",
            DeathKind::NoVertexFunction => "display.pipeline.no_vertex_function",
            DeathKind::NoFragmentFunction => "display.pipeline.no_fragment_function",
            DeathKind::NoGammaFragmentFunction => "display.pipeline.no_gamma_fragment_function",
            DeathKind::PositionBufferRefused => "display.init.position_buffer",
            DeathKind::TexCoordBufferRefused => "display.init.tex_coord_buffer",
            DeathKind::PropertyBufferRefused => "display.init.property_buffer",
            DeathKind::NullContents => "display.buffer.null_contents",
            DeathKind::PresentWithoutNewFrameHandler => "display.present.no_new_frame_handler",
            DeathKind::ModeChangeWithoutQueue => "display.mode_change.no_queue",
            DeathKind::ModeChangeHandlerGone => "display.mode_change.handler_gone",
            DeathKind::GuestDisplaySwapUnbalanced => "display.guest_display.swap_unbalanced",
            DeathKind::NoNewFrameSource => "display.frame.no_source",
            DeathKind::GammaNoEntries => "display.gamma.no_entries",
            DeathKind::GammaTextureRefused => "display.gamma.texture_refused",
            DeathKind::GammaOneEntry => "display.gamma.one_entry",
            DeathKind::GammaTableRead => "display.gamma.table_read",
            DeathKind::EfiOffPresentQueue => "display.efi.off_present_queue",
        }
    }

    /// The kind of death.
    pub(crate) const fn class(self) -> crate::raise::death::DeathKind {
        use crate::raise::death::DeathKind as K;
        match self {
            DeathKind::NullContents
            | DeathKind::ModeChangeWithoutQueue
            | DeathKind::NoNewFrameSource => K::NullDeref,
            DeathKind::ModeChangeHandlerGone => K::NilBlock,
            DeathKind::GammaOneEntry | DeathKind::GammaTableRead => K::OutOfBounds,
            DeathKind::EfiOffPresentQueue => K::DispatchMisuse,
            _ => K::Assert,
        }
    }

    /// The site a death of this kind names (tests).
    #[cfg(test)]
    pub(crate) fn site(self) -> crate::raise::death::DeathSite {
        crate::raise::death::DeathId::from(self).site()
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<DeathKind> for DeathId {
    fn from(k: DeathKind) -> DeathId {
        DeathId::Display(k)
    }
}

/// The state the display mutex guards (the present, the encodes, `set_guest_display`, the
/// completion).
pub struct DisplayLocked<P: Platform> {
    /// The frame size last presented.
    pub width: u16,
    /// Likewise.
    pub height: u16,
    /// The full-screen strip's positions.
    pub pos_buf: Option<P::Buffer>,
    /// Its texture coordinates.
    pub tex_buf: Option<P::Buffer>,
    /// The fragment shader's properties ([`DisplayFragmentShaderProperties`]).
    pub frag_shader_property_buf: Option<P::Buffer>,
    /// The frame presented last, held (`DisplayCore::has_current_frame` mirrors whether it is
    /// set, for the unlocked `signal_current_frame`).
    pub current_frame: Option<P::Texture>,
    /// The strip's vertex count.
    pub vtx_count: u64,
    /// The pipeline the encodes use: the target's pixel format, and whether there is gamma.
    pub pipeline: PipelineKey,
    /// The IOSurface pixel format presented last.
    pub iosurface_pixel_format: u32,
    /// The gamma table's texture.
    pub gamma_table_tex: Option<P::Texture>,
    /// The protection requirements presented last.
    pub protection_requirements: u64,
    /// The guest display's colour space (its ICC profile bytes), taken when the guest marked it
    /// dirty.
    pub guest_color_space: Option<Arc<[u8]>>,
}

impl<P: Platform> DisplayLocked<P> {
    /// Zero and nil.
    fn zeroed() -> Self {
        DisplayLocked {
            width: 0,
            height: 0,
            pos_buf: None,
            tex_buf: None,
            frag_shader_property_buf: None,
            current_frame: None,
            vtx_count: 0,
            pipeline: PipelineKey::default(),
            iosurface_pixel_format: 0,
            gamma_table_tex: None,
            protection_requirements: 0,
            guest_color_space: None,
        }
    }
}

/// The host's reference to a display. The host holds it (`Arc`), the device holds it unretained
/// (`Weak`), and the last drop of the `Arc` is hot-unplug on that thread: detach from the port,
/// then release what the display holds, the device last (module doc, "Two tiers"). It derefs to
/// its [`DisplayCore`].
///
/// The guest display reaches the core, not this reference, so a guest display operation in flight
/// never keeps the display alive: the host's last release runs the unplug on the host's thread,
/// which waits in the detach for the guest display's mutex. Whoever holds only the `Weak` reaches
/// the core through the device ([`DeviceCore::display_core`]), never by upgrading it, which would
/// make its own drop the last release.
pub struct Display<P: Platform> {
    core: Arc<DisplayCore<P>>,
}

impl<P: Platform> std::ops::Deref for Display<P> {
    type Target = DisplayCore<P>;

    fn deref(&self) -> &DisplayCore<P> {
        &self.core
    }
}

impl<P: Platform> Display<P> {
    /// See [`DisplayCore::new`].
    pub fn new(
        platform: &Arc<P>,
        device: Arc<DeviceCore<P>>,
        descriptor: &DisplayDescriptor,
        port: u64,
        serial_num: u32,
    ) -> RaiseResult<Result<Arc<Display<P>>, DisplayRefusal>> {
        DisplayCore::new(platform, device, descriptor, port, serial_num)
    }

    /// The core without the host's reference: what the guest display holds (unretained) for its
    /// client.
    pub fn core(&self) -> &Arc<DisplayCore<P>> {
        &self.core
    }
}

/// The host's handle to a display, and the retain a block that holds the display takes
/// ([`DisplayCore::retain_self`]): one counted reference. There is no `Clone`:
/// [`DisplayHandle::clone_ref`] takes another reference, and when the last one goes its drop
/// runs the display's end (hot-unplug) here, on this thread. A display keeps its device alive.
pub struct DisplayHandle<P: Platform>(Arc<Display<P>>);

impl<P: Platform> DisplayHandle<P> {
    /// Takes over `display`'s reference.
    pub(crate) fn new(display: Arc<Display<P>>) -> DisplayHandle<P> {
        DisplayHandle(display)
    }

    /// One more reference.
    pub fn clone_ref(&self) -> DisplayHandle<P> {
        DisplayHandle(Arc::clone(&self.0))
    }

    /// Whether `self` and `other` are references to one display.
    pub fn ptr_eq(&self, other: &DisplayHandle<P>) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// The display, unretained: the identity the device's port table and the guest display hold.
    pub fn downgrade(&self) -> Weak<Display<P>> {
        Arc::downgrade(&self.0)
    }
}

impl<P: Platform> std::ops::Deref for DisplayHandle<P> {
    type Target = Display<P>;

    fn deref(&self) -> &Display<P> {
        &self.0
    }
}

/// The display as a block it dispatches holds it ([`DisplayCore::block_self`]).
enum BlockSelf<P: Platform> {
    /// The display, retained: the block's end releases it, and a last release there runs the
    /// display's end on that thread.
    Display(DisplayHandle<P>),
    /// The core of a display whose last release has begun: a retain of it takes nothing, and
    /// the block still runs against the core.
    Deallocating(Arc<DisplayCore<P>>),
}

impl<P: Platform> std::ops::Deref for BlockSelf<P> {
    type Target = DisplayCore<P>;

    fn deref(&self) -> &DisplayCore<P> {
        match self {
            BlockSelf::Display(d) => d,
            BlockSelf::Deallocating(c) => c,
        }
    }
}

/// The display's hold on its device: the device's retain (`None` once the device's end has
/// begun) and its core. The display's end drops both, the retain first, so the device's whole
/// teardown runs inside it when the display held the last reference, before the display's
/// destroyed line.
struct DeviceRef<P: Platform> {
    retain: Option<Device<P>>,
    host: Arc<DeviceCore<P>>,
}

impl<P: Platform> DeviceRef<P> {
    /// The retain first (the device's teardown, when it is the last), then the core.
    fn release(self) {
        let DeviceRef { retain, host } = self;
        drop(retain);
        drop(host);
    }
}

/// A display's state and methods (see [`Display`]). Holding it keeps the memory, not the display:
/// the end runs when the last [`Display`] goes.
///
/// The fields drop in their order here when a block still held the core at the display's end
/// (the end releases what it can reach alone first).
pub struct DisplayCore<P: Platform> {
    /// The platform.
    platform: Arc<P>,
    /// The host's reference, for the blocks that retain it; set once it exists (before the
    /// attach).
    this: OnceLock<Weak<Display<P>>>,
    /// Whether the serial number is reserved (until the unplug).
    reserved_serial_num: AtomicBool,
    /// Whether the display is attached to its port (until the unplug).
    attached: AtomicBool,
    /// The device and its retain, both released by the end: see [`DeviceRef`]. A leaf cell,
    /// never held across a call.
    device: Mutex<Option<DeviceRef<P>>>,
    /// The copied descriptor. The SDK properties the display keeps (the size, the queue, the
    /// five handlers) are never in it: they are `props`.
    pub(crate) descriptor: DisplayDescriptor,
    /// The SDK's readonly properties, taken from the descriptor at init.
    props: SdkProps,
    /// The configuration epoch, origin and scale factor (copied from the descriptor at init).
    pub(crate) compositor_parameters: Mutex<(u64, DisplayCoord, u8)>,
    /// The newFrame source (`DATA_ADD` onto the host's queue); the unplug cancels and clears it.
    pub(crate) new_frame_event_source: Mutex<Option<NewFrameSource>>,
    /// The pipelines (always set on a display `new` returned, until the end releases them). They
    /// hold the display side's one reference to the host's `MTLDevice`.
    pub(crate) render_pipeline_cache: Option<DisplayPipelineCache<P>>,
    /// Whether the guest changed its colour space since the last present.
    guest_color_space_dirty: AtomicBool,
    /// Whether the host encoded the current frame (written without the lock).
    current_frame_encoded: AtomicBool,
    /// Whether there is a current frame, for the unlocked `signal_current_frame`.
    pub(crate) has_current_frame: AtomicBool,
    /// Whether the colour matrix the property buffer holds is the identity (written under the
    /// display mutex; `render.rs`, the console's CPU copy).
    pub(crate) color_matrix_identity: AtomicBool,
    /// Whether the gamma table the display holds maps every 8-bit level to itself through the
    /// gamma pass (or there is none; written under the display mutex; `gamma.rs`).
    pub(crate) gamma_identity: AtomicBool,
    /// The mode-list responsiveness: 0 with no guest display, 1 with one, 2 with one that
    /// live-resizes.
    mode_list_responsiveness: AtomicI32,
    /// The protection options (set under the guest display pointer's lock).
    pub(crate) protection_options: AtomicU64,
    /// The modes set last, which the guest display shares. Written under the display mutex, read
    /// without it. A leaf cell.
    mode_list: Mutex<Option<ModeList>>,
    /// Held across a whole `set_mode_list`, so lists set at once reach the guest display in the
    /// order they were stored. Taken before any other lock of the display's.
    mode_list_update: Mutex<()>,
    /// The protection requirements as the unlocked getter reads them: stored with
    /// [`DisplayLocked::protection_requirements`] under the display mutex.
    protection_requirements_unlocked: AtomicU64,
    /// The guest colour space as the unlocked getter reads it: stored with
    /// [`DisplayLocked::guest_color_space`] under the display mutex (a leaf cell).
    guest_color_space_unlocked: Mutex<Option<Arc<[u8]>>>,
    /// The display mutex and what it guards.
    pub(crate) display_mutex: Mutex<DisplayLocked<P>>,
    /// The guest display, unretained, under its own mutex.
    pub(crate) guest_display: Mutex<Option<Weak<GuestDisplay<P>>>>,
    /// Signalled when the last frame in flight completes (waits on the display mutex).
    pub(crate) in_flight_condition: Condvar,
    /// The frames encoded and not yet completed (changed under the display mutex).
    pub(crate) in_flight_frame_count: AtomicU32,
    /// Tests only: called once as the end begins, on its thread.
    #[cfg(test)]
    pub(crate) drop_probe: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

/// A display's mode list, shared with its guest display.
pub type ModeList = Arc<[DisplayMode]>;

impl<P: Platform> DisplayCore<P> {
    /// Frees `serial_num` in the process.
    pub fn unreserve_serial_num(platform: &P, serial_num: u32) {
        let mut db = platform.globals().serial_num_db().lock();
        db.retain(|s| *s != serial_num);
    }

    /// Reserves `serial_num` in the process: `false` if a display of any device has it (the set
    /// lives in the platform's `ProcessState`, one per process; testing.md 7.3). A duplicate logs.
    pub fn reserve_serial_num(platform: &P, serial_num: u32) -> bool {
        let mut db = platform.globals().serial_num_db().lock();
        let before = db.len();
        if !db.contains(&serial_num) {
            db.push(serial_num);
        }
        let after = db.len();
        if before >= after {
            log::warn!("display serial number {serial_num:#x} is already in use in this process");
        }
        before < after
    }

    /// The reason for a refusal ([`DisplayRefusal`]: a bad port, no pipelines, a serial number
    /// another display of the process has, a busy port, checked in that order), each after the
    /// partial end ([`Drop`]). Dies for a buffer Metal refuses. Attaches to the port, which may
    /// link the guest display before this returns.
    #[allow(clippy::new_ret_no_self)] // the host's reference, `Display`
    pub fn new(
        platform: &Arc<P>,
        device: Arc<DeviceCore<P>>,
        descriptor: &DisplayDescriptor,
        port: u64,
        serial_num: u32,
    ) -> RaiseResult<Result<Arc<Display<P>>, DisplayRefusal>> {
        let port_count = device.display_port_count();
        if port >= u64::from(port_count) {
            log::warn!("display refused: port {port} is past the device's {port_count} ports");
            return Ok(Err(DisplayRefusal::PortOutOfRange));
        }
        // The initializer's pool: it pops at the body's end. A raise leaves it pushed; a failure
        // releases the display after it.
        let made = pool::around(&**platform, || {
            Self::init_pooled(platform, device, descriptor, port, serial_num)
        })?;
        Ok(match made {
            Ok(display) => Ok(display),
            Err((failed, why)) => {
                drop(failed);
                Err(why)
            }
        })
    }

    /// The initializer's body, inside its pool: `Err((Some(_), why))` is a failed display the
    /// caller releases after it, `Err((None, why))` one already released here.
    #[allow(clippy::type_complexity)] // RaiseResult<Result<made, failed>>
    fn init_pooled(
        platform: &Arc<P>,
        device: Arc<DeviceCore<P>>,
        descriptor: &DisplayDescriptor,
        port: u64,
        serial_num: u32,
    ) -> RaiseResult<Result<Arc<Display<P>>, (Option<Arc<Display<P>>>, DisplayRefusal)>> {
        // The copies; the handlers are new references to the same blocks.
        let mut desc = descriptor.clone();
        let props = SdkProps::take(&mut desc, port, serial_num);
        let device_retain = device.retain_self();
        let mut source = None;
        // No queue: no source, even with a handler.
        if props.queue().is_some() {
            // The display's own reference to the queue, which the unplug releases.
            props.retain_queue();
            let handler = props.new_frame_event_handler();
            if let Some(handler) = handler
                && let Some(q) = props.queue()
            {
                // The source's handler holds the host's handler.
                source = Some(NewFrameSource::new_add(&q, move |_data: u64| handler()));
            }
        }
        // The pipelines hold the display side's one reference to the host's `MTLDevice`.
        let cache_device = device.mtl_device_ref().clone_ref();
        let cache = DisplayPipelineCache::new(cache_device);
        let display = Display {
            core: Arc::new(DisplayCore {
                platform: platform.clone(),
                this: OnceLock::new(),
                reserved_serial_num: AtomicBool::new(false),
                attached: AtomicBool::new(false),
                device: Mutex::new(Some(DeviceRef {
                    retain: device_retain,
                    host: device,
                })),
                compositor_parameters: Mutex::new((
                    desc.config_epoch,
                    desc.origin,
                    desc.scale_factor,
                )),
                descriptor: desc,
                props,
                new_frame_event_source: Mutex::new(source),
                render_pipeline_cache: cache,
                guest_color_space_dirty: AtomicBool::new(false),
                current_frame_encoded: AtomicBool::new(false),
                has_current_frame: AtomicBool::new(false),
                color_matrix_identity: AtomicBool::new(true),
                gamma_identity: AtomicBool::new(true),
                mode_list_responsiveness: AtomicI32::new(0),
                protection_options: AtomicU64::new(0),
                mode_list: Mutex::new(None),
                mode_list_update: Mutex::new(()),
                protection_requirements_unlocked: AtomicU64::new(0),
                guest_color_space_unlocked: Mutex::new(None),
                display_mutex: Mutex::new(DisplayLocked::zeroed()),
                guest_display: Mutex::new(None),
                in_flight_condition: Condvar::new(),
                in_flight_frame_count: AtomicU32::new(0),
                #[cfg(test)]
                drop_probe: Mutex::new(None),
            }),
        };
        // No pipelines: the partial end, here.
        if display.render_pipeline_cache.is_none() {
            drop(display);
            return Ok(Err((None, DisplayRefusal::Metal)));
        }
        // The pre-warm, the buffers and the matrix pass `OnFault::Die`: a raise out of Metal dies
        // at its send, so none comes back here; were one to, it would die here.
        if let Err(raised) = display.build_render_state() {
            raised.fatal()
        }
        if !Self::reserve_serial_num(platform, serial_num) {
            return Ok(Err((Some(Arc::new(display)), DisplayRefusal::SerialInUse)));
        }
        display.reserved_serial_num.store(true, Ordering::SeqCst);
        let display = Arc::new(display);
        let _ = display.this.set(Arc::downgrade(&display));
        // The host's reference exists: the device and the guest display reach the display from here
        // (the attach hands it to the device, which may hand it to the guest display).
        if let Some(d) = display.device() {
            d.register_display(&display);
        }
        // The attach may link the guest display and call `set_guest_display` inside.
        if !display
            .device()
            .is_some_and(|d| d.attach_display(Arc::downgrade(&display), port))
        {
            return Ok(Err((Some(display), DisplayRefusal::PortInUse)));
        }
        display.attached.store(true, Ordering::SeqCst);
        log::info!(
            "display on port {} created, serial number {:#x}, name {}",
            display.port(),
            display.serial_num(),
            name_arg(display.name())
        );
        Ok(Ok(display))
    }

    /// The host's reference, retained. `None` only before it exists or once its last drop
    /// began.
    fn arc(&self) -> Option<Arc<Display<P>>> {
        self.this.get().and_then(Weak::upgrade)
    }

    /// A retain of the host's display, as a block that holds the display takes it. `None` once
    /// the display's last release has begun: a retain then takes nothing, and a weak load of it
    /// is nil.
    pub fn retain_self(&self) -> Option<DisplayHandle<P>> {
        self.arc().map(DisplayHandle::new)
    }

    /// The display as a block it dispatches holds it: retained while it lives; once its last
    /// release has begun (the end waiting in the detach behind a lock a present, a doorbell or
    /// the device's opcode 1 holds), the core, which the block still reaches. `None` only before
    /// the host's reference exists, or once the end has detached the display.
    fn block_self(&self) -> Option<BlockSelf<P>> {
        match self.retain_self() {
            Some(display) => Some(BlockSelf::Display(display)),
            // The device, still held: the end releases it only after it has deregistered.
            None => self
                .this
                .get()
                .and_then(|this| self.device()?.display_core(this))
                .map(BlockSelf::Deallocating),
        }
    }

    /// The display's queue: the descriptor's, until the unplug releases it. A reference of the
    /// caller's.
    pub fn queue(&self) -> Option<Arc<dyn HostQueue>> {
        self.props.queue()
    }

    /// The device, until the end releases it.
    pub fn device(&self) -> Option<Arc<DeviceCore<P>>> {
        self.device.lock().as_ref().map(|d| d.host.clone())
    }

    /// The host's `MTLDevice`, which the pipelines hold. `None` once the end has released them.
    pub fn mtl_device(&self) -> Option<&P::Device> {
        self.render_pipeline_cache.as_ref().map(|c| &c.mtl_device)
    }

    /// Detaches from the port, unreserves the serial number, cancels the newFrame source and
    /// releases the queue; each once, at the display's end. The guest display pointer stays.
    pub fn unplug(&self) {
        if self.attached.load(Ordering::SeqCst) {
            // By identity: the `Weak` also names a display whose last release has begun.
            if let Some(w) = self.this.get()
                && let Some(device) = self.device()
            {
                device.detach_display(w, self.port());
            }
            self.attached.store(false, Ordering::SeqCst);
        }
        if self.reserved_serial_num.load(Ordering::SeqCst) {
            Self::unreserve_serial_num(&self.platform, self.serial_num());
            self.reserved_serial_num.store(false, Ordering::SeqCst);
        }
        if self.props.queue().is_some() {
            let source = self.new_frame_event_source.lock().take();
            if let Some(source) = &source {
                source.cancel();
            }
            // Cancelling frees nothing here: the source's cancel callout frees its handler (the
            // host's handler with it) on the target queue, after any fire already queued and
            // after this display's end has released its own references and logged; the source
            // lets its queue go after that callout (rt/signal.rs).
            drop(source);
            self.props.release_queue();
        }
    }

    /// Stores `modes` and hands them to the guest display; nothing for an empty list. The guest
    /// display's mode-list state machine raises through it, at the raise: only the host's API
    /// entries, with no handler, set a display's modes.
    ///
    /// The list is stored and the guest display read under the display mutex, which orders it
    /// against `set_guest_display`'s hand-over; the guest display's mutex is taken only once the
    /// display mutex is released, since a present holds the two the other way round.
    pub fn set_mode_list(&self, modes: &[DisplayMode]) -> RaiseResult<()> {
        if modes.is_empty() {
            return Ok(());
        }
        let list: ModeList = Arc::from(modes);
        let _update = self.mode_list_update.lock();
        let guest_display = {
            let _g = self.display_mutex.lock();
            let old = self.mode_list.lock().replace(list.clone());
            drop(old);
            self.guest_display.lock().as_ref().and_then(Weak::upgrade)
        };
        if let Some(n) = guest_display {
            n.set_mode_list(OnFault::Die, list)?;
        }
        Ok(())
    }

    /// The modes set last; empty before a list is set.
    pub fn mode_list(&self) -> Vec<DisplayMode> {
        self.mode_list
            .lock()
            .as_deref()
            .map(<[DisplayMode]>::to_vec)
            .unwrap_or_default()
    }

    /// The modes set last, shared (what a new guest display is handed).
    fn shared_mode_list(&self) -> Option<ModeList> {
        self.mode_list.lock().clone()
    }

    /// Takes the guest display (exactly one of the old and new must be nil) after waiting for
    /// frames in flight and dropping the current frame, hands a new guest display the mode list,
    /// then signals the responsiveness. The guest display's mode-list state machine raises through
    /// it (and then the responsiveness is not signalled), as `on` says: a FIFO command's page
    /// change hands the guest display over under a handler, and a display's attach with none.
    pub fn set_guest_display(
        &self,
        on: OnFault,
        guest_display: Option<Weak<GuestDisplay<P>>>,
    ) -> RaiseResult<()> {
        let mut g = self.display_mutex.lock();
        self.wait_for_pending_frames(&mut g);
        // Whether the host encoded the frame is left as it is.
        g.current_frame = None;
        self.has_current_frame.store(false, Ordering::SeqCst);
        let mut slot = self.guest_display.lock();
        crate::fatal_unless!(
            guest_display.is_none() != slot.is_none(),
            DeathKind::GuestDisplaySwapUnbalanced
        );
        *slot = guest_display;
        if let Some(n) = slot.as_ref().and_then(Weak::upgrade)
            && let Some(list) = self.shared_mode_list()
        {
            n.set_mode_list(on, list)?;
        }
        // With the guest display pointer's lock still held.
        self.signal_responsiveness_with(slot.as_ref());
        Ok(())
    }

    /// The guest display, unretained.
    pub fn guest_display(&self) -> Option<Weak<GuestDisplay<P>>> {
        self.guest_display.lock().clone()
    }

    /// From the guest display's doorbell, under the guest display's mutex.
    pub fn update_responsiveness(&self) {
        let _g = self.display_mutex.lock();
        // The guest display pointer read without holding its lock across the call; the cell is a
        // leaf.
        let guest_display = self.guest_display.lock().clone();
        self.signal_responsiveness_with(guest_display.as_ref());
    }

    /// The responsiveness for the guest display now (the callers hold the display mutex).
    pub fn signal_responsiveness(&self) {
        let guest_display = self.guest_display.lock().clone();
        self.signal_responsiveness_with(guest_display.as_ref());
    }

    /// 0 with no guest display, 1 with one, 2 with one that live-resizes.
    fn signal_responsiveness_with(&self, guest_display: Option<&Weak<GuestDisplay<P>>>) {
        match guest_display {
            None => self.mode_list_responsiveness.store(0, Ordering::SeqCst),
            // A guest display pointer whose guest display is gone reads as not live-resizing.
            Some(w) => {
                let live = w
                    .upgrade()
                    .is_some_and(|n| n.compositor_supports_live_resize());
                self.mode_list_responsiveness
                    .store(if live { 2 } else { 1 }, Ordering::SeqCst);
            }
        }
    }

    /// {0xffff, 0xffff} without a guest display; else the guest display's (which asserts while
    /// paused).
    pub fn cursor_position(&self) -> DisplayCoord {
        let guest_display = self.guest_display.lock();
        match guest_display.as_ref().and_then(Weak::upgrade) {
            Some(n) => n.cursor_position(),
            None => DisplayCoord {
                x: u16::MAX,
                y: u16::MAX,
            },
        }
    }

    /// Sets the protection options (logged when they change) and hands them to the guest display,
    /// under the guest display pointer's lock.
    pub fn set_protection_options(&self, protection_options: u64) {
        let guest_display = self.guest_display.lock();
        let old = self.protection_options.load(Ordering::SeqCst);
        if old != protection_options {
            log::info!(
                "display on port {}: protection options {old:#x} changed to {protection_options:#x}",
                self.port()
            );
            self.protection_options
                .store(protection_options, Ordering::SeqCst);
        }
        if let Some(n) = guest_display.as_ref().and_then(Weak::upgrade) {
            n.set_protection_options(protection_options);
        }
    }

    /// The protection options.
    pub fn protection_options(&self) -> u64 {
        self.protection_options.load(Ordering::SeqCst)
    }

    /// The protection requirements presented last, read without a lock.
    pub fn protection_requirements(&self) -> u64 {
        self.protection_requirements_unlocked.load(Ordering::SeqCst)
    }

    /// The guest colour space taken at the last present, read without a lock.
    pub fn guest_color_space(&self) -> Option<Arc<[u8]>> {
        self.guest_color_space_unlocked.lock().clone()
    }

    /// The descriptor's colour space.
    pub fn color_space(&self) -> DisplayColorSpace {
        self.descriptor.color_space
    }

    /// The configuration epoch.
    pub fn config_epoch(&self) -> u64 {
        self.compositor_parameters.lock().0
    }

    /// The origin.
    pub fn origin(&self) -> DisplayCoord {
        self.compositor_parameters.lock().1
    }

    /// The scale factor.
    pub fn scale_factor(&self) -> u8 {
        self.compositor_parameters.lock().2
    }

    /// The connection type.
    pub fn connection_type(&self) -> i32 {
        self.descriptor.connection_type
    }

    /// The display UUID.
    pub fn display_uuid(&self) -> Option<&str> {
        self.descriptor.display_uuid.as_deref()
    }

    /// The minimum luminance.
    pub fn min_nits(&self) -> f32 {
        self.descriptor.min_nits
    }

    /// The maximum luminance.
    pub fn max_nits(&self) -> f32 {
        self.descriptor.max_nits
    }

    /// The maximum SDR luminance.
    pub fn max_sdr_nits(&self) -> f32 {
        self.descriptor.max_sdr_nits
    }

    /// The name the descriptor had.
    pub fn name(&self) -> Option<&DisplayName> {
        self.descriptor.name.as_ref()
    }

    /// The size in millimetres.
    pub fn size_in_millimeters(&self) -> (f64, f64) {
        self.props.size_in_millimeters()
    }

    /// The port.
    pub fn port(&self) -> u64 {
        self.props.port()
    }

    /// The serial number.
    pub fn serial_num(&self) -> u32 {
        self.props.serial_num()
    }

    /// The texture usage a host's encode target needs.
    pub fn minimum_texture_usage(&self) -> u64 {
        MINIMUM_TEXTURE_USAGE
    }

    /// The guest's presents counted so far.
    pub fn guest_present_count(&self) -> u64 {
        self.props.guest_present_count()
    }

    /// The host's encodes of a new frame counted so far.
    pub fn host_present_count(&self) -> u64 {
        self.props.host_present_count()
    }

    /// The mode-list responsiveness (0, 1 or 2).
    pub fn mode_list_responsiveness(&self) -> i32 {
        self.mode_list_responsiveness.load(Ordering::SeqCst)
    }

    /// Whether the guest changed its colour space since the last present.
    pub fn guest_color_space_dirty(&self) -> bool {
        self.guest_color_space_dirty.load(Ordering::SeqCst)
    }

    /// Marks the guest's colour space changed (or not).
    pub fn set_guest_color_space_dirty(&self, dirty: bool) {
        self.guest_color_space_dirty.store(dirty, Ordering::SeqCst);
    }
}

impl<P: Platform> Drop for Display<P> {
    /// The display's end, on the thread of the last release: the unplug (which releases the queue),
    /// then the releases in a fixed order, then the destroyed line. The guest display's client is
    /// unretained, so a present in flight makes the detach wait for the guest display's mutex here,
    /// and never moves the end elsewhere.
    ///
    /// The order: the position and texture buffers before the name; the five handlers before the
    /// mode list (a host destroy callback runs before the device's teardown and the log); the
    /// current frame (its IOSurface use, and the surface when the frame held its last reference),
    /// the gamma texture and the pipelines (with the `MTLDevice` they hold) before the device; the
    /// device's retain and core together, so its whole teardown runs inside this one when it is
    /// the last ([`DeviceRef`]); the property buffer and the guest colour space after it; the log
    /// last.
    fn drop(&mut self) {
        #[cfg(test)]
        if let Some(probe) = self.core.drop_probe.lock().take() {
            probe();
        }
        // A drop: no handler.
        self.core.unplug();
        // A newFrame source the unplug left (its queue was gone then) is never released:
        // resumed, it keeps its handler and queue for good, and the host's handler's release never
        // comes.
        if let Some(source) = self.core.new_frame_event_source.lock().take() {
            keep_forever(source, Kept::HostReleasesNeverCome);
        }
        // Detached: neither the device nor the guest display reaches the display now. Its entry
        // goes before the releases, so its `Weak` of the core does not keep `get_mut` below from
        // releasing them. The device is still held: it is released below.
        if let Some(device) = self.core.device() {
            device.deregister_display(&self.core);
        }
        // A block still running a handler on the host's queue holds the core; what it may be
        // using goes with it then, and only the device is released here regardless.
        if let Some(core) = Arc::get_mut(&mut self.core) {
            let g = core.display_mutex.get_mut();
            drop(g.pos_buf.take());
            drop(g.tex_buf.take());
            g.pipeline = PipelineKey::default();
            drop(core.descriptor.name.take());
        }
        if let Some(core) = Arc::get_mut(&mut self.core) {
            drop(core.descriptor.display_uuid.take());
        }
        self.core.props.release_handlers();
        drop(self.core.mode_list.lock().take());
        if let Some(core) = Arc::get_mut(&mut self.core) {
            let g = core.display_mutex.get_mut();
            // The detach never takes the guest display from the display, so the last frame
            // presented is still held here.
            drop(g.current_frame.take());
            core.has_current_frame.store(false, Ordering::SeqCst);
            drop(g.gamma_table_tex.take());
            drop(core.render_pipeline_cache.take());
        }
        let device = self.core.device.lock().take();
        if let Some(device) = device {
            device.release();
        }
        if let Some(core) = Arc::get_mut(&mut self.core) {
            let g = core.display_mutex.get_mut();
            drop(g.frag_shader_property_buf.take());
            drop(g.guest_color_space.take());
            drop(core.guest_color_space_unlocked.get_mut().take());
        }
        log::info!(
            "display on port {} destroyed, serial number {:#x}",
            self.core.port(),
            self.core.serial_num()
        );
    }
}
