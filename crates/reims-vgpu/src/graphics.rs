//! [`GraphicsDevice`]: the reims-graphics device (PCI 1234:5247), `reims-vgpu-graphics`'s host side
//! with its display shown in a window of the library's, made from a [`GraphicsConfig`].
//!
//! The device's region (BAR 2) is made here and shared: the device takes the guest's commands
//! from its rings, and the window's sink imports the region's pages under the scanout as a
//! no-copy buffer on the Vulkan platform and blits the scanout out of it into each drawable, on
//! the GPU, with no copy on the CPU (`sink`). The buffer's deallocator holds the region, so it stays mapped until
//! the GPU has let the buffer go, after the device and the window.
//!
//! **The window** ([`WindowScreen`]) is a display window of `reims-vgpu-window` that keeps its size
//! (`WindowConfig::keeps_its_size`): the guest follows it. It opens sized for the config's mode,
//! else 1920 × 1080, and the display then offers the guest one mode, the window's size in pixels as
//! its render thread last drew into it, its width rounded down to a multiple of 16 and its height
//! to even, so each row of a guest surface of the mode's width starts and ends on a 64-byte line of
//! the GPU's memory (`geometry::offered_mode`; a window of other pixels shows the guest's mode
//! unscaled, beside up to 15 columns and a row of black), at the refresh rate of the host screen
//! it opened on; when that size changes (the person resized the window) the device tells the guest
//! (`DISPLAY_CHANGED`). The guest's scanout shows in the window at once and then again at each of
//! the display's refreshes, at the screen's rate, while no presented surface shows in its place;
//! the window draws it at the screen's pace. While a surface shows, a refresh draws nothing: the
//! guest's presents draw it ([`Presented`]). The guest's cursor image is the host's cursor over
//! the image, shown while the guest shows its cursor, at the host's pointer, which moves the
//! guest's. The window takes each place the guest puts its cursor's hot spot (`MOVE_CURSOR`):
//! where the guest moves its cursor with no pointer event of the window's just before (a mouse
//! passed through to the guest with QEMU's `usb-host`, or a warp of the guest's own), the window
//! draws the guest's cursor into its image there itself, and the host's cursor is invisible over
//! the image (`reims_vgpu_window::handle`'s module doc; `sink`, "The guest's cursor").
//!
//! **Without a window** the display offers 1920 × 1080 at 60 Hz and shows nothing.
//!
//! **Several displays** ([`GraphicsConfig::display_count`], docs/graphics-kext/design.md,
//! "Multiple displays"): each display has what display 0 has, its own window (display `i`'s on the
//! window crate's port `i`, titled and placed as that port's), or none where display 0 has none,
//! its own presented surface, colour, host's reads and host cursor, and its mode sized to its
//! scanout room (`reims-vgpu-graphics`' `display::offered_in_room`). The presents of all of them
//! share one queue, each display's in its own order ([`PresentQueue`]). A device of one display
//! is what a device was before the count.
//!
//! **The host is told the guest's cursor**, with a window or without one, where it takes it
//! ([`GraphicsCallbacks`]' `cursor_*`, [`HostCursor`]): its image, whether it shows and where the
//! guest put its hot spot, on the display thread. QEMU's console draws it for QEMU's own displays.
//!
//! **The host reads what the display shows** ([`GraphicsDevice::read_display`], `read`): the
//! window's draw, into a texture of the display's mode, read back once it has finished, with a
//! window or without one; QEMU's console shows it, and its screendump saves it.
//!
//! Every public method runs as an entry ([`crate::entry`]).

use std::borrow::Cow;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use metal2vulkan_runtime::host_time;
use parking_lot::lock_api::ArcRwLockReadGuard;
use parking_lot::{RawRwLock, RwLock};
use reims_vgpu_graphics::ops::RgfxCommandBuffer;
use reims_vgpu_graphics::region::Region;
use reims_vgpu_graphics::wire::{
    RGFX_CMD_ARGUMENT_LAYOUT, RGFX_CMD_BUFFER_OVER_PAGES, RGFX_CMD_COMMAND_BUFFER,
    RGFX_CMD_COMPUTE_PIPELINE_CREATE, RGFX_CMD_DYNAMIC_LIBRARY_CREATE,
    RGFX_CMD_FUNCTION_HANDLE_CREATE, RGFX_CMD_FUNCTION_TABLE_CREATE, RGFX_CMD_HEAP_OVER_PAGES,
    RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN, RGFX_CMD_ICB_CREATE, RGFX_CMD_ICB_SLOTS,
    RGFX_CMD_ICB_SLOTS_IN_PAGES, RGFX_CMD_ICB_WRITE, RGFX_CMD_LIBRARY_CREATE,
    RGFX_CMD_LIBRARY_FROM_REGION, RGFX_CMD_LIBRARY_LOOKUP, RGFX_CMD_MESH_RENDER_PIPELINE_CREATE,
    RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_PAGES_BUFFER_ADDRESS, RGFX_CMD_PIPELINE_ANSWERS,
    RGFX_CMD_RATE_MAP_CREATE, RGFX_CMD_RATE_MAP_QUERY, RGFX_CMD_RATE_MAP_RESET,
    RGFX_CMD_RENDER_PIPELINE_CREATE, RGFX_CMD_SAMPLER_CREATE, RGFX_CMD_SHARED_EVENT_IMPORT,
    RGFX_CMD_SHARED_EVENT_NAME, RGFX_CMD_SHARED_EVENT_UNNAME, RGFX_CONTEXT_OBJ_ID_SHIFT,
    RGFX_DEVICE_INFO_MAX_BUFFER_LENGTH, RGFX_DEVICE_INFO_OFFSET, RGFX_DISPLAYS_MAX,
    RGFX_EVT_FUNCTION_POINTER_STATE, RGFX_EVT_ICB_STATE, RGFX_EVT_RATE_MAP_ANSWER,
    RGFX_EVT_RATE_MAP_STATE, RGFX_EVT_SAMPLER_STATE, RGFX_FEATURE_BUFFER_ESCAPED,
    RGFX_FEATURE_CB_SUBMITTED, RGFX_FEATURE_COMMIT_ORDER, RGFX_FEATURE_CURSOR_COLOR,
    RGFX_FEATURE_DISPLAY_COLOR, RGFX_FEATURE_DISPLAY_POWER, RGFX_FEATURE_DISPLAY_SCALE,
    RGFX_FEATURE_DOORBELL_ASKS, RGFX_FEATURE_DOORBELL_PAGES, RGFX_FEATURE_DYNAMIC_LIBRARIES,
    RGFX_FEATURE_FUNCTION_POINTERS, RGFX_FEATURE_GUEST_TIMES, RGFX_FEATURE_HEAP_OVER_PAGES,
    RGFX_FEATURE_HOST_TIME, RGFX_FEATURE_ICB_SLOTS, RGFX_FEATURE_ICB_SLOTS_IN_PAGES,
    RGFX_FEATURE_IMPORT_ADDRESSES, RGFX_FEATURE_IMPORT_ADDRESSES_LATER,
    RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS, RGFX_FEATURE_MANAGED_AT_COPY,
    RGFX_FEATURE_MANAGED_AT_OFFSET, RGFX_FEATURE_MANAGED_BUFFER_SLABS,
    RGFX_FEATURE_MANAGED_TEXTURE_SLABS, RGFX_FEATURE_MESH_PIPELINES, RGFX_FEATURE_OPENCL_MODE,
    RGFX_FEATURE_PAGES_16K, RGFX_FEATURE_PAGES_BUFFERS, RGFX_FEATURE_PIPELINE_ANSWERS,
    RGFX_FEATURE_PIPELINE_KEYS, RGFX_FEATURE_POINTER_MOTION, RGFX_FEATURE_POOL,
    RGFX_FEATURE_POOL_ADDRESSES, RGFX_FEATURE_PRIVATE_BLOCKS, RGFX_FEATURE_PRIVATE_RENDER_STATE,
    RGFX_FEATURE_RATE_MAPS, RGFX_FEATURE_READ_ONLY_PAGES, RGFX_FEATURE_SCHEDULED_ASK,
    RGFX_FEATURE_SHARED_EVENT_NAMES, RGFX_FEATURE_SMALL_ID_CHUNKS, RGFX_FEATURE_SUBMIT_ROOM,
    RGFX_FEATURE_SURFACE_CACHE, RGFX_FEATURE_SURFACE_CACHE_EVERY,
    RGFX_FEATURE_SURFACE_CACHE_REFUSED, RGFX_FEATURE_SURFACE_DEFERRED,
    RGFX_FEATURE_SURFACE_DEFERRED_ASK, RGFX_FEATURE_SURFACE_DEFERRED_CARRY,
    RGFX_FEATURE_SURFACE_ORDER, RGFX_FEATURE_SURFACE_PAGES, RGFX_FEATURE_SURFACE_REFILL,
    RGFX_FEATURE_TEXTURE_DID_MODIFY_DATA, RGFX_FEATURE_UNCHANGING_PAGES,
    RGFX_FEATURE_UNSHARED_QUEUES, RGFX_FEATURE_UPLOADS_WRITTEN, RGFX_FEATURE_VT_MESSAGES,
    RGFX_RESOURCE_ID_CHUNK, RGFX_RESOURCE_ID_CHUNKS, RGFX_STATUS_INVALID_ARGUMENT,
    RGFX_STATUS_NOT_FOUND, RGFX_STATUS_OK, RGFX_STATUS_REFUSED, RGFX_VIDEO_CAPS_CODECS,
    RGFX_VIDEO_CAPS_OFFSET, RGFX_VIDEO_CAPS_OUTPUT_CACHED, RGFX_VIDEO_CAPS_VERSION,
    RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_DECODE, RGFX_VIDEO_H264_BASELINE,
    RGFX_VIDEO_H264_HIGH, RGFX_VIDEO_H264_MAIN, RGFX_VIDEO_HEVC_MAIN, RGFX_VIDEO_HEVC_MAIN_10,
    RGFX_VT_STATUS_MALFORMED, RgfxBlockGrant, RgfxBlockRelease, RgfxBufferOverPages,
    RgfxCbComplete, RgfxCbScheduled, RgfxCbSubmitted, RgfxColorEntry, RgfxDisplayColor,
    RgfxEventSignalled, RgfxFunctionPointerState, RgfxHeapOverPages, RgfxIcbState,
    RgfxLibraryLookup, RgfxMsgHeader, RgfxPageRun, RgfxPipelineAnswersQuery, RgfxPoolGive,
    RgfxPoolTake, RgfxPrivateGive, RgfxPrivateTake, RgfxRateMapAnswer, RgfxRateMapState,
    RgfxSamplerState, RgfxScheduledAsk, RgfxSharedEventName, RgfxSurfacePages,
    RgfxSurfacePagesLayout, RgfxVideoCaps,
};
use reims_vgpu_graphics::{
    CompleteAnswer, CompletionPoster, CursorImage, DoorbellAt, Host, Mode, PoolAnswer,
    PresentAnswer, PrivateAnswer, QueueClaim, RawDoorbell, RingDegradation, Scanout, Screen,
    VtAnswer, VtLink, VtMessage, VtSurface,
};
use reims_vgpu_video::paravirt;
use reims_vgpu_vulkan::core::host::HostQueue;
use reims_vgpu_vulkan::core::memory::map::MemoryMap as CoreMemoryMap;
use reims_vgpu_vulkan::core::num::ToU64;
use reims_vgpu_vulkan::core::platform::command::CommandBuffer as _;
use reims_vgpu_vulkan::core::platform::device::GpuDevice;
use reims_vgpu_vulkan::core::platform::os::clock::Clock as _;
use reims_vgpu_vulkan::core::platform::resource::Texture;
use reims_vgpu_vulkan::core::platform::stage::{self, FillEnd};
use reims_vgpu_vulkan::core::platform::values::TextureDesc;
use reims_vgpu_vulkan::core::platform::{Handle, Platform};
use reims_vgpu_vulkan::core::raise::{OnFault, Raise, RaiseResult};
use reims_vgpu_vulkan::core::sync::Mutex;
use reims_vgpu_window::{FrameSink, Window, WindowConfig, geometry};
use zerocopy::{FromBytes, IntoBytes};

use crate::DefaultPlatform;
use crate::cursor::{
    CursorGlyph, CursorGlyphAtFn, CursorGlyphFn, CursorMoveAtFn, CursorMoveFn, CursorShowAtFn,
    CursorShowFn,
};
use crate::entry;
use crate::entry::no_catch;
use crate::error::Error;
use crate::graphics::answers::PipelineAnswers;
use crate::graphics::color::{ColorChange, ColorStaging, DisplayColor};
use crate::graphics::compile::{CompileKind, CompileOf, CompilePool, Pending};
use crate::graphics::completion::cb_failed;
use crate::graphics::decode::prepass::{Ahead, Prepass, WalkEnd};
use crate::graphics::decode::shares::ListenSink;
use crate::graphics::decode::worker::{Job, Settle, Unsettled, Worker};
use crate::graphics::decode::{
    AheadSinks, Begun, CompiledPipeline, CompletionSink, DecodeError, Decoders, DrainAnswer,
    FinishedSink, GraphicsDecoder, Joined, LibraryJob, LibraryStart, MadeLibrary, PipelineJob,
    PipelineKey, PipelineReply, ReflectionSink, ScheduledSink, SignalSink, Sinks, SubmittedSink,
    SurfacePresent, Then, compile_refusing, complete, make_library, refused_reply, refusing,
};
use crate::graphics::device_info::device_info;
use crate::graphics::present::{Drawing, LetGo, PresentAnswers, PresentQueue};
use crate::graphics::read::DisplayRead;
use crate::graphics::slow_answers::{
    CloseFrees, CloseMarks, Freed, ReleaseMarks, Split, answering, writing_back,
};
use crate::memory::{GuestMemoryLease, MemoryMap};
use crate::platform::{ApiPlatform, api_state};
use crate::pool::{
    Give, HostBlocks, Pool, PoolAddFn, PoolConfig, PoolHost, PoolLedger, PoolRemoveFn,
    PrivateBlocks, PrivateGive, PrivateTakeAnswer, PrivateTakeError, TAKE_MAX,
};
use crate::window::WindowLink;

mod answers;
mod color;
mod compile;
mod completion;
mod decode;
mod device_info;
#[cfg(windows)]
mod drawn;
mod fault_walk;
#[cfg(target_os = "linux")]
mod nvidia;
mod present;
mod read;
mod rt_window;
mod sink;
mod slow_answers;
mod summary;
#[cfg(any(target_os = "linux", windows))]
mod vaapi;
mod video_choice;
mod video_host;
mod vt;
#[cfg(any(target_os = "linux", windows))]
mod vt_encode;

pub(crate) use video_choice::rt_graphics_video;

pub use decode::{GraphicsVideo, GraphicsVideoSession, PlanePages, VideoDecode, VideoPlane};

/// Raises MSI-X vector `vector`: the device's interrupt.
pub type GraphicsRaiseFn = Arc<dyn Fn(u32) + Send + Sync>;

/// `(route, ring_id, bell)`: with `Some(bell)`, routes a command queue's plain doorbell, a
/// `DOORBELL` write of `ring_id`, into the file `bell` (an eventfd the queue's thread sleeps on)
/// where the host takes such a write with no thread of its running it (QEMU: a KVM
/// `ioeventfd`), as route number `route`, and answers whether it took it; with `None`, drops
/// route `route` (of `ring_id`), whose queue ended, and its answer is not read: once the call
/// returns no write reaches `bell`, as the ring may be a later queue's. A route of a `ring_id`
/// another route holds replaces that one. The host takes a copy of `bell` and may take the
/// route after the call; until then, and where it refuses it, the writes come through
/// [`GraphicsDevice::mmio_write_taking`]. The device offers `RGFX_FEATURE_DOORBELL_ASKS` only
/// where the host gives this function, on Linux, and routes only where the guest's driver took
/// it, at most `reims_vgpu_graphics::device::MAX_DOORBELL_ROUTES` at once, and drops none at its
/// end: those it holds then are the host's to drop. Called with no lock held that a function of
/// the device's waits for: on one of ring 0's threads (`reims-vgpu.graphics-ring0-<i>`) or the
/// control thread (`reims-vgpu.graphics-control`) as a queue is made, destroyed or its process
/// closed, where
/// it may wait for a lock the host holds while it calls the device (QEMU's BQL); and with
/// `None` also on the host's own inside a `STATUS` 0 write or [`GraphicsDevice::reset`], where
/// it must not wait for a lock the host holds around that call.
pub type GraphicsRouteDoorbellFn = Arc<dyn Fn(u64, u32, Option<RawDoorbell>) -> bool + Send + Sync>;

/// `(route, page, ring_id, bell)`: [`GraphicsRouteDoorbellFn`], for a write of `ring_id` at the
/// first byte of doorbell page `page` in BAR 4 (`page * RGFX_DOORBELL_PAGE_SIZE`) in place of
/// `DOORBELL`: the route of a queue whose context has that page, which only a host that maps BAR 4
/// ([`GraphicsConfig::doorbell_pages`]) is asked for. Until it takes the route, and where it
/// refuses it, the writes come through [`GraphicsDevice::doorbell_write`]. A route of a `ring_id`
/// at `page` another route holds replaces that one. Called as [`GraphicsRouteDoorbellFn`] is.
pub type GraphicsRouteDoorbellPageFn =
    Arc<dyn Fn(u64, u32, u32, Option<RawDoorbell>) -> bool + Send + Sync>;

/// Whether the device offers `DOORBELL_ASKS`, on which it routes its queues' plain doorbells to
/// the host ([`GraphicsCallbacks::route_doorbell`]): where the host gives the route function, on
/// Linux, whose eventfds the queues' threads sleep on and KVM's `ioeventfd` writes. A host on
/// macOS (QEMU under HVF or TCG, with no `ioeventfd` of the kernel's), and one with no route
/// function, is not offered it: its guest rings every doorbell as before the bit, every write
/// runs through [`GraphicsDevice::mmio_write_taking`], and nothing changes for the guest.
fn offers_doorbell_asks(callbacks: &GraphicsCallbacks) -> bool {
    cfg!(target_os = "linux") && callbacks.route_doorbell.is_some()
}

/// A graphics device's callbacks. `raise_interrupt` is called on the thread that posts the event
/// it raises for: the device's control thread (`reims-vgpu.graphics-control`), one of ring 0's
/// threads (`reims-vgpu.graphics-ring0-<i>`), which carry out a guest process's control messages,
/// its presents and a block's pages and downloads, each on its lane, a display's refresh thread, a
/// command queue's service thread (`reims-vgpu.graphics-queue-<id>`) or one of its lane's threads
/// (`reims-vgpu.graphics-lane-<id>`), a pipeline compile's thread
/// (`reims-vgpu.graphics-compile-<n>`), the surface worker (`reims-vgpu.graphics-surfaces-<i>`),
/// the indirect command buffers' worker (`reims-vgpu.icb-create-<i>`), the worker that lets go of
/// a closed process's objects and answers its close (`reims-vgpu.graphics-closes-<i>`), or the
/// platform's thread that runs a command buffer's scheduled or completed handler; or the host's
/// own, inside [`GraphicsDevice::mmio_write_taking`].
/// It is called never inside another call of the host's into the device, and with no lock held
/// that a function of the device's waits for, directly or through a thread of the library's, so
/// it may call it at once. It may run under these: a command queue's claim on its next message,
/// on the queue's service thread and inside `mmio_write_taking`; there, at a command buffer's
/// `CB_SUBMITTED` answered on the calling thread, the decoder's contexts' lock, the buffer's
/// context's table read, its queue's answer slot and the reset gate read; the device caches'
/// lock, with the download's answer slot, at a `SURFACE_DOWNLOAD` answered at once; a command
/// buffer's `CB_SUBMITTED` latch, wherever that is posted; and the presents' queue, at a present
/// answered at once with no window. No function of the device takes any of them, and
/// `mmio_write_taking` only tries the claim and the decoder's locks.
///
/// The cursor's three run on the process's display thread (`reims-vgpu.display`, the
/// `platform` module's), one at a time with every display callback of the process, in the order
/// the device took the guest's cursor commands and resets (`HostCursor`): never inside a call of
/// the host's, and the device never waits for them. The guest sees nothing of whether the host
/// takes them. They are display 0's; the `_at` three are those of displays 1 and up, which name
/// the display, called as they are, in order with them.
#[derive(Clone, Default)]
pub struct GraphicsCallbacks {
    /// Raises the device's interrupt. Without it the guest gets no interrupt.
    pub raise_interrupt: Option<GraphicsRaiseFn>,
    /// The guest set its cursor's image: its pixels as straight-alpha RGBA, which the guest
    /// gave premultiplied BGRA (`reims-vgpu-graphics`' `display`), and its hot spot.
    pub cursor_glyph: Option<CursorGlyphFn>,
    /// Whether the cursor shows changed, hidden until the first call: 1 once the guest has an
    /// image while it shows its cursor, 0 once it has none or hides it.
    pub cursor_show: Option<CursorShowFn>,
    /// The guest moved its cursor: [`GraphicsDevice::cursor_position`] reads where.
    pub cursor_move: Option<CursorMoveFn>,
    /// `cursor_glyph` of display 1 and up, `(display, glyph)`.
    pub cursor_glyph_at: Option<CursorGlyphAtFn>,
    /// `cursor_show` of display 1 and up, `(display, show)`.
    pub cursor_show_at: Option<CursorShowAtFn>,
    /// `cursor_move` of display 1 and up, `(display)`: [`GraphicsDevice::cursor_position_at`]
    /// reads where.
    pub cursor_move_at: Option<CursorMoveAtFn>,
    /// Routes a command queue's plain doorbell into its bell, or drops the route. Without it every
    /// doorbell write comes through [`GraphicsDevice::mmio_write_taking`].
    pub route_doorbell: Option<GraphicsRouteDoorbellFn>,
    /// Routes the plain doorbell of a command queue whose context has a doorbell page into its
    /// bell, at the page, or drops the route. Without it such a queue is not routed, and its
    /// writes come through [`GraphicsDevice::doorbell_write`].
    pub route_doorbell_page: Option<GraphicsRouteDoorbellPageFn>,
    /// Maps a piece of the guest's Shared pool into the guest ([`PoolAddFn`]); needed with
    /// [`GraphicsConfig::pool`], on the pool's thread.
    pub pool_add: Option<PoolAddFn>,
    /// Unmaps a piece of the pool ([`PoolRemoveFn`]); needed with [`GraphicsConfig::pool`].
    pub pool_remove: Option<PoolRemoveFn>,
}

/// A graphics device's config.
#[derive(Clone, Default)]
pub struct GraphicsConfig {
    /// Its callbacks.
    pub callbacks: GraphicsCallbacks,
    /// The display's window: its input callbacks and the mode it opens sized for; `None` is no
    /// window. The window keeps its size whatever this says (the module doc). Display 0's on a
    /// device of several displays.
    pub window: Option<WindowConfig>,
    /// How many displays the device has: 0 and 1 are one display, as a config made before the
    /// field has; 2 to `RGFX_DISPLAYS_MAX` (4) make a device of that many, which offers the
    /// guest `RGFX_FEATURE_MULTI_DISPLAY` and `RGFX_FEATURE_POINTER_OVER`. Any other count is
    /// [`Error::InvalidArgument`].
    pub display_count: u32,
    /// The windows of displays 1 to `display_count - 1`, given exactly where
    /// [`Self::window`] is: `display_count - 1` of them with it, none without it; otherwise
    /// [`Error::InvalidArgument`].
    pub display_windows: Vec<WindowConfig>,
    /// The guest's RAM, with the files behind it, through which the GPU uses an IOSurface's own
    /// pages in place (`RGFX_FEATURE_SURFACE_PAGES`, where the host's page is the guest's 4 KiB, or
    /// 16 KiB for a guest that takes `RGFX_FEATURE_PAGES_16K`); `None` keeps each surface's GPU
    /// view a copy of its pages in the region. On a GPU that shares the CPU's memory the device
    /// answers `recommendedMaxWorkingSetSize` from the RAM the map holds (`device_info`'s
    /// `lane_working_set`). The pages stay the guest's RAM, mapped, while the device lives and
    /// after it, as [`MemoryMap::push_with_file`] says: a host that discards guest RAM (a
    /// balloon, hot-unplug) gives none.
    pub memory_map: Option<MemoryMap>,
    /// The guest's architecture, which the device's answer of `isFramebufferReadSupported` follows
    /// (`device_info`'s `lane_flags`).
    pub guest_arch: GuestArch,
    /// The host maps BAR 4, the doorbell pages (`RGFX_PCI_BAR_DOORBELLS`, 32-bit and not
    /// prefetchable, `RGFX_DOORBELLS_SIZE` bytes), and hands each of its writes to
    /// [`GraphicsDevice::doorbell_write`], reading 0 for every read: the device then offers
    /// `RGFX_FEATURE_DOORBELL_PAGES`, on which the kext maps each process's own page, which it
    /// rings its queues' doorbells on with no call into the kext.
    pub doorbell_pages: bool,
    /// The guest physical addresses the host can give the guest's Shared pool, with no RAM
    /// under them, which the device fills with blocks as the pool grows (`crate::pool`); `None`
    /// is no pool. The device takes from their start as many as its GPU's memory heaps hold,
    /// every heap summed ([`PoolConfig::sized`]), at most all of them, and none where the GPU
    /// answers no heap: [`GraphicsDevice::pool_range`] answers the range it took, which the host
    /// keeps free for the device's life. Given, it needs [`GraphicsCallbacks::pool_add`] and
    /// `pool_remove`, a range [`PoolConfig::is_valid`] takes, and none of
    /// [`Self::memory_map`]'s ranges overlapping it: otherwise [`Error::InvalidArgument`].
    pub pool: Option<PoolConfig>,
}

/// The architecture of the macOS guest a graphics device serves ([`GraphicsConfig::guest_arch`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GuestArch {
    /// An x86-64 guest, whose frameworks were written against Intel and AMD Macs' GPUs.
    #[default]
    X86_64,
    /// An arm64 guest, whose frameworks were written against Apple silicon's GPUs.
    Arm64,
}

/// The bytes of the host GPU's memory deferred blocks' device caches may hold, which
/// `DEFERRED_BUDGET_LO` and `_HI` read, of a device offering `features` on a GPU whose own memory's
/// budget is `own_memory` ([`GpuDevice::own_memory_budget`]): a quarter of it, as the wire gives
/// it (`RGFX_REG_DEFERRED_BUDGET_LO`), whether the GPU shares the CPU's memory or not; 0 without
/// `SURFACE_DEFERRED` or `SURFACE_DEFERRED_ASK`.
fn deferred_budget(features: u64, own_memory: u64) -> u64 {
    let deferring =
        u64::from(RGFX_FEATURE_SURFACE_DEFERRED) | u64::from(RGFX_FEATURE_SURFACE_DEFERRED_ASK);
    if features & deferring == 0 {
        return 0;
    }
    own_memory / 4
}

/// The bytes every IOSurface device cache may hold together (`SurfaceCaches::set_cache_budget`),
/// of a GPU whose own memory's budget is `own_memory` ([`GpuDevice::own_memory_budget`], read
/// once at the device's start, past the guest's RAM where QEMU preallocated it): half of it where
/// the GPU shares the CPU's memory (`unified`), two thirds of it where the GPU has memory of its
/// own.
///
/// Where the GPU shares the CPU's memory the caches are host RAM beside the guest's pages. A
/// quarter, the deferred blocks' budget, held Safari's and a WebGL page's surfaces in place on a
/// Radeon 780M with an 8 GiB guest (1385 MiB, 454 blocks past it), and its WebGL frames fell from
/// 33.0 to 24.6 a second; with no bound, the caches held 1762 MiB at most (jobs m3-max-38033 and
/// 38034).
///
/// Where the GPU has memory of its own the caches are that memory, beside everything else the
/// platform places there for the guest. In the per-shape form the budget trims only clean shapes no
/// texture references, never a dirty or plane one (`decode::surface_cache`, "Within a budget"): the
/// caches' buffers and their other shapes are a floor it does not go under. On a 4 GiB Radeon RX
/// 550 (RADV) with no budget, QEMU asked the card for more than it has in a browser's 3D page (by
/// 201-270 MiB) and in a video (by up to 1283 MiB, most of it the host's video decoder's), the
/// caches' buffers and referenced shapes holding 2912-3634 MiB at each such moment (jobs
/// m3-max-63976 and 64138). Two thirds of the card, about 2.7 GB, is under that, so there the
/// budget trims every shape it may; in those walks it would have trimmed elsewhere only where QEMU
/// asked within 415 MiB of the card's room, where half would have trimmed in a scroll that asked
/// 2.2 GB of it too. With it, a walk trimmed 193 MiB, and the 3D page still asked 169-203 MiB past
/// the card: its 462 MiB of unreferenced shapes were dirty or planes, which stay until a copy out
/// writes them back (job m3-max-64350).
fn cache_budget(own_memory: u64, unified: bool) -> u64 {
    if unified {
        own_memory / 2
    } else {
        own_memory / 3 * 2
    }
}

/// The mode a display offers before its window has a size, and without a window: no screen
/// shows it, so it has no scale.
const FALLBACK_MODE: Mode = Mode {
    width: 1920,
    height: 1080,
    refresh_mhz: 60_000,
    scale_120ths: 0,
};

/// The displays `config` makes ([`GraphicsConfig::display_count`]): 0 and 1 are one;
/// [`Error::InvalidArgument`] for a count past `RGFX_DISPLAYS_MAX`, or display windows that are
/// not one for each display from 1 where display 0 has a window, and none where it has none.
fn display_count(config: &GraphicsConfig) -> Result<usize, Error> {
    let count = config.display_count.max(1);
    if count > RGFX_DISPLAYS_MAX {
        return Err(Error::InvalidArgument);
    }
    let count = count as usize;
    let windows = if config.window.is_some() {
        count - 1
    } else {
        0
    };
    if config.display_windows.len() != windows {
        return Err(Error::InvalidArgument);
    }
    Ok(count)
}

/// The pool `config` gives, with the host's side of it: `None` without one;
/// [`Error::InvalidArgument`] for a range [`PoolConfig::is_valid`] refuses, one a range of the
/// memory map overlaps, or one without both of the host's pool callbacks.
fn pool_host(config: &GraphicsConfig) -> Result<Option<(PoolConfig, PoolHost)>, Error> {
    let Some(range) = config.pool else {
        return Ok(None);
    };
    let (Some(add), Some(remove)) = (&config.callbacks.pool_add, &config.callbacks.pool_remove)
    else {
        return Err(Error::InvalidArgument);
    };
    let mut ram = (config.memory_map.iter()).flat_map(|m| m.ranges());
    if !range.is_valid() || ram.any(|r| range.overlaps(r.physical_address, r.physical_length)) {
        return Err(Error::InvalidArgument);
    }
    let host = PoolHost {
        add: Arc::clone(add),
        remove: Arc::clone(remove),
    };
    Ok(Some((range, host)))
}

/// The range the device's pool takes of `room`, the addresses the host can give it, on a GPU
/// whose memory heaps hold `heaps` bytes, every heap summed ([`PoolConfig::sized`]), logged once
/// a device, with the room's size: with a warning where the GPU answers no heap, and so no pool,
/// or where the room holds less than the heaps, and the range stops at the room's end.
fn pool_range(room: PoolConfig, heaps: u64) -> Option<PoolConfig> {
    let Some(range) = room.sized(heaps) else {
        log::warn!("pool: the GPU answers no memory heap: the guest has no Shared pool");
        return None;
    };
    if range.physical_length < heaps {
        log::warn!(
            "pool: the guest's {} MiB of addresses at {:#x} hold less than the GPU's {} MiB of \
             memory heaps: the pool's range stops there",
            room.physical_length >> 20,
            room.physical_address,
            heaps >> 20
        );
    }
    log::info!(
        "pool: {} MiB of guest addresses at {:#x}, of {} MiB the host can give, for the GPU's {} \
         MiB of memory heaps",
        range.physical_length >> 20,
        range.physical_address,
        room.physical_length >> 20,
        heaps >> 20
    );
    Some(range)
}

/// The reims-graphics device.
pub struct GraphicsDevice<P: ApiPlatform = DefaultPlatform> {
    /// The displays' windows, display `i`'s at index `i`, none without them, which close when
    /// their links drop: before the device.
    windows: Vec<WindowLink>,
    /// The device.
    device: reims_vgpu_graphics::Device,
    /// Each display's guest cursor as the host is told it, display `i`'s at index `i`.
    cursors: Vec<Arc<HostCursor>>,
    /// The stage-3 decoder and the device handle it drives, shared with the device's [`Host`],
    /// which feeds it the control ring's stage-3 messages; held here for the tests' look at it.
    #[cfg(any(test, feature = "testkit"))]
    control: Arc<GraphicsControl<P>>,
    /// What display 0's window shows, where there is one: the tests draw it.
    #[cfg(test)]
    shown: Option<Arc<Shown>>,
    /// The host's reads of what each display shows ([`GraphicsDevice::read_display_at`]),
    /// display `i`'s at index `i`.
    reads: Vec<DisplayRead<P>>,
    /// The stage-3 decoder's state while it lives: once the device ended, what holds the GPU
    /// device until its threads, its objects and their GPU work let go ([`ENDING`]).
    ending: Weak<GraphicsControl<P>>,
    /// The guest's Shared pool, where the host reserved one.
    pool: Option<Pool>,
    /// The platform the decoder runs on.
    platform: PhantomData<fn() -> P>,
}

/// How many graphics devices live in the process, of every platform: made and not dropped
/// ([`GraphicsDevice::let_go_of_gpu`]).
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// The decoder states of the graphics devices that ended, while one lives
/// ([`GraphicsDevice::let_go_of_gpu`]'s warning).
static ENDING: Mutex<Vec<Weak<dyn std::any::Any + Send + Sync>>> = Mutex::new(Vec::new());

/// How long the C API's release of the process's last graphics device waits for the device's
/// threads, objects and GPU work to let go of the GPU device ([`GraphicsDevice::let_go_of_gpu`]).
pub const LET_GO_BOUND: std::time::Duration = std::time::Duration::from_secs(5);

impl<P: ApiPlatform> std::fmt::Debug for GraphicsDevice<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphicsDevice")
            .field("device", &self.device)
            .field("windows", &self.windows.len())
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
    /// refuses the region's memory, a pipe or a thread, or has no GPU device the platform can use
    /// (none, or one its driver refused).
    pub fn with_platform(platform: Arc<P>, config: GraphicsConfig) -> Result<Self, Error> {
        entry::run(|_| {
            let count = display_count(&config)?;
            let pool_host = pool_host(&config)?;
            let size = usize::try_from(reims_vgpu_graphics::REGION_SIZE)
                .map_err(|_| Error::HostResources)?;
            let region = Arc::new(Region::new(size).ok_or(Error::HostResources)?);
            // The device the decoder drives, and its caps blob, which the guest bundle reads
            // before it makes any queue. A host with no GPU device cannot run the lane.
            let gpu_device = platform.metal_device(None).ok_or(Error::HostResources)?;
            // Before anything else of this device makes a queue or a texture on it.
            reserve_guest_texture_ids(platform.as_ref(), &gpu_device);
            let clock = Arc::clone(&platform);
            let pool = CompilePool::new(Box::new(move || clock.clock().now_mono()));
            let identity = gpu_device.identity_caps();
            // The GPU's memory heaps, summed, which size the guest's Shared pool's range below.
            let memory_heaps = gpu_device.memory_heaps_total();
            // The source of the guest's pool's blocks and of its contexts' Private blocks: the
            // platform's where it gives blocks (metal2vulkan-runtime's), else the process's own
            // memory for the pool and the host's picks of Private addresses.
            let block_source = platform.block_source(&gpu_device);
            // The guest's Shared pool, its blocks the source's. Its range is what the GPU's
            // memory heaps hold, of the addresses the host can give.
            let shared_pool = match pool_host {
                Some((room, host)) => match pool_range(room, memory_heaps) {
                    Some(range) => {
                        let clock = Arc::clone(&platform);
                        let now = Box::new(move || clock.clock().now_mono());
                        let source =
                            (block_source.clone()).unwrap_or_else(|| Arc::new(HostBlocks::new()));
                        Some(Pool::new(range, host, now, source)?)
                    }
                    None => None,
                },
                None => None,
            };
            // The guest's processes take blocks of it for their buffers' slabs (`POOL`). An
            // arm64 guest's processes use the blocks write-back, as an x86-64 guest's do: arm64
            // boots under TCG and HVF ran on them with no fault (docs/graphics-kext/design.md,
            // "The Shared pool"). A Windows host offers it to an x86-64 guest alone, whose cost
            // under WHP was measured beside the region's (docs/qemu.md, "Windows hosts: the
            // Shared pool"): an arm64 guest runs under TCG there, where it was not measured.
            let pool_ledger = (shared_pool.as_ref())
                .filter(|_| !cfg!(windows) || config.guest_arch == GuestArch::X86_64)
                .map(Pool::ledger);
            // `Managed` resources are one copy where the platform answers that the GPU shares the
            // CPU's memory, else two, which hold Metal's contract on any GPU: the guest's device
            // answers `hasUnifiedMemory` alike, from the blob.
            let unified_memory = identity.flags.has_unified_memory.unwrap_or(false);
            let meshes = device_info::makes_mesh_pipelines(&identity);
            let makes_rate_maps = device_info::makes_rate_maps(&identity);
            // The guest's RAM, as much of it as the host maps: the working set a GPU that shares
            // the CPU's memory answers follows it (`device_info`'s `lane_working_set`).
            let guest_ram = (config.memory_map.as_ref())
                .map(|map| map.ranges().iter().map(|r| r.physical_length).sum::<u64>())
                .filter(|&bytes| bytes > 0);
            let mut info = device_info(
                &gpu_device.caps(),
                identity,
                pool.threads().to_u64(),
                config.guest_arch,
                guest_ram,
            );
            let mut decoder = GraphicsDecoder::new(Arc::clone(&platform), Arc::clone(&region));
            // The device walks the decoder's tables at its loss, placing the driver's fault
            // among their live objects (`fault_walk`).
            fault_walk::watch::<P>(&gpu_device, decoder.table_walk());
            decoder.set_unified_memory(unified_memory);
            if let Some(ledger) = &pool_ledger {
                decoder.set_pool(ledger.clone());
            }
            // The guest's `maxBufferLength`, past which the decoder refuses a buffer over a
            // process's own pages, whose length the guest writes.
            decoder.set_max_buffer_length(info.sizes[RGFX_DEVICE_INFO_MAX_BUFFER_LENGTH]);
            // The platform's lease on the guest's RAM, held by the views of the surfaces' pages
            // made through the map and by the deallocators of the platform's imports of its
            // bytes, so it outlives the last of them.
            if let Some(map) = &config.memory_map {
                let lease: GuestMemoryLease = Arc::from(platform.guest_memory(map));
                let memory = CoreMemoryMap::new(Arc::clone(&platform), &map.to_core())
                    .ok_or(Error::InvalidArgument)?;
                decoder.set_guest_memory(memory, lease);
            }
            // Every host keeps IOSurfaces' device caches where the kext asks (`SURFACE_CACHE`), as
            // a Mac's GPU keeps its own copy of each surface it uses, synced with the pages at the
            // CPU's locks, within a budget of the GPU's memory (`decode::surface_cache`, "Within a
            // budget"). A GPU of its own memory places its `Managed` buffers' CPU copies in the
            // guest's heaps. Every host answers a
            // command buffer's ask for its submission, and a waiting thread's for its scheduling
            // (`SCHEDULED_ASK`), orders command buffers of different queues
            // over the IOSurfaces they share where the guest names the order, commits a process's
            // command buffers across its queues in the order it names, makes a queue the
            // kext asks unshared (WindowServer's) on the platform's unshared queue, and tells a
            // guest producer waiting for room on a submission ring that the device took from it
            // (the device carries SUBMIT_ROOM out). The platform makes indirect command buffers on
            // every host, so every host decodes their messages, slots included. Every host reads a
            // `Managed` texture's or buffer's CPU copy where its Shared buffer's contents lie, so
            // the guest may lay the copies in its slabs. Every host takes the texture resource ids'
            // grants in small chunks (`RESOURCE_ID_SMALL_CHUNK`). Every host lets the guest read
            // the clock its VBLANKs are stamped on (the device answers HOST_TIME). A host that maps
            // the guest's pages maps a process's own pages too, for its no-copy buffers
            // (`BUFFER_OVER_PAGES`), the heaps of its buffers' slabs (`HEAP_OVER_PAGES`) and its
            // indirect command buffers' slots (`ICB_SLOTS_IN_PAGES`), and the pages it maps
            // read-only, which it keeps every object over from writing (`READ_ONLY_PAGES`), those
            // of them no one writes read once (`UNCHANGING_PAGES`). Every
            // host carries the display pipe's colour out (`DISPLAY_COLOR`), its window drawing
            // through it, names the shared events one process hands another
            // (`SHARED_EVENT_NAMES`), and tells the guest its window's screen's scale
            // (`DISPLAY_SCALE`). A host whose GPU answers Metal3 makes mesh render pipelines
            // and draws them (`MESH_PIPELINES`). A host whose device makes compute pipelines of
            // OpenCL mode links an OpenCL kernel's function the bundle names so (`OPENCL_MODE`). A
            // host whose device takes the private render state Apple's OpenGL renderer sets makes
            // the guest's render pipelines with it (`PRIVATE_RENDER_STATE`). Every host carries a
            // texture's `didModifyData` to its platform (`TEXTURE_DID_MODIFY_DATA`), shows black
            // while the guest's display sleeps (`DISPLAY_POWER`), and makes its compute pipeline
            // states' function handles and visible function tables (`FUNCTION_POINTERS`). A host
            // whose platform makes rasterization rate maps makes the guest's on it (`RATE_MAPS`),
            // of the layer counts the device info answers ([`device_info::makes_rate_maps`]): rt
            // makes none. A host that routes queues' doorbells takes the doorbell
            // of a buffer that asks for its `CB_SUBMITTED` apart from the rest (`DOORBELL_ASKS`;
            // [`offers_doorbell_asks`]), so it may take the rest where it runs nothing of the
            // device's. Every host takes the guest's stamps of its command buffers
            // (`GUEST_TIMES`), a diagnostic of the time before it takes each. A host that maps
            // BAR 4 has each process ring its queues on a doorbell page of its own
            // (`DOORBELL_PAGES`). Every host hands the kext the answers it kept of keyed
            // pipeline creates from earlier runs (`PIPELINE_ANSWERS`, `answers`), and takes the
            // bytes a command buffer wrote in its `Managed` chunks of `set*Bytes:` data from the
            // buffer's own stream (`UPLOADS_WRITTEN`, `decode::managed`). Every host shows the
            // guest's cursor converted through the colour chain the guest gives it
            // (`CURSOR_COLOR`), which the device applies before the screen is handed the image.
            // Every host takes the guest's note that a no-copy buffer's identity left its Metal
            // with no message naming the buffer (`BUFFER_ESCAPED`, `decode::escape`), which it
            // marks and counts. A host with the guest's Shared pool lets its processes take
            // blocks of it for their buffers' slabs (`POOL`), where the pool is offered (below).
            // Every host fills a surface's device cache again at the generation it holds where a
            // use asks it, the CPU having written the pages after the lock that moved it
            // (`SURFACE_REFILL`, `decode::surface_cache`, "Refill"). On a platform that imports a
            // pages block's view as a block it knows, a host answers the block's `SURFACE_PAGES`
            // before it makes the view and its import, which it does after the answer, so the
            // guest's call waits for neither (`IMPORT_ADDRESSES_LATER`, `start_surface_pages`).
            // No other platform offers it: its guests' answers carry the view's address as before.
            let view_imports = platform.view_imports();
            let in_place = u64::from(RGFX_FEATURE_SURFACE_PAGES)
                | u64::from(RGFX_FEATURE_PAGES_BUFFERS)
                | u64::from(RGFX_FEATURE_HEAP_OVER_PAGES)
                | RGFX_FEATURE_ICB_SLOTS_IN_PAGES
                | RGFX_FEATURE_READ_ONLY_PAGES
                | RGFX_FEATURE_UNCHANGING_PAGES
                | if view_imports.is_some() {
                    u64::from(RGFX_FEATURE_IMPORT_ADDRESSES_LATER)
                } else {
                    0
                };
            let features = u64::from(RGFX_FEATURE_CB_SUBMITTED)
                | u64::from(RGFX_FEATURE_SURFACE_ORDER)
                | u64::from(RGFX_FEATURE_COMMIT_ORDER)
                | u64::from(RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS)
                | u64::from(RGFX_FEATURE_UNSHARED_QUEUES)
                | u64::from(RGFX_FEATURE_SUBMIT_ROOM)
                | u64::from(RGFX_FEATURE_ICB_SLOTS)
                | u64::from(RGFX_FEATURE_HOST_TIME)
                | u64::from(RGFX_FEATURE_MANAGED_TEXTURE_SLABS)
                | u64::from(RGFX_FEATURE_MANAGED_BUFFER_SLABS)
                | u64::from(RGFX_FEATURE_SMALL_ID_CHUNKS)
                | u64::from(RGFX_FEATURE_DISPLAY_COLOR)
                | u64::from(RGFX_FEATURE_SHARED_EVENT_NAMES)
                | RGFX_FEATURE_SCHEDULED_ASK
                | RGFX_FEATURE_DISPLAY_SCALE
                | RGFX_FEATURE_TEXTURE_DID_MODIFY_DATA
                | RGFX_FEATURE_DISPLAY_POWER
                | if makes_rate_maps {
                    RGFX_FEATURE_RATE_MAPS
                } else {
                    0
                }
                | RGFX_FEATURE_FUNCTION_POINTERS
                | RGFX_FEATURE_DYNAMIC_LIBRARIES
                | RGFX_FEATURE_PIPELINE_KEYS
                | RGFX_FEATURE_PIPELINE_ANSWERS
                | RGFX_FEATURE_GUEST_TIMES
                | RGFX_FEATURE_UPLOADS_WRITTEN
                | RGFX_FEATURE_CURSOR_COLOR
                | RGFX_FEATURE_BUFFER_ESCAPED
                | RGFX_FEATURE_SURFACE_REFILL
                | if offers_doorbell_asks(&config.callbacks) {
                    RGFX_FEATURE_DOORBELL_ASKS
                } else {
                    0
                }
                | if config.doorbell_pages {
                    RGFX_FEATURE_DOORBELL_PAGES
                } else {
                    0
                }
                | if gpu_device.makes_opencl_mode_pipelines() {
                    u64::from(RGFX_FEATURE_OPENCL_MODE)
                } else {
                    0
                }
                | if gpu_device.takes_private_render_state() {
                    u64::from(RGFX_FEATURE_PRIVATE_RENDER_STATE)
                } else {
                    0
                }
                | if meshes {
                    u64::from(RGFX_FEATURE_MESH_PIPELINES)
                } else {
                    0
                }
                | if unified_memory {
                    0
                } else {
                    u64::from(RGFX_FEATURE_MANAGED_AT_OFFSET)
                }
                // A `Managed` buffer at its CPU copy's address ([`managed_at_copy`]).
                | managed_at_copy(unified_memory, P::PINS_IN_BLOCKS, pool_ledger.is_some())
                | if pool_ledger.is_some() {
                    RGFX_FEATURE_POOL | RGFX_FEATURE_POOL_ADDRESSES
                } else {
                    0
                }
                // The contexts' Private blocks, ranges of addresses the host picks, which need
                // no memory: on every host and for every guest.
                | RGFX_FEATURE_PRIVATE_BLOCKS
                // The pages blocks' views' addresses, the host's picks too
                // (`decode::surface_pages`, "Addresses the host gives").
                | RGFX_FEATURE_IMPORT_ADDRESSES
                // A window's captured pointer drives the guest's own mouse while a game hides
                // the cursor ([`crate::window`]); a device with no window has no pointer to.
                | if config.window.is_some() {
                    RGFX_FEATURE_POINTER_MOTION
                } else {
                    0
                };
            // Every host that maps the surfaces' pages keeps a device cache of each surface the
            // kext caches, and keeps fresh surfaces in them alone where the kext defers their
            // pages, the few a process wrote without a lock carried in. A GPU that shares the
            // CPU's memory does so too, as a Mac on an Intel GPU keeps its own copy of each
            // surface the GPU uses and copies it into the pages at a CPU lock after a GPU write
            // (**measured** on a MacBook Pro's UHD 630, docs/graphics-kext/design.md, Decided
            // 14): its textures over the surfaces the GPU draws and the CPU never locks (the
            // compositor's) lie in its own memory rather than over the guest's pages. A platform
            // of the per-shape form, where the pool is offered, tells the guest that every
            // surface's GPU form is its device cache (`SURFACE_CACHE_EVERY`, [`cache_every`]).
            let pages_features = if decoder.surface_pages_available() {
                in_place
                    | u64::from(RGFX_FEATURE_SURFACE_CACHE)
                    | u64::from(RGFX_FEATURE_SURFACE_DEFERRED)
                    | RGFX_FEATURE_SURFACE_DEFERRED_CARRY
                    | cache_every(P::SURFACE_SHAPES, pool_ledger.is_some())
            } else {
                0
            };
            // A host of 16 KiB pages offers the guest RAM's features only to a guest that takes
            // `PAGES_16K`, whose runs are whole 16 KiB pages: the device reads them in
            // `FEATURES_LO` and `_HI` once it is taken, and every other guest reads the features
            // without them, as from a host with no guest RAM
            // (`device::Host::features_with_16k_pages`).
            let (features, features_16k) =
                if pages_features != 0 && decoder.surface_pages_need_16k() {
                    (features | RGFX_FEATURE_PAGES_16K, pages_features)
                } else {
                    (features | pages_features, 0)
                };
            // The decoder carries out each feature the device may offer; the device's registers
            // read the budget only where it offers the deferred blocks.
            let offerable = features | features_16k;
            let own_memory = gpu_device.own_memory_budget();
            let deferred_budget = deferred_budget(offerable, own_memory);
            decoder.set_deferred_budget(deferred_budget);
            // A platform that keeps pins only in its known blocks has the decoder count the
            // others by kind (`decode::unkept`).
            if P::PINS_IN_BLOCKS {
                decoder.set_pins_in_blocks();
            }
            // One that knows a pages block's view's import as a block answers the view's address
            // with the import's (`decode::surface_pages`, "Addresses the host gives").
            if let Some(imports) = view_imports {
                decoder.set_view_imports(imports);
            }
            // One that keeps no view of an IOSurface's pages has its surfaces take device caches
            // whose copies move their bytes (`decode::surface_pages`, "Surfaces with no views").
            if !P::SURFACE_VIEWS {
                decoder.set_surfaces_unviewed();
            }
            // A platform of the per-shape form has each texture over a device cache be a view of
            // its shape's image, the platform's (`ApiPlatform::shape_image`,
            // `decode::surface_shapes`).
            if P::SURFACE_SHAPES {
                let platform = Arc::clone(&platform);
                decoder.set_surface_shapes(
                    Arc::new(
                        move |device: &P::Device, desc: &TextureDesc, placed: bool| match placed {
                            true => platform.placed_shape_image(device, desc),
                            false => platform.shape_image(device, desc),
                        },
                    ),
                    P::PLACED_SHAPE_IMAGES,
                );
                // Their age rule runs on a timer too, so the shapes no texture references go
                // while the guest uses no surface (`decode::surface_cache::shapes`, "The bound").
                decoder.start_shape_age_timer();
            }
            // The reset gate ([`GraphicsControl::reset_gate`]), which the video caches' spare
            // surfaces' makes hold shared too.
            let reset_gate: Arc<RwLock<()>> = Arc::default();
            // A platform whose imports of the host's memory cost each submission stages the
            // caches' copies through the host's CPU (`ApiPlatform::STAGED_SURFACE_COPIES`).
            if P::STAGED_SURFACE_COPIES {
                decoder.set_staged_surface_copies();
            }
            // Every device cache fits one budget together, on any GPU (`decode::surface_cache`,
            // "Within a budget"): where the GPU shares the CPU's memory they are host RAM beside
            // the guest's pages, so the copies stay bounded on a small host; where it has memory
            // of its own they are that memory, beside what the platform places there for the
            // guest, so the caches leave it room.
            let cache_budget = cache_budget(own_memory, unified_memory);
            decoder.set_cache_budget(cache_budget);
            if deferred_budget != 0 {
                log::info!(
                    target: "reims_vgpu::graphics",
                    "IOSurface device caches: {} MiB at most, of the {} MiB of memory the GPU {}",
                    cache_budget >> 20,
                    own_memory >> 20,
                    if unified_memory {
                        "shares with the CPU"
                    } else {
                        "has of its own"
                    }
                );
            }
            if offerable & u64::from(RGFX_FEATURE_SURFACE_DEFERRED_ASK) != 0 {
                decoder.set_deferred_asks();
            }
            if offerable & RGFX_FEATURE_SURFACE_DEFERRED_CARRY != 0 {
                decoder.set_deferred_carries();
            }
            // The processes read the features from the device info, as no register is theirs:
            // those every guest reads, written before any takes `PAGES_16K`.
            info.features_lo = u8::try_from(features & 0xff).unwrap_or_default();
            // A host that offers the surfaces' pages has the guest align their rows as its GPU's
            // linear images take them in place, on any GPU: a texture over a surface lies over its
            // pages, or over a device cache in the pages' layout, and the GPU's own linear image
            // holds it there only at the GPU's pitch (on RADV exactly `round_up(row, 256)`). One
            // that offers its pages only once a guest takes `PAGES_16K` leaves the guest's rows as
            // they were, as the device info is written before it does.
            if decoder.surface_pages_available() && features_16k == 0 {
                info.linear_row_alignment =
                    u32::try_from(gpu_device.linear_row_alignment()).unwrap_or_default();
            }
            // Whether the host's GPU is an integrated one. Nothing in the guest reads it: the
            // bundle's `rgfxMetalNameOf` gives every type the same prefix.
            info.integrated = u8::from(gpu_device.is_integrated());
            region.write(RGFX_DEVICE_INFO_OFFSET, info.as_bytes());
            // The host's hardware video, whose codecs the video caps after the device info list
            // for the guest's video driver, and whose decoders VideoToolbox's paravirtual path
            // lists for an arm64 guest's.
            let mut vt = paravirt::Host::default();
            let video_runtime = crate::video::runtime::PlatformRuntime::of(&platform);
            if let Some(video) = platform.graphics_video(&gpu_device, video_runtime) {
                let caps = video_caps(features, P::SURFACE_SHAPES, &*video);
                let cached = caps.flags & RGFX_VIDEO_CAPS_OUTPUT_CACHED != 0;
                debug_assert!(caps.codec_count as usize <= RGFX_VIDEO_CAPS_CODECS);
                region.write(RGFX_VIDEO_CAPS_OFFSET, caps.as_bytes());
                // VideoToolbox's decoders are listed where the arm lands their pictures in the
                // surfaces the guest lends: each session decodes on the arm.
                // Its encoders, where the arm encodes the source frames the guest lends.
                if video.guest_decodes() {
                    let backend = vt::VtBackend::new(Arc::clone(&video), decoder.lent_views());
                    vt = paravirt::Host::with_backend(
                        vt_decoders(&caps),
                        Arc::new(backend),
                        crate::video::runtime::PlatformRuntime::of(&platform),
                    );
                    let encoders = video.guest_encoders();
                    if !encoders.is_empty() {
                        let backend = vt::VtEncoderBackend::new(
                            Arc::clone(&video),
                            decoder.lent_views(),
                            gpu_device.clone_ref(),
                        );
                        vt = vt.with_encoders(
                            encoders,
                            Arc::new(backend),
                            crate::video::runtime::PlatformRuntime::of(&platform),
                        );
                    }
                }
                if cached && video.imports_device_caches() {
                    decoder.set_exportable_caches();
                } else if cached && video.writes_into_device_caches() {
                    decoder.set_dma_buf_caches();
                }
                #[cfg(windows)]
                if cached && let Some(draw) = video.plane_draws() {
                    decoder.set_drawn_caches(draw, Arc::clone(&reset_gate));
                }
                decoder.set_video(video);
            }
            // Each display's presented surface, display `i`'s at index `i`.
            let presented: Vec<Arc<Presented<P>>> = (0..count)
                .map(|_| {
                    let clock = Arc::clone(&platform);
                    Arc::new(Presented::new(Box::new(move || clock.clock().now_mono())))
                })
                .collect();
            let presents = Arc::new(PresentQueue::new(presented.clone()));
            // A present that waits for its surface's fill is shown as the fill finishes.
            let waiting = Arc::downgrade(&presents);
            decoder.on_fill_done(Arc::new(move || {
                if let Some(presents) = waiting.upgrade() {
                    presents.filled();
                }
            }));
            let answers = PipelineAnswers::open(gpu_device.pipeline_answers_identity().as_deref());
            let control = Arc::new(GraphicsControl {
                decoders: decoder.decoders(),
                device: gpu_device,
                presents,
                queue_sinks: RwLock::default(),
                presented,
                colors: (0..count).map(|_| Mutex::default()).collect(),
                pool,
                pending: Pending::default(),
                answers,
                region: Arc::clone(&region),
                icb_worker: Worker::new("reims-vgpu.icb-create"),
                icbs: Arc::default(),
                pages_buffers: Arc::default(),
                reset_gate,
                inline_walk_max: AtomicUsize::new(INLINE_WALK_MAX),
                inline_walk_alone_max: AtomicUsize::new(INLINE_WALK_ALONE_MAX),
                shared_pool: pool_ledger,
                private_blocks: {
                    let clock = Arc::clone(&platform);
                    PrivateBlocks::new(block_source, Box::new(move || clock.clock().now_mono()))
                },
            });
            let cursors: Vec<Arc<HostCursor>> = (0..count)
                .map(|display| {
                    let (glyph, show, moved) = display_cursor(&config.callbacks, display);
                    Arc::new(HostCursor::new(&platform, glyph, show, moved))
                })
                .collect();
            let host = Box::new(CallbackHost {
                callbacks: config.callbacks,
                features: features | vt_features(&vt),
                features_16k,
                deferred_budget,
                control: Arc::clone(&control),
                vt,
                vt_seen: Arc::default(),
            });
            // Display 0's window, then those of displays 1 and up, or none.
            let window_configs: Vec<Option<WindowConfig>> = match config.window {
                Some(first) => std::iter::once(first)
                    .chain(config.display_windows)
                    .map(Some)
                    .collect(),
                None => (0..count).map(|_| None).collect(),
            };
            let displays = u32::try_from(count).unwrap_or(u32::MAX);
            // Where several displays' windows tell the guest which one the pointer is over, and
            // carry a captured pointer past their edges, and where each window's captured pointer
            // drives the guest's mouse: the device's link, once it is made.
            let pointer = Arc::new(OnceLock::new());
            let desktop = crate::window::Desktop::new(Arc::clone(&pointer));
            let mut by_display = Vec::with_capacity(count);
            let mut windows = Vec::new();
            let mut screens: Vec<Box<dyn Screen>> = Vec::with_capacity(count);
            let mut reads = Vec::with_capacity(count);
            let mut first_shown = None;
            for (display, window_config) in window_configs.into_iter().enumerate() {
                let (screen, shown): (Arc<dyn Screen>, Arc<Shown>) = match window_config {
                    Some(mut window_config) => {
                        crate::window::give_motion(&mut window_config, &pointer);
                        if displays > 1 {
                            let id = u32::try_from(display).unwrap_or(u32::MAX);
                            crate::window::tell_pointer(&mut window_config, id, &pointer);
                            desktop.give(&mut window_config, id);
                        }
                        let (link, screen) = WindowScreen::open(
                            &platform,
                            &region,
                            &control.presented[display],
                            window_config,
                            display.to_u64(),
                        );
                        by_display.push(Some(link.weak()));
                        windows.push(link);
                        let shown = Arc::clone(&screen.shown);
                        (Arc::new(screen), shown)
                    }
                    None => {
                        by_display.push(None);
                        let shown = Arc::new(Shown::default());
                        let screen = HeadlessScreen {
                            shown: Arc::clone(&shown),
                        };
                        (Arc::new(screen), shown)
                    }
                };
                reads.push(DisplayRead::new(
                    &platform,
                    &region,
                    &shown,
                    &control.presented[display],
                    Arc::clone(&screen),
                    displays,
                ));
                first_shown.get_or_insert(shown);
                screens.push(Box::new(SharedScreen {
                    screen,
                    cursor: Arc::clone(&cursors[display]),
                }));
            }
            #[cfg(test)]
            let shown = first_shown.filter(|_| !windows.is_empty());
            #[cfg(not(test))]
            drop(first_shown);
            let device = reims_vgpu_graphics::Device::with_screens(region, host, screens)
                .ok_or(Error::HostResources)?;
            desktop.set_windows(by_display);
            let _ = pointer.set(device.pointer_link());
            LIVE.fetch_add(1, Ordering::SeqCst);
            Ok(GraphicsDevice {
                windows,
                device,
                cursors,
                ending: Arc::downgrade(&control),
                #[cfg(any(test, feature = "testkit"))]
                control,
                #[cfg(test)]
                shown,
                reads,
                pool: shared_pool,
                platform: PhantomData,
            })
        })
    }

    /// The host mapped the pool's piece at `physical_address` it was asked to add
    /// ([`GraphicsCallbacks::pool_add`]): the guest's address space has it from now on. Any
    /// thread, the host's lock held or not; waits for nothing and calls back never.
    /// [`Error::InvalidArgument`] without a pool, or for an address the device asked no add at,
    /// or one the host said it added already.
    pub fn pool_added(&self, physical_address: u64) -> Result<(), Error> {
        (self.pool.as_ref())
            .ok_or(Error::InvalidArgument)?
            .added(physical_address)
    }

    /// The host took the pool's piece at `physical_address` out of the guest's address space,
    /// and no thread of its own touches it: asked to ([`GraphicsCallbacks::pool_remove`]), or of
    /// its own, as it does with every piece before the device's release. Any thread, the host's
    /// lock held or not; waits for nothing and calls back never. [`Error::InvalidArgument`]
    /// without a pool, or for an address the device asked no add at.
    pub fn pool_removed(&self, physical_address: u64) -> Result<(), Error> {
        (self.pool.as_ref())
            .ok_or(Error::InvalidArgument)?
            .removed(physical_address)
    }

    /// The range of guest physical addresses the device's Shared pool took of the addresses its
    /// config gave ([`GraphicsConfig::pool`]); `None` without a pool. Any thread; waits for
    /// nothing and calls back never.
    pub fn pool_range(&self) -> Option<PoolConfig> {
        self.pool.as_ref().map(Pool::range)
    }

    /// Asks for at least `length` more bytes of the guest's Shared pool now, as the guest's own
    /// ask does: the pool's taker takes a block, and the pool's thread asks the host to map it
    /// ([`GraphicsCallbacks::pool_add`]). Waits for nothing and calls back never.
    /// [`Error::InvalidArgument`] without a pool, or for a `length` of 0.
    pub fn pool_grow(&self, length: u64) -> Result<(), Error> {
        let pool = self.pool.as_ref().ok_or(Error::InvalidArgument)?;
        if length == 0 {
            return Err(Error::InvalidArgument);
        }
        pool.grow(length);
        Ok(())
    }

    /// The region's first byte: BAR 2's memory, which the host maps into the guest, write-back.
    /// It stays mapped until the device and its window's GPU work have ended.
    pub fn region_base(&self) -> *mut u8 {
        self.device.region_base()
    }

    /// How many of the guest's completion records were handed over to the queues' threads to
    /// post, from threads that may not post them (the sinks' `post_from_handler`).
    #[cfg(any(test, feature = "testkit"))]
    pub fn handed_over(&self) -> u64 {
        self.device.handed_over()
    }

    /// The objects the stage-3 decoder holds: the tests check a create reached it.
    #[cfg(any(test, feature = "testkit"))]
    pub fn decoder_object_count(&self) -> usize {
        self.control
            .decoders
            .sender(None, Sinks::default())
            .object_count()
    }

    /// The GPU address of the stage-3 decoder's buffer or heap `obj_id`, where it holds one.
    #[cfg(test)]
    pub(crate) fn decoder_gpu_address(&self, obj_id: u64) -> Option<u64> {
        self.control
            .decoders
            .sender(None, Sinks::default())
            .gpu_address_of(obj_id)
    }

    /// The ids that stand for a create the stage-3 decoder refused (`decode::refused`).
    #[cfg(test)]
    pub(crate) fn decoder_refused_ids(&self) -> Vec<u64> {
        self.control
            .decoders
            .sender(None, Sinks::default())
            .refused_ids()
    }

    /// How many command queues have a per-queue ring and service thread: the tests check a
    /// `QUEUE_CREATE` with valid ring parameters made one.
    #[cfg(any(test, feature = "testkit"))]
    pub fn graphics_queue_count(&self) -> usize {
        self.device.queue_count()
    }

    /// By queue id, where the completion sink its command buffers are begun with lies
    /// (`GraphicsControl::queue_sinks`): the tests check a queue's buffers share one.
    #[cfg(test)]
    pub(crate) fn queue_completion_sinks(&self) -> Vec<(u64, usize)> {
        let sinks = self.control.queue_sinks.read();
        let at = |sinks: &Sinks| {
            sinks
                .completion
                .as_ref()
                .map(|c| Arc::as_ptr(c).cast::<()>())
        };
        (sinks.iter())
            .map(|(&queue, (_, s))| (queue, at(s).map_or(0, |p| p as usize)))
            .collect()
    }

    /// The region's size: BAR 2's, a power of two.
    pub fn region_size(&self) -> u64 {
        self.device.region_size()
    }

    /// What the display shows now, read back (`read`'s module doc): the guest's scanout, or the
    /// IOSurface its display pipe presents, through the display's colour, as the window draws it
    /// into a drawable of the display's mode, written into `dst` as rows of `width * 4` bytes,
    /// BGRA8, black where the guest shows nothing. The mode's `(width, height)` is returned
    /// whatever the result. [`Error::InvalidArgument`], with nothing written, where `dst` holds
    /// fewer than `width * height * 4` bytes; [`Error::Metal`] where the platform refuses the
    /// draw or its texture; [`Error::Gpu`] where the draw ended with an error. Waits for the draw
    /// on the GPU, and for no lock a callback of the device's may wait on; the guest sees
    /// nothing of it.
    pub fn read_display(&self, dst: &mut [u8]) -> ((u32, u32), Result<(), Error>) {
        self.read_display_at(0, dst)
    }

    /// [`Self::read_display`] of display `display`, at its own mode, the one the guest is told
    /// (sized to its scanout room on a device of several displays); display 0's is
    /// `read_display`. [`Error::InvalidArgument`], with nothing written and a size of (0, 0),
    /// for a display at or past the device's count.
    pub fn read_display_at(&self, display: u32, dst: &mut [u8]) -> ((u32, u32), Result<(), Error>) {
        entry::run(|_| match self.reads.get(display as usize) {
            Some(read) => read.read(dst),
            None => ((0, 0), Err(Error::InvalidArgument)),
        })
    }

    /// How many displays the device has: 1 to `RGFX_DISPLAYS_MAX`
    /// ([`GraphicsConfig::display_count`]).
    pub fn display_count(&self) -> u32 {
        u32::try_from(self.reads.len()).unwrap_or(u32::MAX)
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
    /// calling `raise_interrupt` inside this call (`reims-vgpu-graphics`'s `Device::mmio_write_taking`),
    /// allocating nothing; the queue's thread then takes it as it takes any. A write of ring 0's
    /// may take and answer the control ring's next message there, where it is a
    /// `SURFACE_DOWNLOAD` whose pages hold the surface's device cache already
    /// ([`GraphicsControl::downloaded_at_doorbell`]), allocating nothing either. It waits for no
    /// lock but the queues' table, as `mmio_write`'s doorbell does.
    pub fn mmio_write_taking(&self, offset: u64, value: u32) {
        entry::run(|_| self.device.mmio_write_taking(offset, value));
    }

    /// A 32-bit write of BAR 4, the doorbell pages, at `offset`, where the host maps it
    /// ([`GraphicsConfig::doorbell_pages`]) and no route took it: a write of a ring's id at a
    /// page's first byte is that `DOORBELL` write through [`Self::mmio_write_taking`] for a ring of
    /// the context the page is bound to, and may answer its next command buffer and call
    /// `raise_interrupt` inside this call; every other is dropped and counted
    /// (`reims-vgpu-graphics`'s `Device::doorbell_write`). It waits for no lock but the queues'
    /// table.
    pub fn doorbell_write(&self, offset: u64, value: u32) {
        entry::run(|_| self.device.doorbell_write(offset, value));
    }

    /// Takes a copy of `doorbell`, the host's `ioeventfd` for `DOORBELL` with ring 0: a file on
    /// Linux and macOS, an event on Windows ([`RawDoorbell`]). False where `doorbell` is not open
    /// or the device has one already.
    pub fn set_doorbell(&self, doorbell: RawDoorbell) -> bool {
        entry::run(|_| self.device.set_doorbell(doorbell))
    }

    /// The host's reset of the device (the guest's bus reset).
    pub fn reset(&self) {
        entry::run(|_| self.device.reset());
    }

    /// Where the guest last put its cursor's hot spot, in the display's pixels, each axis kept
    /// to 0 to 65535: (0, 0) before it moved it and after a reset. The host reads it at its
    /// `cursor_move` ([`GraphicsCallbacks`]).
    pub fn cursor_position(&self) -> (u16, u16) {
        self.cursor_position_at(0)
    }

    /// [`Self::cursor_position`] of display `display`, in its pixels, which the host reads at its
    /// `cursor_move_at` ([`GraphicsCallbacks`]); display 0's is `cursor_position`. (0, 0) for a
    /// display at or past the device's count.
    pub fn cursor_position_at(&self, display: u32) -> (u16, u16) {
        entry::run(|_| {
            (self.cursors.get(display as usize)).map_or((0, 0), |cursor| cursor.position())
        })
    }
}

impl<P: ApiPlatform> Drop for GraphicsDevice<P> {
    /// The windows close, display 0's first, then the device ends; neither waits
    /// (`reims-vgpu-graphics`'s `device`). The device no longer counts among those alive
    /// ([`GraphicsDevice::let_go_of_gpu`]).
    fn drop(&mut self) {
        entry::run(|_| {
            drop(std::mem::take(&mut self.windows));
        });
        // The host removed every piece of the pool before the release: the pool's thread ends,
        // each block goes back to the source once no thread of the device's holds it and the
        // platform freed every import of its bytes (`pool`'s keeps), and the source frees them;
        // past the release's 5 s, one still held stays out, and the source is kept to the exit.
        drop(self.pool.take());
        // The contexts' Private takes in flight end unanswered, and their workers go, each
        // giving back the block it was taking: none calls the source past the release.
        if let Some(control) = self.ending.upgrade() {
            control.private_blocks.close();
        }
        let ending: Weak<dyn std::any::Any + Send + Sync> = self.ending.clone();
        let mut ended = ENDING.lock();
        ended.retain(|state| state.strong_count() > 0);
        ended.push(ending);
        drop(ended);
        LIVE.fetch_sub(1, Ordering::SeqCst);
    }
}

impl<P: ApiPlatform> GraphicsDevice<P> {
    /// Where no graphics device lives in the process, lets go of the GPU devices the platform
    /// keeps for it ([`ApiPlatform::let_go_of_devices`]): what the C API's release of the last
    /// one does, with [`LET_GO_BOUND`], so that no runtime device is live as the process exits
    /// (on NVIDIA's driver an exit with one live can end in `SIGSEGV` inside the driver's exit
    /// handler, `reims_vgpu_vulkan::let_go`). It waits, at most `bound`, for the ended devices'
    /// threads, their objects and their GPU work to let go of the GPU devices; a device callback
    /// still running holds them up, one blocked on a lock the caller holds until the bound. Past
    /// the bound it lets go of nothing, says what still held them in a warning, and answers it as
    /// `Err`. `Ok(false)` where a graphics device lives, letting go of nothing; `Ok(true)` once
    /// let go of. A device made after it runs on a new GPU device of the platform's.
    pub fn let_go_of_gpu(bound: std::time::Duration) -> Result<bool, String> {
        if LIVE.load(Ordering::SeqCst) != 0 {
            return Ok(false);
        }
        entry::run(|_| {
            let let_go = P::process().let_go_of_devices(bound);
            let_go.map(|()| true).map_err(|held| held_up(bound, &held))
        })
    }

    /// Quiesces the process's platform ahead of its exit (core's `Platform::quiesce`), whether a
    /// graphics device lives or was released: what the C API's `rvg_quiesce_for_exit` does, as
    /// the last call of the host's before its exit, on the thread that then exits. Its answer is
    /// whether the host may then call `exit`. The library's threads park at their next step and
    /// the steps in progress on other threads are waited for; on the platform over
    /// metal2vulkan-runtime, then, where a GPU device is still live, rt's threads end and its GPU
    /// work completes, and the process's GPU work after it is refused. Nothing is freed. On the
    /// platform over metal2vulkan-runtime, true where it keeps no GPU device (the release of the
    /// last graphics device let go of it, [`Self::let_go_of_gpu`]) and the steps ended in time;
    /// false where it keeps one, stopped or not, since its runtime supports no `exit` with a device
    /// live. On the Vulkan platform, true only where it never asked its runtime for a GPU device
    /// in the process and the steps ended in time; false once it did, let go of since or not,
    /// since that runtime's exit hook, registered as the device was made, waits for the runtime's
    /// work still in progress (a pipeline's making can outlast the device's let-go) only for a
    /// bound, and on Linux past it ends the process at once, before the host's exit work ran.
    /// False also where a step outlasted the wait, with a warning. On false the process's counts
    /// are written then (`reims_vgpu_vulkan::exit_gate::write_counts_ahead`), since the host ends
    /// with `_exit`, which runs no exit handler. Every wait ends by `bound`. A later call on the
    /// same thread answers at once, with the first answer.
    pub fn quiesce_for_exit(bound: std::time::Duration) -> bool {
        static ANSWER: OnceLock<bool> = OnceLock::new();
        entry::run(|_| {
            *ANSWER.get_or_init(|| {
                let may_exit = P::process().quiesce(bound);
                if !may_exit {
                    reims_vgpu_vulkan::exit_gate::write_counts_ahead();
                }
                may_exit
            })
        })
    }
}

/// The warning of a let-go that waited out its `bound` ([`GraphicsDevice::let_go_of_gpu`]), `held`
/// the platform's account of the devices still held: it names the ended graphics devices whose
/// decoder still lives, too. Logged, and answered.
fn held_up(bound: std::time::Duration, held: &str) -> String {
    let ending = (ENDING.lock().iter())
        .filter(|state| state.strong_count() > 0)
        .count();
    let decoders = match ending {
        0 => String::new(),
        n => format!(
            "the decoder of {n} ended graphics devices still lives (its threads, a callback \
             among them, its objects or their GPU work), and "
        ),
    };
    let line = format!(
        "the GPU device stays for the process: after {bound:?}, {decoders}{held} still held it"
    );
    log::warn!("{line}");
    line
}

/// The stage-3 decoder and the device handle it drives, shared by the device's [`Host`], which
/// feeds it ring 0's object lifecycle on each context's lane (`reims-vgpu-graphics`' device) and the
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
    /// The surface each display's pipe shows, shared with its window's sink, display `i`'s at
    /// index `i`.
    presented: Vec<Arc<Presented<P>>>,
    /// The presents waiting for the command buffers that write their surfaces.
    presents: Arc<PresentQueue<P>>,
    /// By queue id: the sinks a command queue's buffers are begun with, and the poster they were
    /// made for ([`Self::queue_sinks`]). An entry goes with its queue's destroy, its context's
    /// close or the device's reset.
    queue_sinks: RwLock<HashMap<u64, (CompletionPoster, Sinks)>>,
    /// Each display colour's `DISPLAY_COLOR` parts staged since the display's last present, which
    /// its next one takes (`color`): on the presents' lane, in ring order with the presents.
    /// Display `i`'s at index `i`.
    colors: Vec<Mutex<ColorStaging>>,
    /// The threads ring 0's pipeline and library creates compile on (`compile`, `make`).
    pool: CompilePool,
    /// The pipelines being compiled, from their create's hand-over to the pool to their store,
    /// which a command buffer naming one waits for.
    pending: Pending,
    /// The answers it gave keyed pipeline creates, kept from one run of the host to the next
    /// (`answers`): learnt as each make finishes ([`Self::store_made`]), handed to the kext on
    /// `PIPELINE_ANSWERS` ([`Self::pipeline_answers`]).
    answers: PipelineAnswers,
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
    /// The no-copy buffers and heaps whose `BUFFER_OVER_PAGES` or `HEAP_OVER_PAGES` was handed to
    /// the surface worker and is not yet made ([`Self::start_over_pages`]): by their context
    /// ([`pages_context`]), and by their pages block's offset.
    pages_buffers: Arc<Unsettled>,
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
    /// The longest `COMMAND_BUFFER` message, in bytes, carried out on its queue's service thread
    /// where it would wait for nothing ([`Self::carry_out`]): [`INLINE_WALK_MAX`]; the tests that
    /// hold a walk to stand for a long one set it to 0, as a long buffer goes to its lane.
    inline_walk_max: AtomicUsize,
    /// The longest `COMMAND_BUFFER` message, in bytes, carried out on its queue's service thread
    /// where it would wait for nothing and its queue's ring holds nothing more
    /// ([`Self::carry_out`]): [`INLINE_WALK_ALONE_MAX`]; set to 0 with [`Self::inline_walk_max`]
    /// by the tests whose buffers go to their lanes.
    inline_walk_alone_max: AtomicUsize,
    /// The ledger of the contexts' blocks of the guest's Shared pool, where the device offers it
    /// (`POOL_TAKE`, `POOL_GIVE`, [`Self::take_pool_block`]).
    shared_pool: Option<PoolLedger>,
    /// The ledger of the contexts' Private blocks (`PRIVATE_TAKE`, `PRIVATE_GIVE`,
    /// [`Self::take_private_block`]): the host's picks of addresses until the device runs on
    /// metal2vulkan-runtime, whose Private blocks then fill it.
    private_blocks: PrivateBlocks,
}

/// The longest `COMMAND_BUFFER` message, in bytes, that is walked and committed on its queue's
/// service thread where its commit would wait for nothing ([`GraphicsControl::carry_out`]): a
/// few dozen ops, whose walk takes about what a lane thread's wake does (about 2 us an op on the
/// Vulkan platform on an RTX 5080 laptop host, against 80 to 90 us a wake there), so a longer
/// buffer, a compositor's frame, goes to its lane and the queue's next buffers are taken beside
/// its walk.
const INLINE_WALK_MAX: usize = 2048;

/// The longest `COMMAND_BUFFER` message, in bytes, that is walked and committed on its queue's
/// service thread where its commit would wait for nothing and its queue's ring holds nothing
/// more as it is taken ([`GraphicsControl::carry_out`]): no buffer of the queue is there to be
/// taken beside its walk, so a lane's thread would be woken only to walk it while the service
/// thread sleeps. Over Geekbench 7.1's Metal compute pass on an RTX 5080 Laptop host, a compute
/// buffer longer than [`INLINE_WALK_MAX`] went from its begin to its walk's start on its lane in
/// 73.7 us at the mean and 90.3 at the median, most of it the lane thread's wake, where one walked
/// on the service thread started 3.0 us after its begin (4127 and 5655 buffers with copies of
/// their `Managed` bytes in; potluck job m3-max-62499's trace; measured). A walk of 2 to 8 KiB
/// took 51 to 72 us at the median there: about a wake, which is what a buffer the guest
/// publishes during the walk waits for its take, its commit coming after this one's in its turn
/// either way.
const INLINE_WALK_ALONE_MAX: usize = 8192;

/// A `COMMAND_BUFFER` message as its queue's service thread began it
/// ([`GraphicsDecoder::begin_taken`]), with its pre-pass where it hands one
/// ([`GraphicsControl::hand_prepass`]) and where the thread meant its walk to go: what
/// [`GraphicsControl::carry_out`] walks and closes.
type BegunBuffer<P> = (
    RaiseResult<Result<Begun<P>, DecodeError>>,
    Option<Prepass<P>>,
    Walk,
);

/// Where a `COMMAND_BUFFER` message's queue's service thread meant its walk to go as it began
/// it ([`GraphicsControl::begin_command_buffer`]), and what the walk waits for first, for
/// [`GraphicsControl::carry_out`].
struct Walk {
    /// The lane's idle thread was woken for it as its turn was taken
    /// ([`GraphicsControl::goes_to_lane`]), the buffer not alone.
    to_lane: bool,
    /// Its message is [`INLINE_WALK_ALONE_MAX`] bytes at most, and the guest had published
    /// nothing behind it on its queue's ring as it was taken ([`QueueClaim::nothing_published`]).
    alone: bool,
    /// It runs no compute encoder, or its pre-pass found every dispatch it runs made
    /// (`decode::prepass`, "Whether every dispatch is made"): its walk waits for no make, nor
    /// for a stage's fill.
    made: bool,
    /// The pipelines it names that were pending as it was begun ([`Pending::named_by`]), which its
    /// walk waits for on its lane ([`Pending::wait_on_lane`]); empty for most.
    awaited: Vec<u64>,
}

/// A pipeline create compiled off its ring ([`GraphicsControl::compile`]): what its settling
/// needs of it once its make has run ([`GraphicsControl::settle`]).
#[derive(Clone, Copy)]
struct Creating {
    /// The pipeline's id.
    obj_id: u64,
    /// The create's seqno.
    seqno: u64,
    /// The create's key, under which its answer is kept for the next runs.
    key: Option<PipelineKey>,
    /// The clock of its table as the create was prepared ([`PipelineJob::seen_from`]).
    seen_from: u64,
    /// The reset count its compile began at ([`Pending::begin`]).
    began: u64,
}

impl Creating {
    /// The create `job` is of, its compile begun at reset count `began`.
    fn of<P: Platform>(job: &PipelineJob<P>, began: u64) -> Creating {
        Creating {
            obj_id: job.obj_id,
            seqno: job.seqno,
            key: job.key(),
            seen_from: job.seen_from,
            began,
        }
    }
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
    /// Whether every write of the bytes the texture reads is the platform's work, queued behind
    /// a window draw of it scheduled before ([`SurfacePresent`]'s `queued`): a draw of it
    /// then holds the presents' answers until it is scheduled, not until it has finished
    /// ([`DrawnPresents::after`]).
    queued: bool,
}

impl Drop for DisplayHold {
    fn drop(&mut self) {
        for answer in self.answers.get_mut().drain(..) {
            answer();
        }
    }
}

/// A texture over a surface a present shows, with the display's view of it and its hold of its
/// block.
pub(crate) struct ShownTexture<P: Platform> {
    /// The texture.
    pub(crate) texture: P::Texture,
    /// The display's view of it, which every present of the texture shares ([`ShownView`]).
    view: Arc<ShownView<P>>,
    /// The hold of the block it lies in.
    hold: Arc<DisplayHold>,
}

/// A texture the display showed before, by its surface's id, with the display's view of it
/// ([`Presented`]'s `retired`).
type RetiredTexture<P> = (u32, <P as Platform>::Texture, Arc<ShownView<P>>);

/// The texture the display shows, the display's view of it and the hold of its block
/// ([`Presented::shown`]).
pub(crate) type ShownRefs<P> = (
    <P as Platform>::Texture,
    Arc<ShownView<P>>,
    Arc<DisplayHold>,
);

/// The display's own view of a texture over a surface it shows: the whole texture in its own
/// pixel format, which the platform places at one of its own indices, never at one the guest
/// places (on the Vulkan platform, `rt_platform::texture`'s `OwnIndices`). The window's draws
/// that sample the texture (the present shaders' and the colour's) read it through this view
/// (`sink`'s module doc, "The display's view"): the texture's own index is the guest's, which goes
/// back to the guest's kext as the process that made the texture ends (a logout, a window
/// server's restart), and the next texture made at that index takes its element at once, which a
/// draw of the texture through its index would then read. Made at the first such draw of the
/// texture, and kept while the decoder holds the texture (its entry, `decode::tables`'s
/// `Entry::shown_view`) and while the display does (each [`ShownTexture`] of it), so a later
/// present of the texture makes nothing; let go of with the last of them, its index given back
/// by the platform's rule for its own indices.
pub(crate) struct ShownView<P: Platform>(OnceLock<P::Texture>);

impl<P: Platform> Default for ShownView<P> {
    fn default() -> Self {
        ShownView(OnceLock::new())
    }
}

impl<P: Platform> ShownView<P> {
    /// The view of `texture`, the texture this view is of, made at the first ask; `None`, with
    /// nothing kept, where the platform refuses it, so the next draw asks again.
    pub(crate) fn of(&self, texture: &P::Texture) -> Option<&P::Texture> {
        if let Some(view) = self.0.get() {
            return Some(view);
        }
        let view = no_catch(texture.new_texture_view(texture.pixel_format()))?;
        // A draw on another thread that made one first keeps its own: this one goes.
        Some(self.0.get_or_init(|| view))
    }

    /// The view made, if any (for the tests).
    #[cfg(test)]
    pub(crate) fn made(&self) -> Option<&P::Texture> {
        self.0.get()
    }

    /// Whether the view was made (the host's summary, `summary::kept`).
    pub(crate) fn is_made(&self) -> bool {
        self.0.get().is_some()
    }
}

/// What a present shows: a surface's texture (`None`: the scanout), by the surface's id, or what the
/// display shows already (`RGFX_PRESENT_KEEP`), with only its colour changed.
pub(crate) enum Shows<P: Platform> {
    /// The surface `id`'s texture, or the scanout for `None`.
    Surface(u32, Option<ShownTexture<P>>),
    /// What the display shows already.
    Kept,
}

/// The IOSurface the display pipe shows in place of the scanout (`PRESENT_SURFACE`): the texture
/// the decoder made over it, which the window's sink draws into the next drawable (`sink`), and
/// the presents' answers, given at the first refresh of the device's display after their show,
/// once no draw of the window that may read the surface of a present before them holds them: a
/// draw holds them until the guest's next write of that surface is queued behind it, or until it
/// has finished ([`PresentAnswers`], [`DrawnPresents::after`]). The window draws a present when
/// it is made, and again where the window needs a frame (a new layout, the cursor): each present
/// is the guest's own, and none is batched or repeated by the GPU's work. The texture's block is
/// held until the next present, the device's reset or the device's end, and each draw of it until
/// it has finished ([`DisplayHold`]); the texture itself is let go of at the device's refresh
/// after the next present, or at its reset (`retired`).
pub(crate) struct Presented<P: Platform> {
    /// The texture shown; `None` shows the scanout.
    pub(crate) texture: Mutex<Option<ShownTexture<P>>>,
    /// The textures the display showed before, by their surface's id, with the display's views
    /// of them, let go of at the device's next refresh ([`Self::refreshed`]) or its reset: a
    /// present is shown on the platform's thread that runs its writer's completion handlers,
    /// which waits for nothing, and the release of a texture's last hold is the platform's work,
    /// which may wait (a driver that frees an import of the guest's pages).
    retired: Mutex<Vec<RetiredTexture<P>>>,
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
    /// The display's colour (`color`), and how many times a present or the reset changed it,
    /// which the sink's buffers of it are named by.
    color: Mutex<(u64, Arc<DisplayColor>)>,
}

/// For the tests: the timing lines' clock stays at 0.
#[cfg(test)]
impl<P: Platform> Default for Presented<P> {
    fn default() -> Self {
        Presented::new(Box::new(
            reims_vgpu_vulkan::core::platform::os::clock::MonoTime::default,
        ))
    }
}

impl<P: Platform> Presented<P> {
    /// Nothing shown yet: the scanout, with every colour stage off; the presents' timing lines
    /// read `now`, the platform's monotonic clock.
    fn new(now: compile::Now) -> Self {
        Presented {
            texture: Mutex::new(None),
            retired: Mutex::default(),
            holds: Mutex::default(),
            surface_id: AtomicU32::new(0),
            window: OnceLock::new(),
            refused: AtomicBool::new(false),
            formats: Mutex::new(None),
            answers: PresentAnswers::new(now),
            color: Mutex::new((0, Arc::default())),
        }
    }

    /// `texture`, over a surface whose present was taken, held for the display from now, with
    /// `view`, the display's view of it ([`ShownView`]): the block at `span` it lies in is not
    /// given back until the display is done with it ([`DisplayHold`]). `queued`: every write of
    /// the bytes it reads is the platform's work ([`DisplayHold`]'s `queued`).
    fn hold(
        &self,
        texture: P::Texture,
        view: Arc<ShownView<P>>,
        span: Option<(u64, u64)>,
        queued: bool,
    ) -> ShownTexture<P> {
        let hold = Arc::new(DisplayHold {
            span,
            answers: Mutex::default(),
            queued,
        });
        if span.is_some() {
            let mut holds = self.holds.lock();
            holds.retain(|h| h.strong_count() > 0);
            holds.push(Arc::downgrade(&hold));
        }
        ShownTexture {
            texture,
            view,
            hold,
        }
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
            queued: false,
        });
        for hold in held {
            let join = Arc::clone(&join);
            hold.answers.lock().push(Box::new(move || drop(join)));
        }
        drop(live);
        Box::new(move || drop(join))
    }

    /// Shows `texture` (the surface `id`'s), or the scanout for `None`, or keeps what the display
    /// shows where `shows` is [`Shows::Kept`], with the colour stages `color` sets
    /// ([`DisplayColor::apply`]), and asks the window for a frame; `answer` is given at the vblank
    /// the display first shows it ([`PresentAnswers`]), or at once without a window to draw it (a
    /// headless device, or a window with no sink: [`PresentAnswers::answered_at_show`]), and the
    /// present's timing, from when it was `taken`, logged there at debug and counted in the host's
    /// summary. The hold of
    /// the texture shown before is returned, for the caller to let go of with no lock held: a
    /// release waiting for it may be answered there.
    #[must_use]
    fn show(
        &self,
        shows: Shows<P>,
        color: Option<ColorChange>,
        answer: PresentAnswer,
        taken: present::Taken,
    ) -> Option<Arc<DisplayHold>> {
        if let Some(change) = color {
            let mut shown = self.color.lock();
            let mut next = DisplayColor::clone(&shown.1);
            next.apply(change);
            if next.is_identity() != shown.1.is_identity() {
                log::info!(
                    target: "reims_vgpu::graphics",
                    "the display's colour is {}",
                    if next.is_identity() {
                        "the identity again"
                    } else {
                        "no longer the identity: the window draws through it"
                    }
                );
            }
            *shown = (shown.0.wrapping_add(1), Arc::new(next));
        }
        if let Shows::Surface(_, Some(shown)) = &shows {
            reims_vgpu_vulkan::recorder::present(&shown.texture);
        }
        let hold = match shows {
            Shows::Kept => None,
            Shows::Surface(id, texture) => self.swap(id, texture),
        };
        match self.window.get() {
            Some(window) => {
                self.answers.shown(answer, taken);
                window.new_frame();
            }
            None => self.answers.answered_at_show(answer, taken),
        }
        hold
    }

    /// The display shows `texture` (the surface `id`'s), or the scanout for `None`: the hold of
    /// the texture shown before, for [`Self::show`]'s caller. That texture is retired with the
    /// display's view of it, let go of at the next refresh (`retired`).
    fn swap(&self, id: u32, texture: Option<ShownTexture<P>>) -> Option<Arc<DisplayHold>> {
        let previous = self.surface_id.swap(id, Ordering::AcqRel);
        // A guest that draws into a few IOSurfaces in turn (WindowServer's three) changes the
        // surface shown at every present, so each change is a debug line of its own target, which
        // a run that counts the display's flips names (`reims_vgpu::graphics::shown=debug`); only
        // a change between the scanout and a surface is an info line too.
        if previous != id && (previous == 0 || id == 0) {
            log::info!(
                target: "reims_vgpu::graphics",
                "the display shows IOSurface {id} (was {previous}; 0 is the scanout)"
            );
        } else if previous != id {
            log::debug!(
                target: "reims_vgpu::graphics::shown",
                "the display shows IOSurface {id} (was {previous}; 0 is the scanout)"
            );
        }
        let shown = std::mem::replace(&mut *self.texture.lock(), texture);
        let (shown, hold) = shown.map(|s| ((s.texture, s.view), s.hold)).unzip();
        if let Some((shown, view)) = shown {
            self.retired.lock().push((previous, shown, view));
        }
        hold
    }

    /// Lets go of the textures the display showed before (`retired`), each with the display's
    /// view of it in a refusal scope: one may be the last hold of a closed context's, so a
    /// release the platform dies at is refused, logged, and the display goes on. `at` names the
    /// event, for the log line.
    fn let_go_of_retired(&self, at: &str) {
        let retired = std::mem::take(&mut *self.retired.lock());
        for (id, texture, view) in retired {
            if let Err(refusal) =
                reims_vgpu_vulkan::core::raise::refusals::refusing(move || drop((texture, view)))
            {
                log::warn!(
                    target: "reims_vgpu::graphics", kind:% = refusal;
                    "the texture of IOSurface {id} was let go of at {at}, and its release was \
                     refused: {refusal}"
                );
            }
        }
    }

    /// The display's colour, and the count of its changes, which names it.
    pub(crate) fn color(&self) -> (u64, Arc<DisplayColor>) {
        let color = self.color.lock();
        (color.0, Arc::clone(&color.1))
    }

    /// The device's reset: the display shows the scanout, which the reset keeps, in place of any
    /// surface, whose texture is let go of in a refusal scope, as are those retired before it
    /// (`retired`), with every colour stage off; the
    /// answers still owed are given as they come, and dropped, as each is of a present taken
    /// before the reset. The scanout is the guest's own pixels, still in the region, but the
    /// colour is what the transactions of a display pipe the reset ended set: kept, a reset in the
    /// middle of a fade would leave the window black, with no transaction left to undo it. The
    /// guest's next transaction that sets a stage sets it again.
    fn reset(&self) {
        {
            let mut color = self.color.lock();
            *color = (color.0.wrapping_add(1), Arc::default());
        }
        self.let_go_of_retired("the device's reset");
        let previous = self.surface_id.swap(0, Ordering::AcqRel);
        let shown = self.texture.lock().take();
        let (shown, hold) = shown.map(|s| ((s.texture, s.view), s.hold)).unzip();
        if let Err(refusal) =
            reims_vgpu_vulkan::core::raise::refusals::refusing(move || drop(shown))
        {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = refusal;
                "the texture of IOSurface {previous} was let go of at the device's reset, and its \
                 release was refused: {refusal}"
            );
        }
        drop(hold);
        if let Some(window) = self.window.get() {
            window.new_frame();
        }
    }

    /// The window draws the latest present: the presents shown after it are answered once the
    /// draw lets go of them, as it is scheduled or has finished ([`DrawnPresents::after`]), or
    /// once it is dropped undone.
    pub(crate) fn drawn(self: &Arc<Self>) -> DrawnPresents<P> {
        DrawnPresents {
            answers: HeldAnswers {
                presented: Arc::clone(self),
                drawing: self.answers.drawn(),
                how: LetGo::Finished,
            },
            hold: None,
        }
    }

    /// Whether the display shows a presented surface's texture in place of the scanout.
    fn shows_surface(&self) -> bool {
        self.texture.lock().is_some()
    }

    /// The texture shown, the display's view of it and the hold of its block, for a window draw
    /// of it; `None` for the scanout.
    pub(crate) fn shown(&self) -> Option<ShownRefs<P>> {
        (self.texture.lock().as_ref()).map(|s| {
            let texture = Handle::clone_ref(&s.texture);
            (texture, Arc::clone(&s.view), Arc::clone(&s.hold))
        })
    }

    /// A refresh of the device's display, on its refresh thread: the presents it now shows are
    /// answered, then the textures it showed before them are let go of (`retired`).
    fn refreshed(&self) {
        self.answers.refreshed();
        self.let_go_of_retired("a refresh");
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

/// What a window's draw holds ([`DrawnPresents::after`]): the answers of the presents shown after
/// the latest one when it began, whose surface it may read ([`PresentAnswers::drawn`]), and the
/// hold of the block of the surface it draws. Both go where it is dropped before its commit.
pub(crate) struct DrawnPresents<P: Platform> {
    /// The answers it holds.
    answers: HeldAnswers<P>,
    /// The hold of the block of the surface the draw reads, where it reads one.
    hold: Option<Arc<DisplayHold>>,
}

/// The answers a window's draw holds ([`DrawnPresents`]), let go of as they drop.
struct HeldAnswers<P: Platform> {
    /// Where they are answered.
    presented: Arc<Presented<P>>,
    /// The presents the draw took, and the answers it holds.
    drawing: Drawing,
    /// How the draw let go of them, for their timing lines: [`LetGo::Finished`] but where its
    /// scheduled handler lets go of them.
    how: LetGo,
}

impl<P: Platform> Drop for HeldAnswers<P> {
    fn drop(&mut self) {
        let drawing = std::mem::take(&mut self.drawing);
        self.presented.answers.let_go(drawing, self.how);
    }
}

impl<P: Platform> DrawnPresents<P> {
    /// The draw reads the surface whose block `hold` holds: it holds it until it has finished.
    pub(crate) fn reading(&mut self, hold: Arc<DisplayHold>) {
        self.hold = Some(hold);
    }

    /// Lets go of what the draw holds as `command_buffer`, the draw, which reads the surface,
    /// runs, through its handlers. Called before its commit; where a handler is refused, or the
    /// buffer is never committed, what it holds goes as if it had finished.
    ///
    /// The surface's block goes once the draw has finished on the GPU, its last read of the
    /// block's memory: a completed handler holds it.
    ///
    /// The answers go once the draw is scheduled, in its scheduled handler, where every write of
    /// the surface is the platform's work ([`DisplayHold`]'s `queued`: a device cache's), or
    /// where it reads none. The guest writes a surface again only once an answer the draw held
    /// has freed it (`present`'s module doc), so the write is committed after the draw's
    /// scheduling, and runs after the draw on the GPU, by the rule of the platform's queue: rt
    /// submits every command queue's work, the sink's unshared queue's ([`sink::ScanoutDraw`]'s
    /// `queue`) as the guest's and the caches' transfer queue's, to its device's one `VkQueue`
    /// in commit order, from one thread, each command buffer's segments ending with a global
    /// memory barrier that runs every later command after its own (metal2vulkan-runtime's
    /// README, "Queues, command buffers and events"); and the draw, which presents the window's
    /// drawable (`rt_window`), is never held in a group, and is scheduled only once its
    /// submission has passed `vkQueueSubmit2` (the README, "Presenting"), so a command buffer
    /// committed after it goes in a later submission. A second queue, where the device's queue
    /// family has one, takes only the swapchain's presents. `reims-vgpu-vulkan`'s
    /// `rt_platform::command` test `work_committed_after_a_scheduling_runs_after_it_on_vulkan`
    /// fails where the sink's queue and the guest's stop sharing that order. Where a write of
    /// the surface may go around the queue (the guest's CPU writing the pages or the copy the
    /// texture lies over, the host's video writing a cache), the answers go once the draw has
    /// finished, with the block.
    pub(crate) fn after(self, command_buffer: &P::CommandBuffer) {
        let DrawnPresents { answers, hold } = self;
        let held = !answers.drawing.is_empty();
        let at_scheduling = hold.as_ref().is_none_or(|hold| hold.queued);
        let answers = if held && at_scheduling {
            // rt's one queue (above): the guest's next write of the surface runs after the draw.
            let let_go = move || {
                let mut answers = answers;
                answers.how = LetGo::Scheduled;
                drop(answers);
            };
            no_catch(command_buffer.add_scheduled_handler(OnFault::Die, Box::new(let_go)));
            None
        } else {
            held.then_some(answers)
        };
        if answers.is_none() && hold.is_none() {
            return;
        }
        let finished = Box::new(move |_| drop((answers, hold)));
        no_catch(command_buffer.add_completed_handler(OnFault::Die, finished));
    }
}

impl<P: ApiPlatform> GraphicsControl<P> {
    /// The device made the command queue `poster` posts for, before its create is answered
    /// ([`Host::queue_made`]): what its ring publishes bounds what a present waits for of it, and
    /// what a buffer of another queue waits for of it (`surface_order`, `commit_order`), from
    /// before the guest can name its buffers, however late its service thread has the decoder make
    /// its handle. Nothing of a queue the device's reset ended.
    fn queue_made(&self, poster: &CompletionPoster) {
        let Some(queue) = poster.queue_obj_id() else {
            return;
        };
        let _carried = self.reset_gate.read();
        if poster.stale() {
            return;
        }
        self.presents.queue_made(queue, poster.clone());
        let ring = poster.clone();
        let decoder = self.decoder();
        decoder.surface_order_queue_made(queue, Arc::new(move || ring.published_seqno()));
        let ring = poster.clone();
        decoder.queue_buffers_made(queue, Arc::new(move || ring.published_buffer_seqno()));
    }

    /// Takes every object of context `context` (its process closed the device or died) out of
    /// the decoder's tables, with the surfaces' entries whose objects went with them, and lets go
    /// of them on the decoder's close worker, so the context's lane is done with the close, and
    /// the jobs of other lanes that wait for it go on, while the platform lets them go. `answer`
    /// is given once they are let go of and every command buffer of the context the decoder
    /// committed has finished: the kext then gives the context's blocks, texture resource ids and
    /// GPU addresses out again.
    fn close_context(&self, context: u32, answer: CompleteAnswer) {
        // Its waits' ends and its frees, for a slow answer's line (`slow_answers`).
        let marks = Arc::new(CloseMarks::asked());
        // Its groups of views being formed make nothing (`decode::surface_pages`, "Group
        // dma-bufs"); then after its no-copy buffers being made, so none lands in the table its
        // close empties.
        self.decoder().drop_groups(context);
        self.pages_buffers.wait(pages_context(context));
        marks.pages();
        entry::run(|_| {
            let decoder = self.decoder();
            // Its compiles first, so none stores a pipeline in the table its close empties.
            self.pending.cancel_context(context);
            let (dropped, closed) = decoder.close_tables(&self.device, context);
            self.pool.context_closed(context);
            self.presents.context_gone(context);
            self.queue_sinks
                .write()
                .retain(|&queue, _| decode::context_of(queue) != context);
            // Its blocks of the Shared pool go back to it once its command buffers have finished,
            // before the kext hears of the close's end.
            let gone = (self.shared_pool.as_ref())
                .map(|ledger| (ledger.clone(), ledger.context_closed(context)));
            // And so do its Private blocks.
            let private = self.private_blocks.clone();
            let private_gone = private.context_closed(context);
            let m = Arc::clone(&marks);
            let drain = decoder.drain_context(
                context,
                true,
                Box::new(move || {
                    let drained = host_time();
                    let mut shared = Freed::default();
                    if let Some((ledger, gone)) = gone {
                        shared.blocks = gone.len();
                        ledger.free(&gone);
                    }
                    let shared_freed = host_time();
                    // rt's release of its Private blocks, apart from the ledger's free.
                    let releasing = private.free(&private_gone);
                    let ledger = host_time();
                    let freed = Freed {
                        blocks: releasing.blocks(),
                        bytes: releasing.bytes(),
                    };
                    drop(releasing);
                    let split = m.split(CloseFrees {
                        drained,
                        shared,
                        shared_freed,
                        ledger,
                        released: host_time(),
                        private: freed,
                    });
                    answering(split, || answer.answer(RGFX_STATUS_OK));
                }),
            );
            marks.closed();
            let m = Arc::clone(&marks);
            decoder.let_go_later(closed, drain, move |closed| {
                m.letting_go(closed.objects(), || closed.let_go());
            });
            log::info!(
                target: "reims_vgpu::graphics",
                "context {context} closed: {dropped} objects dropped"
            );
        });
    }

    /// The device's reset (`Host::device_reset`): every context's objects dropped as at its close
    /// ([`Self::close_context`]), with no answer owed, and everything else the decoder keeps for
    /// the guest's processes ([`GraphicsDecoder::reset`]); the pipelines compiling are not stored,
    /// a dispatch that waits for its stage's compile outside the process is refused, that compile
    /// cancelled, the presents waiting are dropped, and the display shows the scanout in place of
    /// any surface. The device is then as one just made, to the guest: the ids it picks, the
    /// texture resource ids and samplers' ids it is handed, and what a library lookup finds.
    fn reset(&self) {
        entry::run(|_| {
            let (contexts, dropped) = {
                // After everything of the guest's that began before the reset has ended, its
                // waits for other queues' buffers cut short, and its dispatches' waits for a
                // compile outside the process (`GpuDevice::reset_begins`).
                self.decoders.resetting();
                self.device.reset_begins();
                // And command buffers' waits on their lanes for pipelines still compiling, which
                // hold the gate.
                self.pending.resetting();
                let _alone = self.reset_gate.write();
                self.pending.reset();
                let decoder = self.decoder();
                // After the closes before it let go of their objects, as the guest's next creates
                // may take the texture resource ids and GPU addresses they held.
                decoder.closes_let_go();
                let (contexts, dropped) = decoder.reset(&self.device);
                for &context in &contexts {
                    self.pool.context_closed(context);
                }
                // Every block of the Shared pool goes back to it, the objects over them dropped.
                if let Some(ledger) = &self.shared_pool {
                    ledger.reset();
                }
                // And every Private block, the objects in them dropped.
                self.private_blocks.reset();
                self.presents.reset();
                self.queue_sinks.write().clear();
                for color in &self.colors {
                    color.lock().reset();
                }
                self.decoders.reset_done();
                self.device.reset_ends();
                (contexts, dropped)
            };
            for presented in &self.presented {
                presented.reset();
            }
            log::info!(
                target: "reims_vgpu::graphics",
                "the device was reset: {dropped} objects of {} contexts dropped",
                contexts.len()
            );
            // What the boot before the reset degraded, and the lines it repeated
            // (docs/conventions.md).
            reims_vgpu_vulkan::answer::write_counts();
        });
    }

    /// `BLOCK_RELEASE` (`Host::block_released`): refused at once where an object the decoder
    /// holds lies in the block, else answered once every command buffer of the block's context
    /// the decoder committed has finished, every one another holder's context committed before
    /// its `BLOCK_UNSHARE`, its close, or the release where neither came first, which the device
    /// hands on as an unshare ([`Self::unshare_block`]), and once the display no longer shows
    /// or draws a surface's texture over the block ([`DisplayHold`]).
    fn release_block(&self, release: &RgfxBlockRelease, answer: CompleteAnswer) {
        // Its waits' ends, for a slow answer's line (`slow_answers`).
        let marks = Arc::new(ReleaseMarks::asked());
        // After the no-copy buffers being made over it, which keep it.
        self.pages_buffers.wait(release.offset);
        marks.pages();
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
        // Once no display shows or draws a surface over the block.
        let m = Arc::clone(&marks);
        let answer: DrainAnswer = Box::new(move || {
            let split = m.split();
            answering(split, || answer.answer(RGFX_STATUS_OK));
        });
        let answer = (self.presented.iter()).fold(answer, |answer, presented| {
            presented.after_shown(release.offset, release.size, answer)
        });
        let m = Arc::clone(&marks);
        let answer: DrainAnswer = Box::new(move || {
            m.unshared();
            answer();
        });
        let answer = decoder.after_unshares(release.offset, answer);
        let m = Arc::clone(&marks);
        let answer: DrainAnswer = Box::new(move || {
            m.drained();
            answer();
        });
        let drain = decoder.drain_context(release.context, false, answer);
        // No buffer lies over the block's surface view any more: it goes with the block, and its
        // device cache once the surface's pages hold what was written of it, which the answer
        // waits for. The copy is made once the buffers still walked that may write the cache
        // have finished, on the last one's thread: the context's lane goes on.
        let held = Arc::clone(&drain);
        let (flags, offset) = (release.flags, release.offset);
        let m = Arc::clone(&marks);
        entry::run(|_| {
            decoder.block_released(
                &self.device,
                offset,
                flags,
                || marks.settled(),
                Box::new(move || {
                    m.written();
                    writing_back(|| drop(held));
                }),
                Box::new(move |raise| {
                    log::warn!(
                        target: "reims_vgpu::graphics", kind:% = raise;
                        "the block at {offset:#x}: the platform raised at its device cache's \
                         write back ({raise:?})"
                    );
                }),
            );
        });
        marks.released();
        drop(decoder);
        drop(drain);
    }

    /// `POOL_TAKE` (`Host::pool_take`): context `take.context`'s block of the Shared pool, carved
    /// now where a piece has room and answered from the context's take worker once it is zeroed
    /// ([`PoolLedger::take`]); refused where the pool has no room, or the device offers none.
    fn take_pool_block(&self, take: &RgfxPoolTake, answer: PoolAnswer) {
        let Some(ledger) = &self.shared_pool else {
            answer.answer(RGFX_STATUS_REFUSED, 0, 0, 0);
            return;
        };
        if take.length == 0 || take.length > TAKE_MAX {
            answer.answer(RGFX_STATUS_INVALID_ARGUMENT, 0, 0, 0);
            return;
        }
        let (context, length) = (take.context, take.length);
        let given = ledger.take(
            context,
            length,
            Box::new(move |block| match block {
                Some(block) => answer.answer(
                    RGFX_STATUS_OK,
                    block.offset,
                    block.length,
                    block.gpu_address,
                ),
                None => {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "context {context} asked for {length:#x} bytes of the Shared pool, which \
                         has no room for them: refused"
                    );
                    answer.answer(RGFX_STATUS_REFUSED, 0, 0, 0);
                }
            }),
        );
        debug_assert!(given.is_ok(), "a take's length was checked");
    }

    /// `POOL_GIVE` (`Host::pool_give`): not found where context `give.context` holds no such block
    /// of the Shared pool, refused at once where an object the decoder holds lies in it, else
    /// answered, and the block freed, once every command buffer of the context the decoder
    /// committed has finished ([`PoolLedger::give`]).
    fn give_pool_block(&self, give: &RgfxPoolGive, answer: CompleteAnswer) {
        // For a slow answer's line (`slow_answers`).
        let asked = host_time();
        let Some(ledger) = &self.shared_pool else {
            answer.answer(RGFX_STATUS_NOT_FOUND);
            return;
        };
        let decoder = self.decoder();
        let given = ledger.give(
            give.context,
            give.pool_offset,
            give.length,
            |offset, length| {
                decode::POOL_SPANS
                    .checked_add(offset)
                    .is_some_and(|span| decoder.span_in_use(span, length))
            },
        );
        match given {
            Give::NotFound => answer.answer(RGFX_STATUS_NOT_FOUND),
            Give::InUse => {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "context {} gave back its block of the Shared pool of {:#x} bytes at {:#x} \
                     while an object lies in it: the block is kept",
                    give.context, give.length, give.pool_offset
                );
                answer.answer(RGFX_STATUS_REFUSED);
            }
            Give::Given(carve) => {
                let given = host_time();
                let length = give.length;
                let ledger = ledger.clone();
                let drain = decoder.drain_context(
                    give.context,
                    false,
                    Box::new(move || {
                        let drained = host_time();
                        ledger.free(&[carve]);
                        let split = Split::Give {
                            asked,
                            given,
                            drained,
                            freed: host_time(),
                            bytes: length,
                        };
                        answering(split, || answer.answer(RGFX_STATUS_OK));
                    }),
                );
                drop(decoder);
                drop(drain);
            }
        }
    }

    /// `PRIVATE_TAKE` (`Host::private_take`): context `take.context`'s new Private block
    /// ([`PrivateBlocks::take`]): the host's pick answered at once, the source's block on the
    /// context's take worker, after its earlier takes; invalid for a length of 0 or past the
    /// wire's longest, refused where no block or address is left for it.
    fn take_private_block(&self, take: &RgfxPrivateTake, answer: PrivateAnswer) {
        let (context, asked) = (take.context, take.length);
        let answer: PrivateTakeAnswer = Box::new(move |taken| match taken {
            Ok((gpu_address, length)) => {
                static TOLD: AtomicBool = AtomicBool::new(false);
                if !TOLD.swap(true, Ordering::Relaxed) {
                    log::info!(
                        target: "reims_vgpu::graphics",
                        "the first Private block: context {context}'s, {length:#x} bytes at \
                         {gpu_address:#x}"
                    );
                }
                answer.answer(RGFX_STATUS_OK, gpu_address, length);
            }
            Err(PrivateTakeError::Invalid) => answer.answer(RGFX_STATUS_INVALID_ARGUMENT, 0, 0),
            Err(PrivateTakeError::Refused) => {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "context {context} asked for a Private block of {asked:#x} bytes, for which \
                     no block or address is left: refused"
                );
                answer.answer(RGFX_STATUS_REFUSED, 0, 0);
            }
        });
        self.private_blocks.take(context, asked, answer);
    }

    /// `PRIVATE_GIVE` (`Host::private_give`): not found where context `give.context` holds no
    /// such Private block, refused at once where a buffer or heap the decoder holds lies in it,
    /// else answered, and the block freed, once every command buffer of the context the decoder
    /// committed has finished ([`PrivateBlocks::give`]).
    fn give_private_block(&self, give: &RgfxPrivateGive, answer: CompleteAnswer) {
        // For a slow answer's line (`slow_answers`).
        let asked = host_time();
        let decoder = self.decoder();
        let given =
            (self.private_blocks).give(give.context, give.gpu_address, give.length, |at, len| {
                decoder.addresses_in_use(at, len)
            });
        match given {
            PrivateGive::NotFound => answer.answer(RGFX_STATUS_NOT_FOUND),
            PrivateGive::InUse => {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "context {} gave back its Private block of {:#x} bytes at {:#x} while an \
                     object lies in it: the block is kept",
                    give.context, give.length, give.gpu_address
                );
                answer.answer(RGFX_STATUS_REFUSED);
            }
            PrivateGive::Given(block) => {
                let given = host_time();
                let blocks = self.private_blocks.clone();
                let drain = decoder.drain_context(
                    give.context,
                    false,
                    Box::new(move || {
                        let drained = host_time();
                        // rt's release of the block, apart from the ledger's free.
                        let releasing = blocks.free(&[block]);
                        let ledger = host_time();
                        let freed = Freed {
                            blocks: releasing.blocks(),
                            bytes: releasing.bytes(),
                        };
                        drop(releasing);
                        let split = Split::PrivateGive {
                            asked,
                            given,
                            drained,
                            ledger,
                            released: host_time(),
                            freed,
                        };
                        answering(split, || answer.answer(RGFX_STATUS_OK));
                    }),
                );
                drop(decoder);
                drop(drain);
            }
        }
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
    /// message that names the block and waited for it is answered after it. A pages block of a
    /// guest that took `IMPORT_ADDRESSES_LATER` is answered as soon as its runs are carried out,
    /// with no address ([`CompleteAnswer::answers_later`]), and its view is made and given its
    /// address after the answer, on the same job, before the block settles: the messages that
    /// name the block wait for that on the host's side, and the guest's call for none of it.
    fn start_surface_pages(
        self: &Arc<Self>,
        pages: RgfxSurfacePages,
        runs: Vec<RgfxPageRun>,
        flags: u32,
        layout: Option<RgfxSurfacePagesLayout>,
        answer: CompleteAnswer,
    ) {
        let begun = self.decoder().surface_pages_begun();
        // After the no-copy buffers being made over the block it describes again, which see it as
        // before, and whose make on the surface worker would else wait for it, behind them there.
        self.pages_buffers.wait(pages.block_offset);
        // After the copies of a released block's device cache into the same pages, which would
        // land over what the new block's buffers write.
        (self.decoder()).wait_copies_into(pages.block_offset, &runs);
        let me = Arc::clone(self);
        let job = move |settle| {
            let wants_address = answer.wants_address();
            let (status, at, answer) = {
                let _carried = me.reset_gate.read_recursive();
                let status = (me.decoder())
                    .carry_out_surface_pages(&me.device, &pages, &runs, flags, layout, &settle);
                // Where the guest took IMPORT_ADDRESSES_LATER, answered now, with no address,
                // before the view is made and given one below.
                let answer = match status {
                    Some(status) if answer.answers_later() => {
                        me.decoder().surface_pages_answered(begun);
                        answer.answer_at(status, 0);
                        None
                    }
                    _ => Some(answer),
                };
                // A pages block's view's address, where the guest took IMPORT_ADDRESSES.
                let at = match status {
                    Some(RGFX_STATUS_OK) if wants_address => {
                        (me.decoder()).pages_address(&me.device, pages.block_offset)
                    }
                    _ => 0,
                };
                (status, at, answer)
            };
            // Nothing is owed where the device was reset since, nor where it was answered above.
            if let (Some(status), Some(answer)) = (status, answer) {
                me.decoder().surface_pages_answered(begun);
                answer.answer_at(status, at);
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
        let (answer, done, failed) = download_answers(block, answer);
        entry::run(|_| {
            (self.decoder()).surface_download(&self.device, block, done, failed);
        });
        drop(answer);
    }

    /// [`Self::download_surface`], on the control thread, where the block's lane has nothing
    /// left ([`Host::try_surface_download`]): carried out here where it waits for nothing
    /// ([`GraphicsDecoder::try_surface_download`]), else `answer` back, for the lane.
    fn try_download_surface(
        &self,
        block: u64,
        answer: CompleteAnswer,
    ) -> Result<(), CompleteAnswer> {
        let (answer, done, failed) = download_answers(block, answer);
        let taken = entry::run(|_| {
            (self.decoder()).try_surface_download(&self.device, block, done, failed)
        });
        if taken {
            return Ok(());
        }
        // Untouched where left: the answers were dropped ungiven.
        answer.lock().take().map_or(Ok(()), Err)
    }

    /// [`Self::try_download_surface`] on a doorbell's thread, the guest's vCPU
    /// ([`Host::surface_downloaded`]): whether the download of the block at `block` is carried
    /// out already, nothing to copy or wait for ([`Decoders::downloaded_at_doorbell`]), with no
    /// reset under way (the reset gate tried). Allocates and frees nothing.
    fn downloaded_at_doorbell(&self, block: u64) -> bool {
        let Some(_gate) = self.reset_gate.try_read() else {
            return false;
        };
        self.decoders.downloaded_at_doorbell(block)
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
    /// pipeline still compiling is begun here all the same, and waits for it on its queue's lane,
    /// where it is then walked and committed ([`Self::begin_command_buffer`]). An indirect command
    /// buffer's create is made on the ICB worker ([`Self::start_icb`]), and the destroy of one
    /// still being made waits for it. A library lookup is answered here
    /// ([`Self::lookup_library`]).
    ///
    /// `bytes` is the message, borrowed, or the device's own copy of a command buffer's
    /// ([`Host::command_buffer`]), which goes to the buffer's lane as it is, with `claim`, its
    /// queue's claim on its messages, which a buffer carried out here lets go once begun
    /// ([`Self::carry_out`]).
    fn handle_stage3(
        self: &Arc<Self>,
        poster: &CompletionPoster,
        bytes: Cow<'_, [u8]>,
        claim: Option<&mut QueueClaim<'_>>,
    ) {
        let message: &[u8] = &bytes;
        let r#type = RgfxMsgHeader::read_from_prefix(message).map_or(0, |(h, _)| h.r#type);
        if (r#type == RGFX_CMD_BUFFER_OVER_PAGES || r#type == RGFX_CMD_HEAP_OVER_PAGES)
            && self.start_over_pages(poster, message)
        {
            return;
        }
        // A message of a context whose no-copy buffer is being made on the surface worker, which
        // it may name, is carried out after it.
        self.wait_for_pages_buffers(poster, message);
        if poster.queue_obj_id().is_none() {
            self.regroup_step(poster, message);
        }
        match r#type {
            RGFX_CMD_RENDER_PIPELINE_CREATE
            | RGFX_CMD_MESH_RENDER_PIPELINE_CREATE
            | RGFX_CMD_COMPUTE_PIPELINE_CREATE => {
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
            RGFX_CMD_PIPELINE_ANSWERS => {
                self.pipeline_answers(poster, message);
                return;
            }
            RGFX_CMD_PAGES_BUFFER_ADDRESS => {
                self.pages_buffer_address(poster, message);
                return;
            }
            RGFX_CMD_RATE_MAP_CREATE | RGFX_CMD_RATE_MAP_RESET => {
                self.rate_map_state(poster, message);
                return;
            }
            RGFX_CMD_RATE_MAP_QUERY => {
                self.rate_map_query(poster, message);
                return;
            }
            RGFX_CMD_FUNCTION_HANDLE_CREATE | RGFX_CMD_FUNCTION_TABLE_CREATE => {
                self.function_pointer_state(poster, message);
                return;
            }
            RGFX_CMD_DYNAMIC_LIBRARY_CREATE => {
                self.dynamic_library_create(poster, message);
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
            RGFX_CMD_SHARED_EVENT_NAME
            | RGFX_CMD_SHARED_EVENT_IMPORT
            | RGFX_CMD_SHARED_EVENT_UNNAME => {
                self.share_event(poster, message);
                return;
            }
            RGFX_CMD_COMMAND_BUFFER => {
                if !foreign_queue(poster, message) {
                    self.begin_command_buffer(poster, bytes.into_owned(), claim);
                    return;
                }
            }
            RGFX_CMD_ICB_WRITE => self.pending.wait_for_icb_write(message),
            RGFX_CMD_ICB_SLOTS => self.pending.wait_for_icb_slots(&self.region, message),
            // Its slots lie in pages blocks, which the sender's decoder finds, as it would carry
            // the message out.
            RGFX_CMD_ICB_SLOTS_IN_PAGES => self.pending.wait_for_named(|| {
                let _carried = self.reset_gate.read();
                if poster.stale() {
                    return Vec::new();
                }
                (self.decoders)
                    .sender(Some(sender_context(poster, message)), Sinks::default())
                    .icb_slots_in_pages_pipelines(&self.device, message)
            }),
            // The destroy of an indirect command buffer still being made (the kext destroys
            // behind a create whose submitting thread was aborted before the answer, or one made
            // for a context that had closed) waits for its create, so it and everything after it
            // keep their places in ring order.
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
                    // A queue's destroy: nothing more of it finishes for a present to wait for,
                    // and no more of its buffers is begun.
                    self.presents.queue_gone(obj_id);
                    self.queue_sinks.write().remove(&obj_id);
                }
                let foreign = foreign_queue(poster, message);
                // An ICB_WRITE's records and an ICB_SLOTS's slots run on the control ring, ahead
                // of the process's later command buffers: how long they take is logged
                // (diagnostics).
                let icb_start = matches!(
                    r#type,
                    RGFX_CMD_ICB_WRITE | RGFX_CMD_ICB_SLOTS | RGFX_CMD_ICB_SLOTS_IN_PAGES
                )
                .then(|| self.pool.now());
                let outcome = if foreign {
                    Ok(Err(DecodeError::BadReference))
                } else {
                    decoder.decode_refusing(&self.device, message)
                };
                if let Some(start) = icb_start {
                    log::debug!(
                        target: "reims_vgpu::graphics::icb",
                        "{} of {} bytes applied in {} us",
                        match r#type {
                            RGFX_CMD_ICB_SLOTS => "ICB_SLOTS",
                            RGFX_CMD_ICB_SLOTS_IN_PAGES => "ICB_SLOTS_IN_PAGES",
                            _ => "ICB_WRITE",
                        },
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
        // Each posted waiting for nothing, on whichever thread finds the buffer complete
        // (`CompletionPoster::post_unwaiting`).
        let completion: CompletionSink = Arc::new(move |record: RgfxCbComplete, then: Then| {
            let seqno = record.header.seqno;
            let (presents, poster) = (Arc::clone(&presents), queue_poster.clone());
            let then: Then = Box::new(move || {
                // A present waiting for the buffer is shown once its completion is posted.
                if let Some(queue) = queue {
                    presents.finished(queue, seqno, &poster);
                }
                then();
            });
            queue_poster.post_unwaiting(record.as_bytes(), Some(then));
        });
        let signal_poster = poster.clone();
        let signal: SignalSink = Arc::new(move |record: RgfxEventSignalled, then: Then| {
            signal_poster.post_unwaiting(record.as_bytes(), Some(then));
        });
        // A buffer's submission handler posts these, on whichever thread runs it
        // ([`post_from_handler`]).
        let scheduled_poster = poster.clone();
        let scheduled: ScheduledSink = Arc::new(move |record: RgfxCbScheduled| {
            post_from_handler(&scheduled_poster, record.as_bytes());
        });
        let submitted_poster = poster.clone();
        let submitted: SubmittedSink = Arc::new(move |record: RgfxCbSubmitted| {
            post_from_handler(&submitted_poster, record.as_bytes());
        });
        let reflection_poster = poster.clone();
        let reflection: ReflectionSink = Arc::new(move |reply: PipelineReply| {
            reflection_poster.post(&reply.bytes());
        });
        // A completion posted at its decode shows no present: one waiting for the buffer is shown
        // at its end, once its decode's copy is done.
        let ahead = queue.map(|queue| {
            let ahead_poster = poster.clone();
            let post: CompletionSink = Arc::new(move |record: RgfxCbComplete, then: Then| {
                ahead_poster.post_unwaiting(record.as_bytes(), Some(then));
            });
            let (presents, finished_poster) = (Arc::clone(&self.presents), poster.clone());
            let finished: FinishedSink = Arc::new(move |seqno| {
                presents.finished(queue, seqno, &finished_poster);
            });
            AheadSinks { post, finished }
        });
        Sinks {
            completion: Some(completion),
            signal: Some(signal),
            scheduled: Some(scheduled),
            submitted: Some(submitted),
            reflection: Some(reflection),
            ahead,
        }
    }

    /// [`Self::sinks`] of a command queue's `poster`, for a buffer begun on the queue's service
    /// thread ([`Self::begin_command_buffer`]): made at the queue's first buffer, and shared by
    /// each buffer after it whose poster posts as theirs does ([`CompletionPoster::posts_as`]).
    /// What sinks post depends on their poster and the presents alone, so shared ones post what
    /// new ones would. Made at each begin, they would cost the begin seven closures and eight
    /// posters (measured with five and six: 0.30 us of a buffer's 14.9 us from its take to its
    /// begin on an Iris Xe's queue thread, in Geekbench 7's Super Resolution), and its lane their
    /// release after the walk.
    fn queue_sinks(&self, poster: &CompletionPoster) -> Sinks {
        let Some(queue) = poster.queue_obj_id() else {
            return self.sinks(poster);
        };
        if let Some((made_for, sinks)) = self.queue_sinks.read().get(&queue)
            && made_for.posts_as(poster)
        {
            return sinks.clone();
        }
        let sinks = self.sinks(poster);
        self.queue_sinks
            .write()
            .insert(queue, (poster.clone(), sinks.clone()));
        sinks
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
                    target: "reims_vgpu::graphics", kind:? = e;
                    "a message (type {ty:#x}, seqno {seqno}, object {obj_id:#x}) was not \
                     carried out: {e:?}"
                );
                e.count();
                if !stood_in {
                    self.fail_queue_command_buffer(poster, message);
                }
            }
            Err(raise) => {
                let (ty, seqno, obj_id) = message_identity(message);
                log::warn!(
                    target: "reims_vgpu::graphics", kind:% = raise;
                    "a message (type {ty:#x}, seqno {seqno}, object {obj_id:#x}) faulted: \
                     {raise}"
                );
                reims_vgpu_vulkan::answer::count_kind(decode::FAULTED);
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
    /// going there as it is. A buffer longer than [`INLINE_WALK_MAX`] has the lane's idle thread
    /// woken for it as its turn is taken, so the thread wakes during the rest of its begin
    /// (`decode::lane`, "Woken as the buffer is begun"), unless it is alone on its queue's ring:
    /// [`INLINE_WALK_ALONE_MAX`] bytes at most, with nothing published behind it as it is taken
    /// ([`QueueClaim::nothing_published`]), so that it is walked here where its commit would wait
    /// for nothing. One that names a pipeline still compiling is begun all the same, and waits for
    /// it on its lane before its walk ([`Self::carry_out`]).
    fn begin_command_buffer(
        self: &Arc<Self>,
        poster: &CompletionPoster,
        message: Vec<u8>,
        claim: Option<&mut QueueClaim<'_>>,
    ) {
        entry::run(|_| {
            // Held to the end of the buffer's work, on its lane ([`Self::reset_gate`]).
            let carried = self.reset_gate.read_arc();
            // Of a queue the device's reset ended, which the decoder's reset dropped, or is to
            // drop: nothing of it is carried out, and nothing is owed.
            if poster.stale() {
                return;
            }
            let sinks = self.queue_sinks(poster);
            let mut decoder = (self.decoders).sender(Some(sender_context(poster, &message)), sinks);
            // Alone on its ring as it is taken: nothing behind it waits for the service thread.
            let alone = message.len() <= self.inline_walk_alone_max.load(Ordering::Relaxed)
                && claim
                    .as_ref()
                    .is_some_and(|claim| claim.nothing_published());
            let to_lane = self.goes_to_lane(&message) && !alone;
            let queue = poster.queue_obj_id();
            // Before the begin takes the message's views of the tables: a pipeline stored between
            // the two is in both, and none is in neither.
            let awaited = self.pending.named_by(&message);
            let begun = refusing(|| decoder.begin_taken(&self.device, &message, queue, to_lane));
            let (prepass, made) = match &begun {
                Ok(Ok(b)) if b.computes() => self.hand_prepass(&decoder, &message),
                _ => (None, true),
            };
            let walk = Walk {
                to_lane,
                alone,
                made,
                awaited,
            };
            let begun = (begun, prepass, walk);
            self.carry_out(decoder, begun, carried, poster.clone(), message, claim);
        });
    }

    /// The pre-pass of the `COMMAND_BUFFER` `message` `decoder` began
    /// ([`GraphicsDecoder::prepass`]), and whether every dispatch the buffer runs is made: each
    /// shape it hands over is one compile of the pool's, which its walk does not wait for. A
    /// shape the pool takes no compile of, its threads gone, is left to the walk.
    fn hand_prepass(
        &self,
        decoder: &GraphicsDecoder<P>,
        message: &[u8],
    ) -> (Option<Prepass<P>>, bool) {
        let Ahead { prepass, made } = decoder.prepass(message);
        if let Some(prepass) = &prepass {
            let of = CompileOf::new(CompileKind::Prepass, prepass.cb_obj_id());
            for _ in 0..prepass.handed() {
                if self.pool.submit(of, prepass.job()).is_err() {
                    break;
                }
            }
        }
        (prepass, made)
    }

    /// The rest of the `COMMAND_BUFFER` `message`'s work once it is begun (`begun`, on `decoder`,
    /// with its pre-pass, where it hands one), on its queue's lane where it took a turn there
    /// ([`GraphicsDecoder::lane`]), else here: it
    /// is walked and committed in its turn ([`GraphicsDecoder::finish_command_buffer`]), its
    /// pre-pass then closed before the decoder lets go of anything ([`Prepass::close`]), stood in
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
    ///
    /// So is a short buffer, its message of [`INLINE_WALK_MAX`] bytes at most, whose commit would
    /// wait for nothing ([`GraphicsDecoder::commits_at_once`]): its turn is next, and no buffer of
    /// its process it is committed after is still to be committed. Its walk takes about what the
    /// lane thread's wake would, and its commit then follows at once, where on the lane each
    /// buffer of a drain (a guest's buffers committed back to back and waited for) paid a wake
    /// for its walk and another for the turn its commit waited on. The queue's next buffer is
    /// begun once it is committed, a walk of a few dozen ops later.
    ///
    /// So is a buffer of [`INLINE_WALK_ALONE_MAX`] bytes at most whose commit would wait for
    /// nothing and which was alone on its queue's ring as it was taken ([`Walk::alone`]): no
    /// buffer of the queue was there to be taken beside a lane's walk of it,
    /// and the lane's thread was not woken for it. Where its commit would wait after all, it
    /// goes to the lane as a short buffer does, its thread woken there.
    ///
    /// None of these is walked here where a dispatch of its runs a pipeline its pre-pass did not
    /// find made ([`Walk::made`]): that walk may wait for the pipeline's make, or for its stage's
    /// fill, a minute or two (`reims_vgpu_vulkan::core::platform::stage`), and here the queue's
    /// next messages would wait behind it. It goes to the lane, whose thread waits out of the
    /// lane's walks (`decode::lane`), the queue's next buffers taken and walked meanwhile. So does
    /// one that names a pipeline still compiling as it was begun ([`Walk::awaited`]), whose
    /// compile may wait for its stage's fill as long: its walk first waits there for the
    /// compile's store, or its cancel, or the device's reset ([`Pending::wait_on_lane`]), and its
    /// views of the tables then name the pipelines stored since
    /// ([`GraphicsDecoder::see_stored`]), as views taken after the stores would.
    ///
    /// A buffer carried out here lets its queue's `claim` go first, once begun
    /// ([`QueueClaim::let_go`]): a buffer the guest published behind it and asked `CB_SUBMITTED`
    /// of is offered to the host then, and one rung for during the walk at its doorbell, each
    /// answered where it waits for nothing, as while a lane's thread walks the one before it.
    /// Held through the walk, the claim kept the doorbell from answering it, and the answer came
    /// only at the service thread's take after the walk and the commit: measured on the RTX
    /// 5080 laptop host, a guest's `commitAndWaitUntilSubmitted` of a tiny buffer behind one of
    /// fills took 32-36 µs at the median and 70-240 at the 90th percentile, against 18 and
    /// 27-35 with that walk on the lane.
    fn carry_out(
        self: &Arc<Self>,
        decoder: GraphicsDecoder<P>,
        (begun, prepass, walk): BegunBuffer<P>,
        carried: ArcRwLockReadGuard<RawRwLock, ()>,
        poster: CompletionPoster,
        message: Vec<u8>,
        claim: Option<&mut QueueClaim<'_>>,
    ) {
        let lane = decoder.lane();
        let expected = walk.to_lane;
        let begun_whole = matches!(&begun, Ok(Ok(_)));
        let here = match &begun {
            Ok(Ok(_)) if !walk.made || !walk.awaited.is_empty() => false,
            Ok(Ok(b)) if b.ordered() => decoder.turn_is_next(),
            Ok(Ok(_)) => {
                (message.len() <= self.inline_walk_max.load(Ordering::Relaxed) || walk.alone)
                    && decoder.commits_at_once()
            }
            _ => false,
        };
        let control = Arc::clone(self);
        let awaited = walk.awaited;
        let job = move || {
            entry::run(|_| {
                let mut decoder = decoder;
                // The pipelines it names that were still compiling as it was begun, waited for out
                // of the lane's walks; its views then name those stored since.
                if matches!(&begun, Ok(Ok(_))) && !awaited.is_empty() {
                    decoder.waiting(|| control.pending.wait_on_lane(&awaited));
                    // Of a queue the device's reset ended as it waited: nothing of it is carried
                    // out, and nothing is owed.
                    if poster.stale() {
                        if let Some(prepass) = prepass {
                            prepass.close(WalkEnd::Unwalked, || control.pool.now());
                        }
                        drop(begun);
                        drop(decoder);
                        drop(carried);
                        return;
                    }
                    decoder.see_stored(&awaited);
                }
                // Its walk makes, at its first dispatch, the shapes no compile thread took.
                decoder.make_at_first_dispatch(prepass.as_ref().map(Prepass::untaken));
                let outcome = match begun {
                    Ok(Ok(begun)) => {
                        refusing(|| decoder.finish_command_buffer(&control.device, &message, begun))
                    }
                    Ok(Err(e)) => Ok(Err(e)),
                    Err(raise) => Err(raise),
                };
                decoder.make_at_first_dispatch(None);
                // Nothing of its pre-pass outlives the views of the table its message holds.
                if let Some(prepass) = prepass {
                    prepass.close(WalkEnd::of(&outcome), || control.pool.now());
                }
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
            lane => {
                // A long buffer walked in its turn here: the lane's thread woken for it as it was
                // begun waits for it no more.
                if let Some(lane) = lane.filter(|_| expected) {
                    lane.withdraw();
                }
                // Begun, its place among the queue's buffers taken: the queue's next buffer may
                // be answered while this one is walked here, as while a lane's thread walks it.
                if let Some(claim) = claim.filter(|_| begun_whole) {
                    claim.let_go();
                }
                job();
            }
        }
    }

    /// Whether the `COMMAND_BUFFER` `message` goes to its queue's lane wherever it is not walked
    /// in its turn with its turn next, nor alone on its ring ([`Self::carry_out`]): it is longer
    /// than the service thread walks itself, [`INLINE_WALK_MAX`] bytes.
    fn goes_to_lane(&self, message: &[u8]) -> bool {
        message.len() > self.inline_walk_max.load(Ordering::Relaxed)
    }

    /// [`Host::answer_command_buffer`]: answers the `COMMAND_BUFFER` `message` of the queue
    /// `poster` posts onto `CB_SUBMITTED`, on a doorbell's thread or on the lane's thread that ends
    /// its queue's thread's wait for ring 0, before the queue's service thread hands it over,
    /// where it gets that at its take and nothing need be waited for or allocated (`decode`'s
    /// [`Decoders::answer_at_doorbell`]): it runs on that queue, no
    /// reset is under way, every lock it takes is free, each tried, no pipeline it names is still
    /// compiling, and no no-copy buffer of its context is being made on the surface worker
    /// ([`Self::start_over_pages`]), which its begin waits for. Its `CB_SUBMITTED` is
    /// posted through `poster` waiting for nothing, and its `CB_SCHEDULED` after it where no buffer
    /// before it on its queue owes its own, once
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
        (self.decoders).answer_at_doorbell(context, message, |record| poster.try_post(record))
    }

    /// [`Host::scheduled_ask`]: a thread of the process of the queue `poster` posts onto waits for
    /// the queue's command buffer the `SCHEDULED_ASK` `message` names to be scheduled, on the
    /// queue's service thread, which began that buffer before it: its `CB_SCHEDULED` is posted
    /// through `poster` where it may be now ([`GraphicsDecoder::ask_scheduled`]). Nothing is
    /// posted for the ask itself, nor where a reset ended the queue.
    fn scheduled_ask(&self, poster: &CompletionPoster, message: &[u8]) {
        let (Ok((ask, _)), Some(queue)) = (
            RgfxScheduledAsk::read_from_prefix(message),
            poster.queue_obj_id(),
        ) else {
            return;
        };
        entry::run(|_| {
            let _carried = self.reset_gate.read();
            if poster.stale() {
                return;
            }
            let decoder =
                (self.decoders).sender(Some(sender_context(poster, message)), Sinks::default());
            decoder.ask_scheduled(queue, ask.cb_seqno, ask.cb_obj_id);
        });
    }

    /// A pipeline create from ring 0: prepared here, on its context's lane, with the decoder's
    /// table (a malformed create or one naming no library is refused here, unanswered, as the
    /// decoder refuses it), marked pending, and compiled on the compile pool, so ring 0 goes on
    /// meanwhile; its reply goes through `poster` when the compile has made its answers, before
    /// its store ([`Self::compile`], [`Self::settle`]).
    fn start_pipeline(self: &Arc<Self>, poster: &CompletionPoster, message: &[u8]) {
        let prepared = entry::run(|_| {
            let decoder = self
                .decoders
                .sender(Some(sender_context(poster, message)), Sinks::default());
            let prepared = decoder
                .prepare_pipeline(message)
                .map(|job| (self.pending.begin(job.obj_id), job));
            // One the bundle posted unanswered stands refused: its draws are dropped, not its
            // command buffers (`decode::refused`). Under the reset gate, as every store is.
            if prepared.is_err() {
                let _carried = self.reset_gate.read();
                if !poster.stale() {
                    let (ty, _, obj_id) = message_identity(message);
                    decoder.refuse(obj_id, ty);
                }
            }
            prepared
        });
        let (began, job) = match prepared {
            Ok(prepared) => prepared,
            Err(e) => {
                let (ty, seqno, obj_id) = message_identity(message);
                log::warn!(
                    target: "reims_vgpu::graphics", kind:? = e;
                    "a pipeline create (type {ty:#x}, seqno {seqno}, object {obj_id:#x}) was not \
                     carried out: {e:?}"
                );
                e.count();
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
            .submit(of, Box::new(move || me.compile(job, began, poster, false)))
        {
            compile();
        }
    }

    /// Compiles a prepared pipeline create, pending since reset count `began`, under a library
    /// entry, with no lock held, and settles it ([`Self::settle`]).
    ///
    /// A compute create whose make the platform refused for a stage whose compile runs outside
    /// the process, naming the stage's fill ([`stage::making`]), is held instead: it stays
    /// pending, the pool's thread let go, until that fill ends ([`Self::held_end`]), and is then
    /// made again, `remade`, once. A remade create settles whatever its make answers, a refusal
    /// counted ([`compile::HELD_CREATE_REFUSED`]).
    fn compile(
        self: &Arc<Self>,
        job: PipelineJob<P>,
        began: u64,
        poster: CompletionPoster,
        remade: bool,
    ) {
        entry::run(|_| {
            let creating = Creating::of(&job, began);
            let again = if remade { None } else { job.again() };
            let (compiled, fill) = stage::making(|| compile_refusing::<P>(&self.device, job));
            match (fill, again) {
                (Some(fill), Some(again)) if compiled.is_refused() => {
                    log::info!(
                        target: "reims_vgpu::graphics",
                        "a pipeline create (seqno {}, object {:#x}) waits for its stage's fill",
                        creating.seqno,
                        creating.obj_id
                    );
                    let me = Arc::clone(self);
                    fill.on_end(move |end| me.held_end(end, again, creating, poster, compiled));
                }
                _ => {
                    if remade && compiled.is_refused() {
                        reims_vgpu_vulkan::answer::count_kind(compile::HELD_CREATE_REFUSED);
                    }
                    self.settle(compiled, creating, &poster)
                }
            }
        });
    }

    /// The end of the fill a held compute create waits for ([`Self::compile`]), on the thread
    /// that ends it, or at once where it had ended. Where it filled the store and no destroy,
    /// close or reset cancelled the create, its make, `again`, is made again on the pool; after a
    /// breach or a failed fill, or the create's cancel, the make's first refusal, `first`,
    /// settles it there ([`Self::settle`]), counted after a breach or a failed fill
    /// ([`compile::HELD_CREATE_REFUSED`]); and after a cancelled fill (the device's reset, the
    /// process's exit) the create is let go of here, unstored and its answer not kept, its
    /// reply posted where the ring has room, waiting for nothing.
    fn held_end(
        self: &Arc<Self>,
        end: FillEnd,
        again: PipelineJob<P>,
        creating: Creating,
        poster: CompletionPoster,
        first: CompiledPipeline<P>,
    ) {
        if end == FillEnd::Cancelled {
            entry::run(|_| {
                self.pending.finish(creating.obj_id, creating.began);
                if !poster.try_post(&first.reply().bytes()) {
                    log::debug!(
                        target: "reims_vgpu::graphics",
                        "a pipeline create (seqno {}, object {:#x}) let go of at its fill's \
                         cancel: its reply found no room",
                        creating.seqno,
                        creating.obj_id
                    );
                }
                drop(again);
                drop(first);
            });
            return;
        }
        let me = Arc::clone(self);
        let settle = move || {
            if end == FillEnd::Filled && me.pending.live(creating.obj_id, creating.began) {
                drop(first);
                me.compile(again, creating.began, poster, true);
            } else {
                entry::run(|_| {
                    drop(again);
                    if end != FillEnd::Filled {
                        reims_vgpu_vulkan::answer::count_kind(compile::HELD_CREATE_REFUSED);
                    }
                    me.settle(first, creating, &poster);
                });
            }
        };
        let of = CompileOf::new(CompileKind::Pipeline, creating.obj_id);
        if let Err(settle) = self.pool.submit(of, Box::new(settle)) {
            settle();
        }
    }

    /// Settles a pipeline create's compile, `compiled`, of the create `creating`, in two steps on
    /// the calling thread, the compile's (the `compile` module's docs, "Order"). The
    /// reply ([`Self::post_reply`]) logs its refusal and posts its reply (its reflection, or a
    /// refusal) through `poster`, the pipeline still pending. The store ([`Self::store_made`])
    /// then finishes its make, keeps a keyed create's answer for the next runs where the make
    /// finished, stores the pipeline where no destroy, close or reset cancelled it meanwhile, and
    /// lets what waits for it go on: the command buffers that name it, the `ICB_WRITE`s,
    /// `ICB_SLOTS` and `ICB_SLOTS_IN_PAGES` that set it, and the `FUNCTION_HANDLE_CREATE`s and
    /// `FUNCTION_TABLE_CREATE`s that name its state.
    fn settle(&self, compiled: CompiledPipeline<P>, creating: Creating, poster: &CompletionPoster) {
        self.post_reply(&compiled, creating, poster);
        self.store_made(compiled, creating);
    }

    /// The reply step of a create's settling ([`Self::settle`]): logs the refusal of `compiled`,
    /// the create `creating`'s compile, and posts its reply through `poster`. Every answer the
    /// reply carries was made by the make's first step; the pipeline stays pending, so whatever
    /// names it from now on waits for its store.
    fn post_reply(
        &self,
        compiled: &CompiledPipeline<P>,
        creating: Creating,
        poster: &CompletionPoster,
    ) {
        let Creating { obj_id, seqno, .. } = creating;
        if let Some(reason) = compiled.refusal() {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = reason;
                "a pipeline create (seqno {seqno}, object {obj_id:#x}) was refused: {reason}"
            );
        }
        poster.post(&compiled.reply().bytes());
    }

    /// The store step of a create's settling ([`Self::settle`]), after its reply: finishes the
    /// make of `compiled`, the create `creating`'s compile, on this thread
    /// ([`CompiledPipeline::finish`]); keeps a keyed create's answer for the next runs where it
    /// was made and its make finished, or drops the kept one where it was refused, answered
    /// otherwise, or its finish failed (the late failure, [`decode::FINISH_FAILED`]); and stores
    /// the pipeline where no destroy, close or reset cancelled it since its compile began, as
    /// made where its finish failed, its draws and dispatches meeting the failure at their encode.
    fn store_made(&self, compiled: CompiledPipeline<P>, creating: Creating) {
        let Creating {
            obj_id,
            seqno,
            key,
            seen_from,
            began,
        } = creating;
        let finished = compiled.finish(seqno);
        if let Some(key) = key {
            let made = (!compiled.is_refused() && finished).then_some(&compiled.reflection);
            self.answers.observe(&key, made);
        }
        // Stored where no destroy, close or reset cancelled it, under the lock of its table, which
        // its destroy removes from after cancelling it; a reset waits for the store.
        let _carried = self.reset_gate.read();
        let live = || self.pending.live(obj_id, began);
        (self.decoder()).finish_pipeline_if(compiled, Some(seen_from), live);
        self.pending.finish(obj_id, began);
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
        // Each guest process's library lookups: a debug line of its own target, which a run that
        // times the guest's launches names (`reims_vgpu::graphics::library=debug`).
        log::debug!(
            target: "reims_vgpu::graphics::library",
            "library lookup (object {obj_id:#x}, {size} bytes): {}",
            if status == RGFX_STATUS_OK { "hit" } else { "miss" }
        );
        if let Some(seqno) = seqno {
            poster.post(complete(seqno, status).as_bytes());
        }
    }

    /// A `PAGES_BUFFER_ADDRESS` from ring 0 (`IMPORT_ADDRESSES_LATER`), answered at once through
    /// `poster` with the GPU address of the buffer over a pages block it names
    /// ([`GraphicsDecoder::pages_buffer_address`]). It is carried out after the sender's messages
    /// before it, the buffer's make on the surface worker among them
    /// ([`Self::wait_for_pages_buffers`]), which waits for its block's view's address: the address
    /// is the one the buffer lies at.
    fn pages_buffer_address(&self, poster: &CompletionPoster, message: &[u8]) {
        let answer = entry::run(|_| {
            (self.decoders)
                .sender(Some(sender_context(poster, message)), Sinks::default())
                .pages_buffer_address(message)
        });
        if let Some(answer) = answer {
            poster.post(answer.as_bytes());
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

    /// A shared event's name, import or unname from ring 0, the kext's for a process that hands
    /// its event to another or makes one of its own from another's, and for the port it gave out
    /// gone (`decode::shares`), a name and an import answered at once through `poster`: a name's
    /// `COMPLETE`, an import's `SHARED_EVENT_VALUE`. The values other holders give the event go to
    /// the sender's listen queue's completion ring, posted there waiting for nothing.
    fn share_event(&self, poster: &CompletionPoster, message: &[u8]) {
        let (ty, _, obj_id) = message_identity(message);
        if ty == RGFX_CMD_SHARED_EVENT_UNNAME {
            let decoder = (self.decoders).sender(Some(0), Sinks::default());
            entry::run(|_| decoder.unname_shared_event(message));
            return;
        }
        // Both messages name the listen queue after the name.
        let listen_queue = RgfxSharedEventName::read_from_prefix(message)
            .map_or(0, |(m, _)| m.listen_queue_obj_id);
        let listen = self.presents.queue_poster(listen_queue).map(|ring| {
            Arc::new(move |m: &RgfxEventSignalled| ring.try_post(m.as_bytes())) as ListenSink
        });
        let decoder =
            (self.decoders).sender(Some(sender_context(poster, message)), Sinks::default());
        if ty == RGFX_CMD_SHARED_EVENT_NAME {
            let (seqno, status) = entry::run(|_| decoder.name_shared_event(message, listen));
            log::debug!(
                target: "reims_vgpu::graphics",
                "shared event {obj_id:#x} named for other processes: status {status}"
            );
            if let Some(seqno) = seqno {
                poster.post(complete(seqno, status).as_bytes());
            }
        } else {
            let answer = entry::run(|_| decoder.import_shared_event(message, listen));
            if let Some(answer) = answer {
                log::debug!(
                    target: "reims_vgpu::graphics",
                    "shared event {obj_id:#x} made from another process's: status {}, value {}",
                    answer.status,
                    answer.value
                );
                poster.post(answer.as_bytes());
            }
        }
    }

    /// A rasterization rate map's create or reset from ring 0, answered at once through `poster`
    /// with the map's state (`RATE_MAP_STATE`), the map stored under its id before the answer
    /// (`decode::rate_map`). A create or reset the platform raises or dies at is answered refused,
    /// as one it refuses, and makes nothing.
    fn rate_map_state(&self, poster: &CompletionPoster, message: &[u8]) {
        let (ty, seqno, obj_id) = message_identity(message);
        let decoder =
            (self.decoders).sender(Some(sender_context(poster, message)), Sinks::default());
        let reset = ty == RGFX_CMD_RATE_MAP_RESET;
        let answer =
            entry::run(|_| refusing(|| decoder.rate_map_state(&self.device, message, reset)));
        let answer = answer.unwrap_or_else(|raise| {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = raise;
                "a rate map's create or reset (type {ty:#x}, seqno {seqno}, object {obj_id:#x}) \
                 faulted: {raise}"
            );
            RgfxMsgHeader::read_from_prefix(message)
                .ok()
                .map(|(header, _)| RgfxRateMapState {
                    header: RgfxMsgHeader {
                        r#type: RGFX_EVT_RATE_MAP_STATE,
                        size: size_of::<RgfxRateMapState>() as u32,
                        seqno: header.seqno,
                    },
                    status: RGFX_STATUS_REFUSED,
                    ..RgfxRateMapState::default()
                })
        });
        let Some(answer) = answer else {
            return;
        };
        if answer.status != RGFX_STATUS_OK {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = answer.status;
                "a rate map's create or reset (type {ty:#x}, seqno {seqno}, object {obj_id:#x}) \
                 was answered status {}",
                answer.status
            );
        }
        poster.post(answer.as_bytes());
    }

    /// A compute pipeline state's function handle or visible function table from ring 0
    /// (`FUNCTION_HANDLE_CREATE`, `FUNCTION_TABLE_CREATE`), made and answered at once through
    /// `poster` (`FUNCTION_POINTER_STATE`, `decode::function_table`), after the compile of the
    /// state it names where that is still under way. A create the platform raises at is answered
    /// refused.
    fn function_pointer_state(&self, poster: &CompletionPoster, message: &[u8]) {
        if let Some(pipeline) = decode::function_table_pipeline(message) {
            self.pending.wait_for_ids(&[pipeline]);
        }
        let (ty, seqno, obj_id) = message_identity(message);
        let decoder =
            (self.decoders).sender(Some(sender_context(poster, message)), Sinks::default());
        let answer = entry::run(|_| refusing(|| decoder.function_pointer_state(message)));
        let answer = answer.unwrap_or_else(|raise| {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = raise;
                "a function handle's or table's create (type {ty:#x}, seqno {seqno}, object \
                 {obj_id:#x}) faulted: {raise}"
            );
            RgfxMsgHeader::read_from_prefix(message)
                .ok()
                .map(|(header, _)| RgfxFunctionPointerState {
                    header: RgfxMsgHeader {
                        r#type: RGFX_EVT_FUNCTION_POINTER_STATE,
                        size: size_of::<RgfxFunctionPointerState>() as u32,
                        seqno: header.seqno,
                    },
                    status: RGFX_STATUS_REFUSED,
                    ..RgfxFunctionPointerState::default()
                })
        });
        let Some(answer) = answer else {
            return;
        };
        // A handle Metal answers nil for (a function the state does not link) is not found, as
        // asked; the rest are logged.
        if answer.status != RGFX_STATUS_OK
            && !(ty == RGFX_CMD_FUNCTION_HANDLE_CREATE && answer.status == RGFX_STATUS_NOT_FOUND)
        {
            log::warn!(
                target: "reims_vgpu::graphics",
                "a function handle's or table's create (type {ty:#x}, seqno {seqno}, object \
                 {obj_id:#x}) was answered status {}",
                answer.status
            );
        }
        poster.post(answer.as_bytes());
    }

    /// A dynamic library's create from ring 0, answered through `poster` (`COMPLETE`,
    /// `decode::dynamic_library`) once the library it names is made. A create the platform raises
    /// at is answered refused.
    fn dynamic_library_create(&self, poster: &CompletionPoster, message: &[u8]) {
        let (ty, seqno, obj_id) = message_identity(message);
        let decoder =
            (self.decoders).sender(Some(sender_context(poster, message)), Sinks::default());
        let answer =
            entry::run(|_| refusing(|| decoder.dynamic_library_create(&self.device, message)));
        let answer = answer.unwrap_or_else(|raise| {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = raise;
                "a dynamic library create (type {ty:#x}, seqno {seqno}, object {obj_id:#x}) \
                 faulted: {raise}"
            );
            RgfxMsgHeader::read_from_prefix(message)
                .ok()
                .map(|(header, _)| (header.seqno, RGFX_STATUS_REFUSED))
        });
        let Some((seqno, status)) = answer else {
            return;
        };
        if status != RGFX_STATUS_OK {
            log::warn!(
                target: "reims_vgpu::graphics",
                "a dynamic library create (type {ty:#x}, seqno {seqno}, object {obj_id:#x}) was \
                 answered status {status}"
            );
        }
        poster.post(complete(seqno, status).as_bytes());
    }

    /// A rasterization rate map's query from ring 0, answered at once through `poster`
    /// (`RATE_MAP_ANSWER`, `decode::rate_map`). A query the platform raises or dies at, where
    /// Metal ends the process, is answered refused.
    fn rate_map_query(&self, poster: &CompletionPoster, message: &[u8]) {
        let (ty, seqno, obj_id) = message_identity(message);
        let decoder =
            (self.decoders).sender(Some(sender_context(poster, message)), Sinks::default());
        let answer = entry::run(|_| refusing(|| decoder.rate_map_answer(message)));
        let answer = answer.unwrap_or_else(|raise| {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = raise;
                "a rate map's query (type {ty:#x}, seqno {seqno}, map {obj_id:#x}) faulted: \
                 {raise}"
            );
            RgfxMsgHeader::read_from_prefix(message)
                .ok()
                .map(|(header, _)| RgfxRateMapAnswer {
                    header: RgfxMsgHeader {
                        r#type: RGFX_EVT_RATE_MAP_ANSWER,
                        size: size_of::<RgfxRateMapAnswer>() as u32,
                        seqno: header.seqno,
                    },
                    status: RGFX_STATUS_REFUSED,
                    ..RgfxRateMapAnswer::default()
                })
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
                target: "reims_vgpu::graphics", kind:% = raise;
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

    /// Waits for the no-copy buffers and heaps of `message`'s context being made on the surface
    /// worker ([`Self::start_over_pages`]), which it may name. A ring-0 message waits on its
    /// context's lane, which hands no later message over meanwhile, so no view behind it can
    /// join the context's group being formed: the group closes once the views handed over before
    /// it have joined, rather than at its window's end (`decode::surface_pages`, "Group
    /// dma-bufs"). A message of a queue's ring leaves the group its window, as a later view of the
    /// context may still come on ring 0.
    fn wait_for_pages_buffers(&self, poster: &CompletionPoster, message: &[u8]) {
        let context = sender_context(poster, message);
        let key = pages_context(context);
        if !self.pages_buffers.any(std::iter::once(key)) {
            return;
        }
        let begun = self.pool.now();
        let ring_0 = poster.queue_obj_id().is_none();
        let waits = ring_0.then(|| self.decoder().wait_for_views(context));
        self.pages_buffers.wait(key);
        drop(waits);
        let (ty, seqno, _) = message_identity(message);
        log::debug!(
            target: "reims_vgpu::graphics::surface_pages",
            "a message (type {ty:#x}, seqno {seqno}) of context {context} on {} waited {} us for \
             its no-copy buffers being made",
            if ring_0 { "ring 0" } else { "its queue's ring" },
            self.pool.now().saturating_duration_since(begun).as_micros()
        );
    }

    /// A `BUFFER_OVER_PAGES` or `HEAP_OVER_PAGES` from ring 0 that makes its pages block's view
    /// (the block's first; a heap over a block of one run makes none), follows its block's
    /// `SURFACE_PAGES` still on the surface worker, or follows another of its context's still
    /// being made: begun here, in ring order, and carried out on the surface
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
    fn start_over_pages(self: &Arc<Self>, poster: &CompletionPoster, message: &[u8]) -> bool {
        let Some((name, block)) = over_pages_block(message) else {
            return false;
        };
        let context = pages_context(sender_context(poster, message));
        let decoder = self.decoder();
        if !self.pages_buffers.any([context, block].into_iter())
            && !decoder.over_pages_makes(message)
        {
            return false;
        }
        let begun = self.pool.now();
        let settles = (
            self.pages_buffers.begin(context, 0, begun),
            self.pages_buffers.begin(block, 0, begun),
        );
        // Counted until its job has joined its context's group or gone on alone, so a message of
        // the context that waits behind it on its lane leaves the group open for it.
        let handed = decoder.hand_over_pages(sender_context(poster, message));
        let (me, poster, message) = (Arc::clone(self), poster.clone(), message.to_vec());
        decoder.on_surface_worker(
            block,
            Box::new(move || {
                let context = sender_context(&poster, &message);
                // Where the platform groups views, the block's view is mapped here and joins its
                // context's group, and the rest waits for the group's seal
                // (`decode::surface_pages`, "Group dma-bufs").
                let joined = {
                    let _carried = me.reset_gate.read_recursive();
                    if poster.stale() {
                        return;
                    }
                    let carry = (poster.clone(), message.clone(), name, block, settles);
                    let then = Arc::clone(&me).over_pages_job(carry);
                    let decoder = (me.decoders).sender(Some(context), Sinks::default());
                    decoder.join_group(&message, then)
                };
                drop(handed);
                match joined {
                    Joined::Now(then) => then(),
                    Joined::Waits => {}
                    Joined::Opened(number) => {
                        let sealer = Arc::clone(&me);
                        let seal = Box::new(move || sealer.seal_group(number));
                        me.decoder().on_surface_worker(group_key(context), seal);
                    }
                }
            }),
        );
        true
    }

    /// At a ring-0 message, before it is carried out (`decode::surface_pages`, "Regroups"): the
    /// regroups of its context made ready, their members' buffers moved into their views here, on
    /// the context's lane, as many as the message's budget takes, what that did logged and traced
    /// (`regroup`: the context, the microseconds, the moves asked, the members taken up and those
    /// left for its next messages); and where a pass is due, the pass handed to the surface
    /// worker, under the reset gate, held shared recursively, as its other jobs are, where the
    /// regroups' views are mapped and their imports made, off the lanes.
    fn regroup_step(self: &Arc<Self>, poster: &CompletionPoster, message: &[u8]) {
        let context = sender_context(poster, message);
        let decoder = self.decoder();
        let (ready, due) = decoder.regroup_step(context, self.pool.now());
        if ready {
            let mover = (self.decoders).sender(Some(context), Sinks::default());
            if let Some(moves) = mover.move_regroups() {
                let us = u64::try_from(moves.took.as_micros()).unwrap_or(u64::MAX);
                reims_vgpu_vulkan::core::trace::event(
                    "regroup",
                    &[
                        u64::from(context),
                        us,
                        moves.rebinds,
                        moves.handled,
                        moves.left,
                    ],
                );
                log::debug!(
                    target: "reims_vgpu::graphics::surface_pages",
                    "context {context}'s regroups: {} moves asked on its lane in {us} us, {} \
                     members taken up, {} left for its next messages",
                    moves.rebinds, moves.handled, moves.left
                );
            }
        }
        if due {
            let me = Arc::clone(self);
            decoder.on_surface_worker(
                REGROUP_KEY,
                Box::new(move || {
                    let _carried = me.reset_gate.read_recursive();
                    if let Err(raise) = me.decoder().regroup_pass(&me.device) {
                        log::debug!(
                            target: "reims_vgpu::graphics::surface_pages",
                            "a regroups' pass ended at a raise: {raise:?}"
                        );
                    }
                }),
            );
        }
    }

    /// The carrying out of a `BUFFER_OVER_PAGES` or `HEAP_OVER_PAGES` handed to the surface
    /// worker ([`Self::start_over_pages`]): `carry` is its poster, the message, its name, its
    /// block and its settles, which drop once it is made. Made under the reset gate, held shared
    /// recursively; where the device was reset since, nothing is made.
    fn over_pages_job(
        self: Arc<Self>,
        carry: (
            CompletionPoster,
            Vec<u8>,
            &'static str,
            u64,
            (Settle, Settle),
        ),
    ) -> Job {
        let (poster, message, name, block, settles) = carry;
        Box::new(move || {
            let outcome = {
                let _carried = self.reset_gate.read_recursive();
                if poster.stale() {
                    return;
                }
                let mut decoder = (self.decoders)
                    .sender(Some(sender_context(&poster, &message)), self.sinks(&poster));
                decoder.decode_refusing(&self.device, &message)
            };
            self.report(&poster, &message, outcome, false);
            log::debug!(
                target: "reims_vgpu::graphics::surface_pages",
                "{name} of the block at {block:#x} made on the worker {} us after its begin",
                self.pool.now().saturating_duration_since(settles.0.begun).as_micros()
            );
            drop(settles);
        })
    }

    /// The seal of the group of pages blocks' views numbered `number` (`decode::surface_pages`,
    /// "Group dma-bufs"), on the surface worker, under its context's key ([`group_key`]): waits
    /// for its window to close, makes the members' imports under the reset gate, held shared
    /// recursively, as the worker's other jobs are, then carries out the messages that waited
    /// for them, in their order, each as [`Self::over_pages_job`] does. Where the group went (its
    /// context's close, the device's reset), nothing.
    fn seal_group(&self, number: u64) {
        let decoder = self.decoder();
        let Some(sealing) = decoder.group_closed(number) else {
            return;
        };
        {
            let _carried = self.reset_gate.read_recursive();
            decoder.seal_group(&self.device, sealing);
        }
        // The members' messages, and those that join a member while they are carried out.
        while let Some(jobs) = decoder.group_jobs(number) {
            for job in jobs {
                job();
            }
        }
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
                target: "reims_vgpu::graphics", kind:% = raise;
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
                target: "reims_vgpu::graphics", kind:% = status;
                "an argument layout (type {ty:#x}, object {obj_id:#x}) was answered status \
                 {status}"
            );
        }
        if let Some(seqno) = seqno {
            poster.post(complete(seqno, status).as_bytes());
        }
    }

    /// A `PIPELINE_ANSWERS` from the kext, answered at once on the control ring: each kept answer
    /// it asks for as a keyed `PIPELINE_REFLECTION` of the query's seqno, newest first, then a
    /// `COMPLETE` of how many went.
    fn pipeline_answers(&self, poster: &CompletionPoster, message: &[u8]) {
        let (_, seqno, _) = message_identity(message);
        let Ok((query, _)) = RgfxPipelineAnswersQuery::read_from_prefix(message) else {
            poster.post(complete(seqno, RGFX_STATUS_INVALID_ARGUMENT).as_bytes());
            return;
        };
        let replies = self.answers.replies(query.first, query.most, seqno);
        for reply in &replies {
            poster.post(reply);
        }
        log::debug!(
            target: "reims_vgpu::graphics::answers",
            "the kext took {} kept pipeline answers from the {}th",
            replies.len(),
            query.first
        );
        let count = u32::try_from(replies.len()).unwrap_or(u32::MAX);
        poster.post(complete(seqno, count).as_bytes());
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
                    target: "reims_vgpu::graphics", kind:? = error;
                    "a library create (type {ty:#x}, seqno {seqno}, object {obj_id:#x}) was not \
                     carried out: {error:?}"
                );
                error.count();
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
                    target: "reims_vgpu::graphics", kind:% = reason;
                    "a library create (seqno {seqno}, object {obj_id:#x}) was refused: {reason}"
                );
            }
            poster.post(complete(seqno, status).as_bytes());
        });
    }

    /// Shows the IOSurface `iosurface_id` (0: the scanout) on display `display` once the command
    /// buffers committed that write its copy have finished, after the display's presents before
    /// it ([`PresentQueue`]), with the colour stages staged for it since its last present
    /// ([`ColorStaging::take`]), answering once its window shows it; answered invalid at once,
    /// with nothing changed and the colour still staged, where the decoder holds no texture over
    /// it, or the device has no such display (which the device never hands on).
    fn present_surface(
        &self,
        display: usize,
        iosurface_id: u32,
        generation: Option<u64>,
        answer: PresentAnswer,
    ) {
        let (Some(presented), Some(staging)) =
            (self.presented.get(display), self.colors.get(display))
        else {
            answer.answer(RGFX_STATUS_INVALID_ARGUMENT);
            return;
        };
        if iosurface_id == 0 {
            let color = staging.lock().take();
            self.presents
                .present(display, 0, None, color, answer, Vec::new(), None);
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
                    target: "reims_vgpu::graphics", kind:% = raise;
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
                view,
                span,
                writers,
                owner,
                queued,
            }) => {
                // The process that made the texture over the surface the display shows owns the
                // scanout (WindowServer): its compiles go first.
                self.pool.favour(owner);
                // The rt platform's boot diagnostics count over-copies a frame, and name the
                // compositor by it.
                reims_vgpu_vulkan::rt_platform::diag::presented(owner);
                // Its block is held from here, before the texture's destroy and the block's
                // release can come: the present shows it later.
                let shown = presented.hold(texture, view, span, queued);
                let color = staging.lock().take();
                self.presents.present(
                    display,
                    iosurface_id,
                    Some(shown),
                    color,
                    answer,
                    writers,
                    fill,
                );
            }
            None => answer.answer(RGFX_STATUS_INVALID_ARGUMENT),
        }
    }

    /// A present that keeps what display `display` shows (`RGFX_PRESENT_KEEP`), with the colour
    /// stages staged for it since its last present, after its presents before it, answered once
    /// its window shows it; at once where the device has no such display (which the device never
    /// hands on).
    fn present_kept(&self, display: usize, answer: PresentAnswer) {
        let Some(staging) = self.colors.get(display) else {
            answer.answer(RGFX_STATUS_OK);
            return;
        };
        let color = staging.lock().take();
        self.presents.present_kept(display, color, answer);
    }

    /// One `DISPLAY_COLOR` part, staged for its display's next present ([`ColorStaging::part`]);
    /// one the staging refuses is logged.
    fn display_color(&self, color: &RgfxDisplayColor, entries: &[RgfxColorEntry]) {
        let Some(staging) = self.colors.get(color.display as usize) else {
            return;
        };
        if let Err(dropped) = staging.lock().part(color, entries) {
            log::warn!(
                target: "reims_vgpu::graphics", kind:? = dropped;
                "a display colour part of transaction {} (stage {}, entries {} on of {}, {} \
                 carried) was dropped ({dropped:?}): the stage keeps its table",
                color.transaction_id,
                color.stage,
                color.first_entry,
                color.entry_count,
                entries.len()
            );
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
        // Ahead of the buffers before it, which may still run: it finishes alone.
        if let Some(queue) = poster.queue_obj_id() {
            self.presents.failed_at_once(queue, header.seqno, poster);
        }
    }
}

/// The host's callbacks and its stage-3 decoder as the device's [`Host`].
struct CallbackHost<P: ApiPlatform> {
    /// The host's callbacks: the device raises its interrupt through them.
    callbacks: GraphicsCallbacks,
    /// The `RGFX_FEATURE_*` the decoder carries out, fixed at the device's make: the guest reads
    /// them on a vCPU, which takes no lock of the decoder's.
    features: u64,
    /// Those it carries out only for a guest that takes `RGFX_FEATURE_PAGES_16K`, also fixed at
    /// the make: the guest RAM's on a host of 16 KiB pages, else none.
    features_16k: u64,
    /// The deferred blocks' budget ([`deferred_budget`]), read once at the device's make, so the
    /// kext's one read holds.
    deferred_budget: u64,
    /// The stage-3 decoder, fed the stage-3 messages of ring 0 and of the queues' rings.
    control: Arc<GraphicsControl<P>>,
    /// The host's side of VideoToolbox's paravirtual messages ([`paravirt`]), over the decoders
    /// of the host's hardware video.
    vt: paravirt::Host,
    /// The kinds of VideoToolbox message each way the log named (diagnostics, [`vt_first`]).
    vt_seen: VtSeen,
}

impl<P: ApiPlatform> Host for CallbackHost<P> {
    fn raise_interrupt(&self, vector: u32) {
        if let Some(f) = &self.callbacks.raise_interrupt {
            f(vector);
        }
    }

    fn route_doorbell(&self, route: u64, at: DoorbellAt, ring_id: u32, bell: RawDoorbell) -> bool {
        match at {
            DoorbellAt::Register => (self.callbacks.route_doorbell)
                .as_ref()
                .is_some_and(|f| f(route, ring_id, Some(bell))),
            DoorbellAt::Page(page) => (self.callbacks.route_doorbell_page)
                .as_ref()
                .is_some_and(|f| f(route, page, ring_id, Some(bell))),
        }
    }

    fn routes_doorbell_pages(&self) -> bool {
        self.callbacks.route_doorbell_page.is_some()
    }

    fn unroute_doorbell(&self, route: u64, at: DoorbellAt, ring_id: u32) {
        match at {
            DoorbellAt::Register => {
                if let Some(f) = &self.callbacks.route_doorbell {
                    f(route, ring_id, None);
                }
            }
            DoorbellAt::Page(page) => {
                if let Some(f) = &self.callbacks.route_doorbell_page {
                    f(route, page, ring_id, None);
                }
            }
        }
    }

    fn trace(&self, kind: &'static str, values: &[u64]) {
        reims_vgpu_vulkan::core::trace::event(kind, values);
    }

    /// Where the trace is on, on rt's threads that run a command buffer's handlers
    /// ([`RT_HANDLER_THREADS`]).
    fn times_posts_here(&self) -> bool {
        reims_vgpu_vulkan::core::trace::on()
            && std::thread::current()
                .name()
                .is_some_and(|name| RT_HANDLER_THREADS.contains(&name))
    }

    fn ring_degraded(&self, kind: RingDegradation) {
        reims_vgpu_vulkan::answer::count_kind(kind.code());
    }

    fn answer_slow(&self, kind: u32, took_ns: u64) {
        slow_answers::slow(kind, took_ns);
    }

    fn present_surface(
        &self,
        display: u32,
        iosurface_id: u32,
        generation: Option<u64>,
        answer: PresentAnswer,
    ) {
        stepped(|| {
            self.control
                .present_surface(display as usize, iosurface_id, generation, answer);
        });
    }

    fn present_keep(&self, display: u32, answer: PresentAnswer) {
        stepped(|| self.control.present_kept(display as usize, answer));
    }

    fn display_color(&self, color: &RgfxDisplayColor, entries: &[RgfxColorEntry]) {
        stepped(|| self.control.display_color(color, entries));
    }

    fn surface_textures(&self, iosurface_id: u32) -> Vec<u64> {
        self.control.decoder().surface_textures(iosurface_id)
    }

    fn refreshed(&self, display: u32) {
        if let Some(presented) = self.control.presented.get(display as usize) {
            stepped(|| presented.refreshed());
        }
    }

    fn context_closed(&self, context: u32, answer: CompleteAnswer) {
        stepped(|| self.control.close_context(context, answer));
    }

    fn block_released(&self, release: &RgfxBlockRelease, answer: CompleteAnswer) {
        stepped(|| self.control.release_block(release, answer));
    }

    fn block_unshared(&self, unshare: &RgfxBlockGrant) {
        stepped(|| self.control.unshare_block(unshare));
    }

    fn pool_range(&self) -> (u64, u64) {
        (self.control.shared_pool.as_ref()).map_or((0, 0), PoolLedger::range)
    }

    fn pool_taken(&self) {
        if let Some(ledger) = &self.control.shared_pool {
            let _ = ledger.keep_headroom();
        }
    }

    fn driver_ready(&self, taken: u64) {
        (self.control.decoder()).set_managed_at_copy(taken & RGFX_FEATURE_MANAGED_AT_COPY != 0);
    }

    fn pool_take(&self, take: &RgfxPoolTake, answer: PoolAnswer) {
        stepped(|| self.control.take_pool_block(take, answer));
    }

    fn pool_give(&self, give: &RgfxPoolGive, answer: CompleteAnswer) {
        stepped(|| self.control.give_pool_block(give, answer));
    }

    fn private_take(&self, take: &RgfxPrivateTake, answer: PrivateAnswer) {
        stepped(|| self.control.take_private_block(take, answer));
    }

    fn private_give(&self, give: &RgfxPrivateGive, answer: CompleteAnswer) {
        stepped(|| self.control.give_private_block(give, answer));
    }

    fn device_reset(&self) {
        stepped(|| self.control.reset());
    }

    fn features(&self) -> u64 {
        self.features
    }

    fn features_with_16k_pages(&self) -> u64 {
        self.features_16k
    }

    fn deferred_budget(&self) -> u64 {
        self.deferred_budget
    }

    fn surface_pages(
        &self,
        pages: &RgfxSurfacePages,
        runs: Vec<RgfxPageRun>,
        flags: u32,
        layout: Option<RgfxSurfacePagesLayout>,
        answer: CompleteAnswer,
    ) {
        stepped(|| {
            self.control
                .start_surface_pages(*pages, runs, flags, layout, answer);
        });
    }

    fn surface_download(&self, block_offset: u64, answer: CompleteAnswer) {
        stepped(|| self.control.download_surface(block_offset, answer));
    }

    fn try_surface_download(
        &self,
        block_offset: u64,
        answer: CompleteAnswer,
    ) -> Result<(), CompleteAnswer> {
        stepped(|| self.control.try_download_surface(block_offset, answer))
    }

    fn surface_downloaded(&self, block_offset: u64) -> bool {
        stepped(|| self.control.downloaded_at_doorbell(block_offset))
    }

    fn stage3_message(&self, completion: &CompletionPoster, message: &[u8]) {
        stepped(|| {
            self.control
                .handle_stage3(completion, Cow::Borrowed(message), None);
        });
    }

    fn queue_made(&self, completion: &CompletionPoster) {
        stepped(|| self.control.queue_made(completion));
    }

    fn command_buffer(
        &self,
        completion: &CompletionPoster,
        message: Vec<u8>,
        claim: &mut QueueClaim<'_>,
    ) {
        stepped(|| (self.control).handle_stage3(completion, Cow::Owned(message), Some(claim)));
    }

    fn answer_command_buffer(&self, completion: &CompletionPoster, message: &[u8]) -> bool {
        stepped(|| self.control.answer_command_buffer(completion, message))
    }

    fn scheduled_ask(&self, completion: &CompletionPoster, message: &[u8]) {
        stepped(|| self.control.scheduled_ask(completion, message));
    }

    fn vt_message(&self, message: VtMessage, answer: VtAnswer, link: VtLink) {
        let length = message.message.length;
        let lends: Vec<u64> = (message.surfaces.iter())
            .map(|surface| surface.surface.lend)
            .collect();
        // Refused: the lends go back, as for a message the device could not read.
        let refuse = |why: &str, answer: VtAnswer| {
            log::warn!(target: "reims_vgpu::graphics", "a VT_MESSAGE of {length} bytes: {why}");
            if !lends.is_empty() {
                link.send(&paravirt::surface_release(&lends));
            }
            answer.answer(RGFX_VT_STATUS_MALFORMED);
        };
        let decoder = self.control.decoder();
        let Some(surfaces) = (message.surfaces.into_iter())
            .map(|surface| with_table_runs(surface, |runs, n| decoder.guest_bytes(runs, n)))
            .collect::<Option<Vec<_>>>()
        else {
            return refuse(
                "a surface's run table lies outside the guest RAM the host maps, or does not \
                 hold its bytes",
                answer,
            );
        };
        let lent: Vec<Arc<dyn paravirt::LentSurface>> = surfaces
            .into_iter()
            .map(|surface| Arc::new(GuestSurface(surface)) as Arc<dyn paravirt::LentSurface>)
            .collect();
        let guest: Arc<dyn paravirt::Link> =
            Arc::new(GuestLink(link.clone(), Arc::clone(&self.vt_seen)));
        let Some(bytes) = decoder.guest_bytes(&message.runs, length) else {
            return refuse("its runs lie outside the guest RAM the host maps", answer);
        };
        let answered = self.vt.deliver_lending(&bytes, lent, Some(guest));
        vt_first(
            &self.vt_seen,
            false,
            &bytes,
            Some((answered.status, answered.replies.len())),
        );
        for reply in &answered.replies {
            vt_first(&self.vt_seen, true, reply, None);
            if !link.send(reply) {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "a {}-byte VideoToolbox message to the guest was not sent",
                    reply.len()
                );
            }
        }
        answer.answer(answered.status);
    }
}

/// `surface` with its own runs: where its record has a run table
/// ([`RGFX_VT_SURFACE_RUN_TABLE`](reims_vgpu_graphics::wire::RGFX_VT_SURFACE_RUN_TABLE)), the
/// table's entries, in order, as many as hold the surface's `length` bytes, of the table's bytes
/// as `read` gives them (the first `n` bytes the runs hold, as the decoder's `guest_bytes` reads
/// them). `None` where the table does not read, an entry is empty, or the
/// entries do not hold `length` bytes.
fn with_table_runs(
    mut surface: VtSurface,
    read: impl FnOnce(&[(u64, u64)], u64) -> Option<Vec<u8>>,
) -> Option<VtSurface> {
    if !surface.table {
        return Some(surface);
    }
    let held = (surface.runs.iter()).try_fold(0u64, |sum, &(_, n)| sum.checked_add(n))?;
    let table = read(&surface.runs, held)?;
    let mut runs = Vec::new();
    let mut sum = 0u64;
    for entry in table.as_chunks::<{ size_of::<RgfxPageRun>() }>().0 {
        if sum >= surface.surface.length {
            break;
        }
        let run = RgfxPageRun::read_from_bytes(entry).ok()?;
        if run.length == 0 {
            return None;
        }
        sum = sum.checked_add(run.length)?;
        runs.push((run.address, run.length));
    }
    (sum >= surface.surface.length).then(|| {
        // Diagnostics: the first surface of a boot whose runs came in a run table.
        static TOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !TOLD.swap(true, std::sync::atomic::Ordering::Relaxed) {
            log::info!(
                target: "reims_vgpu::graphics",
                "VideoToolbox: a lent surface of {} bytes came with a run table of {} runs \
                 (told once)",
                surface.surface.length,
                runs.len()
            );
        }
        surface.runs = runs;
        surface.table = false;
        surface
    })
}

/// A surface a `VT_MESSAGE` lent, as the device read it: its record and its runs of guest RAM,
/// which the host may write until it lets go of the lend.
#[derive(Debug)]
struct GuestSurface(VtSurface);

impl paravirt::LentSurface for GuestSurface {
    fn id(&self) -> u64 {
        self.0.surface.lend
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The device's link, as VideoToolbox's paravirtual host sends on it, and the kinds of message
/// the log named.
struct GuestLink(VtLink, VtSeen);

impl paravirt::Link for GuestLink {
    fn send(&self, message: &[u8]) -> bool {
        vt_first(&self.1, true, message, None);
        self.0.send(message)
    }
}

/// The kinds of VideoToolbox message the log named, each with whether it went to the guest.
type VtSeen = Arc<Mutex<Vec<(bool, [u8; 4])>>>;

/// Logs the first VideoToolbox message of each kind each way at info (diagnostics): a `vtpv`
/// message's type, or a box's (`rels`), its bytes, and for the guest's its status and replies.
fn vt_first(seen: &VtSeen, to_guest: bool, bytes: &[u8], answered: Option<(i32, usize)>) {
    let at = if bytes.get(4..8) == Some(b"vtpv".as_slice()) {
        8
    } else {
        4
    };
    let kind: [u8; 4] = (bytes.get(at..at + 4))
        .and_then(|k| k.try_into().ok())
        .unwrap_or(*b"????");
    {
        let mut seen = seen.lock();
        if seen.contains(&(to_guest, kind)) {
            return;
        }
        seen.push((to_guest, kind));
    }
    let kind = String::from_utf8_lossy(&kind);
    match answered {
        Some((status, replies)) => log::info!(
            target: "reims_vgpu::graphics",
            "VideoToolbox: the guest's first '{kind}' ({} bytes): status {status}, {replies} \
             replies",
            bytes.len()
        ),
        None => log::info!(
            target: "reims_vgpu::graphics",
            "VideoToolbox: the host's first '{kind}' to the guest ({} bytes)",
            bytes.len()
        ),
    }
}

/// The features VideoToolbox's paravirtual path adds: `VT_MESSAGES` where `vt` lists a decoder,
/// whose sessions decode on the host's video arm; an arm64 guest's kext then serves the guest's
/// VideoToolbox. None where it lists none.
fn vt_features(vt: &paravirt::Host) -> u64 {
    if vt.decoders().is_empty() {
        0
    } else {
        RGFX_FEATURE_VT_MESSAGES
    }
}

/// The decoders VideoToolbox's paravirtual path lists ([`paravirt::Host`]): one for each decode
/// entry of `caps` of H.264 or HEVC with a profile, its profiles as VideoToolbox numbers them
/// (H.264's `profile_idc`; HEVC's `general_profile_idc`, Main Still Picture where Main is, whose
/// decoder takes its streams) and its level the caps'.
fn vt_decoders(caps: &RgfxVideoCaps) -> Vec<paravirt::Decoder> {
    let count = usize::try_from(caps.codec_count)
        .unwrap_or(usize::MAX)
        .min(RGFX_VIDEO_CAPS_CODECS);
    let numbered = |bits: u32, numbers: &[(u32, u32)]| -> Vec<u32> {
        numbers
            .iter()
            .filter(|&&(bit, _)| bits & bit != 0)
            .map(|&(_, number)| number)
            .collect()
    };
    caps.codecs
        .iter()
        .take(count)
        .filter(|c| c.direction == RGFX_VIDEO_DECODE)
        .filter_map(|c| {
            let (codec, profiles) = match c.codec {
                RGFX_VIDEO_CODEC_H264 => (
                    paravirt::Codec::H264,
                    numbered(
                        c.profiles,
                        &[
                            (RGFX_VIDEO_H264_BASELINE, 66),
                            (RGFX_VIDEO_H264_MAIN, 77),
                            (RGFX_VIDEO_H264_HIGH, 100),
                        ],
                    ),
                ),
                RGFX_VIDEO_CODEC_HEVC => (
                    paravirt::Codec::Hevc,
                    numbered(
                        c.profiles,
                        &[
                            (RGFX_VIDEO_HEVC_MAIN, 1),
                            (RGFX_VIDEO_HEVC_MAIN_10, 2),
                            (RGFX_VIDEO_HEVC_MAIN, 3),
                        ],
                    ),
                ),
                _ => return None,
            };
            (!profiles.is_empty()).then_some(paravirt::Decoder {
                codec,
                profiles,
                max_level: c.max_level,
            })
        })
        .collect()
}

/// Runs `f`, a callback of the device's that may call the GPU's driver, as a step of the exit
/// gate's ([`reims_vgpu_vulkan::exit_gate`]): the process's exit waits for it, and one called
/// once the exit began parks its thread for good, on any thread but the exiting one (as an entry
/// does, [`entry`]'s module doc).
fn stepped<T>(f: impl FnOnce() -> T) -> T {
    let Some(_step) = reims_vgpu_vulkan::exit_gate::step() else {
        reims_vgpu_vulkan::exit_gate::stay()
    };
    f()
}

/// Posts a command buffer's `CB_SUBMITTED` or `CB_SCHEDULED` `record` through `poster`, waiting
/// for nothing: on a thread the header names for the device's raise (`RvgGraphicsRaiseFn`: the
/// device's and the decoder's own threads, and the platform's that run a buffer's handlers), at
/// once where the ring has room and no record is handed over before it, raising the queue's
/// vector there, else handed to the queue's service thread
/// (`CompletionPoster::post_unwaiting`); handed to it always, which posts it and raises there,
/// where a submission handler runs inside a commit on any other thread ([`foreign_committer`]).
/// The rt platform runs submission handlers on rt's own threads, never inside a commit: on its
/// submit thread, before the buffer's submission goes to the driver (after it, for one that
/// presents a drawable), one held for a shared event's value among them
/// ([`RT_HANDLER_THREADS`]). A platform may run them, other buffers' among them, inside a commit
/// (`reims_vgpu_vulkan::core::platform::command::committing`), as the testkit's does. A
/// buffer's transition to scheduled, and every later one of its queue, waits for its submission
/// handlers, and a wait for room on any of those threads would hold the record's mutex, which the
/// buffer's completion handler takes, across the wait: none waits.
fn post_from_handler(poster: &CompletionPoster, record: &[u8]) {
    if foreign_committer() {
        poster.post_elsewhere(record);
    } else {
        poster.post_unwaiting(record, None);
    }
}

/// Whether this thread runs a submission handler inside a platform commit
/// (`reims_vgpu_vulkan::core::platform::command::inside_commit`) and is none of the threads the
/// header names for the device's raise ([`RAISE_THREADS`]). A handler run outside a commit runs
/// on the platform's own thread.
fn foreign_committer() -> bool {
    reims_vgpu_vulkan::core::platform::command::inside_commit()
        && !std::thread::current()
            .name()
            .is_some_and(|name| RAISE_THREADS.iter().any(|named| name.starts_with(named)))
}

/// rt's threads that run a command buffer's handlers, and so post onto its queue's ring and raise
/// its vector inline: its submit thread, which runs its submission and scheduled handlers, and its
/// completion thread, which runs its completed handlers but those handed on
/// (`reims-vgpu.rt-handed-on`; `reims_vgpu_vulkan::rt_platform::command`'s module doc). The
/// host's trace times the device's posts on them (`qraise`, `qhanded`,
/// `reims_vgpu_graphics::Host::trace`).
const RT_HANDLER_THREADS: [&str; 2] = ["m2v-submit", "m2v-complete"];

/// The names, or their prefixes, of the library's threads `RvgGraphicsRaiseFn` names: the
/// control thread, a context's, the presents' and a block's lanes, the refresh thread, a queue's
/// service thread and its lane's, a compile's thread, the surfaces', ICB makes' and closes'
/// workers' threads, and the threads that move a staged copy of a cache into its pages.
const RAISE_THREADS: [&str; 12] = [
    "reims-vgpu.graphics-control",
    "reims-vgpu.graphics-context-",
    "reims-vgpu.graphics-presents",
    "reims-vgpu.graphics-block-",
    "reims-vgpu.graphics-refresh",
    "reims-vgpu.graphics-queue-",
    "reims-vgpu.graphics-lane-",
    "reims-vgpu.graphics-compile-",
    "reims-vgpu.graphics-surfaces-",
    "reims-vgpu.icb-create-",
    "reims-vgpu.graphics-closes-",
    "reims-vgpu.graphics-into-pages-",
];

/// What the window and its sink share: the scanout the guest set, whether the display's panel is
/// off, the drawable the window's render thread last drew into (its size in pixels, and its
/// pixels a point), and whether the window draws black while the guest shows nothing.
#[derive(Debug, Default)]
pub(crate) struct Shown {
    /// The guest's scanout; `None` shows black.
    pub(crate) scanout: Mutex<Option<Scanout>>,
    /// The display's panel is off (`SET_DISPLAY_POWER`, the guest's display asleep): the window
    /// and the host's reads show black, whatever the scanout or the presented surface.
    pub(crate) asleep: AtomicBool,
    /// The window's drawable, once drawn into.
    pub(crate) drawable: Mutex<Option<Drawable>>,
    /// A display after display 0, whose window draws black while the guest shows nothing on it
    /// (no scanout, no presented surface; `sink`'s module doc, "A display the guest does not
    /// drive"). Display 0's draws nothing then, as before: the option ROM drives it at once.
    pub(crate) black_undriven: bool,
}

impl Shown {
    /// Whether the display's panel is off.
    pub(crate) fn asleep(&self) -> bool {
        self.asleep.load(Ordering::Acquire)
    }
}

/// A drawable the window's render thread drew into.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Drawable {
    /// Its size in pixels, width then height: the window's size.
    pub(crate) pixels: (u64, u64),
    /// Its pixels a point: the backing scale of the screen the window is on.
    pub(crate) scale: f64,
}

/// `scale` pixels a point in 120ths, as `RgfxDisplayScale` carries it: rounded to nearest, and 0
/// where it is not a positive number.
fn scale_in_120ths(scale: f64) -> u32 {
    if !(scale.is_finite() && scale > 0.0) {
        return 0;
    }
    #[expect(
        clippy::as_conversions,
        reason = "a backing scale of a few units, rounded to nearest; a cast saturates"
    )]
    let scale = (scale * 120.0).round() as u32;
    scale
}

/// The display's window as the device's [`Screen`] (the module doc).
pub(crate) struct WindowScreen {
    /// The window.
    window: Arc<Window>,
    /// What it shows.
    shown: Arc<Shown>,
    /// Whether it shows a presented surface in place of the scanout ([`Presented`]).
    shows_surface: Box<dyn Fn() -> bool + Send + Sync>,
    /// The refresh rate of the host screen it opened on, in thousandths of a hertz.
    refresh_mhz: u32,
    /// The guest's cursor, as last told.
    cursor: Mutex<CursorState>,
}

/// The guest's cursor, as the window shows it: over the image while the guest has an image and
/// shows it, and the display's panel is on.
#[derive(Debug)]
struct CursorState {
    /// The guest has an image.
    image: bool,
    /// The guest shows its cursor: until it says otherwise, it does.
    visible: bool,
    /// The display's panel is off: no cursor shows.
    asleep: bool,
    /// What the window was last told, once told.
    shown: Option<bool>,
}

impl CursorState {
    /// Whether the cursor shows, where that changed since it was last told.
    fn changed(&mut self) -> Option<bool> {
        let show = self.image && self.visible && !self.asleep;
        if self.shown == Some(show) {
            return None;
        }
        self.shown = Some(show);
        Some(show)
    }

    /// Tells `window` whether the cursor shows, where that changed.
    fn update(&mut self, window: &Window) {
        if let Some(show) = self.changed() {
            window.cursor_show(show);
        }
    }
}

/// The guest's cursor as the host is told it ([`GraphicsCallbacks`]' `cursor_*`), beside what
/// the screen shows of it. Each change is queued onto the display thread under the state's lock,
/// so the host's callbacks run in the order the device took the guest's commands, and never
/// inside a call of the host's: the device's reset tells the screen inside the host's reset or
/// the guest's `STATUS` write.
struct HostCursor {
    /// The host's `cursor_glyph`.
    glyph: Option<CursorGlyphFn>,
    /// The host's `cursor_show`.
    show: Option<CursorShowFn>,
    /// The host's `cursor_move`.
    moved: Option<CursorMoveFn>,
    /// The display thread, where the host takes any of them.
    queue: Option<Arc<dyn HostQueue>>,
    /// What the host was told: hidden until the guest shows it.
    state: Mutex<CursorState>,
    /// Where the guest last put its cursor's hot spot, each axis kept to 0 to 65535: x in the
    /// high 16 bits, y in the low.
    position: AtomicU32,
}

/// The host's cursor callbacks of display `display` among `callbacks`: display 0's
/// `cursor_glyph`, `cursor_show` and `cursor_move`, and another display's the `_at` three, which
/// name it.
fn display_cursor(
    callbacks: &GraphicsCallbacks,
    display: usize,
) -> (
    Option<CursorGlyphFn>,
    Option<CursorShowFn>,
    Option<CursorMoveFn>,
) {
    if display == 0 {
        return (
            callbacks.cursor_glyph.clone(),
            callbacks.cursor_show.clone(),
            callbacks.cursor_move.clone(),
        );
    }
    let at = u32::try_from(display).unwrap_or(u32::MAX);
    let glyph = callbacks.cursor_glyph_at.clone().map(|f| {
        let f: CursorGlyphFn = Arc::new(move |glyph: &CursorGlyph<'_>| f(at, glyph));
        f
    });
    let show = callbacks.cursor_show_at.clone().map(|f| {
        let f: CursorShowFn = Arc::new(move |show| f(at, show));
        f
    });
    let moved = callbacks.cursor_move_at.clone().map(|f| {
        let f: CursorMoveFn = Arc::new(move || f(at));
        f
    });
    (glyph, show, moved)
}

impl HostCursor {
    /// A display's host cursor, with the host's callbacks `glyph`, `show` and `moved`, run on
    /// `platform`'s display thread, which is made here where the host takes any.
    fn new<P: ApiPlatform>(
        platform: &Arc<P>,
        glyph: Option<CursorGlyphFn>,
        show: Option<CursorShowFn>,
        moved: Option<CursorMoveFn>,
    ) -> HostCursor {
        let takes = glyph.is_some() || show.is_some() || moved.is_some();
        HostCursor {
            glyph,
            show,
            moved,
            queue: takes.then(|| api_state(platform).display_thread(platform)),
            state: Mutex::new(CursorState {
                image: false,
                visible: true,
                asleep: false,
                shown: Some(false),
            }),
            position: AtomicU32::new(0),
        }
    }

    /// Queues `job` onto the display thread.
    fn queue(&self, job: impl FnOnce() + Send + 'static) {
        if let Some(queue) = &self.queue {
            queue.exec_async(Box::new(job));
        }
    }

    /// Tells the host whether the cursor shows, where that changed: each change a debug line of
    /// its own target (`reims_vgpu::graphics::cursor`), which a boot's check of the display's
    /// sleep reads.
    fn tell_shown(&self, state: &mut CursorState) {
        let show = state.changed();
        if let Some(show) = show {
            log::debug!(
                target: "reims_vgpu::graphics::cursor",
                "the host is told the guest's cursor {}",
                if show { "shows" } else { "is hidden" }
            );
        }
        if let Some(show) = show
            && let Some(f) = &self.show
        {
            let f = Arc::clone(f);
            self.queue(move || f(u8::from(show)));
        }
    }

    /// The guest set its cursor image, or removed it (`None`).
    fn image(&self, image: Option<&CursorImage>) {
        let mut state = self.state.lock();
        state.image = image.is_some();
        if let (Some(image), Some(f)) = (image, &self.glyph) {
            let (f, image) = (Arc::clone(f), image.clone());
            self.queue(move || {
                let hot = |v: u32| u16::try_from(v).unwrap_or(u16::MAX);
                f(&CursorGlyph {
                    width: image.width,
                    height: image.height,
                    hot_spot: (hot(image.hot_spot.0), hot(image.hot_spot.1)),
                    rgba: &image.rgba,
                });
            });
        }
        self.tell_shown(&mut state);
    }

    /// The guest moved its cursor's hot spot to (`x`, `y`), and shows it where `visible`.
    fn moved(&self, x: i32, y: i32, visible: bool) {
        let mut state = self.state.lock();
        state.visible = visible;
        let axis = |v: i32| u32::from(u16::try_from(v.max(0)).unwrap_or(u16::MAX));
        self.position
            .store((axis(x) << 16) | axis(y), Ordering::Release);
        if let Some(f) = &self.moved {
            let f = Arc::clone(f);
            self.queue(move || f());
        }
        self.tell_shown(&mut state);
    }

    /// The display's panel went off (`on` false) or on: while it is off, the host is told the
    /// cursor does not show.
    fn power(&self, on: bool) {
        let mut state = self.state.lock();
        state.asleep = !on;
        self.tell_shown(&mut state);
    }

    /// Where the guest last put its cursor's hot spot.
    fn position(&self) -> (u16, u16) {
        let p = self.position.load(Ordering::Acquire);
        let half = |v: u32| u16::try_from(v & 0xffff).unwrap_or(u16::MAX);
        (half(p >> 16), half(p))
    }
}

/// The display's screen without a window: the mode it offers without one, and nothing shown on
/// the host. It keeps the guest's scanout for the host's reads of the display (`read`).
struct HeadlessScreen {
    /// The guest's scanout.
    shown: Arc<Shown>,
}

impl Screen for HeadlessScreen {
    fn mode(&self) -> Mode {
        FALLBACK_MODE
    }

    fn scanout(&self, scanout: Option<Scanout>) {
        *self.shown.scanout.lock() = scanout;
    }

    fn present(&self) {}

    fn cursor_image(&self, _image: Option<CursorImage>) {}

    fn cursor_moved(&self, _x: i32, _y: i32, _visible: bool) {}

    /// The host's reads of the display show black while the panel is off.
    fn power(&self, on: bool) {
        self.shown.asleep.store(!on, Ordering::Release);
    }
}

/// The device's [`Screen`]: each call goes to the screen it holds, which the host's reads of
/// the display share, and the cursor's to the host as well ([`HostCursor`]).
struct SharedScreen {
    /// The window's screen, or the headless one.
    screen: Arc<dyn Screen>,
    /// The cursor as the host is told it.
    cursor: Arc<HostCursor>,
}

impl Screen for SharedScreen {
    fn mode(&self) -> Mode {
        self.screen.mode()
    }

    fn scanout(&self, scanout: Option<Scanout>) {
        self.screen.scanout(scanout);
    }

    fn present(&self) {
        self.screen.present();
    }

    fn cursor_image(&self, image: Option<CursorImage>) {
        self.cursor.image(image.as_ref());
        self.screen.cursor_image(image);
    }

    fn cursor_moved(&self, x: i32, y: i32, visible: bool) {
        self.cursor.moved(x, y, visible);
        self.screen.cursor_moved(x, y, visible);
    }

    fn power(&self, on: bool) {
        self.cursor.power(on);
        self.screen.power(on);
    }
}

impl WindowScreen {
    /// The window of the display on `port` (its id) over `region`, opened with its sink on
    /// `platform` where the platform draws one (no sink: the window never opens), and the screen
    /// over it. The window is titled as the port's, and opens on the port's host screen
    /// (`reims-vgpu-window`'s `handle::title` and `screens::screen_for_port`).
    fn open<P: ApiPlatform>(
        platform: &Arc<P>,
        region: &Arc<Region>,
        presented: &Arc<Presented<P>>,
        mut config: WindowConfig,
        port: u64,
    ) -> (WindowLink, WindowScreen) {
        config.keeps_its_size = true;
        let first = config.mode.unwrap_or((1920, 1080));
        let window = Window::new(config, port, reims_vgpu_window::handle::title(port));
        window.offered_mode(first.0, first.1);
        // In the crate's tests the display runs at 60 Hz, asking the host for no screen: on Linux
        // the ask starts the windows' UI thread.
        let refresh_mhz = if cfg!(test) {
            FALLBACK_MODE.refresh_mhz
        } else {
            reims_vgpu_window::screens::screen_for_port(port)
                .map_or(FALLBACK_MODE.refresh_mhz, |s| hz_to_mhz(s.refresh_hz))
        };
        let shown = Arc::new(Shown {
            black_undriven: port > 0,
            ..Shown::default()
        });
        if let Some(sink) = scanout_sink(platform, region, &shown, presented) {
            window.attach(sink);
            // Presents wait for the window's draw only where it has a sink to draw them.
            let _ = presented.window.set(Arc::clone(&window));
        }
        let presented = Arc::clone(presented);
        let screen = WindowScreen {
            window: Arc::clone(&window),
            shown,
            shows_surface: Box::new(move || presented.shows_surface()),
            refresh_mhz,
            cursor: Mutex::new(CursorState {
                image: false,
                visible: true,
                asleep: false,
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
/// and a few plain textures check it took: one whose index lies inside the guest's ids is logged
/// as an error. The index is a resource id's low word: rt's ids carry its word for the texture's
/// descriptor in the high one (metal2vulkan-runtime's `ResourceId`).
fn reserve_guest_texture_ids<P: Platform>(platform: &P, device: &P::Device) {
    let ids: Vec<u64> = guest_texture_ids().collect();
    if let Err(raise) = refusing(|| device.reserve_resource_indices(0, &ids)) {
        log::error!(
            kind:% = raise;
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
        .map(|t| t.gpu_resource_id() & 0xffff_ffff)
        .find(|id| guest_texture_ids().contains(id))
    {
        log::error!(
            "a texture of the host's took resource index {id}, one of the guest's ({} to {}): \
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
    /// The window's size in pixels as last drawn into, else the mode it opened for, its width
    /// rounded down to a multiple of 16 and its height to even ([`geometry::offered_mode`]), at
    /// its host screen's rate, with the backing scale it was last drawn at (none before its first
    /// draw).
    fn mode(&self) -> Mode {
        let drawn = *self.shown.drawable.lock();
        let (width, height) = geometry::offered_mode(drawn.map_or_else(
            || {
                let (w, h) = self.window.offered().unwrap_or((1920, 1080));
                (u64::from(w), u64::from(h))
            },
            |d| d.pixels,
        ));
        let clamp = |v: u64| u32::try_from(v.max(1)).unwrap_or(u32::MAX);
        Mode {
            width: clamp(width),
            height: clamp(height),
            refresh_mhz: self.refresh_mhz,
            scale_120ths: drawn.map_or(0, |d| scale_in_120ths(d.scale)),
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

    /// The window draws the scanout again where it shows it; while it shows a presented surface,
    /// or black while the display's panel is off, a refresh draws nothing, and the guest's
    /// presents draw (`Presented`).
    fn present(&self) {
        if !self.shown.asleep() && !(self.shows_surface)() {
            self.window.new_frame();
        }
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

    /// The window shows or hides the guest's cursor as the guest does, and takes the place the
    /// guest put its hot spot, each axis kept to 0 to 65535 (`Window::cursor_at`): where the
    /// guest moved it on its own, the window draws it (the module doc).
    fn cursor_moved(&self, x: i32, y: i32, visible: bool) {
        let mut cursor = self.cursor.lock();
        cursor.visible = visible;
        cursor.update(&self.window);
        drop(cursor);
        let axis = |v: i32| u16::try_from(v.max(0)).unwrap_or(u16::MAX);
        self.window.cursor_at((axis(x), axis(y)));
    }

    /// The window shows black, and no guest cursor, while the panel is off, and what the guest
    /// shows again once it is on: a frame is drawn at each change.
    fn power(&self, on: bool) {
        self.shown.asleep.store(!on, Ordering::Release);
        let mut cursor = self.cursor.lock();
        cursor.asleep = !on;
        cursor.update(&self.window);
        drop(cursor);
        self.window.new_frame();
    }
}

/// The window's sink over `region` on `platform` ([`sink::ScanoutDraw`]): the rt platform's
/// swapchain presenter (`rt_window`); `None` on any other platform (the testkit's), or where the
/// platform refuses a command queue.
fn scanout_sink<P: ApiPlatform>(
    platform: &Arc<P>,
    region: &Arc<Region>,
    shown: &Arc<Shown>,
    presented: &Arc<Presented<P>>,
) -> Option<Arc<dyn FrameSink>> {
    let any: Arc<dyn std::any::Any + Send + Sync> = Arc::clone(platform) as _;
    let presented_any: Arc<dyn std::any::Any + Send + Sync> = Arc::clone(presented) as _;
    if let (Ok(rt), Ok(presented)) = (
        Arc::clone(&any).downcast::<reims_vgpu_vulkan::RtPlatform>(),
        Arc::clone(&presented_any).downcast::<Presented<reims_vgpu_vulkan::RtPlatform>>(),
    ) {
        return rt_window::sink(rt, region, shown, &presented);
    }
    let _ = (any, presented_any, region, shown);
    None
}

/// `SURFACE_CACHE_EVERY` where the platform caches every surface in the per-shape form (`shapes`,
/// [`ApiPlatform::SURFACE_SHAPES`]) and the device offers the Shared pool (`pool`): the guest's
/// kext then names every block in place cached, its tables grow rather than refuse, and it and
/// the bundle count each copy block of a surface, a second copy beside the surface's device
/// cache. A platform that lays textures over the surfaces' pages, and a device with no pool,
/// offer it never: their guests keep the surfaces' blocks as before.
fn cache_every(shapes: bool, pool: bool) -> u64 {
    if shapes && pool {
        RGFX_FEATURE_SURFACE_CACHE_EVERY
    } else {
        0
    }
}

/// `MANAGED_AT_COPY` where the GPU shares the CPU's memory (`unified`), the platform keeps a pin
/// only inside a block it knows (`pins_in_blocks`, [`ApiPlatform::PINS_IN_BLOCKS`]) and the device
/// offers the Shared pool (`pool`): a `Managed` buffer is then its bytes, at its CPU copy's
/// address, and every CPU copy lies in a block the platform knows, a slab's in a pool block at the
/// address the host gave it (`POOL_ADDRESSES`), a no-copy one's in a pages block's view at its
/// import's (`IMPORT_ADDRESSES`), so the platform keeps the pin. A GPU of its own memory keeps a
/// `Managed` buffer's GPU copy elsewhere; a platform that keeps every pin where the guest picks
/// (the testkit's) knows no block a CPU copy lies in; and with no pool the slabs' copies lie in
/// heaps over the region, which no block the platform knows holds: none offers it, and their
/// guests pick each buffer an address of its own.
fn managed_at_copy(unified: bool, pins_in_blocks: bool, pool: bool) -> u64 {
    if unified && pins_in_blocks && pool {
        RGFX_FEATURE_MANAGED_AT_COPY
    } else {
        0
    }
}

/// The video caps of the host's hardware video `video` on a device of `features`: its codecs,
/// each with what an encode takes beyond it, and either the layout a decode lands its pictures in
/// place in, or `RGFX_VIDEO_CAPS_OUTPUT_CACHED` and no layout, where the guest asks its outputs
/// cached. Pictures land in device caches only where the kext makes them: of every surface it
/// asks one for (`SURFACE_CACHE`), or of those whose textures would not lie in place in their
/// pages, where the GPU shares the CPU's memory (`SURFACE_CACHE_REFUSED`), and where `video` lands
/// pictures in them. The guest's decoder then asks a cache for its outputs with their layout, as
/// for any other surface, and a picture lands in place only in a surface's pages. A platform of
/// the per-shape form (`shapes`, [`ApiPlatform::SURFACE_SHAPES`]) caches every surface and lays
/// no texture over its pages, so the guest asks its outputs cached wherever it offers video.
fn video_caps<P: Platform>(
    features: u64,
    shapes: bool,
    video: &dyn GraphicsVideo<P>,
) -> RgfxVideoCaps {
    let caches =
        u64::from(RGFX_FEATURE_SURFACE_CACHE) | u64::from(RGFX_FEATURE_SURFACE_CACHE_REFUSED);
    let cached = shapes || (features & caches != 0 && video.lands_in_device_caches());
    let mut caps = RgfxVideoCaps {
        version: RGFX_VIDEO_CAPS_VERSION,
        flags: if cached {
            RGFX_VIDEO_CAPS_OUTPUT_CACHED
        } else {
            0
        },
        ..RgfxVideoCaps::default()
    };
    let slots = caps
        .codecs
        .iter_mut()
        .zip(&mut caps.layouts)
        .zip(&mut caps.encode)
        .zip(&mut caps.decode);
    for ((((slot, layout), encode), decode), codec) in slots.zip(video.caps()) {
        if !cached {
            *layout = video.layout(&codec);
        }
        *encode = video.encode_caps(&codec);
        *decode = video.decode_caps(&codec);
        *slot = codec;
        caps.codec_count += 1;
    }
    caps
}

#[cfg(test)]
mod tests;

/// A download's answer, kept until one of its two ends gives it, and the two: OK once the pages
/// hold the block at `block`'s cache, or refused where the platform raised at the copy, logged.
/// Neither given, the answer is still in the first.
#[allow(clippy::type_complexity)] // the answer and its two ends
fn download_answers(
    block: u64,
    answer: CompleteAnswer,
) -> (
    Arc<Mutex<Option<CompleteAnswer>>>,
    Box<dyn FnOnce() + Send>,
    Box<dyn FnOnce(Raise) + Send>,
) {
    let answer = Arc::new(Mutex::new(Some(answer)));
    let (give, refuse) = (Arc::clone(&answer), Arc::clone(&answer));
    let done: Box<dyn FnOnce() + Send> = Box::new(move || {
        if let Some(answer) = give.lock().take() {
            answer.answer(RGFX_STATUS_OK);
        }
    });
    let failed: Box<dyn FnOnce(Raise) + Send> = Box::new(move |raise| {
        log::warn!(
            target: "reims_vgpu::graphics",
            "the block at {block:#x}: the platform raised at its device cache's download \
             ({raise:?})"
        );
        if let Some(answer) = refuse.lock().take() {
            answer.answer(RGFX_STATUS_REFUSED);
        }
    });
    (answer, done, failed)
}

/// The context (the process) `message` came from: for a message off a queue's ring, the queue's;
/// on ring 0, that of the object it creates or destroys, which the kext checked is the sender's.
fn sender_context(poster: &CompletionPoster, message: &[u8]) -> u32 {
    let obj_id = poster
        .queue_obj_id()
        .unwrap_or_else(|| message_identity(message).2);
    (obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32
}

/// The key the no-copy buffers and heaps of context `context` being made are unsettled under in
/// `GraphicsControl::pages_buffers`, apart from their blocks' offsets, which lie below 2^63.
fn pages_context(context: u32) -> u64 {
    1 << 63 | u64::from(context)
}

/// The surface worker's key the seals of context `context`'s groups of views run under
/// (`GraphicsControl::seal_group`), apart from its blocks' offsets, which lie below 2^63, and the
/// views' teardowns' key (`u64::MAX`).
fn group_key(context: u32) -> u64 {
    0xffff_fffe_0000_0000 | u64::from(context)
}

/// The surface worker's key of the regroups' passes (`decode::surface_pages`, "Regroups"): one at
/// a time, apart from every block's and group's jobs.
const REGROUP_KEY: u64 = 0xffff_fffd_0000_0000;

/// The name of the `BUFFER_OVER_PAGES` or `HEAP_OVER_PAGES` `message`, and its pages block's
/// offset; `None` for any other message.
fn over_pages_block(message: &[u8]) -> Option<(&'static str, u64)> {
    match RgfxMsgHeader::read_from_prefix(message).ok()?.0.r#type {
        RGFX_CMD_BUFFER_OVER_PAGES => RgfxBufferOverPages::read_from_prefix(message)
            .ok()
            .map(|(m, _)| ("BUFFER_OVER_PAGES", m.block_offset)),
        RGFX_CMD_HEAP_OVER_PAGES => RgfxHeapOverPages::read_from_prefix(message)
            .ok()
            .map(|(m, _)| ("HEAP_OVER_PAGES", m.block_offset)),
        _ => None,
    }
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
