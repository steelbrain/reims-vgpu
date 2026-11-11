//! The device caches of IOSurfaces (`RGFX_FEATURE_SURFACE_CACHE`; docs/graphics-kext/design.md,
//! Decided 14, "The device cache", which says each choice's class).
//!
//! A GPU that does not share the CPU's memory keeps the GPU's copy of an IOSurface in its own
//! memory, the surface's device cache, and moves bytes between it and the surface's pages only
//! where IOSurface's locking asks it to. A block the kext marks cached
//! ([`RGFX_SURFACE_PAGES_CACHED`]) gets one here: on a host whose GPU is such a GPU, any surface's
//! block the kext caches; on one whose GPU shares the CPU's memory, only a block the device
//! answered needs one, its textures unable to lie in place over its pages
//! ([`super::surface_pages`], "Device caches where the pages cannot serve"). A cache is a Private
//! buffer of the surface's bytes in
//! the pages' own layout, which every `BUFFER_OVER_SURFACE_PAGES` buffer of the block is, so every
//! texture over the surface, of any shape, plane or format, lies over the same bytes, as the
//! textures over the pages did. The surface's pages are the other end of each copy, through their
//! view ([`super::surface_pages`]) or by their runs (below).
//!
//! - **Fill.** A cache is made as zeros, on the transfer queue before any use, and owes its first
//!   fill from the pages, as of the CPU's generation of the surface in the block's header as it
//!   is made: the kext's count of the surface's read-write locks, which IOSurface marks at the
//!   lock. The first thing that reads the cache makes the fill before it reads
//!   ([`SurfaceCaches::fill_owed`]): a use a command buffer names, a present, a command buffer of
//!   render, compute or blit work in the process that made the first buffer over the block, and
//!   a buffer over the block made by any other. So a texture over the surface that a command
//!   buffer reaches with no use named (through an argument buffer, with no `useResource:`, which
//!   Metal leaves the app to make resident) reads the pages, not zeros, as only a process with a
//!   buffer over the block reaches it. A command buffer whose work is decodes alone, whose decode
//!   writes the whole of both planes of the surface (a decoder's new output), takes the fill as
//!   met instead, and none is made ([`SurfaceCaches::before_one`]): the picture lands over the
//!   zeros, and the cache's copies into the pages copy those planes' rows alone from then on
//!   (Download, below), so every byte of the pages beside them stays as the guest's CPU wrote it
//!   (CoreVideo's header among them). A decode that answers no picture (one the stream's random
//!   access skips) makes the fill, before anything reads the cache. A picture that fails to land,
//!   or lands short of the planes, leaves the texels it did not write zero, where a fill would
//!   have left the pages' bytes: Metal leaves a failed decode's output undefined, and a new
//!   surface's texels are the kernel's zero-filled pages unless its client wrote them first,
//!   which a read-write lock does and the use's newer generation then fills (below); no other
//!   memory's bytes reach the guest, the cache made as zeros. A command
//!   buffer names each cached surface it uses (`RGFX_OP_SURFACE_USE`), with the CPU's generation
//!   of the surface when it was committed. Where the cache holds an older generation, or none,
//!   the cache is filled from the pages before the buffer's work; a buffer of the same generation
//!   decoded later waits for that fill, on any queue. A block named again over the same pages
//!   keeps its cache, filled from the pages again as it is named, as of the generation in the
//!   block's header then, as a new cache is: a texture over it that a command buffer reaches
//!   with no use named reads the pages as they were then. One of an older
//!   generation than the cache's (two queues' buffers committed apart and taken in the other
//!   order) fills nothing. A buffer's write is in the pages before a newer generation's fill
//!   wherever a read-write lock waited for it: a process names its buffer's writes to the kext
//!   before the commit returns, so a lock after the commit waits for them and has them
//!   downloaded, and the download comes before the fill. A commit on one thread racing a lock of
//!   the surface on another, which IOSurface leaves undefined, can name its write after the lock
//!   looked: its write is in no download, and the fill of the newer generation copies the
//!   pages, the CPU's writes, over it. On a block that reads back (Early copy, below), the
//!   write's early copy lands in the pages whenever the GPU runs it, among the CPU's writes, as
//!   a GPU's write lands where the GPU shares the CPU's memory, and the fill waits for it.
//! - **Download.** The pages are written from the cache only at a `SURFACE_DOWNLOAD` (a CPU lock
//!   after a GPU write), at the block's `BLOCK_RELEASE` and where the kext names the block's pages
//!   again (`SURFACE_PAGES`), and only where a buffer that may write the surface was committed
//!   since the cache was last copied either way. A release copies whether or not the surface has
//!   a client left (`RGFX_BLOCK_RELEASE_NO_CLIENT` changes nothing): a mach port or an XPC object
//!   holds a surface with no client, and the process that looks it up from one reads its pages.
//!   Only a release the kext flags `RGFX_BLOCK_RELEASE_UNHELD` (no port or object held the
//!   surface either) whose block's header still counts no client as the copy would be made
//!   ([`CLIENTS_OFFSET`]) drops the cache uncopied: no process holds the surface to read its
//!   pages, which the kext frees or gives out again only once the release is answered, and the
//!   kernel zero-fills what it gives out again.
//!   A copy into the pages (a download's, a release's, an early copy) of a cache whose surface's
//!   layout the kext's ask gave (on a GPU that shares the CPU's memory), or whose first fill a
//!   decode took as met (Fill, above, the layout its planes' textures give), copies the surface's
//!   texels alone, each plane's rows' bytes ([`texel_spans`]): the bytes beside them, between
//!   rows or outside the planes, are the guest's CPU's alone, as no texture's texel lies there
//!   and the device makes no buffer over an IOSurface, so they stay as it wrote them, whatever a
//!   render over the cache leaves in its own. A fill copies every byte in. Nothing else reads
//!   those bytes of the cache, so its buffer declares them unobserved
//!   ([`GpuDevice::new_buffer_of_length_padding_unobserved`]): a texture over it whose rows lie
//!   with bytes between them may hold and change them, and lies over the cache's memory where
//!   the platform would otherwise make it a copy of its own (on the Vulkan platform, metal2vulkan's
//!   shadow, which each command buffer copies in before its use and back after its write).
//!   Where the kext gives the surface's layout (its ask, or on a GPU of its own memory the cached
//!   or deferred block's own `SURFACE_PAGES`), the platform places the surface's first plane's
//!   first byte where its textures over the cache's memory may start in place ([`aligned_byte`],
//!   [`CacheLayout`]), whichever way the cache is made, its exportable one among them: CoreVideo
//!   lays a frame's planes 32 bytes into their surface, off every linear image's alignment
//!   measured, and the textures over a decoder's planes are then images of the cache's memory,
//!   not the runtime's shadows, where the platform binds images in that memory. On the RTX 5080
//!   (driver 615.71) metal2vulkan makes each exportable buffer its own dedicated allocation, to
//!   which it binds no image, so every texture over an exportable cache stays its shadow wherever
//!   the byte lies (**measured** 2026-11-11, a macOS 26.6.1 guest's 4K60 videos in Safari: 2850
//!   of the 4063 shadowed IOSurface textures named the exported memory, the NV12 and P010
//!   planes' 1586 among them). Only the ask's layout makes the copies take texels alone.
//!   A download or a release copies nothing where, as its copy is made (at once, or in the
//!   finish of a buffer begun before it), the CPU's generation in the block's header
//!   ([`CPU_GEN_OFFSET`]) has moved past the cache's: the lock that asked gave up waiting for the
//!   answer, moved the generation, and let the CPU write the pages, whose writes are the newer,
//!   as at a fill. A download or a release that copies nothing is answered once a copy into the
//!   pages still in flight lands, and a `SURFACE_PAGES` over pages a release still copies into
//!   (the kext made a new block of the surface before the release's answer) waits for the copy
//!   before it is carried out ([`Caches::wait_copies_into`]), and for the release itself where
//!   its context's lane has not carried it out yet (`reims_vgpu_graphics`'s `device::lanes`).
//! - **Early copy.** A surface the CPU reads back after each GPU write (a decoder's output, or a
//!   pixel transfer's, that its client locks each frame to read) is copied into its pages as the
//!   write ends, not at the lock that follows it ([`SurfaceCaches::early_copies`]). Once a
//!   download copied a write that was the only one since the cache's last copy, the block reads
//!   back, and each later writer that is again the only one is followed by a command buffer of
//!   the host's that copies the cache into the pages; the download that follows is answered as
//!   that copy lands. A writer that holds a video encoder has it committed on its own queue right
//!   behind it. Any other writer's goes on the host's read-back queue, waiting on the GPU for the
//!   writer's end, as one on the writer's queue would hold its later work behind it (a window
//!   server drawing into a surface that another process captures). A write that no download
//!   followed ends it, until a download copies again; so a surface locked now and then takes at
//!   most one copy no lock asked for. A download of pages the host copies through the CPU goes as
//!   before, the copy running only where the pages lie in a buffer.
//! - **Order.** Fills and downloads run on the host's own transfer queue, in the order they are
//!   asked, each after every command buffer that used the cache before it: each such buffer
//!   signals its queue's end event as it ends, and the copy waits for the latest value of each
//!   queue's. So a frame drawn from the cache before a CPU write still reads what it read. A copy
//!   asked while such a buffer is still walked (begun and not committed) is made in that buffer's
//!   finish, and a buffer begun after it that reads what it writes, or writes what it reads,
//!   waits for it on the GPU, on the cache's own event ([`Caches`]): no begin of a buffer, no
//!   present and no ring-0 copy waits on the CPU for another queue's walk.
//! - **The host's own writes.** A device of the host that writes a surface's pages itself (a
//!   video arm's frame, [`crate::device_writes`]) moves no generation, so each cache's pages
//!   are watched: a write over them makes the cache's next use fill it whatever its generation,
//!   and a download or a release before that copies nothing, the pages being newer than the
//!   cache.
//! - **Deferred blocks.** Where the guest took `RGFX_FEATURE_SURFACE_DEFERRED`, the kext names a
//!   fresh surface only the GPU has touched with no pages
//!   ([`RGFX_SURFACE_PAGES_DEFERRED`](reims_vgpu_graphics::wire::RGFX_SURFACE_PAGES_DEFERRED)), as a
//!   GPU of its own memory keeps such a surface in that memory alone: its cache is made as zeros,
//!   on the transfer queue before any use, at the CPU's generation 0, with no pages to fill from,
//!   download into or watch. The kext names the pages at the first CPU lock that needs them
//!   ([`RGFX_SURFACE_PAGES_ATTACH`](reims_vgpu_graphics::wire::RGFX_SURFACE_PAGES_ATTACH)), and the
//!   cache takes them as they are, keeping its bytes, its generation and whether a buffer wrote
//!   it, so the download that follows copies what the GPU wrote. A release of a block whose pages
//!   were never named copies nothing. Pages the host makes no view of, or no buffer over, at that
//!   naming are taken all the same, copied (`CachePages::Copied`): each fill and download goes
//!   through a Shared buffer of its own, which the host's CPU fills from the pages, or empties into
//!   them once the GPU's copy is done, through its mapping of the guest's RAM, in place of the
//!   refusal that left the CPU lock asking with pages that held neither the GPU's writes nor a
//!   way for its own to reach the GPU. Such caches hold at most the device's budget of them
//!   (`DEFERRED_BUDGET_LO`), from their make to their release, named or not; a block past it, or
//!   one whose cache the platform does not make, is refused with no cache, and the kext names its
//!   pages instead.
//!
//!   On a GPU that shares the CPU's memory, where the guest took
//!   `RGFX_FEATURE_SURFACE_DEFERRED_ASK`, a deferral asks with the surface's layout
//!   ([`RGFX_SURFACE_PAGES_ASK_CACHE`](reims_vgpu_graphics::wire::RGFX_SURFACE_PAGES_ASK_CACHE)), and
//!   the block is deferred only where the surface would need a device cache whatever pages it
//!   comes to have ([`GpuDevice::fresh_surface_never_in_place`]); its cache is then laid out as an
//!   asked one's ([`super::surface_pages`], "Device caches where the pages cannot serve"). Every
//!   other is refused with nothing made, counted, and the kext names its pages asking, as it does
//!   for a block refused past the budget: the pages then decide, as on a device that defers
//!   nothing.
//!
//! - **By runs or by a view.** A cache's copies go between it and the pages' runs in the guest's
//!   RAM where the platform copies by them ([`GpuDevice::copies_guest_ram`]; the block takes no
//!   view, [`super::surface_pages`]), else through a buffer over the surface's view. A copy by
//!   runs the platform refuses makes the view there and then, the copy goes through it, and the
//!   cache copies through it from then on: what a command buffer wrote of the cache reaches the
//!   pages all the same. The runtime orders a copy by runs against nothing made over the view's
//!   mapping of the pages, so the two never share a command buffer: each copy has a transfer
//!   command buffer of its own, and a refused copy by runs encodes nothing in it.
//!
//! [`GpuDevice::copies_guest_ram`]: reims_vgpu_vulkan::core::platform::device::GpuDevice::copies_guest_ram
//! [`GpuDevice::fresh_surface_never_in_place`]: reims_vgpu_vulkan::core::platform::device::GpuDevice::fresh_surface_never_in_place
//! [`GpuDevice::new_buffer_of_length_padding_unobserved`]: reims_vgpu_vulkan::core::platform::device::GpuDevice::new_buffer_of_length_padding_unobserved
//!
//! What the guest observes is IOSurface's contract under its locks: a lock after a GPU write sees
//! it, and a GPU use after a read-write lock sees the CPU's writes. A CPU write made without a
//! read-write lock reaches the GPU at the next fill, as on a GPU of its own memory.
//!
//! **Why a buffer, not an image (measured).** A texture over a Private buffer is a linear image in
//! the GPU's memory on the Vulkan platform, and renders and samples as fast as a Private texture,
//! where one over host pages is a shadow filled across PCIe at each first use. On an RTX 5080
//! Laptop GPU (NVIDIA 615.71.09, Linux 7.2.8, metal2vulkan 45b3427434), a 3840x2400 BGRA8Unorm
//! target, commit to completion, median of 60 frames, two rounds:
//!
//! | frame | Private texture | over a Private buffer | over host pages |
//! |---|---|---|---|
//! | a load-store pass, one quad in a 2400x2400 scissor | 96, 219 us | 105, 169 us | 1497, 1507 us |
//! | the same, a 256x256 scissor | 41 us | 41 us | 276 us |
//! | six full-target draws, each sampling a source of the target's size | 408, 439 us | 436, 440 us | 8563, 8570 us |
//!
//! The runtime renders a linear image in place where the driver reports its format's linear
//! tiling as a color attachment, and on NVIDIA under `VK_NV_linear_color_attachment` where every
//! color attachment of the pass is linear; elsewhere it renders through a stand-in copied within
//! the GPU's memory. Neither is measured on a discrete RADV or ANV device yet.
//!
//! [`RGFX_SURFACE_PAGES_CACHED`]: reims_vgpu_graphics::wire::RGFX_SURFACE_PAGES_CACHED

use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use reims_vgpu_graphics::ops::{RGFX_SURFACE_USE_WRITES, RgfxOpSurfaceUse};
use reims_vgpu_graphics::region::Region;
use reims_vgpu_graphics::wire::{
    RGFX_BLOCK_RELEASE_NO_CLIENT, RGFX_BLOCK_RELEASE_UNHELD, RgfxBufferOverSurfacePages,
};
use reims_vgpu_video::core::device_memory::{MemoryIdentity, MemoryLife};
use reims_vgpu_vulkan::core::host::HostPtr;
use reims_vgpu_vulkan::core::num::{ToU64 as _, ToUsize as _};
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer, CommandQueue, CompletedInfo};
use reims_vgpu_vulkan::core::platform::device::GpuDevice;
use reims_vgpu_vulkan::core::platform::event::{EventRef, SharedEvent as _};
use reims_vgpu_vulkan::core::platform::os::clock::MonoTime;
use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
use reims_vgpu_vulkan::core::platform::types::NsRange;
use reims_vgpu_vulkan::core::platform::values::PagesPlane;
use reims_vgpu_vulkan::core::platform::{Handle, Platform};
use reims_vgpu_vulkan::core::raise::{OnFault, Raise, RaiseResult, refusals};
use reims_vgpu_vulkan::core::sync::{Condvar, Mutex, MutexGuard};

use super::refusing;
use super::surface_pages::{SurfaceView, Unmade, buffer_over};
use crate::device_writes::{self, Watcher};

/// The copies run under a live raise handler (the ring loop's, or the entry of the ring-0 thread
/// that asks for them).
const ON: OnFault = OnFault::Return;

/// `MTLResourceStorageModePrivate`, in `MTLResourceOptions`.
pub(crate) const STORAGE_PRIVATE: u64 = 2 << 4;
/// `MTLResourceStorageModeShared`, in `MTLResourceOptions`.
const STORAGE_SHARED: u64 = 0;
/// `MTLResourceStorageModeMask`.
pub(crate) const STORAGE_MASK: u64 = 0xf << 4;

/// Where a cached block's header holds its surface's CPU generation, from the block's first
/// byte: a `u64` the kext writes at each read-write lock, and stores 0 in as it caches the block
/// (the kext's `kRgfxSurfaceCpuGenOffset`, which the processes read for `RGFX_OP_SURFACE_USE`).
pub(crate) const CPU_GEN_OFFSET: u64 = 16;

/// The key the host's read-back queue's end event is kept under among the command queues' end
/// events ([`SurfaceCaches::early_copies`]): an id no guest queue takes, as a context's object
/// ids count up from 1 and never reach 2^40 - 1 (`RGFX_CONTEXT_OBJ_ID_SHIFT`).
const READ_BACK: u64 = u64::MAX;

/// The CPU's generation of the surface whose cached block is at `block`, as the kext last wrote
/// it in the block's header ([`CPU_GEN_OFFSET`]); `None` where the header is not in `region`.
pub(crate) fn cpu_generation(region: &Region, block: u64) -> Option<u64> {
    region.load_u64(block.checked_add(CPU_GEN_OFFSET)?)
}

/// Where a cached block's header holds the count of its surface's clients, from the block's first
/// byte: a `u32` the kext writes as it caches the block, as a process holds it again, as each
/// client goes, and past 0 where a process takes the surface again while a release flagged
/// `RGFX_BLOCK_RELEASE_UNHELD` is unanswered (the kext's `kRgfxSurfaceClientsOffset`).
pub(crate) const CLIENTS_OFFSET: u64 = 12;

/// The count of the clients of the surface whose cached block is at `block`, as the kext last
/// wrote it in the block's header ([`CLIENTS_OFFSET`]); `None` where the header is not in
/// `region`.
pub(crate) fn surface_clients(region: &Region, block: u64) -> Option<u32> {
    let bytes = region.read(block.checked_add(CLIENTS_OFFSET)?, 4)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

/// One surface's device cache.
pub(crate) struct SurfaceCache<P: Platform> {
    /// The cache: a Private buffer of the surface's bytes, which the block's buffers are.
    device: P::Buffer,
    /// The other end of each copy: the surface's own pages; `None` for a deferred block whose
    /// pages the kext has not named yet.
    pages: Option<CachePages<P>>,
    /// A deferred block's carried pages, while its pages are not named
    /// ([`RGFX_FEATURE_SURFACE_DEFERRED_CARRY`](reims_vgpu_graphics::wire::RGFX_FEATURE_SURFACE_DEFERRED_CARRY)):
    /// what its first fill copies in, over its make's zeros.
    carried: Option<Arc<CarriedPages>>,
    /// The surface's bytes.
    length: u64,
    /// The surface's planes, as the kext's ask laid it out or a decode wrote it whole; 0 where
    /// neither named them. Diagnostics only ([`CopyTrace`]).
    planes: usize,
    /// The bytes of the surface's texels, where the kext's ask laid the surface out (its
    /// planes' rows, [`texel_spans`]): a copy out of the cache copies those alone, so the bytes
    /// of the pages beside them, between rows or outside the planes, stay as the guest's CPU
    /// wrote them, whatever the cache holds there. `None` copies all of the surface's bytes.
    texels: Option<Vec<Range<u64>>>,
    /// The CPU's generation of the surface the cache holds; `None` until its first fill.
    generation: Option<u64>,
    /// The writes of the host's devices over the pages its last fill read ([`Watched::writes`]):
    /// the pages hold one it does not where they count more.
    host_read: u64,
    /// A command buffer that may write the cache was committed since the cache was last copied.
    written: bool,
    /// The transfer event's value at which the cache's last fill is done; 0 for none.
    filled_at: u64,
    /// The CPU's generation of the surface the cache's first fill is of, owed from the cache's
    /// make, which made it zeros, until the first fill of the cache or a decode that takes it as
    /// met (the module doc, "Fill"); `None` then, and for a deferred block's cache, which owes
    /// none but where it carries pages (generation 0, of them alone).
    first_fill: Option<u64>,
    /// The context of the first buffer over the block while its first fill is owed (`Some(None)`
    /// for one of no context): a command buffer of its with work that may reach the cache with no
    /// use named makes the fill ([`SurfaceCaches::fill_reached`]), as does a buffer over the
    /// block of any other context.
    owner: Option<Option<u32>>,
    /// The cache holds its make's zeros where a decode's picture is to land: a decode took its
    /// first fill as met ([`SurfaceCaches::before_one`]), and has not yet answered its picture
    /// ([`SurfaceCaches::decoded`]), which a decode that answers none makes.
    zeros: bool,
    /// Each queue whose command buffers used the cache since its last copy, with the value of the
    /// queue's end event the latest of them signals.
    users: HashMap<u64, u64>,
    /// The copies of the cache asked while command buffers they follow were begun and not
    /// finished ([`Deferred`]): how many were asked, and how many made, each its number in that
    /// order; and the event each signals to its number once it is made, which the command buffers
    /// begun after it that follow it wait for on the GPU, made at the first such buffer.
    asked: u64,
    made: u64,
    event: Option<P::SharedEvent>,
    /// Where the block was deferred, the block's size its `SURFACE_PAGES` gave, which a
    /// `BUFFER_OVER_SURFACE_PAGES` is checked against while the block has no view: its cache
    /// counts against the deferred budget until it goes, its pages named or not.
    deferred: Option<u64>,
    /// Whether the CPU reads the surface back after each GPU write, and the early copy that
    /// follows from it ([`SurfaceCaches::early_copies`]).
    read_back: ReadBack,
    /// The cache's last copy into its pages on the transfer queue ([`SurfaceCaches::write_back`]),
    /// until it lands: a download or a release that copies nothing waits for it, as for an
    /// early copy, and so does a use that writes the cache, on the GPU.
    out: Option<OutCopy>,
    /// The cache's life, where its memory was made for the host's video (exportable, or a
    /// dma-buf: [`CacheMaker::make`]), as the host's video keeps what it makes of the
    /// cache's memory (an import of it, [`SurfaceCaches::identity`]): it ends as the cache goes.
    /// `None` for any other cache, whose memory no device of the host's video reaches.
    life: Option<MemoryLife>,
}

/// A copy of a cache into its pages on the transfer queue ([`SurfaceCaches::write_back`]): the
/// transfer event's value it signals as it ends, and its landing.
struct OutCopy {
    value: u64,
    landing: Arc<Landing>,
}

/// A copy of a cache into pages asked and not yet landed, by the pages' host addresses in the
/// guest's RAM ([`SurfaceCaches::copies_into`]): a release's, from its ask on, as it may wait for
/// the buffers that write the cache before it is made, or one made on the transfer queue.
struct IntoPages {
    runs: Vec<Range<u64>>,
    landing: Arc<Landing>,
}

/// What a cache knows of the CPU's reads of its surface after the GPU's writes, and its early
/// copy into the pages ([`SurfaceCaches::early_copies`]).
#[derive(Default)]
struct ReadBack {
    /// Uses that may write the cache counted since it was last copied into its pages: by a
    /// download, a release or an early copy.
    writes: u32,
    /// A download copied a write that was the only one since the cache's last copy: each later
    /// writer that is again the only one is copied out as it ends.
    on: bool,
    /// An early copy that no download has asked for since.
    unread: bool,
    /// The last early copy, which a download or a release of the cache nothing wrote since waits
    /// for, and a use that writes the cache waits for, until it lands.
    copy: Option<EarlyCopy>,
}

/// An early copy of a cache into its pages: a command buffer of the host's on the queue of the
/// writer it follows, which signals that queue's end event at `value`.
struct EarlyCopy {
    queue: u64,
    value: u64,
    landing: Arc<Landing>,
}

/// A copy into a cache's pages until it lands (an early copy, or one on the transfer queue): the
/// answers waiting for it, given as the platform sees its command buffer done, and the threads
/// waiting for it ([`Caches::wait_copied_out`], [`Caches::wait_copies_into`]).
struct Landing {
    /// The answers waiting; `None` once it landed.
    waiting: Mutex<Option<Vec<Answer>>>,
    landed: Condvar,
    /// What its command buffer's completion gave of the copy, for a slow answer's line.
    times: Mutex<CopyTimes>,
}

/// What a copy into a cache's pages took, as its command buffer completed, for the line of an
/// answer that waited for it ([`CopyTrace`]): diagnostics only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CopyTimes {
    /// Its command buffer's `GPUStartTime` and `GPUEndTime` ([`CompletedInfo::gpu_times_ns`]),
    /// in nanoseconds of the platform's host clock: `None` where the platform gave none.
    pub(crate) gpu: Option<(u64, u64)>,
    /// How long the host's CPU took to copy a staging buffer's bytes into the pages after it
    /// (a copied cache's, [`CachePages::Copied`]); zero where the copy wrote the pages itself.
    pub(crate) staged: std::time::Duration,
}

/// What a `SURFACE_DOWNLOAD`'s or a `BLOCK_RELEASE`'s answer waited for once it was carried out:
/// the cache, how its pages took it, and the copy the answer came with, for the line of an
/// answer that came late ([`super::GraphicsDecoder::timed`]). Diagnostics only: nothing reads it
/// to decide anything.
#[derive(Default)]
pub(crate) struct CopyTrace {
    traced: Mutex<Traced>,
    /// The platform's monotonic clock, which times a staging buffer's copy into the pages;
    /// `None` times none.
    now: Option<Now>,
}

/// A read of the platform's monotonic clock.
pub(crate) type Now = Arc<dyn Fn() -> MonoTime + Send + Sync>;

/// [`CopyTrace`]'s record.
#[derive(Default)]
struct Traced {
    /// The cache's planes, as the kext's ask or a decode laid its surface out (0 where neither
    /// did), and its bytes; `None` where the block had no cache.
    cache: Option<(usize, u64)>,
    /// How the copy reaches the pages ([`CachePages`]): "a view", "runs" or "a staging buffer".
    into: &'static str,
    /// `Some` with the queues whose command buffers that used the cache the copy waited for on
    /// the GPU, where the answer made a copy; `None` where it waited for one in flight, or for
    /// nothing.
    users: Option<usize>,
    /// The copies the answer waited for: its own, or those in flight; it came with the last of
    /// them to end on the GPU.
    landings: Vec<Arc<Landing>>,
}

/// What a late answer's copy phase was ([`CopyTrace::split`]), as its line says it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CopySplit {
    /// The cache's planes and bytes; `None` where the block had no cache.
    pub(crate) cache: Option<(usize, u64)>,
    /// [`Traced::into`].
    pub(crate) into: &'static str,
    /// [`Traced::users`].
    pub(crate) users: Option<usize>,
    /// The copies the answer waited for: 0 where it waited for none.
    pub(crate) copies: usize,
    /// Of the copy phase: before the GPU had run the copy (the GPU's wait for the command
    /// buffers that used the cache, the transfer queue's work ahead of it, and the platform's
    /// sight of its end), the GPU's run of it, and the CPU's copy of a staging buffer after it.
    /// `wait` and `transfer` are `None` where the platform gave no GPU times.
    pub(crate) wait: Option<std::time::Duration>,
    pub(crate) transfer: Option<std::time::Duration>,
    pub(crate) staged: std::time::Duration,
}

impl CopyTrace {
    /// A trace whose copies time a staging buffer's copy into the pages by `now`.
    pub(crate) fn new(now: Now) -> Self {
        CopyTrace {
            traced: Mutex::default(),
            now: Some(now),
        }
    }

    /// The cache of the block at `block` in `caches`, and how its pages take a copy.
    fn cache<P: Platform>(&self, caches: &HashMap<u64, SurfaceCache<P>>, block: u64) {
        let mut traced = self.traced.lock();
        let Some(cache) = caches.get(&block) else {
            return;
        };
        traced.cache = Some((cache.planes, cache.length));
        traced.into = match &cache.pages {
            Some(CachePages::View(_)) => "a view",
            Some(CachePages::Runs { .. }) => "runs",
            Some(CachePages::Copied(_)) => "a staging buffer",
            None => "no pages",
        };
    }

    /// The answer made a copy, after the command buffers of `users` queues on the GPU.
    fn made(&self, users: usize, landing: &Arc<Landing>) {
        let mut traced = self.traced.lock();
        traced.users = Some(users);
        traced.landings = vec![Arc::clone(landing)];
    }

    /// The answer waits for `landings`, copies in flight.
    fn behind(&self, landings: &[Arc<Landing>]) {
        self.traced.lock().landings = landings.to_vec();
    }

    /// The copy phase of the answer, `copy` long, split ([`CopySplit`]).
    pub(crate) fn split(&self, copy: std::time::Duration) -> CopySplit {
        let traced = self.traced.lock();
        let times = (traced.landings.iter())
            .map(|landing| *landing.times.lock())
            .max_by_key(|times| times.gpu.map(|(_, end)| end))
            .unwrap_or_default();
        let transfer = times
            .gpu
            .map(|(start, end)| std::time::Duration::from_nanos(end.saturating_sub(start)));
        let wait =
            transfer.map(|transfer| copy.saturating_sub(transfer).saturating_sub(times.staged));
        CopySplit {
            cache: traced.cache,
            into: traced.into,
            users: traced.users,
            copies: traced.landings.len(),
            wait,
            transfer,
            staged: times.staged,
        }
    }
}

/// What runs once, as what it waits for is done: an answer waiting for a copy to land, or what a
/// copy out of a cache runs as it is carried out.
pub(crate) type Answer = Box<dyn FnOnce() + Send>;

impl Landing {
    /// A copy in flight.
    fn new() -> Arc<Self> {
        Arc::new(Landing {
            waiting: Mutex::new(Some(Vec::new())),
            landed: Condvar::new(),
            times: Mutex::new(CopyTimes::default()),
        })
    }

    /// [`Self::land`], its copy's command buffer having completed with `gpu`, its GPU times, and
    /// its staging buffer copied into the pages in `staged`.
    fn land_timed(&self, gpu: Option<(u64, u64)>, staged: std::time::Duration) {
        *self.times.lock() = CopyTimes { gpu, staged };
        self.land();
    }

    /// Whether the copy landed.
    fn landed(&self) -> bool {
        self.waiting.lock().is_none()
    }

    /// [`Self::landed`], its lock tried: `None` where it is held.
    fn try_landed(&self) -> Option<bool> {
        Some(self.waiting.try_lock()?.is_none())
    }

    /// Gives `done` once the copy landed: here where it has.
    fn then(&self, done: Answer) {
        let mut waiting = self.waiting.lock();
        match waiting.as_mut() {
            Some(answers) => answers.push(done),
            None => {
                drop(waiting);
                done();
            }
        }
    }

    /// The copy landed: the threads waiting are woken, and the answers given.
    fn land(&self) {
        let answers = self.waiting.lock().take();
        self.landed.notify_all();
        for done in answers.into_iter().flatten() {
            done();
        }
    }

    /// Waits for the copy to land, about `timeout` at most, in slices of at most
    /// [`FILL_WAIT_SLICE`] each woken at once by its landing, as a fill's wait does
    /// ([`Fills::wait`]); whether it landed.
    fn wait(&self, timeout: std::time::Duration) -> bool {
        let slice = FILL_WAIT_SLICE.min(timeout);
        let mut slices = (timeout.as_micros() / slice.as_micros().max(1)).max(1);
        let mut waiting = self.waiting.lock();
        while waiting.is_some() {
            if slices == 0 {
                return false;
            }
            if self.landed.wait_for(&mut waiting, slice).timed_out() {
                slices -= 1;
            }
        }
        true
    }
}

/// Gives `done` once every copy of `landings` has landed: at once where none is in flight.
fn after_landings(mut landings: Vec<Arc<Landing>>, done: Answer) {
    match landings.pop() {
        Some(landing) => landing.then(Box::new(move || after_landings(landings, done))),
        None => done(),
    }
}

/// The other end of a cache's copies: the surface's pages (the module doc).
enum CachePages<P: Platform> {
    /// A Shared buffer over the surface's view of its pages.
    View(P::Buffer),
    /// The pages' host addresses in the guest's RAM, in the surface's order, as runs, which the
    /// platform copies by; and what makes the view where it refuses.
    Runs {
        runs: Vec<Range<u64>>,
        unmade: Arc<Unmade<P>>,
    },
    /// A deferred block's pages the host made no view of, or no buffer over, at their naming:
    /// copied through a Shared buffer of each copy's own (the module doc).
    Copied(Arc<CopiedPages>),
}

/// How a deferred block's pages are named for its cache ([`SurfaceCaches::attach`]): by a view
/// of them, or by their runs.
pub(crate) enum Named<P: Platform> {
    View(Arc<SurfaceView<P>>),
    Runs(Vec<Range<u64>>, Arc<Unmade<P>>),
}

/// The pages of a copied cache ([`CachePages::Copied`]): each run of them the guest's RAM
/// mapping holds contiguously, as its host address and its bytes, in the surface's order.
struct CopiedPages {
    runs: Vec<(HostPtr, u64)>,
}

impl CopiedPages {
    /// The pages of `runs` (host address, bytes), merged where one ends where the next starts,
    /// where they hold at least `length` bytes; `None` otherwise.
    fn new(runs: &[(HostPtr, u64)], length: u64) -> Option<Self> {
        let mut merged: Vec<(HostPtr, u64)> = Vec::new();
        let mut total = 0u64;
        for &(at, bytes) in runs {
            total = total.checked_add(bytes)?;
            let follows = |&(start, len): &(HostPtr, u64)| {
                start.addr().to_u64().checked_add(len) == Some(at.addr().to_u64())
            };
            match merged.last_mut() {
                Some(last) if follows(last) => last.1 += bytes,
                _ => merged.push((at, bytes)),
            }
        }
        (total >= length && length != 0).then_some(CopiedPages { runs: merged })
    }

    /// Copies the first `length` bytes of the pages into `to`, or the other way where `into`,
    /// in the surface's order.
    ///
    /// # Safety
    /// `buffer` is `length` bytes the caller may read and write that no GPU work uses now; each
    /// run is the host's mapping of guest RAM pages the kext holds wired while their block is
    /// named, which the guest's CPU may write meanwhile, as on a GPU of its own memory.
    unsafe fn copy(&self, buffer: *mut u8, length: u64, into: bool) {
        let mut at = 0u64;
        for &(host, bytes) in &self.runs {
            if at >= length {
                break;
            }
            let n = bytes.min(length - at);
            let page = host.as_ptr::<u8>();
            // SAFETY: the caller's; `at + n` is at most `length`, and the run holds `n` bytes.
            unsafe {
                let here = buffer.add(at.to_usize());
                if into {
                    std::ptr::copy_nonoverlapping(here, page, n.to_usize());
                } else {
                    std::ptr::copy_nonoverlapping(page, here, n.to_usize());
                }
            }
            at += n;
        }
    }
}

/// A deferred block's carried pages ([`SurfaceCache::carried`]): each run of its surface's
/// resident pages as the surface's byte it starts at, its host address in the guest's RAM, which
/// the kext holds wired until the block's pages are named or its release is answered, and its
/// bytes, in the surface's order.
pub(crate) struct CarriedPages {
    runs: Vec<(u64, HostPtr, u64)>,
}

impl CarriedPages {
    /// The carried pages of a surface of `length` bytes, from `runs` (the surface's byte, host
    /// address, bytes), where each lies in the surface after the one before it and they hold
    /// some bytes; `None` otherwise.
    pub(crate) fn new(runs: Vec<(u64, HostPtr, u64)>, length: u64) -> Option<Self> {
        let mut end = 0u64;
        for &(at, _, bytes) in &runs {
            let run_end = at.checked_add(bytes)?;
            if at < end || bytes == 0 || run_end > length {
                return None;
            }
            end = run_end;
        }
        (!runs.is_empty()).then_some(CarriedPages { runs })
    }

    /// Their bytes.
    fn bytes(&self) -> u64 {
        self.runs.iter().map(|&(_, _, bytes)| bytes).sum()
    }
}

/// The host's transfer queue, on which every copy runs, and the event its fills signal.
struct Transfer<P: Platform> {
    /// Unshared ([`new_command_queue_unshared`]): a present of a surface waits for its fill, and
    /// a frame drawn from a cache for the fill before it ([`FillWait`]).
    ///
    /// [`new_command_queue_unshared`]: reims_vgpu_vulkan::core::platform::device::GpuDevice::new_command_queue_unshared
    queue: P::CommandQueue,
    /// The early copies behind writers that hold no video encoder
    /// ([`SurfaceCaches::early_copies`]), apart from the fills, so neither waits behind the
    /// other's writers; it signals the end event the caches keep under [`READ_BACK`]. `None`
    /// where the platform makes no queue: those writers take no early copy.
    read_back: Option<P::CommandQueue>,
    event: P::Event,
    /// The last value a copy signals.
    signalled: u64,
    /// How far its fills have finished.
    fills: Arc<Fills>,
}

impl<P: Platform> Transfer<P> {
    /// A transfer queue and its event, its fills telling `hook`; `None` where the platform makes
    /// either not.
    fn new(device: &P::Device, hook: &Arc<OnceLock<FillHook>>) -> RaiseResult<Option<Self>> {
        let Some(queue) = device.new_command_queue_unshared() else {
            return Ok(None);
        };
        let Some(event) = device.new_event(ON)? else {
            return Ok(None);
        };
        Ok(Some(Transfer {
            queue,
            read_back: device.new_command_queue(),
            event,
            signalled: 0,
            fills: Arc::new(Fills {
                done: AtomicU64::new(0),
                lock: Mutex::new(()),
                moved: Condvar::new(),
                hook: Arc::clone(hook),
            }),
        }))
    }
}

/// The longest a wait for a fill sleeps at once ([`Fills::wait`]): the slices count the wait's
/// timeout with no clock read; a fill's end wakes it at once.
const FILL_WAIT_SLICE: std::time::Duration = std::time::Duration::from_millis(10);

/// What is told as a fill finishes: the host's presents that wait for one ([`FillWait`]).
pub(crate) type FillHook = Arc<dyn Fn() + Send + Sync>;

/// How far the transfer queue's fills have finished: raised as the platform sees each fill done,
/// which wakes the threads waiting for one ([`Caches::wait_filled`]) and then tells the hook.
pub(crate) struct Fills {
    /// The highest transfer value whose fill has finished.
    done: AtomicU64,
    /// Held around a waiter's look at `done` and around the wake after it rises, so no wake is
    /// missed.
    lock: Mutex<()>,
    moved: Condvar,
    /// The host's presents, set once ([`Caches::on_fill_done`]).
    hook: Arc<OnceLock<FillHook>>,
}

impl Fills {
    /// Whether the fill that signals `value` has finished.
    fn reached(&self, value: u64) -> bool {
        self.done.load(Ordering::Acquire) >= value
    }

    /// The fill that signals `value` finished.
    fn finished(&self, value: u64) {
        self.done.fetch_max(value, Ordering::AcqRel);
        drop(self.lock.lock());
        self.moved.notify_all();
        if let Some(hook) = self.hook.get() {
            hook();
        }
    }

    /// Waits for the fill that signals `value`, about `timeout` at most, in slices of at most
    /// [`FILL_WAIT_SLICE`] each woken at once by a fill's end; whether it finished.
    fn wait(&self, value: u64, timeout: std::time::Duration) -> bool {
        let slice = FILL_WAIT_SLICE.min(timeout);
        let mut slices = (timeout.as_micros() / slice.as_micros().max(1)).max(1);
        let mut guard = self.lock.lock();
        while !self.reached(value) {
            if slices == 0 {
                return false;
            }
            if self.moved.wait_for(&mut guard, slice).timed_out() {
                slices -= 1;
            }
        }
        true
    }
}

/// A present's wait for its surface's fill ([`Caches::fill_now`]): the present is shown once the
/// fill has finished, as once its surface's writers have, waiting for nothing on the control
/// thread.
pub(crate) enum FillWait {
    /// A fill made, done once the transfer queue's fills reach `value`.
    At { fills: Arc<Fills>, value: u64 },
    /// A fill asked while command buffers it follows were begun and not finished, made in the
    /// finish of the last of them ([`SurfaceCaches::run_deferred`]).
    Deferred(Arc<DeferredFill>),
}

impl FillWait {
    /// Whether the fill has finished.
    pub(crate) fn done(&self) -> bool {
        match self {
            FillWait::At { fills, value } => fills.reached(*value),
            FillWait::Deferred(filled) => filled.done(),
        }
    }

    /// Waits for the fill on the calling thread, with no lock of the caches' held: for its make
    /// as long as it takes, as a begin waited for the command buffers it follows, then for its
    /// work about `timeout` at most ([`Fills::wait`]); whether it finished.
    fn wait(&self, timeout: std::time::Duration) -> bool {
        match self {
            FillWait::At { fills, value } => fills.wait(*value, timeout),
            FillWait::Deferred(filled) => match filled.wait_made() {
                Some((fills, value)) => fills.wait(value, timeout),
                None => true,
            },
        }
    }
}

/// What the caches did, for the log ([`SurfaceCaches::counts`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CacheCounts {
    /// Caches made.
    pub(crate) made: u64,
    /// Of the fills, a cache's first, owed since its make ([`SurfaceCaches::fill_owed`]), and
    /// those as its block was named again over the same pages
    /// ([`SurfaceCaches::fill_named_again`]).
    pub(crate) made_fills: u64,
    /// First fills a decode took as met, its picture landing over the cache's zeros
    /// ([`SurfaceCaches::before_one`]).
    pub(crate) decode_firsts: u64,
    /// Fills, and their bytes.
    pub(crate) fills: u64,
    pub(crate) filled_bytes: u64,
    /// Downloads, and their bytes.
    pub(crate) downloads: u64,
    pub(crate) downloaded_bytes: u64,
    /// Downloads answered at once: of a cache nothing wrote since its last copy, or one whose
    /// pages a device of the host wrote since.
    pub(crate) clean: u64,
    /// Fills a device of the host's write over the pages asked, and downloads and releases of a
    /// written cache it kept from copying.
    pub(crate) host_writes: u64,
    /// Releases of a written cache whose surface had no client left
    /// (`RGFX_BLOCK_RELEASE_NO_CLIENT`), copied into the pages all the same.
    pub(crate) unread: u64,
    /// Releases whose surface nothing but the kext held (`RGFX_BLOCK_RELEASE_UNHELD`), every one,
    /// as the kext counts them; of them, those of a written cache, dropped uncopied (its pages
    /// named or not), and their bytes; and those the header said a process took the surface
    /// again, copied.
    pub(crate) unheld_released: u64,
    pub(crate) unheld: u64,
    pub(crate) unheld_bytes: u64,
    pub(crate) retaken: u64,
    /// Downloads and releases of a written cache not copied, as the CPU's generation of the
    /// surface moved past the cache's ([`SurfaceCaches::behind_the_cpu`]).
    pub(crate) behind: u64,
    /// Caches made over their pages' runs, with no view.
    pub(crate) by_runs: u64,
    /// Copies by runs the platform refused, each of which made its cache's view.
    pub(crate) runs_refused: u64,
    /// Deferred blocks' caches made, and refused (past the budget, or not made); of those asked
    /// with their surface's layout, those refused as their textures might lie in place over the
    /// pages they come to have; deferred blocks whose pages were named; and those released with
    /// their pages never named.
    pub(crate) deferred: u64,
    pub(crate) deferred_refused: u64,
    pub(crate) deferred_in_place: u64,
    pub(crate) attached: u64,
    /// Of them, pages the host made no view of, or no buffer over, copied instead.
    pub(crate) attached_copied: u64,
    /// Deferred blocks' caches made carrying pages ([`SurfaceCache::carried`]).
    pub(crate) carried: u64,
    pub(crate) unattached: u64,
    /// Early copies of caches into their pages, behind a writer of a surface that reads back
    /// ([`SurfaceCaches::early_copies`]), and their bytes.
    pub(crate) early: u64,
    pub(crate) early_bytes: u64,
    /// Of them, copies behind a writer that held no video encoder, on the host's read-back queue.
    pub(crate) early_aside: u64,
    /// Downloads that an early copy answered, and early copies that no download followed before
    /// the next write, each of which ended its block's reading back.
    pub(crate) early_read: u64,
    pub(crate) early_unread: u64,
    /// Fills asked while command buffers they follow were begun and not finished, each made in
    /// the finish of the last of them and waited for on the GPU by the uses counted after it
    /// ([`SurfaceCaches::defer_fill`]).
    pub(crate) fills_deferred: u64,
    /// Waits on the CPU, of a command buffer's begin or a decode of the host's, for other command
    /// buffers' uses of a cache: each where the platform made no event to wait on the GPU with
    /// ([`Caches::before_answering`]), or a decode's for a deferred fill ([`Caches::wait_filled`]).
    pub(crate) cpu_waits: u64,
    /// Command buffers that use a cache and ask for their `CB_SUBMITTED`, looked at on a
    /// doorbell's thread with the caches' lock free ([`Caches::waits_for_nothing`]): those that
    /// wait for no copy of their caches, which the doorbell may answer, and those that wait for
    /// one, left to their queue's thread.
    pub(crate) doorbell_free: u64,
    pub(crate) doorbell_waits: u64,
    /// Caches made exportable for the host's video ([`CacheMaker::make`]), and those the
    /// platform made none for, made as any other buffer.
    pub(crate) exportable: u64,
    pub(crate) unexportable: u64,
    /// Caches made with their memory exported as a dma-buf, for the host's video to decode into
    /// ([`CacheMaker::make`]), and those the platform made none for, made as before.
    pub(crate) dma_buf: u64,
    pub(crate) no_dma_buf: u64,
    /// Caches made with their surface's first plane's byte placed for textures in place
    /// ([`CacheMaker::make`], [`aligned_byte`]).
    pub(crate) aligned: u64,
}

/// One cache's pages, host addresses in the guest's RAM, and how many times a device of the host
/// wrote over them ([`device_writes`]): a fill reads the writes counted when it is made
/// ([`SurfaceCache::host_read`]).
struct Pages {
    runs: Vec<Range<u64>>,
    writes: u64,
}

/// Each cache's [`Pages`], by block. Its lock is a leaf.
#[derive(Default)]
struct Watched(Mutex<HashMap<u64, Pages>>);

impl Watched {
    /// The writes of the host's devices over the block's pages so far.
    fn writes(&self, block: u64) -> u64 {
        self.0.lock().get(&block).map_or(0, |p| p.writes)
    }

    /// [`Self::writes`], with the lock tried: `None` where it is busy.
    fn try_writes(&self, block: u64) -> Option<u64> {
        Some(self.0.try_lock()?.get(&block).map_or(0, |p| p.writes))
    }
}

impl Watcher for Watched {
    fn wrote(&self, runs: &[Range<u64>]) {
        for pages in self.0.lock().values_mut() {
            if device_writes::overlap(&pages.runs, runs) {
                pages.writes += 1;
            }
        }
    }
}

/// The device caches of the blocks the kext marked cached, by block offset.
pub(crate) struct SurfaceCaches<P: Platform> {
    /// Whether the transfer queue and the end events are made with the command queues
    /// ([`Caches::queue_made`]): on a host whose GPU has memory of its own, where the kext caches
    /// every surface it may; not on one whose GPU shares the CPU's memory, where a surface takes a
    /// cache only where its textures cannot lie in place over its pages, and the first copy
    /// makes them ([`Self::make_transfer`], [`Self::after`]), so a guest none of whose surfaces
    /// takes one costs no queue.
    early: bool,
    by_block: HashMap<u64, SurfaceCache<P>>,
    transfer: Option<Transfer<P>>,
    /// Each command queue's end event, by the queue's id, and the last value signalled: made with
    /// the queue ([`Caches::queue_made`]), or at its first buffer that uses a cache where it was
    /// not; let go of once the queue is gone and no cache names it any more
    /// ([`Self::queue_ended`]).
    ends: HashMap<u64, (P::Event, u64)>,
    /// The queues gone whose end event a cache still names, for a later copy of it to wait for:
    /// each let go of once none does ([`Self::sweep_gone`]).
    gone: HashSet<u64>,
    /// The caches' pages, watched for the host's devices' writes while this lives.
    watched: Arc<Watched>,
    counts: CacheCounts,
    /// By block: the command buffers begun and not finished that use its cache ([`Caches`]).
    pending: HashMap<u64, Users>,
    /// By block: the copies of its cache asked while command buffers they follow were begun and
    /// not finished, in the order they were asked, each made in the finish of the last of them
    /// ([`Self::run_deferred`]): a fill follows every use counted before it was asked, and a copy
    /// out (a download, a release) every one of them that may write the cache.
    deferred: HashMap<u64, VecDeque<Deferred<P>>>,
    /// The number the next use of a cache is counted under.
    next_user: u64,
    /// The command buffers committed and finished that wait on the GPU for a deferred copy not
    /// yet made, each still counted among its blocks' users, and among those that may write
    /// them, until the last of those copies is ([`Self::leave`]): a copy behind one, made first
    /// on the transfer queue, would wait there for its end, and so for a copy committed behind
    /// it on that queue, which would never run.
    parked: Vec<Parked>,
    /// The deferred copy being made ([`Self::make_deferred`]): its event and number, for the
    /// command buffer its copy commits, if any, to signal ([`Self::fill`], [`Self::write_back`]).
    signal_with: Option<(P::SharedEvent, u64)>,
    /// A deferred fill made nothing, and the presents waiting for one are to be told
    /// ([`Self::take_tell`]).
    tell: bool,
    /// What a fill tells as it finishes, the [`Caches`]' own.
    on_fill_done: Arc<OnceLock<FillHook>>,
    /// The bytes deferred blocks' caches may hold at once (the device's `DEFERRED_BUDGET`), and the
    /// bytes they hold.
    deferred_budget: u64,
    deferred_bytes: u64,
    /// The device offers `SURFACE_DEFERRED_ASK`: a deferred block asks with its surface's layout
    /// ([`Self::set_deferred_asks`]).
    deferred_asks: bool,
    /// Whether a deferral may carry its surface's resident pages, as the device offers
    /// `SURFACE_DEFERRED_CARRY` ([`Self::set_deferred_carries`]).
    deferred_carries: bool,
    /// The copies of caches into pages asked and not yet landed ([`Self::copies_into`]), those
    /// landed let go of as each is added.
    into_pages: Vec<IntoPages>,
    /// Whether caches of surfaces of two planes are made exportable, for the host's video to
    /// import ([`Caches::set_exportable`]).
    exportable: bool,
    /// The blocks whose caches owe their first fill ([`SurfaceCache::first_fill`]).
    owed: HashSet<u64>,
    /// The blocks of the begins answered before the fills they make ([`Caches::before_answering`]),
    /// from the answer until those fills are made: no other begin counts a use of one meanwhile.
    answering: HashSet<u64>,
    /// Whether laid-out caches of two planes are made with their memory a dma-buf, for the host's
    /// video to decode into ([`Caches::set_dma_buf`]).
    dma_buf: bool,
}

/// The command buffers begun and not finished that use one block's cache, by the number each was
/// counted under, oldest first.
#[derive(Default)]
struct Users {
    all: BTreeSet<u64>,
    /// Those that may write it.
    writers: BTreeSet<u64>,
}

/// A command buffer's count among a block's users ([`SurfaceCaches::hold`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Count {
    block: u64,
    /// The number it was counted under: the uses of every cache are numbered in the order they
    /// are counted.
    number: u64,
    /// It may write the cache.
    writes: bool,
}

/// A command buffer finished while a deferred copy it follows is not yet made
/// ([`SurfaceCaches::parked`]).
struct Parked {
    /// Its counts among the blocks' users.
    counts: Vec<Count>,
    /// The deferred copies it follows, each by block and number ([`SurfaceCache::asked`]).
    follows: Vec<(u64, u64)>,
}

/// A copy out of a block's cache, made on the caches ([`Caches::after_writers`]).
type CopyOut<P> = Box<dyn FnOnce(&mut SurfaceCaches<P>) + Send>;

/// A copy of a block's cache asked while command buffers it follows were begun and not finished
/// ([`SurfaceCaches::deferred`]), made in the finish of the last of them.
struct Deferred<P: Platform> {
    /// It follows every use counted below this number; where `writers`, only those that may
    /// write the cache.
    upto: u64,
    writers: bool,
    copy: Asked<P>,
    /// Its number among the cache's deferred copies ([`SurfaceCache::asked`]), which it signals
    /// the cache's event to once made, where `event` holds it.
    number: u64,
    event: Option<P::SharedEvent>,
    /// The device it is made on.
    device: P::Device,
    /// What the uses counted from `upto` on did to the cache while it waits, the cache's own once
    /// it is made.
    since: Since,
}

/// What a deferred copy does.
enum Asked<P: Platform> {
    /// A fill, as of CPU generation `generation`, which the uses counted after it read, and of
    /// the host's devices' writes over the pages counted when it was asked (`host`); `filled` is
    /// told what it made ([`DeferredFill`]).
    Fill {
        generation: u64,
        host: u64,
        filled: Arc<DeferredFill>,
    },
    /// A copy out of the cache.
    Out(CopyOut<P>),
}

/// The uses of a cache counted after a deferred copy was asked, while it waits: the queues whose
/// command buffers used it, each with the end value of the latest; whether one that may write it
/// was committed; and how many such were counted (the cache's [`ReadBack::writes`]).
#[derive(Default)]
struct Since {
    users: HashMap<u64, u64>,
    written: bool,
    writes: u32,
}

/// A fill made: its transfer queue's fills, and the value at which it is done.
type FillMade = (Arc<Fills>, u64);

/// A deferred fill ([`Asked::Fill`]) until it is made: the transfer value its fill signals, a
/// present's wait for it ([`FillWait`]) and a decode's ([`Caches::wait_filled`]).
pub(crate) struct DeferredFill {
    /// `None` until it is made; then its fill, `None` where it made none.
    made: Mutex<Option<Option<FillMade>>>,
    moved: Condvar,
}

impl DeferredFill {
    fn new() -> Arc<Self> {
        Arc::new(DeferredFill {
            made: Mutex::new(None),
            moved: Condvar::new(),
        })
    }

    /// It was made: `fill` is the wait for the fill it committed, `None` where it committed
    /// none.
    fn set(&self, fill: Option<FillMade>) {
        *self.made.lock() = Some(fill);
        self.moved.notify_all();
    }

    /// Whether it was made and its fill has finished.
    fn done(&self) -> bool {
        match &*self.made.lock() {
            None => false,
            Some(None) => true,
            Some(Some((fills, value))) => fills.reached(*value),
        }
    }

    /// Waits, with no lock of the caches' held, until it is made: the wait for its fill.
    fn wait_made(&self) -> Option<FillMade> {
        let mut made = self.made.lock();
        loop {
            if let Some(fill) = &*made {
                return fill.clone();
            }
            self.moved.wait(&mut made);
        }
    }
}

impl<P: Platform> SurfaceCaches<P> {
    /// No caches, each fill telling `on_fill_done` as it finishes.
    fn new(on_fill_done: Arc<OnceLock<FillHook>>) -> Self {
        let watched = Arc::new(Watched::default());
        let watcher: Arc<dyn Watcher> = Arc::clone(&watched) as Arc<dyn Watcher>;
        device_writes::watch(&watcher);
        SurfaceCaches {
            early: true,
            by_block: HashMap::new(),
            transfer: None,
            ends: HashMap::new(),
            gone: HashSet::new(),
            watched,
            counts: CacheCounts::default(),
            pending: HashMap::new(),
            deferred: HashMap::new(),
            next_user: 0,
            parked: Vec::new(),
            signal_with: None,
            tell: false,
            on_fill_done,
            deferred_budget: 0,
            deferred_bytes: 0,
            deferred_asks: false,
            deferred_carries: false,
            into_pages: Vec::new(),
            exportable: false,
            owed: HashSet::new(),
            answering: HashSet::new(),
            dma_buf: false,
        }
    }

    /// What the caches did so far.
    pub(crate) fn counts(&self) -> CacheCounts {
        self.counts
    }

    /// Whether the block at `block` keeps the cache it has as its pages are named again: one of
    /// `length` bytes over the same pages, `host_pages` ([`Caches::make`]).
    fn keeps(&self, block: u64, length: u64, host_pages: &[Range<u64>]) -> bool {
        (self.by_block.get(&block)).is_some_and(|cache| {
            cache.length == length
                && (self.watched.0.lock().get(&block)).is_some_and(|p| p.runs == host_pages)
        })
    }

    /// The cache the block at `block` keeps ([`Self::keeps`]), what a buffer wrote of it copied
    /// into its pages first, filled again from them as of `generation`
    /// ([`Self::fill_named_again`]), as the kext's generations of the surface may count anew.
    fn named_again(
        &mut self,
        device: &P::Device,
        block: u64,
        generation: u64,
    ) -> RaiseResult<bool> {
        if self.written_since_host(block) {
            self.write_back(device, block, Box::new(|| {}), None)?;
        }
        self.fill_named_again(device, block, generation)?;
        Ok(true)
    }

    /// [`Caches::make`] under the caches' lock, of a cache of `length` bytes, its surface laid
    /// out as `layout` where the kext's ask gave it: a block that keeps its cache
    /// ([`Self::keeps`]) has it filled again ([`Self::named_again`]) and takes nothing of `made`;
    /// any other cache it had goes as at a release ([`SurfaceCaches::released`]), and the cache
    /// `made` with the lock let go of ([`CacheMaker::make`]) is taken, its first fill owed as of
    /// `generation`. What is not taken stays in `made`, for the caller to let go of with the lock
    /// let go of.
    #[allow(clippy::too_many_arguments)] // make's, and what was made
    fn make_with(
        &mut self,
        device: &P::Device,
        block: u64,
        length: u64,
        host_pages: Vec<Range<u64>>,
        generation: u64,
        layout: Option<Vec<PagesPlane>>,
        made: &mut Option<RaiseResult<Made<P>>>,
    ) -> RaiseResult<bool> {
        if self.keeps(block, length, &host_pages) {
            return self.named_again(device, block, generation);
        }
        if self.by_block.contains_key(&block) {
            self.released(device, block, 0, None, None, Box::new(|| {}), None)?;
        }
        let Some(made) = made.take() else {
            return Ok(false);
        };
        let Made { cache, pages } = made?;
        let Some((cache, life)) = cache.take(&mut self.counts) else {
            return Ok(false);
        };
        let Some(pages) = pages else {
            return Ok(false);
        };
        let Some(filled_at) = self.zero(device, &cache, length)? else {
            return Ok(false);
        };
        if matches!(pages, CachePages::Runs { .. }) {
            self.counts.by_runs += 1;
        }
        self.by_block.insert(
            block,
            SurfaceCache {
                device: cache,
                pages: Some(pages),
                carried: None,
                length,
                planes: layout.as_ref().map_or(0, Vec::len),
                texels: layout.as_deref().map(texel_spans),
                generation: None,
                host_read: 0,
                written: false,
                filled_at,
                first_fill: Some(generation),
                owner: None,
                zeros: false,
                users: HashMap::new(),
                asked: 0,
                made: 0,
                event: None,
                deferred: None,
                read_back: ReadBack::default(),
                out: None,
                life,
            },
        );
        let pages = Pages {
            runs: host_pages,
            writes: 0,
        };
        self.watched.0.lock().insert(block, pages);
        self.owed.insert(block);
        self.counts.made += 1;
        Ok(true)
    }

    /// The first fill the block at `block`'s cache owes since its make
    /// ([`SurfaceCache::first_fill`]), made before what reads the cache next, as of the
    /// generation it was made at: at once where no command buffer begun and not finished uses the
    /// cache and no copy of it waits, else in the finish of the last of them
    /// ([`Self::defer_fill`]). Nothing where none is owed, or the block has no cache.
    fn fill_owed(&mut self, device: &P::Device, block: u64) -> RaiseResult<()> {
        let Some(generation) = (self.by_block.get(&block)).and_then(|c| c.first_fill) else {
            return Ok(());
        };
        self.fill_whole(device, block, generation)
    }

    /// The cache of the block at `block` filled from its pages as of CPU generation
    /// `generation`, its first fill or one as a new cache's, whatever it holds: at once where no
    /// command buffer begun and not finished uses it and no copy of it waits, else in the finish
    /// of the last of them ([`Self::defer_fill`]), as a use's fill is.
    fn fill_whole(&mut self, device: &P::Device, block: u64, generation: u64) -> RaiseResult<()> {
        if let Some(cache) = self.by_block.get_mut(&block) {
            cache.generation = None;
        }
        if self.pending(block, false) > 0 || self.deferring(block) {
            self.defer_fill(device, block, generation);
        } else {
            self.fill(device, block, generation)?;
        }
        Ok(())
    }

    /// A fill of the block at `block`'s cache asked ([`Self::fill`], [`Self::defer_fill`]): it
    /// copies the whole surface before anything later reads the cache, so no first fill is owed
    /// any more; one that was is counted made ([`CacheCounts::made_fills`]).
    fn owes_none(&mut self, block: u64) {
        if let Some(cache) = self.by_block.get_mut(&block) {
            if cache.first_fill.take().is_some() {
                self.counts.made_fills += 1;
            }
            cache.owner = None;
            cache.zeros = false;
        }
        self.owed.remove(&block);
    }

    /// The first fills owed of the caches whose first buffer context `context` made
    /// ([`SurfaceCache::owner`]), before a command buffer of its with render, compute or blit
    /// work, which may reach a texture over the surface with no use named.
    /// `cb` waits on the GPU for each fill; whether it waits for one.
    fn fill_reached(
        &mut self,
        device: &P::Device,
        cb: &P::CommandBuffer,
        context: Option<u32>,
    ) -> RaiseResult<bool> {
        if self.owed.is_empty() {
            return Ok(false);
        }
        let reached: Vec<u64> = (self.owed.iter())
            .copied()
            .filter(|block| (self.by_block.get(block)).is_some_and(|c| c.owner == Some(context)))
            .collect();
        let mut waits = false;
        for block in reached {
            self.fill_owed(device, block)?;
            waits |= self.wait_last_fill(cb, block)?;
        }
        Ok(waits)
    }

    /// A buffer over the block at `block` made in `context` (`BUFFER_OVER_SURFACE_PAGES`), its
    /// cache the buffer: the first such context, while the cache owes its first fill, is the
    /// cache's owner ([`SurfaceCache::owner`]); another context's buffer makes the fill at
    /// once, as that context may reach the cache in ways a decode of the owner's does not order.
    fn buffer_made(
        &mut self,
        device: &P::Device,
        block: u64,
        context: Option<u32>,
    ) -> RaiseResult<()> {
        let Some(cache) = self.by_block.get_mut(&block) else {
            return Ok(());
        };
        if cache.first_fill.is_none() {
            return Ok(());
        }
        match cache.owner {
            None => cache.owner = Some(context),
            Some(owner) if owner == context => {}
            Some(_) => return self.fill_owed(device, block),
        }
        Ok(())
    }

    /// The picture of a decode into the block at `block`'s surface answered (`picture`), or none
    /// (a picture the stream's random access skips): where a decode took the cache's first fill
    /// as met ([`SurfaceCache::zeros`]), a picture lands over the zeros, and none makes the fill,
    /// before what reads the cache next, as [`Self::fill_owed`] makes one, on `device`; where
    /// none is given (the platform's command buffer named none), it is logged and not made.
    fn decoded(
        &mut self,
        device: Option<&P::Device>,
        block: u64,
        picture: bool,
    ) -> RaiseResult<()> {
        let Some(cache) = self.by_block.get_mut(&block) else {
            return Ok(());
        };
        if !std::mem::take(&mut cache.zeros) || picture {
            return Ok(());
        }
        let Some(device) = device else {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the IOSurface block at {block:#x}: a decode into its new device cache answered \
                 no picture, and its command buffer named no device to fill the cache on: it \
                 holds zeros"
            );
            return Ok(());
        };
        // The first fill it took as met is owed again, and made now.
        let generation = cache.generation.unwrap_or(0);
        cache.first_fill = Some(generation);
        self.fill_whole(device, block, generation)
    }

    /// The cache of the block at `block`, named again over the same pages, filled from them as of
    /// CPU generation `generation`, as a new cache's first fill is ([`Self::fill_owed`]), after
    /// the copy of what a buffer wrote of it into them: a use the guest names no
    /// `RGFX_OP_SURFACE_USE` of reads the pages as they were then, not what the cache held before
    /// the naming. The kext's generations of the surface may count anew, so the cache holds none
    /// but this fill's, and a named use of a newer generation fills again. Filled as a use's fill
    /// is ([`Self::fill_whole`]).
    fn fill_named_again(
        &mut self,
        device: &P::Device,
        block: u64,
        generation: u64,
    ) -> RaiseResult<()> {
        self.counts.made_fills += 1;
        self.fill_whole(device, block, generation)
    }

    /// A cache of zeros for the deferred block at `block` (the module doc), of `length` bytes,
    /// at the CPU's generation 0, its block `block_size` bytes, its surface laid out as `layout`
    /// where its ask gave it (a device that shares the CPU's memory: [`Caches::make`]'s caches
    /// of a layout): the cache `made` with the caches' lock let go of ([`Caches::make_deferred`]),
    /// zeroed on the transfer queue, whose fill every use of it waits for. Where it `carried`
    /// pages (their host addresses in the guest's RAM `host_pages`), it owes a first fill of them
    /// at generation 0, as a new cache owes one of its pages ([`Caches::make`]), which the
    /// zeroing comes before on the same queue, and their pages are watched until the block's are
    /// named. `Ok(false)`, with no cache, where the deferred caches' budget has no room for it
    /// now (what was made then stays in `made`, for the caller to let go of with the lock let go
    /// of), or the platform made no buffer or no transfer queue for it. A cache the block had
    /// goes first, as at a release ([`SurfaceCaches::released`]).
    #[allow(clippy::too_many_arguments)] // the deferral's, its carried pages, and what was made
    fn make_deferred(
        &mut self,
        device: &P::Device,
        block: u64,
        length: u64,
        block_size: u64,
        layout: Option<Vec<PagesPlane>>,
        carried: Option<CarriedPages>,
        host_pages: Vec<Range<u64>>,
        made: &mut Option<RaiseResult<NewCache<P>>>,
    ) -> RaiseResult<bool> {
        if self.by_block.contains_key(&block) {
            self.released(device, block, 0, None, None, Box::new(|| {}), None)?;
        }
        let room = self.deferred_budget.saturating_sub(self.deferred_bytes);
        let made = match made.take_if(|_| length <= room) {
            Some(new) => new?.take(&mut self.counts),
            None => None,
        };
        let zeroed = match &made {
            Some((cache, _)) => self.zero(device, cache, length)?,
            None => None,
        };
        let (Some((cache, life)), Some(filled_at)) = (made, zeroed) else {
            self.counts.deferred_refused += 1;
            return Ok(false);
        };
        let carries = carried.is_some();
        self.by_block.insert(
            block,
            SurfaceCache {
                device: cache,
                pages: None,
                carried: carried.map(Arc::new),
                length,
                planes: layout.as_ref().map_or(0, Vec::len),
                texels: layout.as_deref().map(texel_spans),
                // A carrying block holds no generation until its first fill, as a new cache.
                generation: (!carries).then_some(0),
                host_read: 0,
                written: false,
                filled_at,
                first_fill: carries.then_some(0),
                owner: None,
                zeros: false,
                users: HashMap::new(),
                asked: 0,
                made: 0,
                event: None,
                deferred: Some(block_size),
                read_back: ReadBack::default(),
                out: None,
                life,
            },
        );
        if carries {
            let pages = Pages {
                runs: host_pages,
                writes: 0,
            };
            self.watched.0.lock().insert(block, pages);
            self.owed.insert(block);
            self.counts.carried += 1;
        }
        self.deferred_bytes += length;
        self.counts.made += 1;
        self.counts.deferred += 1;
        Ok(true)
    }

    /// Fills `length` bytes of `cache` with zeros on the transfer queue: the transfer event's
    /// value at which it is done, `None` where the platform made no queue, command buffer or
    /// encoder for it.
    fn zero(
        &mut self,
        device: &P::Device,
        cache: &P::Buffer,
        length: u64,
    ) -> RaiseResult<Option<u64>> {
        if !self.make_transfer(device)? {
            return Ok(None);
        }
        let Some(transfer) = self.transfer.as_mut() else {
            return Ok(None);
        };
        let Some(cb) = transfer.queue.command_buffer(ON)? else {
            return Ok(None);
        };
        let Some(blit) = cb.blit_encoder(ON)? else {
            return Ok(None);
        };
        let range = NsRange::new(0, length);
        blit.encode(
            ON,
            BlitCmd::FillBuffer {
                buffer: cache,
                range,
                value: 0,
            },
        )?;
        blit.encode(ON, BlitCmd::EndEncoding)?;
        transfer.signalled += 1;
        let value = transfer.signalled;
        cb.encode_signal(Some(EventRef::Event(&transfer.event)), value)?;
        let fills = Arc::clone(&transfer.fills);
        // Told as the platform sees the fill done, with no handler thread woken between.
        cb.add_completion_seen_handler(
            ON,
            Box::new(move |_: CompletedInfo| fills.finished(value)),
        )?;
        cb.commit(ON)?;
        Ok(Some(value))
    }

    /// The pages of the deferred block at `block`, `named` by a view or by their runs, whose host
    /// addresses in the guest's RAM are `host_pages` ([`Caches::make`]), taken for its cache, which
    /// keeps its bytes, its generation and whether a buffer wrote it: `Some(true)`; `Some(false)`
    /// where they are fewer than the cache's bytes or the platform made no buffer over the view,
    /// the block keeping none; `None` where the block has no deferred cache whose pages are not
    /// named.
    pub(crate) fn attach(
        &mut self,
        device: &P::Device,
        block: u64,
        named: Named<P>,
        host_pages: Vec<Range<u64>>,
    ) -> RaiseResult<Option<bool>> {
        let Some(length) = self.unattached(block) else {
            return Ok(None);
        };
        let ends = match named {
            Named::View(view) if view.length() >= length => {
                match buffer_over(device, &view, 0, length, 0, None)? {
                    Some(pages) => CachePages::View(pages),
                    None => return Ok(Some(false)),
                }
            }
            Named::Runs(runs, unmade) if unmade.length() >= length => {
                CachePages::Runs { runs, unmade }
            }
            Named::View(_) | Named::Runs(..) => return Ok(Some(false)),
        };
        if let Some(cache) = self.by_block.get_mut(&block) {
            cache.pages = Some(ends);
            cache.carried = None;
        }
        let pages = Pages {
            runs: host_pages,
            writes: 0,
        };
        self.watched.0.lock().insert(block, pages);
        self.counts.attached += 1;
        Ok(Some(true))
    }

    /// The pages of the deferred block at `block`, `copied` from the guest's RAM, whose host
    /// addresses are `host_pages` ([`Caches::make`]), taken for its cache where the host made no
    /// view of them or no buffer over it (`CachePages::Copied`), as [`Self::attach`] takes them:
    /// `Some(true)`; `Some(false)` where they hold fewer bytes than the cache; `None` where the
    /// block has no deferred cache whose pages are not named.
    pub(crate) fn attach_copied(
        &mut self,
        block: u64,
        copied: &[(HostPtr, u64)],
        host_pages: Vec<Range<u64>>,
    ) -> Option<bool> {
        let length = self.unattached(block)?;
        let Some(copied) = CopiedPages::new(copied, length) else {
            return Some(false);
        };
        if let Some(cache) = self.by_block.get_mut(&block) {
            cache.pages = Some(CachePages::Copied(Arc::new(copied)));
            cache.carried = None;
        }
        let pages = Pages {
            runs: host_pages,
            writes: 0,
        };
        self.watched.0.lock().insert(block, pages);
        self.counts.attached += 1;
        self.counts.attached_copied += 1;
        Some(true)
    }

    /// The bytes of the deferred block at `block`'s cache, where its pages are not named yet.
    pub(crate) fn unattached(&self, block: u64) -> Option<u64> {
        (self.by_block.get(&block))
            .filter(|c| c.deferred.is_some() && c.pages.is_none())
            .map(|c| c.length)
    }

    /// The cache of the deferred block `m` names, which has no view while its pages are not
    /// named, nor where they are copied, where the block's size holds the `m.block_size` bytes `m`
    /// names and the cache at least `m.length` bytes, as a view's are checked
    /// ([`super::surface_pages::SurfaceViews::view_for`]).
    pub(crate) fn deferred_buffer(&self, m: &RgfxBufferOverSurfacePages) -> Option<&P::Buffer> {
        (self.by_block.get(&m.block_offset))
            .filter(|c| {
                !matches!(c.pages, Some(CachePages::View(_)))
                    && c.deferred.is_some_and(|size| m.block_size <= size)
                    && m.length != 0
                    && m.length <= c.length
            })
            .map(|c| &c.device)
    }

    /// Sets the bytes deferred blocks' caches may hold at once.
    pub(crate) fn set_deferred_budget(&mut self, budget: u64) {
        self.deferred_budget = budget;
    }

    /// Has a deferred block ask with its surface's layout from now on, as the device offers
    /// `SURFACE_DEFERRED_ASK` (a device whose GPU shares the CPU's memory).
    pub(crate) fn set_deferred_asks(&mut self) {
        self.deferred_asks = true;
    }

    /// Has a deferral carry its surface's resident pages where it has some from now on, as the
    /// device offers `SURFACE_DEFERRED_CARRY`.
    pub(crate) fn set_deferred_carries(&mut self) {
        self.deferred_carries = true;
    }

    /// Whether a deferral may carry pages ([`Self::set_deferred_carries`]).
    pub(crate) fn deferred_carries(&self) -> bool {
        self.deferred_carries
    }

    /// Whether a deferred block asks with its surface's layout ([`Self::set_deferred_asks`]).
    pub(crate) fn deferred_asks(&self) -> bool {
        self.deferred_asks
    }

    /// A deferred block's ask refused, its surface's textures possibly lying in place over the
    /// pages it comes to have.
    pub(crate) fn deferred_in_place(&mut self) {
        self.counts.deferred_in_place += 1;
    }

    /// The cache of the block at `block`, which its buffers are.
    pub(crate) fn buffer(&self, block: u64) -> Option<&P::Buffer> {
        self.by_block.get(&block).map(|c| &c.device)
    }

    /// The bytes of the cache of the block at `block`.
    #[cfg(test)]
    pub(crate) fn length(&self, block: u64) -> Option<u64> {
        self.by_block.get(&block).map(|c| c.length)
    }

    /// The identity of the cache of the block at `block`, which lives as long as the cache: the
    /// host's video keeps what it makes of the cache's memory by it ([`SurfaceCache::life`]).
    /// `None` where the block has no cache, or one whose memory was not made for the host's
    /// video.
    pub(crate) fn identity(&self, block: u64) -> Option<MemoryIdentity> {
        let life = self.by_block.get(&block)?.life.as_ref()?;
        Some(life.identity())
    }

    /// How a new cache's buffer is made as the caches are set now ([`CacheMaker::make`]), with
    /// their lock let go of.
    fn maker(&self) -> CacheMaker {
        CacheMaker {
            exportable: self.exportable,
            dma_buf: self.dma_buf,
        }
    }

    /// The room the deferred caches' budget has for the deferred block at `block`'s cache, the
    /// cache the block has going first ([`Self::make_deferred`]).
    fn deferred_room(&self, block: u64) -> u64 {
        let going = (self.by_block.get(&block))
            .filter(|cache| cache.deferred.is_some())
            .map_or(0, |cache| cache.length);
        let held = self.deferred_bytes.saturating_sub(going);
        self.deferred_budget.saturating_sub(held)
    }

    /// Whether the block at `block` has a cache.
    pub(crate) fn has(&self, block: u64) -> bool {
        self.by_block.contains_key(&block)
    }

    /// The wait for the block at `block`'s last fill: a deferred one where one was asked, else
    /// the transfer event's value at which the last made is done; `None` where it has no cache
    /// or no fill.
    fn last_fill(&self, block: u64) -> Option<FillWait> {
        if let Some(filled) = self.last_deferred_fill(block) {
            return Some(FillWait::Deferred(filled));
        }
        let (cache, t) = (self.by_block.get(&block)?, self.transfer.as_ref()?);
        (cache.filled_at != 0).then(|| FillWait::At {
            fills: Arc::clone(&t.fills),
            value: cache.filled_at,
        })
    }

    /// The last fill of the block at `block`'s cache asked and not yet made.
    fn last_deferred_fill(&self, block: u64) -> Option<Arc<DeferredFill>> {
        (self.deferred.get(&block)?.iter().rev()).find_map(|d| match &d.copy {
            Asked::Fill { filled, .. } => Some(Arc::clone(filled)),
            Asked::Out(_) => None,
        })
    }

    /// What the block at `block`'s cache holds once the fills asked of it are made: the CPU's
    /// generation of the surface, and the host's devices' writes over its pages counted when the
    /// last was asked ([`Watched::writes`]); `None` where it has no cache.
    fn will_hold(&self, block: u64) -> Option<(Option<u64>, u64)> {
        let cache = self.by_block.get(&block)?;
        let asked = (self.deferred.get(&block)).and_then(|q| {
            q.iter().rev().find_map(|d| match &d.copy {
                Asked::Fill {
                    generation, host, ..
                } => Some((Some(*generation), *host)),
                Asked::Out(_) => None,
            })
        });
        Some(asked.unwrap_or((cache.generation, cache.host_read)))
    }

    /// Whether a use or a present of the block at `block` at CPU generation `generation` needs
    /// a fill of its cache first: the cache holds, or will hold once the fills asked of it are
    /// made ([`Self::will_hold`]), an older generation than that, or none, or a device of the
    /// host wrote its pages since. False for a block with no cache.
    fn needs_fill_at(&self, block: u64, generation: u64) -> bool {
        self.will_hold(block).is_some_and(|(held, host)| {
            held.is_none_or(|held| generation > held) || self.watched.writes(block) > host
        })
    }

    /// [`Self::needs_fill_at`] of a use's block and generation.
    fn needs_fill(&self, u: &RgfxOpSurfaceUse) -> bool {
        self.needs_fill_at(u.block_offset, u.cpu_gen)
    }

    /// The use `u` of a block with a cache, before command buffer `cb` runs: its count among the
    /// cache's users ([`Self::hold`]), `None` for a block with no cache; and what `cb` waits for
    /// of the cache ([`Self::follow`]). The cache is filled first where the use needs it
    /// ([`Self::needs_fill`]): at once where no command buffer begun and not finished uses it and
    /// no copy of it waits, else in the finish of the last command buffer begun before it that
    /// uses it ([`Self::defer_fill`]), the use waiting for it on the GPU.
    ///
    /// Where `decoded` gives the planes the buffer's decodes write whole of the block's surface
    /// ([`Reach::decoded`]), the first fill the cache owes is taken as met instead, where it is of
    /// the use's generation or a newer one and no device of the host wrote the pages since the
    /// make ([`Self::take_decoded`]): the use reads nothing of the cache, which the picture
    /// lands over.
    fn before_one(
        &mut self,
        device: &P::Device,
        cb: &P::CommandBuffer,
        u: &RgfxOpSurfaceUse,
        decoded: Option<&[PagesPlane; 2]>,
    ) -> (Option<Count>, RaiseResult<(bool, Option<u64>)>) {
        let block = u.block_offset;
        if let Some(planes) = decoded {
            self.take_decoded(block, u.cpu_gen, planes);
        }
        let Some((held, host)) = self.will_hold(block) else {
            return (None, Ok((false, None)));
        };
        let older = held.is_none_or(|held| u.cpu_gen > held);
        let host_wrote = self.watched.writes(block) > host;
        let mut filled = Ok(());
        if older || host_wrote {
            if !older {
                self.counts.host_writes += 1;
            }
            if self.pending(block, false) > 0 || self.deferring(block) {
                self.defer_fill(device, block, u.cpu_gen);
            } else {
                filled = self.fill(device, block, u.cpu_gen).map(|_| ());
            }
        }
        let count = self.hold(block, u.flags & RGFX_SURFACE_USE_WRITES != 0);
        let follows = filled.and_then(|()| self.follow(cb, count));
        (Some(count), follows)
    }

    /// The first fill the block at `block`'s cache owes taken as met by a decode that writes the
    /// whole of `planes`, the surface's two (the module doc, "Fill"), named by a use of CPU
    /// generation `generation`: where the fill owed is of that generation or a newer one, and no
    /// device of the host wrote the pages since the make. The cache holds that generation, its
    /// zeros until the picture lands ([`SurfaceCache::zeros`]), and copies into the pages only
    /// the planes' rows where the kext's ask gave no layout ([`SurfaceCache::texels`]).
    fn take_decoded(&mut self, block: u64, generation: u64, planes: &[PagesPlane; 2]) {
        let host = self.watched.writes(block);
        let Some(cache) = self.by_block.get_mut(&block) else {
            return;
        };
        let Some(owed) = cache.first_fill.filter(|&owed| generation <= owed) else {
            return;
        };
        if host > cache.host_read {
            return;
        }
        cache.generation = Some(owed);
        cache.first_fill = None;
        cache.owner = None;
        cache.zeros = true;
        if cache.texels.is_none() {
            cache.texels = Some(texel_spans(planes));
            cache.planes = planes.len();
        }
        self.owed.remove(&block);
        self.counts.decode_firsts += 1;
    }

    /// The use `count` of the block's cache, just counted, before command buffer `cb` runs: a use
    /// that may write the cache marks it written, for the copy out that follows (the cache's own
    /// mark, or the deferred copy's before it, [`Since`]), and waits on the GPU for an early copy
    /// still reading the cache. `cb` waits on the GPU for the cache's last fill not yet done, and
    /// for the last deferred copy of the cache the use follows: the last fill, which it reads,
    /// and, where it may write the cache, the last copy out too, which must not copy its write.
    /// It waits for that copy on the cache's event, which [`Caches::before_answering`] gave the cache
    /// before counting the use, or, where the platform made none, waited for the copy instead.
    /// Whether `cb` waits on the GPU for a copy of the cache; and the number of the deferred copy
    /// it follows ([`SurfaceCache::asked`]).
    fn follow(&mut self, cb: &P::CommandBuffer, count: Count) -> RaiseResult<(bool, Option<u64>)> {
        let block = count.block;
        if count.writes {
            match self.since_mut(block, count.number) {
                Some(since) => {
                    since.written = true;
                    since.writes = since.writes.saturating_add(1);
                }
                None => {
                    if let Some(cache) = self.by_block.get_mut(&block) {
                        cache.written = true;
                        cache.read_back.writes = cache.read_back.writes.saturating_add(1);
                    }
                }
            }
            if let Some(cache) = self.by_block.get_mut(&block) {
                let read_back = &mut cache.read_back;
                // An early copy that no download followed: the surface no longer reads back.
                if std::mem::take(&mut read_back.unread) {
                    read_back.on = false;
                    self.counts.early_unread += 1;
                }
                // An early copy still reading the cache: the write waits for it on the GPU, as for
                // a buffer of another queue (a decode, whose picture lands off the buffer, waits
                // for it on the CPU, [`Caches::wait_copied_out`]).
                if let Some(copy) = &read_back.copy
                    && !copy.landing.landed()
                    && let Some((event, _)) = self.ends.get(&copy.queue)
                {
                    cb.encode_wait(Some(EventRef::Event(event)), copy.value)?;
                }
                // So does a copy into the pages on the transfer queue, asked before the write:
                // the pages take the cache as it was then.
                if let Some(out) = &cache.out
                    && !out.landing.landed()
                    && let Some(t) = &self.transfer
                {
                    cb.encode_wait(Some(EventRef::Event(&t.event)), out.value)?;
                }
            }
        }
        let follows = (self.deferred.get(&block))
            .and_then(|q| match count.writes {
                true => q.back(),
                false => q
                    .iter()
                    .rev()
                    .find(|d| matches!(d.copy, Asked::Fill { .. })),
            })
            .map(|d| d.number);
        let mut waits = false;
        if let Some(number) = follows
            && let Some(event) = (self.by_block.get(&block)).and_then(|c| c.event.as_ref())
        {
            cb.encode_wait(Some(EventRef::Shared(event)), number)?;
            waits = true;
        }
        waits |= self.wait_last_fill(cb, block)?;
        Ok((waits, follows))
    }

    /// Has `cb` wait on the GPU for the block at `block`'s last fill made, where it is not yet
    /// done; whether it waits.
    fn wait_last_fill(&self, cb: &P::CommandBuffer, block: u64) -> RaiseResult<bool> {
        if let (Some(t), Some(cache)) = (&self.transfer, self.by_block.get(&block))
            && !t.fills.reached(cache.filled_at)
        {
            cb.encode_wait(Some(EventRef::Event(&t.event)), cache.filled_at)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Whether the use `u` follows a copy of its block's cache that waits for command buffers
    /// begun before it ([`Self::follow`]): a fill it needs that must wait so, or a deferred copy
    /// asked before it.
    fn follows_deferred(&self, u: &RgfxOpSurfaceUse) -> bool {
        let block = u.block_offset;
        let writes = u.flags & RGFX_SURFACE_USE_WRITES != 0;
        let asked = (self.deferred.get(&block))
            .is_some_and(|q| writes || q.iter().any(|d| matches!(d.copy, Asked::Fill { .. })));
        asked || (self.needs_fill(u) && (self.pending(block, false) > 0 || self.deferring(block)))
    }

    /// Whether the use `u` waits on the GPU for no copy of its block's cache where it is begun
    /// now ([`Self::before_one`]): the block has no cache, or its cache needs no fill for it,
    /// no copy of it asked and not made is one it follows, and its last fill is done. `None`
    /// where the lock of the pages' watch is busy: a doorbell's thread tries it
    /// ([`Caches::waits_for_nothing`]).
    fn waits_for_nothing(&self, u: &RgfxOpSurfaceUse) -> Option<bool> {
        let block = u.block_offset;
        let (Some(cache), Some((held, host))) = (self.by_block.get(&block), self.will_hold(block))
        else {
            return Some(true);
        };
        let host_writes = self.watched.try_writes(block)?;
        let fills = held.is_none_or(|held| u.cpu_gen > held) || host_writes > host;
        let writes = u.flags & RGFX_SURFACE_USE_WRITES != 0;
        let follows = (self.deferred.get(&block))
            .is_some_and(|q| writes || q.iter().any(|d| matches!(d.copy, Asked::Fill { .. })));
        let filled = (self.transfer.as_ref()).is_none_or(|t| t.fills.reached(cache.filled_at));
        Some(!fills && !follows && filled)
    }

    /// Whether the block at `block`'s cache has its event, which the command buffers that follow
    /// a deferred copy of it wait on ([`SurfaceCache::event`]).
    fn has_event(&self, block: u64) -> bool {
        (self.by_block.get(&block)).is_some_and(|c| c.event.is_some())
    }

    /// The block at `block`'s cache takes `event`, made with no lock held, where it has none: set
    /// on the CPU to the number of the last of its deferred copies made, whose signals no event
    /// took, and held by each deferred copy asked of it; `event` back where it has one, or no
    /// cache, for the caller to let go of with no lock held.
    fn give_event(&mut self, block: u64, event: P::SharedEvent) -> Option<P::SharedEvent> {
        let cache = self.by_block.get_mut(&block)?;
        if cache.event.is_some() {
            return Some(event);
        }
        if let Err(raise) = event.set_signaled_value(cache.made) {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = raise;
                "the IOSurface block at {block:#x}: the platform raised at the set of its device \
                 cache's new event ({raise:?}); its deferred copies are waited for on the CPU"
            );
            return Some(event);
        }
        for d in self.deferred.get_mut(&block).into_iter().flatten() {
            if d.event.is_none() {
                d.event = Some(event.clone_ref());
            }
        }
        cache.event = Some(event);
        None
    }

    /// The number of the last deferred copy of the block at `block`'s cache made; `u64::MAX` for
    /// a block with no cache, whose copies are all done with.
    fn made_through(&self, block: u64) -> u64 {
        self.by_block.get(&block).map_or(u64::MAX, |c| c.made)
    }

    /// How many command buffers begun and not finished use the block at `block`'s cache; where
    /// `writers`, only those that may write it.
    fn pending(&self, block: u64, writers: bool) -> usize {
        (self.pending.get(&block)).map_or(0, |u| match writers {
            true => u.writers.len(),
            false => u.all.len(),
        })
    }

    /// Whether a command buffer begun and not finished, counted below `upto`, uses the block at
    /// `block`'s cache; where `writers`, one that may write it.
    fn pending_below(&self, block: u64, upto: u64, writers: bool) -> bool {
        let oldest = (self.pending.get(&block)).and_then(|u| match writers {
            true => u.writers.first(),
            false => u.all.first(),
        });
        oldest.is_some_and(|&number| number < upto)
    }

    /// Counts a command buffer's use of the block at `block`'s cache among those begun and not
    /// finished, where it may write it as a writer too, under a number of its own: its count.
    fn hold(&mut self, block: u64, writes: bool) -> Count {
        let number = self.next_user;
        self.next_user += 1;
        let users = self.pending.entry(block).or_default();
        users.all.insert(number);
        if writes {
            users.writers.insert(number);
        }
        Count {
            block,
            number,
            writes,
        }
    }

    /// Whether a copy of the block at `block`'s cache waits for command buffers begun before it.
    fn deferring(&self, block: u64) -> bool {
        self.deferred.contains_key(&block)
    }

    /// What the uses of the block at `block`'s cache counted under `number` did to it while the
    /// deferred copy asked before it waits ([`Since`]): the last one asked at or below
    /// `number`'s; `None` where none was, and the use's are the cache's own.
    fn since_mut(&mut self, block: u64, number: u64) -> Option<&mut Since> {
        (self.deferred.get_mut(&block)?.iter_mut().rev())
            .find(|d| d.upto <= number)
            .map(|d| &mut d.since)
    }

    /// [`Self::since_mut`] is `Some`.
    fn behind_deferred(&self, block: u64, number: u64) -> bool {
        (self.deferred.get(&block)).is_some_and(|q| q.iter().any(|d| d.upto <= number))
    }

    /// Whether a download of the block at `block` asked now would commit its copy at once
    /// ([`Self::download`]): no command buffer begun and not finished may write its cache, no
    /// copy of it waits for one, and one wrote it since its last copy. A download that
    /// would then copy nothing all the same (the pages a device of the host's wrote, or the
    /// CPU's generation past the cache's) is counted as one that copies.
    fn copies_now(&self, block: u64) -> bool {
        if self.pending(block, true) > 0 || self.deferring(block) {
            return false;
        }
        self.by_block.get(&block).is_some_and(|cache| cache.written)
    }

    /// Runs `op` on the caches once every command buffer begun and not finished that may write
    /// the block at `block`'s cache has finished, and after the copies of it deferred before it:
    /// at once where none is, else in the finish of the last of them ([`Self::run_deferred`]).
    fn after_writers(&mut self, device: &P::Device, block: u64, op: CopyOut<P>) {
        if self.pending(block, true) == 0 && !self.deferring(block) {
            op(self);
            return;
        }
        self.defer(device, block, true, Asked::Out(op));
    }

    /// Asks a fill of the block at `block`'s cache, as of CPU generation `generation` or the
    /// newer one it will hold, of command buffers begun and not finished that use it, or behind
    /// a copy of it that waits for such: made in the finish of the last of them, after their
    /// whole use as under one lock, and waited for on the GPU by the uses counted after it. What
    /// it made, once it is.
    fn defer_fill(&mut self, device: &P::Device, block: u64, generation: u64) -> Arc<DeferredFill> {
        let held = self.will_hold(block).and_then(|(held, _)| held);
        let generation = held.map_or(generation, |held| held.max(generation));
        let filled = DeferredFill::new();
        let fill = Asked::Fill {
            generation,
            host: self.watched.writes(block),
            filled: Arc::clone(&filled),
        };
        self.defer(device, block, false, fill);
        self.owes_none(block);
        self.counts.fills_deferred += 1;
        filled
    }

    /// Asks `copy` of the block at `block`'s cache, made once every command buffer counted before
    /// now that uses it (`writers`: that may write it) has finished, after the copies asked before
    /// it ([`Self::run_deferred`]).
    fn defer(&mut self, device: &P::Device, block: u64, writers: bool, copy: Asked<P>) {
        let (number, event) = match self.by_block.get_mut(&block) {
            Some(cache) => {
                cache.asked += 1;
                (cache.asked, cache.event.as_ref().map(Handle::clone_ref))
            }
            None => (0, None),
        };
        (self.deferred.entry(block).or_default()).push_back(Deferred {
            upto: self.next_user,
            writers,
            copy,
            number,
            event,
            device: device.clone_ref(),
            since: Since::default(),
        });
    }

    /// Makes the copies of the block at `block`'s cache whose command buffers have all finished,
    /// in the order they were asked ([`Self::carry_out`]).
    fn run_deferred(&mut self, block: u64) {
        loop {
            let Some(front) = self.deferred.get(&block).and_then(VecDeque::front) else {
                return;
            };
            if self.pending_below(block, front.upto, front.writers) {
                return;
            }
            let Some(queue) = self.deferred.get_mut(&block) else {
                return;
            };
            let deferred = queue.pop_front();
            if queue.is_empty() {
                self.deferred.remove(&block);
            }
            if let Some(deferred) = deferred {
                self.carry_out(block, deferred);
            }
        }
    }

    /// Makes the deferred copy `deferred` of the block at `block`'s cache: its copy, which
    /// signals its event to its number in the command buffer it commits, where it has an event
    /// ([`Self::signal_with`]), or else in one of its own ([`Self::signal_alone`]); then what the
    /// uses counted after it did becomes the cache's own ([`Since`]). A fill the platform raises
    /// at is logged and left unmade, as are the uses that read it then.
    fn carry_out(&mut self, block: u64, deferred: Deferred<P>) {
        let Deferred {
            copy,
            number,
            event,
            device,
            since,
            ..
        } = deferred;
        self.signal_with = event.as_ref().map(|e| (e.clone_ref(), number));
        match copy {
            Asked::Out(op) => op(self),
            Asked::Fill {
                generation, filled, ..
            } => {
                let made = match refusing(|| self.fill(&device, block, generation)) {
                    Ok(made) => made,
                    Err(raise) => {
                        log::warn!(
                            target: "reims_vgpu::graphics", kind:% = raise;
                            "the IOSurface block at {block:#x}: the platform raised at a fill of \
                             its device cache made behind the command buffers it waited for \
                             ({raise:?}); it is not made"
                        );
                        None
                    }
                };
                let fills = self.transfer.as_ref().map(|t| Arc::clone(&t.fills));
                filled.set(made.zip(fills).map(|(value, fills)| (fills, value)));
                // A present waiting for it is told as for a fill that finished.
                self.tell |= made.is_none();
            }
        }
        if let Some(cache) = self.by_block.get_mut(&block) {
            cache.made = cache.made.max(number);
            for (queue, value) in since.users {
                let latest = cache.users.entry(queue).or_insert(value);
                *latest = (*latest).max(value);
            }
            cache.written |= since.written;
            let read_back = &mut cache.read_back;
            read_back.writes = read_back.writes.saturating_add(since.writes);
        }
        if let Some((event, number)) = self.signal_with.take() {
            self.signal_alone(&device, block, &event, number);
        }
    }

    /// The deferred copy numbered `number` of the block at `block`'s cache committed no command
    /// buffer that signals its event (it copied nothing): a command buffer of the transfer queue
    /// signals it alone, after the copy before it and after a copy of the cache into its pages
    /// still in flight (an early copy, or one on the transfer queue), which a use that may write
    /// the cache must not write under. Where the platform makes no such command buffer, the
    /// event is set on the CPU, logged.
    fn signal_alone(
        &mut self,
        device: &P::Device,
        block: u64,
        event: &P::SharedEvent,
        number: u64,
    ) {
        let signalled = refusing(|| {
            if !self.make_transfer(device)? {
                return Ok(false);
            }
            let Some(transfer) = self.transfer.as_ref() else {
                return Ok(false);
            };
            let Some(cb) = transfer.queue.command_buffer(ON)? else {
                return Ok(false);
            };
            if number > 1 {
                cb.encode_wait(Some(EventRef::Shared(event)), number - 1)?;
            }
            let early = (self.by_block.get(&block))
                .and_then(|c| c.read_back.copy.as_ref())
                .filter(|copy| !copy.landing.landed());
            if let Some(copy) = early
                && let Some((end, _)) = self.ends.get(&copy.queue)
            {
                cb.encode_wait(Some(EventRef::Event(end)), copy.value)?;
            }
            if let Some(out) = (self.by_block.get(&block)).and_then(|c| c.out.as_ref())
                && !out.landing.landed()
            {
                cb.encode_wait(Some(EventRef::Event(&transfer.event)), out.value)?;
            }
            cb.encode_signal(Some(EventRef::Shared(event)), number)?;
            cb.commit(ON)?;
            Ok(true)
        });
        if !matches!(signalled, Ok(true)) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the IOSurface block at {block:#x}: no command buffer signalled its device cache's \
                 deferred copy {number} ({signalled:?}); its event is set on the CPU"
            );
            if let Err(raise) = event.set_signaled_value(number) {
                log::warn!(
                    target: "reims_vgpu::graphics", kind:% = raise;
                    "the IOSurface block at {block:#x}: the platform raised at the set of its \
                     device cache's event ({raise:?})"
                );
            }
        }
    }

    /// Whether a deferred fill made nothing since this was last asked: the presents waiting for
    /// one are told, as for a fill that finished, by the caller with the caches' lock let go.
    fn take_tell(&mut self) -> bool {
        std::mem::take(&mut self.tell)
    }

    /// The command buffer whose counts are `counts`, committed, is finished: it leaves them where
    /// every deferred copy it follows (`follows`, each by block and number) is made
    /// ([`Self::unhold`]); else it stays among their users, and among those that may write them,
    /// until the last of those copies is ([`Self::parked`]).
    fn leave(&mut self, counts: &[Count], follows: &[(u64, u64)]) {
        if self.all_made(follows) {
            self.unhold(counts);
            return;
        }
        for c in counts {
            if let Some(users) = self.pending.get_mut(&c.block) {
                users.writers.insert(c.number);
            }
        }
        self.parked.push(Parked {
            counts: counts.to_vec(),
            follows: follows.to_vec(),
        });
    }

    /// Whether each deferred copy of `follows`, by block and number, is made.
    fn all_made(&self, follows: &[(u64, u64)]) -> bool {
        (follows.iter()).all(|&(block, number)| self.made_through(block) >= number)
    }

    /// The command buffer whose counts are `counts` leaves them, and the copies of their caches
    /// that waited for it alone are made ([`Self::run_deferred`]); then so does each parked
    /// buffer whose copies are now all made ([`Self::parked`]), in turn.
    fn unhold(&mut self, counts: &[Count]) {
        self.let_go_of(counts);
        while let Some(at) = (self.parked.iter()).position(|p| self.all_made(&p.follows)) {
            let parked = self.parked.swap_remove(at);
            self.let_go_of(&parked.counts);
        }
    }

    /// `counts` leave their blocks' users ([`Self::forget`]), and the copies of their caches that
    /// waited for them alone are made ([`Self::run_deferred`]).
    fn let_go_of(&mut self, counts: &[Count]) {
        self.forget(counts);
        for c in counts {
            self.run_deferred(c.block);
        }
    }

    /// `counts` leave their blocks' users.
    fn forget(&mut self, counts: &[Count]) {
        for c in counts {
            if let Some(users) = self.pending.get_mut(&c.block) {
                users.all.remove(&c.number);
                users.writers.remove(&c.number);
                if users.all.is_empty() {
                    self.pending.remove(&c.block);
                }
            }
        }
    }

    /// Ends command buffer `cb` of queue `queue`, whose counts among the caches' users are
    /// `counts`: it signals the queue's end event, which a later copy of each cache waits for,
    /// the cache's next or, where the buffer was counted after a deferred copy of it was asked,
    /// the next after that copy ([`Since`]).
    pub(crate) fn after(
        &mut self,
        device: &P::Device,
        queue: u64,
        cb: &P::CommandBuffer,
        counts: &[Count],
    ) -> RaiseResult<()> {
        if counts.is_empty() {
            return Ok(());
        }
        let value = {
            let (event, value) = match self.ends.entry(queue) {
                Entry::Occupied(end) => end.into_mut(),
                Entry::Vacant(end) => match device.new_event(ON)? {
                    Some(event) => end.insert((event, 0)),
                    None => return Ok(()),
                },
            };
            *value += 1;
            cb.encode_signal(Some(EventRef::Event(event)), *value)?;
            *value
        };
        for c in counts {
            match self.since_mut(c.block, c.number) {
                Some(since) => {
                    since.users.insert(queue, value);
                }
                None => {
                    if let Some(cache) = self.by_block.get_mut(&c.block) {
                        cache.users.insert(queue, value);
                    }
                }
            }
        }
        Ok(())
    }

    /// The early copies behind command buffer `writer`, just committed on queue `queue`, whose
    /// counts among the caches' users are `counts` (the module doc): each cache it may write
    /// whose surface reads back ([`ReadBack`]), where its write is the only one since the cache's
    /// last copy and no device of the host wrote the pages since its last fill, is copied into
    /// its pages by a command buffer of the host's. Where the writer holds a video encoder that
    /// buffer is committed on `on`, the writer's own queue, right behind it, so the copy starts as
    /// the writer's work ends, waiting for nothing else, and signals that queue's end event next.
    /// Any other writer's is committed on the host's read-back queue ([`Transfer::read_back`]),
    /// waiting on the GPU for the writer's end value, so the writer's queue goes on behind the
    /// writer, and signals the read-back queue's end event ([`READ_BACK`]). Later copies of the
    /// cache wait for that value, and so does a later writer; the cache's write is the pages'
    /// once the copy lands ([`Landing`]).
    fn early_copies(
        &mut self,
        queue: u64,
        on: Option<&P::CommandQueue>,
        counts: &[Count],
    ) -> RaiseResult<()> {
        // The copies' queue, the end event they signal, and the writer's end it waits for there.
        let (copies_on, signals) = match on {
            Some(on) => (on, queue),
            None => match self.transfer.as_ref().and_then(|t| t.read_back.as_ref()) {
                Some(read_back) => (read_back, READ_BACK),
                None => return Ok(()),
            },
        };
        for c in counts {
            // A writer counted after a deferred copy of the cache was asked takes none: its
            // write is copied out by the download that follows.
            if !c.writes || self.behind_deferred(c.block, c.number) {
                continue;
            }
            let block = c.block;
            let (Some(cache), Some((event, value))) =
                (self.by_block.get(&block), self.ends.get(&signals))
            else {
                continue;
            };
            if self.watched.writes(block) > cache.host_read {
                continue;
            }
            // Off the writer's queue, the copy waits for the writer's end, which it signalled.
            let after = if signals == queue {
                None
            } else {
                match self.ends.get(&queue) {
                    Some((writer, end)) => Some((writer, *end)),
                    None => continue,
                }
            };
            let read_back = &cache.read_back;
            if !read_back.on || read_back.writes != 1 || !cache.written {
                continue;
            }
            // Pages in a buffer, or by their runs: a deferred block's not named yet have none,
            // and copied pages ([`CachePages::Copied`]) take no early copy.
            let view = match &cache.pages {
                Some(CachePages::View(pages)) => Some(pages),
                Some(CachePages::Runs { .. }) => None,
                Some(CachePages::Copied(_)) | None => continue,
            };
            let Some(cb) = copies_on.command_buffer(ON)? else {
                continue;
            };
            if let Some((writer, end)) = after {
                cb.encode_wait(Some(EventRef::Event(writer)), end)?;
            }
            let Some(blit) = cb.blit_encoder(ON)? else {
                continue;
            };
            // A copy by runs the platform refuses encodes nothing: the download that follows
            // copies, through the view it makes then (`copy`).
            if !encode_out::<P>(&blit, cache, view)? {
                blit.encode(ON, BlitCmd::EndEncoding)?;
                continue;
            }
            blit.encode(ON, BlitCmd::EndEncoding)?;
            let landing = Landing::new();
            // The answers waiting go as the platform sees the copy done, on its thread, with no
            // wake of a handler thread between; the buffers live until its completed handler,
            // past the cache's own end at a release.
            let lands = Arc::clone(&landing);
            cb.add_completion_seen_handler(
                ON,
                Box::new(move |info: CompletedInfo| {
                    lands.land_timed(info.gpu_times_ns, std::time::Duration::ZERO);
                }),
            )?;
            let held = (
                cache.device.clone_ref(),
                view.map(|pages| pages.clone_ref()),
            );
            cb.add_completed_handler(ON, Box::new(move |_: CompletedInfo| drop(held)))?;
            // The queue's next end value, the queue's once the buffer is committed.
            let next = value + 1;
            cb.encode_signal(Some(EventRef::Event(event)), next)?;
            cb.commit(ON)?;
            let length = cache.length;
            if let Some((_, value)) = self.ends.get_mut(&signals) {
                *value = next;
            }
            if let Some(cache) = self.by_block.get_mut(&block) {
                cache.written = false;
                cache.users.insert(signals, next);
                let read_back = &mut cache.read_back;
                read_back.writes = 0;
                read_back.unread = true;
                read_back.copy = Some(EarlyCopy {
                    queue: signals,
                    value: next,
                    landing,
                });
            }
            self.counts.early += 1;
            self.counts.early_bytes += length;
            if signals == READ_BACK {
                self.counts.early_aside += 1;
            }
        }
        Ok(())
    }

    /// Fills the cache of the block at `block` from the surface's pages, as of CPU generation
    /// `generation` or the newer one it holds, on the transfer queue, after every command buffer
    /// that used it before, reading the host's devices' writes over the pages so far; where a
    /// deferred copy is being made ([`Self::carry_out`]), the fill's command buffer signals its
    /// event. The transfer event's value at which it is done; `None` where it made none.
    fn fill(
        &mut self,
        device: &P::Device,
        block: u64,
        generation: u64,
    ) -> RaiseResult<Option<u64>> {
        if !self.make_transfer(device)? {
            return Ok(None);
        }
        let host = self.watched.writes(block);
        let signal =
            (self.signal_with.as_ref()).map(|(event, number)| (event.clone_ref(), *number));
        let (Some(transfer), Some(cache)) = (self.transfer.as_mut(), self.by_block.get_mut(&block))
        else {
            return Ok(None);
        };
        // A deferred block's pages, not named yet, hold nothing the cache does not but those it
        // carried: the CPU has not locked the surface for writing.
        if cache.pages.is_none() && cache.carried.is_none() {
            return Ok(None);
        }
        let Some(cb) = transfer.queue.command_buffer(ON)? else {
            return Ok(None);
        };
        wait_before::<P>(&cb, signal.as_ref())?;
        wait_for_users::<P>(&cb, &self.ends, cache)?;
        let staged = copy::<P>(device, &cb, cache, Way::In, block, &mut self.counts)?;
        if let Some(staged) = staged {
            cb.add_completed_handler(ON, Box::new(move |_: CompletedInfo| drop(staged)))?;
        }
        transfer.signalled += 1;
        let value = transfer.signalled;
        cb.encode_signal(Some(EventRef::Event(&transfer.event)), value)?;
        signal_after::<P>(&cb, signal.as_ref())?;
        let fills = Arc::clone(&transfer.fills);
        // Told as the platform sees the fill done, with no handler thread woken between: the
        // presents and decodes waiting for it go on then.
        cb.add_completion_seen_handler(
            ON,
            Box::new(move |_: CompletedInfo| fills.finished(value)),
        )?;
        cb.commit(ON)?;
        if signal.is_some() {
            self.signal_with = None;
        }
        let length = cache.length;
        if let Some(cache) = self.by_block.get_mut(&block) {
            cache.generation = Some(cache.generation.map_or(generation, |g| g.max(generation)));
            cache.host_read = host;
            cache.filled_at = value;
            cache.users.clear();
        }
        self.owes_none(block);
        self.counts.fills += 1;
        self.counts.filled_bytes += length;
        Ok(Some(value))
    }

    /// Fills the cache of the block at `block` where it holds, or will hold once the fills asked
    /// of it are made, an older generation than `generation`, or a device of the host wrote its
    /// pages since ([`Self::needs_fill_at`]): a present of the surface shows the pages as of that
    /// generation. Filled at once where no command buffer begun and not finished uses the cache
    /// and no copy of it waits, else in the finish of the last of them ([`Self::defer_fill`]).
    /// The wait for the cache's last fill, which the present makes in the present queue; `None`
    /// where nothing was filled.
    fn fill_now(
        &mut self,
        device: &P::Device,
        block: u64,
        generation: u64,
    ) -> RaiseResult<Option<FillWait>> {
        let Some((held, host)) = self.will_hold(block) else {
            return Ok(None);
        };
        let older = held.is_none_or(|held| generation > held);
        let host_wrote = self.watched.writes(block) > host;
        if !older && !host_wrote {
            return Ok(None);
        }
        if !older {
            self.counts.host_writes += 1;
        }
        if self.pending(block, false) > 0 || self.deferring(block) {
            let filled = self.defer_fill(device, block, generation);
            return Ok(Some(FillWait::Deferred(filled)));
        }
        self.fill(device, block, generation)?;
        Ok(self.last_fill(block))
    }

    /// Whether a `SURFACE_DOWNLOAD` of the block at `block` asked now is carried out at once with
    /// nothing to copy and nothing to wait for ([`Self::download`] after [`Self::after_writers`]):
    /// no command buffer begun and not finished may write its cache, no copy out of it waits for
    /// one, and it has no cache, or one no command buffer wrote since its last copy whose last
    /// early copy, if any, and last copy on the transfer queue, if any, have landed (their locks
    /// tried). Where so, the download's counts are kept as `download` keeps them; where not,
    /// nothing is changed, and a cache a command buffer wrote is left to `download`, whatever a
    /// device of the host wrote of its pages. Allocates and frees nothing.
    fn downloaded_now(&mut self, block: u64) -> bool {
        if self.pending(block, true) > 0 || self.deferring(block) {
            return false;
        }
        let Some(cache) = self.by_block.get_mut(&block) else {
            return true;
        };
        if cache.written
            || (cache.out.as_ref()).is_some_and(|out| out.landing.try_landed() != Some(true))
        {
            return false;
        }
        let read_back = &mut cache.read_back;
        match read_back.copy.as_ref() {
            Some(copy) => {
                if copy.landing.try_landed() != Some(true) {
                    return false;
                }
                if std::mem::take(&mut read_back.unread) {
                    self.counts.early_read += 1;
                }
            }
            None => self.counts.clean += 1,
        }
        true
    }

    /// `SURFACE_DOWNLOAD` of the block at `block`: the pages written from its cache, after every
    /// command buffer that used it, where one that may write it was committed since its last copy,
    /// no device of the host wrote the pages since its last fill, and the CPU's generation `cpu`
    /// of the surface, as the copy is made, is not past the cache's ([`Self::behind_the_cpu`]);
    /// `done` once the pages hold it, at once where nothing is to copy.
    pub(crate) fn download(
        &mut self,
        device: &P::Device,
        block: u64,
        cpu: Option<u64>,
        done: Box<dyn FnOnce() + Send>,
        trace: Option<&CopyTrace>,
    ) -> RaiseResult<()> {
        if let Some(trace) = trace {
            trace.cache(&self.by_block, block);
        }
        let written = self.written_since_host(block);
        if !written || self.behind_the_cpu(block, cpu, "SURFACE_DOWNLOAD") {
            // An early copy, landed or not, holds the cache's last write: answered as it lands.
            // So does a copy on the transfer queue not yet landed, which an earlier download whose
            // answer the kext stopped waiting for, or a block named again, asked.
            let early = (self.by_block.get_mut(&block)).and_then(|cache| {
                let read_back = &mut cache.read_back;
                let copy = read_back.copy.as_ref()?;
                if std::mem::take(&mut read_back.unread) {
                    self.counts.early_read += 1;
                }
                Some(Arc::clone(&copy.landing))
            });
            let landings: Vec<Arc<Landing>> =
                early.into_iter().chain(self.out_copy(block)).collect();
            if landings.is_empty() && !written && self.by_block.contains_key(&block) {
                self.counts.clean += 1;
            }
            if let Some(trace) = trace {
                trace.behind(&landings);
            }
            after_landings(landings, done);
            return Ok(());
        }
        // The CPU reads back a write that was the only one since the cache's last copy: the next
        // such writer is copied out as it ends ([`Self::early_copies`]).
        if let Some(cache) = self.by_block.get_mut(&block) {
            let read_back = &mut cache.read_back;
            read_back.on = read_back.writes == 1;
            read_back.unread = false;
        }
        if self.unattached(block).is_some() {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the IOSurface block at {block:#x}: a SURFACE_DOWNLOAD of a deferred block whose \
                 pages are not named; nothing is copied"
            );
        }
        self.write_back(device, block, done, trace)
    }

    /// The block at `block` went (its `BLOCK_RELEASE`, of `RGFX_BLOCK_RELEASE_*` `flags`): its
    /// cache goes once the pages hold what a buffer wrote of it, unless a device of the host wrote
    /// them since its last fill, or the CPU's generation `cpu` of the surface, as the copy is
    /// made, is past the cache's ([`Self::behind_the_cpu`]); `done` then, at once where nothing is
    /// to copy. A release whose surface had no client left (`RGFX_BLOCK_RELEASE_NO_CLIENT`) copies
    /// too: a surface with no client can still be looked up again, from a mach port or an XPC
    /// object that holds it (how one process hands a surface it drew to another), and the process
    /// that looks it up then reads the pages. One whose surface nothing but the kext held
    /// (`RGFX_BLOCK_RELEASE_UNHELD`) copies nothing, no process holding the surface to read the
    /// pages, where `clients`, the count of the surface's clients in the block's header as the
    /// copy is made, is still 0; past 0, a process took the surface again before the answer
    /// (the kext marks it so), and the cache is copied. The cache is returned, for the caller to
    /// let go of.
    #[allow(clippy::too_many_arguments)] // the release's, and its trace
    pub(crate) fn released(
        &mut self,
        device: &P::Device,
        block: u64,
        flags: u32,
        cpu: Option<u64>,
        clients: Option<u32>,
        done: Box<dyn FnOnce() + Send>,
        trace: Option<&CopyTrace>,
    ) -> RaiseResult<Option<SurfaceCache<P>>> {
        if let Some(trace) = trace {
            trace.cache(&self.by_block, block);
        }
        let written = self.written_since_host(block);
        let flagged = flags & RGFX_BLOCK_RELEASE_UNHELD != 0;
        let unheld = written && flagged && clients == Some(0);
        if flagged {
            self.counts.unheld_released += 1;
        }
        if unheld {
            let length = (self.by_block.get(&block)).map_or(0, |cache| cache.length);
            self.counts.unheld += 1;
            self.counts.unheld_bytes += length;
        }
        if !written || unheld || self.behind_the_cpu(block, cpu, "BLOCK_RELEASE") {
            // The pages are let go of once a copy into them still in flight lands: an early copy,
            // or one on the transfer queue.
            let early = (self.by_block.get(&block))
                .and_then(|cache| cache.read_back.copy.as_ref())
                .map(|copy| Arc::clone(&copy.landing));
            let landings: Vec<Arc<Landing>> =
                early.into_iter().chain(self.out_copy(block)).collect();
            if let Some(trace) = trace {
                trace.behind(&landings);
            }
            after_landings(landings, done);
        } else {
            if flags & RGFX_BLOCK_RELEASE_NO_CLIENT != 0 {
                self.counts.unread += 1;
            }
            if flagged {
                self.counts.retaken += 1;
            }
            self.write_back(device, block, done, trace)?;
        }
        let gone = self.by_block.remove(&block);
        self.owed.remove(&block);
        if let Some(cache) = gone.as_ref().filter(|c| c.deferred.is_some()) {
            self.deferred_bytes = self.deferred_bytes.saturating_sub(cache.length);
            if cache.pages.is_none() {
                self.counts.unattached += 1;
            }
        }
        self.watched.0.lock().remove(&block);
        Ok(gone)
    }

    /// The landing of the block's last copy into its pages on the transfer queue, where it has
    /// not landed ([`SurfaceCache::out`]).
    fn out_copy(&self, block: u64) -> Option<Arc<Landing>> {
        let out = self.by_block.get(&block)?.out.as_ref()?;
        (!out.landing.landed()).then(|| Arc::clone(&out.landing))
    }

    /// Notes a copy into the pages of host addresses `runs` asked and not yet landed, letting go of
    /// those that have.
    fn note_into(&mut self, runs: Vec<Range<u64>>, landing: &Arc<Landing>) {
        self.into_pages.retain(|c| !c.landing.landed());
        if !runs.is_empty() {
            self.into_pages.push(IntoPages {
                runs,
                landing: Arc::clone(landing),
            });
        }
    }

    /// The copies of caches into pages among host addresses `runs` ([`device_writes::sorted`])
    /// asked and not yet landed: a release's from its ask, or one on the transfer queue.
    fn copies_into(&self, runs: &[Range<u64>]) -> Vec<Arc<Landing>> {
        (self.into_pages.iter())
            .filter(|c| !c.landing.landed() && device_writes::overlap(&c.runs, runs))
            .map(|c| Arc::clone(&c.landing))
            .collect()
    }

    /// Whether the block's cache holds a command buffer's write the pages do not: one was
    /// committed since its last copy, and no device of the host wrote the pages since its last
    /// fill. Where one did, its write is the newer, and the cache's is dropped.
    fn written_since_host(&mut self, block: u64) -> bool {
        let host = self.watched.writes(block);
        let Some(cache) = self.by_block.get_mut(&block) else {
            return false;
        };
        if cache.written && host > cache.host_read {
            cache.written = false;
            self.counts.host_writes += 1;
        }
        cache.written
    }

    /// Whether the CPU's generation `cpu` of the surface whose block is at `block`, read from the
    /// block's header as a copy out of its cache is made (`message`'s), is past the cache's: then
    /// a read-write lock moved it that this copy never ordered (it gave up waiting for the copy's
    /// answer, the kext's `kDownloadAnswerMs`, or for the block's release, and went on), and the
    /// pages hold the CPU's writes under that lock, which the copy would land over. They are
    /// newer than what the cache holds, as for a fill ([`Self::needs_fill`]), so nothing is
    /// copied: the next use fills the cache from them. Counted and logged; false where the block
    /// has no cache or its header was not read.
    fn behind_the_cpu(&mut self, block: u64, cpu: Option<u64>, message: &str) -> bool {
        let (Some(cache), Some(cpu)) = (self.by_block.get(&block), cpu) else {
            return false;
        };
        let held = cache.generation.unwrap_or(0);
        if held >= cpu {
            return false;
        }
        self.counts.behind += 1;
        log::warn!(
            target: "reims_vgpu::graphics", kind:% = message;
            "the IOSurface block at {block:#x}: a {message} of its device cache, of CPU \
             generation {held}, copies nothing: the CPU's generation moved to {cpu} under a lock \
             that did not wait for the copy"
        );
        true
    }

    /// Copies the cache of the block at `block` into the surface's pages on the transfer queue,
    /// after every command buffer that used it; `done` once the copy has finished. Where no copy
    /// can be made `done` is given at once. The copy signals the transfer event, which a use that
    /// writes the cache waits for on the GPU while it is in flight, and lands for the downloads and
    /// releases that copy nothing behind it ([`SurfaceCache::out`]), and for the `SURFACE_PAGES`
    /// of a block over the same pages ([`Self::copies_into`]).
    fn write_back(
        &mut self,
        device: &P::Device,
        block: u64,
        done: Box<dyn FnOnce() + Send>,
        trace: Option<&CopyTrace>,
    ) -> RaiseResult<()> {
        // A deferred block whose pages are not named has nowhere to copy to.
        if self.unattached(block).is_some() || !self.make_transfer(device)? {
            done();
            return Ok(());
        }
        let signal =
            (self.signal_with.as_ref()).map(|(event, number)| (event.clone_ref(), *number));
        let (Some(transfer), Some(cache)) = (self.transfer.as_mut(), self.by_block.get_mut(&block))
        else {
            done();
            return Ok(());
        };
        let Some(cb) = transfer
            .queue
            .command_buffer(ON)?
            .filter(|_| cache.pages.is_some())
        else {
            done();
            return Ok(());
        };
        wait_before::<P>(&cb, signal.as_ref())?;
        let users = wait_for_users::<P>(&cb, &self.ends, cache)?;
        let staged = copy::<P>(device, &cb, cache, Way::Out, block, &mut self.counts)?;
        signal_after::<P>(&cb, signal.as_ref())?;
        // The buffers live until the copy is done, past the cache's own end at a release.
        let view = match &cache.pages {
            Some(CachePages::View(pages)) => Some(pages.clone_ref()),
            Some(CachePages::Runs { .. } | CachePages::Copied(_)) | None => None,
        };
        let length = cache.length;
        let held = (cache.device.clone_ref(), view);
        let now = trace.and_then(|trace| trace.now.clone());
        transfer.signalled += 1;
        let value = transfer.signalled;
        cb.encode_signal(Some(EventRef::Event(&transfer.event)), value)?;
        let landing = Landing::new();
        let lands = Arc::clone(&landing);
        match staged {
            // A copied cache's pages take the staging buffer's bytes on the host's CPU, before
            // `done`: a copy too long for the platform's thread that sees the completion.
            Some(staged) => cb.add_completed_handler(
                ON,
                Box::new(move |info: CompletedInfo| {
                    let at = now.as_ref().map(|now| now());
                    staged.done(length);
                    drop(held);
                    let took = (now.as_ref().zip(at))
                        .map_or(std::time::Duration::ZERO, |(now, at)| {
                            now().saturating_duration_since(at)
                        });
                    lands.land_timed(info.gpu_times_ns, took);
                    done();
                }),
            )?,
            // Else answered as the platform sees the copy done, with no handler thread woken
            // between, as an early copy's landing is; the buffers go in the completed handler.
            None => {
                cb.add_completion_seen_handler(
                    ON,
                    Box::new(move |info: CompletedInfo| {
                        lands.land_timed(info.gpu_times_ns, std::time::Duration::ZERO);
                        done();
                    }),
                )?;
                cb.add_completed_handler(ON, Box::new(move |_: CompletedInfo| drop(held)))?;
            }
        }
        cb.commit(ON)?;
        if let Some(trace) = trace {
            trace.made(users, &landing);
        }
        if signal.is_some() {
            self.signal_with = None;
        }
        let length = cache.length;
        if let Some(cache) = self.by_block.get_mut(&block) {
            cache.written = false;
            cache.users.clear();
            // This copy is the cache's last: an early copy before it is waited for on the GPU.
            cache.read_back.writes = 0;
            cache.read_back.copy = None;
            cache.out = Some(OutCopy {
                value,
                landing: Arc::clone(&landing),
            });
        }
        let runs = (self.watched.0.lock().get(&block)).map_or_else(Vec::new, |p| p.runs.clone());
        self.note_into(runs, &landing);
        self.counts.downloads += 1;
        self.counts.downloaded_bytes += length;
        Ok(())
    }

    /// Makes the transfer queue where the first command queue's make did not
    /// ([`Caches::queue_made`]); whether there is one.
    fn make_transfer(&mut self, device: &P::Device) -> RaiseResult<bool> {
        if self.transfer.is_none() {
            self.transfer = Transfer::new(device, &self.on_fill_done)?;
        }
        Ok(self.transfer.is_some())
    }

    /// Queue `queue` is gone: the end events to let go of, its own where no cache names it
    /// ([`Self::sweep_gone`]), so no copy waits for it any more. One a cache still names stays,
    /// for that cache's next copy, which waits for the queue's last buffer that used it, and goes
    /// at a later sweep once the cache's copy, or its release, names it no more.
    fn queue_ended(&mut self, queue: u64) -> Vec<P::Event> {
        if self.ends.contains_key(&queue) {
            self.gone.insert(queue);
        }
        self.sweep_gone()
    }

    /// Whether a cache names queue `queue`: a command buffer of it used the cache since the
    /// cache's last copy, or since a deferred copy of it was asked, which the next copy waits
    /// for, or the cache's early copy ran on it.
    fn names_queue(&self, queue: u64) -> bool {
        let used = (self.by_block.values()).any(|cache| {
            cache.users.contains_key(&queue)
                || (cache.read_back.copy.as_ref()).is_some_and(|copy| copy.queue == queue)
        });
        used || (self.deferred.values().flatten()).any(|d| d.since.users.contains_key(&queue))
    }

    /// The end events of the queues gone that no cache names any more, taken out to be let go
    /// of: a queue's buffers that used a cache name its queue until the cache's next copy, which
    /// waits for them, or its release, so each gone queue's event is looked at again at each
    /// queue's make and end. Kept until the device's reset, they piled up one a queue that ever
    /// used a cache, as processes came and went.
    fn sweep_gone(&mut self) -> Vec<P::Event> {
        if self.gone.is_empty() {
            return Vec::new();
        }
        let unnamed: Vec<u64> = (self.gone.iter().copied())
            .filter(|&queue| !self.names_queue(queue))
            .collect();
        let mut events = Vec::with_capacity(unnamed.len());
        for queue in unnamed {
            self.gone.remove(&queue);
            if let Some((event, _)) = self.ends.remove(&queue) {
                events.push(event);
            }
        }
        events
    }

    /// The device's reset: every block's cache goes uncopied, as its surface's pages are the
    /// rebooted guest's again, and so do the queues' end events, each buffer and event let go of
    /// in a refusal scope, and every copy of a cache still deferred is dropped unmade, its
    /// answer owed to no one (no command buffer is begun across the reset, so none waits on the
    /// CPU), its event set to its number for the command buffers that wait for it on the GPU
    /// and its present told, and the buffers parked behind them leave their counts. The host's
    /// transfer queue stays, and so do the counts.
    pub(crate) fn reset(&mut self) {
        let caches = std::mem::take(&mut self.by_block);
        let ends = std::mem::take(&mut self.ends);
        let deferred = std::mem::take(&mut self.deferred);
        for parked in std::mem::take(&mut self.parked) {
            self.forget(&parked.counts);
        }
        self.gone.clear();
        self.owed.clear();
        self.deferred_bytes = 0;
        self.watched.0.lock().clear();
        // What waits for a deferred copy, on the GPU or for a present, waits for it no more.
        for d in deferred.values().flatten() {
            if let Some(event) = &d.event {
                let _ = event.set_signaled_value(d.number);
            }
            if let Asked::Fill { filled, .. } = &d.copy {
                filled.set(None);
                self.tell = true;
            }
        }
        if let Err(refusal) = refusals::refusing(move || drop((caches, ends, deferred))) {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = refusal;
                "the IOSurface device caches were let go of at the device's reset, and a release \
                 was refused: {refusal}"
            );
        }
    }

    /// How many blocks have a cache (for the tests).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.by_block.len()
    }
}

/// The device caches behind their lock, and the order of their copies against the command buffers
/// that use them. A command buffer counts among a block's users from its begin to its finish
/// ([`Uses`]), each use under a number of its own in the order they are counted: where its use is
/// the walk's alone, between them, a copy follows it.
///
/// - **A copy into a cache** (a fill, at a command buffer's begin or a present) follows every
///   command buffer begun and not finished that uses the cache. **A copy out of a cache** (a
///   download, a release, the block's pages named again) follows every one of them that may
///   write it.
/// - **Where they are made.** A copy no such buffer holds up, and no copy asked before it, is made
///   at once. Any other is made in the finish of the last buffer it follows
///   ([`SurfaceCaches::run_deferred`]), on that buffer's thread, after its commit and its early
///   copies, in the order the copies were asked, so the thread that asked waits for none of them:
///   a fill at a begin, a present's fill, and a download and a release, which ring 0 asks for (a
///   download on its block's lane, or on the control thread where it commits no copy then,
///   [`Caches::try_download`]; a release on its context's lane) and are answered when their copy
///   is done. The block's pages named again, which the buffers over the block made on ring 0 after
///   it lie over, waits there for the buffers that may write the cache and for the copies asked.
/// - **What follows a deferred copy.** A use counted after a deferred copy was asked is ordered
///   after it on the GPU: its command buffer waits for the cache's event, which the copy's
///   command buffer signals to the copy's number ([`SurfaceCache::event`]), where the use reads
///   what the copy writes (a fill), or may write what the copy reads (any copy). It holds its
///   queue, as a buffer that waits for an event does, where a fill made at once holds nothing
///   ([`Caches::before_answering`]). Its marks of the cache (written) and its end
///   value are kept apart until the copy is made ([`Since`]), so the copy follows none of them.
///   A use that only reads follows no copy out, which leaves the cache as it is.
/// - **A buffer finished before a copy it follows is made** stays counted among the users of
///   every block it used, and among those that may write them, until the copy is made
///   ([`SurfaceCaches::parked`]). The copies are committed on one queue, the transfer queue, in
///   the order they are made, and one made behind such a buffer would wait there for its end,
///   which waits for the copy it follows, committed behind it on that queue: neither would run.
///
/// So a copy comes before or after each such buffer's whole use, as under one lock, and the
/// queues' end events order it on the GPU after the buffers committed before it
/// ([`SurfaceCaches::after`]); the cache's event orders the buffers committed before it after it.
/// A buffer's uses are counted together, after those of every buffer begun before it
/// ([`Self::before_answering`]); a copy follows only buffers begun before it was asked, and a buffer waits
/// only for copies asked before it was begun, and is let go of only once they are made, so no
/// two wait for each other, on the CPU or on the GPU, and the transfer queue runs its copies in
/// an order that waits for nothing committed after them. No begin, no present and no ring-0
/// message waits on the CPU for another buffer's walk, but where the platform makes no event
/// (then the begin waits there for the copy before counting its uses, as it did for the
/// buffers), a decode of the host's for its output's fill ([`Self::wait_filled`]), and the
/// block's pages named again. Nor does one wait for a cache's make: a new cache's buffer, and the
/// buffer over its pages, are made with the lock let go of ([`CacheMaker`]), and only taken under
/// it.
pub(crate) struct Caches<P: Platform> {
    state: Mutex<SurfaceCaches<P>>,
    moved: Condvar,
    /// What each fill tells as it finishes ([`Self::on_fill_done`]).
    on_fill_done: Arc<OnceLock<FillHook>>,
}

/// How a new cache's buffer is made ([`SurfaceCaches::maker`]): read under the caches' lock and
/// used with it let go of, as a new block of the device's memory may take tens of milliseconds
/// (**measured** 2026-11-09, a macOS 26.7.2 guest's Safari scroll on a Radeon 780M: each 64 MiB
/// block of 7.125 MiB dma-buf caches, every 8th cache, took 8 to 28 ms), and every queue's begin
/// takes that lock ([`Caches::before_answering`]). Made under the lock, those makes held 11 of
/// the scroll's begins for 14 to 25 ms each; made outside it, none of the same scroll's begins
/// took 2 ms or more for its caches (**measured** 2026-11-09, the same guest and host).
#[derive(Clone, Copy, Debug)]
struct CacheMaker {
    /// Caches of surfaces of two planes are made exportable, for the host's video to import
    /// ([`Caches::set_exportable`]).
    exportable: bool,
    /// Laid-out caches of two planes are made with their memory a dma-buf
    /// ([`Caches::set_dma_buf`]).
    dma_buf: bool,
}

impl CacheMaker {
    /// A new cache's buffer of `length` bytes, Private: where a layout gave the surface `planes`
    /// and they are two, as the host's video's output (NV12, P010) has, exportable where the
    /// host's video imports caches ([`Caches::set_exportable`]) and the platform makes one; else,
    /// where `layout` gives two planes and the host's video decodes into caches
    /// ([`Caches::set_dma_buf`]), one whose memory is also a dma-buf, laid as the next is, where
    /// the platform makes one; else, where `layout` is given (the kext's ask gave the surface's
    /// planes, so the cache's copies into the pages take its texels alone:
    /// [`SurfaceCache::texels`]), one that declares the bytes between a
    /// texture's rows unobserved, which a texture over the cache whose rows lie with bytes
    /// between them may hold and change in place, with its byte `aligned` placed where the
    /// platform's textures over its memory may start in place where given ([`aligned_byte`]), so
    /// that textures over its surface's planes lie in it rather than in copies of their own
    /// ([`GpuDevice::new_buffer_of_length_padding_unobserved`]); else as any other buffer, its
    /// byte `aligned` placed alike where given and nothing else declared
    /// ([`GpuDevice::new_buffer_of_length_aligned`]). An exportable cache places its byte
    /// `aligned` alike where given ([`GpuDevice::new_exportable_buffer_of_length`]). With the
    /// buffer, the cache's life where its memory was made for the host's video, exportable or a
    /// dma-buf ([`SurfaceCache::life`]).
    fn make<P: Platform>(
        self,
        device: &P::Device,
        length: u64,
        planes: Option<usize>,
        layout: Option<&[PagesPlane]>,
        aligned: Option<u64>,
    ) -> RaiseResult<NewCache<P>> {
        let laid_out = layout.is_some();
        let mut counts = MakerCounts::default();
        let life = || Some(MemoryLife::new());
        #[cfg(unix)]
        if self.exportable && planes == Some(2) {
            let made =
                device.new_exportable_buffer_of_length(ON, length, STORAGE_PRIVATE, aligned)?;
            if let Some(cache) = made {
                counts.exportable = true;
                counts.aligned = aligned.is_some();
                return Ok(NewCache::new(Some((cache, life())), counts));
            }
            counts.unexportable = true;
        }
        #[cfg(unix)]
        if self.dma_buf && layout.is_some_and(|planes| planes.len() == 2) {
            let made = device.new_dma_buf_buffer_of_length_padding_unobserved(
                ON,
                length,
                STORAGE_PRIVATE,
                aligned,
            )?;
            if let Some(cache) = made {
                counts.dma_buf = true;
                counts.aligned = aligned.is_some();
                return Ok(NewCache::new(Some((cache, life())), counts));
            }
            counts.no_dma_buf = true;
        }
        let made = match laid_out {
            true => {
                let made = device.new_buffer_of_length_padding_unobserved(
                    ON,
                    length,
                    STORAGE_PRIVATE,
                    aligned,
                )?;
                counts.aligned = aligned.is_some() && made.is_some();
                made
            }
            false => match aligned {
                Some(byte) => {
                    let made =
                        device.new_buffer_of_length_aligned(ON, length, STORAGE_PRIVATE, byte)?;
                    counts.aligned = made.is_some();
                    made
                }
                None => device.new_buffer_of_length(ON, length, STORAGE_PRIVATE)?,
            },
        };
        Ok(NewCache::new(made.map(|cache| (cache, None)), counts))
    }
}

/// What a cache's make counts ([`CacheCounts`]'s `exportable`, `unexportable`, `dma_buf`,
/// `no_dma_buf` and `aligned`), added as the cache is taken ([`NewCache::take`]).
#[derive(Clone, Copy, Debug, Default)]
struct MakerCounts {
    exportable: bool,
    unexportable: bool,
    dma_buf: bool,
    no_dma_buf: bool,
    aligned: bool,
}

/// A new cache's buffer made with the caches' lock let go of ([`CacheMaker::make`]), with its
/// life where its memory was made for the host's video, and what its make counts. One the caches
/// do not take is let go of by the caller once it has let go of their lock.
pub(crate) struct NewCache<P: Platform> {
    made: Option<(P::Buffer, Option<MemoryLife>)>,
    counts: MakerCounts,
}

impl<P: Platform> NewCache<P> {
    fn new(made: Option<(P::Buffer, Option<MemoryLife>)>, counts: MakerCounts) -> Self {
        NewCache { made, counts }
    }

    /// The buffer and its life, `None` where the platform made none; its make counted into
    /// `counts`.
    fn take(self, counts: &mut CacheCounts) -> Option<(P::Buffer, Option<MemoryLife>)> {
        let made = self.counts;
        counts.exportable += u64::from(made.exportable);
        counts.unexportable += u64::from(made.unexportable);
        counts.dma_buf += u64::from(made.dma_buf);
        counts.no_dma_buf += u64::from(made.no_dma_buf);
        counts.aligned += u64::from(made.aligned);
        self.made
    }
}

/// A new cache made with the caches' lock let go of ([`Caches::make`]): its buffer, and the other
/// end of its copies, made where the buffer was.
pub(crate) struct Made<P: Platform> {
    cache: NewCache<P>,
    pages: Option<CachePages<P>>,
}

/// The blocks a begin answered before its fills holds ([`Caches::before_answering`]), let go of as
/// it is dropped, with the caches' lock not held, and the begins waiting for them woken.
struct Answering<'a, P: Platform> {
    caches: &'a Caches<P>,
    blocks: Vec<u64>,
}

impl<P: Platform> Drop for Answering<'_, P> {
    fn drop(&mut self) {
        if self.blocks.is_empty() {
            return;
        }
        let mut state = self.caches.state.lock();
        for block in &self.blocks {
            state.answering.remove(block);
        }
        drop(state);
        self.caches.moved.notify_all();
    }
}

/// What a command buffer's work reaches of the device caches besides the uses it names
/// ([`Caches::before_answering`]).
#[derive(Debug, Default)]
pub(crate) struct Reach {
    /// The context it was committed in.
    pub(crate) context: Option<u32>,
    /// It holds render, compute or blit work, which may reach a texture over a surface with no
    /// use named (through an argument buffer).
    pub(crate) work: bool,
    /// Where its work is decodes alone: the blocks of the surfaces its decodes write whole, each
    /// with the two planes they write, its luma's and its chroma's, as laid out in the surface's
    /// bytes.
    pub(crate) decoded: Vec<(u64, [PagesPlane; 2])>,
}

/// A command buffer's count among the users of the blocks it uses ([`Caches`]), until its finish
/// ([`Caches::finish`]) or its drop.
pub(crate) struct Uses<P: Platform> {
    caches: Arc<Caches<P>>,
    /// Its count among each block's users.
    blocks: Vec<Count>,
    /// The deferred copies it waits for on the GPU, each by block and number.
    follows: Vec<(u64, u64)>,
}

impl<P: Platform> Drop for Uses<P> {
    fn drop(&mut self) {
        if self.blocks.is_empty() {
            return;
        }
        let tell = {
            let mut state = self.caches.state.lock();
            state.unhold(&self.blocks);
            state.take_tell()
        };
        self.caches.moved.notify_all();
        self.caches.tell(tell);
    }
}

/// The blocks of `uses` that have caches, each once, in block order, with the first use of each.
fn by_block<'u>(
    caches: &SurfaceCaches<impl Platform>,
    uses: &'u [RgfxOpSurfaceUse],
) -> Vec<&'u RgfxOpSurfaceUse> {
    let mut out: Vec<&RgfxOpSurfaceUse> =
        uses.iter().filter(|u| caches.has(u.block_offset)).collect();
    // Stable: the first use of a block stays first.
    out.sort_by_key(|u| u.block_offset);
    out.dedup_by_key(|u| u.block_offset);
    out
}

impl<P: Platform> Caches<P> {
    /// No caches.
    pub(crate) fn new() -> Self {
        let on_fill_done = Arc::default();
        Caches {
            state: Mutex::new(SurfaceCaches::new(Arc::clone(&on_fill_done))),
            moved: Condvar::new(),
            on_fill_done,
        }
    }

    /// Whether the transfer queue and the end events are made with the command queues
    /// ([`Self::queue_made`]): not where the host's GPU shares the CPU's memory, whose first copy
    /// makes them.
    pub(crate) fn set_early(&self, early: bool) {
        self.state.lock().early = early;
    }

    /// Has each fill tell `hook` as it finishes, after the threads waiting for it are woken: the
    /// host's presents that wait for a fill ([`Self::fill_now`]). Set once; a second is ignored.
    pub(crate) fn on_fill_done(&self, hook: FillHook) {
        let _ = self.on_fill_done.set(hook);
    }

    /// Command queue `queue` was made: its end event is made now, and the transfer queue too
    /// where there is none yet, each with no lock held, where they are made early
    /// ([`Self::set_early`]). A command buffer's begin and finish, under the caches' lock, then
    /// make neither ([`SurfaceCaches::after`], [`SurfaceCaches::make_transfer`]): every other
    /// queue's buffers that use a cache would wait for the creation, which the platform's other
    /// threads' work on the device's memory and semaphores can stretch to milliseconds (measured
    /// on the RTX 5080 Laptop GPU host: 0.5 to 3 ms an end event at a queue's first buffer in a
    /// guest's boot, against 0.1 ms where no other thread makes or frees the device's memory or
    /// semaphores).
    pub(crate) fn queue_made(&self, device: &P::Device, queue: u64) -> RaiseResult<()> {
        let (early, transfer, read_back) = {
            let state = self.state.lock();
            (
                state.early,
                state.transfer.is_some(),
                state.ends.contains_key(&READ_BACK),
            )
        };
        if !early {
            return Ok(());
        }
        let mut end = device.new_event(ON)?;
        let mut made = match transfer {
            true => None,
            false => Transfer::new(device, &self.on_fill_done)?,
        };
        // The read-back queue's end event, made again after a reset let go of it.
        let mut read_back_end = match read_back {
            true => None,
            false => device.new_event(ON)?,
        };
        let swept = {
            let mut state = self.state.lock();
            // A queue of the same id is the queue's again, whose event it takes on.
            state.gone.remove(&queue);
            // Kept only where no buffer's begin or finish made the same meanwhile; else let go of
            // below.
            for (key, made_end) in [(queue, &mut end), (READ_BACK, &mut read_back_end)] {
                if let Some(event) = made_end.take() {
                    match state.ends.entry(key) {
                        Entry::Vacant(slot) => {
                            slot.insert((event, 0));
                        }
                        Entry::Occupied(_) => *made_end = Some(event),
                    }
                }
            }
            if state.transfer.is_none() {
                state.transfer = made.take();
            }
            // And the gone queues' events no cache names any more ([`SurfaceCaches::sweep_gone`]).
            state.sweep_gone()
        };
        if (end.is_some() || made.is_some() || read_back_end.is_some() || !swept.is_empty())
            && let Err(refusal) =
                refusals::refusing(move || drop((end, made, read_back_end, swept)))
        {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = refusal;
                "queue {queue:#x}'s spare end event or transfer queue, or a gone queue's end \
                 event, was let go of, and its release was refused: {refusal}"
            );
        }
        Ok(())
    }

    /// Command queue `queue` is gone (its destroy, or its context's close): its end event goes
    /// once no cache names it, and so do the other gone queues' that none names any more
    /// ([`SurfaceCaches::queue_ended`]), let go of with no lock held. A refusal is logged.
    pub(crate) fn queue_ended(&self, queue: u64) {
        let ends = self.state.lock().queue_ended(queue);
        if ends.is_empty() {
            return;
        }
        if let Err(refusal) = refusals::refusing(move || drop(ends)) {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = refusal;
                "queue {queue:#x}'s end event, or another gone queue's, was let go of, and its \
                 release was refused: {refusal}"
            );
        }
    }

    /// The caches, locked: for what orders nothing against command buffers' uses.
    pub(crate) fn lock(&self) -> MutexGuard<'_, SurfaceCaches<P>> {
        self.state.lock()
    }

    /// The identity of the cache of the block at `block` ([`SurfaceCaches::identity`]).
    pub(crate) fn identity(&self, block: u64) -> Option<MemoryIdentity> {
        self.state.lock().identity(block)
    }

    /// Whether the block at `block` has a cache.
    pub(crate) fn has(&self, block: u64) -> bool {
        self.state.lock().has(block)
    }

    /// The device's reset ([`SurfaceCaches::reset`]), which no command buffer is begun across:
    /// the guest's messages carried out before it have ended.
    pub(crate) fn reset(&self) {
        let tell = {
            let mut state = self.state.lock();
            state.reset();
            state.take_tell()
        };
        self.moved.notify_all();
        self.tell(tell);
    }

    /// Tells the presents waiting for a fill, where a deferred fill made nothing
    /// ([`SurfaceCaches::take_tell`]), as a fill that finished tells them; called with no lock of
    /// the caches' held.
    fn tell(&self, tell: bool) {
        if tell && let Some(hook) = self.on_fill_done.get() {
            hook();
        }
    }

    /// Makes each cache of a surface a layout gives two planes from now on exportable where the
    /// platform makes one, for the host's video to import and write its pictures into on its own
    /// GPU work (`graphics::nvidia`): set once, at the device's make, before any cache, where the
    /// device's video imports caches. The video writes only two-plane surfaces (NV12, P010), so
    /// a cache of one plane, or of a surface no layout gave, is made as any other cache.
    pub(crate) fn set_exportable(&self) {
        self.state.lock().exportable = true;
    }

    /// Makes each laid-out cache of two planes from now on with its memory a dma-buf where the
    /// platform makes one, for the host's video to decode its pictures into (`graphics::vaapi`):
    /// set once, at the device's make, before any cache, where the device's video decodes into
    /// caches. The video decodes only into two-plane surfaces (NV12, P010), so a cache of one
    /// plane is made as any other laid-out cache. Made dma-bufs, one-plane caches had the
    /// runtime make their blocks on the thread that makes them, with the kernel's compaction of
    /// memory under it 19 to 23 times a scroll (300 to 570 ms in all), and their exports waited
    /// 47 to 155 ms a scroll behind imports that wait for the lock of the process's GPU address
    /// space, which a block being made holds; made as other laid-out caches, neither (**measured**
    /// 2026-11-09, a macOS 12.7.6 guest's Safari scroll on a Radeon 780M, whose caches all have
    /// one plane).
    pub(crate) fn set_dma_buf(&self) {
        self.state.lock().dma_buf = true;
    }

    /// Sets the bytes deferred blocks' caches may hold at once (the device's `DEFERRED_BUDGET`,
    /// read once at its start); 0, as at the caches' make, defers none.
    pub(crate) fn set_deferred_budget(&self, budget: u64) {
        self.state.lock().set_deferred_budget(budget);
    }

    /// Has a deferred block ask with its surface's layout from now on
    /// ([`SurfaceCaches::set_deferred_asks`]): set once, at the device's make.
    pub(crate) fn set_deferred_asks(&self) {
        self.state.lock().set_deferred_asks();
    }

    /// Has a deferral carry its surface's resident pages from now on
    /// ([`SurfaceCaches::set_deferred_carries`]): set once, at the device's make.
    pub(crate) fn set_deferred_carries(&self) {
        self.state.lock().set_deferred_carries();
    }

    /// Waits, letting the lock go meanwhile, while a command buffer begun and not finished that
    /// may write the block at `block`'s cache is, or a copy of it asked waits for command buffers
    /// ([`SurfaceCaches::deferred`]): for the block's pages named again, whose buffers made on
    /// ring 0 after it lie over what it leaves.
    fn wait_copies(&self, state: &mut MutexGuard<'_, SurfaceCaches<P>>, block: u64) {
        while state.pending(block, true) > 0 || state.deferring(block) {
            self.moved.wait(state);
        }
    }

    /// The tests' [`Self::before_answering`] with no answer: whether `cb` waits on the GPU for a
    /// copy of a cache, and its uses.
    #[cfg(test)]
    pub(crate) fn before(
        self: &Arc<Self>,
        device: &P::Device,
        cb: &P::CommandBuffer,
        uses: &[RgfxOpSurfaceUse],
        reach: &Reach,
    ) -> RaiseResult<(bool, Option<Uses<P>>)> {
        let (waits, _, uses) = self.before_answering(device, cb, uses, reach, None)?;
        Ok((waits, uses))
    }

    /// The blocks `uses` names that have caches, before command buffer `cb` runs, and its count
    /// among their users ([`SurfaceCaches::before_one`]): each filled where it needs it
    /// ([`SurfaceCaches::needs_fill`]), at once or after every command buffer begun and not
    /// finished that uses it; `cb` made to wait on the GPU for each one's last fill not yet done,
    /// and for the deferred copies of it the use follows; and whether it waits for one. A use that
    /// may write marks its cache written.
    ///
    /// Every use is counted with the lock held throughout, so a buffer's counts are numbered
    /// together, after those of every buffer begun before it: a copy follows only buffers begun
    /// before it was asked, and a buffer only copies asked before it, so no two wait for each
    /// other. First, with nothing counted, each cache whose deferred copy the buffer would follow
    /// gets its event, made with the lock let go; where the platform makes none, the copy is
    /// waited for on the CPU, counted ([`CacheCounts::cpu_waits`]).
    ///
    /// Before any of that, a buffer whose `reach` holds render, compute or blit work has the first
    /// fills its context's caches owe made, and waits for them on the GPU
    /// ([`SurfaceCaches::fill_reached`]); a use of a block its decodes write whole takes the first
    /// fill as met ([`SurfaceCaches::before_one`]).
    ///
    /// It answers too whether `cb` follows a deferred copy of a cache ([`Uses`]'s `follows`),
    /// which holds its queue; and, where `answer` is given and `cb` would follow none, `answer` is
    /// called first, before the fills the begin makes, with no lock held: so a buffer's
    /// `CB_SUBMITTED` is posted before the host copies its surfaces' pages into their caches (a
    /// new surface's first fill, at its first use, takes milliseconds), as a fill made at once
    /// waits on the GPU for nothing of the guest's still to come. A copy follows `cb` only where
    /// it waits for buffers begun before it ([`SurfaceCaches::follows_deferred`]); from the answer
    /// until the fills are made, the blocks of `cb`'s uses are held ([`SurfaceCaches::answering`])
    /// and another begin that names one waits for them, so none counts a use of one between and
    /// turns a fill decided at once into one that waits for its buffer. A buffer that follows a
    /// deferred copy is not answered here: its copy waits for buffers begun before it, which may
    /// wait for the guest.
    pub(crate) fn before_answering(
        self: &Arc<Self>,
        device: &P::Device,
        cb: &P::CommandBuffer,
        uses: &[RgfxOpSurfaceUse],
        reach: &Reach,
        answer: Option<&mut dyn FnMut()>,
    ) -> RaiseResult<(bool, bool, Option<Uses<P>>)> {
        if uses.is_empty() && !reach.work {
            if let Some(answer) = answer {
                answer();
            }
            return Ok((false, false, None));
        }
        let mut state = self.state.lock();
        while (uses.iter()).any(|u| state.answering.contains(&u.block_offset)) {
            self.moved.wait(&mut state);
        }
        // Let go of as the begin ends, however it ends (a raise, or a panic its caller catches),
        // so no later begin waits for good.
        let mut answered = Answering {
            caches: self,
            blocks: Vec::new(),
        };
        if let Some(answer) = answer {
            let picked = by_block(&state, uses);
            if !picked.iter().any(|u| state.follows_deferred(u)) {
                answered.blocks = picked.iter().map(|u| u.block_offset).collect();
                state.answering.extend(answered.blocks.iter().copied());
                drop(state);
                answer();
                state = self.state.lock();
            }
        }
        let before = self.before_locked(state, device, cb, uses, reach);
        drop(answered);
        let (waits, uses) = before?;
        let follows = uses.as_ref().is_some_and(|u| !u.follows.is_empty());
        Ok((waits, follows, uses))
    }

    /// [`Self::before_answering`] from its lock on, `state`, which it lets go of.
    fn before_locked<'a>(
        self: &'a Arc<Self>,
        mut state: MutexGuard<'a, SurfaceCaches<P>>,
        device: &P::Device,
        cb: &P::CommandBuffer,
        uses: &[RgfxOpSurfaceUse],
        reach: &Reach,
    ) -> RaiseResult<(bool, Option<Uses<P>>)> {
        let reached = match reach.work {
            true => state.fill_reached(device, cb, reach.context)?,
            false => false,
        };
        if uses.is_empty() {
            return Ok((reached, None));
        }
        let picked: Vec<RgfxOpSurfaceUse> = by_block(&state, uses).into_iter().copied().collect();
        let mut spare = Vec::new();
        let mut no_event = HashSet::new();
        let mut waited = false;
        loop {
            let lacking: Vec<u64> = (picked.iter())
                .filter(|u| state.follows_deferred(u) && !state.has_event(u.block_offset))
                .map(|u| u.block_offset)
                .collect();
            if lacking.is_empty() {
                break;
            }
            let make: Vec<u64> = (lacking.into_iter())
                .filter(|block| !no_event.contains(block))
                .collect();
            if make.is_empty() {
                // No event to wait for a copy with on the GPU: it is waited for here.
                if !std::mem::replace(&mut waited, true) {
                    state.counts.cpu_waits += 1;
                }
                self.moved.wait(&mut state);
                continue;
            }
            drop(state);
            let made: Vec<_> = (make.into_iter())
                .map(|block| (block, device.new_shared_event()))
                .collect();
            state = self.state.lock();
            for (block, event) in made {
                if let Some(event) = event
                    && let Some(event) = state.give_event(block, event)
                {
                    spare.push(event);
                }
                if !state.has_event(block) {
                    no_event.insert(block);
                }
            }
        }
        let mut held = Uses {
            caches: Arc::clone(self),
            blocks: Vec::new(),
            follows: Vec::new(),
        };
        let mut waits = reached;
        let mut raised = None;
        for u in &picked {
            let decoded = (reach.decoded.iter())
                .find(|(block, _)| *block == u.block_offset)
                .map(|(_, planes)| planes);
            let (count, follows) = state.before_one(device, cb, u, decoded);
            // Counted at once, so a later copy of the block follows this buffer.
            held.blocks.extend(count);
            match follows {
                Ok((w, followed)) => {
                    waits |= w;
                    held.follows
                        .extend(followed.map(|number| (u.block_offset, number)));
                }
                Err(raise) => {
                    raised = Some(raise);
                    break;
                }
            }
        }
        drop(state);
        if !spare.is_empty()
            && let Err(refusal) = refusals::refusing(move || drop(spare))
        {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = refusal;
                "a device cache's spare event was let go of, and its release was refused: \
                 {refusal}"
            );
        }
        if let Some(raise) = raised {
            return Err(raise);
        }
        Ok((waits, (!held.blocks.is_empty()).then_some(held)))
    }

    /// Whether a command buffer whose `SURFACE_USE` ops are `uses` waits for no copy of a cache
    /// at its begin ([`Self::before_answering`]), so it gets its `CB_SUBMITTED` at its take: no block of
    /// them that has a cache needs a fill ([`SurfaceCaches::needs_fill`]), or follows a copy of
    /// its cache asked and not yet made ([`SurfaceCaches::follow`]), and each one's last fill is
    /// done. Answered waiting for nothing and allocating nothing (a doorbell's thread), each lock
    /// tried: `None` where one is busy.
    pub(crate) fn waits_for_nothing(
        &self,
        mut uses: impl Iterator<Item = RgfxOpSurfaceUse>,
    ) -> Option<bool> {
        let mut state = self.state.try_lock()?;
        let (mut cached, mut busy) = (false, false);
        let none = uses.all(|u| {
            cached |= state.has(u.block_offset);
            match state.waits_for_nothing(&u) {
                Some(none) => none,
                None => {
                    busy = true;
                    false
                }
            }
        });
        if busy {
            return None;
        }
        // Counted where a block of them has a cache: a use of one with none waits for nothing.
        if cached {
            match none {
                true => state.counts.doorbell_free += 1,
                false => state.counts.doorbell_waits += 1,
            }
        }
        Some(none)
    }

    /// Ends command buffer `cb` of queue `queue`, which `uses` counted among the users of caches
    /// ([`SurfaceCaches::after`]), and `commit`s it with the lock let go; then, with the lock
    /// taken again and the buffer still among their users, commits the early copies behind it on
    /// `on`, its queue, where it has one ([`SurfaceCaches::early_copies`]), and it leaves their
    /// users, the copies of their caches that waited for it alone made there
    /// ([`SurfaceCaches::run_deferred`]), or, while a deferred copy it waits for on the GPU is
    /// not yet made, stays among them until it is ([`SurfaceCaches::leave`]). An early copy the
    /// platform refuses is logged and left unmade: the cache stays written, for the download that
    /// follows.
    ///
    /// The platform's commit call can take milliseconds (a long buffer's recording, a driver's
    /// submit), and under the caches' one lock every other queue's begin of a buffer that uses a
    /// cache waited for it, and so did a doorbell's look at the caches, a present's fill and
    /// ring 0's copies. What a copy needs of the buffer still holds with the lock let go: the
    /// buffer counts among the blocks' users until it leaves them here, after its commit and its
    /// early copies, so a fill of one of them, a copy out after a writer, a block named again and
    /// a present's fill follow both. A copy out of a block the buffer only reads, asked
    /// meanwhile, may be committed first, waiting on the GPU for the buffer's end event's value,
    /// as a wait for a signal committed later on another queue does: Metal's GPU waits for it,
    /// and metal2vulkan's scheduler holds the copy until the signal is submitted (its
    /// `event_timeline.rs`, "What the scheduler knows"). Neither copy writes what the buffer uses,
    /// so no order by commit crosses that wait. The queue's later buffers sign their end values
    /// after it, as each is finished after its commit in turn.
    ///
    /// An early copy needs nothing more of the lock: it is committed on the buffer's own queue
    /// right behind the buffer, as the queue's next buffer is committed only once this one has
    /// finished, and only this queue's finish moves the queue's end value. A writer of the block
    /// begun meanwhile on another queue counts a second write, so the block takes no early copy
    /// and its download copies it, as where that writer began before the buffer's commit.
    pub(crate) fn finish(
        &self,
        device: &P::Device,
        queue: u64,
        on: Option<&P::CommandQueue>,
        cb: &P::CommandBuffer,
        mut uses: Uses<P>,
        commit: impl FnOnce() -> RaiseResult<()>,
    ) -> RaiseResult<()> {
        let signalled = self.state.lock().after(device, queue, cb, &uses.blocks);
        let done = signalled.and_then(|()| commit());
        let mut state = self.state.lock();
        // The read-back queue runs its copies in order, so one behind a buffer that waits for a
        // deferred copy not yet made would hold every copy after it there, one of which that
        // deferred copy may wait for: such a buffer's write is copied out by the download that
        // follows instead. One on its own queue, behind it, holds nothing of another's.
        let early = on.is_some() || state.all_made(&uses.follows);
        if done.is_ok()
            && early
            && let Err(raise) = refusing(|| state.early_copies(queue, on, &uses.blocks))
        {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = raise;
                "queue {queue:#x}: the platform raised at an early copy of a device cache into \
                 its pages ({raise:?}); the lock that follows downloads it"
            );
        }
        if done.is_ok() {
            state.leave(&uses.blocks, &uses.follows);
        } else {
            // Never run, so nothing waits behind it.
            state.unhold(&uses.blocks);
        }
        uses.blocks.clear();
        let tell = state.take_tell();
        drop(state);
        self.moved.notify_all();
        self.tell(tell);
        done
    }

    /// Fills the cache of the block at `block` where a present of it at `generation` needs it
    /// ([`SurfaceCaches::fill_now`]), after every command buffer begun and not finished that uses
    /// it, waiting for none: the wait for the fill, which the caller hands to the present, waiting
    /// for nothing on the GPU; `None` where the present waits for no fill.
    pub(crate) fn fill_now(
        &self,
        device: &P::Device,
        block: u64,
        generation: u64,
    ) -> RaiseResult<Option<FillWait>> {
        let mut state = self.state.lock();
        if !state.needs_fill_at(block, generation) {
            return Ok(None);
        }
        state.fill_now(device, block, generation)
    }

    /// `SURFACE_DOWNLOAD` of the block at `block` ([`SurfaceCaches::download`]), after every
    /// command buffer begun and not finished that may write it, waiting for none
    /// ([`Self::after_writers`]), `ran` as it is carried out then, and what it waited for then in
    /// `trace`: `done` once the pages hold the cache, or `failed` where the platform raised at
    /// the copy.
    #[allow(clippy::too_many_arguments)] // the download's, and its trace
    pub(crate) fn download(
        &self,
        device: &P::Device,
        block: u64,
        region: Arc<Region>,
        ran: Answer,
        trace: Arc<CopyTrace>,
        done: Box<dyn FnOnce() + Send>,
        failed: Box<dyn FnOnce(Raise) + Send>,
    ) {
        self.after_writers(
            device,
            block,
            download_op(device, block, region, ran, trace, done, failed),
        );
    }

    /// [`Self::download`] where it waits for no lock and commits no copy now: the caches' lock
    /// is tried, and the download asked only where it copies nothing at once
    /// ([`SurfaceCaches::copies_now`]): its cache nothing wrote since its last copy (an early
    /// copy, landed or not, holds the write), or a command buffer still walked may write it, in
    /// whose finish it is made. Whether it was asked; where not, `done` and `failed` are dropped
    /// ungiven.
    #[allow(clippy::too_many_arguments)] // the download's, and its trace
    pub(crate) fn try_download(
        &self,
        device: &P::Device,
        block: u64,
        region: Arc<Region>,
        ran: Answer,
        trace: Arc<CopyTrace>,
        done: Box<dyn FnOnce() + Send>,
        failed: Box<dyn FnOnce(Raise) + Send>,
    ) -> bool {
        let Some(mut state) = self.state.try_lock() else {
            return false;
        };
        if state.copies_now(block) {
            return false;
        }
        state.after_writers(
            device,
            block,
            download_op(device, block, region, ran, trace, done, failed),
        );
        drop(state);
        self.moved.notify_all();
        true
    }

    /// [`Self::try_download`] on a doorbell's thread, the guest's vCPU: whether the download is
    /// carried out at once with nothing to copy and nothing to wait for
    /// ([`SurfaceCaches::downloaded_now`]), the caches' lock tried, which the caller then answers.
    /// Allocates and frees nothing.
    pub(crate) fn try_downloaded(&self, block: u64) -> bool {
        let Some(mut state) = self.state.try_lock() else {
            return false;
        };
        let done = state.downloaded_now(block);
        drop(state);
        if done {
            self.moved.notify_all();
        }
        done
    }

    /// The block at `block` went, its `BLOCK_RELEASE` ([`SurfaceCaches::released`]), after every
    /// command buffer begun and not finished that may write its cache, waiting for none
    /// ([`Self::after_writers`]), `ran` as it is carried out then: `done` once the pages hold the
    /// cache, or `failed` where the platform raised at the copy, and the cache handed to
    /// `let_go`, which lets go of it on a thread of its own, with no lock of the caches' held.
    /// Nothing names the block until it is answered: the kext gives it out again only then.
    #[allow(clippy::too_many_arguments)] // the release, its answers, and what the copy reads
    pub(crate) fn release(
        &self,
        device: &P::Device,
        block: u64,
        flags: u32,
        region: Arc<Region>,
        ran: Answer,
        trace: Arc<CopyTrace>,
        done: Box<dyn FnOnce() + Send>,
        failed: Box<dyn FnOnce(Raise) + Send>,
        let_go: Box<dyn FnOnce(SurfaceCache<P>) + Send>,
    ) {
        let on = device.clone_ref();
        // Its copy into the pages lands with its answer, which a `SURFACE_PAGES` over the same
        // pages waits for from here, while the copy waits for the buffers that write the cache.
        let (done, failed) = {
            let mut state = self.state.lock();
            let runs = (state.by_block.contains_key(&block))
                .then(|| (state.watched.0.lock().get(&block)).map(|p| p.runs.clone()))
                .flatten();
            match runs {
                Some(runs) => {
                    let landing = Landing::new();
                    state.note_into(runs, &landing);
                    let lands = Arc::clone(&landing);
                    let done: Box<dyn FnOnce() + Send> = Box::new(move || {
                        lands.land();
                        done();
                    });
                    let failed: Box<dyn FnOnce(Raise) + Send> = Box::new(move |raise| {
                        landing.land();
                        failed(raise);
                    });
                    (done, failed)
                }
                None => (done, failed),
            }
        };
        self.after_writers(
            device,
            block,
            Box::new(move |state| {
                ran();
                let device = on;
                let cpu = cpu_generation(&region, block);
                let clients = surface_clients(&region, block);
                match refusing(|| {
                    state.released(&device, block, flags, cpu, clients, done, Some(&trace))
                }) {
                    Ok(Some(cache)) => let_go(cache),
                    Ok(None) => {}
                    Err(raise) => failed(raise),
                }
            }),
        );
    }

    /// Runs `op` on the caches once every command buffer begun and not finished that may write
    /// the block at `block`'s cache has finished, after the copies of it asked before: here
    /// where none is, else in the last one's finish, on its thread, waiting for nothing here. A
    /// command buffer begun meanwhile whose use may write the cache waits for it on the GPU
    /// ([`SurfaceCaches::follow`]), as one begun after the copy, which the ring-0 thread that
    /// asked for it made before going on, was ordered after it.
    fn after_writers(&self, device: &P::Device, block: u64, op: CopyOut<P>) {
        let mut state = self.state.lock();
        state.after_writers(device, block, op);
        drop(state);
        self.moved.notify_all();
    }

    /// The block at `block` named again without a cache, or over other pages, loses it
    /// ([`SurfaceCaches::released`]), after every command buffer begun and not finished that may
    /// write it and the copies of it asked, waiting for them here: the buffers made over the
    /// block after it lie over its pages. The cache is returned, for the caller to let go of
    /// with the lock let go.
    pub(crate) fn released(
        &self,
        device: &P::Device,
        block: u64,
        done: Box<dyn FnOnce() + Send>,
    ) -> RaiseResult<Option<SurfaceCache<P>>> {
        let mut state = self.state.lock();
        self.wait_copies(&mut state, block);
        state.released(device, block, 0, None, None, done, None)
    }

    /// A cache for the block at `block`, whose surface's pages `view` maps: `Ok(false)` where
    /// the platform made no buffer for it, and the block keeps its view. `host_pages` are the
    /// pages' host addresses in the guest's RAM, [`device_writes::sorted`], which the host's
    /// devices' writes are watched over. A new cache is made as zeros on the transfer queue, and
    /// owes its first fill from the pages as of `generation`, the CPU's generation of the surface
    /// in the block's header (the module doc, "Fill"); `Ok(false)` too where the platform made no
    /// command buffer for the zeros, as for no buffer: no cache holds what the GPU's memory held
    /// before.
    ///
    /// A block that has a cache keeps it where its pages are the same, filled again from them as
    /// of `generation` ([`SurfaceCaches::fill_named_again`]), as the kext's generations of the
    /// surface may count anew; any other goes as at a release ([`SurfaceCaches::released`]).
    /// Either way what a buffer wrote of the cache is copied into the pages first. (The kext names
    /// a block's pages once, before any process holds it, and a block's offset is given out again
    /// only once its release is answered.)
    ///
    /// It follows every command buffer begun and not finished that may write the cache the block
    /// had, and the copies of it asked. The cache's buffer, and the buffer over `view` its copies
    /// take, are made with the caches' lock let go of ([`CacheMaker`]), and taken under it
    /// ([`SurfaceCaches::make_with`]).
    pub(crate) fn make(
        &self,
        device: &P::Device,
        block: u64,
        view: &Arc<SurfaceView<P>>,
        host_pages: Vec<Range<u64>>,
        generation: u64,
        layout: CacheLayout,
    ) -> RaiseResult<bool> {
        let length = view.length();
        let pages =
            || buffer_over(device, view, 0, length, 0, None).map(|b| b.map(CachePages::View));
        self.make_with(device, block, length, host_pages, generation, layout, pages)
    }

    /// [`Self::make`] of a cache of `length` bytes whose copies' other end `pages` makes, placed
    /// and laid out as `layout` says ([`CacheLayout`]): a block that keeps its cache
    /// has it filled again under the lock ([`SurfaceCaches::keeps`]); for any other the cache's
    /// buffer, and then `pages`, are made with the lock let go of and taken under it, and what was
    /// not taken is let go of once the lock is let go of again.
    #[allow(clippy::too_many_arguments)] // make's, and what makes the pages
    fn make_with(
        &self,
        device: &P::Device,
        block: u64,
        length: u64,
        host_pages: Vec<Range<u64>>,
        generation: u64,
        layout: CacheLayout,
        pages: impl FnOnce() -> RaiseResult<Option<CachePages<P>>>,
    ) -> RaiseResult<bool> {
        let mut state = self.state.lock();
        self.wait_copies(&mut state, block);
        if state.keeps(block, length, &host_pages) {
            return state.named_again(device, block, generation);
        }
        let maker = state.maker();
        drop(state);
        let aligned = layout.aligned_byte();
        let planes = layout.planes();
        let layout = layout.into_planes();
        let made = maker.make::<P>(device, length, planes, layout.as_deref(), aligned);
        let made = made.and_then(|cache| {
            let pages = match cache.made.is_some() {
                true => pages()?,
                false => None,
            };
            Ok(Made { cache, pages })
        });
        let mut made = Some(made);
        let mut state = self.state.lock();
        self.wait_copies(&mut state, block);
        let taken = state.make_with(
            device, block, length, host_pages, generation, layout, &mut made,
        );
        drop(state);
        drop(made);
        taken
    }

    /// A cache of zeros for the deferred block at `block` ([`SurfaceCaches::make_deferred`]),
    /// carrying `carried` where given, after every command buffer begun and not finished that may
    /// write the cache it had, and the copies of it asked. Its buffer is made with the caches'
    /// lock let go of ([`CacheMaker`]), where the deferred caches' budget has room for it then
    /// ([`SurfaceCaches::deferred_room`]), and taken under the lock where the budget still has
    /// room for it: one made while other deferrals took that room is refused as a deferral
    /// without room is, and its buffer let go of once the lock is let go of.
    #[allow(clippy::too_many_arguments)] // the deferral's, and its carried pages
    pub(crate) fn make_deferred(
        &self,
        device: &P::Device,
        block: u64,
        length: u64,
        block_size: u64,
        layout: CacheLayout,
        carried: Option<CarriedPages>,
        host_pages: Vec<Range<u64>>,
    ) -> RaiseResult<bool> {
        let (room, maker) = {
            let state = self.state.lock();
            (state.deferred_room(block), state.maker())
        };
        let aligned = layout.aligned_byte();
        let planes = layout.planes();
        let layout = layout.into_planes();
        let mut made = (length <= room)
            .then(|| maker.make::<P>(device, length, planes, layout.as_deref(), aligned));
        let mut state = self.state.lock();
        self.wait_copies(&mut state, block);
        let taken = state.make_deferred(
            device, block, length, block_size, layout, carried, host_pages, &mut made,
        );
        drop(state);
        drop(made);
        taken
    }

    /// The deferred block at `block`'s pages taken for its cache ([`SurfaceCaches::attach`]),
    /// which copies nothing, so waits for no command buffer.
    pub(crate) fn attach(
        &self,
        device: &P::Device,
        block: u64,
        named: Named<P>,
        host_pages: Vec<Range<u64>>,
    ) -> RaiseResult<Option<bool>> {
        self.state.lock().attach(device, block, named, host_pages)
    }

    /// A cache for the block at `block`, named by its pages' runs with no view
    /// ([`super::surface_pages`]): `runs` in the surface's order, and `unmade`, which makes the
    /// view where the platform refuses a copy by them; else as [`Self::make`].
    #[allow(clippy::too_many_arguments)] // make's, and the runs
    pub(crate) fn make_by_runs(
        &self,
        device: &P::Device,
        block: u64,
        runs: Vec<Range<u64>>,
        unmade: Arc<Unmade<P>>,
        host_pages: Vec<Range<u64>>,
        generation: u64,
        layout: CacheLayout,
    ) -> RaiseResult<bool> {
        let length = unmade.length();
        let pages = || Ok(Some(CachePages::Runs { runs, unmade }));
        self.make_with(device, block, length, host_pages, generation, layout, pages)
    }

    /// A buffer over the block at `block` made in `context`, its cache the buffer
    /// ([`SurfaceCaches::buffer_made`]): the first fill the cache owes is made now where another
    /// context than its owner's made it.
    pub(crate) fn buffer_made(
        &self,
        device: &P::Device,
        block: u64,
        context: Option<u32>,
    ) -> RaiseResult<()> {
        self.state.lock().buffer_made(device, block, context)
    }

    /// A decode into the block at `block`'s surface answered its picture (`picture`) or none
    /// ([`SurfaceCaches::decoded`]), the fill for none made on `device`.
    pub(crate) fn decoded(
        &self,
        device: Option<&P::Device>,
        block: u64,
        picture: bool,
    ) -> RaiseResult<()> {
        self.state.lock().decoded(device, block, picture)
    }

    /// The deferred block at `block`'s pages taken copied ([`SurfaceCaches::attach_copied`]),
    /// which copies nothing, so waits for no command buffer.
    pub(crate) fn attach_copied(
        &self,
        block: u64,
        copied: &[(HostPtr, u64)],
        host_pages: Vec<Range<u64>>,
    ) -> Option<bool> {
        self.state.lock().attach_copied(block, copied, host_pages)
    }

    /// Waits, on the calling thread, for the last fill of the block at `block`'s cache to be done,
    /// with no lock held, woken as the platform sees the fill done: for work the host starts
    /// outside a command buffer that writes the cache, which a command buffer's wait does not
    /// order. A deferred fill is waited for until it is made, as a begin waited for the command
    /// buffers before it (counted, [`CacheCounts::cpu_waits`]), and then `timeout` at most, as a
    /// fill made is. True where it is done, or the block has no cache or no fill.
    pub(crate) fn wait_filled(&self, block: u64, timeout: std::time::Duration) -> bool {
        let fill = {
            let mut state = self.state.lock();
            let fill = state.last_fill(block);
            if let Some(FillWait::Deferred(filled)) = &fill
                && !filled.done()
            {
                state.counts.cpu_waits += 1;
            }
            fill
        };
        fill.is_none_or(|fill| fill.wait(timeout))
    }

    /// Waits, on the calling thread, for an early copy of the block at `block`'s cache into its
    /// pages still in flight to land ([`SurfaceCaches::early_copies`]), `timeout` at most, with
    /// no lock held, woken as it lands: for work the host starts outside a command buffer that
    /// writes the cache, which a command buffer's wait does not order, so the copy never reads
    /// what it writes. True where it landed, or the block has no cache or no copy in flight.
    pub(crate) fn wait_copied_out(&self, block: u64, timeout: std::time::Duration) -> bool {
        let landing = (self.state.lock().by_block.get(&block))
            .and_then(|cache| cache.read_back.copy.as_ref())
            .map(|copy| Arc::clone(&copy.landing));
        landing.is_none_or(|landing| landing.wait(timeout))
    }

    /// Waits, on the calling thread, for every copy of a cache into the pages of host addresses
    /// `runs` ([`device_writes::sorted`]) asked and not yet landed
    /// ([`SurfaceCaches::copies_into`]), each `timeout` at most, with no lock held: for a
    /// `SURFACE_PAGES` over pages a released block's cache still copies into, whose new block's
    /// buffers the copy would land over. True where none is left.
    pub(crate) fn wait_copies_into(
        &self,
        runs: &[Range<u64>],
        timeout: std::time::Duration,
    ) -> bool {
        let landings = self.state.lock().copies_into(runs);
        landings.iter().all(|landing| landing.wait(timeout))
    }
}

/// A `SURFACE_DOWNLOAD` of the block at `block` as a copy out of its cache
/// ([`SurfaceCaches::download`]), reading the CPU's generation in the block's header in `region`
/// as it is made, `ran` first: `done` once the pages hold the cache, or `failed` where the
/// platform raised.
fn download_op<P: Platform>(
    device: &P::Device,
    block: u64,
    region: Arc<Region>,
    ran: Answer,
    trace: Arc<CopyTrace>,
    done: Box<dyn FnOnce() + Send>,
    failed: Box<dyn FnOnce(Raise) + Send>,
) -> CopyOut<P> {
    let device = device.clone_ref();
    Box::new(move |state| {
        ran();
        let cpu = cpu_generation(&region, block);
        if let Err(raise) = refusing(|| state.download(&device, block, cpu, done, Some(&trace))) {
            failed(raise);
        }
    })
}

/// Makes `cb` wait for the latest command buffer of each queue that used `cache`; how many it
/// waits for.
fn wait_for_users<P: Platform>(
    cb: &P::CommandBuffer,
    ends: &HashMap<u64, (P::Event, u64)>,
    cache: &SurfaceCache<P>,
) -> RaiseResult<usize> {
    let mut waits = 0;
    for (queue, &value) in &cache.users {
        if let Some((event, _)) = ends.get(queue) {
            cb.encode_wait(Some(EventRef::Event(event)), value)?;
            waits += 1;
        }
    }
    Ok(waits)
}

/// Has `cb`, the command buffer of a deferred copy being made ([`SurfaceCaches::carry_out`]),
/// wait on the GPU for the copy before it, which signals its number on the cache's event, where
/// `signal` is the copy's event and number.
fn wait_before<P: Platform>(
    cb: &P::CommandBuffer,
    signal: Option<&(P::SharedEvent, u64)>,
) -> RaiseResult<()> {
    if let Some((event, number)) = signal
        && *number > 1
    {
        cb.encode_wait(Some(EventRef::Shared(event)), number - 1)?;
    }
    Ok(())
}

/// Has `cb`, as [`wait_before`], signal the copy's number on the cache's event as it ends: the
/// command buffers that follow the copy go on then.
fn signal_after<P: Platform>(
    cb: &P::CommandBuffer,
    signal: Option<&(P::SharedEvent, u64)>,
) -> RaiseResult<()> {
    if let Some((event, number)) = signal {
        cb.encode_signal(Some(EventRef::Shared(event)), *number)?;
    }
    Ok(())
}

/// Which way a copy goes: from the pages into the cache, or back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Way {
    In,
    Out,
}

/// What a new cache knows of its surface's layout ([`Caches::make`], [`Caches::make_deferred`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum CacheLayout {
    /// Nothing: the cache's first byte where the platform puts it, and its copies into the pages
    /// the surface's bytes whole, as of a block named cached with no layout.
    #[default]
    Unknown,
    /// The layout a cached or deferred block's own `SURFACE_PAGES` carries on a device that asks
    /// nothing (its GPU keeps its own memory): how many planes the surface has, and its first
    /// plane's first byte where that is past the surface's first ([`aligned_byte`]), which the
    /// cache places where the platform's textures over its memory may start in place; its copies
    /// into the pages still move the surface's bytes whole.
    Carried {
        /// The surface's planes.
        planes: usize,
        /// Its first plane's first byte, where the cache places one.
        first_plane: Option<u64>,
    },
    /// The planes the kext's ask gave (on a GPU that shares the CPU's memory): the cache places
    /// its first plane's byte as [`Self::Carried`] does, and its copies into the pages take the
    /// surface's texels alone ([`SurfaceCache::texels`]).
    Asked(Vec<PagesPlane>),
}

impl CacheLayout {
    /// The byte the cache places for textures in place ([`aligned_byte`]), where it places one.
    fn aligned_byte(&self) -> Option<u64> {
        match self {
            CacheLayout::Unknown => None,
            CacheLayout::Carried { first_plane, .. } => *first_plane,
            CacheLayout::Asked(planes) => aligned_byte(planes),
        }
    }

    /// How many planes the surface has, where a layout gave them.
    fn planes(&self) -> Option<usize> {
        match self {
            CacheLayout::Unknown => None,
            CacheLayout::Carried { planes, .. } => Some(*planes),
            CacheLayout::Asked(planes) => Some(planes.len()),
        }
    }

    /// The planes the kext's ask gave, where it asked.
    fn into_planes(self) -> Option<Vec<PagesPlane>> {
        match self {
            CacheLayout::Asked(planes) => Some(planes),
            CacheLayout::Unknown | CacheLayout::Carried { .. } => None,
        }
    }
}

/// The bytes of a surface laid out as `planes` that its texels are: each plane's rows'
/// `row_bytes` from each row's start, in the surface's order, merged where one ends where the
/// next starts, so a plane whose rows follow on with no bytes between them is one span. The
/// planes lie in the surface's bytes (`surface_pages::layout_planes`).
pub(crate) fn texel_spans(planes: &[PagesPlane]) -> Vec<Range<u64>> {
    let mut rows: Vec<Range<u64>> = Vec::new();
    for p in planes {
        for row in 0..p.rows {
            let start = p.offset.saturating_add(row.saturating_mul(p.bytes_per_row));
            rows.push(start..start.saturating_add(p.row_bytes));
        }
    }
    rows.sort_by_key(|r| r.start);
    let mut spans: Vec<Range<u64>> = Vec::new();
    for r in rows.into_iter().filter(|r| !r.is_empty()) {
        match spans.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => spans.push(r),
        }
    }
    spans
}

/// The byte of a cache of a surface laid out as `planes` that the platform places where its
/// textures over the cache's memory may start in place: its first plane's first byte, where that
/// is past the surface's first (CoreVideo lays a frame's planes 32 bytes into their surface), for
/// a texture over the plane, and over each plane that follows it a multiple of the platform's
/// boundary on, to lie in the cache's memory rather than in a copy of its own; `None` where the
/// plane starts at the surface's first byte, which lies on the boundary in any buffer, or the
/// surface has no plane. The platform decides the boundary: on the Vulkan platform the runtime's
/// (`BufferDescriptor::with_aligned_byte`), a multiple of 256 bytes, the largest linear image's
/// alignment measured (RADV's 256 on an RX 550; 128 on an RTX 5080 Laptop GPU, whose 4K NV12
/// plane textures over its caches 32 bytes in were the runtime's shadows).
pub(crate) fn aligned_byte(planes: &[PagesPlane]) -> Option<u64> {
    let first = planes.first()?.offset;
    (first != 0).then_some(first)
}

/// The runs of `runs`, host addresses of a surface's bytes in its order, that its bytes `spans`
/// lie in, cut to them, each with its first byte of the surface: in one pass, `spans` ascending
/// and apart, as [`texel_spans`] makes them.
fn runs_at(runs: &[Range<u64>], spans: &[Range<u64>]) -> Vec<(u64, Range<u64>)> {
    let mut out = Vec::new();
    let (mut next, mut at) = (0, 0u64);
    for span in spans {
        while let Some(run) = runs.get(next) {
            let end = at.saturating_add(run.end.saturating_sub(run.start));
            let (from, to) = (span.start.max(at), span.end.min(end));
            if from < to {
                out.push((from, run.start + (from - at)..run.start + (to - at)));
            }
            if end > span.end {
                // The run goes on past the span, into the next one's bytes.
                break;
            }
            (next, at) = (next + 1, end);
        }
    }
    out
}

/// Encodes on `blit` the copy of `cache` out into its pages: of its texels alone where its
/// surface's layout is known ([`SurfaceCache::texels`]), else of all its bytes; into `view`
/// where given, else by the pages' runs where the cache has them, in one copy. False where the
/// platform refuses the copy by runs, with nothing encoded.
fn encode_out<P: Platform>(
    blit: &P::BlitEncoder,
    cache: &SurfaceCache<P>,
    view: Option<&P::Buffer>,
) -> RaiseResult<bool> {
    let whole = 0..cache.length;
    let spans = (cache.texels.as_deref()).unwrap_or(std::slice::from_ref(&whole));
    match (view, &cache.pages) {
        (Some(pages), _) => {
            for span in spans {
                blit.encode(
                    ON,
                    BlitCmd::CopyBufferToBuffer {
                        source_buffer: Some(&cache.device),
                        source_offset: span.start,
                        destination_buffer: Some(pages),
                        destination_offset: span.start,
                        size: span.end - span.start,
                    },
                )?;
            }
            Ok(true)
        }
        (None, Some(CachePages::Runs { runs, .. })) => {
            blit.copy_to_guest_ram(ON, &cache.device, &runs_at(runs, spans))
        }
        (None, _) => Ok(false),
    }
}

/// A copied cache's Shared buffer for one copy ([`CachePages::Copied`]), which lives until the
/// copy's command buffer completes; for a copy out of the cache, with the pages its bytes go
/// into then.
struct Staging<P: Platform> {
    buffer: P::Buffer,
    into: Option<Arc<CopiedPages>>,
}

impl<P: Platform> Staging<P> {
    /// The copy into the buffer is done: a copy out of the cache moves its `length` bytes into
    /// the pages.
    fn done(self, length: u64) {
        if let Some(pages) = &self.into {
            // SAFETY: the buffer's `length` Shared bytes, which the copy into it, now done, was
            // the last GPU work to use; the pages are named, so wired.
            unsafe { pages.copy(self.buffer.contents().as_ptr::<u8>(), length, true) };
        }
    }
}

/// Encodes on `cb` a copy of `cache`'s bytes `way`, between the cache and its pages, of the block
/// at `block`: by the pages' runs where the cache has them and the platform copies by them, else
/// through the view, which a refused copy by runs makes now (the module doc; counted in
/// `counts`); or, for copied pages, through a Shared buffer of the copy's own, which the host's
/// CPU fills from the pages now for a copy in, and which the caller keeps until the command
/// buffer completes, then empties into the pages for a copy out ([`Staging::done`]). A deferred
/// block's carried pages, its pages not named, are copied in the same way, each run to its own
/// bytes of the cache, the rest left as it is; nothing is copied out of such a cache. Where no
/// view or buffer can be made, nothing is copied, as where no encoder is.
fn copy<P: Platform>(
    device: &P::Device,
    cb: &P::CommandBuffer,
    cache: &mut SurfaceCache<P>,
    way: Way,
    block: u64,
    counts: &mut CacheCounts,
) -> RaiseResult<Option<Staging<P>>> {
    let Some(blit) = cb.blit_encoder(ON)? else {
        return Ok(None);
    };
    let by_runs = match (&cache.pages, way) {
        (Some(CachePages::Runs { runs, .. }), Way::In) => {
            blit.copy_from_guest_ram(ON, runs, &cache.device, 0)?
        }
        (Some(CachePages::Runs { .. }), Way::Out) => encode_out::<P>(&blit, cache, None)?,
        (Some(CachePages::View(_) | CachePages::Copied(_)) | None, _) => false,
    };
    let mut staged = None;
    if !by_runs {
        if let Some(CachePages::Runs { unmade, .. }) = &cache.pages {
            counts.runs_refused += 1;
            match unmade.make(device)? {
                Ok((_view, import)) => {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "the IOSurface block at {block:#x}: the platform refused a copy by its \
                         pages' runs; its device cache copies through a view of them from now on"
                    );
                    cache.pages = Some(CachePages::View(import));
                }
                Err(why) => log::warn!(
                    target: "reims_vgpu::graphics", kind:? = why;
                    "the IOSurface block at {block:#x}: the platform refused a copy by its pages' \
                     runs, and no view of them was made ({why:?}); the copy is not made"
                ),
            }
        }
        if let Some(CachePages::View(pages)) = &cache.pages {
            match way {
                // A fill copies all of the surface's bytes: those beside its texels are the
                // cache's own to change.
                Way::In => blit.encode(
                    ON,
                    BlitCmd::CopyBufferToBuffer {
                        source_buffer: Some(pages),
                        source_offset: 0,
                        destination_buffer: Some(&cache.device),
                        destination_offset: 0,
                        size: cache.length,
                    },
                )?,
                Way::Out => {
                    encode_out::<P>(&blit, cache, Some(pages))?;
                }
            }
        }
        if let Some(CachePages::Copied(copied)) = &cache.pages
            && let Some(buffer) = device.new_buffer_of_length(ON, cache.length, STORAGE_SHARED)?
        {
            if way == Way::In {
                // SAFETY: the new buffer's `length` Shared bytes, which no GPU work uses yet;
                // the pages are named, so wired.
                unsafe { copied.copy(buffer.contents().as_ptr::<u8>(), cache.length, false) };
            }
            let (from, to) = match way {
                Way::In => (&buffer, &cache.device),
                Way::Out => (&cache.device, &buffer),
            };
            blit.encode(
                ON,
                BlitCmd::CopyBufferToBuffer {
                    source_buffer: Some(from),
                    source_offset: 0,
                    destination_buffer: Some(to),
                    destination_offset: 0,
                    size: cache.length,
                },
            )?;
            let into = (way == Way::Out).then(|| Arc::clone(copied));
            staged = Some(Staging { buffer, into });
        }
        if cache.pages.is_none()
            && way == Way::In
            && let Some(carried) = &cache.carried
            && let Some(buffer) =
                device.new_buffer_of_length(ON, carried.bytes(), STORAGE_SHARED)?
        {
            let mut from = 0u64;
            for &(at, host, bytes) in &carried.runs {
                // SAFETY: `bytes` of the new buffer's Shared bytes from `from`, which no GPU work
                // uses yet; the carried pages are the kext's, wired until their block's pages are
                // named, which drops them from the cache first.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        host.as_ptr::<u8>(),
                        buffer.contents().as_ptr::<u8>().add(from.to_usize()),
                        bytes.to_usize(),
                    );
                }
                blit.encode(
                    ON,
                    BlitCmd::CopyBufferToBuffer {
                        source_buffer: Some(&buffer),
                        source_offset: from,
                        destination_buffer: Some(&cache.device),
                        destination_offset: at,
                        size: bytes,
                    },
                )?;
                from += bytes;
            }
            staged = Some(Staging { buffer, into: None });
        }
    }
    blit.encode(ON, BlitCmd::EndEncoding)?;
    Ok(staged)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use reims_vgpu_graphics::ops::{
        RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_OP_SUBMITTED, RGFX_OP_SURFACE_USE, RgfxCommandBuffer,
        RgfxOpHeader,
    };
    use reims_vgpu_graphics::wire::{
        RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_QUEUE_CREATE,
        RGFX_PAGE_RUN_ZEROS, RGFX_STATUS_CACHE, RGFX_STATUS_INVALID_ARGUMENT, RGFX_STATUS_OK,
        RGFX_STATUS_REFUSED, RGFX_SURFACE_PAGES_ASK_CACHE, RGFX_SURFACE_PAGES_ATTACH,
        RGFX_SURFACE_PAGES_CACHED, RGFX_SURFACE_PAGES_DEFERRED, RgfxMsgHeader, RgfxObjectDestroy,
        RgfxPageRun, RgfxQueueCreate, RgfxSurfacePages, RgfxSurfacePagesLayout,
    };
    use reims_vgpu_vulkan::core::platform::values::PagesInPlace;
    use reims_vgpu_vulkan::core::testkit::FakePlatform;
    use reims_vgpu_vulkan::core::testkit::calllog::{Arg, Call};
    use reims_vgpu_vulkan::core::testkit::faults::Fault;
    use reims_vgpu_vulkan::core::testkit::objects::CompleteOn;
    use reims_vgpu_vulkan::core::testkit::script::HeldCall;
    use zerocopy::IntoBytes;

    use super::super::surface_pages::tests::{BLOCK, RAM, Rig, rig};
    use super::super::{DecodeError, no_raise};
    use super::*;

    const COPY: &str = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";
    const WAIT: &str = "encodeWaitForEvent:value:";
    const SIGNAL: &str = "encodeSignalEvent:value:";
    /// A command buffer's begin that reaches nothing beyond the uses it names.
    const NO_REACH: Reach = Reach {
        context: None,
        work: false,
        decoded: Vec::new(),
    };

    /// A rig whose block at [`BLOCK`] is two pages of a surface, marked cached, with command
    /// queues 1 and 2.
    fn cached() -> Rig {
        let mut r = rig(8);
        let page = r.page;
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        queues(&mut r);
        r
    }

    /// Command buffer `cb_obj_id` on queue `queue`, which uses each (block, generation, writes)
    /// of `uses` and does nothing else.
    fn use_surfaces(r: &mut Rig, queue: u64, cb_obj_id: u64, uses: &[(u64, u64, bool)]) {
        use_surfaces_asking(r, queue, cb_obj_id, uses, false);
    }

    /// [`use_surfaces`], whose buffer asks for its `CB_SUBMITTED` where it `asks`.
    fn use_surfaces_asking(
        r: &mut Rig,
        queue: u64,
        cb_obj_id: u64,
        uses: &[(u64, u64, bool)],
        asks: bool,
    ) {
        let message = surfaces_message(queue, cb_obj_id, uses, asks);
        assert_eq!(r.dec.decode(&r.device, &message).unwrap(), Ok(()));
    }

    /// The `COMMAND_BUFFER` message of [`use_surfaces_asking`], at seqno `cb_obj_id`.
    fn surfaces_message(
        queue: u64,
        cb_obj_id: u64,
        uses: &[(u64, u64, bool)],
        asks: bool,
    ) -> Vec<u8> {
        let mut ops = Vec::new();
        for &(block_offset, cpu_gen, writes) in uses {
            let body = RgfxOpSurfaceUse {
                block_offset,
                cpu_gen,
                flags: if writes { RGFX_SURFACE_USE_WRITES } else { 0 },
                reserved_14: 0,
            };
            let header = RgfxOpHeader {
                tag: RGFX_OP_SURFACE_USE,
                flags: 0,
                len: (8 + size_of::<RgfxOpSurfaceUse>()) as u32,
            };
            ops.extend_from_slice(header.as_bytes());
            ops.extend_from_slice(body.as_bytes());
        }
        if asks {
            let header = RgfxOpHeader {
                tag: RGFX_OP_SUBMITTED,
                flags: 0,
                len: 8,
            };
            ops.extend_from_slice(header.as_bytes());
        }
        let m = RgfxCommandBuffer {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_COMMAND_BUFFER,
                size: (RGFX_COMMAND_BUFFER_OPS_OFFSET as usize + ops.len()) as u32,
                seqno: cb_obj_id,
            },
            queue_obj_id: queue,
            cb_obj_id,
            op_bytes: ops.len() as u64,
        };
        let mut message = m.as_bytes().to_vec();
        message.extend_from_slice(&ops);
        message
    }

    /// The (event, value) of each call to `sel` in `calls`.
    fn events(calls: &[Call], sel: &str) -> Vec<(u64, u64)> {
        (calls.iter())
            .filter(|c| c.sel == sel)
            .map(|c| {
                let event = c.args[0].as_obj().map_or(0, |o| o.0);
                (event, c.args[1].as_u64().unwrap_or(0))
            })
            .collect()
    }

    /// The (source, destination) of each copy in `calls`.
    fn copies(calls: &[Call]) -> Vec<(u64, u64)> {
        (calls.iter())
            .filter(|c| c.sel == COPY)
            .map(|c| {
                let id = |a: &Arg| a.as_obj().map_or(0, |o| o.0);
                (id(&c.args[0]), id(&c.args[2]))
            })
            .collect()
    }

    /// The id of the block's cache.
    fn cache_id(r: &Rig) -> u64 {
        let caches = r.dec.caches();
        let cache = caches.buffer(BLOCK).unwrap();
        Arg::obj(cache).as_obj().unwrap().0
    }

    /// A block marked cached gets a device cache, which each of its buffers is, on a GPU of its
    /// own memory or one that shares the CPU's alike; a block not marked gets none
    #[test]
    fn a_cached_blocks_buffers_are_its_device_cache() {
        let mut r = cached();
        let page = r.page;
        assert_eq!(r.dec.caches().len(), 1);
        let mark = r.p.call_log().mark();
        assert_eq!(r.buffer(10, BLOCK, 2 * page), Ok(()));
        assert_eq!(r.buffer(11, BLOCK, page), Ok(()));
        assert!(
            r.p.call_log().sels_since(mark).is_empty(),
            "no buffer made: each is the cache"
        );
        assert!(r.dec.span_in_use(BLOCK, 0x1000));
        assert_eq!(r.pages(BLOCK + 0x8000, &[(3, 1)], page), RGFX_STATUS_OK);
        assert_eq!(r.dec.caches().len(), 1, "a block not marked");
        r.dec.set_unified_memory(true);
        assert_eq!(
            r.pages_cached(BLOCK + 0x10000, &[(4, 1)], page, true),
            RGFX_STATUS_OK
        );
        assert_eq!(r.dec.caches().len(), 2, "a GPU of the CPU's memory");
    }

    /// The zero fills in `calls`: each one's buffer, and the range it zeroes.
    fn zeroed(calls: &[Call]) -> Vec<(u64, Arg)> {
        (calls.iter())
            .filter(|c| c.sel == "fillBuffer:range:value:" && c.args[2] == Arg::U8(0))
            .map(|c| (c.args[0].as_obj().map_or(0, |o| o.0), c.args[1].clone()))
            .collect()
    }

    /// A new cache is made as zeros, the whole of it, on the transfer queue, and copies nothing
    /// from the pages at its make: its first fill, as of the CPU's generation in the block's
    /// header, is owed. The first named use of that generation makes it, and waits for it on the
    /// GPU; a use of that generation after it fills nothing more, and one of a newer generation
    /// fills again, once
    #[test]
    fn a_new_cache_is_zeros_until_its_first_use_fills_it() {
        let mut r = rig(8);
        let page = r.page;
        queues(&mut r);
        // The kext's generation in the header, past 0: two read-write locks before the naming.
        assert!(
            r.dec
                .region
                .write(BLOCK + CPU_GEN_OFFSET, &2u64.to_le_bytes())
        );
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        let cache = cache_id(&r);
        let calls = r.p.call_log().since(mark);
        assert!(copies(&calls).is_empty(), "no fill at the make");
        assert_eq!(
            zeroed(&calls),
            [(cache, Arg::val(&NsRange::new(0, 2 * page)))],
            "the whole cache made zeros"
        );
        assert!(
            calls.iter().any(|c| c.sel == "commit"),
            "committed before the naming is answered"
        );
        let counts = r.dec.caches().counts();
        assert_eq!((counts.fills, counts.made_fills), (0, 0));
        // The first named use, of the header's generation: the owed fill, waited for.
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 2, false)]);
        let calls = r.p.call_log().since(mark);
        let filled = copies(&calls);
        assert_eq!(filled.len(), 1, "the first fill");
        assert_eq!(filled[0].1, cache, "into the cache");
        let transfer = events(&calls, SIGNAL)[0];
        assert!(
            events(&calls, WAIT).contains(&transfer),
            "the buffer waits for its fill"
        );
        let counts = r.dec.caches().counts();
        assert_eq!((counts.fills, counts.made_fills), (1, 1));
        // The same generation again fills nothing; a newer one fills again.
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 2, false)]);
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x102, &[(BLOCK, 3, false)]);
        assert_eq!(copies(&r.p.call_log().since(mark)).len(), 1);
        assert_eq!(r.dec.caches().counts().fills, 2);
    }

    /// The answer of a new cache's `SURFACE_PAGES` takes no fill, and the kext's wait for it only
    /// the zeros' commit: a command buffer that uses the surface, issued right after the answer,
    /// waits for the block to settle before it is decoded, then makes the first fill and waits
    /// for it on the GPU, on the transfer event, so it reads the pages, not zeros
    #[test]
    fn a_use_right_after_the_answer_makes_the_first_fill() {
        let mut r = rig(8);
        let page = r.page;
        queues(&mut r);
        let runs = [RgfxPageRun {
            address: RAM + page,
            length: 2 * page,
        }];
        let m = RgfxSurfacePages {
            header: RgfxMsgHeader {
                r#type: reims_vgpu_graphics::wire::RGFX_CMD_SURFACE_PAGES,
                size: size_of::<RgfxSurfacePages>() as u32,
                seqno: 0,
            },
            context: 0,
            run_count: 1,
            block_offset: BLOCK,
            block_size: 0x2000,
            runs_offset: BLOCK + 0x1000,
            length: 2 * page,
        };
        let mark = r.p.call_log().mark();
        let settle = r.dec.begin_surface_pages(BLOCK);
        let status = (r.dec).carry_out_surface_pages(
            &r.device,
            &m,
            &runs,
            RGFX_SURFACE_PAGES_CACHED,
            None,
            &settle,
        );
        assert_eq!(status, Some(RGFX_STATUS_OK), "the answer");
        assert_eq!(r.dec.caches().len(), 1, "the cache made before the answer");
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "no fill before the answer"
        );
        let message = surfaces_message(1, 0x100, &[(BLOCK, 0, false)], false);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            // A lane's decoder, as the context's own messages take.
            let mut dec = r.dec.sender(None, super::super::Sinks::default());
            let device = r.device.clone_ref();
            scope.spawn(move || tx.send(dec.decode(&device, &message).unwrap()).unwrap());
            assert!(
                rx.recv_timeout(std::time::Duration::from_millis(200))
                    .is_err(),
                "the use waits for the block to settle"
            );
            drop(settle);
            assert_eq!(rx.recv().unwrap(), Ok(()), "decoded once the block settled");
        });
        let calls = r.p.call_log().since(mark);
        let filled = copies(&calls);
        assert_eq!(filled.len(), 1, "the first fill alone");
        assert_eq!(filled[0].1, cache_id(&r), "into the cache");
        // The transfer event's signals: the make's zeros, then the fill.
        let signals = events(&calls, SIGNAL);
        let fill = *signals.iter().rfind(|s| s.0 == signals[0].0).unwrap();
        assert_eq!(fill.1, 2, "the fill after the zeros");
        assert!(
            events(&calls, WAIT).contains(&fill),
            "the use waits on the GPU for the first fill"
        );
        let counts = r.dec.caches().counts();
        assert_eq!((counts.fills, counts.made_fills), (1, 1));
    }

    /// The `COMMAND_BUFFER` `cb_obj_id` on queue 1 that names no use and holds an empty blit
    /// encoder: work that may reach a texture over a surface with no use named.
    fn blit_work(cb_obj_id: u64) -> Vec<u8> {
        let mut ops = Vec::new();
        for tag in [
            reims_vgpu_graphics::ops::RGFX_OP_ENCODER_BEGIN_BLIT,
            reims_vgpu_graphics::ops::RGFX_OP_ENCODER_END,
        ] {
            let header = RgfxOpHeader {
                tag,
                flags: 0,
                len: 8,
            };
            ops.extend_from_slice(header.as_bytes());
        }
        let m = RgfxCommandBuffer {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_COMMAND_BUFFER,
                size: (RGFX_COMMAND_BUFFER_OPS_OFFSET as usize + ops.len()) as u32,
                seqno: cb_obj_id,
            },
            queue_obj_id: 1,
            cb_obj_id,
            op_bytes: ops.len() as u64,
        };
        let mut message = m.as_bytes().to_vec();
        message.extend_from_slice(&ops);
        message
    }

    /// A command buffer of blit work that names no use, in the context of the first buffer over a
    /// new cache, makes the cache's first fill and waits for it on the GPU, as its work may reach
    /// a texture over the surface through an argument buffer; one before any buffer over the
    /// block, which no context reaches yet, fills nothing
    #[test]
    fn work_in_the_context_of_a_new_cache_s_buffer_fills_it() {
        let mut r = cached();
        let page = r.page;
        let mark = r.p.call_log().mark();
        assert_eq!(r.dec.decode(&r.device, &blit_work(0x100)).unwrap(), Ok(()));
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "no buffer over it yet"
        );
        assert_eq!(r.buffer(10, BLOCK, 2 * page), Ok(()));
        let mark = r.p.call_log().mark();
        assert_eq!(r.dec.decode(&r.device, &blit_work(0x101)).unwrap(), Ok(()));
        let calls = r.p.call_log().since(mark);
        let filled = copies(&calls);
        assert_eq!(filled.len(), 1, "the first fill");
        assert_eq!(filled[0].1, cache_id(&r), "into the cache");
        let transfer = events(&calls, SIGNAL)[0];
        assert!(
            events(&calls, WAIT).contains(&transfer),
            "the buffer waits for the fill"
        );
        let mark = r.p.call_log().mark();
        assert_eq!(r.dec.decode(&r.device, &blit_work(0x102)).unwrap(), Ok(()));
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "filled once"
        );
        assert_eq!(r.dec.caches().counts().made_fills, 1);
    }

    /// A buffer over a new cache made in another context than the first's makes the cache's
    /// first fill at once, as that context may reach it in ways the first's decodes do not
    /// order; one made again in the first's fills nothing
    #[test]
    fn a_buffer_over_a_new_cache_in_another_context_fills_it() {
        let mut r = cached();
        let page = r.page;
        assert_eq!(r.buffer(10, BLOCK, 2 * page), Ok(()));
        let caches = Arc::clone(&r.dec.surface_caches);
        let mark = r.p.call_log().mark();
        caches.buffer_made(&r.device, BLOCK, None).unwrap();
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "the same context"
        );
        caches.buffer_made(&r.device, BLOCK, Some(7)).unwrap();
        let filled = copies(&r.p.call_log().since(mark));
        assert_eq!(filled.len(), 1, "another context's");
        assert_eq!(filled[0].1, cache_id(&r), "into the cache");
        assert_eq!(r.dec.caches().counts().made_fills, 1);
    }

    /// A use of a newer generation than the cache's, or of any where it holds none (a new cache,
    /// owing its first fill), fills the cache from the pages; one of the generation the cache
    /// holds, or an older one, fills nothing. A buffer behind a fill that has not finished waits
    /// for it
    #[test]
    fn a_use_fills_the_cache_once_a_generation() {
        let mut r = cached();
        let cache = cache_id(&r);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        let calls = r.p.call_log().since(mark);
        let filled = copies(&calls);
        assert_eq!(filled.len(), 1);
        assert_eq!(filled[0].1, cache, "into the cache");
        let transfer = events(&calls, SIGNAL)[0];
        assert_eq!(transfer.1, 2, "after the make's zeros");
        assert!(
            events(&calls, WAIT).contains(&transfer),
            "the buffer waits for its fill"
        );
        for (generation, fills) in [(1, 0), (0, 0), (3, 1), (2, 0), (3, 0)] {
            let mark = r.p.call_log().mark();
            use_surfaces(&mut r, 1, 0x101 + generation, &[(BLOCK, generation, false)]);
            assert_eq!(
                copies(&r.p.call_log().since(mark)).len(),
                fills,
                "generation {generation}"
            );
        }
        assert_eq!(r.dec.caches().counts().fills, 2, "1's, the first, and 3's");
    }

    /// A buffer that asks for its `CB_SUBMITTED` and uses a cache its begin fills at once (no
    /// buffer begun and not finished uses it, no copy of it waits) gets it as the host takes it,
    /// before the fill's copy and its own commit: the fill waits for nothing of the guest's to
    /// come. Once the fill is done, a buffer of the same generation fills nothing and gets it as
    /// it is taken too.
    #[test]
    fn a_buffer_whose_begin_fills_a_cache_is_submitted_before_the_fill() {
        let mut r = cached();
        let at = Arc::new(Mutex::new(Vec::new()));
        let (log, marks) = (Arc::clone(&r.p), Arc::clone(&at));
        r.dec.set_submitted_sink(Arc::new(move |m| {
            marks.lock().push((m.cb_obj_id, log.call_log().mark()));
        }));
        // The first call to `sel` since `from`.
        let first = |r: &Rig, from, sel: &str| {
            (r.p.call_log().since(from).iter())
                .filter(|c| c.sel == sel)
                .map(|c| c.seq)
                .min()
                .unwrap()
        };
        let mark = r.p.call_log().mark();
        use_surfaces_asking(&mut r, 1, 0x100, &[(BLOCK, 1, false)], true);
        assert_eq!(copies(&r.p.call_log().since(mark)).len(), 1, "a fill");
        let (cb, posted) = at.lock()[0];
        assert_eq!(cb, 0x100);
        assert!(posted <= first(&r, mark, COPY), "before the fill");
        assert!(posted <= first(&r, mark, "commit"), "as it is taken");
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        use_surfaces_asking(&mut r, 1, 0x101, &[(BLOCK, 1, false)], true);
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        let (cb, posted) = at.lock()[1];
        assert_eq!(cb, 0x101);
        assert!(posted <= first(&r, mark, "commit"), "as it is taken");
    }

    /// A buffer that asks for its `CB_SUBMITTED` and needs a fill of a cache that a buffer begun
    /// and not finished uses follows the fill, deferred to that buffer's finish
    /// ([`SurfaceCaches::defer_fill`]), which may wait for the guest: it gets its `CB_SUBMITTED`
    /// at its scheduling, after its commit, not as the host takes it.
    #[test]
    fn a_buffer_behind_a_deferred_fill_is_submitted_at_its_scheduling() {
        let mut r = cached();
        let at = Arc::new(Mutex::new(Vec::new()));
        let (log, marks) = (Arc::clone(&r.p), Arc::clone(&at));
        r.dec.set_submitted_sink(Arc::new(move |m| {
            marks.lock().push((m.cb_obj_id, log.call_log().mark()));
        }));
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let queue = r.device.new_command_queue().unwrap();
        let reader = queue.command_buffer(ON).unwrap().unwrap();
        let (_, uses) = (r.dec.surface_caches)
            .before(&r.device, &reader, &reading(BLOCK, 1), &NO_REACH)
            .unwrap();
        let mark = r.p.call_log().mark();
        use_surfaces_asking(&mut r, 1, 0x101, &[(BLOCK, 2, false)], true);
        let calls = r.p.call_log().since(mark);
        assert!(copies(&calls).is_empty(), "the fill deferred");
        let commit = (calls.iter())
            .filter(|c| c.sel == "commit")
            .map(|c| c.seq)
            .max()
            .unwrap();
        drop(uses);
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(r.dec.caches().counts().fills_deferred, 1);
        let (cb, posted) = at.lock()[0];
        assert_eq!(cb, 0x101);
        assert!(posted > commit, "at its scheduling");
    }

    /// A begin answered before its fills holds its uses' blocks from the answer until the fills
    /// are made: another begin over one of them waits meanwhile, so it counts no use the answered
    /// begin's fill, decided at once, would then wait for. It goes on once they are made, and
    /// waits for the fill on the GPU, which is made once.
    #[test]
    fn a_begin_waits_while_another_answered_before_its_fills_makes_them() {
        let r = cached();
        let caches = Arc::clone(&r.dec.surface_caches);
        let queue = r.device.new_command_queue().unwrap();
        let answered = queue.command_buffer(ON).unwrap().unwrap();
        let other = queue.command_buffer(ON).unwrap().unwrap();
        std::thread::scope(|scope| {
            let mut begin = None;
            let mut answer = || {
                begin = Some(scope.spawn(|| {
                    (caches.before(&r.device, &other, &reading(BLOCK, 1), &NO_REACH))
                        .map(|(waits, uses)| (waits, uses.is_some()))
                        .map_err(|_| ())
                }));
                std::thread::sleep(std::time::Duration::from_millis(50));
                assert!(
                    !begin.as_ref().unwrap().is_finished(),
                    "a begin counted a use of a block answered before its fill"
                );
            };
            let (waits, follows, uses) = (caches)
                .before_answering(
                    &r.device,
                    &answered,
                    &reading(BLOCK, 1),
                    &NO_REACH,
                    Some(&mut answer as &mut dyn FnMut()),
                )
                .unwrap();
            assert!(begin.is_some(), "answered");
            assert!(
                waits && !follows && uses.is_some(),
                "waits for its fill alone"
            );
            assert_eq!(
                begin.unwrap().join().unwrap(),
                Ok((true, true)),
                "counted after the fill, waiting for it"
            );
        });
        let counts = r.dec.caches().counts();
        assert_eq!((counts.fills, counts.fills_deferred), (1, 0));
    }

    /// A wait for a cache's fill, as a decode that writes the cache outside a command buffer
    /// takes, lasts while the fill runs (the make's, then a use's) and ends once it is done; a
    /// block with no cache waits for nothing
    #[test]
    fn a_wait_for_a_fill_ends_once_it_is_done() {
        let mut r = cached();
        let short = std::time::Duration::from_millis(1);
        assert!(
            !r.dec.surface_caches.wait_filled(BLOCK, short),
            "the make's fill runs"
        );
        assert!(
            r.dec.surface_caches.wait_filled(BLOCK + 0x8000, short),
            "no cache"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(
            r.dec.surface_caches.wait_filled(BLOCK, short),
            "the make's fill is done"
        );
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        assert!(
            !r.dec.surface_caches.wait_filled(BLOCK, short),
            "the fill runs"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(
            r.dec.surface_caches.wait_filled(BLOCK, short),
            "the fill is done"
        );
    }

    /// A present's fill is committed where the present is taken, which waits for nothing on the
    /// GPU: the present gets the wait for it, done once the fill has finished, which tells the
    /// hook (the host's presents) as it does. A present of a generation the cache holds fills
    /// nothing and waits for nothing
    #[test]
    fn a_present_s_fill_is_waited_for_by_the_present_not_where_it_is_asked() {
        let mut r = cached();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let told = Arc::new(AtomicU64::new(0));
        let tell = Arc::clone(&told);
        r.dec.on_fill_done(Arc::new(move || {
            tell.fetch_add(1, Ordering::SeqCst);
        }));
        let mark = r.p.call_log().mark();
        let caches = Arc::clone(&r.dec.surface_caches);
        let wait = std::thread::scope(|scope| {
            let asked = scope.spawn(|| caches.fill_now(&r.device, BLOCK, 2).unwrap());
            for _ in 0..1000 {
                if asked.is_finished() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert!(
                asked.is_finished(),
                "the present's fill was waited for where it was asked"
            );
            asked.join().unwrap()
        });
        let wait = wait.expect("a wait for the fill");
        assert_eq!(copies(&r.p.call_log().since(mark)).len(), 1, "one fill");
        assert!(!wait.done(), "done before the GPU ran it");
        assert_eq!(told.load(Ordering::SeqCst), 0);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(wait.done());
        assert_eq!(told.load(Ordering::SeqCst), 1, "the hook was told once");
        assert!(
            r.dec
                .surface_caches
                .fill_now(&r.device, BLOCK, 2)
                .unwrap()
                .is_none(),
            "the cache holds the present's generation"
        );
    }

    /// Two queues' buffers of one generation take one fill, whichever the host takes first, and
    /// the other waits for it; a download waits for the latest buffer of each queue that used
    /// the cache, copies the cache into the pages, and is answered once that copy is done
    #[test]
    fn two_queues_take_one_fill_and_a_download_waits_for_both() {
        let mut r = cached();
        let cache = cache_id(&r);
        // Queue 2's buffer is taken first, though queue 1's may have been committed first.
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 2, 0x200, &[(BLOCK, 1, false)]);
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        let calls = r.p.call_log().since(mark);
        assert_eq!(copies(&calls).len(), 1, "one fill");
        let signals = events(&calls, SIGNAL);
        let fill = signals[0];
        assert_eq!(
            events(&calls, WAIT),
            [fill, fill],
            "both buffers wait for the fill"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 1, false)]);
        use_surfaces(&mut r, 2, 0x201, &[(BLOCK, 1, true)]);
        let calls = r.p.call_log().since(mark);
        assert!(copies(&calls).is_empty());
        assert!(events(&calls, WAIT).is_empty(), "the fill is done");
        let mut latest = events(&calls, SIGNAL);
        assert_eq!(latest.len(), 2, "each buffer signals its queue's end");
        let done = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&done);
        let mark = r.p.call_log().mark();
        r.dec.surface_download(
            &r.device,
            BLOCK,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        let calls = r.p.call_log().since(mark);
        let mut waited = events(&calls, WAIT);
        waited.sort_unstable();
        latest.sort_unstable();
        assert_eq!(waited, latest, "after the latest buffer of each queue");
        assert_eq!(copies(&calls)[0].0, cache, "out of the cache");
        assert!(
            !done.load(Ordering::SeqCst),
            "answered once the copy is done"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
    }

    /// A download of a cache nothing wrote since its last copy copies nothing and is answered at
    /// once; a release of a written cache copies it into the pages first, and is done once the
    /// copy is
    #[test]
    fn only_a_written_cache_is_copied_back() {
        let mut r = cached();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        let done = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&done);
        let mark = r.p.call_log().mark();
        r.dec.surface_download(
            &r.device,
            BLOCK,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        assert!(done.load(Ordering::SeqCst), "at once");
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        assert_eq!(r.dec.caches().counts().clean, 1);
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 0, true)]);
        let released = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&released);
        let mark = r.p.call_log().mark();
        r.dec.block_released(
            &r.device,
            BLOCK,
            0,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        assert_eq!(copies(&r.p.call_log().since(mark)).len(), 1);
        assert_eq!(r.dec.caches().len(), 0);
        assert!(!released.load(Ordering::SeqCst));
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(released.load(Ordering::SeqCst));
        assert_eq!(r.dec.caches().counts().downloads, 1);
    }

    /// A release of a written cache whose surface had no client left (its `BLOCK_RELEASE`'s
    /// `RGFX_BLOCK_RELEASE_NO_CLIENT`) still copies the cache into the pages, and is done once
    /// the copy is: a mach port or an XPC object holds such a surface, and the process that
    /// looks it up reads the pages, or fills a new block's cache from them. Dropping the cache
    /// there left the pages as they were before the GPU's write: a Dock icon showed the bytes the
    /// pages held before (another app's icon) until it was drawn again
    #[test]
    fn a_release_with_no_client_left_copies_what_was_written() {
        let mut r = cached();
        let cache = cache_id(&r);
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 0, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let released = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&released);
        let mark = r.p.call_log().mark();
        r.dec.block_released(
            &r.device,
            BLOCK,
            RGFX_BLOCK_RELEASE_NO_CLIENT,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        let copied = copies(&r.p.call_log().since(mark));
        assert_eq!(copied.len(), 1, "the written cache is copied back");
        assert_eq!(
            copied[0].0, cache,
            "out of the device cache, into the pages"
        );
        assert_eq!(r.dec.caches().len(), 0, "the cache goes with the block");
        assert!(
            !released.load(Ordering::SeqCst),
            "the release waits for the copy"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(released.load(Ordering::SeqCst));
        let counts = r.dec.caches().counts();
        assert_eq!((counts.downloads, counts.unread), (1, 1));
    }

    /// A release with no client left of a cache nothing wrote copies nothing, as any release of
    /// one does, and counts nothing
    #[test]
    fn a_release_with_no_client_left_of_an_unwritten_cache_copies_nothing() {
        let mut r = cached();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 0, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let released = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&released);
        let mark = r.p.call_log().mark();
        r.dec.block_released(
            &r.device,
            BLOCK,
            RGFX_BLOCK_RELEASE_NO_CLIENT,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        assert!(released.load(Ordering::SeqCst), "done at once");
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        let counts = r.dec.caches().counts();
        assert_eq!((counts.downloads, counts.unread), (0, 0));
    }

    /// A release of a written cache whose surface nothing but the kext held
    /// (`RGFX_BLOCK_RELEASE_UNHELD`, the header counting no client) drops the cache uncopied and
    /// is answered at once: no process holds the surface to read its pages, which the kext frees
    /// or gives out again only once answered, zero-filled by the kernel. Each control copies: an
    /// old kext's release of the same surface (no client, the bit unset), a release whose bits
    /// the host does not know (as a host that does not know this one), and a release flagged
    /// unheld whose header counts a client, as the kext writes it where a process took the
    /// surface again before the host's copy
    #[test]
    fn an_unheld_release_drops_a_written_cache_uncopied() {
        use reims_vgpu_graphics::wire::RGFX_BLOCK_RELEASE_UNHELD;
        let release_written = |flags: u32, clients: u32| {
            let mut r = cached();
            let cache = cache_id(&r);
            assert!(
                r.dec
                    .region
                    .write(BLOCK + CLIENTS_OFFSET, &clients.to_le_bytes())
            );
            use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 0, true)]);
            r.p.world().complete_all(CompleteOn::Caller);
            let released = Arc::new(AtomicBool::new(false));
            let set = Arc::clone(&released);
            let mark = r.p.call_log().mark();
            r.dec.block_released(
                &r.device,
                BLOCK,
                flags,
                Box::new(move || set.store(true, Ordering::SeqCst)),
                no_raise(),
            );
            let copied = copies(&r.p.call_log().since(mark));
            assert!(copied.iter().all(|&(from, _)| from == cache));
            let at_once = released.load(Ordering::SeqCst);
            r.p.world().complete_all(CompleteOn::Caller);
            assert!(released.load(Ordering::SeqCst), "answered");
            assert_eq!(r.dec.caches().len(), 0, "the cache goes with the block");
            (copied.len(), at_once, r.dec.caches().counts())
        };
        let unheld = RGFX_BLOCK_RELEASE_NO_CLIENT | RGFX_BLOCK_RELEASE_UNHELD;
        let (copied, at_once, counts) = release_written(unheld, 0);
        assert_eq!(
            (copied, at_once),
            (0, true),
            "dropped uncopied, answered at once"
        );
        assert_eq!((counts.unheld, counts.downloads, counts.retaken), (1, 0, 0));
        assert_eq!(counts.unheld_bytes, 2 * cached().page, "the cache's bytes");
        assert_eq!(counts.unheld_released, 1);
        let (copied, at_once, counts) = release_written(RGFX_BLOCK_RELEASE_NO_CLIENT, 0);
        assert_eq!((copied, at_once), (1, false), "an old kext's: copied");
        assert_eq!((counts.unheld, counts.downloads, counts.unread), (0, 1, 1));
        assert_eq!(counts.unheld_released, 0);
        let (copied, _, counts) = release_written(RGFX_BLOCK_RELEASE_NO_CLIENT | 1 << 31, 0);
        assert_eq!(
            (copied, counts.unheld),
            (1, 0),
            "a bit the host does not know: copied"
        );
        let (copied, at_once, counts) = release_written(unheld, 1);
        assert_eq!((copied, at_once), (1, false), "taken again: copied");
        assert_eq!((counts.unheld, counts.retaken, counts.downloads), (0, 1, 1));
        assert_eq!(
            counts.unheld_released, 1,
            "counted among the unheld all the same"
        );
    }

    /// Every release the kext flags unheld is counted, as the kext counts them, so the two logs'
    /// counts of them agree: of a cache nothing wrote, which drops nothing, and of a deferred
    /// block's written cache whose pages were never named, which drops the GPU's write uncopied as
    /// an attached cache's is, and counts among the drops and their bytes
    #[test]
    fn every_unheld_release_is_counted_and_every_written_one_dropped() {
        use reims_vgpu_graphics::wire::RGFX_BLOCK_RELEASE_UNHELD;
        let unheld = RGFX_BLOCK_RELEASE_NO_CLIENT | RGFX_BLOCK_RELEASE_UNHELD;
        let released = |r: &Rig| {
            let done = Arc::new(AtomicBool::new(false));
            let set = Arc::clone(&done);
            r.dec.block_released(
                &r.device,
                BLOCK,
                unheld,
                Box::new(move || set.store(true, Ordering::SeqCst)),
                no_raise(),
            );
            done
        };
        let mut r = cached();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 0, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(released(&r).load(Ordering::SeqCst));
        let counts = r.dec.caches().counts();
        assert_eq!(
            (counts.unheld_released, counts.unheld, counts.downloads),
            (1, 0, 0),
            "unwritten: counted, nothing dropped"
        );

        let mut r = deferring(8);
        let page = r.page;
        assert_eq!(defer(&mut r, BLOCK, 2 * page), RGFX_STATUS_OK);
        assert!(
            r.dec
                .region
                .write(BLOCK + CLIENTS_OFFSET, &0u32.to_le_bytes())
        );
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        assert!(released(&r).load(Ordering::SeqCst), "at once");
        assert!(copies(&r.p.call_log().since(mark)).is_empty(), "no pages");
        let counts = r.dec.caches().counts();
        assert_eq!(
            (counts.unheld_released, counts.unheld, counts.unattached),
            (1, 1, 1),
            "a written cache never named: dropped uncopied"
        );
        assert_eq!(counts.unheld_bytes, 2 * page);
    }

    /// A download asked while the cache's last copy into its pages is still in flight on the
    /// transfer queue (the kext stopped waiting for its answer, and a lock asks again) is answered
    /// once that copy lands, not at once over pages the copy has not reached; so is a release of
    /// the block then. A buffer that writes the cache meanwhile waits for the copy on the GPU, so
    /// the copy takes the cache as it was asked
    #[test]
    fn a_download_or_release_behind_a_copy_in_flight_waits_for_it() {
        for second in ["download", "release"] {
            let mut r = cached();
            use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
            r.p.world().complete_all(CompleteOn::Caller);
            let mark = r.p.call_log().mark();
            let first = download(&r, BLOCK);
            let calls = r.p.call_log().since(mark);
            assert_eq!(copies(&calls).len(), 1, "the first copies");
            let signalled = events(&calls, SIGNAL);
            assert_eq!(signalled.len(), 1, "its copy signals the transfer event");
            assert!(!first.load(Ordering::SeqCst), "in flight");

            let mark = r.p.call_log().mark();
            let then = if second == "download" {
                download(&r, BLOCK)
            } else {
                release(&r, BLOCK)
            };
            assert!(
                copies(&r.p.call_log().since(mark)).is_empty(),
                "{second}: nothing new to copy"
            );
            assert!(
                !then.load(Ordering::SeqCst),
                "{second}: waits for the copy in flight"
            );
            if second == "download" {
                let mark = r.p.call_log().mark();
                use_surfaces(&mut r, 2, 0x200, &[(BLOCK, 0, true)]);
                assert!(
                    events(&r.p.call_log().since(mark), WAIT).contains(&signalled[0]),
                    "a writer waits for the copy on the GPU"
                );
            }
            r.p.world().complete_all(CompleteOn::Caller);
            assert!(first.load(Ordering::SeqCst) && then.load(Ordering::SeqCst));
        }
    }

    /// A download or a release answered [`SLOW_ANSWER`](super::super::SLOW_ANSWER) or more after
    /// it was asked, its copy into the pages that long in flight, is counted as slow (and logged,
    /// with where it waited); one answered sooner is not
    #[test]
    fn a_download_or_release_answered_late_is_counted_slow() {
        let mut r = cached();
        let late = super::super::SLOW_ANSWER;
        for (n, second) in ["download", "release"].into_iter().enumerate() {
            use_surfaces(&mut r, 1, 0x100 + n as u64, &[(BLOCK, 0, true)]);
            r.p.world().complete_all(CompleteOn::Caller);
            let soon = download(&r, BLOCK);
            r.p.world().complete_all(CompleteOn::Caller);
            assert!(soon.load(Ordering::SeqCst));
            assert_eq!(r.dec.slow_answers(), n as u64, "{second}: answered at once");

            // The writer, and the copy of its write into the pages, in flight.
            use_surfaces(&mut r, 1, 0x200 + n as u64, &[(BLOCK, 0, true)]);
            let then = if second == "download" {
                download(&r, BLOCK)
            } else {
                release(&r, BLOCK)
            };
            assert!(!then.load(Ordering::SeqCst), "{second}: its copy in flight");
            r.p.virtual_clock().advance(late);
            r.p.world().complete_all(CompleteOn::Caller);
            assert!(then.load(Ordering::SeqCst));
            assert_eq!(
                r.dec.slow_answers(),
                n as u64 + 1,
                "{second}: answered late"
            );
        }
    }

    /// A slow answer's copy phase is split by its copy's GPU times: the GPU's run of the copy,
    /// and the rest before it (the GPU's wait for the command buffers that used the cache, and the
    /// transfer queue's work before it), with the cache's planes and bytes, how the copy reaches
    /// the pages, and the queues it waited for; with no GPU times from the platform, the phase
    /// stays whole
    #[test]
    fn a_slow_answer_s_copy_phase_is_split_by_its_copy_s_gpu_times() {
        let late = super::super::SLOW_ANSWER;
        let transfer = std::time::Duration::from_millis(30);
        for timed in [true, false] {
            let mut r = cached();
            let length = r.dec.caches().length(BLOCK).unwrap();
            use_surfaces(&mut r, 1, 0x200, &[(BLOCK, 0, true)]);
            let mark = r.p.call_log().mark();
            let answered = download(&r, BLOCK);
            let calls = r.p.call_log().since(mark);
            let commits: Vec<&Call> = calls.iter().filter(|c| c.sel == "commit").collect();
            assert_eq!(commits.len(), 1, "its own copy, on the transfer queue");
            if timed {
                let start = 7_000_000_000;
                let end = start + transfer.as_nanos() as u64;
                assert!(r.p.world().set_gpu_times(commits[0].obj, start, end));
            }
            r.p.virtual_clock().advance(late);
            r.p.world().complete_all(CompleteOn::Caller);
            assert!(answered.load(Ordering::SeqCst));
            let split = r.dec.last_slow_copy().expect("answered late");
            assert_eq!(split.cache, Some((0, length)), "no layout named its planes");
            assert_eq!(
                (split.users, split.copies),
                (Some(1), 1),
                "after queue 1's buffer"
            );
            assert_eq!(split.staged, std::time::Duration::ZERO, "its pages a view");
            match timed {
                true => {
                    assert_eq!(split.transfer, Some(transfer));
                    assert_eq!(split.wait, Some(late - transfer));
                }
                false => assert_eq!((split.transfer, split.wait), (None, None)),
            }
        }
    }

    /// A `SURFACE_PAGES` of a new block over the pages of a released block whose cache is still
    /// to be copied into them (its release waits for a buffer still walked that writes the
    /// cache, as when the kext stopped waiting for its answer and made the new block anyway)
    /// waits for that copy to land before it is carried out: the new block's buffers lie over the
    /// pages, and the copy would land over what they write
    #[test]
    fn a_surface_pages_over_pages_a_release_still_copies_into_waits_for_it() {
        let mut r = cached();
        let page = r.page;
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let queue = r.device.new_command_queue().unwrap();
        let writer = queue.command_buffer(ON).unwrap().unwrap();
        let writes = [RgfxOpSurfaceUse {
            flags: RGFX_SURFACE_USE_WRITES,
            ..reading(BLOCK, 0)[0]
        }];
        let (_, uses) = (r.dec.surface_caches)
            .before(&r.device, &writer, &writes, &NO_REACH)
            .unwrap();
        let released = release(&r, BLOCK);
        assert!(!released.load(Ordering::SeqCst), "behind the writer");
        let other = BLOCK + 0x4000;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            // A decoder of its own, as a lane's is.
            let dec = r.dec.sender(None, super::super::Sinks::default());
            let device = r.device.clone_ref();
            scope.spawn(move || {
                let runs = [RgfxPageRun {
                    address: RAM + page,
                    length: 2 * page,
                }];
                let m = RgfxSurfacePages {
                    header: RgfxMsgHeader {
                        r#type: reims_vgpu_graphics::wire::RGFX_CMD_SURFACE_PAGES,
                        size: size_of::<RgfxSurfacePages>() as u32,
                        seqno: 0,
                    },
                    context: 0,
                    run_count: 1,
                    block_offset: other,
                    block_size: 0x2000,
                    runs_offset: other + 0x1000,
                    length: 2 * page,
                };
                tx.send(dec.surface_pages(&device, &m, &runs, 0)).unwrap();
            });
            assert!(
                rx.recv_timeout(std::time::Duration::from_millis(200))
                    .is_err(),
                "waits for the release's copy"
            );
            (r.dec.surface_caches)
                .finish(&r.device, 1, None, &writer, uses.unwrap(), || {
                    writer.commit(ON)
                })
                .unwrap();
            assert!(!released.load(Ordering::SeqCst), "copied, in flight");
            assert!(
                rx.recv_timeout(std::time::Duration::from_millis(200))
                    .is_err(),
                "waits for the copy to land"
            );
            r.p.world().complete_all(CompleteOn::Caller);
            assert!(released.load(Ordering::SeqCst));
            assert_eq!(
                rx.recv_timeout(std::time::Duration::from_secs(5)),
                Ok(RGFX_STATUS_OK)
            );
        });
    }

    /// A release flagged unheld behind a writer still running is decided as its copy is made, at
    /// the writer's finish: a process that took the surface again meanwhile (the kext counting
    /// it in the header) has the cache copied into the pages before the answer, which its lock
    /// waits for, so it reads the GPU's write; with none taken, nothing is copied
    #[test]
    fn an_unheld_release_copies_where_the_surface_is_taken_again_before_its_copy() {
        use reims_vgpu_graphics::wire::RGFX_BLOCK_RELEASE_UNHELD;
        for taken in [true, false] {
            let mut r = cached();
            let cache = cache_id(&r);
            use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
            r.p.world().complete_all(CompleteOn::Caller);
            let queue = r.device.new_command_queue().unwrap();
            let writer = queue.command_buffer(ON).unwrap().unwrap();
            let writes = [RgfxOpSurfaceUse {
                flags: RGFX_SURFACE_USE_WRITES,
                ..reading(BLOCK, 0)[0]
            }];
            let (_, uses) = (r.dec.surface_caches)
                .before(&r.device, &writer, &writes, &NO_REACH)
                .unwrap();
            let released = Arc::new(AtomicBool::new(false));
            let set = Arc::clone(&released);
            let mark = r.p.call_log().mark();
            r.dec.block_released(
                &r.device,
                BLOCK,
                RGFX_BLOCK_RELEASE_NO_CLIENT | RGFX_BLOCK_RELEASE_UNHELD,
                Box::new(move || set.store(true, Ordering::SeqCst)),
                no_raise(),
            );
            assert!(!released.load(Ordering::SeqCst), "behind the writer");
            if taken {
                assert!(
                    r.dec
                        .region
                        .write(BLOCK + CLIENTS_OFFSET, &1u32.to_le_bytes())
                );
            }
            (r.dec.surface_caches)
                .finish(&r.device, 1, None, &writer, uses.unwrap(), || {
                    writer.commit(ON)
                })
                .unwrap();
            r.p.world().complete_all(CompleteOn::Caller);
            assert!(released.load(Ordering::SeqCst), "answered");
            let copied = copies(&r.p.call_log().since(mark));
            assert_eq!(
                copied.len(),
                usize::from(taken),
                "copied only where taken again"
            );
            assert!(
                copied.iter().all(|&(from, _)| from == cache),
                "out of the cache"
            );
        }
    }

    /// The header's count of a surface's clients lies where the kext writes it
    #[test]
    fn the_clients_offset_is_the_kext_s() {
        let abi = include_str!("../../../../../guest/kext/src/RgfxUserClientABI.hpp");
        let line = format!("kRgfxSurfaceClientsOffset = {CLIENTS_OFFSET},");
        assert!(abi.contains(&line), "the kext's header has no `{line}`");
    }

    /// A block named again over the same pages keeps its cache, filled from the pages again as it
    /// is named, as of the generation in the block's header, once what a buffer wrote of the cache
    /// went into them: the next use of that generation fills nothing more, and one of a newer
    /// generation fills again. One named over other pages, or not cached, loses it, what a buffer
    /// wrote of it going into its pages first, and the new cache over other pages owes its first
    /// fill, which its first use makes. Nothing else is copied
    #[test]
    fn a_block_named_again_keeps_its_cache_filled_from_its_pages() {
        let mut r = cached();
        let page = r.page;
        let cache = cache_id(&r);
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        // The kext's generation in the header as the block is named again: the use's.
        assert!(
            r.dec
                .region
                .write(BLOCK + CPU_GEN_OFFSET, &1u64.to_le_bytes())
        );
        for (round, out) in [(0, 1), (1, 0)] {
            let mark = r.p.call_log().mark();
            assert_eq!(
                r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
                RGFX_STATUS_OK
            );
            let copied = copies(&r.p.call_log().since(mark));
            assert_eq!(copied.len(), out + 1, "round {round}");
            assert!(
                copied[..out].iter().all(|&(from, _)| from == cache),
                "out of the cache first, round {round}"
            );
            assert_eq!(copied[out].1, cache, "then filled, round {round}");
            assert_eq!(cache_id(&r), cache, "kept, round {round}");
            // The next naming waits for the copies into the pages to land.
            r.p.world().complete_all(CompleteOn::Caller);
        }
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 1, false)]);
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "the generation the naming filled"
        );
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x102, &[(BLOCK, 2, true)]);
        let filled = copies(&r.p.call_log().since(mark));
        assert_eq!(filled.len(), 1, "a newer generation fills again");
        assert_eq!(filled[0].1, cache);
        r.p.world().complete_all(CompleteOn::Caller);

        let mark = r.p.call_log().mark();
        assert_eq!(
            r.pages_cached(BLOCK, &[(3, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        let back = copies(&r.p.call_log().since(mark));
        let other = cache_id(&r);
        assert_ne!(other, cache, "a new cache");
        assert_eq!(
            back.len(),
            1,
            "the old cache copied out, the new one owing its first fill"
        );
        assert_eq!(back[0].0, cache, "the old cache, into its pages");
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x103, &[(BLOCK, 1, true)]);
        let filled = copies(&r.p.call_log().since(mark));
        assert_eq!(
            filled.len(),
            1,
            "the new cache's first fill, at its first use"
        );
        assert_eq!(filled[0].1, other, "the new cache, filled from its pages");
        r.p.world().complete_all(CompleteOn::Caller);

        let mark = r.p.call_log().mark();
        assert_eq!(r.pages(BLOCK, &[(5, 1)], page), RGFX_STATUS_OK);
        let back = copies(&r.p.call_log().since(mark));
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].0, other);
        assert_eq!(r.dec.caches().len(), 0, "not cached");
        r.p.world().complete_all(CompleteOn::Caller);
        let counts = r.dec.caches().counts();
        assert_eq!(
            (counts.made, counts.made_fills, counts.downloads),
            (2, 4, 3),
            "two made, each filled at its first use, and filled at the two namings; three \
             copies out"
        );
    }

    /// A block named again over the same pages while a command buffer begun and not finished uses
    /// its cache is filled in that buffer's finish, as a use's fill is: the buffer reads what the
    /// cache held
    #[test]
    fn a_block_named_again_behind_a_walk_is_filled_in_its_finish() {
        let mut r = cached();
        let page = r.page;
        let cache = cache_id(&r);
        r.p.world().complete_all(CompleteOn::Caller);
        let queue = r.device.new_command_queue().unwrap();
        let reader = queue.command_buffer(ON).unwrap().unwrap();
        let (_, uses) = (r.dec.surface_caches)
            .before(&r.device, &reader, &reading(BLOCK, 0), &NO_REACH)
            .unwrap();
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "filled under the walk"
        );
        drop(uses);
        let filled = copies(&r.p.call_log().since(mark));
        assert_eq!(filled.len(), 1, "made in the finish");
        assert_eq!(filled[0].1, cache);
        assert_eq!(r.dec.caches().counts().cpu_waits, 0);
    }

    /// A use of `block` at generation `cpu_gen` that only reads it.
    fn reading(block_offset: u64, cpu_gen: u64) -> [RgfxOpSurfaceUse; 1] {
        [RgfxOpSurfaceUse {
            block_offset,
            cpu_gen,
            flags: 0,
            reserved_14: 0,
        }]
    }

    /// The transfer queue is made with the first command queue, once, and unshared: a present of
    /// a surface waits for its fill. The read-back queue is made with it, plain. The fills make no
    /// queue
    #[test]
    fn the_transfer_queue_is_made_unshared_with_the_first_queue() {
        use reims_vgpu_vulkan::core::testkit::impls::device::UNSHARED_QUEUE;
        let mut r = cached();
        let (unshared, plain) = (
            r.p.call_log().count(UNSHARED_QUEUE),
            r.p.call_log().count("newCommandQueue"),
        );
        assert_eq!(
            (unshared, plain),
            (1, 3),
            "made with queue 1, beside queues 1 and 2 and the read-back queue"
        );
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        assert_eq!(copies(&r.p.call_log().since(mark)).len(), 1, "a fill");
        r.p.world().complete_all(CompleteOn::Caller);
        use_surfaces(&mut r, 2, 0x101, &[(BLOCK, 2, false)]);
        assert_eq!(
            copies(&r.p.call_log().since(mark)).len(),
            2,
            "a second fill"
        );
        assert_eq!(r.p.call_log().count(UNSHARED_QUEUE), unshared);
        assert_eq!(r.p.call_log().count("newCommandQueue"), plain);
    }

    /// A queue's end event is made with the queue, so its first command buffer that uses a cache
    /// makes no event as it fills the cache and signals the end event; at the queue's destroy, the
    /// end event goes where no cache names the queue, and stays where one does, for the cache's
    /// next copy, which waits for the queue's last buffer that used it
    #[test]
    fn a_queues_end_event_is_made_with_it_and_its_first_buffer_makes_none() {
        let mut r = cached();
        let end = |r: &Rig, queue: u64| {
            (r.dec.caches().ends.get(&queue)).map(|(event, _)| Arg::obj(event).as_obj().unwrap().0)
        };
        let one = end(&r, 1).expect("queue 1's end event, made with it");
        assert!(end(&r, 2).is_some(), "queue 2's");
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        let calls = r.p.call_log().since(mark);
        assert_eq!(copies(&calls).len(), 1, "a fill");
        assert!(
            events(&calls, SIGNAL).contains(&(one, 1)),
            "its end event signalled"
        );
        assert!(
            !calls
                .iter()
                .any(|c| c.sel == "newEvent" || c.sel == "newSharedEvent"),
            "no event made at the buffer"
        );
        for queue in [1, 2] {
            let destroy = RgfxObjectDestroy {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_OBJECT_DESTROY,
                    size: size_of::<RgfxObjectDestroy>() as u32,
                    seqno: 0,
                },
                obj_id: queue,
            };
            assert_eq!(r.dec.decode(&r.device, destroy.as_bytes()).unwrap(), Ok(()));
        }
        assert_eq!(end(&r, 1), Some(one), "signalled: kept");
        assert_eq!(end(&r, 2), None, "never signalled: let go of");
    }

    /// A gone queue's end event goes once no cache names it: at the queue's end where every
    /// cache it used was copied since, else at a later queue's make or end, once the cache's copy
    /// or its release names it no more. Kept until the device's reset, one piled up for each
    /// queue that ever used a cache, as processes came and went
    #[test]
    fn a_gone_queues_end_event_goes_once_no_cache_names_it() {
        let mut r = cached();
        let has_end = |r: &Rig, queue: u64| r.dec.caches().ends.contains_key(&queue);
        let destroy = |r: &mut Rig, obj_id: u64| {
            let destroy = RgfxObjectDestroy {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_OBJECT_DESTROY,
                    size: size_of::<RgfxObjectDestroy>() as u32,
                    seqno: 0,
                },
                obj_id,
            };
            assert_eq!(r.dec.decode(&r.device, destroy.as_bytes()).unwrap(), Ok(()));
        };
        // Queue 1 writes the cache, which a download then copies: no cache names queue 1.
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let downloaded = download(&r, BLOCK);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
        destroy(&mut r, 1);
        assert!(!has_end(&r, 1), "named by no cache: let go of at its end");
        // Queue 2 reads the cache, which names it still at its end, for its next copy.
        use_surfaces(&mut r, 2, 0x101, &[(BLOCK, 0, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        destroy(&mut r, 2);
        assert!(has_end(&r, 2), "named by the cache: kept");
        // The block's release lets the cache go; the next queue's make looks again.
        let released = release(&r, BLOCK);
        assert!(released.load(Ordering::SeqCst));
        assert!(has_end(&r, 2), "not looked at again yet");
        let queue = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 0,
            },
            obj_id: 3,
            ..RgfxQueueCreate::default()
        };
        assert_eq!(r.dec.decode(&r.device, queue.as_bytes()).unwrap(), Ok(()));
        assert!(!has_end(&r, 2), "let go of at queue 3's make");
        assert!(has_end(&r, 3), "queue 3's own, made with it");
    }

    /// The id of the block's cache's event, which the command buffers that follow a deferred copy
    /// of it wait on.
    fn event_id(r: &Rig) -> u64 {
        let caches = r.dec.caches();
        let event = caches.by_block[&BLOCK]
            .event
            .as_ref()
            .expect("the cache's event");
        Arg::obj(event).as_obj().unwrap().0
    }

    /// A fill of a cache follows every command buffer begun and not finished that uses it, and
    /// is made once the last of them has finished, as one made after their whole use under one
    /// lock would be (docs/graphics-kext/design.md, "What a walk sees"), in that buffer's finish:
    /// the buffer that asked it is begun at once and waits for it on the GPU, on the cache's
    /// event, which the fill's command buffer signals.
    #[test]
    fn a_fill_follows_the_buffers_using_the_cache() {
        let mut r = cached();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let queue = r.device.new_command_queue().unwrap();
        let reader = queue.command_buffer(ON).unwrap().unwrap();
        let (_, uses) = (r.dec.surface_caches)
            .before(&r.device, &reader, &reading(BLOCK, 1), &NO_REACH)
            .unwrap();
        assert!(uses.is_some(), "counted among its users");
        let filler = queue.command_buffer(ON).unwrap().unwrap();
        let mark = r.p.call_log().mark();
        let (waits, filling) = (r.dec.surface_caches)
            .before(&r.device, &filler, &reading(BLOCK, 2), &NO_REACH)
            .unwrap();
        assert!(waits, "waits for the fill");
        assert!(filling.is_some());
        let event = event_id(&r);
        let calls = r.p.call_log().since(mark);
        assert!(
            copies(&calls).is_empty(),
            "filled under a buffer using the cache"
        );
        assert_eq!(events(&calls, WAIT), [(event, 1)], "the fill, on the GPU");
        assert_eq!(r.dec.caches().counts().cpu_waits, 0);
        let mark = r.p.call_log().mark();
        drop(uses);
        let calls = r.p.call_log().since(mark);
        assert_eq!(copies(&calls).len(), 1, "filled once its user finished");
        assert!(
            events(&calls, SIGNAL).contains(&(event, 1)),
            "the fill signals the buffers that follow it"
        );
        assert_eq!(r.dec.caches().counts().fills_deferred, 1);
    }

    /// The wait the ledger measured (R048, R197b): a buffer of one queue whose walk is long holds
    /// a cache, and another queue's buffer that needs the cache filled is begun. That begin waits
    /// on the CPU for nothing, whether the long walk reads the cache or writes it: its fill is
    /// asked, the buffer waits for it on the GPU, and the fill is made in the long walk's finish,
    /// after its commit, waiting on the GPU for its end. A third buffer of the newer generation,
    /// begun meanwhile, takes the same fill.
    #[test]
    fn a_begin_waits_on_the_cpu_for_no_other_queue_s_long_walk() {
        for writes in [false, true] {
            let mut r = cached();
            use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
            r.p.world().complete_all(CompleteOn::Caller);
            let (long_queue, other_queue) = (
                r.device.new_command_queue().unwrap(),
                r.device.new_command_queue().unwrap(),
            );
            let long = long_queue.command_buffer(ON).unwrap().unwrap();
            let use_of = [RgfxOpSurfaceUse {
                flags: if writes { RGFX_SURFACE_USE_WRITES } else { 0 },
                ..reading(BLOCK, 1)[0]
            }];
            let (_, walking) = (r.dec.surface_caches)
                .before(&r.device, &long, &use_of, &NO_REACH)
                .unwrap();
            let caches = Arc::clone(&r.dec.surface_caches);
            let (asking, taking) = (
                other_queue.command_buffer(ON).unwrap().unwrap(),
                other_queue.command_buffer(ON).unwrap().unwrap(),
            );
            let mark = r.p.call_log().mark();
            let (begun_at_once, asked, taken) = std::thread::scope(|scope| {
                let begin = scope.spawn(|| {
                    let asked = caches.before(&r.device, &asking, &reading(BLOCK, 2), &NO_REACH);
                    let taken = caches.before(&r.device, &taking, &reading(BLOCK, 2), &NO_REACH);
                    (asked.map_err(|_| ()), taken.map_err(|_| ()))
                });
                let mut at_once = false;
                for _ in 0..1000 {
                    if begin.is_finished() {
                        at_once = true;
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                // The long walk ends: its commit, then its finish.
                (r.dec.surface_caches)
                    .finish(&r.device, 1, None, &long, walking.unwrap(), || {
                        long.commit(ON)
                    })
                    .unwrap();
                let (asked, taken) = begin.join().unwrap();
                (at_once, asked.unwrap(), taken.unwrap())
            });
            let counts = r.dec.caches().counts();
            assert!(
                begun_at_once,
                "a begin waited for another queue's walk ({writes})"
            );
            assert_eq!(counts.cpu_waits, 0, "waits on the CPU ({writes})");
            assert!(asked.0 && taken.0, "both wait for the fill on the GPU");
            assert_eq!(counts.fills_deferred, 1, "one fill for both");
            let event = event_id(&r);
            let calls = r.p.call_log().since(mark);
            assert_eq!(
                events(&calls, WAIT)
                    .iter()
                    .filter(|&&e| e == (event, 1))
                    .count(),
                2,
                "both buffers wait for the fill"
            );
            // The fill comes after the long walk's commit, and waits for its end on the GPU.
            let commit = calls.iter().position(|c| c.sel == "commit").unwrap();
            let copied = calls.iter().position(|c| c.sel == COPY).unwrap();
            assert!(commit < copied, "filled before the long walk's commit");
            let end = events(&calls, SIGNAL)[0];
            assert!(
                events(&calls[commit..], WAIT).contains(&end),
                "the fill waits for the long walk's end"
            );
            drop((asked, taken));
        }
    }

    /// A present's fill of a cache a command buffer begun and not finished uses is asked where
    /// the present is, waiting for nothing there: the present gets a wait that is done once the
    /// fill, made in that buffer's finish, has finished, and the hook is told then
    #[test]
    fn a_present_s_fill_behind_a_walk_is_made_in_its_finish() {
        let mut r = cached();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let told = Arc::new(AtomicU64::new(0));
        let tell = Arc::clone(&told);
        r.dec.on_fill_done(Arc::new(move || {
            tell.fetch_add(1, Ordering::SeqCst);
        }));
        let queue = r.device.new_command_queue().unwrap();
        let reader = queue.command_buffer(ON).unwrap().unwrap();
        let (_, uses) = (r.dec.surface_caches)
            .before(&r.device, &reader, &reading(BLOCK, 1), &NO_REACH)
            .unwrap();
        let mark = r.p.call_log().mark();
        let wait = (r.dec.surface_caches)
            .fill_now(&r.device, BLOCK, 2)
            .unwrap()
            .expect("a wait for the fill");
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        assert!(!wait.done(), "done before its fill was made");
        drop(uses);
        assert_eq!(
            copies(&r.p.call_log().since(mark)).len(),
            1,
            "made in the finish"
        );
        assert!(!wait.done(), "done before the GPU ran it");
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(wait.done());
        assert_eq!(told.load(Ordering::SeqCst), 1, "the hook was told once");
        assert_eq!(r.dec.caches().counts().cpu_waits, 0);
    }

    /// A buffer begun while a download waits for a writer of the cache waits for nothing where it
    /// only reads the cache, on the CPU or on the GPU; one whose use may write it waits for the
    /// download on the GPU, so the download copies the first writer's write and not its own, and
    /// its write is still the cache's for the next download, which copies it
    #[test]
    fn a_begin_behind_a_download_waits_on_the_gpu_only_where_it_writes() {
        let mut r = cached();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let queue = r.device.new_command_queue().unwrap();
        let writes = |generation| {
            [RgfxOpSurfaceUse {
                flags: RGFX_SURFACE_USE_WRITES,
                ..reading(BLOCK, generation)[0]
            }]
        };
        let first = queue.command_buffer(ON).unwrap().unwrap();
        let (_, first_uses) = (r.dec.surface_caches)
            .before(&r.device, &first, &writes(1), &NO_REACH)
            .unwrap();
        let downloaded = download(&r, BLOCK);
        let (reader, writer) = (
            queue.command_buffer(ON).unwrap().unwrap(),
            queue.command_buffer(ON).unwrap().unwrap(),
        );
        let mark = r.p.call_log().mark();
        let (reads, read_uses) = (r.dec.surface_caches)
            .before(&r.device, &reader, &reading(BLOCK, 1), &NO_REACH)
            .unwrap();
        assert!(!reads, "a reader waits for the download");
        assert!(events(&r.p.call_log().since(mark), WAIT).is_empty());
        let (written, write_uses) = (r.dec.surface_caches)
            .before(&r.device, &writer, &writes(1), &NO_REACH)
            .unwrap();
        assert!(written, "a writer waits for the download");
        let event = event_id(&r);
        assert_eq!(events(&r.p.call_log().since(mark), WAIT), [(event, 1)]);
        assert_eq!(r.dec.caches().counts().cpu_waits, 0);
        // The first writer ends: the download is made, signalling the writer behind it.
        let mark = r.p.call_log().mark();
        (r.dec.surface_caches)
            .finish(&r.device, 1, None, &first, first_uses.unwrap(), || {
                first.commit(ON)
            })
            .unwrap();
        let calls = r.p.call_log().since(mark);
        assert_eq!(copies(&calls).len(), 1, "the download's copy");
        assert!(events(&calls, SIGNAL).contains(&(event, 1)));
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
        // The second writer's write was kept apart from the download, and reaches the pages: by
        // an early copy behind it (the surface reads back since the download) or by the next
        // download.
        let mark = r.p.call_log().mark();
        drop(read_uses);
        (r.dec.surface_caches)
            .finish(&r.device, 1, None, &writer, write_uses.unwrap(), || {
                writer.commit(ON)
            })
            .unwrap();
        let again = download(&r, BLOCK);
        let copied = copies(&r.p.call_log().since(mark));
        assert_eq!(copied.len(), 1, "the second write copied out once");
        assert_eq!(copied[0].0, cache_id(&r), "out of the cache");
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(again.load(Ordering::SeqCst));
    }

    /// A buffer that waits on the GPU for a deferred copy of one cache, and is finished before
    /// that copy is made, stays among the users of every cache it used until it is: a copy of
    /// another of its caches asked behind it is made after the copy it waits for. Made at its
    /// finish, that copy would go first on the transfer queue and wait there for the buffer's
    /// end, and so for the copy behind it on that queue, and neither would run. Nor does it take
    /// an early copy of a surface that reads back, which would go first on the read-back queue:
    /// the download copies its write out
    #[test]
    fn a_copy_behind_a_buffer_waiting_for_a_copy_is_made_after_that_copy() {
        let mut r = cached();
        let other = BLOCK + 0x10000;
        let page = r.page;
        assert_eq!(r.pages_cached(other, &[(4, 1)], page, true), RGFX_STATUS_OK);
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false), (other, 0, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        // The other block's surface reads back: a download copied its write.
        let read = download(&r, other);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(read.load(Ordering::SeqCst));
        let (long_queue, queue) = (
            r.device.new_command_queue().unwrap(),
            r.device.new_command_queue().unwrap(),
        );
        // A long walk reads the block's cache.
        let long = long_queue.command_buffer(ON).unwrap().unwrap();
        let (_, walking) = (r.dec.surface_caches)
            .before(&r.device, &long, &reading(BLOCK, 1), &NO_REACH)
            .unwrap();
        // A buffer reads a newer generation of it, whose fill waits for the long walk, and writes
        // the other block's cache, whose download is asked behind it.
        let buffer = queue.command_buffer(ON).unwrap().unwrap();
        let uses = [
            reading(BLOCK, 2)[0],
            RgfxOpSurfaceUse {
                flags: RGFX_SURFACE_USE_WRITES,
                ..reading(other, 0)[0]
            },
        ];
        let (waits, using) = (r.dec.surface_caches)
            .before(&r.device, &buffer, &uses, &NO_REACH)
            .unwrap();
        assert!(waits, "waits for the fill");
        let downloaded = download(&r, other);
        let mark = r.p.call_log().mark();
        (r.dec.surface_caches)
            .finish(&r.device, 2, None, &buffer, using.unwrap(), || {
                buffer.commit(ON)
            })
            .unwrap();
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "a copy (the download, or an early copy) was made ahead of the fill the buffer waits \
             for"
        );
        (r.dec.surface_caches)
            .finish(&r.device, 1, None, &long, walking.unwrap(), || {
                long.commit(ON)
            })
            .unwrap();
        let (cache, other_cache) = {
            let caches = r.dec.caches();
            let id = |block| Arg::obj(caches.buffer(block).unwrap()).as_obj().unwrap().0;
            (id(BLOCK), id(other))
        };
        let copied = copies(&r.p.call_log().since(mark));
        assert_eq!(copied.len(), 2, "the fill, then the download");
        assert_eq!(copied[0].1, cache, "the fill first");
        assert_eq!(copied[1].0, other_cache, "then the download");
        let counts = r.dec.caches().counts();
        assert_eq!((counts.cpu_waits, counts.early), (0, 0));
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
    }

    /// A buffer that uses a device cache is answered `CB_SUBMITTED` at its doorbell, before its
    /// queue's thread takes it, where it would get it at its take: its cache's last fill is done,
    /// it needs no fill, and no copy asked of the cache is one it follows (R184, R224: it was
    /// refused for using a cache at all). Not while the cache owes its first fill, nor while that
    /// fill runs, nor where the buffer's generation is newer than the cache's, nor for a writer
    /// behind a download that waits for another writer, which a reader is not behind; nor while
    /// the caches' lock is busy.
    #[test]
    fn a_buffer_using_a_cache_is_answered_at_its_doorbell_where_it_waits_for_no_copy() {
        let mut r = cached();
        let answered = |r: &Rig, message: &[u8]| {
            let yes = (r.dec.decoders()).answer_at_doorbell(0, message, |_| true);
            // What the answer took, let go of, as a begin would take it.
            let took = r
                .dec
                .queue_counts(1)
                .unwrap()
                .answered
                .lock()
                .take()
                .is_some();
            assert_eq!(took, yes, "kept on its queue where answered");
            yes
        };
        let reads = |cb, generation| surfaces_message(1, cb, &[(BLOCK, generation, false)], true);
        let writes = |cb, generation| surfaces_message(1, cb, &[(BLOCK, generation, true)], true);
        assert!(!answered(&r, &reads(0x300, 0)), "the first fill owed");
        // Its begin makes the fill.
        assert_eq!(r.dec.decode(&r.device, &reads(0x300, 0)).unwrap(), Ok(()));
        assert!(!answered(&r, &reads(0x2ff, 0)), "the first fill runs");
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(
            answered(&r, &reads(0x301, 0)),
            "a reader of the cache's generation"
        );
        assert!(answered(&r, &writes(0x302, 0)), "a writer of it");
        assert!(
            !answered(&r, &reads(0x303, 1)),
            "a newer generation, which fills"
        );
        // A writer begun, and a download asked behind it.
        let queue = r.device.new_command_queue().unwrap();
        let writer = queue.command_buffer(ON).unwrap().unwrap();
        let use_of = [RgfxOpSurfaceUse {
            flags: RGFX_SURFACE_USE_WRITES,
            ..reading(BLOCK, 0)[0]
        }];
        let (_, uses) = (r.dec.surface_caches)
            .before(&r.device, &writer, &use_of, &NO_REACH)
            .unwrap();
        let downloaded = download(&r, BLOCK);
        assert!(
            !answered(&r, &writes(0x304, 0)),
            "a writer behind the download"
        );
        assert!(answered(&r, &reads(0x305, 0)), "a reader is not behind it");
        {
            let _held = r.dec.caches();
            assert!(!answered(&r, &reads(0x306, 0)), "the caches' lock busy");
        }
        (r.dec.surface_caches)
            .finish(&r.device, 1, None, &writer, uses.unwrap(), || {
                writer.commit(ON)
            })
            .unwrap();
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
        assert!(
            answered(&r, &writes(0x307, 0)),
            "the download made and done"
        );
        let counts = r.dec.caches().counts();
        assert_eq!(
            (counts.doorbell_free, counts.doorbell_waits),
            (4, 4),
            "the looks with the lock free"
        );
    }

    /// A download and a release asked while a command buffer that may write the cache is begun
    /// and not finished are not waited for where they are asked: each is made, in the order
    /// asked, in that buffer's finish, after its commit, and a buffer begun meanwhile that only
    /// reads the cache waits for neither. Here the download copies the buffer's write out, and
    /// the release then has nothing more to copy: the cache goes, and the release is answered as
    /// the download's copy lands
    #[test]
    fn a_copy_out_of_a_cache_is_made_as_its_writers_finish() {
        let mut r = cached();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let queue = r.device.new_command_queue().unwrap();
        let writer = queue.command_buffer(ON).unwrap().unwrap();
        let writes = [RgfxOpSurfaceUse {
            flags: RGFX_SURFACE_USE_WRITES,
            ..reading(BLOCK, 1)[0]
        }];
        let (_, uses) = (r.dec.surface_caches)
            .before(&r.device, &writer, &writes, &NO_REACH)
            .unwrap();
        let downloaded = Arc::new(AtomicBool::new(false));
        let released = Arc::new(AtomicBool::new(false));
        let mark = r.p.call_log().mark();
        let set = Arc::clone(&downloaded);
        r.dec.surface_download(
            &r.device,
            BLOCK,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        let set = Arc::clone(&released);
        r.dec.block_released(
            &r.device,
            BLOCK,
            0,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "copied while the writer is begun"
        );
        assert!(!downloaded.load(Ordering::SeqCst) && !released.load(Ordering::SeqCst));
        let later = queue.command_buffer(ON).unwrap().unwrap();
        let (waits, reading_uses) = (r.dec.surface_caches)
            .before(&r.device, &later, &reading(BLOCK, 1), &NO_REACH)
            .unwrap();
        assert!(
            !waits && reading_uses.is_some(),
            "begun before the copies were made"
        );
        (r.dec.surface_caches)
            .finish(&r.device, 1, Some(&queue), &writer, uses.unwrap(), || {
                writer.commit(ON)
            })
            .unwrap();
        drop(reading_uses);
        let calls = r.p.call_log().since(mark);
        let commit = calls.iter().position(|c| c.sel == "commit").unwrap();
        let copied = calls.iter().position(|c| c.sel == COPY).unwrap();
        assert!(commit < copied, "copied after the writer's commit");
        assert_eq!(copies(&calls).len(), 1, "the download's copy alone");
        assert!(
            !released.load(Ordering::SeqCst),
            "nothing more to copy: answered as the download's copy lands"
        );
        assert_eq!(r.dec.caches().len(), 0);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst) && released.load(Ordering::SeqCst));
    }

    /// A command buffer's commit is made with the caches' lock let go: while the platform's
    /// commit call takes, another buffer that uses the cache is begun and the doorbell's look at
    /// the caches is answered; a download asked meanwhile, behind the committing buffer's write,
    /// is still made after the commit, and so is a fill asked at another begin, which waits for
    /// nothing there, as the buffer counts among the cache's users until its commit returns
    #[test]
    fn a_commit_is_made_with_the_caches_lock_let_go() {
        let mut r = cached();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let queue = r.device.new_command_queue().unwrap();
        let writer = queue.command_buffer(ON).unwrap().unwrap();
        let writes = [RgfxOpSurfaceUse {
            flags: RGFX_SURFACE_USE_WRITES,
            ..reading(BLOCK, 1)[0]
        }];
        let (_, uses) = (r.dec.surface_caches)
            .before(&r.device, &writer, &writes, &NO_REACH)
            .unwrap();
        let caches = Arc::clone(&r.dec.surface_caches);
        let reader = queue.command_buffer(ON).unwrap().unwrap();
        let filler = queue.command_buffer(ON).unwrap().unwrap();
        let mark = r.p.call_log().mark();
        let (in_commit, committing) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let (go, wait_go) = std::sync::mpsc::channel::<()>();
            let finish = scope.spawn(|| {
                let writer = &writer;
                caches
                    .finish(&r.device, 1, None, writer, uses.unwrap(), move || {
                        in_commit.send(()).unwrap();
                        let _ = wait_go.recv();
                        writer.commit(ON)
                    })
                    .map_err(|_| ())
            });
            committing.recv().unwrap();
            let begin = scope.spawn(|| {
                (caches.before(&r.device, &reader, &reading(BLOCK, 1), &NO_REACH))
                    .map(|(_, uses)| uses.is_some())
                    .map_err(|_| ())
            });
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(
                begin.is_finished(),
                "a begin waited for another buffer's commit"
            );
            assert_eq!(begin.join().unwrap(), Ok(true));
            assert!(
                caches
                    .waits_for_nothing(reading(BLOCK, 1).into_iter())
                    .is_some(),
                "the doorbell's look found the caches' lock held"
            );
            let downloaded = download(&r, BLOCK);
            let fill = scope.spawn(|| {
                (caches.before(&r.device, &filler, &reading(BLOCK, 2), &NO_REACH))
                    .map(|_| ())
                    .map_err(|_| ())
            });
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(
                fill.is_finished(),
                "a begin waited for another buffer's commit"
            );
            assert!(
                !downloaded.load(Ordering::SeqCst),
                "downloaded before the commit"
            );
            assert!(
                copies(&r.p.call_log().since(mark)).is_empty(),
                "copied while the writer commits"
            );
            go.send(()).unwrap();
            assert!(finish.join().unwrap().is_ok());
            assert!(fill.join().unwrap().is_ok());
        });
        let calls = r.p.call_log().since(mark);
        let commit = calls.iter().position(|c| c.sel == "commit").unwrap();
        let copied = calls.iter().position(|c| c.sel == COPY).unwrap();
        assert!(commit < copied, "copied after the writer's commit");
        assert_eq!(
            copies(&calls).len(),
            2,
            "the download's copy and the fill's"
        );
    }

    /// A device of the host's write over a cache's pages has the cache's next use fill it again,
    /// though of the generation it holds, and once; a write beside them does nothing. A cache a
    /// buffer wrote is not copied over such a write, at a download or at a release
    #[test]
    fn a_host_devices_write_over_the_pages_refills_the_cache_and_is_not_overwritten() {
        let mut r = cached();
        let page = r.page;
        let cache = cache_id(&r);
        // The block's surface is guest pages 1 and 2.
        let first = r.ram.host_addr(RAM + page).unwrap().addr().to_u64();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        device_writes::wrote(&[first - page..first, first + 2 * page..first + 3 * page]);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 1, false)]);
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "beside the pages"
        );

        // Every run lies in the rig's own RAM: the watchers are the process's, and a run past
        // it would mark the caches of other tests running beside this one written.
        device_writes::wrote(&[
            first + page + 8..first + page + 16,
            first + 5 * page..first + 7 * page,
        ]);
        let done = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&done);
        let mark = r.p.call_log().mark();
        r.dec.surface_download(
            &r.device,
            BLOCK,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        assert!(done.load(Ordering::SeqCst), "answered at once");
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "the cache is not copied over the host's write"
        );
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x102, &[(BLOCK, 1, false)]);
        let filled = copies(&r.p.call_log().since(mark));
        assert_eq!(filled.len(), 1, "filled again, of the same generation");
        assert_eq!(filled[0].1, cache);
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x103, &[(BLOCK, 1, true)]);
        assert!(copies(&r.p.call_log().since(mark)).is_empty(), "once");

        device_writes::wrote(&[first..first + 1, first + 5 * page..first + 7 * page]);
        let released = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&released);
        let mark = r.p.call_log().mark();
        r.dec.block_released(
            &r.device,
            BLOCK,
            0,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        assert!(released.load(Ordering::SeqCst), "done at once");
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        let counts = r.dec.caches().counts();
        assert_eq!(
            (counts.fills, counts.downloads, counts.host_writes),
            (2, 0, 3),
            "generation 1's, the first, and the host's write's"
        );
    }

    /// A buffer's write that a read-write CPU lock waited for is in the pages before the newer
    /// generation's fill: the lock's download copies it out of the cache, and the fill, after it
    /// on the transfer queue, copies the pages and the CPU's writes back in. A write the kext did
    /// not download before the lock (a commit racing the lock on another thread) is filled over,
    /// the CPU's writes winning, and the cache, still written, is copied out at its next download.
    /// Two writes come before the first download, so the block does not read back: one that does
    /// takes the raced write's early copy, which the fill waits for
    /// (`a_read_back_queue_s_copy_in_flight_holds_the_next_writer`)
    #[test]
    fn a_newer_generation_fills_after_the_download_of_a_write() {
        let mut r = cached();
        let cache = cache_id(&r);
        use_surfaces(&mut r, 1, 0x0ff, &[(BLOCK, 0, true)]);
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 1, false)]);
        let copied = copies(&r.p.call_log().since(mark));
        assert_eq!(copied.len(), 2, "{copied:?}");
        let pages = copied[0].1;
        assert_eq!(
            copied,
            [(cache, pages), (pages, cache)],
            "the download, then the fill"
        );
        // A write at generation 1, then a buffer of generation 2 with no download between.
        use_surfaces(&mut r, 1, 0x102, &[(BLOCK, 1, true)]);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x103, &[(BLOCK, 2, false)]);
        assert_eq!(
            copies(&r.p.call_log().since(mark)),
            [(pages, cache)],
            "filled over"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
        assert_eq!(
            copies(&r.p.call_log().since(mark)),
            [(cache, pages)],
            "still written"
        );
    }

    /// A copy out of a cache that waited for a writer begun before it is not made where the CPU's
    /// generation moved past the cache's meanwhile: the kext's read-write lock gave up waiting for
    /// its answer (2 s), moved the generation and let the CPU write the pages, which the copy
    /// would land over. The download and the release made at the writer's finish copy nothing,
    /// and are answered; the next use of the newer generation fills the cache from the pages. One
    /// whose generation did not move copies, as before
    #[test]
    fn a_copy_out_after_the_cpu_s_generation_moved_copies_nothing() {
        let mut r = cached();
        let cache = cache_id(&r);
        let at = BLOCK + CPU_GEN_OFFSET;
        let set_gen = |r: &Rig, generation: u64| {
            assert!(r.dec.region.write(at, &generation.to_le_bytes()));
        };
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        // A writer of generation 0, begun and not finished (its walk held up, past the kext's
        // wait for the download's answer).
        let begin_writer = |r: &Rig| {
            let queue = r.device.new_command_queue().unwrap();
            let writer = queue.command_buffer(ON).unwrap().unwrap();
            let writes = [RgfxOpSurfaceUse {
                flags: RGFX_SURFACE_USE_WRITES,
                ..reading(BLOCK, 0)[0]
            }];
            let (_, uses) = (r.dec.surface_caches)
                .before(&r.device, &writer, &writes, &NO_REACH)
                .unwrap();
            (writer, uses.unwrap())
        };
        let (writer, uses) = begin_writer(&r);
        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        // The lock gives up, moves the generation, and the CPU writes the pages; then the writer
        // finishes.
        set_gen(&r, 1);
        (r.dec.surface_caches)
            .finish(&r.device, 1, None, &writer, uses, || writer.commit(ON))
            .unwrap();
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst), "answered");
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "the download copied over the CPU's writes"
        );
        assert_eq!(r.dec.caches().counts().behind, 1);
        // The next use, of generation 1, fills the cache from the pages.
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 1, true)]);
        let filled = copies(&r.p.call_log().since(mark));
        assert_eq!(filled.len(), 1);
        assert_eq!(filled[0].1, cache, "into the cache");
        r.p.world().complete_all(CompleteOn::Caller);
        // A release behind a writer, the generation moved meanwhile: nothing is copied.
        let (writer, uses) = begin_writer(&r);
        let mark = r.p.call_log().mark();
        let released = release(&r, BLOCK);
        set_gen(&r, 2);
        (r.dec.surface_caches)
            .finish(&r.device, 1, None, &writer, uses, || writer.commit(ON))
            .unwrap();
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(released.load(Ordering::SeqCst), "answered");
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "the release copied over the CPU's writes"
        );
        assert_eq!(r.dec.caches().counts().behind, 2);
        assert_eq!(r.dec.caches().len(), 0, "the cache went");
    }

    /// The control of [`a_copy_out_after_the_cpu_s_generation_moved_copies_nothing`]: a download
    /// behind a writer, the CPU's generation in the block's header still the cache's, copies the
    /// cache into the pages at the writer's finish
    #[test]
    fn a_copy_out_of_the_cache_s_generation_copies() {
        let mut r = cached();
        let cache = cache_id(&r);
        assert!(
            r.dec
                .region
                .write(BLOCK + CPU_GEN_OFFSET, &1u64.to_le_bytes())
        );
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let queue = r.device.new_command_queue().unwrap();
        let writer = queue.command_buffer(ON).unwrap().unwrap();
        let writes = [RgfxOpSurfaceUse {
            flags: RGFX_SURFACE_USE_WRITES,
            ..reading(BLOCK, 1)[0]
        }];
        let (_, uses) = (r.dec.surface_caches)
            .before(&r.device, &writer, &writes, &NO_REACH)
            .unwrap();
        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        (r.dec.surface_caches)
            .finish(&r.device, 1, None, &writer, uses.unwrap(), || {
                writer.commit(ON)
            })
            .unwrap();
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
        let copied = copies(&r.p.call_log().since(mark));
        assert_eq!(copied.len(), 1);
        assert_eq!(copied[0].0, cache, "out of the cache");
        assert_eq!(r.dec.caches().counts().behind, 0);
    }

    /// The CPU generation's place in a cached block's header is the kext's
    #[test]
    fn the_cpu_generation_s_offset_is_the_kext_s() {
        let abi = include_str!("../../../../../guest/kext/src/RgfxUserClientABI.hpp");
        let line = format!("kRgfxSurfaceCpuGenOffset = {CPU_GEN_OFFSET},");
        assert!(abi.contains(&line), "the kext's header has no `{line}`");
    }

    /// The guest pages of a surface by runs: scattered and out of order, (page, pages) of the
    /// rig's RAM, as the guest's are.
    const SCATTERED: [(u64, u64); 3] = [(5, 1), (1, 2), (0, 1)];

    /// A rig on a device that copies the guest's RAM by its runs, its RAM holding a pattern
    /// (each byte its own), whose block at [`BLOCK`] is [`SCATTERED`]'s four pages of a surface,
    /// marked cached, with command queues 1 and 2.
    fn cached_by_runs() -> Rig {
        let mut r = rig(8);
        let page = r.page;
        r.p.world().copy_guest_ram(true);
        let pattern: Vec<u8> = (0..8 * page).map(|i| (i * 7 + i / page) as u8).collect();
        let at = r.ram.host_addr(RAM).unwrap();
        // SAFETY: the rig's RAM, 8 pages from `at`, which nothing else uses now.
        unsafe {
            std::ptr::copy_nonoverlapping(pattern.as_ptr(), at.as_ptr::<u8>(), pattern.len())
        };
        assert_eq!(
            r.pages_cached(BLOCK, &SCATTERED, 4 * page, true),
            RGFX_STATUS_OK
        );
        queues(&mut r);
        r
    }

    /// The runs each call to `sel` in `calls` copies by, its argument `at`.
    fn copied_runs(calls: &[Call], sel: &str, at: usize) -> Vec<Vec<Range<u64>>> {
        (calls.iter())
            .filter(|c| c.sel == sel)
            .map(|c| match &c.args[at] {
                Arg::U64s(ends) => ends.chunks(2).map(|e| e[0]..e[1]).collect(),
                other => panic!("runs logged as {other:?}"),
            })
            .collect()
    }

    /// The bytes at `runs`, host addresses, in order.
    fn gathered(runs: &[Range<u64>]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for run in runs {
            // SAFETY: the rig's RAM, which the runs lie in, mapped while the rig lives.
            let at = std::ptr::with_exposed_provenance::<u8>(run.start.to_usize());
            bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(at, (run.end - run.start).to_usize())
            });
        }
        bytes
    }

    /// Writes `bytes` at `runs`, host addresses, in order.
    fn scatter(runs: &[Range<u64>], bytes: &[u8]) {
        let mut from = 0;
        for run in runs {
            let n = (run.end - run.start).to_usize();
            // SAFETY: the rig's RAM, which the runs lie in, mapped while the rig lives and used
            // by nothing else now.
            unsafe {
                let at = std::ptr::with_exposed_provenance_mut::<u8>(run.start.to_usize());
                std::ptr::copy_nonoverlapping(bytes[from..].as_ptr(), at, n);
            }
            from += n;
        }
        assert_eq!(from, bytes.len());
    }

    /// The bytes a view of [`SCATTERED`]'s pages holds: an uncached block's, at `block`.
    fn view_bytes(r: &mut Rig, block: u64) -> Vec<u8> {
        let page = r.page;
        assert_eq!(r.pages(block, &SCATTERED, 4 * page), RGFX_STATUS_OK);
        let view = r.view(block);
        // SAFETY: the view's mapping, `length` bytes, which lives while `view` does.
        unsafe { std::slice::from_raw_parts(view.base().as_ptr::<u8>(), view.length().to_usize()) }
            .to_vec()
    }

    /// On a device that copies the guest's RAM by its runs, a cached block takes no view, and its
    /// cache copies by its pages' runs, in the surface's order, both ways: those runs hold, byte
    /// for byte, what a view of the block's pages holds, and what is written through them is
    /// what the view then holds, as the guest's pages lie scattered and out of order
    #[test]
    fn a_cached_block_by_runs_copies_the_bytes_its_view_holds() {
        use reims_vgpu_vulkan::core::testkit::impls::blit::{
            BUFFER_TO_GUEST_RAM, GUEST_RAM_TO_BUFFER,
        };
        let mut r = cached_by_runs();
        let page = r.page;
        assert_eq!(
            r.dec.views_of_surfaces().len(),
            (0, 0),
            "no view, none cached"
        );
        assert_eq!(r.dec.views_of_surfaces().counts().by_runs, 1);
        assert_eq!(r.dec.caches().counts().by_runs, 1);
        assert_eq!(
            r.buffer(10, BLOCK, 4 * page),
            Ok(()),
            "its buffer is its cache"
        );
        let cache = cache_id(&r);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        let calls = r.p.call_log().since(mark);
        assert!(copies(&calls).is_empty(), "no copy through a view");
        let fills = copied_runs(&calls, GUEST_RAM_TO_BUFFER, 0);
        assert_eq!(fills.len(), 1);
        let filled = (calls.iter())
            .find(|c| c.sel == GUEST_RAM_TO_BUFFER)
            .unwrap();
        assert_eq!(filled.args[1].as_obj().map(|o| o.0), Some(cache));
        assert_eq!(filled.args[2].as_u64(), Some(0));
        // Pages 5, then 1 and 2, which follow each other, then 0: three runs.
        assert_eq!(fills[0].len(), 3);
        let filled_bytes = gathered(&fills[0]);
        assert_eq!(filled_bytes.len(), (4 * page).to_usize());
        assert!(
            filled_bytes == view_bytes(&mut r, BLOCK + 0x4000),
            "the view's bytes"
        );

        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 0, true)]);
        let done = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&done);
        let mark = r.p.call_log().mark();
        r.dec.surface_download(
            &r.device,
            BLOCK,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        let calls = r.p.call_log().since(mark);
        assert!(copies(&calls).is_empty(), "no copy through a view");
        let downloads = copied_runs(&calls, BUFFER_TO_GUEST_RAM, 2);
        assert_eq!(downloads, fills, "the same runs, both ways");
        let out = (calls.iter())
            .find(|c| c.sel == BUFFER_TO_GUEST_RAM)
            .unwrap();
        assert_eq!(out.args[0].as_obj().map(|o| o.0), Some(cache));
        assert_eq!(
            out.args[1],
            Arg::U64s(vec![0, page, 3 * page]),
            "one copy, each run from its byte of the cache"
        );
        let written: Vec<u8> = (0..4 * page).map(|i| (i * 3 + 1) as u8).collect();
        scatter(&downloads[0], &written);
        // The fake's views copy the RAM at their mapping, and take its writes at a sync.
        r.p.fake_vm().sync();
        assert!(
            written == view_bytes(&mut r, BLOCK + 0x8000),
            "what the view holds"
        );
        assert!(!done.load(Ordering::SeqCst));
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
        let counts = r.dec.caches().counts();
        assert_eq!(
            (counts.fills, counts.downloads, counts.runs_refused),
            (1, 1, 0),
            "generation 1's, the first"
        );
    }

    /// A copy into a cache by runs that the platform refuses makes the block's view there and
    /// then and fills the cache through it, and the cache copies through the view from then on,
    /// both ways: what a command buffer wrote of the cache still reaches the pages
    #[test]
    fn a_refused_fill_by_runs_fills_through_a_view_from_then_on() {
        use reims_vgpu_vulkan::core::testkit::Step;
        use reims_vgpu_vulkan::core::testkit::impls::blit::{
            BUFFER_TO_GUEST_RAM, GUEST_RAM_TO_BUFFER,
        };
        let mut r = cached_by_runs();
        let cache = cache_id(&r);
        r.p.script().push(GUEST_RAM_TO_BUFFER, Step::value(false));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        let calls = r.p.call_log().since(mark);
        let sels: Vec<&str> = (calls.iter())
            .map(|c| c.sel)
            .filter(|s| [GUEST_RAM_TO_BUFFER, "newBufferWithDescriptor:", COPY].contains(s))
            .collect();
        assert_eq!(
            sels,
            [GUEST_RAM_TO_BUFFER, "newBufferWithDescriptor:", COPY]
        );
        let filled = copies(&calls);
        assert_eq!(filled.len(), 1);
        let (view, into) = filled[0];
        assert_eq!(into, cache);
        assert_eq!(r.dec.caches().counts().runs_refused, 1);

        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 2, true)]);
        let calls = r.p.call_log().since(mark);
        assert!(
            copied_runs(&calls, GUEST_RAM_TO_BUFFER, 0).is_empty(),
            "by the view now"
        );
        assert_eq!(copies(&calls), [(view, cache)]);

        let done = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&done);
        let mark = r.p.call_log().mark();
        r.dec.surface_download(
            &r.device,
            BLOCK,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        let calls = r.p.call_log().since(mark);
        assert!(
            copied_runs(&calls, BUFFER_TO_GUEST_RAM, 2).is_empty(),
            "by the view now"
        );
        assert_eq!(
            copies(&calls),
            [(cache, view)],
            "the write reaches the pages"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
        let counts = r.dec.caches().counts();
        assert_eq!(
            (counts.fills, counts.downloads, counts.runs_refused),
            (2, 1, 1),
            "the first fill, refused by runs and made through the view, then generation 2's"
        );
    }

    /// A cache made for a block that asked with its surface's layout has the platform place its
    /// first plane's first byte where its textures may start in place, where the plane starts
    /// past the surface's first byte, as CoreVideo's planes start 32 bytes into their surface (or
    /// 256 bytes in, which the platform places alike), so textures over the planes lie in the
    /// cache rather than in copies of their own; a cache made with no layout is made as any other
    /// buffer
    #[test]
    fn an_asked_cache_places_its_first_planes_byte() {
        use reims_vgpu_graphics::wire::{
            RGFX_STATUS_CACHE, RgfxSurfacePagesLayout, RgfxSurfacePlaneLayout,
        };
        use reims_vgpu_vulkan::core::platform::values::PagesInPlace;
        use reims_vgpu_vulkan::core::testkit::impls::device::PADDING_UNOBSERVED_BUFFER;
        const NEW_BUFFER: &str = "newBufferWithLength:options:";
        let mut r = rig(16);
        let page = r.page;
        r.dec.set_unified_memory(true);
        let world = Arc::clone(r.p.world());
        world.answer_surface_pages(PagesInPlace::No("off the device's alignment".into()));
        // NV12 as CoreVideo lays it in four pages: luma from 0x20, chroma after its rows.
        let planes = |first: u64| {
            let mut layout = RgfxSurfacePagesLayout {
                plane_count: 2,
                ..Default::default()
            };
            layout.planes[0] = RgfxSurfacePlaneLayout {
                offset: first,
                bytes_per_row: 512,
                row_bytes: 480,
                rows: 16,
            };
            layout.planes[1] = RgfxSurfacePlaneLayout {
                offset: first + 16 * 512,
                bytes_per_row: 512,
                row_bytes: 480,
                rows: 8,
            };
            layout
        };
        let made = |r: &mut Rig, block: u64, first: u64, layout: Option<RgfxSurfacePagesLayout>| {
            let mark = r.p.call_log().mark();
            if layout.is_some() {
                assert_eq!(
                    r.pages_asking(block, &[(first, 4)], 4 * page, layout),
                    RGFX_STATUS_CACHE
                );
            }
            assert_eq!(
                r.pages_cached(block, &[(first, 4)], 4 * page, true),
                RGFX_STATUS_OK
            );
            let calls = r.p.call_log().since(mark);
            let made: Vec<(&str, Vec<Arg>)> = (calls.iter())
                .filter(|c| [NEW_BUFFER, PADDING_UNOBSERVED_BUFFER].contains(&c.sel))
                .map(|c| (c.sel, c.args.clone()))
                .collect();
            assert_eq!(made.len(), 1, "one cache");
            made.into_iter().next().unwrap()
        };
        let (sel, args) = made(&mut r, BLOCK, 1, Some(planes(0x20)));
        assert_eq!(sel, PADDING_UNOBSERVED_BUFFER, "a laid-out cache");
        assert_eq!(args.get(2), Some(&Arg::U64(0x20)), "the first plane's byte");
        assert_eq!(args[0], Arg::U64(4 * page));
        assert_eq!(r.dec.caches().counts().aligned, 1);

        let (sel, args) = made(&mut r, BLOCK + 0x10000, 5, Some(planes(0x100)));
        assert_eq!(sel, PADDING_UNOBSERVED_BUFFER, "a laid-out cache");
        assert_eq!(
            args.get(2),
            Some(&Arg::U64(0x100)),
            "a plane 256 bytes in: the platform's to place"
        );
        let (sel, args) = made(&mut r, BLOCK + 0x20000, 9, None);
        assert_eq!(
            (sel, args.len()),
            (NEW_BUFFER, 2),
            "no layout: as any other buffer"
        );
        assert_eq!(r.dec.caches().counts().aligned, 2);
    }

    /// A cached block's own `SURFACE_PAGES` that carries its surface's layout, on a device that
    /// asks nothing (its GPU keeps its own memory), places the block's cache by it alone: the
    /// cache's buffer has its first plane's byte placed where the platform's textures may start
    /// in place, where the plane starts past the surface's first byte (CoreVideo's 32), with
    /// nothing else declared, and its copies still move the surface's bytes whole, a fill and a
    /// download copying every byte; a deferred block's message places its cache alike. A first
    /// plane at the surface's first byte, a layout that does not lie in the surface's bytes, and
    /// a message with none place nothing, and every one is answered as with none
    #[test]
    fn a_cached_blocks_layout_places_its_cache_alone() {
        use reims_vgpu_graphics::wire::{RgfxSurfacePagesLayout, RgfxSurfacePlaneLayout};
        use reims_vgpu_vulkan::core::testkit::impls::device::{
            ALIGNED_BUFFER, PADDING_UNOBSERVED_BUFFER,
        };
        const NEW_BUFFER: &str = "newBufferWithLength:options:";
        let mut r = rig(24);
        r.dec.set_deferred_budget(16 * r.page);
        queues(&mut r);
        let page = r.page;
        // NV12 as CoreVideo lays it in four pages, its luma plane `first` bytes in.
        let planes = |first: u64, rows: u64| {
            let mut layout = RgfxSurfacePagesLayout {
                plane_count: 2,
                ..Default::default()
            };
            layout.planes[0] = RgfxSurfacePlaneLayout {
                offset: first,
                bytes_per_row: 512,
                row_bytes: 480,
                rows,
            };
            layout.planes[1] = RgfxSurfacePlaneLayout {
                offset: first + rows * 512,
                bytes_per_row: 512,
                row_bytes: 480,
                rows: rows / 2,
            };
            layout
        };
        let cached = RGFX_SURFACE_PAGES_CACHED;
        let deferred = RGFX_SURFACE_PAGES_CACHED | RGFX_SURFACE_PAGES_DEFERRED;
        // The cache each message makes: its maker and arguments.
        let made = |r: &mut Rig, block: u64, at: u64, flags: u32, layout| {
            let mark = r.p.call_log().mark();
            let runs: &[(u64, u64)] = if flags == deferred { &[] } else { &[(at, 4)] };
            assert_eq!(
                r.pages_laid_out(block, 0, runs, 4 * page, flags, layout),
                RGFX_STATUS_OK,
                "answered as with no layout"
            );
            let calls = r.p.call_log().since(mark);
            let made: Vec<(&str, Vec<Arg>)> = (calls.iter())
                .filter(|c| {
                    [NEW_BUFFER, PADDING_UNOBSERVED_BUFFER, ALIGNED_BUFFER].contains(&c.sel)
                })
                .map(|c| (c.sel, c.args.clone()))
                .collect();
            assert_eq!(made.len(), 1, "one cache");
            made.into_iter().next().unwrap()
        };
        let (sel, args) = made(&mut r, BLOCK + 0x10000, 1, cached, Some(planes(0x20, 16)));
        assert_eq!(
            (sel, args.get(2)),
            (ALIGNED_BUFFER, Some(&Arg::U64(0x20))),
            "the first plane's byte placed, nothing declared"
        );
        assert_eq!(args[0], Arg::U64(4 * page));
        let (sel, args) = made(&mut r, BLOCK + 0x20000, 0, deferred, Some(planes(0x20, 16)));
        assert_eq!(
            (sel, args.get(2)),
            (ALIGNED_BUFFER, Some(&Arg::U64(0x20))),
            "a deferred block's cache placed alike"
        );
        assert_eq!(r.dec.caches().counts().aligned, 2);
        for (block, at, layout, why) in [
            (
                BLOCK + 0x30000,
                5,
                Some(planes(0, 16)),
                "a first plane at the first byte",
            ),
            (
                BLOCK + 0x40000,
                9,
                Some(planes(0x20, 4096)),
                "a layout past the surface's bytes",
            ),
            (BLOCK + 0x50000, 13, None, "no layout"),
        ] {
            let (sel, args) = made(&mut r, block, at, cached, layout);
            assert_eq!(
                (sel, args.len()),
                (NEW_BUFFER, 2),
                "{why}: placed as any other"
            );
        }
        assert_eq!(r.dec.caches().counts().aligned, 2);

        // The placed cache's copies move the surface's bytes whole: its first fill copies every
        // byte in, and a download every byte out, as of a cache with no layout.
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK + 0x10000, 0, true)]);
        assert_eq!(
            offsets(&r.p.call_log().since(mark)),
            [(0, 0, 4 * page)],
            "the first fill copies every byte in"
        );
        let mark = r.p.call_log().mark();
        (r.dec).surface_download(&r.device, BLOCK + 0x10000, Box::new(|| {}), no_raise());
        assert_eq!(
            offsets(&r.p.call_log().since(mark)),
            [(0, 0, 4 * page)],
            "the download copies every byte out"
        );
    }

    /// Only a cache whose memory was made for the host's video gives the video its identity
    /// (`VideoPlane::cache`), so that no decode takes for an import a cache the platform made as
    /// any other buffer: where the host's video decodes into caches, a laid-out cache is made a
    /// dma-buf and gives it; a deferred block's, whose ask carries no layout, is made as any
    /// other buffer and gives none, as does a laid-out cache where the video decodes into none
    #[cfg(unix)]
    #[test]
    fn only_a_cache_made_for_the_hosts_video_gives_it_its_identity() {
        use reims_vgpu_graphics::wire::{
            RGFX_STATUS_CACHE, RgfxSurfacePagesLayout, RgfxSurfacePlaneLayout,
        };
        use reims_vgpu_vulkan::core::platform::values::PagesInPlace;
        use reims_vgpu_vulkan::core::testkit::impls::device::DMA_BUF_BUFFER;
        // NV12 as CoreVideo lays it in four pages: luma from 0x20, chroma after its rows.
        let mut layout = RgfxSurfacePagesLayout {
            plane_count: 2,
            ..Default::default()
        };
        layout.planes[0] = RgfxSurfacePlaneLayout {
            offset: 0x20,
            bytes_per_row: 512,
            row_bytes: 480,
            rows: 16,
        };
        layout.planes[1] = RgfxSurfacePlaneLayout {
            offset: 0x20 + 16 * 512,
            bytes_per_row: 512,
            row_bytes: 480,
            rows: 8,
        };
        // Whether a laid-out cache was made a dma-buf, and whether it gives its identity.
        let laid_out = |dma_buf: bool| {
            let mut r = rig(16);
            let page = r.page;
            r.dec.set_unified_memory(true);
            if dma_buf {
                r.dec.set_dma_buf_caches();
            }
            let world = Arc::clone(r.p.world());
            world.answer_surface_pages(PagesInPlace::No("off the device's alignment".into()));
            let mark = r.p.call_log().mark();
            assert_eq!(
                r.pages_asking(BLOCK, &[(1, 4)], 4 * page, Some(layout)),
                RGFX_STATUS_CACHE
            );
            assert_eq!(
                r.pages_cached(BLOCK, &[(1, 4)], 4 * page, true),
                RGFX_STATUS_OK
            );
            let made = r.p.call_log().sels_since(mark).contains(&DMA_BUF_BUFFER);
            assert_eq!(r.dec.caches().len(), 1);
            (made, r.dec.caches().identity(BLOCK).is_some())
        };
        assert_eq!(laid_out(true), (true, true), "a dma-buf cache: given");
        assert_eq!(
            laid_out(false),
            (false, false),
            "a cache as any other: none"
        );

        let mut r = deferring(8);
        let page = r.page;
        r.dec.set_dma_buf_caches();
        let mark = r.p.call_log().mark();
        assert_eq!(defer(&mut r, BLOCK, 2 * page), RGFX_STATUS_OK);
        assert!(
            !r.p.call_log().sels_since(mark).contains(&DMA_BUF_BUFFER),
            "no layout: no dma-buf"
        );
        assert_eq!(r.dec.caches().counts().deferred, 1);
        assert!(
            r.dec.caches().identity(BLOCK).is_none(),
            "a deferred cache: none"
        );
    }

    /// Where the host's video decodes into caches, only a laid-out cache of two planes, as the
    /// video's NV12 and P010 output is, is made a dma-buf and gives the video its identity; a
    /// laid-out cache of one plane is made as any other laid-out cache, its bytes between rows
    /// unobserved, and gives none, as the video writes no surface of one plane
    #[cfg(unix)]
    #[test]
    fn only_a_two_plane_cache_is_made_a_dma_buf_for_the_hosts_video() {
        use super::super::surface_pages::tests::one_plane;
        use reims_vgpu_graphics::wire::{
            RGFX_STATUS_CACHE, RgfxSurfacePagesLayout, RgfxSurfacePlaneLayout,
        };
        use reims_vgpu_vulkan::core::platform::values::PagesInPlace;
        use reims_vgpu_vulkan::core::testkit::impls::device::{
            DMA_BUF_BUFFER, PADDING_UNOBSERVED_BUFFER,
        };
        // Two planes as CoreVideo lays them from 0x20: `sample` bytes a texel, the chroma
        // plane's rows after the luma's, interleaved and half as many.
        let two_planes = |sample: u64| {
            let mut layout = RgfxSurfacePagesLayout {
                plane_count: 2,
                ..Default::default()
            };
            layout.planes[0] = RgfxSurfacePlaneLayout {
                offset: 0x20,
                bytes_per_row: 256 * sample,
                row_bytes: 240 * sample,
                rows: 16,
            };
            layout.planes[1] = RgfxSurfacePlaneLayout {
                offset: 0x20 + 16 * 256 * sample,
                bytes_per_row: 256 * sample,
                row_bytes: 240 * sample,
                rows: 8,
            };
            layout
        };
        // The buffer a laid-out cache was made as, and whether it gives its identity.
        let made = |layout: RgfxSurfacePagesLayout| {
            let mut r = rig(16);
            let page = r.page;
            r.dec.set_unified_memory(true);
            r.dec.set_dma_buf_caches();
            let world = Arc::clone(r.p.world());
            world.answer_surface_pages(PagesInPlace::No("off the device's alignment".into()));
            let mark = r.p.call_log().mark();
            assert_eq!(
                r.pages_asking(BLOCK, &[(1, 4)], 4 * page, Some(layout)),
                RGFX_STATUS_CACHE
            );
            assert_eq!(
                r.pages_cached(BLOCK, &[(1, 4)], 4 * page, true),
                RGFX_STATUS_OK
            );
            let sels = r.p.call_log().sels_since(mark);
            let made: Vec<&str> = (sels.iter().copied())
                .filter(|sel| [DMA_BUF_BUFFER, PADDING_UNOBSERVED_BUFFER].contains(sel))
                .collect();
            assert_eq!(r.dec.caches().len(), 1);
            (made, r.dec.caches().identity(BLOCK).is_some())
        };
        assert_eq!(
            made(two_planes(1)),
            (vec![DMA_BUF_BUFFER], true),
            "NV12: a dma-buf, given"
        );
        assert_eq!(
            made(two_planes(2)),
            (vec![DMA_BUF_BUFFER], true),
            "P010: a dma-buf, given"
        );
        assert_eq!(
            made(one_plane(1024, 960, 16)),
            (vec![PADDING_UNOBSERVED_BUFFER], false),
            "one plane: as any other laid-out cache, none"
        );
    }

    /// Where the host's video imports caches (NVDEC, on a GPU of its own memory, where the kext
    /// caches the blocks and a cached block's pages carry its surface's layout), only the cache
    /// of a surface whose layout gives two planes, as the video's NV12 and P010 output has, is
    /// made exportable and gives the video its identity; a cache of a surface of one plane, or
    /// of one whose pages carry no layout, is made as any other cache and gives none, as the
    /// video writes no such surface
    #[cfg(unix)]
    #[test]
    fn only_a_two_plane_cache_is_made_exportable_for_the_hosts_video() {
        use super::super::surface_pages::tests::one_plane;
        use reims_vgpu_graphics::wire::RgfxSurfacePlaneLayout;
        use reims_vgpu_vulkan::core::testkit::impls::device::EXPORTABLE_BUFFER;
        // Two planes as CoreVideo lays them from 0x20: `sample` bytes a texel, the chroma
        // plane's rows after the luma's, interleaved and half as many.
        let two_planes = |sample: u64| {
            let mut layout = RgfxSurfacePagesLayout {
                plane_count: 2,
                ..Default::default()
            };
            layout.planes[0] = RgfxSurfacePlaneLayout {
                offset: 0x20,
                bytes_per_row: 256 * sample,
                row_bytes: 240 * sample,
                rows: 16,
            };
            layout.planes[1] = RgfxSurfacePlaneLayout {
                offset: 0x20 + 16 * 256 * sample,
                bytes_per_row: 256 * sample,
                row_bytes: 240 * sample,
                rows: 8,
            };
            layout
        };
        // How many exportable buffers a cached block's pages made, and whether its cache gives
        // its identity.
        let made = |layout: Option<RgfxSurfacePagesLayout>| {
            let mut r = rig(16);
            let page = r.page;
            r.dec.set_exportable_caches();
            let mark = r.p.call_log().mark();
            let flags = RGFX_SURFACE_PAGES_CACHED;
            assert_eq!(
                r.pages_laid_out(BLOCK, 0, &[(1, 4)], 4 * page, flags, layout),
                RGFX_STATUS_OK
            );
            let sels = r.p.call_log().sels_since(mark);
            let exportable = sels.iter().filter(|&&sel| sel == EXPORTABLE_BUFFER).count();
            assert_eq!(r.dec.caches().len(), 1);
            (exportable, r.dec.caches().identity(BLOCK).is_some())
        };
        assert_eq!(
            made(Some(two_planes(1))),
            (1, true),
            "NV12: exportable, given"
        );
        assert_eq!(
            made(Some(two_planes(2))),
            (1, true),
            "P010: exportable, given"
        );
        assert_eq!(
            made(Some(one_plane(1024, 960, 16))),
            (0, false),
            "one plane: as any other cache, none"
        );
        assert_eq!(
            made(None),
            (0, false),
            "no layout: as any other cache, none"
        );
    }

    /// A cache made for a block the device answered a cache with its surface's layout copies
    /// out its texels alone: a download of a surface whose rows hold bytes between them copies
    /// each row's texels into the pages, so the pages' bytes between rows, and those past the
    /// last row, stay as the guest's CPU wrote them however the cache's own were scribbled; a
    /// fill still copies all of the surface's bytes in
    #[test]
    fn a_download_of_a_laid_out_cache_leaves_the_bytes_between_rows() {
        use super::super::surface_pages::tests::one_plane;
        use reims_vgpu_graphics::wire::RGFX_STATUS_CACHE;
        use reims_vgpu_vulkan::core::platform::values::PagesInPlace;
        let mut r = rig(16);
        let page = r.page;
        r.dec.set_unified_memory(true);
        let world = Arc::clone(r.p.world());
        world.answer_surface_pages(PagesInPlace::No("the rows end inside a line".into()));
        // Three rows a page apart of 48 bytes fewer texels, in four pages.
        let (pitch, texels) = (page, page - 48);
        let layout = one_plane(pitch, texels, 3);
        assert_eq!(
            r.pages_asking(BLOCK, &[(1, 4)], 4 * page, Some(layout)),
            RGFX_STATUS_CACHE
        );
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 4)], 4 * page, true),
            RGFX_STATUS_OK
        );
        assert!(
            offsets(&r.p.call_log().since(mark)).is_empty(),
            "its fill owed"
        );
        queues(&mut r);
        let cache = cache_id(&r);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        let fill: Vec<(u64, u64, u64)> = offsets(&r.p.call_log().since(mark));
        assert_eq!(
            fill,
            [(0, 0, 4 * page)],
            "the first fill copies every byte in"
        );
        let mark = r.p.call_log().mark();
        r.dec
            .surface_download(&r.device, BLOCK, Box::new(|| {}), no_raise());
        let calls = r.p.call_log().since(mark);
        let out = copies(&calls);
        assert!(!out.is_empty() && out.iter().all(|&(from, _)| from == cache));

        // The pages as the copies leave them: the CPU's bytes (1) where no copy lands, the
        // cache's (0xee, every byte of it scribbled) where one does.
        let mut pages = vec![1u8; (4 * page).to_usize()];
        for (from, to, size) in offsets(&calls) {
            assert_eq!(from, to, "each byte into its own place");
            pages[from.to_usize()..(from + size).to_usize()].fill(0xee);
        }
        for row in 0..4u64 {
            let at = (row * pitch).to_usize();
            let (texel, rest) = pages[at..at + pitch.to_usize()].split_at(texels.to_usize());
            let want = if row < 3 { 0xee } else { 1 };
            assert!(texel.iter().all(|&b| b == want), "row {row}'s texels");
            assert!(
                rest.iter().all(|&b| b == 1),
                "the bytes after row {row}'s texels"
            );
        }
        assert_eq!(
            offsets(&calls),
            [
                (0, 0, texels),
                (pitch, pitch, texels),
                (2 * pitch, 2 * pitch, texels)
            ]
        );
    }

    /// A laid-out cache of a surface whose block names its pages to the end of the last one, as
    /// an arm64 guest's kext names them under `PAGES_16K` (a 1920 x 1080 surface of 8 bytes a
    /// texel, 0xfd2000 bytes, in 16 KiB pages to 0xfd4000), lands no byte past the surface's own
    /// in the pages: its fill reads the whole block's bytes, and neither a download after a GPU
    /// write nor the block's release after another copies past the last row's texels
    #[test]
    fn a_laid_out_cache_writes_nothing_past_its_surface_in_its_last_page() {
        use super::super::surface_pages::tests::one_plane;
        use reims_vgpu_graphics::wire::RGFX_STATUS_CACHE;
        use reims_vgpu_vulkan::core::platform::values::PagesInPlace;
        let mut r = rig(16);
        let page = r.page;
        r.dec.set_unified_memory(true);
        let world = Arc::clone(r.p.world());
        world.answer_surface_pages(PagesInPlace::No("the rows end inside a line".into()));
        // Five rows of half a page, no bytes between them: the surface's own bytes end half way
        // through the third page, and the block names all three.
        let (pitch, rows) = (page / 2, 5);
        let own = pitch * rows;
        let layout = one_plane(pitch, pitch, rows);
        assert_eq!(
            r.pages_asking(BLOCK, &[(1, 3)], 3 * page, Some(layout)),
            RGFX_STATUS_CACHE
        );
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 3)], 3 * page, true),
            RGFX_STATUS_OK
        );
        queues(&mut r);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        assert_eq!(
            offsets(&r.p.call_log().since(mark)),
            [(0, 0, 3 * page)],
            "the fill reads every byte the block names"
        );
        // Every copy into the pages lands inside the surface's own bytes.
        let inside = |calls: &[Call], what: &str| {
            let out = offsets(calls);
            assert!(!out.is_empty(), "{what}: copies");
            for (from, to, size) in out {
                assert_eq!(from, to, "{what}: each byte into its own place");
                assert!(
                    to + size <= own,
                    "{what}: {size:#x} bytes at {to:#x} pass {own:#x}"
                );
            }
        };
        let mark = r.p.call_log().mark();
        r.dec
            .surface_download(&r.device, BLOCK, Box::new(|| {}), no_raise());
        r.p.world().complete_all(CompleteOn::Caller);
        inside(&r.p.call_log().since(mark), "the download");
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 0, true)]);
        let mark = r.p.call_log().mark();
        r.dec
            .block_released(&r.device, BLOCK, 0, Box::new(|| {}), no_raise());
        r.p.world().complete_all(CompleteOn::Caller);
        inside(&r.p.call_log().since(mark), "the release");
    }

    /// A laid-out cache on a device that copies the guest's RAM by its runs copies its rows'
    /// texels out in one copy: each row cut to the runs its bytes lie in, a run two rows lie in
    /// cut for each, every one from its own byte of the cache
    #[test]
    fn a_laid_out_cache_by_runs_copies_its_rows_out_in_one_copy() {
        use super::super::surface_pages::tests::one_plane;
        use reims_vgpu_graphics::wire::RGFX_STATUS_CACHE;
        use reims_vgpu_vulkan::core::platform::values::PagesInPlace;
        use reims_vgpu_vulkan::core::testkit::impls::blit::BUFFER_TO_GUEST_RAM;
        let mut r = rig(16);
        let page = r.page;
        r.dec.set_unified_memory(true);
        r.p.world().copy_guest_ram(true);
        let world = Arc::clone(r.p.world());
        world.answer_surface_pages(PagesInPlace::No("the rows end inside a line".into()));
        // Three rows a page apart of 48 bytes fewer texels, in SCATTERED's four pages.
        let (pitch, texels) = (page, page - 48);
        let layout = one_plane(pitch, texels, 3);
        assert_eq!(
            r.pages_asking(BLOCK, &SCATTERED, 4 * page, Some(layout)),
            RGFX_STATUS_CACHE
        );
        assert_eq!(
            r.pages_cached(BLOCK, &SCATTERED, 4 * page, true),
            RGFX_STATUS_OK
        );
        queues(&mut r);
        let cache = cache_id(&r);
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        let mark = r.p.call_log().mark();
        r.dec
            .surface_download(&r.device, BLOCK, Box::new(|| {}), no_raise());
        let calls = r.p.call_log().since(mark);
        assert!(copies(&calls).is_empty(), "no copy through a view");
        let out: Vec<&Call> = (calls.iter())
            .filter(|c| c.sel == BUFFER_TO_GUEST_RAM)
            .collect();
        assert_eq!(out.len(), 1, "one copy for every row");
        assert_eq!(out[0].args[0].as_obj().map(|o| o.0), Some(cache));
        assert_eq!(out[0].args[1], Arg::U64s(vec![0, pitch, 2 * pitch]));
        // Row 0 in RAM page 5, rows 1 and 2 in pages 1 and 2, one run of the RAM.
        let host = |p: u64| r.ram.host_addr(RAM + p * page).unwrap().addr().to_u64();
        assert_eq!(
            copied_runs(&calls, BUFFER_TO_GUEST_RAM, 2),
            [vec![
                host(5)..host(5) + texels,
                host(1)..host(1) + texels,
                host(2)..host(2) + texels,
            ]]
        );
    }

    /// A cache made for a block the device answered a cache with its surface's layout is a buffer
    /// that declares the bytes between a texture's rows unobserved, as its copies into the pages
    /// take the texels alone, so a texture over it whose rows lie with bytes between them lies in
    /// place; a cache of a block named cached with no layout is made as any other buffer
    #[test]
    fn a_laid_out_caches_buffer_declares_the_bytes_between_rows_unobserved() {
        use super::super::surface_pages::tests::one_plane;
        use reims_vgpu_graphics::wire::RGFX_STATUS_CACHE;
        use reims_vgpu_vulkan::core::platform::values::PagesInPlace;
        use reims_vgpu_vulkan::core::testkit::impls::device::PADDING_UNOBSERVED_BUFFER;
        const NEW_BUFFER: &str = "newBufferWithLength:options:";
        let mut r = rig(16);
        let page = r.page;
        r.dec.set_unified_memory(true);
        let world = Arc::clone(r.p.world());
        world.answer_surface_pages(PagesInPlace::No("the rows end inside a line".into()));
        let made = |r: &Rig, mark: u64| {
            let calls = r.p.call_log().since(mark);
            let count = |sel: &str| calls.iter().filter(|c| c.sel == sel).count();
            (count(PADDING_UNOBSERVED_BUFFER), count(NEW_BUFFER))
        };
        let layout = one_plane(page, page - 48, 3);
        assert_eq!(
            r.pages_asking(BLOCK, &[(1, 4)], 4 * page, Some(layout)),
            RGFX_STATUS_CACHE
        );
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 4)], 4 * page, true),
            RGFX_STATUS_OK
        );
        assert_eq!(made(&r, mark), (1, 0), "laid out: declared");

        let other = BLOCK + 0x8000;
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.pages_cached(other, &[(8, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        assert!(r.dec.caches().has(other));
        assert_eq!(made(&r, mark), (0, 1), "no layout: as any other buffer");
    }

    /// The (source offset, destination offset, size) of each copy in `calls`.
    fn offsets(calls: &[Call]) -> Vec<(u64, u64, u64)> {
        (calls.iter())
            .filter(|c| c.sel == COPY)
            .map(|c| {
                let at = |i: usize| c.args[i].as_u64().unwrap_or(u64::MAX);
                (at(1), at(3), at(4))
            })
            .collect()
    }

    /// The bytes of each plane's rows' texels, merged where rows follow on, in the surface's
    /// order: two planes of rows with bytes between, one whose rows follow on, and rows of no
    /// texels
    #[test]
    #[allow(clippy::single_range_in_vec_init)] // spans of one range
    fn texel_spans_are_each_plane_s_rows() {
        let plane = |offset, bytes_per_row, row_bytes, rows| PagesPlane {
            offset,
            bytes_per_row,
            row_bytes,
            rows,
        };
        assert_eq!(
            texel_spans(&[plane(4096, 64, 48, 2), plane(0, 256, 200, 2)]),
            [0..200, 256..456, 4096..4144, 4160..4208]
        );
        assert_eq!(texel_spans(&[plane(0, 64, 64, 4)]), [0..256]);
        assert_eq!(texel_spans(&[plane(0, 64, 0, 4)]), []);
    }

    /// The runs the spans of a surface's bytes lie in, cut to them, each with its first byte of
    /// the surface: a run that two spans lie in is cut for each, and a run no span reaches is left
    /// out
    #[test]
    #[allow(clippy::single_range_in_vec_init)] // spans of one range
    fn runs_at_spans_are_cut_to_them() {
        let runs = [0x1000..0x2000, 0x8000..0x9000, 0x4000..0x5000];
        assert_eq!(
            runs_at(&runs, &[0x800..0x1800]),
            [(0x800, 0x1800..0x2000), (0x1000, 0x8000..0x8800)]
        );
        assert_eq!(
            runs_at(&runs, &[0x2000..0x2100]),
            [(0x2000, 0x4000..0x4100)]
        );
        assert_eq!(runs_at(&runs, &[0..0x1000]), [(0, 0x1000..0x2000)]);
        assert_eq!(
            runs_at(&runs, &[0x100..0x200, 0x300..0x1100, 0x2f00..0x3000]),
            [
                (0x100, 0x1100..0x1200),
                (0x300, 0x1300..0x2000),
                (0x1000, 0x8000..0x8100),
                (0x2f00, 0x4f00..0x5000),
            ]
        );
        assert_eq!(
            runs_at(&runs, &[0x2f00..0x3100]),
            [(0x2f00, 0x4f00..0x5000)]
        );
        assert_eq!(runs_at(&runs, &[]), []);
    }

    /// A copy out of a cache by runs that the platform refuses, at a download, makes the block's
    /// view there and then and copies the cache into the pages through it, so the command
    /// buffer's write is not lost, and the release that follows has nothing more to copy
    #[test]
    fn a_refused_download_by_runs_copies_through_a_view() {
        use reims_vgpu_vulkan::core::testkit::Step;
        use reims_vgpu_vulkan::core::testkit::impls::blit::BUFFER_TO_GUEST_RAM;
        let mut r = cached_by_runs();
        let cache = cache_id(&r);
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        r.p.script().push(BUFFER_TO_GUEST_RAM, Step::value(false));
        let done = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&done);
        let mark = r.p.call_log().mark();
        r.dec.surface_download(
            &r.device,
            BLOCK,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        let calls = r.p.call_log().since(mark);
        let sels: Vec<&str> = (calls.iter())
            .map(|c| c.sel)
            .filter(|s| [BUFFER_TO_GUEST_RAM, "newBufferWithDescriptor:", COPY].contains(s))
            .collect();
        assert_eq!(
            sels,
            [BUFFER_TO_GUEST_RAM, "newBufferWithDescriptor:", COPY]
        );
        let out = copies(&calls);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, cache, "out of the cache, into the view");
        assert!(!done.load(Ordering::SeqCst));
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst), "once the pages hold the write");
        let released = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&released);
        let mark = r.p.call_log().mark();
        r.dec.block_released(
            &r.device,
            BLOCK,
            0,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        assert!(released.load(Ordering::SeqCst), "nothing more to copy");
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        let counts = r.dec.caches().counts();
        assert_eq!((counts.downloads, counts.runs_refused), (1, 1));
    }

    /// A block named by its runs that the platform makes no cache for takes its view then, and
    /// its buffers lie over the view
    #[test]
    fn a_block_by_runs_with_no_cache_takes_its_view() {
        use reims_vgpu_vulkan::core::testkit::Step;
        let mut r = rig(8);
        let page = r.page;
        r.p.world().copy_guest_ram(true);
        r.p.script().push("newBufferWithLength:options:", Step::Nil);
        assert_eq!(
            r.pages_cached(BLOCK, &SCATTERED, 4 * page, true),
            RGFX_STATUS_OK
        );
        assert_eq!(r.dec.caches().len(), 0);
        assert_eq!(r.dec.views_of_surfaces().len().0, 1, "its view");
        assert_eq!(r.buffer(10, BLOCK, 4 * page), Ok(()));
    }

    /// Each transfer command buffer's copies of the block's cache, as (copies by runs asked,
    /// copies through a view), from `commandBuffer` to its `commit`.
    fn copies_per_command_buffer(calls: &[Call]) -> Vec<(usize, usize)> {
        use reims_vgpu_vulkan::core::testkit::impls::blit::{
            BUFFER_TO_GUEST_RAM, GUEST_RAM_TO_BUFFER,
        };
        let mut out = Vec::new();
        let mut open: Option<(usize, usize)> = None;
        for c in calls {
            match c.sel {
                "commandBuffer" => open = Some((0, 0)),
                GUEST_RAM_TO_BUFFER | BUFFER_TO_GUEST_RAM => {
                    if let Some(o) = open.as_mut() {
                        o.0 += 1;
                    }
                }
                COPY => {
                    if let Some(o) = open.as_mut() {
                        o.1 += 1;
                    }
                }
                "commit" => out.extend(open.take().filter(|&o| o != (0, 0))),
                _ => {}
            }
        }
        out
    }

    /// A copy by runs and one through the view never share a command buffer, as the runtime
    /// orders the one against nothing over the other's mapping: a fill by runs has its own, a
    /// download whose copy by runs is refused copies through the view in its own, the refused
    /// copy having encoded nothing, and every copy after it goes through the view, each in its own
    #[test]
    fn a_copy_by_runs_and_one_through_the_view_never_share_a_command_buffer() {
        use reims_vgpu_vulkan::core::testkit::Step;
        use reims_vgpu_vulkan::core::testkit::impls::blit::BUFFER_TO_GUEST_RAM;
        let mut r = cached_by_runs();
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        r.p.script().push(BUFFER_TO_GUEST_RAM, Step::value(false));
        let done = download(&r, BLOCK);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
        use_surfaces(&mut r, 2, 0x101, &[(BLOCK, 2, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let done = download(&r, BLOCK);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
        assert_eq!(
            copies_per_command_buffer(&r.p.call_log().since(mark)),
            [(1, 0), (1, 1), (0, 1), (0, 1)],
            "a fill by runs; the refused download, through the view; a fill and a download \
             through it"
        );
        assert_eq!(r.dec.caches().counts().runs_refused, 1);
    }

    /// A deferred block's pages, named on a device that copies the guest's RAM by its runs, are
    /// taken by their runs, with no view: the cache keeps its bytes, and the download that
    /// follows copies the GPU's write out of it by them
    #[test]
    fn a_deferred_blocks_pages_are_taken_by_their_runs() {
        use reims_vgpu_vulkan::core::testkit::impls::blit::BUFFER_TO_GUEST_RAM;
        let mut r = deferring(8);
        let page = r.page;
        r.p.world().copy_guest_ram(true);
        assert_eq!(defer(&mut r, BLOCK, 4 * page), RGFX_STATUS_OK);
        let cache = cache_id(&r);
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            attach(&mut r, BLOCK, 3, &SCATTERED, 4 * page),
            RGFX_STATUS_OK
        );
        assert_eq!(r.dec.views_of_surfaces().len(), (0, 0), "no view");
        assert_eq!(cache_id(&r), cache);
        assert_eq!(r.dec.caches().counts().attached, 1);
        let mark = r.p.call_log().mark();
        let done = download(&r, BLOCK);
        let calls = r.p.call_log().since(mark);
        assert!(copies(&calls).is_empty(), "no copy through a view");
        let out = copied_runs(&calls, BUFFER_TO_GUEST_RAM, 2);
        assert_eq!(out.len(), 1);
        let from = (calls.iter())
            .find(|c| c.sel == BUFFER_TO_GUEST_RAM)
            .unwrap();
        assert_eq!(from.args[0].as_obj().map(|o| o.0), Some(cache));
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
    }

    /// A use of a block with no cache does nothing
    #[test]
    fn a_use_of_a_block_without_a_cache_does_nothing() {
        let mut r = cached();
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK + 0x8000, 0, true)]);
        let calls = r.p.call_log().since(mark);
        assert!(copies(&calls).is_empty());
        assert!(events(&calls, SIGNAL).is_empty());
    }

    /// `fillBuffer:range:value:`.
    const FILL: &str = "fillBuffer:range:value:";

    /// A `SURFACE_PAGES` of the deferred block at `block`, of `length` bytes and no runs; its
    /// answer.
    fn defer(r: &mut Rig, block: u64, length: u64) -> u32 {
        let flags = RGFX_SURFACE_PAGES_CACHED | RGFX_SURFACE_PAGES_DEFERRED;
        r.pages_flagged(block, 0, &[], length, flags)
    }

    /// A `SURFACE_PAGES` naming the pages of the deferred block at `block`, `runs` as
    /// [`Rig::pages`]'s, from context `context`; its answer.
    fn attach(r: &mut Rig, block: u64, context: u32, runs: &[(u64, u64)], length: u64) -> u32 {
        let flags = RGFX_SURFACE_PAGES_CACHED | RGFX_SURFACE_PAGES_ATTACH;
        r.pages_flagged(block, context, runs, length, flags)
    }

    /// A rig of [`rig`]'s with a deferred budget of `budget` pages, and command queues 1 and 2.
    fn deferring(budget: u64) -> Rig {
        let mut r = rig(8);
        r.dec.set_deferred_budget(budget * r.page);
        queues(&mut r);
        r
    }

    /// The `SURFACE_PAGES` of the deferred block at `block`, of `length` bytes and no runs, as
    /// [`defer`] sends it.
    fn deferral(block: u64, length: u64) -> RgfxSurfacePages {
        RgfxSurfacePages {
            header: RgfxMsgHeader {
                r#type: reims_vgpu_graphics::wire::RGFX_CMD_SURFACE_PAGES,
                size: size_of::<RgfxSurfacePages>() as u32,
                seqno: 0,
            },
            context: 0,
            run_count: 0,
            block_offset: block,
            block_size: 0x2000,
            runs_offset: block + 0x1000,
            length,
        }
    }

    /// Holds the next cache's buffer the platform is asked to make (one of no layout, made as
    /// any buffer), as the driver holds a make that takes a new block of the device's memory:
    /// the hold.
    fn hold_next_cache_make(r: &Rig) -> HeldCall {
        let made = HeldCall::new();
        r.p.faults().at(
            "MTLDevice",
            "newBufferWithLength:options:",
            0,
            Fault::Hold(made.clone()),
        );
        made
    }

    /// A cache's buffer is made with the caches' lock let go of: while the platform holds a
    /// deferred block's make, as a driver holds one that takes a new block of the device's
    /// memory (8 to 28 ms on a Radeon 780M), a command buffer that names another block's cache is
    /// decoded, rather than waiting for the make to end; the deferral then gets its cache
    #[test]
    fn a_begin_waits_for_no_cache_s_make() {
        let r = cached();
        let page = r.page;
        r.dec.set_deferred_budget(8 * page);
        let made = hold_next_cache_make(&r);
        let deferred = BLOCK + 0x8000;
        let m = deferral(deferred, 2 * page);
        let flags = RGFX_SURFACE_PAGES_CACHED | RGFX_SURFACE_PAGES_DEFERRED;
        let message = surfaces_message(1, 0x100, &[(BLOCK, 0, false)], true);
        let (tx, rx) = std::sync::mpsc::channel();
        let answer = std::thread::scope(|scope| {
            let deferring = r.dec.sender(None, super::super::Sinks::default());
            let device = r.device.clone_ref();
            let deferral = scope
                .spawn(move || deferring.surface_pages_laid_out(&device, &m, &[], flags, None));
            assert!(
                made.reached(std::time::Duration::from_secs(5)),
                "the deferral's cache is being made"
            );
            let mut begin = r.dec.sender(None, super::super::Sinks::default());
            let device = r.device.clone_ref();
            scope.spawn(move || tx.send(begin.decode(&device, &message).unwrap()).unwrap());
            let decoded = rx.recv_timeout(std::time::Duration::from_secs(2));
            made.open();
            assert_eq!(decoded, Ok(Ok(())), "decoded while the make was held");
            deferral.join().unwrap()
        });
        assert_eq!(answer, RGFX_STATUS_OK, "the deferral's answer");
        let caches = r.dec.caches();
        assert!(caches.has(deferred), "its cache, taken once made");
        assert_eq!(caches.counts().deferred, 1);
    }

    /// A deferred block's cache is made where the deferred caches' budget has room for it, with
    /// the caches' lock let go of, and taken under it where the budget still has room: one whose
    /// room another deferral took while it was made is refused, as a deferral without room is,
    /// counted, and its buffer let go of; the other keeps its cache
    #[test]
    fn a_deferral_whose_room_went_while_its_cache_was_made_is_refused() {
        let r = deferring(2);
        let page = r.page;
        let buffers = r.p.world().live("MTLBuffer").len();
        let made = hold_next_cache_make(&r);
        let (first, second) = (BLOCK, BLOCK + 0x8000);
        let flags = RGFX_SURFACE_PAGES_CACHED | RGFX_SURFACE_PAGES_DEFERRED;
        let (tx, rx) = std::sync::mpsc::channel();
        let answers = std::thread::scope(|scope| {
            let deferring = r.dec.sender(None, super::super::Sinks::default());
            let device = r.device.clone_ref();
            let m = deferral(first, 2 * page);
            let held = scope
                .spawn(move || deferring.surface_pages_laid_out(&device, &m, &[], flags, None));
            assert!(
                made.reached(std::time::Duration::from_secs(5)),
                "the first deferral's cache is being made"
            );
            // The second takes the budget's room meanwhile, its own make not held.
            let deferring = r.dec.sender(None, super::super::Sinks::default());
            let device = r.device.clone_ref();
            let m = deferral(second, 2 * page);
            scope.spawn(move || {
                let answer = deferring.surface_pages_laid_out(&device, &m, &[], flags, None);
                tx.send(answer).unwrap();
            });
            let second_answer = rx.recv_timeout(std::time::Duration::from_secs(2));
            made.open();
            (held.join().unwrap(), second_answer)
        });
        assert_eq!(answers, (RGFX_STATUS_REFUSED, Ok(RGFX_STATUS_OK)));
        let caches = r.dec.caches();
        assert!(!caches.has(first), "no cache for the refused deferral");
        assert!(caches.has(second));
        let counts = caches.counts();
        assert_eq!((counts.deferred, counts.deferred_refused), (1, 1));
        drop(caches);
        assert_eq!(
            r.p.world().live("MTLBuffer").len(),
            buffers + 1,
            "the second's cache alone: the refused one's buffer let go of"
        );
    }

    /// Command queues 1 and 2.
    fn queues(r: &mut Rig) {
        for obj_id in [1, 2] {
            let queue = RgfxQueueCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_QUEUE_CREATE,
                    size: size_of::<RgfxQueueCreate>() as u32,
                    seqno: 0,
                },
                obj_id,
                ..RgfxQueueCreate::default()
            };
            assert_eq!(r.dec.decode(&r.device, queue.as_bytes()).unwrap(), Ok(()));
        }
    }

    /// `SURFACE_DOWNLOAD` of the block at `block`, and whether it was answered.
    fn download(r: &Rig, block: u64) -> Arc<AtomicBool> {
        let done = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&done);
        r.dec.surface_download(
            &r.device,
            block,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        done
    }

    /// `BLOCK_RELEASE` of the block at `block`, and whether it was answered.
    fn release(r: &Rig, block: u64) -> Arc<AtomicBool> {
        let done = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&done);
        r.dec.block_released(
            &r.device,
            block,
            0,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        done
    }

    /// A deferred block gets a device cache and no view: the cache is zeroed on the transfer
    /// queue, which a use waits for, at the CPU's generation 0, so a use of generation 0 fills
    /// nothing; its buffers are the cache. Its release, its pages never named, copies nothing
    #[test]
    fn a_deferred_block_is_a_cache_of_zeros_with_no_pages() {
        let mut r = deferring(8);
        let page = r.page;
        let mark = r.p.call_log().mark();
        assert_eq!(defer(&mut r, BLOCK, 2 * page), RGFX_STATUS_OK);
        let calls = r.p.call_log().since(mark);
        let cache = cache_id(&r);
        let zeroed: Vec<&Call> = calls.iter().filter(|c| c.sel == FILL).collect();
        assert_eq!(zeroed.len(), 1);
        assert_eq!(zeroed[0].args[0].as_obj().map(|o| o.0), Some(cache));
        assert_eq!(zeroed[0].args[1], Arg::val(&NsRange::new(0, 2 * page)));
        assert_eq!(zeroed[0].args[2], Arg::U8(0));
        let zero_done = events(&calls, SIGNAL);
        assert_eq!(zero_done.len(), 1, "the zeroing signals the transfer event");
        assert_eq!(r.dec.views_of_surfaces().len(), (0, 0), "no view");
        assert_eq!(r.dec.caches().counts().deferred, 1);
        assert_eq!(r.dec.caches().unattached(BLOCK), Some(2 * page));

        let mark = r.p.call_log().mark();
        assert_eq!(r.buffer(10, BLOCK, 2 * page), Ok(()));
        assert!(
            r.p.call_log().sels_since(mark).is_empty(),
            "no buffer made: it is the cache"
        );
        assert_eq!(
            r.buffer(11, BLOCK, 3 * page),
            Err(DecodeError::Refused),
            "past the cache"
        );
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        let calls = r.p.call_log().since(mark);
        assert!(copies(&calls).is_empty(), "nothing to fill from");
        assert_eq!(events(&calls, WAIT), zero_done, "after the zeroing");

        let mark = r.p.call_log().mark();
        let released = release(&r, BLOCK);
        assert!(released.load(Ordering::SeqCst), "at once");
        assert!(copies(&r.p.call_log().since(mark)).is_empty(), "no pages");
        assert_eq!(r.dec.caches().len(), 0);
        assert_eq!(r.dec.caches().counts().unattached, 1);
    }

    /// A deferred block's pages, named at the first CPU lock that needs them and from any
    /// context, are taken for its cache, which keeps its bytes and its generation: a use of
    /// generation 0 still fills nothing, and the download that follows copies the GPU's write
    /// from the same cache into the pages. A use after a CPU write's lock then fills the cache
    /// from them, as any cache's
    #[test]
    fn a_deferred_blocks_pages_keep_its_bytes_and_generation() {
        let mut r = deferring(8);
        let page = r.page;
        assert_eq!(defer(&mut r, BLOCK, 2 * page), RGFX_STATUS_OK);
        let cache = cache_id(&r);
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        r.p.world().complete_all(CompleteOn::Caller);

        let mark = r.p.call_log().mark();
        assert_eq!(
            attach(&mut r, BLOCK, 7, &[(1, 2)], 2 * page),
            RGFX_STATUS_OK
        );
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "nothing copied"
        );
        assert_eq!(cache_id(&r), cache, "the same cache");
        assert_eq!(r.dec.caches().unattached(BLOCK), None);
        assert_eq!(r.dec.caches().counts().attached, 1);
        assert_eq!(r.dec.views_of_surfaces().len().0, 1, "its view");

        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 2, 0x101, &[(BLOCK, 0, false)]);
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "generation 0 still"
        );
        r.p.world().complete_all(CompleteOn::Caller);

        let mark = r.p.call_log().mark();
        let done = download(&r, BLOCK);
        let out = copies(&r.p.call_log().since(mark));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, cache, "the GPU's write, out of the cache");
        assert_ne!(out[0].1, cache);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));

        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x102, &[(BLOCK, 1, false)]);
        let filled = copies(&r.p.call_log().since(mark));
        assert_eq!(filled, vec![(out[0].1, cache)], "from the pages");
    }

    /// A `SURFACE_PAGES` of the deferred block at `block`, of `length` bytes and no runs, that
    /// asks with its surface's layout `layout`, as a kext does where the device answers which
    /// surfaces need a device cache; its answer.
    fn defer_asking(
        r: &mut Rig,
        block: u64,
        length: u64,
        layout: Option<RgfxSurfacePagesLayout>,
    ) -> u32 {
        let flags =
            RGFX_SURFACE_PAGES_CACHED | RGFX_SURFACE_PAGES_DEFERRED | RGFX_SURFACE_PAGES_ASK_CACHE;
        r.pages_laid_out(block, 0, &[], length, flags, layout)
    }

    /// A rig of [`deferring`]'s whose GPU shares the CPU's memory, its device answering which
    /// surfaces need a device cache and deferring the fresh ones that would whatever their pages,
    /// as `SURFACE_DEFERRED_ASK` has it.
    fn deferring_asking(budget: u64) -> Rig {
        let mut r = deferring(budget);
        r.dec.set_unified_memory(true);
        r.dec.set_deferred_asks();
        r
    }

    /// One plane of two rows of a page apart whose texels end 16 bytes short of the next row.
    fn short_rows(page: u64) -> (RgfxSurfacePagesLayout, Vec<PagesPlane>) {
        use super::super::surface_pages::tests::one_plane;
        let plane = PagesPlane {
            offset: 0,
            bytes_per_row: page,
            row_bytes: page - 16,
            rows: 2,
        };
        (one_plane(page, page - 16, 2), vec![plane])
    }

    /// On a GPU that shares the CPU's memory, a fresh block that asks to be deferred is asked of
    /// the device with its planes alone, no page named: where textures over them would lie in
    /// place over no import of any pages, it is a cache of zeros with no view, as a deferred
    /// block of a GPU of its own memory is. Where they may lie in place, it is refused with
    /// nothing made, counted, and its pages then ask whether it needs a cache as any block's do
    #[test]
    fn a_fresh_block_asking_is_deferred_only_where_it_would_need_a_cache() {
        let mut r = deferring_asking(8);
        let page = r.page;
        let world = Arc::clone(r.p.world());
        let (layout, planes) = short_rows(page);
        world.answer_fresh_surfaces(Some("the rows end inside a line".into()));
        let mark = r.p.call_log().mark();
        assert_eq!(
            defer_asking(&mut r, BLOCK, 2 * page, Some(layout)),
            RGFX_STATUS_OK
        );
        assert_eq!(world.fresh_surfaces_asked(), [planes]);
        let calls = r.p.call_log().since(mark);
        let cache = cache_id(&r);
        let zeroed: Vec<&Call> = calls.iter().filter(|c| c.sel == FILL).collect();
        assert_eq!(zeroed.len(), 1, "zeroed");
        assert_eq!(zeroed[0].args[0].as_obj().map(|o| o.0), Some(cache));
        assert_eq!(r.dec.views_of_surfaces().len(), (0, 0), "no view");
        assert!(world.surface_pages_asked().is_empty(), "no page asked of");
        assert_eq!(r.dec.caches().unattached(BLOCK), Some(2 * page));
        assert_eq!(r.dec.caches().counts().deferred, 1);

        world.answer_fresh_surfaces(None);
        let second = BLOCK + 0x8000;
        let mark = r.p.call_log().mark();
        assert_eq!(
            defer_asking(&mut r, second, 2 * page, Some(layout)),
            RGFX_STATUS_REFUSED
        );
        assert!(r.p.call_log().sels_since(mark).is_empty(), "nothing made");
        assert!(!r.dec.caches().has(second));
        let counts = r.dec.caches().counts();
        assert_eq!((counts.deferred, counts.deferred_in_place), (1, 1));
        world.answer_surface_pages(PagesInPlace::Yes);
        assert_eq!(
            r.pages_asking(second, &[(5, 2)], 2 * page, Some(layout)),
            RGFX_STATUS_OK,
            "in place"
        );
        assert!(r.dec.views_of_surfaces().view(second).is_some());
        assert!(!r.dec.caches().has(second));
    }

    /// A deferral that asks fails safe at a full deferred budget: it is refused with nothing
    /// made, as a deferral of a GPU of its own memory is, and the block's pages then ask whether
    /// it needs a device cache as they do on a device that defers nothing: answered a cache, its
    /// cached `SURFACE_PAGES` makes one over them. The room a deferred block gave back takes the
    /// next
    #[test]
    fn a_full_deferred_budget_has_the_pages_ask_as_without_deferral() {
        let mut r = deferring_asking(2);
        let page = r.page;
        let world = Arc::clone(r.p.world());
        let (layout, _) = short_rows(page);
        world.answer_fresh_surfaces(Some("the rows end inside a line".into()));
        assert_eq!(
            defer_asking(&mut r, BLOCK, 2 * page, Some(layout)),
            RGFX_STATUS_OK
        );
        let second = BLOCK + 0x8000;
        assert_eq!(
            defer_asking(&mut r, second, 2 * page, Some(layout)),
            RGFX_STATUS_REFUSED,
            "past the budget"
        );
        assert!(!r.dec.caches().has(second));
        assert_eq!(r.dec.caches().counts().deferred_refused, 1);
        assert_eq!(r.dec.views_of_surfaces().len(), (0, 0), "no view");

        world.answer_surface_pages(PagesInPlace::No("the rows end inside a line".into()));
        assert_eq!(
            r.pages_asking(second, &[(5, 2)], 2 * page, Some(layout)),
            RGFX_STATUS_CACHE
        );
        assert_eq!(
            r.pages_cached(second, &[(5, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        assert!(r.dec.caches().has(second));
        assert_eq!(r.dec.caches().unattached(second), None, "over its pages");

        let released = release(&r, BLOCK);
        assert!(released.load(Ordering::SeqCst));
        let third = BLOCK + 0x10000;
        assert_eq!(
            defer_asking(&mut r, third, 2 * page, Some(layout)),
            RGFX_STATUS_OK,
            "the room is back"
        );
    }

    /// Where the device answers which surfaces need a device cache, a deferral that does not
    /// ask, or asks with no layout or one that does not lie in the surface's bytes, is an
    /// invalid argument, and asks the device nothing; where it does not answer them, one that
    /// asks is. Neither makes a cache
    #[test]
    fn a_deferral_asks_exactly_where_the_device_answers_asks() {
        use super::super::surface_pages::tests::one_plane;
        let mut r = deferring_asking(8);
        let page = r.page;
        let world = Arc::clone(r.p.world());
        let (layout, _) = short_rows(page);
        world.answer_fresh_surfaces(Some("the rows end inside a line".into()));
        let invalid = RGFX_STATUS_INVALID_ARGUMENT;
        assert_eq!(defer(&mut r, BLOCK, 2 * page), invalid, "no ask");
        let past = one_plane(page, page, 3);
        let empty = one_plane(page, 0, 2);
        for bad in [Some(past), Some(empty), None] {
            assert_eq!(defer_asking(&mut r, BLOCK, 2 * page, bad), invalid);
        }
        assert!(world.fresh_surfaces_asked().is_empty());
        assert!(!r.dec.caches().has(BLOCK));

        let mut r = deferring(8);
        assert_eq!(
            defer_asking(&mut r, BLOCK, 2 * page, Some(layout)),
            invalid,
            "asked of a device that answers no asks"
        );
        assert!(!r.dec.caches().has(BLOCK));
    }

    /// A deferred block that asked keeps the cache it was given through the CPU's first lock:
    /// the lock's pages are taken for the cache, which keeps its bytes and its generation, and
    /// the download that follows copies the GPU's write from the cache into the pages, the
    /// texels of each row alone, as an asked cache's downloads do; a use after a CPU write's lock
    /// fills the cache from them, as any cache's
    #[test]
    fn a_deferred_block_that_asked_is_read_and_written_through_its_pages() {
        let mut r = deferring_asking(8);
        let page = r.page;
        let (layout, _) = short_rows(page);
        r.p.world()
            .answer_fresh_surfaces(Some("the rows end inside a line".into()));
        assert_eq!(
            defer_asking(&mut r, BLOCK, 2 * page, Some(layout)),
            RGFX_STATUS_OK
        );
        let cache = cache_id(&r);
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        r.p.world().complete_all(CompleteOn::Caller);

        let mark = r.p.call_log().mark();
        assert_eq!(
            attach(&mut r, BLOCK, 7, &[(1, 2)], 2 * page),
            RGFX_STATUS_OK
        );
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "nothing copied"
        );
        assert_eq!(cache_id(&r), cache, "the same cache");
        assert_eq!(r.dec.caches().unattached(BLOCK), None);
        assert_eq!(r.dec.caches().counts().attached, 1);

        let mark = r.p.call_log().mark();
        let done = download(&r, BLOCK);
        let out = copies(&r.p.call_log().since(mark));
        let pages = out[0].1;
        assert_ne!(pages, cache);
        assert_eq!(
            out,
            vec![(cache, pages); 2],
            "the GPU's write, out of the cache, each row's texels as the layout has them"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));

        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 1, false)]);
        let filled = copies(&r.p.call_log().since(mark));
        assert!(!filled.is_empty(), "filled");
        assert!(
            filled.iter().all(|&c| c == (pages, cache)),
            "from the pages: {filled:?}"
        );
    }

    /// Pages named for a block with no deferred cache whose pages are not named (none, one
    /// with pages, or a deferred one named already) are an invalid argument, and make no view
    /// and change no cache; so is a message with both flags, a deferred block's without `CACHED`
    /// or with runs
    #[test]
    fn a_stray_attach_is_an_invalid_argument() {
        let mut r = deferring(8);
        let page = r.page;
        let invalid = RGFX_STATUS_INVALID_ARGUMENT;
        assert_eq!(attach(&mut r, BLOCK, 0, &[(1, 2)], 2 * page), invalid);
        assert_eq!(r.dec.views_of_surfaces().len(), (0, 0));
        assert_eq!(r.dec.caches().len(), 0);

        let other = BLOCK + 0x8000;
        assert_eq!(r.pages_cached(other, &[(3, 1)], page, true), RGFX_STATUS_OK);
        let cache = r.dec.caches().buffer(other).map(Arg::obj);
        assert_eq!(attach(&mut r, other, 0, &[(4, 1)], page), invalid);
        assert_eq!(r.dec.caches().buffer(other).map(Arg::obj), cache);

        assert_eq!(defer(&mut r, BLOCK, 2 * page), RGFX_STATUS_OK);
        assert_eq!(
            attach(&mut r, BLOCK, 0, &[(1, 2)], 2 * page),
            RGFX_STATUS_OK
        );
        assert_eq!(
            attach(&mut r, BLOCK, 0, &[(5, 2)], 2 * page),
            invalid,
            "named already"
        );

        let both =
            RGFX_SURFACE_PAGES_CACHED | RGFX_SURFACE_PAGES_DEFERRED | RGFX_SURFACE_PAGES_ATTACH;
        let third = BLOCK + 0x10000;
        assert_eq!(r.pages_flagged(third, 0, &[], page, both), invalid);
        assert_eq!(
            r.pages_flagged(third, 0, &[], page, RGFX_SURFACE_PAGES_DEFERRED),
            invalid
        );
        let deferred = RGFX_SURFACE_PAGES_CACHED | RGFX_SURFACE_PAGES_DEFERRED;
        assert_eq!(
            r.pages_flagged(third, 0, &[(6, 1)], page, deferred),
            invalid
        );
        assert!(!r.dec.caches().has(third));
    }

    /// A rig of [`deferring`]'s whose device lets a deferral carry its surface's resident pages,
    /// as `SURFACE_DEFERRED_CARRY` has it.
    fn deferring_carrying(budget: u64) -> Rig {
        let r = deferring(budget);
        r.dec.set_deferred_carries();
        r
    }

    /// The runs of a carrying deferral of a surface of `pages` pages: each surface page of
    /// `carried` (the surface's page, a guest RAM page) that page of guest RAM, every other a
    /// run of zeros.
    fn carrying_runs(r: &Rig, pages: u64, carried: &[(u64, u64)]) -> Vec<RgfxPageRun> {
        (0..pages)
            .map(|at| RgfxPageRun {
                address: (carried.iter())
                    .find(|&&(page, _)| page == at)
                    .map_or(RGFX_PAGE_RUN_ZEROS, |&(_, ram)| RAM + ram * r.page),
                length: r.page,
            })
            .collect()
    }

    /// A `SURFACE_PAGES` of the deferred block at `block`, of `pages` pages, carrying `carried`
    /// ([`carrying_runs`]), asking with `layout` where given; its answer.
    fn defer_carrying(
        r: &mut Rig,
        block: u64,
        pages: u64,
        carried: &[(u64, u64)],
        layout: Option<RgfxSurfacePagesLayout>,
    ) -> u32 {
        let runs = carrying_runs(r, pages, carried);
        let ask = if layout.is_some() {
            RGFX_SURFACE_PAGES_ASK_CACHE
        } else {
            0
        };
        let flags = RGFX_SURFACE_PAGES_CACHED | RGFX_SURFACE_PAGES_DEFERRED | ask;
        r.pages_of_runs(block, 0, &runs, pages * r.page, flags, layout)
    }

    /// Each copy in `calls`: its source, source offset, destination, destination offset and size.
    fn copies_at(calls: &[Call]) -> Vec<(u64, u64, u64, u64, u64)> {
        let id = |a: &Arg| a.as_obj().map_or(0, |o| o.0);
        let at = |a: &Arg| a.as_u64().unwrap_or(u64::MAX);
        (calls.iter())
            .filter(|c| c.sel == COPY)
            .map(|c| {
                let a = &c.args;
                (id(&a[0]), at(&a[1]), id(&a[2]), at(&a[3]), at(&a[4]))
            })
            .collect()
    }

    /// A deferral that carries its surface's resident pages is a cache of zeros, with no view,
    /// that owes a first fill of those pages alone: nothing is copied at the deferral, and the
    /// first use that reads the cache, at the CPU's generation 0, copies each carried page in at
    /// its own bytes of the cache, through a Shared buffer the host's CPU filled from the page as
    /// the fill was made, the rest of the cache left as its zeroing left it. The cache then holds
    /// generation 0, so a use after it fills nothing
    #[test]
    fn a_carrying_deferral_fills_its_pages_in_before_a_read() {
        let mut r = deferring_carrying(8);
        let page = r.page;
        let mark = r.p.call_log().mark();
        assert_eq!(
            defer_carrying(&mut r, BLOCK, 3, &[(1, 4)], None),
            RGFX_STATUS_OK
        );
        let calls = r.p.call_log().since(mark);
        let cache = cache_id(&r);
        assert_eq!(calls.iter().filter(|c| c.sel == FILL).count(), 1, "zeroed");
        assert!(copies(&calls).is_empty(), "nothing copied at the deferral");
        assert_eq!(r.dec.views_of_surfaces().len(), (0, 0), "no view");
        let counts = r.dec.caches().counts();
        assert_eq!((counts.deferred, counts.carried), (1, 1));
        assert_eq!(r.dec.caches().unattached(BLOCK), Some(3 * page));

        // The page as the fill finds it, written after the deferral.
        let ram = ram_bytes(&r, 4, page, Some(0x3c));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        let filled = copies_at(&r.p.call_log().since(mark));
        assert_eq!(filled.len(), 1, "the carried page alone: {filled:?}");
        let (staging, from, into, at, size) = filled[0];
        assert_ne!(staging, cache);
        assert_eq!(
            (from, into, at, size),
            (0, cache, page, page),
            "at its own bytes"
        );
        assert_eq!(buffer_bytes(&r, staging, None), ram, "the page's bytes");
        r.p.world().complete_all(CompleteOn::Caller);
        let counts = r.dec.caches().counts();
        assert_eq!((counts.fills, counts.made_fills), (1, 1));

        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 2, 0x101, &[(BLOCK, 0, true)]);
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "generation 0 held"
        );
    }

    /// A carrying deferral's pages that follow one another both in the surface and in guest RAM
    /// fill in with one copy, the host's guest pages of each run merged, and a page apart from the
    /// one before it in either with its own: surface pages 0 and 1 at RAM pages 2 and 3 are one
    /// copy, page 2 at RAM page 5 (apart in RAM) another, and page 4 at RAM page 6 (apart in the
    /// surface) a third, each at its own bytes of the cache and through its own bytes of the
    /// Shared buffer
    #[test]
    fn a_carrying_deferral_copies_each_run_of_its_pages_once() {
        let mut r = deferring_carrying(8);
        let page = r.page;
        assert_eq!(
            defer_carrying(&mut r, BLOCK, 5, &[(0, 2), (1, 3), (2, 5), (4, 6)], None),
            RGFX_STATUS_OK
        );
        let cache = cache_id(&r);
        let mut ram = ram_bytes(&r, 2, 2 * page, Some(0x5a));
        ram.extend(ram_bytes(&r, 5, 2 * page, Some(0xa5)));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        let filled = copies_at(&r.p.call_log().since(mark));
        let staging = filled.first().map_or(0, |c| c.0);
        assert_eq!(
            filled,
            [
                (staging, 0, cache, 0, 2 * page),
                (staging, 2 * page, cache, 2 * page, page),
                (staging, 3 * page, cache, 4 * page, page)
            ],
            "one copy a run"
        );
        assert_eq!(
            buffer_bytes(&r, staging, None),
            ram,
            "the pages' bytes, in order"
        );
    }

    /// A carrying deferral that asks with its surface's layout fills its carried page in whole,
    /// as every fill copies every byte in, those beside the rows' texels among them. Its pages,
    /// named at a CPU lock, are taken for the cache, and the download that follows copies each
    /// row's texels alone, as an asked cache's downloads do
    #[test]
    fn a_carrying_deferral_that_asks_fills_its_pages_in_whole() {
        let mut r = deferring_carrying(8);
        r.dec.set_unified_memory(true);
        r.dec.set_deferred_asks();
        let page = r.page;
        let (layout, _) = short_rows(page);
        r.p.world()
            .answer_fresh_surfaces(Some("the rows end inside a line".into()));
        assert_eq!(
            defer_carrying(&mut r, BLOCK, 2, &[(0, 1)], Some(layout)),
            RGFX_STATUS_OK
        );
        let cache = cache_id(&r);
        let ram = ram_bytes(&r, 1, page, Some(0x3c));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        let filled = copies_at(&r.p.call_log().since(mark));
        assert_eq!(filled.len(), 1, "{filled:?}");
        let (staging, from, into, at, size) = filled[0];
        assert_eq!(
            (from, into, at, size),
            (0, cache, 0, page),
            "the whole page, the bytes beside its row's texels too"
        );
        assert_eq!(buffer_bytes(&r, staging, None), ram);
        r.p.world().complete_all(CompleteOn::Caller);

        assert_eq!(
            attach(&mut r, BLOCK, 7, &[(1, 2)], 2 * page),
            RGFX_STATUS_OK
        );
        assert_eq!(r.dec.caches().unattached(BLOCK), None);
        let mark = r.p.call_log().mark();
        let done = download(&r, BLOCK);
        let out = copies(&r.p.call_log().since(mark));
        let pages = out[0].1;
        assert_ne!(pages, cache);
        assert_eq!(out, vec![(cache, pages); 2], "each row's texels alone");
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
    }

    /// A carrying deferral whose pages are named before anything read its cache still owes its
    /// first fill, now of all its pages: the carried ones go with the naming, and the first use
    /// fills the cache from the pages' view, whole
    #[test]
    fn a_carrying_deferrals_naming_owes_its_first_fill_of_all_its_pages() {
        let mut r = deferring_carrying(8);
        let page = r.page;
        assert_eq!(
            defer_carrying(&mut r, BLOCK, 2, &[(0, 1)], None),
            RGFX_STATUS_OK
        );
        let cache = cache_id(&r);
        assert_eq!(
            attach(&mut r, BLOCK, 0, &[(1, 2)], 2 * page),
            RGFX_STATUS_OK
        );
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        let filled = copies_at(&r.p.call_log().since(mark));
        assert_eq!(filled.len(), 1, "{filled:?}");
        let (pages, from, into, at, size) = filled[0];
        assert_ne!(pages, cache);
        assert_eq!(
            (from, into, at, size),
            (0, cache, 0, 2 * page),
            "the whole surface, from its pages"
        );
    }

    /// A carrying deferral whose runs are not whole guest pages summing to its surface's bytes,
    /// lie outside the guest's RAM the host maps, or carry no page, is an invalid argument and
    /// makes nothing; so is a deferral with runs where the device lets none carry. One past the
    /// budget is refused with nothing made, as any deferral is, so the kext names its pages
    /// instead
    #[test]
    fn a_carrying_deferral_carries_whole_pages_of_its_surface_or_is_refused() {
        let mut r = deferring_carrying(8);
        let page = r.page;
        let flags = RGFX_SURFACE_PAGES_CACHED | RGFX_SURFACE_PAGES_DEFERRED;
        let run = |address, length| RgfxPageRun { address, length };
        let zeros = RGFX_PAGE_RUN_ZEROS;
        let bad = [
            vec![run(RAM + page, page)],
            vec![run(RAM + page, page), run(zeros, page), run(zeros, page)],
            vec![run(zeros, page), run(zeros, page)],
            vec![
                run(RAM + page, page),
                run(zeros, page - 0x800),
                run(zeros, 0x800),
            ],
            vec![run(RAM + page + 0x800, page), run(zeros, page)],
            vec![run(RAM + 64 * page, page), run(zeros, page)],
            vec![run(RAM + page, page), run(zeros, 0), run(zeros, page)],
        ];
        for runs in bad {
            assert_eq!(
                r.pages_of_runs(BLOCK, 0, &runs, 2 * page, flags, None),
                RGFX_STATUS_INVALID_ARGUMENT,
                "{runs:?}"
            );
        }
        assert!(!r.dec.caches().has(BLOCK));
        assert_eq!(r.dec.caches().counts().deferred, 0);

        let mut plain = deferring(8);
        assert_eq!(
            defer_carrying(&mut plain, BLOCK, 2, &[(0, 1)], None),
            RGFX_STATUS_INVALID_ARGUMENT,
            "a device that lets no deferral carry"
        );
        assert!(!plain.dec.caches().has(BLOCK));

        let mut full = deferring_carrying(1);
        assert_eq!(
            defer_carrying(&mut full, BLOCK, 2, &[(0, 1)], None),
            RGFX_STATUS_REFUSED,
            "past the budget"
        );
        assert!(!full.dec.caches().has(BLOCK));
        assert_eq!(full.dec.caches().counts().deferred_refused, 1);
    }

    /// A deferred block the budget has no room for, or whose cache the platform does not make, is
    /// refused and leaves nothing behind, so the kext's
    /// `SURFACE_PAGES` of its pages that follows makes its cache as any block's. A release gives
    /// its room back
    #[test]
    fn a_deferred_block_not_made_is_refused_with_nothing_left() {
        use reims_vgpu_vulkan::core::testkit::Step;
        let mut r = deferring(3);
        let page = r.page;
        let refused = RGFX_STATUS_REFUSED;
        assert_eq!(defer(&mut r, BLOCK, 2 * page), RGFX_STATUS_OK);
        let other = BLOCK + 0x8000;
        assert_eq!(defer(&mut r, other, 2 * page), refused, "past the budget");
        assert!(!r.dec.caches().has(other));
        assert_eq!(r.dec.views_of_surfaces().len(), (0, 0));
        assert_eq!(
            r.pages_cached(other, &[(3, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        assert!(r.dec.caches().has(other), "named as any block");
        assert_eq!(r.dec.caches().unattached(other), None);

        assert!(release(&r, BLOCK).load(Ordering::SeqCst));
        let third = BLOCK + 0x10000;
        r.p.script().push("newBufferWithLength:options:", Step::Nil);
        assert_eq!(defer(&mut r, third, 2 * page), refused, "no buffer");
        assert!(!r.dec.caches().has(third));
        assert_eq!(
            defer(&mut r, third, 2 * page),
            RGFX_STATUS_OK,
            "the room is back"
        );
        assert_eq!(r.dec.caches().counts().deferred_refused, 2);

        let mut none = deferring(0);
        assert_eq!(defer(&mut none, BLOCK, page), refused, "no budget");
    }

    /// The live Shared buffer `id`'s bytes, each set to `with` first where given.
    fn buffer_bytes(r: &Rig, id: u64, with: Option<u8>) -> Vec<u8> {
        use reims_vgpu_vulkan::core::platform::ObjId;
        use reims_vgpu_vulkan::core::testkit::objects::kind;
        let buffer = (r.p.world().find(ObjId(id)))
            .and_then(|o| o.cast::<kind::Buffer>())
            .expect("a live buffer");
        let (contents, length) = (buffer.contents(), buffer.length());
        assert!(!contents.is_null(), "a Shared buffer");
        // SAFETY: a live Shared buffer's `length` bytes, which no work of the test's uses now.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(contents.as_ptr::<u8>(), length.to_usize()) };
        if let Some(value) = with {
            bytes.fill(value);
        }
        bytes.to_vec()
    }

    /// The `length` bytes of guest RAM from page `first`, read, or written with `with`.
    fn ram_bytes(r: &Rig, first: u64, length: u64, with: Option<u8>) -> Vec<u8> {
        let at = r
            .ram
            .host_addr(RAM + first * r.page)
            .unwrap()
            .as_ptr::<u8>();
        // SAFETY: the rig's RAM, `length` bytes from page `first` in it, which no work uses now.
        let bytes = unsafe { std::slice::from_raw_parts_mut(at, length.to_usize()) };
        if let Some(value) = with {
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = value ^ (i as u8);
            }
        }
        bytes.to_vec()
    }

    /// A deferred block's pages that the host makes no buffer over, at the CPU lock that names
    /// them, are taken copied, not refused: a download copies the cache into a Shared buffer of
    /// its own, whose bytes the host's CPU moves into the pages once that copy is done, before
    /// the answer; a fill after the CPU's write copies the pages into another such buffer, which
    /// the GPU copies into the cache; the block's buffers are its cache. Runs that are not the
    /// surface's bytes are refused still
    #[test]
    fn a_deferred_blocks_pages_with_no_view_are_copied() {
        use reims_vgpu_vulkan::core::testkit::Step;
        let mut r = deferring(8);
        let page = r.page;
        let length = 2 * page;
        assert_eq!(defer(&mut r, BLOCK, length), RGFX_STATUS_OK);
        let cache = cache_id(&r);
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        r.p.world().complete_all(CompleteOn::Caller);

        let before = ram_bytes(&r, 1, length, Some(0x5a));
        assert_eq!(
            attach(&mut r, BLOCK, 0, &[(1, 1)], length),
            RGFX_STATUS_REFUSED,
            "runs short of the surface are refused"
        );
        assert_eq!(r.dec.caches().unattached(BLOCK), Some(length));
        // The platform makes no buffer over the pages' view (`buffer_over`).
        r.p.script().push("newBufferWithDescriptor:", Step::Nil);
        let attached = r.p.call_log().mark();
        assert_eq!(
            attach(&mut r, BLOCK, 0, &[(1, 2)], length),
            RGFX_STATUS_OK,
            "no buffer over the view: copied, not refused"
        );
        assert_eq!(r.dec.caches().unattached(BLOCK), None);
        assert_eq!(r.dec.caches().counts().attached_copied, 1);
        assert_eq!(cache_id(&r), cache, "the same cache");
        // The view no buffer was made over goes, and its mapping with it on the surface worker
        // (`SurfaceView`'s drop): done before the next mark, which it would land after.
        r.dec.surface_worker().flush();
        assert!(
            r.p.call_log()
                .sels_since(attached)
                .contains(&"mach_vm_deallocate"),
            "the view's mapping is let go of"
        );
        let mark = r.p.call_log().mark();
        assert_eq!(r.buffer(0x300, BLOCK, length), Ok(()));
        assert!(
            r.p.call_log().sels_since(mark).is_empty(),
            "no buffer made: its buffers are the cache"
        );

        // The download: the GPU copies the cache into a staging buffer; the pages take it once
        // that copy is done, then the answer.
        let mark = r.p.call_log().mark();
        let done = download(&r, BLOCK);
        let out = copies(&r.p.call_log().since(mark));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, cache, "out of the cache");
        assert_ne!(out[0].1, cache);
        assert!(!done.load(Ordering::SeqCst), "not before the copy is done");
        assert_eq!(
            ram_bytes(&r, 1, length, None),
            before,
            "the pages wait for it"
        );
        // As the GPU's copy would, the staging buffer takes the cache's bytes.
        assert_eq!(
            buffer_bytes(&r, out[0].1, Some(0xc3)).len(),
            length.to_usize()
        );
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(
            done.load(Ordering::SeqCst),
            "answered once the pages hold it"
        );
        assert!(
            ram_bytes(&r, 1, length, None).iter().all(|&b| b == 0xc3),
            "the pages hold what the GPU copied out of the cache"
        );

        // The CPU writes the pages under a read-write lock: the next use fills the cache from
        // them, through a staging buffer the host's CPU filled.
        let written = ram_bytes(&r, 1, length, Some(0x11));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 1, false)]);
        let filled = copies(&r.p.call_log().since(mark));
        assert_eq!(filled.len(), 1);
        assert_eq!(filled[0].1, cache, "into the cache");
        assert_ne!(filled[0].0, out[0].1, "from a staging buffer of its own");
        assert_eq!(
            buffer_bytes(&r, filled[0].0, None),
            written,
            "the CPU's write, in the fill's staging buffer before the GPU copies it"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        let released = release(&r, BLOCK);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(released.load(Ordering::SeqCst));
        assert_eq!(r.dec.caches().len(), 0);
    }

    /// The platform's command queue.
    type Queue = <FakePlatform as Platform>::CommandQueue;

    /// A command buffer on `queue`, of id `id`, that writes `BLOCK` at generation 0, begun and
    /// finished on the caches as the decoder does one that holds a video encoder: its queue is
    /// handed to [`Caches::finish`], for the early copies behind it.
    fn decode_into(r: &Rig, id: u64, queue: &Queue) {
        let cb = queue.command_buffer(ON).unwrap().unwrap();
        let writes = [RgfxOpSurfaceUse {
            flags: RGFX_SURFACE_USE_WRITES,
            ..reading(BLOCK, 0)[0]
        }];
        let caches = &r.dec.surface_caches;
        let (_, uses) = caches.before(&r.device, &cb, &writes, &NO_REACH).unwrap();
        (caches.finish(&r.device, id, Some(queue), &cb, uses.unwrap(), || {
            cb.commit(ON)
        }))
        .unwrap();
    }

    /// [`decode_into`], its completion, then a download: whether that was answered once done,
    /// and the copies it made. The surface is read back after the write.
    fn decode_then_lock(r: &Rig, id: u64, queue: &Queue) -> (bool, Vec<(u64, u64)>) {
        decode_into(r, id, queue);
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        let done = download(r, BLOCK);
        let copied = copies(&r.p.call_log().since(mark));
        r.p.world().complete_all(CompleteOn::Caller);
        (done.load(Ordering::SeqCst), copied)
    }

    /// A surface whose decoded picture the CPU locks to read (a decoder's output, read by its
    /// client) is copied into its pages as the next decode into it ends, not at the lock after
    /// it: the decode is followed on its own queue by a command buffer of the host's, committed
    /// right behind it, which copies the cache into the pages and signals the queue's end event
    /// next; the download that follows copies nothing more and is answered once that copy lands
    #[test]
    fn a_surface_read_back_is_copied_out_behind_its_next_writer() {
        let r = cached();
        let cache = cache_id(&r);
        let queue = r.device.new_command_queue().unwrap();
        let (answered, copied) = decode_then_lock(&r, 7, &queue);
        assert!(answered);
        assert_eq!(copied.len(), 1, "the lock's download copies");
        let pages = copied[0].1;
        assert_eq!(copied[0].0, cache);

        let mark = r.p.call_log().mark();
        decode_into(&r, 7, &queue);
        let calls = r.p.call_log().since(mark);
        assert_eq!(
            copies(&calls),
            [(cache, pages)],
            "copied out behind the writer"
        );
        let commits: Vec<usize> = (calls.iter().enumerate())
            .filter(|(_, c)| c.sel == "commit")
            .map(|(i, _)| i)
            .collect();
        let copy = calls.iter().position(|c| c.sel == COPY).unwrap();
        assert_eq!(commits.len(), 2, "the writer, then the copy");
        assert!(commits[0] < copy && copy < commits[1]);
        let signals = events(&calls, SIGNAL);
        assert_eq!(signals.len(), 2);
        assert_eq!(
            signals[1],
            (signals[0].0, signals[0].1 + 1),
            "the queue's end, next"
        );

        let mark = r.p.call_log().mark();
        let done = download(&r, BLOCK);
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "nothing more"
        );
        assert!(!done.load(Ordering::SeqCst), "answered as the copy lands");
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
        let counts = r.dec.caches().counts();
        assert_eq!(
            (
                counts.downloads,
                counts.early,
                counts.early_read,
                counts.early_unread
            ),
            (1, 1, 1, 0)
        );
        assert!(
            download(&r, BLOCK).load(Ordering::SeqCst),
            "and at once after"
        );
    }

    /// A read-back surface whose cache copies by its pages' runs is copied out behind its next
    /// writer by them, as one with a view is through it: the host's command buffer behind the
    /// writer copies the cache into the runs, and the download that follows copies nothing more
    #[test]
    fn a_surface_read_back_by_runs_is_copied_out_by_them_behind_its_next_writer() {
        use reims_vgpu_vulkan::core::testkit::impls::blit::BUFFER_TO_GUEST_RAM;
        let r = cached_by_runs();
        let cache = cache_id(&r);
        let queue = r.device.new_command_queue().unwrap();
        decode_into(&r, 7, &queue);
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        let done = download(&r, BLOCK);
        let lock = copied_runs(&r.p.call_log().since(mark), BUFFER_TO_GUEST_RAM, 2);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
        assert_eq!(lock.len(), 1, "the lock's download copies by the runs");

        let mark = r.p.call_log().mark();
        decode_into(&r, 7, &queue);
        let calls = r.p.call_log().since(mark);
        assert!(copies(&calls).is_empty(), "no copy through a view");
        assert_eq!(
            copied_runs(&calls, BUFFER_TO_GUEST_RAM, 2),
            lock,
            "copied out by the runs behind the writer"
        );
        let from = (calls.iter())
            .find(|c| c.sel == BUFFER_TO_GUEST_RAM)
            .unwrap();
        assert_eq!(from.args[0].as_obj().map(|o| o.0), Some(cache));
        let commits: Vec<usize> = (calls.iter().enumerate())
            .filter(|(_, c)| c.sel == "commit")
            .map(|(i, _)| i)
            .collect();
        let copy = (calls.iter())
            .position(|c| c.sel == BUFFER_TO_GUEST_RAM)
            .unwrap();
        assert_eq!(commits.len(), 2, "the writer, then the copy");
        assert!(commits[0] < copy && copy < commits[1]);

        let mark = r.p.call_log().mark();
        let done = download(&r, BLOCK);
        let calls = r.p.call_log().since(mark);
        assert!(
            copies(&calls).is_empty() && copied_runs(&calls, BUFFER_TO_GUEST_RAM, 2).is_empty(),
            "nothing more"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
        let counts = r.dec.caches().counts();
        assert_eq!(
            (counts.early, counts.early_read, counts.runs_refused),
            (1, 1, 0)
        );
    }

    /// A download of a read-back surface asked while its writer is committed, with the caches'
    /// lock let go, waits for the writer, still among the cache's users, and so for the early
    /// copy made behind it once the commit has returned: the download copies nothing of its own
    /// and is answered as that copy lands ([`Caches::finish`])
    #[test]
    fn a_download_asked_during_a_read_back_writer_s_commit_waits_for_its_early_copy() {
        let r = cached();
        let cache = cache_id(&r);
        let queue = r.device.new_command_queue().unwrap();
        let (answered, copied) = decode_then_lock(&r, 7, &queue);
        assert!(answered);
        assert_eq!(copied.len(), 1, "the lock's download copies");
        let pages = copied[0].1;

        let cb = queue.command_buffer(ON).unwrap().unwrap();
        let writes = [RgfxOpSurfaceUse {
            flags: RGFX_SURFACE_USE_WRITES,
            ..reading(BLOCK, 0)[0]
        }];
        let caches = &r.dec.surface_caches;
        let (_, uses) = caches.before(&r.device, &cb, &writes, &NO_REACH).unwrap();
        // The download as the decoder asks it ([`download`]), on a thread of its own: the
        // decoder is not shared between threads, the caches are.
        let region = Arc::clone(&r.dec.shared.region);
        let done = Arc::new(AtomicBool::new(false));
        let ask = || {
            let set = Arc::clone(&done);
            let answer = Box::new(move || set.store(true, Ordering::SeqCst));
            caches.download(
                &r.device,
                BLOCK,
                region,
                Box::new(|| {}),
                Arc::default(),
                answer,
                no_raise(),
            );
        };
        let mark = r.p.call_log().mark();
        let (in_commit, committing) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let (go, wait_go) = std::sync::mpsc::channel::<()>();
            let finish = scope.spawn(|| {
                let (cb, queue) = (&cb, &queue);
                caches
                    .finish(&r.device, 7, Some(queue), cb, uses.unwrap(), move || {
                        in_commit.send(()).unwrap();
                        let _ = wait_go.recv();
                        cb.commit(ON)
                    })
                    .map_err(|_| ())
            });
            committing.recv().unwrap();
            let asked = scope.spawn(ask);
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(
                asked.is_finished(),
                "a download waited for the writer's commit"
            );
            asked.join().unwrap();
            assert!(
                !done.load(Ordering::SeqCst),
                "answered before the writer's commit"
            );
            drop(go);
            assert_eq!(finish.join().unwrap(), Ok(()));
        });
        let calls = r.p.call_log().since(mark);
        assert_eq!(copies(&calls), [(cache, pages)], "the early copy alone");
        assert!(!done.load(Ordering::SeqCst), "answered as the copy lands");
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
        let counts = r.dec.caches().counts();
        assert_eq!(
            (counts.downloads, counts.early, counts.early_read),
            (1, 1, 1)
        );
    }

    /// Only a write that was the only one since the cache's last copy, read back, starts the
    /// early copies; a write that no download followed ends them, and the lock after it
    /// downloads as before, which starts them again. A writer whose queue the decoder does not
    /// hand over (one that holds no video encoder) takes its copy on the host's read-back queue,
    /// which waits there on the GPU for the writer's end
    #[test]
    fn a_write_no_lock_followed_ends_the_early_copies() {
        let mut r = cached();
        let queue = r.device.new_command_queue().unwrap();
        // Two writes, then the lock: no reading back.
        use_surfaces(&mut r, 2, 0x200, &[(BLOCK, 0, true)]);
        let (_, copied) = decode_then_lock(&r, 7, &queue);
        assert_eq!(copied.len(), 1);
        let mark = r.p.call_log().mark();
        decode_into(&r, 7, &queue);
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "not read back"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        let (_, copied) = decode_then_lock(&r, 7, &queue);
        assert_eq!(copied.len(), 1, "two writes since the last copy");

        // One write, then the lock: the next write is copied out early, a writer that holds no
        // video encoder on the read-back queue, after the writer's end; its lock copies nothing
        // more.
        let (_, copied) = decode_then_lock(&r, 7, &queue);
        assert_eq!(copied.len(), 1);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 2, 0x201, &[(BLOCK, 0, true)]);
        let calls = r.p.call_log().since(mark);
        assert_eq!(copies(&calls).len(), 1, "copied out on the read-back queue");
        let writer = events(&calls, SIGNAL)[0];
        assert!(
            events(&calls, WAIT).contains(&writer),
            "after the writer's end, on the GPU"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        let done = download(&r, BLOCK);
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "its lock copies nothing more"
        );
        assert!(done.load(Ordering::SeqCst));
        assert_eq!(r.dec.caches().counts().early_aside, 1);
        let mark = r.p.call_log().mark();
        decode_into(&r, 7, &queue);
        assert_eq!(copies(&r.p.call_log().since(mark)).len(), 1, "early");
        r.p.world().complete_all(CompleteOn::Caller);
        // No lock reads that write: the next one is not copied out, and its lock downloads.
        let mark = r.p.call_log().mark();
        let (answered, copied) = decode_then_lock(&r, 7, &queue);
        assert!(answered);
        let all = copies(&r.p.call_log().since(mark));
        assert_eq!(
            (all.len(), copied.len()),
            (1, 1),
            "the lock's download alone"
        );
        assert_eq!(r.dec.caches().counts().early_unread, 1);
        let mark = r.p.call_log().mark();
        decode_into(&r, 7, &queue);
        assert_eq!(copies(&r.p.call_log().since(mark)).len(), 1, "early again");
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(r.dec.caches().counts().early, 3);
    }

    /// An early copy on the host's read-back queue still in flight holds the cache's next writer
    /// on the GPU, one on the first writer's own queue too, at the copy's end value, and a fill
    /// of a newer generation waits for it there as well
    #[test]
    fn a_read_back_queue_s_copy_in_flight_holds_the_next_writer() {
        let mut r = cached();
        use_surfaces(&mut r, 2, 0x200, &[(BLOCK, 0, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let done = download(&r, BLOCK);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 2, 0x201, &[(BLOCK, 0, true)]);
        let calls = r.p.call_log().since(mark);
        assert_eq!(copies(&calls).len(), 1, "copied out early");
        let copied = events(&calls, SIGNAL)[1];
        assert_eq!(copied.1, 1, "the read-back queue's first copy");

        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 2, 0x202, &[(BLOCK, 0, true)]);
        let calls = r.p.call_log().since(mark);
        assert!(events(&calls, WAIT).contains(&copied), "the writer waits");
        assert!(
            copies(&calls).is_empty(),
            "read by no lock: no copy behind it"
        );
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x203, &[(BLOCK, 1, false)]);
        let calls = r.p.call_log().since(mark);
        assert_eq!(copies(&calls).len(), 1, "a fill");
        assert!(events(&calls, WAIT).contains(&copied), "the fill waits");
        r.p.world().complete_all(CompleteOn::Caller);
    }

    /// An early copy still in flight is waited for: a release of the cache, which nothing wrote
    /// since, is answered as it lands, copying nothing; a writer of another queue waits for it on
    /// the GPU, at the copy's end value; and the host's own work on the cache (a decode's
    /// picture) waits for it on the CPU
    #[test]
    fn an_early_copy_in_flight_is_waited_for() {
        let mut r = cached();
        let queue = r.device.new_command_queue().unwrap();
        decode_then_lock(&r, 7, &queue);
        let mark = r.p.call_log().mark();
        decode_into(&r, 7, &queue);
        let early = events(&r.p.call_log().since(mark), SIGNAL)[1];
        let wait = std::time::Duration::from_millis(20);
        assert!(
            !r.dec.surface_caches.wait_copied_out(BLOCK, wait),
            "in flight"
        );

        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 2, 0x200, &[(BLOCK, 0, true)]);
        let calls = r.p.call_log().since(mark);
        assert_eq!(events(&calls, WAIT), [early], "after the copy, on the GPU");
        assert!(copies(&calls).is_empty());
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(r.dec.surface_caches.wait_copied_out(BLOCK, wait), "landed");

        // The lock after that write downloads it, the only one since the early copy: read back
        // again, and an early copy in flight at the release.
        let mark = r.p.call_log().mark();
        let done = download(&r, BLOCK);
        assert_eq!(copies(&r.p.call_log().since(mark)).len(), 1);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
        decode_into(&r, 7, &queue);
        let mark = r.p.call_log().mark();
        let released = release(&r, BLOCK);
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "nothing more"
        );
        assert!(
            !released.load(Ordering::SeqCst),
            "answered as the copy lands"
        );
        assert_eq!(r.dec.caches().len(), 0);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(released.load(Ordering::SeqCst));
    }

    /// `SURFACE_DOWNLOAD` of the block at `block` where it waits for nothing and commits no
    /// copy now ([`GraphicsDecoder::try_surface_download`]): whether it was asked, and whether it
    /// was answered.
    fn try_download(r: &Rig, block: u64) -> (bool, Arc<AtomicBool>) {
        let done = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&done);
        let asked = r.dec.try_surface_download(
            &r.device,
            block,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        (asked, done)
    }

    /// A download that commits no copy as it is asked is carried out where it is offered (the
    /// control thread): one of a cache nothing wrote since its last copy is answered at once,
    /// one whose early copy is in flight as that lands, and one whose writer is still walked in
    /// that writer's finish, which copies. One that would copy now, or that finds the caches'
    /// lock held, is left, unanswered and copying nothing, for the block's lane
    #[test]
    fn a_download_that_copies_nothing_now_is_carried_out_where_it_is_offered() {
        let mut r = cached();
        let cache = cache_id(&r);
        let queue = r.device.new_command_queue().unwrap();
        let (asked, done) = try_download(&r, BLOCK);
        assert!(asked && done.load(Ordering::SeqCst), "nothing written");

        // A lock after a decode reads it back: the next decode is copied out behind it.
        let (_, copied) = decode_then_lock(&r, 7, &queue);
        assert_eq!(copied.len(), 1);
        decode_into(&r, 7, &queue);
        r.p.world().complete_all(CompleteOn::Caller);
        // The early copy behind that decode landed: answered at once, copying nothing.
        let mark = r.p.call_log().mark();
        let (asked, done) = try_download(&r, BLOCK);
        assert!(
            asked && done.load(Ordering::SeqCst),
            "the early copy landed"
        );
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        // One in flight: answered as it lands.
        decode_into(&r, 7, &queue);
        let (asked, done) = try_download(&r, BLOCK);
        assert!(
            asked && !done.load(Ordering::SeqCst),
            "the early copy is in flight"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));

        // Two writes, the first copied out early and read by no lock, so the second is not: the
        // download would copy now.
        use_surfaces(&mut r, 2, 0x200, &[(BLOCK, 0, true)]);
        use_surfaces(&mut r, 2, 0x201, &[(BLOCK, 0, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        let (asked, done) = try_download(&r, BLOCK);
        assert!(!asked && !done.load(Ordering::SeqCst), "left for the lane");
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        // With the caches' lock held elsewhere, left too: a try never waits for it.
        {
            let held = r.dec.surface_caches.lock();
            let (asked, _) = try_download(&r, BLOCK);
            assert!(!asked, "the lock is held");
            drop(held);
        }
        let done = download(&r, BLOCK);
        assert_eq!(copies(&r.p.call_log().since(mark)), [(cache, copied[0].1)]);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));

        // A writer begun and not finished: asked, and copied in its finish.
        let cb = queue.command_buffer(ON).unwrap().unwrap();
        let writes = [RgfxOpSurfaceUse {
            flags: RGFX_SURFACE_USE_WRITES,
            ..reading(BLOCK, 0)[0]
        }];
        let caches = &r.dec.surface_caches;
        let (_, uses) = caches.before(&r.device, &cb, &writes, &NO_REACH).unwrap();
        let mark = r.p.call_log().mark();
        let (asked, done) = try_download(&r, BLOCK);
        assert!(asked, "made in the writer's finish");
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        (caches.finish(&r.device, 7, None, &cb, uses.unwrap(), || cb.commit(ON))).unwrap();
        assert_eq!(
            copies(&r.p.call_log().since(mark)).len(),
            1,
            "in its finish"
        );
        assert!(!done.load(Ordering::SeqCst));
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
    }

    /// The handlers added to the command buffer of the `nth` commit in `calls`, in order.
    fn handlers_of_commit(calls: &[Call], nth: usize) -> Vec<&'static str> {
        let commit = (calls.iter())
            .filter(|c| c.sel == "commit")
            .nth(nth)
            .unwrap();
        (calls.iter())
            .filter(|c| c.obj == commit.obj && c.seq < commit.seq)
            .map(|c| c.sel)
            .filter(|sel| ["addCompletionSeenHandler", "addCompletedHandler:"].contains(sel))
            .collect()
    }

    /// A download's copy is answered, and a fill's waiters told, as the platform sees its
    /// command buffer done, by a completion-seen handler, with no handler thread woken between
    /// on a platform that has one; the completed handler only lets go of the copy's buffers. The
    /// fill is the one made as the cache is made ([`SurfaceCaches::fill_made`]), which the
    /// writer's first use of the generation waits for
    #[test]
    fn a_copy_s_answer_goes_as_the_platform_sees_it_done() {
        let mut r = rig(8);
        let page = r.page;
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        queues(&mut r);
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, true)]);
        let calls = r.p.call_log().since(mark);
        assert_eq!(
            copies(&calls).len(),
            1,
            "the fill at the make, then the writer"
        );
        assert_eq!(handlers_of_commit(&calls, 0), ["addCompletionSeenHandler"]);
        r.p.world().complete_all(CompleteOn::Caller);

        let mark = r.p.call_log().mark();
        let done = download(&r, BLOCK);
        let calls = r.p.call_log().since(mark);
        assert_eq!(copies(&calls).len(), 1);
        assert_eq!(
            handlers_of_commit(&calls, 0),
            ["addCompletionSeenHandler", "addCompletedHandler:"]
        );
        assert!(!done.load(Ordering::SeqCst));
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
    }
}
