//! The device caches of IOSurfaces (`RGFX_FEATURE_SURFACE_CACHE`; docs/graphics-kext/design.md,
//! Decided 14, "The device cache", which says each choice's class).
//!
//! A GPU that does not share the CPU's memory keeps the GPU's copy of an IOSurface in its own
//! memory, the surface's device cache, and moves bytes between it and the surface's pages only
//! where IOSurface's locking asks it to. Where the host's GPU is such a GPU, a block the kext marks
//! cached ([`RGFX_SURFACE_PAGES_CACHED`]) gets one here: a Private buffer of the surface's bytes in
//! the pages' own layout, which every `BUFFER_OVER_SURFACE_PAGES` buffer of the block is, so every
//! texture over the surface, of any shape, plane or format, lies over the same bytes, as the
//! textures over the pages did. The surface's view of its pages ([`super::surface_pages`]) is the
//! other end of each copy.
//!
//! - **Fill.** A command buffer names each cached surface it uses (`RGFX_OP_SURFACE_USE`), with
//!   the CPU's generation of the surface when it was committed: the kext's count of the surface's
//!   read-write locks, which IOSurface marks at the lock. Where the cache holds an older
//!   generation, or none, the cache is filled from the pages before the buffer's work; a buffer
//!   of the same generation decoded later waits for that fill, on any queue. One of an older
//!   generation than the cache's (two queues' buffers committed apart and taken in the other
//!   order) fills nothing. A buffer's write is in the pages before a newer generation's fill
//!   wherever a read-write lock waited for it: a process names its buffer's writes to the kext
//!   before the commit returns, so a lock after the commit waits for them and has them
//!   downloaded, and the download comes before the fill. A commit on one thread racing a lock of
//!   the surface on another, which IOSurface leaves undefined, can name its write after the lock
//!   looked: its write is in no download, and the fill of the newer generation copies the
//!   pages, the CPU's writes, over it.
//! - **Download.** The pages are written from the cache only at a `SURFACE_DOWNLOAD` (a CPU lock
//!   after a GPU write), at the block's `BLOCK_RELEASE` and where the kext names the block's pages
//!   again (`SURFACE_PAGES`), and only where a buffer that may write the surface was committed
//!   since the cache was last copied either way. A release copies whether or not the surface has
//!   a client left (`RGFX_BLOCK_RELEASE_NO_CLIENT` changes nothing): a mach port or an XPC object
//!   holds a surface with no client, and the process that looks it up from one reads its pages.
//!   A download or a release copies nothing where, as its copy is made (at once, or in the
//!   finish of a buffer begun before it), the CPU's generation in the block's header
//!   ([`CPU_GEN_OFFSET`]) has moved past the cache's: the lock that asked gave up waiting for the
//!   answer, moved the generation, and let the CPU write the pages, whose writes are the newer,
//!   as at a fill.
//! - **Early copy.** A decoder's output that the CPU reads back after each picture (a client
//!   that locks each frame to read it) is copied into its pages as the decode ends, not at the
//!   lock that follows it ([`SurfaceCaches::early_copies`]). Once a download copied a write that
//!   was the only one since the cache's last copy, the block reads back, and each later writer
//!   that holds a video encoder and is again the only one has a command buffer of the host's
//!   committed on its own queue right behind it, which copies the cache into the pages; the
//!   download that follows is answered as that copy lands. A write that no download followed
//!   ends it, until a download copies again. Other writers' queues take no such copy, as one
//!   would hold their later work behind it (a window server drawing into a surface that another
//!   process captures), and their downloads go as before; so does a download of pages the host
//!   copies through the CPU, the copy running only where the pages lie in a buffer.
//! - **Order.** Fills and downloads run on the host's own transfer queue, in the order they are
//!   asked, each after every command buffer that used the cache before it: each such buffer
//!   signals its queue's end event as it ends, and the copy waits for the latest value of each
//!   queue's. So a frame drawn from the cache before a CPU write still reads what it read.
//! - **The host's own writes.** A device of the host that writes a surface's pages itself (a
//!   video device's frame, [`crate::device_writes`]) moves no generation, so each cache's pages
//!   are watched: a write over them makes the cache's next use fill it whatever its generation,
//!   and a download or a release before that copies nothing, the pages being newer than the
//!   cache.
//! - **Deferred blocks.** Where the guest took `RGFX_FEATURE_SURFACE_DEFERRED`, the kext names a
//!   fresh surface only the GPU has touched with no pages
//!   ([`RGFX_SURFACE_PAGES_DEFERRED`](reims_graphics::wire::RGFX_SURFACE_PAGES_DEFERRED)), as a
//!   GPU of its own memory keeps such a surface in that memory alone: its cache is made as zeros,
//!   on the transfer queue before any use, at the CPU's generation 0, with no pages to fill from,
//!   download into or watch. The kext names the pages at the first CPU lock that needs them
//!   ([`RGFX_SURFACE_PAGES_ATTACH`](reims_graphics::wire::RGFX_SURFACE_PAGES_ATTACH)), and the
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

use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use reims_graphics::ops::{RGFX_SURFACE_USE_WRITES, RgfxOpSurfaceUse};
use reims_graphics::region::Region;
use reims_graphics::wire::RgfxBufferOverSurfacePages;
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::num::{ToU64 as _, ToUsize as _};
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue, CompletedInfo};
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::event::EventRef;
use reims_vgpu_core::platform::resource::Buffer as _;
use reims_vgpu_core::platform::types::NsRange;
use reims_vgpu_core::platform::{Handle, Platform};
use reims_vgpu_core::raise::{OnFault, Raise, RaiseResult, refusals};
use reims_vgpu_core::sync::{Condvar, Mutex, MutexGuard};

use super::refusing;
use super::surface_pages::{SurfaceView, buffer_over};
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

/// The CPU's generation of the surface whose cached block is at `block`, as the kext last wrote
/// it in the block's header ([`CPU_GEN_OFFSET`]); `None` where the header is not in `region`.
pub(crate) fn cpu_generation(region: &Region, block: u64) -> Option<u64> {
    region.load_u64(block.checked_add(CPU_GEN_OFFSET)?)
}

/// One surface's device cache.
pub(crate) struct SurfaceCache<P: Platform> {
    /// The cache: a Private buffer of the surface's bytes, which the block's buffers are.
    device: P::Buffer,
    /// The surface's pages, the other end of each copy; `None` for a deferred block whose pages
    /// the kext has not named yet.
    pages: Option<CachePages<P>>,
    /// The surface's bytes.
    length: u64,
    /// The CPU's generation of the surface the cache holds; `None` until its first fill.
    generation: Option<u64>,
    /// A command buffer that may write the cache was committed since the cache was last copied.
    written: bool,
    /// The transfer event's value at which the cache's last fill is done; 0 for none.
    filled_at: u64,
    /// Each queue whose command buffers used the cache since its last copy, with the value of the
    /// queue's end event the latest of them signals.
    users: HashMap<u64, u64>,
    /// Where the block was deferred, the block's size its `SURFACE_PAGES` gave, which a
    /// `BUFFER_OVER_SURFACE_PAGES` is checked against while the block has no view: its cache
    /// counts against the deferred budget until it goes, its pages named or not.
    deferred: Option<u64>,
    /// Whether the CPU reads the surface back after each GPU write, and the early copy that
    /// follows from it ([`SurfaceCaches::early_copies`]).
    read_back: ReadBack,
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

/// An early copy until it lands: the answers waiting for it, given as the platform sees its
/// command buffer done, and the threads waiting for it ([`Caches::wait_copied_out`]).
struct Landing {
    /// The answers waiting; `None` once it landed.
    waiting: Mutex<Option<Vec<Answer>>>,
    landed: Condvar,
}

/// An answer waiting for an early copy to land.
type Answer = Box<dyn FnOnce() + Send>;

impl Landing {
    /// A copy in flight.
    fn new() -> Arc<Self> {
        Arc::new(Landing {
            waiting: Mutex::new(Some(Vec::new())),
            landed: Condvar::new(),
        })
    }

    /// Whether the copy landed.
    fn landed(&self) -> bool {
        self.waiting.lock().is_none()
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

/// A cache's other end: the surface's pages.
enum CachePages<P: Platform> {
    /// A Shared buffer over the pages, the surface's view of them.
    View(P::Buffer),
    /// A deferred block's pages the host made no view of, or no buffer over, at their naming:
    /// copied through a Shared buffer of each copy's own (the module doc).
    Copied(Arc<CopiedPages>),
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

/// The host's transfer queue, on which every copy runs, and the event its fills signal.
struct Transfer<P: Platform> {
    /// Unshared ([`new_command_queue_unshared`]): a present of a surface waits for its fill, and
    /// a frame drawn from a cache for the fill before it ([`FillWait`]).
    ///
    /// [`new_command_queue_unshared`]: reims_vgpu_core::platform::device::GpuDevice::new_command_queue_unshared
    queue: P::CommandQueue,
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

/// How far the transfer queue's fills have finished: raised by each fill's completed handler,
/// which wakes the threads waiting for one ([`Caches::wait_filled`]) and then tells the hook.
struct Fills {
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
pub(crate) struct FillWait {
    fills: Arc<Fills>,
    value: u64,
}

impl FillWait {
    /// Whether the fill has finished.
    pub(crate) fn done(&self) -> bool {
        self.fills.reached(self.value)
    }
}

/// What the caches did, for the log ([`SurfaceCaches::counts`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CacheCounts {
    /// Caches made.
    pub(crate) made: u64,
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
    /// Downloads and releases of a written cache not copied, as the CPU's generation of the
    /// surface moved past the cache's ([`SurfaceCaches::behind_the_cpu`]).
    pub(crate) behind: u64,
    /// Deferred blocks' caches made, and refused (past the budget, or not made); deferred blocks
    /// whose pages were named; and those released with their pages never named.
    pub(crate) deferred: u64,
    pub(crate) deferred_refused: u64,
    pub(crate) attached: u64,
    /// Of them, pages the host made no view of, or no buffer over, copied instead.
    pub(crate) attached_copied: u64,
    pub(crate) unattached: u64,
    /// Early copies of caches into their pages, behind a writer of a surface that reads back
    /// ([`SurfaceCaches::early_copies`]), and their bytes.
    pub(crate) early: u64,
    pub(crate) early_bytes: u64,
    /// Downloads that an early copy answered, and early copies that no download followed before
    /// the next write, each of which ended its block's reading back.
    pub(crate) early_read: u64,
    pub(crate) early_unread: u64,
}

/// One cache's pages, host addresses in the guest's RAM, and whether a device of the host wrote
/// them since the cache's last fill ([`device_writes`]).
struct Pages {
    runs: Vec<Range<u64>>,
    written: bool,
}

/// Each cache's [`Pages`], by block. Its lock is a leaf.
#[derive(Default)]
struct Watched(Mutex<HashMap<u64, Pages>>);

impl Watched {
    /// Whether a device of the host wrote the block's pages since its last fill.
    fn written(&self, block: u64) -> bool {
        self.0.lock().get(&block).is_some_and(|p| p.written)
    }

    /// [`Watched::written`], cleared: the fill about to be made reads the write.
    fn take(&self, block: u64) -> bool {
        (self.0.lock().get_mut(&block)).is_some_and(|p| std::mem::take(&mut p.written))
    }
}

impl Watcher for Watched {
    fn wrote(&self, runs: &[Range<u64>]) {
        for pages in self.0.lock().values_mut() {
            if !pages.written && device_writes::overlap(&pages.runs, runs) {
                pages.written = true;
            }
        }
    }
}

/// The device caches of the blocks the kext marked cached, by block offset.
pub(crate) struct SurfaceCaches<P: Platform> {
    /// Whether the device keeps caches: its GPU does not share the CPU's memory.
    on: bool,
    by_block: HashMap<u64, SurfaceCache<P>>,
    transfer: Option<Transfer<P>>,
    /// Each command queue's end event, by the queue's id, and the last value signalled: made with
    /// the queue ([`Caches::queue_made`]), or at its first buffer that uses a cache where it was
    /// not; let go of at the queue's end where no buffer signalled it ([`Self::queue_ended`]).
    ends: HashMap<u64, (P::Event, u64)>,
    /// The caches' pages, watched for the host's devices' writes while this lives.
    watched: Arc<Watched>,
    counts: CacheCounts,
    /// By block: the command buffers begun and not finished that use its cache ([`Caches`]).
    pending: HashMap<u64, Users>,
    /// By block: the fills of its cache waiting for those command buffers to finish.
    filling: HashMap<u64, usize>,
    /// By block: the copies out of its cache asked on ring 0 (a download, a release)
    /// that wait for the command buffers that may write it begun before each was asked, in the
    /// order they were asked ([`Caches::after_writers`]).
    deferred: HashMap<u64, VecDeque<Deferred<P>>>,
    /// The number the next use that may write a cache is counted under.
    next_writer: u64,
    /// What a fill tells as it finishes, the [`Caches`]' own, kept across [`Caches::set_on`].
    on_fill_done: Arc<OnceLock<FillHook>>,
    /// The bytes deferred blocks' caches may hold at once (the device's `DEFERRED_BUDGET`), kept
    /// across [`Caches::set_on`], and the bytes they hold.
    deferred_budget: u64,
    deferred_bytes: u64,
}

/// The command buffers begun and not finished that use one block's cache.
#[derive(Default)]
struct Users {
    users: usize,
    /// Those that may write it, by the number each was counted under, oldest first.
    writers: BTreeSet<u64>,
}

/// A command buffer's count among a block's users ([`SurfaceCaches::hold`]): the block, and the
/// number of its write, where it may write it.
type Count = (u64, Option<u64>);

/// A copy out of a block's cache, made on the caches ([`Caches::after_writers`]).
type CopyOut<P> = Box<dyn FnOnce(&mut SurfaceCaches<P>) + Send>;

/// A copy out of a block's cache waiting for the buffers that may write it ([`Caches::after_writers`]).
struct Deferred<P: Platform> {
    /// It waits for every write counted below this number.
    upto: u64,
    run: CopyOut<P>,
}

impl<P: Platform> SurfaceCaches<P> {
    /// No caches; `on` keeps them where the kext asks, each fill telling `on_fill_done` as it
    /// finishes.
    fn new(on: bool, on_fill_done: Arc<OnceLock<FillHook>>) -> Self {
        let watched = Arc::new(Watched::default());
        let watcher: Arc<dyn Watcher> = Arc::clone(&watched) as Arc<dyn Watcher>;
        device_writes::watch(&watcher);
        SurfaceCaches {
            on,
            by_block: HashMap::new(),
            transfer: None,
            ends: HashMap::new(),
            watched,
            counts: CacheCounts::default(),
            pending: HashMap::new(),
            filling: HashMap::new(),
            deferred: HashMap::new(),
            next_writer: 0,
            on_fill_done,
            deferred_budget: 0,
            deferred_bytes: 0,
        }
    }

    /// Whether the device keeps caches.
    pub(crate) fn on(&self) -> bool {
        self.on
    }

    /// What the caches did so far.
    pub(crate) fn counts(&self) -> CacheCounts {
        self.counts
    }

    /// A cache for the block at `block`, whose surface's pages `view` maps: `Ok(false)` where
    /// the platform made no buffer for it, and the block keeps its view. `host_pages` are the
    /// pages' host addresses in the guest's RAM, [`device_writes::sorted`], which the host's
    /// devices' writes are watched over.
    ///
    /// A block that has a cache keeps it where its pages are the same, filled again at its next
    /// use, as the kext's generations of the surface may count anew; any other goes as at a
    /// release ([`SurfaceCaches::released`]). Either way what a buffer wrote of the cache is
    /// copied into the pages first. (The kext names a block's pages once, before any process
    /// holds it, and a block's offset is given out again only once its release is answered.)
    pub(crate) fn make(
        &mut self,
        device: &P::Device,
        block: u64,
        view: &Arc<SurfaceView<P>>,
        host_pages: Vec<Range<u64>>,
    ) -> RaiseResult<bool> {
        let length = view.length();
        if let Some(cache) = self.by_block.get(&block) {
            let same = cache.length == length
                && (self.watched.0.lock().get(&block)).is_some_and(|p| p.runs == host_pages);
            if same {
                if self.written_since_host(block) {
                    self.write_back(device, block, Box::new(|| {}))?;
                }
                if let Some(cache) = self.by_block.get_mut(&block) {
                    cache.generation = None;
                }
                return Ok(true);
            }
            self.released(device, block, false, None, Box::new(|| {}))?;
        }
        let Some(cache) = device.new_buffer_of_length(ON, length, STORAGE_PRIVATE)? else {
            return Ok(false);
        };
        let Some(pages) = buffer_over(device, view, 0, length, 0, None)? else {
            return Ok(false);
        };
        self.by_block.insert(
            block,
            SurfaceCache {
                device: cache,
                pages: Some(CachePages::View(pages)),
                length,
                generation: None,
                written: false,
                filled_at: 0,
                users: HashMap::new(),
                deferred: None,
                read_back: ReadBack::default(),
            },
        );
        let pages = Pages {
            runs: host_pages,
            written: false,
        };
        self.watched.0.lock().insert(block, pages);
        self.counts.made += 1;
        Ok(true)
    }

    /// A cache of zeros for the deferred block at `block` (the module doc), of `length` bytes,
    /// at the CPU's generation 0, its block `block_size` bytes: zeroed on the transfer queue,
    /// whose fill every use of it waits for. `Ok(false)`, with no cache, where the device keeps
    /// none, the deferred caches' budget has no room for it, or the platform made no buffer or
    /// no transfer queue for it. A cache the block had goes first, as at a release
    /// ([`SurfaceCaches::released`]).
    pub(crate) fn make_deferred(
        &mut self,
        device: &P::Device,
        block: u64,
        length: u64,
        block_size: u64,
    ) -> RaiseResult<bool> {
        if !self.on {
            return Ok(false);
        }
        if self.by_block.contains_key(&block) {
            self.released(device, block, false, None, Box::new(|| {}))?;
        }
        let room = self.deferred_budget.saturating_sub(self.deferred_bytes);
        let made = if length <= room {
            device.new_buffer_of_length(ON, length, STORAGE_PRIVATE)?
        } else {
            None
        };
        let zeroed = match &made {
            Some(cache) => self.zero(device, cache, length)?,
            None => None,
        };
        let (Some(cache), Some(filled_at)) = (made, zeroed) else {
            self.counts.deferred_refused += 1;
            return Ok(false);
        };
        self.by_block.insert(
            block,
            SurfaceCache {
                device: cache,
                pages: None,
                length,
                generation: Some(0),
                written: false,
                filled_at,
                users: HashMap::new(),
                deferred: Some(block_size),
                read_back: ReadBack::default(),
            },
        );
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
        cb.add_completed_handler(ON, Box::new(move |_: CompletedInfo| fills.finished(value)))?;
        cb.commit(ON)?;
        Ok(Some(value))
    }

    /// The pages of the deferred block at `block`, which `view` maps and whose host addresses in
    /// the guest's RAM are `host_pages` ([`Self::make`]), taken for its cache, which keeps its
    /// bytes, its generation and whether a buffer wrote it: `Some(true)`; `Some(false)` where the
    /// view is shorter than the cache or the platform made no buffer over it, the block keeping
    /// none; `None` where the block has no deferred cache whose pages are not named.
    pub(crate) fn attach(
        &mut self,
        device: &P::Device,
        block: u64,
        view: &Arc<SurfaceView<P>>,
        host_pages: Vec<Range<u64>>,
    ) -> RaiseResult<Option<bool>> {
        let Some(length) = self.unattached(block) else {
            return Ok(None);
        };
        if view.length() < length {
            return Ok(Some(false));
        }
        let Some(pages) = buffer_over(device, view, 0, length, 0, None)? else {
            return Ok(Some(false));
        };
        if let Some(cache) = self.by_block.get_mut(&block) {
            cache.pages = Some(CachePages::View(pages));
        }
        let pages = Pages {
            runs: host_pages,
            written: false,
        };
        self.watched.0.lock().insert(block, pages);
        self.counts.attached += 1;
        Ok(Some(true))
    }

    /// The pages of the deferred block at `block`, `copied` from the guest's RAM, whose host
    /// addresses are `host_pages` ([`Self::make`]), taken for its cache where the host made no
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
        }
        let pages = Pages {
            runs: host_pages,
            written: false,
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

    /// The cache of the block at `block`, which its buffers are.
    pub(crate) fn buffer(&self, block: u64) -> Option<&P::Buffer> {
        self.by_block.get(&block).map(|c| &c.device)
    }

    /// Whether the block at `block` has a cache.
    pub(crate) fn has(&self, block: u64) -> bool {
        self.by_block.contains_key(&block)
    }

    /// The transfer event's value at which the block at `block`'s last fill is done, and the
    /// value done so far; `None` where it has no cache or no fill.
    fn last_fill(&self, block: u64) -> Option<FillWait> {
        let (cache, t) = (self.by_block.get(&block)?, self.transfer.as_ref()?);
        (cache.filled_at != 0).then(|| FillWait {
            fills: Arc::clone(&t.fills),
            value: cache.filled_at,
        })
    }

    /// Whether a present of the block at `block` at CPU generation `generation` needs a fill of
    /// its cache first ([`Self::fill_now`]). False for a block with no cache.
    fn needs_fill_at(&self, block: u64, generation: u64) -> bool {
        self.by_block.get(&block).is_some_and(|cache| {
            cache.generation.is_none_or(|held| generation > held) || self.watched.written(block)
        })
    }

    /// Whether a use of the block at `u.block_offset`'s cache needs a fill first: the cache holds
    /// an older generation than the use's, or none, or a device of the host wrote its pages since
    /// its last fill. False for a block with no cache.
    fn needs_fill(&self, u: &RgfxOpSurfaceUse) -> bool {
        self.by_block.get(&u.block_offset).is_some_and(|cache| {
            cache.generation.is_none_or(|held| u.cpu_gen > held)
                || self.watched.written(u.block_offset)
        })
    }

    /// The use `u` of a block with a cache, before command buffer `cb` runs: the cache filled
    /// where it needs it ([`Self::needs_fill`]), its fill marked written where `u` may write it,
    /// and `cb` made to wait for its last fill not yet done; whether it waits. The caller made
    /// sure no command buffer begun and not finished uses the cache where it fills
    /// ([`Caches::before`]).
    fn before_one(
        &mut self,
        device: &P::Device,
        cb: &P::CommandBuffer,
        u: &RgfxOpSurfaceUse,
    ) -> RaiseResult<bool> {
        let Some(cache) = self.by_block.get(&u.block_offset) else {
            return Ok(false);
        };
        let older = cache.generation.is_none_or(|held| u.cpu_gen > held);
        let host_wrote = self.watched.take(u.block_offset);
        if older || host_wrote {
            if !older {
                self.counts.host_writes += 1;
            }
            self.fill(device, u.block_offset, u.cpu_gen)?;
        }
        let Some(cache) = self.by_block.get_mut(&u.block_offset) else {
            return Ok(false);
        };
        if u.flags & RGFX_SURFACE_USE_WRITES != 0 {
            cache.written = true;
            let read_back = &mut cache.read_back;
            read_back.writes = read_back.writes.saturating_add(1);
            // An early copy that no download followed: the surface no longer reads back.
            if std::mem::take(&mut read_back.unread) {
                read_back.on = false;
                self.counts.early_unread += 1;
            }
            // An early copy still reading the cache: the write waits for it on the GPU, as for a
            // buffer of another queue (a decode, whose picture lands off the buffer, waits for it
            // on the CPU, [`Caches::wait_copied_out`]).
            if let Some(copy) = &read_back.copy
                && !copy.landing.landed()
                && let Some((event, _)) = self.ends.get(&copy.queue)
            {
                cb.encode_wait(Some(EventRef::Event(event)), copy.value)?;
            }
        }
        if let Some(t) = &self.transfer
            && !t.fills.reached(cache.filled_at)
        {
            cb.encode_wait(Some(EventRef::Event(&t.event)), cache.filled_at)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// How many command buffers begun and not finished use the block at `block`'s cache; where
    /// `writers`, only those that may write it.
    fn pending(&self, block: u64, writers: bool) -> usize {
        self.pending
            .get(&block)
            .map_or(0, |u| if writers { u.writers.len() } else { u.users })
    }

    /// Counts a command buffer's uses `blocks` (each block, and whether it may write it) among
    /// those begun and not finished: its counts, each write under a number of its own.
    fn hold(&mut self, blocks: &[(u64, bool)]) -> Vec<Count> {
        let mut counts = Vec::with_capacity(blocks.len());
        for &(block, writes) in blocks {
            let users = self.pending.entry(block).or_default();
            users.users += 1;
            let write = writes.then(|| {
                let number = self.next_writer;
                self.next_writer += 1;
                users.writers.insert(number);
                number
            });
            counts.push((block, write));
        }
        counts
    }

    /// Whether a copy out of the block at `block`'s cache waits for the buffers that may write it.
    fn deferring(&self, block: u64) -> bool {
        self.deferred.contains_key(&block)
    }

    /// Runs `op` on the caches once every command buffer begun and not finished that may write
    /// the block at `block`'s cache has finished, and after the copies out of it deferred before
    /// it: at once where none is, else in the finish of the last of them
    /// ([`Self::run_deferred`]).
    fn after_writers(&mut self, block: u64, op: CopyOut<P>) {
        if self.pending(block, true) == 0 && !self.deferring(block) {
            op(self);
            return;
        }
        let upto = self.next_writer;
        (self.deferred.entry(block).or_default()).push_back(Deferred { upto, run: op });
    }

    /// Runs the copies out of the block at `block`'s cache whose writers have all finished, in
    /// the order they were asked.
    fn run_deferred(&mut self, block: u64) {
        loop {
            let oldest = (self.pending.get(&block)).and_then(|u| u.writers.first().copied());
            let Some(queue) = self.deferred.get_mut(&block) else {
                return;
            };
            if (queue.front()).is_some_and(|op| oldest.is_some_and(|number| number < op.upto)) {
                return;
            }
            let Some(op) = queue.pop_front() else {
                self.deferred.remove(&block);
                return;
            };
            if queue.is_empty() {
                self.deferred.remove(&block);
            }
            (op.run)(self);
        }
    }

    /// A fill of the block at `block` waits no more.
    fn unfill(&mut self, block: u64) {
        if let Some(n) = self.filling.get_mut(&block) {
            *n -= 1;
            if *n == 0 {
                self.filling.remove(&block);
            }
        }
    }

    /// The command buffer whose counts are `counts` is finished: it leaves them, and the copies
    /// out of their caches that waited for it alone run ([`Self::run_deferred`]).
    fn unhold(&mut self, counts: &[Count]) {
        for &(block, write) in counts {
            if let Some(users) = self.pending.get_mut(&block) {
                users.users = users.users.saturating_sub(1);
                if let Some(number) = write {
                    users.writers.remove(&number);
                }
                if users.users == 0 {
                    self.pending.remove(&block);
                }
            }
            if write.is_some() {
                self.run_deferred(block);
            }
        }
    }

    /// Ends command buffer `cb` of queue `queue`, which used the caches of `used`: it signals the
    /// queue's end event, which a later copy of each waits for.
    pub(crate) fn after(
        &mut self,
        device: &P::Device,
        queue: u64,
        cb: &P::CommandBuffer,
        used: &[u64],
    ) -> RaiseResult<()> {
        if used.is_empty() {
            return Ok(());
        }
        let (event, value) = match self.ends.entry(queue) {
            Entry::Occupied(end) => end.into_mut(),
            Entry::Vacant(end) => match device.new_event(ON)? {
                Some(event) => end.insert((event, 0)),
                None => return Ok(()),
            },
        };
        *value += 1;
        cb.encode_signal(Some(EventRef::Event(event)), *value)?;
        for block in used {
            if let Some(cache) = self.by_block.get_mut(block) {
                cache.users.insert(queue, *value);
            }
        }
        Ok(())
    }

    /// The early copies behind command buffer `writer`, just committed on queue `queue`, whose
    /// counts among the caches' users are `counts` (the module doc): each cache it may write
    /// whose surface reads back ([`ReadBack`]), where its write is the only one since the cache's
    /// last copy and no device of the host wrote the pages since its last fill, is copied into
    /// its pages by a command buffer of the host's committed on `on`, the writer's own queue,
    /// right behind it, so the copy starts as the writer's work ends, waiting for nothing else.
    /// That buffer signals the queue's end event next, which later copies of the cache wait for;
    /// the cache's write is then the pages', once the copy lands ([`Landing`]).
    fn early_copies(
        &mut self,
        queue: u64,
        on: &P::CommandQueue,
        counts: &[Count],
    ) -> RaiseResult<()> {
        for &(block, write) in counts {
            if write.is_none() || self.watched.written(block) {
                continue;
            }
            let (Some(cache), Some((event, value))) =
                (self.by_block.get(&block), self.ends.get(&queue))
            else {
                continue;
            };
            let read_back = &cache.read_back;
            if !read_back.on || read_back.writes != 1 || !cache.written {
                continue;
            }
            // Pages in a buffer alone: a deferred block's not named yet have none, and copied
            // pages ([`CachePages::Copied`]) take no early copy.
            let Some(CachePages::View(pages)) = &cache.pages else {
                continue;
            };
            let Some(cb) = on.command_buffer(ON)? else {
                continue;
            };
            let Some(blit) = cb.blit_encoder(ON)? else {
                continue;
            };
            blit.encode(
                ON,
                BlitCmd::CopyBufferToBuffer {
                    source_buffer: Some(&cache.device),
                    source_offset: 0,
                    destination_buffer: Some(pages),
                    destination_offset: 0,
                    size: cache.length,
                },
            )?;
            blit.encode(ON, BlitCmd::EndEncoding)?;
            let landing = Landing::new();
            // The answers waiting go as the platform sees the copy done, on its thread, with no
            // wake of a handler thread between; the buffers live until its completed handler,
            // past the cache's own end at a release.
            let lands = Arc::clone(&landing);
            cb.add_completion_seen_handler(ON, Box::new(move |_: CompletedInfo| lands.land()))?;
            let held = (cache.device.clone_ref(), pages.clone_ref());
            cb.add_completed_handler(ON, Box::new(move |_: CompletedInfo| drop(held)))?;
            // The queue's next end value, the queue's once the buffer is committed.
            let next = value + 1;
            cb.encode_signal(Some(EventRef::Event(event)), next)?;
            cb.commit(ON)?;
            let length = cache.length;
            if let Some((_, value)) = self.ends.get_mut(&queue) {
                *value = next;
            }
            if let Some(cache) = self.by_block.get_mut(&block) {
                cache.written = false;
                cache.users.insert(queue, next);
                let read_back = &mut cache.read_back;
                read_back.writes = 0;
                read_back.unread = true;
                read_back.copy = Some(EarlyCopy {
                    queue,
                    value: next,
                    landing,
                });
            }
            self.counts.early += 1;
            self.counts.early_bytes += length;
        }
        Ok(())
    }

    /// Fills the cache of the block at `block` from the surface's pages, as of CPU generation
    /// `generation` or the newer one it holds, on the transfer queue, after every command buffer
    /// that used it before.
    fn fill(&mut self, device: &P::Device, block: u64, generation: u64) -> RaiseResult<()> {
        if !self.make_transfer(device)? {
            return Ok(());
        }
        let (Some(transfer), Some(cache)) = (self.transfer.as_mut(), self.by_block.get(&block))
        else {
            return Ok(());
        };
        // A deferred block's pages, not named yet, hold nothing the cache does not: the CPU has
        // not locked the surface for writing.
        let Some(pages) = &cache.pages else {
            return Ok(());
        };
        let Some(cb) = transfer.queue.command_buffer(ON)? else {
            return Ok(());
        };
        wait_for_users::<P>(&cb, &self.ends, cache)?;
        if !pages_into_cache::<P>(device, &cb, pages, &cache.device, cache.length)? {
            return Ok(());
        }
        transfer.signalled += 1;
        let value = transfer.signalled;
        cb.encode_signal(Some(EventRef::Event(&transfer.event)), value)?;
        let fills = Arc::clone(&transfer.fills);
        cb.add_completed_handler(ON, Box::new(move |_: CompletedInfo| fills.finished(value)))?;
        cb.commit(ON)?;
        let length = cache.length;
        if let Some(cache) = self.by_block.get_mut(&block) {
            cache.generation = Some(cache.generation.map_or(generation, |g| g.max(generation)));
            cache.filled_at = value;
            cache.users.clear();
        }
        self.counts.fills += 1;
        self.counts.filled_bytes += length;
        Ok(())
    }

    /// Fills the cache of the block at `block` where it holds an older generation than
    /// `generation`, or a device of the host wrote its pages since its last fill: a present of
    /// the surface shows the pages as of that generation. The wait for the cache's last fill,
    /// which the present makes in the present queue; `None` where nothing was filled.
    fn fill_now(
        &mut self,
        device: &P::Device,
        block: u64,
        generation: u64,
    ) -> RaiseResult<Option<FillWait>> {
        let Some(cache) = self.by_block.get(&block) else {
            return Ok(None);
        };
        let older = cache.generation.is_none_or(|held| generation > held);
        let host_wrote = self.watched.take(block);
        if !older && !host_wrote {
            return Ok(None);
        }
        if !older {
            self.counts.host_writes += 1;
        }
        self.fill(device, block, generation)?;
        Ok(self.last_fill(block))
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
    ) -> RaiseResult<()> {
        let written = self.written_since_host(block);
        if !written || self.behind_the_cpu(block, cpu, "SURFACE_DOWNLOAD") {
            // An early copy, landed or not, holds the cache's last write: answered as it lands.
            let early = (self.by_block.get_mut(&block)).and_then(|cache| {
                let read_back = &mut cache.read_back;
                let copy = read_back.copy.as_ref()?;
                if std::mem::take(&mut read_back.unread) {
                    self.counts.early_read += 1;
                }
                Some(Arc::clone(&copy.landing))
            });
            match early {
                Some(landing) => landing.then(done),
                None => {
                    if !written && self.by_block.contains_key(&block) {
                        self.counts.clean += 1;
                    }
                    done();
                }
            }
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
        self.write_back(device, block, done)
    }

    /// The block at `block` went (its `BLOCK_RELEASE`): its cache goes once the pages hold what a
    /// buffer wrote of it, unless a device of the host wrote them since its last fill, or the
    /// CPU's generation `cpu` of the surface, as the copy is made, is past the cache's
    /// ([`Self::behind_the_cpu`]); `done` then, at once where nothing is to copy. A release whose
    /// surface had no client left
    /// (`unread`, `RGFX_BLOCK_RELEASE_NO_CLIENT`) copies too: a surface with no client can still
    /// be looked up again, from a mach port or an XPC object that holds it (how one process hands
    /// a surface it drew to another), and the process that looks it up then reads the pages. The
    /// cache is returned, for the caller to let go of.
    pub(crate) fn released(
        &mut self,
        device: &P::Device,
        block: u64,
        unread: bool,
        cpu: Option<u64>,
        done: Box<dyn FnOnce() + Send>,
    ) -> RaiseResult<Option<SurfaceCache<P>>> {
        if !self.written_since_host(block) || self.behind_the_cpu(block, cpu, "BLOCK_RELEASE") {
            // The pages are let go of once an early copy into them still in flight lands.
            let copy = (self.by_block.get(&block))
                .and_then(|cache| cache.read_back.copy.as_ref())
                .map(|copy| Arc::clone(&copy.landing));
            match copy {
                Some(landing) => landing.then(done),
                None => done(),
            }
        } else {
            if unread {
                self.counts.unread += 1;
            }
            self.write_back(device, block, done)?;
        }
        let gone = self.by_block.remove(&block);
        if let Some(cache) = gone.as_ref().filter(|c| c.deferred.is_some()) {
            self.deferred_bytes = self.deferred_bytes.saturating_sub(cache.length);
            if cache.pages.is_none() {
                self.counts.unattached += 1;
            }
        }
        self.watched.0.lock().remove(&block);
        Ok(gone)
    }

    /// Whether the block's cache holds a command buffer's write the pages do not: one was
    /// committed since its last copy, and no device of the host wrote the pages since its last
    /// fill. Where one did, its write is the newer, and the cache's is dropped.
    fn written_since_host(&mut self, block: u64) -> bool {
        let Some(cache) = self.by_block.get_mut(&block) else {
            return false;
        };
        if cache.written && self.watched.written(block) {
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
            target: "reims_vgpu::graphics",
            "the IOSurface block at {block:#x}: a {message} of its device cache, of CPU \
             generation {held}, copies nothing: the CPU's generation moved to {cpu} under a lock \
             that did not wait for the copy"
        );
        true
    }

    /// Copies the cache of the block at `block` into the surface's pages on the transfer queue,
    /// after every command buffer that used it; `done` once the copy has finished. Where no copy
    /// can be made `done` is given at once.
    fn write_back(
        &mut self,
        device: &P::Device,
        block: u64,
        done: Box<dyn FnOnce() + Send>,
    ) -> RaiseResult<()> {
        // A deferred block whose pages are not named has nowhere to copy to.
        if self.unattached(block).is_some() || !self.make_transfer(device)? {
            done();
            return Ok(());
        }
        let (Some(transfer), Some(cache)) = (self.transfer.as_mut(), self.by_block.get(&block))
        else {
            done();
            return Ok(());
        };
        let (Some(pages), Some(cb)) = (&cache.pages, transfer.queue.command_buffer(ON)?) else {
            done();
            return Ok(());
        };
        // A copied cache's pages take the staging buffer's bytes once the GPU's copy into it is
        // done, before `done`.
        let (to, copied) = match pages {
            CachePages::View(pages) => (pages.clone_ref(), None),
            CachePages::Copied(copied) => {
                let Some(staging) =
                    device.new_buffer_of_length(ON, cache.length, STORAGE_SHARED)?
                else {
                    done();
                    return Ok(());
                };
                (staging, Some(Arc::clone(copied)))
            }
        };
        wait_for_users::<P>(&cb, &self.ends, cache)?;
        copy::<P>(&cb, &cache.device, &to, cache.length)?;
        // The buffers live until the copy is done, past the cache's own end at a release.
        let length = cache.length;
        let held = (cache.device.clone_ref(), to);
        cb.add_completed_handler(
            ON,
            Box::new(move |_: CompletedInfo| {
                if let Some(copied) = copied {
                    let contents = held.1.contents().as_ptr::<u8>();
                    // SAFETY: the staging buffer's `length` Shared bytes, which the copy into it,
                    // now done, was the last GPU work to use; the pages are named, so wired.
                    unsafe { copied.copy(contents, length, true) };
                }
                drop(held);
                done();
            }),
        )?;
        cb.commit(ON)?;
        let length = cache.length;
        if let Some(cache) = self.by_block.get_mut(&block) {
            cache.written = false;
            cache.users.clear();
            // This copy is the cache's last: an early copy before it is waited for on the GPU.
            cache.read_back.writes = 0;
            cache.read_back.copy = None;
        }
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

    /// Queue `queue` is gone: its end event, where no command buffer of it signalled it, so no
    /// copy waits for it, to let go of. One a buffer signalled stays, for the copies that wait
    /// for the queue's last buffers.
    fn queue_ended(&mut self, queue: u64) -> Option<P::Event> {
        match self.ends.entry(queue) {
            Entry::Occupied(end) if end.get().1 == 0 => Some(end.remove().0),
            _ => None,
        }
    }

    /// The device's reset: every block's cache goes uncopied, as its surface's pages are the
    /// rebooted guest's again, and so do the queues' end events, each buffer and event let go of
    /// in a refusal scope, and every copy out of a cache still deferred is dropped unmade, its
    /// answer owed to no one (no command buffer is begun across the reset, so none waits). The
    /// host's transfer queue stays, and so do the counts.
    pub(crate) fn reset(&mut self) {
        let caches = std::mem::take(&mut self.by_block);
        let ends = std::mem::take(&mut self.ends);
        self.deferred.clear();
        self.deferred_bytes = 0;
        self.watched.0.lock().clear();
        if let Err(refusal) = refusals::refusing(move || drop((caches, ends))) {
            log::warn!(
                target: "reims_vgpu::graphics",
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
/// ([`Uses`]): where its use is the walk's alone, between them, a copy waits for it.
///
/// - **A copy into a cache** (a fill, at a command buffer's begin or a present) waits for every
///   command buffer begun and not finished that uses the cache, and a later command buffer's use
///   of the cache waits for a fill waiting so, so fills are not held off for good.
/// - **A copy out of a cache** (a download, a release, the block's pages named again) waits for
///   every one of them that may write it. A download and a release, which ring 0 asks for (a
///   download on its block's lane, a release on its context's lane) and are answered when
///   their copy is done, do not wait there: each is made in the
///   finish of the last buffer that may write the cache begun before it was asked
///   ([`Caches::after_writers`]), in the order they were asked, and a command buffer begun after
///   it waits at its begin until it is made. The block's pages named again, which the buffers
///   over the block made on ring 0 after it lie over, waits there.
///
/// So a copy comes before or after each such buffer's whole use, as under one lock, and the
/// queues' end events order it on the GPU after the buffers committed before it
/// ([`SurfaceCaches::after`]). A command buffer counts its blocks in their order, and waits only
/// for a block past every block it counts (a fill or a copy out of it waits only for the buffers
/// begun before it, which wait for no block past them), so no two wait for each other. The waits
/// hold no lock (the cache's lock is let go while they wait).
pub(crate) struct Caches<P: Platform> {
    state: Mutex<SurfaceCaches<P>>,
    moved: Condvar,
    /// What each fill tells as it finishes ([`Self::on_fill_done`]).
    on_fill_done: Arc<OnceLock<FillHook>>,
}

/// A command buffer's count among the users of the blocks it uses ([`Caches`]), until its finish
/// ([`Caches::finish`]) or its drop.
pub(crate) struct Uses<P: Platform> {
    caches: Arc<Caches<P>>,
    /// Each block, and the number of its write where the buffer may write it.
    blocks: Vec<Count>,
}

impl<P: Platform> Drop for Uses<P> {
    fn drop(&mut self) {
        if self.blocks.is_empty() {
            return;
        }
        self.caches.state.lock().unhold(&self.blocks);
        self.caches.moved.notify_all();
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
    /// No caches; `on` keeps them where the kext asks.
    pub(crate) fn new(on: bool) -> Self {
        let on_fill_done = Arc::default();
        Caches {
            state: Mutex::new(SurfaceCaches::new(on, Arc::clone(&on_fill_done))),
            moved: Condvar::new(),
            on_fill_done,
        }
    }

    /// Has each fill tell `hook` as it finishes, after the threads waiting for it are woken: the
    /// host's presents that wait for a fill ([`Self::fill_now`]). Set once; a second is ignored.
    pub(crate) fn on_fill_done(&self, hook: FillHook) {
        let _ = self.on_fill_done.set(hook);
    }

    /// Command queue `queue` was made: where the device keeps caches, its end event is made now,
    /// and the transfer queue too where there is none yet, each with no lock held. A command
    /// buffer's begin and finish, under the caches' lock, then make neither
    /// ([`SurfaceCaches::after`], [`SurfaceCaches::make_transfer`]): every other queue's buffers
    /// that use a cache would wait for the creation, which the platform's other threads' work on
    /// the device's memory and semaphores can stretch to milliseconds (measured on the RTX 5080
    /// Laptop GPU host: 0.5 to 3 ms an end event at a queue's first buffer in a guest's boot,
    /// against 0.1 ms where no other thread makes or frees the device's memory or semaphores).
    pub(crate) fn queue_made(&self, device: &P::Device, queue: u64) -> RaiseResult<()> {
        let (on, transfer) = {
            let state = self.state.lock();
            (state.on, state.transfer.is_some())
        };
        if !on {
            return Ok(());
        }
        let mut end = device.new_event(ON)?;
        let mut made = match transfer {
            true => None,
            false => Transfer::new(device, &self.on_fill_done)?,
        };
        {
            let mut state = self.state.lock();
            // Kept only where the caches are still on, and where no buffer's begin or finish made
            // the same meanwhile; else let go of below.
            if state.on {
                if let Some(event) = end.take() {
                    match state.ends.entry(queue) {
                        Entry::Vacant(slot) => {
                            slot.insert((event, 0));
                        }
                        Entry::Occupied(_) => end = Some(event),
                    }
                }
                if state.transfer.is_none() {
                    state.transfer = made.take();
                }
            }
        }
        if (end.is_some() || made.is_some())
            && let Err(refusal) = refusals::refusing(move || drop((end, made)))
        {
            log::warn!(
                target: "reims_vgpu::graphics",
                "queue {queue:#x}'s spare end event or transfer queue was let go of, and its \
                 release was refused: {refusal}"
            );
        }
        Ok(())
    }

    /// Command queue `queue` is gone (its destroy, or its context's close): its end event goes
    /// where no buffer of it signalled it ([`SurfaceCaches::queue_ended`]), let go of with no
    /// lock held. A refusal is logged.
    pub(crate) fn queue_ended(&self, queue: u64) {
        let Some(end) = self.state.lock().queue_ended(queue) else {
            return;
        };
        if let Err(refusal) = refusals::refusing(move || drop(end)) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "queue {queue:#x}'s end event was let go of, and its release was refused: \
                 {refusal}"
            );
        }
    }

    /// The caches, locked: for what orders nothing against command buffers' uses.
    pub(crate) fn lock(&self) -> MutexGuard<'_, SurfaceCaches<P>> {
        self.state.lock()
    }

    /// Whether the block at `block` has a cache.
    pub(crate) fn has(&self, block: u64) -> bool {
        self.state.lock().has(block)
    }

    /// The device's reset ([`SurfaceCaches::reset`]), which no command buffer is begun across:
    /// the guest's messages carried out before it have ended.
    pub(crate) fn reset(&self) {
        self.state.lock().reset();
        self.moved.notify_all();
    }

    /// Starts again with no caches; `on` keeps them where the kext asks. The deferred caches'
    /// budget stays.
    pub(crate) fn set_on(&self, on: bool) {
        let mut state = self.state.lock();
        let budget = state.deferred_budget;
        *state = SurfaceCaches::new(on, Arc::clone(&self.on_fill_done));
        state.deferred_budget = budget;
    }

    /// Sets the bytes deferred blocks' caches may hold at once (the device's `DEFERRED_BUDGET`,
    /// read once at its start); 0, as at the caches' make, defers none.
    pub(crate) fn set_deferred_budget(&self, budget: u64) {
        self.state.lock().set_deferred_budget(budget);
    }

    /// Waits, letting the lock go meanwhile, while a command buffer begun and not finished uses
    /// the block at `block`'s cache (`writers`: one that may write it).
    fn wait_users(&self, state: &mut MutexGuard<'_, SurfaceCaches<P>>, block: u64, writers: bool) {
        while state.pending(block, writers) > 0 {
            self.moved.wait(state);
        }
    }

    /// Waits, letting the lock go meanwhile, while a fill of the block at `block` waits.
    fn wait_fills(&self, state: &mut MutexGuard<'_, SurfaceCaches<P>>, block: u64) {
        while state.filling.get(&block).is_some_and(|&n| n > 0) {
            self.moved.wait(state);
        }
    }

    /// The blocks `uses` names that have caches, before command buffer `cb` runs, and its count
    /// among their users: each filled where it needs it ([`SurfaceCaches::needs_fill`]), after
    /// every command buffer begun and not finished that uses it, and `cb` made to wait for each
    /// one's last fill not yet done; and whether it waits for one. A use that may write marks its
    /// cache written.
    pub(crate) fn before(
        self: &Arc<Self>,
        device: &P::Device,
        cb: &P::CommandBuffer,
        uses: &[RgfxOpSurfaceUse],
    ) -> RaiseResult<(Vec<u64>, bool, Option<Uses<P>>)> {
        if uses.is_empty() {
            return Ok((Vec::new(), false, None));
        }
        let mut state = self.state.lock();
        let mut held = Uses {
            caches: Arc::clone(self),
            blocks: Vec::new(),
        };
        let mut waits = false;
        let picked: Vec<RgfxOpSurfaceUse> = by_block(&state, uses).into_iter().copied().collect();
        for u in &picked {
            let block = u.block_offset;
            // After the copies out of the cache asked before this begin, which wait only for the
            // buffers begun before them ([`Self::after_writers`]).
            while state.deferring(block) {
                self.moved.wait(&mut state);
            }
            // A fill waits for the block's users, and a use behind a fill waiting for it; what
            // the use needs is read again after each wait.
            loop {
                if state.needs_fill(u) {
                    *state.filling.entry(block).or_default() += 1;
                    self.wait_users(&mut state, block, false);
                    state.unfill(block);
                    self.moved.notify_all();
                    break;
                }
                if !state.filling.contains_key(&block) {
                    break;
                }
                self.wait_fills(&mut state, block);
            }
            if !state.has(block) {
                continue;
            }
            let outcome = state.before_one(device, cb, u);
            // Counted at once, so a later fill of the block waits for this buffer.
            let counted = state.hold(&[(block, u.flags & RGFX_SURFACE_USE_WRITES != 0)]);
            held.blocks.extend(counted);
            match outcome {
                Ok(w) => waits |= w,
                Err(raise) => {
                    drop(state);
                    return Err(raise);
                }
            }
        }
        drop(state);
        let used = held.blocks.iter().map(|&(b, _)| b).collect();
        Ok((used, waits, (!held.blocks.is_empty()).then_some(held)))
    }

    /// Whether none of the blocks `blocks` has a cache, so a command buffer that uses only them
    /// needs no fill and counts among no cache's users: answered waiting for nothing and
    /// allocating nothing (a doorbell's thread), `None` where the lock is busy.
    pub(crate) fn none_cached(&self, mut blocks: impl Iterator<Item = u64>) -> Option<bool> {
        let state = self.state.try_lock()?;
        Some(blocks.all(|block| !state.has(block)))
    }

    /// Ends command buffer `cb` of queue `queue`, which `uses` counted among the users of the
    /// caches of `used` ([`SurfaceCaches::after`]), and `commit`s it, with the lock held, so a
    /// later copy finds its end event's value committed; then commits the early copies behind it
    /// on `on`, its queue, where it has one ([`SurfaceCaches::early_copies`]), and it leaves
    /// their users. An early copy the platform refuses is logged and left unmade: the cache stays
    /// written, for the download that follows.
    #[expect(
        clippy::too_many_arguments,
        reason = "the buffer's queue, as the guest and the platform name it, and its counts"
    )]
    pub(crate) fn finish(
        &self,
        device: &P::Device,
        queue: u64,
        on: Option<&P::CommandQueue>,
        cb: &P::CommandBuffer,
        mut uses: Uses<P>,
        used: &[u64],
        commit: impl FnOnce() -> RaiseResult<()>,
    ) -> RaiseResult<()> {
        let mut state = self.state.lock();
        let done = state.after(device, queue, cb, used).and_then(|()| commit());
        if done.is_ok()
            && let Some(on) = on
            && let Err(raise) = refusing(|| state.early_copies(queue, on, &uses.blocks))
        {
            log::warn!(
                target: "reims_vgpu::graphics",
                "queue {queue:#x}: the platform raised at an early copy of a device cache into \
                 its pages ({raise:?}); the lock that follows downloads it"
            );
        }
        state.unhold(&uses.blocks);
        uses.blocks.clear();
        drop(state);
        self.moved.notify_all();
        done
    }

    /// Fills the cache of the block at `block` where a present of it at `generation` needs it
    /// ([`SurfaceCaches::fill_now`]), after every command buffer begun and not finished that uses
    /// it: the wait for the fill, which the caller hands to the present, waiting for nothing on
    /// the GPU; `None` where the present waits for no fill.
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
        *state.filling.entry(block).or_default() += 1;
        self.wait_users(&mut state, block, false);
        let filled = state.fill_now(device, block, generation);
        state.unfill(block);
        drop(state);
        self.moved.notify_all();
        filled
    }

    /// `SURFACE_DOWNLOAD` of the block at `block` ([`SurfaceCaches::download`]), after every
    /// command buffer begun and not finished that may write it, waiting for none
    /// ([`Self::after_writers`]): `done` once the pages hold the cache, or `failed` where the
    /// platform raised at the copy.
    pub(crate) fn download(
        &self,
        device: &P::Device,
        block: u64,
        region: Arc<Region>,
        done: Box<dyn FnOnce() + Send>,
        failed: Box<dyn FnOnce(Raise) + Send>,
    ) {
        let device = device.clone_ref();
        self.after_writers(
            block,
            Box::new(move |state| {
                let cpu = cpu_generation(&region, block);
                if let Err(raise) = refusing(|| state.download(&device, block, cpu, done)) {
                    failed(raise);
                }
            }),
        );
    }

    /// The block at `block` went, its `BLOCK_RELEASE` ([`SurfaceCaches::released`]), after every
    /// command buffer begun and not finished that may write its cache, waiting for none
    /// ([`Self::after_writers`]): `done` once the pages hold the cache, or `failed` where the
    /// platform raised at the copy, and the cache handed to `let_go`, which lets go of it on a
    /// thread of its own, with no lock of the caches' held. Nothing names the block until it is
    /// answered: the kext gives it out again only then.
    #[allow(clippy::too_many_arguments)] // the release, its answers, and what the copy reads
    pub(crate) fn release(
        &self,
        device: &P::Device,
        block: u64,
        unread: bool,
        region: Arc<Region>,
        done: Box<dyn FnOnce() + Send>,
        failed: Box<dyn FnOnce(Raise) + Send>,
        let_go: Box<dyn FnOnce(SurfaceCache<P>) + Send>,
    ) {
        let device = device.clone_ref();
        self.after_writers(
            block,
            Box::new(move |state| {
                let cpu = cpu_generation(&region, block);
                match refusing(|| state.released(&device, block, unread, cpu, done)) {
                    Ok(Some(cache)) => let_go(cache),
                    Ok(None) => {}
                    Err(raise) => failed(raise),
                }
            }),
        );
    }

    /// Runs `op` on the caches once every command buffer begun and not finished that may write
    /// the block at `block`'s cache has finished, after the copies out of it asked before: here
    /// where none is, else in the last one's finish, on its thread, waiting for nothing here. A
    /// command buffer begun meanwhile that uses the cache waits at its begin until `op` has run
    /// ([`Self::before`]), as one begun after the copy, which the ring-0 thread that asked for it
    /// made before going on, waited for it.
    fn after_writers(&self, block: u64, op: CopyOut<P>) {
        let mut state = self.state.lock();
        state.after_writers(block, op);
        drop(state);
        self.moved.notify_all();
    }

    /// The block at `block` named again without a cache, or over other pages, loses it
    /// ([`SurfaceCaches::released`]), after every command buffer begun and not finished that may
    /// write it, waiting for them here: the buffers made over the block after it lie over its
    /// pages. The cache is returned, for the caller to let go of with the lock let go.
    pub(crate) fn released(
        &self,
        device: &P::Device,
        block: u64,
        unread: bool,
        done: Box<dyn FnOnce() + Send>,
    ) -> RaiseResult<Option<SurfaceCache<P>>> {
        let mut state = self.state.lock();
        self.wait_users(&mut state, block, true);
        state.released(device, block, unread, None, done)
    }

    /// A cache for the block at `block` ([`SurfaceCaches::make`]), after every command buffer
    /// begun and not finished that may write the cache it had.
    pub(crate) fn make(
        &self,
        device: &P::Device,
        block: u64,
        view: &Arc<SurfaceView<P>>,
        host_pages: Vec<Range<u64>>,
    ) -> RaiseResult<bool> {
        let mut state = self.state.lock();
        self.wait_users(&mut state, block, true);
        state.make(device, block, view, host_pages)
    }

    /// A cache of zeros for the deferred block at `block` ([`SurfaceCaches::make_deferred`]),
    /// after every command buffer begun and not finished that may write the cache it had.
    pub(crate) fn make_deferred(
        &self,
        device: &P::Device,
        block: u64,
        length: u64,
        block_size: u64,
    ) -> RaiseResult<bool> {
        let mut state = self.state.lock();
        self.wait_users(&mut state, block, true);
        state.make_deferred(device, block, length, block_size)
    }

    /// The deferred block at `block`'s pages taken for its cache ([`SurfaceCaches::attach`]),
    /// which copies nothing, so waits for no command buffer.
    pub(crate) fn attach(
        &self,
        device: &P::Device,
        block: u64,
        view: &Arc<SurfaceView<P>>,
        host_pages: Vec<Range<u64>>,
    ) -> RaiseResult<Option<bool>> {
        self.state.lock().attach(device, block, view, host_pages)
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
    /// `timeout` at most, with no lock held, woken by the fill's completed handler: for work the
    /// host starts outside a command buffer that writes the cache, which a command buffer's wait
    /// does not order. True where it is done, or the block has no cache or no fill.
    pub(crate) fn wait_filled(&self, block: u64, timeout: std::time::Duration) -> bool {
        let Some(fill) = self.state.lock().last_fill(block) else {
            return true;
        };
        fill.fills.wait(fill.value, timeout)
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
}

/// Makes `cb` wait for the latest command buffer of each queue that used `cache`.
fn wait_for_users<P: Platform>(
    cb: &P::CommandBuffer,
    ends: &HashMap<u64, (P::Event, u64)>,
    cache: &SurfaceCache<P>,
) -> RaiseResult<()> {
    for (queue, &value) in &cache.users {
        if let Some((event, _)) = ends.get(queue) {
            cb.encode_wait(Some(EventRef::Event(event)), value)?;
        }
    }
    Ok(())
}

/// Encodes on `cb` a copy of the surface's `length` bytes from its pages into `cache`: from its
/// view, or, where they are copied, from a Shared buffer of the copy's own that the host's CPU
/// fills from the pages now, the CPU's writes before the command buffer that asked the fill in
/// them, and that lives until the copy is done. False, with nothing encoded, where the platform
/// made no such buffer.
fn pages_into_cache<P: Platform>(
    device: &P::Device,
    cb: &P::CommandBuffer,
    pages: &CachePages<P>,
    cache: &P::Buffer,
    length: u64,
) -> RaiseResult<bool> {
    let copied = match pages {
        CachePages::View(view) => {
            copy::<P>(cb, view, cache, length)?;
            return Ok(true);
        }
        CachePages::Copied(copied) => copied,
    };
    let Some(staging) = device.new_buffer_of_length(ON, length, STORAGE_SHARED)? else {
        return Ok(false);
    };
    // SAFETY: the new buffer's `length` Shared bytes, which no GPU work uses yet; the pages are
    // named, so wired.
    unsafe { copied.copy(staging.contents().as_ptr::<u8>(), length, false) };
    copy::<P>(cb, &staging, cache, length)?;
    cb.add_completed_handler(ON, Box::new(move |_: CompletedInfo| drop(staging)))?;
    Ok(true)
}

/// Encodes a copy of `length` bytes from `from` to `to` on `cb`.
fn copy<P: Platform>(
    cb: &P::CommandBuffer,
    from: &P::Buffer,
    to: &P::Buffer,
    length: u64,
) -> RaiseResult<()> {
    let Some(blit) = cb.blit_encoder(ON)? else {
        return Ok(());
    };
    blit.encode(
        ON,
        BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(from),
            source_offset: 0,
            destination_buffer: Some(to),
            destination_offset: 0,
            size: length,
        },
    )?;
    blit.encode(ON, BlitCmd::EndEncoding)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use reims_graphics::ops::{
        RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_OP_SUBMITTED, RGFX_OP_SURFACE_USE, RgfxCommandBuffer,
        RgfxOpHeader,
    };
    use reims_graphics::wire::{
        RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_QUEUE_CREATE,
        RGFX_STATUS_INVALID_ARGUMENT, RGFX_STATUS_OK, RGFX_STATUS_REFUSED,
        RGFX_SURFACE_PAGES_ATTACH, RGFX_SURFACE_PAGES_CACHED, RGFX_SURFACE_PAGES_DEFERRED,
        RgfxMsgHeader, RgfxObjectDestroy, RgfxQueueCreate,
    };
    use reims_vgpu_core::testkit::calllog::{Arg, Call};
    use reims_vgpu_core::testkit::objects::CompleteOn;
    use zerocopy::IntoBytes;

    use super::super::surface_pages::tests::{BLOCK, RAM, Rig, rig};
    use super::super::{DecodeError, no_raise};
    use super::*;

    const COPY: &str = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";
    const WAIT: &str = "encodeWaitForEvent:value:";
    const SIGNAL: &str = "encodeSignalEvent:value:";

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
        assert_eq!(r.dec.decode(&r.device, &message).unwrap(), Ok(()));
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

    /// A block marked cached gets a device cache, which each of its buffers is; a block not
    /// marked, or any block of a GPU that shares the CPU's memory, gets none
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
        assert_eq!(r.dec.caches().len(), 0, "a GPU of the CPU's memory");
    }

    /// The first use fills the cache from the pages, and so does a use of a newer generation;
    /// one of the generation the cache holds, or an older one, fills nothing. A buffer behind a
    /// fill that has not finished waits for it
    #[test]
    fn a_use_fills_the_cache_once_a_generation() {
        let mut r = cached();
        let cache = cache_id(&r);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        let calls = r.p.call_log().since(mark);
        let filled = copies(&calls);
        assert_eq!(filled.len(), 1);
        assert_eq!(filled[0].1, cache, "into the cache");
        let transfer = events(&calls, SIGNAL)[0];
        assert_eq!(transfer.1, 1);
        assert!(
            events(&calls, WAIT).contains(&transfer),
            "the buffer waits for its fill"
        );
        for (generation, fills) in [(0, 0), (3, 1), (2, 0), (3, 0)] {
            let mark = r.p.call_log().mark();
            use_surfaces(&mut r, 1, 0x101 + generation, &[(BLOCK, generation, false)]);
            assert_eq!(
                copies(&r.p.call_log().since(mark)).len(),
                fills,
                "generation {generation}"
            );
        }
        assert_eq!(r.dec.caches().counts().fills, 2);
    }

    /// A buffer that asks for its `CB_SUBMITTED` and uses a cache the host fills first waits for
    /// the fill, so it gets it at its scheduling, after its commit (the fake schedules a buffer as
    /// it is committed), not as the host takes it; once the fill is done, a buffer of the same
    /// generation waits for nothing and gets it as it is taken, before its commit.
    #[test]
    fn a_buffer_behind_a_fill_is_submitted_at_its_scheduling() {
        let mut r = cached();
        let at = Arc::new(Mutex::new(Vec::new()));
        let (log, marks) = (Arc::clone(&r.p), Arc::clone(&at));
        r.dec.set_submitted_sink(Arc::new(move |m| {
            marks.lock().push((m.cb_obj_id, log.call_log().mark()));
        }));
        // The buffer's own commit, the last of those since `from`.
        let commit = |r: &Rig, from| {
            (r.p.call_log().since(from).iter())
                .filter(|c| c.sel == "commit")
                .map(|c| c.seq)
                .max()
                .unwrap()
        };
        let mark = r.p.call_log().mark();
        use_surfaces_asking(&mut r, 1, 0x100, &[(BLOCK, 0, false)], true);
        assert_eq!(copies(&r.p.call_log().since(mark)).len(), 1, "a fill");
        let (cb, posted) = at.lock()[0];
        assert_eq!(cb, 0x100);
        assert!(posted > commit(&r, mark), "at its scheduling");
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        use_surfaces_asking(&mut r, 1, 0x101, &[(BLOCK, 0, false)], true);
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        let (cb, posted) = at.lock()[1];
        assert_eq!(cb, 0x101);
        assert!(posted <= commit(&r, mark), "as it is taken");
    }

    /// A wait for a cache's fill, as a decode that writes the cache outside a command buffer
    /// takes, lasts while the fill runs and ends once it is done; a block with no cache, or none
    /// filled, waits for nothing
    #[test]
    fn a_wait_for_a_fill_ends_once_it_is_done() {
        let mut r = cached();
        let short = std::time::Duration::from_millis(1);
        assert!(
            r.dec.surface_caches.wait_filled(BLOCK, short),
            "no fill yet"
        );
        assert!(
            r.dec.surface_caches.wait_filled(BLOCK + 0x8000, short),
            "no cache"
        );
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
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
            false,
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
            true,
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
            true,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        assert!(released.load(Ordering::SeqCst), "done at once");
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        let counts = r.dec.caches().counts();
        assert_eq!((counts.downloads, counts.unread), (0, 0));
    }

    /// A block named again over the same pages keeps its cache, which its next use fills again;
    /// one named over other pages, or not cached, loses it. What a buffer wrote of the cache goes
    /// into its pages first each time, and nothing else is copied
    #[test]
    fn a_block_named_again_keeps_what_its_cache_holds() {
        let mut r = cached();
        let page = r.page;
        let cache = cache_id(&r);
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, true)]);
        r.p.world().complete_all(CompleteOn::Caller);
        for (round, copied) in [(0, 1), (1, 0)] {
            let mark = r.p.call_log().mark();
            assert_eq!(
                r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
                RGFX_STATUS_OK
            );
            let back = copies(&r.p.call_log().since(mark));
            assert_eq!(back.len(), copied, "round {round}");
            assert!(
                back.iter().all(|&(from, _)| from == cache),
                "out of the cache"
            );
            assert_eq!(cache_id(&r), cache, "kept, round {round}");
        }
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 1, true)]);
        let filled = copies(&r.p.call_log().since(mark));
        assert_eq!(filled.len(), 1, "filled again, of the generation it held");
        assert_eq!(filled[0].1, cache);
        r.p.world().complete_all(CompleteOn::Caller);

        let mark = r.p.call_log().mark();
        assert_eq!(
            r.pages_cached(BLOCK, &[(3, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        let back = copies(&r.p.call_log().since(mark));
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].0, cache, "the old cache, into its pages");
        let other = cache_id(&r);
        assert_ne!(other, cache, "a new cache");
        use_surfaces(&mut r, 1, 0x102, &[(BLOCK, 1, true)]);
        r.p.world().complete_all(CompleteOn::Caller);

        let mark = r.p.call_log().mark();
        assert_eq!(r.pages(BLOCK, &[(5, 1)], page), RGFX_STATUS_OK);
        let back = copies(&r.p.call_log().since(mark));
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].0, other);
        assert_eq!(r.dec.caches().len(), 0, "not cached");
        r.p.world().complete_all(CompleteOn::Caller);
        let counts = r.dec.caches().counts();
        assert_eq!((counts.made, counts.downloads), (2, 3));
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
    /// a surface waits for its fill. The fills make no queue
    #[test]
    fn the_transfer_queue_is_made_unshared_with_the_first_queue() {
        use reims_vgpu_core::testkit::impls::device::UNSHARED_QUEUE;
        let mut r = cached();
        let (unshared, plain) = (
            r.p.call_log().count(UNSHARED_QUEUE),
            r.p.call_log().count("newCommandQueue"),
        );
        assert_eq!(
            (unshared, plain),
            (1, 2),
            "made with queue 1, beside queues 1 and 2"
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
    /// end event goes where no buffer of it signalled it, and stays where one did, for the copies
    /// that wait for the queue's last buffers
    #[test]
    fn a_queues_end_event_is_made_with_it_and_its_first_buffer_makes_none() {
        let mut r = cached();
        let end = |r: &Rig, queue: u64| {
            (r.dec.caches().ends.get(&queue)).map(|(event, _)| Arg::obj(event).as_obj().unwrap().0)
        };
        let one = end(&r, 1).expect("queue 1's end event, made with it");
        assert!(end(&r, 2).is_some(), "queue 2's");
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
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

    /// A fill of a cache waits for every command buffer begun and not finished that uses it, and
    /// is made once the last of them has finished, as one made after their whole use under one
    /// lock would be (docs/graphics-kext/design.md, "What a walk sees").
    #[test]
    fn a_fill_waits_for_the_buffers_using_the_cache() {
        let mut r = cached();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let queue = r.device.new_command_queue().unwrap();
        let reader = queue.command_buffer(ON).unwrap().unwrap();
        let (_, _, uses) = (r.dec.surface_caches)
            .before(&r.device, &reader, &reading(BLOCK, 1))
            .unwrap();
        assert!(uses.is_some(), "counted among its users");
        let caches = Arc::clone(&r.dec.surface_caches);
        let filler = queue.command_buffer(ON).unwrap().unwrap();
        let mark = r.p.call_log().mark();
        std::thread::scope(|scope| {
            let fill = scope.spawn(|| {
                caches
                    .before(&r.device, &filler, &reading(BLOCK, 2))
                    .map(|_| ())
                    .map_err(|_| ())
            });
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(!fill.is_finished(), "filled under a buffer using the cache");
            assert!(copies(&r.p.call_log().since(mark)).is_empty());
            drop(uses);
            assert!(fill.join().unwrap().is_ok());
        });
        assert_eq!(
            copies(&r.p.call_log().since(mark)).len(),
            1,
            "filled once its user finished"
        );
    }

    /// A download and a release asked while a command buffer that may write the cache is begun
    /// and not finished are not waited for where they are asked: each is made, in the order
    /// asked, in that buffer's finish, after its commit, and a buffer begun meanwhile that uses
    /// the cache waits for them at its begin. Here the
    /// download copies the buffer's write out, and the release then has nothing more to copy:
    /// the cache goes, and the buffer begun after it finds none
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
        let (used, _, uses) = (r.dec.surface_caches)
            .before(&r.device, &writer, &writes)
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
            false,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "copied while the writer is begun"
        );
        assert!(!downloaded.load(Ordering::SeqCst) && !released.load(Ordering::SeqCst));
        let caches = Arc::clone(&r.dec.surface_caches);
        let later = queue.command_buffer(ON).unwrap().unwrap();
        std::thread::scope(|scope| {
            let begin = scope.spawn(|| {
                (caches.before(&r.device, &later, &reading(BLOCK, 1)))
                    .map(|(_, _, uses)| uses.is_some())
                    .map_err(|_| ())
            });
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(!begin.is_finished(), "begun before the copies were made");
            (r.dec.surface_caches)
                .finish(
                    &r.device,
                    1,
                    Some(&queue),
                    &writer,
                    uses.unwrap(),
                    &used,
                    || writer.commit(ON),
                )
                .unwrap();
            assert_eq!(begin.join().unwrap(), Ok(false), "the cache was released");
        });
        let calls = r.p.call_log().since(mark);
        let commit = calls.iter().position(|c| c.sel == "commit").unwrap();
        let copied = calls.iter().position(|c| c.sel == COPY).unwrap();
        assert!(commit < copied, "copied after the writer's commit");
        assert_eq!(copies(&calls).len(), 1, "the download's copy alone");
        assert!(released.load(Ordering::SeqCst), "nothing more to copy");
        assert_eq!(r.dec.caches().len(), 0);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
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
            false,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            no_raise(),
        );
        assert!(released.load(Ordering::SeqCst), "done at once");
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        let counts = r.dec.caches().counts();
        assert_eq!(
            (counts.fills, counts.downloads, counts.host_writes),
            (2, 0, 3)
        );
    }

    /// A buffer's write that a read-write CPU lock waited for is in the pages before the newer
    /// generation's fill: the lock's download copies it out of the cache, and the fill, after it
    /// on the transfer queue, copies the pages and the CPU's writes back in. A write the kext did
    /// not download before the lock (a commit racing the lock on another thread) is filled over,
    /// the CPU's writes winning, and the cache, still written, is copied out at its next download
    #[test]
    fn a_newer_generation_fills_after_the_download_of_a_write() {
        let mut r = cached();
        let cache = cache_id(&r);
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
            let (used, _, uses) = (r.dec.surface_caches)
                .before(&r.device, &writer, &writes)
                .unwrap();
            (writer, used, uses.unwrap())
        };
        let (writer, used, uses) = begin_writer(&r);
        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        // The lock gives up, moves the generation, and the CPU writes the pages; then the writer
        // finishes.
        set_gen(&r, 1);
        (r.dec.surface_caches)
            .finish(&r.device, 1, &writer, uses, &used, || writer.commit(ON))
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
        let (writer, used, uses) = begin_writer(&r);
        let mark = r.p.call_log().mark();
        let released = release(&r, BLOCK);
        set_gen(&r, 2);
        (r.dec.surface_caches)
            .finish(&r.device, 1, &writer, uses, &used, || writer.commit(ON))
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
        let (used, _, uses) = (r.dec.surface_caches)
            .before(&r.device, &writer, &writes)
            .unwrap();
        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        (r.dec.surface_caches)
            .finish(&r.device, 1, &writer, uses.unwrap(), &used, || {
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
            false,
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

    /// A deferred block the budget has no room for, or whose cache the platform does not make,
    /// or on a device that keeps no caches, is refused and leaves nothing behind, so the kext's
    /// `SURFACE_PAGES` of its pages that follows makes its cache as any block's. A release gives
    /// its room back
    #[test]
    fn a_deferred_block_not_made_is_refused_with_nothing_left() {
        use reims_vgpu_core::testkit::Step;
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

        let fourth = BLOCK + 0x18000;
        r.dec.set_unified_memory(true);
        r.dec.set_deferred_budget(8 * page);
        assert_eq!(defer(&mut r, fourth, page), refused, "no caches");
        let mut none = deferring(0);
        assert_eq!(defer(&mut none, BLOCK, page), refused, "no budget");
    }

    /// The live Shared buffer `id`'s bytes, each set to `with` first where given.
    fn buffer_bytes(r: &Rig, id: u64, with: Option<u8>) -> Vec<u8> {
        use reims_vgpu_core::platform::ObjId;
        use reims_vgpu_core::testkit::objects::kind;
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
        use reims_vgpu_core::testkit::Step;
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
        assert_eq!(
            attach(&mut r, BLOCK, 0, &[(1, 2)], length),
            RGFX_STATUS_OK,
            "no buffer over the view: copied, not refused"
        );
        assert_eq!(r.dec.caches().unattached(BLOCK), None);
        assert_eq!(r.dec.caches().counts().attached_copied, 1);
        assert_eq!(cache_id(&r), cache, "the same cache");
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
        let (used, _, uses) = caches.before(&r.device, &cb, &writes).unwrap();
        (caches.finish(
            &r.device,
            id,
            Some(queue),
            &cb,
            uses.unwrap(),
            &used,
            || cb.commit(ON),
        ))
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

    /// Only a write that was the only one since the cache's last copy, read back, starts the
    /// early copies; a write that no download followed ends them, and the lock after it
    /// downloads as before, which starts them again. A writer whose queue the decoder does not
    /// hand over (one that holds no video encoder) takes none
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

        // One write, then the lock: the next write is copied out early, but not a writer that
        // holds no video encoder.
        let (_, copied) = decode_then_lock(&r, 7, &queue);
        assert_eq!(copied.len(), 1);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 2, 0x201, &[(BLOCK, 0, true)]);
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "no video encoder"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        let done = download(&r, BLOCK);
        assert_eq!(
            copies(&r.p.call_log().since(mark)).len(),
            1,
            "its lock downloads"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
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
        assert_eq!(r.dec.caches().counts().early, 2);
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
}
