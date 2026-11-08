//! IOSurfaces whose GPU view is their own guest pages (`SURFACE_PAGES`,
//! `BUFFER_OVER_SURFACE_PAGES`; docs/graphics-kext/design.md, Decided 14).
//!
//! The accelerator kext wires an IOSurface's allocation while a process holds the surface's
//! shared block, and lists its guest RAM runs in the block (`SURFACE_PAGES`). The host maps the
//! runs as one view through the memory map ([`MemoryMap::new_range_mapping`]), and a `BUFFER_OVER_SURFACE_PAGES` buffer lies over the
//! view, so the GPU reads and writes the surface's own pages in place: no copy of them is made
//! or synced. The view is the block's until its `BLOCK_RELEASE`; each buffer over it holds it
//! until the platform frees the buffer.
//!
//! **The cache.** WindowServer lets a surface's last texture go and makes one again every frame,
//! so the kext gives the surface a fresh block each time (measured on the msi 5080 host,
//! macOS 26 guest, a window dragged: 975 fresh blocks in 30 s, 729 of them one 8 MiB surface's).
//! The [`CACHED_VIEWS`] views made last are kept, by their runs, each with a platform buffer over
//! the whole view that holds the runtime's import of its pages: a block over the same pages takes
//! that view, and buffers over it the live import, with no mapping or import made again. A view
//! maps the guest's pages themselves, so it never goes stale: the memory map's pages stay the
//! guest's RAM while the device lives (`GraphicsConfig::memory_map`). A view costs a kernel
//! mapping a run where the platform counts them, so the cached views no block holds (a video
//! session's outputs once it ends) keep no more mappings, a mapping a run, than the platform
//! answers for them ([`HostVm::idle_view_mappings`]; on Linux the views' budget's headroom): as
//! a view is cached, the oldest of them go until the rest fit. **Measured** 2026-11-07 on the
//! msi 5080 host (`vm.max_map_count` 65,530), an 8 GiB x86-64 macOS 26 guest decoding 1080p
//! H.264, 4K HEVC and 1080p H.264 into views: 31 views made and none taken from the cache, which
//! kept every one, ended sessions' among them, so the process's mappings only grew, from 3,220 at
//! the desktop to 43,949, past the headroom.
//!
//! That bound holds once a view is cached, and the views of a session that just ended keep their
//! mappings until then, so before a view is made, where the platform's room for views' mappings
//! ([`HostVm::view_room`]; on Linux the views' budget's room short of its headroom, past which
//! a view is mapped through dma-bufs that a device pinning its pages cannot import) is less than
//! the view adds, a mapping a run and one at each end, cached views no block holds go first, until
//! the mappings they hold ([`HostVm::view_mappings`]: a mapping a run, fewer where the platform
//! mapped the view another way) cover what it lacks: those whose mapping nothing else holds
//! first, then those a host's video device holds too, the oldest first among each; and none where
//! all of them would not cover it, as the view is then mapped another way whatever goes. Their
//! mappings are torn down where their last holds go ([`Cached::let_go_now`]): on the surface
//! worker where nothing else holds them, else on the runtime's thread that frees their import or
//! the thread that drops a device's import of them, and the view waits for that room,
//! [`ROOM_WAIT_STEPS`] steps of [`ROOM_WAIT_STEP`] at most, before it is mapped; a pages block's
//! view alike. So a view a block needs is not mapped another way while an unheld one holds the
//! room. **Measured** 2026-11-08 on the same host and
//! guest, at 3648 x 2052, with NVDEC's pictures made to land in their pages (a diagnostic build:
//! they land in device caches, which take no view): a 4K HEVC session's views took about 19,000
//! mappings at once (6 to 7 NV12 surfaces of about 3,041 runs), and the views it left kept about
//! 19,000 more until the next view was made. With one such session running and two more run one
//! after the other beside it, room was made 11 times, a view let go of each time: its mappings
//! were never gone at once, gone in 1.1 to 5.1 ms in 8, and not within 50 ms in 3. The two
//! running sessions' live views, which no block lets go of, still passed the headroom: one NV12
//! surface of each later session was mapped through dma-bufs, and its pictures copied by the CPU
//! (200 of 1,200, and 1,500 of 9,000). Run again with the holds of each let-go view and its
//! mapping traced (the same host, guest and sessions), room was made 11 times, one NV12 view of
//! 2,891 to 3,040 runs let go of each time, before views were weighed by their mappings: in 7,
//! NVDEC's hold of the view's memory (its CUDA registration) kept the mapping until a thread of
//! the video crate let go of it, 1.3 to 8.3 ms on; in 1, nothing else held it, and it went at
//! once; in 3, the view
//! had been mapped through dma-bufs and gave back 0 to 3 mappings, and the view being made waited
//! out the 20 ms for room that could not come.
//!
//! [`HostVm::idle_view_mappings`]: reims_vgpu_vulkan::core::platform::os::vm::HostVm::idle_view_mappings
//! [`HostVm::view_room`]: reims_vgpu_vulkan::core::platform::os::vm::HostVm::view_room
//! [`HostVm::view_mappings`]: reims_vgpu_vulkan::core::platform::os::vm::HostVm::view_mappings
//!
//! The runs are the guest's: each must be whole pages of the guest's (4 KiB) and of the host's
//! from a page, lie in guest RAM, and the runs must sum to the surface's length; else the block
//! gets no view, `SURFACE_PAGES` is refused, and the kext gives the surface a copy block instead.
//!
//! **Pages blocks** (`RGFX_FEATURE_PAGES_BUFFERS`). A `SURFACE_PAGES` marked
//! `RGFX_SURFACE_PAGES_PRIVATE` names a block of one process's own, which holds nothing but the
//! runs of pages of that process's the kext wired for it (its `PagesBlock` call): an app's memory
//! under a no-copy buffer. Its runs are checked as a surface's, and must lie in guest RAM, at its
//! `SURFACE_PAGES`; its view is made at its first `BUFFER_OVER_PAGES`, so a pages block no buffer
//! is laid over costs no mapping or import. The view is mapped as a surface's, with a buffer over
//! the whole of it that holds the runtime's import of its pages, and is the block's alone: it stays
//! out of the cache, as no other block names those pages and a view of a GiB would push every
//! surface's out of it. A `BUFFER_OVER_PAGES` of the context the block was made for lies over a
//! part of the view, at the guest's GPU address, which keeps the app pointer's place in its page.
//! The view goes at the block's `BLOCK_RELEASE` or its context's close, once the buffers over it
//! have let it go.
//!
//! **Heaps over pages** (`RGFX_FEATURE_HEAP_OVER_PAGES`). A pages block, a slab of the guest's,
//! can hold a `HEAP_OVER_PAGES`, the buffers placed in it costing what a heap over host memory's
//! do. A block of one run, a 2 MiB superpage, holds it over that run where the guest's RAM is
//! mapped for the host (the memory map's), not over the view, so the platform imports it as it
//! does guest RAM; the RAM stays mapped as long as the device. A block of more than one run, a
//! slab of plain pages the guest took where its kernel had no superpage left (most of a guest's
//! uptime: they run out minutes after its boot), holds it over the block's view, made as a
//! `BUFFER_OVER_PAGES`'s is, on the surface worker where it is not made yet. Either way the heap
//! lies over the block, so the block is not released while it lives (its `BLOCK_RELEASE` is
//! refused), and a context's close lets go of its objects before its pages blocks' views: the view
//! outlives the heap with no hold of its own.
//!
//! **By runs.** A cached block (`RGFX_SURFACE_PAGES_CACHED`) on a device that copies the guest's
//! RAM by its runs ([`GpuDevice::copies_guest_ram`]) takes no view: its `SURFACE_PAGES` checks
//! its runs and finds their pages' host addresses in the guest's RAM, in the surface's order, and
//! its device cache copies by those ([`super::surface_cache`]), so the block costs no mapping and
//! no import. What would make its view is kept with it ([`Unmade`]): the cache makes the view
//! where the platform refuses a copy, or where it makes no cache, and the block then lies over
//! the view as any other. A block whose runs leave the RAM's mapping takes its view as before.
//!
//! [`GpuDevice::copies_guest_ram`]: reims_vgpu_vulkan::core::platform::device::GpuDevice::copies_guest_ram
//!
//! **Device caches where the pages cannot serve** (`RGFX_FEATURE_SURFACE_CACHE_REFUSED`). On a
//! device whose GPU shares the CPU's memory a surface's textures lie in its pages, but not every
//! surface's can: the platform takes its own copy of the pages instead (a shadow) where the GPU
//! cannot place a texture over them as the guest laid them out (rows that end inside the cache
//! line the GPU writes back by, on an Intel GPU; more pieces than a dma-buf takes, on RADV), and
//! that copy goes back and forth at every command buffer. Such a surface takes a device cache
//! instead, which is copied only at the CPU's locks and unlocks, as on a GPU of its own memory. A
//! block in place asks for it at its `SURFACE_PAGES` (`RGFX_SURFACE_PAGES_ASK_CACHE`), with its
//! surface's planes, which must lie in its bytes ([`layout_planes`]). Where the device answers
//! first that textures over the surface would lie in place over no buffer of its pages, whatever
//! it imports them as ([`GpuDevice::surface_pages_never_in_place`]), the block is answered
//! `RGFX_STATUS_CACHE` at once, with no view made ([`SurfaceViews::cache_unmade`]), where its runs
//! pass a view's checks. Otherwise its view is made as any block's, and the device asked whether
//! textures over it lie in place ([`GpuDevice::surface_pages_in_place`]). Where they do, or where
//! the device's verdict is not known yet, the block keeps its view and its buffers lie over its
//! pages, as a block that asks nothing; where they do not, the block is answered
//! `RGFX_STATUS_CACHE` with no view, its view left in the cache of views. Either way the kext then
//! posts the block again marked cached, which takes a device cache ([`super::surface_cache`]),
//! and a view where it is not named by its runs, the one left in the cache where there is one.
//! The counts keep how each ask was answered. **Measured** 2026-11-08 on the Iris Xe host,
//! macOS 26.7.2 guest, Safari scrolling a 96,482 px page down to its end and back at 60 wheel
//! clicks a second (jobs m3-max-8402 and m3-max-8403): each 7264 KiB WebKit tile (1,815 runs)
//! asked and was answered a cache. With a view made for each ask and the cache's first fill
//! committed before the cached naming was answered, the ask took 7.66 ms and the naming 2.12 ms
//! on the kext's wait (210 tiles in 16.7 s); with the answer before any view and the fill after
//! the naming's answer ([`super::surface_cache`]), 0.11 ms and 0.23 ms (294 tiles), the fill
//! 1.43 ms on the surface worker, which 292 messages naming the blocks waited for, 1.2 ms on
//! average and 9.1 ms at the 99th percentile. The surface workers' time in a second of 20 or more
//! namings went from 223 ms to 58 ms, and the page's `requestAnimationFrame` callbacks over the 5 s
//! of the scroll down from 80 to 199.
//!
//! A block that asks is answered `RGFX_STATUS_CACHE` with no view made where the platform has no
//! room for its view's mappings ([`HostVm::view_room`]) once the cached views no block holds went
//! where they cover it ([`PagesRefusal::NoRoom`]): past that room a view is mapped another way,
//! and on a device that imports a view's pages only mapped from their files (Intel's ANV), every
//! buffer over such a view is copied rather than imported, which costs each command buffer a
//! compare of its bytes. **Measured** 2026-11-08 on homelab-ms01-1 (Iris Xe, ANV), an 8 GiB
//! x86-64 macOS 26.7.2 guest at 3648 x 2038 with Safari playing two 4K HEVC clips at once, then
//! three: every view's pages were imported, so the views held all they mapped, about 66,000
//! mappings by the desktop and 273,000 to 282,000 with the clips, at the host's
//! `vm.max_map_count` of 1,048,576, and the clips presented 52 to 56 frames a second. With the
//! library's limit set to a stock host's 65,530 (a diagnostic build), the views stayed within the
//! 40,954 short of the headroom, 116 buffers of 1.7 GiB were copied, and the clips presented 29
//! to 31 frames a second two at once, and 7 to 30 three at once.
//!
//! [`GpuDevice::surface_pages_in_place`]: reims_vgpu_vulkan::core::platform::device::GpuDevice::surface_pages_in_place
//! [`GpuDevice::surface_pages_never_in_place`]: reims_vgpu_vulkan::core::platform::device::GpuDevice::surface_pages_never_in_place
//!
//! **Off ring 0's threads.** A view costs a kernel mapping per run, and the runtime's import of
//! them: real surfaces' runs are single 4 KiB pages, most laid right below the run before (measured
//! on the msi 5080 host, macOS 26 guest, Safari scrolling: every run one page, 74% of them
//! downwards), so a 3840x2400 layer takes 9000 mappings, made in 16 to 478 ms and imported in 18 to
//! 21 ms. Ring 0 waits for none of it: the block's lane hands each `SURFACE_PAGES` to the surface
//! worker, `reims-vgpu.graphics-surfaces-<i>` ([`Worker`]), whose threads make each block's views
//! in the order they are handed over (its ring order), other blocks' beside them, and answer each
//! once its block's view, and device cache, are made. Until then the block is unsettled
//! ([`Unsettled`]): every later message that names it, on ring 0 or a queue's ring, waits for it
//! first, and a command buffer that uses it is not answered at a doorbell, so it sees the block as
//! it would have right after the `SURFACE_PAGES`; a pages block's `SURFACE_PAGES`, which only
//! checks its runs, is carried out
//! there too, in its order with the rest. The guest's kext names a block only after the answer, so
//! in its order nothing waits but for the block's device cache's first fill, made after the
//! answer and before the block settles (`surface_cache`, "Fill"). A pages block's view, which its first `BUFFER_OVER_PAGES` or
//! `HEAP_OVER_PAGES` makes, is made on the worker as well (`GraphicsControl::start_over_pages`):
//! neither message has an answer, so its context's later messages, its block's release and its
//! context's close wait for it instead, and none of its context's command buffers is answered at
//! a doorbell meanwhile. What lets a view go is dropped on the worker too, where
//! it may be the last hold of an import or a mapping: a surface's buffer
//! (`BUFFER_OVER_SURFACE_PAGES`) at its `OBJECT_DESTROY`, and a block's views and device cache at
//! its `BLOCK_RELEASE`. A texture and a pages block's buffer go at once, as the guest may pick
//! their id or their GPU address again in its next message; where one is the last hold of a view,
//! the view's mapping still goes on the worker ([`SurfaceView`]), as does every view's, wherever
//! its last hold goes.
//!
//! [`Worker`]: super::worker::Worker
//! [`Unsettled`]: super::worker::Unsettled

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use reims_vgpu_graphics::wire::{
    RGFX_SURFACE_PAGES_PLANES, RgfxBufferOverPages, RgfxBufferOverSurfacePages, RgfxHeapOverPages,
    RgfxPageRun, RgfxSurfacePages, RgfxSurfacePagesLayout,
};
use reims_vgpu_video::core::device_memory::{DeviceMemory, HostView, MemoryIdentity, MemoryLife};
#[cfg(target_os = "linux")]
use reims_vgpu_video::core::device_memory::{DmaBufError, SurfaceDmaBuf};
use reims_vgpu_video::core::surfaces::SurfaceError;
use reims_vgpu_vulkan::core::host::{HostPtr, PhysRange};
use reims_vgpu_vulkan::core::memory::PAGE_SIZE as MAPPING_PAGE;
use reims_vgpu_vulkan::core::memory::map::MemoryMap;
use reims_vgpu_vulkan::core::memory::mapping::Mapping;
use reims_vgpu_vulkan::core::num::{ToU64 as _, ToUsize as _};
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::os::clock::Clock as _;
use reims_vgpu_vulkan::core::platform::values::{BufferDesc, HeapDesc, OptionSet, PagesPlane};
use reims_vgpu_vulkan::core::platform::{Handle, HostRange, Platform};
use reims_vgpu_vulkan::core::raise::{RaiseResult, refusals};
use reims_vgpu_vulkan::core::sync::Mutex;

use super::worker::{Settle, Worker};

/// The guest's page: every run is whole pages of it.
pub(crate) const GUEST_PAGE: u64 = 4096;

/// How many views the cache keeps, the last made or taken.
pub(crate) const CACHED_VIEWS: usize = 32;

/// The most bytes of guest pages the cache's views map, past which the oldest go.
pub(crate) const CACHED_BYTES: u64 = 1 << 30;

/// The surface worker's key of a view's teardown ([`SurfaceView`]'s drop): no block's offset,
/// so a teardown never waits behind a block's jobs, nor they behind it, and the teardowns run
/// one at a time, on one of the worker's threads at most.
const TEARDOWN: u64 = u64::MAX;

/// One surface's view of its guest pages, which the buffers of its blocks lie over.
///
/// Its last hold may go on any thread: a block's release or the cache's eviction on the surface
/// worker, a context's close on the closes' worker, or the deallocator of a buffer over it,
/// which the platform runs where the buffer's last release is, ring 0's thread among them (a
/// texture's or a pages block's buffer's `OBJECT_DESTROY`, which lets go of its object at once,
/// the module doc). So the view's mapping, whose teardown is one unmap of every run it maps, is
/// let go of on the surface worker ([`TEARDOWN`]) wherever that is, never on the thread that let
/// go of the view last.
///
/// A device of the host's video reaches the view as its memory ([`Self::memory`]): one mapping
/// of the surface's pages, whose identity ends as the view goes, and whose hold keeps the
/// mapping, not the view, so a device's import of it (a CUDA registration, a dma-buf) holds the
/// pages mapped until the device has let go of it, which it does once the identity ends.
pub(crate) struct SurfaceView<P: Platform> {
    /// The view's first byte, and its bytes: its mapping's.
    base: HostPtr,
    length: u64,
    /// The mapping, which the devices' imports of the view hold too.
    held: Arc<MappingHold<P>>,
    /// The view's life, which ends as it goes ([`Self::memory`]).
    life: MemoryLife,
}

/// A view's mapping: the runs mapped in order, and the platform's hold on the guest's RAM, kept
/// until the last hold goes, then let go of on the surface worker ([`TEARDOWN`]), or where the
/// last hold goes once the cache let go of the view to make room ([`Cached::let_go_now`]).
struct MappingHold<P: Platform> {
    /// Taken as it goes.
    held: Option<(Arc<Mapping<P>>, Lease)>,
    /// Where `held` is let go of; `None` where it goes where the hold does (the costs' runs).
    teardown: Option<Arc<Worker>>,
    /// Whether `held` goes where the last hold does, whatever `teardown` says: a view the cache
    /// let go of to make room, whose room a view being made waits for, never on ring 0's
    /// threads, as no ring holds such a view.
    here: AtomicBool,
}

impl<P: Platform> Drop for MappingHold<P> {
    fn drop(&mut self) {
        let Some(held) = self.held.take() else {
            return;
        };
        match self.teardown.take() {
            Some(worker) if !self.here.load(Ordering::Acquire) => {
                worker.drop_later(TEARDOWN, 0, held);
            }
            _ => drop(held),
        }
    }
}

/// A view's memory as the host's video devices take it ([`SurfaceView::memory`]).
struct ViewMemory<P: Platform> {
    identity: MemoryIdentity,
    /// The mapping's first byte, a page's, and its bytes.
    base: HostPtr,
    length: usize,
    held: Arc<MappingHold<P>>,
}

impl<P: Platform> DeviceMemory for ViewMemory<P> {
    fn identity(&self) -> MemoryIdentity {
        self.identity.clone()
    }

    fn length(&self) -> usize {
        self.length
    }

    /// A dma-buf of the view's pages behind `ranges`: the guest's RAM pages the view maps, which
    /// the Vulkan platform's backing table resolves to the files behind them, made by udmabuf
    /// (`platform/vulkan/lent_memory.rs`). The view is mapped while this lives.
    #[cfg(target_os = "linux")]
    fn dma_buf_of(&self, ranges: &[(usize, usize)]) -> Result<SurfaceDmaBuf, DmaBufError> {
        let first = NonNull::new(self.base.as_ptr::<u8>())
            .ok_or_else(|| DmaBufError::Refused("a view at no address".to_owned()))?;
        crate::platform::host_dma_buf(first, self.length, ranges)
    }

    /// The whole mapping, the surface's first byte its first, held mapped by the view's mapping
    /// hold, which outlives the view while the caller keeps it.
    fn host_view(&self) -> Result<HostView, SurfaceError> {
        let first = NonNull::new(self.base.as_ptr::<u8>())
            .ok_or_else(|| SurfaceError::Unreachable("a view at no address".to_owned()))?;
        Ok(HostView {
            first,
            size: self.length,
            offset: 0,
            hold: Box::new(Arc::clone(&self.held)),
        })
    }
}

/// The platform's hold on the guest's RAM the views are made of.
pub(crate) type Lease = Arc<dyn Any + Send + Sync>;

impl<P: Platform> SurfaceView<P> {
    /// The view of `mapping`, which holds `lease` until it goes, let go of on `teardown`.
    pub(crate) fn new(
        mapping: Arc<Mapping<P>>,
        lease: Lease,
        teardown: Option<Arc<Worker>>,
    ) -> Self {
        SurfaceView {
            base: mapping.virtual_address(),
            length: mapping.length(),
            held: Arc::new(MappingHold {
                held: Some((mapping, lease)),
                teardown,
                here: AtomicBool::new(false),
            }),
            life: MemoryLife::new(),
        }
    }

    /// The view as the host's video devices take memory: its identity, which ends as the view
    /// goes, and its whole mapping, held while the caller keeps what it makes of it; `None` for
    /// a view of no bytes.
    pub(crate) fn memory(&self) -> Option<Arc<dyn DeviceMemory>> {
        let length = usize::try_from(self.length).ok().filter(|&n| n > 0)?;
        if self.base.is_null() {
            return None;
        }
        Some(Arc::new(ViewMemory {
            identity: self.life.identity(),
            base: self.base,
            length,
            held: Arc::clone(&self.held),
        }))
    }

    /// The view's bytes: the surface's.
    pub(crate) fn length(&self) -> u64 {
        self.length
    }

    /// The view's first byte.
    pub(crate) fn base(&self) -> HostPtr {
        self.base
    }
}

/// Why a `SURFACE_PAGES` left its block with no view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PagesRefusal {
    /// The device has no memory map: the host was given no guest RAM.
    NoMemoryMap,
    /// The host's or the guest's page is not one the runs can be mapped in.
    PageSize,
    /// No runs, or a run that is empty or not whole pages from a page.
    Unaligned,
    /// The runs' lengths do not sum to the surface's.
    Length,
    /// A run lies outside guest RAM, or the host refused the view (its views' budget).
    NotMapped,
    /// The platform made no buffer over the view.
    NoBuffer,
    /// Not a refusal: the block asked whether it needs a device cache, and the platform has no
    /// room for its view's mappings ([`HostVm::view_room`]), the cached views no block holds let
    /// go of first where they cover it, so it takes a device cache instead (the module doc,
    /// "Device caches where the pages cannot serve").
    ///
    /// [`HostVm::view_room`]: reims_vgpu_vulkan::core::platform::os::vm::HostVm::view_room
    NoRoom,
}

/// What the views did, for the log ([`SurfaceViews::counts`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ViewCounts {
    /// Views made.
    pub(crate) made: u64,
    /// Blocks given a cached view.
    pub(crate) reused: u64,
    /// `SURFACE_PAGES` refused for runs that are not whole pages, or do not sum to the length.
    pub(crate) bad_runs: u64,
    /// `SURFACE_PAGES` refused for runs outside guest RAM, or a view or buffer the host did not
    /// make.
    pub(crate) not_mapped: u64,
    /// Pages blocks' views made (`RGFX_SURFACE_PAGES_PRIVATE`), each at its block's first
    /// `BUFFER_OVER_PAGES`, which `made` does not count.
    pub(crate) private: u64,
    /// Cached views no block held let go of as their mappings passed the platform's bound for
    /// them (the module doc).
    pub(crate) idle_returned: u64,
    /// Cached blocks named by their runs, with no view (the module doc).
    pub(crate) by_runs: u64,
    /// Blocks that asked whether they need a device cache and stay in place (the module doc,
    /// "Device caches where the pages cannot serve"), the device's verdict known, which
    /// `asked_pending` does not count.
    pub(crate) asked_in_place: u64,
    /// Such blocks kept in place while the device's verdict for their pages was pending.
    pub(crate) asked_pending: u64,
    /// Such blocks answered `RGFX_STATUS_CACHE`.
    pub(crate) asked_cache: u64,
    /// Of `asked_cache`, those answered so with no view made, the platform having no room for
    /// its mappings ([`PagesRefusal::NoRoom`]).
    pub(crate) asked_room: u64,
}

/// How a block that asked whether it needs a device cache was answered, for the counts
/// ([`SurfaceViews::asked`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Asked {
    /// In place: the device's textures lie in place over its pages.
    InPlace,
    /// In place, the device's verdict for its pages pending.
    Pending,
    /// A device cache.
    Cache,
    /// A device cache, with no view made, the platform having no room for its mappings
    /// ([`PagesRefusal::NoRoom`]).
    Room,
}

/// The planes of the surface `layout` gives ([`RgfxSurfacePagesLayout`]), where it has 1 to
/// [`RGFX_SURFACE_PAGES_PLANES`] planes, each of at least one row of texels, its rows apart by
/// at least a row's texels where it has more than one, and its texels in the surface's `length`
/// bytes; `None` otherwise.
pub(crate) fn layout_planes(
    layout: &RgfxSurfacePagesLayout,
    length: u64,
) -> Option<Vec<PagesPlane>> {
    let count = usize::try_from(layout.plane_count).ok()?;
    if !(1..=RGFX_SURFACE_PAGES_PLANES as usize).contains(&count) {
        return None;
    }
    let planes: Vec<PagesPlane> = (layout.planes[..count].iter())
        .map(|p| PagesPlane {
            offset: p.offset,
            bytes_per_row: p.bytes_per_row,
            row_bytes: p.row_bytes,
            rows: p.rows,
        })
        .collect();
    let lies = |p: &PagesPlane| {
        p.rows != 0
            && p.row_bytes != 0
            && (p.rows == 1 || p.row_bytes <= p.bytes_per_row)
            && p.end().is_some_and(|end| end <= length)
    };
    planes.iter().all(lies).then_some(planes)
}

/// How a surface's runs lie in guest RAM, for the log: each run takes a kernel mapping of its own
/// in the view, so their count is the view's cost.
struct Layout {
    runs: usize,
    /// Runs of one host page.
    single: usize,
    /// Runs that end where the run before them starts: the guest's pages laid out downwards.
    downwards: usize,
    /// The longest run's bytes.
    longest: u64,
}

impl Layout {
    fn of(runs: &[RgfxPageRun], page: u64) -> Layout {
        Layout {
            runs: runs.len(),
            single: runs.iter().filter(|r| r.length == page).count(),
            downwards: (runs.windows(2))
                .filter(|w| w[1].address.checked_add(w[1].length) == Some(w[0].address))
                .count(),
            longest: runs.iter().map(|r| r.length).max().unwrap_or(0),
        }
    }
}

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} runs ({} of one page, {} right below the run before, the longest {:#x} bytes)",
            self.runs, self.single, self.downwards, self.longest
        )
    }
}

/// A view the cache keeps, with the buffer that holds the runtime's import of its pages.
struct Cached<P: Platform> {
    /// The runs the view maps, in order.
    runs: Vec<(u64, u64)>,
    view: Arc<SurfaceView<P>>,
    import: P::Buffer,
}

impl<P: Platform> Cached<P> {
    /// Whether the cache alone holds the view: its own hold and its import's, whose deallocator
    /// holds the view (`buffer_over`). Under the views' lock, which every new hold of a cached
    /// view takes, such a view stays so until the lock is let go.
    fn idle(&self) -> bool {
        Arc::strong_count(&self.view) <= 2
    }

    /// Lets go of the view and its import, to make room ([`SurfaceViews::make_room_for`]), with
    /// the view's mapping torn down where its last hold goes rather than on the surface worker:
    /// here, where the platform frees the import at once and no device's import of the view's
    /// memory holds it; else on the runtime's thread that frees the import once the command
    /// buffers that use it completed, or on the thread that drops a device's import as the
    /// view's life ends (a CUDA registration's reaper). The view being made waits for the room
    /// on a surface worker's thread, which none of those needs.
    fn let_go_now(self) {
        let Cached { view, import, .. } = self;
        view.held.here.store(true, Ordering::Release);
        drop((import, view));
    }
}

/// How many steps a view being made waits at most for the room the cached views let go of for it
/// give back ([`SurfaceViews::make_room_for`]): 20 ms in all, four times the longest such wait
/// that gave the room back (the module doc).
const ROOM_WAIT_STEPS: u32 = 100;

/// One step of that wait.
const ROOM_WAIT_STEP: std::time::Duration = std::time::Duration::from_micros(200);

/// A pages block (the module doc): its runs, and the view its `BUFFER_OVER_PAGES` buffers lie
/// over.
pub(crate) struct Private<P: Platform> {
    /// The runs, checked, in order.
    runs: Vec<(u64, u64)>,
    /// The bytes they hold, the view's length.
    length: u64,
    /// The block's size, as its `SURFACE_PAGES` gave it.
    block_size: u64,
    /// The context the block was made for, the one that may name it.
    context: u32,
    /// The view, made at the block's first `BUFFER_OVER_PAGES`, with a buffer over the whole of
    /// it that holds the runtime's import of its pages for the buffers over its parts.
    view: Option<(Arc<SurfaceView<P>>, P::Buffer)>,
    /// Where the guest's RAM mapping holds the block's pages, where they are one run: what a
    /// `HEAP_OVER_PAGES` lies over, which then makes no view (a block of more runs holds its heap
    /// over the view).
    ram: Option<(HostPtr, u64)>,
}

/// What makes the view of a block named by its runs (the module doc): its runs, checked, the
/// guest's RAM they are mapped through, and the platform's hold on it.
pub(crate) struct Unmade<P: Platform> {
    memory: Arc<MemoryMap<P>>,
    lease: Lease,
    /// Where the view's mapping is let go of ([`SurfaceView`]).
    teardown: Option<Arc<Worker>>,
    key: Vec<(u64, u64)>,
    length: u64,
}

impl<P: Platform> Unmade<P> {
    /// The surface's bytes.
    pub(crate) fn length(&self) -> u64 {
        self.length
    }

    /// The view, mapped now, and a Shared buffer over the whole of it on `device`, which holds
    /// the view and the runtime's import of its pages; the refusal where either is not made.
    pub(crate) fn make(&self, device: &P::Device) -> RaiseResult<Result<Made<P>, PagesRefusal>> {
        let view = match map_view(
            &self.memory,
            Arc::clone(&self.lease),
            self.teardown.clone(),
            &self.key,
            self.length,
        ) {
            Ok(view) => view,
            Err(why) => return Ok(Err(why)),
        };
        Ok(
            match buffer_over(device, &view, 0, view.length(), 0, None)? {
                Some(import) => Ok((view, import)),
                None => Err(PagesRefusal::NoBuffer),
            },
        )
    }
}

/// A block named by its runs ([`SurfaceViews::by_runs`]): its pages' host addresses in the
/// guest's RAM, in the surface's order, as runs, and what makes its view.
pub(crate) type NamedByRuns<P> = (Vec<Range<u64>>, Arc<Unmade<P>>);

/// A block named by its runs, with no view (the module doc).
struct ByRuns<P: Platform> {
    /// The pages' host addresses in the guest's RAM, in the surface's order, as runs.
    runs: Vec<Range<u64>>,
    unmade: Arc<Unmade<P>>,
    /// The size of the block's part its holders map.
    size: u64,
}

/// The views of the surfaces whose blocks name their pages, by block offset, and the cache.
pub(crate) struct SurfaceViews<P: Platform> {
    /// The guest's RAM, through which the views are made, and the platform's hold on it; `None`
    /// makes none.
    memory: Option<(Arc<MemoryMap<P>>, Lease)>,
    /// Where each view's mapping is let go of ([`SurfaceView`]): the surface worker.
    teardown: Option<Arc<Worker>>,
    /// Each block's view, and the size of the block's part its holders map.
    by_block: HashMap<u64, (Arc<SurfaceView<P>>, u64)>,
    /// Each block named by its runs, with no view.
    by_runs: HashMap<u64, ByRuns<P>>,
    /// Each pages block's view, by block offset.
    private: HashMap<u64, Private<P>>,
    /// The layout of each block answered that it needs a device cache, by block offset, until
    /// the block's cached `SURFACE_PAGES` takes it for its cache ([`Self::take_layout`]).
    laid_out: HashMap<u64, Vec<PagesPlane>>,
    /// The views made or taken last, the latest at the back.
    cache: VecDeque<Cached<P>>,
    counts: ViewCounts,
    /// The device's resets so far: a `SURFACE_PAGES` begun before the last makes no view
    /// ([`Settle`]).
    resets: u64,
}

/// What a `SURFACE_PAGES` found under the views' lock ([`SurfaceViews::surface_pages`]).
enum Found<P: Platform> {
    /// The block's view, the cache's, or the refusal.
    Done(Result<Arc<SurfaceView<P>>, PagesRefusal>),
    /// A view to make of the runs `key` outside the lock, through `memory`.
    Make {
        memory: Arc<MemoryMap<P>>,
        lease: Lease,
        teardown: Option<Arc<Worker>>,
        key: Vec<(u64, u64)>,
    },
}

/// What a block's `BLOCK_RELEASE` takes out of the views ([`SurfaceViews::released`]): its hold
/// of its view, or its pages block, for the caller to let go of with no lock held.
pub(crate) struct Released<P: Platform> {
    _view: Option<Arc<SurfaceView<P>>>,
    _private: Option<Private<P>>,
    _by_runs: Option<ByRuns<P>>,
}

impl<P: Platform> SurfaceViews<P> {
    /// No views, made through `memory`, each let go of on `teardown` ([`SurfaceView`]).
    pub(crate) fn new(
        memory: Option<(Arc<MemoryMap<P>>, Lease)>,
        teardown: Option<Arc<Worker>>,
    ) -> Self {
        SurfaceViews {
            memory,
            teardown,
            by_block: HashMap::new(),
            by_runs: HashMap::new(),
            laid_out: HashMap::new(),
            private: HashMap::new(),
            cache: VecDeque::new(),
            counts: ViewCounts::default(),
            resets: 0,
        }
    }

    /// Whether views can be made: a memory map, and a host page the guest's runs are whole
    /// pages of ([`GUEST_PAGE`]); the device's [`RGFX_FEATURE_SURFACE_PAGES`] says so.
    ///
    /// [`RGFX_FEATURE_SURFACE_PAGES`]: reims_vgpu_graphics::wire::RGFX_FEATURE_SURFACE_PAGES
    pub(crate) fn available(&self, platform: &P) -> bool {
        self.memory.is_some()
            && platform.vm().page_size() == GUEST_PAGE
            && MAPPING_PAGE == GUEST_PAGE
    }

    /// What the views did so far.
    pub(crate) fn counts(&self) -> ViewCounts {
        self.counts
    }

    /// `SURFACE_PAGES`: the view of `runs` for the block `pages` names, in place of any it had:
    /// the cache's over the same runs, else one made, with a buffer over it on `device`; or, where
    /// `by_runs`, no view, the block named by its runs (the module doc), where every page lies in
    /// the guest RAM's mapping. The
    /// mapping and the import are made with `views` unlocked, so a message that asks of the
    /// views meanwhile does not wait for them; the views and the cache are changed only here,
    /// by one `SURFACE_PAGES` of a block at a time (the surface [`Worker`]), and at a release,
    /// which names another block, and a reset, which waits for it. Two blocks' at once change
    /// only their own entries, and two views made at once over the same runs are both cached.
    /// What leaves the views goes once the lock is let go, on this thread. A block that `ask`s
    /// whether it needs a device cache, whose view the platform has no room for, gets none
    /// ([`PagesRefusal::NoRoom`]). `None` where the device was reset since `settle` began it.
    ///
    /// [`Worker`]: super::worker::Worker
    #[allow(clippy::too_many_arguments)] // the block, its runs, and how its message asks
    pub(crate) fn surface_pages(
        views: &Mutex<SurfaceViews<P>>,
        platform: &P,
        device: &P::Device,
        pages: &RgfxSurfacePages,
        runs: &[RgfxPageRun],
        settle: &Settle,
        by_runs: bool,
        ask: bool,
    ) -> RaiseResult<Option<Result<(), PagesRefusal>>> {
        let (old, found) = {
            let mut v = views.lock();
            if v.resets != settle.resets {
                return Ok(None);
            }
            let old = (
                v.by_block.remove(&pages.block_offset),
                v.private.remove(&pages.block_offset),
                v.by_runs.remove(&pages.block_offset),
            );
            if by_runs && v.name_by_runs(platform, pages, runs) {
                drop(v);
                drop(old);
                return Ok(Some(Ok(())));
            }
            (old, v.find(platform, pages, runs))
        };
        drop(old);
        let made = match found {
            Found::Done(view) => view,
            Found::Make {
                memory,
                lease,
                teardown,
                key,
            } => {
                Self::make_room_for(views, platform, key.len());
                if ask && !Self::room_for(platform, key.len()) {
                    return Ok(Some(Err(PagesRefusal::NoRoom)));
                }
                match make_view(
                    platform, device, pages, runs, &memory, lease, teardown, &key,
                )? {
                    Ok((view, import)) => {
                        let idle_mappings = platform.vm().idle_view_mappings();
                        let evicted = views.lock().keep(key, &view, import, idle_mappings);
                        drop(evicted);
                        Ok(view)
                    }
                    Err(why) => Err(why),
                }
            }
        };
        let mut v = views.lock();
        Ok(Some(match made {
            Ok(view) => {
                v.by_block
                    .insert(pages.block_offset, (view, pages.block_size));
                Ok(())
            }
            Err(why) => Err(v.refused(why)),
        }))
    }

    /// An ask whether the block `pages` names needs a device cache
    /// (`RGFX_SURFACE_PAGES_ASK_CACHE`), answered that it does before any view of `runs` is made,
    /// the device's textures lying in place over no buffer of its pages
    /// ([`GpuDevice::surface_pages_never_in_place`]), so a view made for the ask would be left at
    /// once: where the runs pass a view's checks and lie in the guest RAM's mapping, any view or
    /// naming the block had goes, once the lock is let go, as at such an answer over a view, the
    /// ask is counted, and its layout `planes` kept for the block's cached `SURFACE_PAGES`
    /// ([`Self::take_layout`]): `Some(true)`. `Some(false)`, with nothing changed, where the runs
    /// fail them, for the view's path to refuse them as before; `None` where the device was reset
    /// since `settle` began it.
    ///
    /// [`GpuDevice::surface_pages_never_in_place`]: reims_vgpu_vulkan::core::platform::device::GpuDevice::surface_pages_never_in_place
    pub(crate) fn cache_unmade(
        views: &Mutex<SurfaceViews<P>>,
        platform: &P,
        pages: &RgfxSurfacePages,
        runs: &[RgfxPageRun],
        planes: &[PagesPlane],
        settle: &Settle,
    ) -> Option<bool> {
        let old = {
            let mut v = views.lock();
            if v.resets != settle.resets {
                return None;
            }
            let Ok(key) = v.checked_runs(platform, pages, runs) else {
                return Some(false);
            };
            let in_ram =
                (v.memory.as_ref()).is_some_and(|(memory, _)| in_order(memory, &key).is_some());
            if !in_ram {
                return Some(false);
            }
            let old = v.released(pages.block_offset);
            v.asked(Asked::Cache);
            v.laid_out(pages.block_offset, planes.to_vec());
            old
        };
        drop(old);
        Some(true)
    }

    /// `SURFACE_PAGES` of a deferred block (`RGFX_SURFACE_PAGES_DEFERRED`), which has no pages
    /// yet, and so no view: any view the block had goes, once the lock is let go, on this thread.
    /// False where the device was reset since `settle` began it.
    pub(crate) fn deferred(views: &Mutex<SurfaceViews<P>>, block: u64, settle: &Settle) -> bool {
        let old = {
            let mut v = views.lock();
            if v.resets != settle.resets {
                return false;
            }
            (
                v.by_block.remove(&block),
                v.private.remove(&block),
                v.by_runs.remove(&block),
            )
        };
        drop(old);
        true
    }

    /// `SURFACE_PAGES` of a pages block (`RGFX_SURFACE_PAGES_PRIVATE`, the module doc): `runs`,
    /// checked and in guest RAM, for the block `pages` names, in place of any view it had; its
    /// view is made at its first `BUFFER_OVER_PAGES` ([`Self::private_view`]). What the block
    /// had goes once the lock is let go, on this thread. `None` where the device was reset since
    /// `settle` began it.
    pub(crate) fn private_pages(
        views: &Mutex<SurfaceViews<P>>,
        platform: &P,
        pages: &RgfxSurfacePages,
        runs: &[RgfxPageRun],
        settle: &Settle,
    ) -> Option<Result<(), PagesRefusal>> {
        let mut v = views.lock();
        if v.resets != settle.resets {
            return None;
        }
        let old = (
            v.by_block.remove(&pages.block_offset),
            v.private.remove(&pages.block_offset),
        );
        let checked = match v.checked_runs(platform, pages, runs) {
            Ok(key) if !v.in_ram(&key) => Err(PagesRefusal::NotMapped),
            checked => checked,
        };
        let made = match checked {
            Ok(runs) => {
                let ram = match (runs.as_slice(), &v.memory) {
                    (&[(address, length)], Some((memory, _))) => {
                        let host = memory.virtual_address_for_physical(address, length);
                        (!host.is_null()).then_some((host, length))
                    }
                    _ => None,
                };
                v.private.insert(
                    pages.block_offset,
                    Private {
                        runs,
                        length: pages.length,
                        block_size: pages.block_size,
                        context: pages.context,
                        view: None,
                        ram,
                    },
                );
                Ok(())
            }
            Err(why) => Err(v.refused(why)),
        };
        drop(v);
        drop(old);
        Some(made)
    }

    /// Whether every byte of the runs `key` lies in the guest RAM the memory map holds, each run
    /// split across the map's ranges in their order, as [`MemoryMap::new_range_mapping`] maps it.
    fn in_ram(&self, key: &[(u64, u64)]) -> bool {
        let Some((memory, _)) = &self.memory else {
            return false;
        };
        key.iter().all(|&(address, length)| {
            let Some(end) = address.checked_add(length) else {
                return false;
            };
            let mut at = address;
            for r in memory.ranges() {
                if at >= r.physical_address && at < r.physical_end {
                    at = r.physical_end.min(end);
                    if at == end {
                        return true;
                    }
                }
            }
            false
        })
    }

    /// `why`, counted.
    fn refused(&mut self, why: PagesRefusal) -> PagesRefusal {
        if matches!(why, PagesRefusal::Unaligned | PagesRefusal::Length) {
            self.counts.bad_runs += 1;
        } else {
            self.counts.not_mapped += 1;
        }
        why
    }

    /// Names the block `pages` names by `runs` (the module doc), with no view: false, naming
    /// nothing, where the runs are refused or a page is not in the guest RAM's mapping, for the
    /// caller to make its view (whose refusal is counted there).
    fn name_by_runs(
        &mut self,
        platform: &P,
        pages: &RgfxSurfacePages,
        runs: &[RgfxPageRun],
    ) -> bool {
        let Ok(key) = self.checked_runs(platform, pages, runs) else {
            return false;
        };
        let Some((memory, lease)) = self.memory.as_ref() else {
            return false;
        };
        let Some(host) = in_order(memory, &key) else {
            return false;
        };
        let unmade = Unmade {
            memory: Arc::clone(memory),
            lease: Arc::clone(lease),
            teardown: self.teardown.clone(),
            key,
            length: pages.length,
        };
        let block = ByRuns {
            runs: host,
            unmade: Arc::new(unmade),
            size: pages.block_size,
        };
        self.by_runs.insert(pages.block_offset, block);
        self.counts.by_runs += 1;
        true
    }

    /// The runs of `pages`, checked (the module doc), as (address, length) pairs.
    fn checked_runs(
        &self,
        platform: &P,
        pages: &RgfxSurfacePages,
        runs: &[RgfxPageRun],
    ) -> Result<Vec<(u64, u64)>, PagesRefusal> {
        if self.memory.is_none() {
            return Err(PagesRefusal::NoMemoryMap);
        }
        // Whole pages of the host's, of the guest's and of the length core's mappings take.
        let page = platform.vm().page_size().max(MAPPING_PAGE).max(GUEST_PAGE);
        if !page.is_power_of_two() {
            return Err(PagesRefusal::PageSize);
        }
        if runs.is_empty()
            || runs.iter().any(|r| {
                r.length == 0 || !r.address.is_multiple_of(page) || !r.length.is_multiple_of(page)
            })
        {
            return Err(PagesRefusal::Unaligned);
        }
        let total = runs
            .iter()
            .try_fold(0u64, |sum, r| sum.checked_add(r.length));
        if total != Some(pages.length) {
            return Err(PagesRefusal::Length);
        }
        Ok(runs.iter().map(|r| (r.address, r.length)).collect())
    }

    /// The cache's view of `runs`, taken to the back, the refusal of runs that cannot be one,
    /// or what the view to make takes.
    fn find(&mut self, platform: &P, pages: &RgfxSurfacePages, runs: &[RgfxPageRun]) -> Found<P> {
        let key = match self.checked_runs(platform, pages, runs) {
            Ok(key) => key,
            Err(why) => return Found::Done(Err(why)),
        };
        let hit = self.cache.iter().position(|c| c.runs == key);
        if let Some(cached) = hit.and_then(|at| self.cache.remove(at)) {
            let view = Arc::clone(&cached.view);
            self.cache.push_back(cached);
            self.counts.reused += 1;
            return Found::Done(Ok(view));
        }
        let Some((memory, lease)) = self.memory.as_ref() else {
            return Found::Done(Err(PagesRefusal::NoMemoryMap));
        };
        Found::Make {
            memory: Arc::clone(memory),
            lease: Arc::clone(lease),
            teardown: self.teardown.clone(),
            key,
        }
    }

    /// Caches `view`, just made of the runs `key` with `import` over it, at the back; what
    /// leaves the cache past [`CACHED_VIEWS`] views or [`CACHED_BYTES`], the oldest first, and
    /// the oldest views no block holds past `idle_mappings` mappings, a mapping a run (the module
    /// doc), is returned, for the caller to let go of with no lock held.
    fn keep(
        &mut self,
        key: Vec<(u64, u64)>,
        view: &Arc<SurfaceView<P>>,
        import: P::Buffer,
        idle_mappings: Option<u64>,
    ) -> Vec<Cached<P>> {
        self.counts.made += 1;
        self.cache.push_back(Cached {
            runs: key,
            view: Arc::clone(view),
            import,
        });
        let mut evicted = Vec::new();
        let mut bytes: u64 = self.cache.iter().map(|c| c.view.length()).sum();
        while self.cache.len() > CACHED_VIEWS || (bytes > CACHED_BYTES && self.cache.len() > 1) {
            if let Some(old) = self.cache.pop_front() {
                bytes -= old.view.length();
                evicted.push(old);
            }
        }
        if let Some(bound) = idle_mappings {
            let mut mappings: u64 = (self.cache.iter())
                .filter(|c| c.idle())
                .map(|c| c.runs.len().to_u64())
                .sum();
            while mappings > bound {
                let Some(at) = self.cache.iter().position(Cached::idle) else {
                    break;
                };
                let Some(old) = self.cache.remove(at) else {
                    break;
                };
                mappings -= old.runs.len().to_u64();
                self.counts.idle_returned += 1;
                evicted.push(old);
            }
        }
        evicted
    }

    /// Room in `platform`'s mappings for a view of `runs` runs, made before the view is (the
    /// module doc): where the room the platform answers is less than the most the view adds, a
    /// mapping a run and one at each end, cached views no block holds let go of on this thread, a
    /// surface worker's, until the mappings they hold ([`HostVm::view_mappings`]) cover what it
    /// lacks; none where all of them would not.
    ///
    /// [`HostVm::view_mappings`]: reims_vgpu_vulkan::core::platform::os::vm::HostVm::view_mappings
    fn make_room_for(views: &Mutex<SurfaceViews<P>>, platform: &P, runs: usize) {
        let needed = runs.to_u64().saturating_add(2);
        let vm = platform.vm();
        let Some(room) = vm.view_room().filter(|&room| room < needed) else {
            return;
        };
        let held = |view: &SurfaceView<P>| vm.view_mappings(view.base, view.length).unwrap_or(0);
        let evicted = views.lock().make_room(needed - room, held);
        if evicted.is_empty() {
            return;
        }
        for cached in evicted {
            cached.let_go_now();
        }
        // Their mappings go here, or a little later on the threads that hold them last.
        for _ in 0..ROOM_WAIT_STEPS {
            if vm.view_room().is_none_or(|room| room >= needed) {
                break;
            }
            std::thread::sleep(ROOM_WAIT_STEP);
        }
    }

    /// Whether `platform` has room for the mappings of a view of `runs` runs, a mapping a run and
    /// one at each end, as [`Self::make_room_for`] weighs them; always where it counts none.
    fn room_for(platform: &P, runs: usize) -> bool {
        let needed = runs.to_u64().saturating_add(2);
        platform.vm().view_room().is_none_or(|room| room >= needed)
    }

    /// Views no block holds, out of the cache, until the mappings each holds (`held`) add up to
    /// `short`: first those whose mapping nothing but the view holds, which goes at once, then
    /// those a host's video device holds too ([`SurfaceView::memory`]), which goes once the device
    /// lets go of it, the oldest first among each; none where all of them hold fewer, as the view
    /// is then mapped another way whatever goes (the module doc). Counted as let go of for their
    /// mappings; for the caller to let go of with no lock held ([`Cached::let_go_now`]).
    fn make_room(&mut self, short: u64, held: impl Fn(&SurfaceView<P>) -> u64) -> Vec<Cached<P>> {
        let mut idle: Vec<(usize, u64, bool)> = (self.cache.iter().enumerate())
            .filter(|(_, c)| c.idle())
            .map(|(at, c)| {
                (
                    at,
                    held(c.view.as_ref()),
                    Arc::strong_count(&c.view.held) > 1,
                )
            })
            .collect();
        // Stable: the oldest first among each.
        idle.sort_by_key(|&(_, _, shared)| shared);
        let mut chosen = Vec::new();
        let mut freed: u64 = 0;
        for (at, mappings, _) in idle {
            if freed >= short {
                break;
            }
            freed = freed.saturating_add(mappings);
            chosen.push(at);
        }
        if freed < short {
            return Vec::new();
        }
        // From the back, so the places of those still to go stay as they were.
        chosen.sort_unstable_by_key(|&at| std::cmp::Reverse(at));
        let evicted: Vec<Cached<P>> = (chosen.into_iter())
            .filter_map(|at| self.cache.remove(at))
            .collect();
        self.counts.idle_returned += evicted.len().to_u64();
        evicted
    }

    /// The buffer over the whole of the view of the block at `block` that holds its import, one
    /// more reference to it, where the block has a view the cache of views keeps.
    pub(crate) fn import_of(&self, block: u64) -> Option<P::Buffer> {
        let (view, _) = self.by_block.get(&block)?;
        let cached = self.cache.iter().find(|c| Arc::ptr_eq(&c.view, view))?;
        Some(cached.import.clone_ref())
    }

    /// A block that asked whether it needs a device cache was answered `how`, counted.
    pub(crate) fn asked(&mut self, how: Asked) {
        match how {
            Asked::InPlace => self.counts.asked_in_place += 1,
            Asked::Pending => self.counts.asked_pending += 1,
            Asked::Cache => self.counts.asked_cache += 1,
            Asked::Room => {
                self.counts.asked_cache += 1;
                self.counts.asked_room += 1;
            }
        }
    }

    /// The block at `block` was answered that it needs a device cache, its surface laid out as
    /// `planes`, which its cached `SURFACE_PAGES` takes for the cache ([`Self::take_layout`]).
    pub(crate) fn laid_out(&mut self, block: u64, planes: Vec<PagesPlane>) {
        self.laid_out.insert(block, planes);
    }

    /// The layout the block at `block` was answered a device cache with ([`Self::laid_out`]),
    /// taken; `None` for a block no such answer laid out.
    pub(crate) fn take_layout(&mut self, block: u64) -> Option<Vec<PagesPlane>> {
        self.laid_out.remove(&block)
    }

    /// The device's resets so far ([`Settle`]).
    pub(crate) fn resets(&self) -> u64 {
        self.resets
    }

    /// Whether the block at `m.block_offset` is named, by a view or by its runs, where its
    /// holders' part holds the `m.block_size` bytes `m` names and its pages at least `m.length`
    /// bytes.
    pub(crate) fn names(&self, m: &RgfxBufferOverSurfacePages) -> bool {
        let fits =
            |size: u64, length: u64| m.block_size <= size && m.length != 0 && m.length <= length;
        self.view_for(m).is_some()
            || (self.by_runs.get(&m.block_offset)).is_some_and(|b| fits(b.size, b.unmade.length()))
    }

    /// The runs and the view's maker of the block at `block`, where it is named by its runs.
    pub(crate) fn by_runs(&self, block: u64) -> Option<NamedByRuns<P>> {
        (self.by_runs.get(&block)).map(|b| (b.runs.clone(), Arc::clone(&b.unmade)))
    }

    /// The block at `block`, named by its runs, lies over `view` from now on: its cache made no
    /// copy by runs, or none was made.
    pub(crate) fn viewed(&mut self, block: u64, view: Arc<SurfaceView<P>>) {
        if let Some(b) = self.by_runs.remove(&block) {
            self.by_block.insert(block, (view, b.size));
        }
    }

    /// The view of the block at `m.block_offset`, where its holders' part holds the
    /// `m.block_size` bytes `m` names and the view at least `m.length` bytes.
    pub(crate) fn view_for(&self, m: &RgfxBufferOverSurfacePages) -> Option<Arc<SurfaceView<P>>> {
        self.by_block
            .get(&m.block_offset)
            .filter(|(v, size)| m.block_size <= *size && m.length != 0 && m.length <= v.length())
            .map(|(v, _)| Arc::clone(v))
    }

    /// The view of the block at `block`.
    pub(crate) fn view(&self, block: u64) -> Option<Arc<SurfaceView<P>>> {
        self.by_block.get(&block).map(|(v, _)| Arc::clone(v))
    }

    /// The view of the pages block at `block`, where it was made for `context` (any context
    /// where `None`, the decoder's own tests) and holds the `block_size` bytes a message names:
    /// made with its import on `device` the first time, the module doc, with `views` unlocked,
    /// so a message that asks of the views meanwhile does not wait for the mapping or the
    /// import, after room is made in `platform`'s mappings for it. `None` where the block is no
    /// such pages block, or the view or its import is not made, which is counted, or the block
    /// went while it was made.
    pub(crate) fn private_view(
        views: &Mutex<SurfaceViews<P>>,
        platform: &P,
        device: &P::Device,
        block: u64,
        block_size: u64,
        context: Option<u32>,
    ) -> RaiseResult<Option<Arc<SurfaceView<P>>>> {
        let (memory, lease, teardown, runs, length) = {
            let mut v = views.lock();
            let Some(p) = (v.private.get(&block))
                .filter(|p| block_size <= p.block_size && context.is_none_or(|c| c == p.context))
            else {
                return Ok(None);
            };
            if let Some((view, _)) = &p.view {
                return Ok(Some(Arc::clone(view)));
            }
            let (runs, length) = (p.runs.clone(), p.length);
            let Some((memory, lease)) = v.memory.as_ref() else {
                v.refused(PagesRefusal::NoMemoryMap);
                return Ok(None);
            };
            let teardown = v.teardown.clone();
            (
                Arc::clone(memory),
                Arc::clone(lease),
                teardown,
                runs,
                length,
            )
        };
        Self::make_room_for(views, platform, runs.len());
        let made = match map_view(&memory, lease, teardown, &runs, length) {
            Ok(view) => match buffer_over(device, &view, 0, view.length(), 0, None)? {
                Some(import) => Ok((view, import)),
                None => Err(PagesRefusal::NoBuffer),
            },
            Err(why) => Err(why),
        };
        let mut v = views.lock();
        let (view, import) = match made {
            Ok(made) => made,
            Err(why) => {
                v.refused(why);
                return Ok(None);
            }
        };
        // The block's entry as it was taken, else it went meanwhile (a release, its context's
        // close): the view goes with no lock held.
        let Some(p) = (v.private.get_mut(&block)).filter(|p| p.runs == runs && p.view.is_none())
        else {
            drop(v);
            drop((view, import));
            return Ok(None);
        };
        p.view = Some((Arc::clone(&view), import));
        v.counts.private += 1;
        Ok(Some(view))
    }

    /// Where the guest's RAM mapping holds the pages of the pages block at `block`, where they are
    /// one run and the block was made for `context` (any where `None`) and holds the `block_size`
    /// bytes a message names: its first byte and length.
    pub(crate) fn private_ram(
        &self,
        block: u64,
        block_size: u64,
        context: Option<u32>,
    ) -> Option<(HostPtr, u64)> {
        self.private
            .get(&block)
            .filter(|p| block_size <= p.block_size && context.is_none_or(|c| c == p.context))
            .and_then(|p| p.ram)
    }

    /// Whether the block at `block` is a pages block whose view is not made yet: its first
    /// `BUFFER_OVER_PAGES` makes it ([`Self::private_view`]).
    pub(crate) fn private_unmade(&self, block: u64) -> bool {
        self.private.get(&block).is_some_and(|p| p.view.is_none())
    }

    /// The host addresses of the guest pages of `runs` in the mapping of the guest's RAM the
    /// memory map names, as runs ([`crate::device_writes::sorted`]): what the host's devices'
    /// writes of them name. A page the map does not hold is left out.
    pub(crate) fn host_runs(&self, runs: &[RgfxPageRun]) -> Vec<Range<u64>> {
        let Some((memory, _)) = &self.memory else {
            return Vec::new();
        };
        let mut pages = Vec::new();
        for r in runs {
            for at in (r.address..r.address.saturating_add(r.length)).step_by(GUEST_PAGE.to_usize())
            {
                let host = memory.virtual_address_for_physical(at, GUEST_PAGE);
                if !host.is_null() {
                    pages.push(host.addr().to_u64());
                }
            }
        }
        crate::device_writes::page_runs(&pages, GUEST_PAGE)
    }

    /// Each guest page of `runs`, in their order, as its host address in the guest's RAM mapping
    /// and its bytes (a run's last may be short): what a cache the host's CPU copies reads and
    /// writes ([`super::surface_cache`]'s copied pages). `None` where the host has no mapping of
    /// the guest's RAM, or a page of them lies outside it.
    pub(crate) fn host_pages_in_order(&self, runs: &[RgfxPageRun]) -> Option<Vec<(HostPtr, u64)>> {
        let (memory, _) = self.memory.as_ref()?;
        let mut pages = Vec::new();
        for r in runs {
            let end = r.address.checked_add(r.length)?;
            let mut at = r.address;
            while at < end {
                let bytes = (GUEST_PAGE - at % GUEST_PAGE).min(end - at);
                let host = memory.virtual_address_for_physical(at, bytes);
                if host.is_null() {
                    return None;
                }
                pages.push((host, bytes));
                at += bytes;
            }
        }
        Some(pages)
    }

    /// The block at `offset` went (its `BLOCK_RELEASE`): its view goes once the buffers over it
    /// and the cache have let it go, and what the block held of it is returned, for the caller to
    /// let go of with no lock held.
    pub(crate) fn released(&mut self, offset: u64) -> Option<Released<P>> {
        let view = self.by_block.remove(&offset).map(|(view, _)| view);
        let private = self.private.remove(&offset);
        let by_runs = self.by_runs.remove(&offset);
        self.laid_out.remove(&offset);
        (view.is_some() || private.is_some() || by_runs.is_some()).then_some(Released {
            _view: view,
            _private: private,
            _by_runs: by_runs,
        })
    }

    /// Context `context` closed: the views of its pages blocks, which the kext frees with no
    /// `BLOCK_RELEASE` once the close is answered, taken out for the caller to let go of with no
    /// lock held. Each goes once the buffers over it have let it go.
    pub(crate) fn context_closed(&mut self, context: u32) -> Vec<Private<P>> {
        let blocks: Vec<u64> = (self.private.iter())
            .filter(|(_, p)| p.context == context)
            .map(|(&at, _)| at)
            .collect();
        blocks
            .iter()
            .filter_map(|at| self.private.remove(at))
            .collect()
    }

    /// The device's reset: every block's view and every cached one go, each buffer over them
    /// let go of in a refusal scope, as no block names its pages any more; what the views did is
    /// still counted.
    pub(crate) fn reset(&mut self) {
        self.resets += 1;
        let views = std::mem::take(&mut self.by_block);
        let private = std::mem::take(&mut self.private);
        let cache = std::mem::take(&mut self.cache);
        self.by_runs.clear();
        self.laid_out.clear();
        if let Err(refusal) = refusals::refusing(move || drop((views, private, cache))) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the IOSurface views were let go of at the device's reset, and a release was \
                 refused: {refusal}"
            );
        }
    }

    /// How many blocks have a view, and how many views the cache keeps (for the tests).
    #[cfg(test)]
    pub(crate) fn len(&self) -> (usize, usize) {
        (self.by_block.len(), self.cache.len())
    }

    /// How many pages blocks have a view (for the tests).
    #[cfg(test)]
    pub(crate) fn private_len(&self) -> usize {
        self.private.len()
    }
}

/// A view just made, and the buffer over the whole of it that holds its import.
pub(crate) type Made<P> = (Arc<SurfaceView<P>>, <P as Platform>::Buffer);

/// The host addresses of the guest pages of `key` in `memory`'s mapping of the guest's RAM, in
/// their order, as runs, each page that follows the last one's end in the same run; `None` where
/// a page is not in the mapping.
fn in_order<P: Platform>(memory: &MemoryMap<P>, key: &[(u64, u64)]) -> Option<Vec<Range<u64>>> {
    let mut runs: Vec<Range<u64>> = Vec::new();
    for &(address, length) in key {
        let end = address.checked_add(length)?;
        for at in (address..end).step_by(GUEST_PAGE.to_usize()) {
            let host = memory.virtual_address_for_physical(at, GUEST_PAGE);
            if host.is_null() {
                return None;
            }
            let host = host.addr().to_u64();
            match runs.last_mut() {
                Some(last) if last.end == host => last.end += GUEST_PAGE,
                _ => runs.push(host..host + GUEST_PAGE),
            }
        }
    }
    Some(runs)
}

/// A view of the runs `key` through `memory`, which holds `lease`, `length` bytes, let go of on
/// `teardown`.
fn map_view<P: Platform>(
    memory: &MemoryMap<P>,
    lease: Lease,
    teardown: Option<Arc<Worker>>,
    key: &[(u64, u64)],
    length: u64,
) -> Result<Arc<SurfaceView<P>>, PagesRefusal> {
    let ranges: Vec<PhysRange> = key
        .iter()
        .map(|&(physical_address, physical_length)| PhysRange {
            physical_address,
            physical_length,
        })
        .collect();
    // Writable: the GPU draws into a surface as it samples one.
    let Some(mapping) = memory.new_range_mapping(&ranges, length, 0) else {
        return Err(PagesRefusal::NotMapped);
    };
    Ok(Arc::new(SurfaceView::new(mapping, lease, teardown)))
}

/// The view of the runs `key` of the block `pages` names, through `memory`, which holds `lease`,
/// let go of on `teardown`, with a buffer over the whole of it on `device`, which holds the
/// runtime's import of its pages; logs what each cost.
#[allow(clippy::too_many_arguments)] // the block, its runs, and what the view is made through
fn make_view<P: Platform>(
    platform: &P,
    device: &P::Device,
    pages: &RgfxSurfacePages,
    runs: &[RgfxPageRun],
    memory: &MemoryMap<P>,
    lease: Lease,
    teardown: Option<Arc<Worker>>,
    key: &[(u64, u64)],
) -> RaiseResult<Result<Made<P>, PagesRefusal>> {
    let clock = platform.clock();
    let start = clock.now_mono();
    let view = match map_view(memory, lease, teardown, key, pages.length) {
        Ok(view) => view,
        Err(why) => return Ok(Err(why)),
    };
    let mapped = clock.now_mono();
    let Some(import) = buffer_over(device, &view, 0, view.length(), 0, None)? else {
        return Ok(Err(PagesRefusal::NoBuffer));
    };
    let imported = clock.now_mono();
    log::debug!(
        target: "reims_vgpu::graphics::surface_pages",
        "the block at {:#x}: a view of {:#x} bytes in {}; mapped in {} us, imported in {} us",
        pages.block_offset,
        pages.length,
        Layout::of(runs, GUEST_PAGE),
        mapped.saturating_duration_since(start).as_micros(),
        imported.saturating_duration_since(mapped).as_micros()
    );
    Ok(Ok((view, import)))
}

/// A Shared buffer of `options` over the `length` bytes at `offset` in `view`, at GPU address
/// `pin` where given, which holds the view until the platform frees it.
pub(crate) fn buffer_over<P: Platform>(
    device: &P::Device,
    view: &Arc<SurfaceView<P>>,
    offset: u64,
    length: u64,
    options: u64,
    pin: Option<u64>,
) -> RaiseResult<Option<P::Buffer>> {
    debug_assert!(
        offset
            .checked_add(length)
            .is_some_and(|end| end <= view.length())
    );
    // SAFETY: the deallocator holds the view, so its pages stay mapped until the platform frees
    // the buffer; the `length` bytes at `offset` lie in the view.
    let bytes = unsafe { HostRange::new(view.base().wrapping_add(offset), length) };
    let keep = Arc::clone(view);
    device.new_buffer(BufferDesc {
        length,
        options: vec![OptionSet::ResourceOptions(options)],
        pinned_gpu_address: pin,
        contents: Some(bytes),
        no_copy: true,
        deallocator: Some(Box::new(move |_, _| drop(keep))),
    })
}

impl<P: Platform> super::GraphicsDecoder<P> {
    /// `BUFFER_OVER_SURFACE_PAGES`: a Shared buffer over the view of the surface whose shared
    /// block the sender holds, lying over the block's part the holders map (`spans`), as a
    /// buffer over host memory lies over its block. A block the sender does not hold is a guest
    /// protocol error; one with no view, or a length past it, is refused.
    pub(super) fn buffer_over_surface_pages(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), super::DecodeError>> {
        let clock = self.platform.clock();
        let start = clock.now_mono();
        let made = self.over_surface_pages(device, message);
        if let Some(m) = super::read::<RgfxBufferOverSurfacePages>(message) {
            log::debug!(
                target: "reims_vgpu::graphics::surface_pages",
                "BUFFER_OVER_SURFACE_PAGES of the block at {:#x} ({:#x} bytes): {} in {} us",
                m.block_offset,
                m.length,
                if matches!(made, Ok(Ok(()))) { "made" } else { "not made" },
                clock.now_mono().saturating_duration_since(start).as_micros()
            );
        }
        made
    }

    /// [`Self::buffer_over_surface_pages`], untimed.
    fn over_surface_pages(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), super::DecodeError>> {
        let Some(m) = super::read::<RgfxBufferOverSurfacePages>(message) else {
            return Ok(Err(super::DecodeError::Malformed));
        };
        if !self.in_own_block(m.block_offset, m.block_size) {
            return Ok(Err(super::DecodeError::BadReference));
        }
        self.settle_surface(m.block_offset);
        let (names, view) = {
            let views = self.shared.surface_views.lock();
            (views.names(&m), views.view_for(&m))
        };
        // A block with a device cache: the buffer is the cache, over the surface's bytes in the
        // pages' layout (`surface_cache`); a block named by its runs has one, or took its view. A
        // deferred block whose pages are not named has its cache, and neither.
        let cache = {
            let caches = self.surface_caches.lock();
            match names {
                true => caches.buffer(m.block_offset),
                false => caches.deferred_buffer(&m),
            }
            .map(|c| c.clone_ref())
        };
        let buffer = match (cache, view) {
            (Some(cache), _) => cache,
            (None, Some(view)) => {
                match buffer_over(device, &view, 0, m.length, m.options.into(), None)? {
                    Some(buffer) => buffer,
                    None => return Ok(Err(super::DecodeError::Refused)),
                }
            }
            (None, None) => return Ok(Err(super::DecodeError::Refused)),
        };
        let span = Some((m.block_offset, m.block_size));
        let entry = super::tables::Entry::of(super::ObjHandle::Buffer(buffer)).over(span);
        self.store(m.obj_id, entry.over_view());
        Ok(Ok(()))
    }

    /// `BUFFER_OVER_PAGES`: a Shared buffer over the `length` bytes at `offset` in the view of a
    /// pages block of the sender's (the module doc), at the guest's GPU address, lying over the
    /// block, so the block is not released while it lives, its contents those bytes of the view.
    /// A block the sender does not hold is a guest protocol error; one that is no pages block of
    /// its context's, or a range past its view, is refused.
    pub(super) fn buffer_over_pages(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), super::DecodeError>> {
        let Some(m) = super::read::<RgfxBufferOverPages>(message) else {
            return Ok(Err(super::DecodeError::Malformed));
        };
        if !self.in_own_block(m.block_offset, m.block_size) {
            return Ok(Err(super::DecodeError::BadReference));
        }
        // Its block's `SURFACE_PAGES` is carried out on the surface worker.
        self.settle_surface(m.block_offset);
        let view = SurfaceViews::private_view(
            &self.shared.surface_views,
            &*self.platform,
            device,
            m.block_offset,
            m.block_size,
            self.context,
        )?;
        let Some(view) = view.filter(|v| {
            m.length != 0 && (m.offset.checked_add(m.length)).is_some_and(|end| end <= v.length())
        }) else {
            return Ok(Err(super::DecodeError::Refused));
        };
        // At the guest's address, which its pointers to the buffer hold, as a
        // BUFFER_OVER_HOST_MEMORY's is.
        let pin = (m.gpu_address != 0).then_some(m.gpu_address);
        let Some(buffer) = buffer_over(device, &view, m.offset, m.length, m.options.into(), pin)?
        else {
            return Ok(Err(super::DecodeError::Refused));
        };
        // Its contents are the view's bytes at the offset, which the buffer keeps mapped.
        let bytes = super::GuestBytes {
            ram: Some(view.base()),
            at: m.offset,
            length: m.length,
        };
        let entry = super::tables::Entry::of(super::ObjHandle::Buffer(buffer))
            .over(Some((m.block_offset, m.block_size)))
            .with_bytes(Some(bytes));
        self.store(m.obj_id, entry);
        Ok(Ok(()))
    }

    /// `HEAP_OVER_PAGES`: a Shared heap over the pages of a pages block of the sender's (the
    /// module doc), in place, as a `HEAP_OVER_HOST_MEMORY` is over the region: its memory is the
    /// `size_bytes` from the pages' first byte, rounded up to Metal's 16 KiB page, where the
    /// guest's RAM is mapped for a block of one run, else in the block's view. It lies over the
    /// block, and records its bytes, where a `BUFFER_MANAGED_AT_OFFSET` in it finds its CPU copy.
    /// A block the sender does not hold is a guest protocol error; one that is no pages block of
    /// its context's, has no view, or is smaller than the heap, is refused.
    pub(super) fn heap_over_pages(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), super::DecodeError>> {
        let Some(m) = super::read::<RgfxHeapOverPages>(message) else {
            return Ok(Err(super::DecodeError::Malformed));
        };
        if !self.in_own_block(m.block_offset, m.block_size) {
            return Ok(Err(super::DecodeError::BadReference));
        }
        // Its block's `SURFACE_PAGES` is carried out on the surface worker.
        self.settle_surface(m.block_offset);
        let ram = (self.shared.surface_views.lock()).private_ram(
            m.block_offset,
            m.block_size,
            self.context,
        );
        // One run: the guest's RAM itself. More: the block's view, made here where it is not yet
        // (on the surface worker, `GraphicsControl::start_over_pages`).
        let pages = match ram {
            Some(run) => Some(run),
            None => SurfaceViews::private_view(
                &self.shared.surface_views,
                &*self.platform,
                device,
                m.block_offset,
                m.block_size,
                self.context,
            )?
            .map(|view| (view.base(), view.length())),
        };
        let Some((at, length)) = pages.and_then(|(at, room)| {
            let length = m.size_bytes.checked_next_multiple_of(super::HEAP_PAGE)?;
            (m.size_bytes != 0 && length <= room).then_some((at, length))
        }) else {
            return Ok(Err(super::DecodeError::Refused));
        };
        // SAFETY: the bytes lie in the run or the view (checked above): the mapping of the
        // guest's RAM, which stays mapped as long as the device, or the block's view, which its
        // entry holds until the block's release, refused while the heap lies over the block, or
        // its context's close, which lets go of the heap first (the module doc).
        let bytes = unsafe { HostRange::new(at, length) };
        let desc = HeapDesc {
            size: m.size_bytes,
            heap_type: super::host_heap_type(m.heap_type),
            options: vec![OptionSet::ResourceOptions(m.options.into())],
            address_ranges: None,
            pinned_gpu_address: (m.gpu_address != 0).then_some(m.gpu_address),
        };
        match device.new_heap_over_host_memory(bytes, &desc)?.heap {
            Some(heap) => {
                let mut entry = super::tables::Entry::of(super::ObjHandle::Heap(heap))
                    .over(Some((m.block_offset, m.block_size)));
                entry.ram = Some((at, length));
                self.store(m.obj_id, entry);
                Ok(Ok(()))
            }
            None => {
                super::heap_refused(
                    m.obj_id,
                    m.size_bytes,
                    m.heap_type,
                    m.options,
                    m.gpu_address,
                );
                Ok(Err(super::DecodeError::Refused))
            }
        }
    }
}

#[cfg(test)]
mod costs;

#[cfg(test)]
pub(super) mod tests {
    use reims_vgpu_graphics::region::Region;
    use reims_vgpu_graphics::wire::{
        RGFX_CMD_BUFFER_AT_OFFSET, RGFX_CMD_BUFFER_OVER_HOST_MEMORY, RGFX_CMD_BUFFER_OVER_PAGES,
        RGFX_CMD_BUFFER_OVER_SURFACE_PAGES, RGFX_CMD_HEAP_OVER_PAGES, RGFX_CMD_OBJECT_DESTROY,
        RGFX_CMD_SURFACE_PAGES, RGFX_CMD_TEXTURE_FROM_BUFFER, RGFX_CONTEXT_ALLOCATIONS_OFFSET,
        RGFX_CONTEXT_OBJ_ID_SHIFT, RGFX_CURSOR_SPAN, RGFX_STATUS_CACHE,
        RGFX_STATUS_INVALID_ARGUMENT, RGFX_STATUS_OK, RGFX_STATUS_REFUSED,
        RGFX_SURFACE_PAGES_ASK_CACHE, RGFX_SURFACE_PAGES_CACHED, RGFX_SURFACE_PAGES_PRIVATE,
        RgfxBufferAtOffset, RgfxBufferOverHostMemory, RgfxMsgHeader, RgfxObjectDestroy,
        RgfxSurfacePagesLayout, RgfxSurfacePlaneLayout, RgfxTextureDescriptor,
        RgfxTextureFromBuffer,
    };
    use reims_vgpu_vulkan::core::memory::descriptor::{GuestPhysicalRange, MemoryMapDescriptor};
    use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
    use reims_vgpu_vulkan::core::platform::values::PagesInPlace;
    use reims_vgpu_vulkan::core::testkit::{Arg, FakePlatform, GuestRam};
    use zerocopy::IntoBytes;

    use super::super::{DecodeError, GraphicsDecoder, no_raise};
    use super::*;

    /// The processes' blocks start here.
    pub(crate) const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x4000;
    /// The guest RAM's first byte.
    pub(crate) const RAM: u64 = 0x10_0000;

    pub(crate) struct Rig {
        pub(crate) p: Arc<FakePlatform>,
        pub(crate) device: <FakePlatform as Platform>::Device,
        pub(crate) ram: GuestRam,
        pub(crate) dec: GraphicsDecoder<FakePlatform>,
        /// The host's page, which the runs are whole pages of here.
        pub(crate) page: u64,
    }

    /// A decoder over a region with a MiB of blocks, and `pages` host pages of guest RAM from
    /// [`RAM`] behind its memory map.
    pub(crate) fn rig(pages: u64) -> Rig {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let page = p.vm().page_size();
        let size = BLOCK + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(Region::new(size as usize).unwrap());
        let mut dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), region);
        let ram = GuestRam::with_range(RAM, pages * page).unwrap();
        p.fake_vm().adopt(&ram);
        let mut desc = MemoryMapDescriptor::new();
        for r in ram.ranges() {
            desc.add_range(GuestPhysicalRange {
                physical_address: r.physical_address,
                physical_length: r.physical().physical_length,
                host_address: r.host(),
            });
        }
        let map = MemoryMap::new(Arc::clone(&p), &desc).unwrap();
        dec.set_guest_memory(map, Arc::new(()));
        Rig {
            p,
            device,
            ram,
            dec,
            page,
        }
    }

    fn header(r#type: u32) -> RgfxMsgHeader {
        RgfxMsgHeader {
            r#type,
            size: 0,
            seqno: 0,
        }
    }

    impl Rig {
        /// A `SURFACE_PAGES` for the block at `block` of `runs`, given as (page, pages) of guest
        /// RAM, over `length` bytes; its answer.
        pub(crate) fn pages(&mut self, block: u64, runs: &[(u64, u64)], length: u64) -> u32 {
            self.pages_cached(block, runs, length, false)
        }

        /// [`Self::pages`], the block marked `cached`.
        pub(crate) fn pages_cached(
            &mut self,
            block: u64,
            runs: &[(u64, u64)],
            length: u64,
            cached: bool,
        ) -> u32 {
            let flags = if cached { RGFX_SURFACE_PAGES_CACHED } else { 0 };
            self.pages_flagged(block, 0, runs, length, flags)
        }

        /// A `SURFACE_PAGES` for the block at `block` of context `context`, of `runs` (as
        /// [`Self::pages`]) over `length` bytes, with `flags`; its answer.
        pub(crate) fn pages_flagged(
            &mut self,
            block: u64,
            context: u32,
            runs: &[(u64, u64)],
            length: u64,
            flags: u32,
        ) -> u32 {
            self.pages_laid_out(block, context, runs, length, flags, None)
        }

        /// A `SURFACE_PAGES` for the block at `block` of `runs` (as [`Self::pages`]) over
        /// `length` bytes that asks whether it needs a device cache, with the layout `layout`; its
        /// answer.
        pub(crate) fn pages_asking(
            &mut self,
            block: u64,
            runs: &[(u64, u64)],
            length: u64,
            layout: Option<RgfxSurfacePagesLayout>,
        ) -> u32 {
            let flags = RGFX_SURFACE_PAGES_ASK_CACHE;
            self.pages_laid_out(block, 0, runs, length, flags, layout)
        }

        /// [`Self::pages_flagged`], the message carrying `layout`.
        fn pages_laid_out(
            &mut self,
            block: u64,
            context: u32,
            runs: &[(u64, u64)],
            length: u64,
            flags: u32,
            layout: Option<RgfxSurfacePagesLayout>,
        ) -> u32 {
            let runs: Vec<RgfxPageRun> = runs
                .iter()
                .map(|&(at, n)| RgfxPageRun {
                    address: RAM + at * self.page,
                    length: n * self.page,
                })
                .collect();
            let m = RgfxSurfacePages {
                header: header(RGFX_CMD_SURFACE_PAGES),
                context,
                run_count: runs.len() as u32,
                block_offset: block,
                block_size: 0x2000,
                runs_offset: block + 0x1000,
                length,
            };
            (self.dec).surface_pages_laid_out(&self.device, &m, &runs, flags, layout)
        }

        /// A `BUFFER_OVER_SURFACE_PAGES` of `length` bytes over the block at `block`.
        pub(crate) fn buffer(
            &mut self,
            obj_id: u64,
            block: u64,
            length: u64,
        ) -> Result<(), DecodeError> {
            let m = RgfxBufferOverSurfacePages {
                header: header(RGFX_CMD_BUFFER_OVER_SURFACE_PAGES),
                obj_id,
                block_offset: block,
                block_size: 0x1000,
                length,
                options: 0,
                reserved_34: 0,
            };
            self.dec.decode(&self.device, m.as_bytes()).unwrap()
        }

        /// The view the block at `block` has.
        pub(crate) fn view(&self, block: u64) -> Arc<SurfaceView<FakePlatform>> {
            Arc::clone(&self.dec.views_of_surfaces().by_block[&block].0)
        }

        /// The view the pages block at `block` has: its first buffer made it.
        fn private_view(&self, block: u64) -> Arc<SurfaceView<FakePlatform>> {
            let views = self.dec.views_of_surfaces();
            Arc::clone(&views.private[&block].view.as_ref().expect("no view yet").0)
        }

        /// Whether the pages block at `block` has a view.
        fn has_private_view(&self, block: u64) -> bool {
            self.dec.views_of_surfaces().private[&block].view.is_some()
        }

        /// A `BUFFER_OVER_PAGES` of `length` bytes at `offset` in the view of the pages block at
        /// `block` (of [`Self::pages_flagged`]'s size), at `gpu_address`.
        fn pages_buffer(
            &mut self,
            obj_id: u64,
            gpu_address: u64,
            block: u64,
            offset: u64,
            length: u64,
        ) -> Result<(), DecodeError> {
            let m = RgfxBufferOverPages {
                header: header(RGFX_CMD_BUFFER_OVER_PAGES),
                obj_id,
                gpu_address,
                block_offset: block,
                block_size: 0x2000,
                offset,
                length,
                options: 0,
                reserved_44: 0,
            };
            self.dec.decode(&self.device, m.as_bytes()).unwrap()
        }

        /// A `HEAP_OVER_PAGES` of `size` bytes over the pages block at `block` (of
        /// [`Self::pages_flagged`]'s size), at an address of its id's.
        pub(crate) fn pages_heap(
            &mut self,
            obj_id: u64,
            block: u64,
            size: u64,
        ) -> Result<(), DecodeError> {
            let m = RgfxHeapOverPages {
                header: header(RGFX_CMD_HEAP_OVER_PAGES),
                obj_id,
                gpu_address: 0x2_0000_0000 + (obj_id & 0xff) * 0x100_0000,
                block_offset: block,
                block_size: 0x2000,
                size_bytes: size,
                options: 0,
                heap_type: 1,
            };
            self.dec.decode(&self.device, m.as_bytes()).unwrap()
        }

        /// The host's first byte of guest RAM.
        pub(crate) fn ram_base(&self) -> HostPtr {
            self.ram.ranges()[0].host()
        }

        /// The GPU address and the contents of the buffer `obj_id` names.
        fn buffer_at(&self, obj_id: u64) -> (u64, HostPtr) {
            match self.dec.object(obj_id) {
                Some(super::super::ObjHandle::Buffer(b)) => (b.gpu_address(), b.contents()),
                _ => panic!("no buffer {obj_id:#x}"),
            }
        }
    }

    /// A surface's runs, scattered and out of order in guest RAM, are one view in their order,
    /// over the guest's own pages: a buffer over the block lies over it, and over the block's
    /// part its holders map, so the block is not released while it lives
    #[test]
    fn a_surface_s_runs_are_one_view_its_buffers_lie_over() {
        let mut r = rig(8);
        let page = r.page;
        // Pages 5 and 6, then page 1.
        assert!(r.ram.write(RAM + 5 * page, b"fifth"));
        assert!(r.ram.write(RAM + page, b"first"));
        assert_eq!(r.pages(BLOCK, &[(5, 2), (1, 1)], 3 * page), RGFX_STATUS_OK);
        let view = r.view(BLOCK);
        assert_eq!(view.length(), 3 * page);
        let fake = r.p.fake_vm();
        assert_eq!(fake.read(view.base(), 5).unwrap(), b"fifth");
        assert_eq!(
            fake.read(view.base().wrapping_add(2 * page), 5).unwrap(),
            b"first"
        );
        // Writes through the view reach the guest's page.
        assert!(fake.write(view.base().wrapping_add(2 * page), b"GPU"));
        fake.sync();
        let mut got = [0u8; 3];
        assert!(r.ram.read(RAM + page, &mut got));
        assert_eq!(&got, b"GPU");

        let mark = r.p.call_log().mark();
        assert_eq!(r.buffer(10, BLOCK, 3 * page), Ok(()));
        assert_eq!(
            r.p.call_log().sels_since(mark),
            ["newBufferWithDescriptor:"]
        );
        assert!(r.dec.span_in_use(BLOCK, 0x1000));
        // Past the view, or a block with none: refused.
        assert_eq!(r.buffer(11, BLOCK, 3 * page + 1), Err(DecodeError::Refused));
        assert_eq!(
            r.buffer(12, BLOCK + 0x10000, page),
            Err(DecodeError::Refused)
        );
        // The buffer holds the view after the block's release.
        r.dec
            .block_released(&r.device, BLOCK, 0, Box::new(|| ()), no_raise());
        assert_eq!(r.buffer(13, BLOCK, page), Err(DecodeError::Refused));
        assert!(
            Arc::strong_count(&view) > 2,
            "the buffer and the cache hold it"
        );
        let destroy = RgfxObjectDestroy {
            header: header(RGFX_CMD_OBJECT_DESTROY),
            obj_id: 10,
        };
        assert_eq!(r.dec.decode(&r.device, destroy.as_bytes()).unwrap(), Ok(()));
        assert!(!r.dec.span_in_use(BLOCK, 0x1000));
    }

    /// A block made again over the same runs takes the cached view, with no mapping made; the
    /// cache keeps [`CACHED_VIEWS`] views, the last taken
    #[test]
    fn a_block_over_the_same_runs_takes_the_cached_view() {
        let mut r = rig(64);
        let page = r.page;
        assert_eq!(r.pages(BLOCK, &[(3, 1), (0, 1)], 2 * page), RGFX_STATUS_OK);
        let first = r.view(BLOCK);
        r.dec
            .block_released(&r.device, BLOCK, 0, Box::new(|| ()), no_raise());
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.pages(BLOCK + 0x8000, &[(3, 1), (0, 1)], 2 * page),
            RGFX_STATUS_OK
        );
        assert!(
            r.p.call_log().sels_since(mark).is_empty(),
            "no mapping or buffer made"
        );
        assert!(Arc::ptr_eq(&first, &r.view(BLOCK + 0x8000)));
        assert_eq!(r.dec.views_of_surfaces().counts().reused, 1);
        // The same pages in another order are another view.
        assert_eq!(r.pages(BLOCK, &[(0, 1), (3, 1)], 2 * page), RGFX_STATUS_OK);
        assert!(!Arc::ptr_eq(&first, &r.view(BLOCK)));
        // Past CACHED_VIEWS views the oldest goes from the cache.
        for i in 0..CACHED_VIEWS as u64 {
            assert_eq!(r.pages(BLOCK, &[(10 + i, 1)], page), RGFX_STATUS_OK);
        }
        assert_eq!(r.dec.views_of_surfaces().len().1, CACHED_VIEWS);
        r.dec
            .block_released(&r.device, BLOCK + 0x8000, 0, Box::new(|| ()), no_raise());
        // The block's hold goes on the surface worker.
        r.dec.surface_worker().flush();
        assert_eq!(
            Arc::strong_count(&first),
            1,
            "only the test holds the first view"
        );
    }

    /// The cached views no block holds keep no more mappings, a mapping a run, than the
    /// platform answers for them: as a view is cached, the oldest of them go until the rest fit,
    /// and a view its block holds stays, however old
    #[test]
    fn the_unheld_cached_views_keep_the_platform_s_mappings_at_most() {
        let mut r = rig(64);
        let page = r.page;
        r.p.fake_vm().set_idle_view_mappings(Some(4));
        let block = |i: u64| BLOCK + i * 0x8000;
        // The oldest, of three runs, its block held throughout.
        assert_eq!(
            r.pages(block(0), &[(0, 1), (2, 1), (4, 1)], 3 * page),
            RGFX_STATUS_OK
        );
        let held = Arc::downgrade(&r.view(block(0)));
        // Three more, of two, two and one runs, whose blocks go.
        let sets: [&[(u64, u64)]; 3] = [&[(6, 1), (8, 1)], &[(10, 1), (12, 1)], &[(14, 1)]];
        let mut unheld = Vec::new();
        for (at, runs) in (1u64..).map(block).zip(sets) {
            assert_eq!(
                r.pages(at, runs, runs.len().to_u64() * page),
                RGFX_STATUS_OK
            );
            unheld.push(Arc::downgrade(&r.view(at)));
            r.dec
                .block_released(&r.device, at, 0, Box::new(|| ()), no_raise());
        }
        // The blocks' holds go on the surface worker.
        r.dec.surface_worker().flush();
        assert!(unheld.iter().all(|v| v.upgrade().is_some()), "cached");
        // Five mappings unheld, past the four: the next view cached lets go of the oldest
        // unheld one alone.
        assert_eq!(
            r.pages(block(4), &[(16, 1), (18, 1)], 2 * page),
            RGFX_STATUS_OK
        );
        assert!(held.upgrade().is_some(), "the view its block holds stays");
        assert!(unheld[0].upgrade().is_none(), "the oldest unheld view went");
        assert!(
            unheld[1..].iter().all(|v| v.upgrade().is_some()),
            "the rest fit"
        );
        let views = r.dec.views_of_surfaces();
        assert_eq!(views.counts().idle_returned, 1);
        assert_eq!(views.len(), (2, 4));
    }

    /// A block asking whether it needs a device cache whose view the platform has no room for,
    /// with no unheld cached view to let go of, is answered `RGFX_STATUS_CACHE` with no view
    /// mapped and the device not asked, counted apart, and its cached `SURFACE_PAGES` then gets
    /// a device cache; with room, a block asking takes the device's answer as before
    #[test]
    fn a_block_asking_with_no_room_for_its_view_is_answered_a_cache() {
        let mut r = rig(16);
        let page = r.page;
        r.dec.set_unified_memory(true);
        let world = Arc::clone(r.p.world());
        let layout = one_plane(page, page - 16, 2);
        let runs: [&[(u64, u64)]; 2] = [&[(1, 1), (3, 1)], &[(5, 1), (7, 1)]];

        // Room for the four mappings a view of two runs adds: in place.
        r.p.fake_vm().set_view_room(Some(4));
        assert_eq!(
            r.pages_asking(BLOCK, runs[0], 2 * page, Some(layout)),
            RGFX_STATUS_OK
        );
        assert!(r.dec.views_of_surfaces().view(BLOCK).is_some());
        assert_eq!(world.surface_pages_asked().len(), 1);

        // Room for three, and the one view its block holds: a cache, nothing mapped or asked.
        r.p.fake_vm().set_view_room(Some(3));
        let second = BLOCK + 0x8000;
        let made = r.dec.views_of_surfaces().counts().made;
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.pages_asking(second, runs[1], 2 * page, Some(layout)),
            RGFX_STATUS_CACHE
        );
        let sels = r.p.call_log().sels_since(mark);
        assert!(
            !sels.contains(&"mach_vm_allocate"),
            "no view mapped: {sels:?}"
        );
        assert!(r.dec.views_of_surfaces().view(second).is_none());
        assert!(r.dec.views_of_surfaces().view(BLOCK).is_some());
        assert_eq!(r.dec.views_of_surfaces().counts().made, made);
        assert_eq!(world.surface_pages_asked().len(), 1, "the device not asked");
        let c = r.dec.views_of_surfaces().counts();
        assert_eq!((c.asked_in_place, c.asked_cache, c.asked_room), (1, 1, 1));

        // Its cached SURFACE_PAGES gets a device cache.
        assert_eq!(
            r.pages_cached(second, runs[1], 2 * page, true),
            RGFX_STATUS_OK
        );
        assert!(r.dec.caches().has(second));
    }

    /// Where the platform has too little room for a new view's mappings, cached views no block
    /// holds go before it is made, until the mappings they hold cover what it lacks: first those
    /// whose mapping no device's import holds, the oldest first, torn down at once; then one a
    /// device's import holds, torn down on that device's thread before the new view is mapped. A
    /// view its block holds stays, however old. Where the unheld views hold too few mappings to
    /// cover it (a view the platform mapped another way holds fewer than its runs), none goes,
    /// and the view is made all the same, as the platform maps it. A pages block's view makes
    /// room alike
    #[test]
    fn a_view_short_of_room_lets_unheld_cached_views_go_by_the_mappings_they_hold() {
        let mut r = rig(64);
        let page = r.page;
        let block = |i: u64| BLOCK + i * 0x8000;
        // The oldest, of three runs, its block held throughout.
        assert_eq!(
            r.pages(block(0), &[(0, 1), (2, 1), (4, 1)], 3 * page),
            RGFX_STATUS_OK
        );
        let held = Arc::downgrade(&r.view(block(0)));
        // Three more, of two runs each, whose blocks go.
        let sets: [&[(u64, u64)]; 3] =
            [&[(6, 1), (8, 1)], &[(10, 1), (12, 1)], &[(14, 1), (16, 1)]];
        let mut unheld = Vec::new();
        for (at, runs) in (1u64..).map(block).zip(sets) {
            assert_eq!(r.pages(at, runs, 2 * page), RGFX_STATUS_OK);
            unheld.push(Arc::downgrade(&r.view(at)));
            r.dec
                .block_released(&r.device, at, 0, Box::new(|| ()), no_raise());
        }
        // The oldest unheld view's memory, held by a device that lets go of it on a thread of
        // its own a little after the view's life ends, as a CUDA registration's reaper does.
        let memory = unheld[0]
            .upgrade()
            .and_then(|v| v.memory())
            .expect("a view's memory");
        let threads = r.p.threads().handle();
        memory.identity().when_gone(move || {
            reims_vgpu_vulkan::core::rt::spawn_named(
                &*threads,
                "reims-vgpu.test",
                512 * 1024,
                Box::new(()),
                move || {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    drop(memory);
                },
            );
        });
        // The blocks' holds go on the surface worker.
        r.dec.surface_worker().flush();
        assert!(unheld.iter().all(|v| v.upgrade().is_some()), "cached");
        // The teardowns since `mark`, and whether each came before the next view was mapped.
        let torn_down = |r: &Rig, mark: u64| {
            let sels = r.p.call_log().sels_since(mark);
            let made_at = sels.iter().position(|&s| s == "mach_vm_allocate");
            (sels.iter().enumerate())
                .filter(|&(_, &s)| s == "mach_vm_deallocate")
                .map(|(i, _)| Some(i) < made_at)
                .collect::<Vec<_>>()
        };

        // A view of three runs adds up to five mappings, with room for two: the two unheld views
        // no device holds go, four mappings, torn down before the view is mapped; the oldest,
        // whose mapping a device holds, stays.
        r.p.fake_vm().set_view_room(Some(2));
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.pages(block(4), &[(20, 1), (22, 1), (24, 1)], 3 * page),
            RGFX_STATUS_OK
        );
        assert!(held.upgrade().is_some(), "the view its block holds stays");
        assert!(
            unheld[1..].iter().all(|v| v.upgrade().is_none()),
            "the unheld views no device holds went"
        );
        assert!(
            unheld[0].upgrade().is_some(),
            "the one a device holds stays"
        );
        assert_eq!(torn_down(&r, mark), [true, true]);
        assert_eq!(r.dec.views_of_surfaces().counts().idle_returned, 2);

        // No room at all, three mappings short: the view of block 4, unheld since and no
        // device's, covers it alone.
        r.dec
            .block_released(&r.device, block(4), 0, Box::new(|| ()), no_raise());
        r.dec.surface_worker().flush();
        r.p.fake_vm().set_view_room(Some(0));
        let mark = r.p.call_log().mark();
        assert_eq!(r.pages(block(5), &[(30, 1)], page), RGFX_STATUS_OK);
        assert!(held.upgrade().is_some(), "the view its block holds stays");
        assert!(
            unheld[0].upgrade().is_some(),
            "the one a device holds stays"
        );
        assert_eq!(torn_down(&r, mark), [true]);
        assert_eq!(r.dec.views_of_surfaces().counts().idle_returned, 3);

        // A pages block's view of one run, three short again: the view of block 5, one mapping,
        // and the one a device holds, two, go, its mapping torn down on the device's thread
        // before the view is mapped.
        let last = Arc::downgrade(&r.view(block(5)));
        r.dec
            .block_released(&r.device, block(5), 0, Box::new(|| ()), no_raise());
        r.dec.surface_worker().flush();
        let pages = block(6);
        assert_eq!(
            r.pages_flagged(pages, 0, &[(40, 1)], page, RGFX_SURFACE_PAGES_PRIVATE),
            RGFX_STATUS_OK
        );
        let mark = r.p.call_log().mark();
        assert_eq!(r.pages_buffer(10, 0x1_0000_0000, pages, 0, page), Ok(()));
        assert!(r.has_private_view(pages));
        assert!(last.upgrade().is_none(), "the unheld view went");
        assert!(unheld[0].upgrade().is_none(), "the one a device held went");
        assert!(held.upgrade().is_some(), "the view its block holds stays");
        assert_eq!(torn_down(&r, mark), [true, true]);
        assert_eq!(r.dec.views_of_surfaces().counts().idle_returned, 5);

        // An unheld view of three runs mapped another way, holding one mapping: too few for
        // the three a view of one run lacks, so it stays, and the view is made.
        assert_eq!(
            r.pages(block(7), &[(50, 1), (52, 1), (54, 1)], 3 * page),
            RGFX_STATUS_OK
        );
        let other = r.view(block(7));
        r.p.fake_vm().set_view_mappings(other.base(), Some(1));
        let other = Arc::downgrade(&other);
        r.dec
            .block_released(&r.device, block(7), 0, Box::new(|| ()), no_raise());
        r.dec.surface_worker().flush();
        let mark = r.p.call_log().mark();
        assert_eq!(r.pages(block(8), &[(60, 1)], page), RGFX_STATUS_OK);
        assert!(
            other.upgrade().is_some(),
            "the view mapped another way stays"
        );
        assert!(r.dec.views_of_surfaces().view(block(8)).is_some());
        assert_eq!(torn_down(&r, mark), Vec::<bool>::new());
        assert_eq!(r.dec.views_of_surfaces().counts().idle_returned, 5);
    }

    /// A surface of one plane of `rows` rows of `row_bytes` texel bytes `bytes_per_row` apart.
    pub(crate) fn one_plane(
        bytes_per_row: u64,
        row_bytes: u64,
        rows: u64,
    ) -> RgfxSurfacePagesLayout {
        let mut layout = RgfxSurfacePagesLayout {
            plane_count: 1,
            ..Default::default()
        };
        layout.planes[0] = RgfxSurfacePlaneLayout {
            offset: 0,
            bytes_per_row,
            row_bytes,
            rows,
        };
        layout
    }

    /// A block in place that asks whether it needs a device cache, on a host whose GPU shares the
    /// CPU's memory, takes the device's answer for its view and its planes. Where the device
    /// answers that every texture over them lies in place, or that its verdict is pending, the
    /// block keeps its view in place with no cache, as a block that asks nothing does, and its
    /// buffers lie over the pages. Where it answers no, the block is answered `RGFX_STATUS_CACHE`
    /// and keeps no view, the view staying in the cache of views; the block's cached
    /// `SURFACE_PAGES` that follows takes that view, mapping nothing again, and gets a device
    /// cache, which its buffers are. A layout that does not lie in the surface's bytes, an ask
    /// with no layout, or an ask of a cached block, is refused and makes nothing
    #[test]
    fn a_block_asking_for_a_device_cache_takes_the_devices_answer() {
        let mut r = rig(16);
        let page = r.page;
        r.dec.set_unified_memory(true);
        let world = Arc::clone(r.p.world());
        let layout = one_plane(page, page - 16, 2);
        let planes = vec![PagesPlane {
            offset: 0,
            bytes_per_row: page,
            row_bytes: page - 16,
            rows: 2,
        }];

        // In place: a view, no cache, and the buffers over the pages.
        let made = r.dec.views_of_surfaces().counts().made;
        assert_eq!(
            r.pages_asking(BLOCK, &[(1, 2)], 2 * page, Some(layout)),
            RGFX_STATUS_OK
        );
        assert_eq!(world.surface_pages_asked(), vec![planes.clone()]);
        assert!(r.dec.views_of_surfaces().view(BLOCK).is_some());
        assert!(!r.dec.caches().has(BLOCK));
        assert_eq!(r.buffer(10, BLOCK, 2 * page), Ok(()));
        assert_eq!(r.buffer_at(10).1, r.view(BLOCK).base(), "over the pages");

        // A verdict not known yet: in place too, counted apart.
        let second = BLOCK + 0x8000;
        world.answer_surface_pages(PagesInPlace::Pending);
        assert_eq!(
            r.pages_asking(second, &[(3, 2)], 2 * page, Some(layout)),
            RGFX_STATUS_OK
        );
        assert!(r.dec.views_of_surfaces().view(second).is_some());
        assert!(!r.dec.caches().has(second));

        // Not in place: answered a cache, with no view of its own; its cached SURFACE_PAGES
        // takes the view made for the ask.
        let third = BLOCK + 0x10000;
        world.answer_surface_pages(PagesInPlace::No("the rows end inside a line".into()));
        assert_eq!(
            r.pages_asking(third, &[(5, 2)], 2 * page, Some(layout)),
            RGFX_STATUS_CACHE
        );
        assert!(r.dec.views_of_surfaces().view(third).is_none());
        assert!(!r.dec.caches().has(third));
        assert_eq!(r.dec.views_of_surfaces().counts().made, made + 3);
        assert_eq!(
            r.pages_cached(third, &[(5, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        assert_eq!(
            r.dec.views_of_surfaces().counts().made,
            made + 3,
            "the cached block takes the ask's view"
        );
        assert!(r.dec.caches().has(third));
        let mark = r.p.call_log().mark();
        assert_eq!(r.buffer(11, third, 2 * page), Ok(()));
        assert!(
            r.p.call_log().sels_since(mark).is_empty(),
            "no buffer made: it is the cache"
        );
        let c = r.dec.views_of_surfaces().counts();
        assert_eq!(
            (c.asked_in_place, c.asked_pending, c.asked_cache),
            (1, 1, 1)
        );

        // Refused, nothing made or asked.
        world.answer_surface_pages(PagesInPlace::Yes);
        let asked = world.surface_pages_asked().len();
        let fourth = BLOCK + 0x18000;
        let past = one_plane(page, page, 3);
        let none = one_plane(page, 0, 2);
        for layout in [Some(past), Some(none), None] {
            assert_eq!(
                r.pages_asking(fourth, &[(7, 2)], 2 * page, layout),
                RGFX_STATUS_INVALID_ARGUMENT
            );
        }
        let flags = RGFX_SURFACE_PAGES_ASK_CACHE | RGFX_SURFACE_PAGES_CACHED;
        assert_eq!(
            r.pages_flagged(fourth, 0, &[(7, 2)], 2 * page, flags),
            RGFX_STATUS_INVALID_ARGUMENT
        );
        assert!(r.dec.views_of_surfaces().view(fourth).is_none());
        assert!(!r.dec.caches().has(fourth));
        assert_eq!(world.surface_pages_asked().len(), asked);
    }

    /// Where the device answers, before any view of a block's pages, that textures over them
    /// would lie in place over no buffer of them, a block that asks whether it needs a device
    /// cache is answered `RGFX_STATUS_CACHE` with no view made and nothing asked over one,
    /// counted as answered a cache, its layout kept for its cached `SURFACE_PAGES`, which gets a
    /// device cache. An ask whose runs fail a view's checks is refused as before. Where the device
    /// leaves it to the view, the block is asked over its view and kept in place where the device
    /// answers so, as before
    #[test]
    fn a_block_in_place_over_no_buffer_is_answered_a_cache_with_no_view() {
        let mut r = rig(16);
        let page = r.page;
        r.dec.set_unified_memory(true);
        let world = Arc::clone(r.p.world());
        let layout = one_plane(page, page - 16, 2);
        let planes = vec![PagesPlane {
            offset: 0,
            bytes_per_row: page,
            row_bytes: page - 16,
            rows: 2,
        }];
        world.answer_surface_pages(PagesInPlace::Yes);
        world.answer_surface_pages_unmade(Some("no import's rows lie on its lines".into()));

        let made = r.dec.views_of_surfaces().counts().made;
        assert_eq!(
            r.pages_asking(BLOCK, &[(1, 2)], 2 * page, Some(layout)),
            RGFX_STATUS_CACHE
        );
        {
            let mut views = r.dec.views_of_surfaces();
            assert_eq!(views.counts().made, made, "no view made");
            assert_eq!(views.counts().asked_cache, 1);
            assert!(views.view(BLOCK).is_none());
            let kept = views.take_layout(BLOCK);
            assert_eq!(kept.as_ref(), Some(&planes), "the layout, for the cache");
            views.laid_out(BLOCK, planes.clone());
        }
        assert!(
            world.surface_pages_asked().is_empty(),
            "nothing asked over a view"
        );
        assert!(!r.dec.caches().has(BLOCK));
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        assert!(r.dec.caches().has(BLOCK));
        assert!(
            r.dec.views_of_surfaces().take_layout(BLOCK).is_none(),
            "taken by the cache"
        );

        // Runs past guest RAM: refused, as the view's path refuses them.
        let second = BLOCK + 0x8000;
        assert_eq!(
            r.pages_asking(second, &[(100, 2)], 2 * page, Some(layout)),
            RGFX_STATUS_REFUSED
        );
        assert_eq!(r.dec.views_of_surfaces().counts().asked_cache, 1);

        // Left to the view: asked over it, and in place.
        world.answer_surface_pages_unmade(None);
        let third = BLOCK + 0x10000;
        assert_eq!(
            r.pages_asking(third, &[(5, 2)], 2 * page, Some(layout)),
            RGFX_STATUS_OK
        );
        assert_eq!(world.surface_pages_asked(), vec![planes]);
        assert!(r.dec.views_of_surfaces().view(third).is_some());
        assert!(!r.dec.caches().has(third));
    }

    /// Runs that are not whole pages, that do not sum to the surface's length, or that lie
    /// outside guest RAM make no view, are refused, and are counted
    #[test]
    fn bad_runs_make_no_view() {
        let mut r = rig(8);
        let page = r.page;
        assert_eq!(r.pages(BLOCK, &[], 0), RGFX_STATUS_REFUSED);
        assert_eq!(r.pages(BLOCK, &[(1, 1)], 2 * page), RGFX_STATUS_REFUSED);
        assert_eq!(r.pages(BLOCK, &[(100, 1)], page), RGFX_STATUS_REFUSED);
        let runs = [RgfxPageRun {
            address: RAM + 0x800,
            length: page,
        }];
        let m = RgfxSurfacePages {
            header: header(RGFX_CMD_SURFACE_PAGES),
            run_count: 1,
            block_offset: BLOCK,
            block_size: 0x2000,
            runs_offset: BLOCK + 0x1000,
            length: page,
            ..RgfxSurfacePages::default()
        };
        assert_eq!(
            r.dec.surface_pages(&r.device, &m, &runs, 0),
            RGFX_STATUS_REFUSED
        );
        let c = r.dec.views_of_surfaces().counts();
        assert_eq!((c.made, c.bad_runs, c.not_mapped), (0, 3, 1));
        assert_eq!(r.buffer(10, BLOCK, page), Err(DecodeError::Refused));
        // A good one after them has its view.
        assert_eq!(r.pages(BLOCK, &[(1, 1)], page), RGFX_STATUS_OK);
        assert_eq!(r.buffer(10, BLOCK, page), Ok(()));
    }

    /// What may be the last hold of a view's import or mapping goes on the surface worker: a
    /// buffer over the view at its `OBJECT_DESTROY`, the block's own hold at its `BLOCK_RELEASE`,
    /// and, as the worker makes the views, the cache's as it leaves the cache
    #[test]
    fn a_view_s_last_holds_go_on_the_worker() {
        let mut r = rig(64);
        let page = r.page;
        assert_eq!(r.pages(BLOCK, &[(1, 1)], page), RGFX_STATUS_OK);
        let view = r.view(BLOCK);
        assert_eq!(r.buffer(10, BLOCK, page), Ok(()));
        let held = Arc::strong_count(&view);
        r.dec.surface_worker().hold();
        let destroy = RgfxObjectDestroy {
            header: header(RGFX_CMD_OBJECT_DESTROY),
            obj_id: 10,
        };
        assert_eq!(r.dec.decode(&r.device, destroy.as_bytes()).unwrap(), Ok(()));
        assert!(
            !r.dec.span_in_use(BLOCK, 0x1000),
            "the object is gone at once"
        );
        r.dec
            .block_released(&r.device, BLOCK, 0, Box::new(|| ()), no_raise());
        assert_eq!(
            Arc::strong_count(&view),
            held,
            "its holds wait for the worker"
        );
        r.dec.surface_worker().release();
        r.dec.surface_worker().flush();
        assert_eq!(
            Arc::strong_count(&view),
            held - 2,
            "the buffer's and the block's holds went there"
        );
        for i in 0..CACHED_VIEWS as u64 {
            assert_eq!(
                r.pages(BLOCK + 0x8000, &[(10 + i, 1)], page),
                RGFX_STATUS_OK
            );
        }
        assert_eq!(Arc::strong_count(&view), 1, "the cache let it go");
    }

    /// A device's hold of a view's memory keeps the view's mapping past the view: the view's
    /// identity ends as the view goes, which tells a device to let go of what it made of the
    /// memory (a CUDA registration of its pages), and the mapping is torn down only once the
    /// device's hold goes too, after the device let go, so the pages a registration pinned are
    /// still mapped when it is undone.
    #[test]
    fn a_view_s_mapping_outlives_it_while_a_device_holds_its_memory() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let mut r = rig(64);
        let page = r.page;
        assert_eq!(r.pages(BLOCK, &[(1, 1)], page), RGFX_STATUS_OK);
        let view = r.view(BLOCK);
        let at = view.base().addr();
        let memory = view.memory().expect("a view's memory");
        let identity = memory.identity();
        let gone = Arc::new(AtomicBool::new(false));
        identity.when_gone({
            let gone = Arc::clone(&gone);
            move || gone.store(true, Ordering::SeqCst)
        });
        let host = memory.host_view().expect("the view's host view");
        assert_eq!(
            (host.first.as_ptr().addr(), host.size, host.offset),
            (at, page.to_usize(), 0)
        );
        assert_eq!(memory.length(), page.to_usize());
        drop((view, memory));
        r.dec
            .block_released(&r.device, BLOCK, 0, Box::new(|| ()), no_raise());
        for i in 0..CACHED_VIEWS as u64 {
            assert_eq!(
                r.pages(BLOCK + 0x8000, &[(10 + i, 1)], page),
                RGFX_STATUS_OK
            );
        }
        r.dec.surface_worker().flush();
        assert!(
            gone.load(Ordering::SeqCst),
            "the view went, its identity with it"
        );
        assert!(!identity.is_live());
        let torn_down = || {
            (r.p.call_log().calls_to("mach_vm_deallocate").iter())
                .filter(|c| c.args.first() == Some(&Arg::Ptr(at)))
                .count()
        };
        assert_eq!(torn_down(), 0, "the device's hold keeps the mapping");
        drop(host);
        r.dec.surface_worker().flush();
        assert_eq!(torn_down(), 1, "torn down on the worker once the hold went");
    }

    /// A destroy that gives back a name the guest picked lets go of its object at once, though the
    /// block it lies over has a view and the surface worker is held: a texture over the view's
    /// buffer (its resource id) and a buffer over the block's host memory (its GPU address), which
    /// the guest's next create may take at once. The view's buffer waits for the worker
    #[test]
    fn a_destroy_that_gives_back_a_guest_s_name_waits_for_no_worker() {
        let mut r = rig(64);
        let page = r.page;
        let world = Arc::clone(r.p.world());
        r.dec.hold_resource_ids(0, 1);
        assert_eq!(r.pages(BLOCK, &[(1, 1)], page), RGFX_STATUS_OK);
        assert_eq!(r.buffer(10, BLOCK, page), Ok(()));
        r.dec.surface_worker().hold();
        let send = |r: &mut Rig, bytes: &[u8]| r.dec.decode(&r.device, bytes).unwrap();
        let destroy = |obj_id| RgfxObjectDestroy {
            header: header(RGFX_CMD_OBJECT_DESTROY),
            obj_id,
        };
        let texture = |obj_id| RgfxTextureFromBuffer {
            header: header(RGFX_CMD_TEXTURE_FROM_BUFFER),
            obj_id,
            resource_id: 0x41,
            buffer_obj_id: 10,
            offset: 0,
            bytes_per_row: 64 * 4,
            desc: RgfxTextureDescriptor {
                pixel_format: 80,
                texture_type: 2,
                width: 64,
                height: 16,
                depth: 1,
                mipmap_level_count: 1,
                array_length: 1,
                sample_count: 1,
                usage: 1,
                ..RgfxTextureDescriptor::default()
            },
        };
        const PIN: u64 = 0x1_0000_0000;
        let host_memory = |obj_id| RgfxBufferOverHostMemory {
            header: header(RGFX_CMD_BUFFER_OVER_HOST_MEMORY),
            obj_id,
            gpu_address: PIN,
            region_offset: BLOCK,
            length: page,
            ..RgfxBufferOverHostMemory::default()
        };
        let textures = world.live("MTLTexture").len();
        assert_eq!(send(&mut r, texture(11).as_bytes()), Ok(()));
        assert_eq!(world.live("MTLTexture").len(), textures + 1);
        assert_eq!(send(&mut r, destroy(11).as_bytes()), Ok(()));
        assert_eq!(
            world.live("MTLTexture").len(),
            textures,
            "the texture went at once"
        );
        assert_eq!(
            send(&mut r, texture(12).as_bytes()),
            Ok(()),
            "at the same resource id"
        );
        let buffers = world.live("MTLBuffer").len();
        assert_eq!(send(&mut r, host_memory(13).as_bytes()), Ok(()));
        assert_eq!(world.live("MTLBuffer").len(), buffers + 1);
        assert_eq!(send(&mut r, destroy(13).as_bytes()), Ok(()));
        assert_eq!(
            world.live("MTLBuffer").len(),
            buffers,
            "the buffer went at once"
        );
        assert_eq!(
            send(&mut r, host_memory(14).as_bytes()),
            Ok(()),
            "at the same address"
        );
        assert_eq!(send(&mut r, destroy(12).as_bytes()), Ok(()));
        let buffers = world.live("MTLBuffer").len();
        assert_eq!(send(&mut r, destroy(10).as_bytes()), Ok(()));
        assert_eq!(
            world.live("MTLBuffer").len(),
            buffers,
            "the view's buffer waits"
        );
        r.dec.surface_worker().release();
        r.dec.surface_worker().flush();
        assert_eq!(world.live("MTLBuffer").len(), buffers - 1);
    }

    /// A view's mapping goes on the surface worker wherever its last hold goes: a texture over a
    /// surface's buffer and a pages block's buffer, each the last hold of its view once the
    /// buffer, the blocks and the cache let theirs go, are let go of at once at their
    /// `OBJECT_DESTROY`, on ring 0's thread, and the views' mappings wait for the worker
    #[test]
    fn a_view_s_mapping_goes_on_the_worker_from_a_destroy_on_ring_0() {
        let mut r = rig(64);
        let page = r.page;
        let world = Arc::clone(r.p.world());
        r.dec.hold_resource_ids(0, 1);
        let send = |r: &mut Rig, bytes: &[u8]| r.dec.decode(&r.device, bytes).unwrap();
        let destroy = |obj_id| RgfxObjectDestroy {
            header: header(RGFX_CMD_OBJECT_DESTROY),
            obj_id,
        };
        // A surface's view, a buffer over it and a texture over the buffer.
        assert_eq!(r.pages(BLOCK, &[(1, 1)], page), RGFX_STATUS_OK);
        let surface = Arc::downgrade(&r.view(BLOCK));
        assert_eq!(r.buffer(10, BLOCK, page), Ok(()));
        let texture = RgfxTextureFromBuffer {
            header: header(RGFX_CMD_TEXTURE_FROM_BUFFER),
            obj_id: 11,
            resource_id: 0x41,
            buffer_obj_id: 10,
            offset: 0,
            bytes_per_row: 64 * 4,
            desc: RgfxTextureDescriptor {
                pixel_format: 80,
                texture_type: 2,
                width: 64,
                height: 16,
                depth: 1,
                mipmap_level_count: 1,
                array_length: 1,
                sample_count: 1,
                usage: 1,
                ..RgfxTextureDescriptor::default()
            },
        };
        assert_eq!(send(&mut r, texture.as_bytes()), Ok(()));
        // A pages block's view, which its buffer made.
        const PAGES: u64 = BLOCK + 0x8000;
        assert_eq!(
            r.pages_flagged(PAGES, 0, &[(3, 1)], page, RGFX_SURFACE_PAGES_PRIVATE),
            RGFX_STATUS_OK
        );
        assert_eq!(r.pages_buffer(20, 0x1_0000_0000, PAGES, 0, page), Ok(()));
        let private = Arc::downgrade(&r.private_view(PAGES));
        // Every other hold goes: the surface's buffer, both blocks, and the cache's.
        assert_eq!(send(&mut r, destroy(10).as_bytes()), Ok(()));
        for block in [BLOCK, PAGES] {
            r.dec
                .block_released(&r.device, block, 0, Box::new(|| ()), no_raise());
        }
        for i in 0..CACHED_VIEWS as u64 {
            assert_eq!(
                r.pages(BLOCK + 0x10000, &[(10 + i, 1)], page),
                RGFX_STATUS_OK
            );
        }
        r.dec.surface_worker().flush();
        let bases = [&surface, &private].map(|v| v.upgrade().expect("the view lives").base());
        assert_eq!(
            surface.strong_count(),
            1,
            "the texture's buffer alone holds it"
        );
        assert_eq!(
            private.strong_count(),
            1,
            "the pages block's buffer alone holds it"
        );
        let unmaps = |r: &Rig, mark| -> Vec<usize> {
            (r.p.call_log().since(mark).iter())
                .filter(|c| c.sel == "mach_vm_deallocate")
                .filter_map(|c| match c.args[0] {
                    Arg::Ptr(at) => Some(at),
                    _ => None,
                })
                .filter(|&at| bases.iter().any(|b| b.addr() == at))
                .collect()
        };
        r.dec.surface_worker().hold();
        let mark = r.p.call_log().mark();
        let (textures, buffers) = (
            world.live("MTLTexture").len(),
            world.live("MTLBuffer").len(),
        );
        assert_eq!(send(&mut r, destroy(11).as_bytes()), Ok(()));
        assert_eq!(send(&mut r, destroy(20).as_bytes()), Ok(()));
        assert_eq!(
            world.live("MTLTexture").len(),
            textures - 1,
            "the texture went at once"
        );
        assert_eq!(
            world.live("MTLBuffer").len(),
            buffers - 2,
            "so did both buffers"
        );
        assert_eq!((surface.strong_count(), private.strong_count()), (0, 0));
        assert!(
            unmaps(&r, mark).is_empty(),
            "the mappings wait for the worker"
        );
        r.dec.surface_worker().release();
        r.dec.surface_worker().flush();
        let mut unmapped = unmaps(&r, mark);
        unmapped.sort_unstable();
        let mut want = bases.map(|b| b.addr());
        want.sort_unstable();
        assert_eq!(unmapped, want, "each view's mapping went on the worker");
    }

    /// A block named again, or released and its offset given to another surface, lies over its
    /// new runs' view from then on, never over the view of the pages it named before; a buffer
    /// made before keeps the view it was made over
    #[test]
    fn a_block_named_again_never_lies_over_its_old_view() {
        let mut r = rig(16);
        let page = r.page;
        assert!(r.ram.write(RAM + page, b"old"));
        assert!(r.ram.write(RAM + 2 * page, b"new"));
        let p = Arc::clone(&r.p);
        let fake = p.fake_vm();
        assert_eq!(r.pages(BLOCK, &[(1, 1)], page), RGFX_STATUS_OK);
        let old = r.view(BLOCK);
        assert_eq!(r.buffer(10, BLOCK, page), Ok(()));
        // Named again, over other pages.
        assert_eq!(r.pages(BLOCK, &[(2, 1)], page), RGFX_STATUS_OK);
        let new = r.view(BLOCK);
        assert!(!Arc::ptr_eq(&old, &new));
        assert_eq!(fake.read(new.base(), 3).unwrap(), b"new");
        assert_eq!(
            fake.read(old.base(), 3).unwrap(),
            b"old",
            "buffer 10 keeps its view"
        );
        // Refused: the block keeps no view, not its last.
        assert_eq!(r.pages(BLOCK, &[(100, 1)], page), RGFX_STATUS_REFUSED);
        assert_eq!(r.buffer(11, BLOCK, page), Err(DecodeError::Refused));
        // Released, and given to a surface of other pages.
        let destroy = RgfxObjectDestroy {
            header: header(RGFX_CMD_OBJECT_DESTROY),
            obj_id: 10,
        };
        assert_eq!(r.dec.decode(&r.device, destroy.as_bytes()).unwrap(), Ok(()));
        r.dec
            .block_released(&r.device, BLOCK, 0, Box::new(|| ()), no_raise());
        assert_eq!(r.buffer(12, BLOCK, page), Err(DecodeError::Refused));
        assert_eq!(r.pages(BLOCK, &[(3, 1)], page), RGFX_STATUS_OK);
        let next = r.view(BLOCK);
        assert!(!Arc::ptr_eq(&next, &old) && !Arc::ptr_eq(&next, &new));
        assert_eq!(r.buffer(13, BLOCK, page), Ok(()));
    }

    /// A pages block's runs are one view, its own, made at its first `BUFFER_OVER_PAGES` and kept
    /// for the next: no cache entry is made or taken for it, and a `BUFFER_OVER_PAGES` lies over
    /// the bytes at its offset in the view, at the guest's GPU
    /// address, over the app's own page, and over the block, so the block is not released while
    /// it lives. A range past the view, an empty one, and a `BUFFER_OVER_SURFACE_PAGES` of the
    /// block are refused; after the block's release the buffer still holds the view, and the
    /// block takes no new buffer
    #[test]
    fn a_pages_block_s_buffers_lie_over_its_view_at_their_offsets() {
        let mut r = rig(8);
        let page = r.page;
        // A surface's view in the cache, over page 5 as the pages block's.
        assert_eq!(r.pages(BLOCK + 0x8000, &[(5, 1)], page), RGFX_STATUS_OK);
        assert_eq!(r.dec.views_of_surfaces().len(), (1, 1));
        // The app's bytes, 0x123 into page 5, the first of the block's.
        assert!(r.ram.write(RAM + 5 * page + 0x123, b"app"));
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.pages_flagged(
                BLOCK,
                0,
                &[(5, 2), (1, 1)],
                3 * page,
                RGFX_SURFACE_PAGES_PRIVATE
            ),
            RGFX_STATUS_OK
        );
        assert_eq!(
            r.p.call_log().sels_since(mark),
            Vec::<&str>::new(),
            "no view or import before a buffer is laid over the block"
        );
        assert!(!r.has_private_view(BLOCK));
        let views = r.dec.views_of_surfaces();
        assert_eq!((views.len(), views.private_len()), ((1, 1), 1));
        let c = views.counts();
        assert_eq!((c.made, c.reused, c.private), (1, 0, 0));
        drop(views);

        let pin = 0x1_0000_0123;
        let mark = r.p.call_log().mark();
        assert_eq!(r.pages_buffer(10, pin, BLOCK, 0x123, 2 * page), Ok(()));
        assert_eq!(
            r.p.call_log().sels_since(mark),
            [
                "mach_vm_allocate",
                "mach_vm_remap",
                "mach_vm_remap",
                "mach_vm_protect",
                "newBufferWithDescriptor:",
                "newBufferWithDescriptor:"
            ],
            "the first buffer: a view of the block's two runs and its import, made anew, not the \
             cache's of page 5, then the buffer"
        );
        assert_eq!(r.dec.views_of_surfaces().counts().private, 1);
        let view = r.private_view(BLOCK);
        assert_eq!(view.length(), 3 * page);
        let (gpu_address, contents) = r.buffer_at(10);
        assert_eq!(gpu_address, pin);
        assert_eq!(contents, view.base().wrapping_add(0x123));
        assert_eq!(r.p.fake_vm().read(contents, 3).unwrap(), b"app");
        assert!(r.dec.span_in_use(BLOCK, 0x2000));
        // Its last byte the view's: taken, over the view the first made; one past it, or none,
        // refused.
        let mark = r.p.call_log().mark();
        assert_eq!(r.pages_buffer(11, 0, BLOCK, page, 2 * page), Ok(()));
        assert_eq!(
            r.p.call_log().sels_since(mark),
            ["newBufferWithDescriptor:"]
        );
        assert_eq!(r.dec.views_of_surfaces().counts().private, 1);
        assert_eq!(
            r.pages_buffer(12, 0, BLOCK, page, 2 * page + 1),
            Err(DecodeError::Refused)
        );
        assert_eq!(
            r.pages_buffer(12, 0, BLOCK, u64::MAX, 2),
            Err(DecodeError::Refused)
        );
        assert_eq!(
            r.pages_buffer(12, 0, BLOCK, 0, 0),
            Err(DecodeError::Refused)
        );
        assert_eq!(r.buffer(12, BLOCK, page), Err(DecodeError::Refused));

        r.dec
            .block_released(&r.device, BLOCK, 0, Box::new(|| ()), no_raise());
        assert_eq!(r.dec.views_of_surfaces().private_len(), 0);
        assert_eq!(
            r.pages_buffer(13, 0, BLOCK, 0, page),
            Err(DecodeError::Refused)
        );
        assert!(Arc::strong_count(&view) > 1, "the buffers hold it");
        for obj_id in [10, 11] {
            let destroy = RgfxObjectDestroy {
                header: header(RGFX_CMD_OBJECT_DESTROY),
                obj_id,
            };
            assert_eq!(r.dec.decode(&r.device, destroy.as_bytes()).unwrap(), Ok(()));
        }
        assert!(!r.dec.span_in_use(BLOCK, 0x2000));
        // The block's hold of it goes on the surface worker.
        r.dec.surface_worker().flush();
        assert_eq!(Arc::strong_count(&view), 1, "only the test holds the view");
    }

    /// A `BUFFER_OVER_PAGES` is the sender's own pages block's alone: a block the sender does not
    /// hold is a bad reference, and a surface's block, a block with no view, or a pages block
    /// made for another context, refused
    #[test]
    fn a_pages_buffer_takes_only_the_sender_s_own_pages_block() {
        let mut r = rig(8);
        let page = r.page;
        let (two, three) = (BLOCK, BLOCK + 0x4000);
        r.dec.region.grant(2, two, 0x2000);
        r.dec.region.grant(3, three, 0x2000);
        let private = RGFX_SURFACE_PAGES_PRIVATE;
        assert_eq!(
            r.pages_flagged(two, 2, &[(1, 1)], page, private),
            RGFX_STATUS_OK
        );
        // Context 2's pages block lies in context 3's block: the kext never makes one so.
        assert_eq!(
            r.pages_flagged(three, 2, &[(2, 1)], page, private),
            RGFX_STATUS_OK
        );
        let obj = |context: u64, n: u64| (context << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;

        r.dec.set_context(Some(3));
        assert_eq!(
            r.pages_buffer(obj(3, 1), 0, two, 0, page),
            Err(DecodeError::BadReference),
            "context 2's block"
        );
        assert_eq!(
            r.pages_buffer(obj(3, 1), 0, three, 0, page),
            Err(DecodeError::Refused),
            "a pages block made for context 2"
        );
        r.dec.set_context(Some(2));
        assert_eq!(r.pages_buffer(obj(2, 1), 0, two, 0, page), Ok(()));
        // A surface's block, and one with no view.
        assert_eq!(r.pages_cached(two, &[(3, 1)], page, true), RGFX_STATUS_OK);
        assert_eq!(
            r.pages_buffer(obj(2, 2), 0, two, 0, page),
            Err(DecodeError::Refused)
        );
        r.dec.region.grant(2, BLOCK + 0x8000, 0x2000);
        assert_eq!(
            r.pages_buffer(obj(2, 2), 0, BLOCK + 0x8000, 0, page),
            Err(DecodeError::Refused)
        );
    }

    /// A pages block of one run holds a heap over the guest's RAM itself, not over its view, at
    /// the guest's address, rounded up to Metal's page: a buffer placed in it lies at the heap's
    /// address plus its offset, and the block is in use while either lives. A surface's block,
    /// one with no view, a heap past the run or of no bytes, and another context's block are
    /// refused; a block outside the processes' is a bad reference
    #[test]
    fn a_pages_block_of_one_run_holds_a_heap_over_the_guest_s_ram() {
        let mut r = rig(16);
        let page = r.page;
        let private = RGFX_SURFACE_PAGES_PRIVATE;
        assert_eq!(
            r.pages_flagged(BLOCK, 0, &[(2, 4)], 4 * page, private),
            RGFX_STATUS_OK
        );
        let mark = r.p.call_log().mark();
        assert_eq!(r.pages_heap(20, BLOCK, 4 * page - 0x100), Ok(()));
        assert_eq!(
            r.p.call_log().sels_since(mark),
            ["newHeapWithDescriptor:"],
            "one heap, with no mapping made"
        );
        let calls = r.p.call_log().since(mark);
        let first = r.ram_base().wrapping_add(2 * page);
        assert_eq!(
            calls[0].args[..2],
            [Arg::Ptr(first.addr()), Arg::U64(4 * page)],
            "over the run's pages where the guest's RAM is mapped, whole pages of Metal's"
        );
        assert!(r.dec.span_in_use(BLOCK, 0x2000));
        let at = RgfxBufferAtOffset {
            header: header(RGFX_CMD_BUFFER_AT_OFFSET),
            obj_id: 21,
            heap_obj_id: 20,
            offset: page,
            length: 0x100,
            options: 0,
            reserved_34: 0,
        };
        assert_eq!(r.dec.decode(&r.device, at.as_bytes()).unwrap(), Ok(()));
        let (gpu_address, _) = r.buffer_at(21);
        assert_eq!(gpu_address, 0x2_0000_0000 + 20 * 0x100_0000 + page);

        // Refusals.
        assert_eq!(r.pages(BLOCK + 0x8000, &[(12, 1)], page), RGFX_STATUS_OK);
        assert_eq!(
            r.pages_heap(22, BLOCK + 0x8000, page),
            Err(DecodeError::Refused),
            "a surface's block"
        );
        assert_eq!(
            r.pages_heap(22, BLOCK + 0xc000, page),
            Err(DecodeError::Refused),
            "no view"
        );
        assert_eq!(
            r.pages_heap(22, BLOCK, 4 * page + 1),
            Err(DecodeError::Refused),
            "past the run"
        );
        assert_eq!(r.pages_heap(22, BLOCK, 0), Err(DecodeError::Refused));
        assert_eq!(r.pages_heap(22, 0, page), Err(DecodeError::BadReference));
        r.dec.region.grant(3, BLOCK, 0x2000);
        r.dec.set_context(Some(3));
        let obj = (3 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 1;
        assert_eq!(
            r.pages_heap(obj, BLOCK, page),
            Err(DecodeError::Refused),
            "a pages block made for context 0"
        );
        r.dec.set_context(None);

        // The block is in use while the buffer or the heap lives.
        for obj_id in [20, 21] {
            assert!(r.dec.span_in_use(BLOCK, 0x2000));
            let destroy = RgfxObjectDestroy {
                header: header(RGFX_CMD_OBJECT_DESTROY),
                obj_id,
            };
            assert_eq!(r.dec.decode(&r.device, destroy.as_bytes()).unwrap(), Ok(()));
        }
        assert!(!r.dec.span_in_use(BLOCK, 0x2000));
    }

    /// A pages block of several runs, a slab of plain pages the guest took where its kernel had no
    /// superpage left, holds a heap over its view, made by the heap where no buffer made it
    /// before: the heap's bytes are the view's, whose pages are the guest's runs in their order, a
    /// buffer placed in it lies at the heap's address plus its offset, and the heap records the
    /// view's bytes for a CPU copy placed in it. The block is in use while the heap lives, so its
    /// release, which lets go of the view, is refused until the heap's destroy; a heap past the
    /// view is refused
    #[test]
    fn a_pages_block_of_several_runs_holds_a_heap_over_its_view() {
        let mut r = rig(16);
        let page = r.page;
        let private = RGFX_SURFACE_PAGES_PRIVATE;
        assert_eq!(
            r.pages_flagged(BLOCK, 0, &[(8, 1), (3, 2), (12, 1)], 4 * page, private),
            RGFX_STATUS_OK
        );
        assert!(!r.has_private_view(BLOCK));
        assert_eq!(
            r.pages_heap(20, BLOCK, 4 * page + 1),
            Err(DecodeError::Refused),
            "past the view"
        );
        let mark = r.p.call_log().mark();
        assert_eq!(r.pages_heap(20, BLOCK, 4 * page - 0x100), Ok(()));
        let view = r.private_view(BLOCK);
        assert_eq!(view.length(), 4 * page);
        let heap = r.p.call_log().since(mark);
        let heap = heap
            .iter()
            .rfind(|c| c.sel == "newHeapWithDescriptor:")
            .expect("a heap");
        assert_eq!(
            heap.args[..2],
            [Arg::Ptr(view.base().addr()), Arg::U64(4 * page)],
            "over the view, whole pages of Metal's"
        );
        assert_eq!(r.dec.ram_of(20), Some((view.base(), 4 * page)));
        // The view maps the runs in their order: its second page is guest page 3, and what the
        // GPU writes there reaches it.
        assert!(r.ram.write(RAM + 3 * page, b"third"));
        let fake = r.p.fake_vm();
        assert_eq!(
            fake.read(view.base().wrapping_add(page), 5).unwrap(),
            b"third"
        );
        assert!(fake.write(view.base().wrapping_add(3 * page), b"GPU"));
        fake.sync();
        let mut got = [0u8; 3];
        assert!(r.ram.read(RAM + 12 * page, &mut got));
        assert_eq!(&got, b"GPU");
        let at = RgfxBufferAtOffset {
            header: header(RGFX_CMD_BUFFER_AT_OFFSET),
            obj_id: 21,
            heap_obj_id: 20,
            offset: page,
            length: 0x100,
            options: 0,
            reserved_34: 0,
        };
        assert_eq!(r.dec.decode(&r.device, at.as_bytes()).unwrap(), Ok(()));
        let (gpu_address, _) = r.buffer_at(21);
        assert_eq!(gpu_address, 0x2_0000_0000 + 20 * 0x100_0000 + page);
        // A second heap over the same block takes the view made.
        let mark = r.p.call_log().mark();
        assert_eq!(r.pages_heap(22, BLOCK, page), Ok(()));
        assert_eq!(
            r.p.call_log().sels_since(mark),
            ["newHeapWithDescriptor:"],
            "no mapping made again"
        );
        assert!(Arc::ptr_eq(&r.private_view(BLOCK), &view));
        // The block is in use, its view kept, while a heap or the buffer lives.
        for obj_id in [21, 22, 20] {
            assert!(r.dec.span_in_use(BLOCK, 0x2000));
            let destroy = RgfxObjectDestroy {
                header: header(RGFX_CMD_OBJECT_DESTROY),
                obj_id,
            };
            assert_eq!(r.dec.decode(&r.device, destroy.as_bytes()).unwrap(), Ok(()));
        }
        assert!(!r.dec.span_in_use(BLOCK, 0x2000));
        assert!(r.has_private_view(BLOCK));
    }

    /// A pages block's runs are checked as a surface's: runs outside guest RAM, not whole pages,
    /// or not summing to its length make no view, are refused and counted, and a block flagged
    /// both private and cached is a pages block, with no device cache
    #[test]
    fn bad_pages_block_runs_make_no_view() {
        let mut r = rig(8);
        let page = r.page;
        let private = RGFX_SURFACE_PAGES_PRIVATE;
        assert_eq!(
            r.pages_flagged(BLOCK, 0, &[], 0, private),
            RGFX_STATUS_REFUSED
        );
        assert_eq!(
            r.pages_flagged(BLOCK, 0, &[(1, 1)], 2 * page, private),
            RGFX_STATUS_REFUSED
        );
        assert_eq!(
            r.pages_flagged(BLOCK, 0, &[(100, 1)], page, private),
            RGFX_STATUS_REFUSED,
            "outside guest RAM"
        );
        let c = r.dec.views_of_surfaces().counts();
        assert_eq!((c.private, c.bad_runs, c.not_mapped), (0, 2, 1));
        assert_eq!(r.dec.views_of_surfaces().private_len(), 0);
        assert_eq!(
            r.pages_buffer(10, 0, BLOCK, 0, page),
            Err(DecodeError::Refused)
        );
        let both = private | RGFX_SURFACE_PAGES_CACHED;
        assert_eq!(
            r.pages_flagged(BLOCK, 0, &[(1, 1)], page, both),
            RGFX_STATUS_OK
        );
        assert!(r.dec.caches().buffer(BLOCK).is_none());
        assert_eq!(r.pages_buffer(10, 0, BLOCK, 0, page), Ok(()));
    }

    /// At its context's close a pages block's view goes, as the kext frees the block with no
    /// release; another context's stays
    #[test]
    fn a_closed_context_s_pages_views_go() {
        let mut r = rig(8);
        let page = r.page;
        let private = RGFX_SURFACE_PAGES_PRIVATE;
        assert_eq!(
            r.pages_flagged(BLOCK, 2, &[(1, 1)], page, private),
            RGFX_STATUS_OK
        );
        assert_eq!(
            r.pages_flagged(BLOCK + 0x4000, 3, &[(2, 1)], page, private),
            RGFX_STATUS_OK
        );
        let obj = |context: u64, n: u64| (context << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
        assert_eq!(r.pages_buffer(obj(2, 1), 0, BLOCK, 0, page), Ok(()));
        assert_eq!(
            r.pages_buffer(obj(3, 1), 0, BLOCK + 0x4000, 0, page),
            Ok(())
        );
        let view = r.private_view(BLOCK);
        r.dec.drop_context(&r.device, 2);
        assert_eq!(r.dec.views_of_surfaces().private_len(), 1);
        assert_eq!(Arc::strong_count(&view), 1);
        let _ = r.private_view(BLOCK + 0x4000);
    }

    /// With no memory map the device has no `SURFACE_PAGES` and makes no view
    #[test]
    fn no_memory_map_makes_no_view() {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let size = BLOCK + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(Region::new(size as usize).unwrap());
        let dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), region);
        assert!(!dec.surface_pages_available());
        let m = RgfxSurfacePages {
            header: header(RGFX_CMD_SURFACE_PAGES),
            run_count: 1,
            block_offset: BLOCK,
            block_size: 0x2000,
            runs_offset: BLOCK + 0x1000,
            length: 0x4000,
            ..RgfxSurfacePages::default()
        };
        let runs = [RgfxPageRun {
            address: RAM,
            length: 0x4000,
        }];
        assert_eq!(
            dec.surface_pages(&device, &m, &runs, 0),
            RGFX_STATUS_REFUSED
        );
    }
}
