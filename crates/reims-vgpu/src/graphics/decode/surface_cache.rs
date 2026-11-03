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
//!   order) fills nothing.
//! - **Download.** The pages are written from the cache only at a `SURFACE_DOWNLOAD` (a CPU lock
//!   after a GPU write), at the block's `BLOCK_RELEASE` and where the kext names the block's pages
//!   again (`SURFACE_PAGES`), and only where a buffer that may write the surface was committed
//!   since the cache was last copied either way.
//! - **Order.** Fills and downloads run on the host's own transfer queue, in the order they are
//!   asked, each after every command buffer that used the cache before it: each such buffer
//!   signals its queue's end event as it ends, and the copy waits for the latest value of each
//!   queue's. So a frame drawn from the cache before a CPU write still reads what it read.
//! - **The host's own writes.** A device of the host that writes a surface's pages itself (a
//!   video device's frame, [`crate::device_writes`]) moves no generation, so each cache's pages
//!   are watched: a write over them makes the cache's next use fill it whatever its generation,
//!   and a download or a release before that copies nothing, the pages being newer than the
//!   cache.
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
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use reims_graphics::ops::{RGFX_SURFACE_USE_WRITES, RgfxOpSurfaceUse};
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue, CompletedInfo};
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::event::EventRef;
use reims_vgpu_core::platform::{Handle, Platform};
use reims_vgpu_core::raise::{OnFault, Raise, RaiseResult, refusals};
use reims_vgpu_core::sync::{Condvar, Mutex, MutexGuard};

use super::refusing;
use super::surface_pages::{SurfaceView, buffer_over};
use crate::device_writes::{self, Watcher};

/// The copies run under a live raise handler (the ring loop's, or the control thread's entry).
const ON: OnFault = OnFault::Return;

/// `MTLResourceStorageModePrivate`, in `MTLResourceOptions`.
pub(crate) const STORAGE_PRIVATE: u64 = 2 << 4;
/// `MTLResourceStorageModeMask`.
pub(crate) const STORAGE_MASK: u64 = 0xf << 4;

/// How often [`SurfaceCaches::wait_filled`] looks at a fill.
const FILL_POLL: std::time::Duration = std::time::Duration::from_micros(50);

/// One surface's device cache.
struct SurfaceCache<P: Platform> {
    /// The cache: a Private buffer of the surface's bytes, which the block's buffers are.
    device: P::Buffer,
    /// A Shared buffer over the surface's own pages, the other end of each copy.
    pages: P::Buffer,
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
}

/// The host's transfer queue, on which every copy runs, and the event its fills signal.
struct Transfer<P: Platform> {
    queue: P::CommandQueue,
    event: P::Event,
    /// The last value a copy signals.
    signalled: u64,
    /// The highest value whose copy has finished.
    done: Arc<AtomicU64>,
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
    /// Releases of a written cache not copied, as no process could read the pages.
    pub(crate) unread: u64,
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
    /// Each command queue's end event, by the queue's id, and the last value signalled.
    ends: HashMap<u64, (P::Event, u64)>,
    /// The caches' pages, watched for the host's devices' writes while this lives.
    watched: Arc<Watched>,
    counts: CacheCounts,
    /// By block: the command buffers begun and not finished that use its cache ([`Caches`]).
    pending: HashMap<u64, Users>,
    /// By block: the fills of its cache waiting for those command buffers to finish.
    filling: HashMap<u64, usize>,
    /// By block: the copies out of its cache asked of the control thread (a download, a release)
    /// that wait for the command buffers that may write it begun before each was asked, in the
    /// order they were asked ([`Caches::after_writers`]).
    deferred: HashMap<u64, VecDeque<Deferred<P>>>,
    /// The number the next use that may write a cache is counted under.
    next_writer: u64,
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
    /// No caches; `on` keeps them where the kext asks.
    pub(crate) fn new(on: bool) -> Self {
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
            self.released(device, block, false, Box::new(|| {}))?;
        }
        let Some(cache) = device.new_buffer_of_length(ON, length, STORAGE_PRIVATE)? else {
            return Ok(false);
        };
        let Some(pages) = buffer_over(device, view, length, 0)? else {
            return Ok(false);
        };
        self.by_block.insert(
            block,
            SurfaceCache {
                device: cache,
                pages,
                length,
                generation: None,
                written: false,
                filled_at: 0,
                users: HashMap::new(),
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
    fn last_fill(&self, block: u64) -> Option<(u64, Arc<AtomicU64>)> {
        let (cache, t) = (self.by_block.get(&block)?, self.transfer.as_ref()?);
        Some((cache.filled_at, Arc::clone(&t.done)))
    }

    /// Whether [`Caches::before`] would neither fill a cache `uses` names nor make the command
    /// buffer wait for one's fill: each holds its use's generation or a newer one, no device of
    /// the host wrote its pages since its last fill, and its last fill is done.
    pub(crate) fn ready(&self, uses: &[RgfxOpSurfaceUse]) -> bool {
        uses.iter().all(|u| {
            let Some(cache) = self.by_block.get(&u.block_offset) else {
                return true;
            };
            let filled = self
                .transfer
                .as_ref()
                .is_none_or(|t| cache.filled_at <= t.done.load(Ordering::Acquire));
            cache.generation.is_some_and(|held| u.cpu_gen <= held)
                && !self.watched.written(u.block_offset)
                && filled
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
    /// where it needs it ([`Self::needs_fill`]) and `may_fill`, its fill marked written where `u`
    /// may write it, and `cb` made to wait for its last fill not yet done; whether it waits. The
    /// caller made sure no command buffer begun and not finished uses the cache where it fills
    /// ([`Caches::before`]).
    fn before_one(
        &mut self,
        device: &P::Device,
        cb: &P::CommandBuffer,
        u: &RgfxOpSurfaceUse,
        may_fill: bool,
    ) -> RaiseResult<bool> {
        let Some(cache) = self.by_block.get(&u.block_offset) else {
            return Ok(false);
        };
        let older = cache.generation.is_none_or(|held| u.cpu_gen > held);
        let host_wrote = may_fill && self.watched.take(u.block_offset);
        if may_fill && (older || host_wrote) {
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
        }
        if let Some(t) = &self.transfer
            && cache.filled_at > t.done.load(Ordering::Acquire)
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
        let Some(cb) = transfer.queue.command_buffer(ON)? else {
            return Ok(());
        };
        wait_for_users::<P>(&cb, &self.ends, cache)?;
        copy::<P>(&cb, &cache.pages, &cache.device, cache.length)?;
        transfer.signalled += 1;
        let value = transfer.signalled;
        cb.encode_signal(Some(EventRef::Event(&transfer.event)), value)?;
        let done = Arc::clone(&transfer.done);
        cb.add_completed_handler(
            ON,
            Box::new(move |_: CompletedInfo| {
                done.fetch_max(value, Ordering::AcqRel);
            }),
        )?;
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
    /// the surface shows the pages as of that generation. A command buffer on the transfer queue
    /// behind the fill, committed, which the caller waits for with no lock held; `None` where
    /// nothing was filled.
    fn fill_now(
        &mut self,
        device: &P::Device,
        block: u64,
        generation: u64,
    ) -> RaiseResult<Option<P::CommandBuffer>> {
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
        let Some(t) = &self.transfer else {
            return Ok(None);
        };
        // A command buffer behind the fill on the transfer queue finishes after it.
        let Some(cb) = t.queue.command_buffer(ON)? else {
            return Ok(None);
        };
        cb.encode_wait(Some(EventRef::Event(&t.event)), t.signalled)?;
        cb.commit(ON)?;
        Ok(Some(cb))
    }

    /// `SURFACE_DOWNLOAD` of the block at `block`: the pages written from its cache, after every
    /// command buffer that used it, where one that may write it was committed since its last copy
    /// and no device of the host wrote the pages since its last fill; `done` once the pages hold
    /// it, at once where nothing is to copy.
    pub(crate) fn download(
        &mut self,
        device: &P::Device,
        block: u64,
        done: Box<dyn FnOnce() + Send>,
    ) -> RaiseResult<()> {
        let written = self.written_since_host(block);
        if !written {
            if self.by_block.contains_key(&block) {
                self.counts.clean += 1;
            }
            done();
            return Ok(());
        }
        self.write_back(device, block, done)
    }

    /// The block at `block` went (its `BLOCK_RELEASE`): its cache goes once the pages hold what a
    /// buffer wrote of it, unless a device of the host wrote them since its last fill, or no
    /// process can read them (`unread`: its surface has no client left); `done` then, at once
    /// where nothing is to copy.
    pub(crate) fn released(
        &mut self,
        device: &P::Device,
        block: u64,
        unread: bool,
        done: Box<dyn FnOnce() + Send>,
    ) -> RaiseResult<()> {
        if !self.written_since_host(block) {
            done();
        } else if unread {
            self.counts.unread += 1;
            done();
        } else {
            self.write_back(device, block, done)?;
        }
        self.by_block.remove(&block);
        self.watched.0.lock().remove(&block);
        Ok(())
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

    /// Copies the cache of the block at `block` into the surface's pages on the transfer queue,
    /// after every command buffer that used it; `done` once the copy has finished. Where no copy
    /// can be made `done` is given at once.
    fn write_back(
        &mut self,
        device: &P::Device,
        block: u64,
        done: Box<dyn FnOnce() + Send>,
    ) -> RaiseResult<()> {
        if !self.make_transfer(device)? {
            done();
            return Ok(());
        }
        let (Some(transfer), Some(cache)) = (self.transfer.as_mut(), self.by_block.get(&block))
        else {
            done();
            return Ok(());
        };
        let Some(cb) = transfer.queue.command_buffer(ON)? else {
            done();
            return Ok(());
        };
        wait_for_users::<P>(&cb, &self.ends, cache)?;
        copy::<P>(&cb, &cache.device, &cache.pages, cache.length)?;
        // The buffers live until the copy is done, past the cache's own end at a release.
        let held = (cache.device.clone_ref(), cache.pages.clone_ref());
        cb.add_completed_handler(
            ON,
            Box::new(move |_: CompletedInfo| {
                drop(held);
                done();
            }),
        )?;
        cb.commit(ON)?;
        let length = cache.length;
        if let Some(cache) = self.by_block.get_mut(&block) {
            cache.written = false;
            cache.users.clear();
        }
        self.counts.downloads += 1;
        self.counts.downloaded_bytes += length;
        Ok(())
    }

    /// Makes the transfer queue at its first use; whether there is one.
    fn make_transfer(&mut self, device: &P::Device) -> RaiseResult<bool> {
        if self.transfer.is_none() {
            let Some(queue) = device.new_command_queue() else {
                return Ok(false);
            };
            let Some(event) = device.new_event(ON)? else {
                return Ok(false);
            };
            self.transfer = Some(Transfer {
                queue,
                event,
                signalled: 0,
                done: Arc::default(),
            });
        }
        Ok(true)
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
///   every one of them that may write it. A download and a release, which the control thread
///   asks for and are answered when their copy is done, do not wait there: each is made in the
///   finish of the last buffer that may write the cache begun before it was asked
///   ([`Caches::after_writers`]), in the order they were asked, and a command buffer begun after
///   it waits at its begin until it is made. The block's pages named again, which the buffers
///   over the block made on the control thread after it lie over, waits there.
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
}

/// The blocks of the caches a command buffer uses, and its count among their users
/// ([`Caches::hold_ready`]).
pub(crate) type Held<P> = (Vec<u64>, Option<Uses<P>>);

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
        Caches {
            state: Mutex::new(SurfaceCaches::new(on)),
            moved: Condvar::new(),
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

    /// Starts again with no caches; `on` keeps them where the kext asks.
    pub(crate) fn set_on(&self, on: bool) {
        *self.state.lock() = SurfaceCaches::new(on);
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
    /// cache written. Where not `may_fill` (the caller found them [`Self::hold_ready`]), nothing
    /// is filled.
    pub(crate) fn before(
        self: &Arc<Self>,
        device: &P::Device,
        cb: &P::CommandBuffer,
        uses: &[RgfxOpSurfaceUse],
        may_fill: bool,
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
                if may_fill && state.needs_fill(u) {
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
            let outcome = state.before_one(device, cb, u, may_fill);
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

    /// Whether [`Self::before`] would neither fill a cache `uses` names nor wait for anything, and
    /// so command buffer `cb`'s count among their users where so: taken without waiting for the
    /// lock (`None` where it is busy, or the buffer must wait). Each cache holds its use's
    /// generation or a newer one, no device of the host wrote its pages since its last fill, its
    /// last fill is done, and no fill of it, nor copy out of it, waits.
    pub(crate) fn hold_ready(self: &Arc<Self>, uses: &[RgfxOpSurfaceUse]) -> Option<Held<P>> {
        if uses.is_empty() {
            return Some((Vec::new(), None));
        }
        let mut state = self.state.try_lock()?;
        let picked: Vec<RgfxOpSurfaceUse> = by_block(&state, uses).into_iter().copied().collect();
        if !state.ready(&picked)
            || picked.iter().any(|u| {
                state.filling.contains_key(&u.block_offset) || state.deferring(u.block_offset)
            })
        {
            return None;
        }
        let blocks: Vec<(u64, bool)> = picked
            .iter()
            .map(|u| (u.block_offset, u.flags & RGFX_SURFACE_USE_WRITES != 0))
            .collect();
        for u in &picked {
            if u.flags & RGFX_SURFACE_USE_WRITES != 0
                && let Some(cache) = state.by_block.get_mut(&u.block_offset)
            {
                cache.written = true;
            }
        }
        let blocks = state.hold(&blocks);
        drop(state);
        let used = blocks.iter().map(|&(b, _)| b).collect();
        let held = Uses {
            caches: Arc::clone(self),
            blocks,
        };
        Some((used, (!held.blocks.is_empty()).then_some(held)))
    }

    /// Ends command buffer `cb` of queue `queue`, which `uses` counted among the users of the
    /// caches of `used` ([`SurfaceCaches::after`]), and `commit`s it, with the lock held, so a
    /// later copy finds its end event's value committed; then it leaves their users.
    pub(crate) fn finish(
        &self,
        device: &P::Device,
        queue: u64,
        cb: &P::CommandBuffer,
        mut uses: Uses<P>,
        used: &[u64],
        commit: impl FnOnce() -> RaiseResult<()>,
    ) -> RaiseResult<()> {
        let mut state = self.state.lock();
        let done = state.after(device, queue, cb, used).and_then(|()| commit());
        state.unhold(&uses.blocks);
        uses.blocks.clear();
        drop(state);
        self.moved.notify_all();
        done
    }

    /// Fills the cache of the block at `block` where a present of it at `generation` needs it
    /// ([`SurfaceCaches::fill_now`]), after every command buffer begun and not finished that uses
    /// it, and waits for the fill with no lock held.
    pub(crate) fn fill_now(
        &self,
        device: &P::Device,
        block: u64,
        generation: u64,
    ) -> RaiseResult<()> {
        let behind = {
            let mut state = self.state.lock();
            if !state.needs_fill_at(block, generation) {
                return Ok(());
            }
            *state.filling.entry(block).or_default() += 1;
            self.wait_users(&mut state, block, false);
            let filled = state.fill_now(device, block, generation);
            state.unfill(block);
            drop(state);
            self.moved.notify_all();
            filled?
        };
        match behind {
            Some(cb) => cb.wait_until_completed(ON),
            None => Ok(()),
        }
    }

    /// `SURFACE_DOWNLOAD` of the block at `block` ([`SurfaceCaches::download`]), after every
    /// command buffer begun and not finished that may write it, waiting for none
    /// ([`Self::after_writers`]): `done` once the pages hold the cache, or `failed` where the
    /// platform raised at the copy.
    pub(crate) fn download(
        &self,
        device: &P::Device,
        block: u64,
        done: Box<dyn FnOnce() + Send>,
        failed: Box<dyn FnOnce(Raise) + Send>,
    ) {
        let device = device.clone_ref();
        self.after_writers(
            block,
            Box::new(move |state| {
                if let Err(raise) = refusing(|| state.download(&device, block, done)) {
                    failed(raise);
                }
            }),
        );
    }

    /// The block at `block` went, its `BLOCK_RELEASE` ([`SurfaceCaches::released`]), after every
    /// command buffer begun and not finished that may write its cache, waiting for none
    /// ([`Self::after_writers`]): `done` once the pages hold the cache, or `failed` where the
    /// platform raised at the copy. Nothing names the block until it is answered: the kext gives
    /// it out again only then.
    pub(crate) fn release(
        &self,
        device: &P::Device,
        block: u64,
        unread: bool,
        done: Box<dyn FnOnce() + Send>,
        failed: Box<dyn FnOnce(Raise) + Send>,
    ) {
        let device = device.clone_ref();
        self.after_writers(
            block,
            Box::new(move |state| {
                if let Err(raise) = refusing(|| state.released(&device, block, unread, done)) {
                    failed(raise);
                }
            }),
        );
    }

    /// Runs `op` on the caches once every command buffer begun and not finished that may write
    /// the block at `block`'s cache has finished, after the copies out of it asked before: here
    /// where none is, else in the last one's finish, on its thread, waiting for nothing here. A
    /// command buffer begun meanwhile that uses the cache waits at its begin until `op` has run
    /// ([`Self::before`]), as one begun after the copy, which the control thread made before
    /// going on, waited for it.
    fn after_writers(&self, block: u64, op: CopyOut<P>) {
        let mut state = self.state.lock();
        state.after_writers(block, op);
        drop(state);
        self.moved.notify_all();
    }

    /// The block at `block` named again without a cache, or over other pages, loses it
    /// ([`SurfaceCaches::released`]), after every command buffer begun and not finished that may
    /// write it, waiting for them here: the buffers made over the block after it lie over its
    /// pages.
    pub(crate) fn released(
        &self,
        device: &P::Device,
        block: u64,
        unread: bool,
        done: Box<dyn FnOnce() + Send>,
    ) -> RaiseResult<()> {
        let mut state = self.state.lock();
        self.wait_users(&mut state, block, true);
        state.released(device, block, unread, done)
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

    /// Waits, on the calling thread, for the last fill of the block at `block`'s cache to be done,
    /// polling for about `timeout` at most (each poll sleeps at least [`FILL_POLL`]), with no lock
    /// held: for work the host starts outside a command buffer that writes the cache, which a
    /// command buffer's wait does not order. True where it is done, or the block has no cache or
    /// no fill.
    pub(crate) fn wait_filled(&self, block: u64, timeout: std::time::Duration) -> bool {
        let Some((filled_at, done)) = self.state.lock().last_fill(block) else {
            return true;
        };
        // The fill's completion handler raises `done`: a fill is a copy of the block, rare (a
        // cache's first use, or one after a CPU's write under a lock) and short, so the wait
        // polls it.
        let mut polls = timeout.as_micros() / FILL_POLL.as_micros();
        while filled_at > done.load(Ordering::Acquire) {
            if polls == 0 {
                return false;
            }
            polls -= 1;
            std::thread::sleep(FILL_POLL);
        }
        true
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
        RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_QUEUE_CREATE, RGFX_STATUS_OK, RgfxMsgHeader,
        RgfxQueueCreate,
    };
    use reims_vgpu_core::num::ToU64 as _;
    use reims_vgpu_core::testkit::calllog::{Arg, Call};
    use reims_vgpu_core::testkit::objects::CompleteOn;
    use zerocopy::IntoBytes;

    use super::super::no_raise;
    use super::super::surface_pages::tests::{BLOCK, RAM, Rig, rig};
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

    /// A release of a written cache whose surface no process can read (its `BLOCK_RELEASE`'s
    /// `RGFX_BLOCK_RELEASE_NO_CLIENT`) copies nothing and is done at once; the cache goes all the
    /// same
    #[test]
    fn an_unread_release_copies_nothing() {
        let mut r = cached();
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
        assert!(released.load(Ordering::SeqCst), "done at once");
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        assert_eq!(r.dec.caches().len(), 0);
        let counts = r.dec.caches().counts();
        assert_eq!((counts.downloads, counts.unread), (0, 1));
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

    /// A cache is ready for a buffer taken on a doorbell's thread, which fills none, once it is
    /// filled at its use's generation or a newer one, its fill is done, and no device of the
    /// host's wrote over its pages since; such a buffer's readying fills nothing and leaves a
    /// write since for the next use, which fills the cache again
    #[test]
    fn a_cache_is_ready_once_filled_at_its_use_s_generation_and_untouched() {
        let mut r = cached();
        let page = r.page;
        let first = r.ram.host_addr(RAM + page).unwrap().addr().to_u64();
        let ready = |r: &Rig, cpu_gen| r.dec.caches().ready(&reading(BLOCK, cpu_gen));
        assert!(!ready(&r, 0), "never filled");
        assert!(
            r.dec.caches().ready(&reading(BLOCK + 0x8000, 0)),
            "no cache"
        );
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 1, false)]);
        assert!(!ready(&r, 1), "its fill is not done");
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(ready(&r, 1) && ready(&r, 0));
        assert!(!ready(&r, 2), "a newer generation");

        device_writes::wrote(std::slice::from_ref(&(first..first + 8)));
        assert!(!ready(&r, 1), "the host wrote its pages");
        let queue = r.device.new_command_queue().unwrap();
        let cb = queue.command_buffer(ON).unwrap().unwrap();
        let mark = r.p.call_log().mark();
        let (used, waits, uses) = (r.dec.surface_caches)
            .before(&r.device, &cb, &reading(BLOCK, 1), false)
            .unwrap();
        assert_eq!((used, waits), (vec![BLOCK], false));
        assert!(r.p.call_log().sels_since(mark).is_empty(), "nothing filled");
        // The buffer is done with the cache: a fill no longer waits for it.
        drop(uses);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 1, false)]);
        assert_eq!(
            copies(&r.p.call_log().since(mark)).len(),
            1,
            "the next use fills"
        );
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
            .before(&r.device, &reader, &reading(BLOCK, 1), true)
            .unwrap();
        assert!(uses.is_some(), "counted among its users");
        let caches = Arc::clone(&r.dec.surface_caches);
        let filler = queue.command_buffer(ON).unwrap().unwrap();
        let mark = r.p.call_log().mark();
        std::thread::scope(|scope| {
            let fill = scope.spawn(|| {
                caches
                    .before(&r.device, &filler, &reading(BLOCK, 2), true)
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
    /// the cache waits for them at its begin, and a doorbell's take does not count it. Here the
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
            .before(&r.device, &writer, &writes, true)
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
        assert!(
            r.dec
                .surface_caches
                .hold_ready(&reading(BLOCK, 1))
                .is_none(),
            "a take counted behind the copies"
        );
        let caches = Arc::clone(&r.dec.surface_caches);
        let later = queue.command_buffer(ON).unwrap().unwrap();
        std::thread::scope(|scope| {
            let begin = scope.spawn(|| {
                (caches.before(&r.device, &later, &reading(BLOCK, 1), true))
                    .map(|(_, _, uses)| uses.is_some())
                    .map_err(|_| ())
            });
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(!begin.is_finished(), "begun before the copies were made");
            (r.dec.surface_caches)
                .finish(&r.device, 1, &writer, uses.unwrap(), &used, || {
                    writer.commit(ON)
                })
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

        device_writes::wrote(&[
            first + page + 8..first + page + 16,
            first + 9 * page..u64::MAX,
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

        device_writes::wrote(&[first..first + 1, first + 9 * page..u64::MAX]);
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
}
