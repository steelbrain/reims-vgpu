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

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use reims_graphics::ops::{RGFX_SURFACE_USE_WRITES, RgfxOpSurfaceUse};
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue, CompletedInfo};
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::event::EventRef;
use reims_vgpu_core::platform::{Handle, Platform};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::sync::Mutex;

use super::surface_pages::{SurfaceView, buffer_over};
use crate::device_writes::{self, Watcher};

/// The copies run under a live raise handler (the ring loop's, or the control thread's entry).
const ON: OnFault = OnFault::Return;

/// `MTLResourceStorageModePrivate`, in `MTLResourceOptions`.
pub(crate) const STORAGE_PRIVATE: u64 = 2 << 4;
/// `MTLResourceStorageModeMask`.
pub(crate) const STORAGE_MASK: u64 = 0xf << 4;

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

    /// Whether [`Self::before`] would neither fill a cache `uses` names nor make the command
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

    /// The blocks `uses` names that have caches, before command buffer `cb` of queue `queue`
    /// runs: each filled where it holds an older generation than the use's, or none, or a device
    /// of the host wrote its pages since its last fill, and `cb` made to wait for each one's last
    /// fill not yet done; and whether it waits for one. A use that may write marks its cache
    /// written. Where not `may_fill` (the caller found them [`Self::ready`], with nothing changed
    /// since but the host's writes), nothing is filled: a write of the host's since `ready` is
    /// left for the next use's fill, as one made after this use.
    pub(crate) fn before(
        &mut self,
        device: &P::Device,
        cb: &P::CommandBuffer,
        uses: &[RgfxOpSurfaceUse],
        may_fill: bool,
    ) -> RaiseResult<(Vec<u64>, bool)> {
        let mut used = Vec::new();
        let mut waits = false;
        for u in uses {
            if !self.by_block.contains_key(&u.block_offset) || used.contains(&u.block_offset) {
                continue;
            }
            used.push(u.block_offset);
            let older = self.by_block[&u.block_offset]
                .generation
                .is_none_or(|held| u.cpu_gen > held);
            let host_wrote = may_fill && self.watched.take(u.block_offset);
            if may_fill && (older || host_wrote) {
                if !older {
                    self.counts.host_writes += 1;
                }
                self.fill(device, u.block_offset, u.cpu_gen)?;
            }
            let Some(cache) = self.by_block.get_mut(&u.block_offset) else {
                continue;
            };
            if u.flags & RGFX_SURFACE_USE_WRITES != 0 {
                cache.written = true;
            }
            if let Some(t) = &self.transfer
                && cache.filled_at > t.done.load(Ordering::Acquire)
            {
                cb.encode_wait(Some(EventRef::Event(&t.event)), cache.filled_at)?;
                waits = true;
            }
        }
        Ok((used, waits))
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
    /// `generation`, or a device of the host wrote its pages since its last fill, and waits for
    /// the fill: a present of the surface shows the pages as of that generation.
    pub(crate) fn fill_now(
        &mut self,
        device: &P::Device,
        block: u64,
        generation: u64,
    ) -> RaiseResult<()> {
        let Some(cache) = self.by_block.get(&block) else {
            return Ok(());
        };
        let older = cache.generation.is_none_or(|held| generation > held);
        let host_wrote = self.watched.take(block);
        if !older && !host_wrote {
            return Ok(());
        }
        if !older {
            self.counts.host_writes += 1;
        }
        self.fill(device, block, generation)?;
        let Some(t) = &self.transfer else {
            return Ok(());
        };
        // A command buffer behind the fill on the transfer queue finishes after it.
        if let Some(cb) = t.queue.command_buffer(ON)? {
            cb.encode_wait(Some(EventRef::Event(&t.event)), t.signalled)?;
            cb.commit(ON)?;
            cb.wait_until_completed(ON)?;
        }
        Ok(())
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

    /// How many blocks have a cache (for the tests).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.by_block.len()
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
        let cache = r.dec.surface_caches.buffer(BLOCK).unwrap();
        Arg::obj(cache).as_obj().unwrap().0
    }

    /// A block marked cached gets a device cache, which each of its buffers is; a block not
    /// marked, or any block of a GPU that shares the CPU's memory, gets none
    #[test]
    fn a_cached_blocks_buffers_are_its_device_cache() {
        let mut r = cached();
        let page = r.page;
        assert_eq!(r.dec.surface_caches.len(), 1);
        let mark = r.p.call_log().mark();
        assert_eq!(r.buffer(10, BLOCK, 2 * page), Ok(()));
        assert_eq!(r.buffer(11, BLOCK, page), Ok(()));
        assert!(
            r.p.call_log().sels_since(mark).is_empty(),
            "no buffer made: each is the cache"
        );
        assert!(r.dec.span_in_use(BLOCK, 0x1000));
        assert_eq!(r.pages(BLOCK + 0x8000, &[(3, 1)], page), RGFX_STATUS_OK);
        assert_eq!(r.dec.surface_caches.len(), 1, "a block not marked");
        r.dec.set_unified_memory(true);
        assert_eq!(
            r.pages_cached(BLOCK + 0x10000, &[(4, 1)], page, true),
            RGFX_STATUS_OK
        );
        assert_eq!(r.dec.surface_caches.len(), 0, "a GPU of the CPU's memory");
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
        assert_eq!(r.dec.surface_caches.counts().fills, 2);
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
        r.dec
            .surface_download(
                &r.device,
                BLOCK,
                Box::new(move || set.store(true, Ordering::SeqCst)),
            )
            .unwrap();
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
        r.dec
            .surface_download(
                &r.device,
                BLOCK,
                Box::new(move || set.store(true, Ordering::SeqCst)),
            )
            .unwrap();
        assert!(done.load(Ordering::SeqCst), "at once");
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        assert_eq!(r.dec.surface_caches.counts().clean, 1);
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 0, true)]);
        let released = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&released);
        let mark = r.p.call_log().mark();
        r.dec
            .block_released(
                &r.device,
                BLOCK,
                false,
                Box::new(move || set.store(true, Ordering::SeqCst)),
            )
            .unwrap();
        assert_eq!(copies(&r.p.call_log().since(mark)).len(), 1);
        assert_eq!(r.dec.surface_caches.len(), 0);
        assert!(!released.load(Ordering::SeqCst));
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(released.load(Ordering::SeqCst));
        assert_eq!(r.dec.surface_caches.counts().downloads, 1);
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
        r.dec
            .block_released(
                &r.device,
                BLOCK,
                true,
                Box::new(move || set.store(true, Ordering::SeqCst)),
            )
            .unwrap();
        assert!(released.load(Ordering::SeqCst), "done at once");
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        assert_eq!(r.dec.surface_caches.len(), 0);
        let counts = r.dec.surface_caches.counts();
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
        assert_eq!(r.dec.surface_caches.len(), 0, "not cached");
        r.p.world().complete_all(CompleteOn::Caller);
        let counts = r.dec.surface_caches.counts();
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
        let ready = |r: &Rig, cpu_gen| r.dec.surface_caches.ready(&reading(BLOCK, cpu_gen));
        assert!(!ready(&r, 0), "never filled");
        assert!(
            r.dec.surface_caches.ready(&reading(BLOCK + 0x8000, 0)),
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
        let (used, waits) = (r.dec.surface_caches)
            .before(&r.device, &cb, &reading(BLOCK, 1), false)
            .unwrap();
        assert_eq!((used, waits), (vec![BLOCK], false));
        assert!(r.p.call_log().sels_since(mark).is_empty(), "nothing filled");
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 1, false)]);
        assert_eq!(
            copies(&r.p.call_log().since(mark)).len(),
            1,
            "the next use fills"
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

        device_writes::wrote(&[
            first + page + 8..first + page + 16,
            first + 9 * page..u64::MAX,
        ]);
        let done = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&done);
        let mark = r.p.call_log().mark();
        r.dec
            .surface_download(
                &r.device,
                BLOCK,
                Box::new(move || set.store(true, Ordering::SeqCst)),
            )
            .unwrap();
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
        r.dec
            .block_released(
                &r.device,
                BLOCK,
                false,
                Box::new(move || set.store(true, Ordering::SeqCst)),
            )
            .unwrap();
        assert!(released.load(Ordering::SeqCst), "done at once");
        assert!(copies(&r.p.call_log().since(mark)).is_empty());
        let counts = r.dec.surface_caches.counts();
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
