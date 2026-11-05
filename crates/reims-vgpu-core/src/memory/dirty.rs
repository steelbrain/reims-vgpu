//! The dirty log: which 4 KiB pages of the guest's RAM were written since a stamp, for a GPU
//! runtime that keeps copies of guest bytes in memory of its own (a discrete GPU's VRAM, or an
//! iGPU whose driver cannot reference the guest's memory) and copies again only the pages that
//! changed.
//!
//! **The source.** The host publishes the pages the guest wrote: QEMU's `apple-gfx` harvests
//! KVM's dirty log, re-protecting only the granules watched here. The library stamps the pages it
//! writes itself ([`host_wrote`], [`DirtyLog::wrote`]). A device made without a source has no
//! log: no import gets one, and every copy is whole, as before.
//!
//! **Stamps.** Harvests are numbered from 1, in order, and a page keeps the number of the latest
//! one that saw it written: its stamp. Only watched pages have one. Watches are counted per
//! [`GRANULE`] (64 pages), the unit the host clears its log in: a granule whose first watch
//! begins has every page stamped with the next harvest's number, so it reads as written for any
//! stamp from before the watch, and a granule whose last watch ends loses its stamps. A page with
//! no stamp reads as written at every stamp, and so does every byte outside the guest's RAM.
//!
//! **A harvest**, the host's, from a thread of its own:
//! 1. [`DirtyLog::begin`] numbers the harvest and gives the watched runs, in granules;
//! 2. the host makes every guest write made before the begin visible in its log (on Intel,
//!    flushing each vCPU's page-modification log), and takes and clears the watched runs' bits
//!    (KVM's `KVM_CLEAR_DIRTY_LOG` re-protects them, so a later write shows in a later harvest);
//! 3. [`DirtyLog::end`] stamps the written pages, one bitmap word per granule, and publishes.
//!
//! **Ordering.** [`DirtyLog::request`] answers `started + 1`, the first harvest to begin after
//! the call: a guest write made before the request was made before that harvest's begin, so the
//! harvest sees it, or an earlier one stamped it. Once [`DirtyLog::wait`] for the answer returns
//! true, [`DirtyLog::changed`] shows the page for any stamp from before the write; the runtime's
//! `since` is the stamp it waited for before its last copy, which read the bytes after that wait.
//!
//! **Host writes.** KVM's log sees only the vCPUs' writes. Each write the library makes into the
//! guest's RAM stamps its pages `started + 1` once its bytes are in memory, the bytes ordered
//! before the load of `started` by a full fence. A copy that read the page before the write ended
//! ran after its wait for some `S`, so `started >= S` by then, and the write's stamp is above the
//! copy's `since`: the page is taken again. A write that ends before a harvest's begin is in
//! memory before any copy that waited for that harvest. A writer that writes for long, a video
//! decoder over its lend, brackets its writes instead ([`DirtyLog::begin_write`]): its pages read
//! as written at every stamp until it ends. A batch of runs whose bytes are all in memory
//! ([`DirtyLog::wrote_runs`], [`DirtyLog::wrote_runs_as`]) shares one fence and one load of
//! `started`, and takes the named lock at most once: the runtime reports a command buffer's
//! writes over a surface in scattered guest pages, thousands of runs at a time, and a take of the
//! lock per run was 7% of QEMU's CPU time in a stock-lane present loop on the RTX 5080 Laptop GPU
//! (**measured** 2026-11-05, round 20261105-092958, perf at 499 Hz).
//!
//! A host write costs one relaxed load while no device has a dirty source, and a fence and a load
//! more while no granule is watched: a write before a granule's first watch is in memory before
//! the watch counts it, and the watch stamps the granule as written for every earlier stamp.
//!
//! **Writers.** The runtime names its own writes (a shadow's copy back or merge) with a writer
//! ([`DirtyLog::wrote_as`]), and [`DirtyLog::changed_except`] leaves out pages only that writer
//! wrote after `since`. A page that a named writer wrote keeps, beside its stamp, the writer and
//! the latest stamp of every other write (`others`), in a side table its chunk allocates at its
//! first named write. A write by the page's writer leaves `others`; a write by anyone else raises
//! it; a named write by another writer takes the page over, with `others` its stamp before
//! (which may count the new writer's own earlier writes: more copies, never fewer). The writer
//! and `others` change only under the named lock; a stamp change outside it that finds the side
//! table takes the lock (or holds it already, from an earlier chunk of its batch) and raises
//! `others`, and the side table's publication and the stamp's raise are both sequentially
//! consistent, so one of the two sees the other.
//!
//! **Only "after".** A page keeps only its latest stamps, so the log answers which pages changed
//! after a stamp, never which changed within a window.
//!
//! **Waits never hang.** The harvester needs the host (QEMU's needs the BQL), and the threads a
//! wait runs on are waited on by guest MMIO accesses. So a wait is bounded ([`WAIT_TIMEOUT`]),
//! and false means "take everything":
//! - the device closes the log ([`DirtyLog::close`]) before a call the host makes under its lock
//!   waits on its threads (reset, pause, stop, suspend and resume), and every wait answers false
//!   at once until it reopens; at its last release it retires the log ([`DirtyLog::retire`]);
//! - a wait that times out stalls the log, and later waits answer false at once until the next
//!   publication.
//!
//! **Memory.** The table is a slot per 2 MiB of the guest's RAM (16 bytes a slot), and a chunk of
//! 512 page stamps (8 bytes a page) allocated at the first watch of one of its granules, kept
//! until the log drops so reads never lock: 8 bytes a page ever watched, 16 more where a named
//! writer wrote.
//!
//! The state lock is a leaf apart from the named lock, taken under it; the host's hook is called
//! under a lock of its own that only [`DirtyLog::retire`] also takes, so no call reaches the host
//! after the retire returns.

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering, fence};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::num::{ToU64, ToUsize};
use crate::platform::os::{Clock, Deadline};
use crate::sync::{Condvar, Mutex, MutexGuard};

/// A harvest's number; 0 is before the first.
pub type DirtyStamp = u64;

/// The log's page: 4 KiB, the x86-64 guest's page and KVM's on an x86-64 host.
pub const PAGE_SIZE: u64 = 1 << PAGE_SHIFT;
const PAGE_SHIFT: u32 = 12;
/// What watches are counted in, and what a harvest takes the watched pages in: 64 pages
/// (256 KiB), one `u64` word of a page bitmap, and the unit of KVM's `KVM_CLEAR_DIRTY_LOG`.
pub const GRANULE: u64 = PAGE_SIZE << GRANULE_SHIFT;
const GRANULE_SHIFT: u32 = 6;
const GRANULE_PAGES: usize = 1 << GRANULE_SHIFT;
/// log2 of the pages in a chunk: 512 pages, 2 MiB.
const CHUNK_SHIFT: u32 = 9;
const CHUNK_PAGES: usize = 1 << CHUNK_SHIFT;
const CHUNK_GRANULES: usize = CHUNK_PAGES >> GRANULE_SHIFT;
/// The stamp of a page no import watches: written at every stamp.
const UNKNOWN: DirtyStamp = DirtyStamp::MAX;
/// How long [`DirtyLog::wait`] waits for a publication: about half a frame at 120 Hz. A harvest
/// takes well under a millisecond, so only a host blocked behind the waiter's own chain reaches
/// it.
pub const WAIT_TIMEOUT: Duration = Duration::from_millis(4);

/// What the host does when a stamp is wanted: start a harvest soon, from a thread of its own.
/// Called on any thread, at most once per stamp, with no lock of the log held. It must not block,
/// and must not call the device.
pub type HarvestWanted = Box<dyn Fn() + Send + Sync>;

/// One run of an import: `length` bytes from `offset` in the import, showing the guest's RAM from
/// guest physical `guest`, or memory that is not the guest's RAM (`None`), which reads as written
/// at every stamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    /// Where the run starts in the import.
    pub offset: u64,
    /// The run's length in bytes.
    pub length: u64,
    /// The guest physical address of the run's first byte.
    pub guest: Option<u64>,
}

/// The log's counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DirtyCounts {
    /// The last stamp published.
    pub published: DirtyStamp,
    /// Waits that timed out and stalled the log.
    pub stalls: u64,
    /// Waits that answered false at once: the log closed, retired or stalled.
    pub refused: u64,
    /// Granules watched now.
    pub watched_granules: u64,
}

/// The side table of a chunk's named writes (module docs, "Writers"): each page's writer (0 for
/// none) and the latest stamp of every other write. Read and written under the named lock.
struct Named {
    writer: [AtomicU64; CHUNK_PAGES],
    others: [AtomicU64; CHUNK_PAGES],
}

/// 512 pages of the table.
struct Chunk {
    /// The guest page number of the chunk's first page.
    first_page: u64,
    stamps: [AtomicU64; CHUNK_PAGES],
    /// The watches of each granule, changed under the state lock.
    watches: [AtomicU32; CHUNK_GRANULES],
    /// The side table, set once under the named lock.
    named: OnceLock<Box<Named>>,
}

impl Chunk {
    fn named(&self) -> Option<&Named> {
        self.named.get().map(|n| &**n)
    }
}

/// A slot of the table: the chunk of 2 MiB of the guest's RAM, made at its first watch and kept
/// until the log drops, so reads never lock.
type Slot = OnceLock<Box<Chunk>>;

/// One range of the guest's RAM, in pages, and the table's slots for it.
#[derive(Clone, Copy, Debug)]
struct Ram {
    first_page: u64,
    end_page: u64,
    /// The first slot of its chunks, the chunk of `first_page`.
    first_slot: usize,
}

#[derive(Debug, Default)]
struct State {
    /// The watched runs of the harvest in flight, as [`DirtyLog::begin`] gave them.
    in_flight: Option<Vec<Range<u64>>>,
    stalls: u64,
    refused: u64,
    watched_granules: u64,
}

/// The dirty log of one device's guest RAM (module docs).
pub struct DirtyLog {
    ram: Box<[Ram]>,
    slots: Box<[Slot]>,
    /// The last harvest begun.
    started: AtomicU64,
    /// The last harvest published.
    published: AtomicU64,
    /// The highest stamp the host was told is wanted.
    notified: AtomicU64,
    /// The closes not yet reopened.
    closed: AtomicU32,
    retired: AtomicBool,
    /// A wait timed out since the last publication.
    stalled: AtomicBool,
    /// Writes under way ([`DirtyLog::begin_write`]), counted for a lock-free look.
    pending_count: AtomicUsize,
    pending: Mutex<Vec<(u64, Range<u64>)>>,
    next_pending: AtomicU64,
    state: Mutex<State>,
    cond: Condvar,
    named_lock: Mutex<()>,
    /// The times the named lock was taken, for the tests.
    #[cfg(test)]
    named_takes: AtomicU64,
    wanted: Mutex<Option<HarvestWanted>>,
}

impl Drop for DirtyLog {
    fn drop(&mut self) {
        let left = self.state.get_mut().watched_granules;
        WATCHED_GRANULES.fetch_sub(left, Ordering::SeqCst);
    }
}

impl std::fmt::Debug for DirtyLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirtyLog")
            .field("ram", &self.ram)
            .field("started", &self.started)
            .field("published", &self.published)
            .finish_non_exhaustive()
    }
}

/// The pages `range` touches.
fn pages(range: &Range<u64>) -> Range<u64> {
    if range.end <= range.start {
        return 0..0;
    }
    let first = range.start >> PAGE_SHIFT;
    let end = (range.end >> PAGE_SHIFT).saturating_add(u64::from(range.end & (PAGE_SIZE - 1) != 0));
    first..end
}

/// Appends `range` to `out`, joined to the last range where they touch; an empty one is not.
fn push_joined(out: &mut Vec<Range<u64>>, range: Range<u64>) {
    if range.end <= range.start {
        return;
    }
    if let Some(last) = out.last_mut()
        && last.end == range.start
    {
        last.end = range.end;
        return;
    }
    out.push(range);
}

impl DirtyLog {
    /// An open log over the guest's RAM `ram` (guest physical ranges), with nothing watched and
    /// nothing published, whose host is told through `wanted` when a stamp is wanted.
    pub fn new(ram: &[Range<u64>], wanted: HarvestWanted) -> DirtyLog {
        let mut sorted: Vec<Range<u64>> = ram.iter().filter(|r| r.start < r.end).cloned().collect();
        sorted.sort_by_key(|r| r.start);
        let mut slots = 0usize;
        let ram: Box<[Ram]> = sorted
            .iter()
            .map(|r| {
                let p = pages(r);
                let first_slot = slots;
                let chunks = ((p.end.saturating_sub(1) >> CHUNK_SHIFT)
                    .saturating_sub(p.start >> CHUNK_SHIFT))
                .saturating_add(1);
                slots = slots.saturating_add(chunks.to_usize());
                Ram {
                    first_page: p.start,
                    end_page: p.end,
                    first_slot,
                }
            })
            .collect();
        DirtyLog {
            ram,
            slots: (0..slots).map(|_| OnceLock::new()).collect(),
            started: AtomicU64::new(0),
            published: AtomicU64::new(0),
            notified: AtomicU64::new(0),
            closed: AtomicU32::new(0),
            retired: AtomicBool::new(false),
            stalled: AtomicBool::new(false),
            pending_count: AtomicUsize::new(0),
            pending: Mutex::new(Vec::new()),
            next_pending: AtomicU64::new(0),
            state: Mutex::new(State::default()),
            cond: Condvar::new(),
            named_lock: Mutex::new(()),
            #[cfg(test)]
            named_takes: AtomicU64::new(0),
            wanted: Mutex::new(Some(wanted)),
        }
    }

    /// Walks `pages` (guest page numbers) a chunk at a time: `f(first, end, slot)`, with the
    /// table's slot of the chunk holding `first..end`, or `None` for pages outside the RAM.
    fn walk(&self, pages: Range<u64>, mut f: impl FnMut(u64, u64, Option<&Slot>)) {
        let mut p = pages.start;
        while p < pages.end {
            match self
                .ram
                .iter()
                .find(|r| r.first_page <= p && p < r.end_page)
            {
                Some(r) => {
                    let chunk_end = ((p >> CHUNK_SHIFT).saturating_add(1)) << CHUNK_SHIFT;
                    let end = pages.end.min(chunk_end).min(r.end_page);
                    let slot = r.first_slot.saturating_add(
                        ((p >> CHUNK_SHIFT).saturating_sub(r.first_page >> CHUNK_SHIFT)).to_usize(),
                    );
                    f(p, end, self.slots.get(slot));
                    p = end;
                }
                None => {
                    let next = (self.ram.iter().map(|r| r.first_page))
                        .filter(|&s| s > p)
                        .min()
                        .unwrap_or(u64::MAX);
                    let end = pages.end.min(next);
                    f(p, end, None);
                    p = end;
                }
            }
        }
    }

    /// The chunk in `slot`, made where there is none (under the state lock).
    fn chunk_of(slot: &Slot, first: u64) -> &Chunk {
        slot.get_or_init(|| {
            Box::new(Chunk {
                first_page: first & !((1 << CHUNK_SHIFT) - 1),
                stamps: std::array::from_fn(|_| AtomicU64::new(UNKNOWN)),
                watches: std::array::from_fn(|_| AtomicU32::new(0)),
                named: OnceLock::new(),
            })
        })
    }

    /// The stamp of the first harvest to begin after this call: once it is published, every write
    /// made before this call shows in [`DirtyLog::changed`]. Tells the host it is wanted, once
    /// per stamp. Never waits for the host.
    pub fn request(&self) -> DirtyStamp {
        let stamp = self.started.load(Ordering::SeqCst).saturating_add(1);
        self.want(stamp);
        stamp
    }

    /// Tells the host `stamp` is wanted, unless it was told already or it is published.
    fn want(&self, stamp: DirtyStamp) {
        if self.published.load(Ordering::Acquire) >= stamp
            || self.notified.fetch_max(stamp, Ordering::AcqRel) >= stamp
        {
            return;
        }
        if let Some(wanted) = &*self.wanted.lock() {
            wanted();
        }
    }

    /// Waits until `stamp` is published, for at most [`WAIT_TIMEOUT`]: true once it is, false
    /// when it is not and the caller must take every page as written (module docs).
    pub fn wait(&self, stamp: DirtyStamp, clock: &impl Clock) -> bool {
        self.wait_for(stamp, clock, WAIT_TIMEOUT)
    }

    fn refusing(&self) -> bool {
        self.closed.load(Ordering::SeqCst) > 0
            || self.retired.load(Ordering::SeqCst)
            || self.stalled.load(Ordering::SeqCst)
    }

    /// [`DirtyLog::wait`], for at most `timeout`.
    pub fn wait_for(&self, stamp: DirtyStamp, clock: &impl Clock, timeout: Duration) -> bool {
        if self.published.load(Ordering::Acquire) >= stamp {
            return true;
        }
        self.want(stamp);
        let deadline = Deadline::Mono(clock.now_mono().saturating_add(timeout));
        let mut s = self.state.lock();
        loop {
            if self.published.load(Ordering::Acquire) >= stamp {
                return true;
            }
            if self.refusing() {
                s.refused = s.refused.saturating_add(1);
                return false;
            }
            if clock.wait_until(&self.cond, &mut s, deadline).timed_out() {
                if self.published.load(Ordering::Acquire) >= stamp {
                    return true;
                }
                self.stalled.store(true, Ordering::SeqCst);
                s.stalls = s.stalls.saturating_add(1);
                log::debug!(
                    "dirty log: stamp {stamp} not published within {timeout:?} (published {}); \
                     whole copies until the next publication",
                    self.published.load(Ordering::Acquire)
                );
                return false;
            }
        }
    }

    /// Closes the log until a [`DirtyLog::reopen`] for each close: every wait answers false at
    /// once, those waiting included. Requests and harvests go on.
    pub fn close(&self) {
        let _s = self.state.lock();
        self.closed.fetch_add(1, Ordering::SeqCst);
        self.cond.notify_all();
    }

    /// Ends one [`DirtyLog::close`].
    pub fn reopen(&self) {
        let _s = self.state.lock();
        let _ = (self.closed).try_update(Ordering::SeqCst, Ordering::SeqCst, |c| c.checked_sub(1));
    }

    /// Closes the log for good and drops the host's hook: once this returns, the hook is never
    /// called again. Harvests are still taken.
    pub fn retire(&self) {
        {
            let _s = self.state.lock();
            self.retired.store(true, Ordering::SeqCst);
            self.cond.notify_all();
        }
        drop(self.wanted.lock().take());
    }

    /// The last stamp published.
    pub fn published(&self) -> DirtyStamp {
        self.published.load(Ordering::Acquire)
    }

    /// The log's counts.
    pub fn counts(&self) -> DirtyCounts {
        let s = self.state.lock();
        DirtyCounts {
            published: self.published(),
            stalls: s.stalls,
            refused: s.refused,
            watched_granules: s.watched_granules,
        }
    }

    /// Begins a watch of the pages of `runs` (guest physical addresses): one more watch of each
    /// granule each run touches, and a granule's first stamps every page of it with the next
    /// harvest's number. [`DirtyLog::unwatch`] with the same runs ends it.
    pub fn watch(&self, runs: &[Range<u64>]) {
        let mut s = self.state.lock();
        let fresh = self.started.load(Ordering::SeqCst).saturating_add(1);
        let mut added = 0u64;
        for run in runs {
            self.each_granule(run, |this, slot, first, g| {
                let c = DirtyLog::chunk_of(slot, first);
                let Some(w) = c.watches.get(g) else { return };
                if w.fetch_add(1, Ordering::Relaxed) == 0 {
                    // Counted before it is stamped: a write that skipped the log as nothing was
                    // watched is in memory before the watch ([`host_wrote`]).
                    WATCHED_GRANULES.fetch_add(1, Ordering::SeqCst);
                    this.set_granule(c, g, fresh);
                    added = added.saturating_add(1);
                }
            });
        }
        s.watched_granules = s.watched_granules.saturating_add(added);
    }

    /// Ends one watch of each granule of `runs`, as [`DirtyLog::watch`] began it. A granule no
    /// import watches any more loses its stamps.
    pub fn unwatch(&self, runs: &[Range<u64>]) {
        let mut s = self.state.lock();
        let mut removed = 0u64;
        for run in runs {
            self.each_granule(run, |this, slot, _, g| {
                let Some(c) = slot.get() else {
                    return;
                };
                let Some(w) = c.watches.get(g) else { return };
                let n = w.load(Ordering::Relaxed);
                if n == 0 {
                    return;
                }
                w.store(n.saturating_sub(1), Ordering::Relaxed);
                if n == 1 {
                    this.set_granule(c, g, UNKNOWN);
                    WATCHED_GRANULES.fetch_sub(1, Ordering::SeqCst);
                    removed = removed.saturating_add(1);
                }
            });
        }
        s.watched_granules = s.watched_granules.saturating_sub(removed);
    }

    /// Calls `f(self, slot, first page, granule index in the chunk)` for each granule of the RAM
    /// that `run` touches, once per run.
    fn each_granule(&self, run: &Range<u64>, mut f: impl FnMut(&DirtyLog, &Slot, u64, usize)) {
        self.walk(pages(run), |first, end, slot| {
            let Some(slot) = slot else { return };
            let chunk_mask = (1u64 << CHUNK_SHIFT) - 1;
            let g_first = ((first & chunk_mask) >> GRANULE_SHIFT).to_usize();
            let g_last = ((end.saturating_sub(1) & chunk_mask) >> GRANULE_SHIFT).to_usize();
            for g in g_first..=g_last {
                f(self, slot, first, g);
            }
        });
    }

    /// Sets every page stamp of granule `g` of `c` to `stamp`, and drops its named writes.
    fn set_granule(&self, c: &Chunk, g: usize, stamp: DirtyStamp) {
        let pages =
            g.saturating_mul(GRANULE_PAGES)..g.saturating_add(1).saturating_mul(GRANULE_PAGES);
        if let Some(named) = c.named() {
            let _n = self.lock_named();
            for i in pages.clone() {
                if let Some(w) = named.writer.get(i) {
                    w.store(0, Ordering::SeqCst);
                }
            }
        }
        for i in pages {
            if let Some(st) = c.stamps.get(i) {
                st.store(stamp, Ordering::SeqCst);
            }
        }
    }

    /// Appends to `out` the bytes of the runs `segments` names (an import's) whose pages were
    /// written after `since`, by anyone: offsets in the import, in order, joined where they touch.
    pub fn changed(&self, segments: &[Segment], since: DirtyStamp, out: &mut Vec<Range<u64>>) {
        self.changed_by(segments, since, None, out);
    }

    /// [`DirtyLog::changed`], leaving out the pages only `writer` wrote after `since` (module
    /// docs, "Writers").
    pub fn changed_except(
        &self,
        segments: &[Segment],
        since: DirtyStamp,
        writer: NonZeroU64,
        out: &mut Vec<Range<u64>>,
    ) {
        self.changed_by(segments, since, Some(writer), out);
    }

    fn changed_by(
        &self,
        segments: &[Segment],
        since: DirtyStamp,
        writer: Option<NonZeroU64>,
        out: &mut Vec<Range<u64>>,
    ) {
        let pending: Vec<Range<u64>> = if self.pending_count.load(Ordering::SeqCst) > 0 {
            self.pending.lock().iter().map(|(_, r)| pages(r)).collect()
        } else {
            Vec::new()
        };
        for seg in segments {
            let seg_end = seg.offset.saturating_add(seg.length);
            let Some(guest) = seg.guest else {
                push_joined(out, seg.offset..seg_end);
                continue;
            };
            // The import's offset of guest page `p`'s first byte, clamped to the segment.
            let at = |p: u64| {
                let g = p << PAGE_SHIFT;
                seg.offset
                    .saturating_add(g.saturating_sub(guest))
                    .clamp(seg.offset, seg_end)
            };
            let guest_end = guest.saturating_add(seg.length);
            self.walk(pages(&(guest..guest_end)), |first, end, slot| {
                let chunk = slot.and_then(OnceLock::get);
                let Some(c) = chunk else {
                    push_joined(out, at(first)..at(end));
                    return;
                };
                let named = writer.and_then(|w| c.named().map(|n| (w, n)));
                let _n = named.map(|_| self.lock_named());
                for p in first..end {
                    let i = (p.saturating_sub(c.first_page)).to_usize();
                    let Some(st) = c.stamps.get(i) else { continue };
                    let mut stamp = st.load(Ordering::SeqCst);
                    if let Some((w, n)) = named
                        && stamp != UNKNOWN
                        && n.writer
                            .get(i)
                            .is_some_and(|x| x.load(Ordering::SeqCst) == w.get())
                    {
                        stamp = n
                            .others
                            .get(i)
                            .map_or(UNKNOWN, |o| o.load(Ordering::SeqCst));
                    }
                    if stamp <= since && !pending.iter().any(|r| r.contains(&p)) {
                        continue;
                    }
                    push_joined(out, at(p)..at(p.saturating_add(1)));
                }
            });
        }
    }

    /// The named lock, counted for the tests.
    fn lock_named(&self) -> MutexGuard<'_, ()> {
        #[cfg(test)]
        self.named_takes.fetch_add(1, Ordering::Relaxed);
        self.named_lock.lock()
    }

    /// The library wrote the guest physical bytes of `range`, and they are in memory: each
    /// watched page of it is stamped with the next harvest's number (module docs, "Host writes").
    pub fn wrote(&self, range: Range<u64>) {
        self.wrote_runs(std::iter::once(range));
    }

    /// [`DirtyLog::wrote`] for each of `runs`, whose bytes are all in memory: one fence and one
    /// load of the harvest number for them all, and the named lock taken at most once.
    pub fn wrote_runs(&self, runs: impl IntoIterator<Item = Range<u64>>) {
        fence(Ordering::SeqCst);
        let stamp = self.started.load(Ordering::SeqCst).saturating_add(1);
        let mut named = None;
        for run in runs {
            self.stamp(&run, stamp, &mut named);
        }
    }

    /// [`DirtyLog::wrote`], as `writer`'s write (module docs, "Writers"):
    /// [`DirtyLog::changed_except`] for `writer` leaves it out.
    pub fn wrote_as(&self, range: Range<u64>, writer: NonZeroU64) {
        self.wrote_runs_as(std::iter::once(range), writer);
    }

    /// [`DirtyLog::wrote_as`] for each of `runs`, whose bytes are all in memory, under one take
    /// of the named lock and one load of the harvest number.
    pub fn wrote_runs_as(&self, runs: impl IntoIterator<Item = Range<u64>>, writer: NonZeroU64) {
        let _n = self.lock_named();
        fence(Ordering::SeqCst);
        let stamp = self.started.load(Ordering::SeqCst).saturating_add(1);
        for run in runs {
            self.stamp_as(&run, stamp, writer);
        }
    }

    /// Stamps each watched page of `range` with `stamp`, as `writer`'s write, under the named
    /// lock.
    fn stamp_as(&self, range: &Range<u64>, stamp: DirtyStamp, writer: NonZeroU64) {
        self.walk(pages(range), |first, end, slot| {
            let Some(c) = slot.and_then(OnceLock::get) else {
                return;
            };
            let named = c.named.get_or_init(|| {
                Box::new(Named {
                    writer: std::array::from_fn(|_| AtomicU64::new(0)),
                    others: std::array::from_fn(|_| AtomicU64::new(0)),
                })
            });
            // The side table is published before any stamp of this chunk is read here; a
            // lock-free stamp fences after its raise and before it looks for the side table
            // (module docs, "Writers").
            fence(Ordering::SeqCst);
            for p in first..end {
                let i = (p.saturating_sub(c.first_page)).to_usize();
                let (Some(st), Some(w), Some(o)) =
                    (c.stamps.get(i), named.writer.get(i), named.others.get(i))
                else {
                    continue;
                };
                if w.load(Ordering::SeqCst) != writer.get() {
                    w.store(writer.get(), Ordering::SeqCst);
                    // After the writer's store: a lock-free stamp either raised this, or finds
                    // the side table and raises `others` once we let the lock go.
                    o.store(st.load(Ordering::SeqCst), Ordering::SeqCst);
                }
                if st.load(Ordering::SeqCst) == UNKNOWN {
                    w.store(0, Ordering::SeqCst);
                    continue;
                }
                st.fetch_max(stamp, Ordering::SeqCst);
            }
        });
    }

    /// Stamps each watched page of `range` with `stamp`, by no named writer. `named` holds the
    /// named lock once a chunk with a side table took it, and keeps it for the rest of a batch.
    fn stamp<'a>(
        &'a self,
        range: &Range<u64>,
        stamp: DirtyStamp,
        named: &mut Option<MutexGuard<'a, ()>>,
    ) {
        self.walk(pages(range), |first, end, slot| {
            let Some(c) = slot.and_then(OnceLock::get) else {
                return;
            };
            self.stamp_in(c, first, end, stamp, named);
        });
    }

    /// Stamps the watched pages `first..end` of `c` with `stamp`, by no named writer, taking the
    /// named lock into `held` where the chunk has a side table and it is not held yet.
    fn stamp_in<'a>(
        &'a self,
        c: &Chunk,
        first: u64,
        end: u64,
        stamp: DirtyStamp,
        held: &mut Option<MutexGuard<'a, ()>>,
    ) {
        let lo = (first.saturating_sub(c.first_page)).to_usize();
        let hi = (end.saturating_sub(c.first_page))
            .to_usize()
            .min(CHUNK_PAGES);
        // A raise even where the stamp is later already: it is the modification the fence
        // orders before the look for the side table (module docs, "Writers"). Under the lock no
        // named write runs between the two, and the look sees every side table published.
        for st in c.stamps.get(lo..hi).unwrap_or(&[]) {
            st.fetch_max(stamp, Ordering::SeqCst);
        }
        if held.is_none() {
            fence(Ordering::SeqCst);
        }
        if let Some(named) = c.named() {
            if held.is_none() {
                *held = Some(self.lock_named());
            }
            for i in lo..hi {
                if let (Some(w), Some(o)) = (named.writer.get(i), named.others.get(i))
                    && w.load(Ordering::SeqCst) != 0
                {
                    o.fetch_max(stamp, Ordering::SeqCst);
                }
            }
        }
    }

    /// Begins a write of the guest physical bytes of `range` that lasts (a device's writes over
    /// a lend): its pages read as written at every stamp, to every asker, until
    /// [`DirtyLog::end_write`] with the number this returns.
    pub fn begin_write(&self, range: Range<u64>) -> u64 {
        let id = self.next_pending.fetch_add(1, Ordering::Relaxed);
        let mut p = self.pending.lock();
        p.push((id, range));
        self.pending_count.store(p.len(), Ordering::SeqCst);
        id
    }

    /// Ends the write `id`: its bytes are in memory, and its pages are stamped as
    /// [`DirtyLog::wrote`] stamps them before they stop reading as written.
    pub fn end_write(&self, id: u64) {
        let range = {
            let p = self.pending.lock();
            p.iter().find(|(w, _)| *w == id).map(|(_, r)| r.clone())
        };
        let Some(range) = range else { return };
        self.wrote(range);
        let mut p = self.pending.lock();
        p.retain(|(w, _)| *w != id);
        self.pending_count.store(p.len(), Ordering::SeqCst);
    }

    /// Begins a harvest (module docs, step 1): numbers it, and fills `watched` with the watched
    /// runs, guest physical, in [`GRANULE`]s, ascending and joined. `None` while one is in
    /// flight. Never blocks on a waiter.
    pub fn begin(&self, watched: &mut Vec<Range<u64>>) -> Option<DirtyStamp> {
        watched.clear();
        let mut s = self.state.lock();
        if s.in_flight.is_some() {
            return None;
        }
        let stamp = self
            .started
            .fetch_add(1, Ordering::SeqCst)
            .saturating_add(1);
        for c in self.slots.iter().filter_map(OnceLock::get) {
            for (g, w) in c.watches.iter().enumerate() {
                if w.load(Ordering::Relaxed) == 0 {
                    continue;
                }
                let start =
                    (c.first_page.saturating_add((g << GRANULE_SHIFT).to_u64())) << PAGE_SHIFT;
                push_joined(watched, start..start.saturating_add(GRANULE));
            }
        }
        s.in_flight = Some(watched.clone());
        Some(stamp)
    }

    /// Ends the harvest `stamp` (module docs, step 3): `written` holds one word per [`GRANULE`]
    /// of the runs [`DirtyLog::begin`] gave, in order, bit `i` set where page `i` of the granule
    /// was written. Stamps those pages, publishes `stamp` and wakes the waits. False, changing
    /// nothing, where `stamp` is not the harvest in flight or the words do not match its runs.
    pub fn end(&self, stamp: DirtyStamp, written: &[u64]) -> bool {
        {
            let mut s = self.state.lock();
            let Some(runs) = s.in_flight.as_ref() else {
                return false;
            };
            if self.started.load(Ordering::SeqCst) != stamp {
                return false;
            }
            let granules: u64 = (runs.iter())
                .map(|r| r.end.saturating_sub(r.start) >> (PAGE_SHIFT + GRANULE_SHIFT))
                .fold(0, u64::saturating_add);
            if granules != written.len().to_u64() {
                return false;
            }
            let mut words = written.iter();
            for run in runs {
                let mut g = run.start;
                while g < run.end {
                    let word = words.next().copied().unwrap_or(!0);
                    if word != 0 {
                        self.stamp_word(g >> PAGE_SHIFT, word, stamp);
                    }
                    g = g.saturating_add(GRANULE);
                }
            }
            s.in_flight = None;
            self.published.store(stamp, Ordering::SeqCst);
            self.stalled.store(false, Ordering::SeqCst);
        }
        self.cond.notify_all();
        true
    }

    /// Stamps the pages of the granule from guest page `first` whose bits `word` sets, a run of
    /// set bits at a time, taking the named lock at most once.
    fn stamp_word(&self, first: u64, word: u64, stamp: DirtyStamp) {
        let mut named = None;
        let mut bits = word;
        while bits != 0 {
            let i = bits.trailing_zeros();
            let run = (!(bits >> i)).trailing_zeros();
            let end = i.saturating_add(run);
            let p = first.saturating_add(u64::from(i));
            let q = first.saturating_add(u64::from(end));
            self.stamp(&((p << PAGE_SHIFT)..(q << PAGE_SHIFT)), stamp, &mut named);
            bits = if end >= 64 { 0 } else { bits & (!0u64 << end) };
        }
    }
}

/// What resolves a host address range to the guest physical runs behind it, and stamps each in
/// its device's log (`f(log, run)`): the platform's, where it knows the files behind its views of
/// the guest's RAM.
pub type Resolver = Box<dyn Fn(u64, u64, &mut dyn FnMut(&Arc<DirtyLog>, Range<u64>)) + Send + Sync>;

static RESOLVER: OnceLock<Resolver> = OnceLock::new();

/// Whether any log can take a host write: while none can, [`host_wrote`] costs one relaxed load.
static RESOLVING: AtomicBool = AtomicBool::new(false);

/// The granules every log in the process watches: while none, [`host_wrote`] resolves nothing.
static WATCHED_GRANULES: AtomicU64 = AtomicU64::new(0);

/// Installs the process's resolver, once; false where one was installed already.
pub fn install_resolver(resolver: Resolver) -> bool {
    RESOLVER.set(resolver).is_ok()
}

/// Says whether any log can take a host write (the platform's, as it registers and drops the
/// logs of the guest's RAM it resolves to).
pub fn set_resolving(resolving: bool) {
    RESOLVING.store(resolving, Ordering::SeqCst);
}

/// The library wrote `length` bytes at host address `at`, and they are in memory: stamps the
/// guest pages behind them in their device's log (module docs, "Host writes"). One relaxed load
/// while no device has a dirty source.
#[inline]
pub fn host_wrote(at: u64, length: u64) {
    if RESOLVING.load(Ordering::Relaxed) {
        host_wrote_slow(at, length);
    }
}

#[cold]
fn host_wrote_slow(at: u64, length: u64) {
    // A write before the first watch of a granule is in memory before the watch counts it, and
    // the watch stamps the granule as written for every stamp before it.
    fence(Ordering::SeqCst);
    if WATCHED_GRANULES.load(Ordering::SeqCst) == 0 {
        return;
    }
    if let Some(resolver) = RESOLVER.get() {
        resolver(at, length, &mut |log, run| log.wrote(run));
    }
}

/// A write that lasts into the guest's RAM behind a host range ([`host_write`]): its pages read
/// as written at every stamp until it drops, and are stamped then.
#[derive(Default)]
pub struct HostWrite {
    writes: Vec<(Arc<DirtyLog>, u64)>,
}

impl std::fmt::Debug for HostWrite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostWrite")
            .field("writes", &self.writes.len())
            .finish()
    }
}

impl Drop for HostWrite {
    fn drop(&mut self) {
        for (log, id) in &self.writes {
            log.end_write(*id);
        }
    }
}

/// Begins a write that lasts into the `length` bytes at host address `at` (a device's writes
/// over a lend): hold what it returns until the writes are done. Empty, at one load, while no
/// device has a dirty source.
pub fn host_write(at: u64, length: u64) -> HostWrite {
    let mut writes = Vec::new();
    if RESOLVING.load(Ordering::Relaxed)
        && let Some(resolver) = RESOLVER.get()
    {
        resolver(at, length, &mut |log, run| {
            let id = log.begin_write(run);
            writes.push((Arc::clone(log), id));
        });
    }
    HostWrite { writes }
}

#[cfg(test)]
mod tests;
