//! Regroups: a context's lone pages blocks' views, two seconds old or older, moved into one view
//! of all their pages, so that the device imports one file where it imported one a view
//! (`decode::surface_pages`, "Regroups").
//!
//! - **Why after the make.** A group dma-buf is formed as views are made, in a burst
//!   ("Group dma-bufs"), but a context that makes its no-copy buffers one at a time, each followed
//!   by a ring-0 message that waits for it, closes each group with one member, which is made
//!   alone. **Measured** on homelab-gem10-1's Radeon 780M, a macOS 26.7.2 guest's desktop walk
//!   (Finder, then Launchpad three times; m3-max-36781, the trace's views and their imports'
//!   lives): at the second Launchpad's midpoint 241 pages blocks' views of several runs held a
//!   live import each, every one 2 s old or older and 207 of them 5 s or older (23 s at the
//!   median), WindowServer's no-copy buffers of 12 KiB at the median, against those that went
//!   before it, which lived 0.26 s at the median. On homelab-ms01-1's Iris Xe, a macOS 26.7.2
//!   guest's animated page (m3-max-36792), every one of the 80 to 83 such live views was 10 s old
//!   or older.
//! - **What a pass does.** Every [`REGROUP_EVERY`] (a ring-0 message of any context finds a pass
//!   due), the surface worker gathers each context's lone views made [`REGROUP_AGE`] ago or more,
//!   of [`REGROUP_MEMBER_MOST`] bytes or fewer, in their order, into regroups of at most
//!   [`HostVm::regroup_limits`]' runs and bytes, at least two views each; maps a view of each
//!   regroup's pages, its members' runs one after another, each member at a page of its own; and
//!   makes a no-copy buffer over the whole of it, which holds the runtime's import of its pages:
//!   the import is made there, off the context's lanes. A context whose regroups wait for its lane
//!   is passed over until they moved, so a context that sends no ring-0 message holds one pass's
//!   regroups at most, and no view is in two.
//! - **What its lane does.** The context's ring-0 lane then, at its next messages, moves each
//!   member's buffers into the regroup's view ([`Buffer::rebind_no_copy`]): the block's buffer
//!   that holds its view's import, then each buffer its `BUFFER_OVER_PAGES` made, each lying over
//!   the same guest pages as before, at the same offset in its page, its object and GPU address as
//!   they were. Each move lies in the regroup's import: the platform imports nothing then. A
//!   buffer lies in the import of the regroup's view from then on, and the block's view's own
//!   import goes with the last buffer over it. A block whose view changed since the pass (a
//!   release, a reuse of its offset) is left out, and so is one whose view's import buffer the
//!   platform refuses to move, which no regroup takes again. A buffer of the guest's the platform
//!   refuses to move (one a command buffer not yet committed used, one not yet submitted whose
//!   encoder the runtime cannot encode again, or one a dispatch left to the host's step binds, a
//!   GPU-written indirect one among them, until it completed: `EncodedUses`) keeps lying over the
//!   view it lay over, which the block keeps, and is asked again at its context's ring-0
//!   messages, [`REGROUP_EVERY`] apart, up to [`REGROUP_TRIES`] times. No such buffer is left out
//!   of a regroup beforehand: the refusal is the platform's, and fails no command buffer.
//! - **A few members a message.** A message's moves stop at the first member past
//!   [`REGROUP_BUDGET`] of the lane's time, the rest left for the context's next ring-0 messages.
//!   **Measured** on homelab-um790-xtx-1's Radeon 780M, a macOS 26.7.2 guest's desktop walk
//!   (m3-max-37248, every member moved at one message): 13 of its 17 messages moved about 720
//!   buffers in 29.7 ms of WindowServer's ring-0 lane, 22 to 108 us a buffer and 39 us at the
//!   median over the seconds that moved them (the runtime's count of buffers bound in an import
//!   over each second, against the lane's time), and one message took 6.0 ms.
//! - **One move at a time.** The platform's move of a buffer allows encodes and commits of it on
//!   other threads, and no other move of it, nor a texture made over its bytes, while it runs.
//!   The moves of every context run one at a time, under one lock of the device's decoders
//!   (`Shared::regroup_moves`), and a texture over a buffer's bytes (`TEXTURE_FROM_BUFFER`) is
//!   made on its context's ring-0 lane, where the context's moves run.
//! - **What it costs.** A regroup's import holds every member's pages until its last member's
//!   buffers go: a member whose block was released keeps its pages pinned by the file and mapped
//!   in the device's import, though no object covers them ([`Record`]). They are the guest's RAM,
//!   resident anyway, but counted against the GPU's GTT and mapped in its address space; hence
//!   [`REGROUP_MEMBER_MOST`]. **Measured** on homelab-um790-xtx-1's RX 7900 XTX, a macOS 26.7.2
//!   guest's Geekbench 7.1.0 Metal run (m3-max-37250, members of any size): its contexts' lone
//!   views of many MiB went into regroups of up to 64 MiB, 680 MiB at one pass, and 7 live
//!   regroups held 174.6 MiB, 156.6 MiB of it members gone, each workload's time within 0.15 s of
//!   the run's with no regroups (m3-max-37249); the desktop walk's members were all smaller than
//!   1 MiB. A replay of the 780M walk's trace (every second, views 2 s old or older into
//!   regroups of 1,024 runs and 64 MiB at most) cut the live imports of such views at the second
//!   and third Launchpad from 268 to 41, its regroups holding 41 to 43 MiB, 14 to 17 MiB of it of
//!   members gone; with 5 s, 60 and 8 to 10 MiB. On the Iris Xe's page, 110 to 21-25, with 14 to
//!   23 MiB of members gone.
//!
//! [`HostVm::regroup_limits`]: reims_vgpu_vulkan::core::platform::os::vm::HostVm::regroup_limits
//! [`Buffer::rebind_no_copy`]: reims_vgpu_vulkan::core::platform::resource::Buffer::rebind_no_copy

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use reims_vgpu_vulkan::core::host::HostPtr;
use reims_vgpu_vulkan::core::platform::os::clock::{Clock as _, MonoTime};
use reims_vgpu_vulkan::core::platform::os::vm::GroupLimits;
use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
use reims_vgpu_vulkan::core::platform::{Handle as _, HostRange, Platform};
use reims_vgpu_vulkan::core::raise::RaiseResult;
use reims_vgpu_vulkan::core::sync::Mutex;

use super::super::ObjHandle;
use super::{SurfaceView, SurfaceViews, buffer_over};

/// How old a lone view is at least as a pass takes it: past the lives of most views that go
/// (0.26 s at the median on the 780M's desktop, the module doc).
pub(crate) const REGROUP_AGE: Duration = Duration::from_secs(2);

/// How often a pass is due, and how long after a refusal a buffer is asked to move again.
pub(crate) const REGROUP_EVERY: Duration = Duration::from_secs(1);

/// How many times a member's buffer is asked to move, the first included, before it is left over
/// the view it lay over.
pub(crate) const REGROUP_TRIES: u32 = 8;

/// How much of its lane's time a context's message spends on moves before it leaves the rest for
/// its next messages: a member's moves past it end the message's (the module doc).
pub(crate) const REGROUP_BUDGET: Duration = Duration::from_micros(500);

/// The bytes of the largest view a pass takes: a member gone holds its pages in the regroup's
/// import (the module doc, "What it costs").
pub(crate) const REGROUP_MEMBER_MOST: u64 = 1 << 20;

/// What a regroup's view holds: its members' bytes, and those of members whose block is live, so
/// that the bytes of members gone are counted while the regroup lives.
#[derive(Debug)]
pub(crate) struct Record {
    bytes: u64,
    live: AtomicU64,
}

/// A pages block's place in a regroup ([`Record`]): its bytes leave the live ones as it goes.
#[derive(Debug)]
pub(crate) struct Member {
    record: Arc<Record>,
    bytes: u64,
    /// The objects of its `BUFFER_OVER_PAGES` the platform refused to move, by id, to be asked
    /// again.
    pending: Vec<u64>,
    /// The moves of them asked so far.
    tries: u32,
}

impl Drop for Member {
    fn drop(&mut self) {
        self.record.live.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// A regroup a pass made, for its context's ring-0 lane to move its members into.
pub(crate) struct Prepared<P: Platform> {
    /// The view of the members' pages, one after another.
    view: Arc<SurfaceView<P>>,
    /// A buffer over the whole view, which holds the runtime's import of its pages until the
    /// members' buffers lie in it.
    anchor: P::Buffer,
    record: Arc<Record>,
    /// Each member: its block, the view it had at the pass, and its offset in `view`.
    members: Vec<(u64, Arc<SurfaceView<P>>, u64)>,
}

/// What the regroups did, for the log.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RegroupCounts {
    /// Passes made.
    pub(crate) passes: u64,
    /// Regroups' views made, with their import.
    pub(crate) made: u64,
    /// Pages blocks whose buffers moved into a regroup's view.
    pub(crate) members: u64,
    /// Buffers moved.
    pub(crate) moved: u64,
    /// Refusals of the platform to move a buffer, each asked again up to [`REGROUP_TRIES`] times.
    pub(crate) refused: u64,
    /// Buffers moved when asked again.
    pub(crate) retried: u64,
    /// Buffers left over their block's view after [`REGROUP_TRIES`] refusals.
    pub(crate) stayed: u64,
    /// Members left out: their block's view changed after the pass, or the platform refused to
    /// move its import buffer.
    pub(crate) left: u64,
    /// The longest a context's lane took to move a regroup's members, in microseconds.
    pub(crate) most_us: u64,
}

/// What a context's lane did at one message ([`super::super::GraphicsDecoder::move_regroups`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Moves {
    /// What it took, the wait for another context's moves included.
    pub(crate) took: Duration,
    /// The platform's moves asked, refused ones included.
    pub(crate) rebinds: u64,
    /// The members, and the members' blocks asked again, it took up.
    pub(crate) handled: u64,
    /// Those it left for the context's next messages, its budget spent.
    pub(crate) left: u64,
}

/// The views' state of regroups.
pub(crate) struct Regroups<P: Platform> {
    /// When the next pass is due; `None` until a ring-0 message first asks.
    next: Option<MonoTime>,
    /// A pass handed to the surface worker and not yet done.
    passing: bool,
    /// The regroups made, by context, until its ring-0 lane moves their members.
    ready: HashMap<u32, Vec<Prepared<P>>>,
    /// The blocks with buffers to be asked to move again, by context, each with when.
    retry: HashMap<u32, Vec<(MonoTime, u64)>>,
    /// The regroups whose views may live.
    records: Vec<Weak<Record>>,
    /// A message's time for moves ([`REGROUP_BUDGET`]).
    budget: Duration,
    pub(crate) counts: RegroupCounts,
}

impl<P: Platform> Default for Regroups<P> {
    fn default() -> Self {
        Regroups {
            next: None,
            passing: false,
            ready: HashMap::new(),
            retry: HashMap::new(),
            records: Vec::new(),
            budget: REGROUP_BUDGET,
            counts: RegroupCounts::default(),
        }
    }
}

/// What the live regroups hold: how many, their bytes, and the bytes of members gone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RegroupsHeld {
    pub(crate) regroups: u64,
    pub(crate) bytes: u64,
    pub(crate) gone: u64,
}

/// A candidate of a pass: its block, its view, its runs and its bytes.
type Candidate<P> = (u64, Arc<SurfaceView<P>>, Vec<(u64, u64)>, u64);

/// The one call that moves a buffer into a regroup's view ([`Buffer::rebind_no_copy`]): the
/// `length` bytes at `at` in `view`, held mapped by the deallocator until the platform let go of
/// them.
///
/// [`Buffer::rebind_no_copy`]: reims_vgpu_vulkan::core::platform::resource::Buffer::rebind_no_copy
fn rebind<P: Platform>(
    buffer: &P::Buffer,
    view: &Arc<SurfaceView<P>>,
    at: u64,
    length: u64,
) -> Result<(), String> {
    if at.checked_add(length).is_none_or(|end| end > view.length()) {
        return Err("past the view".to_owned());
    }
    // SAFETY: the deallocator holds the view, whose mapping holds the regroup's, so the bytes
    // stay mapped until the platform let go of them; they lie in the view (checked above).
    let bytes = unsafe { HostRange::new(view.base().wrapping_add(at), length) };
    let keep = Arc::clone(view);
    buffer.rebind_no_copy(bytes, Some(Box::new(move || drop(keep))))
}

impl<P: Platform> SurfaceViews<P> {
    /// The pages block at `block` made the object `id` over its view (`BUFFER_OVER_PAGES`).
    pub(crate) fn buffer_made(&mut self, block: u64, id: u64) {
        if let Some(p) = self.private.get_mut(&block)
            && !p.buffers.contains(&id)
        {
            p.buffers.push(id);
        }
    }

    /// A heap lies over the view of the pages block at `block`.
    pub(crate) fn heap_made(&mut self, block: u64) {
        if let Some(p) = self.private.get_mut(&block) {
            p.heaped = true;
        }
    }

    /// Whether a pass is due at `now`: one is taken as due and the next set
    /// [`REGROUP_EVERY`] later, none while one is being made.
    pub(crate) fn regroup_due(&mut self, now: MonoTime) -> bool {
        let r = &mut self.regroups;
        match r.next {
            _ if r.passing => false,
            None => {
                r.next = Some(now.saturating_add(REGROUP_EVERY));
                false
            }
            Some(at) if now.as_nanos() >= at.as_nanos() => {
                r.next = Some(now.saturating_add(REGROUP_EVERY));
                r.passing = true;
                true
            }
            Some(_) => false,
        }
    }

    /// Whether regroups of `context`, or buffers of its members to be asked again, wait for its
    /// lane at `now`.
    pub(crate) fn regroups_ready(&self, context: u32, now: MonoTime) -> bool {
        let r = &self.regroups;
        r.ready.contains_key(&context)
            || (r.retry.get(&context))
                .is_some_and(|blocks| blocks.iter().any(|(at, _)| now.as_nanos() >= at.as_nanos()))
    }

    /// Sets a message's time for moves (for the tests; [`REGROUP_BUDGET`] otherwise).
    #[cfg(test)]
    pub(crate) fn set_regroup_budget(&mut self, budget: Duration) {
        self.regroups.budget = budget;
    }

    /// Context `context` closed: its regroups made and not yet moved, for the caller to let go
    /// of with no lock held, and its members' buffers to be asked again forgotten.
    pub(crate) fn regroups_closed(&mut self, context: u32) -> Vec<Prepared<P>> {
        self.regroups.retry.remove(&context);
        self.regroups.ready.remove(&context).unwrap_or_default()
    }

    /// The device's reset ([`SurfaceViews::reset`]): every regroup made and not yet moved, for the
    /// caller to let go of, and the buffers to be asked again forgotten, as no block is left.
    pub(super) fn regroups_reset(&mut self) -> HashMap<u32, Vec<Prepared<P>>> {
        self.regroups.retry.clear();
        std::mem::take(&mut self.regroups.ready)
    }

    /// What the regroups did so far.
    pub(crate) fn regroup_counts(&self) -> RegroupCounts {
        self.regroups.counts
    }

    /// What the live regroups hold now.
    pub(crate) fn regroups_held(&mut self) -> RegroupsHeld {
        let mut held = RegroupsHeld::default();
        self.regroups.records.retain(|r| {
            let Some(r) = r.upgrade() else {
                return false;
            };
            held.regroups += 1;
            held.bytes += r.bytes;
            held.gone += r.bytes.saturating_sub(r.live.load(Ordering::Relaxed));
            true
        });
        held
    }

    /// The lone pages blocks' views a pass takes at `now`, by context, each in regroups within
    /// `limits` of two members or more: views of several runs and [`REGROUP_MEMBER_MOST`] bytes or
    /// fewer, made [`REGROUP_AGE`] ago or more, in no group or regroup, made or moving, with no
    /// heap over them and pages their process maps writable, and not in the views' cache, whose
    /// entry shares the block's import buffer; none of a context whose regroups wait for its
    /// lane.
    fn regroup_candidates(
        &self,
        now: MonoTime,
        limits: GroupLimits,
    ) -> Vec<(u32, Vec<Candidate<P>>)> {
        let cached: HashSet<*const SurfaceView<P>> =
            (self.cache.iter()).map(|c| Arc::as_ptr(&c.view)).collect();
        let mut by_context: HashMap<u32, Vec<(MonoTime, Candidate<P>)>> = HashMap::new();
        // The trace's `regcand`: the blocks with no view, then how many blocks each check below
        // left out, the first that did, in their order, how many were taken, and the views the
        // cache holds (a diagnostic).
        let mut left = [0u64; 16];
        for (&block, p) in &self.private {
            let Some((view, _)) = &p.view else {
                left[0] += 1;
                continue;
            };
            let old = p
                .made
                .is_some_and(|made| now.saturating_duration_since(made) >= REGROUP_AGE);
            let out = [
                p.made.is_none(),
                !old,
                p.runs.len() < 2,
                p.length > REGROUP_MEMBER_MOST,
                p.heaped,
                p.stays,
                p.queued,
                p.access.read_only(),
                p.regrouped.is_some(),
                self.regroups.ready.contains_key(&p.context),
                view.group.lock().is_some(),
                cached.contains(&Arc::as_ptr(view)),
                view.length() != p.length,
            ];
            if let Some(first) = out.iter().position(|&o| o) {
                left[first + 1] += 1;
                continue;
            }
            left[14] += 1;
            let made = p.made.unwrap_or(now);
            let candidate = (block, Arc::clone(view), p.runs.clone(), p.length);
            by_context
                .entry(p.context)
                .or_default()
                .push((made, candidate));
        }
        left[15] = self.cache.len().to_u64();
        reims_vgpu_vulkan::core::trace::event("regcand", &left);
        let mut out = Vec::new();
        for (context, mut found) in by_context {
            found.sort_by_key(|(made, c)| (made.as_nanos(), c.0));
            let mut open: Vec<Candidate<P>> = Vec::new();
            let (mut runs, mut bytes) = (0, 0);
            for (_, c) in found {
                let (r, b) = (c.2.len().to_u64(), c.3);
                if r > limits.runs || b > limits.bytes {
                    continue;
                }
                if runs + r > limits.runs || bytes + b > limits.bytes {
                    if open.len() > 1 {
                        out.push((context, std::mem::take(&mut open)));
                    }
                    open.clear();
                    (runs, bytes) = (0, 0);
                }
                (runs, bytes) = (runs + r, bytes + b);
                open.push(c);
            }
            if open.len() > 1 {
                out.push((context, open));
            }
        }
        out
    }

    /// A pass, on the surface worker ([`Self::regroup_due`]): each context's regroups, their
    /// views mapped and their imports made here, with `views` unlocked, handed to the context's
    /// ring-0 lane ([`super::super::GraphicsDecoder::move_regroups`]).
    pub(crate) fn regroup_pass(
        views: &Mutex<SurfaceViews<P>>,
        platform: &P,
        device: &P::Device,
    ) -> RaiseResult<()> {
        let clock = platform.clock();
        let now = clock.now_mono();
        let limits = platform.vm().regroup_limits();
        let found = match limits {
            Some(limits) => views.lock().regroup_candidates(now, limits),
            None => Vec::new(),
        };
        let mut made = Vec::new();
        for (context, members) in found {
            let runs: Vec<(u64, u64)> = members.iter().flat_map(|m| m.2.iter().copied()).collect();
            let length: u64 = members.iter().map(|m| m.3).sum();
            Self::make_room_for(views, platform, runs.len());
            let Ok(view) = Self::map_runs(views, &runs, length) else {
                continue;
            };
            let Some(anchor) = buffer_over(device, &view, 0, length, 0, None)? else {
                continue;
            };
            let mut at = 0;
            let members = (members.into_iter())
                .map(|(block, old, _, bytes)| {
                    let member = (block, old, at);
                    at += bytes;
                    member
                })
                .collect();
            let record = Arc::new(Record {
                bytes: length,
                live: AtomicU64::new(0),
            });
            made.push((
                context,
                Prepared {
                    view,
                    anchor,
                    record,
                    members,
                },
            ));
        }
        let bytes: u64 = made.iter().map(|(_, p)| p.record.bytes).sum();
        let took = clock.now_mono().saturating_duration_since(now);
        // The pass's regroups, their bytes and its time on the surface worker, for the trace: the
        // first submission after it makes the device map its imports.
        reims_vgpu_vulkan::core::trace::event(
            "regpass",
            &[
                made.len().to_u64(),
                bytes,
                u64::try_from(took.as_micros()).unwrap_or(u64::MAX),
            ],
        );
        let mut v = views.lock();
        v.regroups.passing = false;
        v.regroups.counts.passes += 1;
        for (context, prepared) in made {
            for (block, _, _) in &prepared.members {
                if let Some(p) = v.private.get_mut(block) {
                    p.queued = true;
                }
            }
            v.regroups.counts.made += 1;
            v.regroups.records.push(Arc::downgrade(&prepared.record));
            v.regroups.ready.entry(context).or_default().push(prepared);
        }
        Ok(())
    }
}

impl<P: Platform> super::super::GraphicsDecoder<P> {
    /// At a ring-0 message of `context`, at `now`: whether regroups of it wait for its lane, and
    /// whether a pass is due, which is taken as begun ([`SurfaceViews::regroup_due`]); neither
    /// where the platform makes no regroups.
    pub(crate) fn regroup_step(&self, context: u32, now: MonoTime) -> (bool, bool) {
        if self.platform.vm().regroup_limits().is_none() {
            return (false, false);
        }
        let mut v = self.shared.surface_views.lock();
        (v.regroups_ready(context, now), v.regroup_due(now))
    }

    /// A pass ([`SurfaceViews::regroup_pass`]) of the views, its imports made on `device`.
    pub(crate) fn regroup_pass(&self, device: &P::Device) -> RaiseResult<()> {
        SurfaceViews::regroup_pass(&self.shared.surface_views, &*self.platform, device)
    }

    /// The current context's regroups made ready, their members' buffers moved into each
    /// regroup's view, and its members' buffers refused before asked again, on its ring-0 lane at
    /// a message (the module doc), one context's at a time, until the message's budget is spent
    /// ([`REGROUP_BUDGET`]): what it did; `None` where nothing waited.
    pub(crate) fn move_regroups(&self) -> Option<Moves> {
        let context = self.context?;
        let clock = self.platform.clock();
        let start = clock.now_mono();
        let _one = self.shared.regroup_moves.lock();
        let (ready, retry, budget) = {
            let mut v = self.shared.surface_views.lock();
            let r = &mut v.regroups;
            let ready = r.ready.remove(&context).unwrap_or_default();
            let mut due = Vec::new();
            if let Some(blocks) = r.retry.get_mut(&context) {
                blocks.retain(|&(at, block)| {
                    let now = start.as_nanos() >= at.as_nanos();
                    if now {
                        due.push(block);
                    }
                    !now
                });
                if blocks.is_empty() {
                    r.retry.remove(&context);
                }
            }
            (ready, due, r.budget)
        };
        if ready.is_empty() && retry.is_empty() {
            return None;
        }
        // Spent once a member was taken up and the budget is gone.
        let spent = |handled: u64| {
            handled != 0 && clock.now_mono().saturating_duration_since(start) >= budget
        };
        let mut moves = Moves::default();
        let (mut again, mut rest) = (Vec::new(), Vec::new());
        let mut ready = ready.into_iter();
        while let Some(prepared) = ready.next() {
            let (asked, left) = self.move_regroup(prepared, &mut moves, &spent);
            again.extend(asked);
            if let Some(left) = left {
                rest.push(left);
                rest.extend(ready.by_ref());
            }
        }
        let mut later = Vec::new();
        let mut retry = retry.into_iter();
        while let Some(block) = retry.next() {
            if spent(moves.handled) {
                later.push(block);
                later.extend(retry.by_ref());
                break;
            }
            moves.handled += 1;
            again.extend(self.move_again(block, &mut moves));
        }
        moves.left += (rest.iter().map(|p| p.members.len()).sum::<usize>() + later.len()).to_u64();
        let now = clock.now_mono();
        moves.took = now.saturating_duration_since(start);
        let mut v = self.shared.surface_views.lock();
        if !rest.is_empty() {
            // Ahead of any a pass made meanwhile.
            let ready = v.regroups.ready.entry(context).or_default();
            rest.append(ready);
            *ready = rest;
        }
        if !(again.is_empty() && later.is_empty()) {
            let blocks = v.regroups.retry.entry(context).or_default();
            let next = now.saturating_add(REGROUP_EVERY);
            blocks.extend(again.into_iter().map(|block| (next, block)));
            blocks.extend(later.into_iter().map(|block| (start, block)));
        }
        let us = u64::try_from(moves.took.as_micros()).unwrap_or(u64::MAX);
        v.regroups.counts.most_us = v.regroups.counts.most_us.max(us);
        Some(moves)
    }

    /// Moves the objects `ids` the member at `block` made over its view at `old` into `view`, its
    /// place in its regroup's view, each that is still the one its `BUFFER_OVER_PAGES` made (its
    /// span the block's, its bytes in that view): how many moved, the ids refused, and the ids
    /// found.
    fn move_buffers(
        &self,
        (block, block_size): (u64, u64),
        view: &Arc<SurfaceView<P>>,
        old: HostPtr,
        ids: &[u64],
    ) -> (u64, Vec<u64>, Vec<u64>) {
        let (mut moved, mut refused, mut found) = (0, Vec::new(), Vec::new());
        for &id in ids {
            let over = self
                .entry(id)
                .and_then(|e| match (&e.handle, e.span, &e.bytes) {
                    (ObjHandle::Buffer(b), Some(span), Some(bytes))
                        if span == (block, block_size) && bytes.ram == Some(old) =>
                    {
                        Some((b.clone_ref(), bytes.at, bytes.length))
                    }
                    _ => None,
                });
            let Some((buffer, at, length)) = over else {
                continue;
            };
            found.push(id);
            match rebind::<P>(&buffer, view, at, length) {
                Ok(()) => moved += 1,
                Err(why) => {
                    log::debug!(
                        target: "reims_vgpu::graphics::surface_pages",
                        "object {id:#x} over the pages block at {block:#x} stays over its view, \
                         not moved into a regroup's view: {why}"
                    );
                    refused.push(id);
                }
            }
        }
        (moved, refused, found)
    }

    /// One regroup's members moved ([`Self::move_regroups`]) until `spent` answers so for the
    /// members taken up: the blocks with buffers to be asked again, and the regroup with the
    /// members left.
    fn move_regroup(
        &self,
        prepared: Prepared<P>,
        moves: &mut Moves,
        spent: &dyn Fn(u64) -> bool,
    ) -> (Vec<u64>, Option<Prepared<P>>) {
        let Prepared {
            view,
            anchor,
            record,
            members,
        } = prepared;
        let mut again = Vec::new();
        let mut members = members.into_iter();
        while let Some((block, old, at)) = members.next() {
            if spent(moves.handled) {
                let mut left = vec![(block, old, at)];
                left.extend(members);
                let prepared = Prepared {
                    view,
                    anchor,
                    record,
                    members: left,
                };
                return (again, Some(prepared));
            }
            moves.handled += 1;
            // The block as the pass found it, else it is left out; taken up, no longer queued.
            let taken = {
                let mut v = self.shared.surface_views.lock();
                (v.private.get_mut(&block)).and_then(|p| {
                    p.queued = false;
                    let same = (p.view.as_ref()).is_some_and(|(made, _)| Arc::ptr_eq(made, &old));
                    (same && p.regrouped.is_none()).then(|| {
                        let import =
                            (p.view.as_ref()).and_then(|(_, i)| i.as_ref().map(|i| i.clone_ref()));
                        (import, p.buffers.clone(), p.length, p.block_size)
                    })
                })
            };
            let Some((Some(import), ids, length, block_size)) = taken else {
                self.shared.surface_views.lock().regroups.counts.left += 1;
                continue;
            };
            let slice = Arc::new(SurfaceView::slice(&view, at, length));
            // The view's import buffer first: refused, the block stays as it was, its pages in
            // the regroup's view those of a member gone, and no regroup takes it again.
            moves.rebinds += 1;
            if let Err(why) = rebind::<P>(&import, &slice, 0, length) {
                log::debug!(
                    target: "reims_vgpu::graphics::surface_pages",
                    "the pages block at {block:#x} stays out of regroups: its view's import \
                     buffer was not moved into a regroup's view: {why}"
                );
                let mut v = self.shared.surface_views.lock();
                v.regroups.counts.refused += 1;
                v.regroups.counts.left += 1;
                if let Some(p) = v.private.get_mut(&block) {
                    p.stays = true;
                }
                continue;
            }
            let (moved, refused, found) =
                self.move_buffers((block, block_size), &slice, old.base(), &ids);
            moves.rebinds += found.len().to_u64();
            record.live.fetch_add(length, Ordering::Relaxed);
            let mut v = self.shared.surface_views.lock();
            let c = &mut v.regroups.counts;
            c.members += 1;
            c.moved += moved + 1;
            c.refused += refused.len().to_u64();
            match v.private.get_mut(&block) {
                Some(p)
                    if p.view
                        .as_ref()
                        .is_some_and(|(made, _)| Arc::ptr_eq(made, &old)) =>
                {
                    if !refused.is_empty() {
                        again.push(block);
                    }
                    p.buffers = found;
                    p.regrouped = Some(Member {
                        record: Arc::clone(&record),
                        bytes: length,
                        pending: refused,
                        tries: 1,
                    });
                    if let Some((made, _)) = p.view.as_mut() {
                        let was = std::mem::replace(made, slice);
                        p.retired.push(was);
                    }
                }
                // Released meanwhile: its bytes are a member's gone.
                _ => drop(Member {
                    record: Arc::clone(&record),
                    bytes: length,
                    pending: Vec::new(),
                    tries: 1,
                }),
            }
        }
        // The members' buffers hold the import from now on.
        drop(anchor);
        drop(view);
        (again, None)
    }

    /// The objects of the member at `block` a move left where they lay, asked to move again
    /// ([`Self::move_regroups`]): the block where some are to be asked once more.
    fn move_again(&self, block: u64, moves: &mut Moves) -> Option<u64> {
        let taken = {
            let mut v = self.shared.surface_views.lock();
            let p = v.private.get_mut(&block)?;
            let view = Arc::clone(&p.view.as_ref()?.0);
            let old = p.retired.first()?.base();
            let member = p.regrouped.as_mut()?;
            member.tries += 1;
            let pending = std::mem::take(&mut member.pending);
            (view, old, pending, p.block_size, member.tries)
        };
        let (view, old, pending, block_size, tries) = taken;
        let (moved, refused, found) = self.move_buffers((block, block_size), &view, old, &pending);
        moves.rebinds += found.len().to_u64();
        let mut v = self.shared.surface_views.lock();
        let c = &mut v.regroups.counts;
        c.retried += moved;
        c.refused += refused.len().to_u64();
        let last = tries >= REGROUP_TRIES;
        if last {
            c.stayed += refused.len().to_u64();
        }
        let p = v.private.get_mut(&block)?;
        let member = p.regrouped.as_mut()?;
        if last || refused.is_empty() {
            return None;
        }
        member.pending = refused;
        Some(block)
    }
}

use reims_vgpu_vulkan::core::num::ToU64 as _;
