//! Publishing and announcing FIFO completion stamps off the drain worker.
//!
//! # What the guest is actually waiting on
//!
//! `IOGPUEventMachine::waitForStamp(index, target)` in the guest reads the stamp
//! word **straight out of the page this device writes** — a signed
//! `target - current <= 0`, so it is wrap-safe — and if the target has not been
//! reached it builds a **one-second** deadline, sleeps on the stamp word's own
//! address as the wait channel, and re-reads the word on every wake.
//!
//! Two things follow:
//!
//! * The word is the authority. This module stores it only after the timeline
//!   point of the FIFO work it represents has completed.
//! * **The interrupt is the wakeup, not a hint.** Nothing re-checks the word
//!   until something wakes the thread, so a late interrupt is not a late
//!   notification, it is up to a full second of guest stall. An earlier attempt
//!   deferred the announcement to the drain worker's next tranche and measured
//!   exactly that: draws/s 3237 -> 2, presents/s 45 -> 1.
//!
//! The completion thread therefore waits the queue's monotonic timeline,
//! release-stores the shared word, and immediately raises the interrupt. The
//! drain worker only enqueues the checked word and returns.
//!
//! # Why a thread, and why it needs nothing from the device lock
//!
//! The announcement is three operations, and the device already does all three
//! off the drain worker for display VBL (`device::vbl_contended_pulse`):
//! `fetch_or` on the `Arc<AtomicU32>` clone of the interrupt-status register,
//! a push onto the lock-free `prompt_actions` queue, and `notify_actions` —
//! which the ABI documents as safe from any thread. The prompt rail exists for
//! precisely this: its own doc says it is there "so a guest ISR sees its
//! stamp-completion MSI while the drain worker is still rendering later
//! packets".
//!
//! This module owns none of that. It takes an [`AnnounceStamp`] hook the device
//! layer installs, so the engine keeps knowing nothing about `BoundDevice`.
//!
//! # Why a timeline semaphore rather than a fence
//!
//! A second waiter cannot use the ring's fences. `ResourcePools` owns every one
//! of them and resets each at retire, so a thread waiting on a ring fence races
//! the reset and a submission that signalled once can read as unsignalled
//! forever. Giving completion tracking its own fence instead breaks the ring
//! the other way: `vkQueueSubmit` takes exactly one fence, so the slot's fence
//! would never signal and its cleanup would never retire.
//!
//! A timeline semaphore has neither problem. It is signalled *in addition* to
//! the slot's fence, its value is monotonic so nothing has to be reset, and
//! `vkWaitSemaphores` may be called from any thread. Core in Vulkan 1.2, which
//! is this backend's baseline — and still gated, because the fallback is simply
//! the blocking rail every host used before this existed.

use ash::vk;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Pending completions owned by one FIFO before its producer must wait.
///
/// This is the FIFO contract's queue depth, not a tuning knob. Keeping the
/// bound per FIFO matters: pressure on one channel must not consume another
/// channel's completion capacity.
const FIFO_PENDING_STAMP_CAPACITY: usize = 32;

/// One vGPU session's lock-free completion projections.
///
/// The drain worker needs this before taking the engine lock: a CPU-only stamp
/// still has to join the completion queue when an older stamp for the same FIFO
/// is pending, or it can publish ahead and the older completion later moves the
/// guest's fence backwards. The physical queue and completion worker are shared,
/// but a guest FIFO belongs to exactly one vGPU session.
#[derive(Default)]
pub(super) struct SessionState {
    pending_fifo_mask: AtomicU32,
    unsubmitted_fifo_mask: AtomicU32,
    announce: Mutex<Option<AnnounceStamp>>,
}

impl std::fmt::Debug for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionState")
            .field(
                "pending_fifo_mask",
                &self.pending_fifo_mask.load(Ordering::Relaxed),
            )
            .field(
                "unsubmitted_fifo_mask",
                &self.unsubmitted_fifo_mask.load(Ordering::Relaxed),
            )
            .field(
                "announce_installed",
                &self
                    .announce
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_some(),
            )
            .finish()
    }
}

const _: () = assert!(reims_vgpu_core::MAX_CHANNELS <= u32::BITS as usize);

pub(super) fn fifo_has_pending_stamp(state: &SessionState, index: u32) -> bool {
    index < u32::BITS && state.pending_fifo_mask.load(Ordering::Acquire) & (1u32 << index) != 0
}

/// The subset of the session's pending mask whose completion point is a submission
/// **this device has not made yet** — a stamp registered against the open
/// batch's future point by `StampCompletion::queue_for_next_submission`.
///
/// The distinction is the whole difference between a guest waiting on the GPU
/// and a guest waiting on *us*. A `Submitted` stamp is in flight and nothing can
/// make it land sooner. A `NextSubmission` stamp is a word we have promised and
/// then parked in a command buffer that is still recording, and the batch has no
/// time bound — it stays open until a draw claims a slot, a readback arrives, a
/// present runs, or the pending ring fills. So a timeline blocked on one is
/// blocked until unrelated work happens to arrive, which on a quiet channel can
/// be tens of milliseconds.
/// Whether FIFO `index` is owed a stamp that is parked on an unsubmitted batch.
pub(super) fn fifo_has_unsubmitted_stamp(state: &SessionState, index: u32) -> bool {
    index < u32::BITS && state.unsubmitted_fifo_mask.load(Ordering::Acquire) & (1u32 << index) != 0
}

/// Raise the guest-visible interrupt for a completed stamp slot.
///
/// Installed by the device layer, which owns the interrupt-status clone and the
/// prompt action queue. Called from the completion thread with no lock of this
/// crate's held, so an implementation must not reach for the device lock.
pub type AnnounceStamp = Arc<dyn Fn(u32) + Send + Sync>;

/// Install the hook the completion thread announces through. Idempotent; the
/// last caller wins, which is what a device rebind wants.
///
/// There is no uninstall. The hook the device layer installs resolves its
/// device by id every time it is called, so one left behind by a torn-down
/// device holds nothing and announces nothing — an uninstall would only be a
/// second way to reach the same state, and a race against a completion already
/// in flight.
pub(super) fn install_announce(state: &SessionState, hook: AnnounceStamp) {
    *state.announce.lock().unwrap_or_else(|e| e.into_inner()) = Some(hook);
}

fn announce(waiting: &Waiting) {
    let hook = waiting
        .signals
        .announce
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(Arc::clone);
    match hook {
        // Called with no lock of this module's held: the hook reaches the
        // device's prompt queue, and holding the session hook mutex across it
        // would put this module's mutex under the device's.
        Some(hook) => hook(waiting.index),
        None => reims_vgpu_observe::fail(format!(
            "stamp_announce_no_hook reason=stamp_announce_no_hook index={} \
             (a stamp completed with no device bound to raise its interrupt; the guest \
             waiting on it will sleep to its one-second deadline)",
            waiting.index
        )),
    }
}

/// One stamp waiting for its queue point, in FIFO order.
#[derive(Clone, Debug)]
struct Waiting {
    session: super::SessionId,
    signals: Arc<SessionState>,
    /// The exact submission this completion belongs to, before or after that
    /// submission has reached `vkQueueSubmit`.
    point: CompletionPoint,
    /// Stamp slot index, for the interrupt-status bit.
    index: u32,
    /// The checked shared-memory word written before the interrupt is raised.
    word: reims_vgpu_memory::GuestRef,
    /// The FIFO completion value published into `word`.
    stamp: u32,
    /// When this stamp was registered, for the publish-latency census.
    ///
    /// This is the clock the guest actually experiences: it is blocked from the
    /// moment the packet is held on the wait until the word appears, and every
    /// hop in between — the batch reaching `vkQueueSubmit`, the GPU retiring it,
    /// the completion thread waking, the store, the interrupt — is inside this
    /// span. Nothing else in this device measures it end to end, and the drain's
    /// own censuses cannot: they are written by the drain thread, which is not
    /// the thread that finishes the work.
    queued_at: std::time::Instant,
}

/// Association between one guest stamp and one FIFO-owned submission.
///
/// A stamp recorded in an open batch already knows which monotonic point the
/// next reservation will receive. Carrying that point here prevents an older,
/// delayed `vkQueueSubmit` from claiming stamps recorded for a newer batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompletionPoint {
    NextSubmission(u64),
    Submitted(u64),
}

#[derive(Default)]
struct PendingQueue {
    waiting: std::collections::VecDeque<Waiting>,
    per_fifo: std::collections::HashMap<(super::SessionId, u32), usize>,
}

impl PendingQueue {
    fn is_full(&self, session: super::SessionId, index: u32) -> bool {
        self.per_fifo.get(&(session, index)).copied().unwrap_or(0) == FIFO_PENDING_STAMP_CAPACITY
    }

    fn push(&mut self, waiting: Waiting) {
        *self
            .per_fifo
            .entry((waiting.session, waiting.index))
            .or_default() += 1;
        self.waiting.push_back(waiting);
    }

    fn pop_front(&mut self) -> Option<Waiting> {
        let waiting = self.waiting.pop_front()?;
        let key = (waiting.session, waiting.index);
        if let Some(count) = self.per_fifo.get_mut(&key) {
            *count -= 1;
            if *count == 0 {
                self.per_fifo.remove(&key);
            }
        }
        Some(waiting)
    }

    fn has_pending(&self, session: super::SessionId, index: u32) -> bool {
        self.per_fifo.contains_key(&(session, index))
    }

    fn bind_submission(&mut self, timeline: u64) -> usize {
        let mut bound = 0;
        for waiting in &mut self.waiting {
            if waiting.point == CompletionPoint::NextSubmission(timeline) {
                waiting.point = CompletionPoint::Submitted(timeline);
                bound += 1;
            }
        }
        self.republish_unsubmitted();
        bound
    }

    /// Recompute the lock-free projection of which FIFOs still have a stamp
    /// parked on an unsubmitted batch.
    ///
    /// Recomputed from the queue rather than decremented, because one
    /// `bind_submission` can promote several of a FIFO's stamps at once while
    /// leaving others — belonging to a *later* still-open batch — behind, and a
    /// counter stepped per promotion would clear the bit while one of those is
    /// still parked.
    fn republish_unsubmitted(&self) {
        let mut masks: std::collections::HashMap<usize, (Arc<SessionState>, u32)> =
            std::collections::HashMap::new();
        for waiting in &self.waiting {
            let key = Arc::as_ptr(&waiting.signals) as usize;
            let (_, mask) = masks
                .entry(key)
                .or_insert_with(|| (Arc::clone(&waiting.signals), 0));
            if matches!(waiting.point, CompletionPoint::NextSubmission(_))
                && waiting.index < u32::BITS
            {
                *mask |= 1u32 << waiting.index;
            }
        }
        for (_, (signals, mask)) in masks {
            signals.unsubmitted_fifo_mask.store(mask, Ordering::Release);
        }
    }
}

/// The queue the drain worker pushes to and the completion thread drains.
struct Shared {
    queue: Mutex<PendingQueue>,
    /// Woken by a push and by shutdown. The thread waits here only when the
    /// queue is empty; otherwise it is blocked in `vkWaitSemaphores`, which is
    /// where it should be.
    wake: Condvar,
    stop: AtomicBool,
    /// Highest value handed out. The drain worker reserves with `fetch_add`
    /// under the engine lock, so reservation order is submission order.
    next_value: AtomicU64,
    /// Highest timeline point successfully handed to the ordered queue owner.
    ///
    /// A completion stamp may be recorded after the handoff but before the
    /// owner enters `vkQueueSubmit`. It must wait this point, not the preceding
    /// submitted one, or the guest can reuse shared inputs while their reader
    /// is still in the host FIFO.
    latest_queued: AtomicU64,
}

impl Shared {
    /// Point an open batch will receive when it is handed to the queue.
    fn next_submission(&self) -> u64 {
        self.next_value.load(Ordering::Acquire) + 1
    }

    /// Reserve the point previously advertised by [`Self::next_submission`].
    fn reserve_submission(&self) -> u64 {
        self.next_value.fetch_add(1, Ordering::AcqRel) + 1
    }

    fn latest_queued(&self) -> Option<u64> {
        let value = self.latest_queued.load(Ordering::Acquire);
        (value != 0).then_some(value)
    }
}

/// Cloneable publication half handed to the queue owner with one submission.
/// It cannot stop the completion thread or destroy its semaphore; it can only
/// bind pending guest stamps after `vkQueueSubmit` has actually succeeded.
#[derive(Clone)]
pub(crate) struct SubmissionNote {
    shared: Arc<Shared>,
}

#[cfg(test)]
pub(super) struct SubmissionProbe {
    shared: Arc<Shared>,
}

#[cfg(test)]
impl SubmissionProbe {
    pub(super) fn new() -> Self {
        Self {
            shared: Arc::new(Shared {
                queue: Mutex::new(PendingQueue::default()),
                wake: Condvar::new(),
                stop: AtomicBool::new(false),
                next_value: AtomicU64::new(0),
                latest_queued: AtomicU64::new(0),
            }),
        }
    }

    pub(super) fn note(&self) -> SubmissionNote {
        SubmissionNote {
            shared: Arc::clone(&self.shared),
        }
    }

    pub(super) fn latest_queued(&self) -> Option<u64> {
        self.shared.latest_queued()
    }
}

impl SubmissionNote {
    pub(crate) fn queued(&self, value: u64) {
        self.shared
            .latest_queued
            .fetch_max(value, Ordering::Release);
    }

    pub(crate) fn submitted(&self, value: u64) {
        self.queued(value);
        let bound = self
            .shared
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .bind_submission(value);
        if bound != 0 {
            self.shared.wake.notify_all();
        }
    }
}

/// A running completion thread, owned by the device context.
pub(crate) struct StampCompletion {
    shared: Arc<Shared>,
    semaphore: vk::Semaphore,
    join: Option<std::thread::JoinHandle<()>>,
}

impl StampCompletion {
    /// Create the semaphore and start the thread.
    ///
    /// `device` is cloned into the thread. `ash::Device` is a handle plus a
    /// function-pointer table, and the two entry points the thread calls —
    /// `vkWaitSemaphores` and `vkGetSemaphoreCounterValue` — are not externally
    /// synchronized against anything the drain worker does to this semaphore
    /// (only signalling is, and only the queue signals it). What *is* required
    /// is that the thread stop before `vkDestroyDevice`, which [`Self::stop`]
    /// guarantees and `DeviceContext::destroy` calls.
    ///
    /// # Safety
    ///
    /// `device` must outlive the returned value's [`Self::stop`].
    pub(crate) unsafe fn start(device: &ash::Device) -> Result<Self, vk::Result> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let ci = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        let semaphore = unsafe { device.create_semaphore(&ci, None) }?;
        let shared = Arc::new(Shared {
            queue: Mutex::new(PendingQueue::default()),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            next_value: AtomicU64::new(0),
            latest_queued: AtomicU64::new(0),
        });
        let thread_shared = Arc::clone(&shared);
        let thread_device = device.clone();
        let join = std::thread::Builder::new()
            .name("reims-vgpu-stamp".into())
            .spawn(move || run(&thread_device, semaphore, &thread_shared))
            .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
        Ok(Self {
            shared,
            semaphore,
            join: Some(join),
        })
    }

    /// Reserve the timeline point for one FIFO-owned queue submission.
    ///
    /// Reserved under the engine lock, so the values are handed out in
    /// submission order — which is what makes a single-threaded drain of the
    /// queue announce stamps in that same order without any further ordering
    /// machinery.
    pub(crate) fn reserve_submission(&self) -> (vk::Semaphore, u64, SubmissionNote) {
        let value = self.shared.reserve_submission();
        (
            self.semaphore,
            value,
            SubmissionNote {
                shared: Arc::clone(&self.shared),
            },
        )
    }

    /// The newest queue point this device has accepted, including work waiting
    /// in the ordered owner's host FIFO.
    pub(crate) fn latest_queued(&self) -> Option<(vk::Semaphore, u64)> {
        self.shared
            .latest_queued()
            .map(|value| (self.semaphore, value))
    }

    /// Retire one FIFO completion after `timeline` completes.
    pub(super) fn wait_for_stamp(
        &self,
        session: super::SessionId,
        signals: Arc<SessionState>,
        timeline: u64,
        index: u32,
        word: reims_vgpu_memory::GuestRef,
        stamp: u32,
    ) {
        if index as usize >= reims_vgpu_core::MAX_CHANNELS {
            reims_vgpu_observe::fail(format!(
                "stamp_fifo_out_of_range reason=stamp_fifo_out_of_range index={index} \
                 max_channels={}",
                reims_vgpu_core::MAX_CHANNELS
            ));
            return;
        }
        let mut queue = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        while queue.is_full(session, index) && !self.shared.stop.load(Ordering::Acquire) {
            queue = self
                .shared
                .wake
                .wait(queue)
                .unwrap_or_else(|e| e.into_inner());
        }
        if self.shared.stop.load(Ordering::Acquire) {
            return;
        }
        queue.push(Waiting {
            session,
            signals: Arc::clone(&signals),
            point: CompletionPoint::Submitted(timeline),
            index,
            word,
            stamp,

            queued_at: std::time::Instant::now(),
        });
        signals
            .pending_fifo_mask
            .fetch_or(1u32 << index, Ordering::Release);
        drop(queue);
        self.shared.wake.notify_one();
    }

    /// Register a stamp behind the command buffer that is still recording.
    ///
    /// Returns `false` instead of blocking when this FIFO's contract-sized
    /// pending ring is full. The caller owns the open command buffer, so
    /// sleeping there would prevent the very submission that can make room;
    /// it must submit the batch and retry against that concrete point.
    pub(super) fn queue_for_next_submission(
        &self,
        session: super::SessionId,
        signals: Arc<SessionState>,
        index: u32,
        word: reims_vgpu_memory::GuestRef,
        stamp: u32,
    ) -> bool {
        if index as usize >= reims_vgpu_core::MAX_CHANNELS {
            reims_vgpu_observe::fail(format!(
                "stamp_fifo_out_of_range reason=stamp_fifo_out_of_range index={index} \
                 max_channels={}",
                reims_vgpu_core::MAX_CHANNELS
            ));
            return false;
        }
        let mut queue = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        if queue.is_full(session, index) || self.shared.stop.load(Ordering::Acquire) {
            return false;
        }
        // The engine lock serializes this read with reservation. No other
        // FIFO-owned submission can reserve between recording this stamp and
        // flushing its open batch, so this is the batch's exact future point.
        let target = self.shared.next_submission();
        queue.push(Waiting {
            session,
            signals: Arc::clone(&signals),
            point: CompletionPoint::NextSubmission(target),
            index,
            word,
            stamp,

            queued_at: std::time::Instant::now(),
        });
        signals
            .pending_fifo_mask
            .fetch_or(1u32 << index, Ordering::Release);
        signals
            .unsubmitted_fifo_mask
            .fetch_or(1u32 << index, Ordering::Release);
        drop(queue);
        self.shared.wake.notify_one();
        true
    }

    /// Wait until this FIFO has no queued completion word left to publish.
    ///
    /// Used only before the CPU fallback writes a newer value. GPU work may
    /// already be settled while its completion worker has not yet stored the
    /// older word; waiting on the GPU alone would still permit that older store
    /// to land after the fallback and move the guest's fence backwards.
    pub(crate) fn wait_for_fifo_idle(&self, session: super::SessionId, index: u32) {
        if index as usize >= reims_vgpu_core::MAX_CHANNELS {
            return;
        }
        let mut queue = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        while queue.has_pending(session, index) && !self.shared.stop.load(Ordering::Acquire) {
            queue = self
                .shared
                .wake
                .wait(queue)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Stop the thread and destroy the semaphore.
    ///
    /// Must run before `vkDestroyDevice`: the thread holds a cloned `ash::Device`
    /// and is blocked inside it.
    ///
    /// # Safety
    ///
    /// `device` must be the device this was started with, and must not yet be
    /// destroyed.
    pub(crate) unsafe fn stop(&mut self, device: &ash::Device) {
        self.shared.stop.store(true, Ordering::Release);
        // Signal past every reserved value so a thread blocked in
        // `vkWaitSemaphores` returns rather than sitting out its deadline. The
        // thread observes `stop` after the wait and never publishes a word for
        // work this host signal merely skipped over.
        let past_everything = self.shared.next_value.load(Ordering::Acquire) + 1;
        let signal = vk::SemaphoreSignalInfo::default()
            .semaphore(self.semaphore)
            .value(past_everything);
        let _ = unsafe { device.signal_semaphore(&signal) };
        self.shared.wake.notify_all();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
        let queue = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        for waiting in &queue.waiting {
            waiting
                .signals
                .pending_fifo_mask
                .store(0, Ordering::Release);
            waiting
                .signals
                .unsubmitted_fifo_mask
                .store(0, Ordering::Release);
        }
        drop(queue);
        unsafe { device.destroy_semaphore(self.semaphore, None) };
    }
}

/// The completion thread.
///
/// Blocks in `vkWaitSemaphores` while there is a stamp outstanding and on the
/// condvar while there is not, so it costs nothing when the guest is idle and
/// adds no latency when it is not.
fn run(device: &ash::Device, semaphore: vk::Semaphore, shared: &Shared) {
    loop {
        let next = {
            let mut queue = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if shared.stop.load(Ordering::Acquire) {
                    return;
                }
                if let Some(front) = queue.waiting.front().cloned() {
                    if matches!(front.point, CompletionPoint::Submitted(_)) {
                        break Some(front);
                    }
                }
                let (guard, _) = shared
                    .wake
                    .wait_timeout(queue, std::time::Duration::from_millis(250))
                    .unwrap_or_else(|e| e.into_inner());
                queue = guard;
            }
        };
        let Some(waiting) = next else {
            return;
        };
        let semaphores = [semaphore];
        let CompletionPoint::Submitted(timeline) = waiting.point else {
            unreachable!("front was checked as submitted")
        };
        let values = [timeline];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // The same deadline every blocking wait in this backend uses. It is a
        // diagnostic deadline, not permission to discard the association: the
        // guest-visible word means this exact submission completed. Keep the
        // entry at the head and retry until that becomes true.
        let wait = unsafe { device.wait_semaphores(&info, super::context::FENCE_TIMEOUT_NS) };
        match classify_wait(wait, shared.stop.load(Ordering::Acquire)) {
            CompletionWait::Completed => {}
            CompletionWait::Retry => {
                if reims_vgpu_observe::first_sight("stamp_wait_timeout", timeline) {
                    reims_vgpu_observe::fail(format!(
                        "stamp_wait_timeout reason=stamp_wait_timeout index={} value={} \
                         (the submission carrying this stamp's word has not executed within the \
                         fence deadline; its completion remains pending)",
                        waiting.index, timeline
                    ));
                    // Join on the queue's timeline value, not `waiting.index`:
                    // the latter names the guest FIFO whose word is waiting,
                    // while the former is the exact Vulkan submission that
                    // must signal. Treating a FIFO ordinal as a ring slot would
                    // produce plausible-looking evidence for unrelated work.
                    match crate::gpu_hang_trail::submission_for_timeline(timeline) {
                        Some((slot, submission)) => reims_vgpu_observe::fail(format!(
                            "stamp_wait_timeout_submission reason=stamp_wait_timeout \
                             index={} value={} slot={} held=[{}]",
                            waiting.index, timeline, slot, submission
                        )),
                        None => reims_vgpu_observe::fail(format!(
                            "stamp_wait_timeout_submission reason=stamp_wait_timeout \
                             index={} value={} held=none \
                             (no live submission-ring entry carries this timeline point)",
                            waiting.index, timeline
                        )),
                    }
                    if let Some(outstanding) = crate::gpu_hang_trail::outstanding() {
                        reims_vgpu_observe::fail(format!(
                            "stamp_wait_timeout_queue reason=stamp_wait_timeout index={} \
                             value={} {outstanding}",
                            waiting.index, timeline
                        ));
                    }
                    if let Some(trail) = crate::gpu_hang_trail::trail() {
                        reims_vgpu_observe::fail(format!(
                            "stamp_wait_timeout_trail reason=stamp_wait_timeout index={} \
                             value={} {trail}",
                            waiting.index, timeline
                        ));
                    }
                    if let Some(firsts) = crate::gpu_hang_trail::recent_pipeline_firsts() {
                        reims_vgpu_observe::fail(format!(
                            "stamp_wait_timeout_pipes reason=stamp_wait_timeout index={} \
                             value={} {firsts}",
                            waiting.index, timeline
                        ));
                    }
                }
                continue;
            }
            CompletionWait::Failed(e) => {
                reims_vgpu_observe::fail(format!(
                    "stamp_wait_failed reason=stamp_wait_failed index={} value={} err={e:?} \
                     (the completion remains unpublished)",
                    waiting.index, timeline
                ));
                // This thread may not take the engine lock — it exists to
                // announce guest fences while the drain worker holds it — so it
                // latches the loss and the drain's end-of-tranche flush runs the
                // established context recovery. No later stamp may pass this
                // missing point in the meantime.
                super::device_lost::note_device_lost_seen();
                return;
            }
            CompletionWait::Stopping => return,
        }
        if !publish_stamp_word(&waiting) {
            reims_vgpu_observe::fail(format!(
                "stamp_cpu_store_failed reason=stamp_cpu_store_failed index={} value={:#x} \
                 (the completed queue point could not publish its checked shared word; \
                 its interrupt was withheld)",
                waiting.index, waiting.stamp
            ));
            return;
        }
        {
            let mut queue = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            queue.pop_front();
            if !queue.has_pending(waiting.session, waiting.index) {
                waiting
                    .signals
                    .pending_fifo_mask
                    .fetch_and(!(1u32 << waiting.index), Ordering::Release);
            }
        }
        shared.wake.notify_all();
        announce(&waiting);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompletionWait {
    Completed,
    Retry,
    Failed(vk::Result),
    Stopping,
}

fn classify_wait(result: Result<(), vk::Result>, stopping: bool) -> CompletionWait {
    if stopping {
        return CompletionWait::Stopping;
    }
    match result {
        Ok(()) => CompletionWait::Completed,
        Err(vk::Result::TIMEOUT) => CompletionWait::Retry,
        Err(error) => CompletionWait::Failed(error),
    }
}

fn publish_stamp_word(waiting: &Waiting) -> bool {
    note_publish_latency(waiting.queued_at.elapsed());
    waiting.word.store_u32_release(waiting.stamp)
}

/// Band how long a guest stamp took to become visible, from registration to the
/// word landing.
///
/// Banded rather than averaged because the question is a *distribution*: a mean
/// hides whether the guest is losing a little on every stamp or a lot on a few,
/// and those have different repairs. The bands straddle the frame period this
/// rail is judged against (a macos-13 boot runs ~29 Hz, so ~34 ms), so a stamp
/// landing in `lt64ms` has cost the guest most of a frame on its own.
///
/// The top two bands exist for one specific failure. The guest does not poll a
/// stamp slot: it sleeps on the slot's address and is woken by this device's
/// interrupt, with a **one-second deadline** as its only backstop. So a stamp
/// whose interrupt is dropped, or whose word becomes visible *after* the
/// interrupt rather than before it, is not slow by a little — the guest sleeps
/// out the full second and wakes to re-read the page. Any weight at `ge500ms` is
/// that signature and nothing else, and it would not be visible in a mean or in
/// any band that stopped at "slower than a frame".
fn note_publish_latency(elapsed: std::time::Duration) {
    let us = elapsed.as_micros() as u64;
    crate::telemetry::note_route(match us {
        0..=99 => "stamp_publish_lt100us",
        100..=999 => "stamp_publish_lt1ms",
        1_000..=3_999 => "stamp_publish_lt4ms",
        4_000..=15_999 => "stamp_publish_lt16ms",
        16_000..=63_999 => "stamp_publish_lt64ms",
        64_000..=499_999 => "stamp_publish_lt500ms",
        _ => "stamp_publish_ge500ms",
    });
    crate::telemetry::note_route_us("stamp_publish_us", us);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::SessionId;

    fn waiting(index: u32, stamp: u32) -> Waiting {
        let mut word = Box::new(0u32);
        let import = Arc::new(
            reims_vgpu_memory::GuestRamImport::new_host_allocation(
                (&mut *word) as *mut u32 as usize,
                std::mem::size_of::<u32>() as u64,
                std::mem::align_of::<u32>() as u64,
            )
            .expect("test import"),
        );
        // The import deliberately owns no allocation. Leak this one-word test
        // backing so every queued GuestRef remains valid for the test's life.
        Box::leak(word);
        let slice = import.slice(0, 4).expect("stamp word");
        Waiting {
            session: SessionId(1),
            signals: Arc::new(SessionState::default()),
            point: CompletionPoint::Submitted(u64::from(stamp) + 1),
            index,
            word: reims_vgpu_memory::GuestRef::new(import, slice).expect("guest word"),
            stamp,
            queued_at: std::time::Instant::now(),
        }
    }

    fn deferred(index: u32, stamp: u32, submission: u64) -> Waiting {
        let mut waiting = waiting(index, stamp);
        waiting.point = CompletionPoint::NextSubmission(submission);
        waiting
    }

    fn test_shared(next_value: u64) -> Arc<Shared> {
        Arc::new(Shared {
            queue: Mutex::new(PendingQueue::default()),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            next_value: AtomicU64::new(next_value),
            latest_queued: AtomicU64::new(0),
        })
    }

    fn points(shared: &Shared) -> Vec<CompletionPoint> {
        shared
            .queue
            .lock()
            .expect("pending queue")
            .waiting
            .iter()
            .map(|waiting| waiting.point)
            .collect()
    }

    #[test]
    fn pending_capacity_is_per_fifo_and_fifo_order_is_preserved() {
        let mut queue = PendingQueue::default();
        for stamp in 0..FIFO_PENDING_STAMP_CAPACITY as u32 {
            queue.push(waiting(0, stamp));
        }
        let session = SessionId(1);
        assert!(queue.is_full(session, 0));
        assert!(!queue.is_full(session, 1));
        queue.push(waiting(1, 0xfeed));

        for stamp in 0..FIFO_PENDING_STAMP_CAPACITY as u32 {
            let entry = queue.pop_front().expect("root completion");
            assert_eq!((entry.index, entry.stamp), (0, stamp));
        }
        assert!(!queue.has_pending(session, 0));
        assert!(queue.has_pending(session, 1));
        let child = queue.pop_front().expect("child completion");
        assert_eq!((child.index, child.stamp), (1, 0xfeed));
        assert!(!queue.has_pending(session, 1));
    }

    #[test]
    fn identical_fifo_ordinals_in_two_sessions_have_independent_pressure_and_projection() {
        let first = SessionId(1);
        let second = SessionId(2);
        let first_signals = Arc::new(SessionState::default());
        let second_signals = Arc::new(SessionState::default());
        let mut queue = PendingQueue::default();
        for stamp in 0..FIFO_PENDING_STAMP_CAPACITY as u32 {
            let mut entry = deferred(0, stamp, 7);
            entry.session = first;
            entry.signals = Arc::clone(&first_signals);
            queue.push(entry);
        }
        let mut other = deferred(0, 0xfeed, 9);
        other.session = second;
        other.signals = Arc::clone(&second_signals);
        queue.push(other);
        queue.republish_unsubmitted();

        assert!(queue.is_full(first, 0));
        assert!(!queue.is_full(second, 0));
        assert!(fifo_has_unsubmitted_stamp(&first_signals, 0));
        assert!(fifo_has_unsubmitted_stamp(&second_signals, 0));

        assert_eq!(queue.bind_submission(7), FIFO_PENDING_STAMP_CAPACITY);
        assert!(!fifo_has_unsubmitted_stamp(&first_signals, 0));
        assert!(fifo_has_unsubmitted_stamp(&second_signals, 0));
    }

    #[test]
    fn a_completion_announces_only_through_its_own_session_hook() {
        let first_count = Arc::new(AtomicU32::new(0));
        let second_count = Arc::new(AtomicU32::new(0));
        let first_signals = Arc::new(SessionState::default());
        let second_signals = Arc::new(SessionState::default());
        install_announce(&first_signals, {
            let count = Arc::clone(&first_count);
            Arc::new(move |_| {
                count.fetch_add(1, Ordering::Relaxed);
            })
        });
        install_announce(&second_signals, {
            let count = Arc::clone(&second_count);
            Arc::new(move |_| {
                count.fetch_add(1, Ordering::Relaxed);
            })
        });
        let mut completed = waiting(0, 1);
        completed.signals = first_signals;

        announce(&completed);

        assert_eq!(first_count.load(Ordering::Relaxed), 1);
        assert_eq!(second_count.load(Ordering::Relaxed), 0);
    }

    /// The unsubmitted projection clears only when a FIFO has no stamp left on
    /// an unmade submission — not merely when one of them is bound.
    ///
    /// A FIFO can hold stamps against two different open batches, and
    /// `bind_submission` promotes only the one whose point matches. Stepping a
    /// counter per promotion would clear the bit while the later batch's stamp
    /// is still parked, and a timeline blocked on *that* one would then never
    /// get its batch submitted. Hence the recompute.
    #[test]
    fn the_unsubmitted_projection_survives_a_partial_bind() {
        let mut queue = PendingQueue::default();

        // Two batches' worth of stamps on one FIFO, plus a sibling's.
        let mut early = waiting(1, 0x10);
        let signals = Arc::clone(&early.signals);
        early.point = CompletionPoint::NextSubmission(7);
        let mut late = waiting(1, 0x11);
        late.signals = Arc::clone(&signals);
        late.point = CompletionPoint::NextSubmission(9);
        let mut other = waiting(2, 0x20);
        other.signals = Arc::clone(&signals);
        other.point = CompletionPoint::NextSubmission(9);
        queue.push(early);
        queue.push(late);
        queue.push(other);
        queue.republish_unsubmitted();
        assert!(fifo_has_unsubmitted_stamp(&signals, 1));
        assert!(fifo_has_unsubmitted_stamp(&signals, 2));

        // Submitting batch 7 binds only the early one.
        assert_eq!(queue.bind_submission(7), 1);
        assert!(
            fifo_has_unsubmitted_stamp(&signals, 1),
            "FIFO 1 still has a stamp on batch 9, so its batch must still be \
             submittable on demand"
        );
        assert!(fifo_has_unsubmitted_stamp(&signals, 2));

        // Submitting batch 9 binds the rest, and both bits clear.
        assert_eq!(queue.bind_submission(9), 2);
        assert!(
            !fifo_has_unsubmitted_stamp(&signals, 1),
            "with everything in flight there is nothing left to submit early"
        );
        assert!(!fifo_has_unsubmitted_stamp(&signals, 2));
    }

    /// Reservation order is submission order; reserving alone does not publish
    /// a point that a completion stamp could observe.
    #[test]
    fn reservations_are_monotonic_and_handoff_is_published_separately() {
        let shared = Shared {
            queue: Mutex::new(PendingQueue::default()),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            next_value: AtomicU64::new(0),
            latest_queued: AtomicU64::new(0),
        };
        for n in 1u64..=100 {
            assert_eq!(shared.next_submission(), n);
            assert_eq!(
                shared.reserve_submission(),
                n,
                "the point recorded into an open-batch stamp is exactly the point subsequently reserved"
            );
        }
        assert_eq!(shared.latest_queued.load(Ordering::Acquire), 0);
    }

    #[test]
    fn queued_point_orders_a_stamp_before_driver_submission() {
        let shared = Arc::new(Shared {
            queue: Mutex::new(PendingQueue::default()),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            next_value: AtomicU64::new(1),
            latest_queued: AtomicU64::new(0),
        });
        let note = SubmissionNote {
            shared: Arc::clone(&shared),
        };

        note.queued(1);

        assert_eq!(shared.latest_queued.load(Ordering::Acquire), 1);
    }

    /// The initial value is 0 and the first reservation is 1, so no submission
    /// ever signals the value the semaphore was created at. A first reservation
    /// of 0 would be already-signalled at creation and its stamp would be
    /// announced before the GPU had run anything.
    #[test]
    fn no_reservation_can_collide_with_the_semaphores_initial_value() {
        let next = AtomicU64::new(0);
        assert_eq!(next.fetch_add(1, Ordering::AcqRel) + 1, 1);
    }

    #[test]
    fn completed_waiting_entry_publishes_its_word() {
        let mut words = [0u32; 2];
        let import = Arc::new(
            reims_vgpu_memory::GuestRamImport::new_host_allocation(
                words.as_mut_ptr() as usize,
                std::mem::size_of_val(&words) as u64,
                std::mem::align_of_val(&words) as u64,
            )
            .expect("test import"),
        );
        let slice = import.slice(4, 4).expect("second word");
        let word = reims_vgpu_memory::GuestRef::new(import, slice).expect("guest word");
        let waiting = Waiting {
            session: SessionId(1),
            signals: Arc::new(SessionState::default()),
            point: CompletionPoint::Submitted(7),
            index: 2,
            word,
            stamp: 0x89ab_cdef,
            queued_at: std::time::Instant::now(),
        };

        assert!(publish_stamp_word(&waiting));
        assert_eq!(words, [0, 0x89ab_cdefu32.to_le()]);
    }

    #[test]
    fn only_a_completed_wait_may_publish_and_retire_its_entry() {
        assert_eq!(classify_wait(Ok(()), false), CompletionWait::Completed);
        assert_eq!(
            classify_wait(Err(vk::Result::TIMEOUT), false),
            CompletionWait::Retry,
            "a diagnostic timeout retains the exact submission association"
        );
        assert_eq!(
            classify_wait(Err(vk::Result::ERROR_DEVICE_LOST), false),
            CompletionWait::Failed(vk::Result::ERROR_DEVICE_LOST)
        );
        assert_eq!(
            classify_wait(Ok(()), true),
            CompletionWait::Stopping,
            "the host signal used for teardown is not guest completion"
        );
    }

    #[test]
    fn an_open_batch_stamp_binds_only_when_submission_succeeds() {
        let shared = Arc::new(Shared {
            queue: Mutex::new(PendingQueue::default()),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            next_value: AtomicU64::new(0),
            latest_queued: AtomicU64::new(0),
        });
        let mut submitted = waiting(0, 1);
        submitted.point = CompletionPoint::Submitted(7);
        let mut deferred_a = waiting(0, 2);
        deferred_a.point = CompletionPoint::NextSubmission(11);
        let mut deferred_b = waiting(1, 3);
        deferred_b.point = CompletionPoint::NextSubmission(11);
        {
            let mut queue = shared.queue.lock().expect("pending queue");
            queue.push(submitted);
            queue.push(deferred_a);
            queue.push(deferred_b);
        }
        let note = SubmissionNote {
            shared: Arc::clone(&shared),
        };

        note.queued(11);
        let before_submit: Vec<CompletionPoint> = shared
            .queue
            .lock()
            .expect("pending queue")
            .waiting
            .iter()
            .map(|w| w.point)
            .collect();
        assert_eq!(
            before_submit,
            vec![
                CompletionPoint::Submitted(7),
                CompletionPoint::NextSubmission(11),
                CompletionPoint::NextSubmission(11),
            ]
        );

        note.submitted(11);

        let points: Vec<CompletionPoint> = shared
            .queue
            .lock()
            .expect("pending queue")
            .waiting
            .iter()
            .map(|w| w.point)
            .collect();
        assert_eq!(
            points,
            vec![
                CompletionPoint::Submitted(7),
                CompletionPoint::Submitted(11),
                CompletionPoint::Submitted(11),
            ]
        );
    }

    /// A delayed older submit must not claim stamps that the drain worker
    /// recorded for a newer batch while the queue owner was inside the driver.
    #[test]
    fn delayed_submission_binds_only_its_own_open_batch_stamps() {
        let shared = Arc::new(Shared {
            queue: Mutex::new(PendingQueue::default()),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            next_value: AtomicU64::new(0),
            latest_queued: AtomicU64::new(0),
        });
        let mut older = waiting(0, 1);
        older.point = CompletionPoint::NextSubmission(1);
        let mut newer = waiting(0, 2);
        newer.point = CompletionPoint::NextSubmission(2);
        {
            let mut queue = shared.queue.lock().expect("pending queue");
            queue.push(older);
            queue.push(newer);
        }
        let note = SubmissionNote {
            shared: Arc::clone(&shared),
        };

        note.submitted(1);

        let points: Vec<CompletionPoint> = shared
            .queue
            .lock()
            .expect("pending queue")
            .waiting
            .iter()
            .map(|w| w.point)
            .collect();
        assert_eq!(
            points,
            vec![
                CompletionPoint::Submitted(1),
                CompletionPoint::NextSubmission(2),
            ]
        );
    }

    /// Exercise the ownership relation as a matrix rather than one observed
    /// pair: every batch has stamps on several FIFOs, and each successful
    /// submission may transition exactly its own cells.
    #[test]
    fn every_submission_binds_exactly_its_stamps_across_batches_and_fifos() {
        const BATCHES: u64 = 8;
        const FIFOS: u32 = 4;
        let mut queue = PendingQueue::default();
        let mut owners = Vec::new();

        // A concrete pressure-path point is mixed into the same queue. No
        // open-batch submission notification may rewrite it.
        queue.push(waiting(FIFOS, 0xf000));
        owners.push(None);
        for batch in 1..=BATCHES {
            for fifo in 0..FIFOS {
                queue.push(deferred(fifo, (batch as u32) * 16 + fifo, batch));
                owners.push(Some(batch));
            }
        }
        let fifo_levels = queue.per_fifo.clone();

        for submitted in 1..=BATCHES {
            assert_eq!(
                queue.bind_submission(submitted),
                FIFOS as usize,
                "one notification binds every FIFO stamp in its batch and no other"
            );
            for (waiting, owner) in queue.waiting.iter().zip(&owners) {
                let expected = match owner {
                    None => CompletionPoint::Submitted(0xf001),
                    Some(batch) if *batch <= submitted => CompletionPoint::Submitted(*batch),
                    Some(batch) => CompletionPoint::NextSubmission(*batch),
                };
                assert_eq!(waiting.point, expected, "owner={owner:?} after={submitted}");
            }
            assert_eq!(
                queue.bind_submission(submitted),
                0,
                "a duplicate driver-success notification is idempotent"
            );
            assert_eq!(
                queue.per_fifo, fifo_levels,
                "binding changes state, never FIFO occupancy"
            );
        }
    }

    /// Exact identity, rather than arrival order, owns a stamp. The real queue
    /// owner is ordered, but keeping this true under an adversarial call order
    /// prevents a future refactor from quietly restoring bind-all semantics.
    #[test]
    fn out_of_order_notifications_still_cannot_cross_batch_ownership() {
        let mut queue = PendingQueue::default();
        queue.push(deferred(0, 1, 1));
        queue.push(deferred(0, 2, 2));
        queue.push(deferred(0, 3, 3));

        assert_eq!(queue.bind_submission(2), 1);
        assert_eq!(
            queue.waiting.iter().map(|w| w.point).collect::<Vec<_>>(),
            vec![
                CompletionPoint::NextSubmission(1),
                CompletionPoint::Submitted(2),
                CompletionPoint::NextSubmission(3),
            ]
        );
        assert_eq!(queue.bind_submission(1), 1);
        assert_eq!(queue.bind_submission(3), 1);
        assert!(queue
            .waiting
            .iter()
            .all(|waiting| matches!(waiting.point, CompletionPoint::Submitted(_))));
    }

    /// Reproduce the actual two-thread shape deterministically: the queue
    /// worker is held inside the older submit while the drain side records a
    /// newer stamp, then the older driver call returns.
    #[test]
    fn queue_worker_return_cannot_claim_a_stamp_recorded_during_its_driver_call() {
        let shared = test_shared(0);
        shared
            .queue
            .lock()
            .expect("pending queue")
            .push(deferred(0, 1, 1));
        let gate = Arc::new(std::sync::Barrier::new(2));
        let worker_gate = Arc::clone(&gate);
        let note = SubmissionNote {
            shared: Arc::clone(&shared),
        };
        let worker = std::thread::spawn(move || {
            worker_gate.wait();
            note.submitted(1);
        });

        shared
            .queue
            .lock()
            .expect("pending queue")
            .push(deferred(1, 2, 2));
        gate.wait();
        worker.join().expect("queue worker");

        assert_eq!(
            points(&shared),
            vec![
                CompletionPoint::Submitted(1),
                CompletionPoint::NextSubmission(2),
            ]
        );
    }
}
