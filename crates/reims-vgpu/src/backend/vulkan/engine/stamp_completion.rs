//! Announcing a completion stamp the GPU wrote, off the drain worker.
//!
//! # What the guest is actually waiting on
//!
//! `IOGPUEventMachine::waitForStamp(index, target)` in the guest reads the stamp
//! word **straight out of the page this device writes** — a signed
//! `target - current <= 0`, so it is wrap-safe — and if the target has not been
//! reached it builds a **one-second** deadline, sleeps on the stamp word's own
//! address as the wait channel, and re-reads the word on every wake.
//!
//! Two things follow, and the second is the one that cost a rebuild:
//!
//! * The word is the authority. Writing it from the GPU, ordered behind the
//!   writeback copies by a barrier, is a sound way to move a fence — the guest
//!   never asks this device whether the value is real.
//! * **The interrupt is the wakeup, not a hint.** Nothing re-checks the word
//!   until something wakes the thread, so a late interrupt is not a late
//!   notification, it is up to a full second of guest stall. An earlier attempt
//!   deferred the announcement to the drain worker's next tranche and measured
//!   exactly that: draws/s 3237 -> 2, presents/s 45 -> 1.
//!
//! So the stamp word may be handed to the GPU, and the interrupt may not be
//! handed to anything that runs on a schedule. It has to be raised the moment
//! the submission completes, which is what this module is.
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
//! # What it measured
//!
//! Back-to-back on one machine against the parent commit, testufo animating,
//! 30 census windows each:
//!
//! ```text
//!                      before     after
//! presents/s             44.0      64.0     +45%
//! draws/s              3206.0    3346.5
//! busy_us/s          909958.5  830495.0      -9%
//! max_tranche_us      37250.5   25458.0     -32%
//! ```
//!
//! The presents ranges do not overlap — 41-47 against 54-69 — which is what
//! makes this readable at all, because the same config measured across an hour
//! of other work read 51, 53 and 63. Take a reading from this rail only against
//! a run of its own parent on the same quiet machine.
//!
//! **The drain worker's block did not go away, and the gain is not from its
//! removal.** `fence/s` still tracks `flushes/s` (399 -> 421), because root slot
//! 0 stays on the blocking rail. What changed is what that block costs: by the
//! time a root stamp quiesces, the child stamps ahead of it have already ordered
//! the same copies on the queue, so the wait finds them done. The worker does 9%
//! less work and delivers 45% more frames.
//!
//! # Why a timeline semaphore rather than a fence
//!
//! A second waiter cannot use the ring's fences. `ResourcePools` owns every one
//! of them and resets each at retire, so a thread waiting on a ring fence races
//! the reset and a submission that signalled once can read as unsignalled
//! forever. Giving the stamp submission its own fence instead breaks the ring
//! the other way: `vkQueueSubmit` takes exactly one fence, so the slot's fence
//! would never signal and its cleanup would never retire.
//!
//! A timeline semaphore has neither problem. It is signalled *in addition* to
//! the slot's fence, its value is monotonic so nothing has to be reset, and
//! `vkWaitSemaphores` may be called from any thread. Core in Vulkan 1.2, which
//! is this backend's baseline — and still gated, because the fallback is simply
//! the blocking rail every host used before this existed.

use ash::vk;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Raise the guest-visible interrupt for a completed stamp slot.
///
/// Installed by the device layer, which owns the interrupt-status clone and the
/// prompt action queue. Called from the completion thread with no lock of this
/// crate's held, so an implementation must not reach for the device lock.
pub type AnnounceStamp = Arc<dyn Fn(u32) + Send + Sync>;

/// The installed announcement hook.
///
/// A global rather than a constructor argument because the two events have no
/// order between them: the device layer binds when QEMU realizes the device, and
/// the engine builds its context lazily on the first draw. Whichever happens
/// first, the thread reads the hook when it has something to announce.
///
/// **A stamp completing with no hook installed is a lost wakeup**, so it is
/// fail-visible rather than silent — it means a submission reached the GPU
/// before this device was bound, which nothing should be able to arrange.
static HOOK: std::sync::Mutex<Option<AnnounceStamp>> = std::sync::Mutex::new(None);

/// Install the hook the completion thread announces through. Idempotent; the
/// last caller wins, which is what a device rebind wants.
///
/// There is no uninstall. The hook the device layer installs resolves its
/// device by id every time it is called, so one left behind by a torn-down
/// device holds nothing and announces nothing — an uninstall would only be a
/// second way to reach the same state, and a race against a completion already
/// in flight.
pub fn install_announce(hook: AnnounceStamp) {
    *HOOK.lock().unwrap_or_else(|e| e.into_inner()) = Some(hook);
}

fn announce(index: u32) {
    let hook = HOOK
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(Arc::clone);
    match hook {
        // Called with no lock of this module's held: the hook reaches the
        // device's prompt queue, and holding `HOOK` across it would put this
        // module's mutex under the device's.
        Some(hook) => hook(index),
        None => crate::observe::fail(format!(
            "stamp_announce_no_hook reason=stamp_announce_no_hook index={index} \
             (a stamp completed with no device bound to raise its interrupt; the guest \
             waiting on it will sleep to its one-second deadline)"
        )),
    }
}

/// One stamp waiting on the GPU, in submission order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Waiting {
    /// Timeline value the submission carrying this stamp's word will signal.
    value: u64,
    /// Stamp slot index, for the interrupt-status bit.
    index: u32,
}

/// The queue the drain worker pushes to and the completion thread drains.
struct Shared {
    queue: Mutex<std::collections::VecDeque<Waiting>>,
    /// Woken by a push and by shutdown. The thread waits here only when the
    /// queue is empty; otherwise it is blocked in `vkWaitSemaphores`, which is
    /// where it should be.
    wake: Condvar,
    stop: AtomicBool,
    /// Highest value handed out. The drain worker reserves with `fetch_add`
    /// under the engine lock, so reservation order is submission order.
    next_value: AtomicU64,
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
            queue: Mutex::new(std::collections::VecDeque::new()),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            next_value: AtomicU64::new(0),
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

    /// The semaphore a stamp submission signals, and the value to signal.
    ///
    /// Reserved under the engine lock, so the values are handed out in
    /// submission order — which is what makes a single-threaded drain of the
    /// queue announce stamps in that same order without any further ordering
    /// machinery.
    pub(crate) fn reserve(&self, index: u32) -> (vk::Semaphore, u64) {
        let value = self.shared.next_value.fetch_add(1, Ordering::AcqRel) + 1;
        self.shared
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(Waiting { value, index });
        self.shared.wake.notify_one();
        (self.semaphore, value)
    }

    /// Drop a reservation whose submission never happened.
    ///
    /// A submit that fails after [`Self::reserve`] leaves a value nothing will
    /// ever signal, and the thread would then block on it until the deadline and
    /// hold every later stamp behind it. Signalling the value from the host is
    /// the repair: it is exactly what the queue would have done, so the thread's
    /// wait completes and the announcement still reaches the guest — which is
    /// the safe direction, because the alternative is a guest asleep on a fence
    /// with a one-second deadline and nothing coming.
    ///
    /// # Safety
    ///
    /// `device` must be the device this was started with.
    pub(crate) unsafe fn abandon(&self, device: &ash::Device, value: u64) {
        let signal = vk::SemaphoreSignalInfo::default()
            .semaphore(self.semaphore)
            .value(value);
        if let Err(e) = unsafe { device.signal_semaphore(&signal) } {
            crate::observe::fail(format!(
                "stamp_signal_abandon_failed reason=stamp_signal_abandon_failed value={value} \
                 err={e:?} (a stamp submission failed and the host could not stand in for it; \
                 the guest waiting on this stamp will sleep to its deadline)"
            ));
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
        // stamps it then announces are ones whose bytes may not have landed —
        // deliberately, and for the reason `abandon` gives: this device is going
        // away, so a withheld announcement is a guest that never wakes.
        let past_everything = self.shared.next_value.load(Ordering::Acquire) + 1;
        let signal = vk::SemaphoreSignalInfo::default()
            .semaphore(self.semaphore)
            .value(past_everything);
        let _ = unsafe { device.signal_semaphore(&signal) };
        self.shared.wake.notify_all();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
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
                if let Some(front) = queue.front().copied() {
                    break Some(front);
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
        let values = [waiting.value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // The same deadline every blocking wait in this backend uses. Reaching
        // it means the queue has not run this submission, which is a device
        // fault rather than a slow frame — announce anyway and say so, because
        // the guest's own deadline is one second and a withheld stamp costs it
        // that whether the GPU is wedged or not.
        match unsafe { device.wait_semaphores(&info, super::context::FENCE_TIMEOUT_NS) } {
            Ok(()) => {}
            Err(vk::Result::TIMEOUT) => crate::observe::fail(format!(
                "stamp_wait_timeout reason=stamp_wait_timeout index={} value={} \
                 (the submission carrying this stamp's word has not executed within the \
                 fence deadline; announcing it anyway so the guest is not left asleep)",
                waiting.index, waiting.value
            )),
            Err(e) => crate::observe::fail(format!(
                "stamp_wait_failed reason=stamp_wait_failed index={} value={} err={e:?} \
                 (announcing regardless, for the reason a timeout does)",
                waiting.index, waiting.value
            )),
        }
        shared
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front();
        announce(waiting.index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reservation order is submission order, and the queue keeps it. A single
    /// thread draining a FIFO is the whole of this rail's ordering guarantee —
    /// a stamp announced out of order moves a guest fence past a completion it
    /// has not been told about.
    ///
    /// Device-free: drives the ledger, not the wait.
    #[test]
    fn reservations_are_monotonic_and_queue_in_submission_order() {
        let shared = Shared {
            queue: Mutex::new(std::collections::VecDeque::new()),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
            next_value: AtomicU64::new(0),
        };
        for (n, index) in [(1u64, 3u32), (2, 0), (3, 3)] {
            let value = shared.next_value.fetch_add(1, Ordering::AcqRel) + 1;
            assert_eq!(value, n, "timeline values start at 1 and never repeat");
            shared
                .queue
                .lock()
                .unwrap()
                .push_back(Waiting { value, index });
        }
        let queue = shared.queue.lock().unwrap();
        assert_eq!(
            queue.iter().copied().collect::<Vec<_>>(),
            vec![
                Waiting { value: 1, index: 3 },
                Waiting { value: 2, index: 0 },
                Waiting { value: 3, index: 3 },
            ],
            "the same slot may be stamped twice and both must be announced, in order"
        );
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
}
