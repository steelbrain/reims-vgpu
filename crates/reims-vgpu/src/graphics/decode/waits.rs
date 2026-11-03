//! Diagnostics for a command buffer that waits on an event (`encodeWaitForEvent:value:`): one not
//! finished [`WAIT_LOGGED_AFTER`] after its commit is named in the log, with each event and value it
//! waits for, and named again once it finishes. Metal waits without end for a value nothing reaches
//! (measured on the M3 Max: the buffer stays `Committed`, never `Scheduled`, until a CPU set
//! reaches the value), and so does the host: this only says so in the log, and changes nothing the
//! guest or the host's platform sees.
//!
//! One thread per decoder, started at the first watch, takes the buffers in commit order, each at
//! its deadline on the platform's clock, and ends with the decoder ([`WaitWatch`]'s drop). The
//! deadlines are in the queue's order, so a watch wakes the thread only where the queue was empty:
//! otherwise the thread already sleeps until an earlier deadline, and takes the new buffer after.

use std::collections::VecDeque;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::AtomicU32;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::Duration;

use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::os::clock::{Clock, Deadline, MonoTime};
use reims_vgpu_core::sync::{Condvar, Mutex};

/// How long a waiting command buffer goes unfinished before the log names it.
pub(crate) const WAIT_LOGGED_AFTER: Duration = Duration::from_secs(5);

/// The waits a decoder logs; past them, one line says the rest are not.
const WAITS_LOGGED: u32 = 32;

/// A watched buffer not yet logged or finished.
const RUNNING: u8 = 0;
/// A watched buffer the log has named as waiting.
const LOGGED: u8 = 1;
/// A watched buffer that finished.
const FINISHED: u8 = 2;

/// One command buffer's waits: what it waits for, and whether it has finished.
pub(crate) struct WaitingBuffer {
    cb_obj_id: u64,
    seqno: u64,
    context: Option<u32>,
    /// Each (event, value) the buffer waits for, in its op stream's order.
    waits: Vec<(u64, u64)>,
    /// [`RUNNING`], [`LOGGED`] or [`FINISHED`].
    state: AtomicU8,
    /// The log named the buffer as waiting.
    logged: AtomicBool,
}

impl WaitingBuffer {
    /// The buffer finished (its completed handler ran): the log says so where it named the wait.
    pub(crate) fn finished(&self) {
        if self.state.swap(FINISHED, Ordering::SeqCst) == LOGGED {
            log::info!(
                target: "reims_vgpu::graphics",
                "{} finished after waiting for {}",
                self.named(),
                self.waits_named()
            );
        }
    }

    /// Whether the log named the buffer as waiting, finished since or not.
    #[cfg(test)]
    pub(crate) fn was_logged(&self) -> bool {
        self.logged.load(Ordering::SeqCst)
    }

    fn named(&self) -> String {
        match self.context {
            Some(context) => format!(
                "context {context}'s command buffer {:#x} (seqno {})",
                self.cb_obj_id, self.seqno
            ),
            None => format!(
                "command buffer {:#x} (seqno {})",
                self.cb_obj_id, self.seqno
            ),
        }
    }

    fn waits_named(&self) -> String {
        self.waits
            .iter()
            .map(|(event, value)| format!("event {event:#x} to reach {value}"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// What the watcher thread and the decoder share.
struct Shared<P: Platform> {
    platform: Arc<P>,
    state: Mutex<State>,
    wake: Condvar,
    /// The watches that woke the thread.
    #[cfg(test)]
    woken: AtomicU32,
}

#[derive(Default)]
struct State {
    /// The watched buffers, in commit order, each with its deadline.
    queue: VecDeque<(MonoTime, Arc<WaitingBuffer>)>,
    /// The thread was started.
    started: bool,
    /// The decoder went: the thread ends.
    closed: bool,
    /// The waits logged so far.
    logged: u32,
}

/// The decoder's watch over its waiting command buffers ([`Self::watch`]).
pub(crate) struct WaitWatch<P: Platform> {
    shared: Arc<Shared<P>>,
}

impl<P: Platform> WaitWatch<P> {
    /// A watch on `platform`'s clock, its thread not yet started.
    pub(crate) fn new(platform: Arc<P>) -> Self {
        WaitWatch {
            shared: Arc::new(Shared {
                platform,
                state: Mutex::new(State::default()),
                wake: Condvar::new(),
                #[cfg(test)]
                woken: AtomicU32::new(0),
            }),
        }
    }

    /// Watches the committed command buffer `cb_obj_id` of `context`, which waits for each of
    /// `waits`; its completed handler calls [`WaitingBuffer::finished`] on what this returns.
    pub(crate) fn watch(
        &self,
        cb_obj_id: u64,
        seqno: u64,
        context: Option<u32>,
        waits: Vec<(u64, u64)>,
    ) -> Arc<WaitingBuffer> {
        let buffer = Arc::new(WaitingBuffer {
            cb_obj_id,
            seqno,
            context,
            waits,
            state: AtomicU8::new(RUNNING),
            logged: AtomicBool::new(false),
        });
        let mut state = self.shared.state.lock();
        // Read under the lock, so the queue's deadlines are in its order.
        let deadline = self
            .shared
            .platform
            .clock()
            .now_mono()
            .saturating_add(WAIT_LOGGED_AFTER);
        let wakes = state.queue.is_empty();
        state.queue.push_back((deadline, Arc::clone(&buffer)));
        if !state.started {
            let shared = Arc::clone(&self.shared);
            let spawned = std::thread::Builder::new()
                .name("reims-vgpu.graphics-event-waits".into())
                .spawn(move || run(&shared));
            // Without the thread nothing is logged; the buffers run as before.
            state.started = spawned.is_ok();
            if spawned.is_err() {
                state.queue.clear();
            }
        }
        drop(state);
        if wakes {
            #[cfg(test)]
            self.shared.woken.fetch_add(1, Ordering::SeqCst);
            self.shared.wake.notify_one();
        }
        buffer
    }
}

impl<P: Platform> Drop for WaitWatch<P> {
    fn drop(&mut self) {
        self.shared.state.lock().closed = true;
        self.shared.wake.notify_one();
    }
}

/// The watcher thread: each buffer at its deadline, logged where it has not finished.
fn run<P: Platform>(shared: &Shared<P>) {
    let clock = shared.platform.clock();
    let mut state = shared.state.lock();
    loop {
        if state.closed {
            return;
        }
        let Some((deadline, buffer)) = state.queue.front().cloned() else {
            clock.wait(&shared.wake, &mut state);
            continue;
        };
        if buffer.state.load(Ordering::SeqCst) == FINISHED {
            state.queue.pop_front();
            continue;
        }
        if clock.now_mono() < deadline {
            clock.wait_until(&shared.wake, &mut state, Deadline::Mono(deadline));
            continue;
        }
        state.queue.pop_front();
        if state.logged < WAITS_LOGGED
            && buffer
                .state
                .compare_exchange(RUNNING, LOGGED, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        {
            state.logged += 1;
            buffer.logged.store(true, Ordering::SeqCst);
            log::warn!(
                target: "reims_vgpu::graphics",
                "{} has waited {} s for {}: nothing has signalled or set those values yet, and \
                 its queue's later command buffers wait behind it",
                buffer.named(),
                WAIT_LOGGED_AFTER.as_secs(),
                buffer.waits_named()
            );
            if state.logged == WAITS_LOGGED {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "{WAITS_LOGGED} command buffers' waits logged: further ones are not"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::testkit::FakePlatform;

    use super::*;

    /// Polls `done` for up to two seconds of real time: the watcher runs on a thread of its own.
    fn eventually(done: impl Fn() -> bool) -> bool {
        for _ in 0..2000 {
            if done() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        done()
    }

    /// A buffer still waiting at its deadline is logged; one that finished first is not, and
    /// neither is one whose deadline has not come.
    #[test]
    fn a_buffer_waiting_past_its_deadline_is_logged() {
        let p = Arc::new(FakePlatform::new());
        let watch = WaitWatch::new(Arc::clone(&p));
        let finished = watch.watch(1, 1, Some(2), vec![(6, 1)]);
        finished.finished();
        let waiting = watch.watch(2, 2, Some(2), vec![(6, 1), (7, 4)]);
        p.virtual_clock().advance(WAIT_LOGGED_AFTER);
        let later = watch.watch(3, 3, None, vec![(7, 5)]);
        assert!(eventually(|| waiting.was_logged()));
        // The queue is taken in order: the finished one went before.
        assert!(!finished.was_logged());
        std::thread::sleep(Duration::from_millis(20));
        assert!(!later.was_logged());
        waiting.finished();
        assert!(waiting.was_logged());
    }

    /// A watch wakes the thread only where its queue was empty: one behind others needs it no
    /// sooner than the deadline it sleeps until. Each is still logged at its own deadline.
    #[test]
    fn a_watch_wakes_the_thread_only_onto_an_empty_queue() {
        let p = Arc::new(FakePlatform::new());
        let watch = WaitWatch::new(Arc::clone(&p));
        let first = watch.watch(1, 1, None, vec![(6, 1)]);
        let behind: Vec<_> = (2..10)
            .map(|n| watch.watch(n, n, None, vec![(6, n)]))
            .collect();
        assert_eq!(watch.shared.woken.load(Ordering::SeqCst), 1);
        p.virtual_clock().advance(WAIT_LOGGED_AFTER);
        assert!(eventually(
            || first.was_logged() && behind.iter().all(|b| b.was_logged())
        ));
        assert!(eventually(|| watch.shared.state.lock().queue.is_empty()));
        // The queue emptied: the next watch wakes the thread again.
        let after = watch.watch(10, 10, None, vec![(6, 10)]);
        assert_eq!(watch.shared.woken.load(Ordering::SeqCst), 2);
        p.virtual_clock().advance(WAIT_LOGGED_AFTER);
        assert!(eventually(|| after.was_logged()));
    }

    /// The decoder's watch ends its thread when it goes.
    #[test]
    fn the_watch_ends_with_its_decoder() {
        let p = Arc::new(FakePlatform::new());
        let watch = WaitWatch::new(Arc::clone(&p));
        let waiting = watch.watch(1, 1, None, vec![(6, 1)]);
        let shared = Arc::downgrade(&watch.shared);
        drop(watch);
        assert!(eventually(|| shared.strong_count() == 0));
        assert!(!waiting.was_logged());
    }
}
