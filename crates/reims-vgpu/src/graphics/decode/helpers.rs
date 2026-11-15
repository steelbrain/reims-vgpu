//! The threads that walk a command buffer's later range beside the lane thread that walks the
//! rest of it ([`super::split`], "Two ranges at once").
//!
//! - **Hands.** A walk hands its later range over as it starts ([`Helpers::hand`]): an idle
//!   helper is woken for it, else one is started, up to [`HELPERS_MOST`] across the device. A
//!   range no helper has started when the walk of the rest ends is taken back
//!   ([`Hand::take_back`]) and walked there, as a buffer walked whole is: so a walk never waits
//!   for a helper to come free, and one with none spare is walked as it was.
//! - **Threads.** Each helper gives itself the walk hints as it starts, as a lane's thread does
//!   ([`super::lane`]), takes the hands in the order they came, and goes once it has had none for
//!   [`KEEP`], 10 s. A helper is no lane's thread: what a lane counts and starts, and every
//!   turn, ticket and commit, is as it was. A helper walks and commits nothing; the lane's
//!   thread commits what it walked, in the buffer's order.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use reims_vgpu_graphics::keep::KEEP;
use reims_vgpu_vulkan::core::platform::os::Threads;
use reims_vgpu_vulkan::core::sync::{Condvar, Mutex};

/// The most helpers the device runs at once: each walks one buffer's later range, so as many
/// buffers walk two ranges at once, the rest of them walked as before.
pub(crate) const HELPERS_MOST: usize = 4;

/// A range's walk, for a helper.
pub(crate) type Job = Box<dyn FnOnce() + Send>;

/// The device's helpers (the module doc).
pub(crate) struct Helpers {
    state: Mutex<State>,
    work: Condvar,
    /// The platform's threads, whose walk hints each helper gives itself as it starts; none in
    /// the tests that make no platform.
    hints: Option<Arc<dyn Threads>>,
    /// How long a helper with no hand waits for one before it goes.
    idle_for: Duration,
}

/// What [`Helpers`]' lock guards.
#[derive(Default)]
struct State {
    /// The hands no helper has taken yet, in the order they came.
    hands: VecDeque<Arc<Hand>>,
    /// The helpers running.
    threads: usize,
    /// The helpers waiting for a hand.
    idle: usize,
}

/// A range handed to the helpers: waiting for one, walked by one, or done (walked, or taken
/// back).
pub(crate) struct Hand {
    state: Mutex<HandState>,
    done: Condvar,
}

enum HandState {
    Waiting(Job),
    Running,
    Done,
}

impl Helpers {
    /// The device's helpers, none running yet, each giving itself `hints`' walk hints.
    pub(crate) fn new(hints: Option<Arc<dyn Threads>>) -> Arc<Helpers> {
        Helpers::with_idle(hints, KEEP)
    }

    /// [`Self::new`], a helper going after `idle_for` with no hand.
    fn with_idle(hints: Option<Arc<dyn Threads>>, idle_for: Duration) -> Arc<Helpers> {
        Arc::new(Helpers {
            state: Mutex::new(State::default()),
            work: Condvar::new(),
            hints,
            idle_for,
        })
    }

    /// Hands `job` to a helper: one idle is woken for it, else one is started where fewer than
    /// [`HELPERS_MOST`] run; else it waits for the first to come free, unless it is taken back
    /// first. Waits for nothing.
    pub(crate) fn hand(self: &Arc<Self>, job: Job) -> Arc<Hand> {
        let hand = Arc::new(Hand {
            state: Mutex::new(HandState::Waiting(job)),
            done: Condvar::new(),
        });
        let mut state = self.state.lock();
        state.hands.push_back(Arc::clone(&hand));
        if state.hands.len() <= state.idle {
            self.work.notify_one();
            return hand;
        }
        if state.threads >= HELPERS_MOST {
            return hand;
        }
        state.threads += 1;
        drop(state);
        let helpers = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name("reims-vgpu.graphics-helper".to_owned())
            .spawn(move || helpers.serve());
        if spawned.is_err() {
            self.state.lock().threads -= 1;
            log::warn!(
                target: "reims_vgpu::graphics",
                "no helper thread could be started: a command buffer's later range is walked with \
                 the rest"
            );
        }
        hand
    }

    /// A helper: the walk hints, then the hands, in the order they came, until none comes for
    /// `idle_for`.
    fn serve(&self) {
        if let Some(threads) = &self.hints {
            threads.give_walk_hints();
        }
        let mut state = self.state.lock();
        loop {
            if let Some(hand) = state.hands.pop_front() {
                drop(state);
                hand.run();
                state = self.state.lock();
                continue;
            }
            state.idle += 1;
            let waited = self.work.wait_for(&mut state, self.idle_for);
            state.idle -= 1;
            if waited.timed_out() && state.hands.is_empty() {
                state.threads -= 1;
                return;
            }
        }
    }

    /// The helpers running (for the tests).
    #[cfg(test)]
    pub(crate) fn threads(&self) -> usize {
        self.state.lock().threads
    }

    /// Holds every helper busy until the answer drops, so a hand made meanwhile waits (for the
    /// tests).
    #[cfg(test)]
    pub(crate) fn hold_all(self: &Arc<Self>) -> HeldAll {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let hands: Vec<_> = (0..HELPERS_MOST)
            .map(|_| {
                let gate = Arc::clone(&gate);
                self.hand(Box::new(move || {
                    let mut open = gate.0.lock();
                    while !*open {
                        gate.1.wait(&mut open);
                    }
                }))
            })
            .collect();
        assert!(
            crate::graphics::tests::eventually(|| {
                (hands.iter()).all(|h| matches!(*h.state.lock(), HandState::Running))
            }),
            "every helper took a hand"
        );
        HeldAll { gate, hands }
    }
}

/// Every helper held busy ([`Helpers::hold_all`]), until it drops.
#[cfg(test)]
pub(crate) struct HeldAll {
    gate: Arc<(Mutex<bool>, Condvar)>,
    hands: Vec<Arc<Hand>>,
}

#[cfg(test)]
impl Drop for HeldAll {
    fn drop(&mut self) {
        *self.gate.0.lock() = true;
        self.gate.1.notify_all();
        for hand in &self.hands {
            hand.wait();
        }
    }
}

impl Hand {
    /// Runs the job, where it was not taken back.
    fn run(&self) {
        let job = {
            let mut state = self.state.lock();
            match std::mem::replace(&mut *state, HandState::Running) {
                HandState::Waiting(job) => job,
                other => {
                    *state = other;
                    return;
                }
            }
        };
        #[cfg(test)]
        self.done.notify_all();
        job();
        *self.state.lock() = HandState::Done;
        self.done.notify_all();
    }

    /// The job, where no helper has started it: it is the caller's from now on, and no helper
    /// runs it. `None` where one has.
    pub(crate) fn take_back(&self) -> Option<Job> {
        let mut state = self.state.lock();
        match std::mem::replace(&mut *state, HandState::Done) {
            HandState::Waiting(job) => Some(job),
            other => {
                *state = other;
                None
            }
        }
    }

    /// Waits until a helper has started the job, or it was taken back (for the tests).
    #[cfg(test)]
    pub(crate) fn wait_started(&self) {
        let mut state = self.state.lock();
        while matches!(*state, HandState::Waiting(_)) {
            self.done.wait(&mut state);
        }
    }

    /// Waits, with no lock of the caller's held, until a helper's run of the job has ended; at
    /// once where it was taken back.
    pub(crate) fn wait(&self) {
        let mut state = self.state.lock();
        while matches!(*state, HandState::Running) {
            self.done.wait(&mut state);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A hand runs on a helper, once, and its wait returns after it ran; one taken back before a
    /// helper took it never runs there.
    #[test]
    fn a_hand_runs_once_on_a_helper_or_is_taken_back() {
        let helpers = Helpers::with_idle(None, Duration::from_millis(50));
        let ran = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&ran);
        let hand = helpers.hand(Box::new(move || {
            counted.fetch_add(1, Ordering::SeqCst);
        }));
        hand.wait();
        assert!(crate::graphics::tests::eventually(|| ran
            .load(Ordering::SeqCst)
            == 1));
        hand.wait();
        assert!(hand.take_back().is_none(), "run: nothing to take back");

        // Every helper busy: a later hand waits, and is taken back.
        let held = helpers.hold_all();
        let counted = Arc::clone(&ran);
        let late = helpers.hand(Box::new(move || {
            counted.fetch_add(1, Ordering::SeqCst);
        }));
        let job = late.take_back().expect("no helper was free");
        late.wait();
        drop(held);
        assert_eq!(
            ran.load(Ordering::SeqCst),
            1,
            "the hand taken back never ran"
        );
        job();
        assert_eq!(ran.load(Ordering::SeqCst), 2);
        assert!(
            crate::graphics::tests::eventually(|| helpers.threads() == 0),
            "idle helpers go"
        );
    }
}
