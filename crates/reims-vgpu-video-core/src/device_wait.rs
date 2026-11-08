//! How long a backend waits for its device, and what it keeps of work its
//! device has not finished by then.
//!
//! A decode, an encode or a copy runs on the device after the driver call
//! that submits it returns, and the backend waits for it before it reads
//! what the device wrote. No such wait has a bound of its own: a driver's
//! blocking wait (NVDEC's map, NVENC's lock, CUDA's synchronous copy,
//! VA-API's sync) returns when the device is done, and a device that never
//! finishes would hold the caller's thread for good. So each backend waits
//! by asking the driver's own non-blocking question until the work is done
//! or [`DEVICE_WAIT`] has passed ([`poll_until`]).
//!
//! **Past it**, the work fails as the backend's malfunction
//! ([`crate::backend::BackendError::Malfunction`]). A backend whose device
//! missed a deadline fails each later call on the same decoder or encoder
//! at once, the same way, so that no later frame waits its own deadline
//! behind it.
//!
//! **What is kept.** The device may still be writing, so nothing it may
//! write is handed back or used again until it is done:
//! the backend's own memory the work writes (a decode surface, staging
//! memory, an encoder's buffers, an import of a guest buffer the device
//! writes itself) and the handle it runs on, which the backend gives to
//! [`Lingering`] with its question.
//!
//! [`Lingering`]'s thread asks each kept work's question every
//! [`LINGER_POLL`], and drops it once the device is done: its memory is
//! freed and its handle destroyed. Work a device never finishes is kept
//! until the process ends: that is what is leaked,
//! on purpose, since freeing it would hand memory the device may still
//! write to the host's allocator or back to the guest.
//!
//! What stays bounded: the caller's thread is answered [`DEVICE_WAIT`]
//! after the work was due at most, and so is not left waiting on a stuck
//! call.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::runtime::{Bell, MonoTime, Runtime};
use crate::sync::Mutex;

/// How long a backend waits for its device to finish one frame's work: 5 s
/// on the runtime's clock, from the frame's first wait. Ours.
///
/// The slowest frame measured on any of the hosts' GPUs is the first decode
/// after an RX 7900 XTX has idled 5 s, 0.88 s, most of it the GPU waking
/// from runtime suspend (docs/video/bugs.md, B61). Five seconds is over
/// five times that.
pub const DEVICE_WAIT: Duration = Duration::from_secs(5);

/// The pause between two of a poll's questions while the work may be
/// about to finish: the first [`QUICK_POLLS_FOR`] of the wait.
pub const QUICK_POLL: Duration = Duration::from_micros(100);

/// How long a poll asks at [`QUICK_POLL`]'s pace: longer than any frame's
/// work measured on the hosts' GPUs but the first after an idle GPU wakes.
pub const QUICK_POLLS_FOR: Duration = Duration::from_millis(50);

/// The pause between two of a poll's questions after [`QUICK_POLLS_FOR`].
pub const SLOW_POLL: Duration = Duration::from_millis(1);

/// How often [`Lingering`]'s thread asks whether kept work is done.
pub const LINGER_POLL: Duration = Duration::from_millis(100);

/// `check`'s answer, asked now and then after each pause until it has one
/// or `deadline` has passed on `runtime`'s clock: `None` past it. `check`
/// is asked once more at the deadline, so work done by then is taken
/// however late the poll wakes. The pauses are [`QUICK_POLL`] for the
/// first [`QUICK_POLLS_FOR`], then [`SLOW_POLL`]: each a timed wait on the
/// runtime's clock, a sleep on the host's.
pub fn poll_until<T>(
    runtime: &dyn Runtime,
    deadline: MonoTime,
    mut check: impl FnMut() -> Option<T>,
) -> Option<T> {
    let started = runtime.now();
    let pauses = Bell::new();
    loop {
        if let Some(done) = check() {
            return Some(done);
        }
        let now = runtime.now();
        if now >= deadline {
            return None;
        }
        let pause = pause_after(now.saturating_duration_since(started));
        // No one rings it: the wait ends at its deadline.
        pauses.wait_until(
            runtime,
            now.saturating_add(pause).min(deadline),
            || None::<()>,
        );
    }
}

/// The pause before a poll's next question, `polled` into its wait.
fn pause_after(polled: Duration) -> Duration {
    if polled < QUICK_POLLS_FOR {
        QUICK_POLL
    } else {
        SLOW_POLL
    }
}

/// A deadline [`DEVICE_WAIT`] from now on `runtime`'s clock.
pub fn deadline(runtime: &dyn Runtime) -> MonoTime {
    runtime.deadline_in(DEVICE_WAIT)
}

/// Work a device did not finish by its deadline, with everything it may
/// still write: kept by [`Lingering`] until [`Unfinished::finished`] says
/// the device is done, then dropped there, which lets all of it go.
pub trait Unfinished: Send {
    /// Whether the device is done with the work: the driver's non-blocking
    /// question, asked on [`Lingering`]'s thread. `false` while the answer
    /// says nothing of the device's writes, an error's included, so that
    /// nothing the device may still write is let go.
    fn finished(&mut self) -> bool;
}

/// The name of [`Lingering`]'s thread (ours).
const LINGERING_THREAD_NAME: &str = "reims-vgpu-video.lingering";

/// Work devices did not finish by their deadlines, kept until they do
/// (module docs): one list for the process, asked after on a thread of its
/// own while it holds any, which the runtime it was made with starts.
pub struct Lingering {
    runtime: Arc<dyn Runtime>,
    shared: Arc<Mutex<LingeringState>>,
}

#[derive(Default)]
struct LingeringState {
    work: Vec<Kept>,
    /// How many works a sweep has taken out of `work` to ask after.
    asking: usize,
    /// Whether the thread runs.
    running: bool,
    /// Work kept since the list was made, for the log.
    ever_kept: u64,
}

struct Kept {
    what: String,
    since: MonoTime,
    work: Box<dyn Unfinished>,
}

impl Lingering {
    /// The process's list, made with `runtime` the first time it is asked
    /// for: work outlives the session, and the backend, that kept it.
    pub fn process(runtime: &Arc<dyn Runtime>) -> &'static Lingering {
        static PROCESS: OnceLock<Lingering> = OnceLock::new();
        PROCESS.get_or_init(|| Lingering::new(runtime))
    }

    /// A list of its own, on `runtime`: for a test, whose work must not mix
    /// with the rest's.
    pub fn new(runtime: &Arc<dyn Runtime>) -> Self {
        Self {
            runtime: Arc::clone(runtime),
            shared: Arc::new(Mutex::new(LingeringState::default())),
        }
    }

    /// Keeps `work`, which `what` names for the log, until its device is
    /// done with it. Where no thread starts, the caller's thread asks after
    /// each kept work now, and what is not done stays kept, to be asked
    /// after again with the next work kept.
    pub fn keep(&self, what: impl Into<String>, work: Box<dyn Unfinished>) {
        let what = what.into();
        let start = {
            let mut state = self.shared.lock();
            state.ever_kept += 1;
            log::error!(
                "{what}: the device did not finish it in {:.1} s; it is kept, with what the \
                 device may still write, until the device is done ({} kept now, {} ever)",
                DEVICE_WAIT.as_secs_f64(),
                state.work.len() + state.asking + 1,
                state.ever_kept
            );
            state.work.push(Kept {
                what,
                since: self.runtime.now(),
                work,
            });
            !std::mem::replace(&mut state.running, true)
        };
        if !start {
            return;
        }
        let shared = Arc::clone(&self.shared);
        let runtime = Arc::clone(&self.runtime);
        let started = self.runtime.spawn(
            LINGERING_THREAD_NAME,
            Box::new(move || run(&shared, &*runtime)),
        );
        if let Err(error) = started {
            log::warn!("no thread asks after work a device did not finish: {error}");
            sweep(&self.shared, &*self.runtime);
            self.shared.lock().running = false;
        }
    }

    /// How many works are kept now, those a sweep is asking after among
    /// them.
    pub fn count(&self) -> usize {
        let state = self.shared.lock();
        state.work.len() + state.asking
    }
}

/// The thread: asks after the work every [`LINGER_POLL`] on the runtime's
/// clock, and ends once none is kept.
fn run(shared: &Mutex<LingeringState>, runtime: &dyn Runtime) {
    let pauses = Bell::new();
    while !sweep(shared, runtime) {
        // No one rings it: the wait ends at its deadline.
        pauses.wait_until(runtime, runtime.deadline_in(LINGER_POLL), || None::<()>);
    }
}

/// Asks each kept work whether its device is done, and drops those that
/// are, with no lock held, which lets go of all they kept. `true` where
/// none is kept now, the thread marked as ended under the same lock, so
/// that the next work kept starts another.
fn sweep(shared: &Mutex<LingeringState>, runtime: &dyn Runtime) -> bool {
    let asked = {
        let mut state = shared.lock();
        state.asking = state.work.len();
        std::mem::take(&mut state.work)
    };
    let mut still = Vec::with_capacity(asked.len());
    for mut kept in asked {
        if kept.work.finished() {
            log::warn!(
                "{}: the device is done with it after {:.1} s, and it is let go",
                kept.what,
                runtime.elapsed_since(kept.since).as_secs_f64()
            );
            // No longer counted from here: what is let go is being let go.
            {
                let mut state = shared.lock();
                state.asking = state.asking.saturating_sub(1);
            }
            drop(kept);
        } else {
            still.push(kept);
        }
    }
    let mut state = shared.lock();
    state.asking = 0;
    state.work.append(&mut still);
    let empty = state.work.is_empty();
    if empty {
        state.running = false;
    }
    empty
}

#[cfg(test)]
mod tests;
