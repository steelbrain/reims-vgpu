//! How long a backend waits for its device, and what it keeps of work its
//! device has not finished by then.
//!
//! A decode, an encode, a motion estimate or a copy runs on the device
//! after the driver call that submits it returns, and the backend waits
//! for it before it reads what the device wrote. No such wait has a bound
//! of its own: a driver's blocking wait (NVDEC's map, NVENC's lock, CUDA's
//! synchronous copy, VA-API's sync) returns when the device is done, and a
//! device that never finishes would hold the session's thread for good. So
//! each backend waits by asking the driver's own non-blocking question
//! until the work is done or [`DEVICE_WAIT`] has passed ([`poll_until`]).
//!
//! **What the guest sees past it.** The frame fails as the backend's
//! malfunction ([`crate::backend::BackendError::Malfunction`]), the status
//! Apple's host gives a frame its hardware fails: a decoded frame is
//! emitted with kVTVideoDecoderMalfunctionErr (-12911) and no image, its
//! buffer named in a `Dfub`; an encoded frame with
//! kVTVideoEncoderMalfunctionErr (-12912) and no sample; a motion estimate
//! with -17691 and no vectors (docs/video/backends.md, "How errors reach
//! the guest"). A backend whose device missed a deadline fails each later
//! call on the same decoder, encoder or estimator at once, the same way,
//! so that no later frame waits its own deadline behind it.
//!
//! **What is kept.** The device may still be writing, so nothing it may
//! write is handed back or used again until it is done:
//! - the backend's own memory the work writes (a decode surface, staging
//!   memory, an encoder's buffers) and the handle it runs on, which the
//!   backend gives to [`Lingering`] with its question;
//! - a guest buffer a device writes itself (zero copy): its hold
//!   ([`GuestHold`], from [`crate::device_memory::GuestTarget::hold`]),
//!   which the backend keeps with the rest. While a hold lives the session
//!   sends no `rels` for the buffer, so the guest sees it in use and its
//!   pages are neither reused nor unmapped (the mapper answers an unmap of
//!   a held surface Busy). Once the last of the hold goes, the device's
//!   write is told of ([`GuestHold::report_late_write`]), so that a copy
//!   of the pages the host keeps elsewhere is not left stale.
//!
//! [`Lingering`]'s thread asks each kept work's question every
//! [`LINGER_POLL`], and drops it once the device is done: its memory is
//! freed, its handle destroyed and its guest buffer let go, whose `rels`
//! the session's next sweep sends. Work a device never finishes is kept
//! until the process ends: that is what is leaked, on purpose, since
//! freeing it would hand memory the device may still write to the host's
//! allocator or back to the guest.
//!
//! What stays bounded: the session's thread answers the guest
//! [`DEVICE_WAIT`] after the work was due at most, and so the session's
//! end, which waits for the message in hand 10 s at most
//! (`crate::host::session_thread`), is not left waiting on a stuck call.
//! A device reset ends the host session with work still kept: its guest
//! buffers stay held by this host, though the guest's driver has let them
//! go with the reset.

use std::any::Any;
use std::fmt;
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
/// five times that. It stays under the guest's own wait for a codec
/// operation, 10 s (`kVTParavirtualizationCodecOperationReplyTimeout`, the
/// wait of a synchronous `dfrm` and of a `Demt`), so that the guest hears
/// the malfunction before its wait runs out, and under the 10 s a session's
/// end waits for the message in hand.
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

/// Keeps a guest buffer held while it lives, as a session's own use of it
/// does: no `rels` goes for the buffer, so the guest sees it in use, and
/// its pages are neither reused nor unmapped (module docs). Made by core
/// for each buffer it lends a device ([`crate::device_memory::GuestTarget::hold`]).
///
/// **A write past the deadline.** A device that missed its deadline may
/// still write the buffer, after the call that lent it has answered the
/// guest and told the host nothing of that write; the host may keep a copy
/// of those pages that the write makes stale
/// ([`crate::surfaces::GuestSurface::device_wrote`]). So the call's owner,
/// where the call fails, gives the hold what tells of the write
/// ([`GuestHold::report_late_write`]): it runs once every holder has let
/// go, the work a backend kept until its device is done among them
/// ([`Lingering`]), and before what the hold holds goes.
#[derive(Clone)]
pub struct GuestHold(Option<Arc<Held>>);

/// What a [`GuestHold`] holds, and what tells of a write past the deadline.
struct Held {
    _held: Arc<dyn Any + Send + Sync>,
    late_write: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl Drop for Held {
    /// The last holder let go: a write past the deadline is told of, then
    /// what was held goes.
    fn drop(&mut self) {
        let wrote = self.late_write.lock().take();
        if let Some(wrote) = wrote {
            wrote();
        }
    }
}

impl GuestHold {
    /// A hold on whatever `held` holds while it lives.
    pub fn new(held: Arc<dyn Any + Send + Sync>) -> Self {
        Self(Some(Arc::new(Held {
            _held: held,
            late_write: Mutex::new(None),
        })))
    }

    /// A hold of nothing: a target a test makes over memory no session
    /// holds.
    pub fn nothing() -> Self {
        Self(None)
    }

    /// The call this hold was made for failed, and a device may still write
    /// the buffer: `wrote` tells of that write once every holder has let go
    /// (the type's docs), in place of what was given before. A hold of
    /// nothing tells at once.
    pub fn report_late_write(&self, wrote: Box<dyn FnOnce() + Send>) {
        match &self.0 {
            Some(held) => *held.late_write.lock() = Some(wrote),
            None => wrote(),
        }
    }
}

impl fmt::Debug for GuestHold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.0.is_some() {
            "GuestHold(held)"
        } else {
            "GuestHold(nothing)"
        })
    }
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
