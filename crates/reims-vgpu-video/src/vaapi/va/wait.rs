//! The crate's waits on the GPU, each bounded (core's `device_wait`): the
//! driver is asked whether the work is done, with no wait, until it is or
//! [`DEVICE_WAIT`] has passed, and work past that is kept, with what it may
//! still write and the handles it runs on, until the GPU is done with it
//! ([`keep`]).
//!
//! | wait | asked with | in place of |
//! |---|---|---|
//! | a decode | `vaSyncSurface2` of its surface, with a timeout of 0, until it answers other than `VA_STATUS_ERROR_TIMEDOUT` (`super::surface`) | `vaSyncSurface` |
//! | a video-processing run: the staged copy back's, zero copy's, the trial runs at start | for the copy back and zero copy, into host memory, a `poll` of the dma-bufs it writes until the deadline (`super::fence`); then `vaSyncSurface2` of its target, as a decode's | the poll for 1 s, then `vaSyncSurface` |
//! | an encode | `vaSyncBuffer` of its coded buffer, with a timeout of 0 (`super::encoder`) | `vaSyncBuffer` with `VA_TIMEOUT_INFINITE` |
//! | a driver call that may wait for the GPU itself: `vaGetImage`, `vaMapBuffer` of a derived image, `vaSyncSurface` where the driver has no `vaSyncSurface2`, a decoder's first `vaEndPicture` (`super::buffer`) | its answer, the call made on the display's call thread (`super::calls`) | the call on the caller's thread |
//! | every other driver call, while one of those is in the driver | the display's gate, held exclusively by the call thread meanwhile (`super::calls`) | the driver's own lock, which a call that waits holds while it does |
//!
//! **Measured** on the RX 7900 XTX and the Radeon 780M (radeonsi, Mesa
//! 26.2.4, libva 2.24.1, 2026-10-25), each question asked again and again
//! before each of ffmpeg n9.0.2's own syncs, for a 4K HEVC stream's
//! decode, its scaling to 1080p and its encode at 20 Mbit/s: it answered
//! `VA_STATUS_ERROR_TIMEDOUT` while the work ran and success once it was
//! done, the frames came out the same as without the questions, and the
//! sync after it returned in 1 µs at most. From the sync on, the first
//! decode was done in 1.6 to 3.3 ms and the later ones at once, a run in
//! 0.3 to 0.7 ms at the median and 2.0 ms at most, and an encode in 5 to
//! 14 ms at the median and 26 ms at most. Which calls wait inside the
//! driver is measured in `super::calls`.
//!
//! **One deadline a frame.** A session's call (a decode, an encode) and the
//! read backs of the frames a decode gives share one deadline
//! ([`Deadline`]), [`DEVICE_WAIT`] from the first of their waits, a wait
//! for the display's gate or call thread among them: each is made by the
//! deadline of the frame in hand on its thread ([`by`], [`current`]). So a frame is answered within one deadline of its first
//! wait, however many calls and questions it makes: for a decoder's first
//! frame, its first `vaEndPicture`, whose answer is waited for. A question
//! asked past the deadline is asked once: work done by then is done.
//!
//! What a decoder or an encoder does once a wait of its has passed its
//! deadline: that frame fails as a malfunction ([`Error::Stalled`]), and
//! each later call on it fails at once the same way, for as long as it
//! lives, so that each frame is answered without a wait of its own.
//! Nothing of the work is destroyed until the GPU is done with it: a
//! decoder's surfaces, its staged copy back's memory and its context, an
//! encoder's session, and zero copy's import of a guest buffer. Nothing a
//! call left on the call thread uses is destroyed
//! before it returns: the display's releases are made behind it there
//! (`super::calls`).

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

pub(crate) use crate::core::device_wait::Unfinished;
use crate::core::device_wait::{self, DEVICE_WAIT, Lingering};
use crate::core::runtime::MonoTime;

use super::display::{Display, Error, check};
use super::ffi::{self, encode as enc};

/// What a wait is on: the surface a decode or a video-processing run
/// writes, the coded buffer an encode writes, or the display's calls that
/// wait for the GPU (`super::calls`), done once none is in hand.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Work {
    Surface(ffi::VASurfaceID),
    Buffer(ffi::VABufferID),
    Calls,
}

impl Work {
    /// The driver's question, made with the display's handle `raw`.
    fn question(self, raw: ffi::VADisplay) -> ffi::VAStatus {
        match self {
            Self::Surface(surface) => {
                #[cfg(test)]
                if tests::NO_SYNC_SURFACE2.get() {
                    return ffi::VA_STATUS_ERROR_UNIMPLEMENTED;
                }
                // SAFETY: a surface of this display, which the caller holds
                // for the call; a timeout of 0 returns at once (va.h).
                unsafe { ffi::vaSyncSurface2(raw, surface, 0) }
            }
            // SAFETY: as above, a buffer.
            Self::Buffer(buffer) => unsafe { enc::vaSyncBuffer(raw, buffer, 0) },
            // Asked once the display's gate is had: none is in hand.
            Self::Calls => ffi::VA_STATUS_SUCCESS,
        }
    }

    /// The driver's answer now, asked once the display's gate is had by
    /// `deadline`: `None` while the work runs, `Err` past the deadline.
    fn ask(self, display: &Display, deadline: MonoTime) -> Result<Option<ffi::VAStatus>, Error> {
        let status = display.call_by(deadline, |raw| self.question(raw))?;
        Ok((status != ffi::VA_STATUS_ERROR_TIMEDOUT).then_some(status))
    }

    /// The driver's answer now, with no wait for the display's gate: `None`
    /// while the work runs, or while a call that waits holds the display.
    fn ask_now(self, display: &Display) -> Option<ffi::VAStatus> {
        let status = display.try_call(|raw| self.question(raw))?;
        (status != ffi::VA_STATUS_ERROR_TIMEDOUT).then_some(status)
    }
}

/// One deadline that several waits share, [`wait`] from the first of them
/// to ask (module docs). A clone shares it.
#[derive(Clone, Debug, Default)]
pub(crate) struct Deadline(Arc<OnceLock<MonoTime>>);

impl Deadline {
    /// The deadline, on `display`'s clock: [`wait`] from the first time it
    /// was asked for.
    pub(crate) fn at(&self, display: &Display) -> MonoTime {
        *self.0.get_or_init(|| deadline(display))
    }
}

thread_local! {
    /// The deadline of the frame in hand on this thread ([`by`]).
    static CURRENT: RefCell<Option<Deadline>> = const { RefCell::new(None) };
}

/// Puts the thread's deadline back as it was, as a [`by`] ends.
struct Restore(Option<Deadline>);

impl Drop for Restore {
    fn drop(&mut self) {
        let before = self.0.take();
        CURRENT.with_borrow_mut(|current| *current = before);
    }
}

/// `body`, with `deadline` the deadline of the frame in hand on this
/// thread, which each of its waits and the display's calls it makes take
/// ([`current`]).
pub(crate) fn by<T>(deadline: &Deadline, body: impl FnOnce() -> T) -> T {
    let before = CURRENT.with_borrow_mut(|current| current.replace(deadline.clone()));
    let _restore = Restore(before);
    body()
}

/// The deadline of the frame in hand on this thread ([`by`]); where none
/// is, [`wait`] from now.
pub(crate) fn current(display: &Display) -> MonoTime {
    CURRENT
        .with_borrow(|current| current.as_ref().map(|d| d.at(display)))
        .unwrap_or_else(|| deadline(display))
}

/// How long a frame's work is waited for: [`DEVICE_WAIT`], or what a test
/// of the hardware sets on its thread.
fn wait() -> Duration {
    #[cfg(test)]
    if let Some(wait) = tests::WAIT.get() {
        return wait;
    }
    DEVICE_WAIT
}

/// The deadline of a frame's work due now, on `display`'s clock.
pub(crate) fn deadline(display: &Display) -> MonoTime {
    display.runtime().deadline_in(wait())
}

/// Whether a driver's missing `vaSyncSurface2` was logged.
static SYNC_LOGGED: AtomicBool = AtomicBool::new(false);

/// Waits for `work` until it is done or `deadline` has passed: `Ok` once
/// the driver answers success, the driver's error where it answers one, and
/// [`Error::Stalled`], with `what`, past the deadline. A surface of a
/// driver without `vaSyncSurface2` is waited for in `vaSyncSurface`, made
/// on the display's call thread until the deadline.
pub(crate) fn wait_for(
    display: &Display,
    work: Work,
    deadline: MonoTime,
    what: &str,
) -> Result<(), Error> {
    let mut late = None;
    let answer = device_wait::poll_until(&**display.runtime(), deadline, || {
        match work.ask(display, deadline) {
            Ok(answer) => answer,
            Err(error) => {
                late = Some(error);
                Some(ffi::VA_STATUS_SUCCESS)
            }
        }
    });
    if let Some(error) = late {
        return Err(error);
    }
    match (answer, work) {
        (None, _) => Err(Error::Stalled(what.to_owned())),
        (Some(ffi::VA_STATUS_ERROR_UNIMPLEMENTED), Work::Surface(surface)) => {
            if !SYNC_LOGGED.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "the VA driver has no vaSyncSurface2, so its decodes and runs are waited for \
                     in vaSyncSurface, on the display's call thread"
                );
            }
            // A late call's error says "in time" itself.
            let call = what.strip_suffix(" in time").unwrap_or(what);
            let synced = display.wait_call_by(
                deadline,
                call,
                // SAFETY: a surface of this display, which the caller holds
                // until the call returns: its destroy is made behind it.
                move |raw| unsafe { ffi::vaSyncSurface(raw, surface) },
                |_, _| {},
            )?;
            check("vaSyncSurface", synced)
        }
        (Some(status), Work::Surface(_)) => check("vaSyncSurface2", status),
        (Some(status), Work::Buffer(_)) => check("vaSyncBuffer", status),
        (Some(_), Work::Calls) => Ok(()),
    }
}

/// Whether the driver answers success for each of `works`, asked with no
/// wait, the display's gate's among them.
pub(crate) fn all_done(display: &Display, works: &[Work]) -> bool {
    works
        .iter()
        .all(|work| work.ask_now(display) == Some(ffi::VA_STATUS_SUCCESS))
}

/// Keeps `holding`, which `what` names for the log, until the driver
/// answers success for each of `works` with no wait, then drops it
/// ([`keep_work`]): everything the works may still write, and the handles
/// they run on. An error answered keeps it until the process ends.
pub(crate) fn keep(
    display: &Display,
    what: impl Into<String>,
    works: Vec<Work>,
    holding: impl Send + 'static,
) {
    let kept = Kept {
        _holding: holding,
        works,
        display: display.clone_ref(),
    };
    keep_work(display, what, Box::new(kept));
}

/// Keeps `work`, which `what` names for the log, until it is finished, on
/// core's list of work kept for the process (`Lingering::process`), which
/// asks after it on a thread of its own.
pub(crate) fn keep_work(display: &Display, what: impl Into<String>, work: Box<dyn Unfinished>) {
    #[cfg(test)]
    tests::KEPT_EVER.fetch_add(1, Ordering::SeqCst);
    Lingering::process(display.runtime()).keep(what, work);
}

/// Work [`keep`] keeps: dropped in field order, what it holds first.
struct Kept<T> {
    _holding: T,
    /// Those not yet answered done.
    works: Vec<Work>,
    display: Display,
}

impl<T: Send> Unfinished for Kept<T> {
    fn finished(&mut self) -> bool {
        while let Some(&work) = self.works.last() {
            if work.ask_now(&self.display) != Some(ffi::VA_STATUS_SUCCESS) {
                return false;
            }
            self.works.pop();
        }
        true
    }
}

/// How many works are kept now, process-wide.
#[cfg(test)]
pub(crate) fn kept(display: &Display) -> usize {
    Lingering::process(display.runtime()).count()
}

#[cfg(test)]
mod gpu;
#[cfg(test)]
mod held;

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::Cell;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    thread_local! {
        /// A wait a test of the hardware sets in place of
        /// [`super::DEVICE_WAIT`] for the waits on its thread, so that work
        /// in flight passes its deadline.
        pub(crate) static WAIT: Cell<Option<Duration>> = const { Cell::new(None) };

        /// Set by a test, `vaSyncSurface2` is answered on its thread as a
        /// driver without it answers, `VA_STATUS_ERROR_UNIMPLEMENTED`, so
        /// that its waits are made in `vaSyncSurface` on the display's
        /// call thread.
        pub(crate) static NO_SYNC_SURFACE2: Cell<bool> = const { Cell::new(false) };
    }

    /// How many works were kept, process-wide, ever.
    pub(crate) static KEPT_EVER: AtomicUsize = AtomicUsize::new(0);

    /// Whether `ready` holds within 5 s, asked every millisecond.
    pub(crate) fn within_5_s(mut ready: impl FnMut() -> bool) -> bool {
        for _ in 0..5000 {
            if ready() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        ready()
    }
}
