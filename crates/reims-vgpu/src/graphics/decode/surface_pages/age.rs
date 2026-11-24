//! The timer of the view cache's age bound (the parent's module doc, "The cache", Age): a thread
//! of its own ([`VIEWS_AGE_THREAD`]) looks at the cache ([`SurfaceViews::age`]) every
//! [`AGE_LOOK`] while it keeps a view, so a view no block takes goes though the guest makes no new
//! one, which no other bound acts on. While the cache keeps none the thread waits with no
//! deadline; the next view cached tells it ([`ViewsAge::cached`]), its first look then
//! [`AGE_LOOK`] on, and each look that leaves a view in the cache sets the next. A message's
//! thread that caches a view does nothing more for it than an atomic load once a look is due.
//! Times are the platform's monotonic clock's, in nanoseconds.
//!
//! [`SurfaceViews::age`]: super::SurfaceViews::age

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::os::clock::{Clock as _, Deadline, MonoTime};
use reims_vgpu_vulkan::core::sync::{Condvar, Mutex, MutexGuard};

use super::super::surface_cache::Now;
use super::AGE_LOOK;

/// The name of the age's thread.
pub(crate) const VIEWS_AGE_THREAD: &str = "reims-vgpu.graphics-views-age";

/// The age's timer: when the next look is due, and whether its thread is to end.
pub(crate) struct ViewsAge {
    state: Mutex<AgeState>,
    /// `state`'s look is due: a view cached leaves it ([`Self::cached`]).
    armed: AtomicBool,
    moved: Condvar,
    /// The platform's clock, for the look a view cached sets.
    now: Now,
    /// The looks the thread made.
    looks: AtomicU64,
}

/// [`ViewsAge`]'s state.
struct AgeState {
    /// When the next look is due, where the cache keeps a view.
    due: Option<u64>,
    /// The views went ([`AgeStop`]): the thread ends.
    stopped: bool,
}

/// Stops the age's timer as the views go: its thread ends at its next look, which this wakes it
/// for.
pub(crate) struct AgeStop(pub(crate) Arc<ViewsAge>);

impl Drop for AgeStop {
    fn drop(&mut self) {
        self.0.state.lock().stopped = true;
        self.0.moved.notify_all();
    }
}

impl ViewsAge {
    /// A timer with no look due, whose looks are set on `now`.
    pub(crate) fn new(now: Now) -> ViewsAge {
        ViewsAge {
            state: Mutex::new(AgeState {
                due: None,
                stopped: false,
            }),
            armed: AtomicBool::new(false),
            moved: Condvar::new(),
            now,
            looks: AtomicU64::new(0),
        }
    }

    /// A view was cached: where no look is due, one [`AGE_LOOK`] on, its thread told. Called
    /// under the views' lock: it takes the timer's alone.
    pub(crate) fn cached(&self) {
        if self.armed.load(Ordering::Acquire) {
            return;
        }
        let at = (self.now)().as_nanos();
        let mut state = self.state.lock();
        if state.due.is_some() || state.stopped {
            return;
        }
        state.due = Some(at.saturating_add(AGE_LOOK));
        self.armed.store(true, Ordering::Release);
        drop(state);
        self.moved.notify_all();
    }

    /// The thread: waits for each look on `platform`'s clock and makes it then, `look` given the
    /// clock's time and answering whether the cache still keeps a view, for which the next is
    /// due [`AGE_LOOK`] on, or `None` once the views went; ends then, or once they went
    /// ([`AgeStop`]).
    pub(crate) fn run<P: Platform>(&self, platform: &P, mut look: impl FnMut(u64) -> Option<bool>) {
        let clock = platform.clock();
        let mut state = self.state.lock();
        loop {
            if state.stopped {
                return;
            }
            let Some(due) = state.due else {
                clock.wait(&self.moved, &mut state);
                continue;
            };
            let now = clock.now_mono().as_nanos();
            if now < due {
                let deadline = Deadline::Mono(MonoTime::from_nanos(due));
                let _ = clock.wait_until(&self.moved, &mut state, deadline);
                continue;
            }
            state.due = None;
            self.armed.store(false, Ordering::Release);
            let more = MutexGuard::unlocked(&mut state, || crate::entry::run(|_| look(now)));
            self.looks.fetch_add(1, Ordering::Relaxed);
            match more {
                None => return,
                Some(true) => {
                    let next = now.saturating_add(AGE_LOOK);
                    state.due = Some(state.due.map_or(next, |due| due.min(next)));
                    self.armed.store(true, Ordering::Release);
                }
                Some(false) => {}
            }
        }
    }

    /// When the next look is due, and the looks made (for the tests).
    #[cfg(test)]
    pub(crate) fn due_and_looks(&self) -> (Option<u64>, u64) {
        let due = self.state.lock().due;
        (due, self.looks.load(Ordering::Relaxed))
    }
}
