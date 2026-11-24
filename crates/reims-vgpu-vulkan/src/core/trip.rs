//! The platform's stamps of a command buffer's trip (reims-vgpu-graphics' `trip`): when rt's
//! completed handler entered, and the span of rt's commit call, each on the clock the host
//! installs ([`set_clock`], the device's), handed to the decoder on the thread that runs them;
//! and, from rt's own times of the command buffer (on rt's host clock), how long the GPU sat idle
//! before it and its pickup ([`gpu`]). A diagnostic: before a clock is installed nothing is
//! stamped, and each site costs one load.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// The clock the stamps read, in nanoseconds.
static CLOCK: OnceLock<fn() -> u64> = OnceLock::new();

/// The latest GPU end of the command buffers whose handlers ran, on rt's host clock.
static LAST_GPU_END: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// When rt's completed handler this thread runs entered, and whether it was handed on from
    /// rt's thread.
    static HANDLER: Cell<(u64, bool)> = const { Cell::new((0, false)) };
    /// Its command buffer's idle before it and its pickup ([`gpu`]).
    static GPU: Cell<Option<(u64, u64)>> = const { Cell::new(None) };
    /// When this thread's last commit called rt's, and how long that call took.
    static COMMIT: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

/// Stamps from now on, on `clock`; the first clock installed stays.
pub fn set_clock(clock: fn() -> u64) {
    let _ = CLOCK.set(clock);
}

/// The installed clock's time, 0 before one is installed.
pub(crate) fn stamp() -> u64 {
    CLOCK.get().map_or(0, |clock| clock())
}

/// rt's completed handler this thread runs entered at `at` (0: none), handed on where `handed`,
/// its command buffer's [`gpu`] `gpu`.
pub(crate) fn set_handler(at: u64, handed: bool, gpu: Option<(u64, u64)>) {
    HANDLER.set((at, handed));
    GPU.set(gpu);
}

/// A command buffer's idle and pickup from rt's times of it, in nanoseconds of rt's host clock:
/// its submission's end (`kernel_end`, the submit call's return) and its GPU's start and end. The
/// idle is its GPU start less the latest GPU end of those before it whose handlers ran, 0 where
/// it starts before that; the pickup its GPU start less its submission's end. `None` where rt
/// holds no GPU times of it, or no command buffer's end came before.
pub(crate) fn gpu_times(kernel_end: u64, start: u64, end: u64) -> Option<(u64, u64)> {
    if start == 0 || end == 0 || kernel_end == 0 {
        return None;
    }
    let last = LAST_GPU_END.fetch_max(end, Ordering::Relaxed);
    (last != 0).then(|| (start.saturating_sub(last), start.saturating_sub(kernel_end)))
}

/// The idle and pickup ([`gpu_times`]) of the command buffer whose completed handler this thread
/// runs.
pub fn gpu() -> Option<(u64, u64)> {
    GPU.get()
}

/// When rt's completed handler this thread runs entered (0 before a clock is installed), and
/// whether it was handed on from rt's thread.
pub fn handler() -> (u64, bool) {
    HANDLER.get()
}

/// This thread's commit called rt's at `at`, which took `ns`.
pub(crate) fn set_commit(at: u64, ns: u64) {
    COMMIT.set((at, ns));
}

/// When this thread's last commit called rt's and how long that call took, taken: `None` where it
/// made none since the last take, or before a clock is installed.
pub fn take_commit() -> Option<(u64, u64)> {
    let (at, ns) = COMMIT.replace((0, 0));
    (at != 0).then_some((at, ns))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A commit's stamps are taken once, on the thread that made them
    #[test]
    fn a_commits_stamps_are_taken_once() {
        assert_eq!(take_commit(), None);
        set_commit(5, 7);
        std::thread::scope(|scope| {
            scope.spawn(|| assert_eq!(take_commit(), None));
        });
        assert_eq!(take_commit(), Some((5, 7)));
        assert_eq!(take_commit(), None);
    }

    /// A command buffer's idle runs from the latest end before it, and its pickup from its
    /// submission's end
    #[test]
    fn a_command_buffers_idle_runs_from_the_latest_end_before_it() {
        // The first: no end before it.
        assert_eq!(gpu_times(10, 20, 30), None);
        assert_eq!(gpu_times(40, 50, 60), Some((20, 10)));
        // One that starts before the latest end: no idle.
        assert_eq!(gpu_times(45, 55, 70), Some((0, 10)));
        // No GPU times: nothing.
        assert_eq!(gpu_times(80, 0, 0), None);
    }
}
