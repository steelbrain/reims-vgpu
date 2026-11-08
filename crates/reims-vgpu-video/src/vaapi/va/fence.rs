//! Waiting for a video-processing run's writes with no lock of the
//! driver's held: on the dma-bufs the run writes, before its sync
//! (`super::wait`).
//!
//! Mesa's VA frontend waits for a video-processing run in `vaSyncSurface`
//! with its display's lock held: radeonsi's runs leave their fence as the
//! output surface's `pipe_fence`, which `_vlVaSyncSurface` waits for under
//! `drv->mutex`, where a decode's fence it waits for under the context's
//! lock alone (**read** in Mesa 26.2.4's `src/gallium/frontends/va/surface.c`).
//! Every VA call on the display takes that lock, so while one thread waits
//! there for a run, the session's next decode and every other session's
//! calls wait behind it. The kernel adds a run's fence to each dma-buf the
//! run writes, so a `poll` of those dma-bufs for writing waits for the run
//! with no lock held, and the sync after it finds the fence signalled and
//! returns at once.
//!
//! **Measured** on the RX 7900 XTX and the Radeon 780M (radeonsi, Mesa
//! 26.2.4, Linux 7.2.9, 2026-10-25), 4K runs into a guest's buffer and into
//! the staged copy back's memory: the wait moved into the poll whole,
//! `vaSyncSurface` taking 0.0 to 0.3 ms after it, and with decoding ahead
//! the decoder's `decode` took 0.3 to 0.9 ms a frame where it had taken 0.9
//! to 5.0 ms, with the CPU copying (docs/video/vaapi.md, "The driver's
//! lock"). A
//! driver that adds no fence leaves the poll nothing to wait for, and the
//! sync waits as before: it stays the wait the frame's correctness rests
//! on. The poll ends at the frame's deadline, and the sync after it then
//! asks once, with no wait, whether the run is done.

use std::ffi::{c_int, c_short, c_ulong};
use std::os::fd::{AsRawFd, BorrowedFd};

use crate::core::runtime::{MonoTime, Runtime};

unsafe extern "C" {
    fn poll(fds: *mut PollFd, count: c_ulong, timeout: c_int) -> c_int;
}

/// `struct pollfd`.
#[repr(C)]
struct PollFd {
    fd: c_int,
    events: c_short,
    revents: c_short,
}

/// `POLLOUT`: on a dma-buf, every fence on it signalled.
const POLLOUT: c_short = 0x4;

/// Waits until the GPU's work on each of `dma_bufs` is done, holding no
/// lock (module docs), or until `deadline` has passed on `runtime`'s clock,
/// past which the rest are not waited for.
pub(crate) fn wait_for_writes<'a>(
    runtime: &dyn Runtime,
    deadline: MonoTime,
    dma_bufs: impl IntoIterator<Item = BorrowedFd<'a>>,
) {
    for dma_buf in dma_bufs {
        // A wait a signal interrupts is made again, for what is left.
        loop {
            let left = deadline.saturating_duration_since(runtime.now());
            if left.is_zero() {
                return;
            }
            // Whole milliseconds, rounded up, so that the last wait does
            // not end before the bound.
            let timeout = c_int::try_from(left.as_micros().div_ceil(1000)).unwrap_or(c_int::MAX);
            let mut one = PollFd {
                fd: dma_buf.as_raw_fd(),
                events: POLLOUT,
                revents: 0,
            };
            // SAFETY: one `pollfd` of an open descriptor, for the call.
            let ready = unsafe { poll(&mut one, 1, timeout) };
            if ready == 0 {
                return;
            }
            if ready > 0
                || std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted
            {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{ErrorKind, Write};
    use std::os::fd::AsFd;
    use std::os::unix::net::UnixStream;
    use std::time::Duration;

    use crate::core::runtime::StdRuntime;

    use super::wait_for_writes;

    /// A socket whose send buffer is full: never ready for writing while
    /// its peer reads nothing, as a dma-buf whose fence never signals.
    fn full() -> (UnixStream, UnixStream) {
        let (a, b) = UnixStream::pair().unwrap_or_else(|_| unreachable!());
        a.set_nonblocking(true).unwrap_or_else(|_| unreachable!());
        let chunk = [0u8; 4096];
        loop {
            match (&a).write(&chunk) {
                Ok(_) => {}
                Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                Err(_) => unreachable!(),
            }
        }
        (a, b)
    }

    #[test]
    fn the_wait_is_bounded_for_all_the_dma_bufs_together() {
        let pairs: Vec<_> = (0..5).map(|_| full()).collect();
        let most = Duration::from_millis(100);
        let runtime = StdRuntime::shared();
        let started = runtime.now();
        let deadline = runtime.deadline_in(most);
        wait_for_writes(&*runtime, deadline, pairs.iter().map(|(a, _)| a.as_fd()));
        let waited = runtime.elapsed_since(started);
        // One bound for the five, where a bound each waited 500 ms.
        assert!(waited >= most, "{waited:?}");
        assert!(waited < most * 4, "{waited:?}");
    }

    #[test]
    fn a_ready_dma_buf_is_not_waited_for() {
        let (a, _b) = UnixStream::pair().unwrap_or_else(|_| unreachable!());
        let runtime = StdRuntime::shared();
        let started = runtime.now();
        let deadline = runtime.deadline_in(Duration::from_secs(10));
        wait_for_writes(&*runtime, deadline, [a.as_fd()]);
        assert!(runtime.elapsed_since(started) < Duration::from_secs(5));
    }
}
