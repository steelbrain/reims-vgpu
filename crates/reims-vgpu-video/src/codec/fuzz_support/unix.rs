//! The fuzzers' harness on Linux and macOS: the CPU time from
//! `clock_gettime`, and a SIGABRT handler that names the input in hand.

use std::io::Write;
use std::sync::atomic::Ordering;
use std::time::Duration;

use super::{Accounting, CASE_INDEX, CASE_NAME, CASE_SEED};

/// [`Accounting::cpu_time`]: `CLOCK_THREAD_CPUTIME_ID` or
/// `CLOCK_PROCESS_CPUTIME_ID`.
pub(super) fn cpu_time(accounting: Accounting) -> Duration {
    let clock = match accounting {
        Accounting::Thread => libc::CLOCK_THREAD_CPUTIME_ID,
        Accounting::Process => libc::CLOCK_PROCESS_CPUTIME_ID,
    };
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a timespec the call writes, and nothing else.
    let read = unsafe { libc::clock_gettime(clock, &mut now) };
    assert_eq!(read, 0, "clock_gettime({clock}) failed");
    Duration::new(
        u64::try_from(now.tv_sec).unwrap_or(0),
        u32::try_from(now.tv_nsec).unwrap_or(0),
    )
}

/// Names the input in hand on stderr as the process aborts, as it does
/// after a thread overflows its stack (Rust's handler prints the thread
/// and aborts) and past [`HARD_LIMIT`](super::HARD_LIMIT): abort raises
/// SIGABRT, whose handler this is, and ends the process once it returns.
pub(super) fn install_abort_report() {
    unsafe extern "C" {
        fn signal(signal: i32, handler: extern "C" fn(i32)) -> usize;
    }
    /// SIGABRT, 6 on Linux and on macOS.
    const ABORT: i32 = 6;
    // SAFETY: the handler touches only atomics and a buffer on its own
    // stack, and writes it to stderr.
    unsafe {
        signal(ABORT, report_abort);
    }
}

/// SIGABRT's handler: the input in hand, written without allocating.
extern "C" fn report_abort(_: i32) {
    unsafe extern "C" {
        fn write(fd: std::ffi::c_int, bytes: *const std::ffi::c_void, count: usize) -> isize;
    }
    let name = CASE_NAME.load(Ordering::SeqCst);
    // SAFETY: null, or a name `run` leaked, which lives for ever.
    let name = unsafe { name.as_ref() }.map_or(&b""[..], |name| name.as_bytes());
    let mut line = [0u8; 300];
    let mut cursor = std::io::Cursor::new(&mut line[..]);
    let _ = cursor.write_all(b"fuzz ");
    let _ = cursor.write_all(name);
    let _ = writeln!(
        cursor,
        ": the process aborted in input {} of seed {:#x}; RVGV_FUZZ_SEED and \
         RVGV_FUZZ_INDEX replay it",
        CASE_INDEX.load(Ordering::SeqCst),
        CASE_SEED.load(Ordering::SeqCst),
    );
    let end = usize::try_from(cursor.position()).unwrap_or(0);
    // SAFETY: `end` bytes of `line` are initialised.
    unsafe {
        write(2, line.as_ptr().cast(), end);
    }
}
