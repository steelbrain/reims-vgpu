//! A capture of the `log` crate's lines, for a test whose point is a line: [`capture`].
//!
//! The first capture installs the test logger, once per process: `log` takes one logger per
//! process, and whichever is installed first stays. A capture holds a process-wide lock, so the
//! tests that capture run one at a time, and while it is held every line up to `trace` is
//! recorded, from every thread. Outside a capture `log`'s level is `off`, so no line is kept and
//! no info or debug line evaluates its arguments.
//!
//! A test that captures relies on nextest running each test in its own process: only there are
//! the lines it takes its own test's, and is the test logger sure to be the process's rather than
//! [`crate::logging`]'s. Under `cargo test`'s threads, a line another test writes during the
//! capture is taken with it.
//!
//! ```ignore
//! let logs = testkit::logs::capture();
//! run_the_op();
//! assert_eq!(
//!     logs.take(),
//!     [(log::Level::Warn, "reims_vgpu::graphics".to_owned(), "stalled".to_owned())]
//! );
//! ```

#![allow(clippy::panic)] // a test harness reports failure by panicking (17.4: test-only code)

use std::sync::OnceLock;

use log::{Level, LevelFilter, Log, Metadata, Record};

use crate::sync::{Mutex, MutexGuard};

/// One line: its level, its target and its message.
pub type Line = (Level, String, String);

/// The test logger: every line it is given, in order.
struct TestLogger {
    lines: Mutex<Vec<Line>>,
}

impl Log for TestLogger {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &Record<'_>) {
        let line = (
            record.level(),
            record.target().to_owned(),
            record.args().to_string(),
        );
        self.lines.lock().push(line);
    }

    fn flush(&self) {}
}

static LOGGER: TestLogger = TestLogger {
    lines: Mutex::new(Vec::new()),
};

/// Held by the capture in progress.
static SERIAL: Mutex<()> = Mutex::new(());

/// A capture in progress: lines are recorded until it drops. Made by [`capture`].
#[must_use = "lines are recorded only while the capture is held"]
pub struct Capture {
    _serial: MutexGuard<'static, ()>,
}

/// Starts a capture: waits for any other to end, installs the test logger if it is not yet, drops
/// the lines recorded before, and turns every level on. Panics if the process has another
/// `log` logger.
pub fn capture() -> Capture {
    static INSTALLED: OnceLock<bool> = OnceLock::new();
    let serial = SERIAL.lock();
    if !*INSTALLED.get_or_init(|| log::set_logger(&LOGGER).is_ok()) {
        panic!("another `log` logger is installed in this process");
    }
    LOGGER.lines.lock().clear();
    log::set_max_level(LevelFilter::Trace);
    Capture { _serial: serial }
}

impl Capture {
    /// The lines recorded since the capture began or the last `take`, in order.
    pub fn take(&self) -> Vec<Line> {
        std::mem::take(&mut *LOGGER.lines.lock())
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        log::set_max_level(LevelFilter::Off);
        LOGGER.lines.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A capture takes each line with its level and target, in order, once; nothing is kept
    /// outside a capture
    #[test]
    fn a_capture_takes_its_lines_in_order() {
        log::warn!(target: "reims_vgpu::graphics", "before");
        let logs = capture();
        log::warn!(target: "reims_vgpu::graphics", "stalled {}", 3);
        log::debug!(target: "reims_vgpu::tasks", "detail");
        assert_eq!(
            logs.take(),
            [
                (
                    Level::Warn,
                    "reims_vgpu::graphics".to_owned(),
                    "stalled 3".to_owned()
                ),
                (
                    Level::Debug,
                    "reims_vgpu::tasks".to_owned(),
                    "detail".to_owned()
                ),
            ]
        );
        assert!(logs.take().is_empty());
        drop(logs);
        assert_eq!(log::max_level(), LevelFilter::Off);
        log::error!(target: "reims_vgpu::graphics", "after");
        assert!(capture().take().is_empty());
    }
}
