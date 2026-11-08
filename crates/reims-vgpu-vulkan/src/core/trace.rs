//! A timeline of the device's work for profiling, one line per event, written to the file
//! `REIMS_VGPU_TRACE` names: `<ns> <thread> <kind> <values...>`, the time in nanoseconds of the
//! clock the platform installs ([`set_clock`]), the thread a number of this process's (its name
//! written once, as `thread <number> <name>`).
//!
//! It is a diagnostic: with the variable unset, or before a platform installed its clock, no
//! event is written and [`on`] answers `false`, and no event reads anything with an effect. Each
//! site's comment names its event and its values.
//!
//! Lines are buffered, and written when the buffer fills or with the first event a quarter second
//! or more after the last write. Lines still buffered when events stop wait for the next event,
//! or for a death ([`flush_before_death`]); a process that ends otherwise loses them.

use std::cell::Cell;
use std::ffi::OsStr;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::sync::Mutex;

/// The environment variable naming the file.
pub const ENV_VAR: &str = "REIMS_VGPU_TRACE";

/// How long a line may wait in the buffer, in nanoseconds.
const FLUSH_NS: u64 = 250_000_000;

/// The open file and when it was last written, or `None` with the variable unset or the file
/// refused.
static WRITER: OnceLock<Option<Mutex<Writer>>> = OnceLock::new();

/// The platform's clock, in nanoseconds.
static CLOCK: OnceLock<fn() -> u64> = OnceLock::new();

/// The next thread number.
static THREADS: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// This thread's number, 0 before its first event.
    static THREAD: Cell<u64> = const { Cell::new(0) };
}

struct Writer {
    out: BufWriter<File>,
    flushed: u64,
}

/// Installs the clock events are timed by: the platform's, once per process. Without it nothing
/// is written.
pub fn set_clock(clock: fn() -> u64) {
    let _ = CLOCK.set(clock);
}

/// The clock now, 0 without one.
pub fn now() -> u64 {
    CLOCK.get().map_or(0, |clock| clock())
}

/// Whether events are written.
pub fn on() -> bool {
    CLOCK.get().is_some() && writer().is_some()
}

fn writer() -> Option<&'static Mutex<Writer>> {
    WRITER
        .get_or_init(|| open(&std::env::var_os(ENV_VAR)?))
        .as_ref()
}

/// The file at `path`, created afresh; `None`, with a warning, where it is refused.
fn open(path: &OsStr) -> Option<Mutex<Writer>> {
    let file = File::create(path)
        .inspect_err(|e| log::warn!("{ENV_VAR}: {}: {e}", path.to_string_lossy()))
        .ok()?;
    Some(Mutex::new(Writer {
        out: BufWriter::with_capacity(1 << 20, file),
        flushed: 0,
    }))
}

/// Writes the lines the buffer holds to the file, for a death, which ends the process with no
/// `Drop` run ([`crate::core::raise::death`]). It opens no file, and takes the file's lock only where
/// it is free: the dying thread may hold it, or a thread the death stops before it lets go.
pub fn flush_before_death() {
    if let Some(mut w) = WRITER
        .get()
        .and_then(Option::as_ref)
        .and_then(Mutex::try_lock)
    {
        let _ = w.out.flush();
    }
}

/// Writes one event, `kind` and `values`, timed now.
pub fn event(kind: &str, values: &[u64]) {
    if !on() {
        return;
    }
    event_at(now(), kind, values);
}

/// Writes one event, `kind` and `values`, timed `at`.
pub fn event_at(at: u64, kind: &str, values: &[u64]) {
    let mut line = String::with_capacity(96);
    let _ = write!(line, "{at} {} {kind}", thread_number());
    for v in values {
        let _ = write!(line, " {v}");
    }
    line.push('\n');
    write_line(&line);
}

/// Writes one event whose last value is text (a label: its spaces become `_`).
pub fn event_text(kind: &str, values: &[u64], text: &str) {
    if !on() {
        return;
    }
    let mut line = String::with_capacity(96);
    let _ = write!(line, "{} {} {kind}", now(), thread_number());
    for v in values {
        let _ = write!(line, " {v}");
    }
    line.push(' ');
    line.extend(
        text.chars()
            .map(|c| if c.is_whitespace() { '_' } else { c }),
    );
    line.push('\n');
    write_line(&line);
}

fn write_line(line: &str) {
    let Some(writer) = writer() else {
        return;
    };
    let mut w = writer.lock();
    let _ = w.out.write_all(line.as_bytes());
    let t = now();
    if t.saturating_sub(w.flushed) >= FLUSH_NS {
        let _ = w.out.flush();
        w.flushed = t;
    }
}

/// This thread's number; its first event writes its name.
fn thread_number() -> u64 {
    THREAD.with(|n| {
        if n.get() == 0 {
            let number = THREADS.fetch_add(1, Ordering::Relaxed);
            n.set(number);
            let name = std::thread::current()
                .name()
                .unwrap_or("unnamed")
                .replace(char::is_whitespace, "_");
            write_line(&format!("{} {number} thread {number} {name}\n", now()));
        }
        n.get()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without the variable nothing is written, whatever the clock
    #[test]
    fn off_without_the_variable() {
        // nextest runs each test in a process of its own, where the variable is not set
        assert!(std::env::var_os(ENV_VAR).is_none());
        set_clock(|| 7);
        assert!(!on());
        event("cmd", &[1, 2]);
        event_text("cbcommit", &[1], "a label");
        assert_eq!(now(), 7);
    }

    /// With a file, each event is a line, its thread's name written once before its first, a
    /// text's spaces as `_`; a death's flush writes what the buffer held
    #[test]
    fn a_line_per_event() {
        // nextest runs each test in a process of its own, where the file is this one
        let path = std::env::temp_dir().join(format!("rvg-trace-{}.txt", std::process::id()));
        assert!(WRITER.set(open(path.as_os_str())).is_ok());
        set_clock(|| 7);
        assert!(on());
        event("imp", &[6, 0x4000, 64, 0x4000]);
        event_text("cbcommit", &[1], "a label");
        event("impfree", &[0x4000]);
        flush_before_death();
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let name = std::thread::current()
            .name()
            .unwrap_or("unnamed")
            .replace(char::is_whitespace, "_");
        assert_eq!(
            text,
            format!(
                "7 1 thread 1 {name}\n7 1 imp 6 16384 64 16384\n7 1 cbcommit 1 a_label\n\
                 7 1 impfree 16384\n"
            )
        );
    }
}
