//! A timeline of the device's work for profiling, one line per event, written to the file
//! `REIMS_VGPU_TRACE` names: `<ns> <thread> <kind> <values...>`, the time in nanoseconds of the
//! clock the platform installs ([`set_clock`]), the thread a number of this process's (its name
//! written once, as `thread <number> <name>`).
//!
//! It is a diagnostic: with the variable unset, or before a platform installed its clock, no
//! event is written and [`on`] answers `false`, and no event reads anything with an effect. Each
//! site's comment names its event and its values.
//!
//! An event only appends its line to a buffer, under a lock it holds for the append. A thread of
//! the trace's own (`reims-vgpu.trace`) writes the buffer to the file every quarter second: it
//! swaps the buffer for an empty one under that lock, and writes with the lock let go, so no
//! traced thread ever waits for the file. Lines are in the file in the order their events took
//! the lock. A file whose writes stall keeps its lines in memory until it takes them. Lines still
//! buffered as the process ends are lost, but for a death's ([`flush_before_death`]).
//!
//! A write made holding the events' lock holds every traced thread for as long as the file
//! stalls. In a traced boot on homelab-um870-1 (2026-11-15), with the file on a workspace disk
//! whose writes stalled for up to 219 ms, the trace went silent for 31 to 219 ms at a time right
//! after an event of the control thread, every traced thread going on within 50 us of the others
//! after it, while the guest's command buffers waited for ring 0 behind the control thread.

use std::cell::Cell;
use std::ffi::OsStr;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write as _;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::core::sync::Mutex;

/// The environment variable naming the file.
pub const ENV_VAR: &str = "REIMS_VGPU_TRACE";

/// How long a line may wait in the buffer.
const FLUSH: Duration = Duration::from_millis(250);

/// How long a death's flush waits for the trace's thread to finish a write, and for an event's
/// append ([`flush_before_death`]).
const DEATH_WAIT: Duration = Duration::from_millis(500);

/// The buffer and its file, or `None` with the variable unset or the file refused.
static WRITER: OnceLock<Option<Writer>> = OnceLock::new();

/// The platform's clock, in nanoseconds.
static CLOCK: OnceLock<fn() -> u64> = OnceLock::new();

/// The next thread number.
static THREADS: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// This thread's number, 0 before its first event.
    static THREAD: Cell<u64> = const { Cell::new(0) };
}

/// The lines events appended and the file they go to. A write takes `out`, then `lines` only to
/// swap the buffer out; an event takes `lines` alone.
struct Writer {
    lines: Mutex<Vec<u8>>,
    out: Mutex<Box<dyn std::io::Write + Send>>,
}

impl Writer {
    fn new(out: Box<dyn std::io::Write + Send>) -> Self {
        Self {
            lines: Mutex::new(Vec::with_capacity(1 << 20)),
            out: Mutex::new(out),
        }
    }

    /// Writes the lines the buffer holds: swaps them for `spare` (emptied, its capacity kept)
    /// under the events' lock, and writes them with only the file's held.
    fn write_out(&self, spare: &mut Vec<u8>) {
        let mut out = self.out.lock();
        std::mem::swap(&mut *self.lines.lock(), spare);
        if !spare.is_empty() {
            let _ = out.write_all(spare);
            let _ = out.flush();
            spare.clear();
        }
    }
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

fn writer() -> Option<&'static Writer> {
    let mut made = false;
    let writer = WRITER
        .get_or_init(|| {
            made = true;
            open(&std::env::var_os(ENV_VAR)?)
        })
        .as_ref();
    if made && let Some(writer) = writer {
        start(writer);
    }
    writer
}

/// Makes `writer` the process's, and starts its thread; `false` where the process chose its file
/// already.
#[cfg(any(test, feature = "testkit"))]
fn install(writer: Option<Writer>) -> bool {
    if WRITER.set(writer).is_err() {
        return false;
    }
    if let Some(writer) = WRITER.get().and_then(Option::as_ref) {
        start(writer);
    }
    true
}

/// Writes events to the file at `path`, created afresh, in place of the one the variable names,
/// for a test of a site's events; `false` where the process chose its file already (nextest runs
/// each test in a process of its own). The clock is still the one installed ([`set_clock`]).
#[cfg(any(test, feature = "testkit"))]
pub fn to_file(path: &std::path::Path) -> bool {
    install(open(path.as_os_str()))
}

/// The file at `path`, created afresh; `None`, with a warning, where it is refused.
fn open(path: &OsStr) -> Option<Writer> {
    let file = File::create(path)
        .inspect_err(|e| log::warn!("{ENV_VAR}: {}: {e}", path.to_string_lossy()))
        .ok()?;
    Some(Writer::new(Box::new(file)))
}

/// Starts the thread that writes `writer`'s lines every [`FLUSH`]. Where it cannot be made, the
/// lines wait for a death's flush, with a warning.
fn start(writer: &'static Writer) {
    // The trace's thread runs no Objective-C and none of the platform's work: it sleeps and
    // writes a file, for a static the platform never sees.
    #[allow(clippy::disallowed_methods)]
    let spawned = std::thread::Builder::new()
        .name("reims-vgpu.trace".into())
        .spawn(move || {
            crate::core::rt::thread::become_ours();
            let mut spare = Vec::with_capacity(1 << 20);
            loop {
                std::thread::sleep(FLUSH);
                writer.write_out(&mut spare);
            }
        });
    if let Err(e) = spawned {
        log::warn!("{ENV_VAR}: no thread to write the trace: {e}");
    }
}

/// Writes the lines the buffer holds to the file, for a death, which ends the process with no
/// `Drop` run ([`crate::core::raise::death`]). It opens no file. It waits up to [`DEATH_WAIT`] for
/// the trace's thread to finish a write it is in, and for an event's append, and writes nothing
/// where either takes longer: a file that stalls never holds the death up for longer.
pub fn flush_before_death() {
    let Some(writer) = WRITER.get().and_then(Option::as_ref) else {
        return;
    };
    let Some(mut out) = writer.out.try_lock_for(DEATH_WAIT) else {
        return;
    };
    let Some(mut lines) = writer.lines.try_lock_for(DEATH_WAIT) else {
        return;
    };
    let taken = std::mem::take(&mut *lines);
    drop(lines);
    let _ = out.write_all(&taken);
    let _ = out.flush();
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

/// Appends `line` to the buffer, for the trace's thread to write.
fn write_line(line: &str) {
    if let Some(writer) = writer() {
        writer.lines.lock().extend_from_slice(line.as_bytes());
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
        assert!(to_file(&path));
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

    /// A file whose write stalls holds up no traced thread: while the trace's thread is stuck in
    /// the write, events on another thread still return, and once the write goes on every line
    /// reaches the file, in the order of the events
    #[test]
    fn a_stalled_write_never_blocks_a_tracer() {
        use std::sync::{Arc, mpsc};
        /// A file whose first write waits for `go`, after telling `entered`.
        struct Stalling {
            entered: mpsc::Sender<()>,
            go: mpsc::Receiver<()>,
            stalled: bool,
            got: Arc<Mutex<Vec<u8>>>,
        }
        impl std::io::Write for Stalling {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if !self.stalled {
                    self.stalled = true;
                    let _ = self.entered.send(());
                    let _ = self.go.recv();
                }
                self.got.lock().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let (entered_tx, entered) = mpsc::channel();
        let (go, go_rx) = mpsc::channel();
        let got = Arc::new(Mutex::new(Vec::new()));
        // nextest runs each test in a process of its own, whose trace is this one
        assert!(install(Some(Writer::new(Box::new(Stalling {
            entered: entered_tx,
            go: go_rx,
            stalled: false,
            got: Arc::clone(&got),
        })))));
        set_clock(|| 7);
        event("first", &[0]);
        // The trace's thread takes the line within a quarter second, and stalls in its write.
        entered
            .recv_timeout(Duration::from_secs(10))
            .expect("the trace's thread wrote");
        let (done_tx, done) = mpsc::channel();
        #[allow(clippy::disallowed_methods)] // test helper thread: no Objective-C runs here
        std::thread::spawn(move || {
            for i in 1..=1000 {
                event("next", &[i]);
            }
            let _ = done_tx.send(());
        });
        done.recv_timeout(Duration::from_secs(10))
            .expect("a traced thread waited for the stalled write");
        go.send(()).unwrap();
        flush_before_death();
        let text = String::from_utf8(got.lock().clone()).unwrap();
        let values: Vec<(&str, u64)> = text
            .lines()
            .filter_map(|l| {
                let f: Vec<&str> = l.split(' ').collect();
                matches!(f[2], "first" | "next").then(|| (f[2], f[3].parse().unwrap()))
            })
            .collect();
        let want: Vec<(&str, u64)> = std::iter::once(("first", 0))
            .chain((1..=1000).map(|i| ("next", i)))
            .collect();
        assert_eq!(values, want);
    }
}
