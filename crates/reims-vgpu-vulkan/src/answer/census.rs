//! Census mode, a diagnostic for inventory runs, never set in a product run: on Linux, with
//! `REIMS_VGPU_CENSUS_DIR=<dir>`, a degradation of [`DIES`](super::DIES) is recorded and relayed
//! where it would die, so one sweep of guests finds every kind the host's device announces,
//! instead of one death per boot. Without it, every death stays a death. On macOS the variable is
//! ignored, a warning saying so once ([`ON_THIS_OS`]): there the device runs on Metal (MoltenVK),
//! and where a run dies is the contract's, so no variable moves it.
//!
//! A run in census mode is not faithful to Metal: it goes on where Metal's answer is not met. So
//! the mode says so on stderr, and in the log as an error, once per process as it turns on, which
//! is as the process adopts its first device ([`super::native::device_made`] reads the variable),
//! before any degradation ([`announce_on`]).
//!
//! Each kind is recorded with how many times it was announced, the first object it was announced
//! for, and the selectors of the calls it was announced inside, each with its count (a notice
//! outside every call is counted under `(outside every call)`); a kind a native device relays
//! ([`super::native`]) is recorded too, without selectors. The kind's first occurrence relays the
//! runtime's line, as a relayed kind's.
//!
//! The record is written as JSON to `<dir>/census-<pid>.json` (through a temporary file renamed
//! over it), by the library's thread `reims-vgpu.census`: when a kind or a selector is first met,
//! and every 30 seconds while counts change; and by an exit handler (`atexit`, registered as the
//! writer starts, at the first record) as the process exits. A failed write is logged once and
//! tried again at the next.
//!
//! The exit's write is skipped where the runtime's exit guard ends the process at once. On Linux
//! with the GNU C library, an exit while the Vulkan driver runs the runtime's work (another
//! thread's call, a pipeline's making, a CPU device's commands) ends in the guard's exit handler,
//! with `_exit`, and one line on stderr saying so; the exit handlers registered before the guard's
//! last registration do not run. The runtime registers that handler again after later calls of its
//! own, so where one of those came after this handler's registration, the census is not written
//! at the exit. The file then holds the writer's last write: its counts up to 30 seconds old, and
//! without a kind or a selector first met in the moments before the exit.

use std::path::{Path, PathBuf};
use std::sync::{Once, OnceLock};
use std::time::Duration;

use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::os::{Clock as _, Deadline};
use reims_vgpu_core::sync::{Condvar, Mutex};

use super::native::{self, Placement};

/// The census's directory.
const CENSUS_VAR: &str = "REIMS_VGPU_CENSUS_DIR";

/// The longest time between two writes while counts change.
const PERIOD: Duration = Duration::from_secs(30);

/// The selector a notice outside every call is counted under.
pub const OUTSIDE: &str = "(outside every call)";

/// Whether census mode can be on in this build: on Linux alone (module docs).
const ON_THIS_OS: bool = cfg!(target_os = "linux");

/// The census's directory, read once per process ([`chosen`]).
fn dir() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| chosen(std::env::var(CENSUS_VAR).ok()))
        .as_deref()
}

/// The census's directory for `value`, the variable's: `None` (census off) where it is unset,
/// empty or not UTF-8, and, with a warning, where census mode cannot be on ([`ON_THIS_OS`]).
/// Census mode turning on says so on stderr ([`announce_on`]).
fn chosen(value: Option<String>) -> Option<PathBuf> {
    let dir = value.filter(|d| !d.is_empty()).map(PathBuf::from)?;
    if !ON_THIS_OS {
        log::warn!("census: {CENSUS_VAR} is ignored on macOS, where every death stays a death");
        return None;
    }
    announce_on(&mut std::io::stderr(), &dir);
    Some(dir)
}

/// Says on `w`, and in the log as an error, that census mode is on, recording into `dir`: every
/// degradation that dies elsewhere is relayed and counted, so the run is not faithful to Metal.
fn announce_on(w: &mut impl std::io::Write, dir: &Path) {
    let line = format!(
        "reims-vgpu: census mode is on ({CENSUS_VAR}={}): every degradation that would end the \
         process is relayed and counted instead, so this run is NOT faithful to Metal; never use \
         it for a product run",
        dir.display()
    );
    log::error!("{line}");
    // One write, so the line never interleaves with another thread's.
    let _ = w.write_all(format!("{line}\n").as_bytes());
}

/// Whether census mode is on.
pub fn on() -> bool {
    dir().is_some()
}

/// One kind's record.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Entry {
    kind: rt::Degradation,
    /// Whether a native device relays it ([`native::NATIVE_RELAYED`]), rather than a death.
    relayed_natively: bool,
    count: u64,
    first_object: String,
    /// Each selector it was announced inside, in the order first met, with its count.
    selectors: Vec<(String, u64)>,
}

/// The process's census.
#[derive(Debug, Default)]
struct Census {
    entries: Vec<Entry>,
    /// Whether it changed since its last write.
    dirty: bool,
}

impl Census {
    /// An occurrence of `kind` for `object`: whether it is the kind's first.
    fn record(&mut self, kind: rt::Degradation, object: &str, relayed_natively: bool) -> bool {
        self.dirty = true;
        if let Some(e) = self.entries.iter_mut().find(|e| e.kind == kind) {
            e.count = e.count.saturating_add(1);
            return false;
        }
        self.entries.push(Entry {
            kind,
            relayed_natively,
            count: 1,
            first_object: object.to_owned(),
            selectors: Vec::new(),
        });
        true
    }

    /// `kind`, recorded, announced inside a call standing for `sel`: whether `sel` is new for it.
    fn selector(&mut self, kind: rt::Degradation, sel: &str) -> bool {
        self.dirty = true;
        let Some(e) = self.entries.iter_mut().find(|e| e.kind == kind) else {
            return false;
        };
        if let Some((_, n)) = e.selectors.iter_mut().find(|(s, _)| s == sel) {
            *n = n.saturating_add(1);
            return false;
        }
        e.selectors.push((sel.to_owned(), 1));
        true
    }

    /// The census as JSON, for process `pid` under `placement`.
    fn json(&self, pid: u32, placement: Placement) -> String {
        let placement = match placement {
            Placement::OnMetal => "on-metal",
            Placement::Native => "native",
        };
        let mut out = format!("{{\"pid\": {pid}, \"placement\": \"{placement}\", \"kinds\": [");
        for (i, e) in self.entries.iter().enumerate() {
            out.push_str(if i == 0 { "\n  " } else { ",\n  " });
            out.push_str(&format!(
                "{{\"kind\": {}, \"name\": {}, \"placement\": \"{}\", \"count\": {}, \
                 \"first_object\": {}, \"selectors\": [",
                string(e.kind.code()),
                string(&format!("{:?}", e.kind)),
                if e.relayed_natively {
                    "relayed"
                } else {
                    "dies"
                },
                e.count,
                string(&e.first_object),
            ));
            for (j, (sel, n)) in e.selectors.iter().enumerate() {
                if j > 0 {
                    out.push_str(", ");
                }
                out.push_str(&format!(
                    "{{\"selector\": {}, \"count\": {n}}}",
                    string(sel)
                ));
            }
            out.push_str("]}");
        }
        out.push_str(if self.entries.is_empty() {
            "]}\n"
        } else {
            "\n]}\n"
        });
        out
    }
}

/// `s` as a JSON string.
fn string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The process's census.
static CENSUS: Mutex<Census> = Mutex::new(Census {
    entries: Vec::new(),
    dirty: false,
});

/// Whether the writer has a write to make now: a kind or a selector first met.
static WAKE: Mutex<bool> = Mutex::new(false);
/// Wakes the writer.
static WAKE_CV: Condvar = Condvar::new();

/// An occurrence of a degradation of [`DIES`](super::DIES) for `object`, in census mode, or one
/// a native device relays ([`native`]): recorded, and the writer started at the first. Whether it
/// is the kind's first, which the caller relays. Does nothing, answering `false`, with census
/// mode off.
pub fn record(kind: rt::Degradation, object: &str, relayed_natively: bool) -> bool {
    let Some(dir) = dir() else {
        return false;
    };
    start(dir);
    let first = CENSUS.lock().record(kind, object, relayed_natively);
    if first {
        wake();
    }
    first
}

/// `kind`, recorded, announced inside a call standing for `sel` ([`OUTSIDE`] for none).
pub fn selector(kind: rt::Degradation, sel: &str) {
    if dir().is_none() {
        return;
    }
    if CENSUS.lock().selector(kind, sel) {
        wake();
    }
}

/// Has the writer write now.
fn wake() {
    *WAKE.lock() = true;
    WAKE_CV.notify_one();
}

/// Starts the writer, and registers the write at exit, once per process.
fn start(dir: &'static Path) {
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        let made = std::thread::Builder::new()
            .name("reims-vgpu.census".into())
            .spawn(move || write_forever(dir));
        if let Err(e) = made {
            log::warn!("census: no writer thread: {e}; the census is written at exit alone");
        }
        // SAFETY: `at_exit` is a plain function that never unwinds (`write_if_dirty` does not
        // panic), registered once.
        if unsafe { atexit(at_exit) } != 0 {
            log::warn!("census: atexit refused: the census is not written at exit");
        }
    });
}

unsafe extern "C" {
    /// The C library's `atexit`.
    fn atexit(f: extern "C" fn()) -> i32;
}

/// The write at the process's exit, where the exit runs it (the module docs).
extern "C" fn at_exit() {
    if let Some(dir) = dir() {
        write_if_dirty(dir);
    }
}

/// The writer: each time it is woken, and every [`PERIOD`], the census if it changed.
fn write_forever(dir: &Path) -> ! {
    let clock = crate::os::OsClock::new();
    loop {
        {
            let mut wake = WAKE.lock();
            if !*wake {
                let deadline = Deadline::Mono(clock.now_mono().saturating_add(PERIOD));
                let _ = clock.wait_until(&WAKE_CV, &mut wake, deadline);
            }
            *wake = false;
        }
        write_if_dirty(dir);
    }
}

/// Writes the census into `dir` if it changed since its last write.
fn write_if_dirty(dir: &Path) {
    let json = {
        let mut census = CENSUS.lock();
        if !census.dirty {
            return;
        }
        census.dirty = false;
        census.json(std::process::id(), native::placement())
    };
    if let Err(e) = write(dir, &json) {
        CENSUS.lock().dirty = true;
        static SAID: Once = Once::new();
        SAID.call_once(|| log::warn!("census: cannot write into {}: {e}", dir.display()));
    }
}

/// `json` into `dir`'s file for this process, through a temporary file renamed over it.
fn write(dir: &Path, json: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let pid = std::process::id();
    let file = dir.join(format!("census-{pid}.json"));
    let temporary = dir.join(format!("census-{pid}.json.tmp"));
    std::fs::write(&temporary, json)?;
    std::fs::rename(&temporary, &file)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A kind's first occurrence is recorded with its object, later ones counted; a selector is
    /// counted under the kind, each in the order first met
    #[test]
    fn a_kind_is_recorded_once_and_counted() {
        let mut c = Census::default();
        let kind = rt::Degradation::ReleasedCommandState;
        assert!(c.record(kind, "command buffer 3", false));
        assert!(!c.record(kind, "command buffer 4", false));
        assert!(c.selector(kind, "commit"));
        assert!(!c.selector(kind, "commit"));
        assert!(c.selector(kind, OUTSIDE));
        assert!(!c.selector(rt::Degradation::DeviceInterpolation, "commit"));
        assert!(c.dirty);
        assert_eq!(
            c.entries,
            [Entry {
                kind,
                relayed_natively: false,
                count: 2,
                first_object: "command buffer 3".to_owned(),
                selectors: vec![("commit".to_owned(), 2), (OUTSIDE.to_owned(), 1)],
            }]
        );
    }

    /// The census as JSON: the process, its placement, and each kind with its counts, strings
    /// escaped
    #[test]
    fn the_census_is_json() {
        let mut c = Census::default();
        assert_eq!(
            c.json(7, Placement::OnMetal),
            "{\"pid\": 7, \"placement\": \"on-metal\", \"kinds\": []}\n"
        );
        let interpolation = rt::Degradation::DeviceInterpolation;
        c.record(interpolation, "pipeline \"blit\"\n", true);
        c.record(interpolation, "pipeline 2", true);
        let released = rt::Degradation::ReleasedCommandState;
        c.record(released, "a\\b", false);
        c.selector(released, "commit");
        c.selector(released, OUTSIDE);
        let want = format!(
            "{{\"pid\": 9, \"placement\": \"native\", \"kinds\": [\n  \
             {{\"kind\": \"{}\", \"name\": \"DeviceInterpolation\", \"placement\": \"relayed\", \
             \"count\": 2, \"first_object\": \"pipeline \\\"blit\\\"\\n\", \"selectors\": []}},\n  \
             {{\"kind\": \"{}\", \"name\": \"ReleasedCommandState\", \"placement\": \"dies\", \
             \"count\": 1, \"first_object\": \"a\\\\b\", \"selectors\": [{{\"selector\": \
             \"commit\", \"count\": 1}}, {{\"selector\": \"(outside every call)\", \"count\": 1}}]}}\n\
             ]}}\n",
            interpolation.code(),
            released.code()
        );
        assert_eq!(c.json(9, Placement::Native), want);
        assert_eq!(string("\u{1}"), "\"\\u0001\"");
    }

    /// The variable turns census mode on on Linux alone; unset or empty, it is off everywhere
    #[test]
    fn census_mode_is_linux_s_alone() {
        let linux = cfg!(target_os = "linux");
        assert_eq!(
            chosen(Some("/census".to_owned())),
            linux.then(|| PathBuf::from("/census"))
        );
        assert_eq!(chosen(Some(String::new())), None);
        assert_eq!(chosen(None), None);
    }

    /// Census mode turning on says, in one line, that the run is not faithful
    #[test]
    fn census_mode_says_the_run_is_not_faithful() {
        let mut w = Vec::new();
        announce_on(&mut w, Path::new("/census"));
        let line = String::from_utf8(w).unwrap();
        assert_eq!(line.lines().count(), 1, "{line}");
        assert!(line.contains("REIMS_VGPU_CENSUS_DIR=/census"), "{line}");
        assert!(line.contains("NOT faithful to Metal"), "{line}");
    }

    /// The file is written whole into the directory, made where missing, and named by the pid
    #[test]
    fn the_file_is_written_by_pid() {
        let dir = std::env::temp_dir().join(format!("rvg-census-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write(&dir.join("sub"), "{}\n").unwrap();
        let file = dir
            .join("sub")
            .join(format!("census-{}.json", std::process::id()));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "{}\n");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
