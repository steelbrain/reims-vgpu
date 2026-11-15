//! Making an arm64 VM: macOS restored from an IPSW onto a new disk, by reims-restore's pipeline
//! (crates/reims-restore). The pipeline writes the disk itself, on any host: it reads the IPSW
//! where it lies, has Apple's signing server (TSS) sign the VM's tickets, and leaves the VM's disk
//! (`disk.img`), its auxiliary storage (`aux.img`) and its ECID (`machine-identifier.bin`) in the
//! VM's directory, as a fresh restore by Virtualization.framework leaves them, with the VM's
//! LocalPolicies signed by the project's built-in keys, and `config.cpio` beside them, Ambrosia's
//! bundle of the patches that swap the guest's anchors for the project's. No VM runs, and none of
//! Apple's code. reims-ctl gives it the IPSW, the directory and the disk's size, shows
//! its steps and their progress, and writes the VM's settings (create.rs). The pipeline's notes
//! go only to the restore's log, `restore.log` in the VM's directory ([`LOG`]).
//!
//! The pipeline checks the space it writes once the IPSW's OS image is open, before it writes
//! anything. An IPSW to download is checked first, for the download and the restore together
//! ([`check_download_space`]), from an estimate of what the restore writes ([`space_needed`]).
//!
//! A restore stops when it fails, at Ctrl-C in the CLI, and when the interface quits
//! ([`cancel`]): the create removes the VM's directory, so nothing of the VM is left.

use crate::paths;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// What a restore is given.
pub struct Request<'a> {
    pub ipsw: &'a Path,
    /// The VM's directory, made and empty but for create's mark.
    pub dir: &'a Path,
    /// The disk's size, in bytes.
    pub disk_size: u64,
}

/// What a restore made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Restored {
    pub ecid: u64,
    /// The macOS restored: its version and build, such as 27.2.1 and 26B131.
    pub version: String,
    pub build: String,
    /// What the VM's first boot does: for macOS 14, it makes the Data volume, and reims-restore
    /// finishes the VM after it.
    pub first_boot: FirstBoot,
}

pub use reims_restore::pipeline::FirstBoot;

/// Restores macOS onto a VM's disk: reims-restore's [`Pipeline`], or a test's.
pub trait Restorer {
    /// Where its IPSWs come from.
    fn ipsws(&self) -> &dyn crate::ipsw::Source;
    fn restore(
        &self,
        r: &Request,
        progress: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
    ) -> Result<Restored>;
}

/// The restore's log in the VM's directory: each step, progress line and note, with the seconds
/// since the restore started.
pub const LOG: &str = "restore.log";

/// reims-restore's pipeline.
pub struct Pipeline;

impl Restorer for Pipeline {
    fn ipsws(&self) -> &dyn crate::ipsw::Source {
        &crate::ipsw::Apple
    }

    fn restore(
        &self,
        r: &Request,
        progress: &mut dyn FnMut(&str),
        cancel: &AtomicBool,
    ) -> Result<Restored> {
        use reims_restore::pipeline;
        let job = pipeline::Restore {
            ipsw: r.ipsw.to_path_buf(),
            dir: r.dir.to_path_buf(),
            disk_size: r.disk_size,
            ecid: None,
        };
        let mut shown = Shown::default();
        let mut log = Log::create(&r.dir.join(LOG))?;
        let done = pipeline::run(
            &job,
            &mut |e| on_event(e, &mut shown, &mut log, progress),
            cancel,
        )?;
        Ok(Restored {
            ecid: done.ecid,
            version: done.version,
            build: done.build,
            first_boot: done.first_boot,
        })
    }
}

/// Shows a step, and a progress line a whole percent on, and logs them; a note is only logged.
fn on_event(
    e: reims_restore::pipeline::Event,
    shown: &mut Shown,
    log: &mut Log,
    progress: &mut dyn FnMut(&str),
) {
    use reims_restore::pipeline::Event;
    match e {
        Event::Step(s) => {
            log.line(&s);
            progress(&s);
        }
        Event::Note(s) => log.line(&s),
        Event::Progress { what, done, total } => {
            if let Some(line) = shown.line(&what, done, total) {
                log.line(&line);
                progress(&line);
            }
        }
    }
}

/// The restore's log ([`LOG`]): a write that fails is left out, as the restore goes on.
struct Log {
    file: std::io::BufWriter<std::fs::File>,
    start: Instant,
}

impl Log {
    fn create(path: &Path) -> Result<Log> {
        let file = std::fs::File::create(path).with_context(|| format!("{}", path.display()))?;
        Ok(Log {
            file: std::io::BufWriter::new(file),
            start: Instant::now(),
        })
    }

    fn line(&mut self, text: &str) {
        use std::io::Write;
        let t = self.start.elapsed().as_secs_f64();
        let _ = writeln!(self.file, "[{t:8.1}] {text}");
        let _ = self.file.flush();
    }
}

/// Where the logs of failed restores are kept, a directory for each VM: the cache directory's
/// `restore/logs`.
pub fn logs_dir() -> PathBuf {
    paths::cache_dir().join("restore/logs")
}

/// A progress line for each whole percent, so a long step doesn't print a line per read.
#[derive(Default)]
pub struct Shown {
    last: Option<(String, u64)>,
}

impl Shown {
    pub fn line(&mut self, what: &str, done: u64, total: u64) -> Option<String> {
        let percent = (done.min(total) * 100).checked_div(total).unwrap_or(0);
        let now = (what.to_string(), percent);
        if self.last.as_ref() == Some(&now) {
            return None;
        }
        self.last = Some(now);
        Some(format!("{what}: {percent}%"))
    }
}

// ------------------------------------------------------------------ space

/// An estimate of the space a restore of `ipsw_len` bytes of IPSW writes, for a download's check
/// before the IPSW is here: its disk takes about what the IPSW holds (measured: a fresh restore
/// of 27.2.1 (26B131) by Virtualization.framework took 25.1 GiB of disk from a 23.8 GiB IPSW), so
/// a quarter more, and the auxiliary storage. The pipeline's own check, once the IPSW is here, is
/// exact; this reads low for an IPSW that keeps its images deflated, as 14's do.
pub fn space_needed(ipsw_len: u64) -> u64 {
    ipsw_len + ipsw_len / 4 + (64 << 20)
}

/// Refuses to download an IPSW of `total` bytes, `have` of which a stopped download left in
/// `downloads`, and restore it into `dir`, where the space free is short of what both write.
pub fn check_download_space(
    what: &str,
    total: u64,
    have: u64,
    downloads: &Path,
    dir: &Path,
) -> Result<()> {
    let rest = total.saturating_sub(have) + (64 << 20);
    let restore = space_needed(total);
    use reims_restore::pipeline::free_space;
    let (free_dl, free_vm) = (free_space(downloads)?, free_space(dir)?);
    if same_file_system(downloads, dir) {
        if free_vm < rest + restore {
            bail!(
                "downloading {what} and restoring it needs about {} free where the VM goes ({}): \
                 {} for the rest of its IPSW and {} for the VM, and there are {}: free some \
                 space, or put the VM on another disk (--dir)",
                gib(rest + restore),
                paths::tilde(dir),
                gib(rest),
                gib(restore),
                gib(free_vm)
            );
        }
        return Ok(());
    }
    if free_dl < rest {
        bail!(
            "downloading {what} needs about {} free in {}, and there are {}: free some space",
            gib(rest),
            paths::tilde(downloads),
            gib(free_dl)
        );
    }
    if free_vm < restore {
        bail!(
            "restoring {what} needs about {} free where the VM goes ({}), and there are {}: \
             free some space, or put the VM on another disk (--dir)",
            gib(restore),
            paths::tilde(dir),
            gib(free_vm)
        );
    }
    Ok(())
}

/// Whether `a` and `b`, or the nearest directories above them that exist, are on one file
/// system.
fn same_file_system(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt as _;
    let dev = |p: &Path| {
        p.ancestors()
            .find_map(|q| std::fs::metadata(q).ok())
            .map(|m| m.dev())
    };
    dev(a).is_some() && dev(a) == dev(b)
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

// ------------------------------------------------------------------ stopping

/// Set to stop every restore of this process: at Ctrl-C ([`stop_at_interrupt`]), or as the
/// interface quits.
static CANCEL: AtomicBool = AtomicBool::new(false);

/// The creates of arm64 VMs running in this process, each from its restore's start to its
/// directory's removal where it fails ([`Busy`]).
static RUNNING: AtomicUsize = AtomicUsize::new(0);

/// The flag a restore watches.
pub fn cancel_flag() -> &'static AtomicBool {
    &CANCEL
}

/// Stops every restore of this process.
pub fn cancel() {
    CANCEL.store(true, Ordering::SeqCst);
}

/// The arm64 creates running in this process.
pub fn running() -> usize {
    RUNNING.load(Ordering::SeqCst)
}

/// Waits up to `timeout` for every arm64 create to end; whether they did.
pub fn wait_stopped(timeout: Duration) -> bool {
    let end = Instant::now() + timeout;
    while running() > 0 {
        if Instant::now() >= end {
            return false;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    true
}

/// Held by an arm64 create while it runs, its restore and its cleanup.
pub struct Busy(());

impl Busy {
    pub fn new() -> Busy {
        RUNNING.fetch_add(1, Ordering::SeqCst);
        Busy(())
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        RUNNING.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Ctrl-C (SIGINT) and SIGTERM stop the restores of this process, and the downloads of their
/// IPSWs: a restore's VM directory is removed, a download keeps its part, and the create says the
/// VM was not made.
pub fn stop_at_interrupt() {
    extern "C" fn stop(_: libc::c_int) {
        // An atomic store is all a signal handler may do here.
        CANCEL.store(true, Ordering::SeqCst);
    }
    let handler = stop as extern "C" fn(libc::c_int) as libc::sighandler_t;
    // SAFETY: the handler only stores to an atomic.
    unsafe {
        libc::signal(libc::SIGINT, handler);
        libc::signal(libc::SIGTERM, handler);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_is_a_line_a_percent() {
        let mut s = Shown::default();
        assert_eq!(
            s.line("Installing macOS", 0, 200).as_deref(),
            Some("Installing macOS: 0%")
        );
        assert_eq!(s.line("Installing macOS", 1, 200), None);
        assert_eq!(
            s.line("Installing macOS", 2, 200).as_deref(),
            Some("Installing macOS: 1%")
        );
        assert_eq!(
            s.line("Verifying", 2, 200).as_deref(),
            Some("Verifying: 1%")
        );
        assert_eq!(
            s.line("Verifying", 300, 200).as_deref(),
            Some("Verifying: 100%")
        );
        assert_eq!(
            s.line("Unknown size", 5, 0).as_deref(),
            Some("Unknown size: 0%")
        );
    }

    /// Steps and progress are shown and logged; the pipeline's notes are only logged.
    #[test]
    fn notes_go_to_the_log_alone() {
        use reims_restore::pipeline::Event;
        let path = std::env::temp_dir().join(format!("reims-ctl-log-{}", std::process::id()));
        let mut log = Log::create(&path).unwrap();
        let mut shown = Shown::default();
        let mut seen = Vec::new();
        for e in [
            Event::Step("Signing the VM's tickets".into()),
            Event::Note("TSS answered (8 KiB)".into()),
            Event::Progress {
                what: "Writing the system volume".into(),
                done: 1,
                total: 2,
            },
        ] {
            on_event(e, &mut shown, &mut log, &mut |s| seen.push(s.to_string()));
        }
        drop(log);
        assert_eq!(
            seen,
            ["Signing the VM's tickets", "Writing the system volume: 50%"]
        );
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let lines: Vec<&str> = text
            .lines()
            .map(|l| &l[l.find("] ").unwrap() + 2..])
            .collect();
        assert_eq!(
            lines,
            [
                "Signing the VM's tickets",
                "TSS answered (8 KiB)",
                "Writing the system volume: 50%"
            ]
        );
        assert!(text.starts_with("[     0.0] "), "{text}");
    }

    /// A download and its restore on one file system need the space of both there.
    #[test]
    fn a_download_and_its_restore_need_the_space_of_both() {
        let dir = std::env::temp_dir().join(format!("reims-ctl-dlspace-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let free = reims_restore::pipeline::free_space(&dir).unwrap();
        // more than is free, for the restore alone
        let big = free;
        let e = check_download_space(
            "macOS 31.0 (30A100)",
            big,
            big,
            &dir.join("dl"),
            &dir.join("vm"),
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.starts_with("downloading macOS 31.0 (30A100) and restoring it needs about"),
            "{e}"
        );
        assert!(check_download_space("m", 1 << 20, 0, &dir.join("dl"), &dir.join("vm")).is_ok());
        assert!(same_file_system(&dir.join("a/b"), &dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_restore_is_estimated_at_the_ipsws_size_and_a_quarter() {
        assert_eq!(space_needed(20 << 30), (25 << 30) + (64 << 20));
    }
}
