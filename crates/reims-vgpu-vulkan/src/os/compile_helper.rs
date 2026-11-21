//! Pipeline compiles in a helper process: the library's side of starting one, capping it and
//! ending it.
//!
//! rt refuses, in the VM's process, a pipeline whose stage weighs past its stage bound and whose
//! compile is not in its store yet, and its refusal carries an opaque request. The library then
//! starts the helper, a process that opens a runtime device of its own on the same card and
//! compiles the stage, and hands the compile's cache back to rt in the VM's process, which keeps it
//! in its store for the VM's later creates. The helper's memory and time are capped ([`Limits`]).
//! Past either it is killed, and the caller records the breach with rt, so later creates refuse
//! without starting a helper again.
//!
//! **One at a time on the host.** A helper first takes a lock file that all the host's processes
//! share ([`HelperRun::lock`]; `File::try_lock`, which is `flock` on Linux and macOS and
//! `LockFileEx` on Windows), and holds it until its process has ended, so the VMs of one host
//! never compile two stages at once.
//!
//! **No caller's thread waits.** [`Helper::start`] returns at once. The helper's own thread,
//! `reims-vgpu.compile-helper`, waits for the lock, starts the process and watches it. Once the
//! process has ended and the lock is free, it calls the caller's `on_end` with how the process
//! ended ([`HelperEnd`]). [`Helper::wait`] is for a caller with nothing else to do. Dropping a
//! [`Helper`] cancels it, and waits for nothing.
//!
//! **The talk.** rt owns what the two processes say: in the VM's process `Device::fill_stage`
//! writes the request to the helper's stdin and reads the compile's cache from its stdout, and the
//! helper's main serves it. [`Helper::start`] runs the caller's `talk` with those pipes on a thread
//! of its own, `reims-vgpu.compile-helper-talk`, started only once the process is watched, so none
//! of the helper's work on its request runs unwatched. A kill ends the talk too: its pipes then
//! read their end, and the talk returns. Its result goes to `on_end` with the end. On Linux and
//! macOS the talk's thread blocks `SIGPIPE`, so a write to a helper that has ended fails with
//! `EPIPE` rather than signalling the host's process.
//!
//! **The memory cap** has one form on every OS: every [`POLL`] the helper's thread reads the
//! process's footprint, and kills it once that is past [`Limits::memory`]:
//! - Linux: resident and swapped memory (`VmRSS` and `VmSwap` of `/proc/<pid>/status`), so swap,
//!   zram's among it, does not hide what the helper holds. An address-space limit (`RLIMIT_AS`)
//!   would count the virtual ranges a GPU driver reserves at device creation, which no memory
//!   backs.
//! - macOS: its physical footprint (`ri_phys_footprint` of `proc_pid_rusage`), the figure the
//!   system's memory pressure acts on. macOS enforces neither `RLIMIT_AS` nor `RLIMIT_DATA`.
//! - Windows: its private commit (`PrivateUsage` of `GetProcessMemoryInfo`).
//!
//! The footprint read that crossed the cap is in the end, [`HelperEnd::OverMemory`], so a run
//! can tell how far past the cap a helper got within one poll.
//!
//! **The time cap** is wall time from the process's start, [`Limits::time`], on every OS.
//!
//! **The library's end ends the helper**, by each OS's own means, so no helper outlives the
//! process that started it, and nothing at that process's exit waits for one:
//! - Linux: `PR_SET_PDEATHSIG` (`SIGKILL`), set in the child before it runs the helper. It fires
//!   when the thread that started the child ends, and that thread is the helper's own, which
//!   lives until the process has ended.
//! - macOS, which has no such signal: the helper watches its parent's exit (`kqueue`'s
//!   `NOTE_EXIT`), armed by [`end_with_parent`] as its main begins, and exits at once.
//! - Windows: a Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` and no other limit, whose
//!   one handle the library's process holds. Its end closes the job, which ends the helper. The
//!   process joins the job just after its start, before the talk begins.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "its caller, the handler of rt's stage refusal, lands with the helper's binary"
    )
)]

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(test)]
mod tests;
#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

#[cfg(target_os = "linux")]
use linux as host;
#[cfg(target_os = "macos")]
use macos as host;
#[cfg(windows)]
use windows as host;

use std::ffi::OsString;
use std::fs::{File, TryLockError};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::core::platform::os::{Clock, Deadline};
use crate::core::sync::{Condvar, Mutex, MutexGuard};
use crate::os::OsClock;

/// The most memory a helper may hold on any host: 8 GiB.
const MEMORY_MOST: u64 = 8 << 30;

/// The longest a helper may run: 10 minutes.
const TIME_MOST: Duration = Duration::from_secs(10 * 60);

/// How often the helper's thread reads the process's footprint, and looks for its end and for a
/// cancel. A cancel wakes it at once.
pub(crate) const POLL: Duration = Duration::from_millis(100);

/// A helper's caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Limits {
    /// The most its footprint may reach, in bytes (the module's doc says what each OS counts).
    pub(crate) memory: u64,
    /// The longest it may run, in wall time from its start.
    pub(crate) time: Duration,
}

impl Limits {
    /// The caps on this host: the smaller of 8 GiB and half the host's memory, and 10 minutes.
    pub(crate) fn of_host() -> Limits {
        Limits::for_host_memory(host::host_memory())
    }

    /// The caps on a host with `total` bytes of memory; 8 GiB where the OS does not say, as a cap
    /// of nothing would kill every helper, and record a breach for every stage it compiles.
    fn for_host_memory(total: Option<u64>) -> Limits {
        let half = total.map_or(MEMORY_MOST, |total| {
            total / const { std::num::NonZeroU64::new(2).unwrap() }
        });
        Limits {
            memory: MEMORY_MOST.min(half),
            time: TIME_MOST,
        }
    }
}

/// What a helper runs.
#[derive(Debug)]
pub(crate) struct HelperRun {
    /// The helper's program.
    pub(crate) program: PathBuf,
    /// Its arguments.
    pub(crate) args: Vec<OsString>,
    /// The variables added to the environment it inherits.
    pub(crate) envs: Vec<(OsString, OsString)>,
    /// Its caps.
    pub(crate) limits: Limits,
    /// The lock file that every helper on the host takes before it starts, created where it is
    /// missing.
    pub(crate) lock: PathBuf,
}

/// How a helper ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HelperEnd {
    /// It ran to its end, with this status.
    Ended(ExitStatus),
    /// It was killed when its footprint, `bytes` at that read, was past its memory cap.
    OverMemory {
        /// The footprint read that crossed the cap, in bytes.
        bytes: u64,
    },
    /// It was killed at its time cap.
    OverTime,
    /// It was cancelled: killed, or never started where the cancel came while it waited for the
    /// lock.
    Cancelled,
    /// It never started: the lock, the process, its watch or its talk's thread failed, as said.
    NotStarted(String),
    /// It was killed because its footprint or its status could not be read, as said, so its caps
    /// could not hold.
    Unwatched(String),
}

/// A helper's state, which its thread and its [`Helper`] share.
#[derive(Debug, Default)]
struct State {
    /// Whether it was cancelled.
    cancelled: bool,
    /// Its process's id, once the process has started.
    pid: Option<u32>,
    /// How it ended, once it has.
    end: Option<HelperEnd>,
}

/// [`State`] under its lock, and the condition its changes are signalled on.
#[derive(Debug, Default)]
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
}

impl Shared {
    /// Waits on `state` until [`POLL`] from now, or until a change is signalled.
    fn wait_a_poll(&self, clock: &OsClock, state: &mut MutexGuard<'_, State>) {
        let deadline = Deadline::Mono(clock.now_mono().saturating_add(POLL));
        clock.wait_until(&self.changed, state, deadline);
    }
}

/// A helper process, from its start until it has ended: the caller's handle on it.
#[derive(Debug)]
pub(crate) struct Helper {
    shared: Arc<Shared>,
}

impl Helper {
    /// Starts the helper `run` names on a thread of its own, which takes the host's lock, starts
    /// and watches the process, and runs `talk` with the process's stdin and stdout on another
    /// thread. Once the process has ended, the talk has returned and the lock is free, it calls
    /// `on_end` with how the process ended and the talk's result: `None` where no talk ran, or
    /// where it panicked. Returns at once; fails only where the helper's thread cannot start.
    pub(crate) fn start<T: Send + 'static>(
        run: HelperRun,
        talk: impl FnOnce(ChildStdin, ChildStdout) -> T + Send + 'static,
        on_end: impl FnOnce(&HelperEnd, Option<T>) + Send + 'static,
    ) -> io::Result<Helper> {
        let shared = Arc::new(Shared::default());
        let theirs = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("reims-vgpu.compile-helper".to_owned())
            .spawn(move || {
                let (end, said) = watch(run, talk, &theirs);
                theirs.state.lock().end = Some(end.clone());
                theirs.changed.notify_all();
                on_end(&end, said);
            })?;
        Ok(Helper { shared })
    }

    /// The process's id, once it has started.
    pub(crate) fn pid(&self) -> Option<u32> {
        self.shared.state.lock().pid
    }

    /// Cancels the helper: its thread kills the process at once, or starts none where it still
    /// waits for the lock. The end is [`HelperEnd::Cancelled`] unless the process ended first.
    pub(crate) fn cancel(&self) {
        self.shared.state.lock().cancelled = true;
        self.shared.changed.notify_all();
    }

    /// Waits for the helper's end.
    pub(crate) fn wait(&self) -> HelperEnd {
        let clock = OsClock::new();
        let mut state = self.shared.state.lock();
        loop {
            if let Some(end) = &state.end {
                return end.clone();
            }
            clock.wait(&self.shared.changed, &mut state);
        }
    }
}

impl Drop for Helper {
    /// Cancels the helper, and waits for nothing: its thread kills the process and frees the lock.
    fn drop(&mut self) {
        self.cancel();
    }
}

/// The helper's thread: takes the lock, runs the process and its talk until both have ended, then
/// frees the lock.
fn watch<T: Send + 'static>(
    run: HelperRun,
    talk: impl FnOnce(ChildStdin, ChildStdout) -> T + Send + 'static,
    shared: &Shared,
) -> (HelperEnd, Option<T>) {
    let clock = OsClock::new();
    let lock = match take_lock(&run.lock, shared, &clock) {
        Ok(Some(lock)) => lock,
        Ok(None) => return (HelperEnd::Cancelled, None),
        Err(e) => {
            let end = HelperEnd::NotStarted(format!("the lock {}: {e}", run.lock.display()));
            return (end, None);
        }
    };
    let ended = run_watched(run, talk, shared, &clock);
    drop(lock);
    ended
}

/// Takes the host's lock file at `path`, polling it until it is free; `None` where the helper was
/// cancelled first.
fn take_lock(path: &Path, shared: &Shared, clock: &OsClock) -> io::Result<Option<File>> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    let mut state = shared.state.lock();
    loop {
        if state.cancelled {
            return Ok(None);
        }
        match file.try_lock() {
            Ok(()) => return Ok(Some(file)),
            Err(TryLockError::WouldBlock) => shared.wait_a_poll(clock, &mut state),
            Err(TryLockError::Error(e)) => return Err(e),
        }
    }
}

/// Starts the process and watches it, starts its talk, and polls the process until it ends or is
/// killed; then waits for the talk's end.
fn run_watched<T: Send + 'static>(
    run: HelperRun,
    talk: impl FnOnce(ChildStdin, ChildStdout) -> T + Send + 'static,
    shared: &Shared,
    clock: &OsClock,
) -> (HelperEnd, Option<T>) {
    let mut command = Command::new(&run.program);
    command
        .args(&run.args)
        .envs(run.envs.iter().map(|(k, v)| (k, v)))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        // The helper's lines go to the host's stderr, as the library's do.
        .stderr(Stdio::inherit());
    host::prepare(&mut command);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            let end = HelperEnd::NotStarted(format!("{}: {e}", run.program.display()));
            return (end, None);
        }
    };
    let started = clock.now_mono();
    let watched = match host::Watched::new(&child) {
        Ok(watched) => watched,
        Err(e) => {
            end_process(&mut child);
            return (HelperEnd::NotStarted(format!("its watch: {e}")), None);
        }
    };
    shared.state.lock().pid = Some(child.id());
    let talking = match start_talk(&mut child, talk) {
        Ok(talking) => talking,
        Err(e) => {
            end_process(&mut child);
            return (HelperEnd::NotStarted(format!("its talk: {e}")), None);
        }
    };
    let deadline = Deadline::Mono(started.saturating_add(run.limits.time));
    let end = poll(&mut child, &watched, run.limits, deadline, shared, clock);
    // The talk ends once the process has: its pipes read their end, and its writes fail.
    let said = talking.join().ok();
    // `watched` goes before `child`, whose process handle it reads on Windows.
    (end, said)
}

/// Runs `talk` with the process's pipes on a thread of its own.
fn start_talk<T: Send + 'static>(
    child: &mut Child,
    talk: impl FnOnce(ChildStdin, ChildStdout) -> T + Send + 'static,
) -> io::Result<JoinHandle<T>> {
    let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
        return Err(io::Error::other("the process has no pipes"));
    };
    std::thread::Builder::new()
        .name("reims-vgpu.compile-helper-talk".to_owned())
        .spawn(move || {
            host::mask_broken_pipes();
            talk(stdin, stdout)
        })
}

/// Polls the process every [`POLL`] until it ends, or kills it at a cancel, past its memory cap,
/// at its time cap, or where its footprint cannot be read.
fn poll(
    child: &mut Child,
    watched: &host::Watched,
    limits: Limits,
    deadline: Deadline,
    shared: &Shared,
    clock: &OsClock,
) -> HelperEnd {
    loop {
        if let Some(end) = ended(child) {
            return end;
        }
        let killed = if shared.state.lock().cancelled {
            Some(HelperEnd::Cancelled)
        } else {
            match watched.footprint() {
                Ok(bytes) if bytes > limits.memory => Some(HelperEnd::OverMemory { bytes }),
                Ok(_) => deadline.reached(clock).then_some(HelperEnd::OverTime),
                // A process that ended since the look above has no footprint to read.
                Err(e) => match ended(child) {
                    Some(end) => return end,
                    None => Some(HelperEnd::Unwatched(format!("its footprint: {e}"))),
                },
            }
        };
        if let Some(end) = killed {
            end_process(child);
            return end;
        }
        let mut state = shared.state.lock();
        if !state.cancelled {
            shared.wait_a_poll(clock, &mut state);
        }
    }
}

/// How the process ended, where it has. A look that fails kills it, as its caps cannot hold.
fn ended(child: &mut Child) -> Option<HelperEnd> {
    match child.try_wait() {
        Ok(Some(status)) => Some(HelperEnd::Ended(status)),
        Ok(None) => None,
        Err(e) => {
            end_process(child);
            Some(HelperEnd::Unwatched(format!("its status: {e}")))
        }
    }
}

/// Kills the process and reaps it.
fn end_process(child: &mut Child) {
    // A process that has already ended refuses the kill, and is reaped all the same.
    let _ = child.kill();
    let _ = child.wait();
}

/// The helper's side: arms what ends the helper with the library's process where the OS needs the
/// helper's part, on macOS (the module's doc). The helper's main calls it first.
pub(crate) fn end_with_parent() -> io::Result<()> {
    host::watch_parent()
}
