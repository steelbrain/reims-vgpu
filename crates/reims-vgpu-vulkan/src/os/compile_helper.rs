//! Pipeline compiles in a helper process: the library's side of starting one, capping it and
//! ending it.
//!
//! rt refuses, in the VM's process, a compute pipeline whose kernel weighs past the kernels' stage
//! bound and whose compile is not in its store yet, and its refusal carries an opaque request (a
//! render stage compiles in the process up to the render stages' own bound, and is refused past it,
//! as no helper compiles one: the counts line (`crate::rt_platform::device`) carries the render
//! makes past the kernels' bound). The library then starts the helper, a process that opens a
//! runtime device of its own on the same card and compiles the stage, and hands the compile's cache
//! back to rt in the VM's process, which keeps it in its store for the VM's later creates. The
//! helper's memory and time are capped ([`Limits`]), its memory at most at the ceiling,
//! [`Limits::memory`], and lower where the host's room at its seat is less ([`seat_cap`]). Past the
//! ceiling or its time cap it is killed, and the caller records the breach with rt, so later
//! creates refuse without starting a helper again. Past a cap the room set below the ceiling it is
//! killed too, [`HelperEnd::OverRoom`], and nothing is recorded: the same stage fits on a host with
//! room.
//!
//! **Side by side, as the host's memory allows.** The host's helpers share its seats ([`Seats`]),
//! one lock file per logical CPU of the host ([`host_cpus`]) that all its processes share
//! (`File::try_lock`, which is `flock` on Linux and macOS and `LockFileEx` on Windows). A helper
//! holds a seat from before its process starts until the process has ended. It takes one beside
//! running helpers only where their memory caps and its own fit in what the host has free beside
//! the guest's RAM ([`admits`]), so no helper's growth leaves another, or the guest, short of
//! memory; with no helper running it always takes one, so one helper always runs, at a cap no
//! larger than the room read then ([`seat_cap`]), and nothing waits on the room for it. A helper
//! counts the seats held and takes one under one more lock file, the seats' guard, so no two
//! processes count the same room at once.
//!
//! **In request order.** A process's helpers take seats in the order they were started
//! ([`Helper::start`]), which is the order of the stages' requests: only the first of its helpers
//! that wait tries for a seat, every [`POLL`], reading the host's room again at most every
//! [`ROOM_POLL`] while the room keeps it waiting.
//!
//! **The room** ([`room_of_host`]) is what the host can give now less what of the guest's RAM is
//! not resident yet, which the guest takes as it touches its pages: the host ranges of the guest's
//! RAM are named to it while a device made over them lives ([`hold_guest_ram`]). What this
//! process's running helpers hold now counts toward the room too, as their caps already count it.
//! - Linux: `MemAvailable` of `/proc/meminfo`, and no more than the headroom of each
//!   memory-capped cgroup the process is in (its `memory.max` less its `memory.current`, with its
//!   inactive page cache free), as a container or a capped service unit would kill past it; less
//!   the guest's pages `mincore` finds not resident. reims-ctl's x86-64 VMs preallocate their RAM
//!   (`prealloc=on`), all of it resident from the VM's start.
//! - macOS: the free (less the speculative), file-backed and purgeable pages of
//!   `host_statistics64`, less the guest's pages `mincore` finds not resident.
//! - Windows: the smaller of the available physical memory and the physical memory no commit
//!   charge has taken (`GlobalMemoryStatusEx`): the guest's pagefile-backed section is charged in
//!   full at its creation, so nothing of the guest's is taken off again.
//!
//! Where the OS does not say, there is no room: one helper runs at a time, at the ceiling.
//!
//! **No caller's thread waits.** [`Helper::start`] returns at once. The helper's own thread,
//! `reims-vgpu.compile-helper`, waits for its seat, starts the process and watches it. Once the
//! process has ended and the seat is free, it calls the caller's `on_end` with how the process
//! ended ([`HelperEnd`]). Dropping a [`Helper`] cancels it, and waits for nothing.
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
//! **The memory cap** is the helper's own, set at its seat ([`seat_cap`]): the ceiling,
//! [`Limits::memory`], for a helper beside others, which [`admits`] fit in the room, and where the
//! OS says no room; for a helper alone, the smaller of the ceiling and the room read at its seat,
//! however little that is: where the room is under what the helper needs, it passes its cap at
//! its first reads and is killed. It has one form on every OS: every [`POLL`] the helper's thread
//! reads the process's footprint, and kills it once that is past its cap:
//! - Linux: resident and swapped memory (`VmRSS` and `VmSwap` of `/proc/<pid>/status`), so swap,
//!   zram's among it, does not hide what the helper holds. An address-space limit (`RLIMIT_AS`)
//!   would count the virtual ranges a GPU driver reserves at device creation, which no memory
//!   backs.
//! - macOS: its physical footprint (`ri_phys_footprint` of `proc_pid_rusage`), the figure the
//!   system's memory pressure acts on. macOS enforces neither `RLIMIT_AS` nor `RLIMIT_DATA`.
//! - Windows: its private commit (`PrivateUsage` of `GetProcessMemoryInfo`).
//!
//! The footprint read that crossed the cap is in the end, [`HelperEnd::OverMemory`] where the cap
//! was the ceiling and [`HelperEnd::OverRoom`] where the room set it lower, so a run can tell how
//! far past the cap a helper got within one poll. Every end comes with what the reads saw
//! ([`Footprints`]): the helper's cap, the largest footprint, and the largest growth from one read
//! to the next, which bounds how far past the cap a helper can get before its kill.
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
//!
//! **A helper's own end and its death.** A helper that ends of its own accord having served no
//! reply, as the helper's main does where it finds nothing to serve the request on, exits with a
//! code of its own, [`SERVED_NONE`]. A helper died ([`HelperEnd::died`]) where a fault ended it
//! with no kill of the library's: a fault's signal on Linux and macOS, an exception's status on
//! Windows, or a panic of its main thread. No other end is a death: an exit with a code, an end
//! another process asks for (a terminal's Ctrl+C or its close, a service's stop, the OOM killer),
//! a host policy's kill, or a start the loader failed on Linux or Windows. The library keeps a
//! death before the helper's reply as its stage's breach, and asks again at the VM's next start
//! after any other end (`rt_platform::stage`).
//!
//! **A helper's crash writes no core file.** The helper's main first sets its own core-file
//! limit to none ([`no_core_files`]), whatever the limit it inherits from the library's process,
//! with no switch. A helper that crashes is said by its end ([`HelperEnd::Ended`], its signal
//! where it had one), which the library logs with its fill's key, and by the stderr it shares with
//! the library's process; a crash before its reply is a death, which its fill keeps as a breach
//! (above).
//! - Linux and macOS: `RLIMIT_CORE` at 0, soft and hard (`setrlimit`). The kernel writes no core
//!   file under it, and `systemd-coredump`, where a `core_pattern` pipe hands it the crash, keeps
//!   no core of a process whose limit is below a page.
//! - Windows: nothing, as Windows has no core-file limit of a process's own; whether a crash
//!   leaves a dump is Windows Error Reporting's, by the host's settings.

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

use std::collections::VecDeque;
use std::ffi::OsString;
use std::fmt;
use std::fs::{File, TryLockError};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::core::num::ToU64 as _;
use crate::core::platform::os::{Clock, Deadline, MonoTime};
use crate::core::sync::{Condvar, Mutex, MutexGuard};
use crate::os::OsClock;

/// The most memory a helper may hold on any host: 3 GiB.
const MEMORY_MOST: u64 = 3 << 30;

/// The longest a helper may run: 10 minutes.
const TIME_MOST: Duration = Duration::from_secs(10 * 60);

/// How often the helper's thread reads the process's footprint, and looks for its end and for a
/// cancel. A cancel wakes it at once.
pub(crate) const POLL: Duration = Duration::from_millis(100);

/// How often the first helper that waits reads the host's room again while the room keeps it
/// waiting: a read walks the guest's RAM's pages (`mincore`).
const ROOM_POLL: Duration = Duration::from_secs(1);

/// The exit code of a helper that ends of its own accord having served no reply: the helper's
/// main where it found nothing to serve the request on or its pipes failed
/// (`rt_platform::stage::helper_main`), and on macOS its watch of its parent
/// (`macos::watch_parent`), at the parent's exit or where the watch fails. 75, `EX_TEMPFAIL` of
/// `sysexits.h`: a later ask may be served. No fault ends a process with it ([`HelperEnd::died`]).
pub(crate) const SERVED_NONE: u8 = 75;

/// Rust's exit code of a process whose main thread panicked.
const PANICKED: i32 = 101;

/// A helper's caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Limits {
    /// The most its footprint may reach, in bytes, the ceiling (the module's doc says what each OS
    /// counts): a helper alone on the host's seats is held lower where the room at its seat is less
    /// ([`seat_cap`]).
    pub(crate) memory: u64,
    /// The longest it may run, in wall time from its start.
    pub(crate) time: Duration,
}

impl Limits {
    /// The caps on this host: the smaller of 3 GiB and half the host's memory, and 10 minutes.
    pub(crate) fn of_host() -> Limits {
        Limits::for_host_memory(host::host_memory())
    }

    /// The caps on a host with `total` bytes of memory; 3 GiB where the OS does not say, as a cap
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
    /// The host's seats, one of which it takes before it starts.
    pub(crate) seats: Seats,
    /// What its lines name it by.
    pub(crate) label: String,
}

/// The host's room beside the guest's RAM, in bytes, as [`Seats::room`] reads it: `None` where
/// the OS does not say.
pub(crate) type Room = Arc<dyn Fn() -> Option<u64> + Send + Sync>;

/// The host's seats ([`Seats::count`] lock files beside the guard, `<guard's stem>.<n>.lock`,
/// created where they are missing), and the room a helper beside running ones is admitted by
/// ([`admits`]).
#[derive(Clone)]
pub(crate) struct Seats {
    /// The guard: the lock file a helper holds while it counts the seats held and takes one.
    pub(crate) guard: PathBuf,
    /// How many seats there are, one at the least.
    pub(crate) count: usize,
    /// The host's room beside the guest's RAM.
    pub(crate) room: Room,
}

impl fmt::Debug for Seats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Seats")
            .field("guard", &self.guard)
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

impl Seats {
    /// The host's seats at `guard`: one per logical CPU of the host ([`host_cpus`]), and the
    /// host's room ([`room_of_host`]).
    pub(crate) fn of_host(guard: PathBuf) -> Seats {
        let count = host_cpus()
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(1);
        Seats {
            guard,
            count,
            room: Arc::new(room_of_host),
        }
    }

    /// Seat `n`'s lock file.
    fn seat(&self, n: usize) -> PathBuf {
        self.guard.with_extension(format!("{n}.lock"))
    }

    /// Removes the guard's and the seats' files (tests only).
    #[cfg(test)]
    pub(crate) fn remove_files(&self) {
        let _ = std::fs::remove_file(&self.guard);
        for n in 0..self.count.max(1) {
            let _ = std::fs::remove_file(self.seat(n));
        }
    }
}

/// The host's logical CPUs, every one it runs, whatever the process's own affinity or quota;
/// `None` where the OS does not say.
fn host_cpus() -> Option<u64> {
    host::host_cpus().filter(|&n| n > 0)
}

/// Whether a helper whose memory cap is `cap` may take a seat beside `held` helpers running on
/// the host, where the host has `room` bytes free beside the guest's RAM ([`room_of_host`]) and
/// this process's running helpers hold `ours` bytes now: always where none runs, so one helper
/// always can, at a cap no larger than the room ([`seat_cap`]); otherwise where the caps of
/// `held + 1` helpers fit in the room and what this process's helpers hold, which their caps count
/// again. No room said is none.
fn admits(held: usize, cap: u64, room: Option<u64>, ours: u64) -> bool {
    held == 0
        || room.is_some_and(|room| {
            cap.saturating_mul(held.to_u64().saturating_add(1)) <= room.saturating_add(ours)
        })
}

/// The memory cap of a helper whose ceiling is `ceiling` ([`Limits::memory`]), seated as `beside`
/// says: alone, the smaller of the ceiling and the room read at its seat, so a helper never holds
/// more than the host had free then, a room of nothing among it; the ceiling beside others, which
/// [`admits`] fit in the room, and where the OS says no room, as nothing then says the host is
/// short.
fn seat_cap(ceiling: u64, beside: Beside) -> u64 {
    match (beside.held, beside.room) {
        (0, Some(room)) => ceiling.min(room),
        _ => ceiling,
    }
}

/// The host ranges of a hold's guest RAM, each its host address and length.
type Ranges = Vec<(u64, u64)>;

/// The host ranges of the guest's RAM this process maps, by the id of the hold that named them
/// ([`hold_guest_ram`]).
static GUEST_RAM: Mutex<Vec<(u64, Ranges)>> = Mutex::new(Vec::new());

/// The next hold's id.
static NEXT_HOLD: AtomicU64 = AtomicU64::new(0);

/// The guest's RAM, named to the helpers' room until it drops ([`hold_guest_ram`]).
#[derive(Debug)]
pub struct GuestRamHeld {
    id: u64,
}

/// Names the host ranges of the guest's RAM, each its host address and its length in bytes, to
/// the helpers' room ([`room_of_host`]) until the hold drops: what of them is not resident yet is
/// the guest's to take, and no helper's.
pub fn hold_guest_ram(ranges: impl IntoIterator<Item = (u64, u64)>) -> GuestRamHeld {
    let id = NEXT_HOLD.fetch_add(1, Ordering::Relaxed);
    GUEST_RAM.lock().push((id, ranges.into_iter().collect()));
    GuestRamHeld { id }
}

impl Drop for GuestRamHeld {
    fn drop(&mut self) {
        GUEST_RAM.lock().retain(|(id, _)| *id != self.id);
    }
}

/// The host's room beside the guest's RAM, in bytes (the module's doc): what the OS says it can
/// give now, less what of the guest's RAM named to it is not resident; `None` where the OS does
/// not say.
pub(crate) fn room_of_host() -> Option<u64> {
    let free = host::free_memory()?;
    let mut ranges: Ranges = (GUEST_RAM.lock().iter())
        .flat_map(|(_, ranges)| ranges.iter().copied())
        .collect();
    // Two devices over one guest's RAM name its ranges twice.
    ranges.sort_unstable();
    ranges.dedup();
    Some(free.saturating_sub(host::unresident(&ranges)))
}

/// This process's helpers on one host's seats ([`Seats::guard`]): those that wait for a seat, in
/// the order they were started, and the last footprint read of each that holds one.
#[derive(Debug)]
struct Line {
    guard: PathBuf,
    waiting: VecDeque<u64>,
    running: Vec<(u64, u64)>,
}

/// This process's lines, one per seats' guard.
static LINES: Mutex<Vec<Line>> = Mutex::new(Vec::new());

/// The next helper's place.
static NEXT_PLACE: AtomicU64 = AtomicU64::new(0);

/// A helper's place in its process's line on its seats, from its start until it has ended: it
/// waits, then runs; its drop leaves the line.
#[derive(Debug)]
struct Place {
    guard: PathBuf,
    id: u64,
}

impl Place {
    /// A place at the back of the line on `guard`.
    fn join(guard: &Path) -> Place {
        let id = NEXT_PLACE.fetch_add(1, Ordering::Relaxed);
        let mut lines = LINES.lock();
        match lines.iter_mut().find(|line| line.guard == guard) {
            Some(line) => line.waiting.push_back(id),
            None => lines.push(Line {
                guard: guard.to_path_buf(),
                waiting: VecDeque::from([id]),
                running: Vec::new(),
            }),
        }
        Place {
            guard: guard.to_path_buf(),
            id,
        }
    }

    /// `f` of its line.
    fn with<R: Default>(&self, f: impl FnOnce(&mut Line) -> R) -> R {
        (LINES.lock().iter_mut())
            .find(|line| line.guard == self.guard)
            .map_or_else(R::default, f)
    }

    /// Whether it is the first of the helpers that wait.
    fn first(&self) -> bool {
        self.with(|line| line.waiting.front() == Some(&self.id))
    }

    /// What this process's helpers that hold a seat on its guard hold now, in bytes.
    fn ours(&self) -> u64 {
        self.with(|line| {
            (line.running.iter())
                .map(|&(_, bytes)| bytes)
                .fold(0, u64::saturating_add)
        })
    }

    /// It took a seat.
    fn seated(&self) {
        self.with(|line| {
            line.waiting.retain(|&id| id != self.id);
            line.running.push((self.id, 0));
        });
    }

    /// Its process's footprint read `bytes`.
    fn read(&self, bytes: u64) {
        self.with(|line| {
            if let Some(run) = line.running.iter_mut().find(|(id, _)| *id == self.id) {
                run.1 = bytes;
            }
        });
    }
}

impl Drop for Place {
    fn drop(&mut self) {
        let mut lines = LINES.lock();
        if let Some(line) = lines.iter_mut().find(|line| line.guard == self.guard) {
            line.waiting.retain(|&id| id != self.id);
            line.running.retain(|&(id, _)| id != self.id);
        }
        lines.retain(|line| !line.waiting.is_empty() || !line.running.is_empty());
    }
}

/// How a helper ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HelperEnd {
    /// It ran to its end, with this status.
    Ended(ExitStatus),
    /// It was killed when its footprint, `bytes` at that read, was past its memory cap at the
    /// ceiling ([`Limits::memory`]).
    OverMemory {
        /// The footprint read that crossed the cap, in bytes.
        bytes: u64,
    },
    /// It was killed when its footprint, `bytes` at that read, was past its memory cap `cap`,
    /// which the host's room at its seat set below the ceiling ([`seat_cap`]).
    OverRoom {
        /// The footprint read that crossed the cap, in bytes.
        bytes: u64,
        /// The cap, in bytes.
        cap: u64,
    },
    /// It was killed at its time cap.
    OverTime,
    /// It was cancelled: killed, or never started where the cancel came while it waited for a
    /// seat.
    Cancelled,
    /// It never started: its seats, the process, its watch or its talk's thread failed, as said.
    NotStarted(String),
    /// It was killed because its footprint or its status could not be read, as said, so its caps
    /// could not hold.
    Unwatched(String),
}

impl HelperEnd {
    /// Whether the helper died of a fault, with no kill of the library's (the module's doc, "A
    /// helper's own end and its death"): it ended ([`HelperEnd::Ended`]) by a panic of its main
    /// thread, Rust's exit code 101, or by a fault: a fault's signal on Linux and macOS
    /// (`unix::faulted`), an exception's status on Windows (`windows::faulted`). No other end is a
    /// death: an exit with a code of its own, [`SERVED_NONE`] among them, Linux's dynamic
    /// loader's 127 and Windows' `STATUS_DLL_NOT_FOUND` at a start the loader failed, and every
    /// other signal or status, among them those another process ends it with: `SIGKILL` (the OOM
    /// killer's among them), `SIGTERM` (a service's stop), `SIGINT` and `SIGHUP` (a terminal's
    /// Ctrl+C and its close, which reach the helper in the library's process group), Windows'
    /// `STATUS_CONTROL_C_EXIT`, and a seccomp filter's `SIGSYS`, a host's policy. macOS's `dyld`
    /// ends a start it failed with `SIGABRT`, a death; the helper links only the system's
    /// libraries there, and opens the Vulkan loader and its driver with `dlopen`.
    pub(crate) fn died(&self) -> bool {
        match self {
            HelperEnd::Ended(status) => status.code() == Some(PANICKED) || host::faulted(*status),
            _ => false,
        }
    }
}

/// What the helper's thread read of a process's footprint, every [`POLL`] from its start to its
/// end (the module's doc says what each OS counts), and the cap they were held to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Footprints {
    /// The process's memory cap, in bytes ([`seat_cap`]); 0 where it never started.
    pub(crate) cap: u64,
    /// The largest footprint read, in bytes: the one past the cap, for a process killed at it.
    pub(crate) peak: u64,
    /// The largest growth from one read to the next, in bytes.
    pub(crate) most_growth: u64,
    /// How many reads there were.
    pub(crate) reads: u64,
    /// The last read, in bytes.
    last: u64,
}

impl Footprints {
    /// Takes a read of `bytes`.
    fn read(&mut self, bytes: u64) {
        if self.reads > 0 {
            self.most_growth = self.most_growth.max(bytes.saturating_sub(self.last));
        }
        self.peak = self.peak.max(bytes);
        self.last = bytes;
        self.reads = self.reads.saturating_add(1);
    }
}

/// A helper's state, which its thread and its [`Helper`] share.
#[derive(Debug, Default)]
struct State {
    /// Whether it was cancelled.
    cancelled: bool,
    /// Its process's id, once the process has started.
    pid: Option<u32>,
    /// How it ended, once it has. The library takes it in its `on_end`.
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
    /// Starts the helper `run` names on a thread of its own, which takes a seat of the host's
    /// in this process's order of starts, starts and watches the process, and runs `talk` with
    /// the process's stdin and stdout on another thread. Once the process has ended, the talk
    /// has returned and the seat is free, it calls `on_end` with how the process ended, what its
    /// footprint's reads saw (none where it never started) and the talk's result: `None` where no
    /// talk ran, or where it panicked. Returns at once; fails only where the helper's thread
    /// cannot start.
    pub(crate) fn start<T: Send + 'static>(
        run: HelperRun,
        talk: impl FnOnce(ChildStdin, ChildStdout) -> T + Send + 'static,
        on_end: impl FnOnce(&HelperEnd, Footprints, Option<T>) + Send + 'static,
    ) -> io::Result<Helper> {
        let shared = Arc::new(Shared::default());
        let theirs = Arc::clone(&shared);
        // Its place is taken here, on the caller's thread, so the helpers wait in the order they
        // were started, whatever order their threads run in.
        let place = Place::join(&run.seats.guard);
        std::thread::Builder::new()
            .name("reims-vgpu.compile-helper".to_owned())
            .spawn(move || {
                let (end, footprints, said) = watch(run, talk, place, &theirs);
                theirs.state.lock().end = Some(end.clone());
                theirs.changed.notify_all();
                on_end(&end, footprints, said);
            })?;
        Ok(Helper { shared })
    }

    /// The process's id, once it has started.
    #[cfg(test)]
    pub(crate) fn pid(&self) -> Option<u32> {
        self.shared.state.lock().pid
    }

    /// Cancels the helper: its thread kills the process at once, or starts none where it still
    /// waits for a seat. The end is [`HelperEnd::Cancelled`] unless the process ended first.
    pub(crate) fn cancel(&self) {
        self.shared.state.lock().cancelled = true;
        self.shared.changed.notify_all();
    }

    /// Waits for the helper's end.
    #[cfg(test)]
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
    /// Cancels the helper, and waits for nothing: its thread kills the process and frees its seat.
    fn drop(&mut self) {
        self.cancel();
    }
}

/// The helper's thread: takes a seat, runs the process and its talk until both have ended, then
/// frees the seat and leaves its line.
fn watch<T: Send + 'static>(
    run: HelperRun,
    talk: impl FnOnce(ChildStdin, ChildStdout) -> T + Send + 'static,
    place: Place,
    shared: &Shared,
) -> (HelperEnd, Footprints, Option<T>) {
    let clock = OsClock::new();
    let (seat, cap) = match take_seat(&run, &place, shared, &clock) {
        Ok(Some(seated)) => seated,
        Ok(None) => return (HelperEnd::Cancelled, Footprints::default(), None),
        Err(e) => {
            let guard = run.seats.guard.display();
            let end = HelperEnd::NotStarted(format!("the seats of {guard}: {e}"));
            return (end, Footprints::default(), None);
        }
    };
    let ended = run_watched(run, cap, talk, &place, shared, &clock);
    drop(seat);
    drop(place);
    ended
}

/// What a helper took a seat beside: how many seats other helpers held, the room read where it
/// was read, and what this process's running helpers held then.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Beside {
    held: usize,
    room: Option<u64>,
    ours: u64,
}

/// What a try for a seat found.
#[derive(Debug)]
enum Tried {
    /// It holds seat `n`, taken beside what it says.
    Seated(usize, Beside),
    /// The guard or every seat is held.
    Full,
    /// A seat is free, but the room does not admit it beside the others: `Some` where the room
    /// was read, `None` where it was read too lately to read again.
    NoRoom(Option<Beside>),
}

/// Takes a seat of `run`'s for the helper at `place`: once it is the first of this process's
/// helpers that wait, a seat is free and [`admits`] lets it start beside the running helpers,
/// tried every [`POLL`], the room read at most every [`ROOM_POLL`] beside them; with the helper's
/// memory cap ([`seat_cap`]); `None` where it was cancelled first.
fn take_seat(
    run: &HelperRun,
    place: &Place,
    shared: &Shared,
    clock: &OsClock,
) -> io::Result<Option<(File, u64)>> {
    let seats = &run.seats;
    if let Some(dir) = seats.guard.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let guard = open_lock(&seats.guard)?;
    let mut files = (0..seats.count.max(1))
        .map(|n| open_lock(&seats.seat(n)))
        .collect::<io::Result<Vec<File>>>()?;
    let mut room_read: Option<MonoTime> = None;
    let mut said_waiting = false;
    let mut state = shared.state.lock();
    loop {
        if state.cancelled {
            return Ok(None);
        }
        if place.first() {
            let now = clock.now_mono();
            let may_read =
                room_read.is_none_or(|at| now.saturating_duration_since(at) >= ROOM_POLL);
            let ceiling = run.limits.memory;
            let tried = MutexGuard::unlocked(&mut state, || {
                try_seat(seats, ceiling, place, &guard, &files, may_read)
            })?;
            match tried {
                Tried::Seated(n, beside) => {
                    let cap = seat_cap(ceiling, beside);
                    // Alone at the ceiling at debug; beside others, or below the ceiling, at info.
                    let level = if beside.held == 0 && cap == ceiling {
                        log::Level::Debug
                    } else {
                        log::Level::Info
                    };
                    log::log!(
                        level,
                        "{}: its helper takes a seat beside {} of the host's {}: {}",
                        run.label,
                        beside.held,
                        seats.count,
                        room_line(beside, cap)
                    );
                    return Ok(Some((files.swap_remove(n), cap)));
                }
                Tried::NoRoom(Some(beside)) => {
                    room_read = Some(now);
                    // The first wait at info; each read after it, once a ROOM_POLL, at debug.
                    let level = if said_waiting {
                        log::Level::Debug
                    } else {
                        log::Level::Info
                    };
                    said_waiting = true;
                    log::log!(
                        level,
                        "{}: its helper waits for room beside {} of the host's {} seats: {}",
                        run.label,
                        beside.held,
                        seats.count,
                        room_line(beside, ceiling)
                    );
                }
                Tried::NoRoom(None) | Tried::Full => {}
            }
            if state.cancelled {
                return Ok(None);
            }
        }
        shared.wait_a_poll(clock, &mut state);
    }
}

/// The room a helper whose cap is `cap` was admitted or kept waiting by, for its line.
fn room_line(beside: Beside, cap: u64) -> String {
    let mib = |bytes: u64| bytes >> 20;
    let room = match beside.room {
        None => "none said".to_owned(),
        Some(room) => format!("{} MiB", mib(room)),
    };
    format!(
        "the host's room beside the guest's RAM {room}, this process's helpers holding {} MiB, \
         a helper's cap {} MiB",
        mib(beside.ours),
        mib(cap)
    )
}

/// Opens the lock file at `path`, created where it is missing.
fn open_lock(path: &Path) -> io::Result<File> {
    File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}

/// One try for a seat of `seats`, whose lock files are `files`, under their guard, `guard`: the
/// seats held counted, and one taken where one is free and [`admits`] lets a helper whose cap is
/// `cap` start beside them, the room read beside them only where `may_read` says so, and always
/// where none is held, for the cap of a helper alone ([`seat_cap`]).
fn try_seat(
    seats: &Seats,
    cap: u64,
    place: &Place,
    guard: &File,
    files: &[File],
    may_read: bool,
) -> io::Result<Tried> {
    match guard.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => return Ok(Tried::Full),
        Err(TryLockError::Error(e)) => return Err(e),
    }
    let tried = count_and_take(seats, cap, place, files, may_read);
    // A guard that stays locked frees at the files' close, at the end of the wait.
    let _ = guard.unlock();
    tried
}

/// [`try_seat`]'s work under the guard.
fn count_and_take(
    seats: &Seats,
    cap: u64,
    place: &Place,
    files: &[File],
    may_read: bool,
) -> io::Result<Tried> {
    let mut free = Vec::new();
    for (n, file) in files.iter().enumerate() {
        match file.try_lock() {
            Ok(()) => free.push(n),
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Error(e)) => {
                unlock(files, &free);
                return Err(e);
            }
        }
    }
    let held = files.len().saturating_sub(free.len());
    let Some((&seat, others)) = free.split_first() else {
        return Ok(Tried::Full);
    };
    let beside = match held {
        // No seat is held, so none of this process's helpers runs.
        0 => Beside {
            held,
            room: (seats.room)(),
            ours: 0,
        },
        _ if !may_read => {
            unlock(files, &free);
            return Ok(Tried::NoRoom(None));
        }
        _ => Beside {
            held,
            room: (seats.room)(),
            ours: place.ours(),
        },
    };
    if !admits(held, cap, beside.room, beside.ours) {
        unlock(files, &free);
        return Ok(Tried::NoRoom(Some(beside)));
    }
    unlock(files, others);
    place.seated();
    Ok(Tried::Seated(seat, beside))
}

/// Frees the seats `taken` of `files`. A seat that stays locked frees at the files' close.
fn unlock(files: &[File], taken: &[usize]) {
    for &n in taken {
        if let Some(file) = files.get(n) {
            let _ = file.unlock();
        }
    }
}

/// Starts the process and watches it, starts its talk, and polls the process, whose memory cap is
/// `cap`, until it ends or is killed; then waits for the talk's end.
fn run_watched<T: Send + 'static>(
    run: HelperRun,
    cap: u64,
    talk: impl FnOnce(ChildStdin, ChildStdout) -> T + Send + 'static,
    place: &Place,
    shared: &Shared,
    clock: &OsClock,
) -> (HelperEnd, Footprints, Option<T>) {
    let none = Footprints::default();
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
            return (end, none, None);
        }
    };
    let started = clock.now_mono();
    let watched = match host::Watched::new(&child) {
        Ok(watched) => watched,
        Err(e) => {
            end_process(&mut child);
            return (HelperEnd::NotStarted(format!("its watch: {e}")), none, None);
        }
    };
    shared.state.lock().pid = Some(child.id());
    let talking = match start_talk(&mut child, talk) {
        Ok(talking) => talking,
        Err(e) => {
            end_process(&mut child);
            return (HelperEnd::NotStarted(format!("its talk: {e}")), none, None);
        }
    };
    let deadline = Deadline::Mono(started.saturating_add(run.limits.time));
    let caps = Caps {
        memory: cap,
        ceiling: run.limits.memory,
        deadline,
    };
    let (end, footprints) = poll(&mut child, &watched, caps, place, shared, clock);
    // The talk ends once the process has: its pipes read their end, and its writes fail.
    let said = talking.join().ok();
    // `watched` goes before `child`, whose process handle it reads on Windows.
    (end, footprints, said)
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

/// A running helper's caps.
#[derive(Debug, Clone, Copy)]
struct Caps {
    /// Its memory cap, in bytes ([`seat_cap`]).
    memory: u64,
    /// The ceiling, [`Limits::memory`]: a cap below it is the room's.
    ceiling: u64,
    /// Its time cap.
    deadline: Deadline,
}

impl Caps {
    /// The end of a helper whose footprint read `bytes` past its memory cap: at the ceiling, over
    /// memory; below it, over the room.
    fn past_memory(&self, bytes: u64) -> HelperEnd {
        if self.memory < self.ceiling {
            HelperEnd::OverRoom {
                bytes,
                cap: self.memory,
            }
        } else {
            HelperEnd::OverMemory { bytes }
        }
    }
}

/// Polls the process every [`POLL`] until it ends, or kills it at a cancel, past its memory cap,
/// at its time cap, or where its footprint cannot be read; with what its footprint's reads saw,
/// each read also kept at its `place` for the room of the helpers that wait.
fn poll(
    child: &mut Child,
    watched: &host::Watched,
    caps: Caps,
    place: &Place,
    shared: &Shared,
    clock: &OsClock,
) -> (HelperEnd, Footprints) {
    let mut footprints = Footprints {
        cap: caps.memory,
        ..Footprints::default()
    };
    let end = loop {
        if let Some(end) = ended(child) {
            break end;
        }
        let killed = if shared.state.lock().cancelled {
            Some(HelperEnd::Cancelled)
        } else {
            let read = watched.footprint().inspect(|&bytes| {
                footprints.read(bytes);
                place.read(bytes);
            });
            match read {
                Ok(bytes) if bytes > caps.memory => Some(caps.past_memory(bytes)),
                Ok(_) => caps.deadline.reached(clock).then_some(HelperEnd::OverTime),
                // A process that ended since the look above has no footprint to read.
                Err(e) => match ended(child) {
                    Some(end) => break end,
                    None => Some(HelperEnd::Unwatched(format!("its footprint: {e}"))),
                },
            }
        };
        if let Some(end) = killed {
            end_process(child);
            break end;
        }
        let mut state = shared.state.lock();
        if !state.cancelled {
            shared.wait_a_poll(clock, &mut state);
        }
    };
    (end, footprints)
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
/// helper's part, on macOS (the module's doc). The helper's main calls it as its work begins.
pub(crate) fn end_with_parent() -> io::Result<()> {
    host::watch_parent()
}

/// The helper's side: sets the calling process's core-file limit to none, soft and hard, on Linux
/// and macOS, and does nothing on Windows (the module's doc, "A helper's crash writes no core
/// file"). The helper's main calls it first.
pub(crate) fn no_core_files() -> io::Result<()> {
    host::no_core_files()
}
