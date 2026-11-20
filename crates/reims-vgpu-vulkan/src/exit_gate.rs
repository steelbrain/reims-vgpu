//! The exit gate: no GPU work on the library's own threads overlaps the destructors a process
//! runs as it exits.
//!
//! A process exits (`exit`, or the end of `main`) on one thread, which runs the exit handlers
//! and the loaded libraries' destructors in the reverse order of their registration, while every
//! other thread goes on. On Windows the C library's `exit` and the end of `main` run the exit
//! handlers too, before the system ends the other threads, but `std::process::exit` is
//! `ExitProcess` there, which runs none of them: a process that ends through it never runs the
//! hook (QEMU ends through the C library's). A call into what a driver's destructors freed ends the process: on the
//! RTX 5080 Laptop GPU (NVIDIA 615.71.09), the awake thread's touches segfaulted 79 times in a
//! six-minute run of the workspace's tests, in processes whose tests had passed (`crate::awake`),
//! and test processes ended with `SIGSEGV` inside `libnvidia-glvkspirv.so`, the driver's SPIR-V
//! compiler, after their tests passed. The runtime's own exit guard (`rt::process_exiting`) waits
//! for the runtime's calls on its own threads; the library's calls on its own threads (making
//! objects and pipelines, importing pages, encoding, committing, presenting, decoding video) are
//! the library's to keep out of that window. This gate does it:
//! - **Steps.** Each job of a thread of the library's that may call a GPU's driver runs as a step
//!   ([`step`]; [`making`] for a pipeline's or a library's compile): every entry of
//!   `reims-vgpu`'s (its C API's calls, the device's callbacks, the jobs of its lanes, compile
//!   threads and surface workers), a window's frame, a video picture landed or a frame encoded,
//!   each call the video crate's threads make, and the awake thread's touches. Steps are counted.
//!   A step taken once the gate closed is refused, and its thread parks for good ([`stay`]): what
//!   it holds stays held, and the process's end frees it. A step inside a step is part of the
//!   outer one and is never refused. Neither is a step on the thread that ran the hook, the
//!   exiting thread: the exit runs there, one handler and destructor after another, never beside
//!   it.
//! - **The hook.** An exit handler ([`arm`]) closes the gate and waits for the steps in progress
//!   on other threads to end, never for the exiting thread's own (an exit from inside a step): at
//!   most [`BOUND`], or [`MAKING_BOUND`] while a making is among them, as the runtime's guard
//!   waits for its own makings (a cold compile can outlast seconds under load). It says so on
//!   standard error once the wait passes half its bound, and past the bound the exit goes on.
//!   The gate also counts as closed once the runtime's guard closed (`rt::process_exiting`): no
//!   step starts after either hook ran.
//! - **Ahead of the exit.** A host that quiesces the platform before it exits
//!   (`reims-vgpu`'s `GraphicsDevice::quiesce_for_exit`, the C API's `rvg_quiesce_for_exit`)
//!   closes the gate then ([`close_ahead`]), on the thread that then exits, and waits for the
//!   steps in progress until its own deadline. The hook still runs at the exit: it finds no step
//!   in progress where the wait ended in time, and writes the counts. Where the quiesce answers
//!   false, the host ends with `_exit`, which runs no exit handler, so the counts are written then
//!   ([`write_counts_ahead`]). The close logs what it found: at debug, how long the close and the
//!   wait took and how many steps were in progress on other threads, with where the outermost
//!   began; at warn, where the deadline passed first, the steps still in progress and where the
//!   outermost began.
//! - **Where a step began.** Each step records its caller's site (`#[track_caller]`): a thread's
//!   outermost step takes a slot of [`SITE_SLOTS`] as it begins and empties it as it ends, so that
//!   the close can name the steps it waits for. Past [`SITE_SLOTS`] outermost steps at once, the
//!   rest are counted and not named.
//! - **Its order.** The C library runs exit handlers in the reverse order of their registration,
//!   so the hook runs before every destructor registered before it. It is registered as a device
//!   is adopted (`crate::gpu::device`) and as a compile pool starts, after the driver was loaded;
//!   again as the video crate loads a driver (`reims-vgpu`'s video runtime's `arm_exit`); again
//!   as each making ends, after what the driver's compiler registered at its first use of a path,
//!   as the runtime's guard does for its own makings; and again as each thread's count of its
//!   steps ended reaches each power of two, for what a later call registered, at most 64 times a
//!   thread. A run of an
//!   earlier registration finds the hook ran and does nothing. Left open: a destructor
//!   registered by a call still in progress as the exit begins, or by a call after the hook's last
//!   registration, which runs before the hook.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::io::Write as _;
use std::marker::PhantomData;
use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::time::Duration;

use metal2vulkan_runtime as rt;

use crate::core::platform::os::clock::{Clock as _, Deadline, MonoTime};

/// The longest the hook waits for the steps in progress, while none is a making.
pub const BOUND: Duration = Duration::from_secs(2);

/// The longest the hook waits while a making is among the steps in progress: the runtime's guard
/// waits as long for its own makings.
pub const MAKING_BOUND: Duration = Duration::from_secs(60);

/// How often the hook looks at the steps in progress.
const POLL: Duration = Duration::from_millis(1);

/// Whether the gate closed: the hook ran, or a host closed it ahead of the exit ([`close_ahead`]).
static CLOSED: AtomicBool = AtomicBool::new(false);

/// Whether the hook ran.
static HOOK_RAN: AtomicBool = AtomicBool::new(false);

/// The steps in progress, on every thread, those inside others included.
static IN_PROGRESS: AtomicU64 = AtomicU64::new(0);

/// Of them, the makings.
static MAKINGS: AtomicU64 = AtomicU64::new(0);

/// The log target of the close's lines.
const TARGET: &str = "reims_vgpu_vulkan::exit_gate";

/// How many outermost steps in progress at once the gate names by where they began ([`SITES`]).
pub const SITE_SLOTS: usize = 64;

/// One slot of [`SITES`]: the site of the outermost step that took it, or null. On a cache line
/// of its own, so that steps of threads in neighbouring slots write no line in common.
#[repr(align(128))]
struct SiteSlot(AtomicPtr<Location<'static>>);

/// Where each outermost step in progress began (the module docs): a thread's outermost step takes
/// a free slot as it begins, after it was counted in [`IN_PROGRESS`], and empties it before its
/// count ends. Only `Location::caller`'s `'static` sites are stored.
static SITES: [SiteSlot; SITE_SLOTS] =
    [const { SiteSlot(AtomicPtr::new(std::ptr::null_mut())) }; SITE_SLOTS];

thread_local! {
    /// The steps this thread is inside.
    static DEPTH: Cell<u64> = const { Cell::new(0) };
    /// The slot of [`SITES`] this thread's outermost step in progress took, if it took one.
    static SLOT: Cell<Option<usize>> = const { Cell::new(None) };
    /// The slot this thread took last, where its next outermost step looks first.
    static HINT: Cell<usize> = const { Cell::new(0) };
    /// The steps of this thread's that ended, which register the hook again at each power of
    /// two: counted by thread, so no step's end writes what every thread's does.
    static ENDED: Cell<u64> = const { Cell::new(0) };
    /// Whether this thread runs the process's exit: it ran the hook.
    static EXITING_HERE: Cell<bool> = const { Cell::new(false) };
}

/// A step in progress (the module docs): it ends as this drops, on its thread.
#[must_use = "a step ends as it drops"]
#[derive(Debug)]
pub struct Step {
    making: bool,
    /// The slot of [`SITES`] this step took, where it is its thread's outermost and found one.
    slot: Option<usize>,
    _not_send: PhantomData<*const ()>,
}

/// A step of this thread's: `None` once the process's exit closed the gate, unless the step is
/// inside one this thread is in, or this thread runs the exit. Where it began is its caller's
/// site (the module docs).
#[track_caller]
pub fn step() -> Option<Step> {
    enter(false)
}

/// A step that compiles a pipeline or a library: as [`step`], and the hook waits for it as long
/// as [`MAKING_BOUND`], and registers again as it ends.
#[track_caller]
pub fn making() -> Option<Step> {
    enter(true)
}

/// Whether the process's exit closed the gate: the hook ran, a host closed it ahead of the exit
/// ([`close_ahead`]), or the runtime's guard closed.
pub fn closed() -> bool {
    CLOSED.load(Ordering::SeqCst) || rt::process_exiting()
}

/// Parks this thread for good: a step was refused, the process exiting (the module docs).
pub fn stay() -> ! {
    loop {
        std::thread::park();
    }
}

/// Closes the gate ahead of the process's exit (the module docs), on the thread that then exits
/// the process, which counts as the exiting one from here on: its own steps pass. The hook is
/// registered once more first, so that the exit writes the counts. Then waits for the steps in
/// progress on other threads until `until`, a making's included. True once none is in progress;
/// false where one still is at `until`, whose thread may still be in a GPU's driver. A later call
/// closes nothing more and waits again.
///
/// Logs what it found (the module docs): at debug, the close's time (the hook's registration
/// included), the steps in progress on other threads then, how many were makings and where the
/// outermost began, and the wait's time; at warn, where `until` came first, the steps still in
/// progress and where the outermost began.
pub fn close_ahead(until: MonoTime) -> bool {
    let clock = crate::os::OsClock::new();
    let began = clock.now_mono();
    arm();
    let _ = EXITING_HERE.try_with(|e| e.set(true));
    CLOSED.store(true, Ordering::SeqCst);
    let closed = clock.now_mono();
    let at_close = others_in_progress();
    let makings = MAKINGS.load(Ordering::SeqCst);
    let sites = if at_close > 0 {
        begun_at(&sites_of_others())
    } else {
        String::new()
    };
    let mut now = closed;
    while others_in_progress() > 0 {
        if now >= until {
            log::warn!(
                target: TARGET,
                "the gate closed ahead of the exit waited {} us, to its deadline, with {} of the \
                 library's steps still in progress on other threads, steps inside steps counted \
                 ({} making){}",
                now.saturating_duration_since(closed).as_micros(),
                others_in_progress(),
                MAKINGS.load(Ordering::SeqCst),
                begun_at(&sites_of_others())
            );
            return false;
        }
        clock.sleep_until(Deadline::Mono(now.saturating_add(POLL).min(until)));
        now = clock.now_mono();
    }
    log::debug!(
        target: TARGET,
        "the gate closed ahead of the exit in {} us, the hook's registration included, with \
         {at_close} of the library's steps in progress on other threads, steps inside steps \
         counted ({makings} making){sites}; they ended {} us after the close",
        closed.saturating_duration_since(began).as_micros(),
        now.saturating_duration_since(closed).as_micros()
    );
    true
}

/// `sites` ([`sites_of_others`]) as the close's lines end with them: `, the outermost begun at
/// <sites>`, or nothing where there are none.
fn begun_at(sites: &str) -> String {
    if sites.is_empty() {
        String::new()
    } else {
        format!(", the outermost begun at {sites}")
    }
}

/// Where the outermost steps in progress on other threads began (the module docs), each site once
/// with how many began there, `file:line:column (n)`, in order; empty where none took a slot.
fn sites_of_others() -> String {
    let own = SLOT.try_with(Cell::get).ok().flatten();
    let mut at: BTreeMap<(&'static str, u32, u32), u64> = BTreeMap::new();
    for (i, slot) in SITES.iter().enumerate() {
        let site = slot.0.load(Ordering::SeqCst);
        if site.is_null() || own == Some(i) {
            continue;
        }
        // SAFETY: a slot holds only `Location::caller`'s sites, which live for the process.
        let site: &'static Location<'static> = unsafe { &*site };
        let n = at
            .entry((site.file(), site.line(), site.column()))
            .or_insert(0);
        *n = n.saturating_add(1);
    }
    at.iter()
        .map(|((file, line, column), n)| format!("{file}:{line}:{column} ({n})"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Registers the hook (the module docs), after every exit handler registered so far. Once the
/// gate closed it registers nothing.
pub fn arm() {
    if CLOSED.load(Ordering::SeqCst) {
        return;
    }
    // SAFETY: `at_exit` is a plain function that never unwinds.
    if unsafe { atexit(at_exit) } != 0 {
        log::warn!(
            target: TARGET,
            "atexit refused: the library's GPU work may run beside the driver's exit"
        );
    }
}

#[track_caller]
fn enter(making: bool) -> Option<Step> {
    let depth = DEPTH.with(Cell::get);
    // Counted before the gate is read, and the hook closes it before it reads the count: either
    // the hook sees this step, or this step sees the gate closed.
    IN_PROGRESS.fetch_add(1, Ordering::SeqCst);
    if depth == 0 && closed() && !EXITING_HERE.with(Cell::get) {
        IN_PROGRESS.fetch_sub(1, Ordering::SeqCst);
        return None;
    }
    if making {
        MAKINGS.fetch_add(1, Ordering::SeqCst);
    }
    DEPTH.with(|d| d.set(depth + 1));
    // Read here, not in a closure: a closure's `Location::caller` is the closure's own site.
    let site = Location::caller();
    let slot = if depth == 0 { take_slot(site) } else { None };
    Some(Step {
        making,
        slot,
        _not_send: PhantomData,
    })
}

/// Takes a free slot of [`SITES`] for `site`, this thread's outermost step's, looking first where
/// this thread's last one was: the slot, or `None` where every slot is taken.
fn take_slot(site: &'static Location<'static>) -> Option<usize> {
    let site = std::ptr::from_ref(site).cast_mut();
    let first = HINT.with(Cell::get).min(SITE_SLOTS - 1);
    let taken = (first..SITE_SLOTS).chain(0..first).find(|&i| {
        SITES.get(i).is_some_and(|slot| {
            slot.0.load(Ordering::Relaxed).is_null()
                && (slot.0)
                    .compare_exchange(
                        std::ptr::null_mut(),
                        site,
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                    )
                    .is_ok()
        })
    })?;
    HINT.with(|h| h.set(taken));
    SLOT.with(|s| s.set(Some(taken)));
    Some(taken)
}

impl Drop for Step {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        if let Some(slot) = self.slot.and_then(|i| SITES.get(i)) {
            slot.0.store(std::ptr::null_mut(), Ordering::SeqCst);
            let _ = SLOT.try_with(|s| s.set(None));
        }
        if self.making {
            MAKINGS.fetch_sub(1, Ordering::SeqCst);
        }
        IN_PROGRESS.fetch_sub(1, Ordering::SeqCst);
        let ended = ENDED.with(|e| {
            let ended = e.get().saturating_add(1);
            e.set(ended);
            ended
        });
        if self.making || ended.is_power_of_two() {
            arm();
        }
    }
}

unsafe extern "C" {
    /// The C library's `atexit`.
    fn atexit(f: extern "C" fn()) -> i32;
}

/// Writes the process's degradation counts and the log's count of its repeated lines now, as the
/// hook would at the exit (`crate::answer::write_counts`), for a process that then ends with
/// `_exit`, which runs no exit handler: what the quiesce ahead of the exit does where it answers
/// false (`reims-vgpu`'s `GraphicsDevice::quiesce_for_exit`), after it closed the gate
/// ([`close_ahead`]). The hook then finds it ran, and so does a later call.
pub fn write_counts_ahead() {
    if HOOK_RAN.swap(true, Ordering::SeqCst) {
        return;
    }
    crate::answer::write_counts();
}

/// The hook (the module docs): closes the gate, then waits for the steps in progress on other
/// threads, then writes the process's degradation counts and the log's count of its repeated
/// lines (`crate::answer::write_counts`). A later run finds it ran and returns; a gate closed
/// ahead of the exit ([`close_ahead`]) leaves it its wait, which finds no step where that one
/// ended in time, and its counts.
extern "C" fn at_exit() {
    if HOOK_RAN.swap(true, Ordering::SeqCst) {
        return;
    }
    let _ = EXITING_HERE.try_with(|e| e.set(true));
    CLOSED.store(true, Ordering::SeqCst);
    wait_for_others();
    crate::answer::write_counts();
}

/// The hook's wait for the steps in progress on threads other than the exiting one (the module
/// docs).
fn wait_for_others() {
    let others = others_in_progress;
    if others() == 0 {
        return;
    }
    let clock = crate::os::OsClock::new();
    let start = clock.now_mono();
    let mut said = false;
    while others() > 0 {
        let bound = if MAKINGS.load(Ordering::SeqCst) > 0 {
            MAKING_BOUND
        } else {
            BOUND
        };
        let waited = clock.now_mono().saturating_duration_since(start);
        if waited >= bound {
            let _ = writeln!(
                std::io::stderr(),
                "reims-vgpu: the exit goes on with {} of the library's GPU steps still in progress \
                 after {} ms",
                others(),
                waited.as_millis()
            );
            return;
        }
        if !said && waited >= bound / 2 {
            said = true;
            let _ = writeln!(
                std::io::stderr(),
                "reims-vgpu: the exit waits for {} of the library's GPU steps in progress ({} \
                 making), {} ms at most",
                others(),
                MAKINGS.load(Ordering::SeqCst),
                bound.as_millis()
            );
        }
        clock.sleep_until(Deadline::Mono(clock.now_mono().saturating_add(POLL)));
    }
}

/// The steps in progress on threads other than this one.
fn others_in_progress() -> u64 {
    let own = DEPTH.try_with(Cell::get).unwrap_or(0);
    IN_PROGRESS.load(Ordering::SeqCst).saturating_sub(own)
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use super::*;

    /// The test binary run again for `test` alone, with `RVG_EXIT_CHILD` naming it: its output.
    fn child(test: &str) -> std::process::Output {
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture", "--test-threads=1"])
            .env("RVG_EXIT_CHILD", test)
            .output()
            .unwrap()
    }

    /// Whether this process is the child of `test`.
    fn is_child(test: &str) -> bool {
        std::env::var("RVG_EXIT_CHILD").as_deref() == Ok(test)
    }

    unsafe extern "C" {
        /// The C library's `exit`, which runs the exit handlers, as a host's exit does: what
        /// `std::process::exit` calls on Unix, where on Windows it calls `ExitProcess`, which runs
        /// none (the module docs).
        fn exit(status: i32) -> !;
    }

    /// Writes `line` to standard error at once.
    fn say(line: &str) {
        let _ = writeln!(std::io::stderr(), "{line}");
    }

    /// The exit waits for a step in progress on another thread, and a step taken after it began
    /// is refused, its thread parked: the child exits while one thread is 300 ms into a step and
    /// another asks for one once the gate closed
    #[test]
    fn an_exit_waits_for_a_step_in_progress_and_refuses_the_next() {
        const TEST: &str =
            "exit_gate::tests::an_exit_waits_for_a_step_in_progress_and_refuses_the_next";
        if is_child(TEST) {
            arm();
            let (began, at_begin) = std::sync::mpsc::channel();
            std::thread::Builder::new()
                .name("in a step".into())
                .spawn(move || {
                    let held = step().unwrap();
                    began.send(()).unwrap();
                    std::thread::sleep(Duration::from_millis(300));
                    say("the step in progress ended");
                    drop(held);
                })
                .unwrap();
            std::thread::Builder::new()
                .name("after the close".into())
                .spawn(|| {
                    while !closed() {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    match step() {
                        Some(_) => say("a step began after the close"),
                        None => {
                            say("the step after the close was refused");
                            stay();
                        }
                    }
                })
                .unwrap();
            at_begin.recv().unwrap();
            // SAFETY: the C library's exit, which never returns.
            unsafe { exit(0) }
        }
        let out = child(TEST);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(0), "{out:?}");
        assert!(stderr.contains("the step in progress ended"), "{stderr}");
        assert!(
            stderr.contains("the step after the close was refused"),
            "{stderr}"
        );
        assert!(!stderr.contains("a step began after the close"), "{stderr}");
    }

    /// The exit does not wait for the exiting thread's own step, and the steps of the thread
    /// that ran the hook pass once the gate closed: a child exits from inside a step, and another
    /// takes a step in an exit handler registered before the hook, so run after it
    #[test]
    fn the_exiting_threads_steps_pass_and_are_not_waited_for() {
        const TEST: &str =
            "exit_gate::tests::the_exiting_threads_steps_pass_and_are_not_waited_for";
        static EXITED_AT: AtomicU64 = AtomicU64::new(0);
        extern "C" fn after_the_hook() {
            let clock = crate::os::OsClock::new();
            let waited = clock
                .now_mono()
                .as_nanos()
                .saturating_sub(EXITED_AT.load(Ordering::SeqCst));
            say(&format!("the hook took {} ms", waited / 1_000_000));
            say(&format!("closed {}", closed()));
            match step() {
                Some(_) => say("a step on the exiting thread passed"),
                None => say("a step on the exiting thread was refused"),
            }
        }
        if is_child(TEST) {
            // SAFETY: a plain function that never unwinds.
            assert_eq!(unsafe { atexit(after_the_hook) }, 0);
            arm();
            let inside = std::env::var_os("RVG_EXIT_INSIDE_A_STEP").is_some();
            let outer = inside.then(|| step().unwrap());
            let clock = crate::os::OsClock::new();
            EXITED_AT.store(clock.now_mono().as_nanos(), Ordering::SeqCst);
            // The exit runs no destructor: the step stays in progress.
            let _kept = outer;
            // SAFETY: the C library's exit, which never returns.
            unsafe { exit(0) }
        }
        for inside in [true, false] {
            let mut command = Command::new(std::env::current_exe().unwrap());
            command
                .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
                .env("RVG_EXIT_CHILD", TEST);
            if inside {
                command.env("RVG_EXIT_INSIDE_A_STEP", "1");
            }
            let out = command.output().unwrap();
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert_eq!(out.status.code(), Some(0), "{out:?}");
            assert!(stderr.contains("closed true"), "{stderr}");
            assert!(
                stderr.contains("a step on the exiting thread passed"),
                "{stderr}"
            );
            let took = stderr
                .lines()
                .find_map(|l| l.strip_prefix("the hook took "))
                .and_then(|l| l.strip_suffix(" ms"))
                .and_then(|ms| ms.parse::<u128>().ok())
                .unwrap_or_else(|| panic!("no hook time: {stderr}"));
            assert!(
                took < BOUND.as_millis() / 2,
                "inside a step {inside}: {stderr}"
            );
        }
    }

    /// A gate closed ahead of the exit waits for a step in progress on another thread until its
    /// deadline, answering false where the step outlasts it and true once it ended, and logs
    /// each wait with where the step began, the first at warn and the second at debug; it refuses
    /// a later step of another thread, passes the closing thread's own, and leaves the hook its
    /// counts at the exit: the child closes it while one thread is 300 ms into a step, first with
    /// a 50 ms deadline, then with a 5 s one, then exits
    #[test]
    fn a_gate_closed_ahead_waits_until_its_deadline_and_the_exit_still_counts() {
        const TEST: &str = "exit_gate::tests::\
                            a_gate_closed_ahead_waits_until_its_deadline_and_the_exit_still_counts";
        const CODE: &str = "exit-gate-test-closed-ahead";
        if is_child(TEST) {
            arm();
            crate::answer::count_kind(CODE);
            let (began, at_begin) = std::sync::mpsc::channel();
            std::thread::Builder::new()
                .name("in a step".into())
                .spawn(move || {
                    let (held, here) = (step().unwrap(), Location::caller());
                    began.send(here).unwrap();
                    std::thread::sleep(Duration::from_millis(300));
                    say("the step in progress ended");
                    drop(held);
                })
                .unwrap();
            let here = at_begin.recv().unwrap();
            say(&format!(
                "the step began at {}:{}:",
                here.file(),
                here.line()
            ));
            let logs = crate::core::testkit::logs::capture();
            let clock = crate::os::OsClock::new();
            let short = close_ahead(clock.now_mono().saturating_add(Duration::from_millis(50)));
            say(&format!("a 50 ms deadline answered {short}"));
            let long = close_ahead(clock.now_mono().saturating_add(Duration::from_secs(5)));
            say(&format!("a 5 s deadline answered {long}"));
            for (level, target, line) in logs.take() {
                say(&format!("logged {level} {target}: {line}"));
            }
            drop(logs);
            say(&format!("closed {}", closed()));
            match step() {
                Some(_) => say("the closing thread's step passed"),
                None => say("the closing thread's step was refused"),
            }
            let (refused, at_refusal) = std::sync::mpsc::channel();
            std::thread::Builder::new()
                .name("after the close".into())
                .spawn(move || match step() {
                    Some(_) => {
                        say("a step began after the close");
                        refused.send(()).unwrap();
                    }
                    None => {
                        say("the step after the close was refused");
                        refused.send(()).unwrap();
                        stay();
                    }
                })
                .unwrap();
            at_refusal.recv().unwrap();
            // SAFETY: the C library's exit, which never returns.
            unsafe { exit(0) }
        }
        let out = child(TEST);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(0), "{out:?}");
        let at = |line: &str| {
            stderr
                .find(line)
                .unwrap_or_else(|| panic!("no {line:?}: {stderr}"))
        };
        assert!(at("a 50 ms deadline answered false") < at("the step in progress ended"));
        assert!(at("the step in progress ended") < at("a 5 s deadline answered true"));
        let site = stderr
            .lines()
            .find_map(|l| l.strip_prefix("the step began at "))
            .unwrap_or_else(|| panic!("no site: {stderr}"));
        let logged = |level: &str, words: &str| {
            stderr.lines().any(|l| {
                l.starts_with(&format!("logged {level} {TARGET}: "))
                    && l.contains(words)
                    && l.contains(&format!("the outermost begun at {site}"))
            })
        };
        assert!(
            logged("WARN", "to its deadline, with 1 of the library's steps"),
            "{stderr}"
        );
        assert!(
            logged("DEBUG", "with 1 of the library's steps in progress"),
            "{stderr}"
        );
        at("closed true");
        at("the closing thread's step passed");
        at("the step after the close was refused");
        let counts = stderr
            .lines()
            .find(|l| l.starts_with("reims-vgpu: degradation counts:"));
        assert!(
            counts.is_some_and(|l| l.contains(&format!("{CODE} 1"))),
            "{stderr}"
        );
    }

    /// The sites of the steps in progress on other threads name where each other thread's
    /// outermost step began, once with how many began there, and neither the steps inside it nor
    /// this thread's own; a step that ended is not named
    #[test]
    fn the_sites_name_where_other_threads_outermost_steps_began() {
        let (own, own_here) = (step().unwrap(), Location::caller());
        let (began, at_begin) = std::sync::mpsc::channel();
        let (end, at_end) = std::sync::mpsc::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("in a step".into())
            .spawn(move || {
                let (outer, here) = (step().unwrap(), Location::caller());
                let inner = making().unwrap();
                began.send(here).unwrap();
                at_end.recv().unwrap();
                drop(inner);
                drop(outer);
            })
            .unwrap();
        let here = at_begin.recv().unwrap();
        let at = |l: &Location<'_>| format!("{}:{}:", l.file(), l.line());
        let sites = sites_of_others();
        end.send(()).unwrap();
        thread.join().unwrap();
        let named: Vec<&str> = (sites.split(", "))
            .filter(|s| s.starts_with(&at(here)))
            .collect();
        assert!(named.len() == 1 && named[0].ends_with(" (1)"), "{sites}");
        assert!(!sites.contains(&at(own_here)), "{sites}");
        assert!(!sites_of_others().contains(&at(here)));
        drop(own);
    }

    /// A step inside a step is part of it: both count until they end, and a making counts among
    /// the makings
    #[test]
    fn a_step_inside_a_step_counts_with_it() {
        let depth = DEPTH.with(Cell::get);
        let outer = step().unwrap();
        let inner = making().unwrap();
        assert_eq!(DEPTH.with(Cell::get), depth + 2);
        assert!(IN_PROGRESS.load(Ordering::SeqCst) >= 2);
        assert!(MAKINGS.load(Ordering::SeqCst) >= 1);
        drop(inner);
        drop(outer);
        assert_eq!(DEPTH.with(Cell::get), depth);
    }
}
