//! The exit gate: no GPU work on the library's own threads overlaps the destructors a process
//! runs as it exits.
//!
//! A process exits (`exit`, or the end of `main`) on one thread, which runs the exit handlers
//! and the loaded libraries' destructors in the reverse order of their registration, while every
//! other thread goes on. A call into what a driver's destructors freed ends the process: on the
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
//! - **Its order.** The C library runs exit handlers in the reverse order of their registration,
//!   so the hook runs before every destructor registered before it. It is registered as a device
//!   is adopted (`crate::gpu::device`) and as a compile pool starts, after the driver was loaded;
//!   again as the video crate loads a driver (`reims-vgpu`'s video runtime's `arm_exit`); again
//!   as each making ends, after what the driver's compiler registered at its first use of a path,
//!   as the runtime's guard does for its own makings; and again as each thread's count of its
//!   steps ended reaches each power of two, for what a later call registered, at most 64 times a
//!   thread. A run of an
//!   earlier registration finds the gate closed and does nothing. Left open: a destructor
//!   registered by a call still in progress as the exit begins, or by a call after the hook's last
//!   registration, which runs before the hook.

use std::cell::Cell;
use std::io::Write as _;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use metal2vulkan_runtime as rt;

use crate::core::platform::os::clock::{Clock as _, Deadline};

/// The longest the hook waits for the steps in progress, while none is a making.
pub const BOUND: Duration = Duration::from_secs(2);

/// The longest the hook waits while a making is among the steps in progress: the runtime's guard
/// waits as long for its own makings.
pub const MAKING_BOUND: Duration = Duration::from_secs(60);

/// How often the hook looks at the steps in progress.
const POLL: Duration = Duration::from_millis(1);

/// Whether the hook ran.
static CLOSED: AtomicBool = AtomicBool::new(false);

/// The steps in progress, on every thread, those inside others included.
static IN_PROGRESS: AtomicU64 = AtomicU64::new(0);

/// Of them, the makings.
static MAKINGS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The steps this thread is inside.
    static DEPTH: Cell<u64> = const { Cell::new(0) };
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
    _not_send: PhantomData<*const ()>,
}

/// A step of this thread's: `None` once the process's exit closed the gate, unless the step is
/// inside one this thread is in, or this thread runs the exit.
pub fn step() -> Option<Step> {
    enter(false)
}

/// A step that compiles a pipeline or a library: as [`step`], and the hook waits for it as long
/// as [`MAKING_BOUND`], and registers again as it ends.
pub fn making() -> Option<Step> {
    enter(true)
}

/// Whether the process's exit closed the gate: the hook ran, or the runtime's guard closed.
pub fn closed() -> bool {
    CLOSED.load(Ordering::SeqCst) || rt::process_exiting()
}

/// Parks this thread for good: a step was refused, the process exiting (the module docs).
pub fn stay() -> ! {
    loop {
        std::thread::park();
    }
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
            target: "reims_vgpu_vulkan::exit_gate",
            "atexit refused: the library's GPU work may run beside the driver's exit"
        );
    }
}

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
    Some(Step {
        making,
        _not_send: PhantomData,
    })
}

impl Drop for Step {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
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

/// The hook (the module docs): closes the gate, then waits for the steps in progress on other
/// threads. A later run finds it closed and returns.
extern "C" fn at_exit() {
    if CLOSED.load(Ordering::SeqCst) {
        return;
    }
    let _ = EXITING_HERE.try_with(|e| e.set(true));
    CLOSED.store(true, Ordering::SeqCst);
    let own = DEPTH.try_with(Cell::get).unwrap_or(0);
    let others = || IN_PROGRESS.load(Ordering::SeqCst).saturating_sub(own);
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
            std::process::exit(0);
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
            std::process::exit(0);
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
