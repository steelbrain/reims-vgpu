//! The process's main thread, which AppKit runs on: the work posted to it ([`post_main`]), the
//! state only it touches ([`MainOnly`]), and the sends the AppKit side shares ([`responds`],
//! [`shared_app`], [`ns_text`]), each through the message layer (D35).
//!
//! [`post_main`] runs a closure on the main thread, in the order posted, through the main run
//! loop: `CFRunLoopPerformBlock` in the common modes, then `CFRunLoopWakeUp`. So it runs under
//! QEMU's plain `CFRunLoopRun`, under `[NSApp run]`, and while a window is resized live or a menu
//! is open, whose event-tracking mode is a common mode. Nothing here waits for the main thread:
//! QEMU realizes its devices on the main thread, holding its big lock, before it runs its run
//! loop.
//!
//! **The application's start is posted alone.** The start runs `[NSApp run]` inside a run-loop
//! block, and never returns. The run loop runs the blocks queued for it as one batch, taken all
//! at once, so a block queued behind the start in that batch would never run. So until the
//! application has started, the work posted is held here, in order, and the start ([`start`]) is
//! the only block of the window's in the run loop. The application runs the held work once it
//! has launched ([`drain`]); from then on work is posted to the run loop directly, and a post
//! made while the held work runs is held behind it, so the order posted is the order run.
//!
//! Measured on the M3 Max, macOS 26.7 (25G229): starting `[NSApp run]` from a block on the main
//! dispatch queue starves that queue for good, so the blocks posted to it afterwards, and
//! `NSOperationQueue.mainQueue`'s, never ran; started from `CFRunLoopPerformBlock`, the main
//! queue, later run-loop blocks and the main operation queue all ran. Nothing here uses the main
//! queue.

use std::cell::Cell;
use std::collections::VecDeque;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::{NSObjectProtocol, Sel};
use objc2::{MainThreadMarker, Message};
use objc2_app_kit::{NSApp, NSApplication};
use objc2_core_foundation::{CFRunLoop, kCFRunLoopCommonModes};
use objc2_foundation::NSString;
use reims_vgpu_vulkan::core::rt::BlockEnv;
use reims_vgpu_vulkan::core::sync::Mutex;
use reims_vgpu_vulkan::os::msg;

/// Work for the main thread.
type Job = Box<dyn FnOnce(MainThreadMarker) + Send>;

/// Where the application's start is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Nothing started it: work posted is held.
    Idle,
    /// Its start is posted, or running: work posted is held.
    Starting,
    /// It has launched and run the held work: work is posted to the run loop.
    Running,
}

/// The phase, and the work held until the application runs.
struct Queue {
    phase: Phase,
    held: VecDeque<Job>,
}

/// The process's queue of work for the main thread.
static QUEUE: Mutex<Queue> = Mutex::new(Queue {
    phase: Phase::Idle,
    held: VecDeque::new(),
});

/// Runs `f` on the main thread, after the work posted before it, as a callout of the
/// library's: a panic in it aborts before it unwinds into AppKit or the run loop (6.5). It waits
/// for nothing: before the application has launched, `f` is held until it has.
pub(crate) fn post_main(f: impl FnOnce(MainThreadMarker) + Send + 'static) {
    let job: Job = Box::new(move |mtm| {
        let _env = BlockEnv::foreign();
        f(mtm);
    });
    let mut queue = QUEUE.lock();
    if queue.phase == Phase::Running {
        drop(queue);
        perform(job);
    } else {
        queue.held.push_back(job);
    }
}

/// Posts the application's start to the main run loop, alone: the first call does, later ones
/// nothing. `launch` makes the application, and calls [`drain`] once it runs; it returns the
/// application to run when the library made it, which then runs, for good, outside the
/// callout's environment: the host's own work on the main thread runs inside it from then on.
pub(crate) fn start(launch: fn(MainThreadMarker) -> Option<Retained<NSApplication>>) {
    {
        let mut queue = QUEUE.lock();
        if queue.phase != Phase::Idle {
            return;
        }
        queue.phase = Phase::Starting;
    }
    perform(Box::new(move |mtm| {
        let app = {
            let _env = BlockEnv::foreign();
            launch(mtm)
        };
        if let Some(app) = app {
            msg::get("run", || app.run());
        }
    }));
}

/// Runs the work held until the application ran, in the order posted, then lets later work go
/// to the run loop directly.
pub(crate) fn drain(mtm: MainThreadMarker) {
    loop {
        let held = {
            let mut queue = QUEUE.lock();
            if queue.held.is_empty() {
                queue.phase = Phase::Running;
                return;
            }
            std::mem::take(&mut queue.held)
        };
        for job in held {
            job(mtm);
        }
    }
}

/// Queues `job` on the main run loop in its common modes, and wakes it.
///
/// The block is the run loop's until it runs: when the main thread never runs its run loop (a
/// test process, a host that never does), it and what `job` holds are never released.
fn perform(job: Job) {
    let Some(main) = CFRunLoop::main() else {
        return;
    };
    let job = Cell::new(Some(job));
    let block = RcBlock::new(move || {
        // The main run loop runs its blocks on the main thread.
        let Some(mtm) = MainThreadMarker::new() else {
            return;
        };
        if let Some(job) = job.take() {
            job(mtm);
        }
    });
    // SAFETY: `kCFRunLoopCommonModes` is a constant CoreFoundation defines.
    let modes = unsafe { kCFRunLoopCommonModes };
    // SAFETY: the block takes nothing and returns nothing, as the run loop calls it, and the run
    // loop copies it. It runs on the main thread, where the job's `MainThreadMarker` is true.
    unsafe { main.perform_block(modes.map(|m| &**m), Some(&block)) };
    main.wake_up();
}

/// A value in a `static` that only the main thread reaches, through [`MainOnly::get`]. A static
/// is never dropped, so an AppKit object in one is never released as the process exits.
pub(crate) struct MainOnly<T>(T);

// SAFETY: the value is reached only through `get`, whose `MainThreadMarker` proves the caller is
// on the main thread: no other thread ever touches it.
unsafe impl<T> Sync for MainOnly<T> {}

impl<T> MainOnly<T> {
    /// `value`, for the main thread alone.
    pub(crate) const fn new(value: T) -> MainOnly<T> {
        MainOnly(value)
    }

    /// The value, on the main thread.
    pub(crate) fn get(&self, _mtm: MainThreadMarker) -> &T {
        &self.0
    }
}

/// Whether `object` answers `sel`: a method the library's oldest macOS (11) may lack. objc2's
/// checks end a debug build at a send the receiver does not answer.
pub(crate) fn responds<T: NSObjectProtocol + Message>(object: &T, sel: Sel) -> bool {
    msg::get("respondsToSelector:", || object.respondsToSelector(sel))
}

/// The shared application (`NSApp`), made when there is none.
pub(crate) fn shared_app(mtm: MainThreadMarker) -> Retained<NSApplication> {
    msg::get("sharedApplication", || NSApp(mtm))
}

/// `text` as an `NSString`.
pub(crate) fn ns_text(text: &str) -> Retained<NSString> {
    msg::get("initWithBytes:length:encoding:", || {
        NSString::from_str(text)
    })
}
