//! `Threads`: the threads core runs on, as plain pthreads.
//!
//! **Why pthreads and not `std::thread`.** `std::thread` wraps its body in `catch_unwind`, which
//! Rust's personality reports as a handler to an Objective-C exception's search phase: an exception
//! no call catches (every send and host call catches its own, so one out of a release's `dealloc`)
//! would then unwind every Rust frame of the thread, dropping guards and values, before Rust aborts.
//! A pthread whose start routine is `extern "C-unwind"` has no handler at its base, so such an
//! exception ends the process in its search phase with no drops (D35). Measured with objc2 0.6.4
//! on macOS 26.7: an uncaught raise on a `std::thread` dropped the thread's frames, then Rust
//! aborted; on a pthread with an `extern "C-unwind"` start, or a GCD worker, it ended the process
//! as an uncaught Objective-C exception, with no drops.

use std::ffi::{CString, c_void};
use std::mem::MaybeUninit;
use std::sync::Arc;

use reims_vgpu_core::platform::os::{Threads, TimerClock};
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal};

use super::clock::MacClock;

unsafe extern "C" {
    /// `pthread_create`, with a start routine that may unwind (a foreign exception's search
    /// phase passes it and finds no handler).
    #[link_name = "pthread_create"]
    fn pthread_create_unwinding(
        thread: *mut libc::pthread_t,
        attr: *const libc::pthread_attr_t,
        start: unsafe extern "C-unwind" fn(*mut c_void) -> *mut c_void,
        arg: *mut c_void,
    ) -> libc::c_int;
}

/// What a new thread starts with.
struct Start {
    name: &'static str,
    body: Box<dyn FnOnce() + Send>,
}

/// The start routine: names the thread and runs its body. No frame of it catches.
unsafe extern "C-unwind" fn start(arg: *mut c_void) -> *mut c_void {
    // SAFETY: `arg` is the `Box<Start>` `spawn_pthread` leaked, taken back once, here.
    let Start { name, body } = *unsafe { Box::from_raw(arg.cast::<Start>()) };
    if let Ok(name) = CString::new(name) {
        // SAFETY: a NUL-terminated name; macOS names only the calling thread.
        unsafe { libc::pthread_setname_np(name.as_ptr()) };
    }
    body();
    std::ptr::null_mut()
}

/// `stack_size` rounded up to whole pages, as `pthread_attr_setstacksize` requires.
fn page_rounded(stack_size: usize) -> usize {
    // SAFETY: `vm_page_size` is a libsystem_kernel global, set before `main`.
    let page = unsafe { libc::vm_page_size };
    stack_size.div_ceil(page).saturating_mul(page)
}

/// A detached pthread named `name` with `stack_size` bytes of stack (rounded up to pages) that
/// runs `body`. A refused thread ends the process.
fn spawn_pthread(name: &'static str, stack_size: usize, body: Box<dyn FnOnce() + Send>) {
    let arg = Box::into_raw(Box::new(Start { name, body }));
    let mut attr = MaybeUninit::<libc::pthread_attr_t>::uninit();
    let mut thread = MaybeUninit::<libc::pthread_t>::uninit();
    // SAFETY: the attribute is initialised before use and destroyed after; the start routine
    // takes `arg` back exactly once, and only if the thread was created.
    let rc = unsafe {
        libc::pthread_attr_init(attr.as_mut_ptr());
        libc::pthread_attr_setstacksize(attr.as_mut_ptr(), page_rounded(stack_size));
        libc::pthread_attr_setdetachstate(attr.as_mut_ptr(), libc::PTHREAD_CREATE_DETACHED);
        let rc = pthread_create_unwinding(thread.as_mut_ptr(), attr.as_ptr(), start, arg.cast());
        libc::pthread_attr_destroy(attr.as_mut_ptr());
        rc
    };
    if rc != 0 {
        // SAFETY: no thread took it.
        drop(unsafe { Box::from_raw(arg) });
        fatal(DeathId::Platform {
            kind: DeathKind::Abort,
            call: "pthread_create",
        })
    }
}

/// The process's threads, and its clock for timers.
#[derive(Debug, Default)]
pub struct MacThreads;

impl MacThreads {
    /// The process's Threads.
    pub fn new() -> MacThreads {
        MacThreads
    }
}

impl Threads for MacThreads {
    fn spawn(&self, name: &'static str, stack_size: usize, body: Box<dyn FnOnce() + Send>) {
        spawn_pthread(name, stack_size, body)
    }

    /// A worker named after its queue, with a GCD worker's 512 KiB of stack.
    fn spawn_worker(&self, label: &'static str, body: Box<dyn FnOnce() + Send>) {
        spawn_pthread(label, reims_vgpu_core::rt::queue::WORKER_STACK, body)
    }

    fn handle(&self) -> Arc<dyn Threads> {
        Arc::new(MacThreads)
    }

    fn timer_clock(&self) -> Option<Arc<dyn TimerClock>> {
        Some(Arc::new(MacClock::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    const WAIT: Duration = Duration::from_secs(10);

    /// The running thread's pthread name.
    fn current_name() -> String {
        let mut buf = [0 as libc::c_char; 64];
        // SAFETY: a live buffer of the length given.
        let rc =
            unsafe { libc::pthread_getname_np(libc::pthread_self(), buf.as_mut_ptr(), buf.len()) };
        assert_eq!(rc, 0);
        // SAFETY: pthread_getname_np NUL-terminates within the buffer.
        unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    /// A spawned thread is named, has the stack asked for (rounded up to pages), and is detached
    /// (joining it is refused)
    #[test]
    fn a_thread_is_named_sized_and_detached() {
        // SAFETY: `vm_page_size` is a libsystem_kernel global, set before `main`.
        let page = unsafe { libc::vm_page_size };
        assert_eq!(page_rounded(512 * 1024), 512 * 1024);
        assert_eq!(page_rounded(512 * 1024 + 1), 512 * 1024 + page);
        let (tx, rx) = mpsc::channel();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        MacThreads::new().spawn(
            "reims-vgpu.test",
            512 * 1024 + 1,
            Box::new(move || {
                // SAFETY: queries of the running thread.
                let (me, size) = unsafe {
                    (
                        libc::pthread_self() as usize,
                        libc::pthread_get_stacksize_np(libc::pthread_self()),
                    )
                };
                tx.send((current_name(), me, size)).unwrap();
                let _ = go_rx.recv_timeout(WAIT);
            }),
        );
        let (name, me, size) = rx.recv_timeout(WAIT).unwrap();
        assert_eq!(name, "reims-vgpu.test");
        assert!(size >= 512 * 1024 + page, "{size}");
        // SAFETY: the thread is alive (it waits on `go`); a detached thread refuses the join
        // with EINVAL instead of blocking.
        let rc = unsafe { libc::pthread_join(me as libc::pthread_t, std::ptr::null_mut()) };
        assert_eq!(rc, libc::EINVAL);
        go_tx.send(()).unwrap();
    }

    /// A queue's worker is named after the queue
    #[test]
    fn a_worker_is_named_after_its_queue() {
        let (tx, rx) = mpsc::channel();
        MacThreads::new().spawn_worker(
            "reims-vgpu.test-queue",
            Box::new(move || tx.send(current_name()).unwrap()),
        );
        assert_eq!(rx.recv_timeout(WAIT).unwrap(), "reims-vgpu.test-queue");
    }

    /// An exception out of a block of core's, which has no frame that catches: the search phase
    /// finds no handler, so nothing unwinds (a capture's drop never
    /// runs) and the process dies through the host's uncaught-exception handler. On a named
    /// thread, a queue's worker and a timer's handler.
    mod a_raising_block_dies_uncaught {
        use super::*;
        use reims_vgpu_core::rt::{SerialQueue, Timer};

        /// Prints `DROPPED` when dropped: a block's capture.
        struct Tracer;
        impl Drop for Tracer {
            fn drop(&mut self) {
                eprintln!("DROPPED");
            }
        }

        unsafe extern "C" {
            /// Foundation's.
            fn NSSetUncaughtExceptionHandler(
                handler: Option<unsafe extern "C" fn(*mut objc2::runtime::AnyObject)>,
            );
            #[link_name = "_exit"]
            fn libc_exit(status: std::ffi::c_int) -> !;
        }

        /// The host's uncaught-exception handler: reports and ends the process with status 42.
        unsafe extern "C" fn uncaught(_exception: *mut objc2::runtime::AnyObject) {
            eprintln!("UNCAUGHT");
            // SAFETY: ends the process at once.
            unsafe { libc_exit(42) }
        }

        /// The host's code: raises an NSException.
        fn host_raises() -> ! {
            let name = objc2_foundation::NSString::from_str("HostRaise");
            let reason = objc2_foundation::NSString::from_str("the host raised");
            // SAFETY: a name and a reason, no userInfo.
            let e = unsafe {
                objc2_foundation::NSException::exceptionWithName_reason_userInfo(
                    &name,
                    Some(&reason),
                    None,
                )
            };
            // SAFETY: an NSException is an Objective-C object, which `Exception` wraps.
            objc2::exception::throw(unsafe {
                objc2::rc::Retained::cast_unchecked::<objc2::exception::Exception>(e)
            })
        }

        /// Runs `which` in a child process of this test binary and checks its death.
        fn dies_uncaught(which: &str) {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    &format!("os::macos::threads::tests::a_raising_block_dies_uncaught::{which}"),
                    "--nocapture",
                    "--test-threads=1",
                ])
                .output()
                .unwrap();
            let err = String::from_utf8_lossy(&out.stderr);
            assert_eq!(out.status.code(), Some(42), "{out:?}");
            assert!(err.contains("UNCAUGHT"), "{err}");
            assert!(!err.contains("DROPPED"), "{err}");
            assert!(!err.contains("SURVIVED"), "{err}");
        }

        #[test]
        #[ignore = "run by named_thread in a child process"]
        fn child_thread() {
            // SAFETY: a handler of the right signature, for the process's life.
            unsafe { NSSetUncaughtExceptionHandler(Some(uncaught)) };
            let tracer = Tracer;
            MacThreads::new().spawn(
                "reims-vgpu.test",
                512 * 1024,
                Box::new(move || {
                    let _held = &tracer;
                    host_raises()
                }),
            );
            std::thread::sleep(WAIT);
            eprintln!("SURVIVED");
        }

        #[test]
        #[ignore = "run by async_block in a child process"]
        fn child_async() {
            // SAFETY: as above.
            unsafe { NSSetUncaughtExceptionHandler(Some(uncaught)) };
            let q = SerialQueue::new(&MacThreads::new(), "dev.reims.vgpu.t.raise");
            let tracer = Tracer;
            q.exec_async(move || {
                let _held = &tracer;
                host_raises()
            });
            std::thread::sleep(WAIT);
            eprintln!("SURVIVED");
        }

        #[test]
        #[ignore = "run by timer_handler in a child process"]
        fn child_timer() {
            // SAFETY: as above.
            unsafe { NSSetUncaughtExceptionHandler(Some(uncaught)) };
            let q = SerialQueue::new(&MacThreads::new(), "dev.reims.vgpu.t.raise");
            let tracer = Tracer;
            use reims_vgpu_core::platform::os::{Clock as _, Deadline};
            let _t = Timer::start(
                &q,
                Deadline::Wall(MacClock::new().now_wall()),
                Duration::from_millis(20),
                Duration::ZERO,
                move |_| {
                    let _held = &tracer;
                    host_raises()
                },
            );
            std::thread::sleep(WAIT);
            eprintln!("SURVIVED");
        }

        /// A named thread's body.
        #[test]
        fn named_thread() {
            dies_uncaught("child_thread");
        }

        /// An async block, on the queue's worker.
        #[test]
        fn async_block() {
            dies_uncaught("child_async");
        }

        /// A timer's handler.
        #[test]
        fn timer_handler() {
            dies_uncaught("child_timer");
        }
    }
}
