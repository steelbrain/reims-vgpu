//! A unit test's rt devices go as the test's thread ends, before its process exits. rt supports
//! no `exit` with a device of its live (its README, "Quiesce and exit"), and the process's
//! platform ([`crate::singleton::platform`]) keeps its device until a host's release of its
//! last graphics device lets go of it, which a test does not do. A test process that exits so
//! races rt's threads, still freeing the device's memory, against the drivers' exit handlers: on
//! an RTX 3090 (NVIDIA 620.70.06, Linux), two tiled-texture tests printed ok, then died of
//! `SIGSEGV`, rt's memory thread inside `vkDestroyBuffer` while `exit` tore down the driver's EGL
//! state on the main thread (**measured**, 2 of 2067 tests of one run).
//!
//! The test's thread: nextest runs each test in a process of its own, as `<binary> --exact
//! <name>`, and libtest runs it on a thread named after it, which it joins before `main` returns
//! and the process exits. In a process run with `--exact`, that thread keeps a hold of the
//! process's platform at its first use there ([`process_platform`]) and of each platform a test
//! makes of its own ([`platform`]) in a thread-local, whose destructor runs once the test's
//! function returned or unwound, before the join ends. It lets go of each one's device as a
//! host's release does (`RtPlatform::let_go_of_device`: it waits for the handles other threads
//! still hold, a decoder's or a video session's ending as the test's drops reached them, then
//! quiesces the device, waits for its teardown and drops the instance it was made on, which
//! closes the Vulkan loader where that was the last), on a thread of its own, since this
//! thread's other thread-locals may be gone by then. Where a device is still held after
//! `BOUND`, the process writes a line saying so and ends with `_exit` with the status of a
//! failed test, running no exit handler: the test fails, rather than its process exiting with
//! the device live.
//!
//! The hold is strong, so that a platform's last hold is never a thread's that nothing joins: a
//! thread the test's drops left holding the platform (a decoder's watch over its waiting
//! command buffers, a VA display's call thread) drops it after the let-go, when its drop finds
//! no device. Held weakly, the hold missed a drop already running elsewhere, as an `Arc`'s last
//! drop takes its strong count to zero before the value's drop runs: on an RTX 3090 (NVIDIA
//! 620.70.06), `graphics::decode::cmdbuf`'s test of the platform's own indices printed ok, then
//! died of `SIGSEGV` with the decoder's `reims-vgpu.graphics-event-waits` thread inside
//! `vkDestroyDevice`, in the platform's drop and its let-go, while the main thread ran
//! `libEGL_nvidia.so.0`'s exit handler (**measured** from its core, 1 of 2440 tests under the
//! suite's load, 2026-11-20); on a 780M, NVIDIA's driver installed beside RADV, the same test
//! died after ok with its main thread fetching an instruction from an unmapped address
//! (**measured**, the kernel's log, 1 of 2427 tests). And on an
//! i9-13900H's Iris Xe, with NVIDIA 620.70.06 installed beside Mesa 26.2.6, a VA display's call
//! thread so dropped the last hold of `graphics::vaapi`'s tests of writes into dma-buf caches,
//! its let-go closing the Vulkan loader beside the exit's handlers: they printed ok and died of
//! `SIGSEGV` in 3 of 9 processes, at `libEGL_nvidia.so.0` + 0xa28c2 after its link map was
//! destroyed (**measured**, `LD_DEBUG=files`, 2026-11-20; docs/lessons.md). The NVIDIA crate
//! keeps the first runtime it is opened with, and so the platform in it, for the process's life
//! (its waits' `RUNTIME`): an NVDEC or NVENC test's platform is let go of here and never
//! dropped.
//!
//! Unix only: Windows runs a thread's destructors under the loader lock, which a thread started
//! or joined there waits for. There a test's process exits with the device as before.

use std::sync::Arc;

use reims_vgpu_vulkan::RtPlatform;

#[cfg(unix)]
pub(crate) use end::process_platform;

/// A platform of the test's own, in place of `Arc::new(RtPlatform::new())`: its device goes as
/// the test's thread ends where a handle of it outlives the test's own (a video session's
/// thread, a decoder's watch, a pool's clock).
pub(crate) fn platform() -> Arc<RtPlatform> {
    let platform = Arc::new(RtPlatform::new());
    #[cfg(unix)]
    end::keep(&platform);
    platform
}

/// The let-go at the end of the test's thread.
#[cfg(unix)]
mod end {
    use std::cell::{Cell, RefCell};
    use std::io::Write as _;
    use std::sync::{Arc, OnceLock};
    use std::time::Duration;

    use reims_vgpu_vulkan::RtPlatform;

    /// How long a test's end waits for each device's let-go.
    const BOUND: Duration = Duration::from_secs(30);

    /// The status the process ends with where a device outlived [`BOUND`]: libtest's for a
    /// failed test.
    const FAILED: i32 = 101;

    thread_local! {
        /// The platforms whose devices go as this thread ends: the test's thread's alone.
        static AT_THE_END: RefCell<Option<AtTheEnd>> = const { RefCell::new(None) };
        /// Whether this thread looked at the process's platform before, so that each later use
        /// costs one read.
        static LOOKED: Cell<bool> = const { Cell::new(false) };
    }

    /// What the test's thread lets go of as it ends.
    struct AtTheEnd {
        /// The platforms, each held to the let-go, which leaves it no device for a later drop on
        /// another thread to tear down (the module docs).
        platforms: Vec<Arc<RtPlatform>>,
    }

    impl Drop for AtTheEnd {
        fn drop(&mut self) {
            let platforms = std::mem::take(&mut self.platforms);
            if platforms.is_empty() {
                return;
            }
            let ended = std::thread::Builder::new()
                .name("rt test end".to_owned())
                .spawn(move || {
                    (platforms.iter()).try_for_each(|platform| platform.let_go_of_device(BOUND))
                })
                .map(|thread| thread.join());
            let why = match ended {
                Ok(Ok(Ok(()))) => return,
                Ok(Ok(Err(held))) => format!("after {BOUND:?}, {held} still held it"),
                Ok(Err(_)) => "its let-go panicked".to_owned(),
                Err(error) => format!("no thread could let go of it ({error})"),
            };
            let _ = writeln!(
                std::io::stderr(),
                "reims-vgpu: the test's end left an rt device live: {why}"
            );
            // SAFETY: `_exit` ends the process at once, running no exit handler, as rt's README
            // asks of a process that ends with a device live.
            unsafe { libc::_exit(FAILED) }
        }
    }

    /// Whether this thread is the one libtest runs the process's test on: the process was run
    /// with `--exact`, as nextest runs each test, and the thread is named as one of its
    /// arguments.
    fn is_the_test_s_thread() -> bool {
        static NAMES: OnceLock<Vec<String>> = OnceLock::new();
        let names = NAMES.get_or_init(|| {
            let args: Vec<String> = (std::env::args_os())
                .filter_map(|arg| arg.into_string().ok())
                .collect();
            if args.iter().any(|arg| arg == "--exact") {
                args
            } else {
                Vec::new()
            }
        });
        (std::thread::current().name()).is_some_and(|name| names.iter().any(|arg| arg == name))
    }

    /// Holds `platform` to the let-go of its device at the end of the test's thread, where this
    /// is it.
    pub(super) fn keep(platform: &Arc<RtPlatform>) {
        if !is_the_test_s_thread() {
            return;
        }
        let _ = AT_THE_END.try_with(|end| {
            (end.borrow_mut().get_or_insert_with(|| AtTheEnd {
                platforms: Vec::new(),
            }))
            .platforms
            .push(Arc::clone(platform));
        });
    }

    /// The process's platform, at each use ([`crate::singleton::platform`]): kept at its first
    /// use on the test's thread.
    pub(crate) fn process_platform(platform: &Arc<RtPlatform>) {
        if !LOOKED.replace(true) {
            keep(platform);
        }
    }
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use std::io::Write as _;
    use std::process::Command;
    use std::sync::{Arc, OnceLock};
    use std::time::Duration;

    use metal2vulkan_runtime::{Counters, Gauges};
    use reims_vgpu_vulkan::RtPlatform;
    use reims_vgpu_vulkan::core::platform::Handle as _;
    use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
    use reims_vgpu_vulkan::core::platform::types::NsRange;
    use reims_vgpu_vulkan::core::raise::OnFault;
    use reims_vgpu_vulkan::core::testkit::death::{exit_signal, no_core_at_spawn};
    use reims_vgpu_vulkan::rt_platform::{RtDevice, RtObj};

    use crate::platform::ApiPlatform;

    const PROCESS_S: &str =
        "rt_test_end::tests::on_vulkan_a_test_s_rt_device_goes_before_its_process_exits";
    const A_THREAD_S: &str =
        "rt_test_end::tests::on_vulkan_a_platform_a_thread_drops_last_has_no_device_by_the_exit";
    const CHILD: &str = "RVG_RT_TEST_END_CHILD";
    /// The child's line from its exit handler.
    const AT_EXIT: &str = "the rt device at the exit: torn down";
    /// How long the thread of [`a_thread_s_child`] holds the device after its hold of the
    /// platform went.
    const DEVICE_HELD: Duration = Duration::from_millis(300);

    /// The child's device's counters, for its exit handler.
    static COUNTERS: OnceLock<Counters> = OnceLock::new();

    /// The child's exit handler, registered by its test after the device's work: whether rt's
    /// gauges of the device read zero, as they do once it is torn down.
    extern "C" fn at_exit() {
        let torn_down = (COUNTERS.get()).map(|counters| counters.gauges() == Gauges::default());
        let _ = writeln!(std::io::stderr(), "{AT_EXIT} {torn_down:?}");
    }

    /// `platform`'s device, once GPU work ran on it; its counters kept and the exit handler
    /// registered.
    fn worked(platform: &RtPlatform) -> RtObj<RtDevice> {
        const LEN: u64 = 1 << 20;
        let device = platform.metal_device(None).expect("the rt device");
        let buffer = device
            .new_buffer_of_length(OnFault::Die, LEN, 0)
            .unwrap()
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let cb = queue.command_buffer(OnFault::Die).unwrap().unwrap();
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        let fill = BlitCmd::FillBuffer {
            buffer: &buffer,
            range: NsRange::new(0, LEN),
            value: 0x5a,
        };
        blit.encode(OnFault::Die, fill).unwrap();
        blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(COUNTERS.set(device.rt().counters()).is_ok());
        // SAFETY: a handler that takes nothing and returns, as `atexit` asks.
        assert_eq!(unsafe { libc::atexit(at_exit) }, 0);
        device
    }

    /// The child of the process's platform: a test that takes the process's rt device as the
    /// tests do, runs GPU work on it and returns, its own handles dropped and the platform
    /// keeping the device.
    fn process_s_child() {
        drop(worked(&<RtPlatform as ApiPlatform>::process()));
    }

    /// The child of a thread's last hold: a test that makes a platform of its own, runs GPU work
    /// on its device and leaves a hold of each to a thread nothing joins. Once the test's own
    /// holds went, as its function returns, the thread drops its hold of the platform, then,
    /// [`DEVICE_HELD`] later, the device's handle. Where the test's end held the platform
    /// weakly, the thread's hold was then the platform's last, and the platform's drop on that
    /// thread let go of the device beside the exit, after the end found the platform gone.
    fn a_thread_s_child() {
        const POLL: Duration = Duration::from_millis(1);
        let platform = super::platform();
        let device = worked(&platform);
        let (go, went) = std::sync::mpsc::channel::<()>();
        let held = (Arc::clone(&platform), device.clone_ref());
        let thread = std::thread::Builder::new()
            .name("rt test end's other hold".to_owned())
            .spawn(move || {
                let (platform, device) = held;
                let _ = went.recv();
                drop(platform);
                std::thread::sleep(DEVICE_HELD);
                drop(device);
            });
        // Nothing joins it.
        drop(thread.unwrap());
        drop(device);
        let weak = Arc::downgrade(&platform);
        drop(platform);
        let before = weak.strong_count();
        go.send(()).unwrap();
        let mut waited = Duration::ZERO;
        while weak.strong_count() >= before {
            assert!(waited < Duration::from_secs(10), "the thread kept its hold");
            std::thread::sleep(POLL);
            waited += POLL;
        }
    }

    /// Runs `test` again as nextest runs it, as the child: its exit handler found the device
    /// torn down, and it passed.
    fn its_child_s_device_went_before_the_exit(test: &str) {
        let out = no_core_at_spawn(&mut Command::new(std::env::current_exe().unwrap()))
            .args(["--exact", test, "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains(&format!("{AT_EXIT} Some(true)")),
            "the device went before the exit: {:?}, signal {:?}\n{stderr}",
            out.status,
            exit_signal(out.status),
        );
        assert_eq!(out.status.code(), Some(0), "{stderr}");
    }

    /// A test that ends with the process's rt device kept leaves its process no device by the
    /// exit's handlers: the end of its thread let go of it. The child is this test run again as
    /// nextest runs it, which writes from an exit handler whether rt tore the device down.
    #[test]
    fn on_vulkan_a_test_s_rt_device_goes_before_its_process_exits() {
        if std::env::var_os(CHILD).is_some() {
            process_s_child();
            return;
        }
        its_child_s_device_went_before_the_exit(PROCESS_S);
    }

    /// A test's platform whose last hold, after the test's own, is a thread's that nothing joins
    /// leaves its process no device by the exit's handlers: the end of the test's thread held the
    /// platform, and let go of its device once the thread's handle of it went.
    #[test]
    fn on_vulkan_a_platform_a_thread_drops_last_has_no_device_by_the_exit() {
        if std::env::var_os(CHILD).is_some() {
            a_thread_s_child();
            return;
        }
        its_child_s_device_went_before_the_exit(A_THREAD_S);
    }
}
