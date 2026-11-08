//! The process-wide platform and the panic-hook install.
//!
//! The process-wide state (the serial number set, the latest IOSurface host device) lives in one
//! platform instance (D17), so every public constructor goes through the process's: [`platform`],
//! one `VulkanPlatform` per process, made on first use. Making it installs the aborting panic hook
//! first (6.5), then the stderr logger if `REIMS_VGPU_LOG` asks for it
//! (`reims_vgpu_core::logging::init`). Every function of the C API installs the hook too, before
//! its body runs, since some make no platform.

use std::sync::{Arc, Once, OnceLock};

use reims_vgpu_core::rt::is_our_thread;
use reims_vgpu_vulkan::VulkanPlatform;

static PLATFORM: OnceLock<Arc<VulkanPlatform>> = OnceLock::new();
static HOOK: Once = Once::new();

/// The process's platform (D17), made on first use: the panic hook first, the logger, then one
/// `VulkanPlatform`.
pub fn platform() -> Arc<VulkanPlatform> {
    PLATFORM
        .get_or_init(|| {
            install_panic_hook();
            reims_vgpu_core::logging::init();
            Arc::new(VulkanPlatform::new())
        })
        .clone()
}

/// The line the hook writes before it aborts.
pub(crate) fn hook_message(info: &dyn std::fmt::Display) -> String {
    format!("reims-vgpu: panic: {info}\n")
}

/// Installs the panic hook of architecture 6.5, once per process. A panic is a bug, and a panic
/// in the library's code (on one of its threads, in one of its queue blocks, in a Metal completion
/// handler, inside a public entry point or a C export, each marked by `rt::enter_ours`) ends the
/// process as a death does: one line on stderr, then `abort()`, before anything unwinds, so no
/// panic crosses into the host's frames and no drop runs first. Inside a refusal scope (the
/// graphics lane's ops) it is refused as a death there is: the line is written, and the panic
/// unwinds to the scope, which refuses the op (`reims_vgpu_core::raise::refusals`, "Panics"). A
/// panic anywhere else is the host's own, and goes to the previous hook.
pub fn install_panic_hook() {
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if is_our_thread() {
                use std::io::Write as _;
                let _ = std::io::stderr().write_all(hook_message(info).as_bytes());
                if reims_vgpu_core::raise::refusals::refuses_panics() {
                    return;
                }
                #[allow(clippy::disallowed_methods)] // a panic's abort (6.5)
                std::process::abort();
            }
            previous(info);
        }));
    });
}

#[cfg(test)]
mod tests {

    use std::os::unix::process::ExitStatusExt as _;
    use std::process::Command;

    use super::*;

    /// Re-runs this test binary on one test with `var` set.
    fn child(test: &str, var: &str) -> std::process::Output {
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture", "--test-threads=1"])
            .env(var, "1")
            .output()
            .unwrap()
    }

    /// A panic of ours inside a refusal scope is refused there, its line written, and the process
    /// goes on; outside every scope the hook still ends the process. In a child, with the
    /// capture off, as the product runs.
    #[test]
    fn a_panic_in_a_refusal_scope_is_refused_and_the_process_goes_on() {
        use reims_vgpu_core::raise::death::DeathId;
        use reims_vgpu_core::raise::refusals::refusing;
        if std::env::var_os("RVG_SCOPED_PANIC_CHILD").is_some() {
            reims_vgpu_core::testkit::capture::set_capture_enabled(false);
            install_panic_hook();
            let _ours = reims_vgpu_core::rt::enter_ours();
            let r: Result<(), _> = refusing(|| panic!("in a guest op"));
            assert_eq!(r.unwrap_err().id, DeathId::Panic);
            let _ = std::io::Write::write_all(&mut std::io::stderr(), b"the process goes on\n");
            panic!("outside every scope");
        }
        let out = child(
            "singleton::tests::a_panic_in_a_refusal_scope_is_refused_and_the_process_goes_on",
            "RVG_SCOPED_PANIC_CHILD",
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.signal(), Some(6), "{out:?}");
        let refused = stderr
            .find("reims-vgpu: panic: ")
            .expect("the refused panic's line");
        let goes_on = stderr
            .find("the process goes on")
            .expect("the process went on");
        let ended = stderr
            .rfind("outside every scope")
            .expect("the panic outside");
        assert!(refused < goes_on && goes_on < ended, "{stderr}");
    }

    /// platform() is one per process (D17). In a child: making the platform installs the aborting
    /// hook, which must not meet the death-capture panics of this binary's fake-platform tests.
    #[test]
    fn one_platform_per_process() {
        if std::env::var_os("RVG_PLATFORM_CHILD").is_some() {
            let a = platform();
            let b = platform();
            assert!(Arc::ptr_eq(&a, &b));
            // The hook is in: a panic on our thread now aborts.
            let _ours = reims_vgpu_core::rt::enter_ours();
            panic!("after platform()");
        }
        let out = child(
            "singleton::tests::one_platform_per_process",
            "RVG_PLATFORM_CHILD",
        );
        assert_eq!(out.status.signal(), Some(6), "{out:?}");
        assert!(String::from_utf8_lossy(&out.stderr).contains("reims-vgpu: "));
    }

    /// The hook message names the crate and the panic
    #[test]
    fn hook_message_text() {
        assert_eq!(hook_message(&"boom"), "reims-vgpu: panic: boom\n");
    }

    /// A panic on our thread aborts before unwinding (6.5, D9): the child panics inside
    /// `enter_ours` holding a value whose drop would print.
    #[test]
    fn panic_on_our_thread_aborts_before_unwinding() {
        if std::env::var_os("RVG_HOOK_OURS_CHILD").is_some() {
            struct Loud;
            impl Drop for Loud {
                fn drop(&mut self) {
                    println!("DROPPED");
                }
            }
            install_panic_hook();
            let _ours = reims_vgpu_core::rt::enter_ours();
            let _l = Loud;
            panic!("ours panic");
        }
        let out = child(
            "singleton::tests::panic_on_our_thread_aborts_before_unwinding",
            "RVG_HOOK_OURS_CHILD",
        );
        assert_eq!(out.status.signal(), Some(6), "{out:?}");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!stdout.contains("DROPPED"), "{stdout}");
        assert!(
            stderr.contains("reims-vgpu: ") && stderr.contains("ours panic"),
            "{stderr}"
        );
    }

    /// A panic on another thread goes to the previous hook and unwinds (the host's own panic is not
    /// ours to end)
    #[test]
    fn panic_elsewhere_defers_to_the_previous_hook() {
        if std::env::var_os("RVG_HOOK_OTHER_CHILD").is_some() {
            struct Loud;
            impl Drop for Loud {
                fn drop(&mut self) {
                    println!("DROPPED");
                }
            }
            install_panic_hook();
            install_panic_hook(); // once only: a second install changes nothing
            let _l = Loud;
            panic!("host panic");
        }
        let out = child(
            "singleton::tests::panic_elsewhere_defers_to_the_previous_hook",
            "RVG_HOOK_OTHER_CHILD",
        );
        assert_eq!(out.status.signal(), None, "{out:?}");
        assert!(!out.status.success());
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stdout.contains("DROPPED"), "{stdout}");
        assert!(!stderr.contains("reims-vgpu: "), "{stderr}");
        assert!(stderr.contains("host panic"), "{stderr}");
    }
}
