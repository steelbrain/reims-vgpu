//! The process-wide platform and the panic-hook install.
//!
//! The process-wide state (the serial number set, the latest IOSurface host device) lives in one
//! platform instance (D17), so every public constructor goes through the process's: on macOS
//! [`platform`], one `MetalPlatform` per process, made on first use. Making it installs the
//! aborting panic hook first (6.5), then the stderr logger if `REIMS_VGPU_LOG` asks for it
//! (`reims_vgpu_core::logging::init`). Every function of the C API installs the hook too, before
//! its body runs, since some make no platform.
//!
//! With the `vulkan` feature, `vulkan_platform` is the process's `VulkanPlatform`, made the
//! same way on its first use: the only platform on Linux.

// The docs name macOS's platform, which other OSes have none of: their links resolve, and are
// checked, on macOS.
#![cfg_attr(not(target_os = "macos"), allow(rustdoc::broken_intra_doc_links))]

use std::sync::{Arc, Once, OnceLock};

use reims_vgpu_core::platform::os::Preferences;
use reims_vgpu_core::rt::is_our_thread;
use reims_vgpu_core::tasks::isolation::TaskThreads;
#[cfg(target_os = "macos")]
use reims_vgpu_metal::MetalPlatform;
#[cfg(feature = "vulkan")]
use reims_vgpu_vulkan::VulkanPlatform;

#[cfg(target_os = "macos")]
static PLATFORM: OnceLock<Arc<MetalPlatform>> = OnceLock::new();
#[cfg(feature = "vulkan")]
static VULKAN_PLATFORM: OnceLock<Arc<VulkanPlatform>> = OnceLock::new();
static HOOK: Once = Once::new();

/// The process's platform (D17) on macOS, made on first use: the panic hook first, the logger,
/// then one `MetalPlatform` whose launcher runs each isolated task's service on threads of its own
/// ([`TaskThreads`]).
#[cfg(target_os = "macos")]
pub fn platform() -> Arc<MetalPlatform> {
    platform_ref().clone()
}

/// [`platform`], borrowed for the life of the process.
#[cfg(target_os = "macos")]
pub fn platform_ref() -> &'static Arc<MetalPlatform> {
    PLATFORM.get_or_init(|| {
        install_panic_hook();
        reims_vgpu_core::logging::init();
        Arc::new(MetalPlatform::new(Box::new(TaskThreads::new())))
    })
}

/// The process's Vulkan platform, made on first use as [`platform`] makes the Metal one: the
/// panic hook first, the logger, then one `VulkanPlatform` whose launcher runs each isolated
/// task's service on threads of its own ([`TaskThreads`]).
#[cfg(feature = "vulkan")]
pub fn vulkan_platform() -> Arc<VulkanPlatform> {
    VULKAN_PLATFORM
        .get_or_init(|| {
            install_panic_hook();
            reims_vgpu_core::logging::init();
            Arc::new(VulkanPlatform::new(Box::new(TaskThreads::new())))
        })
        .clone()
}

/// The process's settings reader, for [`crate::DisplayConfig::apply_preferences`] and
/// [`crate::DisplayMode::with_preferences`]: every key is absent, as the library reads no
/// preferences domain.
pub fn preferences() -> &'static dyn Preferences {
    &NoPreferences
}

/// A reader with no settings.
struct NoPreferences;

impl Preferences for NoPreferences {
    fn app_bool(&self, _key: &str) -> (bool, bool) {
        (false, false)
    }

    fn any_user_f32(&self, _key: &str) -> Option<f32> {
        None
    }
}

/// The line the hook writes before it aborts.
pub(crate) fn hook_message(info: &dyn std::fmt::Display) -> String {
    format!("reims-vgpu: panic: {info}\n")
}

/// Installs the panic hook of architecture 6.5, once per process. A panic is a bug, and a panic
/// in the library's code (on the FIFO thread, in one of its queue blocks, in a Metal completion
/// handler, inside a public entry point or a C export, each marked by `rt::enter_ours`) ends the
/// process as a death does: one line on stderr, then `abort()`, before anything unwinds, so no
/// panic crosses into the host's frames and no drop runs first. A panic anywhere else is the
/// host's own, and goes to the previous hook.
pub fn install_panic_hook() {
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if is_our_thread() {
                use std::io::Write as _;
                let _ = std::io::stderr().write_all(hook_message(info).as_bytes());
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

    /// platform() is one per process (D17). In a child: making the platform installs the aborting
    /// hook, which must not meet the death-capture panics of this binary's fake-platform tests.
    #[cfg(target_os = "macos")]
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

    /// vulkan_platform() is one per process, and installs the aborting hook as platform() does;
    /// in a child, for the reason above.
    #[cfg(feature = "vulkan")]
    #[test]
    fn one_vulkan_platform_per_process() {
        if std::env::var_os("RVG_VULKAN_PLATFORM_CHILD").is_some() {
            let a = vulkan_platform();
            let b = vulkan_platform();
            assert!(Arc::ptr_eq(&a, &b));
            let _ours = reims_vgpu_core::rt::enter_ours();
            panic!("after vulkan_platform()");
        }
        let out = child(
            "singleton::tests::one_vulkan_platform_per_process",
            "RVG_VULKAN_PLATFORM_CHILD",
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
