//! The boundary's tests on every OS; each OS's own catch has its tests beside them.

use std::os::unix::process::ExitStatusExt as _;
use std::process::Command;

use reims_vgpu_core::raise::death::{DeathKind, DeathSite};
use reims_vgpu_core::rt::is_our_thread;

use super::*;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;

/// The death of a host raise at `call`.
fn host_raised(call: &'static str) -> DeathSite {
    DeathSite {
        kind: DeathKind::HostRaised,
        file: call,
        line: 0,
    }
}

/// A call that returns answers its value
#[test]
fn a_host_call_answers_its_value() {
    assert_eq!(host_call("count", || 3u64), 3);
}

/// An export's body runs marked as the library's; the mark nests, and the one found is back
/// when the export returns
#[test]
fn an_export_marks_its_thread_for_the_call() {
    assert!(!is_our_thread());
    let got = export(|| {
        assert!(is_our_thread());
        export(|| assert!(is_our_thread()));
        assert!(is_our_thread());
        7
    });
    assert_eq!(got, 7);
    assert!(!is_our_thread());
}

/// Re-runs this test binary on one test with `var` set.
fn child(test: &str, var: &str) -> std::process::Output {
    Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture", "--test-threads=1"])
        .env(var, "1")
        .output()
        .unwrap()
}

/// Prints when dropped.
struct Loud;

impl Drop for Loud {
    fn drop(&mut self) {
        println!("DROPPED");
    }
}

/// A panic under an export ends the process in the library's hook before anything unwinds,
/// though nothing made the platform first: the export installs the hook itself. In a child,
/// since the hook is the process's.
#[test]
fn a_panic_under_an_export_aborts_before_unwinding() {
    if std::env::var_os("RVG_FFI_PANIC_CHILD").is_some() {
        export(|| {
            let _l = Loud;
            panic!("under an export");
        });
        return;
    }
    let out = child(
        "ffi::tests::a_panic_under_an_export_aborts_before_unwinding",
        "RVG_FFI_PANIC_CHILD",
    );
    assert_eq!(out.status.signal(), Some(6), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stdout.contains("DROPPED"), "{stdout}");
    assert!(
        stderr.contains("reims-vgpu: panic: ") && stderr.contains("under an export"),
        "{stderr}"
    );
}

/// A destroy that panics.
unsafe extern "C-unwind" fn panicking_destroy(_ctx: *mut std::ffi::c_void) {
    panic!("in the host's destroy");
}

/// A panic in a host callback that the first export of the process runs (a refused create's
/// destroy, before any platform exists) ends in the library's hook: the export installed it
/// and marked the thread. In a child, since the hook is the process's.
#[test]
fn a_panic_in_a_callback_of_the_first_export_ends_in_the_hook() {
    if std::env::var_os("RVG_FFI_CALLBACK_CHILD").is_some() {
        let callbacks = crate::graphics::RvgGraphicsCallbacks {
            struct_size: crate::sized::size_u32::<crate::graphics::RvgGraphicsCallbacks>(),
            ctx: std::ptr::null_mut(),
            destroy: Some(panicking_destroy),
            raise_interrupt: None,
            scanout_written: None,
            cursor_glyph: None,
            cursor_show: None,
            cursor_move: None,
        };
        let config = crate::graphics::RvgGraphicsDeviceConfig {
            struct_size: crate::sized::size_u32::<crate::graphics::RvgGraphicsDeviceConfig>(),
            callbacks: &callbacks,
            window: std::ptr::null(),
            memory_map: std::ptr::null(),
            memory_files: std::ptr::null(),
        };
        // SAFETY: a live config; a NULL out-pointer, which the create refuses.
        let _ =
            unsafe { crate::graphics::rvg_graphics_device_create(&config, std::ptr::null_mut()) };
        return;
    }
    let out = child(
        "ffi::tests::a_panic_in_a_callback_of_the_first_export_ends_in_the_hook",
        "RVG_FFI_CALLBACK_CHILD",
    );
    assert_eq!(out.status.signal(), Some(6), "{out:?}");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("reims-vgpu: panic: ") && stderr.contains("in the host's destroy"),
        "{stderr}"
    );
}
