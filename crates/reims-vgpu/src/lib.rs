//! Reims vGPU host path — single crate.
//!
//! | Module | Role |
//! | --- | --- |
//! | [`model`] | Live guest-visible state (regs, rings, objects, present) |
//! | [`runtime`] | Composition, drain, resolution, and executor adaptation |
//! | [`qemu`] | QEMU C ABI surface only |
//!
//! The product executor is the sibling `reims-vgpu-vulkan` crate.
//!
//! # The two supported pathways
//!
//! Both pathways use the same backend and differ only in the Vulkan loader:
//!
//! | Arm | `cfg` | Host GPU API |
//! | --- | --- | --- |
//! | Vulkan / MoltenVK | `target_os = "macos"` | MoltenVK |
//! | Vulkan / native | `target_os = "linux"` | native ICD |
//!
//! **Gate the host on `target_os` and nothing else.** `macos` and `linux` are
//! the only two values this crate names, so the pathways differ in one term
//! each and a reader greps one key to find every host gate.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(rust_2018_idioms)]

// Vulkan reaches the GPU through MoltenVK on macOS and a native ICD on Linux.
// Any other host is untested rather than known-broken — name it here so a new
// port is a deliberate edit to this list, not an accident.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
compile_error!(
    "the Vulkan backend is supported on target_os = \"macos\" (MoltenVK) and \
     target_os = \"linux\" (native ICD) only"
);

/// Every environment variable this device reads, and the rule that an override
/// may only narrow what it does — see the module doc.
pub mod env;
#[cfg(test)]
mod iosurface_contract_tests;
pub mod model;
/// Crate-wide observability: the always-on fail sink and the decline
/// vocabulary. Above `runtime/` because every subsystem owes the reader a
/// reason, and `translate/` + `caps/` must be able to name one without
/// depending on `runtime/`.
pub mod observe;
pub mod runtime;

pub mod qemu;

/// Host-owned presentation window (winit + VkSurfaceKHR) — see
/// [[host-window]]. The `host-window` feature adds the windowing adapter and is
/// enabled for every verification command the x86 pathway is checked with.
#[cfg(feature = "host-window")]
pub mod host_window;

/// The device registry and the entry surface `qemu::abi` wraps. Private, with
/// the names that surface reaches re-exported below — the shape
/// `display_surface` and `window_publish` already use.
mod device;
pub(crate) use device::{
    backend_name, device_console_feed, device_create, device_cursor_glyph_copy,
    device_cursor_glyph_info, device_destroy, device_drain, device_efi_console_copy,
    device_gfx_read, device_gfx_write, device_iosfc_read, device_iosfc_write, device_poll,
    device_pop_action, device_reset, device_scanout_copy, device_scanout_may_paint,
    device_window_run_main, device_window_set_early_fb, device_window_start, device_window_stop,
    unwind_safe, ConsoleFeed, CursorGlyphInfo,
};
