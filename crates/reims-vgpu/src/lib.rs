//! Reims vGPU host path — single crate.
//!
//! | Module | Role |
//! | --- | --- |
//! | [`contract`] | Stable facts: formats, layouts, pure arithmetic |
//! | [`model`] | Live guest-visible state (regs, rings, objects, present) |
//! | [`runtime`] | Drain / parse / resolve / plan / HostActions |
//! | [`backend`] | Trait + self-contained [`backend::metal`] / [`backend::vulkan`] |
//! | [`qemu`] | QEMU C ABI surface only |
//!
//! Features: exactly one of `backend-metal` (default) or `backend-vulkan`.
//! Vulkan product path is self-contained `ash` ([`backend::vulkan::engine`]).
//!
//! # The three supported arms
//!
//! A build is exactly one of these, and the guards below reject anything else:
//!
//! | Arm | `cfg` | Host GPU API |
//! | --- | --- | --- |
//! | Metal | `all(feature = "backend-metal", target_os = "macos")` | native Metal |
//! | Vulkan / MoltenVK | `all(feature = "backend-vulkan", target_os = "macos")` | MoltenVK |
//! | Vulkan / native | `all(feature = "backend-vulkan", target_os = "linux")` | native ICD |
//!
//! **Gate the host on `target_os` and nothing else.** `macos` and `linux` are
//! the only two values this crate names, so the three arms differ in one term
//! each and a reader greps one key to find every host gate.
//!
//! There is **no** host-stub Metal arm. `backend-metal` off macOS has no Metal
//! to call, so it is a compile error rather than a binary that links and cannot
//! draw.
//!
//! The consequence the rest of the crate relies on: **the Metal arm and the
//! Vulkan arms partition every buildable configuration.** So the engine path is
//! spelled positively as `feature = "backend-vulkan"` and the Metal path as
//! `all(feature = "backend-metal", target_os = "macos")`, with no negation
//! of one standing in for the other. Do not reintroduce
//! `not(all(feature = "backend-metal", target_os = "macos"))` as a spelling
//! of "the engine path" — it says what the build is *not*, which stops being
//! equivalent the moment a fourth arm exists.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(rust_2018_idioms)]

#[cfg(all(feature = "backend-metal", feature = "backend-vulkan"))]
compile_error!("select exactly one of backend-metal or backend-vulkan");

#[cfg(not(any(feature = "backend-metal", feature = "backend-vulkan")))]
compile_error!("select exactly one of backend-metal or backend-vulkan");

#[cfg(all(feature = "backend-metal", not(target_os = "macos")))]
compile_error!(
    "backend-metal requires target_os = \"macos\": there is no host-stub Metal \
     arm. Use --no-default-features --features backend-vulkan,host-window on \
     any other host."
);

// Vulkan reaches the GPU through MoltenVK on macOS and a native ICD on
// Linux; Windows hosts use their native ICDs (NVIDIA/AMD/Intel ship
// VK_KHR_win32_surface and Vulkan 1.2+). Any other host is untested rather
// than known-broken — name it here so a new port is a deliberate edit to this
// list, not an accident.
#[cfg(all(
    feature = "backend-vulkan",
    not(any(target_os = "macos", target_os = "linux", target_os = "windows"))
))]
compile_error!(
    "backend-vulkan is supported on target_os = \"macos\" (MoltenVK), \
     target_os = \"linux\", and target_os = \"windows\" (native ICDs)"
);

/// The backend-neutral protocol vocabulary, in the crate that owns it.
///
/// Re-exported under the path every caller already writes
/// (`crate::contract::…`). See `reims_vgpu_contract` for what the crate
/// boundary makes true that the module boundary only asserted.
pub use reims_vgpu_contract as contract;
/// Every environment variable this device reads, and the rule that an override
/// may only narrow what it does — see the module doc.
/// Operator switches, in the crate that owns their names and their parse.
///
/// Re-exported under the path every caller already writes (`crate::env::…`) so
/// moving the module out did not move a call site. See `reims_vgpu_env` for why
/// a switch may only narrow what this device does.
pub use reims_vgpu_env as env;
pub mod model;
/// Crate-wide observability: the always-on fail sink and the decline
/// vocabulary. Above `runtime/` because every subsystem owes the reader a
/// reason, and `translate/` + `caps/` must be able to name one without
/// depending on `runtime/`.
pub mod observe;
pub mod runtime;

pub mod backend;
pub mod qemu;

/// Host-owned presentation window (winit + VkSurfaceKHR) — see
/// [[host-window]]. The `host-window` feature implies `backend-vulkan`, and is
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
