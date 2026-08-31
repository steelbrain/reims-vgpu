//! Host-owned presentation window ([[host-window]]) — a Rust-owned `winit`
//! window with its own `VkSurfaceKHR`/swapchain that replaces QEMU's built-in UI
//! and presents the engine frame directly, keeping the C/QEMU side thin.
//!
//! Gated behind the `host-window` cargo feature, which implies `backend-vulkan`.
//! It is the display path the x86 pathway is verified on, so the feature is
//! enabled in every `cargo` command `AGENTS.md` gives for the Vulkan arm.
//!
//! Three pieces:
//! - [`input_map`] — winit event → neutral [`crate::runtime::HostAction`]. Pure
//!   mapping, no window state, unit-tested off-VM.
//! - [`keyboard`] — which keys the guest believes are held, and whether the
//!   compositor's own shortcuts are being captured. Pure state machine; it owns
//!   the rule that every key-down is eventually closed by a key-up.
//! - [`capture`] — the per-platform request that stops the desktop from
//!   consuming shortcuts before the window sees them. A typed refusal where the
//!   platform cannot honour it.
//! - [`present`] — the window itself: event loop, surface, swapchain, and the
//!   acquire → blit → present loop. It also drives [`input_map`] and hands each
//!   action to an `InputSink`; `lib.rs` wires that to the device's prompt action
//!   queue through QEMU's thread-safe `notify_actions` callback.
//! - [`viewport`] — letterbox/scale arithmetic mapping guest framebuffer extent
//!   to window extent, shared by the blit and by input coordinate translation so
//!   a click lands where the pixel was drawn.

pub mod capture;
pub mod input_map;
pub mod keyboard;
pub mod present;
pub mod viewport;
