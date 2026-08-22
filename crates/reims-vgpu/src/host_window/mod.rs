//! Host-owned presentation window ([[host-window]]) — a Rust-owned `winit`
//! window with its own `VkSurfaceKHR`/swapchain that replaces QEMU's built-in UI
//! and presents the engine frame directly, keeping the C/QEMU side thin.
//!
//! Gated behind the `host-window` cargo feature; Vulkan itself is unconditional.
//! It is the display path the x86 pathway is verified on, so the feature is
//! enabled in every `cargo` command `AGENTS.md` gives for the Vulkan arm.
//!
//! Three pieces:
//! - [`input_map`] — winit event → neutral [`crate::runtime::HostAction`]. Pure
//!   mapping, no window state, unit-tested off-VM.
//! - [`present`] — the window itself: event loop, surface, swapchain, and the
//!   acquire → blit → present loop. It also drives [`input_map`] and hands each
//!   action to an `InputSink`; `lib.rs` wires that to the device's prompt action
//!   queue through QEMU's thread-safe `notify_actions` callback.
//! - [`viewport`] — letterbox/scale arithmetic mapping guest framebuffer extent
//!   to window extent, shared by the blit and by input coordinate translation so
//!   a click lands where the pixel was drawn.

pub mod input_map;
pub mod present;
pub mod viewport;
