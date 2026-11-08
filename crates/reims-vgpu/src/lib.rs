//! The library's Rust API (docs/api.md section 3): the reims-graphics device ([`GraphicsDevice`]),
//! which the C API (`reims-vgpu-c`) wraps one to one.
//!
//! A device is made from one config ([`GraphicsConfig`]) holding closures for its callbacks, and
//! answers a [`Result`] with what refused it ([`Error`]). It is generic over the platform with
//! [`DefaultPlatform`] as the default, so its forwarding is unit-tested on core's fake platform.
//! The Vulkan platform, `VulkanPlatform`, over metal2vulkan's runtime, draws the device's window
//! into a swapchain of its window, on macOS one over the window's `CAMetalLayer`.
//!
//! **By OS.** On macOS the default platform is `MetalPlatform`, and the `vulkan` feature adds
//! `VulkanPlatform` beside it. On Linux the library is the Vulkan platform's, with or without the
//! feature, and `VulkanPlatform` is the default. The build script sets `cfg(vulkan_platform)`
//! wherever the library has `VulkanPlatform`, and the code is gated on it.
//!
//! The library logs through the `log` crate: a Rust host installs its own logger.

// With the Vulkan platform, the proof that a window's presenter is `Send` (its swapchain, the
// swapchains it retired, and the runtime's types under them) nests deeper than rustc's default
// limit of 128 (E0275), as the C crate's does.
#![recursion_limit = "256"]

/// The unit tests' allocator, which counts a thread's allocations where a test asks.
#[cfg(test)]
mod counted;
#[cfg(test)]
#[global_allocator]
static ALLOCATOR: counted::Counted = counted::Counted;

mod cursor;
mod device_writes;
mod entry;
mod error;
mod graphics;
mod memory;
pub mod platform;
pub mod singleton;
mod video;
mod window;

pub use cursor::{CursorGlyph, CursorGlyphFn, CursorMoveFn, CursorShowFn};
pub use error::Error;
pub use graphics::{
    GraphicsCallbacks, GraphicsConfig, GraphicsDevice, GraphicsRaiseFn, GraphicsVideo,
    GraphicsVideoSession, PlanePages, VideoDecode, VideoPlane,
};
pub use memory::{MemoryFile, MemoryMap};
pub use platform::ApiPlatform;
#[cfg(target_os = "macos")]
pub use reims_vgpu_metal::MetalPlatform;
#[cfg(vulkan_platform)]
pub use reims_vgpu_vulkan::VulkanPlatform;

/// The platform every handle is made on by default: `MetalPlatform` on macOS.
#[cfg(target_os = "macos")]
pub type DefaultPlatform = MetalPlatform;
/// The platform every handle is made on by default: `VulkanPlatform` on Linux.
#[cfg(target_os = "linux")]
pub type DefaultPlatform = VulkanPlatform;
pub use reims_vgpu_window::{
    BUTTON_EXTRA, BUTTON_LEFT, BUTTON_MIDDLE, BUTTON_RIGHT, BUTTON_SIDE, CloseFn, KeyFn,
    MomentumPhase, PointerFn, ScrollEvent, ScrollFn, ScrollPhase, WheelFn, WindowCallbacks,
    WindowConfig,
};
