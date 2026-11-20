//! The library's Rust API (docs/api.md section 3): the reims-graphics device ([`GraphicsDevice`]),
//! which the C API (`reims-vgpu-c`) wraps one to one.
//!
//! A device is made from one config ([`GraphicsConfig`]) holding closures for its callbacks, and
//! answers a [`Result`] with what refused it ([`Error`]). It is generic over the platform with
//! [`DefaultPlatform`] as the default, so its forwarding is unit-tested on core's fake platform.
//! The platform is `RtPlatform`, over metal2vulkan-runtime, on the host's Vulkan driver on Linux and
//! Windows and on macOS on Mesa's KosmicKrisp; it draws the device's window into a swapchain of its
//! window, on macOS one over the window's `CAMetalLayer`.
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
#[cfg(all(windows, target_arch = "x86_64"))]
mod faults;
mod graphics;
mod memory;
pub mod platform;
mod pool;
/// A unit test's rt devices, let go of as the test's thread ends.
#[cfg(test)]
mod rt_test_end;
pub mod singleton;
mod video;
mod window;

pub use cursor::{
    CursorGlyph, CursorGlyphAtFn, CursorGlyphFn, CursorMoveAtFn, CursorMoveFn, CursorShowAtFn,
    CursorShowFn,
};
pub use error::Error;
pub use graphics::{
    GraphicsCallbacks, GraphicsConfig, GraphicsDevice, GraphicsRaiseFn, GraphicsRouteDoorbellFn,
    GraphicsRouteDoorbellPageFn, GraphicsVideo, GraphicsVideoSession, GuestArch, LET_GO_BOUND,
    PlanePages, VideoDecode, VideoPlane,
};
pub use memory::{HostFile, MemoryFile, MemoryMap};
pub use platform::ApiPlatform;
pub use pool::{BLOCK_LEN, PIECE_ALIGN, PoolAddFn, PoolConfig, PoolRemoveFn, RELEASE_DELAY};
pub use reims_vgpu_graphics::RawDoorbell;
pub use reims_vgpu_vulkan::RtPlatform;

/// The platform every handle is made on: `RtPlatform`, over metal2vulkan-runtime.
pub type DefaultPlatform = RtPlatform;
pub use reims_vgpu_window::{
    BUTTON_EXTRA, BUTTON_LEFT, BUTTON_MIDDLE, BUTTON_RIGHT, BUTTON_SIDE, CloseFn, KeyFn,
    MomentumPhase, PointerFn, ScrollEvent, ScrollFn, ScrollPhase, WheelFn, WindowCallbacks,
    WindowConfig,
};
