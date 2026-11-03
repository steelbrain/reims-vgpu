//! The library's Rust API (docs/api.md section 3): owned Rust types over core, which the C API
//! (`reims-vgpu-c`) wraps one to one.
//!
//! Each object is made from one config ([`DeviceConfig`], [`DisplayConfig`],
//! [`IoSurfaceDeviceConfig`]) holding closures for its callbacks, and answers a [`Result`] with
//! the check that refused it ([`Error`]). [`Device`], [`Display`], [`FrameTarget`] and
//! [`IoSurfaceDevice`] are counted references with `clone_ref()` and no `Clone`: the last drop
//! of a device, display or IOSurface device is its end, run on the releasing thread. They are
//! generic over the platform with [`DefaultPlatform`] as the default, so their forwarding is
//! unit-tested on core's fake platform. The `vulkan` feature adds `VulkanPlatform`, over
//! metal2vulkan's runtime, whose window displays draw each frame into a swapchain of their
//! window, on macOS one over the window's `CAMetalLayer`.
//!
//! **By OS.** On macOS the default platform is `MetalPlatform`, the product. On Linux the library
//! is the Vulkan platform's: it builds only with the `vulkan` feature, and `VulkanPlatform` is the
//! default.
//!
//! Behaviour is core's; this crate adds none. Its closures are called synchronously on the
//! thread core calls them on, in core's order, which is the order of VM-memory operations and
//! releases the host saw before. What it adds is where callbacks run: a display's on the
//! process's display thread ([`platform`]), in place of the host's dispatch queue; and a
//! display's own window, for a display created with one ([`DisplayConfig::window`]), which has
//! every display callback as the guest reads them (`window.rs`).
//!
//! The library logs through the `log` crate: a Rust host installs its own logger.

// With the `vulkan` feature, the proof that a window's presenter is `Send` (its swapchain, the
// swapchains it retired, and the runtime's types under them) nests deeper than rustc's default
// limit of 128 (E0275), as the C crate's does.
#![recursion_limit = "256"]

#[cfg(all(target_os = "linux", not(feature = "vulkan")))]
compile_error!("on Linux the library is the Vulkan platform's: build it with the `vulkan` feature");

/// The unit tests' allocator, which counts a thread's allocations where a test asks.
#[cfg(test)]
mod counted;
#[cfg(test)]
#[global_allocator]
static ALLOCATOR: counted::Counted = counted::Counted;

mod device;
mod device_writes;
pub mod dirty;
mod display;
mod entry;
mod error;
mod frame;
mod graphics;
mod iosurface;
pub mod platform;
mod retained;
mod scaler;
pub mod singleton;
mod stats;
mod video;
mod window;

pub use device::{Device, DeviceConfig, HostCallbacks, MemoryFile, MemoryMap};
pub use dirty::{DirtyHost, DirtySource, KvmSlot};
pub use display::{
    CursorGlyph, CursorGlyphFn, CursorMoveFn, CursorShowFn, Display, DisplayCallbacks,
    DisplayConfig, DisplayMode, ModeChangeFn, NewFrameFn,
};
pub use error::{Error, ResumeError};
pub use frame::{Frame, FrameTarget, MAX_FRAME_TARGET_SIZE, PixelFormat, Region};
pub use graphics::{
    GraphicsCallbacks, GraphicsConfig, GraphicsDevice, GraphicsRaiseFn, GraphicsVideo,
    GraphicsVideoSession, ScanoutWrittenFn, VideoDecode, VideoPlane,
};
pub use iosurface::{IoSurfaceDevice, IoSurfaceDeviceConfig};
pub use platform::ApiPlatform;
pub use reims_vgpu_core::host::{
    AddTraceRange, CreateTask, DestroyTask, HostPtr, HostTask, HostTraceRange, MapMemory,
    PhysicalMemoryRange, RaiseInterrupt, ReadMemory, RemoveTraceRange, TraceRangeHandler,
    UnmapMemory,
};
pub use reims_vgpu_core::platform::os::Preferences;
#[cfg(target_os = "macos")]
pub use reims_vgpu_metal::MetalPlatform;
/// reims-vgpu-video-core, for the tests of crates above this one: the scaler's CPU stand-in and
/// its capabilities, which they make devices on.
#[cfg(feature = "testkit")]
pub use reims_vgpu_video_core;
#[cfg(feature = "vulkan")]
pub use reims_vgpu_vulkan::VulkanPlatform;
pub use scaler::{GuestMemory, ScalerDevice, ScalerDeviceConfig, ScalerOutcome};
pub use video::{
    ALLOCATION_FAILED, INVALID_SESSION, MAX_MESSAGE_SIZE, SendToGuest, SurfacePlane, VideoDevice,
    VideoDeviceConfig,
};

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

/// The most display ports a device has.
pub const MAX_DISPLAY_PORTS: u32 = reims_vgpu_core::device::config::MAX_DISPLAY_PORT_COUNT;
/// The size of the device's MMIO region: the PCI BAR, or vmapple's first region.
pub const DEVICE_MMIO_SIZE: u64 = reims_vgpu_core::device::config::DEFAULT_MMIO_LENGTH;
/// The PCI vendor ID of the device.
pub const PCI_VENDOR_ID: u16 = 0x106b;
/// The PCI device ID of the device.
pub const PCI_DEVICE_ID: u16 = 0xeeee;
