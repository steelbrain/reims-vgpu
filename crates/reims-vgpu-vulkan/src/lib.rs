//! The `Platform` implementation over `metal2vulkan-runtime`, a Metal-shaped API on Vulkan
//! (architecture D1): the library's only platform, natively on Linux, and over MoltenVK on macOS,
//! a build for development and tests.
//!
//! - **Core** ([`core`](crate::core)): what the reims-graphics device and its platform share,
//!   generic over one `P: Platform`: the platform traits, the runtime, raises and deaths, the
//!   guest's RAM and the testkit. It uses nothing else of this crate; the rest of the crate is
//!   the platform it is generic over.
//! - **GPU objects** are the runtime's ([`gpu`]): each handle is an [`Obj`] over the runtime's
//!   object and what core's trait asks of it beside (the device it came from, the IOSurface it
//!   retains).
//! - **The runtime's typed errors** become core's outcomes in one place, [`answer`]: a raise the
//!   send's policy decides, `None` where Metal answers nil, an `NsErrorText`, or a death naming
//!   the selector where the runtime refuses what Metal does.
//! - **A method not built yet** dies at its selector ([`answer::unbuilt`]), so a run's death line
//!   names the first selector the platform lacks; none answers a made-up value.
//! - **The OS services** ([`os`]): on macOS this crate's over the system's (threads, clocks, Mach
//!   VM, the autorelease pool, IOSurface and the Objective-C message layer), beside MoltenVK; on Linux this crate's own, over the host's
//!   Vulkan driver.
//! - **A display window's swapchain** (`swapchain`), on the runtime's own device: the window's
//!   render thread presents the display's frames through it, with no copy, on either OS.

// The proof that the runtime's types are `Send` and `Sync` nests deeper than rustc's default
// limit of 128 (E0275): in the library itself, where a `Debug` derive asks it of an encoder's
// indirect command buffers, as in the crates that build core over this platform.
#![recursion_limit = "256"]

pub mod answer;
pub mod api;
pub(crate) mod awake;
pub mod core;
pub mod gpu;
#[cfg(target_os = "linux")]
pub mod host_gpus;
pub mod os;
pub mod platform;
pub(crate) mod records;
pub mod remap;
pub mod swapchain;
pub(crate) mod work_log;

pub use platform::{Obj, VulkanPlatform};
