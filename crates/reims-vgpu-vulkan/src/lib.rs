//! The `Platform` implementation over metal2vulkan-runtime, a Metal-shaped runtime on Vulkan
//! (architecture D1): the library's only platform, [`RtPlatform`], on the host's Vulkan driver on
//! Linux and Windows, and on macOS on Mesa's KosmicKrisp.
//!
//! - **Core** ([`core`]): what the reims-graphics device and its platform share,
//!   generic over one `P: Platform`: the platform traits, the runtime, raises and deaths, the
//!   guest's RAM and the testkit. It uses nothing else of this crate; the rest of the crate is
//!   the platform it is generic over.
//! - **The platform** ([`rt_platform`]): each of core's objects over rt's, and what core's traits
//!   ask of them beside.
//! - **The answer layer** ([`answer`]): the degradation and refusal counts a boot logs, and the
//!   deaths that name a selector where the platform refuses what Metal does.
//! - **The OS services** ([`os`]): on macOS this crate's over the system's (threads, clocks, Mach
//!   VM, the autorelease pool, IOSurface and the Objective-C message layer); on Linux and on
//!   Windows this crate's own.
//! - **A display window's swapchain**: what the rt platform's swapchain
//!   (`rt_platform::swapchain`) takes of the window system's rules (`swapchain`).
//! - **metal2vulkan-runtime's seam** ([`seam`]): our side of rt's contract.
//! - **The device's choice** ([`device_choice`]), the recorder ([`recorder`]) and the process's
//!   exit gate ([`exit_gate`]).
//! - **The host GPU's render node** (`host_gpus`, on Linux), which the device's VA-API video
//!   opens the GPU by.

// The proof that the platform's types are `Send` and `Sync` nests deeper than rustc's default
// limit of 128 (E0275), as in the crates that build core over this platform.
#![recursion_limit = "256"]

pub mod answer;
pub mod core;
pub mod device_choice;
pub mod exit_gate;
pub mod gpu;
#[cfg(target_os = "linux")]
pub mod host_gpus;
pub mod os;
pub mod recorder;
pub mod rt_platform;
pub mod seam;
pub mod swapchain;

pub use rt_platform::RtPlatform;
