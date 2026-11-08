//! The C API (docs/api.md section 2): a thin FFI layer over `reims-vgpu`'s Rust API, from which
//! cbindgen writes `include/reims_vgpu.h`: the graphics device (`rvg_graphics_device_*`), its
//! window and its cursor. It converts and forwards; it decides nothing. Every callback it builds
//! calls the host's function synchronously, on the thread the library calls it on, so the order
//! and the threads of every call to the host are the library's.
//!
//! Conventions, which the header's preamble repeats:
//! - every function is `extern "C"`, and runs its body inside [`ffi::export`]: nothing unwinds
//!   out of it, and a panic under it ends the process before anything unwinds ([`ffi`]);
//! - handles are opaque ([`handle`]); a create writes one only on `RVG_OK`, a release of NULL
//!   does nothing, and any other NULL handle is `RVG_ERR_INVALID_ARGUMENT` where the function
//!   answers an [`RvgStatus`] and ends the process where it does not;
//! - every struct the host writes starts with `struct_size` ([`sized`]);
//! - every callback table has one `ctx` and one `destroy` ([`table`]).
//!
//! The exports are made over [`Plat`]: on macOS the product's `MetalPlatform`, or with the
//! `vulkan` feature `VulkanPlatform`; on Linux `VulkanPlatform`, with or without the feature,
//! since Metal is macOS's; and core's testkit platform in this crate's unit tests, which then run
//! the product's code.

// On the Vulkan platform, a debug build's proof that the runtime's types are `Send` and `Sync`
// nests deeper than rustc's default limit of 128 (E0275).
#![recursion_limit = "256"]

// cbindgen writes the header's items in the order it parses them: these modules' order.
pub mod status;

pub mod cursor;
pub mod memory;
pub mod window;

pub mod graphics;

mod ffi;
mod handle;
mod sized;
mod table;

pub use status::RvgStatus;

/// The library's allocator on the targets that ship, x86-64 Linux and arm64 macOS: jemalloc, for
/// every Rust allocation in the library, metal2vulkan's runtime's among them. A command queue's
/// threads allocate memory that the runtime's threads free, and glibc's allocator refills its
/// arena for each such block on the allocating thread (measured on ms01's Iris Xe with jemalloc
/// preloaded into all of QEMU: a drag at 36 frames/s under glibc, 42 under jemalloc). Its
/// symbols are prefixed (`_rjem_`), so the host's C code and the drivers keep the system's
/// `malloc`, and no block crosses between the two: the host is handed no block, and the bytes
/// the library gives Metal to free are `malloc`'s own.
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// The platform every handle is made on: the product's, on macOS.
#[cfg(not(any(test, feature = "vulkan", target_os = "linux")))]
pub(crate) type Plat = reims_vgpu::MetalPlatform;
/// The platform every handle is made on, on Linux and with the `vulkan` feature: the Vulkan
/// platform, over metal2vulkan's runtime (architecture 2.2).
#[cfg(all(not(test), any(feature = "vulkan", target_os = "linux")))]
pub(crate) type Plat = reims_vgpu::VulkanPlatform;
/// The platform every handle is made on: core's testkit platform in this crate's unit tests.
#[cfg(test)]
pub(crate) type Plat = reims_vgpu_core::testkit::FakePlatform;

/// Ends the process: `call` was given a NULL handle it cannot answer without one.
pub(crate) fn null_handle(call: &'static str) -> ! {
    use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal};
    fatal(DeathId::Platform {
        kind: DeathKind::NullDeref,
        call,
    })
}
