//! The C API (docs/api.md section 2): a thin FFI layer over `reims-vgpu`'s Rust API, from which
//! cbindgen writes `include/reims_vgpu.h`. It converts and forwards; it decides nothing. Every
//! callback it builds calls the host's function synchronously, on the thread core calls it on,
//! so the order and the threads of every call to the host are core's.
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
//! The exports are made over [`Plat`]: the product's `MetalPlatform`, or with the `vulkan`
//! feature the proof of concept's `VulkanPlatform`, and core's testkit platform in this crate's
//! unit tests, which then run the product's code. On Linux the crate builds only with the
//! `vulkan` feature, since Metal is macOS's.

// With the `vulkan` feature, a debug build's proof that the runtime's types are `Send` and `Sync`
// nests deeper than rustc's default limit of 128 (E0275).
#![recursion_limit = "256"]

// cbindgen writes the header's functions in the order it parses them: these modules' order,
// each in a group of its own so that rustfmt keeps it.
pub mod status;

pub mod log;

pub mod device;
pub mod display;
pub mod frame;
pub mod iosurface;
pub mod trace;
pub mod video;
pub mod window;

// The scaler device after the rest, so that the functions before it keep their places.
pub mod scaler;

mod ffi;
mod handle;
mod sized;
mod table;

pub use status::RvgStatus;

/// The library's allocator on the targets that ship, x86-64 Linux and arm64 macOS: jemalloc, for
/// every Rust allocation in the library, metal2vulkan's runtime's among them. FIFO 1's commands
/// allocate memory that the runtime's threads free, and glibc's allocator refills its arena for
/// each such block on the allocating thread (measured on ms01's Iris Xe with jemalloc preloaded
/// into all of QEMU, not this build: a drag at 36 frames/s under glibc, 42 under jemalloc). Its
/// symbols are prefixed (`_rjem_`), so the host's C code and the drivers keep the system's
/// `malloc`, and no block crosses between the two: the one block the host is handed,
/// [`device::rvg_bytes_free`]'s, goes back through that function, and the bytes the library gives
/// Metal to free are `malloc`'s own.
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(
    any(target_os = "linux", target_os = "windows"),
    not(feature = "vulkan")
))]
compile_error!(
    "on Linux and Windows, reims-vgpu-c builds only with the `vulkan` feature (VulkanPlatform)"
);

/// The platform every handle is made on.
#[cfg(not(any(test, feature = "vulkan")))]
pub(crate) type Plat = reims_vgpu::MetalPlatform;
/// The platform every handle is made on, with the `vulkan` feature: the proof of concept over
/// metal2vulkan's runtime (architecture 2.2).
#[cfg(all(not(test), feature = "vulkan"))]
pub(crate) type Plat = reims_vgpu::VulkanPlatform;
/// The platform every handle is made on: core's testkit platform in this crate's unit tests.
#[cfg(test)]
pub(crate) type Plat = reims_vgpu_core::testkit::FakePlatform;

/// The most display ports a device has.
pub const RVG_MAX_DISPLAY_PORTS: u32 = 8;

/// The size of the device's MMIO region: the PCI BAR, or vmapple's first region.
pub const RVG_DEVICE_MMIO_SIZE: u64 = 16384;

/// The PCI vendor ID of the device.
pub const RVG_PCI_VENDOR_ID: u16 = 4203;

/// The PCI device ID of the device.
pub const RVG_PCI_DEVICE_ID: u16 = 61166;

/// The BAR that holds the device's MMIO region.
pub const RVG_PCI_BAR_MMIO: u32 = 0;

/// The MSI vectors the device uses.
pub const RVG_PCI_MSI_VECTORS: u32 = 1;

const _: () = assert!(RVG_MAX_DISPLAY_PORTS == reims_vgpu::MAX_DISPLAY_PORTS);
const _: () = assert!(RVG_DEVICE_MMIO_SIZE == reims_vgpu::DEVICE_MMIO_SIZE);
const _: () = assert!(RVG_PCI_VENDOR_ID == reims_vgpu::PCI_VENDOR_ID);
const _: () = assert!(RVG_PCI_DEVICE_ID == reims_vgpu::PCI_DEVICE_ID);

/// Ends the process: `call` was given a NULL handle it cannot answer without one.
pub(crate) fn null_handle(call: &'static str) -> ! {
    use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal};
    fatal(DeathId::Platform {
        kind: DeathKind::NullDeref,
        call,
    })
}

#[cfg(test)]
mod testkit;
