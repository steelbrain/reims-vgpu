//! The OS services on macOS: core's threads ([`threads`]) and the scheduling hint of the
//! graphics device's walk threads ([`hints`]), clocks ([`clock`]), Mach VM ([`vm`]) and the
//! Objective-C autorelease pool ([`autorelease`]); and the message layer the Objective-C sends go
//! through ([`msg`]), which the window's AppKit side uses too; and the rt platform's instance on
//! the Vulkan driver the platform runs on here ([`rt_driver`]).

mod autorelease;
mod clock;
mod hints;
pub mod msg;
mod rt_driver;
mod threads;
mod vm;

use std::ffi::CStr;

use objc2::rc::Retained;
use objc2::runtime::{NSObjectProtocol as _, ProtocolObject, Sel};
use objc2::{msg_send, sel};
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

use crate::core::platform::os::vm::{HostAddr, HostVm as _, KernReturn};

use self::clock::MacClock;
use self::threads::MacThreads;
use self::vm::MacVm;

pub(crate) use rt_driver::rt_instance;

/// The clocks: [`clock`]'s.
pub type OsClock = MacClock;
/// The threads: [`threads`]'.
pub type OsThreads = MacThreads;
/// Mach VM: [`vm`]'s.
pub type OsVm = MacVm;

/// Installs what the process's platform needs before it runs: the Objective-C autorelease pool
/// core's pools push, which MoltenVK and Foundation autorelease into.
pub fn install_process_services() {
    autorelease::install();
}

/// The instance's surface extension a display's window presents over, besides `VK_KHR_surface`,
/// asked for where the loader lists it (`crate::os`'s `wsi`): `VK_EXT_metal_surface`, over the
/// window's `CAMetalLayer`, AppKit's view's or SDL's Metal view's alike.
pub(crate) const WINDOW_SURFACE_EXTENSIONS: [&CStr; 1] = [ash::ext::metal_surface::NAME];

/// A compile-pool make ended after `took` of wall time: nothing on macOS. Handing back the heap a
/// heavy make left free is glibc's on Linux (`os/linux/heap_trim.rs`); macOS's allocator differs,
/// and is out of its scope.
pub fn make_ended(_took: std::time::Duration) {}

/// `registryID` of the system default `MTLDevice` (`MTLCreateSystemDefaultDevice()`), asked
/// of Metal once, on the first call that finds a device: `None` where Metal has none. It is
/// the registry ID of the device MoltenVK runs the runtime's default device on where the Mac
/// has one GPU, as Metal's default and the runtime's are then the same GPU. (MoltenVK names
/// its `MTLDevice` only through `VK_EXT_metal_objects`, which the runtime asks no device of.)
pub fn default_registry_id() -> Option<u64> {
    static ID: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    if let Some(id) = ID.get() {
        return Some(*id);
    }
    let id = default_device().map(|d| msg::get("registryID", || d.registryID()))?;
    Some(*ID.get_or_init(|| id))
}

/// `MTLCreateSystemDefaultDevice()`: Metal's system default device, `None` where Metal has none.
fn default_device() -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    msg::get(
        "MTLCreateSystemDefaultDevice()",
        MTLCreateSystemDefaultDevice,
    )
}

/// The value of `getter`, an `MTLDeviceSPI` getter of `d`, sent only where `d` responds to it,
/// else `None`.
fn device_spi<R>(
    d: &ProtocolObject<dyn MTLDevice>,
    getter: Sel,
    name: &'static str,
    get: impl FnOnce() -> R,
) -> Option<R> {
    msg::get("respondsToSelector:", || d.respondsToSelector(getter)).then(|| msg::get(name, get))
}

/// `featureProfile`, `halfFPConfig`, `singleFPConfig` and `doubleFPConfig` of the system default
/// `MTLDevice`, asked of Metal once: the answers of the GPU's own driver for the device
/// [`default_registry_id`] names, which no Vulkan query reports. `None` for each where Metal has
/// no device, or its device does not respond to it.
pub fn default_driver_sizes() -> [Option<u64>; 4] {
    static SIZES: std::sync::OnceLock<[Option<u64>; 4]> = std::sync::OnceLock::new();
    *SIZES.get_or_init(|| {
        default_device().map_or([None; 4], |d| {
            // SAFETY: each is a `Q@:` getter of `MTLDeviceSPI`, sent only where `d` responds.
            [
                device_spi(&d, sel!(featureProfile), "featureProfile", || unsafe {
                    msg_send![&*d, featureProfile]
                }),
                device_spi(&d, sel!(halfFPConfig), "halfFPConfig", || unsafe {
                    msg_send![&*d, halfFPConfig]
                }),
                device_spi(&d, sel!(singleFPConfig), "singleFPConfig", || unsafe {
                    msg_send![&*d, singleFPConfig]
                }),
                device_spi(&d, sel!(doubleFPConfig), "doubleFPConfig", || unsafe {
                    msg_send![&*d, doubleFPConfig]
                }),
            ]
        })
    })
}

/// What a device holds of the guest's RAM while it lives: nothing, as Mach makes a view of any
/// memory by remapping its addresses (`mach_vm_remap`), with no file behind it.
#[derive(Debug)]
pub struct GuestMemoryLease;

/// The lease of the guest's RAM a device holds: nothing on Mach ([`GuestMemoryLease`]).
pub fn lease_guest_memory(_: impl IntoIterator<Item = super::GuestRamFile>) -> GuestMemoryLease {
    GuestMemoryLease
}

/// A fresh range of `len` bytes that a view can be made of: on Mach, any allocation.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn allocate_aliasable(len: u64) -> Result<HostAddr, KernReturn> {
    MacVm::new().allocate_anywhere(len)
}

/// The host's page, which the tests lay their pages out by: Mach's 16 KiB on arm64, 4 KiB on
/// x86-64.
#[cfg(test)]
pub(crate) const PAGE: u64 = if cfg!(target_arch = "aarch64") {
    0x4000
} else {
    0x1000
};

/// Whether `len` bytes from `at` could be made writable, for tests: Mach refuses to raise a
/// maximum protection lowered to read.
#[cfg(test)]
pub(crate) fn writable(at: HostAddr, len: u64) -> bool {
    use crate::core::platform::os::vm::VmProt;
    let vm = MacVm::new();
    vm.protect_max(at, len, VmProt::READ | VmProt::WRITE)
        .is_ok()
}

/// Whether the page at `at` is mapped, for tests: a maximum protection of all three is refused
/// for an unmapped range with `KERN_INVALID_ADDRESS` (1), and changes nothing of a mapped one,
/// whose maximum is all three already or which refuses the raise.
#[cfg(test)]
pub(crate) fn mapped(at: HostAddr) -> bool {
    let vm = MacVm::new();
    vm.protect_max(at, vm.page_size(), crate::core::platform::os::vm::VmProt(7))
        != Err(KernReturn(1))
}
