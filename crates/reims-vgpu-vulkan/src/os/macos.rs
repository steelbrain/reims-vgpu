//! The OS services on macOS: core's threads ([`threads`]) and the scheduling hint of the
//! graphics device's walk threads ([`hints`]), clocks ([`clock`]), Mach VM ([`vm`]) and the
//! Objective-C autorelease pool ([`autorelease`]); and the message layer the Objective-C sends go
//! through ([`msg`]), which the window's AppKit side uses too. `VulkanPlatform` holds them; its
//! Mach VM reports the pages it changes to the runtime ([`crate::remap::VulkanVm`]).

mod autorelease;
mod clock;
mod drawables;
mod hints;
pub mod msg;
mod threads;
mod vm;

use std::ffi::CStr;
use std::path::PathBuf;

use metal2vulkan_runtime as rt;
use objc2::rc::Retained;
use objc2::runtime::{NSObjectProtocol as _, ProtocolObject, Sel};
use objc2::{msg_send, sel};
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

use crate::core::platform::os::vm::{HostAddr, HostVm as _, KernReturn};

use self::clock::MacClock;
use self::threads::MacThreads;
use self::vm::MacVm;

pub(crate) use drawables::Drawables;

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

/// The device extensions a window's swapchain needs besides `VK_KHR_swapchain`:
/// `VK_EXT_metal_objects`, whose export of a swapchain image's Metal texture fetches its drawable
/// on the window's render thread ([`Drawables`]). The runtime enables it only where it is asked
/// for, as it enables any other device extension of the options.
pub(crate) const WINDOW_DEVICE_EXTENSIONS: [&CStr; 1] = [ash::ext::metal_objects::NAME];

/// The Vulkan loader, searched for as the runtime searches for it (its `default_loaders`, which
/// it does not export): by name, where `dlopen` finds it, then in `$VULKAN_SDK/lib`,
/// `/opt/homebrew/lib` and `/usr/local/lib`, where Homebrew's `vulkan-loader` lies beside its
/// `molten-vk`, out of `dlopen`'s default search. The last refusal where none opens.
pub(crate) fn open_loader() -> Result<ash::Entry, ash::LoadingError> {
    let sdk = std::env::var_os("VULKAN_SDK").map(|sdk| PathBuf::from(sdk).join("lib"));
    let directories = sdk
        .into_iter()
        .chain(["/opt/homebrew/lib", "/usr/local/lib"].map(PathBuf::from))
        .map(|directory| directory.join("libvulkan.1.dylib"));
    // SAFETY: opening the loader runs its initialization, and it is kept open while the entry
    // lives; so for each below.
    let mut opened = unsafe { ash::Entry::load_from("libvulkan.dylib") };
    for path in std::iter::once(PathBuf::from("libvulkan.1.dylib")).chain(directories) {
        if opened.is_ok() {
            break;
        }
        // SAFETY: as above.
        opened = unsafe { ash::Entry::load_from(&path) };
    }
    opened
}

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

/// `gpuCoreCount` of the system default `MTLDevice`, asked of Metal once, on the first call that
/// finds a device: the cores of the GPU [`default_registry_id`] names, which MoltenVK reports to
/// no Vulkan query. `None` where Metal has no device, or its device does not respond to
/// `gpuCoreCount`.
pub fn default_gpu_core_count() -> Option<u64> {
    static COUNT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    if let Some(count) = COUNT.get() {
        return *count;
    }
    let d = default_device()?;
    // SAFETY: the runtime answers `gpuCoreCount` as `unsigned int` (`I@:`).
    let count = device_spi(&d, sel!(gpuCoreCount), "gpuCoreCount", || unsafe {
        let n: u32 = msg_send![&*d, gpuCoreCount];
        u64::from(n)
    });
    *COUNT.get_or_init(|| count)
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

/// A graphics device's region in a mapping of a file in memory: none on macOS, where the region is
/// private memory of the library's ([`super::PrivateMemory`]), which Metal and MoltenVK import in
/// place. Uninhabited.
#[derive(Debug)]
pub enum RegionMemory {}

impl RegionMemory {
    /// None on macOS ([`RegionMemory`]).
    pub fn new(_len: u64) -> Option<RegionMemory> {
        None
    }

    /// Its first byte: never asked, as there is none.
    pub fn base(&self) -> *mut u8 {
        match *self {}
    }
}

/// Whether the runtime makes read-only memory, and memory made with `MTLResourceOptions` bit 24,
/// over a copy of it whose GPU writes it drops: not on macOS, where MoltenVK hands Metal the
/// pages, or a read-only view of them, and Metal's GPU answers its writes there itself, as the
/// runtime's device answers yes to `drops_gpu_writes_to_read_only_memory`.
pub(crate) const COPIES_READ_ONLY: bool = false;

/// The memory the runtime imports for some bytes: on macOS those bytes themselves, which MoltenVK
/// hands to Metal, which reads read-only pages.
#[derive(Debug)]
pub(crate) struct ImportView {
    at: std::ptr::NonNull<u8>,
}

// SAFETY: an address only, which nothing here reads through: it is handed to the runtime.
unsafe impl Send for ImportView {}

// SAFETY: as for `Send`: a shared view only answers where it is.
unsafe impl Sync for ImportView {}

impl ImportView {
    /// The first byte the runtime imports.
    pub(crate) fn at(&self) -> std::ptr::NonNull<u8> {
        self.at
    }

    /// Whether the runtime makes the object over a copy of read-only pages whose GPU writes it
    /// drops: never on macOS, where read-only pages stay read-only to the GPU.
    pub(crate) fn read_only(&self) -> bool {
        false
    }

    /// Whether the runtime is told the import's pages are writable: never on macOS, where it
    /// asks the kernel nothing of them either way.
    pub(crate) fn vouched(&self) -> bool {
        false
    }

    /// The word that the runtime's import was made: nothing on macOS, where no view of the bytes
    /// is folded.
    pub(crate) fn made(&self) -> ImportMade {
        ImportMade
    }

    /// The runtime's deallocator of the import, and what keeps the import's memory mapped beside
    /// the bytes themselves: `inner`, as it is, and nothing, as the runtime imports the bytes.
    pub(crate) fn release(
        self,
        inner: Option<rt::Deallocator>,
    ) -> (Option<rt::Deallocator>, Option<rt::KeepAlive>) {
        (inner, None)
    }
}

/// The word that an import was made ([`ImportView::made`]): nothing on macOS.
#[derive(Debug)]
pub(crate) struct ImportMade;

impl ImportMade {
    /// The runtime's call that imports the bytes on `device` returned: nothing on macOS.
    pub(crate) fn made(self, _device: &rt::Device) {}
}

/// The memory the runtime imports for `length` bytes at `at`: those bytes ([`ImportView`]),
/// read-only ones too.
pub(crate) fn import_view(at: std::ptr::NonNull<u8>, _length: u64) -> ImportView {
    ImportView { at }
}

/// [`import_view`] of a heap's bytes: the same on macOS.
pub(crate) fn import_heap_view(at: std::ptr::NonNull<u8>, length: u64) -> ImportView {
    import_view(at, length)
}

/// The runtime's host memory at `at`, the first byte of an import ([`ImportView::at`]): the bytes
/// alone, which MoltenVK imports as Metal does.
///
/// # Safety
///
/// `rt::HostMemory::new`'s contract, for the memory at `at`.
pub(crate) unsafe fn host_memory(at: std::ptr::NonNull<u8>) -> rt::HostMemory {
    // SAFETY: the caller's.
    let memory = unsafe { rt::HostMemory::new(at) };
    #[cfg(test)]
    if REFUSING_IMPORTS.with(std::cell::Cell::get) {
        return memory.with_import_gate(std::sync::Arc::new(|_| false));
    }
    memory
}

#[cfg(test)]
thread_local! {
    /// Whether this thread's imports carry a gate that refuses them ([`refusing_imports`]).
    static REFUSING_IMPORTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `call` with every import [`host_memory`] makes on this thread refused at a gate of its
/// own, so the runtime makes each buffer or heap over a copy of the pages, as Linux's
/// views' gate has it past the headroom of their budget (`os/linux/import.rs`): the copies'
/// coherence, on a host whose imports never fail.
#[cfg(test)]
pub(crate) fn refusing_imports<T>(call: impl FnOnce() -> T) -> T {
    /// Puts back the outer setting on drop, an unwinding call's included.
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            REFUSING_IMPORTS.with(|r| r.set(self.0));
        }
    }
    let _restore = Restore(REFUSING_IMPORTS.with(|r| r.replace(true)));
    call()
}

/// [`host_memory`]: on macOS the runtime asks nothing of the pages' permissions, so a vouch
/// changes nothing.
///
/// # Safety
///
/// [`host_memory`]'s.
pub(crate) unsafe fn host_memory_vouched(
    at: std::ptr::NonNull<u8>,
    _vouched: bool,
) -> rt::HostMemory {
    // SAFETY: the caller's.
    unsafe { host_memory(at) }
}

/// Why this thread's last dma-buf source made none: never, on macOS, where imports carry none.
pub(crate) fn dma_buf_refusal() -> Option<String> {
    None
}

/// A fresh range of `len` bytes that a view can be made of: on Mach, any allocation.
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
