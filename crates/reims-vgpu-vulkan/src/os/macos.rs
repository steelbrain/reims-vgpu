//! The OS services on macOS: the metal crate's, as they are. Core's threads, clocks, Mach VM and
//! autorelease pool are `MetalPlatform`'s own (`VulkanPlatform` holds them; its Mach VM then
//! reports the pages it changes to the runtime, [`crate::remap::VulkanVm`]), and an IOSurface is
//! the metal crate's `MetalSurface`, one CF reference whose retain count the guest sees.
//!
//! Only IOSurface creation names the platform in its trait (`IoSurfaces<P>` answers
//! `P::Surface`), and both platforms' surface is `MetalSurface`: [`VulkanSurfaces`] answers each
//! call with the metal crate's `MetalSurfaces`, unchanged, and records what each surface it made
//! over the guest's pages, or at an offset into its allocation, was made over ([`ClientPages`]).

mod drawables;

use std::collections::BTreeMap;
use std::ffi::{CStr, c_void};
use std::path::PathBuf;
use std::sync::Weak;

use metal2vulkan_runtime as rt;

use reims_vgpu_core::platform::Handle;
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::os::surface::{
    BackingSurfaceSpec, IoSurfaces, Surface, SurfaceSpec,
};
use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm as _, KernReturn};
use reims_vgpu_metal::MetalPlatform;
use reims_vgpu_metal::os::clock::MetalClock;
use reims_vgpu_metal::os::iosurface::{MetalSurface, MetalSurfaces};
use reims_vgpu_metal::os::threads::MetalThreads;
use reims_vgpu_metal::os::vm::MetalVm;

use super::{MadeOver, READ_ONLY_DIRECTION, page_runs};
use crate::platform::VulkanPlatform;

pub(crate) use drawables::Drawables;

/// The clocks: the metal crate's.
pub type OsClock = MetalClock;
/// The threads: the metal crate's.
pub type OsThreads = MetalThreads;
/// Mach VM: the metal crate's.
pub type OsVm = MetalVm;
/// An IOSurface: the metal crate's, one CF reference.
pub type OsSurface = MetalSurface;
/// IOSurface creation: [`VulkanSurfaces`].
pub type OsSurfaces = VulkanSurfaces;

/// Installs what the process's platform needs before it runs: the Objective-C autorelease pool
/// core's pools push, as `MetalPlatform::new` does. IOSurface and Foundation still autorelease
/// into it.
pub fn install_process_services() {
    reims_vgpu_metal::os::autorelease::install();
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
    let id = MetalPlatform::system_default_device().map(|d| d.caps().registry_id)?;
    Some(*ID.get_or_init(|| id))
}

/// `gpuCoreCount` of the system default `MTLDevice`, asked of Metal once, on the first call that
/// finds a device: the cores of the GPU [`default_registry_id`] names, which MoltenVK reports to
/// no Vulkan query. `None` where Metal has no device, or its device does not respond to
/// `gpuCoreCount` (the metal crate's caps).
pub fn default_gpu_core_count() -> Option<u64> {
    static COUNT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    if let Some(count) = COUNT.get() {
        return *count;
    }
    let count = MetalPlatform::system_default_device().map(|d| d.caps().gpu_core_count)?;
    *COUNT.get_or_init(|| count)
}

/// `featureProfile`, `halfFPConfig`, `singleFPConfig` and `doubleFPConfig` of the system default
/// `MTLDevice`, asked of Metal once: the answers of the GPU's own driver for the device
/// [`default_registry_id`] names, which no Vulkan query reports. `None` for each where Metal has
/// no device, or its device does not respond to it.
pub fn default_driver_sizes() -> [Option<u64>; 4] {
    static SIZES: std::sync::OnceLock<[Option<u64>; 4]> = std::sync::OnceLock::new();
    *SIZES.get_or_init(|| {
        MetalPlatform::system_default_device().map_or([None; 4], |d| {
            let z = d.identity_caps().sizes;
            [
                z.feature_profile,
                z.half_fp_config,
                z.single_fp_config,
                z.double_fp_config,
            ]
        })
    })
}

/// What a device holds of the guest's RAM while it lives: nothing, as Mach makes a view of any
/// memory by remapping its addresses (`mach_vm_remap`), with no file behind it.
#[derive(Debug)]
pub struct GuestMemoryLease;

/// The lease of the guest's RAM a device holds: nothing on Mach ([`GuestMemoryLease`]). A
/// device's dirty log is not used: the Mac's GPUs read the guest's RAM in place.
pub fn lease_guest_memory(
    _: impl IntoIterator<Item = super::GuestRamFile>,
    _: Option<std::sync::Arc<reims_vgpu_core::memory::dirty::DirtyLog>>,
) -> GuestMemoryLease {
    GuestMemoryLease
}

/// A graphics device's region in memory of the platform's own: none on macOS, where the region's
/// private anonymous memory is what Metal and MoltenVK import in place. Uninhabited.
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

/// A fresh range of `len` bytes of zeros that a view can be made of: on Mach, any allocation.
pub(crate) fn allocate_zeros(len: u64) -> Result<HostAddr, KernReturn> {
    allocate_aliasable(len)
}

/// Whether the runtime imports a surface's read-only alias through a writable twin of its files:
/// not on macOS, where it imports the bytes themselves ([`import_view`]).
pub(crate) const TWINS_READ_ONLY: bool = false;

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

    /// Whether the import is a writable twin of read-only pages: never on macOS.
    pub(crate) fn twinned(&self) -> bool {
        false
    }

    /// Whether the runtime is told the import's pages are writable: never on macOS, where it
    /// asks the kernel nothing of them either way.
    pub(crate) fn vouched(&self) -> bool {
        false
    }

    /// The runtime's deallocator of the import, and what keeps the import's memory mapped beside
    /// the bytes themselves: `inner`, as it is, and nothing, as the runtime imports the bytes.
    pub(crate) fn release(
        self,
        _original: std::ptr::NonNull<u8>,
        inner: Option<rt::Deallocator>,
    ) -> (Option<rt::Deallocator>, Option<rt::KeepAlive>) {
        (inner, None)
    }
}

/// The memory the runtime imports for `length` bytes at `at`: those bytes ([`ImportView`]),
/// read-only ones too, whether a twin is asked for (`twin`, a surface's read-only alias) or not.
pub(crate) fn import_view(at: std::ptr::NonNull<u8>, _length: u64, _twin: bool) -> ImportView {
    ImportView { at }
}

/// The runtime's host memory at `at`, the first byte of an import ([`ImportView::at`]): the bytes
/// alone, which MoltenVK imports as Metal does.
///
/// # Safety
///
/// `rt::HostMemory::new`'s contract, for the memory at `at`.
pub(crate) unsafe fn host_memory(at: std::ptr::NonNull<u8>, _logged: bool) -> rt::HostMemory {
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
/// own, so the runtime makes each buffer, heap or rebind over a copy of the pages, as Linux's
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
    logged: bool,
    _vouched: bool,
) -> rt::HostMemory {
    // SAFETY: the caller's.
    unsafe { host_memory(at, logged) }
}

/// Why this thread's last dma-buf source made none: never, on macOS, where imports carry none.
pub(crate) fn dma_buf_refusal() -> Option<String> {
    None
}

/// The address whose bytes the runtime's memory at `at` shows: `at`, on macOS.
pub(crate) fn shown_at(at: std::ptr::NonNull<u8>) -> std::ptr::NonNull<u8> {
    at
}

/// A fresh range of `len` bytes that a view can be made of: on Mach, any allocation.
pub(crate) fn allocate_aliasable(len: u64) -> Result<HostAddr, KernReturn> {
    MetalVm::new().allocate_anywhere(len)
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
    use reims_vgpu_core::platform::os::vm::VmProt;
    let vm = MetalVm::new();
    vm.protect_max(at, len, VmProt::READ | VmProt::WRITE)
        .is_ok()
}

/// Whether the page at `at` is mapped, for tests: a maximum protection of all three is refused
/// for an unmapped range with `KERN_INVALID_ADDRESS` (1), and changes nothing of a mapped one,
/// whose maximum is all three already or which refuses the raise.
#[cfg(test)]
pub(crate) fn mapped(at: HostAddr) -> bool {
    let vm = MetalVm::new();
    vm.protect_max(
        at,
        vm.page_size(),
        reims_vgpu_core::platform::os::vm::VmProt(7),
    ) != Err(KernReturn(1))
}

// IOSurface's plane and value queries, which the metal crate's surface does not answer.
#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    fn IOSurfaceGetBaseAddressOfPlane(buffer: *const c_void, plane_index: usize) -> *mut c_void;
    fn IOSurfaceGetBytesPerRowOfPlane(buffer: *const c_void, plane_index: usize) -> usize;
    fn IOSurfaceGetWidthOfPlane(buffer: *const c_void, plane_index: usize) -> usize;
    fn IOSurfaceGetHeightOfPlane(buffer: *const c_void, plane_index: usize) -> usize;
    fn IOSurfaceGetBytesPerElementOfPlane(buffer: *const c_void, plane_index: usize) -> usize;
    /// `CFTypeRef IOSurfaceCopyValue(IOSurfaceRef, CFStringRef)`: +1 or NULL.
    fn IOSurfaceCopyValue(buffer: *const c_void, key: *const c_void) -> *const c_void;
    fn IOSurfaceGetID(buffer: *const c_void) -> u32;
    /// `IOSurfaceRef IOSurfaceLookup(IOSurfaceID)`: +1 or NULL.
    fn IOSurfaceLookup(csid: u32) -> *const c_void;
}

// The CoreFoundation calls that read a surface's value as a string.
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithBytes(
        alloc: *const c_void,
        bytes: *const u8,
        num_bytes: isize,
        encoding: u32,
        is_external_representation: u8,
    ) -> *const c_void;
    fn CFGetTypeID(cf: *const c_void) -> usize;
    fn CFStringGetTypeID() -> usize;
    fn CFStringGetCString(
        string: *const c_void,
        buffer: *mut u8,
        buffer_size: isize,
        encoding: u32,
    ) -> u8;
    fn CFRelease(cf: *const c_void);
}

/// `kCFStringEncodingUTF8`.
const UTF8: u32 = 0x0800_0100;

/// The `IOSurfaceRef` of `surface`: its identity is the reference's address (`MetalSurface`'s
/// `Handle::id`), live while `surface` holds its reference.
fn surface_ref(surface: &MetalSurface) -> Option<*const c_void> {
    let address = usize::try_from(surface.id().0).ok()?;
    Some(std::ptr::with_exposed_provenance::<c_void>(address))
}

/// Plane `plane` of `surface`: its bytes' offset from the surface's base address
/// (`IOSurfaceGetBaseAddressOfPlane`) and its bytes per row; `None` for a plane the surface lacks.
pub(crate) fn plane_at(surface: &MetalSurface, plane: u64) -> Option<(u64, u64)> {
    if plane >= surface.plane_count() {
        return None;
    }
    let index = usize::try_from(plane).ok()?;
    let surface_ref = surface_ref(surface)?;
    // SAFETY: a live IOSurfaceRef, and a plane index under its plane count.
    let (base, bytes_per_row) = unsafe {
        (
            IOSurfaceGetBaseAddressOfPlane(surface_ref, index),
            IOSurfaceGetBytesPerRowOfPlane(surface_ref, index),
        )
    };
    let offset = base.addr().checked_sub(surface.base_address().addr())?;
    Some((
        u64::try_from(offset).ok()?,
        u64::try_from(bytes_per_row).ok()?,
    ))
}

/// The width, height and bytes per element of plane `plane` of `surface`, which has planes;
/// `None` for a plane it lacks.
pub(crate) fn plane_shape(surface: &MetalSurface, plane: u64) -> Option<(u64, u64, u64)> {
    if plane >= surface.plane_count() {
        return None;
    }
    let index = usize::try_from(plane).ok()?;
    let surface_ref = surface_ref(surface)?;
    // SAFETY: a live IOSurfaceRef, and a plane index under its plane count.
    let shape = unsafe {
        (
            IOSurfaceGetWidthOfPlane(surface_ref, index),
            IOSurfaceGetHeightOfPlane(surface_ref, index),
            IOSurfaceGetBytesPerElementOfPlane(surface_ref, index),
        )
    };
    Some((
        u64::try_from(shape.0).ok()?,
        u64::try_from(shape.1).ok()?,
        u64::try_from(shape.2).ok()?,
    ))
}

/// What keeps `surface`'s memory mapped while a device imports it at its base address, the
/// surface gone or not: nothing here. IOSurface's memory is the surface's own, and nothing but a
/// reference of the surface, which the guest counts, keeps it.
pub(crate) fn surface_memory_hold(_surface: &MetalSurface) -> impl Send + 'static {}

/// The alias of `surface`'s allocation its buffers lie over where it is the surface's own
/// mapping (`crate::os::surface_alias`): none on macOS, where IOSurface's mapping goes with the
/// surface's last release, which a buffer's alias may outlive.
pub(crate) fn own_alias(_surface: &MetalSurface) -> Option<crate::gpu::backing::Alias> {
    None
}

/// Whether `surface`'s base address lies in a view of the guest's pages the CPU may write through
/// as it writes the pages (`crate::os::lend_surface_bytes`): never here. IOSurface's own mapping
/// of several pages takes the CPU's writes under IOSurface's lock (reims-vgpu-metal's
/// `os/surface_bytes.rs`), which a lend here does not take, and is read-only for a surface made
/// read-only for its clients (`crate::os::gpu_read_only`).
pub(crate) fn base_views_pages(_surface: &MetalSurface) -> bool {
    false
}

/// The surface's key that names its YCbCr matrix (`kIOSurfaceYCbCrMatrix`).
pub(crate) const MATRIX_KEY: &str = "IOSurfaceYCbCrMatrix";

/// The YCbCr matrix `surface` names now (`kIOSurfaceYCbCrMatrix`, [`MATRIX_KEY`]): `None` for no
/// value there, a value that is no string the runtime's `Other`.
pub(crate) fn ycbcr_matrix(surface: &MetalSurface) -> Option<rt::SurfaceYCbCrMatrix> {
    match surface_value(surface, MATRIX_KEY) {
        SurfaceValue::Absent => None,
        SurfaceValue::Text(t) => Some(rt::SurfaceYCbCrMatrix::from_attachment(&t)),
        SurfaceValue::Other => Some(rt::SurfaceYCbCrMatrix::Other),
    }
}

/// A value of a surface's (`IOSurfaceCopyValue`), as far as a reader of strings sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SurfaceValue {
    /// The surface has no value at the key.
    Absent,
    /// A string.
    Text(String),
    /// A value of another type, or a string with no UTF-8 form.
    Other,
}

/// The value of `surface` at `key` (`IOSurfaceCopyValue`).
fn surface_value(surface: &MetalSurface, key: &str) -> SurfaceValue {
    let Some(surface_ref) = surface_ref(surface) else {
        return SurfaceValue::Other;
    };
    let Ok(length) = isize::try_from(key.len()) else {
        return SurfaceValue::Other;
    };
    // SAFETY: `key`'s bytes, of `length`, are live for the call, which copies them.
    let cf_key =
        unsafe { CFStringCreateWithBytes(std::ptr::null(), key.as_ptr(), length, UTF8, 0) };
    if cf_key.is_null() {
        return SurfaceValue::Other;
    }
    // SAFETY: a live IOSurfaceRef and a CFString; the value is +1, released below.
    let value = unsafe { IOSurfaceCopyValue(surface_ref, cf_key) };
    // SAFETY: the key's +1 reference, made above.
    unsafe { CFRelease(cf_key) };
    if value.is_null() {
        return SurfaceValue::Absent;
    }
    let mut text = [0u8; 256];
    // SAFETY: a live CF object; a string is read into `text`, whose length the call is given.
    let read = unsafe {
        CFGetTypeID(value) == CFStringGetTypeID()
            && CFStringGetCString(value, text.as_mut_ptr(), 256, UTF8) != 0
    };
    // SAFETY: the value's +1 reference, copied above.
    unsafe { CFRelease(value) };
    if !read {
        return SurfaceValue::Other;
    }
    let end = text.iter().position(|&b| b == 0).unwrap_or(text.len());
    match std::str::from_utf8(&text[..end]) {
        Ok(s) => SurfaceValue::Text(s.to_owned()),
        Err(_) => SurfaceValue::Other,
    }
}

/// What [`ClientPages`] recorded of `surface`, `change` given the record to change; `None` for a
/// surface it did not record.
pub(crate) fn recorded_mut<T>(
    surface: &MetalSurface,
    change: impl FnOnce(&mut MadeOver) -> T,
) -> Option<T> {
    let key = key(surface)?;
    CLIENT_PAGES.lock().entries.get_mut(&key).map(change)
}

/// The key of a surface's record, its `IOSurfaceID`: every object over the same IOSurface
/// answers it, each a reference of its own. Measured on the M3 Max: a lookup in this process
/// (`IOSurfaceLookupFromMachPort`, which an isolated task's surface is, and `IOSurfaceLookup`)
/// answers a new reference each time, of the same ID and base address.
type SurfaceKey = u32;

/// `surface`'s [`SurfaceKey`].
fn key(surface: &MetalSurface) -> Option<SurfaceKey> {
    let at = surface_ref(surface)?;
    // SAFETY: a live surface.
    Some(unsafe { IOSurfaceGetID(at) })
}

/// What each surface [`VulkanSurfaces::create`] made over the guest's pages, or at an offset
/// into its allocation, was made over, by its `IOSurfaceID` ([`SurfaceKey`]). An ID names one
/// live surface, so a surface made here drops the entry of one gone that had its ID
/// ([`ClientPages::forget`]), and entries whose ID no surface holds are dropped as it grows.
/// Each record a command's objects read at its encode is found by its ID.
struct ClientPages {
    entries: BTreeMap<SurfaceKey, MadeOver>,
    /// The length at which the next record drops the entries of surfaces gone.
    prune_at: usize,
}

static CLIENT_PAGES: reims_vgpu_core::sync::Mutex<ClientPages> =
    reims_vgpu_core::sync::Mutex::new(ClientPages {
        entries: BTreeMap::new(),
        prune_at: 64,
    });

impl ClientPages {
    /// Drops the entry of `surface`'s ID, one of a surface gone where `surface` was just made.
    fn forget(&mut self, surface: &MetalSurface) {
        if let Some(key) = key(surface) {
            self.entries.remove(&key);
        }
    }

    /// Records `surface`, made over `pages` (host first byte and length of each), its bytes
    /// `offset` bytes into its allocation, the GPU's writes through it reaching them unless
    /// `gpu_read_only`.
    fn record(
        &mut self,
        surface: &MetalSurface,
        pages: &[(u64, u64)],
        offset: u64,
        gpu_read_only: bool,
    ) {
        let Some(key) = key(surface) else {
            return;
        };
        let runs = page_runs(pages.iter().copied());
        self.entries.remove(&key);
        if self.entries.len() >= self.prune_at {
            self.entries.retain(|&id, _| alive(id));
            self.prune_at = (self.entries.len() * 2).max(64);
        }
        self.entries.insert(
            key,
            MadeOver {
                pages: runs,
                offset,
                gpu_read_only,
                alias: Weak::new(),
                lent: false,
            },
        );
    }
}

/// Whether a surface of `IOSurfaceID` `id` lives: one made here, or one another process made
/// since, whose ID's entry goes when a surface made here takes the ID.
fn alive(id: u32) -> bool {
    // SAFETY: a lookup by ID, which answers +1 or NULL.
    let found = unsafe { IOSurfaceLookup(id) };
    if found.is_null() {
        return false;
    }
    // SAFETY: the lookup's +1 reference.
    unsafe { CFRelease(found) };
    true
}

/// `IOSurfaceCreate` and `[[IOSurface alloc] initWithProperties:]`: the metal crate's. A surface
/// made over the guest's pages, or at an offset into its allocation, is recorded with them
/// ([`crate::os::surface_memory`], [`crate::os::surface_offset`]), and with whether the GPU's
/// writes through it reach them ([`crate::os::gpu_read_only`]); every lookup of it answers the
/// record too. Each surface made here drops the record of a surface gone that had its ID.
#[derive(Debug, Default)]
pub struct VulkanSurfaces(MetalSurfaces);

impl VulkanSurfaces {
    /// The process's IoSurfaces.
    pub fn new() -> VulkanSurfaces {
        VulkanSurfaces(MetalSurfaces::new())
    }
}

impl IoSurfaces<VulkanPlatform> for VulkanSurfaces {
    fn create(&self, spec: &SurfaceSpec) -> Option<MetalSurface> {
        let made = IoSurfaces::<MetalPlatform>::create(&self.0, spec)?;
        let mut recorded = CLIENT_PAGES.lock();
        recorded.forget(&made);
        if spec.client_address_ranges.is_some() || spec.offset != 0 {
            let pages: Vec<(u64, u64)> = (spec.client_address_ranges.iter().flatten())
                .map(|r| (r.physical_address, r.physical_length))
                .collect();
            let gpu_read_only = spec.client_address_ranges.is_some()
                && spec.client_direction == Some(READ_ONLY_DIRECTION);
            recorded.record(&made, &pages, spec.offset, gpu_read_only);
        }
        drop(recorded);
        Some(made)
    }

    fn create_backing(&self, spec: &BackingSurfaceSpec) -> Option<MetalSurface> {
        let made = IoSurfaces::<MetalPlatform>::create_backing(&self.0, spec)?;
        CLIENT_PAGES.lock().forget(&made);
        Some(made)
    }

    fn describe(&self, spec: &SurfaceSpec) -> String {
        IoSurfaces::<MetalPlatform>::describe(&self.0, spec)
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::num::ToU64;

    use super::*;
    use crate::os::tests::{bgra_surface, bgra_surface_over_at, free};
    use crate::os::{surface_bytes_memory, surface_memory, surface_offset};

    /// A surface over the guest's pages at an offset into its allocation (`kIOSurfaceOffset`)
    /// has its base address that offset into its allocation, which is another mapping of the
    /// pages: a write through its base address shows in the pages at the offset, and its
    /// memory and its bytes' are found there
    #[test]
    fn an_offset_surface_is_at_its_offset_into_its_pages() {
        use reims_vgpu_core::platform::os::vm::HostVm as _;
        for offset in [0x40, 0x1000, 0x4000] {
            let (s, pages, length) =
                crate::gpu::ycbcr::measure::surface_420_over_at(b"x420", (32, 16), offset);
            let (first, base) = (
                pages.addr().to_u64(),
                s.base_address().as_ptr::<u8>().addr().to_u64(),
            );
            assert_eq!(s.alloc_size(), length, "{offset:#x}");
            // SAFETY: the surface's first byte, which nothing else uses.
            unsafe { s.base_address().as_ptr::<u8>().write(0xab) };
            // SAFETY: the test's pages, mapped until they are freed below.
            let at = unsafe {
                pages
                    .as_ptr::<u8>()
                    .add(usize::try_from(offset).unwrap())
                    .read()
            };
            assert_eq!(
                at, 0xab,
                "{offset:#x}: the base address is the offset into the pages"
            );
            let memory = surface_memory(&s);
            assert_eq!(
                memory[0],
                base - offset..base - offset + length,
                "{offset:#x}"
            );
            let (mapped, read) = (first..first + length, 0x10..0x20);
            assert_eq!(memory[1..], [mapped], "{offset:#x}");
            assert_eq!(
                surface_bytes_memory(&s, &[read]),
                [
                    base + 0x10..base + 0x20,
                    first + offset + 0x10..first + offset + 0x20
                ],
                "{offset:#x}"
            );
            drop(s);
            reims_vgpu_metal::os::vm::MetalVm::new().deallocate(pages, length);
        }
    }

    /// The records of live surfaces outlast the dropping of those of surfaces gone, which runs
    /// as the records grow past 64
    #[test]
    fn records_of_live_surfaces_outlast_the_pruning() {
        let made: Vec<_> = (0..80).map(|_| bgra_surface_over_at(4, 4, 0x40)).collect();
        let offsets: Vec<u64> = made.iter().map(|(s, ..)| surface_offset(s)).collect();
        assert_eq!(offsets, [0x40; 80]);
        for (s, pages, length) in made {
            drop(s);
            free(pages, length);
        }
    }

    /// A surface made with the ID of one gone, which IOSurface hands out again at once, answers
    /// no record of that one's, made by `IOSurfaceCreate` or by `initWithProperties:`. Another
    /// process's surface can take the freed ID first: with 8 tries, 2 of 5 runs alone found no
    /// ID handed out again on the M3 Max (macOS 27.0.1), and 0 of 40 with 64
    #[test]
    fn a_surface_taking_the_id_of_one_gone_takes_none_of_its_record() {
        use reims_vgpu_core::platform::os::surface::{BackingPlaneSpec, BackingSurfaceSpec};
        let backing = BackingSurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            planes: vec![BackingPlaneSpec {
                width: 16,
                height: 16,
                bytes_per_element: 4,
                bytes_per_row: 64,
                base: 0,
                offset: 0,
            }],
            width: 16,
            height: 16,
        };
        let makers: [&dyn Fn() -> MetalSurface; 2] = [&|| bgra_surface(16, 16), &|| {
            VulkanSurfaces::new().create_backing(&backing).unwrap()
        }];
        for make in makers {
            let reused = (0..64).any(|_| {
                let (gone, pages, length) = bgra_surface_over_at(16, 16, 0x100);
                let id = key(&gone);
                drop(gone);
                free(pages, length);
                let made = make();
                let same = key(&made) == id;
                assert_eq!(surface_offset(&made), 0);
                assert!(surface_memory(&made).len() <= 1);
                same
            });
            assert!(reused, "IOSurface handed out no ID again");
        }
    }

    /// A host-only surface is the metal crate's: one CF reference, the spec's size
    #[test]
    fn surfaces_are_the_metal_crates() {
        let spec = SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            alloc_size: 16 * 64,
            width: 16,
            height: 16,
            bytes_per_element: 4,
            bytes_per_row: 64,
            element_width: 1,
            element_height: 1,
            ..SurfaceSpec::default()
        };
        let s = VulkanSurfaces::new().create(&spec).unwrap();
        assert_eq!((s.width(), s.height(), s.bytes_per_row()), (16, 16, 64));
        assert_eq!(s.retain_count(), 1);
        assert!(!VulkanSurfaces::new().describe(&spec).is_empty());
    }
}
