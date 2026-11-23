//! What the API needs of a platform beyond core's [`Platform`] ([`ApiPlatform`]), and the state
//! it keeps once per platform instance, which in the product is once per process: the display
//! thread.
//!
//! **The display thread.** A graphics device queues its host's cursor callbacks onto one core
//! [`SerialQueue`], made at the process's first device whose host takes them (`graphics`'
//! `HostCursor`), so every display callback of the process runs one at a time in the order it
//! was queued. The queue's worker exits after [`reims_vgpu_vulkan::core::rt::queue::IDLE_EXIT`] idle and
//! the next job starts another. The library never waits for it, and a queue's drop joins nothing.
//!
//! **By platform.** The rt platform's `ApiPlatform` is `platform/rt.rs`; on Linux the dma-bufs
//! of host memory mapped from files, which a device imports, are `platform/dma_buf.rs`.

use std::any::Any;
use std::sync::{Arc, OnceLock, Weak};

use reims_vgpu_vulkan::core::host::{HostPtr, HostQueue};
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::values::{PagesAccess, TextureDesc};
use reims_vgpu_vulkan::core::raise::OnFault;
use reims_vgpu_vulkan::core::rt::{Executor, Job, SerialQueue};
use reims_vgpu_vulkan::core::sync::Mutex;

use crate::memory::MemoryMap;

/// A platform's import of a pages block's view as a block it knows
/// ([`ApiPlatform::view_imports`]): given the device, the view's first byte, on a host page, its
/// length, how the block's process maps its pages (read-only: the import then drops the GPU's
/// writes to them, as Metal's GPU does; and unchanging: it may then read them once), and what
/// runs once the platform freed the import (which holds the view's mapping until then), the
/// import's GPU address, at the view's first byte, and what forgets the block, which runs as the
/// last buffer over the view at that address goes. `None` where the platform refuses it, counted.
pub type ViewImports<P> = Arc<
    dyn Fn(
            &<P as Platform>::Device,
            HostPtr,
            u64,
            PagesAccess,
            Box<dyn FnOnce() + Send>,
        ) -> Option<(u64, Box<dyn FnOnce() + Send>)>
        + Send
        + Sync,
>;

/// What the API asks of a platform beyond core's: the process's instance, its GPU device, a
/// texture's pixels, what it holds of the guest's RAM, a graphics device's region and its video.
pub trait ApiPlatform: Platform {
    /// Whether a graphics device on the platform keeps the per-shape form of the textures over
    /// its IOSurface device caches (`graphics::decode::surface_shapes`): each a view of its
    /// shape's image ([`ApiPlatform::shape_image`]) rather than a texture over the cache's buffer,
    /// and every surface cached. False by default.
    const SURFACE_SHAPES: bool = false;

    /// Whether the platform places a shape's image in memory of its own whose bytes are as they
    /// are ([`ApiPlatform::placed_shape_image`]): a graphics device on it asks for one where a new
    /// shape takes its first fill with its make (`graphics::decode::surface_cache`'s `shapes`,
    /// "Placed images"). False by default.
    const PLACED_SHAPE_IMAGES: bool = false;

    /// Whether the platform keeps a pinned buffer's or heap's GPU address only inside a block it
    /// knows, and places it at an address of its own elsewhere (metal2vulkan-runtime's): a graphics
    /// device on it counts each pin not kept by the kind of object
    /// (`graphics::decode::unkept`). False by default: the platform keeps every pin.
    const PINS_IN_BLOCKS: bool = false;

    /// Whether a graphics device on the platform stages its IOSurface device caches' copies
    /// (`graphics::decode::surface_cache`, "Staged copies"): through the host's CPU and a Shared
    /// buffer of each copy's own, with no import of a surface's pages, where an import of the
    /// host's memory may cost each of the platform's submissions while it lives. False by default.
    const STAGED_SURFACE_COPIES: bool = false;

    /// Whether a graphics device on the platform makes views of an IOSurface's pages
    /// (`graphics::decode::surface_pages`): where false, an IOSurface block's `SURFACE_PAGES`, and
    /// a deferred block's pages, make no view and import nothing, however the guest's RAM is
    /// mapped, and the surface takes a device cache whose copies move its bytes at the guest's
    /// events. A pages block, an app's own no-copy memory, still takes its view. True by default.
    const SURFACE_VIEWS: bool = true;

    /// The process's platform, made on first use.
    fn process() -> Arc<Self>;

    /// The Metal device a config names: `None` is the system's default device, `Some(id)` the
    /// device whose registry ID is `id`. `None` when there is no such device.
    fn metal_device(&self, registry_id: Option<u64>) -> Option<Self::Device>;

    /// Copies the whole of `texture` into `dst`: `height` rows of `width * 4` bytes, each
    /// `bytes_per_row` after the last. The caller has checked that `dst` holds them.
    fn read_pixels(&self, texture: &Self::Texture, dst: &mut [u8], bytes_per_row: usize);

    /// A texture of `desc` on `device` of the platform's own, for a helper the guest never
    /// reaches (the window's glyph of the guest's cursor, `graphics::sink`'s `cursor`), so that no
    /// address or ID the guest reads depends on it: a plain texture of a descriptor of `desc` by
    /// default. `None` where the platform refuses it.
    fn own_texture(&self, device: &Self::Device, desc: &TextureDesc) -> Option<Self::Texture> {
        let descriptor = self.texture_descriptor(desc);
        crate::entry::no_catch(device.new_texture(OnFault::Die, &descriptor))
    }

    /// A shape's image of `desc` on `device`, in the per-shape form
    /// ([`ApiPlatform::SURFACE_SHAPES`]): the texture that the guest's textures over one shape of
    /// an IOSurface plane are views of, each at its own resource index, so that no address or ID
    /// the guest reads depends on the image. The platform's own texture by default
    /// ([`ApiPlatform::own_texture`]). `None` where the platform refuses it.
    fn shape_image(&self, device: &Self::Device, desc: &TextureDesc) -> Option<Self::Texture> {
        self.own_texture(device, desc)
    }

    /// A shape's image of `desc` on `device` as [`ApiPlatform::shape_image`] makes one, placed in
    /// memory whose bytes are as they are, which the caller fills before anything reads them
    /// (`decode::surface_cache`'s `shapes`, "Placed images"). `None` where the platform places
    /// none, as by default: the caller then makes the shape's image as before.
    fn placed_shape_image(
        &self,
        device: &Self::Device,
        desc: &TextureDesc,
    ) -> Option<Self::Texture> {
        let _ = (device, desc);
        None
    }

    /// What the platform holds of the guest's RAM while a graphics device made over `map` lives:
    /// the views of the surfaces' pages made through the map hold it, and so do the deallocators
    /// of the platform's imports of the map's bytes, until the platform has run them. Nothing by
    /// default.
    fn guest_memory(&self, map: &MemoryMap) -> Box<dyn Any + Send + Sync> {
        let _ = map;
        Box::new(())
    }

    /// The source of the blocks of a graphics device on `device`: the guest's Shared pool's
    /// (`crate::pool`) and its contexts' Private blocks (`pool::PrivateBlocks`). `None` by
    /// default, where the pool's blocks are the process's own memory (`pool::HostBlocks`) and a
    /// Private block is a range of addresses the host picks.
    fn block_source(&self, device: &Self::Device) -> Option<Arc<dyn crate::pool::BlockSource>> {
        let _ = device;
        None
    }

    /// What imports a pages block's view as a block the platform knows, on a platform that keeps
    /// a pin only inside a block it knows ([`ApiPlatform::PINS_IN_BLOCKS`]): a graphics device on
    /// it answers a pages block's `SURFACE_PAGES` with its view's import's own GPU address
    /// (`graphics::decode::surface_pages`, "Addresses the host gives"). `None` by default, where
    /// the host picks the address and the platform keeps a pin anywhere.
    fn view_imports(&self) -> Option<ViewImports<Self>>
    where
        Self: Sized,
    {
        None
    }

    /// Lets go of the GPU devices the platform keeps for the process, once nothing beyond its own
    /// keeps holds one, waiting at most `bound` for that: a device made after it is a new one.
    /// `Err` names what still held one when the wait ended, and nothing is let go then. Nothing
    /// to let go of by default.
    fn let_go_of_devices(&self, bound: std::time::Duration) -> Result<(), String> {
        let _ = bound;
        Ok(())
    }

    /// A graphics device's region of `len` bytes (BAR 2, `reims_vgpu_graphics::region::Region`)
    /// in memory of the platform's own, where its GPUs import that better than the region's own
    /// private anonymous memory; `None` by default, and where the host gives none, for the
    /// region's own ([`crate::GraphicsDevice::with_platform`]).
    fn graphics_region(&self, len: usize) -> Option<reims_vgpu_graphics::region::Region> {
        let _ = len;
        None
    }

    /// The host's hardware video for a graphics device on `device`, the guest's video driver's
    /// (`crate::GraphicsVideo`), whose host APIs' threads and waits are `runtime`'s; `None` where
    /// the platform has none, and by default.
    fn graphics_video(
        &self,
        device: &Self::Device,
        runtime: Arc<dyn reims_vgpu_video::core::runtime::Runtime>,
    ) -> Option<Arc<dyn crate::GraphicsVideo<Self>>> {
        let _ = (device, runtime);
        None
    }
}

/// A graphics device's region of `len` bytes for a test on a platform, as the device takes one
/// ([`crate::GraphicsDevice::with_platform`]): the platform's ([`ApiPlatform::graphics_region`]),
/// else the region's own private anonymous mapping.
#[cfg(test)]
pub(crate) fn test_region<P: ApiPlatform>(
    platform: &P,
    len: usize,
) -> reims_vgpu_graphics::region::Region {
    (platform.graphics_region(len))
        .or_else(|| reims_vgpu_graphics::region::Region::new(len))
        .expect("a region")
}

/// This process's count of the degradation kind `code`
/// (`reims_vgpu_vulkan::answer::degradation_counts`), which another test in it may have counted
/// before: a test reads it before and after what it checks.
#[cfg(test)]
pub(crate) fn counted(code: &str) -> u64 {
    (reims_vgpu_vulkan::answer::degradation_counts().into_iter())
        .find(|(c, _)| c == code)
        .map_or(0, |(_, n)| n)
}

/// The API's state for one platform instance.
pub(crate) struct ApiState<P: Platform> {
    /// The display thread, made at the first device whose host takes the cursor.
    display_thread: Mutex<Option<DisplayThread>>,
    /// The platform the state is of.
    platform: std::marker::PhantomData<fn() -> P>,
}

/// A platform instance's state, found by the platform's identity: the platform, held weakly,
/// and its [`ApiState`].
type Entry = (Weak<dyn Any + Send + Sync>, Arc<dyn Any + Send + Sync>);

/// Every live platform instance's state (one in the product; one per test platform).
static STATES: OnceLock<Mutex<Vec<Entry>>> = OnceLock::new();

/// The state of `platform`, made on first use. Entries of platforms that are gone are dropped
/// here, so an address is never matched to a platform it no longer holds.
pub(crate) fn api_state<P: Platform>(platform: &Arc<P>) -> Arc<ApiState<P>> {
    let states = STATES.get_or_init(|| Mutex::new(Vec::new()));
    let mut states = states.lock();
    states.retain(|(weak, _)| weak.strong_count() > 0);
    let me = Arc::as_ptr(platform).cast::<()>();
    for (weak, state) in states.iter() {
        if weak.as_ptr().cast::<()>() == me
            && let Ok(state) = Arc::clone(state).downcast::<ApiState<P>>()
        {
            return state;
        }
    }
    let state = Arc::new(ApiState::<P> {
        display_thread: Mutex::new(None),
        platform: std::marker::PhantomData,
    });
    let weak: Weak<dyn Any + Send + Sync> = Arc::downgrade(platform) as Weak<dyn Any + Send + Sync>;
    states.push((weak, state.clone() as Arc<dyn Any + Send + Sync>));
    state
}

impl<P: Platform> ApiState<P> {
    /// The display thread, made now if this is the process's first graphics device whose host
    /// takes the cursor.
    pub(crate) fn display_thread(&self, platform: &P) -> Arc<dyn HostQueue> {
        let mut slot = self.display_thread.lock();
        let thread = slot.get_or_insert_with(|| {
            DisplayThread(SerialQueue::new(platform.threads(), DISPLAY_THREAD_LABEL))
        });
        Arc::new(thread.clone())
    }
}

/// The display thread's label: a diagnostic.
pub const DISPLAY_THREAD_LABEL: &str = "reims-vgpu.display";

/// The executor the cursor callbacks run on: one serial queue for the process. Another handle is
/// the same queue.
#[derive(Clone, Debug)]
struct DisplayThread(SerialQueue);

impl Executor for DisplayThread {
    fn exec_async(&self, job: Job) {
        self.0.exec_async(job);
    }
}

impl HostQueue for DisplayThread {
    /// Another handle to the same queue.
    fn dispatch_retain(&self) -> Arc<dyn HostQueue> {
        Arc::new(self.clone())
    }
}

#[cfg(target_os = "linux")]
mod dma_buf;
mod rt;
/// A dma-buf of host memory the library mapped from files, for a device to import.
#[cfg(target_os = "linux")]
pub(crate) use dma_buf::host_dma_buf;

#[cfg(any(test, feature = "testkit"))]
mod fake {
    use reims_vgpu_vulkan::core::testkit::FakePlatform;
    use reims_vgpu_vulkan::core::testkit::platform::FAKE_REGISTRY_ID;

    use super::*;

    thread_local! {
        /// Each test thread's process.
        static PLATFORM: Arc<FakePlatform> = Arc::new(FakePlatform::new());
    }

    /// The fake processes that import their pages blocks' views as blocks they know, each with
    /// what imports them (`fake_view_imports`, in tests).
    static VIEW_IMPORTS: Mutex<Vec<(std::sync::Weak<FakePlatform>, ViewImports<FakePlatform>)>> =
        parking_lot::const_mutex(Vec::new());

    /// Has `platform` import each pages block's view as a block it knows, through `imports`, as
    /// metal2vulkan-runtime's does ([`ApiPlatform::view_imports`]): a graphics device made on it
    /// from now on offers what goes with them. The entries of processes gone are dropped here.
    #[cfg(test)]
    pub(crate) fn fake_view_imports(
        platform: &Arc<FakePlatform>,
        imports: ViewImports<FakePlatform>,
    ) {
        let mut held = VIEW_IMPORTS.lock();
        held.retain(|(p, _)| p.strong_count() > 0);
        held.push((Arc::downgrade(platform), imports));
    }

    /// The testkit's platform: a process per test thread, its one device (registry ID
    /// [`FAKE_REGISTRY_ID`]), and pixels that read as zero. A process a test gave view imports
    /// (`fake_view_imports`, in tests) answers them.
    impl ApiPlatform for FakePlatform {
        fn process() -> Arc<FakePlatform> {
            PLATFORM.with(Arc::clone)
        }

        fn view_imports(&self) -> Option<ViewImports<Self>> {
            let held = VIEW_IMPORTS.lock();
            // A process gone may have left its entry, at an address a new one can take.
            let mine =
                (held.iter()).find(|(p, _)| p.strong_count() > 0 && std::ptr::eq(p.as_ptr(), self));
            mine.map(|(_, imports)| Arc::clone(imports))
        }

        fn metal_device(&self, registry_id: Option<u64>) -> Option<<Self as Platform>::Device> {
            match registry_id {
                None => Some(self.device()),
                Some(id) => (id == FAKE_REGISTRY_ID).then(|| self.device()),
            }
        }

        fn read_pixels(
            &self,
            _texture: &<Self as Platform>::Texture,
            dst: &mut [u8],
            _bytes_per_row: usize,
        ) {
            dst.fill(0);
        }
    }
}
#[cfg(test)]
pub(crate) use fake::fake_view_imports;

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use reims_vgpu_vulkan::core::testkit::FakePlatform;

    use super::*;

    /// One state per platform instance; a platform that is gone leaves no entry to match
    #[test]
    fn one_state_per_platform() {
        let a = Arc::new(FakePlatform::new());
        let b = Arc::new(FakePlatform::new());
        let sa = api_state(&a);
        assert!(Arc::ptr_eq(&sa, &api_state(&a)));
        assert!(!Arc::ptr_eq(&sa, &api_state(&b)));
        drop(b);
        let c = Arc::new(FakePlatform::new());
        assert!(!Arc::ptr_eq(&sa, &api_state(&c)));
    }

    /// Every display gets the same thread: jobs through two handles run in one order, on the
    /// queue's worker
    #[test]
    fn one_display_thread() {
        let p = Arc::new(FakePlatform::new());
        let s = api_state(&p);
        let a = s.display_thread(&p);
        let b = s.display_thread(&p).dispatch_retain();
        let (tx, rx) = mpsc::channel();
        for i in 0..8 {
            let tx = tx.clone();
            let q = if i % 2 == 0 { &a } else { &b };
            q.exec_async(Box::new(move || tx.send(i).unwrap()));
        }
        let got: Vec<i32> = (0..8).map(|_| rx.recv().unwrap()).collect();
        assert_eq!(got, (0..8).collect::<Vec<_>>());
    }

    /// The platform's devices: the default one, the same object again by the registry ID the
    /// platform answers for it, and none for an ID no device has (`UINT64_MAX`)
    #[test]
    fn on_vulkan_the_devices_by_registry_id() {
        use reims_vgpu_vulkan::core::platform::Handle as _;

        let p = crate::rt_test_end::platform();
        let d = p.metal_device(None).expect("an rt device");
        let id = d.facts().registry_id;
        assert_eq!(p.metal_device(Some(id)).map(|o| o.id()), Some(d.id()));
        assert_eq!(p.metal_device(None).map(|o| o.id()), Some(d.id()));
        assert!(p.metal_device(Some(u64::MAX)).is_none());
    }

    /// A test build's shader cache is the tests' own, `reims-vgpu-vulkan-tests` in the user's
    /// cache directory (`~/.cache` on Linux and macOS): this crate's tests turn on the Vulkan
    /// crate's `testkit` feature, which makes them a test build there too, so no test reads or
    /// writes the cache the library's hosts share, nor the device checks kept beside it
    #[test]
    fn a_test_build_s_shader_cache_is_the_tests_own() {
        let dir = reims_vgpu_vulkan::os::cache_directory().expect("a shader cache directory");
        let cache = reims_vgpu_vulkan::os::user_cache_directory().expect("a user cache directory");
        assert_eq!(dir, cache.join("reims-vgpu-vulkan-tests"));
    }

    /// The platform's lease of the guest's RAM lets views be made of the pages the host
    /// names a file behind, showing the file from its offset, while it lives; never of a range
    /// without a file
    #[cfg(target_os = "linux")]
    #[test]
    fn the_lease_lets_views_be_made_of_the_guests_ram() {
        use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};

        use reims_vgpu_vulkan::RtPlatform;
        use reims_vgpu_vulkan::core::platform::os::vm::{HostAddr, HostVm as _};
        use reims_vgpu_vulkan::os::OsVm;

        use crate::memory::MemoryFile;

        const LEN: u64 = 0x4000;
        let p = RtPlatform::new();
        let vm = OsVm::new();
        // The guest's RAM as QEMU maps it: a shared mapping of a memfd, from an offset into it.
        // SAFETY: a NUL-terminated name; the descriptor is owned from here on.
        let fd = unsafe { libc::memfd_create(c"rvg-ram".as_ptr(), libc::MFD_CLOEXEC) };
        assert!(fd >= 0);
        // SAFETY: a descriptor this test just made, owned once.
        let fd = Arc::new(unsafe { OwnedFd::from_raw_fd(fd) });
        // SAFETY: sizing the file this test owns.
        assert_eq!(
            unsafe { libc::ftruncate(fd.as_raw_fd(), 2 * LEN as libc::off_t) },
            0
        );
        // SAFETY: a new shared mapping of the file's second `LEN` bytes, unmapped below.
        let ram = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                LEN as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                LEN as libc::off_t,
            )
        };
        assert_ne!(ram, libc::MAP_FAILED);
        let ram = HostAddr::from_ptr(ram.cast::<u8>());
        let plain = vm.allocate_anywhere(LEN).unwrap();
        let mut map = MemoryMap::new();
        let file = MemoryFile {
            file: Arc::clone(&fd),
            offset: LEN,
        };
        // SAFETY: both ranges stay mapped while the lease lives, and no device is made over them.
        unsafe {
            map.push_with_file(0, LEN, ram.as_ptr(), Some(file));
            map.push(LEN, LEN, plain.as_ptr());
        }
        // A view of `LEN` bytes from `from`: the byte it shows at 0x10, if one can be made.
        let view = |from: HostAddr| {
            let at = vm.allocate_anywhere(LEN).unwrap();
            let shown = vm.remap_shared_overwrite(at, from, LEN).ok().map(|_| {
                // SAFETY: the view was just mapped readable.
                unsafe { at.as_ptr::<u8>().add(0x10).read_volatile() }
            });
            assert!(vm.deallocate(at, LEN).is_success());
            shown
        };
        // SAFETY: the guest's RAM, mapped writable above.
        unsafe { ram.as_ptr::<u8>().add(0x10).write_volatile(0x5a) };
        assert_eq!(view(ram), None, "no file is known before the lease");
        let lease = p.guest_memory(&map);
        assert_eq!(view(ram), Some(0x5a));
        assert_eq!(view(plain), None, "a range without a file");
        drop(lease);
        assert_eq!(view(ram), None, "the lease's registration went with it");
        drop(map);
        // SAFETY: the mapping made above, which nothing uses any more.
        assert_eq!(unsafe { libc::munmap(ram.as_ptr(), LEN as usize) }, 0);
        assert!(vm.deallocate(plain, LEN).is_success());
    }
}
