//! What the API needs of a platform beyond core's [`Platform`] ([`ApiPlatform`]), and the state
//! it keeps once per platform instance, which in the product is once per process: the display
//! thread.
//!
//! **The display thread.** A graphics device queues its host's cursor callbacks onto one core
//! [`SerialQueue`], made at the process's first device whose host takes them (`graphics`'
//! `HostCursor`), so every display callback of the process runs one at a time in the order it
//! was queued. The queue's worker exits after [`reims_vgpu_core::rt::queue::IDLE_EXIT`] idle and
//! the next job starts another. The library never waits for it, and a queue's drop joins nothing.
//!
//! **By platform.** Each platform's `ApiPlatform` is a file of its own: `platform/metal.rs`
//! (macOS) and `platform/vulkan.rs` (Linux, and macOS with the `vulkan` feature).

use std::any::Any;
use std::sync::{Arc, OnceLock, Weak};

use reims_vgpu_core::host::HostQueue;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::rt::{Executor, Job, SerialQueue};
use reims_vgpu_core::sync::Mutex;

use crate::memory::MemoryMap;

/// What the API asks of a platform beyond core's: the process's instance, its GPU device, a
/// texture's pixels, what it holds of the guest's RAM, a graphics device's region and its video.
pub trait ApiPlatform: Platform {
    /// The process's platform, made on first use.
    fn process() -> Arc<Self>;

    /// The Metal device a config names: `None` is the system's default device, `Some(id)` the
    /// device whose registry ID is `id`. `None` when there is no such device.
    fn metal_device(&self, registry_id: Option<u64>) -> Option<Self::Device>;

    /// Copies the whole of `texture` into `dst`: `height` rows of `width * 4` bytes, each
    /// `bytes_per_row` after the last. The caller has checked that `dst` holds them.
    fn read_pixels(&self, texture: &Self::Texture, dst: &mut [u8], bytes_per_row: usize);

    /// What the platform holds of the guest's RAM while a graphics device made over `map` lives:
    /// the views of the surfaces' pages made through the map hold it. Nothing by default.
    fn guest_memory(&self, map: &MemoryMap) -> Box<dyn Any + Send + Sync> {
        let _ = map;
        Box::new(())
    }

    /// A graphics device's region of `len` bytes (BAR 2, `reims_graphics::region::Region`) in
    /// memory of the platform's own, where its GPUs import that better than private anonymous
    /// memory; `None` by default, and where the host gives none, for the region's own mapping.
    fn graphics_region(&self, len: usize) -> Option<reims_graphics::region::Region> {
        let _ = len;
        None
    }

    /// The host's hardware video for a graphics device on `device`, the guest's video driver's
    /// (`crate::GraphicsVideo`); `None` where the platform has none, and by default.
    fn graphics_video(&self, device: &Self::Device) -> Option<Arc<dyn crate::GraphicsVideo<Self>>> {
        let _ = device;
        None
    }
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

#[cfg(target_os = "macos")]
mod metal;
#[cfg(vulkan_platform)]
mod vulkan;

#[cfg(any(test, feature = "testkit"))]
mod fake {
    use reims_vgpu_core::testkit::FakePlatform;
    use reims_vgpu_core::testkit::platform::FAKE_REGISTRY_ID;

    use super::*;

    thread_local! {
        /// Each test thread's process.
        static PLATFORM: Arc<FakePlatform> = Arc::new(FakePlatform::new());
    }

    /// The testkit's platform: a process per test thread, its one device (registry ID
    /// [`FAKE_REGISTRY_ID`]), and pixels that read as zero.
    impl ApiPlatform for FakePlatform {
        fn process() -> Arc<FakePlatform> {
            PLATFORM.with(Arc::clone)
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
mod tests {
    use std::sync::mpsc;

    use reims_vgpu_core::testkit::FakePlatform;

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

    /// The Vulkan platform's devices: the default one, the same object again by the registry ID
    /// the platform answers for it, and none for an ID no device has (the suite's `UINT64_MAX`)
    #[cfg(vulkan_platform)]
    #[test]
    fn vulkan_devices_by_registry_id() {
        use reims_vgpu_core::platform::Handle as _;
        use reims_vgpu_core::tasks::isolation::TaskThreads;
        use reims_vgpu_vulkan::VulkanPlatform;

        let p = VulkanPlatform::new(Box::new(TaskThreads::new()));
        let d = p.metal_device(None).expect("a Vulkan device (MoltenVK)");
        let id = d.registry_id;
        assert_eq!(p.metal_device(Some(id)).map(|o| o.id()), Some(d.id()));
        assert_eq!(p.metal_device(None).map(|o| o.id()), Some(d.id()));
        assert!(p.metal_device(Some(u64::MAX)).is_none());
    }

    /// The Vulkan platform's lease of the guest's RAM lets views be made of the pages the host
    /// names a file behind, showing the file from its offset, while it lives; never of a range
    /// without a file
    #[cfg(target_os = "linux")]
    #[test]
    fn the_vulkan_lease_lets_views_be_made_of_the_guests_ram() {
        use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};

        use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm as _};
        use reims_vgpu_core::tasks::isolation::TaskThreads;
        use reims_vgpu_vulkan::VulkanPlatform;
        use reims_vgpu_vulkan::os::OsVm;

        use crate::memory::MemoryFile;

        const LEN: u64 = 0x4000;
        let p = VulkanPlatform::new(Box::new(TaskThreads::new()));
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
            fd: Arc::clone(&fd),
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
