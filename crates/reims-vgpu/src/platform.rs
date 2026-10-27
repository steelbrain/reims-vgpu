//! What the API needs of a platform beyond core's [`Platform`] ([`ApiPlatform`]), and the state
//! it keeps once per platform instance, which in the product is once per process: the display
//! thread and the frame queues.
//!
//! **The display thread.** Every display is given one executor in place of a host's dispatch
//! queue: a core [`SerialQueue`] made at the process's first display (docs/api.md section 5).
//! Core queues onto it exactly the jobs it queued onto the host's queue, at the same points, so
//! every display callback of the process runs one at a time in the order they were queued, as
//! they did on the main queue QEMU and the runner gave every display. The queue's worker exits
//! after [`reims_vgpu_core::rt::queue::IDLE_EXIT`] idle and the next job starts another. It is
//! not a dispatch queue, so core models the new-frame source with its `CoalescingSignal`. The
//! library never waits for it: no core path `sync`s onto a display's queue, and a queue's drop
//! joins nothing.
//!
//! **The frame queues.** One command queue per Metal device, made the first time a device is
//! created on that Metal device, before core builds the device, and kept for the life of the
//! process. Every frame's command buffer comes from it, a window's among them.
//!
//! **Windows.** A window display's host screen and what draws its frames are chosen by the
//! platform's type (`window::host_screen`, `window::window_sink`): `MetalPlatform` answers from
//! the host's screens and with a sink over Metal; the Vulkan platform (feature `vulkan`), from
//! the host's screens and, on macOS, with a sink that has Metal copy the runtime's frames into
//! the window; the testkit's platform, a fixed screen (`FAKE_HOST_SCREEN`) and no sink, so its
//! window displays never open a window. So the trait below names no Objective-C type.
//!
//! **By platform.** Each platform's `ApiPlatform` is a file of its own: `platform/metal.rs`
//! (macOS) and `platform/vulkan.rs` (feature `vulkan`).

use std::any::Any;
use std::sync::{Arc, OnceLock, Weak};

use reims_vgpu_core::host::HostQueue;
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::{Handle, ObjId, Platform};
use reims_vgpu_core::rt::{Executor, Job, SerialQueue};
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_video_core::device_memory::DeviceMemory;

use crate::device::MemoryMap;

/// What the API asks of a platform beyond core's: the process's instance, the Metal device a
/// config names, a frame target's pixels, and what it holds of the guest's RAM.
pub trait ApiPlatform: Platform {
    /// The process's platform, made on first use.
    fn process() -> Arc<Self>;

    /// The Metal device a config names: `None` is the system's default device, `Some(id)` the
    /// device whose registry ID is `id`. `None` when there is no such device.
    fn metal_device(&self, registry_id: Option<u64>) -> Option<Self::Device>;

    /// Copies the whole of `texture` into `dst`: `height` rows of `width * 4` bytes, each
    /// `bytes_per_row` after the last. The caller has checked that `dst` holds them.
    fn read_pixels(&self, texture: &Self::Texture, dst: &mut [u8], bytes_per_row: usize);

    /// Calls `read` with the bytes [`read_pixels`](Self::read_pixels) would copy, where the CPU
    /// reaches them in place: the first byte of the texture's first row, the bytes from there to
    /// the end of its last row's `width * 4`, and the bytes from one row to the next. False,
    /// without calling it, where the texture is read only by copying (`read_pixels`), as by
    /// default. The bytes may change under the reader as the GPU's or the guest's writes reach
    /// them, as they may under `read_pixels`: read them through the pointer, never as a slice.
    fn read_in_place(
        &self,
        texture: &Self::Texture,
        read: &mut dyn FnMut(*const u8, usize, usize),
    ) -> bool {
        let _ = (texture, read);
        false
    }

    /// What the platform holds of the guest's RAM while a device made over `map` lives: the
    /// device takes it as it is created and drops it after core's teardown. Nothing by default.
    fn guest_memory(&self, map: &MemoryMap) -> Box<dyn Any + Send + Sync> {
        let _ = map;
        Box::new(())
    }

    /// The planes of `surface`, in order: what a video device's codecs read of a guest pixel
    /// buffer's layout beyond core's `Surface`. Empty for a surface without planes, whose layout
    /// is the surface's own, and by default, for a platform that does not say.
    fn surface_planes(&self, surface: &Self::Surface) -> Vec<crate::video::SurfacePlane> {
        let _ = surface;
        Vec::new()
    }

    /// Lends `surface`'s bytes, from its first byte to the end of its allocation, to `write`:
    /// what a video device's decoders write a guest pixel buffer's frames through, in place where
    /// they can be. `Err` with why where they cannot be lent, and by default, for a platform that
    /// lends none.
    fn lend_surface_bytes(
        &self,
        surface: &Self::Surface,
        write: &mut dyn FnMut(&mut [u8]),
    ) -> Result<(), &'static str> {
        let _ = (surface, write);
        Err("this platform lends no surface's bytes")
    }

    /// Lends `surface`'s bytes, laid out as [`ApiPlatform::lend_surface_bytes`] lends them, to
    /// `read`: what a video device's scaler and encoders read a guest pixel buffer through,
    /// writing nothing back, so a surface the guest made read-only for its clients is lent too.
    /// `Err` with why where they cannot be lent, and by default, for a platform that lends none.
    fn lend_surface_bytes_for_reading(
        &self,
        surface: &Self::Surface,
        read: &mut dyn FnMut(&[u8]),
    ) -> Result<(), &'static str> {
        let _ = (surface, read);
        Err("this platform lends no surface's bytes")
    }

    /// What a video device's decoders lend a device of `surface`'s memory, so that the device
    /// writes a guest pixel buffer's frames into it with no copy by the CPU: the bytes
    /// [`ApiPlatform::lend_surface_bytes`] lends, on Linux as a dma-buf of their pages and as a
    /// view of them in the host's memory, on macOS as the surface's own IOSurface, and an
    /// identity that lives as long as the surface does, so that a device's imports of them go
    /// with it. The caller holds the surface while it uses them. `Err` with why where none is
    /// lent: a protected surface, one the guest made read-only for its clients, and by default,
    /// for a platform that lends none.
    fn surface_device_memory(
        &self,
        surface: &Self::Surface,
    ) -> Result<Box<dyn DeviceMemory>, &'static str> {
        let _ = surface;
        Err("this platform lends no surface's memory to devices")
    }
}

/// The API's state for one platform instance.
pub(crate) struct ApiState<P: Platform> {
    /// The display thread, made at the first display.
    display_thread: Mutex<Option<DisplayThread>>,
    /// The frame queue of each Metal device a device was created on, by the device's identity
    /// (`None`: Metal made none).
    frame_queues: Mutex<Vec<(ObjId, Option<P::CommandQueue>)>>,
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
        frame_queues: Mutex::new(Vec::new()),
    });
    let weak: Weak<dyn Any + Send + Sync> = Arc::downgrade(platform) as Weak<dyn Any + Send + Sync>;
    states.push((weak, state.clone() as Arc<dyn Any + Send + Sync>));
    state
}

impl<P: Platform> ApiState<P> {
    /// The display thread, made now if this is the process's first display.
    pub(crate) fn display_thread(&self, platform: &P) -> Arc<dyn HostQueue> {
        let mut slot = self.display_thread.lock();
        let thread = slot.get_or_insert_with(|| {
            DisplayThread(SerialQueue::new(platform.threads(), DISPLAY_THREAD_LABEL))
        });
        Arc::new(thread.clone())
    }

    /// The frame queue for `device`, made now if none was made for it.
    pub(crate) fn frame_queue(&self, device: &P::Device) -> Option<P::CommandQueue> {
        let id = device.id();
        let mut queues = self.frame_queues.lock();
        if let Some((_, q)) = queues.iter().find(|(d, _)| *d == id) {
            return q.as_ref().map(Handle::clone_ref);
        }
        let queue = device.new_command_queue();
        let found = queue.as_ref().map(Handle::clone_ref);
        queues.push((id, queue));
        found
    }
}

/// The display thread's label: a diagnostic.
pub const DISPLAY_THREAD_LABEL: &str = "reims-vgpu.display";

/// The executor every display is given in place of a host's queue: one serial queue for the
/// process. Another handle is the same queue.
#[derive(Clone, Debug)]
struct DisplayThread(SerialQueue);

impl Executor for DisplayThread {
    fn exec_async(&self, job: Job) {
        self.0.exec_async(job);
    }
}

impl HostQueue for DisplayThread {
    /// Another handle to the same queue: a display holds it until its unplug.
    fn dispatch_retain(&self) -> Arc<dyn HostQueue> {
        Arc::new(self.clone())
    }
}

#[cfg(target_os = "macos")]
mod metal;
#[cfg(feature = "vulkan")]
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

    /// The host screen of the testkit's platform, whatever the port: a Retina screen of 3024 by
    /// 1964 pixels at 2 pixels a point, at 120 Hz.
    pub const FAKE_HOST_SCREEN: reims_vgpu_window::HostScreen = reims_vgpu_window::HostScreen {
        id: 1,
        pixels: (3024, 1964),
        scale: 2.0,
        refresh_hz: 120.0,
    };

    /// The testkit's platform: a process per test thread, its one device (registry ID
    /// [`FAKE_REGISTRY_ID`]), pixels that read as zero, one host screen
    /// ([`FAKE_HOST_SCREEN`]) and no window sink, so a window display's window is never opened.
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

#[cfg(any(test, feature = "testkit"))]
pub use fake::FAKE_HOST_SCREEN;

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
    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_devices_by_registry_id() {
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
    #[cfg(all(feature = "vulkan", target_os = "linux"))]
    #[test]
    fn the_vulkan_lease_lets_views_be_made_of_the_guests_ram() {
        use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};

        use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm as _};
        use reims_vgpu_core::tasks::isolation::TaskThreads;
        use reims_vgpu_vulkan::VulkanPlatform;
        use reims_vgpu_vulkan::os::OsVm;

        use crate::device::MemoryFile;

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

    /// One frame queue per Metal device, made once
    #[test]
    fn one_frame_queue_per_device() {
        let p = Arc::new(FakePlatform::new());
        let s = api_state(&p);
        let d = p.device();
        let q1 = s.frame_queue(&d).unwrap();
        let q2 = s.frame_queue(&d).unwrap();
        assert_eq!(q1.id(), q2.id());
        assert_eq!(p.metal_device(Some(0)).map(|d| d.id()), None);
        assert_eq!(p.metal_device(None).map(|d| d.id()), Some(d.id()));
    }
}
