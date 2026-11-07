//! Measured on Linux, as `no_copy_remap.rs` measures on macOS against Metal: what a no-copy
//! buffer's GPU work reads and writes once other pages are mapped over its range, untold
//! (`HostVm::remap_shared_overwrite`, `mmap(MAP_FIXED)` of their file). Metal wires the pages as it
//! makes the buffer, so its GPU keeps reading and writing them. The runtime imports the pages as
//! it makes the buffer too, and what the import keeps depends on how it imported them, which the
//! device decides and the tests read from the runtime's counts ([`Path`]):
//! - **Dma-bufs of the pages**, where the device refuses their pointer (RADV refuses any page with
//!   a file behind it): the dma-buf pinned the file's pages, so the GPU reads and fills the old
//!   pages, which the range no longer shows, on any driver. Measured on the Radeon 780M and the RX
//!   7900 XTX (RADV, Mesa 26.2.4, Linux 7.2.9), 2026-10-22.
//! - **Their pointer** (`VK_EXT_external_memory_host`): the driver's. NVIDIA's import keeps the
//!   pages it was made over: the GPU reads and fills the old pages, whether the range was ever
//!   touched, and whether or not the GPU used the buffer before the remap (an RTX 5080 Laptop GPU,
//!   NVIDIA 615.71.09, Linux 7.2, 2026-10-22). Where the device's own check finds its pointer
//!   imports follow their address (`Device::pointer_imports_follow_their_address`, which the
//!   runtime measures once per driver, release and kernel: ANV, RADV and lavapipe answer yes), the
//!   GPU reads the pages mapped there at each submission, which `HostMemory::new`'s contract
//!   forbids changing under a live buffer, and the runtime's own tests check
//!   (`tests::pointer_residency` there). And where the device refuses a file's pages by their
//!   pointer, it imports by their pointer only pages with no file behind them (RADV: amdgpu's
//!   userptr, of private anonymous pages), and a file's pages mapped under such an import fail
//!   every later submission of the device: amdgpu gets the import's pages again at each
//!   submission and refuses them, whatever the submission's commands use, and the runtime answers
//!   `DeviceLost` (both RADV GPUs above). Neither case is measured ([`unmeasured`], said on
//!   stderr): the first test maps nothing under such an import, and the second checks nothing
//!   the GPU read through one.
//!
//! The library's own unmaps wait for the imports over their pages (`crate::remap`'s `UnmapHold`),
//! so these measure the ground `crate::remap`'s docs give for pages the library does not unmap.
//!
//! A buffer made after such a remap, over exactly the bytes of a live one the GPU used, takes the
//! new pages where the runtime was told of the remap: by the platform's VM
//! (`VulkanPlatform::vm`), or by the host's blocks the platform wraps (`Platform::host_blocks`).
//! Untold, it lies in the live one's import, over the old pages where that import keeps them.
//!
//! On Linux a view can be made only of pages with a file behind them (`os::linux::vm`), so every
//! range remapped here comes from the platform's `allocate_aliasable`, and the remaps no one is
//! told of are made through the OS's VM alone (`os::OsVm`), which reports nothing.

#![cfg(target_os = "linux")]
// The proof that a command buffer's result is well-formed walks the runtime's types for `Send` and
// `Sync`, deeper than rustc's default limit of 128, as in the library (`src/lib.rs`).
#![recursion_limit = "256"]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    missing_docs,
    unsafe_code
)]

use std::sync::{Arc, mpsc};
use std::time::Duration;

use reims_vgpu_core::host::{HostBlocks, HostPtr, HostTask, PhysicalMemoryRange};
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
use reims_vgpu_core::platform::device::{GpuDevice, NoCopyDeallocator};
use reims_vgpu_core::platform::os::vm::HostVm;
use reims_vgpu_core::platform::resource::Buffer;
use reims_vgpu_core::platform::types::NsRange;
use reims_vgpu_core::platform::{HostRange, Platform};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_core::tasks::launcher::{HostSide, ServiceLink, TaskLauncher};
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::os::OsVm;
use reims_vgpu_vulkan::remap::VulkanVm;

type P = VulkanPlatform;
type Device = <P as Platform>::Device;
type BufferOf = <P as Platform>::Buffer;

const LEN: u64 = 0x8000;

/// A launcher no test launches through.
struct NoLauncher;

impl TaskLauncher<P> for NoLauncher {
    fn launch(&self, _: &Arc<P>, _: Arc<dyn HostSide<P>>) -> Arc<dyn ServiceLink<P>> {
        panic!("no task is launched in these tests")
    }
}

/// A fresh range of `LEN` bytes a view can be made of, every one `byte`.
fn region(byte: u8) -> HostPtr {
    let p = VulkanVm::new().allocate_aliasable(LEN).unwrap();
    // SAFETY: `p..p + LEN` was just allocated, readable and writable.
    unsafe { std::ptr::write_bytes(p.as_ptr::<u8>(), byte, usize::try_from(LEN).unwrap()) };
    p
}

/// The first and last bytes at `p`.
fn ends(p: HostPtr) -> [u8; 2] {
    let last = usize::try_from(LEN - 1).unwrap();
    // SAFETY: `p..p + LEN` is a live mapping.
    unsafe { [*p.as_ptr::<u8>(), *p.as_ptr::<u8>().add(last)] }
}

/// The GPU's copy of `src`'s first and last bytes, then a GPU fill of `src` with `fill`.
fn gpu_read_then_fill(dev: &Device, src: &BufferOf, fill: u8) -> [u8; 2] {
    let out = dev
        .new_buffer_of_length(OnFault::Return, LEN, 0)
        .unwrap()
        .unwrap();
    let cb = dev
        .new_command_queue()
        .unwrap()
        .command_buffer(OnFault::Return)
        .unwrap()
        .unwrap();
    let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    let copy = BlitCmd::CopyBufferToBuffer {
        source_buffer: Some(src),
        source_offset: 0,
        destination_buffer: Some(&out),
        destination_offset: 0,
        size: LEN,
    };
    let range = NsRange::new(0, LEN);
    for cmd in [
        copy,
        BlitCmd::FillBuffer {
            buffer: src,
            range,
            value: fill,
        },
        BlitCmd::EndEncoding,
    ] {
        e.encode(OnFault::Return, cmd).unwrap();
    }
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    ends(out.contents())
}

/// How the runtime imported a no-copy buffer's pages (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    /// By their pointer.
    Pointer,
    /// As dma-bufs of them, where the device refused their pointer.
    DmaBuf,
}

/// The imports of host pages the device made so far: by their pointer, and of dma-bufs of them.
/// Counts that only grow, as the runtime may free an earlier buffer's import on a handler thread
/// meanwhile, after its deallocator ran.
fn imports_made(dev: &Device) -> (u64, u64) {
    let counts = dev.raw.work_counts();
    (counts.host_pointer_imports_made, counts.dma_buf_imports)
}

/// Whether `dev` imports a file's pages by their pointer: a no-copy buffer over a fresh range a
/// view can be made of, made and let go of.
fn files_by_pointer(dev: &Device) -> bool {
    let va = region(0);
    let (buffer, gone, path) = no_copy(dev, va);
    drop(buffer);
    released(&[gone]);
    assert!(OsVm::new().deallocate(va, LEN).is_success());
    path == Some(Path::Pointer)
}

/// Why GPU work after a file's pages were mapped, untold, under an import of `path` is not
/// measured on `dev` (module docs), or `None` where it is: always for dma-bufs; for a pointer,
/// where the device's check finds its pointer imports follow their address, or the device refuses
/// a file's pages by their pointer. The check answers within milliseconds of the device's making,
/// waited for here up to 10 s; unanswered, it reads as not following.
fn unmeasured(dev: &Device, path: Path) -> Option<&'static str> {
    let _ = dev.raw.settle_submit_checks(Duration::from_secs(10));
    let follows = dev.raw.pointer_imports_follow_their_address();
    match path {
        Path::DmaBuf => None,
        Path::Pointer if follows => {
            Some("the device's check finds its pointer imports follow their address")
        }
        Path::Pointer if !files_by_pointer(dev) => Some(
            "the device refuses a file's pages by their pointer, and fails its later submissions \
             where a file's pages lie under a pointer import",
        ),
        Path::Pointer => None,
    }
}

/// A no-copy buffer over `va..va + LEN` whose deallocator says so on the answer's receiver, and
/// how the runtime imported its pages: `None` where it lies in a live import.
fn no_copy(dev: &Device, va: HostPtr) -> (BufferOf, mpsc::Receiver<()>, Option<Path>) {
    let (released, gone) = mpsc::channel();
    let deallocator: NoCopyDeallocator = Box::new(move || {
        let _ = released.send(());
    });
    // SAFETY: `va..va + LEN` stays mapped and readable until the buffer's deallocator runs: each
    // caller deallocates it only after that, and a remap replaces its pages in one call, never
    // leaving it unmapped. Under the runtime's buffer the range's pages may change only where the
    // runtime is told before a buffer over them is made, which these tests break on purpose, as
    // it is what they measure; nothing touches freed memory for it: the pages the buffer was made
    // over stay pinned by the driver's import, and the test reaches the range only through `va`.
    let range = unsafe { HostRange::new(va, LEN) };
    let before = imports_made(dev);
    let buffer = dev
        .new_buffer_over_host_memory(OnFault::Return, range, 0, Some(deallocator))
        .unwrap()
        .unwrap();
    let after = imports_made(dev);
    let path = match (after.0 > before.0, after.1 > before.1) {
        (true, _) => Some(Path::Pointer),
        (_, true) => Some(Path::DmaBuf),
        _ => None,
    };
    (buffer, gone, path)
}

/// The buffers' deallocators ran.
fn released(gone: &[mpsc::Receiver<()>]) {
    for gone in gone {
        assert_eq!(
            gone.recv_timeout(Duration::from_secs(10)),
            Ok(()),
            "the buffer's deallocator ran"
        );
    }
}

/// What the GPU sees of a no-copy buffer over `va` (`LEN` bytes) with other pages of `0xe5`
/// remapped over `va`, untold, after `before` ran on the buffer: the first and last bytes it
/// copies out of the buffer, then, after it fills the buffer with `0x7c`, those of `va` and of the
/// new pages; `None`, with no remap, where that is not measured under the buffer's import
/// ([`unmeasured`], said on stderr with `label`). Both ranges are deallocated once the buffer's
/// deallocator ran.
fn remapped(
    dev: &Device,
    label: &str,
    va: HostPtr,
    before: impl Fn(&BufferOf),
) -> Option<[[u8; 2]; 3]> {
    let vm = OsVm::new();
    let (buffer, gone, path) = no_copy(dev, va);
    let path = path.unwrap();
    before(&buffer);
    let new = region(0xe5);
    let seen = match unmeasured(dev, path) {
        Some(why) => {
            eprintln!("{label}: not remapped under its import ({path:?}): {why}");
            None
        }
        None => {
            vm.remap_shared_overwrite(va, new, LEN).unwrap();
            assert_eq!(ends(va), [0xe5; 2], "the CPU sees the new pages");
            let read = gpu_read_then_fill(dev, &buffer, 0x7c);
            Some([read, ends(va), ends(new)])
        }
    };
    drop(buffer);
    released(&[gone]);
    for mapping in [va, new] {
        assert!(vm.deallocate(mapping, LEN).is_success());
    }
    seen
}

/// The GPU reads the old pages (`old`), and fills them where the range no longer shows them.
fn not_seen(old: u8) -> [[u8; 2]; 3] {
    [[old; 2], [0xe5; 2], [0xe5; 2]]
}

/// The GPU reads the new pages, and fills them, which the range and the new mapping show.
const SEEN: [[u8; 2]; 3] = [[0xe5; 2], [0x7c; 2], [0x7c; 2]];

/// A remap under a no-copy buffer, made over a fresh range of private anonymous pages never
/// touched, over touched pages of a file (`0xa1`), both after the buffer was made and before its
/// first GPU use, and after that use (`0xc3`): the GPU reads and fills the old pages every time,
/// wherever the remap is measured under the buffer's import
#[test]
fn a_no_copy_buffer_keeps_the_pages_it_was_made_over() {
    let dev = VulkanPlatform::system_default_device().unwrap();
    let untouched = OsVm::new().allocate_anywhere(LEN).unwrap();
    let seen = [
        remapped(&dev, "untouched", untouched, |_| ()),
        remapped(&dev, "touched", region(0xa1), |_| ()),
        remapped(&dev, "used by the GPU", region(0xc3), |b| {
            assert_eq!(gpu_read_then_fill(&dev, b, 0xc3), [0xc3; 2]);
        }),
    ];
    for (seen, old) in seen.into_iter().zip([0, 0xa1, 0xc3]) {
        assert!(seen.is_none_or(|seen| seen == not_seen(old)), "{seen:x?}");
    }
}

/// What the GPU sees of a buffer B made over `va` (`LEN` bytes of `0xa1`) after `remap` put other
/// pages of `0xe5` there, while a buffer A made over the same bytes before, which the GPU read
/// and then filled with `0xc3`, lives: B's first and last bytes copied out, then, after the GPU
/// fills B with `0x7c`, those of `va` and of the new pages (the answer's second); and how the
/// runtime imported A's pages.
fn made_after(dev: &Device, va: HostPtr, remap: impl FnOnce() -> HostPtr) -> ([[u8; 2]; 3], Path) {
    let (a, a_gone, path) = no_copy(dev, va);
    assert_eq!(gpu_read_then_fill(dev, &a, 0xc3), [0xa1; 2]);
    let new = remap();
    assert_eq!(ends(va), [0xe5; 2], "the CPU sees the new pages");
    let (b, b_gone, _) = no_copy(dev, va);
    let seen = [gpu_read_then_fill(dev, &b, 0x7c), ends(va), ends(new)];
    drop((b, a));
    released(&[b_gone, a_gone]);
    assert!(OsVm::new().deallocate(new, LEN).is_success());
    (seen, path.unwrap())
}

/// [`made_after`] a remap through `vm`.
fn made_after_a_remap(dev: &Device, vm: &dyn HostVm) -> ([[u8; 2]; 3], Path) {
    let va = region(0xa1);
    let seen = made_after(dev, va, || {
        let new = region(0xe5);
        vm.remap_shared_overwrite(va, new, LEN).unwrap();
        new
    });
    assert!(vm.deallocate(va, LEN).is_success());
    seen
}

/// Host blocks that do the mapping work as a host does, with the OS's VM of their own:
/// `createTask` reserves the task's range, `mapMemory` remaps each range's pages, whose
/// `physicalAddress` is here the host address of pages to share, from the task's base plus the
/// offset on, `unmapMemory` puts fresh zero pages over its range, and `destroyTask` deallocates
/// the task's range.
fn host_blocks() -> HostBlocks {
    let task: Arc<Mutex<(HostPtr, u64)>> = Arc::default();
    let (t, vm) = (Arc::clone(&task), OsVm::new());
    let create = move |size, base: &mut HostPtr| {
        *base = vm.allocate_anywhere(size).unwrap();
        *t.lock() = (*base, size);
        HostTask::from_addr(1)
    };
    let (t, vm) = (Arc::clone(&task), OsVm::new());
    let destroy = move |_| {
        let (base, size) = *t.lock();
        assert!(vm.deallocate(base, size).is_success());
    };
    let (t, vm) = (Arc::clone(&task), OsVm::new());
    let map = move |_, count: u32, offset: u64, _, ranges: &[PhysicalMemoryRange]| {
        let mut at = t.lock().0.wrapping_add(offset);
        for r in &ranges[..usize::try_from(count).unwrap()] {
            let src = HostPtr::from_addr(usize::try_from(r.physical_address).unwrap());
            vm.remap_shared_overwrite(at, src, r.physical_length)
                .unwrap();
            at = at.wrapping_add(r.physical_length);
        }
        true
    };
    let (t, vm) = (Arc::clone(&task), OsVm::new());
    let unmap = move |_, offset: u64, length| {
        let at = t.lock().0.wrapping_add(offset);
        vm.allocate_fixed_overwrite(at, length).is_ok()
    };
    HostBlocks {
        create_task: Some(Arc::new(create)),
        destroy_task: Some(Arc::new(destroy)),
        map_memory: Some(Arc::new(map)),
        unmap_memory: Some(Arc::new(unmap)),
        ..HostBlocks::default()
    }
}

/// [`made_after`] the host's blocks unmapped the task's first `LEN` bytes and mapped other pages
/// there, through `blocks` (the host's, or what the platform answered for them).
fn made_after_the_host_remaps(dev: &Device, blocks: &HostBlocks) -> ([[u8; 2]; 3], Path) {
    let (create, destroy) = (blocks.create_task.as_ref(), blocks.destroy_task.as_ref());
    let (map, unmap) = (blocks.map_memory.as_ref(), blocks.unmap_memory.as_ref());
    let pages = |at: HostPtr| PhysicalMemoryRange {
        physical_address: at.addr().try_into().unwrap(),
        physical_length: LEN,
    };
    let mut base = HostPtr::NULL;
    let task = create.unwrap()(4 * LEN, &mut base);
    let old = region(0xa1);
    assert!(map.unwrap()(task, 1, 0, false, &[pages(old)]));
    let seen = made_after(dev, base, || {
        assert!(unmap.unwrap()(task, 0, LEN));
        let new = region(0xe5);
        assert!(map.unwrap()(task, 1, 0, false, &[pages(new)]));
        new
    });
    destroy.unwrap()(task);
    assert!(OsVm::new().deallocate(old, LEN).is_success());
    seen
}

/// A buffer made over the bytes of a live one after a remap there takes the new pages where the
/// runtime was told of it, by the platform's VM or the host's blocks the platform wraps; untold,
/// it lies in the live one's import, and where that import keeps its pages ([`unmeasured`]) the
/// GPU reads what that one wrote and fills the old pages
#[test]
fn a_buffer_made_after_a_remap_the_runtime_was_told_of_takes_the_new_pages() {
    let dev = VulkanPlatform::system_default_device().unwrap();
    let p = VulkanPlatform::new(Box::new(NoLauncher));
    assert_eq!(made_after_a_remap(&dev, p.vm()).0, SEEN);
    let blocks = p.host_blocks(host_blocks());
    assert_eq!(made_after_the_host_remaps(&dev, &blocks).0, SEEN);
    let untold = [
        ("the OS's VM", made_after_a_remap(&dev, &OsVm::new())),
        (
            "the host's blocks",
            made_after_the_host_remaps(&dev, &host_blocks()),
        ),
    ];
    for (label, (seen, path)) in untold {
        match unmeasured(&dev, path) {
            None => assert_eq!(seen, not_seen(0xc3), "{label}"),
            Some(why) => eprintln!("{label}: not measured ({path:?}): {why}"),
        }
    }
}
