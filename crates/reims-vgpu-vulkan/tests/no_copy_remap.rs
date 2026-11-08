//! Measured: a no-copy buffer keeps the pages its range held when it was made, on Metal and on
//! this crate's runtime alike, which imports them then, as Metal wires
//! them. A `mach_vm_remap` of other pages over the range (`HostVm::remap_shared_overwrite`) after
//! that is not seen by the GPU, whether the range was ever touched, and whether or not the GPU
//! used the buffer before the remap: a GPU read answers the old pages' bytes, and a GPU fill
//! writes the old pages, which the range no longer shows. So a buffer's backing cannot be
//! gathered by remapping under it. Measured on `AppleM3Max-40c-25G229`.
//!
//! A buffer made after such a remap, over exactly the bytes of a live one the GPU used, takes the
//! new pages on Metal, which wires them as it makes each buffer. On this crate's runtime it lies
//! in the live one's import, over the old pages, unless the runtime was told of the remap: the
//! platform tells it of what its Mach VM changes (`VulkanPlatform::vm`), and then the new buffer
//! takes the new pages too. A no-copy buffer's deallocator may call the platform's Mach VM, which tells the runtime,
//! whichever thread releases the buffer.

#![cfg(target_os = "macos")]
// Compares with Metal's answers, recorded on macOS.
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

use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
use reims_vgpu_core::platform::device::{GpuDevice, NoCopyDeallocator};
use reims_vgpu_core::platform::event::{EventRef, SharedEvent};
use reims_vgpu_core::platform::os::vm::HostVm;
use reims_vgpu_core::platform::resource::Buffer;
use reims_vgpu_core::platform::types::NsRange;
use reims_vgpu_core::platform::{HostRange, Platform};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::os::OsVm;

const LEN: u64 = 0x8000;

/// A fresh mapping of `LEN` bytes, every one `byte`.
fn region(vm: &dyn HostVm, byte: u8) -> HostPtr {
    let p = vm.allocate_anywhere(LEN).unwrap();
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
fn gpu_read_then_fill<P: Platform>(dev: &P::Device, src: &P::Buffer, fill: u8) -> [u8; 2] {
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

/// What the GPU sees of a no-copy buffer over `va` (a mapping of `LEN` bytes) with other pages
/// of `0xe5` remapped over `va` after `before` ran on the buffer: the first and last bytes it
/// copies out of the buffer, then, after it fills the buffer with `0x7c`, those of `va` and of
/// the new pages. Both mappings are deallocated once the buffer's deallocator ran.
fn remapped<P: Platform>(
    dev: &P::Device,
    va: HostPtr,
    before: impl Fn(&P::Buffer),
) -> [[u8; 2]; 3] {
    let vm = OsVm::new();
    let (released, gone) = mpsc::channel();
    let deallocator: NoCopyDeallocator = Box::new(move || {
        let _ = released.send(());
    });
    // SAFETY: what `HostRange::new` asks holds: `va..va + LEN` stays mapped and readable until
    // the buffer's deallocator runs, as the mapping is deallocated only after it ran (below) and
    // the remap replaces its pages in one call, never leaving it unmapped. What
    // `HostMemory::new` asks beyond that, under the Vulkan runtime's buffer, does not: the range
    // keeps its backing and is never remapped meanwhile. The remap below breaks that on purpose,
    // while the buffer lives, as it is what this test measures. Nothing touches freed memory for
    // it: the pages the buffer was made over stay held by the GPU's import where it took them
    // (the GPU still reads and writes them, the measurement), the test reaches the range only
    // through `va`'s raw pointer, and the platforms make no CPU access to the buffer's bytes here
    // (no texture over it, no `didModifyRange:`).
    let range = unsafe { HostRange::new(va, LEN) };
    let buffer = dev
        .new_buffer_over_host_memory(OnFault::Return, range, 0, Some(deallocator))
        .unwrap()
        .unwrap();
    before(&buffer);
    let new = region(&vm, 0xe5);
    vm.remap_shared_overwrite(va, new, LEN).unwrap();
    assert_eq!(ends(va), [0xe5; 2], "the CPU sees the new pages");
    let read = gpu_read_then_fill::<P>(dev, &buffer, 0x7c);
    let seen = [read, ends(va), ends(new)];
    drop(buffer);
    assert_eq!(
        gone.recv_timeout(Duration::from_secs(10)),
        Ok(()),
        "the buffer's deallocator ran"
    );
    for mapping in [va, new] {
        assert!(vm.deallocate(mapping, LEN).is_success());
    }
    seen
}

/// What the GPU sees in each order ([`remapped`]): a remap over a reservation never touched and
/// over touched pages (`0xa1`), both after the buffer was made and before its first GPU use,
/// and one after that use (`0xc3`).
fn every_order<P: Platform>(dev: &P::Device) -> [[[u8; 2]; 3]; 3] {
    let vm = OsVm::new();
    [
        remapped::<P>(dev, vm.allocate_anywhere(LEN).unwrap(), |_| ()),
        remapped::<P>(dev, region(&vm, 0xa1), |_| ()),
        remapped::<P>(dev, region(&vm, 0xc3), |b| {
            assert_eq!(gpu_read_then_fill::<P>(dev, b, 0xc3), [0xc3; 2]);
        }),
    ]
}

/// The GPU reads the old pages (`old`), and fills them where the range no longer shows them.
fn not_seen(old: u8) -> [[u8; 2]; 3] {
    [[old; 2], [0xe5; 2], [0xe5; 2]]
}

/// The GPU reads the new pages, and fills them, which the range and the new mapping show.
const SEEN: [[u8; 2]; 3] = [[0xe5; 2], [0x7c; 2], [0x7c; 2]];

/// What every order of [`every_order`] answers, on Metal and on this crate's platform: the old
/// pages.
fn kept() -> [[[u8; 2]; 3]; 3] {
    [not_seen(0), not_seen(0xa1), not_seen(0xc3)]
}

#[test]
fn the_vulkan_runtime_keeps_a_no_copy_buffers_pages_from_its_creation() {
    let dev = VulkanPlatform::system_default_device().unwrap();
    assert_eq!(every_order::<VulkanPlatform>(&dev), kept());
}

/// A no-copy buffer over `va..va + LEN` whose deallocator says so on the answer's receiver.
fn no_copy<P: Platform>(dev: &P::Device, va: HostPtr) -> (P::Buffer, mpsc::Receiver<()>) {
    let (released, gone) = mpsc::channel();
    let deallocator: NoCopyDeallocator = Box::new(move || {
        let _ = released.send(());
    });
    // SAFETY: `va..va + LEN` stays mapped and readable until the buffer's deallocator runs: each
    // caller deallocates it only after that, and a remap replaces its pages in one call, never
    // leaving it unmapped. Under the Vulkan runtime's buffer the range's pages may change only
    // where the runtime is told before a buffer over them is made, which these tests break on
    // purpose in their controls; nothing touches freed memory for it, as in `remapped`.
    let range = unsafe { HostRange::new(va, LEN) };
    let buffer = dev
        .new_buffer_over_host_memory(OnFault::Return, range, 0, Some(deallocator))
        .unwrap()
        .unwrap();
    (buffer, gone)
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

/// What the GPU sees of a buffer B made over `va` (`LEN` bytes of `0xa1`) after `remap` put other
/// pages of `0xe5` there, while a buffer A made over the same bytes before, which the GPU read
/// and then filled with `0xc3`, lives: B's first and last bytes copied out, then, after the GPU
/// fills B with `0x7c`, those of `va` and of the new pages (the answer's second).
fn made_after<P: Platform>(
    dev: &P::Device,
    va: HostPtr,
    remap: impl FnOnce() -> HostPtr,
) -> [[u8; 2]; 3] {
    let (a, a_gone) = no_copy::<P>(dev, va);
    assert_eq!(gpu_read_then_fill::<P>(dev, &a, 0xc3), [0xa1; 2]);
    let new = remap();
    assert_eq!(ends(va), [0xe5; 2], "the CPU sees the new pages");
    let (b, b_gone) = no_copy::<P>(dev, va);
    let seen = [gpu_read_then_fill::<P>(dev, &b, 0x7c), ends(va), ends(new)];
    drop((b, a));
    released(&[b_gone, a_gone]);
    assert!(OsVm::new().deallocate(new, LEN).is_success());
    seen
}

/// [`made_after`] a remap through `vm`, a Mach VM.
fn made_after_a_remap<P: Platform>(dev: &P::Device, vm: &dyn HostVm) -> [[u8; 2]; 3] {
    let va = region(vm, 0xa1);
    let seen = made_after::<P>(dev, va, || {
        let new = region(vm, 0xe5);
        vm.remap_shared_overwrite(va, new, LEN).unwrap();
        new
    });
    assert!(vm.deallocate(va, LEN).is_success());
    seen
}

#[test]
fn a_buffer_made_after_a_remap_the_runtime_was_told_of_takes_the_new_pages() {
    let dev = VulkanPlatform::system_default_device().unwrap();
    let p = VulkanPlatform::new();
    assert_eq!(made_after_a_remap::<VulkanPlatform>(&dev, p.vm()), SEEN);
    // Untold, as through Mach VM of the test's own, the buffer lies in the live one's import: the
    // GPU reads what A wrote, and fills the old pages.
    assert_eq!(
        made_after_a_remap::<VulkanPlatform>(&dev, &OsVm::new()),
        not_seen(0xc3)
    );
}

/// A no-copy buffer whose deallocator deallocates its range through the platform's Mach VM, which
/// tells the runtime: its last release by the test's thread, and by the runtime's once the GPU
/// ran a fill of it that waited for the test's thread to release its own reference first. Each
/// deallocator runs to its end. Another no-copy buffer, over other memory the GPU used, lives
/// throughout, so that each report looks the runtime's noted memory up under its lock, as with
/// none noted the runtime returns before it: the deallocator runs under no lock the report takes.
#[test]
fn a_deallocator_may_deallocate_through_the_platforms_vm() {
    let dev = VulkanPlatform::system_default_device().unwrap();
    let p = Arc::new(VulkanPlatform::new());
    let queue = dev.new_command_queue().unwrap();
    let other = region(p.vm(), 0x11);
    let (live, live_gone) = no_copy::<VulkanPlatform>(&dev, other);
    assert_eq!(
        gpu_read_then_fill::<VulkanPlatform>(&dev, &live, 0x22),
        [0x11; 2]
    );
    for by_the_gpu in [false, true] {
        let va = region(p.vm(), 0x5a);
        let (done, gone) = mpsc::channel();
        let platform = Arc::clone(&p);
        let deallocator: NoCopyDeallocator = Box::new(move || {
            let _ = done.send(platform.vm().deallocate(va, LEN).is_success());
        });
        // SAFETY: `va..va + LEN` stays mapped, its pages unchanged, until the buffer's
        // deallocator runs, which deallocates it.
        let range = unsafe { HostRange::new(va, LEN) };
        let buffer = dev
            .new_buffer_over_host_memory(OnFault::Return, range, 0, Some(deallocator))
            .unwrap()
            .unwrap();
        assert_eq!(
            gpu_read_then_fill::<VulkanPlatform>(&dev, &buffer, 0x6b),
            [0x5a; 2]
        );
        if by_the_gpu {
            let event = dev.new_shared_event().unwrap();
            let cb = queue.command_buffer(OnFault::Return).unwrap().unwrap();
            cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
            let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            for cmd in [
                BlitCmd::FillBuffer {
                    buffer: &buffer,
                    range: NsRange::new(0, LEN),
                    value: 0x7d,
                },
                BlitCmd::EndEncoding,
            ] {
                e.encode(OnFault::Return, cmd).unwrap();
            }
            cb.commit(OnFault::Return).unwrap();
            drop((e, buffer));
            assert!(
                gone.try_recv().is_err(),
                "the command buffer holds the buffer"
            );
            event.set_signaled_value(1).unwrap();
            // Released by its completion, with the command buffer still held here.
            assert_eq!(gone.recv_timeout(Duration::from_secs(10)), Ok(true));
            cb.wait_until_completed(OnFault::Return).unwrap();
        } else {
            drop(buffer);
            assert_eq!(gone.recv_timeout(Duration::from_secs(10)), Ok(true));
        }
    }
    drop(live);
    released(&[live_gone]);
    assert!(p.vm().deallocate(other, LEN).is_success());
}
