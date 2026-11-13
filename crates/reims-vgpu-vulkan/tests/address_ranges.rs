//! Heaps over address ranges, as a heap over the guest's pages is made, on this crate's
//! platform against Metal's answers, measured on `AppleM3Max-40c-25G229`: which ranges make one,
//! past the heap's size ([`past_the_size`]) and over pages without access, unmapped or read-only
//! ([`without_access`]), and ranges off whole pages ([`unaligned`]): whether the heap is made,
//! and what `contents`, a GPU copy and a GPU fill of a buffer placed over them reach.

#![cfg(target_os = "macos")]
// Compares with Metal's answers, recorded on macOS.
// Every platform object is `Send` and `Sync` (core's `Handle`, and `Obj`'s bound, which an
// encoder's drop asks of `PredicatedEncoder`), and the proof walks the object's fields: an
// encoder's command buffer, its predication, and the runtime objects that holds, 141 to 150
// levels deep (measured: rustc warns at a limit of 140, not at 150). That is deeper than
// rustc's default of 128, as in the library (`src/lib.rs`).
#![recursion_limit = "256"]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    missing_docs,
    unsafe_code
)]

use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::core::host::HostPtr;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer, CommandQueue};
use reims_vgpu_vulkan::core::platform::device::GpuDevice;
use reims_vgpu_vulkan::core::platform::os::vm::{HostVm, VmProt};
use reims_vgpu_vulkan::core::platform::resource::{Buffer, Heap};
use reims_vgpu_vulkan::core::platform::types::{NsRange, ResourceAddressRange};
use reims_vgpu_vulkan::core::platform::values::{HeapDesc, OptionSet};
use reims_vgpu_vulkan::core::raise::OnFault;
use reims_vgpu_vulkan::remap::VulkanVm;

const PAGE: u64 = 0x4000;

/// Fresh pages of the host, each byte `byte`.
fn pages(count: u64, byte: u8) -> HostPtr {
    let at = VulkanVm::new().allocate_anywhere(count * PAGE).unwrap();
    // SAFETY: the pages were just allocated, readable and writable.
    unsafe {
        std::ptr::write_bytes(
            at.as_ptr::<u8>(),
            byte,
            usize::try_from(count * PAGE).unwrap(),
        );
    };
    at
}

/// Word `index` of the 32-bit words at `at`.
fn word(at: HostPtr, index: usize) -> u32 {
    // SAFETY: the test's own mapped memory.
    unsafe { at.as_ptr::<u32>().add(index).read_volatile() }
}

fn range(at: HostPtr, offset: u64, length: u64) -> ResourceAddressRange {
    ResourceAddressRange {
        address: u64::try_from(at.addr()).unwrap() + offset,
        length,
    }
}

/// What the platform does, step by step: each step's name and what it answered or showed.
type Steps = Vec<(&'static str, Vec<u64>)>;

/// A blit encoder of `cb` with `commands`, ended.
fn blit<P: Platform>(cb: &P::CommandBuffer, commands: Vec<BlitCmd<'_, P>>) {
    let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    for command in commands {
        e.encode(OnFault::Return, command).unwrap();
    }
    e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
}

/// `cb` committed and waited for: its status and its error's code (0 for none).
fn finish<P: Platform>(cb: &P::CommandBuffer) -> Vec<u64> {
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    vec![
        cb.status().0,
        cb.error().map_or(0, |e| u64::try_from(e.code).unwrap()),
    ]
}

/// A command buffer of `queue`.
fn command_buffer<P: Platform>(queue: &P::CommandQueue) -> P::CommandBuffer {
    queue.command_buffer(OnFault::Return).unwrap().unwrap()
}

/// Frees `length` bytes of the test's pages at `at`.
fn vm_free(at: HostPtr, length: u64) -> bool {
    VulkanVm::new().deallocate(at, length).is_success()
}

/// Two fresh pages of the host after `case`: 0 leaves them readable and writable, 1 takes all
/// access from the second, 2 from both, 3 unmaps the second, 4 leaves both read-only.
fn pages_after(case: u8) -> HostPtr {
    let vm = VulkanVm::new();
    let at = pages(2, 0x5a);
    let second = at.wrapping_add(PAGE);
    match case {
        1 => vm.protect_max(second, PAGE, VmProt(0)).unwrap(),
        2 => vm.protect_max(at, 2 * PAGE, VmProt(0)).unwrap(),
        3 => assert!(vm.deallocate(second, PAGE).is_success()),
        4 => vm.protect_max(at, 2 * PAGE, VmProt::READ).unwrap(),
        _ => {}
    }
    at
}

/// Lets go of what [`pages_after`] left mapped of its two pages at `at` for `case`: both
/// pages, but only the first after case 3, whose second page it already unmapped. Unmapping
/// that second page again would unmap whatever the process has mapped there since, as a new
/// thread's malloc cache or the GPU driver's shared memory, and the process's next touch of it
/// faults.
fn let_go_pages_after(case: u8, at: HostPtr) {
    let length = if case == 3 { PAGE } else { 2 * PAGE };
    let _ = VulkanVm::new().deallocate(at, length);
}

/// The heaps cases' options: Shared, and read-only (resource option bit 24) or not.
fn heap_options(read_only: bool) -> Vec<OptionSet> {
    let mut options = vec![OptionSet::StorageMode(0)];
    if read_only {
        options.push(OptionSet::ResourceOptionsOr(0x100_0000));
    }
    options
}

/// What a platform makes over the two pages of each of [`pages_after`]'s cases: a heap over
/// their address range, read-only or not; 1 for each made, by case.
fn without_access<P: Platform>(p: &P, dev: &P::Device) -> Steps {
    let cases = [0, 1, 2, 3, 4];
    let mut steps: Steps = Vec::new();
    for read_only in [false, true] {
        let heaps = cases.map(|case| {
            let at = pages_after(case);
            let array = p
                .new_address_range_array(&[range(at, 0, 2 * PAGE)])
                .unwrap();
            let made = dev
                .new_heap(&HeapDesc {
                    size: 2 * PAGE,
                    heap_type: 1,
                    options: heap_options(read_only),
                    address_ranges: Some(&array),
                    pinned_gpu_address: None,
                })
                .unwrap();
            let made = u64::from(made.heap.is_some());
            let_go_pages_after(case, at);
            made
        });
        steps.push((
            if read_only {
                "read-only heaps"
            } else {
                "heaps"
            },
            heaps.to_vec(),
        ));
    }
    steps
}

/// What Metal makes over pages without access, measured on
/// `AppleM3Max-40c-25G229`: nothing over a page mapped without access or unmapped, as over
/// read-only pages what it makes over readable and writable ones.
fn expected_without_access() -> Steps {
    vec![
        ("heaps", vec![1, 0, 0, 0, 1]),
        ("read-only heaps", vec![1, 0, 0, 0, 1]),
    ]
}

/// What a platform makes as a heap of one page over ranges of two pages, read-only or not, the
/// second page past its size: one range over both of [`pages_after`]'s pages, then one range
/// of a page each, with the second readable and writable, without access, and unmapped (its
/// cases 0, 1 and 3), and a first range followed by a NULL one; 1 for each made.
fn past_the_size<P: Platform>(p: &P, dev: &P::Device) -> Steps {
    let vm = VulkanVm::new();
    let mut steps: Steps = Vec::new();
    for read_only in [false, true] {
        let mut made = Vec::new();
        for split in [false, true] {
            for case in [0, 1, 3] {
                let at = pages_after(case);
                let ranges = match split {
                    false => vec![range(at, 0, 2 * PAGE)],
                    true => vec![range(at, 0, PAGE), range(at, PAGE, PAGE)],
                };
                made.push(heap_over::<P>(p, dev, &ranges, read_only));
                let_go_pages_after(case, at);
            }
        }
        let at = pages(1, 0x5a);
        let null = ResourceAddressRange {
            address: 0,
            length: PAGE,
        };
        made.push(heap_over::<P>(
            p,
            dev,
            &[range(at, 0, PAGE), null],
            read_only,
        ));
        let _ = vm.deallocate(at, PAGE);
        let name = match read_only {
            false => "heaps past the size",
            true => "read-only heaps past the size",
        };
        steps.push((name, made));
    }
    steps
}

/// 1 where a platform makes a Shared heap of one page over `ranges`, read-only or not, else 0.
fn heap_over<P: Platform>(
    p: &P,
    dev: &P::Device,
    ranges: &[ResourceAddressRange],
    read_only: bool,
) -> u64 {
    let array = p.new_address_range_array(ranges).unwrap();
    let made = dev
        .new_heap(&HeapDesc {
            size: PAGE,
            heap_type: 1,
            options: heap_options(read_only),
            address_ranges: Some(&array),
            pinned_gpu_address: None,
        })
        .unwrap();
    u64::from(made.heap.is_some())
}

/// What Metal makes past a heap's size, measured on `AppleM3Max-40c-25G229`: as
/// within it, nothing where a page of the ranges past it is mapped without access or unmapped,
/// in one range or a range of its own, and nothing after a NULL range.
fn expected_past_the_size() -> Steps {
    vec![
        ("heaps past the size", vec![1, 0, 0, 1, 0, 0, 0]),
        ("read-only heaps past the size", vec![1, 0, 0, 1, 0, 0, 0]),
    ]
}

/// This crate in [`past_the_size`], in a child copy of this test binary under strict mode
/// (`REIMS_VGPU_STRICT=1`), where a death ends the process as in production: Metal's heaps, and
/// nothing dies. The read-only ones (resource option bit 24) are made as Metal makes them: the
/// runtime imports their pages through a read-only view of them, which drops the GPU's writes as
/// Metal's memory of the option does
#[test]
fn pages_past_the_size_on_vulkan() {
    const CHILD: &str = "RVG_PAGES_PAST_THE_SIZE";
    if std::env::var_os(CHILD).is_some() {
        reims_vgpu_vulkan::core::testkit::capture::set_capture_enabled(false);
        let dev = VulkanPlatform::system_default_device().unwrap();
        let p = VulkanPlatform::new();
        println!("made {:?}", past_the_size(&p, &dev));
        std::process::exit(0);
    }
    if VulkanPlatform::system_default_device().is_none() {
        return;
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "pages_past_the_size_on_vulkan", "--nocapture"])
        .args(["--test-threads=1"])
        .env(CHILD, "1")
        .env("REIMS_VGPU_STRICT", "1")
        .output()
        .unwrap();
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let death = stderr
        .lines()
        .find(|l| l.starts_with("reims-vgpu: fatal: "));
    assert!(out.status.success() && death.is_none(), "{out:?}");
    let metal = expected_past_the_size();
    assert!(stdout.contains(&format!("made {metal:?}")), "{stdout}");
}

/// This crate in [`without_access`], in a child copy of this test binary under strict mode
/// (`REIMS_VGPU_STRICT=1`), where a death ends the process as in production: Metal's heaps, and
/// nothing dies. The read-only ones (resource option bit 24) are made as Metal makes them: the
/// runtime imports their pages through a read-only view of them, which drops the GPU's writes as
/// Metal's memory of the option does
#[test]
fn pages_without_access_on_vulkan() {
    const CHILD: &str = "RVG_PAGES_WITHOUT_ACCESS";
    if std::env::var_os(CHILD).is_some() {
        reims_vgpu_vulkan::core::testkit::capture::set_capture_enabled(false);
        let dev = VulkanPlatform::system_default_device().unwrap();
        let p = VulkanPlatform::new();
        println!("made {:?}", without_access(&p, &dev));
        std::process::exit(0);
    }
    if VulkanPlatform::system_default_device().is_none() {
        return;
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "pages_without_access_on_vulkan", "--nocapture"])
        .args(["--test-threads=1"])
        .env(CHILD, "1")
        .env("REIMS_VGPU_STRICT", "1")
        .output()
        .unwrap();
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let death = stderr
        .lines()
        .find(|l| l.starts_with("reims-vgpu: fatal: "));
    assert!(out.status.success() && death.is_none(), "{out:?}");
    let metal = expected_without_access();
    assert!(stdout.contains(&format!("made {metal:?}")), "{stdout}");
}

/// How many shapes [`unaligned_shape`] holds.
const UNALIGNED: usize = 8;

/// The address ranges off whole pages [`unaligned`] takes, by index, over guest pages whose
/// 32-bit words hold their own offset there, top bit set, and the pages of the heap or buffer
/// over them:
/// 0. one range from 0x40 into a page, of two pages;
/// 1. one from 0x1000 into a page, of two pages;
/// 2. a range of 0x1000 bytes before one of two pages;
/// 3. a range of a page and 0x40 bytes before one of a page;
/// 4. a range of a page before one of a page from 0x40 into its page;
/// 5. the same before a third range, of a page, over three pages: whether the second takes the
///    pages its bytes lie in (pages 2 and 3, the third range past the size) or its length rounded
///    up (page 2, then the third range's page 5);
/// 6. a range of a page and 0x40 bytes before one whose bytes follow on from it, to the end of
///    page 2, over three pages: whether the two are one run of bytes (pages 0 to 2) or each takes
///    its own pages (pages 0 and 1, then 1 again);
/// 7. a range of a page, one of 0x1000 bytes from a page's start and one of a page, over three
///    pages.
fn unaligned_shape(guest: HostPtr, index: usize) -> (Vec<ResourceAddressRange>, u64) {
    let r = |offset, length| range(guest, offset, length);
    match index {
        0 => (vec![r(0x40, 2 * PAGE)], 2),
        1 => (vec![r(0x1000, 2 * PAGE)], 2),
        2 => (vec![r(0, 0x1000), r(2 * PAGE, 2 * PAGE)], 2),
        3 => (vec![r(0, PAGE + 0x40), r(3 * PAGE, PAGE)], 2),
        4 => (vec![r(0, PAGE), r(2 * PAGE + 0x40, PAGE)], 2),
        5 => (
            vec![r(0, PAGE), r(2 * PAGE + 0x40, PAGE), r(5 * PAGE, PAGE)],
            3,
        ),
        6 => (vec![r(0, PAGE + 0x40), r(PAGE + 0x40, 2 * PAGE - 0x40)], 3),
        _ => (vec![r(0, PAGE), r(2 * PAGE, 0x1000), r(4 * PAGE, PAGE)], 3),
    }
}

/// The guest pages [`unaligned`] lies over.
const UNALIGNED_PAGES: u64 = 8;

/// What `P` does with the shape `index` ([`unaligned_shape`]) as a placement heap's address
/// ranges: whether it is made; then, of a buffer of the shape's pages placed at 0 in it, the
/// words `contents` reads and those a GPU copy reads at the start, 0x40 and 0x1000 into each page
/// and its last word, and the runs of the guest's words a GPU fill of the whole buffer reached,
/// as guest offsets.
fn unaligned<P: Platform>(p: &P, dev: &P::Device, index: usize) -> Steps {
    let guest = pages(UNALIGNED_PAGES, 0);
    for at in (0..UNALIGNED_PAGES * PAGE).step_by(4) {
        // SAFETY: a word of the test's own pages, which no GPU work uses yet.
        unsafe {
            guest
                .wrapping_add(at)
                .as_ptr::<u32>()
                .write(0x8000_0000 | u32::try_from(at).unwrap());
        }
    }
    let (ranges, count) = unaligned_shape(guest, index);
    let array = p.new_address_range_array(&ranges).unwrap();
    let length = count * PAGE;
    let mut steps: Steps = Vec::new();
    let made = dev
        .new_heap(&HeapDesc {
            size: length,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(0)],
            address_ranges: Some(&array),
            pinned_gpu_address: None,
        })
        .unwrap();
    steps.push(("heap made", vec![u64::from(made.heap.is_some())]));
    let Some(heap) = made.heap else {
        assert!(vm_free(guest, UNALIGNED_PAGES * PAGE));
        return steps;
    };
    let buffer = heap.new_buffer_of_length(length, 0, 0).unwrap();
    steps.push(("placed", vec![u64::from(buffer.is_some())]));
    let Some(buffer) = buffer else {
        drop(heap);
        assert!(vm_free(guest, UNALIGNED_PAGES * PAGE));
        return steps;
    };
    let offsets: Vec<u64> = (0..count)
        .flat_map(|page| [0, 0x40, 0x1000, PAGE - 4].map(|at| page * PAGE + at))
        .collect();
    let words = |at: HostPtr| -> Vec<u64> {
        offsets
            .iter()
            .map(|&offset| u64::from(word(at.wrapping_add(offset), 0)))
            .collect()
    };
    steps.push(("contents", words(buffer.contents())));
    let out = dev
        .new_buffer_of_length(OnFault::Return, length, 0)
        .unwrap()
        .unwrap();
    let queue = dev.new_command_queue().unwrap();
    let cb = command_buffer::<P>(&queue);
    let copy = BlitCmd::CopyBufferToBuffer {
        source_buffer: Some(&buffer),
        source_offset: 0,
        destination_buffer: Some(&out),
        destination_offset: 0,
        size: length,
    };
    blit::<P>(&cb, vec![copy]);
    steps.push(("copy", finish::<P>(&cb)));
    steps.push(("the GPU read", words(out.contents())));
    let cb = command_buffer::<P>(&queue);
    let fill = BlitCmd::FillBuffer {
        buffer: &buffer,
        range: NsRange::new(0, length),
        value: 0x5c,
    };
    blit::<P>(&cb, vec![fill]);
    steps.push(("fill", finish::<P>(&cb)));
    let mut runs: Vec<u64> = Vec::new();
    for at in (0..UNALIGNED_PAGES * PAGE).step_by(4) {
        let filled = word(guest.wrapping_add(at), 0) == 0x5c5c_5c5c;
        match (filled, runs.len() % 2) {
            (true, 0) | (false, 1) => runs.push(at),
            _ => {}
        }
    }
    if runs.len() % 2 == 1 {
        runs.push(UNALIGNED_PAGES * PAGE);
    }
    steps.push(("filled guest runs", runs));
    drop((cb, queue, out, buffer, heap));
    assert!(vm_free(guest, UNALIGNED_PAGES * PAGE));
    steps
}

/// Address ranges off whole pages ([`unaligned_shape`]), as a heap's, on this platform as on
/// Metal: shapes 2 to 7 alike, step by step. A first range starting off its page (shapes 0 and 1)
/// dies at the creator here, where Metal's `contents` answers its bytes from the range's own
/// address and its GPU reads them from its page's start (`src/gpu/backing.rs`'s
/// `off_the_page`), which Metal's first words show
#[test]
fn unaligned_ranges_on_vulkan_are_metals() {
    use reims_vgpu_vulkan::core::raise::death::DeathKind;
    use reims_vgpu_vulkan::core::testkit::capture::catch_death;
    let dev = VulkanPlatform::system_default_device().unwrap();
    let p = VulkanPlatform::new();
    for index in 0..UNALIGNED {
        let case = format!("heap {index}");
        if index < 2 {
            let site = catch_death(|| unaligned(&p, &dev, index)).unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, "newHeapWithDescriptor:"),
                "{case}"
            );
            continue;
        }
        let ours = unaligned(&p, &dev, index);
        reims_vgpu_vulkan::metal_check!(&case, ours);
    }
}
