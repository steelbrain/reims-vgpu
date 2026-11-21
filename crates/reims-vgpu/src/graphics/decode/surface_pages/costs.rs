//! What a new IOSurface's view of its pages costs the thread that makes it, measured on a real
//! platform (ignored: a measurement, not a check). Run it alone, holding the host's GPU locks:
//!
//! ```sh
//! cargo nextest run -p reims-vgpu -E 'test(/surface_view_costs/)' \
//!   --run-ignored only --no-capture
//! ```
//!
//! The guest's RAM is what QEMU gives the device: on Linux a shared mapping of a memfd, whose file
//! the platform's lease registers; on Windows a view of a pagefile-backed section, whose handle it
//! registers; on macOS anonymous memory. Each surface is fresh pages, as a browser's are: its runs
//! are slots of the RAM picked at random, so no two surfaces name the same runs and the cache of
//! views takes none. Each step is timed on the thread that carries it out (the surface worker, for
//! `SURFACE_PAGES`):
//! - `SURFACE_PAGES`: the view (a range mapping of the runs) and the buffer over all of it; for a
//!   cached block also the device cache;
//! - `BUFFER_OVER_SURFACE_PAGES`, its `OBJECT_DESTROY`, and the block's `BLOCK_RELEASE`;
//! - and apart, the same steps one at a time: the range mapping, the first buffer over it, a
//!   second, and each one's drop.
//!
//! `RGK_BENCH_ITERS` surfaces a case (default 40, past the cache's 32 views, so its drops are
//! counted), `RGK_BENCH_RAM_MIB` MiB of RAM (default 768), `RGK_BENCH_THP=1` asks for huge pages
//! of the memfd, and `RGK_BENCH_CASES` a comma list of the cases' names to run.
//!
//! **A pages slab's take** (`pages_take_costs_*`, run as above with `test(/pages_take_costs/)`):
//! what each of a process's 2 MiB pages slabs costs the thread that carries its messages out, as
//! the bundle's pages slab source sends them (`RgfxPagesSlabSource`): the pages block's
//! `SURFACE_PAGES` of its plain pages' runs, the `HEAP_OVER_PAGES` over it, which makes the
//! block's view (a mapping of each run) and its import where the block is more than one run, and
//! a first `BUFFER_AT_OFFSET` in the heap. The guest's next take waits for them: its `PagesBlock`
//! call is answered after the messages before it. The RAM is fresh, as a booted guest's is where
//! no process was yet: only each slab's own pages are written, as the guest's zero fill writes
//! them, just before its take, and the rest of the RAM is never touched. A slab's 512 guest pages
//! are taken from the RAM's pages in three orders, which the kext lists as runs, a page right
//! after the one before continuing its run: upwards (each page after the one before: one run, as
//! a kernel with free RAM in one piece gives them), downwards (each page right below the one
//! before: 512 runs of a page, as a kernel's free list may give them) and scattered (pages picked
//! at random over the RAM: 512 runs). `RGK_BENCH_TAKES` slabs an order
//! (default 256, 512 MiB), `RGK_BENCH_RAM_MIB` the RAM (at least twice the takes'),
//! `RGK_BENCH_TAKE_GAP_US` a pause before each take (default 0, a burst), and `RGK_BENCH_ORDERS` a
//! comma list of the orders' names to run.
//!
//! **A large pages slab's take** (`large_take_costs_*`, `test(/large_take_costs/)`): what a large
//! Shared buffer of the process's own plain pages costs, as the bundle sends it: the pages block's
//! `SURFACE_PAGES` of its pages' runs, then a `BUFFER_OVER_PAGES` over the whole block, which
//! makes the block's view (a mapping of each run) and the import of it, and the buffer's destroy
//! and the block's release. The pages are taken in the three orders above, so a slab is one run
//! (upwards) or one run a page (downwards, scattered). `RGK_BENCH_LARGE_MIB` MiB a slab (default
//! 64), `RGK_BENCH_LARGE_TAKES` slabs an order (default 8), and `RGK_BENCH_ORDERS` as above.

use std::any::Any;
use std::sync::Arc;
use std::time::{Duration, Instant};

use reims_vgpu_graphics::wire::{
    RGFX_CMD_BUFFER_AT_OFFSET, RGFX_CMD_BUFFER_OVER_PAGES, RGFX_CMD_BUFFER_OVER_SURFACE_PAGES,
    RGFX_CMD_HEAP_OVER_PAGES, RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_SURFACE_PAGES,
    RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN, RGFX_STATUS_OK, RGFX_SURFACE_PAGES_CACHED,
    RGFX_SURFACE_PAGES_PRIVATE, RgfxBufferAtOffset, RgfxBufferOverPages,
    RgfxBufferOverSurfacePages, RgfxHeapOverPages, RgfxMsgHeader, RgfxObjectDestroy, RgfxPageRun,
    RgfxSurfacePages,
};
use reims_vgpu_vulkan::core::host::PhysRange;
use reims_vgpu_vulkan::core::memory::map::MemoryMap as CoreMemoryMap;
use reims_vgpu_vulkan::core::num::{ToU64 as _, ToUsize as _};
use zerocopy::IntoBytes;

use super::super::{GraphicsDecoder, no_raise};
use super::{GUEST_PAGE, Lease, SurfaceView, buffer_over};
use crate::memory::MemoryMap;
use crate::platform::ApiPlatform;

/// The block every surface's messages name, one at a time.
const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x4000;

/// The guest physical address of the RAM's first byte.
pub(in crate::graphics::decode) const RAM: u64 = 0x1_0000_0000;

const MIB: u64 = 1 << 20;

/// One case: a surface of `length` bytes in runs of `run` bytes.
struct Case {
    name: &'static str,
    length: u64,
    run: u64,
}

/// The surfaces measured: a 4K BGRA layer (the msi panel's 3840x2400), a tile of 1 and of 4 MiB,
/// and a 48 MP image's 8000x6000 BGRA pages, each in runs of one guest page, and the layer also in
/// runs of 64 KiB and of 2 MiB.
fn cases(page: u64) -> Vec<Case> {
    let layer = (3840 * 2400 * 4u64).next_multiple_of(page);
    let image = (8000 * 6000 * 4u64).next_multiple_of(page);
    vec![
        Case {
            name: "tile-1m",
            length: MIB,
            run: page,
        },
        Case {
            name: "tile-4m",
            length: 4 * MIB,
            run: page,
        },
        Case {
            name: "layer-4k",
            length: layer,
            run: page,
        },
        Case {
            name: "layer-64k",
            length: layer,
            run: (64 << 10u64).max(page),
        },
        Case {
            name: "layer-2m",
            length: layer,
            run: 2 * MIB,
        },
        Case {
            name: "image-48mp",
            length: image,
            run: page,
        },
    ]
}

pub(in crate::graphics::decode) fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A small PRNG, so each run picks the same surfaces.
pub(in crate::graphics::decode) struct XorShift(pub(in crate::graphics::decode) u64);

impl XorShift {
    pub(in crate::graphics::decode) fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// `length` bytes of the RAM's `ram` bytes as runs of `run` bytes (the last one shorter), each at
/// a slot of the RAM picked at random, in the order picked.
pub(in crate::graphics::decode) fn surface_runs(
    rng: &mut XorShift,
    ram: u64,
    length: u64,
    run: u64,
) -> Vec<RgfxPageRun> {
    let slots = ram / run;
    let count = length.div_ceil(run);
    assert!(
        count <= slots,
        "{length:#x} bytes in runs of {run:#x} do not fit the RAM"
    );
    let mut order: Vec<u64> = (0..slots).collect();
    for i in 0..count.to_usize() {
        let j = i + (rng.next() % (slots - i.to_u64())).to_usize();
        order.swap(i, j);
    }
    let mut left = length;
    order[..count.to_usize()]
        .iter()
        .map(|&slot| {
            let n = left.min(run);
            left -= n;
            RgfxPageRun {
                address: RAM + slot * run,
                length: n,
            }
        })
        .collect()
}

/// Durations of one step.
#[derive(Default)]
struct Step(Vec<Duration>);

impl Step {
    #[allow(clippy::disallowed_methods)] // a measurement of real platforms: the wall clock's time
    fn time<T>(&mut self, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let out = f();
        self.0.push(start.elapsed());
        out
    }

    fn line(&self, case: &str, step: &str) -> String {
        let mut v = self.0.clone();
        v.sort();
        let at = |q: f64| -> f64 {
            let i = ((v.len() as f64 - 1.0) * q).round() as usize;
            v.get(i).map_or(0.0, |d| d.as_secs_f64() * 1e3)
        };
        let sum: f64 = v.iter().map(Duration::as_secs_f64).sum::<f64>() * 1e3;
        format!(
            "{case:<11} {step:<26} n {:>3}  p50 {:>8.3}  p90 {:>8.3}  max {:>8.3}  sum {:>9.1} ms",
            v.len(),
            at(0.5),
            at(0.9),
            at(1.0),
            sum
        )
    }
}

/// The guest's RAM as the host maps it, and what keeps it: its mapping, which the leases on it
/// hold too ([`Ram::lease`]), so it is unmapped once the last of them and the RAM have gone, as
/// QEMU's RAM stays mapped while anything of the device's may name it.
pub(in crate::graphics::decode) struct Ram {
    pub(in crate::graphics::decode) map: MemoryMap,
    pub(in crate::graphics::decode) at: *mut u8,
    mapping: Arc<RamMapping>,
}

impl Ram {
    /// `held`, the platform's lease on the RAM (its `guest_memory` of [`Self::map`]), as the
    /// decoder takes it, holding the RAM's mapping too: the views made through it, and the
    /// platform's imports of the RAM, whose releases hold the lease or a view, keep the RAM
    /// mapped until the last of them went, however late the platform runs them. The platform's
    /// lease goes first ([`LeasedRam`]).
    pub(in crate::graphics::decode) fn lease(&self, held: Box<dyn Any + Send + Sync>) -> Lease {
        Arc::new(LeasedRam {
            held: Some(held),
            _mapping: Arc::clone(&self.mapping),
        })
    }
}

/// The RAM's own mapping: `length` bytes at `at`, unmapped as it goes.
struct RamMapping {
    at: usize,
    length: u64,
}

impl Drop for RamMapping {
    /// Unmaps the RAM, which the last of its holds, the RAM and the leases on it, let go of.
    fn drop(&mut self) {
        if self.at == 0 {
            return;
        }
        unmap(std::ptr::with_exposed_provenance_mut(self.at), self.length);
    }
}

/// The platform's lease on the RAM as the decoder takes it ([`Ram::lease`]).
struct LeasedRam {
    /// The platform's lease, taken as it goes.
    held: Option<Box<dyn Any + Send + Sync>>,
    _mapping: Arc<RamMapping>,
}

impl Drop for LeasedRam {
    /// Lets go of the platform's lease before its hold of the RAM's mapping, which may be the
    /// last: no registration of the platform's names the RAM's addresses once another mapping
    /// may take them.
    fn drop(&mut self) {
        drop(self.held.take());
    }
}

/// Unmaps the RAM's own mapping, which no device, view or import uses any more.
#[cfg(unix)]
fn unmap(at: *mut u8, length: u64) {
    // SAFETY: the RAM's own mapping, as the caller says.
    unsafe { libc::munmap(at.cast(), length.to_usize()) };
}

/// [`unmap`] on Windows: the RAM's view of its section.
#[cfg(windows)]
fn unmap(at: *mut u8, _length: u64) {
    use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
    // SAFETY: the RAM's own view, as the caller says.
    unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: at.cast() }) };
}

/// `length` bytes of guest RAM at [`RAM`], every page touched where `touched`, else none: a
/// memfd's shared mapping on Linux (with huge pages asked for where `RGK_BENCH_THP=1`), sealed as
/// QEMU's `memory-backend-memfd` seals it (`F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL`), since
/// udmabuf refuses a memfd not sealed against shrinking; on Windows a view of a pagefile-backed
/// section, as QEMU's Windows build backs the guest's RAM; anonymous memory on macOS.
pub(in crate::graphics::decode) fn guest_ram(length: u64, touched: bool) -> Ram {
    #[cfg(target_os = "linux")]
    let (at, file) = {
        use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
        // SAFETY: a NUL-terminated name; the descriptor is owned from here on.
        let fd = unsafe {
            libc::memfd_create(
                c"rvg-bench-ram".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        assert!(fd >= 0);
        // SAFETY: a descriptor just made, owned once.
        let fd = Arc::new(unsafe { OwnedFd::from_raw_fd(fd) });
        // SAFETY: sizing the file the test owns.
        assert_eq!(
            unsafe { libc::ftruncate(fd.as_raw_fd(), length as libc::off_t) },
            0
        );
        let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
        // SAFETY: sealing the file the test owns, sized just above.
        assert_eq!(
            unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals) },
            0
        );
        // SAFETY: a new shared mapping of the whole file, unmapped at the drop.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length.to_usize(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        assert_ne!(at, libc::MAP_FAILED);
        if env_u64("RGK_BENCH_THP", 0) == 1 {
            // SAFETY: advice on the mapping just made.
            unsafe { libc::madvise(at, length.to_usize(), libc::MADV_HUGEPAGE) };
        }
        let file = crate::memory::MemoryFile {
            file: fd,
            offset: 0,
        };
        (at.cast::<u8>(), file)
    };
    #[cfg(windows)]
    let (at, file) = {
        use std::os::windows::io::{AsRawHandle as _, FromRawHandle as _, OwnedHandle};

        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_ALL_ACCESS, MapViewOfFile, PAGE_READWRITE, SEC_COMMIT,
        };
        let (high, low) = ((length >> 32) as u32, length as u32);
        // SAFETY: a new pagefile-backed section of `length` bytes; its handle is owned from here
        // on.
        let section = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                high,
                low,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: the section's handle, just made, owned once.
        let section = Arc::new(unsafe { OwnedHandle::from_raw_handle(section) });
        // SAFETY: a view of the whole section, unmapped at the drop.
        let view = unsafe {
            MapViewOfFile(
                section.as_raw_handle(),
                FILE_MAP_ALL_ACCESS,
                0,
                0,
                length.to_usize(),
            )
        };
        assert!(!view.Value.is_null());
        let file = crate::memory::MemoryFile {
            file: section,
            offset: 0,
        };
        (view.Value.cast::<u8>(), file)
    };
    #[cfg(target_os = "macos")]
    let (at, file) = {
        // SAFETY: a new anonymous mapping, unmapped at the drop.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length.to_usize(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };
        assert_ne!(at, libc::MAP_FAILED);
        (at.cast::<u8>(), ())
    };
    if touched {
        // SAFETY: the mapping just made, `length` bytes, writable.
        unsafe { std::ptr::write_bytes(at, 0x5a, length.to_usize()) };
    }
    let mut map = MemoryMap::new();
    #[cfg(target_os = "linux")]
    {
        // SAFETY: the RAM stays mapped until the last of `Ram` and the leases on it goes
        // ([`Ram::lease`]), after every device, view and import made over it.
        unsafe { map.push_with_file(RAM, length, at, Some(file)) };
    }
    #[cfg(windows)]
    {
        // SAFETY: as above; the map holds the section.
        unsafe { map.push_with_file(RAM, length, at, Some(file)) };
    }
    #[cfg(target_os = "macos")]
    {
        let () = file;
        // SAFETY: as above.
        unsafe { map.push(RAM, length, at) };
    }
    let mapping = Arc::new(RamMapping {
        at: at.expose_provenance(),
        length,
    });
    Ram { map, at, mapping }
}

fn header(r#type: u32) -> RgfxMsgHeader {
    RgfxMsgHeader {
        r#type,
        size: 0,
        seqno: 0,
    }
}

/// Measures every case on `platform`, printing a line per step.
fn measure<P: ApiPlatform>(label: &str) {
    let platform = P::process();
    let device = platform.metal_device(None).expect("a GPU device");
    let page = platform.vm().page_size().max(GUEST_PAGE);
    let ram_bytes = env_u64("RGK_BENCH_RAM_MIB", 768) * MIB;
    let iters = env_u64("RGK_BENCH_ITERS", 40);
    let only = std::env::var("RGK_BENCH_CASES").ok();
    let ram = guest_ram(ram_bytes, true);
    let lease = ram.lease(platform.guest_memory(&ram.map));
    let core = CoreMemoryMap::new(Arc::clone(&platform), &ram.map.to_core()).expect("a memory map");
    let size = BLOCK + 0x10_0000 + RGFX_CURSOR_SPAN;
    let region = Arc::new(crate::platform::test_region(&*platform, size.to_usize()));
    let mut dec = GraphicsDecoder::<P>::new(Arc::clone(&platform), region);
    dec.set_guest_memory(Arc::clone(&core), Arc::clone(&lease));
    let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
    eprintln!(
        "{label}: {} MiB of guest RAM, {iters} surfaces a case",
        ram_bytes >> 20
    );
    let mut obj = 100u64;
    for case in cases(page) {
        if only
            .as_ref()
            .is_some_and(|o| !o.split(',').any(|n| n == case.name))
        {
            continue;
        }
        if case.length > ram_bytes / 2 {
            eprintln!("{label}: {} skipped: the RAM is too small", case.name);
            continue;
        }
        // Through the decoder, as ring 0's thread carries the messages out.
        for cached in [false, true] {
            let mut steps: [Step; 4] = Default::default();
            let mut refused = 0u64;
            for _ in 0..iters {
                let runs = surface_runs(&mut rng, ram_bytes, case.length, case.run);
                let pages = RgfxSurfacePages {
                    header: header(RGFX_CMD_SURFACE_PAGES),
                    context: 0,
                    run_count: runs.len() as u32,
                    block_offset: BLOCK,
                    block_size: 0x2000,
                    runs_offset: BLOCK + 0x1000,
                    length: case.length,
                };
                let flags = if cached { RGFX_SURFACE_PAGES_CACHED } else { 0 };
                let status = steps[0].time(|| dec.surface_pages(&device, &pages, &runs, flags));
                if status != RGFX_STATUS_OK {
                    refused += 1;
                    continue;
                }
                obj += 1;
                let over = RgfxBufferOverSurfacePages {
                    header: header(RGFX_CMD_BUFFER_OVER_SURFACE_PAGES),
                    obj_id: obj,
                    block_offset: BLOCK,
                    block_size: 0x1000,
                    length: case.length,
                    options: 0,
                    reserved_34: 0,
                };
                let made = steps[1].time(|| dec.decode(&device, over.as_bytes()));
                assert_eq!(
                    made.ok(),
                    Some(Ok(())),
                    "{}: the buffer over its pages",
                    case.name
                );
                let destroy = RgfxObjectDestroy {
                    header: header(RGFX_CMD_OBJECT_DESTROY),
                    obj_id: obj,
                };
                let gone = steps[2].time(|| dec.decode(&device, destroy.as_bytes()));
                assert_eq!(gone.ok(), Some(Ok(())));
                steps[3].time(|| {
                    dec.block_released(&device, BLOCK, 0, Box::new(|| ()), no_raise());
                });
            }
            let mode = if cached { "cached" } else { "plain" };
            if refused != 0 {
                eprintln!(
                    "{label}: {} {mode}: {refused} SURFACE_PAGES refused",
                    case.name
                );
            }
            for (step, name) in steps.iter().zip([
                "SURFACE_PAGES",
                "BUFFER_OVER_SURFACE_PAGES",
                "OBJECT_DESTROY",
                "BLOCK_RELEASE",
            ]) {
                eprintln!(
                    "{label}: {}",
                    step.line(case.name, &format!("{mode} {name}"))
                );
            }
        }
        // The same steps one at a time, the views kept as the cache keeps them.
        let mut steps: [Step; 6] = Default::default();
        let mut kept = std::collections::VecDeque::new();
        for _ in 0..iters {
            let runs = surface_runs(&mut rng, ram_bytes, case.length, case.run);
            let ranges: Vec<PhysRange> = runs
                .iter()
                .map(|r| PhysRange {
                    physical_address: r.address,
                    physical_length: r.length,
                })
                .collect();
            let Some(mapping) = steps[0].time(|| core.new_range_mapping(&ranges, case.length, 0))
            else {
                eprintln!("{label}: {}: a range mapping refused", case.name);
                continue;
            };
            let view = Arc::new(SurfaceView::<P>::new(mapping, Arc::clone(&lease), None));
            let Some(import) = steps[1]
                .time(|| buffer_over(&device, &view, 0, case.length, 0, None))
                .expect("no raise")
            else {
                eprintln!("{label}: {}: a buffer over the view refused", case.name);
                continue;
            };
            let alias = steps[2]
                .time(|| buffer_over(&device, &view, 0, case.length, 0, None))
                .expect("no raise")
                .expect("a second buffer over the view");
            steps[3].time(|| drop(alias));
            kept.push_back((view, import));
            if kept.len() > super::CACHED_VIEWS {
                let (view, import) = kept.pop_front().expect("a view");
                steps[4].time(|| drop(import));
                steps[5].time(|| drop(view));
            }
        }
        for (view, import) in kept.drain(..) {
            steps[4].time(|| drop(import));
            steps[5].time(|| drop(view));
        }
        for (step, name) in steps.iter().zip([
            "range mapping",
            "first buffer (import)",
            "second buffer (shared)",
            "second buffer's drop",
            "first buffer's drop",
            "view's unmap",
        ]) {
            eprintln!("{label}: {}", step.line(case.name, name));
        }
    }
    dec.views_of_surfaces().reset();
    drop(dec);
    drop(core);
    drop(lease);
    drop(ram);
}

/// A pages slab with a heap: the guest's 2 MiB of a process's own plain pages.
const SLAB: u64 = 2 * MIB;

/// The orders a slab's pages are taken in (module docs): of the RAM's `pages` guest pages,
/// `takes` of them.
fn take_orders(pages: u64, takes: u64, rng: &mut XorShift) -> Vec<(&'static str, Vec<u64>)> {
    let mut scattered: Vec<u64> = (0..pages).collect();
    for i in 0..takes.to_usize() {
        let j = i + (rng.next() % (pages - i.to_u64())).to_usize();
        scattered.swap(i, j);
    }
    scattered.truncate(takes.to_usize());
    vec![
        ("upwards", (0..takes).collect()),
        ("downwards", (pages - takes..pages).rev().collect()),
        ("scattered", scattered),
    ]
}

/// The bytes of a pages block whose processes map one page, with room past it for `runs` runs.
fn block_size(runs: u64) -> u64 {
    GUEST_PAGE + (runs * size_of::<RgfxPageRun>().to_u64()).next_multiple_of(GUEST_PAGE)
}

/// Measures a pages slab's takes on `platform` in each order (module docs), printing a line per
/// step and the takes that took 5 ms or more.
fn measure_takes<P: ApiPlatform>(label: &str) {
    let platform = P::process();
    let device = platform.metal_device(None).expect("a GPU device");
    let takes = env_u64("RGK_BENCH_TAKES", 256);
    let ram_bytes = (env_u64("RGK_BENCH_RAM_MIB", 1024) * MIB).max(2 * takes * SLAB);
    let gap = Duration::from_micros(env_u64("RGK_BENCH_TAKE_GAP_US", 0));
    let only = std::env::var("RGK_BENCH_ORDERS").ok();
    let mut rng = XorShift(0x6a09_e667_f3bc_c908);
    let per = SLAB / GUEST_PAGE;
    let orders = take_orders(ram_bytes / GUEST_PAGE, takes * per, &mut rng);
    eprintln!(
        "{label}: {} MiB of fresh guest RAM, {takes} slabs of plain pages an order, {} us apart",
        ram_bytes >> 20,
        gap.as_micros()
    );
    // Each block holds its runs past a page of its own, room for one run a page.
    let block_size = block_size(per);
    for (name, pages) in orders {
        if only
            .as_ref()
            .is_some_and(|o| !o.split(',').any(|n| n == name))
        {
            continue;
        }
        // Fresh RAM, a lease and a decoder of their own an order, as a booted guest's.
        let ram = guest_ram(ram_bytes, false);
        let lease = ram.lease(platform.guest_memory(&ram.map));
        let core =
            CoreMemoryMap::new(Arc::clone(&platform), &ram.map.to_core()).expect("a memory map");
        let size = BLOCK + takes * block_size + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(crate::platform::test_region(&*platform, size.to_usize()));
        let mut dec = GraphicsDecoder::<P>::new(Arc::clone(&platform), region);
        dec.set_guest_memory(Arc::clone(&core), Arc::clone(&lease));
        let mut steps: [Step; 7] = Default::default();
        let mut slow = Vec::new();
        let mut made = Vec::new();
        let mut run_counts = std::collections::BTreeSet::new();
        for (i, slab_pages) in pages.chunks(per.to_usize()).enumerate() {
            std::thread::sleep(gap);
            let block = BLOCK + i.to_u64() * block_size;
            steps[0].time(|| zero_pages(&ram, slab_pages));
            let runs = page_runs(slab_pages);
            run_counts.insert(runs.len());
            let message = RgfxSurfacePages {
                header: header(RGFX_CMD_SURFACE_PAGES),
                context: 0,
                run_count: u32::try_from(runs.len()).expect("a slab's runs"),
                block_offset: block,
                block_size,
                runs_offset: block + GUEST_PAGE,
                length: SLAB,
            };
            let heap = 0x1000 + 2 * i.to_u64();
            let heap_over = RgfxHeapOverPages {
                header: header(RGFX_CMD_HEAP_OVER_PAGES),
                obj_id: heap,
                gpu_address: 0,
                block_offset: block,
                block_size,
                size_bytes: SLAB,
                options: 0,
                heap_type: 1,
            };
            let at = RgfxBufferAtOffset {
                header: header(RGFX_CMD_BUFFER_AT_OFFSET),
                obj_id: heap + 1,
                heap_obj_id: heap,
                offset: 0,
                length: 64 << 10,
                options: 0,
                reserved_34: 0,
            };
            let status = steps[1]
                .time(|| dec.surface_pages(&device, &message, &runs, RGFX_SURFACE_PAGES_PRIVATE));
            assert_eq!(status, RGFX_STATUS_OK, "{name}: take {i}'s SURFACE_PAGES");
            let heaped = steps[2].time(|| dec.decode(&device, heap_over.as_bytes()));
            assert_eq!(heaped.ok(), Some(Ok(())), "{name}: take {i}'s heap");
            let placed = steps[3].time(|| dec.decode(&device, at.as_bytes()));
            assert_eq!(placed.ok(), Some(Ok(())), "{name}: take {i}'s buffer");
            let take = steps[1].0[i] + steps[2].0[i] + steps[3].0[i];
            steps[4].0.push(take);
            if take >= Duration::from_millis(5) {
                slow.push((i, take));
            }
            made.push((block, heap));
        }
        for &(block, heap) in &made {
            steps[5].time(|| {
                for obj_id in [heap + 1, heap] {
                    let destroy = RgfxObjectDestroy {
                        header: header(RGFX_CMD_OBJECT_DESTROY),
                        obj_id,
                    };
                    assert_eq!(dec.decode(&device, destroy.as_bytes()).ok(), Some(Ok(())));
                }
            });
            steps[6].time(|| {
                dec.block_released(&device, block, 0, Box::new(|| ()), no_raise());
            });
        }
        for (step, step_name) in steps.iter().zip([
            "guest's zero fill",
            "SURFACE_PAGES (pages block)",
            "HEAP_OVER_PAGES",
            "BUFFER_AT_OFFSET",
            "take (the three)",
            "OBJECT_DESTROY (both)",
            "BLOCK_RELEASE",
        ]) {
            eprintln!("{label}: {}", step.line(name, step_name));
        }
        let listed: Vec<String> = (slow.iter().take(40))
            .map(|(i, d)| format!("{i}:{:.1}", d.as_secs_f64() * 1e3))
            .collect();
        eprintln!(
            "{label}: {name:<11} takes of 5 ms or more: {} of {} (take:ms) {}; runs a slab: \
             {run_counts:?}",
            slow.len(),
            made.len(),
            listed.join(" ")
        );
        dec.views_of_surfaces().reset();
        drop(dec);
        drop(core);
        drop(lease);
        drop(ram);
    }
}

/// The runs of the guest pages at `pages` of the RAM's pages, in that order, as the kext lists a
/// block's pages: a page right after the one before continues its run.
fn page_runs(pages: &[u64]) -> Vec<RgfxPageRun> {
    let mut runs: Vec<RgfxPageRun> = Vec::new();
    for &page in pages {
        let address = RAM + page * GUEST_PAGE;
        match runs.last_mut() {
            Some(last) if last.address + last.length == address => last.length += GUEST_PAGE,
            _ => runs.push(RgfxPageRun {
                address,
                length: GUEST_PAGE,
            }),
        }
    }
    runs
}

/// Writes zeros over the guest pages at `pages` of `ram`, as the guest's zero fill of a slab's
/// fresh pages does.
fn zero_pages(ram: &Ram, pages: &[u64]) {
    for &page in pages {
        // SAFETY: the page's bytes, in the RAM's mapping, which nothing else uses.
        unsafe {
            std::ptr::write_bytes(
                ram.at.add((page * GUEST_PAGE).to_usize()),
                0,
                GUEST_PAGE.to_usize(),
            );
        }
    }
}

/// Measures large pages slabs' takes on `platform` in each order (module docs), printing a line
/// per step.
fn measure_large_takes<P: ApiPlatform>(label: &str) {
    let platform = P::process();
    let device = platform.metal_device(None).expect("a GPU device");
    let slab = env_u64("RGK_BENCH_LARGE_MIB", 64) * MIB;
    let takes = env_u64("RGK_BENCH_LARGE_TAKES", 8);
    let per = slab / GUEST_PAGE;
    let ram_bytes = (env_u64("RGK_BENCH_RAM_MIB", 1024) * MIB).max(2 * takes * slab);
    let only = std::env::var("RGK_BENCH_ORDERS").ok();
    let mut rng = XorShift(0xbb67_ae85_84ca_a73b);
    let orders = take_orders(ram_bytes / GUEST_PAGE, takes * per, &mut rng);
    eprintln!(
        "{label}: {} MiB of fresh guest RAM, {takes} slabs of {} MiB of plain pages an order",
        ram_bytes >> 20,
        slab >> 20
    );
    let block_size = block_size(per);
    for (name, pages) in orders {
        if only
            .as_ref()
            .is_some_and(|o| !o.split(',').any(|n| n == name))
        {
            continue;
        }
        let ram = guest_ram(ram_bytes, false);
        let lease = ram.lease(platform.guest_memory(&ram.map));
        let core =
            CoreMemoryMap::new(Arc::clone(&platform), &ram.map.to_core()).expect("a memory map");
        let size = BLOCK + takes * block_size + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(crate::platform::test_region(&*platform, size.to_usize()));
        let mut dec = GraphicsDecoder::<P>::new(Arc::clone(&platform), region);
        dec.set_guest_memory(Arc::clone(&core), Arc::clone(&lease));
        let mut steps: [Step; 6] = Default::default();
        let mut run_counts = Vec::new();
        for (i, slab_pages) in pages.chunks(per.to_usize()).enumerate() {
            let block = BLOCK + i.to_u64() * block_size;
            steps[0].time(|| zero_pages(&ram, slab_pages));
            let runs = page_runs(slab_pages);
            run_counts.push(runs.len());
            let message = RgfxSurfacePages {
                header: header(RGFX_CMD_SURFACE_PAGES),
                context: 0,
                run_count: u32::try_from(runs.len()).expect("a slab's runs"),
                block_offset: block,
                block_size,
                runs_offset: block + GUEST_PAGE,
                length: slab,
            };
            let over = RgfxBufferOverPages {
                header: header(RGFX_CMD_BUFFER_OVER_PAGES),
                obj_id: 0x1000 + i.to_u64(),
                gpu_address: 0,
                block_offset: block,
                block_size,
                offset: 0,
                length: slab,
                options: 0,
                reserved_44: 0,
            };
            let status = steps[1]
                .time(|| dec.surface_pages(&device, &message, &runs, RGFX_SURFACE_PAGES_PRIVATE));
            assert_eq!(status, RGFX_STATUS_OK, "{name}: slab {i}'s SURFACE_PAGES");
            let made = steps[2].time(|| dec.decode(&device, over.as_bytes()));
            assert_eq!(made.ok(), Some(Ok(())), "{name}: slab {i}'s buffer");
            let take = steps[1].0[i] + steps[2].0[i];
            steps[3].0.push(take);
            let destroy = RgfxObjectDestroy {
                header: header(RGFX_CMD_OBJECT_DESTROY),
                obj_id: over.obj_id,
            };
            let gone = steps[4].time(|| dec.decode(&device, destroy.as_bytes()));
            assert_eq!(gone.ok(), Some(Ok(())));
            steps[5].time(|| {
                dec.block_released(&device, block, 0, Box::new(|| ()), no_raise());
            });
        }
        for (step, step_name) in steps.iter().zip([
            "guest's zero fill",
            "SURFACE_PAGES (pages block)",
            "BUFFER_OVER_PAGES",
            "take (the two)",
            "OBJECT_DESTROY",
            "BLOCK_RELEASE",
        ]) {
            eprintln!("{label}: {}", step.line(name, step_name));
        }
        eprintln!(
            "{label}: {name:<11} runs a slab: {:?}",
            run_counts
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
        );
        dec.views_of_surfaces().reset();
        drop(dec);
        drop(core);
        drop(lease);
        drop(ram);
    }
}

/// The range mappings alone, with no GPU device made: what a profiler of the views' own code
/// runs, outside the GPU's locks (`RGK_BENCH_MAP_ONLY=1` with the Vulkan test).
fn map_only<P: ApiPlatform>(label: &str) {
    let platform = P::process();
    let page = platform.vm().page_size().max(GUEST_PAGE);
    let ram_bytes = env_u64("RGK_BENCH_RAM_MIB", 768) * MIB;
    let iters = env_u64("RGK_BENCH_ITERS", 40);
    let only = std::env::var("RGK_BENCH_CASES").ok();
    let ram = guest_ram(ram_bytes, true);
    let lease = ram.lease(platform.guest_memory(&ram.map));
    let core = CoreMemoryMap::new(Arc::clone(&platform), &ram.map.to_core()).expect("a memory map");
    let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
    for case in cases(page) {
        if only
            .as_ref()
            .is_some_and(|o| !o.split(',').any(|n| n == case.name))
        {
            continue;
        }
        if case.length > ram_bytes / 2 {
            continue;
        }
        let mut steps: [Step; 2] = Default::default();
        let keep = env_u64("RGK_BENCH_KEEP", 0).to_usize();
        let mut kept = std::collections::VecDeque::new();
        for _ in 0..iters {
            let runs = surface_runs(&mut rng, ram_bytes, case.length, case.run);
            let ranges: Vec<PhysRange> = runs
                .iter()
                .map(|r| PhysRange {
                    physical_address: r.address,
                    physical_length: r.length,
                })
                .collect();
            let mapping = steps[0]
                .time(|| core.new_range_mapping(&ranges, case.length, 0))
                .expect("a view");
            kept.push_back(mapping);
            if kept.len() > keep {
                let old = kept.pop_front();
                steps[1].time(|| drop(old));
            }
        }
        kept.clear();
        for (step, name) in steps.iter().zip(["range mapping", "its unmap"]) {
            eprintln!("{label}: {}", step.line(case.name, name));
        }
    }
    drop(core);
    drop(lease);
    drop(ram);
}

/// The rt platform's costs (module docs).
#[test]
#[ignore = "a measurement: run it alone, under the GPU locks"]
fn surface_view_costs_vulkan() {
    if env_u64("RGK_BENCH_MAP_ONLY", 0) == 1 {
        map_only::<reims_vgpu_vulkan::RtPlatform>("rt");
        return;
    }
    measure::<reims_vgpu_vulkan::RtPlatform>("rt");
}

/// The rt platform's takes of pages slabs (module docs).
#[test]
#[ignore = "a measurement: run it alone, under the GPU locks"]
fn pages_take_costs_vulkan() {
    measure_takes::<reims_vgpu_vulkan::RtPlatform>("rt");
}

/// The rt platform's takes of large pages slabs (module docs).
#[test]
#[ignore = "a measurement: run it alone, under the GPU locks"]
fn large_take_costs_vulkan() {
    measure_large_takes::<reims_vgpu_vulkan::RtPlatform>("rt");
}
