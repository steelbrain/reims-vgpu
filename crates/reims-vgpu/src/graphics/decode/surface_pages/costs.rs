//! What a new IOSurface's view of its pages costs the thread that makes it, measured on a real
//! platform (ignored: a measurement, not a check). Run it alone, holding the host's GPU locks:
//!
//! ```sh
//! cargo nextest run -p reims-vgpu --features vulkan -E 'test(/surface_view_costs/)' \
//!   --run-ignored only --no-capture
//! ```
//!
//! The guest's RAM is what QEMU gives the device: on Linux a shared mapping of a memfd, whose file
//! the platform's lease registers; on macOS anonymous memory. Each surface is fresh pages, as a
//! browser's are: its runs are slots of the RAM picked at random, so no two surfaces name the same
//! runs and the cache of views takes none. Each step is timed as the control thread carries it
//! out:
//! - `SURFACE_PAGES`: the view (a range mapping of the runs) and the buffer over all of it, which
//!   holds the runtime's import; for a cached block also the device cache;
//! - `BUFFER_OVER_SURFACE_PAGES`, its `OBJECT_DESTROY`, and the block's `BLOCK_RELEASE`;
//! - and apart, the same steps one at a time: the range mapping, the first buffer over it (the
//!   import), a second (which shares it), and each one's drop.
//!
//! `RGK_BENCH_ITERS` surfaces a case (default 40, past the cache's 32 views, so its drops are
//! counted), `RGK_BENCH_RAM_MIB` MiB of RAM (default 768), `RGK_BENCH_THP=1` asks for huge pages
//! of the memfd, and `RGK_BENCH_CASES` a comma list of the cases' names to run.

use std::sync::Arc;
use std::time::{Duration, Instant};

use reims_graphics::region::Region;
use reims_graphics::wire::{
    RGFX_CMD_BUFFER_OVER_SURFACE_PAGES, RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_SURFACE_PAGES,
    RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN, RGFX_STATUS_OK, RGFX_SURFACE_PAGES_CACHED,
    RgfxBufferOverSurfacePages, RgfxMsgHeader, RgfxObjectDestroy, RgfxPageRun, RgfxSurfacePages,
};
use reims_vgpu_core::host::PhysRange;
use reims_vgpu_core::memory::map::MemoryMap as CoreMemoryMap;
use reims_vgpu_core::num::{ToU64 as _, ToUsize as _};
use zerocopy::IntoBytes;

use super::super::{GraphicsDecoder, no_raise};
use super::{GUEST_PAGE, Lease, SurfaceView, buffer_over};
use crate::device::MemoryMap;
use crate::platform::ApiPlatform;

/// The block every surface's messages name, one at a time.
const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x4000;

/// The guest physical address of the RAM's first byte.
const RAM: u64 = 0x1_0000_0000;

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

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A small PRNG, so each run picks the same surfaces.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
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
fn surface_runs(rng: &mut XorShift, ram: u64, length: u64, run: u64) -> Vec<RgfxPageRun> {
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

/// The guest's RAM as the host maps it, and what keeps it.
struct Ram {
    map: MemoryMap,
    at: *mut u8,
    length: u64,
    /// The memfd behind it, on Linux.
    #[cfg(target_os = "linux")]
    fd: Arc<std::os::fd::OwnedFd>,
}

impl Drop for Ram {
    fn drop(&mut self) {
        // SAFETY: the RAM's own mapping, which no device or view uses any more.
        unsafe { libc::munmap(self.at.cast(), self.length.to_usize()) };
    }
}

/// `length` bytes of guest RAM at [`RAM`], every page touched: a memfd's shared mapping on Linux
/// (with huge pages asked for where `RGK_BENCH_THP=1`), anonymous memory on macOS.
fn guest_ram(length: u64) -> Ram {
    #[cfg(target_os = "linux")]
    let (at, file) = {
        use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
        // SAFETY: a NUL-terminated name; the descriptor is owned from here on.
        let fd = unsafe { libc::memfd_create(c"rvg-bench-ram".as_ptr(), libc::MFD_CLOEXEC) };
        assert!(fd >= 0);
        // SAFETY: a descriptor just made, owned once.
        let fd = Arc::new(unsafe { OwnedFd::from_raw_fd(fd) });
        // SAFETY: sizing the file the test owns.
        assert_eq!(
            unsafe { libc::ftruncate(fd.as_raw_fd(), length as libc::off_t) },
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
        let file = crate::device::MemoryFile { fd, offset: 0 };
        (at.cast::<u8>(), file)
    };
    #[cfg(not(target_os = "linux"))]
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
    // SAFETY: the mapping just made, `length` bytes, writable.
    unsafe { std::ptr::write_bytes(at, 0x5a, length.to_usize()) };
    let mut map = MemoryMap::new();
    #[cfg(target_os = "linux")]
    {
        let fd = Arc::clone(&file.fd);
        // SAFETY: the RAM stays mapped until `Ram`'s drop, after every device and view made over
        // it.
        unsafe { map.push_with_file(RAM, length, at, Some(file)) };
        Ram {
            map,
            at,
            length,
            fd,
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let () = file;
        // SAFETY: as above.
        unsafe { map.push(RAM, length, at) };
        Ram { map, at, length }
    }
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
    let ram = guest_ram(ram_bytes);
    let lease: Lease = Arc::from(platform.guest_memory(&ram.map, None));
    let core = CoreMemoryMap::new(Arc::clone(&platform), &ram.map.to_core()).expect("a memory map");
    let size = BLOCK + 0x10_0000 + RGFX_CURSOR_SPAN;
    let region = Arc::new(Region::new(size.to_usize()).expect("a region"));
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
        // Through the decoder, as the control thread carries the messages out.
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
                    dec.block_released(&device, BLOCK, false, Box::new(|| ()), no_raise());
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
            let view = Arc::new(SurfaceView::<P> {
                mapping,
                _lease: Arc::clone(&lease),
            });
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

/// The range mappings alone, with no GPU device made: what a profiler of the views' own code
/// runs, outside the GPU's locks (`RGK_BENCH_MAP_ONLY=1` with the Vulkan test).
#[cfg(feature = "vulkan")]
fn map_only<P: ApiPlatform>(label: &str) {
    let platform = P::process();
    let page = platform.vm().page_size().max(GUEST_PAGE);
    let ram_bytes = env_u64("RGK_BENCH_RAM_MIB", 768) * MIB;
    let iters = env_u64("RGK_BENCH_ITERS", 40);
    let only = std::env::var("RGK_BENCH_CASES").ok();
    let ram = guest_ram(ram_bytes);
    let lease: Lease = Arc::from(platform.guest_memory(&ram.map, None));
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

/// The Vulkan platform's costs (module docs).
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "a measurement: run it alone, under the GPU locks"]
fn surface_view_costs_vulkan() {
    if env_u64("RGK_BENCH_MAP_ONLY", 0) == 1 {
        map_only::<reims_vgpu_vulkan::VulkanPlatform>("vulkan");
        return;
    }
    measure::<reims_vgpu_vulkan::VulkanPlatform>("vulkan");
    #[cfg(target_os = "linux")]
    raw::measure();
}

/// The same views and imports made with nothing of ours between: the kernel's `mmap` of each
/// run, and the runtime's import of the view, as it checks the pages' write permission
/// (`PROCMAP_QUERY` a mapping) and as it is told they are writable (`assume_writable`).
#[cfg(all(feature = "vulkan", target_os = "linux"))]
mod raw {
    use std::os::fd::AsRawFd as _;
    use std::ptr::NonNull;

    use metal2vulkan_runtime as rt;

    use super::*;

    pub(super) fn measure() {
        let device = rt::Device::system_default().expect("a runtime device");
        let page = 4096u64;
        let ram_bytes = env_u64("RGK_BENCH_RAM_MIB", 768) * MIB;
        let iters = env_u64("RGK_BENCH_ITERS", 40);
        let only = std::env::var("RGK_BENCH_CASES").ok();
        let ram = guest_ram(ram_bytes);
        let mut rng = XorShift(0x2545_f491_4f6c_dd1d);
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
            let mut steps: [Step; 6] = Default::default();
            for _ in 0..iters {
                let runs = surface_runs(&mut rng, ram_bytes, case.length, case.run);
                let len = case.length.to_usize();
                let view = steps[0].time(|| {
                    // SAFETY: a fresh reservation, every run then mapped over it at its place.
                    unsafe {
                        let at = libc::mmap(
                            std::ptr::null_mut(),
                            len,
                            libc::PROT_NONE,
                            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                            -1,
                            0,
                        );
                        assert_ne!(at, libc::MAP_FAILED);
                        let mut offset = 0usize;
                        for r in &runs {
                            let got = libc::mmap(
                                at.cast::<u8>().add(offset).cast(),
                                r.length.to_usize(),
                                libc::PROT_READ | libc::PROT_WRITE,
                                libc::MAP_SHARED | libc::MAP_FIXED,
                                ram.fd.as_raw_fd(),
                                (r.address - RAM) as libc::off_t,
                            );
                            assert_ne!(got, libc::MAP_FAILED);
                            offset += r.length.to_usize();
                        }
                        at.cast::<u8>()
                    }
                });
                let pointer = NonNull::new(view).expect("a view");
                let import = |vouched: bool| {
                    // SAFETY: the view stays mapped, writable, until after the buffer's drop.
                    let memory = unsafe {
                        let m = rt::HostMemory::new(pointer);
                        if vouched { m.assume_writable() } else { m }
                    };
                    let contents = rt::BufferContents::NoCopy {
                        memory,
                        deallocator: None,
                        keep_alive: None,
                    };
                    device.new_buffer_with_descriptor(rt::BufferDescriptor::new(
                        case.length,
                        rt::ResourceOptions::shared(),
                        contents,
                    ))
                };
                let checked = steps[1].time(|| import(false)).expect("an import");
                steps[2].time(|| drop(checked));
                let vouched = steps[3].time(|| import(true)).expect("an import");
                steps[4].time(|| drop(vouched));
                // SAFETY: the view made above, which no buffer lies over any more.
                steps[5].time(|| unsafe { libc::munmap(view.cast(), len) });
            }
            for (step, name) in steps.iter().zip([
                "raw mmap of each run",
                "runtime import, checked",
                "its free",
                "runtime import, vouched",
                "its free",
                "raw munmap",
            ]) {
                eprintln!("raw: {}", step.line(case.name, name));
            }
        }
    }
}

/// The Metal platform's costs, for comparison (module docs).
#[cfg(target_os = "macos")]
#[test]
#[ignore = "a measurement: run it alone, under the GPU locks"]
fn surface_view_costs_metal() {
    measure::<reims_vgpu_metal::MetalPlatform>("metal");
}
