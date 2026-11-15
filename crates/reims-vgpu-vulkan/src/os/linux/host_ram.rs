//! The sources of dma-bufs of the guest's RAM's pages on Linux, which the RAM handed to a device
//! whole carries ([`with_ram_sources`]; `crate::os`'s `host_ram` hands it).
//!
//! - **By pointer or as dma-bufs.** Where `/dev/udmabuf` opens, the RAM has a source of dma-bufs
//!   of its pages ([`dma_bufs`]): udmabuf's, through the backing table, as a no-copy import's
//!   (`super::import`), each within udmabuf's limits as the host has them (`size_limit_mb`, 64 MiB
//!   by default, which the library never raises). The runtime imports a chunk as a dma-buf where
//!   the device's held pointer imports add to every submission's cost, or where it refuses the
//!   chunk's pointer, and as long as one dma-buf holds there (`rt::HostRam::with_largest_dma_buf`);
//!   by its pointer otherwise, and where the source refuses. Without udmabuf, by pointer alone.
//!   **Measured** 2026-11-07, an x86-64 macOS 26 guest on the reims-vgpu device: CachyOS's
//!   7.3.1 kernels answer `size_limit_mb` 2147483647 as they boot, and its 7.2.8 kernel the
//!   default 64. NVIDIA's driver never asks the source: on an RTX 5080 (7.2.8), whose held
//!   pointer imports cost a submission nothing, the runtime took a 16 GiB guest's RAM by
//!   pointer, in 64 MiB chunks.
//! - **Lists of runs.** With it, the RAM has a source of one dma-buf of a list of runs of its
//!   pages ([`dma_bufs_of_runs`], `rt::HostRam::with_dma_buf_of_runs`), as many runs a list as
//!   udmabuf's `list_limit` takes. The runtime calls it only for a copy by runs on a device whose
//!   held imports cost every submission (RADV's, the Iris Xe's): the copy imports its own pages,
//!   in lists, held until its command buffer completes, and a chunk of the RAM is imported only
//!   for a copy the lists do not take (one of whose runs is the whole chunk, or one whose list's
//!   import failed), the one import objects over that chunk then lie in. A chunk's import pins
//!   its pages, which then stay resident: before the lists, a Radeon 780M
//!   (RADV, 7.3.1) handed the RAM took a 16 GiB guest's RAM in chunks of about 2.5 to 3 GiB for
//!   its copies, and 12.3 GiB of it was resident by the desktop and at the end, where views alone
//!   kept 7.9 to 8.0 GiB by the desktop and 10.4 to 10.8 GiB at the end.

use std::sync::Arc;

use crate::core::num::ToU64 as _;
use metal2vulkan_runtime as rt;

/// `ram` with its sources of dma-bufs of its pages, where `/dev/udmabuf` opens (module docs): one
/// of a chunk's pages ([`dma_bufs`]), within udmabuf's limits, and one of a list of runs
/// ([`dma_bufs_of_runs`]); `ram` as it is otherwise.
pub(crate) fn with_ram_sources(ram: rt::HostRam) -> rt::HostRam {
    match (dma_bufs(), dma_bufs_of_runs()) {
        (Some(source), Some(lists)) => {
            let limits = super::dma_buf::limits();
            (ram.with_dma_buf(source))
                .with_largest_dma_buf(limits.bytes)
                .with_dma_buf_of_runs(lists, limits.pieces.try_into().unwrap_or(usize::MAX))
        }
        _ => ram,
    }
}

/// Whether the RAM has sources of dma-bufs ([`with_ram_sources`]): where `/dev/udmabuf` opens.
#[cfg(test)]
pub(crate) fn ram_has_sources() -> bool {
    super::dma_buf::opens()
}

/// The RAM's source of one dma-buf of a list of runs of its pages
/// (`rt::HostRam::with_dma_buf_of_runs`), where `/dev/udmabuf` opens: each list answered by
/// [`ram_dma_buf_of_runs`].
fn dma_bufs_of_runs() -> Option<rt::DmaBufRunsSource> {
    if !super::dma_buf::opens() {
        return None;
    }
    Some(Arc::new(move |requests: &[rt::DmaBufRequest]| {
        let ranges: Vec<(u64, u64)> = (requests.iter())
            .map(|r| (r.start.as_ptr().addr().to_u64(), r.length))
            .collect();
        ram_dma_buf_of_runs(&ranges)
    }))
}

/// The dma-buf [`dma_bufs_of_runs`]'s source answers for `ranges` of the RAM's pages, each a host
/// address and a length, whole pages by address and apart: those pages one after another,
/// resolved through the backing table (`super::dma_buf::of_ranges`). A refusal (a page the table
/// resolves to no file, a limit, udmabuf's own) answers an error, which leaves the copy to what it
/// takes without the source.
fn ram_dma_buf_of_runs(ranges: &[(u64, u64)]) -> std::io::Result<rt::DmaBufChunk> {
    let length = ranges.iter().map(|&(_, n)| n).sum();
    match super::dma_buf::of_ranges(ranges) {
        Ok(fd) => Ok(rt::DmaBufChunk {
            file: Arc::new(fd),
            offset: 0,
            length,
        }),
        Err(refusal) => {
            log::debug!(
                target: "reims_vgpu_vulkan::os",
                "no dma-buf of {} runs of the guest's RAM, {length:#x} bytes: {refusal}",
                ranges.len()
            );
            Err(std::io::Error::other(refusal.to_string()))
        }
    }
}

/// The RAM's source of dma-bufs of its pages (`rt::HostRam::with_dma_buf`), where `/dev/udmabuf`
/// opens: each request answered by [`ram_dma_bufs`].
fn dma_bufs() -> Option<rt::DmaBufSource> {
    if !super::dma_buf::opens() {
        return None;
    }
    Some(Arc::new(move |request: rt::DmaBufRequest| {
        let first = request.start.as_ptr().addr().to_u64();
        ram_dma_bufs(first, request.length, request.granule)
    }))
}

/// The dma-bufs [`dma_bufs`]'s source answers for `length` bytes of the RAM's pages from `first`,
/// host addresses, at `granule`: those pages resolved through the backing table and cut within
/// udmabuf's limits at multiples of `granule` from `first` (`super::dma_buf::of_pages`), as a
/// no-copy import's source makes them. A refusal (pages the table resolves to no file, a limit,
/// udmabuf's own) answers an error, which leaves the chunk to its pointer.
fn ram_dma_bufs(first: u64, length: u64, granule: u64) -> std::io::Result<Vec<rt::DmaBufChunk>> {
    super::dma_buf::of_pages(first, length, granule).map_err(|refusal| {
        log::debug!(
            target: "reims_vgpu_vulkan::os",
            "no dma-buf of the guest's RAM's {length:#x} bytes at {first:#x}: {refusal}"
        );
        std::io::Error::other(refusal.to_string())
    })
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsRawFd as _;

    use crate::core::host::HostPtr;
    use crate::core::num::{ToU64 as _, ToUsize as _};
    use crate::core::platform::HostRange;
    use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
    use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::os::vm::HostVm as _;
    use crate::core::platform::types::NsRange;
    use crate::core::raise::OnFault;

    use super::super::import::imports_need_files;
    use super::super::vm::memfd;
    use super::super::{PAGE, lease_guest_memory};
    use crate::gpu::device::tests::dev;
    use crate::os::GuestRamFile;
    use crate::remap::VulkanVm;

    const MIB: u64 = 1 << 20;

    /// What a slab of guest RAM costs on this device (arch-shared-mem's design): placement heaps
    /// over one 2 MiB run of the leased RAM each, 4 KiB buffers placed in one of them, heaps over
    /// views of 2 and 32 runs apart, and a command buffer's round trip with one heap alive and with
    /// all of them. Timings only, printed as `slab:` lines: run alone, holding the GPU.
    #[test]
    #[ignore = "a measurement: run alone, holding the GPU"]
    fn slab_heap_costs() {
        use crate::core::platform::os::{Clock as _, MonoTime};
        use crate::core::platform::resource::Heap as _;
        use crate::core::platform::values::{HeapDesc, OptionSet};
        let clock = crate::os::OsClock::new();
        let now = || clock.now_mono();
        let since = |t: MonoTime| clock.now_mono().saturating_duration_since(t);
        const SHARED_TRACKED: u64 = 0x200;
        const PLACEMENT: i64 = 1;
        let device = dev();
        let length = 4608 * MIB;
        let fd = memfd(length).unwrap();
        // SAFETY: a fresh shared mapping of the file, unmapped at the end.
        let start = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length.to_usize(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        assert_ne!(start, libc::MAP_FAILED);
        // Every page touched first, as the guest's zero fill touches a slab's pages.
        // SAFETY: the mapping above, `length` bytes.
        unsafe { std::ptr::write_bytes(start.cast::<u8>(), 0, length.to_usize()) };
        let at = start.addr().to_u64();
        let lease = lease_guest_memory([GuestRamFile {
            host: at,
            guest: 0,
            length,
            file: std::sync::Arc::clone(&fd),
            offset: 0,
        }]);
        let counts = || {
            let c = device.raw.work_counts();
            (
                c.host_ram_chunks_imported,
                c.chunk_bound_buffers_made,
                c.host_pointer_imports_made,
                c.copied_heaps_made,
            )
        };
        let heap_at = |addr: u64, len: u64| {
            // SAFETY: mapped pages of the test's, mapped until after every heap is dropped.
            let bytes = unsafe { HostRange::new(HostPtr::from_addr(addr.to_usize()), len) };
            let desc = HeapDesc {
                size: len,
                heap_type: PLACEMENT,
                options: vec![OptionSet::ResourceOptions(SHARED_TRACKED)],
                pinned_gpu_address: None,
            };
            device
                .new_heap_over_host_memory(bytes, &desc)
                .unwrap()
                .heap
                .unwrap()
        };
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            (v[v.len() / 2], v[v.len() - 1])
        };
        let before = counts();
        let mut heaps = Vec::new();
        let mut made = Vec::new();
        let mut first = 0.0;
        for i in 0..2000u64 {
            let t = now();
            heaps.push(heap_at(at + 2 * MIB * i, 2 * MIB));
            let us = since(t).as_secs_f64() * 1e6;
            if i == 0 {
                first = us;
            }
            made.push(us);
        }
        let (p50, max) = median(&mut made);
        let after = counts();
        println!(
            "slab: one-run heaps n=2000 first_us={first:.1} p50_us={p50:.1} max_us={max:.1} \
             chunks_imported={} chunk_bound={} host_pointer_imports={} copied_heaps={} \
             imports_need_files={}",
            after.0 - before.0,
            after.1 - before.1,
            after.2 - before.2,
            after.3 - before.3,
            imports_need_files()
        );
        let mut placed = Vec::new();
        let mut times = Vec::new();
        for k in 0..512u64 {
            let t = now();
            let b = heaps[1]
                .new_buffer_of_length(4096, SHARED_TRACKED, k * 4096)
                .unwrap();
            times.push(since(t).as_secs_f64() * 1e6);
            placed.push(b.unwrap());
        }
        let (p50, max) = median(&mut times);
        println!("slab: placed 4 KiB buffers n=512 p50_us={p50:.2} max_us={max:.1}");
        // Views of runs: 4 KiB runs 8 KiB apart from 4,300 MiB on (512 runs, 2 MiB; 4,096 runs,
        // 16 MiB), as a slab of the guest's plain pages is where its kernel gives them scattered,
        // the common case; 2 MiB runs 4 MiB apart from 4,100 MiB on (2 runs, 4 MiB; 32 runs,
        // 64 MiB), as a large slab is where they come in 2 MiB pieces.
        let view = |runs: u64, run: u64, from: u64| {
            let len = runs * run;
            // SAFETY: a fresh reservation, then each run mapped over it from the file.
            let base = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len.to_usize(),
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert_ne!(base, libc::MAP_FAILED);
            for r in 0..runs {
                let off = from + 2 * run * r;
                // SAFETY: within the reservation above.
                let p = unsafe {
                    libc::mmap(
                        base.cast::<u8>().add((run * r).to_usize()).cast(),
                        run.to_usize(),
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED | libc::MAP_FIXED,
                        fd.as_raw_fd(),
                        off as libc::off_t,
                    )
                };
                assert_ne!(p, libc::MAP_FAILED);
            }
            (base, len)
        };
        let mut views = Vec::new();
        for (runs, run, from) in [
            (2u64, 2 * MIB, 4100 * MIB),
            (32, 2 * MIB, 4100 * MIB),
            (512, 4096, 4300 * MIB),
            (4096, 4096, 4300 * MIB),
        ] {
            let t = now();
            let (base, len) = view(runs, run, from);
            let map_us = since(t).as_secs_f64() * 1e6;
            let c0 = counts();
            let t = now();
            let h = heap_at(base.addr().to_u64(), len);
            let us = since(t).as_secs_f64() * 1e6;
            let c1 = counts();
            let t = now();
            let b = h
                .new_buffer_of_length(4096, SHARED_TRACKED, len - 4096)
                .unwrap()
                .unwrap();
            let place_us = since(t).as_secs_f64() * 1e6;
            println!(
                "slab: view heap runs={runs} run={run} bytes={len} map_us={map_us:.1} heap_us={us:.1} \
                 place_us={place_us:.1} chunk_bound={} host_pointer_imports={} copied_heaps={}",
                c1.1 - c0.1,
                c1.2 - c0.2,
                c1.3 - c0.3
            );
            views.push((h, b, base, len));
        }
        // A command buffer's round trip: a 4-byte fill of one placed buffer, committed and waited.
        let q = device.new_command_queue().unwrap();
        let round = |n: usize| {
            let mut v = Vec::new();
            for _ in 0..n {
                let t = now();
                let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
                let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
                let fill = BlitCmd::FillBuffer {
                    buffer: &placed[0],
                    range: NsRange::new(0, 4),
                    value: 0x5a,
                };
                e.encode(OnFault::Die, fill).unwrap();
                e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
                cb.commit(OnFault::Die).unwrap();
                cb.wait_until_completed(OnFault::Die).unwrap();
                assert!(cb.error().is_none(), "{:?}", cb.error());
                v.push(since(t).as_secs_f64() * 1e6);
            }
            median(&mut v)
        };
        // Rounds of 20 in a row after heaps are made, each round's median with the time since the
        // makes ended: whether a submission's cost after many makes follows their count or the time.
        let series = |label: &str, made_at: MonoTime| {
            let mut out = String::new();
            for _ in 0..8 {
                let (p50, _) = round(20);
                let ms = since(made_at).as_secs_f64() * 1e3;
                out.push_str(&format!(" {p50:.0}us@{ms:.0}ms"));
            }
            println!("slab: cb round trips {label}:{out}");
        };
        series("after making 2,004 heaps", now());
        let all = round(100);
        drop(views);
        heaps.truncate(200);
        let some = round(100);
        heaps.truncate(2);
        let few = round(100);
        println!(
            "slab: cb round trip p50/max us: {:.1}/{:.1} with 2,004 heaps, {:.1}/{:.1} with 200, \
             {:.1}/{:.1} with 2",
            all.0, all.1, some.0, some.1, few.0, few.1
        );
        let mut again = Vec::new();
        for i in 2..2000u64 {
            let t = now();
            heaps.push(heap_at(at + 2 * MIB * i, 2 * MIB));
            again.push(since(t).as_secs_f64() * 1e6);
        }
        let (again50, againmax) = median(&mut again);
        println!("slab: 1,998 heaps made again at p50 {again50:.1} us, max {againmax:.1}");
        series("after making 1,998 again", now());
        heaps.truncate(2);
        for i in 2..2000u64 {
            heaps.push(heap_at(at + 2 * MIB * i, 2 * MIB));
        }
        let made = now();
        std::thread::sleep(std::time::Duration::from_millis(300));
        series("after making 1,998 again, then 300 ms idle", made);
        heaps.truncate(2);
        for i in 2..2000u64 {
            heaps.push(heap_at(at + 2 * MIB * i, 2 * MIB));
            if i % 100 == 0 {
                round(1);
            }
        }
        series(
            "after making 1,998 again with a command buffer every 100",
            now(),
        );
        let keep = heaps;
        // SAFETY: the RAM, which no GPU work uses now.
        let byte = unsafe {
            std::ptr::with_exposed_provenance::<u8>((at + 2 * MIB).to_usize()).read_volatile()
        };
        assert_eq!(byte, 0x5a, "the fill reached the RAM");
        drop((q, placed, keep, lease));
        // Once the runtime freed its imports of the RAM's chunks (the RAM's keep-alive), through
        // the platform's VM, which leaves pages a buffer's import still holds mapped until the
        // runtime has freed it: the device lives on past the lease.
        let start_at = HostPtr::from_ptr(start.cast::<u8>());
        crate::remap::free_unimported(start.cast(), length, || {
            assert!(VulkanVm::new().deallocate(start_at, length).is_success());
        });
    }

    /// What a slab's buffers cost as no-copy buffers over its pages, each at an offset of a buffer
    /// over the whole slab that holds its import first, as a pages block's view holds one
    /// (arch-shared-mem phase 2b): over one 2 MiB run of the leased RAM, over a view mapping one
    /// 2 MiB run of the RAM's file, and over a view of 512 4 KiB runs apart. The 4 KiB buffers are
    /// made at the host's address, then pinned in a freed range, as the guest pins them; a GPU
    /// fill through the last pinned one reaches the file's page. Timings only, printed as `slab:`
    /// lines: run alone, holding the GPU.
    #[test]
    #[ignore = "a measurement: run alone, holding the GPU"]
    fn slab_nocopy_costs() {
        use crate::core::platform::os::{Clock as _, MonoTime};
        use crate::core::platform::resource::Buffer as _;
        use crate::core::platform::values::{BufferDesc, OptionSet};
        let clock = crate::os::OsClock::new();
        let now = || clock.now_mono();
        let since = |t: MonoTime| clock.now_mono().saturating_duration_since(t);
        const SHARED_TRACKED: u64 = 0x200;
        let device = dev();
        let length = 256 * MIB;
        let fd = memfd(length).unwrap();
        // SAFETY: a fresh shared mapping of the file, unmapped at the end.
        let start = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length.to_usize(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        assert_ne!(start, libc::MAP_FAILED);
        // SAFETY: the mapping above, `length` bytes.
        unsafe { std::ptr::write_bytes(start.cast::<u8>(), 0, length.to_usize()) };
        let at = start.addr().to_u64();
        let lease = lease_guest_memory([GuestRamFile {
            host: at,
            guest: 0,
            length,
            file: std::sync::Arc::clone(&fd),
            offset: 0,
        }]);
        let counts = || {
            let c = device.raw.work_counts();
            (
                c.host_ram_chunks_imported,
                c.chunk_bound_buffers_made,
                c.host_pointer_imports_made,
                c.copied_heaps_made,
            )
        };
        let buffer = |addr: u64, len: u64, pin: Option<u64>| {
            // SAFETY: mapped pages of the test's, mapped until after every buffer is dropped.
            let bytes = unsafe { HostRange::new(HostPtr::from_addr(addr.to_usize()), len) };
            device
                .new_buffer(BufferDesc {
                    length: len,
                    options: vec![OptionSet::ResourceOptions(SHARED_TRACKED)],
                    pinned_gpu_address: pin,
                    contents: Some(bytes),
                    no_copy: true,
                    deallocator: None,
                })
                .unwrap()
        };
        // A GPU range of `len` bytes no buffer holds: a buffer's, freed.
        let freed_range = |len: u64| {
            let b = device
                .new_buffer(BufferDesc {
                    length: len,
                    options: vec![OptionSet::ResourceOptions(SHARED_TRACKED)],
                    pinned_gpu_address: None,
                    contents: None,
                    no_copy: false,
                    deallocator: None,
                })
                .unwrap()
                .unwrap();
            b.gpu_address()
        };
        let median = |v: &mut Vec<f64>| {
            if v.is_empty() {
                return (0.0, 0.0);
            }
            v.sort_by(f64::total_cmp);
            (v[v.len() / 2], v[v.len() - 1])
        };
        // `runs` runs of `run` bytes, `2 * run` apart in the file from `from`, mapped in order.
        let view = |runs: u64, run: u64, from: u64| {
            let len = runs * run;
            // SAFETY: a fresh reservation, then each run mapped over it from the file.
            let base = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len.to_usize(),
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert_ne!(base, libc::MAP_FAILED);
            for r in 0..runs {
                let off = from + 2 * run * r;
                // SAFETY: within the reservation above.
                let p = unsafe {
                    libc::mmap(
                        base.cast::<u8>().add((run * r).to_usize()).cast(),
                        run.to_usize(),
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED | libc::MAP_FIXED,
                        fd.as_raw_fd(),
                        off as libc::off_t,
                    )
                };
                assert_ne!(p, libc::MAP_FAILED);
            }
            base.addr().to_u64()
        };
        let one_run = view(1, 2 * MIB, 64 * MIB);
        let scattered = view(512, 4096, 128 * MIB);
        let mut held = Vec::new();
        for (label, base) in [
            ("ram run", at + 2 * MIB),
            ("view of 1 run", one_run),
            ("view of 512 runs", scattered),
        ] {
            let c0 = counts();
            let t = now();
            let whole = buffer(base, 2 * MIB, None).unwrap();
            let whole_us = since(t).as_secs_f64() * 1e6;
            let c1 = counts();
            let mut times = Vec::new();
            for k in 0..512u64 {
                let t = now();
                held.push(buffer(base + k * 4096, 4096, None).unwrap());
                times.push(since(t).as_secs_f64() * 1e6);
            }
            let pin = freed_range(2 * MIB);
            let (mut pinned, mut nil) = (Vec::new(), 0);
            for k in 0..512u64 {
                let t = now();
                match buffer(base + k * 4096, 4096, Some(pin + k * 4096)) {
                    Some(b) => {
                        pinned.push(since(t).as_secs_f64() * 1e6);
                        assert_eq!(b.gpu_address(), pin + k * 4096);
                        held.push(b);
                    }
                    None => nil += 1,
                }
            }
            let c3 = counts();
            let (p50, max) = median(&mut times);
            let (pin50, pinmax) = median(&mut pinned);
            println!(
                "slab: no-copy {label} whole_us={whole_us:.1} sub4k_p50_us={p50:.2} \
                 sub4k_max_us={max:.1} pinned_p50_us={pin50:.2} pinned_max_us={pinmax:.1} \
                 pinned_nil={nil} whole(chunks={} bound={} ptr_imports={} copied={}) \
                 subs(chunks={} bound={} ptr_imports={} copied={})",
                c1.0 - c0.0,
                c1.1 - c0.1,
                c1.2 - c0.2,
                c1.3 - c0.3,
                c3.0 - c1.0,
                c3.1 - c1.1,
                c3.2 - c1.2,
                c3.3 - c1.3,
            );
            held.push(whole);
        }
        // The GPU's fill through the last pinned buffer of the one-run view lands in the file's
        // page it maps: the view's last page, at 64 MiB + 2 MiB - 4 KiB of the RAM.
        let last = held
            .iter()
            .rev()
            .find(|b| b.contents().addr().to_u64() == one_run + 2 * MIB - 4096)
            .unwrap();
        let q = device.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        let fill = BlitCmd::FillBuffer {
            buffer: last,
            range: NsRange::new(16, 32),
            value: 0x5a,
        };
        e.encode(OnFault::Die, fill).unwrap();
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        // SAFETY: the RAM, mapped readable, which no GPU work uses now.
        let read = |offset: u64| unsafe {
            std::ptr::with_exposed_provenance::<u8>((at + offset).to_usize()).read_volatile()
        };
        let page = 64 * MIB + 2 * MIB - 4096;
        assert_eq!(
            [
                read(page + 15),
                read(page + 16),
                read(page + 47),
                read(page + 48)
            ],
            [0, 0x5a, 0x5a, 0]
        );
        drop((cb, e, q, held, lease));
    }
    /// The RAM's source of dma-bufs ([`super::dma_bufs`]) is there where `/dev/udmabuf` opens, and
    /// answers a request over the RAM's pages ([`super::ram_dma_bufs`]) with dma-bufs of exactly
    /// those pages, in order, split only at multiples of the request's granule and each within
    /// udmabuf's limits: what the CPU wrote through the RAM reads through them, and what is
    /// written through them reaches the RAM. A request over pages no file is behind (anonymous
    /// memory, not the RAM's) is refused with an error, which leaves the chunk to its pointer.
    /// Without udmabuf the RAM has no source, and the test checks nothing more.
    #[test]
    fn the_rams_source_answers_dma_bufs_of_its_pages() {
        let opens = super::super::dma_buf::opens();
        assert_eq!(super::dma_bufs().is_some(), opens);
        if !opens {
            eprintln!("skipped: /dev/udmabuf does not open on this host");
            return;
        }
        let length = 8 * MIB;
        let fd = memfd(length).unwrap();
        // SAFETY: a fresh shared mapping of the file, which the test unmaps after the lease and
        // every dma-buf over it.
        let start = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length.to_usize(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        assert_ne!(start, libc::MAP_FAILED);
        let at = start.addr().to_u64();
        let lease = lease_guest_memory([GuestRamFile {
            host: at,
            guest: 0,
            length,
            file: fd,
            offset: 0,
        }]);
        let ram =
            |offset: u64| std::ptr::with_exposed_provenance_mut::<u8>((at + offset).to_usize());
        // SAFETY: bytes of the RAM, mapped readable and writable until the end of the test.
        unsafe {
            ram(2 * MIB + 5).write_volatile(0x11);
            ram(6 * MIB - 1).write_volatile(0x22);
        }
        // A dma-buf's bytes, mapped shared for `f` and unmapped after.
        let through = |chunk: &metal2vulkan_runtime::DmaBufChunk, f: &mut dyn FnMut(*mut u8)| {
            // SAFETY: a fresh shared mapping of the dma-buf, unmapped once `f` returns.
            unsafe {
                let mapped = libc::mmap(
                    std::ptr::null_mut(),
                    chunk.length.to_usize(),
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    chunk.file.as_raw_fd(),
                    0,
                );
                assert_ne!(mapped, libc::MAP_FAILED);
                f(mapped.cast());
                assert_eq!(libc::munmap(mapped, chunk.length.to_usize()), 0);
            }
        };
        let limit = super::super::dma_buf::limits().bytes;
        for (offset, bytes, granule) in [(2 * MIB, 4 * MIB, 4 * MIB), (MIB, 6 * MIB, MIB)] {
            let chunks = super::ram_dma_bufs(at + offset, bytes, granule).unwrap();
            let mut ends = 0;
            for chunk in &chunks {
                assert!(
                    chunk.length > 0 && chunk.length <= limit,
                    "{}",
                    chunk.length
                );
                ends += chunk.length;
                assert!(
                    ends == bytes || ends.is_multiple_of(granule),
                    "a cut at {ends:#x}"
                );
            }
            assert_eq!(ends, bytes, "the request's pages, no more");
        }
        let chunks = super::ram_dma_bufs(at + 2 * MIB, 4 * MIB, 4 * MIB).unwrap();
        assert_eq!(chunks.len(), 1);
        through(&chunks[0], &mut |bytes| {
            // SAFETY: the dma-buf's mapping, 4 MiB long.
            unsafe {
                assert_eq!(bytes.add(5).read_volatile(), 0x11);
                assert_eq!(bytes.add((4 * MIB - 1).to_usize()).read_volatile(), 0x22);
                bytes.add(0x100).write_volatile(0x33);
            }
        });
        // SAFETY: a byte of the RAM, mapped readable.
        assert_eq!(unsafe { ram(2 * MIB + 0x100).read_volatile() }, 0x33);
        // Anonymous memory is no file's: the source refuses it.
        // SAFETY: a fresh private anonymous mapping, unmapped below.
        let anonymous = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                (2 * MIB).to_usize(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(anonymous, libc::MAP_FAILED);
        let refused = super::ram_dma_bufs(anonymous.addr().to_u64(), 2 * MIB, 2 * MIB);
        assert!(refused.is_err());
        drop((chunks, lease));
        // SAFETY: the mappings made above, which nothing uses any more.
        unsafe {
            assert_eq!(libc::munmap(anonymous, (2 * MIB).to_usize()), 0);
            assert_eq!(libc::munmap(start, length.to_usize()), 0);
        }
    }

    /// The RAM's source of lists of runs ([`super::dma_bufs_of_runs`]) is there where
    /// `/dev/udmabuf` opens, and answers a list of runs of the RAM's pages
    /// ([`super::ram_dma_buf_of_runs`]) with one dma-buf of those pages one after another, its
    /// length theirs: what the CPU wrote through the RAM reads through it, and what is written
    /// through it reaches the RAM, in each run's own pages. A list with pages no file is behind
    /// (anonymous memory, not the RAM's) is refused with an error, which leaves the copy to what
    /// it takes without the source.
    #[test]
    fn the_rams_source_of_lists_answers_one_dma_buf_of_the_runs_in_order() {
        let opens = super::super::dma_buf::opens();
        assert_eq!(super::dma_bufs_of_runs().is_some(), opens);
        if !opens {
            eprintln!("skipped: /dev/udmabuf does not open on this host");
            return;
        }
        let length = 8 * MIB;
        let fd = memfd(length).unwrap();
        // SAFETY: a fresh shared mapping of the file, which the test unmaps after the lease and
        // the dma-buf over it.
        let start = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length.to_usize(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        assert_ne!(start, libc::MAP_FAILED);
        let at = start.addr().to_u64();
        let lease = lease_guest_memory([GuestRamFile {
            host: at,
            guest: 0,
            length,
            file: fd,
            offset: 0,
        }]);
        let ram =
            |offset: u64| std::ptr::with_exposed_provenance_mut::<u8>((at + offset).to_usize());
        // Three runs, by address and apart: pages 1 and 2, page 5, and the page 7 MiB in.
        let runs = [
            (at + PAGE, 2 * PAGE),
            (at + 5 * PAGE, PAGE),
            (at + 7 * MIB, PAGE),
        ];
        // SAFETY: bytes of the RAM, mapped readable and writable until the end of the test.
        unsafe {
            ram(PAGE + 3).write_volatile(0x11);
            ram(5 * PAGE).write_volatile(0x22);
            ram(7 * MIB + PAGE - 1).write_volatile(0x33);
        }
        let chunk = super::ram_dma_buf_of_runs(&runs).unwrap();
        assert_eq!(chunk.length, 4 * PAGE, "the runs' pages, no more");
        // SAFETY: a fresh shared mapping of the dma-buf, unmapped below.
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                chunk.length.to_usize(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                chunk.file.as_raw_fd(),
                0,
            )
        };
        assert_ne!(mapped, libc::MAP_FAILED);
        let bytes = mapped.cast::<u8>();
        // SAFETY: the dma-buf's mapping, four pages long; then a byte of the RAM, mapped readable.
        unsafe {
            assert_eq!(
                bytes.add((PAGE + 3).to_usize()).read_volatile(),
                0,
                "page 2's"
            );
            assert_eq!(bytes.add(3).read_volatile(), 0x11, "page 1's, first");
            assert_eq!(bytes.add((2 * PAGE).to_usize()).read_volatile(), 0x22);
            assert_eq!(bytes.add((4 * PAGE - 1).to_usize()).read_volatile(), 0x33);
            bytes.add((3 * PAGE + 8).to_usize()).write_volatile(0x44);
            assert_eq!(libc::munmap(mapped, chunk.length.to_usize()), 0);
            assert_eq!(
                ram(7 * MIB + 8).read_volatile(),
                0x44,
                "into the last run's page"
            );
        }
        // Anonymous memory is no file's: the source refuses a list with it.
        // SAFETY: a fresh private anonymous mapping, unmapped below.
        let anonymous = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                PAGE.to_usize(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(anonymous, libc::MAP_FAILED);
        let mut with = vec![(anonymous.addr().to_u64(), PAGE)];
        with.extend_from_slice(&runs);
        with.sort_unstable();
        assert!(super::ram_dma_buf_of_runs(&with).is_err());
        drop((chunk, lease));
        // SAFETY: the mappings made above, which nothing uses any more.
        unsafe {
            assert_eq!(libc::munmap(anonymous, PAGE.to_usize()), 0);
            assert_eq!(libc::munmap(start, length.to_usize()), 0);
        }
    }
}
