//! The guest's RAM handed to a device whole, where the device takes no view of the guest's pages
//! through dma-bufs (`super::import`'s `imports_need_files`, NVIDIA's 615 driver): the runtime
//! imports it by pointer itself, in chunks, as objects over it are made, and takes each no-copy
//! object's pages as runs of that RAM ([`runs`]) instead of an import of the object's own view,
//! so such an object takes no kernel mapping a page.
//!
//! - **What is handed.** Each lease's ranges of the guest's RAM (`super::files`'s
//!   `lease_guest_memory`): the host's own shared mappings of the files it named, writable, mapped
//!   while the lease lives. Nothing else: read-only memory, files of zeros, twins and the library's
//!   own memory answer no runs, and keep the imports they have.
//! - **To whom.** Every device that needs imported pages mapped from their files, as each such
//!   device is learned ([`device_needs_ram`]) and as each lease is made ([`lease`]): a device made
//!   after a lease gets its RAM then, and a lease made after the device gets it at once.
//! - **For how long.** A device's use of a lease's RAM goes with the lease ([`LeasedRam`]'s drop),
//!   which drops after core's teardown released every object over the RAM; the runtime frees the
//!   RAM's imports once no command buffer holds them.
//! - **Nothing may discard the RAM's pages** while a device uses it (a balloon, free page
//!   reporting, memory hot-unplug): the device holds the pages it imported, and would go on
//!   reading the old ones. The VMs `reims-ctl` makes have none.
//!
//! - **Copies by runs.** A device that uses a lease's RAM copies between its buffers and runs of
//!   the RAM by their host addresses ([`copies`], [`ram_over`]), so a surface's device cache needs
//!   no view of the surface's pages (`reims-vgpu`'s `graphics/decode/surface_cache.rs`). The
//!   runtime gathers and scatters the runs' 4 KiB pages from its imports of the RAM's chunks; a
//!   copy it refuses encodes nothing, and the cache takes its view then.
//!
//! The registry's lock is a leaf: the runtime is called after it is let go.

use std::ops::Range;
use std::ptr::NonNull;
use std::sync::{Arc, Weak};

use metal2vulkan_runtime as rt;
use reims_vgpu_core::sync::Mutex;

use super::files;

/// One lease's RAM as the devices are handed it: the runtime's RAM of its ranges (`None` where
/// the runtime refused them), and each device's use of it, which ends with the lease, before the
/// RAM is let go.
#[derive(Debug)]
pub(crate) struct LeasedRam {
    uses: Mutex<Vec<rt::HostRamUse>>,
    /// The registry IDs of the devices that use it, one for each of `uses`.
    users: Mutex<Vec<u64>>,
    /// Its ranges, each a host address and a length.
    ranges: Vec<(u64, u64)>,
    ram: Option<Arc<rt::HostRam>>,
}

impl LeasedRam {
    /// Whether `run`, host addresses, lies in one of its ranges.
    fn holds(&self, run: &Range<u64>) -> bool {
        run.start < run.end
            && (self.ranges.iter())
                .any(|&(at, n)| run.start >= at && run.end <= at.saturating_add(n))
    }
}

/// The devices that take the RAM, and the leases' RAM alive now.
struct Registry {
    devices: Vec<rt::Device>,
    leases: Vec<Weak<LeasedRam>>,
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry {
    devices: Vec::new(),
    leases: Vec::new(),
});

/// A lease's RAM, `ranges` (each a host address and a length), with the answers of its dirty log
/// where `logged` (the device's log over the RAM, `super::dirty`), handed to every device that
/// takes it now, and to each one learned while the answer lives.
pub(crate) fn lease(ranges: &[(u64, u64)], logged: bool) -> Arc<LeasedRam> {
    let ram = Arc::new(LeasedRam {
        uses: Mutex::new(Vec::new()),
        users: Mutex::new(Vec::new()),
        ranges: ranges.to_vec(),
        ram: runtime_ram(ranges, logged).map(Arc::new),
    });
    let devices = {
        let mut registry = REGISTRY.lock();
        registry.leases.retain(|l| l.strong_count() > 0);
        registry.leases.push(Arc::downgrade(&ram));
        registry.devices.clone()
    };
    for device in &devices {
        hand(device, &ram);
    }
    ram
}

/// Hands `device`, which needs imported pages mapped from their files, every lease's RAM alive
/// now, and every one made after.
pub(crate) fn device_needs_ram(device: &rt::Device) {
    let leases: Vec<Arc<LeasedRam>> = {
        let mut registry = REGISTRY.lock();
        registry.devices.push(device.clone());
        registry.leases.iter().filter_map(Weak::upgrade).collect()
    };
    for ram in &leases {
        hand(device, ram);
    }
}

/// The runtime's RAM of `ranges`, with the dirty log's answers for all of it where `logged`;
/// `None`, with a warning, where the runtime refuses them.
fn runtime_ram(ranges: &[(u64, u64)], logged: bool) -> Option<rt::HostRam> {
    let pairs: Vec<(NonNull<u8>, u64)> = (ranges.iter())
        .filter_map(|&(at, length)| {
            Some((
                NonNull::new(std::ptr::with_exposed_provenance_mut(at.try_into().ok()?))?,
                length,
            ))
        })
        .collect();
    // SAFETY: each range is the host's own mapping of the guest's RAM, which the host keeps
    // mapped, readable and writable, with the same backing, while any device made over its
    // memory map lives (`docs/api.md`, `RvgMemoryRange`); every device's use of it goes with the
    // lease, which drops after core's teardown, and the runtime holds the RAM no longer than its
    // uses and the objects and command buffers over it.
    let ram = match unsafe { rt::HostRam::new(&pairs, None) } {
        Ok(ram) => ram,
        Err(refusal) => {
            log::warn!(
                target: "reims_vgpu_vulkan::os",
                "the guest's RAM is not handed to devices whole ({refusal}): buffers over its \
                 pages keep imports of their own"
            );
            return None;
        }
    };
    let dirty = logged
        .then(|| {
            ranges
                .first()
                .and_then(|&(at, _)| super::dirty::ImportDirty::of_ram(at))
        })
        .flatten();
    Some(match dirty {
        Some(dirty) => ram.with_dirty_pages(Arc::new(dirty)),
        None => ram,
    })
}

/// Hands `device` the lease's RAM, keeping its use with the lease. A device that takes no RAM
/// (no import of host memory by its pointer) keeps today's imports, said once.
fn hand(device: &rt::Device, ram: &LeasedRam) {
    let Some(host_ram) = &ram.ram else {
        return;
    };
    match device.use_host_ram(host_ram) {
        Ok(used) => {
            ram.uses.lock().push(used);
            ram.users.lock().push(device.registry_id());
        }
        Err(refusal) => {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                log::info!(
                    target: "reims_vgpu_vulkan::os",
                    "{}: the guest's RAM is not used whole ({refusal}): buffers over its pages \
                     keep imports of their own",
                    device.properties().name
                );
            });
        }
    }
}

/// The leases' RAM alive now, taken with the registry's lock let go.
fn live_leases() -> Vec<Arc<LeasedRam>> {
    let registry = REGISTRY.lock();
    registry.leases.iter().filter_map(Weak::upgrade).collect()
}

/// Whether `device` copies between its buffers and the guest's RAM by the RAM's runs: it uses a
/// lease's RAM ([`copy_from_host_ram`], [`copy_to_host_ram`]).
pub(crate) fn copies(device: &rt::Device) -> bool {
    let id = device.registry_id();
    live_leases().iter().any(|l| l.users.lock().contains(&id))
}

/// The RAM `device` uses whose ranges hold every one of `runs`, host addresses: what a copy by
/// the RAM's runs goes through. `None` where no lease's RAM the device uses holds them all.
pub(crate) fn ram_over(device: &rt::Device, runs: &[Range<u64>]) -> Option<Arc<rt::HostRam>> {
    let id = device.registry_id();
    live_leases().into_iter().find_map(|l| {
        let ram = l.ram.as_ref()?;
        let used = l.users.lock().contains(&id);
        (used && runs.iter().all(|r| l.holds(r))).then(|| Arc::clone(ram))
    })
}

/// The runtime refused a copy by runs of a device's RAM.
#[derive(Debug)]
pub(crate) struct RunsCopyRefused;

/// The runtime's copy of `runs` of `ram`, host addresses in the bytes' order, into `destination`
/// from `destination_offset`, on `encoder`; one it refuses encodes nothing.
pub(crate) fn copy_from_host_ram(
    encoder: &mut rt::BlitCommandEncoder,
    ram: &rt::HostRam,
    runs: &[Range<u64>],
    destination: &rt::Buffer,
    destination_offset: u64,
) -> Result<(), RunsCopyRefused> {
    (encoder.copy_from_host_ram(ram, runs, destination, destination_offset)).map_err(refused)
}

/// A refusal of a copy by runs, logged.
fn refused(error: rt::Error) -> RunsCopyRefused {
    log::warn!(target: "reims_vgpu::vulkan", "the runtime refused a copy by runs: {error}");
    RunsCopyRefused
}

/// The runtime's copy of `source` from `source_offset` into `runs` of `ram`, on `encoder`, as
/// [`copy_from_host_ram`] copies the other way.
pub(crate) fn copy_to_host_ram(
    encoder: &mut rt::BlitCommandEncoder,
    source: &rt::Buffer,
    source_offset: u64,
    ram: &rt::HostRam,
    runs: &[Range<u64>],
) -> Result<(), RunsCopyRefused> {
    (encoder.copy_to_host_ram(source, source_offset, ram, runs)).map_err(refused)
}

/// The runs of the guest's RAM behind `length` bytes from `start`, which the runtime asks of a
/// no-copy object's import window (`super::import`'s `host_memory`): host addresses in the leases'
/// ranges, in the bytes' order; `None` where a byte is no writable page of the RAM
/// (`super::files`'s `guest_ram_runs`), and the object keeps today's import.
pub(crate) fn runs(start: u64, length: u64) -> Option<Vec<Range<u64>>> {
    let runs = files::guest_ram_runs(start, length)?;
    Some(runs.into_iter().map(|(at, n)| at..at + n).collect())
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsRawFd as _;

    use reims_vgpu_core::host::HostPtr;
    use reims_vgpu_core::num::{ToU64 as _, ToUsize as _};
    use reims_vgpu_core::platform::HostRange;
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::device::GpuDevice as _;
    use reims_vgpu_core::platform::resource::Buffer as _;
    use reims_vgpu_core::platform::types::NsRange;
    use reims_vgpu_core::raise::OnFault;

    use super::super::import::imports_need_files;
    use super::super::vm::memfd;
    use super::super::{PAGE, lease_guest_memory};
    use crate::gpu::device::tests::dev;
    use crate::os::GuestRamFile;

    const MIB: u64 = 1 << 20;

    /// Where the device needs imported pages mapped from their files (NVIDIA's), it uses the
    /// guest's RAM whole: a no-copy buffer over one run of it is bound in the RAM's chunk, which
    /// the device imports once, a second buffer in the chunk importing nothing more, and the GPU's
    /// fill through each reaches the RAM. A device that takes views through dma-bufs is handed no
    /// RAM, and its buffers import their pages as before. Either way the bytes are the GPU's. The
    /// flag is the process's, which nextest runs this test alone in.
    ///
    /// The runtime's own thread imports the RAM's first chunk as the device is handed the RAM,
    /// ahead of any object over it, and a buffer made before that import ends is bound in an
    /// import of its own run, waiting for none (metal2vulkan's `host_ram.rs`, "No object waits
    /// for a chunk"). So the test waits for the chunk's import before it makes the buffers.
    #[test]
    fn a_buffer_over_one_run_of_the_ram_is_bound_in_its_chunk() {
        let device = dev();
        let length = 8 * MIB;
        let fd = memfd(length).unwrap();
        // SAFETY: a fresh shared mapping of the file, which the test unmaps after the lease and
        // every buffer over it.
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
        let counts = || {
            let c = device.raw.work_counts();
            (
                c.host_ram_chunks_imported,
                c.chunk_bound_buffers_made,
                c.run_bound_buffers_made,
            )
        };
        let before = counts();
        let lease = lease_guest_memory(
            [GuestRamFile {
                host: at,
                guest: 0,
                length,
                fd,
                offset: 0,
            }],
            None,
        );
        if imports_need_files() {
            // The RAM is one chunk (8 MiB): the runtime's thread is given 20 s to import it.
            let imported = (0..4000).any(|_| {
                let done = counts().0 > before.0;
                if !done {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                done
            });
            assert!(
                imported,
                "the runtime's thread imported no chunk of the RAM"
            );
        }
        let over = |offset: u64, length: u64| {
            // SAFETY: pages of the test's RAM, mapped until after the buffer is dropped.
            let bytes =
                unsafe { HostRange::new(HostPtr::from_addr((at + offset).to_usize()), length) };
            (device.new_buffer_over_host_memory(OnFault::Die, bytes, 0, None))
                .unwrap()
                .unwrap()
        };
        let a = over(MIB, 2 * PAGE);
        let b = over(3 * MIB, PAGE);
        let now = counts();
        let made = (now.0 - before.0, now.1 - before.1, now.2 - before.2);
        match imports_need_files() {
            true => assert_eq!(made, (1, 2, 0), "bound in one chunk"),
            false => assert_eq!(made, (0, 0, 0), "no RAM handed"),
        }
        let q = device.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        for (buffer, value) in [(&a, 0x5a), (&b, 0xa5)] {
            let fill = BlitCmd::FillBuffer {
                buffer,
                range: NsRange::new(0x10, PAGE - 0x20),
                value,
            };
            e.encode(OnFault::Die, fill).unwrap();
        }
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        // SAFETY: the RAM, mapped readable, which no GPU work uses now.
        let read = |offset: u64| unsafe {
            std::ptr::with_exposed_provenance::<u8>((at + offset).to_usize()).read_volatile()
        };
        let a_bytes = [
            read(MIB + 0xf),
            read(MIB + 0x10),
            read(MIB + PAGE - 0x11),
            read(MIB + PAGE - 0x10),
        ];
        assert_eq!(a_bytes, [0, 0x5a, 0x5a, 0]);
        let b_bytes = [
            read(3 * MIB + 0xf),
            read(3 * MIB + 0x10),
            read(3 * MIB + PAGE - 0x11),
        ];
        assert_eq!(b_bytes, [0, 0xa5, 0xa5]);
        drop((cb, e, q, a, b, lease));
        // SAFETY: the mapping made above, which nothing uses any more.
        assert_eq!(unsafe { libc::munmap(start, length.to_usize()) }, 0);
    }
    /// A device that copies the RAM by its runs ([`super::copies`]) gathers runs of it into a
    /// buffer, and scatters a buffer back into them, byte for byte in the runs' order, scattered
    /// and out of order as a surface's pages lie; one that does not refuses both, encoding
    /// nothing, the RAM as it was. The flag is the process's, which nextest runs this test alone in.
    #[test]
    fn a_copy_by_runs_gathers_and_scatters_the_rams_pages() {
        let device = dev();
        let length = 8 * MIB;
        let fd = memfd(length).unwrap();
        // SAFETY: a fresh shared mapping of the file, which the test unmaps after the lease and
        // every buffer over it.
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
        let lease = lease_guest_memory(
            [GuestRamFile {
                host: at,
                guest: 0,
                length,
                fd,
                offset: 0,
            }],
            None,
        );
        // SAFETY: the RAM's first pages, mapped until the end, which no GPU work uses now.
        let ram =
            unsafe { std::slice::from_raw_parts_mut(start.cast::<u8>(), (8 * PAGE).to_usize()) };
        for (i, b) in ram.iter_mut().enumerate() {
            *b = (i * 7 + i / PAGE.to_usize()) as u8;
        }
        let before = ram.to_vec();
        // Page 5, then 1 and 2, then 0: a surface's pages, out of order.
        let pages = [(5, 1), (1, 2), (0, 1)];
        let runs: Vec<_> = (pages.iter())
            .map(|&(p, n)| at + p * PAGE..at + (p + n) * PAGE)
            .collect();
        let gathered: Vec<u8> = (pages.iter())
            .flat_map(|&(p, n)| before[(p * PAGE).to_usize()..((p + n) * PAGE).to_usize()].to_vec())
            .collect();
        let total = 4 * PAGE;
        let q = device.new_command_queue().unwrap();
        let shared = |len: u64| {
            (device.new_buffer_of_length(OnFault::Die, len, 0))
                .unwrap()
                .unwrap()
        };
        let into = shared(total);
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        let gather = e
            .copy_from_guest_ram(OnFault::Die, &runs, &into, 0)
            .unwrap();
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        assert_eq!(gather, super::copies(&device.raw));
        // SAFETY: the buffer's Shared contents, `total` bytes, which no GPU work uses now.
        let got =
            unsafe { std::slice::from_raw_parts(into.contents().as_ptr::<u8>(), total.to_usize()) };
        if gather {
            assert!(got == gathered.as_slice(), "the pages, in the runs' order");
        } else {
            assert!(got.iter().all(|&b| b == 0), "nothing copied");
        }

        let from = shared(total);
        // SAFETY: as above, written before the GPU uses it.
        let src = unsafe {
            std::slice::from_raw_parts_mut(from.contents().as_ptr::<u8>(), total.to_usize())
        };
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i * 13 + 1) as u8;
        }
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        let scatter = e.copy_to_guest_ram(OnFault::Die, &from, 0, &runs).unwrap();
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        assert_eq!(scatter, gather);
        let mut want = before.clone();
        if scatter {
            let mut next = 0;
            for &(p, n) in &pages {
                let span = (p * PAGE).to_usize()..((p + n) * PAGE).to_usize();
                let len = span.len();
                want[span].copy_from_slice(&src[next..next + len]);
                next += len;
            }
        }
        assert!(
            ram == want.as_slice(),
            "the buffer, in the runs' order; the rest as it was"
        );
        drop((cb, e, q, into, from, lease));
        // SAFETY: the mapping made above, which nothing uses any more.
        assert_eq!(unsafe { libc::munmap(start, length.to_usize()) }, 0);
    }

    /// What a slab of guest RAM costs on this device (arch-shared-mem's design): placement heaps
    /// over one 2 MiB run of the leased RAM each, 4 KiB buffers placed in one of them, heaps over
    /// views of 2 and 32 runs apart, and a command buffer's round trip with one heap alive and with
    /// all of them. Timings only, printed as `slab:` lines: run alone, holding the GPU.
    #[test]
    #[ignore = "a measurement: run alone, holding the GPU"]
    fn slab_heap_costs() {
        use reims_vgpu_core::platform::os::{Clock as _, MonoTime};
        use reims_vgpu_core::platform::resource::Heap as _;
        use reims_vgpu_core::platform::values::{HeapDesc, OptionSet};
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
        // Every page touched first, as a guest's superpages are.
        // SAFETY: the mapping above, `length` bytes.
        unsafe { std::ptr::write_bytes(start.cast::<u8>(), 0, length.to_usize()) };
        let at = start.addr().to_u64();
        let lease = lease_guest_memory(
            [GuestRamFile {
                host: at,
                guest: 0,
                length,
                fd: std::sync::Arc::clone(&fd),
                offset: 0,
            }],
            None,
        );
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
                address_ranges: None,
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
        // Views of runs: 2 MiB runs 4 MiB apart from 4,100 MiB on (2 runs, 4 MiB; 32 runs, 64 MiB), as
        // superpage slabs of several superpages are; 4 KiB runs 8 KiB apart from 4,300 MiB on (512
        // runs, 2 MiB; 4,096 runs, 16 MiB), as a slab of scattered guest pages is.
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
        // SAFETY: the mapping made above, which nothing uses any more.
        assert_eq!(unsafe { libc::munmap(start, length.to_usize()) }, 0);
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
        use reims_vgpu_core::platform::os::{Clock as _, MonoTime};
        use reims_vgpu_core::platform::resource::Buffer as _;
        use reims_vgpu_core::platform::values::{BufferDesc, OptionSet};
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
        let lease = lease_guest_memory(
            [GuestRamFile {
                host: at,
                guest: 0,
                length,
                fd: std::sync::Arc::clone(&fd),
                offset: 0,
            }],
            None,
        );
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
}
