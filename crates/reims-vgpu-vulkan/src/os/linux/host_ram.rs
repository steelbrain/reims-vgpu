//! The guest's RAM handed to a device whole. Where the device takes no view of the guest's pages
//! through dma-bufs (`super::import`'s `imports_need_files`, NVIDIA's 615 driver), the runtime
//! imports it itself, in chunks, as objects over it are made, and takes each no-copy object's
//! pages as runs of that RAM ([`runs`]) instead of an import of the object's own view, so such an
//! object takes no kernel mapping a page. Where it takes views through dma-bufs (RADV), the RAM
//! serves copies by runs alone: objects keep the imports of their own views.
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
//!   held imports cost every submission (RADV): the copy imports its own pages, in lists, held
//!   until its command buffer completes, and the RAM's chunks are never imported. A chunk's
//!   import would pin its pages, which then stay resident: before the lists, a Radeon 780M
//!   (RADV, 7.3.1) handed the RAM took a 16 GiB guest's RAM in chunks of about 2.5 to 3 GiB for
//!   its copies, and 12.3 GiB of it was resident by the desktop and at the end, where views alone
//!   kept 7.9 to 8.0 GiB by the desktop and 10.4 to 10.8 GiB at the end.
//!
//! - **What is handed.** Each lease's ranges of the guest's RAM (`super::files`'s
//!   `lease_guest_memory`): the host's own shared mappings of the files it named, writable, mapped
//!   while the lease lives. Nothing else: read-only memory, files of zeros, twins and the library's
//!   own memory answer no runs, and keep the imports they have.
//! - **To whom.** Every device whose imports of views were learned (`super::import`'s
//!   `learn_view_imports`), as each is learned ([`hand_ram_to`]) and as each lease is made
//!   ([`lease`]): a device made after a lease gets its RAM then, and a lease made after the device
//!   gets it at once. One that takes views through dma-bufs takes the RAM where udmabuf opens
//!   alone, as its views do, so its copies by runs always have the source of lists.
//! - **For how long.** A device's use of a lease's RAM goes with the lease ([`LeasedRam`]'s drop),
//!   which drops after core's teardown released every object over the RAM; the runtime frees the
//!   RAM's imports once no command buffer holds them, and then drops the RAM's keep-alive, a keep
//!   of each range (`crate::remap::ImportKeep`), which an owner that unmaps the RAM waits for
//!   first (`crate::remap::wait_unimported`, a test's RAM).
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

use crate::core::num::ToU64 as _;
use crate::core::sync::Mutex;
use metal2vulkan_runtime as rt;

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

/// A lease's RAM, `ranges` (each a host address and a length), handed to every device that takes
/// it now, and to each one learned while the answer lives.
pub(crate) fn lease(ranges: &[(u64, u64)]) -> Arc<LeasedRam> {
    let ram = Arc::new(LeasedRam {
        uses: Mutex::new(Vec::new()),
        users: Mutex::new(Vec::new()),
        ranges: ranges.to_vec(),
        ram: runtime_ram(ranges).map(Arc::new),
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

/// Takes the devices that take the RAM, for the platform's let-go of its devices
/// ([`crate::let_go`]): a device adopted after it is handed the RAM as it learns its imports.
pub(crate) fn take_devices() -> Vec<rt::Device> {
    std::mem::take(&mut REGISTRY.lock().devices)
}

/// Hands `device`, whose imports of views were learned (`super::import`'s `learn_view_imports`),
/// every lease's RAM alive now, and every one made after.
pub(crate) fn hand_ram_to(device: &rt::Device) {
    let leases: Vec<Arc<LeasedRam>> = {
        let mut registry = REGISTRY.lock();
        registry.devices.push(device.clone());
        registry.leases.iter().filter_map(Weak::upgrade).collect()
    };
    for ram in &leases {
        hand(device, ram);
    }
}

/// The runtime's RAM of `ranges`; `None`, with a warning, where the runtime refuses them.
fn runtime_ram(ranges: &[(u64, u64)]) -> Option<rt::HostRam> {
    let pairs: Vec<(NonNull<u8>, u64)> = (ranges.iter())
        .filter_map(|&(at, length)| {
            Some((
                NonNull::new(std::ptr::with_exposed_provenance_mut(at.try_into().ok()?))?,
                length,
            ))
        })
        .collect();
    // The keep-alive the runtime drops once no device holds a chunk of the RAM: a keep of each
    // range, which a wait for the imports over the RAM reads (`crate::remap::wait_unimported`).
    let keeps: Vec<crate::remap::ImportKeep> = (pairs.iter())
        .map(|&(at, length)| crate::remap::ImportKeep::new(at, length))
        .collect();
    // SAFETY: each range is the host's own mapping of the guest's RAM, which the host keeps
    // mapped, readable and writable, with the same backing, while any device made over its
    // memory map lives (`docs/api.md`, `RvgMemoryRange`), and until the runtime let go of the
    // keep-alive where the host unmaps it (`crate::remap`, "Memory the library does not
    // unmap"); every device's use of it goes with the lease, which drops after core's teardown,
    // and the runtime holds the RAM no longer than its uses and the objects and command buffers
    // over it.
    let keep = crate::gpu::backing::keep_alive(keeps);
    let ram = match unsafe { rt::HostRam::new(&pairs, Some(keep)) } {
        Ok(ram) => ram,
        Err(refusal) => {
            log::warn!(
                target: "reims_vgpu_vulkan::os", kind:% = refusal;
                "the guest's RAM is not handed to devices whole ({refusal}): buffers over its \
                 pages keep imports of their own"
            );
            return None;
        }
    };
    Some(match (dma_bufs(), dma_bufs_of_runs()) {
        (Some(source), Some(lists)) => {
            let limits = super::dma_buf::limits();
            (ram.with_dma_buf(source))
                .with_largest_dma_buf(limits.bytes)
                .with_dma_buf_of_runs(lists, limits.pieces.try_into().unwrap_or(usize::MAX))
        }
        _ => ram,
    })
}

/// The RAM's source of one dma-buf of a list of runs of its pages
/// (`rt::HostRam::with_dma_buf_of_runs`), where `/dev/udmabuf` opens: each list answered by
/// [`ram_dma_buf_of_runs`].
fn dma_bufs_of_runs() -> Option<rt::DmaBufRunsSource> {
    if !super::dma_buf::opens() {
        return None;
    }
    let zeros = super::dma_buf::Zeros::default();
    Some(Arc::new(move |requests: &[rt::DmaBufRequest]| {
        let ranges: Vec<(u64, u64)> = (requests.iter())
            .map(|r| (r.start.as_ptr().addr().to_u64(), r.length))
            .collect();
        ram_dma_buf_of_runs(&ranges, &zeros)
    }))
}

/// The dma-buf [`dma_bufs_of_runs`]'s source answers for `ranges` of the RAM's pages, each a host
/// address and a length, whole pages by address and apart: those pages one after another,
/// resolved through the backing table (`super::dma_buf::of_ranges`). A refusal (a page the table
/// resolves to no file, a limit, udmabuf's own) answers an error, which leaves the copy to what it
/// takes without the source. The RAM's pages are never the block of zeros', so `zeros` stays
/// empty.
fn ram_dma_buf_of_runs(
    ranges: &[(u64, u64)],
    zeros: &super::dma_buf::Zeros,
) -> std::io::Result<rt::DmaBufChunk> {
    let length = ranges.iter().map(|&(_, n)| n).sum();
    match super::dma_buf::of_ranges(ranges, zeros) {
        Ok(fd) => Ok(rt::DmaBufChunk { fd, length }),
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
    let zeros = super::dma_buf::Zeros::default();
    Some(Arc::new(move |request: rt::DmaBufRequest| {
        let first = request.start.as_ptr().addr().to_u64();
        ram_dma_bufs(first, request.length, request.granule, &zeros)
    }))
}

/// The dma-bufs [`dma_bufs`]'s source answers for `length` bytes of the RAM's pages from `first`,
/// host addresses, at `granule`: those pages resolved through the backing table and cut within
/// udmabuf's limits at multiples of `granule` from `first` (`super::dma_buf::of_pages`), as a
/// no-copy import's source makes them. A refusal (pages the table resolves to no file, a limit,
/// udmabuf's own) answers an error, which leaves the chunk to its pointer. The RAM's pages are its
/// files' and never the block of zeros', so `zeros`, the source's files of zeros, stays empty.
fn ram_dma_bufs(
    first: u64,
    length: u64,
    granule: u64,
    zeros: &super::dma_buf::Zeros,
) -> std::io::Result<Vec<rt::DmaBufChunk>> {
    super::dma_buf::of_pages(first, length, granule, zeros).map_err(|refusal| {
        log::debug!(
            target: "reims_vgpu_vulkan::os",
            "no dma-buf of the guest's RAM's {length:#x} bytes at {first:#x}: {refusal}"
        );
        std::io::Error::other(refusal.to_string())
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
pub(crate) fn ram_over<'a>(
    device: &rt::Device,
    runs: impl Iterator<Item = &'a Range<u64>> + Clone,
) -> Option<Arc<rt::HostRam>> {
    let id = device.registry_id();
    live_leases().into_iter().find_map(|l| {
        let ram = l.ram.as_ref()?;
        let used = l.users.lock().contains(&id);
        (used && runs.clone().all(|r| l.holds(r))).then(|| Arc::clone(ram))
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
    log::warn!(
        target: "reims_vgpu::vulkan", kind:% = error;
        "the runtime refused a copy by runs: {error}"
    );
    RunsCopyRefused
}

/// The runtime's copy of `source` into `runs` of `ram`, each a byte of `source` and a run of
/// host addresses that takes the bytes from that byte on, on `encoder`: one call however many
/// runs, as [`copy_from_host_ram`] copies the other way.
pub(crate) fn copy_to_host_ram(
    encoder: &mut rt::BlitCommandEncoder,
    source: &rt::Buffer,
    ram: &rt::HostRam,
    runs: &[(u64, Range<u64>)],
) -> Result<(), RunsCopyRefused> {
    (encoder.copy_runs_to_host_ram(source, ram, runs)).map_err(refused)
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

    use crate::core::host::HostPtr;
    use crate::core::num::{ToU64 as _, ToUsize as _};
    use crate::core::platform::HostRange;
    use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
    use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::os::vm::HostVm as _;
    use crate::core::platform::resource::Buffer as _;
    use crate::core::platform::types::NsRange;
    use crate::core::raise::OnFault;

    use super::super::import::imports_need_files;
    use super::super::vm::memfd;
    use super::super::{PAGE, lease_guest_memory};
    use crate::gpu::device::tests::{dev, settle_device_checks};
    use crate::os::GuestRamFile;
    use crate::remap::VulkanVm;

    const MIB: u64 = 1 << 20;

    /// Where the device needs imported pages mapped from their files (NVIDIA's), it uses the
    /// guest's RAM whole: a no-copy buffer over one run of it is bound in the RAM's chunk, which
    /// the device imports once, a second buffer in the chunk importing nothing more, and the GPU's
    /// fill through each reaches the RAM. A device that takes views through dma-bufs is handed the
    /// RAM for its copies by runs alone, and its buffers import their pages as before. Either way
    /// the bytes are the GPU's. The flag is the process's, which nextest runs this test alone in.
    ///
    /// The runtime imports the chunk as the first object over it asks, where the RAM has a source
    /// of dma-bufs (`/dev/udmabuf` opens), or on its own thread ahead of every object, where it has
    /// none. An object made before the chunk's import ends is bound in an import of its own run,
    /// waiting for none, but where the device's barriers do not order blits through two imports
    /// of the same pages (ANV's, MoltenVK's), where it is bound in the chunk (metal2vulkan's
    /// `host_ram.rs`, "No object waits for a chunk"). So the test makes the first buffer, waits
    /// for the chunk's import, and makes the second, which lies in the chunk; the first is bound
    /// once, in one of the two.
    #[test]
    fn a_buffer_over_one_run_of_the_ram_is_bound_in_its_chunk() {
        let device = dev();
        // The device's own checks end first: the chunk's import and the counts wait for none.
        settle_device_checks(&device);
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
        let lease = lease_guest_memory([GuestRamFile {
            host: at,
            guest: 0,
            length,
            fd,
            offset: 0,
        }]);
        let over = |offset: u64, length: u64| {
            // SAFETY: pages of the test's RAM, mapped until after the buffer is dropped.
            let bytes =
                unsafe { HostRange::new(HostPtr::from_addr((at + offset).to_usize()), length) };
            (device.new_buffer_over_host_memory(OnFault::Die, bytes, 0, None))
                .unwrap()
                .unwrap()
        };
        let a = over(MIB, 2 * PAGE);
        let after_a = counts();
        if imports_need_files() {
            // The RAM is one chunk (8 MiB): its import is given 20 s to end.
            let imported = (0..4000).any(|_| {
                let done = counts().0 > before.0;
                if !done {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                done
            });
            assert!(imported, "the runtime imported no chunk of the RAM");
        }
        let imported = counts();
        let b = over(3 * MIB, PAGE);
        let now = counts();
        let first = (after_a.1 - before.1, after_a.2 - before.2);
        let second = (now.0 - imported.0, now.1 - imported.1, now.2 - imported.2);
        match imports_need_files() {
            true => {
                assert_eq!(now.0 - before.0, 1, "the chunk imported once");
                assert!(
                    first == (1, 0) || first == (0, 1),
                    "the first buffer bound once, in the chunk or its run: {first:?}"
                );
                assert_eq!(second, (0, 1, 0), "the second bound in the chunk");
            }
            false => {
                let made = (now.0 - before.0, now.1 - before.1, now.2 - before.2);
                assert_eq!(made, (0, 0, 0), "the RAM serves copies alone");
            }
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
        // Once the runtime freed its imports of the RAM's chunks (the RAM's keep-alive), through
        // the platform's VM, which leaves pages a buffer's import still holds mapped until the
        // runtime has freed it: the device lives on past the lease.
        let start_at = HostPtr::from_ptr(start.cast::<u8>());
        crate::remap::free_unimported(start.cast(), length, || {
            assert!(VulkanVm::new().deallocate(start_at, length).is_success());
        });
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
        let lease = lease_guest_memory([GuestRamFile {
            host: at,
            guest: 0,
            length,
            fd,
            offset: 0,
        }]);
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
        let mut next = 0;
        let into: Vec<(u64, std::ops::Range<u64>)> = (runs.iter())
            .map(|run| {
                next += run.end - run.start;
                (next - (run.end - run.start), run.clone())
            })
            .collect();
        let scatter = e.copy_to_guest_ram(OnFault::Die, &from, &into).unwrap();
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
        // Once the runtime freed its imports of the RAM's chunks (the RAM's keep-alive), through
        // the platform's VM, which leaves pages a buffer's import still holds mapped until the
        // runtime has freed it: the device lives on past the lease.
        let start_at = HostPtr::from_ptr(start.cast::<u8>());
        crate::remap::free_unimported(start.cast(), length, || {
            assert!(VulkanVm::new().deallocate(start_at, length).is_success());
        });
    }

    /// A device that copies the RAM by its runs scatters a buffer into runs that start and end
    /// inside pages, as a laid-out device cache's copy out does (`reims-vgpu`'s
    /// `graphics/decode/surface_cache.rs`, `texel_spans`): each of its surface's rows' texels
    /// alone, cut to the pages they lie in, all of them in one copy, each run from its own byte of
    /// the buffer. Here a plane of 2-byte texels 1,023 wide, from byte 16 at a pitch of 2,048
    /// bytes, over four pages out of order: each row starts on 16 bytes (the guest aligns a
    /// plane's offset and pitch to its linear texture alignment, never less) and ends 2,046 bytes
    /// on, off a 4-byte word, some rows crossing from one page into another. The runs' bytes are
    /// the buffer's, and every byte beside them, the guest CPU's, is as it was. One that does not
    /// copy the RAM refuses the copy, encoding nothing.
    #[test]
    fn a_copy_by_runs_scatters_runs_inside_pages() {
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
        let lease = lease_guest_memory([GuestRamFile {
            host: at,
            guest: 0,
            length,
            fd,
            offset: 0,
        }]);
        // SAFETY: the RAM's first pages, mapped until the end, which no GPU work uses now.
        let ram =
            unsafe { std::slice::from_raw_parts_mut(start.cast::<u8>(), (8 * PAGE).to_usize()) };
        for (i, b) in ram.iter_mut().enumerate() {
            *b = (i * 7 + i / PAGE.to_usize()) as u8;
        }
        let before = ram.to_vec();
        // The surface's pages, in its order: page 5, then 1, 2 and 0 of the RAM.
        let pages = [5u64, 1, 2, 0];
        let (offset, pitch, row_bytes, rows) = (16u64, 2048u64, 2046u64, 7u64);
        // Each row's texels as runs of the RAM, cut where the surface's pages end.
        let runs_of = |span: std::ops::Range<u64>| -> Vec<std::ops::Range<u64>> {
            let mut out = Vec::new();
            let mut s = span.start;
            while s < span.end {
                let (page, within) = (s / PAGE, s % PAGE);
                let n = (PAGE - within).min(span.end - s);
                let host = at + pages[page.to_usize()] * PAGE + within;
                out.push(host..host + n);
                s += n;
            }
            out
        };
        let spans: Vec<std::ops::Range<u64>> = (0..rows)
            .map(|r| offset + r * pitch..offset + r * pitch + row_bytes)
            .collect();
        assert!(
            spans.iter().any(|s| runs_of(s.clone()).len() == 2),
            "a row crosses from one page into another"
        );
        let total = 4 * PAGE;
        let q = device.new_command_queue().unwrap();
        let from = (device.new_buffer_of_length(OnFault::Die, total, 0))
            .unwrap()
            .unwrap();
        // SAFETY: the buffer's Shared contents, `total` bytes, written before the GPU uses it.
        let src = unsafe {
            std::slice::from_raw_parts_mut(from.contents().as_ptr::<u8>(), total.to_usize())
        };
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i * 13 + 1) as u8;
        }
        let copies = super::copies(&device.raw);
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        let mut into = Vec::new();
        for span in &spans {
            let mut s = span.start;
            for run in runs_of(span.clone()) {
                let n = run.end - run.start;
                into.push((s, run));
                s += n;
            }
        }
        let scattered = (e.copy_to_guest_ram(OnFault::Die, &from, &into)).unwrap();
        assert_eq!(scattered, copies, "every row's runs in one copy");
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        let mut want = before.clone();
        if copies {
            for span in &spans {
                let mut s = span.start;
                for run in runs_of(span.clone()) {
                    let n = run.end - run.start;
                    let into = (run.start - at).to_usize();
                    want[into..into + n.to_usize()]
                        .copy_from_slice(&src[s.to_usize()..(s + n).to_usize()]);
                    s += n;
                }
            }
        }
        assert!(
            ram == want.as_slice(),
            "the rows' texels in their runs; every byte beside them as it was"
        );
        drop((cb, e, q, from, lease));
        // Once the runtime freed its imports of the RAM's chunks (the RAM's keep-alive), through
        // the platform's VM, which leaves pages a buffer's import still holds mapped until the
        // runtime has freed it: the device lives on past the lease.
        let start_at = HostPtr::from_ptr(start.cast::<u8>());
        crate::remap::free_unimported(start.cast(), length, || {
            assert!(VulkanVm::new().deallocate(start_at, length).is_success());
        });
    }

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
            fd: std::sync::Arc::clone(&fd),
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
            fd: std::sync::Arc::clone(&fd),
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
            fd,
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
                    chunk.fd.as_raw_fd(),
                    0,
                );
                assert_ne!(mapped, libc::MAP_FAILED);
                f(mapped.cast());
                assert_eq!(libc::munmap(mapped, chunk.length.to_usize()), 0);
            }
        };
        let zeros = super::super::dma_buf::Zeros::default();
        let limit = super::super::dma_buf::limits().bytes;
        for (offset, bytes, granule) in [(2 * MIB, 4 * MIB, 4 * MIB), (MIB, 6 * MIB, MIB)] {
            let chunks = super::ram_dma_bufs(at + offset, bytes, granule, &zeros).unwrap();
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
        let chunks = super::ram_dma_bufs(at + 2 * MIB, 4 * MIB, 4 * MIB, &zeros).unwrap();
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
        let refused = super::ram_dma_bufs(anonymous.addr().to_u64(), 2 * MIB, 2 * MIB, &zeros);
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
            fd,
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
        let zeros = super::super::dma_buf::Zeros::default();
        let chunk = super::ram_dma_buf_of_runs(&runs, &zeros).unwrap();
        assert_eq!(chunk.length, 4 * PAGE, "the runs' pages, no more");
        // SAFETY: a fresh shared mapping of the dma-buf, unmapped below.
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                chunk.length.to_usize(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                chunk.fd.as_raw_fd(),
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
        assert!(super::ram_dma_buf_of_runs(&with, &zeros).is_err());
        drop((chunk, lease));
        // SAFETY: the mappings made above, which nothing uses any more.
        unsafe {
            assert_eq!(libc::munmap(anonymous, PAGE.to_usize()), 0);
            assert_eq!(libc::munmap(start, length.to_usize()), 0);
        }
    }
}
