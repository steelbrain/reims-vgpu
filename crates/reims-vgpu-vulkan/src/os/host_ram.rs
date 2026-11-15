//! The guest's RAM handed to a device whole, on Linux and on Windows. The runtime takes each
//! no-copy object's pages as runs of that RAM ([`runs`], which each OS's `import` gives every
//! import) and binds the object in the RAM's chunk that holds them, where the chunk is imported,
//! instead of an import of the object's own view, so such an object takes no import or kernel
//! mapping of its own. Where the imports the device holds cost a submission nothing (NVIDIA's 615
//! driver), the runtime imports the RAM itself, in chunks, ahead of the objects over it. Where
//! each costs every submission and the RAM has a source of dma-bufs (RADV's, the Iris Xe's: the
//! runtime's listed plan), the RAM's chunks are imported for copies alone (below): an object lies
//! in a chunk a copy imported, and over any other takes the import of its own view, so that no
//! object pins a chunk's pages.
//!
//! - **By pointer, or as dma-bufs on Linux.** The runtime imports the RAM's chunks by their
//!   pointers (`VK_EXT_external_memory_host`). On Linux the RAM also has sources of dma-bufs of
//!   its pages where `/dev/udmabuf` opens, which the runtime takes where they cost a submission
//!   less or the pointer is refused (`super::linux`'s `host_ram`, which says when); each OS adds
//!   its own ([`runtime_ram`]), and Windows has none.
//! - **What is handed.** Each lease's ranges of the guest's RAM (each OS's `lease_guest_memory`):
//!   the host's own shared mappings of the files (on Windows, the sections) it named, writable,
//!   mapped while the lease lives. Nothing else: read-only memory and the library's own memory
//!   answer no runs, and keep the imports they have.
//! - **To whom.** Every device the platform adopts, on Linux once its imports of views were
//!   learned (`super::linux`'s `import`, `learn_view_imports`), on Windows as it is adopted
//!   (`crate::gpu::device::adopt`), there being nothing to learn there ([`hand_ram_to`]); and every
//!   such device as each lease is made ([`lease`]): a device made after a lease gets its RAM then,
//!   and a lease made after the device gets it at once. On Linux, one that takes views through
//!   dma-bufs takes the RAM where udmabuf opens alone, as its views do, so its copies by runs
//!   always have the source of lists.
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

use crate::core::sync::Mutex;
use metal2vulkan_runtime as rt;

use crate::os::files;

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

/// Hands `device` every lease's RAM alive now, and every one made after: on Linux once its imports
/// of views were learned (`super::linux`'s `import`, `learn_view_imports`), on Windows as it is
/// adopted (module docs, "To whom").
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
    // The OS's sources of the RAM's pages, where it has any (module docs).
    Some(super::host_os::with_ram_sources(ram))
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

/// The runtime's copy of `runs` of `ram`, each a byte of `destination` and a run of host
/// addresses whose bytes land from that byte on, on `encoder`: one call however many runs, as
/// [`copy_to_host_ram`] copies the other way; one it refuses encodes nothing.
pub(crate) fn copy_runs_from_host_ram(
    encoder: &mut rt::BlitCommandEncoder,
    ram: &rt::HostRam,
    runs: &[(u64, Range<u64>)],
    destination: &rt::Buffer,
) -> Result<(), RunsCopyRefused> {
    (encoder.copy_runs_from_host_ram(ram, runs, destination)).map_err(refused)
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
/// no-copy object's import window (each OS's `import`, `host_memory`): host addresses in the
/// leases' ranges, in the bytes' order; `None` where a byte is no writable page of the RAM
/// (`super::files`'s `guest_ram_runs`), and the object keeps today's import.
pub(crate) fn runs(start: u64, length: u64) -> Option<Vec<Range<u64>>> {
    let runs = files::guest_ram_runs(start, length)?;
    Some(runs.into_iter().map(|(at, n)| at..at + n).collect())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::core::host::HostPtr;
    use crate::core::num::{ToU64 as _, ToUsize as _};
    use crate::core::platform::HostRange;
    use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
    use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::resource::Buffer as _;
    use crate::core::platform::types::NsRange;
    use crate::core::raise::OnFault;

    use crate::gpu::device::tests::{dev, settle_device_checks};
    use crate::os::{GuestRamFile, HostFile, PAGE, lease_guest_memory};

    const MIB: u64 = 1 << 20;

    /// A file of `length` bytes of zeros for a test's RAM, and the host's own shared, writable
    /// mapping of all of it, as a VMM maps its RAM: a memfd's, which the test unmaps after the
    /// lease and every buffer over it ([`unmap_host_ram`]).
    #[cfg(target_os = "linux")]
    fn host_ram(length: u64) -> (Arc<HostFile>, *mut u8) {
        use std::os::fd::AsRawFd as _;
        let fd = super::super::host_os::memfd(length).unwrap();
        // SAFETY: a fresh shared mapping of the file.
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
        (fd, start.cast())
    }

    /// A section of `length` bytes of zeros for a test's RAM, and the host's own writable view of
    /// all of it (`MapViewOfFile`), as QEMU maps its RAM on Windows, which the test unmaps after
    /// the lease and every buffer over it ([`unmap_host_ram`]).
    #[cfg(windows)]
    fn host_ram(length: u64) -> (Arc<HostFile>, *mut u8) {
        use std::os::windows::io::AsRawHandle as _;
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let file = super::super::host_os::section(length).unwrap();
        // SAFETY: a whole view of a section the test holds open.
        let view = unsafe { MapViewOfFile(file.as_raw_handle(), FILE_MAP_WRITE, 0, 0, 0) };
        assert!(!view.Value.is_null());
        (file, view.Value.cast())
    }

    /// Unmaps [`host_ram`]'s mapping of `length` bytes at `start` once the runtime freed its
    /// imports of the RAM's chunks (the RAM's keep-alive): on Linux through the platform's VM,
    /// which leaves pages a buffer's import still holds mapped until the runtime has freed it; on
    /// Windows the view, which the VM did not map.
    fn unmap_host_ram(start: *mut u8, length: u64) {
        crate::remap::free_unimported(start, length, || {
            #[cfg(target_os = "linux")]
            {
                use crate::core::platform::os::vm::HostVm as _;
                let at = HostPtr::from_ptr(start);
                assert!(
                    crate::remap::VulkanVm::new()
                        .deallocate(at, length)
                        .is_success()
                );
            }
            #[cfg(windows)]
            {
                use windows_sys::Win32::System::Memory::{
                    MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile,
                };
                let view = MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: start.cast(),
                };
                // SAFETY: the section's view, which nothing uses now, unmapped once.
                assert_ne!(unsafe { UnmapViewOfFile(view) }, 0);
            }
        });
    }

    /// Where the device imports the guest's RAM ahead of its objects (NVIDIA's), a no-copy buffer
    /// over one run of it is bound in the RAM's chunk, which the device imports once, a second
    /// buffer in the chunk importing nothing more, and the GPU's fill through each reaches the
    /// RAM. Where its plan is listed, no copy imported the chunk here, so its buffers import their
    /// pages as before ([`a_buffer_over_a_chunk_a_copy_imported_lies_in_it`] has one). Either way
    /// the bytes are the GPU's. The flag is the process's, which nextest runs this test alone in.
    ///
    /// The runtime imports the chunk as the first object over it asks, where the RAM has a source
    /// of dma-bufs (on Linux, where `/dev/udmabuf` opens), or on its own thread ahead of every
    /// object, where it has none. An object made before the chunk's import ends imports the chunk
    /// or waits for the import under way, and lies in it: no object is bound in an import of its
    /// own run (metal2vulkan's `host_ram.rs`, "An object over a chunk not imported yet"). The test
    /// makes the first buffer, waits for the chunk's import, and makes the second; both lie in the
    /// chunk, which is imported once. Where the plan is listed instead (the RAM has a source of
    /// dma-bufs and each held import costs every submission, the device's submission check
    /// measured: `releases_idle_pointer_imports`), a chunk is imported only for a copy over it, and
    /// an object over a chunk not imported takes its own import and asks for none: neither buffer
    /// lies in the RAM, and no chunk is imported.
    #[test]
    fn a_buffer_over_one_run_of_the_ram_is_bound_in_its_chunk() {
        let device = dev();
        // The device's own checks end first: the chunk's import and the counts wait for none.
        settle_device_checks(&device);
        let length = 8 * MIB;
        let (fd, start) = host_ram(length);
        let at = start.addr().to_u64();
        let counts = || {
            let c = device.raw.work_counts();
            (c.host_ram_chunks_imported, c.chunk_bound_buffers_made)
        };
        let before = counts();
        let lease = lease_guest_memory([GuestRamFile {
            host: at,
            guest: 0,
            length,
            file: fd,
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
        // The plan is fixed as the RAM's use begins, after the device's checks answered.
        let listed =
            super::super::host_os::ram_has_sources() && device.raw.releases_idle_pointer_imports();
        let a = over(MIB, 2 * PAGE);
        let after_a = counts();
        if !listed {
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
        let first = after_a.1 - before.1;
        let second = (now.0 - imported.0, now.1 - imported.1);
        match listed {
            false => {
                assert_eq!(now.0 - before.0, 1, "the chunk imported once");
                assert_eq!(first, 1, "the first bound in the chunk");
                assert_eq!(second, (0, 1), "the second bound in the chunk");
            }
            true => {
                let made = (now.0 - before.0, first, second.1);
                assert_eq!(made, (0, 0, 0), "no chunk imported, no buffer in the RAM");
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
        // The device lives on past the lease.
        unmap_host_ram(start, length);
    }
    /// A no-copy buffer over a chunk of the RAM a copy imported lies in that chunk, on every device
    /// that uses the RAM: each OS's `import` gives each import the RAM's runs, so the buffer takes
    /// no import of its own, and the GPU's fill through it reaches the RAM. A copy whose run is the
    /// whole chunk (the RAM's 8 MiB here) imports it where the device's plan is listed (RADV's,
    /// the Iris Xe's), whose objects never ask for a chunk; elsewhere the runtime's thread imports
    /// it ahead. A device that does not use the RAM refuses the copy and takes no buffer
    /// into a chunk. The flag is the process's, which nextest runs this test alone in.
    #[test]
    fn a_buffer_over_a_chunk_a_copy_imported_lies_in_it() {
        let device = dev();
        // The device's own checks end first: the chunk's import and the counts wait for none.
        settle_device_checks(&device);
        let length = 8 * MIB;
        let (fd, start) = host_ram(length);
        let at = start.addr().to_u64();
        let counts = || {
            let c = device.raw.work_counts();
            (c.host_ram_chunks_imported, c.chunk_bound_buffers_made)
        };
        let before = counts();
        let lease = lease_guest_memory([GuestRamFile {
            host: at,
            guest: 0,
            length,
            file: fd,
            offset: 0,
        }]);
        let q = device.new_command_queue().unwrap();
        let whole = (device.new_buffer_of_length(OnFault::Die, length, 0))
            .unwrap()
            .unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        let gathered = e
            .copy_from_guest_ram(
                OnFault::Die,
                std::slice::from_ref(&(at..at + length)),
                &whole,
                0,
            )
            .unwrap();
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        assert_eq!(gathered, super::copies(&device.raw));
        let mut kept = None;
        if gathered {
            // The RAM is one chunk: its import is given 20 s to end.
            let imported = (0..4000).any(|_| {
                let done = counts().0 > before.0;
                if !done {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                done
            });
            assert!(imported, "no chunk of the RAM imported for the copy");
            let imported = counts();
            // SAFETY: a page of the test's RAM, mapped until after the buffer is dropped.
            let bytes =
                unsafe { HostRange::new(HostPtr::from_addr((at + 2 * MIB).to_usize()), PAGE) };
            let over = (device.new_buffer_over_host_memory(OnFault::Die, bytes, 0, None))
                .unwrap()
                .unwrap();
            let now = counts();
            assert_eq!(
                (now.0 - imported.0, now.1 - imported.1),
                (0, 1),
                "the buffer bound in the chunk, importing nothing more"
            );
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            let fill = BlitCmd::FillBuffer {
                buffer: &over,
                range: NsRange::new(0x10, PAGE - 0x20),
                value: 0x3c,
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
            let bytes = [
                read(2 * MIB + 0xf),
                read(2 * MIB + 0x10),
                read(2 * MIB + PAGE - 0x11),
                read(2 * MIB + PAGE - 0x10),
            ];
            assert_eq!(bytes, [0, 0x3c, 0x3c, 0]);
            kept = Some((cb, e, over));
        } else {
            assert_eq!(counts(), before, "no chunk imported, no buffer in one");
        }
        drop((kept, cb, e, whole, q, lease));
        // The device lives on past the lease.
        unmap_host_ram(start, length);
    }
    /// A device that copies the RAM by its runs ([`super::copies`]) gathers runs of it into a
    /// buffer, and scatters a buffer back into them, byte for byte in the runs' order, scattered
    /// and out of order as a surface's pages lie; one that does not refuses both, encoding
    /// nothing, the RAM as it was. The flag is the process's, which nextest runs this test alone in.
    #[test]
    fn a_copy_by_runs_gathers_and_scatters_the_rams_pages() {
        let device = dev();
        let length = 8 * MIB;
        let (fd, start) = host_ram(length);
        let at = start.addr().to_u64();
        let lease = lease_guest_memory([GuestRamFile {
            host: at,
            guest: 0,
            length,
            file: fd,
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
        // The device lives on past the lease.
        unmap_host_ram(start, length);
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
        let (fd, start) = host_ram(length);
        let at = start.addr().to_u64();
        let lease = lease_guest_memory([GuestRamFile {
            host: at,
            guest: 0,
            length,
            file: fd,
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
        // The device lives on past the lease.
        unmap_host_ram(start, length);
    }
}
