//! The device: [`RvgDeviceConfig`], [`RvgHostCallbacks`], and the `rvg_device_*` functions,
//! suspend and resume among them.
//!
//! The host's functions become the Rust API's closures, each calling its function
//! synchronously on the thread core calls it on, with the arguments core passes, so the order
//! of VM-memory operations is core's (docs/api.md section 5). Each call into the host runs under
//! `ffi::host_call`: a host function that raises ends the process at the call.

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::num::NonZeroU64;
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;
use std::time::Duration;

use reims_vgpu::{
    AddTraceRange, CreateTask, DestroyTask, Device, DeviceConfig, Error, HostCallbacks, HostPtr,
    HostTask, HostTraceRange, MapMemory, MemoryFile, MemoryMap, PhysicalMemoryRange,
    RaiseInterrupt, ReadMemory, RemoveTraceRange, UnmapMemory,
};
use reims_vgpu::{DirtyHost, DirtySource, KvmSlot};

use crate::ffi;
use crate::handle;
use crate::sized::{self, HostStruct};
use crate::status::{RVG_ERR_INVALID_ARGUMENT, RVG_OK, RvgStatus, status};
use crate::table::{RvgDestroyFn, Table, WithTable};
use crate::trace::RvgTraceRangeHandler;
use crate::{Plat, null_handle};

/// Why a resume refused its state: one of RVG_RESUME_ERROR_*.
pub type RvgResumeError = u32;

// The constants below are RvgResumeError values, typed `u32` so that cbindgen writes them with the
// header's other constants, ahead of its types.

/// No reason stored.
pub const RVG_RESUME_ERROR_NONE: u32 = 4294967295;

/// An internal fault.
pub const RVG_RESUME_ERROR_INTERNAL_FAULT: u32 = 0;

/// The state's version is not one the device reads.
pub const RVG_RESUME_ERROR_INVALID_SUSPEND_STATE_VERSION: u32 = 1;

/// The state is malformed, or absent.
pub const RVG_RESUME_ERROR_INVALID_CONTENT: u32 = 2;

/// The guest's binary version in the state is not one the device supports.
pub const RVG_RESUME_ERROR_INVALID_GUEST_VERSION: u32 = 3;

/// The state was taken on a device this one cannot stand in for.
pub const RVG_RESUME_ERROR_INCOMPATIBLE_DEVICE: u32 = 4;

/// The state's display port count differs from the device's.
pub const RVG_RESUME_ERROR_INVALID_DISPLAY_PORT_COUNT: u32 = 5;

const _: () = assert!(RVG_RESUME_ERROR_NONE == u32::MAX);

/// One range of guest physical memory.
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RvgPhysicalMemoryRange {
    pub physical_address: u64,
    pub physical_length: u64,
}

const _: () = assert!(size_of::<RvgPhysicalMemoryRange>() == size_of::<PhysicalMemoryRange>());
const _: () = assert!(align_of::<RvgPhysicalMemoryRange>() == align_of::<PhysicalMemoryRange>());

impl From<PhysicalMemoryRange> for RvgPhysicalMemoryRange {
    fn from(r: PhysicalMemoryRange) -> Self {
        RvgPhysicalMemoryRange {
            physical_address: r.physical_address,
            physical_length: r.physical_length,
        }
    }
}

impl From<RvgPhysicalMemoryRange> for PhysicalMemoryRange {
    fn from(r: RvgPhysicalMemoryRange) -> Self {
        PhysicalMemoryRange {
            physical_address: r.physical_address,
            physical_length: r.physical_length,
        }
    }
}

/// `create_task(ctx, vm_size, &base_address)`: reserve `vm_size` bytes of host address space
/// for a new GPU task. Returns the host's task, or NULL for a failure, and writes the base
/// address of the reservation through `base_address`.
///
/// Thread: a library thread, or the host's thread inside a call marked "may call back".
/// Calls for different tasks may run at once.
/// May call: rvg_status_message only.
pub type RvgCreateTaskFn = Option<
    unsafe extern "C-unwind" fn(
        ctx: *mut c_void,
        vm_size: u64,
        base_address: *mut *mut c_void,
    ) -> *mut c_void,
>;

/// `destroy_task(ctx, task)`: the task's last call. `task` is NULL after a failed root map.
///
/// Thread, and what it may call: as RvgCreateTaskFn. Every call for `task` has returned.
pub type RvgDestroyTaskFn =
    Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, task: *mut c_void)>;

/// `map_memory(ctx, task, range_count, virtual_offset, read_only, ranges)`: map `range_count`
/// guest ranges, one after another, at `virtual_offset` in the task's reservation. `ranges` is
/// borrowed for the call. Returns false for a failure.
///
/// Thread, and what it may call: as RvgCreateTaskFn. One task's map and unmap calls come in the
/// order the device makes them.
pub type RvgMapMemoryFn = Option<
    unsafe extern "C-unwind" fn(
        ctx: *mut c_void,
        task: *mut c_void,
        range_count: u32,
        virtual_offset: u64,
        read_only: bool,
        ranges: *const RvgPhysicalMemoryRange,
    ) -> bool,
>;

/// `unmap_memory(ctx, task, virtual_offset, length)`. Returns false for a failure.
///
/// Thread, and what it may call: as RvgMapMemoryFn.
pub type RvgUnmapMemoryFn = Option<
    unsafe extern "C-unwind" fn(
        ctx: *mut c_void,
        task: *mut c_void,
        virtual_offset: u64,
        length: u64,
    ) -> bool,
>;

/// `read_memory(ctx, physical_address, length, dst)`: copy guest memory into `dst`. Returns false
/// for a failure. The device reads guest physical memory this way, at most one guest page (16 KiB
/// on arm64, 4 KiB on x86-64) a call: a task's page tables, the task memory it copies through
/// them, and each task's root node at a resume. With a memory map it reads through the map, and a
/// config that gives read_memory is refused. Without a map and without read_memory, it maps the
/// page into its root task (map_memory, then unmap_memory) and copies from there; while the guest
/// has no root page, the device has no root task, and the read fails.
///
/// Thread, and what it may call: as RvgCreateTaskFn. It may wait for another host thread, unless
/// that thread is itself waiting in a call on this device that may call back.
pub type RvgReadMemoryFn = Option<
    unsafe extern "C-unwind" fn(
        ctx: *mut c_void,
        physical_address: u64,
        length: u64,
        dst: *mut c_void,
    ) -> bool,
>;

/// `raise_interrupt(ctx, vector)`: raise the guest interrupt `vector`.
///
/// Thread: a library thread (a FIFO's or the device's reporting thread), or the host's thread
/// inside a call marked "may call back". The device's signal lock is held: one device's raises
/// never run at once. It must not wait for a host thread that is in a call on this device.
/// May call: rvg_status_message only. Without it, the device's first interrupt ends the
/// process.
pub type RvgRaiseInterruptFn = Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, vector: u32)>;

/// `add_trace_range(ctx, range, handler)`: watch guest writes to `range` (the EFI framebuffer)
/// and call `handler` with the dirtied part. Returns the host's trace range, or NULL for none:
/// the EFI display then presents every 100 ms instead of after each write. `handler` is borrowed
/// for the call: keep it with rvg_trace_range_handler_retain. The handler must not be invoked
/// from inside this call.
///
/// Thread: the host's thread inside a call marked "may call back", or a library thread.
/// May call: rvg_trace_range_handler_retain, rvg_status_message.
pub type RvgAddTraceRangeFn = Option<
    unsafe extern "C-unwind" fn(
        ctx: *mut c_void,
        range: RvgPhysicalMemoryRange,
        handler: *mut RvgTraceRangeHandler,
    ) -> *mut c_void,
>;

/// `remove_trace_range(ctx, trace_range)`: stop watching `trace_range`, a range add_trace_range
/// returned. It comes once for each non-NULL range, when the EFI display stops presenting from
/// it: at the guest's next write to an EFI framebuffer register, when EFI goes off, when port 0
/// has no display to present to, or at the device's end. A NULL answer gets no call.
///
/// Thread: as RvgAddTraceRangeFn.
/// May call: rvg_trace_range_handler_release, rvg_status_message.
pub type RvgRemoveTraceRangeFn =
    Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, trace_range: *mut c_void)>;

/// The device's callbacks. Without a memory map, the device calls the task and memory
/// callbacks; with one, it calls none of them and a config that gives any is refused. Without
/// `add_trace_range` the device calls neither trace function.
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RvgHostCallbacks {
    /// sizeof(RvgHostCallbacks) as the host was compiled.
    pub struct_size: u32,
    pub ctx: *mut c_void,
    /// Runs once after the device's end, when no callback of the table runs again.
    pub destroy: RvgDestroyFn,
    pub create_task: RvgCreateTaskFn,
    pub destroy_task: RvgDestroyTaskFn,
    pub map_memory: RvgMapMemoryFn,
    pub unmap_memory: RvgUnmapMemoryFn,
    pub read_memory: RvgReadMemoryFn,
    pub raise_interrupt: RvgRaiseInterruptFn,
    pub add_trace_range: RvgAddTraceRangeFn,
    pub remove_trace_range: RvgRemoveTraceRangeFn,
}

// SAFETY: repr(C), a u32 first; pointers and optional function pointers.
unsafe impl HostStruct for RvgHostCallbacks {}

/// One range of guest physical memory and the host memory behind it.
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RvgMemoryRange {
    pub physical_address: u64,
    pub physical_length: u64,
    /// `physical_length` bytes, mapped (`mach_vm_allocate`d or equivalent) for as long as any
    /// device or IOSurface device made over the map lives.
    pub host_address: *mut c_void,
}

/// The guest's RAM, as the host maps it. The ranges are copied at the create call, in order.
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RvgMemoryMap {
    pub ranges: *const RvgMemoryRange,
    pub range_count: usize,
}

/// The file behind one range of a memory map: the range's host memory is a shared mapping
/// (MAP_SHARED) of `fd` from byte `offset`. Plain data.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RvgMemoryRangeFile {
    /// An open file descriptor, which the library duplicates during the create (the host may
    /// close its own afterwards), or -1 for a range with no file.
    pub fd: i32,
    /// 0; other values are refused.
    pub flags: u32,
    /// The file offset of the range's first byte, a multiple of the host's page size.
    pub offset: u64,
}

const _: () = assert!(size_of::<RvgMemoryRangeFile>() == 16);

/// `start(ctx)`: start logging the guest's RAM's dirty pages, with KVM's manual dirty-log
/// protection and every page logged as written at its start (KVM_DIRTY_LOG_INITIALLY_SET).
/// Called once, at the first harvest. It must not block: the log may start later, and until it
/// does `for_each_slot` lists none of the RAM's slots. Without it, the log is taken as started.
pub type RvgDirtyStartFn = Option<unsafe extern "C-unwind" fn(ctx: *mut c_void)>;

/// `flush_vcpus(ctx, timeout_ns)`: wait until each vCPU that was inside KVM_RUN has returned from
/// it once, so that KVM has drained the CPU's buffer of the vCPU's writes (Intel's
/// page-modification log) into the slots' logs, `timeout_ns` at most; false at the timeout.
/// Called only where `pml` of kvm_intel or kvm_amd is on. Without it, every harvest there takes
/// every watched page as written.
pub type RvgDirtyFlushVcpusFn =
    Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, timeout_ns: u64) -> bool>;

/// `for_each_slot(ctx, slot_ctx)`: calls `rvg_dirty_slot(slot_ctx, ...)` for each memory slot of
/// the VM that logs dirty pages, holding the host's lock on its slots, so none changes before it
/// returns; false, calling none, where the host reads and clears the log itself (QEMU's global
/// dirty tracking).
pub type RvgDirtyForEachSlotFn =
    Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, slot_ctx: *mut c_void) -> bool>;

/// `take_host_writes(ctx, host_offset, length, words)`: ORs into `words`, one bit a 4 KiB page
/// from `host_offset`, the pages of `length` bytes the host itself wrote since its last take (its
/// DMA into the guest's RAM), and clears them in its record, not in KVM's log. Called only inside
/// `for_each_slot`'s `slot`, with `host_offset` and `length` multiples of 256 KiB. Without it,
/// the host writes none of the guest's RAM.
pub type RvgDirtyTakeHostWritesFn = Option<
    unsafe extern "C-unwind" fn(ctx: *mut c_void, host_offset: u64, length: u64, words: *mut u64),
>;

/// The host's dirty source (RvgDeviceConfig.dirty_source): QEMU under KVM on Linux, which tells
/// the device which of the guest's pages the guest wrote, so that the library's Vulkan runtime
/// copies again, into a discrete GPU's memory or an iGPU's that cannot reference the guest's,
/// only those. The device reads KVM's dirty log itself, on its thread `reims-vgpu.dirty`, from
/// its first harvest on, which calls the functions below, one at a time; none may wait for the
/// library, or take a lock the host holds while it calls the library (QEMU's BQL). Nothing else
/// may clear the log of the guest's RAM while the device lives (QEMU's migration blocker). A
/// source with `kvm_vm_fd` below 0, or without `for_each_slot`, is none. `destroy` runs once,
/// after its last call: at the device's end, or before the create returns where the device takes
/// none (no memory map, or a refusal).
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RvgDirtySource {
    /// sizeof(RvgDirtySource) as the host was compiled.
    pub struct_size: u32,
    /// The KVM VM's file descriptor, open while the device lives.
    pub kvm_vm_fd: i32,
    pub ctx: *mut c_void,
    pub destroy: RvgDestroyFn,
    pub start: RvgDirtyStartFn,
    pub flush_vcpus: RvgDirtyFlushVcpusFn,
    pub for_each_slot: RvgDirtyForEachSlotFn,
    pub take_host_writes: RvgDirtyTakeHostWritesFn,
}

// SAFETY: repr(C), a u32 first; an i32, a pointer and optional function pointers.
unsafe impl HostStruct for RvgDirtySource {}

/// The device's config.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RvgDeviceConfig {
    /// sizeof(RvgDeviceConfig) as the host was compiled.
    pub struct_size: u32,
    /// 1 to RVG_MAX_DISPLAY_PORTS; 0 is refused.
    pub display_port_count: u32,
    /// The Metal device's registry ID; 0 is the system's default Metal device.
    pub metal_registry_id: u64,
    /// The feature level: 0 and 4 offer every feature; 1, 2 and 3 cap the binary version the
    /// guest may choose at 62, 113 and 209, and trim the device info it reads. A level above 4 is
    /// accepted here; the guest's first binary version write then ends the process, as do
    /// rvg_device_reset and rvg_device_will_resume once it applies the state's version.
    pub feature_level: u64,
    /// The display port connection types the guest reads (0: none set); below binary version 245
    /// it reads their low 2 bits. A resume of a state from binary version 238 or later refuses one
    /// taken with other types (RVG_RESUME_ERROR_INTERNAL_FAULT).
    pub display_port_connection_types: u32,
    /// Map guest IOSurfaces through the process's latest RvgIOSurfaceDevice (vmapple).
    pub use_iosurface_mapper: bool,
    /// Run each isolated guest task with the task service's semantics (needs a memory map).
    pub process_isolation: bool,
    /// Withhold argument buffers from the guest. They are offered only with process isolation, on
    /// a Metal device that can detach backing and is not itself paravirtual, and, with
    /// `process_isolation_follows_guest`, only while the IOSurface device has a descriptor table.
    pub disable_argument_buffers: bool,
    /// The guest's RAM, or NULL to reach it through the host callbacks.
    pub memory_map: *const RvgMemoryMap,
    /// NULL is a table with every callback absent.
    pub callbacks: *const RvgHostCallbacks,
    /// With `memory_map`: `memory_map->range_count` entries, the i-th naming the file behind
    /// the map's i-th range in the order the host listed them; NULL for none. On Linux the
    /// library builds its views of guest pages from these files (a range without one gets no
    /// view, and a guest use that needs one fails as a failed remap does, as does one over a
    /// file of huge pages, hugetlbfs, which the checks below accept and no view of a page of the
    /// host's size can map: give such a range fd -1); on macOS they are checked and not used. Files without a map are refused, and so is an entry whose `fd` is
    /// below -1, whose `flags` are not 0, or, for an `fd` of 0 or more, whose file `fstat`
    /// cannot read, holds fewer than `offset` + the range's length bytes, or whose `offset` is
    /// not a multiple of the host's page size: RVG_ERR_INVALID_ARGUMENT, in that order, before
    /// any file is duplicated.
    pub memory_files: *const RvgMemoryRangeFile,
    /// With `process_isolation`: isolate a guest task other than task 0 only when, as the guest
    /// defines it, the RvgIOSurfaceDevice the device maps through (`use_iosurface_mapper`) has a
    /// descriptor table base other than 0; otherwise the task runs as without isolation. The
    /// guest's mapper driver writes the base at its start, before Apple's arm64 GPU driver makes
    /// a task. Apple's x86-64 driver never touches the IOSurface device, and references
    /// IOSurfaces through task 0's backings, which an isolated task does not serve. Device info's
    /// argument-buffer answer follows the same rule, read when the guest asks, since only
    /// isolated tasks serve argument buffers. Without `process_isolation` it changes nothing.
    pub process_isolation_follows_guest: bool,
    /// With `memory_map`: the host's dirty source, or NULL for none.
    pub dirty_source: *const RvgDirtySource,
}

// SAFETY: repr(C), a u32 first; integers, bools and pointers.
unsafe impl HostStruct for RvgDeviceConfig {
    /// The first version ends at `callbacks`: a host built against it passes 48, and has no
    /// `memory_files`.
    const V1_SIZE: usize = 48;
}

const _: () =
    assert!(std::mem::offset_of!(RvgDeviceConfig, memory_files) == RvgDeviceConfig::V1_SIZE);

/// The device: one paravirtualized GPU and its display ports.
pub struct RvgDevice {
    pub(crate) device: WithTable<Device<Plat>>,
}

/// Bytes the library allocated; free them with rvg_bytes_free.
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
#[repr(C)]
#[derive(Debug)]
pub struct RvgBytes {
    pub data: *mut u8,
    pub length: usize,
}

/// The Rust API's map of a host's `RvgMemoryMap`, with the files behind its ranges
/// (`RvgDeviceConfig.memory_files`, NULL for none). `Err(RVG_ERR_INVALID_ARGUMENT)`, in this
/// order: ranges that are NULL but counted; files without a map; a file entry
/// [`file_refusal`] refuses, the first range's first; a file that cannot be duplicated. Each
/// file is duplicated only once every entry has passed, so a refusal before then touches no
/// file. A refusal of the files logs a line saying which.
///
/// # Safety
/// `map` is NULL or a live map whose `ranges` hold `range_count` ranges; `files` is NULL or
/// holds `range_count` entries.
pub(crate) unsafe fn memory_map(
    map: *const RvgMemoryMap,
    files: *const RvgMemoryRangeFile,
) -> Result<Option<MemoryMap>, RvgStatus> {
    // SAFETY: the caller's contract.
    let Some(map) = (unsafe { map.as_ref() }) else {
        if !files.is_null() {
            log::warn!("device not created: memory files without a memory map");
            return Err(RVG_ERR_INVALID_ARGUMENT);
        }
        return Ok(None);
    };
    let ranges: &[RvgMemoryRange] = if map.range_count == 0 {
        &[]
    } else if map.ranges.is_null() {
        return Err(RVG_ERR_INVALID_ARGUMENT);
    } else {
        // SAFETY: the caller's contract.
        unsafe { std::slice::from_raw_parts(map.ranges, map.range_count) }
    };
    let files: &[RvgMemoryRangeFile] = if files.is_null() || ranges.is_empty() {
        &[]
    } else {
        // SAFETY: the caller's contract.
        unsafe { std::slice::from_raw_parts(files, ranges.len()) }
    };
    let page = host_page_size();
    for (i, (r, f)) in ranges.iter().zip(files).enumerate() {
        if let Some(why) = file_refusal(r, f, page) {
            log::warn!("device not created: the file of memory range {i} {why}");
            return Err(RVG_ERR_INVALID_ARGUMENT);
        }
    }
    let mut m = MemoryMap::new();
    for (i, r) in ranges.iter().enumerate() {
        let file = match files.get(i) {
            Some(f) if f.fd >= 0 => match duplicate(f.fd) {
                Ok(fd) => Some(MemoryFile {
                    fd: Arc::new(fd),
                    offset: f.offset,
                }),
                Err(e) => {
                    log::warn!(
                        "device not created: the file of memory range {i}, fd {}, was not \
                         duplicated: {e}",
                        f.fd
                    );
                    return Err(RVG_ERR_INVALID_ARGUMENT);
                }
            },
            _ => None,
        };
        // SAFETY: the header's contract on `host_address`: mapped while any device made over
        // the map lives, and, with a file, a shared mapping of it from `offset`.
        unsafe {
            m.push_with_file(
                r.physical_address,
                r.physical_length,
                r.host_address.cast(),
                file,
            );
        }
    }
    Ok(Some(m))
}

/// Why the file entry `f` of range `r` is refused, as the end of a log line, or `None`: an `fd`
/// below -1, `flags` other than 0, and for an `fd` of 0 or more, a file `fstat` cannot read, one
/// holding fewer than `offset + physical_length` bytes, or an `offset` that is not a multiple of
/// `page`, in that order.
fn file_refusal(r: &RvgMemoryRange, f: &RvgMemoryRangeFile, page: NonZeroU64) -> Option<String> {
    if f.fd < -1 {
        return Some(format!("has fd {}, below -1", f.fd));
    }
    if f.flags != 0 {
        return Some(format!("has flags {:#x}, not 0", f.flags));
    }
    if f.fd == -1 {
        return None;
    }
    let size = match file_size(f.fd) {
        Ok(size) => size,
        Err(e) => return Some(format!("has fd {}, which fstat cannot read: {e}", f.fd)),
    };
    if f.offset
        .checked_add(r.physical_length)
        .is_none_or(|end| size < end)
    {
        return Some(format!(
            "holds {size:#x} bytes, fewer than its offset {:#x} and the range's {:#x}",
            f.offset, r.physical_length
        ));
    }
    if f.offset % page != 0 {
        return Some(format!(
            "has offset {:#x}, not a multiple of the page size {page:#x}",
            f.offset
        ));
    }
    None
}

/// The host's page size (`sysconf(_SC_PAGESIZE)`, which does not fail): 16 KiB on arm64 macOS,
/// 4 KiB on x86-64.
fn host_page_size() -> NonZeroU64 {
    // SAFETY: a query of a constant; it reads and writes no memory of the caller's.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(page)
        .ok()
        .and_then(NonZeroU64::new)
        .unwrap_or(NonZeroU64::MIN)
}

/// The size of the file `fd` names (`fstat`).
fn file_size(fd: i32) -> std::io::Result<u64> {
    let mut st = MaybeUninit::<libc::stat>::zeroed();
    // SAFETY: `st` is a writable `struct stat`; an `fd` that names no open file is an error the
    // call returns.
    if unsafe { libc::fstat(fd, st.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `fstat` succeeded, so it filled `st`; it was zeroed before.
    let st = unsafe { st.assume_init() };
    Ok(u64::try_from(st.st_size).unwrap_or(0))
}

/// A duplicate of `fd`, close-on-exec (`fcntl(F_DUPFD_CLOEXEC)`), which the map holds.
fn duplicate(fd: i32) -> std::io::Result<OwnedFd> {
    // SAFETY: `fcntl` reads and writes no memory of the caller's; an `fd` that names no open
    // file is an error the call returns.
    let new = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    if new < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `new` is a descriptor this call opened, owned by nothing else.
    Ok(unsafe { OwnedFd::from_raw_fd(new) })
}

/// The host's task pointer as the Rust API's value, and back.
fn task_ptr(t: HostTask) -> *mut c_void {
    std::ptr::with_exposed_provenance_mut(t.addr())
}

// `HostPtr` is one pointer-sized word, so `create_task` can write the caller's own.
const _: () = assert!(
    size_of::<HostPtr>() == size_of::<*mut c_void>()
        && align_of::<HostPtr>() == align_of::<*mut c_void>()
);

/// The Rust API's closures over a host table: each holds the table, and calls its function
/// with the table's `ctx`.
/// The host's dirty source as the Rust API's, its functions called under `ffi::host_call`; `None`
/// for a source that is none (`RvgDirtySource`).
fn dirty_source(t: &Arc<Table>, d: &RvgDirtySource) -> Option<DirtySource> {
    if d.kvm_vm_fd < 0 || d.for_each_slot.is_none() {
        return None;
    }
    Some(DirtySource {
        kvm_vm_fd: d.kvm_vm_fd,
        host: Arc::new(CDirtyHost {
            table: t.clone(),
            source: *d,
        }),
    })
}

/// A host's `RvgDirtySource`, and its table.
struct CDirtyHost {
    table: Arc<Table>,
    source: RvgDirtySource,
}

// SAFETY: the header's contract: the source's functions are called from the library's thread,
// with its `ctx`.
unsafe impl Send for CDirtyHost {}
// SAFETY: as above.
unsafe impl Sync for CDirtyHost {}

/// One memory slot of the host's KVM VM that logs dirty pages, which a dirty source's
/// `for_each_slot` reports with the `slot_ctx` it was given, inside that call
/// (RvgDirtyForEachSlotFn). `slot` is as KVM_GET_DIRTY_LOG and KVM_CLEAR_DIRTY_LOG take it, the
/// address space's ID << 16 | the slot's ID; `length` is a multiple of 4 KiB; `host_offset` is
/// where its first page lies in the host's own record of its writes (RvgDirtyTakeHostWritesFn):
/// QEMU's ram_addr_t. Called anywhere else, it is undefined behaviour.
///
/// # Safety
///
/// `slot_ctx` is the one `for_each_slot` was given, and the call is inside that call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_dirty_slot(
    slot_ctx: *mut c_void,
    slot: u32,
    guest_physical_address: u64,
    length: u64,
    host_offset: u64,
) {
    ffi::export(|| {
        // SAFETY: `for_each_slot` passes back the `slot_ctx` the library gave it, live for its
        // call, which this call is inside.
        let f = unsafe { &mut *slot_ctx.cast::<&mut dyn FnMut(KvmSlot)>() };
        f(KvmSlot {
            slot,
            guest_physical_address,
            length,
            host_offset,
        });
    })
}

impl DirtyHost for CDirtyHost {
    fn start(&self) {
        if let Some(f) = self.source.start {
            let ctx = self.table.ctx();
            // SAFETY: the host's function over its ctx.
            ffi::host_call("start", || unsafe { f(ctx) });
        }
    }

    fn flush_vcpus(&self, timeout: Duration) -> bool {
        let Some(f) = self.source.flush_vcpus else {
            return false;
        };
        let (ctx, ns) = (
            self.table.ctx(),
            u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX),
        );
        // SAFETY: the host's function over its ctx.
        ffi::host_call("flush_vcpus", || unsafe { f(ctx, ns) })
    }

    fn for_each_slot(&self, slot: &mut dyn FnMut(KvmSlot)) -> bool {
        let Some(f) = self.source.for_each_slot else {
            return false;
        };
        let mut slot = slot;
        let (ctx, slot_ctx) = (self.table.ctx(), (&raw mut slot).cast::<c_void>());
        // SAFETY: the host's function over its ctx; `slot_ctx` is `rvg_dirty_slot`'s, live for
        // the call.
        ffi::host_call("for_each_slot", || unsafe { f(ctx, slot_ctx) })
    }

    fn take_host_writes(&self, host_offset: u64, length: u64, words: &mut [u64]) {
        if let Some(f) = self.source.take_host_writes {
            let (ctx, out) = (self.table.ctx(), words.as_mut_ptr());
            // SAFETY: the host's function over its ctx; `words` holds a bit for each page of
            // `length`, the caller's contract.
            ffi::host_call("take_host_writes", || unsafe {
                f(ctx, host_offset, length, out)
            });
        }
    }
}

pub(crate) fn host_callbacks(t: &Arc<Table>, c: &RvgHostCallbacks) -> HostCallbacks {
    HostCallbacks {
        create_task: c.create_task.map(|f| {
            let t = t.clone();
            let c: CreateTask = Arc::new(move |vm_size, base: &mut HostPtr| {
                let out = std::ptr::from_mut(base).cast::<*mut c_void>();
                // SAFETY: the host's function over its ctx; `out` is the caller's `HostPtr`, a
                // pointer-sized word (the assert above), live and untouched for the call.
                let task = ffi::host_call("create_task", || unsafe { f(t.ctx(), vm_size, out) });
                HostTask::from_addr(task.expose_provenance())
            });
            c
        }),
        destroy_task: c.destroy_task.map(|f| {
            let t = t.clone();
            let c: DestroyTask = Arc::new(move |task| {
                // SAFETY: the host's function over its ctx.
                ffi::host_call("destroy_task", || unsafe { f(t.ctx(), task_ptr(task)) });
            });
            c
        }),
        map_memory: c.map_memory.map(|f| {
            let t = t.clone();
            let c: MapMemory = Arc::new(
                move |task, count, offset, read_only, ranges: &[PhysicalMemoryRange]| {
                    let ranges = ranges.as_ptr().cast::<RvgPhysicalMemoryRange>();
                    // SAFETY: the host's function over its ctx; `ranges` is live for the call
                    // and has RvgPhysicalMemoryRange's layout (the asserts above).
                    ffi::host_call("map_memory", || unsafe {
                        f(t.ctx(), task_ptr(task), count, offset, read_only, ranges)
                    })
                },
            );
            c
        }),
        unmap_memory: c.unmap_memory.map(|f| {
            let t = t.clone();
            let c: UnmapMemory = Arc::new(move |task, offset, length| {
                // SAFETY: the host's function over its ctx.
                ffi::host_call("unmap_memory", || unsafe {
                    f(t.ctx(), task_ptr(task), offset, length)
                })
            });
            c
        }),
        read_memory: c.read_memory.map(|f| {
            let t = t.clone();
            let c: ReadMemory = Arc::new(move |address, length, dst: HostPtr| {
                // SAFETY: the host's function over its ctx; `dst` holds `length` bytes.
                ffi::host_call("read_memory", || unsafe {
                    f(t.ctx(), address, length, dst.as_ptr())
                })
            });
            c
        }),
        raise_interrupt: c.raise_interrupt.map(|f| {
            let t = t.clone();
            let c: RaiseInterrupt = Arc::new(move |vector| {
                // SAFETY: the host's function over its ctx.
                ffi::host_call("raise_interrupt", || unsafe { f(t.ctx(), vector) });
            });
            c
        }),
        add_trace_range: c.add_trace_range.map(|f| {
            let t = t.clone();
            let c: AddTraceRange = Arc::new(move |range, handler| {
                // A host that keeps the handler retains it; otherwise it goes with this frame.
                let h = handle::Lent::new(RvgTraceRangeHandler::new(handler));
                let p = h.as_ptr();
                // SAFETY: the host's function over its ctx; `p` is live for the call.
                let r =
                    ffi::host_call("add_trace_range", || unsafe { f(t.ctx(), range.into(), p) });
                HostTraceRange::from_addr(r.expose_provenance())
            });
            c
        }),
        remove_trace_range: c.remove_trace_range.map(|f| {
            let t = t.clone();
            let c: RemoveTraceRange = Arc::new(move |r: HostTraceRange| {
                let r = std::ptr::with_exposed_provenance_mut(r.addr());
                // SAFETY: the host's function over its ctx.
                ffi::host_call("remove_trace_range", || unsafe { f(t.ctx(), r) });
            });
            c
        }),
    }
}

/// The device behind a handle, or the process's end for NULL.
///
/// # Safety
/// `device` is NULL or a live device handle.
unsafe fn live<'a>(device: *const RvgDevice, call: &'static str) -> &'a RvgDevice {
    // SAFETY: the caller's contract.
    unsafe { handle::get(device) }.unwrap_or_else(|| null_handle(call))
}

/// A new device. Refusals, checked in this order: RVG_ERR_INVALID_ARGUMENT for the config
/// itself (the files behind its memory map's ranges among it), RVG_ERR_UNSUPPORTED,
/// RVG_ERR_DISPLAY_PORT_COUNT, RVG_ERR_NO_METAL_DEVICE, RVG_ERR_MEMORY_CONFLICT,
/// RVG_ERR_MEMORY_MAP, RVG_ERR_INVALID_ARGUMENT (`add_trace_range` without
/// `remove_trace_range`), RVG_ERR_METAL, RVG_ERR_ISOLATION_WITHOUT_MAP. The first call
/// given a Metal device makes the process's frame queue for it, before the device is built, even
/// when the device is then refused. Calls back: never.
///
/// # Safety
/// `config` is NULL or a live config (header preamble); `device` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_device_create(
    config: *const RvgDeviceConfig,
    device: *mut *mut RvgDevice,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let c = match unsafe { sized::read(config) } {
            Ok(Some(c)) => c,
            Ok(None) => return RVG_ERR_INVALID_ARGUMENT,
            Err(s) => return s,
        };
        // SAFETY: the config's contract: NULL or a live table.
        let callbacks = match unsafe { sized::read_or_zero(c.callbacks) } {
            Ok(t) => t,
            Err(s) => return s,
        };
        // SAFETY: the config's contract: NULL or a live source.
        let dirty = match unsafe { sized::read(c.dirty_source) } {
            Ok(d) => d,
            Err(s) => return s,
        };
        // From here the tables are the library's: a refusal ends them before it returns.
        let table = Table::new(callbacks.ctx, callbacks.destroy);
        let dirty = dirty.map(|d| (Table::new(d.ctx, d.destroy), d));
        let refuse = |s: RvgStatus| {
            table.end();
            if let Some((t, _)) = &dirty {
                t.end();
            }
            s
        };
        if device.is_null() {
            return refuse(RVG_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: the config's contract: NULL or a live map, and NULL or its files.
        let memory_map = match unsafe { memory_map(c.memory_map, c.memory_files) } {
            Ok(m) => m,
            Err(s) => return refuse(s),
        };
        let config = DeviceConfig {
            metal_registry_id: (c.metal_registry_id != 0).then_some(c.metal_registry_id),
            display_port_count: c.display_port_count,
            feature_level: c.feature_level,
            display_port_connection_types: c.display_port_connection_types,
            memory_map,
            use_iosurface_mapper: c.use_iosurface_mapper,
            process_isolation: c.process_isolation,
            process_isolation_follows_guest: c.process_isolation_follows_guest,
            argument_buffers: !c.disable_argument_buffers,
            callbacks: host_callbacks(&table, &callbacks),
            dirty_source: dirty.as_ref().and_then(|(t, d)| dirty_source(t, d)),
        };
        match Device::with_platform(<Plat as reims_vgpu::ApiPlatform>::process(), config) {
            Ok(d) => {
                let h = handle::new(RvgDevice {
                    device: WithTable::new(d, table),
                });
                // SAFETY: checked non-NULL; the caller's contract.
                unsafe { device.write(h) };
                RVG_OK
            }
            Err(e) => refuse(status(e)),
        }
    })
}

/// Drops the host's reference; the last reference (see the preamble) is the device's teardown,
/// which stops the device and may block. May call back.
///
/// # Safety
/// `device` is NULL or a live device handle, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_device_release(device: *mut RvgDevice) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { handle::release(device) }
    })
}

/// A 32-bit read at `offset` in the device's MMIO region. May block and may call back.
///
/// # Safety
/// `device` is a live device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_device_mmio_read(device: *mut RvgDevice, offset: u64) -> u32 {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_device_mmio_read") }
            .device
            .mmio_read(offset)
    })
}

/// A 32-bit write at `offset` in the device's MMIO region. May block: a root FIFO write waits
/// for the FIFO's thread. May call back.
///
/// # Safety
/// `device` is a live device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_device_mmio_write(device: *mut RvgDevice, offset: u64, value: u32) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_device_mmio_write") }
            .device
            .mmio_write(offset, value);
    })
}

/// Resets the device as the guest's bus reset does: stops it as rvg_device_stop does, sets the
/// guest's binary version back to 0 and turns EFI off, and returns once it is quiesced. May
/// block: it waits as rvg_device_stop does, and for the EFI display's present thread. At a
/// `feature_level` above 4 it ends the process. May call back.
///
/// # Safety
/// `device` is a live device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_device_reset(device: *mut RvgDevice) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_device_reset") }.device.reset();
    })
}

/// Pauses the device: each FIFO finishes the commands it started and stops, then the guest's
/// displays pause. May block: it waits for each FIFO's thread to stop and for those commands to
/// complete, and hangs while one never does. Where the guest has given a port a display page, a
/// second pause before rvg_device_unpause ends the process. May call back.
///
/// # Safety
/// `device` is a live device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_device_pause(device: *mut RvgDevice) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_device_pause") }.device.pause();
    })
}

/// Resumes the device after rvg_device_pause: unpauses the guest's displays, starts every FIFO
/// and raises one interrupt. May block: it waits while the FIFOs start on the library's FIFO
/// control thread. Where the guest has given a port a display page, an unpause while the guest's
/// displays are not paused ends the process: rvg_device_pause and rvg_device_finish_suspend
/// pause them, and rvg_device_will_suspend does not. May call back.
///
/// # Safety
/// `device` is a live device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_device_unpause(device: *mut RvgDevice) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_device_unpause") }
            .device
            .unpause();
    })
}

/// Stops the device as a root page of 0 does: its FIFOs, the guest's tasks and the guest's
/// display pages are torn down. May block: it waits for each FIFO's thread to stop and for the
/// commands it started to complete. May call back.
///
/// # Safety
/// `device` is a live device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_device_stop(device: *mut RvgDevice) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_device_stop") }.device.stop();
    })
}

/// Starts a suspend: from here the device raises no interrupt, and drops those not yet raised.
/// It quiesces the FIFOs without waiting for them, and pauses none of the guest's displays. May
/// call back.
///
/// # Safety
/// `device` is a live device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_device_will_suspend(device: *mut RvgDevice) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_device_will_suspend") }
            .device
            .will_suspend();
    })
}

/// Ends a suspend: on RVG_OK, writes the device's state through `state`; free it with
/// rvg_bytes_free. The device always has a state to give, so RVG_ERR_SUSPEND never comes.
/// Without rvg_device_will_suspend before it, it ends the process. May block: it waits for each
/// FIFO's thread to stop and for the commands it started to complete, then flushes and ends the
/// guest's tasks and pauses the guest's displays. May call back.
///
/// # Safety
/// `device` is NULL or a live device handle; `state` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_device_finish_suspend(
    device: *mut RvgDevice,
    state: *mut RvgBytes,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let Some(d) = (unsafe { handle::get(device) }) else {
            return RVG_ERR_INVALID_ARGUMENT;
        };
        if state.is_null() {
            return RVG_ERR_INVALID_ARGUMENT;
        }
        match d.device.finish_suspend() {
            Ok(bytes) => {
                let bytes = Box::into_raw(bytes.into_boxed_slice());
                let out = RvgBytes {
                    data: bytes.cast::<u8>(),
                    length: bytes.len(),
                };
                // SAFETY: checked non-NULL; the caller's contract.
                unsafe { state.write(out) };
                RVG_OK
            }
            Err(e) => status(e),
        }
    })
}

/// Frees what rvg_device_finish_suspend wrote. `{NULL, 0}` does nothing.
///
/// # Safety
/// `bytes` is `{NULL, 0}` or what rvg_device_finish_suspend wrote, not freed before.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_bytes_free(bytes: RvgBytes) {
    ffi::export(|| {
        if bytes.data.is_null() {
            return;
        }
        let slice = std::ptr::slice_from_raw_parts_mut(bytes.data, bytes.length);
        // SAFETY: the caller's contract: the boxed slice rvg_device_finish_suspend made.
        drop(unsafe { Box::from_raw(slice) });
    })
}

/// Offers the device a state rvg_device_finish_suspend gave, before the guest runs again.
/// `state` NULL is no state (the device refuses it with RVG_RESUME_ERROR_INVALID_CONTENT); a
/// non-NULL `state` with `state_length` 0 is an empty state. RVG_OK, or RVG_ERR_RESUME. When
/// `error` is not NULL, the device's reason is written there, or RVG_RESUME_ERROR_NONE where it
/// stored none (a refusal may store none, and RVG_OK stores none). From this call until
/// rvg_device_did_resume, whether or not it accepted the state, a new mapping of guest memory
/// (the guest's root page write, for one) ends the process. May block: a state whose binary
/// version has no EFI display turns EFI off, which waits for the EFI display's present thread.
/// May call back.
///
/// # Safety
/// `device` is NULL or a live device handle; `state` is NULL or holds `state_length` bytes;
/// `error` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_device_will_resume(
    device: *mut RvgDevice,
    state: *const u8,
    state_length: usize,
    error: *mut RvgResumeError,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let Some(d) = (unsafe { handle::get(device) }) else {
            return RVG_ERR_INVALID_ARGUMENT;
        };
        let state = if state.is_null() {
            None
        } else {
            // SAFETY: the caller's contract.
            Some(unsafe { std::slice::from_raw_parts(state, state_length) })
        };
        let (s, reason) = match d.device.will_resume(state) {
            Ok(()) => (RVG_OK, RVG_RESUME_ERROR_NONE),
            Err(Error::Resume(reason)) => (
                status(Error::Resume(reason)),
                reason.map_or(RVG_RESUME_ERROR_NONE, |r| r.code()),
            ),
            Err(e) => (status(e), RVG_RESUME_ERROR_NONE),
        };
        // SAFETY: the caller's contract.
        unsafe { handle::write_out(error, reason) };
        s
    })
}

/// Ends a resume that rvg_device_will_resume accepted: rebuilds the device from the state, starts
/// its FIFOs and raises one interrupt. May block: rebuilding the EFI display waits for its
/// present thread, and a root the device still has is torn down as rvg_device_stop does. May
/// call back.
///
/// # Safety
/// `device` is a live device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_device_did_resume(device: *mut RvgDevice) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_device_did_resume") }
            .device
            .did_resume();
    })
}

#[cfg(test)]
mod tests;
