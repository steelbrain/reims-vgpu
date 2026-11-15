//! `HostVm` on Windows: Mach VM's calls, as core makes them, over views of pagefile-backed
//! sections.
//!
//! Windows cannot show private memory at a second address as `mach_vm_remap` does; it can map a
//! section (`CreateFileMappingW`) at any number of addresses, and, since Windows 10 1803, map a
//! view into a reserved range, a placeholder, at the 4 KiB page's granularity (`VirtualAlloc2`
//! with `MEM_RESERVE_PLACEHOLDER`, `MapViewOfFile3` with `MEM_REPLACE_PLACEHOLDER`; read,
//! Microsoft's documentation, and measured on the UHD Graphics 630 host with Windows 11 26H2,
//! 2026-11-15: a view of 1,024 scattered pages, a placeholder split and a view mapped for each,
//! took 5.0 us a page). So every range the VM makes is a view of a section of its own, which the
//! backing table records as a run of that section's pages ([`files`]), and a view of guest pages
//! maps the sections the table names for them, as Linux's maps files:
//!
//! - **`allocate_anywhere`** and `allocate_aliasable` are the same: a new pagefile-backed section
//!   of the length (`SEC_COMMIT`), mapped whole, read and write, into a placeholder of its own. So
//!   every range can be the source of a view, as on Mach, where Linux's anonymous memory cannot.
//! - **A view, a fresh range or an unmap over part of the VM's ranges** first makes room
//!   ([`make_room`]): each Windows view it meets is unmapped back to a placeholder
//!   (`UnmapViewOfFile2` with `MEM_PRESERVE_PLACEHOLDER`), the placeholder is split at the range's
//!   ends (`VirtualFree` with `MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER`, each split off the front
//!   of a placeholder), and what of it lies outside the range is mapped again from the same section
//!   at the same offsets, with the protections the table holds, so those bytes keep their pages.
//!   A view then maps each piece of its source into the placeholders left, a Windows view for each
//!   piece in each placeholder ([`place`]); an unmap releases them (`VirtualFree` with
//!   `MEM_RELEASE`).
//! - **`allocate_fixed_overwrite`** over bytes no view of the VM's holds, free address space as
//!   a deallocate leaves it, first takes them: each run reserved at its place and mapped from a
//!   new section, as a view of the VM's ([`take_free`]), so the call meets views alone, as
//!   Mach's fixed overwrite takes unmapped bytes. Windows reserves address space from its
//!   allocation granularity (64 KiB) alone, so a run that starts inside such a block is refused,
//!   as are bytes in use outside the VM, with nothing changed.
//! - **While a call makes room**, the bytes of the Windows views it meets are unmapped for a
//!   moment. A thread that touches one then faults, and the VM's exception handler holds it while
//!   a call has the page in transit, then runs the access again where the page now allows it,
//!   looking again where a call began or ended while it looked ([`transit`]): no thread sees the
//!   gap, as none sees one on Mach or under Linux's `MAP_FIXED`, however many calls come between
//!   its fault and its handler.
//! - **`remap_shared_overwrite`** fails with `KERN_INVALID_ADDRESS` where a byte of the source has
//!   no section behind it, or a byte of the target lies outside the VM's views, before it changes
//!   anything. A piece Windows refuses fails the call with nothing changed: the pieces mapped
//!   before it are unmapped and the target's former contents mapped back.
//! - **`protect_max`** is `VirtualProtect` of each Windows view's part (a call takes one view at
//!   most), the current protection standing in for the maximum as on Linux. Windows has no
//!   write-only page: write alone is read and write, as on x86-64 Linux, and the table holds what
//!   the page has.
//! - **Whole pages**, of the host's 4 KiB (`GetSystemInfo`), taken as Linux's VM takes them
//!   (`os/linux/vm.rs`'s module docs).
//!
//! Windows counts no mappings against a limit as Linux's `vm.max_map_count`, so the views' budget,
//! dma-bufs and folds are Linux's alone. A range is charged against the host's commit limit for
//! its whole length when it is made, as a `SEC_COMMIT` section is, where Linux's anonymous memory
//! is charged nothing until touched: a reservation the views replace is charged until they do.

use std::collections::BTreeMap;
use std::os::windows::io::{AsRawHandle as _, FromRawHandle as _, OwnedHandle};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
    EXCEPTION_POINTERS,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER,
    MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS,
    MapViewOfFile3, PAGE_EXECUTE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_GUARD,
    PAGE_NOACCESS, PAGE_PROTECTION_FLAGS, PAGE_READONLY, PAGE_READWRITE, SEC_COMMIT,
    UnmapViewOfFile2, VirtualAlloc2, VirtualFree, VirtualProtect, VirtualQuery,
};
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use crate::core::num::{ToU64, ToUsize};
use crate::core::platform::os::vm::{HostAddr, HostVm, KernReturn, VmProt};
use crate::core::sync::Mutex;
use crate::os::HostFile;
use crate::os::files::{self, Backing, Piece};

/// `KERN_INVALID_ADDRESS`.
const INVALID_ADDRESS: KernReturn = KernReturn(1);
/// `KERN_PROTECTION_FAILURE`.
const PROTECTION_FAILURE: KernReturn = KernReturn(2);
/// `KERN_NO_SPACE`.
const NO_SPACE: KernReturn = KernReturn(3);
/// `KERN_INVALID_ARGUMENT`.
const INVALID_ARGUMENT: KernReturn = KernReturn(4);
/// `KERN_FAILURE`.
const FAILURE: KernReturn = KernReturn(5);

/// `ERROR_ACCESS_DENIED`.
const ACCESS_DENIED: i32 = 5;
/// `ERROR_NOT_ENOUGH_MEMORY`.
const NOT_ENOUGH_MEMORY: i32 = 8;
/// `ERROR_INVALID_PARAMETER`.
const INVALID_PARAMETER: i32 = 87;
/// `ERROR_COMMITMENT_LIMIT`.
const COMMITMENT_LIMIT: i32 = 1455;
/// `ERROR_INVALID_ADDRESS`.
const WINDOWS_INVALID_ADDRESS: i32 = 487;

/// One Windows view the VM mapped into a placeholder of its own: its end, and the section and
/// offset its first byte shows.
#[derive(Debug, Clone)]
struct View {
    end: u64,
    backing: Backing,
}

/// The VM's Windows views by first byte, which never overlap, and which every change to them makes
/// under this lock, one at a time. The backing table, a leaf lock, is changed inside it.
static VIEWS: Mutex<BTreeMap<u64, View>> = Mutex::new(BTreeMap::new());

/// The host's page size.
pub(crate) fn page_size() -> u64 {
    let mut info: SYSTEM_INFO = SYSTEM_INFO::default();
    // SAFETY: a live out-parameter; the call cannot fail.
    unsafe { GetSystemInfo(&mut info) };
    match info.dwPageSize {
        0 => 4096,
        size => u64::from(size),
    }
}

/// The whole pages `len` bytes at `addr` lie in: the first byte of the first, and their length;
/// `None` where they pass the end of the address space. Those `protect_max` and `deallocate`
/// take (module docs).
fn pages(addr: HostAddr, len: u64) -> Option<(u64, u64)> {
    let mask = page_size() - 1;
    let start = addr.addr().to_u64();
    let first = start & !mask;
    let end = start.checked_add(len)?.checked_add(mask)? & !mask;
    Some((first, end - first))
}

/// The pages a call of `len` bytes at `addr` takes where Mach takes the length rounded up to
/// whole pages from the page of the address (`allocate_fixed_overwrite`,
/// `remap_shared_overwrite`): the first byte of that page, and the length; `None` where they pass
/// the end of the address space.
fn length_pages(addr: HostAddr, len: u64) -> Option<(u64, u64)> {
    let mask = page_size() - 1;
    let first = addr.addr().to_u64() & !mask;
    let length = len.checked_add(mask)? & !mask;
    first.checked_add(length)?;
    Some((first, length))
}

/// A pointer to `at` in the process's address space.
pub(crate) fn ptr(at: u64) -> *mut core::ffi::c_void {
    std::ptr::with_exposed_provenance_mut(at.to_usize())
}

/// The protection Windows gives a page of `prot`: write alone is read and write, as on x86-64.
fn page_protection(prot: VmProt) -> PAGE_PROTECTION_FLAGS {
    let bits = prot.raw();
    let (read, write, exec) = (
        bits & VmProt::READ.raw() != 0,
        bits & VmProt::WRITE.raw() != 0,
        bits & EXECUTE != 0,
    );
    match (write, read, exec) {
        (true, _, false) => PAGE_READWRITE,
        (true, _, true) => PAGE_EXECUTE_READWRITE,
        (false, true, false) => PAGE_READONLY,
        (false, true, true) => PAGE_EXECUTE_READ,
        (false, false, false) => PAGE_NOACCESS,
        (false, false, true) => PAGE_EXECUTE,
    }
}

/// `prot` as a page of it has it ([`page_protection`]), which the table holds: write alone
/// with read.
fn held(prot: VmProt) -> VmProt {
    let bits = prot.raw() & (VmProt::READ.raw() | VmProt::WRITE.raw() | EXECUTE);
    if bits & VmProt::WRITE.raw() != 0 {
        VmProt(bits | VmProt::READ.raw())
    } else {
        VmProt(bits)
    }
}

/// `VM_PROT_EXECUTE`, which core's `VmProt` names no constant for.
const EXECUTE: i32 = 0x4;

/// Read and write, as a range is made.
const READ_WRITE: VmProt = VmProt(VmProt::READ.raw() | VmProt::WRITE.raw());

/// The Windows error of the last call that failed.
fn last_error() -> std::io::Error {
    std::io::Error::last_os_error()
}

/// A Mach answer for the Windows error of a refused call, which `what` names: no memory or commit
/// left is `KERN_NO_SPACE`, logged, as the host's memory running out
/// (`crate::answer::out_of_host_memory`, counted; in strict mode a death).
fn refused(what: &'static str, error: &std::io::Error) -> KernReturn {
    match error.raw_os_error() {
        Some(NOT_ENOUGH_MEMORY | COMMITMENT_LIMIT) => {
            crate::answer::out_of_host_memory(what, || {
                format!("Windows refused {what} for want of memory or commit charge: {error}")
            });
            NO_SPACE
        }
        Some(INVALID_PARAMETER) => INVALID_ARGUMENT,
        Some(ACCESS_DENIED) => PROTECTION_FAILURE,
        Some(WINDOWS_INVALID_ADDRESS) => INVALID_ADDRESS,
        _ => FAILURE,
    }
}

/// A new pagefile-backed section of `length` bytes of zeros, read and write, committed whole
/// (`SEC_COMMIT`).
pub(crate) fn section(length: u64) -> Result<Arc<HostFile>, KernReturn> {
    let high = u32::try_from(length >> 32).map_err(|_| NO_SPACE)?;
    let low = u32::try_from(length & 0xffff_ffff).map_err(|_| NO_SPACE)?;
    // SAFETY: a pagefile-backed section (no file), unnamed, with no security attributes.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            PAGE_READWRITE | SEC_COMMIT,
            high,
            low,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        return Err(refused("CreateFileMappingW", &last_error()));
    }
    // SAFETY: a handle just made, owned from here on.
    Ok(Arc::new(unsafe { OwnedHandle::from_raw_handle(handle) }))
}

/// A new placeholder of `length` bytes anywhere: its first byte.
fn reserve(length: u64) -> Result<u64, KernReturn> {
    // SAFETY: a fresh reservation of address space, at no address asked for.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            std::ptr::null(),
            length.to_usize(),
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            std::ptr::null_mut(),
            0,
        )
    };
    if at.is_null() {
        return Err(refused("VirtualAlloc2", &last_error()));
    }
    Ok(at.addr().to_u64())
}

/// Splits the `length` bytes from `at`, the front of a longer placeholder, off it, as a
/// placeholder of their own.
fn split_front(at: u64, length: u64) -> Result<(), KernReturn> {
    // SAFETY: the range is the front of a placeholder of the VM's (the caller's).
    if unsafe {
        VirtualFree(
            ptr(at),
            length.to_usize(),
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        )
    } == 0
    {
        return Err(refused("VirtualFree (split)", &last_error()));
    }
    Ok(())
}

/// Releases the placeholder at `at`, whole.
fn release(at: u64) -> Result<(), KernReturn> {
    // SAFETY: a placeholder of the VM's (the caller's), which nothing maps.
    if unsafe { VirtualFree(ptr(at), 0, MEM_RELEASE) } == 0 {
        return Err(refused("VirtualFree (release)", &last_error()));
    }
    Ok(())
}

/// Maps `length` bytes of `backing`'s section over the placeholder of exactly those bytes at `at`,
/// read and write, then lowered to `prot` where it is less. A section the process may only read
/// is mapped read-only where `prot` has no write.
fn map(at: u64, length: u64, backing: &Backing, prot: VmProt) -> Result<(), KernReturn> {
    let process = {
        // SAFETY: the process's pseudo-handle, which needs no closing.
        unsafe { GetCurrentProcess() }
    };
    let handle = backing.file.as_raw_handle();
    let map_as = |protection: PAGE_PROTECTION_FLAGS| {
        // SAFETY: a view of a section the backing holds open, over the placeholder of exactly
        // its bytes (the caller's).
        let mapped = unsafe {
            MapViewOfFile3(
                handle,
                process,
                ptr(at),
                backing.offset,
                length.to_usize(),
                MEM_REPLACE_PLACEHOLDER,
                protection,
                std::ptr::null_mut(),
                0,
            )
        };
        mapped.Value.addr().to_u64() == at
    };
    let prot = held(prot);
    let writes = prot.raw() & VmProt::WRITE.raw() != 0;
    if !map_as(PAGE_READWRITE) {
        let error = last_error();
        if writes || error.raw_os_error() != Some(ACCESS_DENIED) || !map_as(page_protection(prot)) {
            return Err(refused("MapViewOfFile3", &error));
        }
        return Ok(());
    }
    if prot != READ_WRITE {
        protect(at, length, prot)?;
    }
    Ok(())
}

/// Unmaps the Windows view at `at` back to a placeholder of its bytes.
fn unmap(at: u64) -> Result<(), KernReturn> {
    let view = MEMORY_MAPPED_VIEW_ADDRESS { Value: ptr(at) };
    // SAFETY: a view the VM mapped into a placeholder (the caller's).
    if unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) } == 0 {
        return Err(refused("UnmapViewOfFile2", &last_error()));
    }
    Ok(())
}

/// `VirtualProtect` of `length` bytes from `at`, which lie in one Windows view or allocation.
fn protect(at: u64, length: u64, prot: VmProt) -> Result<(), KernReturn> {
    let mut old: PAGE_PROTECTION_FLAGS = 0;
    // SAFETY: lowering or raising a range's protection is the caller's decision; a later access
    // it forbids faults (a process death).
    if unsafe { VirtualProtect(ptr(at), length.to_usize(), page_protection(prot), &mut old) } == 0 {
        return Err(refused("VirtualProtect", &last_error()));
    }
    Ok(())
}

/// Maps the placeholder of `length` bytes at `at` from `backing`, read and write, then sets each
/// piece the table holds there to its protection: the remainder of a Windows view that a call
/// made room beside, whose bytes keep their pages and protections ([`make_room`]).
fn remap_remainder(at: u64, length: u64, backing: &Backing) -> Result<(), KernReturn> {
    map(at, length, backing, READ_WRITE)?;
    for piece in files::pieces(at, length).unwrap_or_default() {
        if piece.prot != READ_WRITE {
            protect(at + piece.at, piece.length, piece.prot)?;
        }
    }
    Ok(())
}

/// The bytes of the Windows views a call is making room in ([`make_room`]), while it does: the
/// exception handler's ([`transit`]).
static TRANSIT: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// The calls' transits begun and ended, one count each: odd while a call holds [`TRANSIT`], which
/// it sets before its count and clears after its end's ([`transit`]). The calls make room one at a
/// time, under [`VIEWS`].
static TRANSITS: AtomicU64 = AtomicU64::new(0);
/// The lowest first byte and the highest end of every call's transit so far, each widened before
/// the call sets [`TRANSIT`]: the bytes whose access violations the handler takes.
static TRANSIT_BOUNDS: [AtomicU64; 2] = [AtomicU64::new(u64::MAX), AtomicU64::new(0)];

/// The bytes `first..end` in transit until the answer drops (module docs): a thread whose access
/// faults inside them waits for the drop, then runs the access again where the page allows it
/// ([`held_fault`]), its handler installed with the first.
struct Transit;

/// Marks `first..end` in transit ([`Transit`]).
fn transit(first: u64, end: u64) -> Transit {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        // SAFETY: a handler that reads the fault's record and the VM's atomics alone.
        let handler = unsafe { AddVectoredExceptionHandler(1, Some(held_fault)) };
        if handler.is_null() {
            log::error!(
                target: "reims_vgpu_vulkan::os",
                "the VM's exception handler was refused ({}): a thread that touches a view while \
                 a call makes room beside it faults",
                last_error()
            );
        }
    });
    TRANSIT_BOUNDS[0].fetch_min(first, Ordering::SeqCst);
    TRANSIT_BOUNDS[1].fetch_max(end, Ordering::SeqCst);
    TRANSIT[0].store(first, Ordering::SeqCst);
    TRANSIT[1].store(end, Ordering::SeqCst);
    TRANSITS.fetch_add(1, Ordering::SeqCst);
    Transit
}

impl Drop for Transit {
    fn drop(&mut self) {
        TRANSITS.fetch_add(1, Ordering::SeqCst);
        TRANSIT[1].store(0, Ordering::SeqCst);
        TRANSIT[0].store(0, Ordering::SeqCst);
    }
}

/// The access an access violation's record names (`ExceptionInformation[0]`): 0 a read, 1 a
/// write, 8 an execution.
fn allows(protection: PAGE_PROTECTION_FLAGS, access: usize) -> bool {
    if protection & PAGE_GUARD != 0 {
        return false;
    }
    let base = protection & 0xff;
    match access {
        1 => base == PAGE_READWRITE || base == PAGE_EXECUTE_READWRITE,
        8 => base == PAGE_EXECUTE || base == PAGE_EXECUTE_READ || base == PAGE_EXECUTE_READWRITE,
        _ => base != PAGE_NOACCESS && base != PAGE_EXECUTE,
    }
}

/// The VM's exception handler (module docs): an access violation inside the bytes calls have made
/// room in ([`TRANSIT_BOUNDS`]) waits while a call has its page in transit, then runs again where
/// the page now allows the access. Where it does not, and no call began or ended while the
/// handler looked ([`TRANSITS`]), the access violation goes on to the next handler, as does any
/// outside those bytes; else the handler looks again. So a thread whose access faulted while a
/// call unmapped its page for a moment runs on however many calls began and ended before its
/// handler ran, or between the handler's wait and its look.
unsafe extern "system" fn held_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: Windows hands the handler a live record of the exception.
    let record = unsafe { &*(*info).ExceptionRecord };
    if record.ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let (access, at) = (
        record.ExceptionInformation[0],
        record.ExceptionInformation[1].to_u64(),
    );
    if at < TRANSIT_BOUNDS[0].load(Ordering::SeqCst)
        || at >= TRANSIT_BOUNDS[1].load(Ordering::SeqCst)
    {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let in_transit = || {
        let end = TRANSIT[1].load(Ordering::SeqCst);
        end != 0 && TRANSIT[0].load(Ordering::SeqCst) <= at && at < end
    };
    loop {
        let transits = TRANSITS.load(Ordering::SeqCst);
        if transits % 2 == 1 && in_transit() {
            std::thread::yield_now();
            continue;
        }
        let mut info = MEMORY_BASIC_INFORMATION::default();
        let size = std::mem::size_of::<MEMORY_BASIC_INFORMATION>();
        // SAFETY: a query of the faulting page into a live buffer of its size.
        let read = unsafe { VirtualQuery(ptr(at), &mut info, size) };
        if read == size && info.State == MEM_COMMIT && allows(info.Protect, access) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if TRANSITS.load(Ordering::SeqCst) == transits {
            return EXCEPTION_CONTINUE_SEARCH;
        }
    }
}

/// The placeholders a call made room for, in order, which together are its range.
type Placeholders = Vec<(u64, u64)>;

/// The Windows views a call mapped, in order, each its first byte and end.
type Placed = Vec<(u64, u64)>;

/// Makes room over `first..end` (module docs): each Windows view meeting it unmapped, split at
/// the range's ends, and what of it lies outside mapped again; the placeholders left over the
/// range, in order, and the transit, which the caller holds until it has mapped them. Where
/// `whole`, every byte of the range must lie in a view of the VM's, else `KERN_INVALID_ADDRESS`
/// with nothing changed; otherwise bytes outside them are left out. With [`VIEWS`] held.
fn make_room(
    views: &mut BTreeMap<u64, View>,
    first: u64,
    end: u64,
    whole: bool,
) -> Result<(Placeholders, Transit), KernReturn> {
    let meeting: Vec<(u64, View)> = views
        .range(..end)
        .rev()
        .take_while(|(_, v)| v.end > first)
        .map(|(&at, v)| (at, v.clone()))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if whole {
        let mut covered = first;
        for (at, view) in &meeting {
            if *at > covered {
                break;
            }
            covered = covered.max(view.end);
        }
        if covered < end {
            log::debug!(
                target: "reims_vgpu_vulkan::os",
                "{:#x} bytes at {first:#x}: a byte at {covered:#x} lies in no view of the VM's",
                end - first
            );
            return Err(INVALID_ADDRESS);
        }
    }
    let span = match (meeting.first(), meeting.last()) {
        (Some((at, _)), Some((_, last))) => (*at, last.end),
        _ => (first, first),
    };
    let transit = transit(span.0, span.1);
    let mut room = Placeholders::with_capacity(meeting.len());
    for (at, view) in meeting {
        let (lo, hi) = (at.max(first), view.end.min(end));
        unmap(at)?;
        views.remove(&at);
        if at < lo {
            split_front(at, lo - at)?;
            remap_remainder(at, lo - at, &view.backing)?;
            views.insert(
                at,
                View {
                    end: lo,
                    backing: view.backing.clone(),
                },
            );
        }
        if hi < view.end {
            split_front(lo, hi - lo)?;
            let rest = Backing {
                file: Arc::clone(&view.backing.file),
                offset: view.backing.offset + (hi - at),
            };
            remap_remainder(hi, view.end - hi, &rest)?;
            views.insert(
                hi,
                View {
                    end: view.end,
                    backing: rest,
                },
            );
        }
        room.push((lo, hi));
    }
    Ok((room, transit))
}

/// Maps `pieces`, which follow on from `target`, into `room`, the placeholders over them, a
/// Windows view for each piece in each placeholder, each split off the front of the placeholder
/// it lies in: the views mapped, in order. On a refusal, the views mapped before it and the code;
/// `room` then holds the placeholders left, those views' own among them once the caller unmaps
/// them ([`unplace`]).
fn place(
    views: &mut BTreeMap<u64, View>,
    room: &mut Placeholders,
    target: u64,
    pieces: &[Piece],
) -> Result<Placed, (Placed, KernReturn)> {
    let mut placed = Vec::new();
    let mut holes = std::mem::take(room).into_iter();
    let mut hole = holes.next();
    for piece in pieces {
        let (mut at, end) = (target + piece.at, target + piece.at + piece.length);
        while at < end {
            let Some((lo, hi)) = hole else {
                room.extend(holes);
                return Err((placed, INVALID_ADDRESS));
            };
            debug_assert_eq!(lo, at, "the placeholders follow the pieces");
            let to = hi.min(end);
            if to < hi
                && let Err(kr) = split_front(lo, to - lo)
            {
                room.push((lo, hi));
                room.extend(holes);
                return Err((placed, kr));
            }
            let backing = Backing {
                file: Arc::clone(&piece.backing.file),
                offset: piece.backing.offset + (at - (target + piece.at)),
            };
            if let Err(kr) = map(at, to - at, &backing, piece.prot) {
                room.push((at, to));
                if to < hi {
                    room.push((to, hi));
                }
                room.extend(holes);
                return Err((placed, kr));
            }
            views.insert(at, View { end: to, backing });
            placed.push((at, to));
            at = to;
            hole = if to < hi {
                Some((to, hi))
            } else {
                holes.next()
            };
        }
    }
    room.extend(hole);
    room.extend(holes);
    Ok(placed)
}

/// Unmaps the views `placed` back to placeholders, adding them to `room` in order: a refused
/// call's put-back.
fn unplace(views: &mut BTreeMap<u64, View>, room: &mut Placeholders, placed: Placed) {
    for (at, end) in placed {
        if let Err(kr) = unmap(at) {
            log::error!(
                target: "reims_vgpu_vulkan::os",
                "a refused call's view of {:#x} bytes at {at:#x} could not be unmapped ({kr:?})",
                end - at
            );
            continue;
        }
        views.remove(&at);
        room.push((at, end));
    }
    room.sort_unstable();
}

/// Puts `pieces`, the target's former contents from `target`, back over `room` after a refused
/// call, logging what Windows refuses there too, whose bytes the table then forgets.
fn put_back(
    views: &mut BTreeMap<u64, View>,
    room: &mut Placeholders,
    target: u64,
    pieces: &[Piece],
) {
    if let Err((_, kr)) = place(views, room, target, pieces) {
        log::error!(
            target: "reims_vgpu_vulkan::os",
            "a refused call's put-back at {target:#x} was refused too ({kr:?}): {} placeholders \
             are released and forgotten",
            room.len()
        );
        for (at, end) in room.drain(..) {
            let _ = release(at);
            files::forget(at, end - at);
        }
    }
}

/// Takes the bytes of `first..end` no view of the VM's holds, free address space as a deallocate
/// leaves it: each run of them reserved at its place as a placeholder (`VirtualAlloc2` with
/// `MEM_RESERVE_PLACEHOLDER` at its first byte) and mapped from a new section of zeros, read and
/// write, as a view of the VM's, so a call over `first..end` meets views alone. The runs taken,
/// each its first byte and end. Where Windows refuses one (bytes in use outside the VM), those
/// taken before it are given back ([`give_back`]) and its code answered. With [`VIEWS`] held.
fn take_free(
    views: &mut BTreeMap<u64, View>,
    first: u64,
    end: u64,
) -> Result<Vec<(u64, u64)>, KernReturn> {
    let before = (views.range(..first).next_back())
        .filter(|(_, v)| v.end > first)
        .map(|(&at, v)| (at, v.end));
    let meeting = before
        .into_iter()
        .chain(views.range(first..end).map(|(&at, v)| (at, v.end)));
    let mut free = Vec::new();
    let mut at = first;
    for (lo, hi) in meeting {
        if lo > at {
            free.push((at, lo));
        }
        at = at.max(hi);
    }
    if at < end {
        free.push((at, end));
    }
    let mut taken = Vec::with_capacity(free.len());
    for (lo, hi) in free {
        match take(lo, hi - lo) {
            Ok(backing) => {
                views.insert(
                    lo,
                    View {
                        end: hi,
                        backing: backing.clone(),
                    },
                );
                files::record(lo, hi - lo, backing, READ_WRITE);
                taken.push((lo, hi));
            }
            Err(kr) => {
                give_back(views, &taken);
                return Err(kr);
            }
        }
    }
    Ok(taken)
}

/// A new section of `length` bytes mapped, read and write, into a placeholder reserved at `at`,
/// free address space: the section's backing.
fn take(at: u64, length: u64) -> Result<Backing, KernReturn> {
    let backing = Backing {
        file: section(length)?,
        offset: 0,
    };
    // SAFETY: a reservation of address space at `at`, which Windows refuses where any of it is
    // in use.
    let reserved = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            ptr(at),
            length.to_usize(),
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            std::ptr::null_mut(),
            0,
        )
    };
    if reserved.is_null() {
        return Err(refused("VirtualAlloc2 (fixed)", &last_error()));
    }
    if reserved.addr().to_u64() != at {
        let _ = release(reserved.addr().to_u64());
        return Err(NO_SPACE);
    }
    if let Err(kr) = map(at, length, &backing, READ_WRITE) {
        let _ = release(at);
        return Err(kr);
    }
    Ok(backing)
}

/// Gives back the runs [`take_free`] took, each `(first, end)`, as a deallocate does: unmapped,
/// released and forgotten by the table. With [`VIEWS`] held.
fn give_back(views: &mut BTreeMap<u64, View>, taken: &[(u64, u64)]) {
    for &(first, end) in taken {
        match make_room(views, first, end, false) {
            Ok((room, transit)) => {
                files::forget(first, end - first);
                for (at, _) in room {
                    let _ = release(at);
                }
                drop(transit);
            }
            Err(kr) => log::error!(
                target: "reims_vgpu_vulkan::os",
                "a refused fixed overwrite's {:#x} bytes at {first:#x} could not be given back \
                 ({kr:?})",
                end - first
            ),
        }
    }
}

/// Maps a new section of `length` bytes into a placeholder of its own, read and write, recording
/// the view: its first byte, and the section's backing.
fn fresh(views: &mut BTreeMap<u64, View>, length: u64) -> Result<(u64, Backing), KernReturn> {
    let backing = Backing {
        file: section(length)?,
        offset: 0,
    };
    let at = reserve(length)?;
    if let Err(kr) = map(at, length, &backing, READ_WRITE) {
        let _ = release(at);
        return Err(kr);
    }
    views.insert(
        at,
        View {
            end: at + length,
            backing: backing.clone(),
        },
    );
    Ok((at, backing))
}

/// A fresh range of `len` bytes, rounded up to whole pages, of zeros, that a view can be made of:
/// every range of the VM's is one (module docs).
pub(crate) fn allocate_aliasable(len: u64) -> Result<HostAddr, KernReturn> {
    WindowsVm::new().allocate_anywhere(len)
}

/// Memory of the library's own that a view can be made of and that only its holder frees: a
/// section's view, registered in the backing table ([`files::Registration`]), so that
/// `deallocate` refuses it: a graphics device's region ([`RegionMemory`]).
#[derive(Debug)]
pub(crate) struct OwnMemory {
    first: u64,
    length: u64,
    registration: Option<files::Registration>,
}

impl OwnMemory {
    /// `len` bytes, rounded up to whole pages, of zeros; `None` for none, or where the host gives
    /// no section or address space, as when memory runs out.
    pub(crate) fn new(len: u64) -> Option<OwnMemory> {
        let (_, length) = pages(HostAddr::NULL, len)?;
        if length == 0 {
            return None;
        }
        let mut views = VIEWS.lock();
        let (first, backing) = fresh(&mut views, length).ok()?;
        Some(OwnMemory {
            first,
            length,
            registration: Some(files::Registration::new(first, length, backing)),
        })
    }

    /// The first byte.
    pub(crate) fn first(&self) -> u64 {
        self.first
    }
}

impl Drop for OwnMemory {
    /// Drops the registration, then the view, through the platform's VM, which waits for a
    /// device's import of its pages to be freed first, as Vulkan requires of imported host memory
    /// (`VK_EXT_external_memory_host`; [`crate::remap::UnmapHold`]).
    fn drop(&mut self) {
        drop(self.registration.take());
        let at = HostAddr::from_addr(self.first.to_usize());
        let _ = crate::remap::VulkanVm::new().deallocate(at, self.length);
    }
}

/// A graphics device's region (BAR 2) in memory of the library's own ([`OwnMemory`]): a view of a
/// section of its own, so views of its pages can be made as of any range's.
#[derive(Debug)]
pub struct RegionMemory(OwnMemory);

impl RegionMemory {
    /// `len` bytes of zeros, rounded up to whole pages; `None` where the host gives no section or
    /// address space.
    pub fn new(len: u64) -> Option<RegionMemory> {
        OwnMemory::new(len).map(RegionMemory)
    }

    /// Its first byte.
    pub fn base(&self) -> *mut u8 {
        ptr(self.0.first()).cast()
    }
}

/// Mach VM's calls on Windows (module docs).
#[derive(Debug, Default)]
pub struct WindowsVm;

impl WindowsVm {
    /// The process's HostVm.
    pub fn new() -> WindowsVm {
        WindowsVm
    }

    /// Whether every page of the `len` bytes at `addr` is mapped with read access: by the backing
    /// table's protections where it knows every byte, else as Windows answers each region over
    /// them (`VirtualQuery`). A `len` of 0 is `true`.
    pub fn readable(&self, addr: HostAddr, len: u64) -> bool {
        if len == 0 {
            return true;
        }
        let Some((first, length)) = pages(addr, len) else {
            return false;
        };
        files::readable(first, length).unwrap_or_else(|| regions_allow(first, first + length, 0))
    }
}

/// Whether every page of `first..end` is committed with a protection that allows `access` (0 a
/// read, 1 a write), as Windows answers each region over them.
pub(crate) fn regions_allow(first: u64, end: u64, access: usize) -> bool {
    let mut at = first;
    while at < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        let size = std::mem::size_of::<MEMORY_BASIC_INFORMATION>();
        // SAFETY: a query into a live buffer of its size.
        if unsafe { VirtualQuery(ptr(at), &mut info, size) } != size
            || info.State != MEM_COMMIT
            || !allows(info.Protect, access)
        {
            return false;
        }
        let region_end = info.BaseAddress.addr().to_u64() + info.RegionSize.to_u64();
        if region_end <= at {
            return false;
        }
        at = region_end;
    }
    true
}

/// The views of `runs`, each `(source, length)` whole pages, laid one after another from
/// `target`, as one remap at a time leaves them (`HostVm::remap_runs_shared_overwrite`).
fn one_at_a_time(
    vm: &WindowsVm,
    dst: HostAddr,
    runs: &[(HostAddr, u64)],
) -> Result<(), (usize, KernReturn)> {
    let mut at = dst;
    for (i, &(src, len)) in runs.iter().enumerate() {
        vm.remap_shared_overwrite(at, src, len)
            .map_err(|kr| (i, kr))?;
        at = at.wrapping_add(len);
    }
    Ok(())
}

/// Maps `pieces`, which follow on from `target`, over `length` bytes of the VM's views there, all
/// or nothing, and records them in the table. With [`VIEWS`] held.
fn map_view(
    views: &mut BTreeMap<u64, View>,
    target: u64,
    length: u64,
    pieces: Vec<Piece>,
) -> Result<(), KernReturn> {
    let former = files::pieces(target, length);
    let (mut room, _transit) = make_room(views, target, target + length, true)?;
    if let Err((placed, kr)) = place(views, &mut room, target, &pieces) {
        unplace(views, &mut room, placed);
        match former {
            Some(former) => put_back(views, &mut room, target, &former),
            None => {
                for (at, end) in room.drain(..) {
                    let _ = release(at);
                    files::forget(at, end - at);
                }
            }
        }
        return Err(kr);
    }
    files::record_pieces(target, pieces);
    Ok(())
}

impl HostVm for WindowsVm {
    fn page_size(&self) -> u64 {
        page_size()
    }

    fn page_shift(&self) -> u32 {
        page_size().trailing_zeros()
    }

    /// A new section's view anywhere (module docs); a length of 0 answers address 0, as Mach's
    /// does.
    fn allocate_anywhere(&self, len: u64) -> Result<HostAddr, KernReturn> {
        let Some((_, length)) = pages(HostAddr::NULL, len) else {
            return Err(NO_SPACE);
        };
        if length == 0 {
            return Ok(HostAddr::NULL);
        }
        let mut views = VIEWS.lock();
        let (at, backing) = fresh(&mut views, length)?;
        files::record(at, length, backing, READ_WRITE);
        Ok(HostAddr::from_addr(at.to_usize()))
    }

    /// A new section's view over the pages of the range's length from the page of its address
    /// (module docs): over the VM's views it meets, and over free address space between them
    /// ([`take_free`]), as Mach's fixed overwrite takes unmapped bytes. Bytes in use outside the
    /// VM refuse it, with nothing changed.
    fn allocate_fixed_overwrite(&self, addr: HostAddr, len: u64) -> Result<(), KernReturn> {
        let (first, length) = length_pages(addr, len).ok_or(NO_SPACE)?;
        if length == 0 {
            return Ok(());
        }
        let backing = Backing {
            file: section(length)?,
            offset: 0,
        };
        let piece = Piece {
            at: 0,
            length,
            backing: backing.clone(),
            prot: READ_WRITE,
        };
        let mut views = VIEWS.lock();
        let taken = take_free(&mut views, first, first + length)?;
        let mapped = map_view(&mut views, first, length, vec![piece]);
        if mapped.is_err() {
            give_back(&mut views, &taken);
        }
        mapped
    }

    /// The source's pages of the length, from the page of its first byte, mapped from their
    /// sections over as many from `dst`'s page, each with the protection the source has there:
    /// that page's first byte, which Mach writes back (module docs).
    fn remap_shared_overwrite(
        &self,
        dst: HostAddr,
        src: HostAddr,
        len: u64,
    ) -> Result<HostAddr, KernReturn> {
        if len == 0 {
            return Err(INVALID_ARGUMENT);
        }
        let (target, length) = length_pages(dst, len).ok_or(INVALID_ARGUMENT)?;
        let (source, _) = length_pages(src, len).ok_or(INVALID_ADDRESS)?;
        let mut views = VIEWS.lock();
        let Some(pieces) = files::pieces(source, length) else {
            log::debug!(
                target: "reims_vgpu_vulkan::os",
                "remap of {length:#x} bytes at {source:#x}: no section behind them"
            );
            return Err(INVALID_ADDRESS);
        };
        map_view(&mut views, target, length, pieces)?;
        Ok(HostAddr::from_addr(target.to_usize()))
    }

    /// The runs' pages mapped as one view, whole pages each ([`map_view`]). A run with no section
    /// behind a byte is refused after the runs before it are mapped, as one remap at a time leaves
    /// them; a map Windows refuses leaves none of them mapped, answered at the first run. Runs
    /// that are not whole pages from a page, or a view not from one, are remapped one at a time.
    fn remap_runs_shared_overwrite(
        &self,
        dst: HostAddr,
        runs: &[(HostAddr, u64)],
    ) -> Result<(), (usize, KernReturn)> {
        let page = page_size();
        let on_pages = |at: HostAddr| at.addr().to_u64().is_multiple_of(page);
        if !on_pages(dst)
            || (runs.iter())
                .any(|&(src, len)| len == 0 || !len.is_multiple_of(page) || !on_pages(src))
        {
            return one_at_a_time(self, dst, runs);
        }
        let target = dst.addr().to_u64();
        let sources: Vec<(u64, u64)> = (runs.iter())
            .map(|&(src, len)| (src.addr().to_u64(), len))
            .collect();
        let mut views = VIEWS.lock();
        match files::view_pieces(&sources) {
            Ok(all) => {
                let length = sources.iter().map(|&(_, len)| len).sum();
                map_view(&mut views, target, length, all).map_err(|kr| (0, kr))
            }
            Err((i, before)) => {
                let (source, len) = sources[i];
                log::debug!(
                    target: "reims_vgpu_vulkan::os",
                    "remap of {len:#x} bytes at {source:#x}: no section behind them"
                );
                if !before.is_empty() {
                    let length = sources[..i].iter().map(|&(_, len)| len).sum();
                    map_view(&mut views, target, length, before).map_err(|kr| (0, kr))?;
                }
                Err((i, INVALID_ADDRESS))
            }
        }
    }

    /// `VirtualProtect` of the range's pages, a call for each Windows view's part and for each
    /// stretch outside them; none for no bytes, and none where the table holds the protection
    /// already (module docs).
    fn protect_max(&self, addr: HostAddr, len: u64, prot: VmProt) -> Result<(), KernReturn> {
        if len == 0 {
            return Ok(());
        }
        let (first, length) = pages(addr, len).ok_or(INVALID_ARGUMENT)?;
        let end = first + length;
        let prot = held(prot);
        let views = VIEWS.lock();
        if files::protected(first, length, prot) {
            return Ok(());
        }
        let mut at = first;
        let meeting = views
            .range(..end)
            .rev()
            .take_while(|(_, v)| v.end > first)
            .map(|(&start, v)| (start.max(first), v.end.min(end)))
            .collect::<Vec<_>>();
        for (lo, hi) in meeting.into_iter().rev() {
            if at < lo {
                protect(at, lo - at, prot)?;
            }
            protect(lo, hi - lo, prot)?;
            at = hi;
        }
        if at < end {
            protect(at, end - at, prot)?;
        }
        files::protect(first, length, prot);
        Ok(())
    }

    /// The range's pages unmapped and released, which show no section from then on; none for no
    /// bytes, and bytes outside the VM's views left as they are (module docs). A range of memory
    /// the library registered and does not own (the guest's RAM) is refused, unmapped by nothing.
    fn deallocate(&self, addr: HostAddr, len: u64) -> KernReturn {
        if len == 0 {
            return KernReturn::SUCCESS;
        }
        let Some((first, length)) = pages(addr, len) else {
            return INVALID_ARGUMENT;
        };
        if files::meets_registration(first, length) {
            log::error!(
                target: "reims_vgpu_vulkan::os",
                "deallocate of {length:#x} bytes at {first:#x}: registered memory, not unmapped"
            );
            return INVALID_ARGUMENT;
        }
        let mut views = VIEWS.lock();
        let (room, transit) = match make_room(&mut views, first, first + length, false) {
            Ok(made) => made,
            Err(kr) => return kr,
        };
        files::forget(first, length);
        let mut answer = KernReturn::SUCCESS;
        for (at, _) in room {
            if let Err(kr) = release(at) {
                answer = kr;
            }
        }
        drop(transit);
        answer
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn vm() -> WindowsVm {
        WindowsVm::new()
    }

    fn page() -> u64 {
        page_size()
    }

    fn read(a: HostAddr, off: u64) -> u8 {
        // SAFETY: test memory this test mapped readable.
        unsafe { a.wrapping_add(off).as_ptr::<u8>().read_volatile() }
    }

    fn write(a: HostAddr, off: u64, v: u8) {
        // SAFETY: test memory this test mapped writable.
        unsafe { a.wrapping_add(off).as_ptr::<u8>().write_volatile(v) }
    }

    /// The host's page, as the tests take it.
    pub(crate) const PAGE: u64 = 0x1000;

    /// Whether every page of `len` bytes from `at` is writable now, as Windows answers its regions.
    pub(crate) fn writable(at: HostAddr, len: u64) -> bool {
        let first = at.addr().to_u64();
        regions_allow(first, first.saturating_add(len), 1)
    }

    /// Whether the page of `at` is committed, as Windows answers it.
    pub(crate) fn mapped(at: HostAddr) -> bool {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        let size = std::mem::size_of::<MEMORY_BASIC_INFORMATION>();
        // SAFETY: a query into a live buffer of its size.
        let read = unsafe { VirtualQuery(at.as_ptr::<core::ffi::c_void>(), &mut info, size) };
        read == size && info.State == MEM_COMMIT
    }

    /// The VM's Windows views inside `first..end`.
    fn views_in(first: u64, end: u64) -> Vec<(u64, u64)> {
        (VIEWS.lock().range(first..end))
            .map(|(&at, v)| (at, v.end))
            .collect()
    }

    /// A range is zeros, read and write, a view of a section of its own, which the table names;
    /// its deallocation unmaps it and forgets it
    #[test]
    fn a_range_is_a_sections_view() {
        let v = vm();
        let p = page();
        assert_eq!(p, PAGE);
        let a = v.allocate_anywhere(3 * p).unwrap();
        let first = a.addr().to_u64();
        assert_eq!(read(a, 0), 0);
        write(a, 2 * p + 7, 0x5a);
        assert_eq!(views_in(first, first + 3 * p), [(first, first + 3 * p)]);
        let pieces = files::pieces(first, 3 * p).unwrap();
        assert_eq!(pieces.len(), 1);
        assert!(writable(a, 3 * p) && mapped(a));
        assert_eq!(v.deallocate(a, 3 * p), KernReturn::SUCCESS);
        assert!(!mapped(a));
        assert!(files::pieces(first, p).is_none());
        assert!(views_in(first, first + 3 * p).is_empty());
    }

    /// A remap shows the source's pages at the target, both ways, and keeps showing them after the
    /// source goes; the target's bytes outside it keep theirs
    #[test]
    fn a_remap_aliases_and_outlives_its_source() {
        let v = vm();
        let p = page();
        let src = v.allocate_anywhere(4 * p).unwrap();
        let dst = v.allocate_anywhere(4 * p).unwrap();
        for i in 0..4 {
            write(src, i * p, 0x10 + u8::try_from(i).unwrap());
            write(dst, i * p, 0x20 + u8::try_from(i).unwrap());
        }
        let at = dst.wrapping_add(p);
        assert_eq!(
            v.remap_shared_overwrite(at, src.wrapping_add(2 * p), 2 * p),
            Ok(at)
        );
        assert_eq!(
            [
                read(dst, 0),
                read(dst, p),
                read(dst, 2 * p),
                read(dst, 3 * p)
            ],
            [0x20, 0x12, 0x13, 0x23]
        );
        write(dst, p + 1, 0x77);
        assert_eq!(read(src, 2 * p + 1), 0x77, "the same pages");
        assert_eq!(v.deallocate(src, 4 * p), KernReturn::SUCCESS);
        assert_eq!(read(dst, p), 0x12, "kept after the source went");
        let first = dst.addr().to_u64();
        assert_eq!(views_in(first, first + 4 * p).len(), 3);
        assert_eq!(v.deallocate(dst, 4 * p), KernReturn::SUCCESS);
    }

    /// A view of scattered runs is mapped whole, each run its own Windows view, in order
    #[test]
    fn a_view_of_runs_maps_each_run() {
        let v = vm();
        let p = page();
        let src = v.allocate_anywhere(8 * p).unwrap();
        for i in 0..8 {
            write(src, i * p, u8::try_from(i).unwrap() + 1);
        }
        let dst = v.allocate_anywhere(4 * p).unwrap();
        let runs: Vec<(HostAddr, u64)> = [5, 1, 6, 0]
            .iter()
            .map(|&i| (src.wrapping_add(i * p), p))
            .collect();
        assert_eq!(v.remap_runs_shared_overwrite(dst, &runs), Ok(()));
        assert_eq!(
            [
                read(dst, 0),
                read(dst, p),
                read(dst, 2 * p),
                read(dst, 3 * p)
            ],
            [6, 2, 7, 1]
        );
        let first = dst.addr().to_u64();
        assert_eq!(views_in(first, first + 4 * p).len(), 4);
        assert_eq!(v.deallocate(dst, 4 * p), KernReturn::SUCCESS);
        assert_eq!(v.deallocate(src, 8 * p), KernReturn::SUCCESS);
    }

    /// A remap of bytes no section is behind, or onto bytes outside the VM's views, changes
    /// nothing and answers `KERN_INVALID_ADDRESS`
    #[test]
    fn a_remap_of_no_section_changes_nothing() {
        let v = vm();
        let p = page();
        let dst = v.allocate_anywhere(2 * p).unwrap();
        write(dst, 0, 9);
        let stack = [0u8; 64];
        let foreign = HostAddr::from_addr(std::ptr::from_ref(&stack).addr());
        assert_eq!(
            v.remap_shared_overwrite(dst, foreign, p),
            Err(INVALID_ADDRESS)
        );
        assert_eq!(read(dst, 0), 9);
        let src = v.allocate_anywhere(p).unwrap();
        assert_eq!(
            v.remap_shared_overwrite(foreign, src, p),
            Err(INVALID_ADDRESS)
        );
        assert_eq!(v.deallocate(dst, 2 * p), KernReturn::SUCCESS);
        assert_eq!(v.deallocate(src, p), KernReturn::SUCCESS);
    }

    /// A fresh range over part of a range is zeros, and the rest keeps its bytes; a protect sets
    /// each view's part and survives a cut beside it
    #[test]
    fn a_fixed_overwrite_and_protect_cut_views() {
        let v = vm();
        let p = page();
        let a = v.allocate_anywhere(4 * p).unwrap();
        for i in 0..4 {
            write(a, i * p, 0x40 + u8::try_from(i).unwrap());
        }
        assert_eq!(
            v.protect_max(a, p, VmProt::READ),
            Ok(()),
            "the first page read-only"
        );
        assert!(!writable(a, p) && writable(a.wrapping_add(p), p));
        assert_eq!(v.allocate_fixed_overwrite(a.wrapping_add(2 * p), p), Ok(()));
        assert_eq!(
            [read(a, 0), read(a, p), read(a, 2 * p), read(a, 3 * p)],
            [0x40, 0x41, 0, 0x43]
        );
        assert!(!writable(a, p), "the remainder keeps its protection");
        assert_eq!(v.protect_max(a, 4 * p, READ_WRITE), Ok(()));
        assert!(writable(a, 4 * p));
        assert_eq!(v.deallocate(a.wrapping_add(p), p), KernReturn::SUCCESS);
        assert!(!mapped(a.wrapping_add(p)) && mapped(a) && mapped(a.wrapping_add(2 * p)));
        assert_eq!(v.deallocate(a, 4 * p), KernReturn::SUCCESS);
    }

    /// A fixed overwrite over a range the VM deallocated whole takes the free address space
    /// again, fresh zeros; over a page freed inside a range, which starts no block of Windows's
    /// allocation granularity, it is refused, the bytes beside it as they were
    #[test]
    fn a_fixed_overwrite_takes_freed_address_space() {
        let v = vm();
        let p = page();
        let a = v.allocate_anywhere(4 * p).unwrap();
        write(a, 0, 7);
        assert_eq!(v.deallocate(a, 4 * p), KernReturn::SUCCESS);
        assert!(!mapped(a));
        assert_eq!(v.allocate_fixed_overwrite(a, 2 * p), Ok(()));
        assert!(mapped(a) && writable(a, 2 * p) && !mapped(a.wrapping_add(2 * p)));
        assert_eq!([read(a, 0), read(a, p)], [0, 0]);
        write(a, p, 5);
        assert_eq!(read(a, p), 5);
        let inner = a.wrapping_add(p);
        assert_eq!(v.deallocate(inner, p), KernReturn::SUCCESS);
        assert!(!mapped(inner));
        assert!(v.allocate_fixed_overwrite(inner, p).is_err());
        assert!(!mapped(inner) && mapped(a));
        assert_eq!(read(a, 0), 0);
        assert_eq!(v.deallocate(a, 2 * p), KernReturn::SUCCESS);
    }

    /// A thread that reads a page while other calls unmap it for a moment, again and again (a
    /// remap beside it in its Windows view, which maps it again as a remainder, and a remap over
    /// it of the same page), never faults, and reads the byte it holds
    #[test]
    fn a_reader_beside_a_cut_never_faults() {
        let v = vm();
        let p = page();
        let a = v.allocate_anywhere(2 * p).unwrap();
        let pair = v.allocate_anywhere(2 * p).unwrap();
        let other = v.allocate_anywhere(p).unwrap();
        write(pair, 0, 0x3c);
        assert_eq!(v.remap_shared_overwrite(a, pair, 2 * p), Ok(a));
        let stop = std::sync::atomic::AtomicBool::new(false);
        let reads = AtomicU64::new(0);
        std::thread::scope(|s| {
            s.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    assert_eq!(read(a, 0), 0x3c);
                    reads.fetch_add(1, Ordering::Relaxed);
                }
            });
            for i in 0..2000u64 {
                if i % 2 == 0 {
                    let at = a.wrapping_add(p);
                    assert_eq!(v.remap_shared_overwrite(at, other, p), Ok(at));
                } else {
                    assert_eq!(v.remap_shared_overwrite(a, pair, 2 * p), Ok(a));
                }
            }
            stop.store(true, Ordering::Relaxed);
        });
        assert!(reads.load(Ordering::Relaxed) > 0);
        for range in [a, pair] {
            assert_eq!(v.deallocate(range, 2 * p), KernReturn::SUCCESS);
        }
        assert_eq!(v.deallocate(other, p), KernReturn::SUCCESS);
    }

    /// An access that faulted while a call unmapped its page for a moment runs again once the
    /// page allows it, though calls over other bytes began and ended before its handler ran, as a
    /// thread the host's scheduler held between its fault and its handler meets them; an access
    /// its page does not allow, with no call under way, goes on to the next handler
    #[test]
    fn a_fault_handled_after_calls_elsewhere_runs_again_where_its_page_allows_it() {
        use windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_RECORD;
        let v = vm();
        let p = page();
        let a = v.allocate_anywhere(2 * p).unwrap();
        let b = v.allocate_anywhere(2 * p).unwrap();
        let other = v.allocate_anywhere(p).unwrap();
        // A call that makes room over `a`'s view, its first page among the bytes in transit, then
        // two over `b`'s alone.
        let a1 = a.wrapping_add(p);
        assert_eq!(v.remap_shared_overwrite(a1, other, p), Ok(a1));
        let b1 = b.wrapping_add(p);
        for _ in 0..2 {
            assert_eq!(v.remap_shared_overwrite(b1, other, p), Ok(b1));
        }
        let handled = |at: HostAddr, access: usize| {
            let mut record = EXCEPTION_RECORD {
                ExceptionCode: EXCEPTION_ACCESS_VIOLATION,
                NumberParameters: 2,
                ..EXCEPTION_RECORD::default()
            };
            record.ExceptionInformation[0] = access;
            record.ExceptionInformation[1] = at.addr();
            let mut info = EXCEPTION_POINTERS {
                ExceptionRecord: &raw mut record,
                ContextRecord: std::ptr::null_mut(),
            };
            // SAFETY: a live record of an access violation, which the handler only reads.
            unsafe { held_fault(&raw mut info) }
        };
        assert_eq!(handled(a, 0), EXCEPTION_CONTINUE_EXECUTION, "a read of `a`");
        assert_eq!(v.protect_max(a, p, VmProt::READ), Ok(()));
        assert_eq!(
            handled(a, 1),
            EXCEPTION_CONTINUE_SEARCH,
            "a write of `a`, read-only"
        );
        assert_eq!(
            handled(a, 0),
            EXCEPTION_CONTINUE_EXECUTION,
            "a read of `a`, read-only"
        );
        for (range, len) in [(a, 2 * p), (b, 2 * p), (other, p)] {
            assert_eq!(v.deallocate(range, len), KernReturn::SUCCESS);
        }
    }

    /// Registered memory is never deallocated
    #[test]
    fn registered_memory_is_refused() {
        let v = vm();
        let own = OwnMemory::new(2 * page()).unwrap();
        let at = HostAddr::from_addr(own.first().to_usize());
        assert_eq!(v.deallocate(at, page()), INVALID_ARGUMENT);
        assert!(mapped(at));
        drop(own);
        assert!(!mapped(at));
    }
}
