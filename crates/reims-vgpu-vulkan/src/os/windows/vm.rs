//! Mach-style shared aliases implemented by Windows placeholder page mappings.
//!
//! The table owns mappings, not copies of bytes. Every source page resolves to its section and
//! offset before any destination changes, including overlapping remaps. Partial unmaps remove
//! only their page views; neighbouring imports keep the same virtual and physical addresses.

use super::mapping::{PAGE, PageArena, Section};
use super::reservation::Reservation;
use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm, KernReturn, VmProt};
use reims_vgpu_core::sync::Mutex;
use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock};

const INVALID_ADDRESS: KernReturn = KernReturn(1);
const PROTECTION_FAILURE: KernReturn = KernReturn(2);
const NO_SPACE: KernReturn = KernReturn(3);
const INVALID_ARGUMENT: KernReturn = KernReturn(4);
const FAILURE: KernReturn = KernReturn(5);

/// Owned allocations and their maximum page protections. A cleared page has no table entry.
struct Allocation {
    arena: PageArena,
    maximum: Vec<i32>,
}

static ALLOCATIONS: LazyLock<Mutex<BTreeMap<usize, Allocation>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));
static RESERVATIONS: Mutex<BTreeMap<usize, Reservation>> = Mutex::new(BTreeMap::new());

/// A destination is owned memory or part of a task's sparse reservation, never borrowed RAM.
#[derive(Clone, Copy)]
enum Destination {
    Owned(usize, usize),
    Reserved(usize, usize),
}
type Backing = (Arc<Section>, usize, i32);

fn destination(
    table: &BTreeMap<usize, Allocation>,
    reservations: &BTreeMap<usize, Reservation>,
    address: usize,
) -> Result<Destination, KernReturn> {
    if let Some((base, page)) = location(table, address) {
        return Ok(Destination::Owned(base, page));
    }
    let (&base, reservation) = reservations
        .range(..=address)
        .next_back()
        .ok_or(INVALID_ADDRESS)?;
    reservation
        .contains(address - base)
        .then_some(Destination::Reserved(base, address - base))
        .ok_or(INVALID_ADDRESS)
}

fn previous(
    table: &BTreeMap<usize, Allocation>,
    reservations: &BTreeMap<usize, Reservation>,
    destination: Destination,
) -> Option<Backing> {
    match destination {
        Destination::Owned(base, page) => {
            let allocation = &table[&base];
            let (section, offset) = allocation.arena.backing(page);
            Some((section, offset, allocation.maximum[page]))
        }
        Destination::Reserved(base, offset) => reservations[&base].backing(offset),
    }
}

fn replace(
    table: &mut BTreeMap<usize, Allocation>,
    reservations: &mut BTreeMap<usize, Reservation>,
    destination: Destination,
    backing: Option<Backing>,
) -> std::io::Result<()> {
    match destination {
        Destination::Owned(base, page) => {
            let (section, offset, maximum) = backing.unwrap_or_else(|| {
                super::failure::abort("owned allocations always retain backing")
            });
            let allocation = table
                .get_mut(&base)
                .unwrap_or_else(|| super::failure::abort(concat!(module_path!(), ":", line!())));
            allocation.arena.replace(page, section, offset)?;
            allocation.arena.protect_access(page, maximum)?;
            allocation.maximum[page] = maximum;
            Ok(())
        }
        Destination::Reserved(base, offset) => reservations
            .get_mut(&base)
            .unwrap_or_else(|| super::failure::abort(concat!(module_path!(), ":", line!())))
            .replace(offset, backing),
    }
}

/// Validate destinations before any change; restore every changed page on a kernel refusal.
fn overwrite(
    table: &mut BTreeMap<usize, Allocation>,
    reservations: &mut BTreeMap<usize, Reservation>,
    start: usize,
    sources: Vec<Backing>,
) -> Result<(), KernReturn> {
    let targets = (0..sources.len())
        .map(|i| destination(table, reservations, start + i * PAGE))
        .collect::<Result<Vec<_>, _>>()?;
    let old: Vec<_> = targets
        .iter()
        .map(|&d| previous(table, reservations, d))
        .collect();
    for (i, (target, source)) in targets.iter().copied().zip(sources).enumerate() {
        if let Err(error) = replace(table, reservations, target, Some(source)) {
            for (target, backing) in targets.iter().copied().zip(old).take(i + 1) {
                replace(table, reservations, target, backing).unwrap_or_else(|error| {
                    super::failure::failed(error, "Windows failed to restore a remap")
                });
            }
            return Err(refusal(error));
        }
    }
    Ok(())
}

/// Guest RAM registered by the device. Aliases keep the section after a registration ends.
struct GuestRange {
    host: usize,
    length: usize,
    section: Arc<Section>,
    offset: usize,
    id: u64,
}
static GUESTS: Mutex<Vec<GuestRange>> = Mutex::new(Vec::new());

/// Registers file-backed host RAM without remapping it or changing its contents.
/// QEMU keeps this shared view read/write until the device releases its lease;
/// protection changes through HostVm apply only to separately owned aliases.
pub(super) fn register_guest(
    host: usize,
    length: usize,
    file: &std::fs::File,
    offset: usize,
) -> std::io::Result<u64> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    // This is the boundary receiving QEMU's memory map. Refuse malformed extents before
    // publishing an address that another task could resolve or hand to the GPU.
    let file_size = usize::try_from(file.metadata()?.len()).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "RAM file extent overflow")
    })?;
    if length == 0
        || !host.is_multiple_of(PAGE)
        || !length.is_multiple_of(PAGE)
        || !offset.is_multiple_of(PAGE)
        || host.checked_add(length).is_none()
        || offset.checked_add(length).is_none_or(|end| end > file_size)
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid guest RAM extent",
        ));
    }
    let section = Section::of_file(file)?;
    // Validate the borrowed mapping once, at registration. VirtualQuery scans to
    // the end of a populated file view, even when asked about a single 4 KiB page.
    // Repeating it in gather_short for thousands of RAM pages stalls GPU jobs.
    if !super::mapping::writable(host as u64, (host + length) as u64) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "guest RAM must remain mapped read/write for the device lease",
        ));
    }
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    GUESTS.lock().push(GuestRange {
        host,
        length,
        section,
        offset,
        id,
    });
    Ok(id)
}

/// Ends only this lease's registration; other devices may hold an identical range.
pub(super) fn unregister_guest(id: u64) {
    GUESTS.lock().retain(|range| range.id != id);
}

/// Answer only for a complete range covered by a live, validated RAM lease.
/// No protection result is cached for aliases, holes, or arbitrary host memory.
fn registered_ram_covers(start: usize, end: usize) -> bool {
    GUESTS
        .lock()
        .iter()
        .any(|range| range.host <= start && end <= range.host + range.length)
}

/// Convert an OS refusal to a real Mach error; never substitute successful zero-filled memory.
fn refusal(error: std::io::Error) -> KernReturn {
    match error.raw_os_error() {
        Some(8 | 14 | 1450 | 1455) => NO_SPACE,
        Some(5) => PROTECTION_FAILURE,
        Some(87) => INVALID_ARGUMENT,
        Some(487) => INVALID_ADDRESS,
        _ => FAILURE,
    }
}

/// Mach allocate/remap rounds length from the address's first page, not its final byte.
fn length_pages(address: HostAddr, length: u64) -> Result<(usize, usize), KernReturn> {
    let length = usize::try_from(length).map_err(|_| INVALID_ARGUMENT)?;
    let rounded = length.checked_add(PAGE - 1).ok_or(INVALID_ARGUMENT)? / PAGE * PAGE;
    let first = address.addr() / PAGE * PAGE;
    first.checked_add(rounded).ok_or(INVALID_ARGUMENT)?;
    Ok((first, rounded))
}

/// Mach protect/deallocate covers every page touched by the supplied byte range.
fn touched_pages(address: HostAddr, length: u64) -> Result<(usize, usize), KernReturn> {
    if length == 0 {
        return Ok((address.addr() / PAGE * PAGE, 0));
    }
    let count = usize::try_from(length).map_err(|_| INVALID_ARGUMENT)?;
    let end = address
        .addr()
        .checked_add(count)
        .and_then(|n| n.checked_add(PAGE - 1))
        .ok_or(INVALID_ARGUMENT)?
        / PAGE
        * PAGE;
    let first = address.addr() / PAGE * PAGE;
    Ok((first, end - first))
}

/// Resolve an address to a live owned page; holes are not addresses of a surviving allocation.
fn location(table: &BTreeMap<usize, Allocation>, address: usize) -> Option<(usize, usize)> {
    let (&base, allocation) = table.range(..=address).next_back()?;
    let index = (address - base) / PAGE;
    allocation.arena.contains(index).then_some((base, index))
}

/// Snapshot the backing and maximum protection before changing possibly overlapping pages.
fn resolve(
    table: &BTreeMap<usize, Allocation>,
    reservations: &BTreeMap<usize, Reservation>,
    address: usize,
) -> Result<(Arc<Section>, usize, i32), KernReturn> {
    if let Some((base, index)) = location(table, address) {
        let allocation = &table[&base];
        let (section, offset) = allocation.arena.backing(index);
        return Ok((section, offset, allocation.maximum[index]));
    }
    if let Ok(Destination::Reserved(base, offset)) = destination(table, reservations, address) {
        return reservations[&base].backing(offset).ok_or(INVALID_ADDRESS);
    }
    let guests = GUESTS.lock();
    let range = guests
        .iter()
        .find(|range| {
            address >= range.host
                && address
                    .checked_add(PAGE)
                    .is_some_and(|end| end <= range.host + range.length)
        })
        .ok_or(INVALID_ADDRESS)?;
    Ok((
        Arc::clone(&range.section),
        range.offset + address - range.host,
        3,
    ))
}

/// The Windows implementation of core's HostVm contract.
#[derive(Debug, Default)]
pub struct WindowsVm;

impl WindowsVm {
    /// All instances resolve the same process-wide mappings.
    pub fn new() -> Self {
        Self
    }

    /// Registered RAM has a stable read/write lifetime; query changing aliases and
    /// other CPU pages through Windows so holes and protection changes remain visible.
    pub fn readable(&self, address: HostAddr, length: u64) -> bool {
        let Some(end) = (address.addr() as u64).checked_add(length) else {
            return false;
        };
        if registered_ram_covers(address.addr(), end as usize) {
            return true;
        }
        super::mapping::readable(address.addr() as u64, end)
    }
}

impl HostVm for WindowsVm {
    fn page_size(&self) -> u64 {
        PAGE as u64
    }
    fn page_shift(&self) -> u32 {
        PAGE.trailing_zeros()
    }

    fn reserve_noaccess(&self, length: u64) -> Result<HostAddr, KernReturn> {
        let (_, length) = length_pages(HostAddr::NULL, length)?;
        if length == 0 {
            return Ok(HostAddr::NULL);
        }
        let reservation = Reservation::new(length).map_err(refusal)?;
        let base = reservation.address();
        RESERVATIONS.lock().insert(base, reservation);
        Ok(HostAddr::from_addr(base))
    }

    fn allocate_anywhere(&self, length: u64) -> Result<HostAddr, KernReturn> {
        let (_, length) = length_pages(HostAddr::NULL, length)?;
        if length == 0 {
            return Ok(HostAddr::NULL);
        }
        let arena = PageArena::new(length).map_err(refusal)?;
        let base = arena.address();
        let maximum = vec![3; length / PAGE];
        ALLOCATIONS
            .lock()
            .insert(base, Allocation { arena, maximum });
        Ok(HostAddr::from_addr(base))
    }

    fn allocate_fixed_overwrite(&self, address: HostAddr, length: u64) -> Result<(), KernReturn> {
        let (first, length) = length_pages(address, length)?;
        if length == 0 {
            return Ok(());
        }
        let section = Section::new(length).map_err(refusal)?;
        let mut table = ALLOCATIONS.lock();
        let mut reservations = RESERVATIONS.lock();
        let sources = (0..length)
            .step_by(PAGE)
            .map(|offset| (Arc::clone(&section), offset, 3))
            .collect();
        overwrite(&mut table, &mut reservations, first, sources)
    }

    fn remap_shared_overwrite(
        &self,
        destination: HostAddr,
        source: HostAddr,
        length: u64,
    ) -> Result<HostAddr, KernReturn> {
        let (dst, length) = length_pages(destination, length)?;
        let src = source.addr() / PAGE * PAGE;
        if length == 0 {
            return Err(INVALID_ARGUMENT);
        }
        let mut table = ALLOCATIONS.lock();
        dst.checked_add(length).ok_or(INVALID_ADDRESS)?;
        src.checked_add(length).ok_or(INVALID_ADDRESS)?;
        let mut reservations = RESERVATIONS.lock();
        let sources = (0..length)
            .step_by(PAGE)
            .map(|offset| resolve(&table, &reservations, src + offset))
            .collect::<Result<Vec<_>, _>>()?;
        overwrite(&mut table, &mut reservations, dst, sources)?;
        Ok(HostAddr::from_addr(dst))
    }

    fn protect_max(
        &self,
        address: HostAddr,
        length: u64,
        protection: VmProt,
    ) -> Result<(), KernReturn> {
        let (first, length) = touched_pages(address, length)?;
        if protection.raw() & !3 != 0 {
            return Err(INVALID_ARGUMENT);
        }
        let mut table = ALLOCATIONS.lock();
        let mut reservations = RESERVATIONS.lock();
        let pages = (0..length)
            .step_by(PAGE)
            .map(|offset| {
                let target = destination(&table, &reservations, first + offset)?;
                let maximum = match target {
                    Destination::Owned(base, page) => table[&base].maximum[page],
                    Destination::Reserved(base, offset) => {
                        reservations[&base].maximum(offset).unwrap_or_else(|| {
                            super::failure::abort(concat!(module_path!(), ":", line!()))
                        })
                    }
                };
                if protection.raw() & !maximum != 0 {
                    return Err(PROTECTION_FAILURE);
                }
                Ok((target, maximum))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (target, _) in pages {
            match target {
                Destination::Owned(base, index) => {
                    let allocation = table.get_mut(&base).unwrap_or_else(|| {
                        super::failure::abort(concat!(module_path!(), ":", line!()))
                    });
                    allocation
                        .arena
                        .protect_access(index, protection.raw())
                        .map_err(refusal)?;
                    allocation.maximum[index] = protection.raw();
                }
                Destination::Reserved(base, offset) => reservations
                    .get_mut(&base)
                    .unwrap_or_else(|| super::failure::abort(concat!(module_path!(), ":", line!())))
                    .protect(offset, protection.raw())
                    .map_err(refusal)?,
            }
        }
        Ok(())
    }

    fn deallocate(&self, address: HostAddr, length: u64) -> KernReturn {
        let (first, length) = match touched_pages(address, length) {
            Ok(span) => span,
            Err(e) => return e,
        };
        let mut table = ALLOCATIONS.lock();
        let end = first + length;
        // Walk allocations/spans, not every address of a mostly empty multi-gigabyte task.
        let bases: Vec<_> = table
            .range(..end)
            .filter(|(base, a)| **base + a.arena.length() > first)
            .map(|(&base, _)| base)
            .collect();
        for base in bases {
            let allocation = table
                .get_mut(&base)
                .unwrap_or_else(|| super::failure::abort(concat!(module_path!(), ":", line!())));
            let start = first.max(base) - base;
            let stop = end.min(base + allocation.arena.length()) - base;
            for offset in (start..stop).step_by(PAGE) {
                allocation.arena.release(offset / PAGE);
            }
            if allocation.arena.is_empty() {
                table.remove(&base);
            }
        }
        let mut reservations = RESERVATIONS.lock();
        let bases: Vec<_> = reservations
            .range(..end)
            .filter(|(base, r)| **base + r.length() > first)
            .map(|(&base, _)| base)
            .collect();
        for base in bases {
            let reservation = reservations
                .get_mut(&base)
                .unwrap_or_else(|| super::failure::abort(concat!(module_path!(), ":", line!())));
            if let Err(error) = reservation.release(
                first.max(base) - base,
                end.min(base + reservation.length()) - base,
            ) {
                return refusal(error);
            }
            if reservation.is_empty() {
                reservations.remove(&base);
            }
        }
        KernReturn::SUCCESS
    }
}

/// Zeroed, aliasable storage used by IOSurface and detach allocations.
pub(crate) fn allocate_aliasable(length: u64) -> Result<HostAddr, KernReturn> {
    WindowsVm::new().allocate_anywhere(length)
}

/// A writable GPU-import view of readable pages, retaining their exact section offsets.
/// EFI and read-only task ranges keep their original CPU protection. As on Linux,
/// the GPU sees a separate writable view of the same backing, never a byte snapshot.
/// The caller records that writes through this view are kept, a Metal semantic difference.
pub(super) fn writable_import_alias(
    address: HostAddr,
    length: u64,
) -> Result<Option<OwnMemory>, KernReturn> {
    let (first, length) = touched_pages(address, length)?;
    if length == 0
        || registered_ram_covers(first, first + length)
        || super::mapping::writable(first as u64, (first + length) as u64)
        || !super::mapping::readable(first as u64, (first + length) as u64)
    {
        return Ok(None);
    }
    let sources = {
        let table = ALLOCATIONS.lock();
        let reservations = RESERVATIONS.lock();
        (0..length)
            .step_by(PAGE)
            .map(|offset| {
                let (section, section_offset, _) = resolve(&table, &reservations, first + offset)?;
                Ok((section, section_offset, 3))
            })
            .collect::<Result<Vec<_>, KernReturn>>()?
    };
    let alias = OwnMemory {
        first: WindowsVm::new().allocate_anywhere(length as u64)?,
        length: length as u64,
    };
    {
        let mut table = ALLOCATIONS.lock();
        let mut reservations = RESERVATIONS.lock();
        overwrite(&mut table, &mut reservations, alias.first.addr(), sources)?;
    }
    Ok(Some(alias))
}

/// IOSurface's allocation owner; mapped bytes live until its last reference/import is released.
#[derive(Debug)]
pub(crate) struct OwnMemory {
    first: HostAddr,
    length: u64,
}

impl OwnMemory {
    pub(crate) fn new(length: u64) -> Option<Self> {
        Some(Self {
            first: allocate_aliasable(length).ok()?,
            length,
        })
    }
    pub(crate) fn first(&self) -> u64 {
        self.first.addr() as u64
    }
}

impl Drop for OwnMemory {
    fn drop(&mut self) {
        assert!(
            WindowsVm::new()
                .deallocate(self.first, self.length)
                .is_success()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reproduce gather_short's repeated 4 KiB probes within QEMU's one large RAM
    /// view. VirtualQuery scans the remaining view even for a one-page request;
    /// a live guest registration must make this cost independent of total RAM.
    #[test]
    fn registered_ram_page_checks_do_not_rescan_large_view() {
        use std::ffi::c_void;
        use std::os::windows::io::AsRawHandle;
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn DeviceIoControl(
                h: *mut c_void,
                code: u32,
                input: *const c_void,
                input_len: u32,
                output: *mut c_void,
                output_len: u32,
                returned: *mut u32,
                overlapped: *mut c_void,
            ) -> i32;
            fn CreateFileMappingW(
                file: *mut c_void,
                attrs: *const c_void,
                access: u32,
                hi: u32,
                lo: u32,
                name: *const u16,
            ) -> *mut c_void;
            fn MapViewOfFile(
                section: *mut c_void,
                access: u32,
                hi: u32,
                lo: u32,
                length: usize,
            ) -> *mut c_void;
            fn UnmapViewOfFile(base: *mut c_void) -> i32;
            fn CloseHandle(handle: *mut c_void) -> i32;
        }
        let path =
            std::env::temp_dir().join(format!("reims-query-cost-{}.tmp", std::process::id()));
        let file = std::fs::File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let length = 8usize << 30;
        let mut returned = 0;
        // SAFETY: the fixture exclusively owns this file and these valid output pointers.
        // Mark it sparse before extending it: the test consumes address space, not 8 GiB of disk.
        assert_ne!(
            unsafe {
                DeviceIoControl(
                    file.as_raw_handle(),
                    0x900c4,
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    0,
                    &mut returned,
                    std::ptr::null_mut(),
                )
            },
            0
        );
        file.set_len(length as u64).unwrap();
        // SAFETY: this is the same read/write file-view arrangement used by QEMU RAM.
        let section = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                4,
                0,
                0,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        let pointer = unsafe { MapViewOfFile(section, 6, 0, 0, length) };
        assert!(!pointer.is_null());
        let lease = register_guest(pointer as usize, length, &file, 0).unwrap();
        let vm = WindowsVm::new();
        let queries_before = super::super::mapping::QUERY_COUNT.with(|count| count.get());
        let start = std::time::Instant::now();
        for page in 0..64 {
            let at = HostAddr::from_addr(pointer as usize + (512 << 20) + page * PAGE);
            assert!(vm.readable(at, PAGE as u64));
        }
        let elapsed = start.elapsed();
        let queries = super::super::mapping::QUERY_COUNT.with(|count| count.get()) - queries_before;
        unregister_guest(lease);
        // Removing the last lease immediately restores real kernel permission checks.
        let after_lease = super::super::mapping::QUERY_COUNT.with(|count| count.get());
        assert!(vm.readable(HostAddr::from_addr(pointer as usize), PAGE as u64));
        assert_eq!(
            super::super::mapping::QUERY_COUNT.with(|count| count.get()),
            after_lease + 1
        );
        // SAFETY: no aliases or accesses remain; release the fixture before asserting timing.
        assert_ne!(unsafe { UnmapViewOfFile(pointer) }, 0);
        assert_ne!(unsafe { CloseHandle(section) }, 0);
        drop(file);
        std::fs::remove_file(path).unwrap();
        eprintln!("64 registered RAM page probes: {elapsed:?}");
        // An untouched sparse view may query quickly, unlike QEMU's populated RAM.
        // Counting the actual OS boundary makes the regression deterministic on both.
        assert_eq!(
            queries, 0,
            "registered RAM page checks entered VirtualQuery"
        );
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "page checks rescanned the multi-gigabyte view: {elapsed:?}"
        );
    }

    /// The original EFI view stays read-only while updates through its RAM backing
    /// immediately reach the import; the import also survives the original views.
    #[test]
    fn readonly_import_is_live_without_elevating_original_permissions() {
        let vm = WindowsVm::new();
        let size = (PAGE * 2) as u64;
        let ram = vm.allocate_anywhere(size).unwrap();
        assert!(writable_import_alias(ram, size).unwrap().is_none());
        let readonly = vm.allocate_anywhere(size).unwrap();
        vm.remap_shared_overwrite(readonly, ram, size).unwrap();
        vm.protect_max(readonly, size, VmProt::READ).unwrap();
        let alias = writable_import_alias(readonly.wrapping_add(23), size - 23)
            .unwrap()
            .unwrap();
        assert!(!super::super::mapping::writable(
            readonly.addr() as u64,
            readonly.addr() as u64 + size
        ));
        // SAFETY: these are the test's live shared views, with no concurrent access.
        unsafe {
            ram.wrapping_add(PAGE as u64)
                .as_ptr::<u8>()
                .write_volatile(0x5a);
            assert_eq!(
                ((alias.first() as usize + PAGE) as *const u8).read_volatile(),
                0x5a
            );
            (alias.first() as *mut u8).write_volatile(0x72);
            assert_eq!(readonly.as_ptr::<u8>().read_volatile(), 0x72);
        }
        assert!(vm.deallocate(readonly, size).is_success());
        assert!(vm.deallocate(ram, size).is_success());
        // SAFETY: OwnMemory and its section views retain the backing until alias drops.
        assert_eq!(
            unsafe { (alias.first() as *const u8).read_volatile() },
            0x72
        );
    }

    /// Exercise the real task lifecycle without committing its 64 GiB of unused addresses.
    #[test]
    fn sparse_task_remap_reset_and_teardown_follow_host_vm() {
        let vm = WindowsVm::new();
        let length = 64 * 1024 * 1024 * 1024u64;
        let task = vm.reserve_noaccess(length).unwrap();
        assert!(!vm.readable(task, 1));
        let source = vm.allocate_anywhere(PAGE as u64).unwrap();
        let last = task.wrapping_add(length - PAGE as u64);
        vm.remap_shared_overwrite(last, source, PAGE as u64)
            .unwrap();
        // SAFETY: source and last are live aliases exclusively accessed by this test.
        unsafe {
            source.as_ptr::<u8>().write_volatile(0x6e);
            assert_eq!(last.as_ptr::<u8>().read_volatile(), 0x6e);
        }
        vm.allocate_fixed_overwrite(last, PAGE as u64).unwrap();
        // SAFETY: overwrite created fresh writable zero-filled backing for this one page.
        assert_eq!(unsafe { last.as_ptr::<u8>().read_volatile() }, 0);
        vm.protect_max(last, PAGE as u64, VmProt::NONE).unwrap();
        assert!(!vm.readable(last, 1));
        assert!(vm.deallocate(task, length).is_success());
        assert!(vm.deallocate(source, PAGE as u64).is_success());
    }

    /// Guest file-backed RAM and a GPU alias must observe the very same page in both directions.
    #[test]
    fn guest_file_alias_survives_registration_and_original_mapping() {
        let path = std::env::temp_dir().join(format!("reims-guest-ram-{}.tmp", std::process::id()));
        let file = std::fs::File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.set_len((4 * PAGE) as u64).unwrap();
        let mut guest = PageArena::new(PAGE).unwrap();
        guest
            .replace(0, Section::of_file(&file).unwrap(), PAGE)
            .unwrap();
        let lease = register_guest(guest.address(), PAGE, &file, PAGE).unwrap();
        let vm = WindowsVm::new();
        let alias = vm.allocate_anywhere(PAGE as u64).unwrap();
        vm.remap_shared_overwrite(alias, HostAddr::from_addr(guest.address()), PAGE as u64)
            .unwrap();
        // SAFETY: this test owns both live mappings and is their only accessor.
        unsafe {
            (guest.address() as *mut u8).write_volatile(0xa1);
            assert_eq!(alias.as_ptr::<u8>().read_volatile(), 0xa1);
            alias.as_ptr::<u8>().write_volatile(0x5c);
            assert_eq!((guest.address() as *const u8).read_volatile(), 0x5c);
        }
        unregister_guest(lease);
        drop(guest);
        drop(file);
        // SAFETY: the alias retains the file section after the host registration is gone.
        assert_eq!(unsafe { alias.as_ptr::<u8>().read_volatile() }, 0x5c);
        assert!(vm.deallocate(alias, PAGE as u64).is_success());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn readable_tracks_guard_and_deallocated_pages() {
        let vm = WindowsVm::new();
        let a = vm.allocate_anywhere(2 * PAGE as u64).unwrap();
        assert!(vm.readable(a, 2 * PAGE as u64));
        vm.protect_max(a.wrapping_add(PAGE as u64), PAGE as u64, VmProt::NONE)
            .unwrap();
        assert!(vm.readable(a, PAGE as u64));
        assert!(!vm.readable(a, 2 * PAGE as u64));
        assert!(vm.deallocate(a, 2 * PAGE as u64).is_success());
        assert!(!vm.readable(a, 1));
    }

    /// Exercise core's real HostVm calls, including alias lifetime and a hole beside a live page.
    #[test]
    fn shared_alias_survives_partial_source_deallocation() {
        let vm = WindowsVm::new();
        let source = vm.allocate_anywhere(3 * PAGE as u64).unwrap();
        let alias = vm.allocate_anywhere(PAGE as u64).unwrap();
        vm.remap_shared_overwrite(alias, source.wrapping_add(PAGE as u64), PAGE as u64)
            .unwrap();
        // SAFETY: these live mappings are exclusively used by this test.
        unsafe { alias.as_ptr::<u8>().write_volatile(0x73) };
        assert!(
            vm.deallocate(source.wrapping_add(PAGE as u64), PAGE as u64)
                .is_success()
        );
        // SAFETY: the alias owns the same section independently of the removed source page.
        assert_eq!(unsafe { alias.as_ptr::<u8>().read_volatile() }, 0x73);
        assert_eq!(
            vm.remap_shared_overwrite(alias, source.wrapping_add(PAGE as u64), PAGE as u64),
            Err(INVALID_ADDRESS)
        );
        assert!(vm.deallocate(source, 3 * PAGE as u64).is_success());
        assert!(vm.deallocate(alias, PAGE as u64).is_success());
    }

    /// Mach's length rounding differs from protect's touched pages; maximum protection is final.
    #[test]
    fn length_rounding_and_maximum_protection_match_the_contract() {
        let vm = WindowsVm::new();
        let source = vm.allocate_anywhere(3 * PAGE as u64).unwrap();
        let target = vm.allocate_anywhere(3 * PAGE as u64).unwrap();
        let late = (PAGE - 64) as u64;
        vm.remap_shared_overwrite(
            target.wrapping_add(late),
            source.wrapping_add(late),
            PAGE as u64,
        )
        .unwrap();
        vm.protect_max(target.wrapping_add(late), 128, VmProt::READ)
            .unwrap();
        assert_eq!(
            vm.protect_max(target, PAGE as u64, VmProt::READ | VmProt::WRITE),
            Err(PROTECTION_FAILURE)
        );
        vm.protect_max(
            target.wrapping_add(2 * PAGE as u64),
            PAGE as u64,
            VmProt::READ | VmProt::WRITE,
        )
        .unwrap();
        assert!(vm.deallocate(source, 3 * PAGE as u64).is_success());
        assert!(vm.deallocate(target, 3 * PAGE as u64).is_success());
    }
}
