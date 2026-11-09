//! `HostVm` on Linux: Mach VM's calls, as core makes them, over `mmap`.
//!
//! Linux cannot remap an address range's pages to another address as `mach_vm_remap` does, so
//! a view of guest pages ("virtual memory layouts") maps their files instead: the address space
//! is reserved first (core's `allocate_anywhere`), then each run of the source's bytes is mapped
//! over it with `mmap(MAP_SHARED | MAP_FIXED)` from the file and offset the backing table names
//! for it ([`super::files`]), and the view is recorded there as runs of its own. A view of a view
//! maps the file too, so it keeps showing the pages after the first view goes, as Mach's aliases
//! do; runs whose file offsets follow on are one VMA to the kernel.
//!
//! - **`allocate_anywhere`** is private anonymous memory (`MAP_NORESERVE`), zero-filled and in no
//!   file, so no view can be made of it: it serves core's reservations, the detach buffer's
//!   temporary and the address space views are mapped over. A range a view must be made of comes
//!   from [`allocate_aliasable`], a shared mapping of a file of its own: the detach heap's
//!   temporary, which Metal's heap is made over, is one (core's `HostVm::allocate_aliasable`),
//!   and so is an alias whose zeros past the guest's pages the runtime imports with them
//!   (`crate::gpu::backing::gather`), as a dma-buf takes file pages alone.
//! - **`remap_shared_overwrite`** fails with `KERN_INVALID_ADDRESS` where a byte of the source
//!   has no file behind it, as Mach fails for an unmapped source, before it maps anything. A
//!   piece the kernel refuses fails the call with nothing changed, as Mach's remap is all or
//!   nothing: the pieces mapped before it are put back as the target was ([`put_back`]).
//! - **`protect_max`** is `mprotect`: Linux has no maximum protection, and the current one
//!   stands in for it, so a lowered protection can be raised again, which Mach refuses. A
//!   forbidden write is a `SIGSEGV`, where Mach's is a `SIGBUS`: a death either way.
//! - **Whole pages**, of the host's size (`sysconf(_SC_PAGESIZE)`), as Mach takes them (measured
//!   on the M3 Max, macOS 27.0.1, 16 KiB pages; `crate::os`'s
//!   `the_vm_spans_its_calls_as_mach_does` checks it there): `protect_max` and `deallocate` take
//!   every page the range's bytes touch, from the page of its first byte to the page past its
//!   last; `allocate_fixed_overwrite` and `remap_shared_overwrite` take the length rounded up to
//!   whole pages from the page of the address, the remap's source from the page of its first
//!   byte too ([`length_pages`]), so a range that starts into a page and runs into the next takes
//!   one page fewer than its bytes touch, and the remap answers the target's page. A call of no
//!   bytes changes nothing, wherever it starts: the remap's is `KERN_INVALID_ARGUMENT`, the
//!   others' success.
//!
//! **Mappings.** A run of a view mapped from its file is a VMA, so such a view of scattered pages
//! costs one per run. The views share a budget short of `vm.max_map_count` ([`super::budget`]): a
//! remap, a range a view can be made of or a stand-in's memory that would pass it is refused
//! before anything is mapped, so the rest of the process keeps its mappings: a remap or a range
//! answers `KERN_NO_SPACE` and a stand-in no memory, as Mach's and IOSurface's calls answer when
//! memory runs out, which core answers as Metal does then (`crate::answer::out_of_host_memory`,
//! counted; in strict mode a death).
//!
//! A view of many runs is resolved in one pass of the backing table and recorded over its fresh
//! reservation with no run to cut (`files::view_pieces`, `files::record_pieces`), so the kernel's
//! maps are most of what it costs. **Measured** 2026-11-08 on homelab-um870-1 (Ryzen 7 8745H, Linux
//! 7.3.1, release build): a view of 7,261 one-page runs of a registered memfd, three in four the
//! page below the one before (an IOSurface's, as an x86-64 macOS guest hands its pages out), took
//! 1.86 µs a run beside 66,000 one-page views, and 2.06 µs beside 200,000, against 2.06 and 2.29 µs
//! with each run resolved and recorded on its own. On homelab-ms01-1 (Core i9-13900H, Linux 7.3.1),
//! beside 66,000, it took 1.21 µs a run on a performance core and 2.11 µs on an efficiency core,
//! against 1.50 and 2.58 µs a run at a time: that processor gives the kernel no capacity to place a
//! thread by (`hints`), so a view is mapped on either. A C loop of the same maps took 2.0 to 2.2 µs
//! a run on homelab-um790-xtx-1 (Ryzen 7 7840HS, Linux 7.3.1) beside 66,000 one-page mappings of
//! the file, and 1.09 to 1.26 µs on homelab-ms01-1, each `MAP_FIXED` cutting the reservation it
//! lands in (`vms_gather_munmap_vmas` and `__split_vma` in its profile); `MAP_FIXED_NOREPLACE` into
//! a hole took 1.6 to 1.7 µs and 0.90 to 1.01 µs, but a hole is no reservation: another thread's
//! `mmap` may land in it while the runs go in.
//!
//! **Views through dma-bufs.** An x86-64 macOS guest hands out its pages in 4 KiB pieces, each
//! 16 KiB clump of them backwards (its free queue refills each CPU's list last in, first out), and
//! a mapping of a file can only run forwards through it, so a view of a guest's buffer mapped from
//! the file takes a VMA a page. Measured on the RTX 5080 Laptop host with a 16 GiB macOS 26.6.1
//! guest at 3840 × 2160 HiDPI (2026-10-27): of the views' VMAs, 99.97% were single 4 KiB pages and
//! 75% of neighbours the page before in the file, none the page after; the views held up to 1.03
//! million VMAs, about 4 GiB of the guest's pages, each page in about one view, within 190 s of a
//! drag, a resize and Safari's scroll, and passed the budget of a `vm.max_map_count` of 1,048,576.
//! So a view of more than one run of file pages made at once (`remap_runs_shared_overwrite`, which
//! core's memory map, the surface stand-in and the platform's aliases make their views with) is
//! mapped as dma-bufs of those pages instead, made by udmabuf (`UDMABUF_CREATE_LIST`, in any order)
//! within its limits, one after another, each one VMA ([`super::dma_buf::of_view`]): at udmabuf's
//! stock `list_limit` of 1,024 a view of scattered pages takes a VMA per 4 MiB. Such a mapping
//! shows the same pages as the file's, both ways (measured on Linux 7.2.8: a dma-buf of 1,024
//! pages backwards, made in 0.1 ms and mapped in 5 µs, read and written through either), and holds
//! them while it lives, as an import of them does. The backing table still names each run's file,
//! which a view of the view and a dma-buf of its pages are made of, and the dma-buf it lies in
//! ([`super::files`]). The kernel maps a dma-buf as `VM_PFNMAP`, which nothing can pin, so a view
//! is mapped from its files where udmabuf is not the process's to open, where it refuses a
//! piece's file (one not a memfd sealed against shrinking), and for a run alone.
//!
//! **Imports of views.** A device given a view's address pins its pages as they are mapped (a
//! runtime import by pointer, NVIDIA's `cuMemHostRegister`), and none pins a `VM_PFNMAP` mapping:
//! NVIDIA's 615 driver refused a pointer into a udmabuf's mapping, and the dma-buf itself, on every
//! memory type (measured 2026-10-27, RTX 5080 Laptop, Linux 7.2.8). A device that imports every
//! page with a file behind it by a dma-buf of the pages (RADV, `super::import`'s module docs)
//! pins none. So where a device of the process refuses such a view (`super::import`'s
//! `learn_view_imports`, as the device is made), views are mapped from their files from then on,
//! a mapping a run, and an import of a view made through dma-bufs before maps what of its pages
//! lies in them from their files for as long as the view lives ([`unfold`]). Where those
//! mappings would come within the headroom of the views' budget (`super::budget`'s `claim`), a
//! view is mapped through dma-bufs there too, and no view ends the process. A no-copy buffer's or
//! heap's import of pages left in dma-bufs for want of mappings, which the runtime asks the
//! views' gate about, maps them from their files only short of that headroom, the views no import
//! holds folded back first where that may free some ([`fold_short_due`]); past it the gate
//! refuses them, and the runtime makes the buffer or heap over a copy of them, which takes no
//! mapping (`super::import`'s `import_gate`); a rebind's alike, which the runtime then makes over
//! a copy too.
//!
//! **Folding back.** At the views' budget, before anything is refused, the views' runs mapped
//! from their files that no import a device may pin holds (`super::import`'s `Pinned`) are mapped
//! through dma-bufs of their pages instead, a mapping per 4 MiB ([`fold_idle`]), where udmabuf is
//! the process's. The same pages are mapped over them, so nothing reading or writing them sees a
//! change; every change the library makes to its views' mappings (a view, an unfold, a protect,
//! an overwrite, an unmap) holds the views' lock to read ([`views`]), and a fold holds it to
//! write, so none of them runs while a fold maps, and a fold never maps over a range another
//! changes.
//!
//! An import's pages are not mapped back through dma-bufs once the import goes: the guest's
//! surfaces are imported again and again, the display's 32 MiB at 3840 × 2160 a frame, held about
//! 30 ms of each 60 (RTX 5080 Laptop, 2026-10-27), and mapping a view's pages from their files
//! took 1.2 µs a page (Linux 7.2.8, 1,024 pages backwards in 1.23 ms, back through a dma-buf in
//! 0.44 ms), 13 ms of the process's map lock a frame for that surface. Where a device imports
//! such a view, it is never mapped from its files.
//!
//! A map the kernel refuses for want of mappings, the process at `vm.max_map_count` (its other
//! mappings past the reserve), first folds the views no import holds back into dma-bufs and runs
//! the call again, the first try having changed nothing ([`reclaiming`]); refused again, it
//! answers `KERN_NO_SPACE`, as a refusal for want of memory does, counted as the host's memory
//! running out and naming how many mappings the process has and the limit ([`no_space`]; in
//! strict mode a death). One refused for want of memory or address space is `KERN_NO_SPACE`, as
//! Mach's is. A file the kernel gives none of for the library's memory ([`memfd`]) is refused as
//! memory is, counted alike, naming the open-file limit.

use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::sync::Arc;

use crate::core::num::{ToU64, ToUsize};
use crate::core::platform::os::vm::{HostAddr, HostVm, KernReturn, VmProt};
use crate::core::sync::{ReadGuard, WriterPrefRwLock};

use super::budget;
use super::files::{self, Backing, Through};

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

/// The most bytes `deallocate` forgets and unmaps at once, a multiple of every page size. The
/// kernel holds the process's lock on its mappings across each `munmap`, which every thread that
/// maps or faults in a page waits on, and a view of the guest's scattered pages is a mapping a
/// run of them. One `munmap` of a 1 GiB IOSurface's view, of 254,000 to 262,000 runs, took 0.66
/// to 0.99 s, 2.5 to 3.8 us a mapping. A vCPU waited through it in an `mmap`, and QEMU's main
/// loop waited for QEMU's lock, which that vCPU held, up to 0.89 s (homelab-gem10, Linux 7.2.9,
/// 2026-10-24). A piece of 1 MiB is at most 256 mappings of 4 KiB pages, so the lock is let go
/// about every millisecond, and the backing table's, which `forget` holds, as often.
const DEALLOCATE_PIECE: u64 = 1 << 20;

/// The views' lock (module docs, "Folding back"): read by every change to the views' mappings,
/// written by a fold ([`fold_idle`]).
static VIEWS: WriterPrefRwLock = WriterPrefRwLock::new(());

thread_local! {
    /// How many reads of [`VIEWS`] the thread holds: a fold is never made while it holds one,
    /// which the fold's write would wait for.
    static IN_VIEWS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// A read of the views' lock ([`views`]).
struct Views {
    _read: ReadGuard<'static>,
}

impl Drop for Views {
    fn drop(&mut self) {
        IN_VIEWS.with(|n| n.set(n.get() - 1));
    }
}

/// The views' lock read, for a change to the views' mappings, until the answer drops.
fn views() -> Views {
    IN_VIEWS.with(|n| n.set(n.get() + 1));
    Views {
        _read: VIEWS.read(),
    }
}

/// Folds back into dma-bufs the views' runs mapped from their files that no import a device may
/// pin holds (module docs, "Folding back"): how many mappings it freed. Nothing where udmabuf is
/// not the process's, or where the thread holds the views' lock.
pub(super) fn fold_idle() -> u64 {
    fold_within(0..u64::MAX)
}

/// [`fold_idle`], of the runs in `within`.
fn fold_within(within: std::ops::Range<u64>) -> u64 {
    if IN_VIEWS.with(std::cell::Cell::get) > 0 || !super::dma_buf::opens() {
        return 0;
    }
    let _all = VIEWS.write();
    // A dma-buf's map the kernel refuses at its limit puts the runs back, and is answered.
    let outer = AT_LIMIT.with(|a| a.replace(Some(false)));
    let before = files::mappings();
    let pinned = super::import::pinned();
    for span in files::file_spans() {
        let span = span.start.max(within.start)..span.end.min(within.end);
        for idle in outside(span, &pinned) {
            fold(idle);
        }
    }
    AT_LIMIT.with(|a| a.set(outer));
    let freed = before.saturating_sub(files::mappings());
    if freed > 0 {
        log::info!(
            target: "reims_vgpu_vulkan::os",
            "at the views' budget (vm.max_map_count): {freed} of the views' {before} mappings \
             freed by mapping the pages no import holds through dma-bufs"
        );
    }
    freed
}

/// The views' mappings and the imports freed (`super::import`'s `unpinned`) as the last fold for
/// a claim short of the headroom ended ([`fold_short`]); `u64::MAX` before the first.
static SHORT_FOLDED: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(u64::MAX),
    std::sync::atomic::AtomicU64::new(u64::MAX),
];

/// Whether a claim short of the headroom the budget refused may fold the views no import holds
/// back into dma-bufs ([`fold_short`]): only where the views' mappings grew or an import was freed
/// since the last such fold ended, as only those give a fold more to free, and a fold reads every
/// run of the backing table, a million at a `vm.max_map_count` of 1,048,576, holding the views'
/// lock to write.
fn fold_short_due() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let [mappings, unpinned] = &SHORT_FOLDED;
    let last = mappings.load(Relaxed);
    last == u64::MAX
        || files::mappings() > last
        || super::import::unpinned() != unpinned.load(Relaxed)
}

/// [`fold_idle`], for a claim short of the headroom: the gated import of guest pages (a no-copy
/// buffer's or heap's, `super::import`'s `import_view`) folds the views no import holds before
/// the runtime copies it for want of mappings, as a copy costs each command buffer a compare of
/// its bytes, and a fold only a remap of the pages it folds.
fn fold_short() {
    use std::sync::atomic::Ordering::Relaxed;
    let unpinned = super::import::unpinned();
    fold_idle();
    SHORT_FOLDED[0].store(files::mappings(), Relaxed);
    SHORT_FOLDED[1].store(unpinned, Relaxed);
}

/// Claims `adding` mappings short of the headroom of the views' budget for a gated import's twin
/// (`super::import`'s `twin`), the views no import holds folded first where the budget refuses and
/// a fold may free something ([`fold_short_due`]).
pub(super) fn claim_short(adding: u64) -> Option<budget::Claim> {
    let reach = budget::Reach::ShortOfHeadroom;
    budget::claim(adding, reach).or_else(|| {
        fold_short_due().then(fold_short)?;
        budget::claim(adding, reach)
    })
}

thread_local! {
    /// Whether the thread's call is [`reclaiming`]'s first try: `Some`, set to `Some(true)` where
    /// the kernel refused a map at the process's limit of mappings ([`no_space`]); `None`
    /// outside one.
    static AT_LIMIT: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// A test's process at its limit of mappings, on the thread that sets it.
    static TEST_AT_LIMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `call`, which changes nothing where the kernel refuses one of its maps: where it refused one at
/// the process's limit of mappings, the views no import holds folded back into dma-bufs
/// ([`fold_idle`]) and `call` again, whose refusal at the limit then answers as the host's
/// memory running out ([`no_space`]; module docs). Inside the views' lock, which a fold waits
/// for, and inside another's first try, `call` once, the outer one trying again.
pub(super) fn reclaiming<T>(mut call: impl FnMut() -> T) -> T {
    let armed = AT_LIMIT.with(std::cell::Cell::get).is_some();
    if armed || IN_VIEWS.with(std::cell::Cell::get) > 0 {
        return call();
    }
    /// The first try's mark, taken away however the try ends.
    struct FirstTry;
    impl Drop for FirstTry {
        fn drop(&mut self) {
            AT_LIMIT.with(|a| a.set(None));
        }
    }
    let (answer, hit) = {
        AT_LIMIT.with(|a| a.set(Some(false)));
        let _first = FirstTry;
        (call(), AT_LIMIT.with(std::cell::Cell::get) == Some(true))
    };
    if !hit {
        return answer;
    }
    drop(answer);
    fold_idle();
    call()
}

/// The parts of `span` that none of `pinned` (in order, apart) meets.
fn outside(
    span: std::ops::Range<u64>,
    pinned: &[std::ops::Range<u64>],
) -> Vec<std::ops::Range<u64>> {
    let mut parts = Vec::new();
    let mut from = span.start;
    for p in pinned
        .iter()
        .filter(|p| p.start < span.end && span.start < p.end)
    {
        if from < p.start {
            parts.push(from..p.start);
        }
        from = from.max(p.end);
    }
    if from < span.end {
        parts.push(from..span.end);
    }
    parts
}

/// Maps the runs of `range`, mapped from their files, through dma-bufs of their pages, where that
/// takes fewer mappings ([`fold_idle`], with the views' lock written).
fn fold(range: std::ops::Range<u64>) {
    let runs = files::mappings_between(range.start, range.end);
    if runs < 3 {
        return;
    }
    let Some(pieces) = files::pieces(range.start, range.end - range.start) else {
        return;
    };
    let Ok(bufs) = super::dma_buf::of_view(&pieces) else {
        return;
    };
    // Two more where the fold cuts a mapping that runs on past either end.
    if bufs.len().to_u64() + 2 >= runs {
        return;
    }
    // A refused map leaves what was mapped before it, and is logged where it is.
    let _ = through(range.start, pieces, bufs, false);
}

/// The host's page size.
pub(super) fn page_size() -> u64 {
    // SAFETY: a sysconf query, which cannot fail for this name.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(size).unwrap_or(4096)
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
/// `remap_shared_overwrite`; module docs): the first byte of that page, and the length; `None`
/// where they pass the end of the address space.
fn length_pages(addr: HostAddr, len: u64) -> Option<(u64, u64)> {
    let mask = page_size() - 1;
    let first = addr.addr().to_u64() & !mask;
    let length = len.checked_add(mask)? & !mask;
    first.checked_add(length)?;
    Some((first, length))
}

/// A Mach answer for the `errno` of a refused call, which `what` names: `ENOMEM` is
/// `KERN_NO_SPACE` for a map ([`no_space`]), `KERN_INVALID_ADDRESS` for a range not mapped.
fn refused(what: &str, unmapped_is_enomem: bool) -> KernReturn {
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    match errno {
        libc::ENOMEM if unmapped_is_enomem => INVALID_ADDRESS,
        libc::ENOMEM => no_space(what),
        libc::EINVAL => INVALID_ARGUMENT,
        libc::EACCES | libc::EPERM => PROTECTION_FAILURE,
        _ => FAILURE,
    }
}

/// `KERN_NO_SPACE` for a map the kernel refused for want of memory or address space, logged with
/// the process's mappings and their limit (`vm.max_map_count`). One it refused for want of
/// mappings, the process at the limit ([`at_the_limit`]), is tried again once the views are
/// folded where it is [`reclaiming`]'s first try; past that, it is the host's memory running out
/// (`crate::answer::out_of_host_memory`), counted, its line naming the numbers, and in strict mode
/// it ends the process there (module docs).
fn no_space(what: &str) -> KernReturn {
    let (mappings, limit) = map_census();
    #[cfg(test)]
    let at_limit = at_the_limit(&mappings, &limit) || TEST_AT_LIMIT.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let at_limit = at_the_limit(&mappings, &limit);
    if at_limit && AT_LIMIT.with(|a| a.replace(a.get().map(|_| true)).is_some()) {
        log::warn!(
            target: "reims_vgpu_vulkan::os",
            "{what}: the kernel refused the map at the process's limit of mappings, {mappings} of \
             vm.max_map_count {limit} held, {} of them the library's views of memory: folding the \
             views no import holds, to try again",
            budget::held()
        );
        return NO_SPACE;
    }
    if at_limit {
        crate::answer::out_of_host_memory("mmap", || {
            format!(
                "the process ran out of kernel mappings: the kernel refused {what} (ENOMEM) with \
                 {mappings} of vm.max_map_count {limit} held, {} of them the library's views of \
                 memory; macOS has no such limit, and `sysctl -w vm.max_map_count=` a higher \
                 value gives room",
                budget::held()
            )
        });
        return NO_SPACE;
    }
    log::error!(
        target: "reims_vgpu_vulkan::os",
        "{what}: the kernel refused the map (ENOMEM): the process has {mappings} mappings, \
         vm.max_map_count is {limit}"
    );
    NO_SPACE
}

/// Whether a process holding `mappings` mappings is at `vm.max_map_count` `limit`, each as
/// [`map_census`] read it: within the 2 a map may add, splitting a mapping in two. `false` where
/// either does not read.
fn at_the_limit(mappings: &str, limit: &str) -> bool {
    match (mappings.parse::<u64>(), limit.parse::<u64>()) {
        (Ok(mappings), Ok(limit)) => mappings.saturating_add(2) >= limit,
        _ => false,
    }
}

/// The process's mappings (`/proc/self/maps`' lines) and `vm.max_map_count`, each as read, or
/// `"?"` where it does not read.
pub(super) fn map_census() -> (String, String) {
    let mappings = std::fs::read_to_string("/proc/self/maps")
        .map_or_else(|_| "?".to_owned(), |m| m.lines().count().to_string());
    let limit = std::fs::read_to_string("/proc/sys/vm/max_map_count")
        .map_or_else(|_| "?".to_owned(), |l| l.trim().to_owned());
    (mappings, limit)
}

/// `VmProt` as `PROT_*`: the same bits.
fn prot_bits(prot: VmProt) -> i32 {
    prot.raw() & (libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC)
}

/// A pointer to `at` in the process's address space.
pub(super) fn ptr(at: u64) -> *mut libc::c_void {
    std::ptr::with_exposed_provenance_mut(at.to_usize())
}

/// Maps `length` bytes of anonymous zero-filled memory, read and write, at `at` (`MAP_FIXED`)
/// or anywhere: its first byte, or the `errno`'s answer.
pub(super) fn anonymous(at: Option<u64>, length: u64, what: &str) -> Result<u64, KernReturn> {
    let fixed = if at.is_some() { libc::MAP_FIXED } else { 0 };
    // SAFETY: a fresh private mapping; at a fixed address, replacing that range is the caller's
    // decision (core's reservation of its own).
    let mapped = unsafe {
        libc::mmap(
            ptr(at.unwrap_or(0)),
            length.to_usize(),
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | fixed,
            -1,
            0,
        )
    };
    if mapped == libc::MAP_FAILED {
        return Err(refused(what, false));
    }
    Ok(mapped.addr().to_u64())
}

/// Maps `length` bytes of `backing`'s file, shared, with protection `prot`, at `at`
/// (`MAP_FIXED`) or anywhere: its first byte, or the `errno`'s answer.
pub(super) fn file(
    at: Option<u64>,
    length: u64,
    backing: &Backing,
    prot: i32,
) -> Result<u64, KernReturn> {
    let fixed = if at.is_some() { libc::MAP_FIXED } else { 0 };
    let offset = libc::off_t::try_from(backing.offset).map_err(|_| INVALID_ARGUMENT)?;
    // SAFETY: a shared mapping of an open file; at a fixed address, replacing that range is the
    // caller's decision (a view over core's reservation).
    let mapped = unsafe {
        libc::mmap(
            ptr(at.unwrap_or(0)),
            length.to_usize(),
            prot,
            libc::MAP_SHARED | fixed,
            backing.fd.as_raw_fd(),
            offset,
        )
    };
    if mapped == libc::MAP_FAILED {
        return Err(refused("a view of guest pages", false));
    }
    Ok(mapped.addr().to_u64())
}

/// A fresh range of `len` bytes, rounded up to whole pages, of zeros, that a view can be made of:
/// a shared mapping of a file of its own, recorded in the backing table. Freed by `deallocate`,
/// as any other range. The block of zeros a detached object reads, and the tests' stand-in for
/// guest memory, are such ranges.
pub(crate) fn allocate_aliasable(len: u64) -> Result<HostAddr, KernReturn> {
    let Some((_, length)) = pages(HostAddr::NULL, len) else {
        return Err(NO_SPACE);
    };
    if length == 0 {
        return Ok(HostAddr::NULL);
    }
    reclaiming(|| {
        if !budget::admit(1, "a range a view can be made of") {
            return Err(NO_SPACE);
        }
        let fd = memfd(length).map_err(|_| NO_SPACE)?;
        let backing = Backing { fd, offset: 0 };
        let prot = libc::PROT_READ | libc::PROT_WRITE;
        let at = file(None, length, &backing, prot)?;
        files::record(at, length, backing, prot);
        Ok(HostAddr::from_addr(at.to_usize()))
    })
}

/// Memory of the library's own that a view can be made of and that only its holder frees: a
/// shared mapping of a new file, read and write, registered in the backing table
/// ([`files::Registration`]), so that `deallocate` refuses it. An IOSurface stand-in's memory.
#[derive(Debug)]
pub(crate) struct OwnMemory {
    first: u64,
    length: u64,
    registration: Option<files::Registration>,
}

impl OwnMemory {
    /// `len` bytes, rounded up to whole pages, of zeros; `None` for none, or where the host
    /// gives no file, no address space or no mapping within the views' budget, as an IOSurface's
    /// creation answers nil when memory runs out.
    pub(crate) fn new(len: u64) -> Option<OwnMemory> {
        let (_, length) = pages(HostAddr::NULL, len)?;
        if length == 0 {
            return None;
        }
        let (first, backing) = reclaiming(|| {
            if !budget::admit(1, "an IOSurface stand-in's memory") {
                return None;
            }
            let backing = Backing {
                fd: memfd(length).ok()?,
                offset: 0,
            };
            let first = file(None, length, &backing, libc::PROT_READ | libc::PROT_WRITE).ok()?;
            Some((first, backing))
        })?;
        budget::own_hold(1, true);
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
    /// Drops the registration, then the mapping, through the platform's VM: pages a device's
    /// import still holds stay mapped until the runtime has freed it, as Vulkan requires of
    /// imported host memory (`VK_EXT_external_memory_host`), and go as the import's hold does
    /// ([`crate::remap::UnmapHold`]). A region's last reference may go in a no-copy buffer's
    /// deallocator (the scanout's, `reims-vgpu`'s `graphics::sink`), which the runtime runs at the
    /// buffer's release, before it frees the buffer's import of the region's pages.
    fn drop(&mut self) {
        drop(self.registration.take());
        let at = HostAddr::from_addr(self.first.to_usize());
        let _ = crate::remap::VulkanVm::new().deallocate(at, self.length);
        budget::own_hold(1, false);
    }
}

/// A graphics device's region (BAR 2) in memory of the library's own ([`OwnMemory`]): a shared
/// mapping of a memfd, so a device that refuses the region's pointer imports dma-bufs of its pages
/// instead (`super::import`'s dma-buf source), as RADV refuses every page with a file behind it.
///
/// RADV imports host memory by pointer as an amdgpu userptr, and amdgpu walks every page of every
/// userptr import a device holds in each of its submissions, where a dma-buf import costs a
/// submission the same at any size (**measured** 2026-11-07, Radeon 780M, RADV of Mesa 26.2.6,
/// Linux 7.3.1, raw Vulkan, the median of 300 empty submissions: 6.9 us with none held; 256 MiB of
/// pointer imports in 4 pieces 90.7 us, in 128 pieces 171 us; 256 MiB of dma-buf imports in 4
/// pieces 12.2 us, in 128 pieces 66.8 us). The region's blocks are the guest's Shared buffers,
/// upload chunks and heaps, each imported as its buffer or heap is made.
#[derive(Debug)]
pub struct RegionMemory(OwnMemory);

impl RegionMemory {
    /// `len` bytes of zeros, rounded up to whole pages; `None` where the host gives no file, no
    /// address space or no mapping within the views' budget.
    pub fn new(len: u64) -> Option<RegionMemory> {
        OwnMemory::new(len).map(RegionMemory)
    }

    /// Its first byte.
    pub fn base(&self) -> *mut u8 {
        ptr(self.0.first()).cast()
    }
}

/// A new file of `length` bytes of zeros in memory (`memfd_create`), sealed against growing and
/// shrinking, as `/dev/udmabuf` takes only a memfd sealed against shrinking
/// ([`super::dma_buf`]). Where the kernel gives none, the kernel's error, the host's memory
/// running out ([`no_file`]): the caller answers as when memory runs out (module docs).
pub(crate) fn memfd(length: u64) -> Result<Arc<OwnedFd>, std::io::Error> {
    #[cfg(test)]
    if TEST_NO_FILE.with(std::cell::Cell::get) {
        let error = std::io::Error::from_raw_os_error(libc::EMFILE);
        return Err(no_file("memfd_create", length, error));
    }
    let name = c"reims-vgpu";
    // SAFETY: a NUL-terminated name.
    let raw =
        unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
    if raw < 0 {
        let error = std::io::Error::last_os_error();
        return Err(no_file("memfd_create", length, error));
    }
    // SAFETY: a descriptor just made, owned from here on.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    let Ok(size) = libc::off_t::try_from(length) else {
        let error = std::io::Error::from_raw_os_error(libc::EFBIG);
        return Err(no_file("ftruncate", length, error));
    };
    // SAFETY: an open memfd.
    if unsafe { libc::ftruncate(fd.as_raw_fd(), size) } != 0 {
        let error = std::io::Error::last_os_error();
        return Err(no_file("ftruncate", length, error));
    }
    let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK;
    // SAFETY: an open memfd made to take seals.
    if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals) } != 0 {
        let error = std::io::Error::last_os_error();
        return Err(no_file("fcntl(F_ADD_SEALS)", length, error));
    }
    Ok(Arc::new(fd))
}

#[cfg(test)]
thread_local! {
    /// A test's kernel that gives no file, on the thread that sets it ([`memfd`]).
    pub(crate) static TEST_NO_FILE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `error`, the kernel's refusal at `call` of a file of `length` bytes ([`memfd`]), the host's
/// memory running out (`crate::answer::out_of_host_memory`): counted, its line naming the error
/// and the open-file limit, and in strict mode the process ends at `call`.
fn no_file(call: &'static str, length: u64, error: std::io::Error) -> std::io::Error {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `limit` is a valid rlimit to fill.
    let read = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } == 0;
    let limit = if read {
        limit.rlim_cur.to_string()
    } else {
        "?".to_owned()
    };
    crate::answer::out_of_host_memory(call, || {
        format!(
            "the kernel gave no file of {length:#x} bytes for the library's memory ({error}); \
             the open-file limit is {limit}; IOSurface's and Mach's calls need no file on macOS"
        )
    });
    error
}

/// Maps `length` bytes from `at`, a piece of a remap that failed after mapping it, back to what
/// they showed before it, so the failed remap changes nothing: the files of the table's runs and
/// registrations there, at the protection each was set to (the remap records nothing until every
/// piece is mapped), and fresh zeros with no access over bytes of no file.
///
/// Bytes of no file in a remap's target are a reservation: a mapping task's, which has no access,
/// or a fresh range its caller frees after the failure (core's range mappings, an alias, a
/// surface's view), so the protection of those zeros never shows. No import of the runtime's
/// lies over them, as the runtime imports no page without read access and nothing imports a
/// fresh range before its remaps succeed, so putting zeros there is reported to no one
/// ([`crate::remap`]); the table's files are mapped back as they were, the same pages. A
/// put-back the kernel refuses too (`vm.max_map_count`) leaves the range forgotten by the table,
/// logged.
fn put_back(at: u64, length: u64) {
    let page = page_size();
    let has_file = |first: u64| files::pieces(first, page).is_some();
    let end = at + length;
    let mut from = at;
    while from < end {
        // The pages from `from` that all have a file, or all have none.
        let filed = has_file(from);
        let mut to = from + page;
        while to < end && has_file(to) == filed {
            to += page;
        }
        let restored = if filed {
            files::pieces(from, to - from).is_some_and(|pieces| {
                let mapped = (pieces.iter())
                    .all(|p| file(Some(from + p.at), p.length, &p.backing, p.prot).is_ok());
                // Mapped from their files now, though the table may have them through a dma-buf.
                if mapped {
                    files::record_pieces(from, pieces);
                }
                mapped
            })
        } else {
            anonymous(Some(from), to - from, "a failed remap's put-back").is_ok()
                // SAFETY: the zeros just mapped, a reservation of the remap's caller (above).
                && unsafe { libc::mprotect(ptr(from), (to - from).to_usize(), libc::PROT_NONE) } == 0
        };
        if !restored {
            log::error!(
                target: "reims_vgpu_vulkan::os",
                "a failed remap's put-back of {:#x} bytes at {from:#x} was refused: {}; the \
                 range is forgotten",
                to - from,
                std::io::Error::last_os_error()
            );
            files::forget(from, to - from);
        }
        from = to;
    }
}

/// The most mappings a view of `pieces` adds: one for each run of them the kernel holds as one
/// (each piece following the one before in the same file at the same protection), and one more
/// where the view cuts a run of its target in two.
pub(super) fn mappings_added(pieces: &[files::Piece]) -> u64 {
    let follows = |a: &files::Piece, b: &files::Piece| {
        Arc::ptr_eq(&a.backing.fd, &b.backing.fd)
            && a.backing.offset + a.length == b.backing.offset
            && a.prot == b.prot
    };
    let apart = pieces.windows(2).filter(|w| !follows(&w[0], &w[1])).count();
    apart.to_u64() + 2
}

/// Maps `pieces`, which follow on from `target`, each from its file, or, where they are more than
/// one, through dma-bufs of their pages where the host makes them: where no device of the process
/// needs imported pages mapped from their files, and where one does, once the pieces from their
/// files would come within the headroom of the views' budget ([`budget::claim`]; module docs,
/// "Views through dma-bufs" and "Imports of views"). Records them: all or nothing, the pieces
/// mapped before a refusal put back.
fn map_view(target: u64, pieces: Vec<files::Piece>) -> Result<(), KernReturn> {
    let adding = mappings_added(&pieces);
    // Held until the pieces are recorded, which counts them.
    let claim = super::import::imports_need_files()
        .then(|| budget::claim(adding, budget::Reach::ShortOfHeadroom))
        .flatten();
    if pieces.len() > 1 && claim.is_none() {
        match super::dma_buf::of_view(&pieces) {
            Ok(bufs) => return map_through(target, pieces, bufs),
            Err(refusal) => no_dma_bufs(&refusal),
        }
    }
    // Past the budget, `KERN_NO_SPACE`, nothing mapped, as Mach's remap answers when memory runs
    // out (module docs).
    if claim.is_none() && !budget::admit(adding, "a view of guest pages") {
        return Err(NO_SPACE);
    }
    let _views = views();
    for (done, piece) in pieces.iter().enumerate() {
        if let Err(kr) = file(
            Some(target + piece.at),
            piece.length,
            &piece.backing,
            piece.prot,
        ) {
            for mapped in pieces.iter().take(done) {
                put_back(target + mapped.at, mapped.length);
            }
            return Err(kr);
        }
    }
    files::record_pieces(target, pieces);
    Ok(())
}

/// Logs, once per kind, why a view of several runs is mapped from their files, a mapping a run,
/// rather than through dma-bufs of their pages.
fn no_dma_bufs(refusal: &super::dma_buf::Refusal) {
    use std::sync::atomic::{AtomicBool, Ordering};
    // A file open read-only is refused from its files alike.
    if matches!(refusal, super::dma_buf::Refusal::ReadOnlyFile) {
        return;
    }
    static DEVICE: AtomicBool = AtomicBool::new(false);
    static OTHER: AtomicBool = AtomicBool::new(false);
    let once = match refusal {
        super::dma_buf::Refusal::Device(_) => &DEVICE,
        _ => &OTHER,
    };
    if !once.swap(true, Ordering::Relaxed) {
        log::warn!(
            target: "reims_vgpu_vulkan::os",
            "views of scattered guest pages take a kernel mapping a run of pages, counted against \
             vm.max_map_count, as {refusal}"
        );
    }
}

/// Maps `bufs`, dma-bufs of the pages `pieces` show from `target` ([`super::dma_buf::of_view`]),
/// one after another, each a mapping of its own, and records the pieces as mapped through them:
/// all or nothing, those mapped before a refusal put back. The descriptors go once mapped, the
/// mappings holding the dma-bufs, weighed against the budget first.
fn map_through(
    target: u64,
    pieces: Vec<files::Piece>,
    bufs: Vec<super::dma_buf::ViewBuf>,
) -> Result<(), KernReturn> {
    if !budget::admit(bufs.len().to_u64() + 2, "a view of guest pages") {
        return Err(NO_SPACE);
    }
    let _views = views();
    through(target, pieces, bufs, true)
}

/// [`map_through`]'s maps and record, weighed and with the views' lock held by the caller: all
/// or nothing where `whole`; otherwise, a fold's, what the dma-bufs mapped before a refused one
/// show stays so, recorded, as it shows the same pages as before, and mapping their files back
/// could be refused too.
fn through(
    target: u64,
    mut pieces: Vec<files::Piece>,
    mut bufs: Vec<super::dma_buf::ViewBuf>,
    whole: bool,
) -> Result<(), KernReturn> {
    let failed = bufs.iter().position(|buf| {
        // SAFETY: a shared mapping of a dma-buf of the pages the view shows, over the caller's
        // range, which a view replaces (`remap_shared_overwrite`'s contract).
        let mapped = unsafe {
            libc::mmap(
                ptr(target + buf.at),
                buf.length.to_usize(),
                buf.prot,
                libc::MAP_SHARED | libc::MAP_FIXED,
                buf.fd.as_raw_fd(),
                0,
            )
        };
        mapped == libc::MAP_FAILED
    });
    let mut refusal = None;
    if let Some(done) = failed {
        let kr = refused("a view of guest pages", false);
        if whole {
            for mapped in bufs.iter().take(done) {
                put_back(target + mapped.at, mapped.length);
            }
            return Err(kr);
        }
        // The pieces before the refused dma-buf's first byte, which those mapped show.
        let end = bufs[done].at;
        pieces.retain(|p| p.at < end);
        if let Some(last) = pieces.last_mut() {
            last.length = last.length.min(end - last.at);
        }
        bufs.truncate(done);
        refusal = Some(kr);
    }
    // Each piece where it lies in the dma-bufs, cut where one ends.
    let mut through = Vec::with_capacity(pieces.len());
    let mut bufs = bufs
        .iter()
        .map(|b| (b.at, b.at + b.length, Through::new_buf()));
    let mut buf = bufs.next();
    for piece in pieces {
        let mut piece = piece;
        while let Some((first, end, id)) = buf {
            let take = piece.length.min(end - piece.at);
            let rest = files::Piece {
                at: piece.at + take,
                length: piece.length - take,
                backing: Backing {
                    fd: Arc::clone(&piece.backing.fd),
                    offset: piece.backing.offset + take,
                },
                prot: piece.prot,
            };
            let offset = piece.at - first;
            through.push((
                files::Piece {
                    length: take,
                    ..piece
                },
                Through { buf: id, offset },
            ));
            if piece.at + take == end {
                buf = bufs.next();
            }
            if rest.length == 0 {
                break;
            }
            piece = rest;
        }
    }
    files::record_through(target, through);
    refusal.map_or(Ok(()), Err)
}

/// Maps what of `first..end` is mapped through dma-bufs from its files instead, for an import of
/// those bytes by their pointer (module docs, "Imports of views"); refused maps are logged, and
/// leave those bytes as they were. Where the mappings would come within the headroom of the
/// views' budget ([`budget::claim`]), nothing is mapped, and the answer is how many they are: the
/// views' gate then refuses the import. One at a time, so two
/// imports of the same pages map them once.
pub(crate) fn unfold(first: u64, end: u64) -> Result<(), u64> {
    reclaiming(|| unfold_once(first, end))
}

/// [`unfold`], once: what a refused map leaves through dma-bufs a second try maps.
fn unfold_once(first: u64, end: u64) -> Result<(), u64> {
    static UNFOLDING: crate::core::sync::Mutex<()> = crate::core::sync::Mutex::new(());
    let _one = UNFOLDING.lock();
    let mut held = views();
    let mut pieces = files::through_pieces(first, end);
    if pieces.is_empty() {
        return Ok(());
    }
    let mut adding = mappings_added(&pieces);
    // Held until the pieces are recorded, which counts them. An import folds the views no import
    // holds first, its own pages among those held (module docs), where a fold may free something
    // since the last ([`fold_short`]).
    let reach = budget::Reach::ShortOfHeadroom;
    let mut claim = budget::claim(adding, reach);
    if claim.is_none() && fold_short_due() {
        drop(held);
        fold_short();
        held = views();
        pieces = files::through_pieces(first, end);
        if pieces.is_empty() {
            return Ok(());
        }
        adding = mappings_added(&pieces);
        claim = budget::claim(adding, reach);
    }
    let Some(_claim) = claim else {
        return Err(adding);
    };
    let mut mapped = Vec::with_capacity(pieces.len());
    for piece in pieces {
        if file(
            Some(first + piece.at),
            piece.length,
            &piece.backing,
            piece.prot,
        )
        .is_err()
        {
            log::error!(
                target: "reims_vgpu_vulkan::os",
                "an import's view of {:#x} bytes at {:#x} stays mapped through a dma-buf: the \
                 kernel refused its file's map: {}",
                piece.length,
                first + piece.at,
                std::io::Error::last_os_error()
            );
            break;
        }
        mapped.push(piece);
    }
    files::record_pieces(first, mapped);
    drop(held);
    Ok(())
}

/// `runs` remapped one at a time, as `HostVm::remap_runs_shared_overwrite` describes.
fn one_at_a_time(
    vm: &LinuxVm,
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

/// Mach VM's calls on Linux (module docs).
#[derive(Debug, Default)]
pub struct LinuxVm;

impl LinuxVm {
    /// The process's HostVm.
    pub fn new() -> LinuxVm {
        LinuxVm
    }

    /// Whether every page of the `len` bytes at `addr` is mapped with read access: by the
    /// backing table's protections where it knows every byte, else by reading a byte of each
    /// page through the kernel (`process_vm_readv`, which refuses a page unmapped or mapped
    /// without read access). A `len` of 0 is `true`.
    pub fn readable(&self, addr: HostAddr, len: u64) -> bool {
        if len == 0 {
            return true;
        }
        let Some((first, length)) = pages(addr, len) else {
            return false;
        };
        files::readable(first, length).unwrap_or_else(|| read_probe(first, length))
    }
}

/// Whether a byte of each page of `length` bytes from `first` reads, through the kernel.
fn read_probe(first: u64, length: u64) -> bool {
    let page = page_size();
    let count = (length / page).to_usize();
    let mut sink = vec![0u8; count.min(1024)];
    let mut at = first;
    let mut left = count;
    while left > 0 {
        let now = left.min(sink.len());
        let local = libc::iovec {
            iov_base: sink.as_mut_ptr().cast(),
            iov_len: now,
        };
        let remote: Vec<libc::iovec> = (0..now.to_u64())
            .map(|i| libc::iovec {
                iov_base: ptr(at + i * page),
                iov_len: 1,
            })
            .collect();
        let Ok(count) = libc::c_ulong::try_from(remote.len()) else {
            return false;
        };
        // SAFETY: the local buffer holds `now` bytes; the kernel reads the remote ones and
        // refuses those it cannot, writing only into the local buffer.
        let read =
            unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, remote.as_ptr(), count, 0) };
        if usize::try_from(read).ok() != Some(now) {
            return false;
        }
        at += now.to_u64() * page;
        left -= now;
    }
    true
}

impl HostVm for LinuxVm {
    fn page_size(&self) -> u64 {
        page_size()
    }

    fn page_shift(&self) -> u32 {
        page_size().trailing_zeros()
    }

    /// Anonymous memory anywhere; a length of 0 answers address 0, as Mach's does.
    fn allocate_anywhere(&self, len: u64) -> Result<HostAddr, KernReturn> {
        let Some((_, length)) = pages(HostAddr::NULL, len) else {
            return Err(NO_SPACE);
        };
        if length == 0 {
            return Ok(HostAddr::NULL);
        }
        let at = reclaiming(|| anonymous(None, length, "allocate_anywhere"))?;
        files::forget(at, length);
        Ok(HostAddr::from_addr(at.to_usize()))
    }

    /// Fresh anonymous memory over the pages of the range's length from the page of its address
    /// (module docs), which then show no file.
    fn allocate_fixed_overwrite(&self, addr: HostAddr, len: u64) -> Result<(), KernReturn> {
        let (first, length) = length_pages(addr, len).ok_or(NO_SPACE)?;
        if length == 0 {
            return Ok(());
        }
        reclaiming(|| {
            let _views = views();
            anonymous(Some(first), length, "allocate_fixed_overwrite")?;
            files::forget(first, length);
            Ok(())
        })
    }

    /// The source's pages of the length, from the page of its first byte, mapped from their files
    /// over as many from `dst`'s page, each with the protection the source has there: that
    /// page's first byte, which Mach writes back (module docs).
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
        reclaiming(|| {
            let Some(pieces) = files::pieces(source, length) else {
                log::debug!(
                    target: "reims_vgpu_vulkan::os",
                    "remap of {length:#x} bytes at {source:#x}: no file behind them"
                );
                return Err(INVALID_ADDRESS);
            };
            map_view(target, pieces)?;
            Ok(HostAddr::from_addr(target.to_usize()))
        })
    }

    /// The runs' pages mapped as one view, whole pages each ([`map_view`]): through dma-bufs of
    /// them where they are more than one piece of a file and the host makes those (module docs,
    /// "Views through dma-bufs"). A run with no file behind a byte is refused after the runs
    /// before it are mapped, as one remap at a time leaves them; a map the kernel refuses leaves
    /// none of them mapped, answered at the first run. Runs that are not whole pages from a page,
    /// or a view not from one, are remapped one at a time.
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
        reclaiming(|| {
            let target = dst.addr().to_u64();
            let sources: Vec<(u64, u64)> = (runs.iter())
                .map(|&(src, len)| (src.addr().to_u64(), len))
                .collect();
            match files::view_pieces(&sources) {
                Ok(all) => map_view(target, all).map_err(|kr| (0, kr)),
                Err((i, before)) => {
                    let (source, len) = sources[i];
                    log::debug!(
                        target: "reims_vgpu_vulkan::os",
                        "remap of {len:#x} bytes at {source:#x}: no file behind them"
                    );
                    if !before.is_empty() {
                        map_view(target, before).map_err(|kr| (0, kr))?;
                    }
                    Err((i, INVALID_ADDRESS))
                }
            }
        })
    }

    /// `mprotect` of the range's pages; none for no bytes (module docs).
    fn protect_max(&self, addr: HostAddr, len: u64, prot: VmProt) -> Result<(), KernReturn> {
        if len == 0 {
            return Ok(());
        }
        let (first, length) = pages(addr, len).ok_or(INVALID_ARGUMENT)?;
        let bits = prot_bits(prot);
        let _views = views();
        // Core protects each mapping it made once its views are in, read and write as the views
        // already are unless it is read-only: no call where the views have the protection.
        if files::protected(first, length, bits) {
            return Ok(());
        }
        // SAFETY: lowering or raising a range's protection is the caller's decision; a later
        // access it forbids faults (a process death).
        if unsafe { libc::mprotect(ptr(first), length.to_usize(), bits) } != 0 {
            return Err(refused("protect_max", true));
        }
        files::protect(first, length, bits);
        Ok(())
    }

    /// `munmap` of the range's pages, which show no file from then on; none for no bytes (module
    /// docs). A range of memory the library registered and does not own (the guest's RAM) is
    /// refused, unmapped by nothing. It goes a piece at a time ([`DEALLOCATE_PIECE`]), each
    /// forgotten and then unmapped.
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
        // `pages` answered, so the range ends inside the address space.
        let end = first + length;
        let mut at = first;
        while at < end {
            let now = (end - at).min(DEALLOCATE_PIECE);
            let _views = views();
            files::forget(at, now);
            // SAFETY: the range is one core allocated and no longer uses (its owner's drop).
            if unsafe { libc::munmap(ptr(at), now.to_usize()) } != 0 {
                return refused("deallocate", true);
            }
            at += now;
        }
        KernReturn::SUCCESS
    }

    /// The budget's headroom ([`budget::HEADROOM`]): what the budget keeps short of itself for
    /// what has no way but mappings of its own. A view kept only to be taken again has another,
    /// being made again, so the views no object holds take no more of the room below the budget
    /// than that.
    fn idle_view_mappings(&self) -> Option<u64> {
        Some(budget::HEADROOM)
    }

    /// The views' budget's room ([`budget::room`]): short of its headroom where a device of the
    /// process pins imported pages, as a view past it is mapped through dma-bufs there.
    fn view_room(&self) -> Option<u64> {
        Some(budget::room())
    }

    /// The mappings the backing table's runs in the view take ([`files::mappings_between`]), as
    /// the budget counts them: a mapping a run mapped from its file, one a dma-buf ("Views
    /// through dma-bufs").
    fn view_mappings(&self, at: HostAddr, len: u64) -> Option<u64> {
        let first = at.addr().to_u64();
        Some(files::mappings_between(first, first.saturating_add(len)))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::core::raise::death::DeathKind;

    use super::*;

    fn vm() -> LinuxVm {
        LinuxVm::new()
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

    /// The host's page, which the tests lay their pages out by: x86-64's 4 KiB.
    pub(crate) const PAGE: u64 = 0x1000;

    /// Whether every page of `len` bytes from `at` is writable now, as `/proc/self/maps` lists
    /// them: Linux has no maximum protection, so the current one is what refuses a write.
    pub(crate) fn writable(at: HostAddr, len: u64) -> bool {
        let first = at.addr().to_u64();
        let end = first.saturating_add(len);
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        let mut next = first;
        // The kernel lists the mappings in address order.
        for line in maps.lines() {
            let mut fields = line.split_whitespace();
            let (Some(range), Some(perms)) = (fields.next(), fields.next()) else {
                continue;
            };
            let (low, high) = range.split_once('-').unwrap();
            let low = u64::from_str_radix(low, 16).unwrap();
            let high = u64::from_str_radix(high, 16).unwrap();
            if (low..high).contains(&next) {
                if !perms.contains('w') {
                    return false;
                }
                next = high;
                if next >= end {
                    return true;
                }
            }
        }
        false
    }

    /// Memory of the library's own whose pages a device's import holds stays mapped past its
    /// drop, until the import's hold goes, as a region's whose last reference goes in a no-copy
    /// buffer's deallocator, before the runtime frees the buffer's import
    #[test]
    fn own_memory_an_import_holds_is_unmapped_once_its_hold_goes() {
        let memory = OwnMemory::new(2 * page()).unwrap();
        let at = HostAddr::from_addr(memory.first().to_usize());
        let held = at.wrapping_add(page());
        let hold = crate::remap::UnmapHold::new(
            std::ptr::NonNull::new(held.as_ptr::<u8>()).unwrap(),
            page(),
        );
        write(held, 0, 0x5a);
        drop(memory);
        assert!(crate::remap::unmap_waits(at, 2 * page()), "the unmap waits");
        assert_eq!(read(held, 0), 0x5a, "the held page stays mapped");
        drop(hold);
        assert!(!crate::remap::unmap_waits(at, 2 * page()), "unmapped");
    }

    /// Whether the page at `at` is mapped at all (`mincore` refuses an unmapped one).
    pub(crate) fn mapped(at: HostAddr) -> bool {
        let mut resident = 0u8;
        let first = at.addr().to_u64() & !(page() - 1);
        // SAFETY: a query; the kernel writes one byte for one page.
        unsafe { libc::mincore(ptr(first), page().to_usize(), &mut resident) == 0 }
    }

    /// The mappings `/proc/self/maps` lists inside `first..end`.
    pub(crate) fn kernel_mappings(first: u64, end: u64) -> u64 {
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        let inside = maps.lines().filter(|line| {
            let range = line.split_whitespace().next().unwrap();
            let (low, high) = range.split_once('-').unwrap();
            let low = u64::from_str_radix(low, 16).unwrap();
            let high = u64::from_str_radix(high, 16).unwrap();
            first <= low && high <= end
        });
        inside.count().to_u64()
    }

    /// The table's count of a view's mappings is the kernel's: over a view of pages in and out of
    /// order, a protect inside it, the protection set back and an unmap of part of it, the
    /// mappings `/proc/self/maps` lists in the view's range are as many as the table counts there
    #[test]
    fn the_table_counts_a_views_mappings_as_the_kernel_maps_them() {
        let v = vm();
        let p = page();
        let src = allocate_aliasable(8 * p).unwrap();
        let at = v.allocate_anywhere(8 * p).unwrap();
        // Pages 0-3 in order, then 6 and 7, then 4 and 5: three runs of the file.
        for (to, from) in [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 6),
            (5, 7),
            (6, 4),
            (7, 5),
        ] {
            v.remap_shared_overwrite(at.wrapping_add(to * p), src.wrapping_add(from * p), p)
                .unwrap();
        }
        let (first, end) = (at.addr().to_u64(), at.addr().to_u64() + 8 * p);
        let both = || {
            (
                kernel_mappings(first, end),
                files::mappings_between(first, end),
            )
        };
        assert_eq!(both(), (3, 3), "the view");
        v.protect_max(at.wrapping_add(p), p, VmProt::READ).unwrap();
        assert_eq!(both(), (5, 5), "its second page read-only");
        v.protect_max(at.wrapping_add(p), p, VmProt::READ | VmProt::WRITE)
            .unwrap();
        assert_eq!(both(), (3, 3), "read and write again");
        assert!(v.deallocate(at.wrapping_add(2 * p), p).is_success());
        assert_eq!(both(), (4, 4), "its third page unmapped");
        assert!(v.deallocate(at, 8 * p).is_success());
        assert!(v.deallocate(src, 8 * p).is_success());
    }

    /// What `view_mappings` answers for a view is the kernel's count in its range, which its
    /// teardown gives back: a mapping a page for pages remapped one at a time backwards, and,
    /// for the same pages remapped at once, one where views are mapped through dma-bufs
    #[test]
    fn a_views_mappings_are_the_kernels_in_its_range() {
        let v = vm();
        let p = page();
        let n = 16u64;
        let src = allocate_aliasable(n * p).unwrap();
        let one = v.allocate_anywhere(n * p).unwrap();
        let at_once = v.allocate_anywhere(n * p).unwrap();
        // Backwards, a page a run.
        let runs: Vec<_> = (0..n)
            .map(|i| (src.wrapping_add((n - 1 - i) * p), p))
            .collect();
        for (i, &(from, len)) in runs.iter().enumerate() {
            v.remap_shared_overwrite(one.wrapping_add(i.to_u64() * p), from, len)
                .unwrap();
        }
        v.remap_runs_shared_overwrite(at_once, &runs).unwrap();
        let kernel = |at: HostAddr| {
            let first = at.addr().to_u64();
            kernel_mappings(first, first + n * p)
        };
        assert_eq!(v.view_mappings(one, n * p), Some(n));
        assert_eq!(v.view_mappings(one, n * p), Some(kernel(one)));
        let through = super::super::import::views_through_dma_bufs();
        let want = if through { 1 } else { n };
        assert_eq!(v.view_mappings(at_once, n * p), Some(want));
        assert_eq!(v.view_mappings(at_once, n * p), Some(kernel(at_once)));
        for at in [one, at_once, src] {
            assert!(v.deallocate(at, n * p).is_success());
        }
    }

    /// Whether this process opens `/dev/udmabuf`, which views through dma-bufs need.
    fn udmabuf_opens() -> bool {
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/udmabuf")
            .is_ok()
    }

    /// A view of runs made at once shows their pages in order, both ways, and so does a view of
    /// it. Where udmabuf opens, a view of pages out of order takes one mapping, as one dma-buf of
    /// them, and a protect or an unmap of part of it cuts it as the kernel does, the table's count
    /// the kernel's; where it does not, a mapping a page. A run with no file behind it is refused
    /// at its index, the runs before it mapped
    #[test]
    fn a_view_of_runs_takes_a_mapping_a_dma_buf() {
        let v = vm();
        let p = page();
        let n = 64u64;
        let src = allocate_aliasable(n * p).unwrap();
        for i in 0..n {
            write(src, i * p, i as u8);
        }
        let at = v.allocate_anywhere(n * p).unwrap();
        // Backwards, a page a run.
        let runs: Vec<_> = (0..n)
            .map(|i| (src.wrapping_add((n - 1 - i) * p), p))
            .collect();
        v.remap_runs_shared_overwrite(at, &runs).unwrap();
        assert!((0..n).all(|i| read(at, i * p) == (n - 1 - i) as u8));
        write(at, 3, 0x5a);
        assert_eq!(read(src, (n - 1) * p + 3), 0x5a);
        write(src, 9, 0xa5);
        assert_eq!(read(at, (n - 1) * p + 9), 0xa5);
        let (first, end) = (at.addr().to_u64(), at.addr().to_u64() + n * p);
        let both = || {
            (
                kernel_mappings(first, end),
                files::mappings_between(first, end),
            )
        };
        let through = udmabuf_opens();
        assert_eq!(both(), if through { (1, 1) } else { (n, n) }, "the view");
        v.protect_max(at.wrapping_add(p), p, VmProt::READ).unwrap();
        assert_eq!(
            both(),
            if through { (3, 3) } else { (n, n) },
            "a page read-only"
        );
        assert!(v.deallocate(at.wrapping_add(5 * p), p).is_success());
        let left = if through { (4, 4) } else { (n - 1, n - 1) };
        assert_eq!(both(), left, "a page unmapped");
        // A view of the view: the pages its runs show.
        let again = v.allocate_anywhere(2 * p).unwrap();
        v.remap_runs_shared_overwrite(again, &[(at.wrapping_add(2 * p), p), (at, p)])
            .unwrap();
        assert_eq!((read(again, 0), read(again, p + 3)), ((n - 3) as u8, 0x5a));
        // A run of no file: refused at its index, the run before it mapped.
        let none = v.allocate_anywhere(p).unwrap();
        let other = v.allocate_anywhere(2 * p).unwrap();
        assert_eq!(
            v.remap_runs_shared_overwrite(other, &[(src, p), (none, p)]),
            Err((1, INVALID_ADDRESS))
        );
        assert_eq!(read(other, 9), 0xa5);
        for (a, len) in [(at, n), (again, 2), (none, 1), (other, 2), (src, n)] {
            assert!(v.deallocate(a, len * p).is_success());
        }
    }

    /// A fold maps a view's runs from their files through dma-bufs, the same pages both ways,
    /// but for the pages an import may pin, which stay mapped from their files
    #[test]
    fn a_fold_maps_the_pages_no_import_holds_through_dma_bufs() {
        if !udmabuf_opens() {
            return;
        }
        let v = vm();
        let p = page();
        let n = 64u64;
        let src = allocate_aliasable(n * p).unwrap();
        let byte = |i: u64| u8::try_from(i).unwrap();
        for i in 0..n {
            write(src, i * p, byte(i));
        }
        let at = v.allocate_anywhere(n * p).unwrap();
        // Backwards, a remap and a mapping a page.
        for i in 0..n {
            let from = src.wrapping_add((n - 1 - i) * p);
            v.remap_shared_overwrite(at.wrapping_add(i * p), from, p)
                .unwrap();
        }
        let (first, end) = (at.addr().to_u64(), at.addr().to_u64() + n * p);
        let both = || {
            (
                kernel_mappings(first, end),
                files::mappings_between(first, end),
            )
        };
        assert_eq!(both(), (n, n));
        // Its last 8 pages held by an import.
        let pinned = super::super::import::Pinned::of(end - 8 * p, end);
        assert!(fold_within(first..end) >= n - 8 - 3);
        let (kernel, table) = both();
        assert_eq!(kernel, table);
        assert!(kernel <= 8 + 2, "{kernel}");
        assert!((0..n).all(|i| read(at, i * p) == byte(n - 1 - i)));
        write(at, 3, 0x5a);
        assert_eq!(read(src, (n - 1) * p + 3), 0x5a);
        drop(pinned);
        fold_within(first..end);
        assert!(both().0 <= 2, "the pin gone");
        for (a, len) in [(at, n), (src, n)] {
            assert!(v.deallocate(a, len * p).is_success());
        }
    }

    /// A map the kernel refuses at the process's limit of mappings folds the views no import
    /// holds and is made again, the first try having changed nothing; refused again, or outside
    /// a call that can try again, it ends the process in strict mode
    #[test]
    fn a_map_refused_at_the_limit_folds_the_views_and_tries_again() {
        use crate::core::testkit::capture::catch_death;
        if !udmabuf_opens() {
            return;
        }
        let dies = |site: Result<KernReturn, crate::core::raise::death::DeathSite>| {
            site.is_err_and(|site| (site.kind, site.file) == (DeathKind::Abort, "mmap"))
        };
        let at_limit = |on: bool| TEST_AT_LIMIT.with(|t| t.set(on));
        let v = vm();
        let p = page();
        let n = 64u64;
        let src = allocate_aliasable(n * p).unwrap();
        let at = v.allocate_anywhere(n * p).unwrap();
        // Backwards, a remap and a mapping a page.
        for i in 0..n {
            let from = src.wrapping_add((n - 1 - i) * p);
            v.remap_shared_overwrite(at.wrapping_add(i * p), from, p)
                .unwrap();
        }
        let (first, end) = (at.addr().to_u64(), at.addr().to_u64() + n * p);
        assert_eq!(kernel_mappings(first, end), n);
        let tries = std::cell::Cell::new(0);
        let answer = reclaiming(|| {
            tries.set(tries.get() + 1);
            if tries.get() > 1 {
                return KernReturn::SUCCESS;
            }
            at_limit(true);
            let kr = no_space("a test's map");
            at_limit(false);
            kr
        });
        assert_eq!((answer, tries.get()), (KernReturn::SUCCESS, 2));
        assert!(kernel_mappings(first, end) <= 2, "folded");
        let refused = || no_space("a test's map");
        at_limit(true);
        assert!(dies(catch_death(|| reclaiming(refused))), "refused again");
        assert!(dies(catch_death(refused)), "no second try");
        at_limit(false);
        for a in [at, src] {
            assert!(v.deallocate(a, n * p).is_success());
        }
    }

    /// A map refused at the process's limit of mappings outside a call that can try again
    /// answers `KERN_NO_SPACE` by default, counted, as Mach's does when memory runs out
    #[test]
    fn a_map_refused_at_the_limit_answers_no_space_by_default() {
        let before = crate::answer::out_of_host_memory_count();
        TEST_AT_LIMIT.with(|t| t.set(true));
        let answer = crate::answer::relaxed(|| no_space("a test's map"));
        TEST_AT_LIMIT.with(|t| t.set(false));
        assert_eq!(answer, NO_SPACE);
        assert_eq!(crate::answer::out_of_host_memory_count(), before + 1);
    }

    /// Where the kernel gives no file, a range a view can be made of answers `KERN_NO_SPACE` and
    /// a stand-in's memory none, as Mach's and IOSurface's calls do when memory runs out,
    /// counted; in strict mode the process ends at the call that made no file
    #[test]
    fn no_file_answers_as_memory_runs_out() {
        use crate::core::testkit::capture::catch_death;
        let p = page();
        TEST_NO_FILE.with(|t| t.set(true));
        let site = catch_death(|| allocate_aliasable(p));
        assert_eq!(
            site.map_err(|site| (site.kind, site.file)),
            Err((DeathKind::Abort, "memfd_create"))
        );
        let before = crate::answer::out_of_host_memory_count();
        let (range, stand_in) =
            crate::answer::relaxed(|| (allocate_aliasable(p), OwnMemory::new(p)));
        TEST_NO_FILE.with(|t| t.set(false));
        assert_eq!(range, Err(NO_SPACE));
        assert!(stand_in.is_none());
        assert_eq!(crate::answer::out_of_host_memory_count(), before + 2);
        let at = allocate_aliasable(p).unwrap();
        assert!(vm().deallocate(at, p).is_success());
    }

    /// A graphics region's memory is zeros of one file of its own, mapped read and write, so a
    /// device that refuses its pointer imports a dma-buf of its pages (`super::import`); none is
    /// made of 0 bytes
    #[test]
    fn a_region_s_memory_is_one_file_s_pages() {
        let len = 4 * page();
        let memory = RegionMemory::new(len).unwrap();
        let at = HostAddr::from_addr(memory.base().addr());
        assert_eq!(at.addr().to_u64() % page(), 0);
        assert_eq!((read(at, 0), read(at, len - 1)), (0, 0));
        write(at, 5, 0x5a);
        assert_eq!(read(at, 5), 0x5a);
        let first = at.addr().to_u64();
        assert_eq!(files::pieces(first, len).map(|p| p.len()), Some(1));
        assert_eq!(files::writable(first, len), Some(true));
        assert!(RegionMemory::new(0).is_none());
    }

    /// The host page is 4 KiB on x86-64 Linux, its shift 12
    #[test]
    fn the_page_is_the_hosts() {
        let v = vm();
        assert_eq!(v.page_size(), 1 << v.page_shift());
        #[cfg(target_arch = "x86_64")]
        assert_eq!((v.page_size(), v.page_shift()), (4096, 12));
    }

    /// An allocation anywhere is page-aligned and zero-filled, and no view can be made of it;
    /// one of 0 bytes is address 0, as Mach's is; one the address space cannot hold is refused
    #[test]
    fn an_allocation_is_zeros_no_view_is_made_of() {
        let v = vm();
        let len = 2 * page();
        let a = v.allocate_anywhere(len).unwrap();
        assert_eq!(a.addr().to_u64() % page(), 0);
        assert_eq!((read(a, 0), read(a, len - 1)), (0, 0));
        write(a, 7, 0xa5);
        let dst = v.allocate_anywhere(len).unwrap();
        assert_eq!(v.remap_shared_overwrite(dst, a, len), Err(INVALID_ADDRESS));
        assert_eq!(v.allocate_anywhere(0), Ok(HostAddr::NULL));
        assert!(
            !v.allocate_anywhere(u64::MAX - page())
                .unwrap_err()
                .is_success()
        );
        assert!(v.deallocate(a, len).is_success());
        assert!(v.deallocate(dst, len).is_success());
        assert!(!mapped(a));
    }

    /// A view shows its source's pages both ways, from the page of the source's first byte; a
    /// view of the view shows them too, and keeps showing them after the first view is unmapped
    #[test]
    fn a_view_of_a_view_outlasts_the_first() {
        let v = vm();
        let len = 4 * page();
        let src = allocate_aliasable(len).unwrap();
        let first = v.allocate_anywhere(len).unwrap();
        let second = v.allocate_anywhere(2 * page()).unwrap();
        write(src, page() + 5, 0x11);
        assert_eq!(
            v.remap_shared_overwrite(first, src.wrapping_add(0x10), len - 0x10),
            Ok(first)
        );
        assert_eq!(read(first, page() + 5), 0x11);
        write(first, 9, 0x33);
        assert_eq!(read(src, 9), 0x33);
        v.remap_shared_overwrite(second, first.wrapping_add(page()), 2 * page())
            .unwrap();
        assert_eq!(read(second, 5), 0x11);
        assert!(v.deallocate(first, len).is_success());
        write(src, page() + 6, 0x22);
        assert_eq!(read(second, 6), 0x22);
        write(second, page(), 0x44);
        assert_eq!(read(src, 2 * page()), 0x44);
        assert!(v.deallocate(second, 2 * page()).is_success());
        assert!(v.deallocate(src, len).is_success());
    }

    /// A view over another view replaces it; an overwrite or an unmap of part of a view leaves
    /// the rest showing its pages; a source with a byte of no file maps nothing
    #[test]
    fn views_are_overwritten_and_unmapped_in_part() {
        let v = vm();
        let (a, b) = (
            allocate_aliasable(2 * page()).unwrap(),
            allocate_aliasable(2 * page()).unwrap(),
        );
        write(a, 0, 1);
        write(a, page(), 2);
        write(b, 0, 3);
        let view = v.allocate_anywhere(2 * page()).unwrap();
        v.remap_shared_overwrite(view, a, 2 * page()).unwrap();
        v.remap_shared_overwrite(view, b, page()).unwrap();
        assert_eq!((read(view, 0), read(view, page())), (3, 2));
        v.allocate_fixed_overwrite(view.wrapping_add(page() + 1), 1)
            .unwrap();
        assert_eq!((read(view, 0), read(view, page())), (3, 0));
        let copy = v.allocate_anywhere(2 * page()).unwrap();
        assert_eq!(
            v.remap_shared_overwrite(copy, view, 2 * page()),
            Err(INVALID_ADDRESS)
        );
        assert_eq!(read(copy, 0), 0, "nothing was mapped");
        v.remap_shared_overwrite(copy, view, page()).unwrap();
        assert_eq!(read(copy, 0), 3);
        for (at, len) in [(view, 2), (copy, 2), (a, 2), (b, 2)] {
            assert!(v.deallocate(at, len * page()).is_success());
        }
    }

    /// A view of more mappings than a piece of a deallocation holds, and a part piece past them,
    /// is unmapped and forgotten whole, and the pages either side of it stay mapped
    #[test]
    fn a_view_of_many_mappings_is_deallocated_whole() {
        let v = vm();
        let pages = 2 * (DEALLOCATE_PIECE / page()) + 3;
        let length = pages * page();
        let (a, b) = (
            allocate_aliasable(length).unwrap(),
            allocate_aliasable(length).unwrap(),
        );
        let reserved = v.allocate_anywhere(length + 2 * page()).unwrap();
        let view = reserved.wrapping_add(page());
        // Every other page from the other file, so each page is a mapping of its own.
        for i in 0..pages {
            let source = if i % 2 == 0 { a } else { b };
            write(source, i * page(), 7);
            v.remap_shared_overwrite(
                view.wrapping_add(i * page()),
                source.wrapping_add(i * page()),
                page(),
            )
            .unwrap();
        }
        let first = view.addr().to_u64();
        assert_eq!(kernel_mappings(first, first + length), pages);
        assert_eq!(read(view, length - page()), 7);
        assert!(v.deallocate(view, length).is_success());
        assert_eq!(kernel_mappings(first, first + length), 0);
        assert!((0..pages).all(|i| !mapped(view.wrapping_add(i * page()))));
        assert!(files::pieces(first, page()).is_none());
        assert!(files::pieces(first + length - page(), page()).is_none());
        assert!(mapped(reserved) && mapped(view.wrapping_add(length)));
        for (at, len) in [
            (reserved, page()),
            (view.wrapping_add(length), page()),
            (a, length),
            (b, length),
        ] {
            assert!(v.deallocate(at, len).is_success());
        }
    }

    /// A file of one page whose bytes are `byte`, its descriptor read-only (opened again through
    /// `/proc/self/fd`) when `read_only`, so a shared writable map of it is refused (`EACCES`).
    fn page_file(byte: u8, read_only: bool) -> Arc<OwnedFd> {
        let fd = memfd(page()).unwrap();
        let at = file(
            None,
            page(),
            &Backing {
                fd: Arc::clone(&fd),
                offset: 0,
            },
            libc::PROT_READ | libc::PROT_WRITE,
        )
        .unwrap();
        // SAFETY: the page just mapped writable, unmapped right after.
        unsafe {
            ptr(at).cast::<u8>().write_bytes(byte, page().to_usize());
            libc::munmap(ptr(at), page().to_usize());
        }
        if !read_only {
            return fd;
        }
        let path = std::ffi::CString::new(format!("/proc/self/fd/{}", fd.as_raw_fd())).unwrap();
        // SAFETY: a NUL-terminated path; the descriptor is owned from here on.
        let raw = unsafe { libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
        assert!(raw >= 0);
        // SAFETY: a descriptor just opened, owned once.
        Arc::new(unsafe { OwnedFd::from_raw_fd(raw) })
    }

    /// A remap whose second piece the kernel refuses changes nothing, as Mach's: its first piece
    /// is put back as the target showed it, a view's file there, and zeros with no access over
    /// a reservation's bytes of no file; the table answers as before
    #[test]
    fn a_remap_that_fails_part_way_changes_nothing() {
        let v = vm();
        let p = page();
        let rw = libc::PROT_READ | libc::PROT_WRITE;
        // The source: a run of a file of 0x11s, then a registration of a file of 0x22s whose
        // descriptor is read-only, which the kernel refuses to map shared and writable.
        let src = v.allocate_anywhere(2 * p).unwrap();
        let first = src.addr().to_u64();
        let ones = Backing {
            fd: page_file(0x11, false),
            offset: 0,
        };
        file(Some(first), p, &ones, rw).unwrap();
        files::record(first, p, ones, rw);
        let twos = Backing {
            fd: page_file(0x22, true),
            offset: 0,
        };
        file(Some(first + p), p, &twos, libc::PROT_READ).unwrap();
        let held = files::Registration::new(first + p, p, twos);
        assert_eq!(files::pieces(first, 2 * p).map(|ps| ps.len()), Some(2));

        // A target whose first page is a view of a file of 0x33s.
        let view = v.allocate_anywhere(2 * p).unwrap();
        let threes = allocate_aliasable(p).unwrap();
        write(threes, 0, 0x33);
        v.remap_shared_overwrite(view, threes, p).unwrap();
        assert_eq!(
            v.remap_shared_overwrite(view, src, 2 * p),
            Err(PROTECTION_FAILURE)
        );
        assert_eq!(read(view, 0), 0x33, "the view's file is back");
        write(view, 1, 0x34);
        assert_eq!(read(threes, 1), 0x34, "the same pages");
        let shown = files::pieces(view.addr().to_u64(), p).unwrap();
        assert_eq!((shown.len(), shown[0].prot), (1, rw));
        assert!(files::pieces(view.addr().to_u64() + p, p).is_none());

        // A target of a reservation's bytes of no file.
        let reserved = v.allocate_anywhere(2 * p).unwrap();
        assert_eq!(
            v.remap_shared_overwrite(reserved, src, 2 * p),
            Err(PROTECTION_FAILURE)
        );
        assert!(files::pieces(reserved.addr().to_u64(), p).is_none());
        assert!(
            mapped(reserved) && !v.readable(reserved, p),
            "zeros, no access"
        );
        assert!(v.readable(reserved.wrapping_add(p), p), "untouched");

        drop(held);
        files::forget(first, p);
        for (at, len) in [(src, 2), (view, 2), (threes, 1), (reserved, 2)] {
            assert!(v.deallocate(at, len * p).is_success());
        }
    }

    /// The protection set is the one the table answers for reads; a range of no access is not
    /// readable, by the table or by the kernel
    #[test]
    fn readable_follows_the_protection() {
        let v = vm();
        let p = page();
        let a = allocate_aliasable(2 * p).unwrap();
        let anon = v.allocate_anywhere(2 * p).unwrap();
        assert!(v.readable(a, 2 * p) && v.readable(anon, 2 * p) && v.readable(a, 0));
        v.protect_max(a.wrapping_add(p), p, VmProt::NONE).unwrap();
        v.protect_max(anon.wrapping_add(p), p, VmProt::NONE)
            .unwrap();
        assert!(v.readable(a, p) && !v.readable(a, 2 * p));
        assert!(v.readable(anon, p) && !v.readable(anon, 2 * p));
        v.protect_max(a, 2 * p, VmProt::READ).unwrap();
        assert!(v.readable(a, 2 * p));
        assert!(v.deallocate(anon, 2 * p).is_success());
        assert!(!v.readable(anon, 1));
        assert_eq!(v.protect_max(anon, p, VmProt::READ), Err(INVALID_ADDRESS));
        assert!(v.deallocate(a, 2 * p).is_success());
    }

    /// Deallocating nothing succeeds, a wrapping range is an invalid argument, and registered
    /// memory is never unmapped
    #[test]
    fn deallocate_answers_as_mach_and_keeps_registered_memory() {
        let v = vm();
        let a = v.allocate_anywhere(page()).unwrap();
        assert_eq!(v.deallocate(a, 0), KernReturn::SUCCESS);
        assert_eq!(
            v.deallocate(HostAddr::from_addr(usize::MAX & !0xfff), 2 * page()),
            INVALID_ARGUMENT
        );
        let fd = memfd(page()).unwrap();
        let at = file(
            None,
            page(),
            &Backing {
                fd: Arc::clone(&fd),
                offset: 0,
            },
            3,
        )
        .unwrap();
        let held = files::Registration::new(at, page(), Backing { fd, offset: 0 });
        let ram = HostAddr::from_addr(at.to_usize());
        assert_eq!(v.deallocate(ram, page()), INVALID_ARGUMENT);
        assert!(mapped(ram));
        drop(held);
        assert!(v.deallocate(ram, page()).is_success());
        assert!(v.deallocate(a, page()).is_success());
    }

    /// A map the kernel refuses for want of memory is `KERN_NO_SPACE`, logged with the process's
    /// mappings and their limit; in a child, whose address space is capped
    #[test]
    fn a_refused_map_is_no_space_with_the_mapping_census() {
        const CHILD: &str = "RVG_LINUX_VM_NO_SPACE_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let cap = libc::rlimit {
                rlim_cur: 1 << 36,
                rlim_max: 1 << 36,
            };
            // SAFETY: lowering this child's own address-space limit.
            assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_AS, &cap) }, 0);
            assert_eq!(vm().allocate_anywhere(1 << 37), Err(NO_SPACE));
            let (mappings, limit) = map_census();
            assert!(mappings.parse::<u64>().unwrap() > 0, "{mappings}");
            assert!(limit.parse::<u64>().unwrap() > 0, "{limit}");
            return;
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "os::linux::vm::tests::a_refused_map_is_no_space_with_the_mapping_census",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        assert!(
            String::from_utf8_lossy(&out.stdout).contains(" 1 passed"),
            "{out:?}"
        );
    }

    /// A refusal is for want of mappings where the process holds the limit, or the 2 a map may
    /// add short of it; not below, nor where either count does not read
    #[test]
    fn at_the_limit_is_within_a_maps_two() {
        assert!(at_the_limit("65530", "65530"));
        assert!(at_the_limit("65528", "65530"));
        assert!(!at_the_limit("65527", "65530"));
        assert!(!at_the_limit("?", "65530"));
        assert!(!at_the_limit("65530", "?"));
    }

    /// An alias of a guest page with zeros past it, as a heap of less than the runtime's 16 KiB
    /// is made over, has a file behind every byte, writable or read-only, so a dma-buf can be
    /// made of all of it, as the runtime asks where the device refuses file pages as host memory
    /// (`crate::gpu::backing::gather`, `super::dma_buf`)
    #[test]
    fn an_alias_with_zeros_past_its_pages_has_a_file_behind_every_byte() {
        use crate::core::platform::types::ResourceAddressRange;
        use crate::gpu::backing::{REPLACE, gather};
        let p = page();
        for read_only in [false, true] {
            let guest = allocate_aliasable(p).unwrap();
            write(guest, 0, 0x5a);
            let ranges = [ResourceAddressRange {
                address: guest.addr().to_u64(),
                length: p,
            }];
            let alias = gather(REPLACE, &ranges, 4 * p, read_only).unwrap();
            let first = alias.first().as_ptr().addr().to_u64();
            assert!(
                files::pieces(first, 4 * p).is_some(),
                "read-only {read_only}"
            );
            let at = HostAddr::from_addr(alias.first().as_ptr().addr());
            assert_eq!(
                (read(at, 0), read(at, p), read(at, 4 * p - 1)),
                (0x5a, 0, 0)
            );
            drop(alias);
            assert!(vm().deallocate(guest, p).is_success());
        }
    }
}
