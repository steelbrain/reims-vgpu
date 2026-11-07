//! The budget of kernel mappings (VMAs) the library's views of memory may hold.
//!
//! A view of scattered guest pages is a mapping per run of them ([`super::vm`]), and the kernel
//! refuses any map past `vm.max_map_count` mappings in the process: past that, not only views
//! fail but every map the process makes, QEMU's, the C library's and the GPU driver's among them.
//! So the views stop short of the limit: a view, a range a view can be made of, an import's
//! writable twin or an IOSurface stand-in's memory that would take the mappings the library holds
//! past `vm.max_map_count - RESERVE` is refused before anything is mapped ([`admit`]), and the
//! mappings the rest of the process makes are never refused for the views' sake. macOS has no
//! such limit, so no answer of Metal's fits exactly: the call answers as it does on the Metal
//! platform when memory runs out, a remap or a range `KERN_NO_SPACE`, a stand-in no memory, a
//! twin none (so the runtime refuses its import as it refuses read-only pages), counted and
//! logged (`crate::answer::out_of_host_memory`), and in strict mode (`REIMS_VGPU_STRICT=1`) it
//! ends the process, naming what it would have mapped, the mappings and the limit.
//!
//! The mappings the library holds are counted exactly where it makes them: the backing table's
//! runs, each protect or unmap that cuts one included ([`super::files::mappings`]), the import
//! twins' pieces ([`super::import`]) and the stand-ins' memory ([`super::vm::OwnMemory`]). What it
//! does not count is the reserve's to cover ([`RESERVE`]).
//!
//! Where the library has another way than mappings of its own, it takes it short of the budget,
//! [`HEADROOM`] below it ([`claim`], [`Reach::ShortOfHeadroom`]): a view a device would need from
//! its files goes through dma-bufs instead, and a no-copy buffer's or heap's import of such a
//! view's pages, or of read-only pages with no twin, is refused at the runtime's gate, the buffer
//! or heap made over a copy (`super::vm`'s module docs, "Imports of views"), as a rebind's is.
//! What has no other way has the headroom to itself: a view through dma-bufs, a range, a
//! stand-in, and a twin where no device needs imported pages mapped from their files
//! ([`Reach::Whole`]). A device lend's pages are mapped from their files only short of the
//! headroom: past it the device declines the frame, which the CPU copies instead
//! (`super::import`'s `pages_from_files`).
//!
//! What would pass the budget first folds the views' runs mapped from their files that no import
//! a device may pin holds back into dma-bufs (`super::vm::fold_idle`), a mapping per 4 MiB of
//! their pages, and weighs again: only where that freed too little does it refuse.
//!
//! The limit is read when the first view is weighed, and again when one would pass it, as it may
//! have been raised since. A guest whose RAM, mapped from its files a mapping a 4 KiB page, would
//! pass the budget is logged where the budget bites, with the setting that raises it: as its
//! memory is leased where udmabuf is not the process's, as views past the budget are refused
//! there ([`warn_if_short`]), and as a device that needs imported pages mapped from their files is
//! made, as buffers and heaps past the headroom are copied there ([`warn_if_imports_short`]).

use std::sync::Once;
use std::sync::atomic::{AtomicU64, Ordering};

use super::files;

/// The mappings left to the rest of the process: QEMU's own, the C library's, the GPU driver's
/// and the runtime's (device memory maps), threads' stacks, and the library's anonymous
/// reservations with the gaps a view leaves in one, none of which the library counts.
///
/// No rule bounds them: what QEMU, the driver and the runtime map grows with the guest's work and
/// differs by driver and release, and nothing the library can read beforehand says how far. So
/// the reserve is a margin, not a bound: a quarter of the kernel's default `vm.max_map_count`
/// (65,530), so that the views may hold three quarters of a default limit. One measurement backs
/// it, no more: on a 12 GiB x86-64 guest's boot and a Safari and Maps session on an RTX 5080
/// laptop host, the
/// process held about 1,700 mappings before the guest's GPU work began and at most 4,600 besides
/// its views of guest memory. Where the views run out of it they are refused, naming the numbers
/// ([`admit`]); where the rest of the process does, the kernel refuses its maps, and the
/// library's own refusal for want of mappings answers the same way ([`super::vm`]'s
/// `no_space`).
pub(super) const RESERVE: u64 = 16_384;

/// The mappings kept below the budget for what has no way but mappings of its own, where the
/// library chooses a view's or an import's mappings ([`claim`]): views through dma-bufs, each a
/// mapping per 4 MiB of scattered pages at udmabuf's stock `list_limit` and one at each end
/// (`super::vm`'s `map_through`), the ranges views are made of, IOSurface stand-ins' memory, and
/// twins where no device needs imported pages mapped from their files ([`Reach::Whole`]). A
/// margin,
/// as [`RESERVE`] is, sized from the views it is for: views through dma-bufs of all of a 16 GiB
/// guest's pages take 4,096 mappings and two for each view, and the
/// views held at the 3840 × 2160 death (RTX 5080 Laptop, 16 GiB guest, 2026-10-27), 391 surfaces
/// over 3.9 GiB, would take about 1,800 through dma-bufs. Claims made at once are weighed together
/// ([`claim`]), so the headroom need cover none of them.
pub(super) const HEADROOM: u64 = 8_192;

/// The piece guest memory comes in for the warnings' estimate: 4 KiB. An x86-64 macOS guest
/// hands out its pages in 16 KiB clumps laid backwards, so a view of them mapped from their files
/// takes a mapping a 4 KiB page (`super::vm`'s module docs, "Views through dma-bufs"): every view
/// where udmabuf is not the process's, and where a device needs imported pages mapped from their
/// files ("Imports of views").
const GUEST_PIECE: u64 = 0x1000;

/// `vm.max_map_count` as last read; 0 before the first read.
static LIMIT: AtomicU64 = AtomicU64::new(0);

/// The setting the warnings name.
const SETTING: &str = "vm.max_map_count";

#[cfg(test)]
thread_local! {
    /// A test's limit, in place of the kernel's, on the thread that sets it.
    static TEST_LIMIT: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// Sets the calling thread's limit in place of the kernel's, or the kernel's again with `None`.
#[cfg(test)]
pub(super) fn set_test_limit(limit: Option<u64>) {
    TEST_LIMIT.with(|l| l.set(limit));
}

/// `vm.max_map_count` read now; `u64::MAX` where it does not read, leaving the kernel's own
/// refusal.
fn read_limit() -> u64 {
    std::fs::read_to_string("/proc/sys/vm/max_map_count")
        .ok()
        .and_then(|l| l.trim().parse().ok())
        .unwrap_or(u64::MAX)
}

/// The limit as last read, reading it the first time; a test's, where the thread set one.
fn limit() -> u64 {
    #[cfg(test)]
    if let Some(limit) = TEST_LIMIT.with(std::cell::Cell::get) {
        return limit;
    }
    match LIMIT.load(Ordering::Relaxed) {
        0 => reread(),
        limit => limit,
    }
}

/// The limit read again, and kept; a test's, where the thread set one.
fn reread() -> u64 {
    #[cfg(test)]
    if let Some(limit) = TEST_LIMIT.with(std::cell::Cell::get) {
        return limit;
    }
    let limit = read_limit();
    LIMIT.store(limit, Ordering::Relaxed);
    limit
}

/// The mappings the library's views may hold under `limit`.
fn budget(limit: u64) -> u64 {
    limit.saturating_sub(RESERVE)
}

/// The import twins' mappings.
static TWIN_MAPPINGS: AtomicU64 = AtomicU64::new(0);

/// The mappings claimed ([`claim`]) and not yet counted where they are made.
static CLAIMED: AtomicU64 = AtomicU64::new(0);

/// The IOSurface stand-ins' mappings of their own memory.
static OWN_MAPPINGS: AtomicU64 = AtomicU64::new(0);

/// Counts `count` more import twin mappings, or fewer.
pub(super) fn twins_hold(count: u64, holding: bool) {
    count_on(&TWIN_MAPPINGS, count, holding);
}

/// Counts `count` more stand-in mappings, or fewer.
pub(super) fn own_hold(count: u64, holding: bool) {
    count_on(&OWN_MAPPINGS, count, holding);
}

/// Adds `count` to `counter` while `holding`, else takes it away.
fn count_on(counter: &AtomicU64, count: u64, holding: bool) {
    if holding {
        counter.fetch_add(count, Ordering::Relaxed);
    } else {
        counter.fetch_sub(count, Ordering::Relaxed);
    }
}

/// The mappings the library's views hold now (module docs).
pub(super) fn held() -> u64 {
    files::mappings()
        .saturating_add(TWIN_MAPPINGS.load(Ordering::Relaxed))
        .saturating_add(OWN_MAPPINGS.load(Ordering::Relaxed))
}

/// Whether the views may add `adding` mappings to those they hold and those claimed within the
/// budget less `kept`, at the limit read again where they would pass it as last read: the
/// mappings held and claimed, and the limit, where they may not.
fn weigh(adding: u64, kept: u64) -> Result<(), (u64, u64)> {
    let held = held().saturating_add(CLAIMED.load(Ordering::Acquire));
    let wanted = held.saturating_add(adding);
    if wanted <= budget(limit()).saturating_sub(kept) {
        return Ok(());
    }
    let limit = reread();
    if wanted <= budget(limit).saturating_sub(kept) {
        return Ok(());
    }
    Err((held, limit))
}

/// Mappings claimed for something the library makes, given back as it drops, which is once they
/// are counted where they are made, or once they are not to be.
#[must_use]
pub(super) struct Claim(u64);

impl Drop for Claim {
    fn drop(&mut self) {
        // Release: whoever claims after this sees what the claim was counted as.
        CLAIMED.fetch_sub(self.0, Ordering::Release);
    }
}

/// How far into the budget a [`claim`] may reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Reach {
    /// Short of it by [`HEADROOM`]: for what has another way than mappings of its own, which it
    /// takes past that (module docs).
    ShortOfHeadroom,
    /// All of it, the headroom included: for what has no other way, an import's twin where no
    /// device needs imported pages mapped from their files (`super::import`'s `twin`).
    Whole,
}

/// Claims `adding` mappings, where they fit as far into the budget as `reach` lets them with
/// those held and claimed, at the limit read again where they would not as last read; `None`
/// where they do not, and the caller takes its other way, or refuses. Claims made at once are
/// weighed together: each counts itself before it weighs, so the last of them to weigh sees the
/// others, and a claim never lets two pass that do not fit together.
pub(super) fn claim(adding: u64, reach: Reach) -> Option<Claim> {
    // Acquire: a claim given back before this one counts sees the mappings it was counted as.
    CLAIMED.fetch_add(adding, Ordering::AcqRel);
    let claim = Claim(adding);
    let kept = match reach {
        Reach::ShortOfHeadroom => HEADROOM,
        Reach::Whole => 0,
    };
    weigh(0, kept).is_ok().then_some(claim)
}

/// Whether `what` may add up to `adding` mappings: not where the views would pass the budget with
/// them, at the limit read again, though the views no import holds were folded back into
/// dma-bufs (`super::vm::fold_idle`) (module docs). A refusal is the host's memory running out
/// (`crate::answer::out_of_host_memory`): counted, its line naming `what`, the mappings the
/// views hold and the process holds, the limit, the reserve and the setting that gives room, and
/// in strict mode the process ends there. Threads that weigh views at once may pass it by what
/// they add between them, which the reserve covers.
#[must_use]
pub(super) fn admit(adding: u64, what: &'static str) -> bool {
    if weigh(adding, 0).is_ok() {
        return true;
    }
    super::vm::fold_idle();
    let Err((held, limit)) = weigh(adding, 0) else {
        return true;
    };
    crate::answer::out_of_host_memory("mmap", || {
        let (mappings, _) = super::vm::map_census();
        exhausted(what, adding, held, limit, 0, &mappings)
    });
    false
}

/// The line of [`admit`]'s refusal: `what` would add `adding` mappings to the `held` the views hold
/// under `limit`, `kept` short of the budget, the process holding `mappings`.
fn exhausted(what: &str, adding: u64, held: u64, limit: u64, kept: u64, mappings: &str) -> String {
    let room = held
        .saturating_add(adding)
        .saturating_add(RESERVE)
        .saturating_add(kept)
        .next_power_of_two()
        .max(1 << 20);
    let short = match kept {
        0 => String::new(),
        kept => format!(" and the {kept} kept for what has no other way"),
    };
    format!(
        "the library's views of memory ran out of kernel mappings: {what} would add up to \
         {adding} to the {held} they hold, past their budget of {}, {SETTING} {limit} less the \
         {RESERVE} kept for the rest of the process (which holds {mappings} in all){short}; \
         Metal's call goes on where macOS has no such limit, and `sysctl -w {SETTING}={room}` or \
         more gives room",
        budget(limit).saturating_sub(kept),
    )
}

/// The guest's RAM, as it was leased ([`warn_if_short`]).
static GUEST_RAM: AtomicU64 = AtomicU64::new(0);

/// Warns once, as a guest's RAM of `guest_ram` bytes is leased, where udmabuf is not the
/// process's and views of all of it mapped from its files, in 4 KiB pieces, would pass the
/// budget: past that much of it, a view is refused ([`admit`]). Where udmabuf is, a view
/// past the budget is mapped through dma-bufs instead, and [`warn_if_imports_short`] warns of
/// what that costs, here where its device was made before the lease.
pub(super) fn warn_if_short(guest_ram: u64) {
    GUEST_RAM.store(guest_ram, Ordering::Relaxed);
    if !super::dma_buf::opens() {
        short("views of guest memory are refused", 0, log::Level::Warn);
    } else if super::import::imports_need_files() {
        warn_if_imports_short();
    }
}

/// Tells once, where a device of the process needs imported pages mapped from their files
/// (`super::import`'s `learn_view_imports`) and imports of all of the guest's RAM, in 4 KiB pieces,
/// would come within the headroom of the budget: past that much of it, buffers and heaps over
/// guest memory are copied rather than imported (`super::import`'s `import_gate`), which works and
/// costs the copies, so it is information, not a warning: the setting is a speed-up. Nothing
/// before the guest's RAM is leased ([`warn_if_short`] warns then).
pub(super) fn warn_if_imports_short() {
    short(
        "buffers and heaps over guest memory are copied rather than imported, which is slower,",
        HEADROOM,
        log::Level::Info,
    );
}

/// The line of [`warn_if_short`] and [`warn_if_imports_short`], once for the process at
/// `level`, where the guest's RAM in 4 KiB pieces would pass the budget less `kept`: past it,
/// `what`.
fn short(what: &str, kept: u64, level: log::Level) {
    let guest_ram = GUEST_RAM.load(Ordering::Relaxed);
    let limit = reread();
    let line = budget(limit).saturating_sub(kept);
    let needed = guest_ram / GUEST_PIECE;
    if needed <= line {
        return;
    }
    static WARNED: Once = Once::new();
    WARNED.call_once(|| {
        let mib = line.saturating_mul(GUEST_PIECE) >> 20;
        log::log!(
            target: "reims_vgpu_vulkan::os",
            level,
            "{SETTING} is {limit}: {what} past {line} mappings, about {mib} MiB of the guest's {} \
             MiB mapped from their files in 4 KiB pieces; `sysctl -w {SETTING}={}` or more lets \
             them cover it",
            guest_ram >> 20,
            needed
                .saturating_add(RESERVE)
                .saturating_add(kept)
                .next_power_of_two()
                .max(1 << 20),
        );
    });
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::num::ToU64 as _;
    use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm as _, VmProt};

    use super::super::{LinuxVm, import, vm};
    use super::*;

    const PAGE: u64 = 0x1000;

    /// Sets the thread's limit for a test, and the kernel's again when it drops.
    struct Limit;

    impl Limit {
        /// Room for `room` more mappings than the views hold now, past the reserve.
        fn room(room: u64) -> Limit {
            set_test_limit(Some(held() + RESERVE + room));
            Limit
        }
    }

    impl Drop for Limit {
        fn drop(&mut self) {
            set_test_limit(None);
        }
    }

    /// Whether `f` ends the process at the budget's death (`mmap`).
    fn dies_at_the_budget<R>(f: impl FnOnce() -> R) -> bool {
        use reims_vgpu_core::raise::death::DeathKind;
        use reims_vgpu_core::testkit::capture::catch_death;
        catch_death(f).is_err_and(|site| (site.kind, site.file) == (DeathKind::Abort, "mmap"))
    }

    /// In strict mode, a view that would take the views past the budget ends the process before
    /// anything is mapped: its target keeps no file and reads its zeros; with room, the same view
    /// is made
    #[test]
    fn a_view_past_the_budget_dies_before_anything_is_mapped() {
        let v = LinuxVm::new();
        let src = vm::allocate_aliasable(64 * PAGE).unwrap();
        let at = v.allocate_anywhere(64 * PAGE).unwrap();
        let page = |base: HostAddr, i: u64| base.wrapping_add(i * PAGE);
        let limit = Limit::room(8);
        // Pages in reverse: a mapping each, until the budget ends the process at one.
        let died = (0..64).find(|&i| {
            dies_at_the_budget(|| v.remap_shared_overwrite(page(at, i), page(src, 63 - i), PAGE))
        });
        let i = died.expect("a view past the budget");
        let target = page(at, i).addr().to_u64();
        assert!(files::pieces(target, PAGE).is_none(), "no file mapped");
        // SAFETY: the reservation's page, mapped read and write.
        assert_eq!(unsafe { page(at, i).as_ptr::<u8>().read_volatile() }, 0);
        drop(limit);
        v.remap_shared_overwrite(page(at, i), page(src, 63 - i), PAGE)
            .unwrap();
        assert!(files::pieces(target, PAGE).is_some());
        assert!(v.deallocate(at, 64 * PAGE).is_success());
        assert!(v.deallocate(src, 64 * PAGE).is_success());
    }

    /// In strict mode, a range a view can be made of, an import's twin and an IOSurface stand-in's
    /// memory end the process past the budget too; with room, the twin is made
    #[test]
    fn ranges_twins_and_stand_ins_past_the_budget_die() {
        let v = LinuxVm::new();
        let at = vm::allocate_aliasable(2 * PAGE).unwrap();
        assert!(v.protect_max(at, 2 * PAGE, VmProt::READ).is_ok());
        let p = std::ptr::NonNull::new(at.as_ptr::<u8>()).unwrap();
        {
            let _limit = Limit::room(0);
            assert!(dies_at_the_budget(|| vm::allocate_aliasable(PAGE)));
            assert!(dies_at_the_budget(|| import::import_view(p, PAGE, true)));
            assert!(dies_at_the_budget(|| vm::OwnMemory::new(PAGE)));
        }
        assert_ne!(
            import::import_view(p, PAGE, true).at(),
            p,
            "a twin with room"
        );
        assert!(v.deallocate(at, 2 * PAGE).is_success());
    }

    /// By default, past the budget, a view of guest pages and a range a view can be made of
    /// answer `KERN_NO_SPACE` with nothing mapped, a stand-in's memory none, and an import's
    /// twin none, its pages given as they are, each counted, as Mach's and IOSurface's calls
    /// answer when memory runs out
    #[test]
    fn past_the_budget_answers_as_memory_runs_out_by_default() {
        use reims_vgpu_core::platform::os::vm::KernReturn;
        let v = LinuxVm::new();
        let src = vm::allocate_aliasable(64 * PAGE).unwrap();
        let at = v.allocate_anywhere(64 * PAGE).unwrap();
        let ro = vm::allocate_aliasable(PAGE).unwrap();
        assert!(v.protect_max(ro, PAGE, VmProt::READ).is_ok());
        let p = std::ptr::NonNull::new(ro.as_ptr::<u8>()).unwrap();
        let page = |base: HostAddr, i: u64| base.wrapping_add(i * PAGE);
        let before = crate::answer::out_of_host_memory_count();
        let limit = Limit::room(8);
        let refused = crate::answer::relaxed(|| {
            // Pages in reverse: a mapping each, until the budget refuses one.
            (0..64).find_map(|i| {
                (v.remap_shared_overwrite(page(at, i), page(src, 63 - i), PAGE)
                    .err())
                .map(|kr| (i, kr))
            })
        });
        let (i, kr) = refused.expect("a view past the budget");
        assert_eq!(kr, KernReturn(3), "KERN_NO_SPACE");
        drop(limit);
        let target = page(at, i).addr().to_u64();
        assert!(files::pieces(target, PAGE).is_none(), "nothing mapped");
        {
            let _limit = Limit::room(0);
            crate::answer::relaxed(|| {
                assert_eq!(vm::allocate_aliasable(PAGE).err(), Some(KernReturn(3)));
                assert!(vm::OwnMemory::new(PAGE).is_none());
                assert_eq!(import::import_view(p, PAGE, true).at(), p, "no twin");
            });
        }
        assert_eq!(crate::answer::out_of_host_memory_count(), before + 4);
        assert!(v.deallocate(at, 64 * PAGE).is_success());
        assert!(v.deallocate(src, 64 * PAGE).is_success());
        assert!(v.deallocate(ro, PAGE).is_success());
    }

    /// The death names what would be mapped, the mappings, the limit, the reserve and a setting
    /// with room
    #[test]
    fn the_death_names_the_numbers() {
        let line = exhausted("a view of guest pages", 3, 49_140, 65_530, 0, "52000");
        for part in [
            "ran out of kernel mappings",
            "a view of guest pages would add up to 3 to the 49140 they hold",
            "past their budget of 49146, vm.max_map_count 65530 less the 16384 kept",
            "holds 52000 in all",
            "`sysctl -w vm.max_map_count=1048576`",
        ] {
            assert!(line.contains(part), "{part}: {line}");
        }
        let line = exhausted("an import's view", 3, 40_000, 65_530, HEADROOM, "52000");
        for part in [
            "past their budget of 40954, vm.max_map_count 65530 less the 16384 kept",
            "(which holds 52000 in all) and the 8192 kept for what has no other way;",
        ] {
            assert!(line.contains(part), "{part}: {line}");
        }
    }

    /// A view past the budget folds the views no import holds back into dma-bufs first, and is
    /// let through where that freed room; where it freed too little, it dies
    #[test]
    fn a_view_past_the_budget_folds_the_idle_views_first() {
        if !super::super::dma_buf::opens() {
            return;
        }
        let v = LinuxVm::new();
        let n = 64;
        let src = vm::allocate_aliasable(n * PAGE).unwrap();
        let at = v.allocate_anywhere(n * PAGE).unwrap();
        // Backwards, a remap and a mapping a page.
        for i in 0..n {
            let (to, from) = (
                at.wrapping_add(i * PAGE),
                src.wrapping_add((n - 1 - i) * PAGE),
            );
            v.remap_shared_overwrite(to, from, PAGE).unwrap();
        }
        let limit = Limit::room(10);
        assert!(!dies_at_the_budget(|| admit(20, "a test's view")));
        drop(limit);
        let first = at.addr().to_u64();
        assert!(
            files::mappings_between(first, first + n * PAGE) < 4,
            "folded"
        );
        let limit = Limit::room(10);
        assert!(
            dies_at_the_budget(|| admit(20, "a test's view")),
            "nothing left"
        );
        drop(limit);
        assert!(v.deallocate(at, n * PAGE).is_success());
        assert!(v.deallocate(src, n * PAGE).is_success());
    }

    /// A claim for what has another way stops the headroom short of the budget; one for what has
    /// none reaches into the headroom, and neither past the budget
    #[test]
    fn only_what_has_no_other_way_claims_the_headroom() {
        let _limit = Limit::room(HEADROOM + 10);
        assert!(claim(11, Reach::ShortOfHeadroom).is_none());
        assert!(claim(10, Reach::ShortOfHeadroom).is_some());
        assert!(claim(HEADROOM + 10, Reach::Whole).is_some());
        assert!(claim(HEADROOM + 11, Reach::Whole).is_none());
    }

    /// The budget is the limit less the reserve, and none where the limit does not read
    #[test]
    fn the_budget_is_the_limit_less_the_reserve() {
        assert_eq!(budget(65_530), 65_530 - 16_384);
        assert_eq!(budget(1_048_576), 1_048_576 - 16_384);
        assert_eq!(budget(100), 0);
        assert_eq!(budget(u64::MAX), u64::MAX - 16_384);
    }
}
