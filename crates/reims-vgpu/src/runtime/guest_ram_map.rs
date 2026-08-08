//! The process's imports of guest RAM: built once from the shim's spans, and
//! the only place a guest physical address becomes a bindable reference.
//!
//! # What this replaces
//!
//! The dma-buf rail had a cache here, and it had to: `UDMABUF_CREATE_LIST`
//! walked every page, took a kernel reference on each, and cost enough that a
//! digest-bucketed LRU bounded by pinned bytes was worth its own module.
//!
//! Under the host-pointer model there is nothing left to cache.
//! [`crate::runtime::guest_ram::GuestRamImport::slice`] is a range check.
//! What *is* worth holding is the small thing the cache was built around: the
//! imports themselves, one per RAMBlock, made once at first use and held for
//! the VM's lifetime. This module is that, and it is a `Vec` of at most a
//! handful of entries rather than a cache with an eviction policy.
//!
//! # Why the imports are built here and not at device create
//!
//! The backend measures the granularity; the runtime holds the
//! [`HostOps`](crate::runtime::HostOps) that can say where guest RAM lives.
//! Neither side has both, and the device context deliberately does not take a
//! host — see the module doc on [`crate::qemu::host_ops`] for why the runtime
//! keeps it. So the granularity is published by the backend through
//! [`crate::runtime::guest_ram::latch_granularity`] and the spans are fetched
//! here, on the first guest-memory reference of a boot.
//!
//! Building lazily rather than eagerly also gets the ordering right for free:
//! the device exists before any guest command is decoded, so the granularity is
//! always published by the time the first reference is asked for.
//!
//! # What a refusal means
//!
//! Every refusal here puts the whole boot on the copying rails for the
//! addresses it covers, so none of them is a slow path and none may be silent.
//! The one *expected* refusal is [`MapRefusal::NoBackendImport`](crate::runtime::guest_ram_map::MapRefusal::NoBackendImport): a host without
//! the extension, or an operator who set
//! [`crate::env::GUEST_IMPORT`](crate::env::GUEST_IMPORT) off. That one is a
//! statement about the host rather than a loss, so it is reported once on the
//! off channel rather than as a failure per reference.

use crate::runtime::guest_ram::{granularity, GuestRamError, GuestRamImport, GuestRef};
use crate::runtime::host::{GuestRamRegionsError, HostOps};
use std::sync::Arc;

/// Why a guest physical address did not become a bindable reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapRefusal {
    /// No backend published an import granularity: this host cannot import
    /// guest RAM, or an operator asked it not to. Expected, and the state every
    /// copying rail exists for.
    NoBackendImport,
    /// The shim could not say where guest RAM lives. Carries the check that
    /// refused.
    HostRefused(GuestRamRegionsError),
    /// The shim answered, but no span survived being bounded to the granularity
    /// — every region was empty, unmapped, malformed, or shorter than one
    /// granule. Distinct from [`Self::HostRefused`] because the host answered
    /// fine and it is our own bound that rejected every span.
    NoUsableRegion { spans: usize },
    /// The address is not inside any imported span. Guest RAM the GPU can reach
    /// exists, and this address is not in it — a device MMIO address, a hole,
    /// or a page the guest named that this machine does not back.
    GpaNotInAnyImport { gpa: u64 },
    /// The address is in a span, and the length asked for leaves it. Carries the
    /// bound's own reason so the check that refused keeps its name.
    OutsideImport(GuestRamError),
    /// A page list that is not one GPA-contiguous stretch.
    ///
    /// Not a statement that the pages are un-importable — they are all inside
    /// one RAMBlock and each is nameable. It is a statement about the *bind*: a
    /// `VkBuffer` range and a Metal buffer offset are each one offset and one
    /// length, so a surface assembled from four stretches is four of them, and
    /// no consumer takes several yet. Named and counted because how often it
    /// fires is what says whether widening them is worth doing.
    ///
    /// `runs` is what says *how much* widening would cost, and it is the number
    /// to read before building it. "Scattered" is one word for both a window in
    /// two stretches — where a second bind is obviously worth it — and a window
    /// in five hundred, where each run is a couple of pages and the region list
    /// starts to rival the copy it replaces. `pages` alone cannot tell those
    /// apart, and a count of *refusals* tells them apart even less: both read as
    /// one line here. Sampled at the point of refusal, so it bands the reach
    /// actually requested rather than the reach some other rail asked for.
    ///
    /// # What it measured, on an x86 guest
    ///
    /// A driven boot — Safari window drag, 25 s, PCI attach — put **every**
    /// window at almost exactly four pages per run, across three orders of
    /// magnitude of size: 2025 pages in 507 runs for each 1920x1080 writeback,
    /// 813/204, 630/158, 588/147, 256/65, 128/32, 45/12. The ratio holds
    /// because it is not a ratio: the runs are 16 KiB each, and the guest backs
    /// a surface in 16 KiB physically-contiguous granules that are unrelated to
    /// each other. Four 4 KiB x86 pages is what one of those granules looks
    /// like from this side.
    ///
    /// Two consequences worth carrying, because both contradict the obvious
    /// guess. Scattering is **not** a fragmentation artifact that a longer
    /// uptime or a quieter guest would improve — it is the allocator's
    /// granularity, so it is the steady state. And the run count scales with
    /// the surface, so the widening this field exists to price is ~500 ranges
    /// for a full-screen flush and not the handful the word "scattered"
    /// suggests.
    Scattered {
        pages: usize,
        runs: usize,
        first: u64,
    },
}

impl crate::observe::Decline for MapRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoBackendImport => "guest_ram_map_no_backend_import",
            Self::HostRefused(_) => "guest_ram_map_host_refused",
            Self::NoUsableRegion { .. } => "guest_ram_map_no_usable_region",
            Self::GpaNotInAnyImport { .. } => "guest_ram_map_gpa_not_in_any_import",
            Self::Scattered { .. } => "guest_ram_map_scattered",
            // The inner reason is the diagnosis; this wrapper only says where
            // it happened, so it forwards rather than adding a slug of its own.
            Self::OutsideImport(inner) => inner.slug(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NoBackendImport => Vec::new(),
            Self::HostRefused(inner) => {
                let mut f = vec![("host_reason", inner.slug().to_string())];
                f.extend(crate::observe::Decline::fields(inner));
                f
            }
            Self::NoUsableRegion { spans } => vec![("spans", spans.to_string())],
            Self::GpaNotInAnyImport { gpa } => vec![("gpa", format!("{gpa:#x}"))],
            Self::Scattered { pages, runs, first } => vec![
                ("pages", pages.to_string()),
                ("runs", runs.to_string()),
                ("first", format!("{first:#x}")),
            ],
            Self::OutsideImport(inner) => inner.fields(),
        }
    }
}

crate::observe::decline_display!(MapRefusal);

/// The greppable event class for this module's refusals.
const EVENT: &str = "guest_ram_map";

/// The imports this process holds, or the refusal that stopped it building any.
///
/// Resolved once and then read. A `Mutex` rather than a `OnceLock` because a
/// device recreate must be able to drop the imports: the backend's handles die
/// with the device, and an import whose identity outlived them would let a
/// stale [`crate::runtime::guest_ram::GuestSlice`] resolve against a
/// `VkDeviceMemory` that no longer exists.
static MAP: std::sync::Mutex<Option<Resolved>> = std::sync::Mutex::new(None);

#[derive(Debug)]
struct Resolved {
    /// One per usable RAMBlock span, in the order the shim reported them.
    /// Ordinary machines have one or two.
    imports: Vec<Arc<GuestRamImport>>,
    /// Set when the resolution refused, so the next reference does not re-ask
    /// the shim for an answer that will not change. A refusal here is about the
    /// host and the granularity, both of which are fixed for the device's life.
    refusal: Option<MapRefusal>,
}

impl Resolved {
    /// Turn one guest physical address and length into a bindable reference.
    ///
    /// The single implementation, so the one-span and the whole-window entry
    /// points cannot disagree about which import owns a GPA or which refusal a
    /// miss earns. Takes no lock of its own — the caller is already inside
    /// [`with_map`], which is what lets a scattered window resolve every run
    /// under one acquisition.
    ///
    /// [`Self::refusal`] is **not** re-checked here: an entry point asks it once
    /// before walking, and asking again per run would emit the same standing
    /// refusal N times for one window.
    fn reference(&self, gpa: u64, len: u64) -> Result<GuestRef, MapRefusal> {
        let import = self
            .imports
            .iter()
            .find(|i| i.contains_gpa(gpa))
            .ok_or(MapRefusal::GpaNotInAnyImport { gpa })
            .map_err(report_once)?;
        // `slice_for_gpa` emits its own named refusal on the fail channel, so
        // the wrapper forwards the reason rather than adding a second line.
        let slice = import
            .slice_for_gpa(gpa, len)
            .map_err(MapRefusal::OutsideImport)?;
        GuestRef::new(Arc::clone(import), slice).map_err(MapRefusal::OutsideImport)
    }
}

/// Forget every import.
///
/// Called when the backend tears its device down. The next reference rebuilds,
/// against fresh identities, so nothing made before the teardown resolves after
/// it — see [`crate::runtime::guest_ram::ImportId`] for why that matters.
pub fn reset() {
    *MAP.lock().unwrap_or_else(|p| p.into_inner()) = None;
}

/// How many RAMBlock spans the shim reported, and how many bytes they cover.
///
/// The denominator for the backend's *imported* count. A backend imports a span
/// at its first reference and not before, so "one imported" means one of these
/// has been touched — which on a two-span machine is a workload fact, not a
/// defect. Reporting the count alone cannot tell those apart, which is why the
/// census line carries both.
///
/// Counts rather than clones: this runs once a census window, and cloning an
/// `Arc` per span to take a length would be a refcount touch per span per
/// second for a number that does not change after the first reference.
///
/// `(0, 0)` before the first reference of a boot and on a host that cannot
/// import, which is the same reading the census suppresses.
pub fn span_census() -> (usize, u64) {
    MAP.lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
        .map(|r| (r.imports.len(), r.imports.iter().map(|i| i.len()).sum()))
        .unwrap_or((0, 0))
}

/// Every import this process holds, for a backend that needs to create or
/// release its device-side handles.
///
/// Empty before the first reference of a boot and on a host that cannot import.
pub fn imports() -> Vec<Arc<GuestRamImport>> {
    MAP.lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
        .map(|r| r.imports.clone())
        .unwrap_or_default()
}

/// Resolve on the first call of a boot, then run `body` against the result.
///
/// The one place the resolution is built, so no entry point can hold a second
/// copy of "have we asked the host yet".
fn with_map<H: HostOps + ?Sized, R>(host: &mut H, body: impl FnOnce(&Resolved) -> R) -> R {
    let mut guard = MAP.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        *guard = Some(resolve(host));
    }
    body(guard.as_ref().expect("just resolved"))
}

/// The refusal the whole rail is standing on, if there is one.
///
/// An entry point that judges the *shape* of what it was given must ask this
/// first. The order is not cosmetic: on a host with no import every window
/// refuses, and one told it was too fragmented sends a reader hunting for a
/// contiguity problem that a contiguous window would not have fixed either —
/// which is exactly what a driven `REIMS_VGPU_GUEST_IMPORT=off` boot logged
/// before [`reference_for_pages`] asked.
///
/// Public because it is also the cheap early-out: a rail whose next step is an
/// `O(pages)` walk should ask this before paying for one it is going to throw
/// away. That caller must ask *this* rather than re-reading
/// [`crate::runtime::guest_ram::granularity`], which is the same answer for one
/// of the four refusals and silence for the other three.
pub fn standing_refusal<H: HostOps + ?Sized>(host: &mut H) -> Option<MapRefusal> {
    with_map(host, |resolved| resolved.refusal)
}

/// Turn a guest physical address and a length into a bindable reference.
///
/// The whole guest-memory rail goes through here. Building the imports on the
/// first call is why `host` is taken: after that it is not touched.
pub fn reference<H: HostOps + ?Sized>(
    host: &mut H,
    gpa: u64,
    len: u64,
) -> Result<GuestRef, MapRefusal> {
    with_map(host, |resolved| {
        if let Some(refusal) = resolved.refusal {
            return Err(report_once(refusal));
        }
        resolved.reference(gpa, len)
    })
}

/// One stretch of a scattered window: where it starts in the window, and the
/// import reference covering it.
///
/// `window_offset` is a byte offset from the first byte the caller asked for,
/// not from the start of a page and not from the start of the import. It is
/// what a copy's source offset is measured in, which is the only thing a
/// consumer needs and the only thing it may not compute for itself.
#[derive(Debug)]
pub struct GuestWindowRun {
    /// Byte offset of this run's first byte within the requested window.
    pub window_offset: u64,
    /// The bindable reference for this run's bytes.
    pub guest: GuestRef,
}

/// [`reference_for_pages`] for a window that is *not* one contiguous stretch:
/// one reference per maximal GPA run, in window order.
///
/// # Why this exists as well as [`reference_for_pages`]
///
/// A driven x86 boot measured every guest surface at four 4 KiB pages per run —
/// the guest backs a surface in 16 KiB physically-contiguous granules — so a
/// 1920x1080 window is 2025 pages in 507 runs and *always* will be. See
/// [`MapRefusal::Scattered`] for the distribution. A rail that only takes one
/// contiguous stretch therefore never runs on a real workload, which is what
/// the boot found: the import was `supported` and bound 8 KiB in 25 seconds.
///
/// # What the caller owes
///
/// Every returned run is a separate bind. A consumer that issues one GPU copy
/// per run is correct; one that concatenates them is not, because nothing
/// relates two runs' import offsets. The runs tile the window exactly — no
/// gaps, no overlaps, ascending — and the tests below assert that rather than
/// leaving it to be re-derived.
///
/// Runs are **not** bounded here. A bound belongs where the cost is, which is
/// the consumer's region array, and a cap in this function would silently hand
/// back a partial window — the failure mode that loses guest work quietly. A
/// consumer that cannot issue N copies must refuse by name on the count it got.
///
/// # One lock for the whole window
///
/// The resolution is behind a mutex, and this runs on the draw-time buffer rail
/// at ~16 000 windows a second of ~16 runs each. Resolving each run through
/// [`reference`] would take and drop that mutex a quarter of a million times a
/// second for an answer that cannot change inside one call, so the walk happens
/// inside a single [`with_map`] instead. [`reference`] keeps its own lock for
/// the callers that resolve exactly one span.
pub fn references_for_runs<H: HostOps + ?Sized>(
    host: &mut H,
    gpas: &[u64],
    page_size: u64,
    in_page: u64,
    len: u64,
) -> Result<Vec<GuestWindowRun>, MapRefusal> {
    if gpas.is_empty() || page_size == 0 || len == 0 {
        return Err(report_once(MapRefusal::Scattered {
            pages: gpas.len(),
            runs: 0,
            first: gpas.first().copied().unwrap_or(0),
        }));
    }
    // Absolute byte range this window occupies, measured from the first byte of
    // `gpas[0]` — the same frame `in_page` is stated in. Page indices and run
    // boundaries are both in this frame, so no step below re-derives it.
    let window_start = in_page;
    let window_end = in_page.checked_add(len).ok_or(MapRefusal::Scattered {
        pages: gpas.len(),
        runs: 0,
        first: gpas[0],
    })?;

    with_map(host, |resolved| {
        if let Some(refusal) = resolved.refusal {
            return Err(report_once(refusal));
        }
        let mut out = Vec::new();
        for run in reims_vgpu_paging::runs::contig_page_runs(gpas, page_size) {
            let run_start = (run.start as u64) * page_size;
            let run_end = (run.end as u64) * page_size;
            // Clip to the window: the first run usually starts before it (the
            // window begins `in_page` bytes in) and the last usually ends after
            // it.
            let start = run_start.max(window_start);
            let end = run_end.min(window_end);
            if start >= end {
                continue;
            }
            // Within a run the GPAs are contiguous by construction, so one add
            // reaches any byte of it.
            let gpa = gpas[run.start] + (start - run_start);
            out.push(GuestWindowRun {
                window_offset: start - window_start,
                guest: resolved.reference(gpa, end - start)?,
            });
        }
        if out.is_empty() {
            return Err(report_once(MapRefusal::Scattered {
                pages: gpas.len(),
                runs: 0,
                first: gpas[0],
            }));
        }
        Ok(out)
    })
}

/// [`reference`] for a decoded page list: `len` bytes starting `in_page` bytes
/// into `gpas[0]`.
///
/// The one implementation of the contiguity rule, so the sampled, buffer and
/// writeback rails cannot disagree about what a bindable page list is.
pub fn reference_for_pages<H: HostOps + ?Sized>(
    host: &mut H,
    gpas: &[u64],
    page_size: u64,
    in_page: u64,
    len: u64,
) -> Result<GuestRef, MapRefusal> {
    if let Some(refusal) = standing_refusal(host) {
        return Err(report_once(refusal));
    }
    let Some(&first) = gpas.first() else {
        return Err(report_once(MapRefusal::Scattered {
            pages: 0,
            runs: 0,
            first: 0,
        }));
    };
    let contiguous = gpas
        .iter()
        .enumerate()
        .all(|(i, gpa)| *gpa == first + (i as u64) * page_size);
    if !contiguous {
        return Err(report_once(MapRefusal::Scattered {
            pages: gpas.len(),
            runs: reims_vgpu_paging::runs::contig_run_count(gpas, page_size),
            first,
        }));
    }
    reference(host, first + in_page, len)
}

/// Ask the host where guest RAM lives and bound every span to the backend's
/// granularity.
fn resolve<H: HostOps + ?Sized>(host: &mut H) -> Resolved {
    let Some(align) = granularity() else {
        return Resolved {
            imports: Vec::new(),
            refusal: Some(MapRefusal::NoBackendImport),
        };
    };
    let spans = match host.guest_ram_regions() {
        Ok(spans) => spans,
        Err(why) => {
            return Resolved {
                imports: Vec::new(),
                refusal: Some(MapRefusal::HostRefused(why)),
            }
        }
    };
    let count = spans.len();
    // A span this device cannot bound is skipped rather than fatal: a machine
    // with one ordinary RAMBlock and one odd sliver should import the RAMBlock.
    // `GuestRamImport::new` names the check that rejected each skipped one on
    // the fail channel, so a partial import is never silent.
    let imports: Vec<Arc<GuestRamImport>> = spans
        .into_iter()
        .filter_map(|span| GuestRamImport::new(span, align).ok().map(Arc::new))
        .collect();
    let refusal = imports
        .is_empty()
        .then_some(MapRefusal::NoUsableRegion { spans: count });
    // Once per boot, because this is what makes `guest_import_levels`'s
    // denominator interpretable. That line reports `imported/reported` and a
    // reader seeing `1/4` cannot tell which three went untouched, or whether the
    // untouched ones are guest RAM at all — on q35 the reported set is the two
    // halves of `-m` either side of the PCI hole plus whatever smaller writable
    // RAM regions the board exposes. Naming each span's base and length answers
    // that from the log instead of from a comment that would go stale when a
    // board changes.
    //
    // `resolve` runs once per boot (and again only after a device teardown), so
    // this is a handful of lines, not a cadence.
    for (n, import) in imports.iter().enumerate() {
        crate::observe::off(format!(
            "guest_ram_span n={n}/{count} gpa={:#x} len={} mib={}",
            import.gpa_base(),
            import.len(),
            import.len() / (1024 * 1024),
        ));
    }
    Resolved { imports, refusal }
}

/// Emit `refusal` and hand it back.
///
/// Deduped by slug: these are per-reference and a decode path that names an
/// unbacked address once will name it every frame.
/// [`MapRefusal::NoBackendImport`] goes to the off channel — it is the host
/// saying what it is, not a loss of guest work — and everything else to the
/// fail channel.
fn report_once(refusal: MapRefusal) -> MapRefusal {
    let line = crate::observe::Emit::decline(EVENT, &refusal);
    match refusal {
        MapRefusal::NoBackendImport => {
            if crate::observe::first_sight("guest_ram_map_no_backend_import", 0) {
                line.off();
            }
        }
        _ => line.fail_once(0),
    }
    refusal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::guest_ram::{forget_granularity, latch_granularity, GuestRamRegion};

    /// The whole module is process-global, and so is the granularity latch.
    /// Every test here takes this and restores both.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct Spans(Vec<GuestRamRegion>);

    impl HostOps for Spans {
        fn mono_ns(&self) -> u64 {
            0
        }
        fn enqueue(&mut self, _action: crate::runtime::host::HostAction) {}
        fn schedule_bh(&mut self) {}
        fn guest_ram_regions(&mut self) -> Result<Vec<GuestRamRegion>, GuestRamRegionsError> {
            Ok(self.0.clone())
        }
    }

    struct Refusing;

    impl HostOps for Refusing {
        fn mono_ns(&self) -> u64 {
            0
        }
        fn enqueue(&mut self, _action: crate::runtime::host::HostAction) {}
        fn schedule_bh(&mut self) {}
        fn guest_ram_regions(&mut self) -> Result<Vec<GuestRamRegion>, GuestRamRegionsError> {
            Err(GuestRamRegionsError::NoRam)
        }
    }

    /// Two spans with a hole between them, which is the shape of an x86 machine
    /// with a PCI hole and the reason the lookup is a search rather than a
    /// single subtraction.
    fn two_spans() -> Spans {
        Spans(vec![
            GuestRamRegion {
                gpa_base: 0,
                host_va: 0x7f00_0000_0000,
                len: 0x8000_0000,
            },
            GuestRamRegion {
                gpa_base: 0x1_0000_0000,
                host_va: 0x7f80_0000_0000,
                len: 0x8000_0000,
            },
        ])
    }

    fn with_granularity<R>(align: Option<u64>, body: impl FnOnce() -> R) -> R {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset();
        match align {
            Some(a) => latch_granularity(a),
            None => forget_granularity(),
        }
        let out = body();
        reset();
        forget_granularity();
        out
    }

    /// An address in either span resolves, and the offset it resolves to is the
    /// one inside *that* span — not a distance from the first one. Getting this
    /// wrong on a machine with a PCI hole binds the GPU 4 GiB away from the
    /// bytes the guest named, inside a live import, where no bound would catch
    /// it.
    #[test]
    fn an_address_resolves_against_the_span_that_backs_it() {
        with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            let low = reference(&mut host, 0x2000, 0x100).expect("in the first span");
            assert_eq!(low.bound().expect("checked").offset, 0x2000);

            let high = reference(&mut host, 0x1_0000_2000, 0x100).expect("in the second span");
            assert_eq!(
                high.bound().expect("checked").offset,
                0x2000,
                "the offset is into the second import, not from the first span's base"
            );
            assert_ne!(low.import().id(), high.import().id());
        });
    }

    /// The span census counts every span the shim reported, not the ones a
    /// workload happened to touch.
    ///
    /// This is the denominator of the `guest_import_levels` census line, and it
    /// is the whole reason that line carries two terms. A backend imports a span
    /// at its first reference, so on this two-span machine — the shape q35 has
    /// with a PCI hole, which is what `vm/boot-x86.sh` boots — a workload that
    /// only ever touches high memory leaves the imported count at one. Reported
    /// against a denominator of one that would read as half of guest RAM having
    /// gone missing; against two it reads as lazy, which is what it is.
    #[test]
    fn the_span_census_counts_what_the_shim_reported_not_what_was_touched() {
        with_granularity(Some(0x1000), || {
            assert_eq!(
                span_census(),
                (0, 0),
                "nothing is asked before the first reference"
            );
            let mut host = two_spans();
            reference(&mut host, 0x1_0000_2000, 0x100).expect("in the second span");
            assert_eq!(
                span_census(),
                (2, 0x1_0000_0000),
                "both spans are reported though only the second was referenced"
            );
        });
    }

    /// The imports are built once. The shim's answer does not change, and
    /// re-asking would be an address-space walk per guest reference.
    #[test]
    fn the_host_is_asked_once_however_many_references_follow() {
        with_granularity(Some(0x1000), || {
            struct Counting(std::cell::Cell<usize>);
            impl HostOps for Counting {
                fn mono_ns(&self) -> u64 {
                    0
                }
                fn enqueue(&mut self, _a: crate::runtime::host::HostAction) {}
                fn schedule_bh(&mut self) {}
                fn guest_ram_regions(
                    &mut self,
                ) -> Result<Vec<GuestRamRegion>, GuestRamRegionsError> {
                    self.0.set(self.0.get() + 1);
                    Ok(vec![GuestRamRegion {
                        gpa_base: 0,
                        host_va: 0x7f00_0000_0000,
                        len: 0x8000,
                    }])
                }
            }
            let mut host = Counting(std::cell::Cell::new(0));
            for _ in 0..5 {
                reference(&mut host, 0x1000, 8).expect("inside");
            }
            assert_eq!(host.0.get(), 1);
            assert_eq!(imports().len(), 1);
        });
    }

    /// A host with no import capability refuses every reference by name, and
    /// never asks the shim at all. This is the ordinary state on a host without
    /// the extension, so it must not read as a failure of the guest's work.
    #[test]
    fn without_a_published_granularity_nothing_is_asked_and_nothing_resolves() {
        with_granularity(None, || {
            let mut host = two_spans();
            assert_eq!(
                reference(&mut host, 0x2000, 8).err(),
                Some(MapRefusal::NoBackendImport)
            );
            assert!(imports().is_empty());
        });
    }

    /// The shim's own refusal is carried through with its name, rather than
    /// collapsing into "no imports". "This machine has no RAM span" and "this
    /// build's shim is too old" are different things to go fix.
    #[test]
    fn the_hosts_own_refusal_keeps_its_name() {
        with_granularity(Some(0x1000), || {
            assert_eq!(
                reference(&mut Refusing, 0x2000, 8).err(),
                Some(MapRefusal::HostRefused(GuestRamRegionsError::NoRam))
            );
        });
    }

    /// An address in the hole between two spans is refused by name. It is not a
    /// bound violation — no import claims it — and reporting it as one would
    /// send a reader looking for arithmetic that is not there.
    #[test]
    fn an_address_no_span_backs_is_refused_by_name() {
        with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            let hole = 0x8000_0000;
            assert_eq!(
                reference(&mut host, hole, 8).err(),
                Some(MapRefusal::GpaNotInAnyImport { gpa: hole })
            );
        });
    }

    /// A length that leaves the span it started in is refused by the bound, with
    /// the bound's own reason. The next span's bytes are elsewhere in host
    /// memory, so running off the end of one import is exactly the stray this
    /// device is bounded against.
    #[test]
    fn a_length_that_runs_off_the_end_of_a_span_is_refused_by_the_bound() {
        with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            let last_page = 0x8000_0000 - 0x1000;
            assert!(reference(&mut host, last_page, 0x1000).is_ok());
            assert!(matches!(
                reference(&mut host, last_page, 0x2000),
                Err(MapRefusal::OutsideImport(
                    GuestRamError::SliceEndPastImport { .. }
                ))
            ));
        });
    }

    /// A span nothing can bound is skipped, and the ones that can be are still
    /// imported. Failing the whole map on one odd sliver would put a machine
    /// with an ordinary RAMBlock beside it on the copying rails for no reason.
    #[test]
    fn one_unusable_span_does_not_cost_the_usable_ones() {
        with_granularity(Some(0x1000), || {
            let mut host = Spans(vec![
                // Shorter than one granule: nothing to import.
                GuestRamRegion {
                    gpa_base: 0,
                    host_va: 0x7f00_0000_0000,
                    len: 0x400,
                },
                GuestRamRegion {
                    gpa_base: 0x1_0000_0000,
                    host_va: 0x7f80_0000_0000,
                    len: 0x8000,
                },
            ]);
            assert!(reference(&mut host, 0x1_0000_0000, 8).is_ok());
            assert_eq!(imports().len(), 1);
        });
    }

    /// Every span unusable is its own refusal, distinct from the host refusing:
    /// the host answered fine and our own bound rejected all of it.
    #[test]
    fn every_span_unusable_is_a_refusal_of_its_own() {
        with_granularity(Some(0x1000), || {
            let mut host = Spans(vec![GuestRamRegion {
                gpa_base: 0,
                host_va: 0x7f00_0000_0000,
                len: 0x400,
            }]);
            assert_eq!(
                reference(&mut host, 0, 8).err(),
                Some(MapRefusal::NoUsableRegion { spans: 1 })
            );
        });
    }

    /// A device recreate drops the imports, and the rebuilt ones carry new
    /// identities. A reference taken before the teardown must not resolve
    /// against the replacement, because the backend handle it named is gone.
    #[test]
    fn a_reset_rebuilds_against_fresh_identities() {
        with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            let before = reference(&mut host, 0x2000, 8).expect("inside");
            reset();
            let after = reference(&mut host, 0x2000, 8).expect("inside");
            assert_ne!(before.import().id(), after.import().id());
            assert!(matches!(
                after
                    .import()
                    .resolve(&before.import().slice(0, 8).unwrap()),
                Err(GuestRamError::SliceForeignImport { .. })
            ));
        });
    }

    /// The refusals reach the always-on log, and the expected one does not
    /// reach the fail channel.
    #[test]
    fn refusals_are_visible_and_the_expected_one_is_not_a_failure() {
        with_granularity(Some(0x1000), || {
            let capture = crate::observe::FailCapture::start();
            let mut host = two_spans();
            let _ = reference(&mut host, 0x8000_0000, 8);
            let line = capture.one(EVENT);
            assert!(
                line.contains("reason=guest_ram_map_gpa_not_in_any_import"),
                "{line}"
            );
            assert!(line.contains("gpa=0x80000000"), "{line}");
            assert!(!line.starts_with("OFF "), "a lost reference is a failure");
        });
        with_granularity(None, || {
            let capture = crate::observe::FailCapture::start();
            let mut host = two_spans();
            let _ = reference(&mut host, 0x2000, 8);
            // Not `capture.one(EVENT)`: an off-channel line's first token is
            // the literal `OFF`, which is the same thing the fail-log reading
            // notes in `AGENTS.md` warn about when ranking `reason=`.
            let lines = capture.lines();
            assert_eq!(
                lines,
                vec!["OFF guest_ram_map reason=guest_ram_map_no_backend_import"],
                "a host without the extension has not lost guest work"
            );
        });
    }

    /// The `runs` a `Scattered` refusal reports is the coalescer's answer, not
    /// a second count that agrees with it today.
    ///
    /// This number decides whether widening the bind to N ranges is worth
    /// building, and how expensive it would be — a window in two stretches and
    /// a window in five hundred both refuse identically without it. So it is
    /// asserted against `reims_vgpu_paging::runs::contig_run_count` on the same input rather
    /// than against a literal: a hand-written expectation here would be a
    /// second implementation of the coalescing rule, and the one that drifts is
    /// always the copy nothing else reads.
    #[test]
    fn a_scattered_refusal_reports_the_run_count_the_coalescer_finds() {
        const PAGE: u64 = 4096;
        // Nine pages in four stretches: 3 + 1 + 2 + 3.
        let gpas: Vec<u64> = vec![
            0x1000, 0x2000, 0x3000, // run 1
            0x9000, // run 2
            0x20000, 0x21000, // run 3
            0x50000, 0x51000, 0x52000, // run 4
        ];
        let expected = reims_vgpu_paging::runs::contig_run_count(&gpas, PAGE);
        assert_eq!(expected, 4, "fixture must actually be four stretches");

        with_granularity(Some(PAGE), || {
            let capture = crate::observe::FailCapture::start();
            let mut host = two_spans();
            let err = reference_for_pages(&mut host, &gpas, PAGE, 0, 8).unwrap_err();
            assert_eq!(
                err,
                MapRefusal::Scattered {
                    pages: gpas.len(),
                    runs: expected,
                    first: 0x1000,
                }
            );
            let line = capture.one(EVENT);
            assert!(line.contains("reason=guest_ram_map_scattered"), "{line}");
            assert!(
                line.contains("runs=4"),
                "the run count reaches the log: {line}"
            );
            assert!(line.contains("pages=9"), "{line}");
        });

        // One stretch is not scattered, so it must not refuse for this reason —
        // otherwise `runs` would be reporting on a population that includes the
        // case the widening is supposed to leave alone.
        let contiguous: Vec<u64> = (0..4).map(|i| 0x1000 + i * PAGE).collect();
        assert_eq!(
            reims_vgpu_paging::runs::contig_run_count(&contiguous, PAGE),
            1
        );
        with_granularity(Some(PAGE), || {
            let mut host = two_spans();
            let out = reference_for_pages(&mut host, &contiguous, PAGE, 0, 8);
            assert!(
                !matches!(out, Err(MapRefusal::Scattered { .. })),
                "one contiguous stretch must not refuse as scattered"
            );
        });
    }

    /// The runs tile the requested window exactly: ascending, no gap, no
    /// overlap, and summing to the length asked for.
    ///
    /// This is the property that keeps a scattered writeback from corrupting
    /// guest memory, and every part of it is a real failure mode rather than a
    /// tidiness rule. A gap leaves a band of the guest's surface holding the
    /// previous frame while the rest updates. An overlap writes one stretch of
    /// guest RAM twice from two different source offsets, so the winner is
    /// whichever copy the GPU retires last. A sum short of `len` is a torn
    /// frame; a sum past it is a write beyond the window the caller was given
    /// — which the import's own bound would catch, but only after the plan had
    /// already decided to make it.
    ///
    /// Asserted as a walk over the returned runs rather than as expected
    /// tuples, so the test states the invariant instead of restating one
    /// fixture's arithmetic.
    #[test]
    fn scattered_runs_tile_the_window_exactly() {
        const PAGE: u64 = 4096;
        // Four stretches, deliberately uneven: 3 + 1 + 2 + 3 pages.
        let gpas: Vec<u64> = vec![
            0x1000, 0x2000, 0x3000, // run 1
            0x9000, // run 2
            0x20000, 0x21000, // run 3
            0x50000, 0x51000, 0x52000, // run 4
        ];
        // Start part-way into the first page and end part-way into the last, so
        // the head and tail clips are both exercised. Neither end is page
        // aligned, which is the normal case for a sample window.
        let in_page = 100u64;
        let len = PAGE * 8 + 55;

        with_granularity(Some(PAGE), || {
            let mut host = two_spans();
            let runs = references_for_runs(&mut host, &gpas, PAGE, in_page, len)
                .expect("a scattered window still resolves");
            assert_eq!(runs.len(), 4, "one reference per contiguous stretch");

            let mut expected_offset = 0u64;
            for run in &runs {
                assert_eq!(
                    run.window_offset, expected_offset,
                    "runs must be ascending and leave no gap"
                );
                let bound = run.guest.bound().expect("each run resolves in its import");
                // `requested` is what the caller asked for; `bound_len` may be
                // larger because the import rounds to its granularity. The tiling
                // is a statement about the former.
                expected_offset += run.guest.requested();
                assert!(bound.len >= run.guest.requested());
            }
            assert_eq!(
                expected_offset, len,
                "the runs together must cover exactly the window asked for"
            );
        });
    }

    /// A window inside one stretch yields one run, so the widened path is not a
    /// different answer for the case that already worked.
    ///
    /// Worth pinning because the two entry points now compute the same thing by
    /// different routes: a divergence would mean a contiguous surface landed
    /// differently depending on which rail asked, which is exactly the "two arms
    /// consume one wire form" class.
    #[test]
    fn a_contiguous_window_is_one_run_and_agrees_with_the_single_reference() {
        const PAGE: u64 = 4096;
        let gpas: Vec<u64> = (0..4).map(|i| 0x1000 + i * PAGE).collect();
        with_granularity(Some(PAGE), || {
            let mut host = two_spans();
            let runs = references_for_runs(&mut host, &gpas, PAGE, 64, PAGE * 3)
                .expect("contiguous resolves");
            assert_eq!(runs.len(), 1);
            assert_eq!(runs[0].window_offset, 0);

            let single = reference_for_pages(&mut host, &gpas, PAGE, 64, PAGE * 3)
                .expect("and so does the single-range entry point");
            assert_eq!(
                runs[0].guest.bound().expect("bound"),
                single.bound().expect("bound"),
                "one stretch must bind identically whichever entry point asked"
            );
        });
    }

    /// A host with no import refuses the widened path by the same name, on the
    /// same channel, as the narrow one.
    ///
    /// The widened path calls `reference` once per run, so a naive
    /// implementation reports the host-wide refusal once per run — five hundred
    /// identical lines for one fact about the machine. `report_once` is what
    /// stops that, and this asserts it still does through the new caller.
    #[test]
    fn no_backend_import_stays_one_line_through_the_widened_path() {
        const PAGE: u64 = 4096;
        let gpas: Vec<u64> = vec![0x1000, 0x2000, 0x9000, 0x20000];
        with_granularity(None, || {
            let capture = crate::observe::FailCapture::start();
            let mut host = two_spans();
            let err = references_for_runs(&mut host, &gpas, PAGE, 0, PAGE).unwrap_err();
            assert_eq!(err, MapRefusal::NoBackendImport);
            let lines = capture.lines();
            assert_eq!(
                lines,
                vec!["OFF guest_ram_map reason=guest_ram_map_no_backend_import"],
                "one statement about the host, not one per run"
            );
        });
    }

    /// Both page-list entry points answer a host with no import the same way.
    ///
    /// They consume one wire form — a decoded page list — and until the
    /// `standing_refusal` gate went in they disagreed about a host, not about
    /// their input: `references_for_runs` reached `reference` and got
    /// `NoBackendImport` on the off channel, while `reference_for_pages`
    /// judged contiguity first and put `guest_ram_map_scattered` on the
    /// **fail** channel — a claim of lost guest work naming a cause that was
    /// not the cause. A driven `REIMS_VGPU_GUEST_IMPORT=off` boot logged
    /// exactly one of those lines, which is what this pins.
    ///
    /// The assertion is that the two arms agree, not that either says a
    /// particular thing, because a divergence is what the fail log cannot
    /// survive.
    #[test]
    fn both_page_list_arms_name_the_host_and_not_the_window() {
        const PAGE: u64 = 4096;
        // Scattered on purpose: this is the input that used to reach the
        // contiguity refusal before anything asked whether an import existed.
        let gpas: Vec<u64> = vec![0x1000, 0x2000, 0x9000, 0x20000];
        with_granularity(None, || {
            let capture = crate::observe::FailCapture::start();
            let mut host = two_spans();

            let narrow = reference_for_pages(&mut host, &gpas, PAGE, 0, PAGE).unwrap_err();
            let wide = references_for_runs(&mut host, &gpas, PAGE, 0, PAGE).unwrap_err();
            assert_eq!(
                narrow, wide,
                "one page list, one host, two entry points: the reason cannot depend on which asked"
            );
            assert_eq!(narrow, MapRefusal::NoBackendImport);

            assert_eq!(
                capture.lines(),
                vec!["OFF guest_ram_map reason=guest_ram_map_no_backend_import"],
                "a host that cannot import has not lost guest work to fragmentation"
            );
        });
    }
}
