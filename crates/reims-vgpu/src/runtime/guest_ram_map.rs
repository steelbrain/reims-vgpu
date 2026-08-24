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
//! imports themselves, made once at first use and held for the VM's lifetime.
//! This module is that, and it is a sorted `Vec` of a dozen or so entries rather
//! than a cache with an eviction policy.
//!
//! A RAMBlock is imported in **chunks** when it is longer than the backend's
//! queried single-allocation limit. A window resolves against whichever import
//! backs its GPA, and one straddling two of them groups into two `VkBuffer`
//! sources, because a RAMBlock boundary could already split one.
//!
//! # Why the imports are built here and not at device create
//!
//! The backend measures the granularity; the runtime holds the
//! [`GuestRamProvider`](crate::runtime::host::GuestRamProvider) that can say where
//! guest RAM lives.
//! Neither side has both, and the device context deliberately does not take a
//! host — see the module doc on [`crate::qemu::host_ops`] for why the runtime
//! keeps it. So the granularity is published by the backend through
//! [`crate::runtime::guest_ram::latch_import_limits`] — together with the
//! largest import that backend's heaps could hold — and the spans are fetched
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

use crate::runtime::guest_ram::{
    granularity, import_budget, import_span_max, GuestRamError, GuestRamImport, GuestRamRegion,
    GuestRef,
};
use crate::runtime::host::{GuestRamProvider, GuestRamRegionsError};
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
    /// The spans are importable and this guest is larger than the roomiest heap
    /// on the host's GPU, so nothing may import any of them.
    ///
    /// # Why the whole map refuses rather than the block that does not fit
    ///
    /// An import is a `VkDeviceMemory` charged to one heap, and a submission
    /// that names it makes the driver keep all of it resident. On a part whose
    /// heaps are a fraction of the guest — an APU with a few gigabytes of
    /// carve-out against a `-m 16G` guest — the kernel refuses to validate the
    /// allocation and the submission fails, which arrives as a **lost device**
    /// and a dead guest rather than as a slow rail. That has been reported from
    /// the field on `radv`/`amdgpu` (`Not enough memory for command submission`,
    /// then a lost context), and the reporter's own fix was to set
    /// [`crate::env::GUEST_IMPORT`] off by hand. This makes the device reach the
    /// same state without being told to.
    ///
    /// Refusing the whole map rather than the oversized block is what makes it
    /// safe: the copying rails are selected by a page having no [`GuestRef`], and
    /// a *partial* import would leave the writeback paths holding references
    /// into one RAMBlock and none into another, which is a hard error at those
    /// sites and not a fallback. All or nothing keeps the boot on the one arm
    /// that is tested end to end.
    ///
    /// The comparison is against the sum, because every import is live at once
    /// for the VM's lifetime and a submission may name any of them.
    ///
    /// # Its relationship to the per-import check, which is the exact one
    ///
    /// The backend publishes this budget as the roomiest heap an import can be
    /// *charged to* — the same population of memory types
    /// `reims-vgpu-vulkan`'s memory selector will choose from, since every
    /// import goes through it carrying one class's required flags. So a sum that
    /// passes here has a heap that each individual chunk fits, and the exact
    /// per-allocation check at the pick — which refuses rather than making a
    /// call Vulkan declares invalid — agrees with this one by construction
    /// rather than by coincidence. Publishing the maximum over *every* heap
    /// instead, which this once did, breaks that: a part whose device-local heap
    /// is twice its host-visible heap passes here with room to spare and then
    /// refuses at every pick, which is the partial import this refusal exists to
    /// prevent.
    ///
    /// This is a heap-*capacity* test and not a residency one, so it is a lower
    /// bound: a host that passes it can still be too full to import. It catches
    /// the direction that has been seen to kill a guest.
    ///
    /// # What would give such a host the fast rail back
    ///
    /// Not this refusal, which governs the optional whole-VM import. Resource-
    /// sized stable aliases are admitted independently by
    /// [`crate::runtime::guest_ram::host_allocation_import_align`], so a guest
    /// allocation that fits can still take the direct rail without making
    /// unrelated RAM resident.
    ImportExceedsHeap { needed: u64, budget: u64 },
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
            Self::ImportExceedsHeap { .. } => "guest_ram_map_import_exceeds_heap",
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
            Self::ImportExceedsHeap { needed, budget } => vec![
                ("needed_mb", (needed >> 20).to_string()),
                ("budget_mb", (budget >> 20).to_string()),
            ],
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
        // Binary search, not a linear scan. The imports are sorted by
        // `gpa_base`, so the last one whose base is at or below `gpa` is the
        // only one that can contain it — `partition_point` names that index and
        // `contains_gpa` still decides, so an address in a hole between two
        // imports is refused exactly as before.
        //
        // A scan was right while a machine had one or two imports. Chunking a
        // RAMBlock at the span ceiling makes it eight to a dozen on an ordinary
        // guest, and this runs once per run of a scattered window — 9 to 32 runs
        // per bind, thousands of binds a second — so the growth would land in
        // the hot path. This makes the count stop mattering instead of trading
        // one host's correctness for another's throughput.
        let import = self
            .imports
            .partition_point(|i| i.gpa_base().is_some_and(|base| base <= gpa))
            .checked_sub(1)
            .map(|last| &self.imports[last])
            .filter(|i| i.contains_gpa(gpa))
            .ok_or(MapRefusal::GpaNotInAnyImport { gpa })
            .map_err(report_once)?;
        // `slice_for_gpa` emits its own named refusal on the fail channel, so
        // the wrapper forwards the reason rather than adding a second line.
        let slice = import
            .slice_for_gpa(gpa, len)
            .map_err(MapRefusal::OutsideImport)?;
        GuestRef::new(Arc::clone(import), slice).map_err(MapRefusal::OutsideImport)
    }

    /// The exclusive end GPA of the import backing `gpa`, or `None` if nothing
    /// backs it.
    ///
    /// Exists so a caller can split a contiguous guest stretch at the seam
    /// between two imports instead of being refused at it — see
    /// [`references_for_runs`]. Deliberately not a public entry point: an import
    /// boundary is this module's own bookkeeping, and the only thing outside it
    /// may do with one is stop at it.
    fn import_end(&self, gpa: u64) -> Option<u64> {
        self.imports
            .partition_point(|i| i.gpa_base().is_some_and(|base| base <= gpa))
            .checked_sub(1)
            .map(|last| &self.imports[last])
            .filter(|i| i.contains_gpa(gpa))
            .and_then(|i| i.gpa_base().map(|base| base + i.len()))
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

/// Take the whole guest-RAM import now, so the guest's first draw does not pay
/// for it.
///
/// # Two steps, and only the second one costs
///
/// Asking the host where its RAMBlocks are ([`resolve`]) is a handful of shim
/// calls. Handing each of those mappings to the GPU is `vkAllocateMemory` with
/// a host pointer chained, which is where a driver that pins takes a reference
/// on every page of guest RAM — seconds, proportional to the RAM the VM was
/// given, and measured per block by
/// [`reims_vgpu_vulkan::engine::warm_guest_ram_imports`].
///
/// Both were lazy and both landed on the guest's first `gather`, inside its
/// first draw, inside a display transaction the guest abandons after 1000 ms.
/// Moving only the first one bought nothing measurable, which is the finding
/// that located the second: `guest_ram_span` moved a second earlier and
/// `gather_us` did not move at all. Called from the guest driver's
/// protocol-version handshake, both now run before the guest has a display pipe
/// to arm a watchdog on, and every later caller finds the answer already there.
///
/// **It must never cache a negative.** [`resolve`] answers `NoBackendImport`
/// when no backend has published a granularity yet, and that answer is latched
/// in `MAP` for the rest of the boot — so warming before the backend is up
/// would turn a capable host into one that refuses every window, which is the
/// opposite of the intent and would look like a host that lacks the extension.
/// The guard is the same question `resolve` asks first, and asking it here
/// leaves the lazy path to handle a backend that is genuinely late.
///
/// The device-side half is Vulkan-only because only Vulkan has a device-side
/// against unified memory and holds no per-RAMBlock import to warm.
pub fn warm<H: GuestRamProvider + ?Sized>(
    host: &mut H,
    executor: &dyn crate::runtime::executor::Executor,
) {
    if granularity().is_none() {
        return;
    }
    let already = MAP.lock().unwrap_or_else(|p| p.into_inner()).is_some();
    if !already {
        with_map(host, |_| ());
    }
    {
        let imports = imports();
        if !imports.is_empty() {
            let (warmed, bytes) = executor.warm_guest_ram_imports(&imports);
            if warmed > 0 {
                crate::observe::off(format!(
                    "guest_ram_warm blocks={warmed} bytes={bytes} spans={}",
                    imports.len()
                ));
            }
        }
    }
}

/// Resolve on the first call of a boot, then run `body` against the result.
///
/// The one place the resolution is built, so no entry point can hold a second
/// copy of "have we asked the host yet".
fn with_map<H: GuestRamProvider + ?Sized, R>(host: &mut H, body: impl FnOnce(&Resolved) -> R) -> R {
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
pub fn standing_refusal<H: GuestRamProvider + ?Sized>(host: &mut H) -> Option<MapRefusal> {
    with_map(host, |resolved| resolved.refusal)
}

/// Turn a guest physical address and a length into a bindable reference.
///
/// The whole guest-memory rail goes through here. Building the imports on the
/// first call is why `host` is taken: after that it is not touched.
pub fn reference<H: GuestRamProvider + ?Sized>(
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
pub use reims_vgpu_memory::GuestWindowRun;

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
pub fn references_for_runs<H: GuestRamProvider + ?Sized>(
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
            // GPA-contiguous is not import-contiguous. A RAMBlock is imported in
            // chunks, so a stretch the guest laid out as one run can cross a
            // seam between two of them — and a `GuestRef` is an offset into one
            // import, so it cannot describe both sides.
            //
            // Split at the seam rather than refuse at it. The consumers already
            // take a list and a RAMBlock boundary has always been able to
            // produce one, so two runs here are indistinguishable from two the
            // guest's own page plan produced. Refusing instead would drop the
            // *whole window* to the copying rail — a named, safe decline, but
            // one that fires on roughly one writeback in 250 for no reason the
            // guest could see, which is a chunk size leaking into throughput.
            //
            // Within a run the GPAs are contiguous by construction, so one add
            // reaches any byte of it.
            let mut piece = start;
            while piece < end {
                let gpa = gpas[run.start] + (piece - run_start);
                // `None` means nothing backs this address at all: hand the whole
                // remainder to `reference` so it names that refusal rather than
                // this loop inventing a second one.
                let piece_end = match resolved.import_end(gpa) {
                    Some(import_end) if import_end > gpa => end.min(piece + (import_end - gpa)),
                    _ => end,
                };
                out.push(GuestWindowRun {
                    window_offset: piece - window_start,
                    guest: resolved.reference(gpa, piece_end - piece)?,
                });
                piece = piece_end;
            }
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
pub fn reference_for_pages<H: GuestRamProvider + ?Sized>(
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
///
/// # This used to run on the guest's first draw, and it is NOT the two seconds
///
/// It was lazy — the first `reference_for_pages` triggered it — and it now runs
/// at the guest driver's protocol handshake instead ([`warm`]). **That move was
/// measured and it did not shift the stall**, which is what located the real
/// one: asking the host where its RAM is costs nothing, and handing those
/// mappings to the GPU costs everything. The evidence is one timestamp —
/// `guest_ram_span`, emitted once per boot by this function, moved from t=56453
/// to t=55342 while `gather_us` on the first frame stayed at 2 180 583 over the
/// same six gathers.
///
/// The seconds are `vkAllocateMemory` with the host pointer chained, measured
/// per RAMBlock at [`reims_vgpu_vulkan::engine::warm_guest_ram_imports`],
/// which is now also warmed from [`warm`]. The table below is the state before
/// that, kept because its second row is what ruled out a per-byte cost and sent
/// the search to one-time setup:
///
/// ```text
///                 draw_stall     stage_us     gather_us  gather_n  gather_b
/// macos-11 first   2 028 844    2 022 252     2 022 259         6   1 176 768
/// macos-11 later           —            —            75        61  13 545 376
/// macos-13 first   1 959 875    1 951 567     1 951 562         4     523 904
/// macos-13 later           —            —           105       104  15 318 048
/// ```
///
/// Six gathers of 1.1 MB taking two seconds and sixty-one gathers of 13 MB
/// taking 75 µs is not a gather cost; it is one-time setup charged to whoever
/// arrives first. The same boots report it as a `sync_exec_lock_hold` of
/// ~2 000 000 µs over one to three draws.
///
/// **The guest has a one-second watchdog behind this.** Its display pipe waits
/// on a submitted display transaction and gives up after 1000 ms, so a first
/// frame that takes two seconds blows it on every boot of every rail. Both
/// rails measured here do blow it; the macos-13 guest recovers and the macos-11
/// guest does not, and on macos-11 that is the whole visible failure — the
/// transaction stays pending, WindowServer stops answering, and the session
/// never starts.
///
/// The same driven boot then timed the two halves of the import separately and
/// read `probe_us=0` beside `alloc_us=2 493 029` for a 15 032 385 536-byte
/// RAMBlock and `alloc_us=309 796` for a 2 146 435 072-byte one — the whole
/// stall, in the one call the first gather was the first to reach. That is what
/// [`warm`] now takes at the handshake.
///
/// Timings above are wall clock on a shared host and are upper bounds; the
/// counts and byte totals are not.
/// Split one RAMBlock into consecutive regions of at most `span_max` bytes.
///
/// The regions tile the block exactly and in ascending order: no byte is
/// dropped, none is covered twice, and the last one is whatever remains. A block
/// already inside the ceiling comes back as itself, so a host that needs no
/// chunking pays one comparison and allocates the same one-element shape it
/// always had.
///
/// Alignment is deliberately *not* applied here. `GuestRamImport::new` trims
/// each region to the device's granularity and names its own refusal when a
/// region cannot survive that — doing it twice would be two spellings of one
/// rule, and this one has no granularity in hand. The consequence is that
/// `span_max` must itself be a multiple of the granularity, which is why the
/// backend masks it before publishing and why a ceiling below the granularity is
/// refused outright rather than clamped.
fn chunk_span(span: GuestRamRegion, span_max: u64) -> Vec<GuestRamRegion> {
    if span_max == 0 || span.len <= span_max {
        return vec![span];
    }
    let mut out = Vec::with_capacity((span.len / span_max) as usize + 1);
    let mut done = 0u64;
    while done < span.len {
        let len = span_max.min(span.len - done);
        out.push(GuestRamRegion {
            host_va: span.host_va + done,
            gpa_base: span.gpa_base + done,
            len,
        });
        done += len;
    }
    out
}

fn resolve<H: GuestRamProvider + ?Sized>(host: &mut H) -> Resolved {
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
    //
    // Each block is imported in chunks no larger than the API-derived span the
    // backend published. Nothing else has to change: a window already resolves
    // against whichever import backs its GPA, and one straddling two of them
    // already groups into two `VkBuffer` sources, because a RAMBlock boundary
    // has always been able to split one.
    let span_max = import_span_max().unwrap_or(u64::MAX);
    let mut imports: Vec<Arc<GuestRamImport>> = spans
        .into_iter()
        .flat_map(|span| chunk_span(span, span_max))
        .filter_map(|span| GuestRamImport::new(span, align).ok().map(Arc::new))
        .collect();
    // `reference` binary-searches this, so the order is load-bearing rather than
    // cosmetic. The shim reports blocks in ascending GPA and `chunk_span` keeps
    // that, so this is a no-op on every machine seen so far — it is here because
    // the search would silently answer `GpaNotInAnyImport` for a live address if
    // a future shim ever reported them out of order.
    imports.sort_by_key(|i| i.gpa_base());
    // Every import is live for the VM's lifetime and any submission may name any
    // of them, so what has to fit is the sum and not the largest block. A guest
    // that does not fit takes the copying rails whole rather than in part — see
    // [`MapRefusal::ImportExceedsHeap`] for why a partial import is the one
    // outcome that is worse than either.
    let needed: u64 = imports.iter().map(|i| i.len()).sum();
    let over_budget = import_budget().filter(|budget| needed > *budget);
    if let Some(budget) = over_budget {
        return Resolved {
            imports: Vec::new(),
            refusal: Some(MapRefusal::ImportExceedsHeap { needed, budget }),
        };
    }
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
            import.gpa_base().expect("RAMBlock imports have a GPA base"),
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
    use crate::runtime::guest_ram::{forget_import_limits, latch_import_limits, GuestRamRegion};

    /// A budget no test span can exceed, so a test about the granularity is not
    /// also a test about the heap. The budget tests name their own.
    const UNBOUNDED: u64 = u64::MAX;

    #[derive(Debug)]
    struct NoopExecutor;

    impl crate::runtime::executor::CapabilityService for NoopExecutor {}
    impl crate::runtime::executor::PresentationService for NoopExecutor {}
    impl crate::runtime::executor::WindowPresentationService for NoopExecutor {}
    impl crate::runtime::executor::GuestPageTransferService for NoopExecutor {}
    impl crate::runtime::executor::ResidentCopyService for NoopExecutor {}
    impl crate::runtime::executor::CompletionService for NoopExecutor {}
    impl crate::runtime::executor::SubmissionBatchService for NoopExecutor {}
    impl crate::runtime::executor::GuestImportService for NoopExecutor {}
    impl crate::runtime::executor::MaintenanceService for NoopExecutor {}
    impl crate::runtime::executor::ObservationService for NoopExecutor {}
    impl crate::runtime::executor::ShaderTranslationService for NoopExecutor {}
    impl crate::runtime::executor::RenderBufferPlanningService for NoopExecutor {}
    impl crate::runtime::executor::GuestImagePlanningService for NoopExecutor {}
    impl crate::runtime::executor::SessionService for NoopExecutor {}
    impl crate::runtime::executor::ReadbackService for NoopExecutor {
        type Error = crate::runtime::executor::DrawError;

        fn read_target(
            &self,
            _identity: &crate::model::TargetIdentity,
        ) -> Result<crate::runtime::executor::TargetReadback, Self::Error> {
            Err(crate::runtime::executor::DrawError::Facade(
                crate::runtime::executor::EngineFacadeDecline::ExecutorServiceUnavailable {
                    service: "target_readback",
                },
            ))
        }
    }
    impl crate::runtime::executor::Executor for NoopExecutor {}
    impl crate::runtime::executor::ResidentService for NoopExecutor {}
    impl crate::runtime::executor::GuestWriteService for NoopExecutor {}
    impl crate::runtime::executor::ComputeResidencyService for NoopExecutor {}

    impl crate::runtime::executor::ExecutionPort for NoopExecutor {
        type Submission = crate::runtime::executor::ResolvedSubmission;
        type Completion = crate::runtime::executor::ExecutionCompletion;
        type Error = crate::runtime::executor::DrawError;

        fn execute(&self, _submission: Self::Submission) -> Result<Self::Completion, Self::Error> {
            Err(crate::runtime::executor::DrawError::Facade(
                crate::runtime::executor::EngineFacadeDecline::ExecutorServiceUnavailable {
                    service: "test",
                },
            ))
        }
    }

    #[derive(Debug, Default)]
    struct RecordingWarmExecutor {
        imports: std::sync::Mutex<std::collections::BTreeSet<reims_vgpu_memory::ImportId>>,
        bytes: std::sync::atomic::AtomicU64,
    }

    impl RecordingWarmExecutor {
        fn bytes(&self) -> u64 {
            self.bytes.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl crate::runtime::executor::CapabilityService for RecordingWarmExecutor {}
    impl crate::runtime::executor::PresentationService for RecordingWarmExecutor {}
    impl crate::runtime::executor::WindowPresentationService for RecordingWarmExecutor {}
    impl crate::runtime::executor::MaintenanceService for RecordingWarmExecutor {}
    impl crate::runtime::executor::SubmissionBatchService for RecordingWarmExecutor {}
    impl crate::runtime::executor::ObservationService for RecordingWarmExecutor {}
    impl crate::runtime::executor::ShaderTranslationService for RecordingWarmExecutor {}
    impl crate::runtime::executor::RenderBufferPlanningService for RecordingWarmExecutor {}
    impl crate::runtime::executor::GuestImagePlanningService for RecordingWarmExecutor {}
    impl crate::runtime::executor::SessionService for RecordingWarmExecutor {}
    impl crate::runtime::executor::ResidentService for RecordingWarmExecutor {}
    impl crate::runtime::executor::GuestWriteService for RecordingWarmExecutor {}
    impl crate::runtime::executor::ComputeResidencyService for RecordingWarmExecutor {}

    impl crate::runtime::executor::GuestPageTransferService for RecordingWarmExecutor {}
    impl crate::runtime::executor::ResidentCopyService for RecordingWarmExecutor {}
    impl crate::runtime::executor::CompletionService for RecordingWarmExecutor {}

    impl crate::runtime::executor::GuestImportService for RecordingWarmExecutor {
        fn warm_guest_ram_imports(
            &self,
            imports: &[std::sync::Arc<crate::runtime::guest_ram::GuestRamImport>],
        ) -> (usize, u64) {
            let mut known = self.imports.lock().unwrap_or_else(|p| p.into_inner());
            let mut warmed = 0;
            let mut bytes = 0u64;
            for import in imports {
                if known.insert(import.id()) {
                    warmed += 1;
                    bytes = bytes.saturating_add(import.len());
                }
            }
            self.bytes
                .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
            (warmed, bytes)
        }
    }

    impl crate::runtime::executor::ReadbackService for RecordingWarmExecutor {
        type Error = crate::runtime::executor::DrawError;

        fn read_target(
            &self,
            _identity: &crate::model::TargetIdentity,
        ) -> Result<crate::runtime::executor::TargetReadback, Self::Error> {
            Err(crate::runtime::executor::DrawError::Facade(
                crate::runtime::executor::EngineFacadeDecline::ExecutorServiceUnavailable {
                    service: "test",
                },
            ))
        }
    }

    impl crate::runtime::executor::ExecutionPort for RecordingWarmExecutor {
        type Submission = crate::runtime::executor::ResolvedSubmission;
        type Completion = crate::runtime::executor::ExecutionCompletion;
        type Error = crate::runtime::executor::DrawError;

        fn execute(&self, _submission: Self::Submission) -> Result<Self::Completion, Self::Error> {
            Err(crate::runtime::executor::DrawError::Facade(
                crate::runtime::executor::EngineFacadeDecline::ExecutorServiceUnavailable {
                    service: "test",
                },
            ))
        }
    }

    impl crate::runtime::executor::Executor for RecordingWarmExecutor {}

    /// Latch a granularity with a budget and a span ceiling that admit
    /// everything, so a test about the granularity is only about that.
    fn latch_granularity(align: u64) {
        latch_import_limits(align, UNBOUNDED, UNBOUNDED);
    }

    fn forget_granularity() {
        forget_import_limits();
    }

    /// The whole module is process-global, and so is the granularity latch.
    /// Every test here takes this and restores both.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct Spans(Vec<GuestRamRegion>);

    impl GuestRamProvider for Spans {
        fn guest_ram_regions(&mut self) -> Result<Vec<GuestRamRegion>, GuestRamRegionsError> {
            Ok(self.0.clone())
        }
    }

    struct Refusing;

    impl GuestRamProvider for Refusing {
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

    /// Run `body` with a granularity and an explicit span ceiling latched, so a
    /// test about chunking is not also a test about the heap.
    fn with_span_max<R>(align: u64, span_max: u64, body: impl FnOnce() -> R) -> R {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset();
        latch_import_limits(align, UNBOUNDED, span_max);
        let out = body();
        reset();
        forget_import_limits();
        out
    }

    /// The chunks must tile the block: every byte covered once, in order, none
    /// invented and none dropped.
    ///
    /// Written as a walk rather than as an expected list because the failure
    /// this guards is an off-by-one at a boundary, and a hand-written list of
    /// expected chunks is exactly as likely to carry the same off-by-one as the
    /// code it checks. The three sizes are the three shapes: an exact multiple, a
    /// remainder, and a block already inside the ceiling.
    #[test]
    fn chunking_tiles_the_block_exactly() {
        for (len, span_max) in [
            (0x8000u64, 0x2000u64), // exact multiple
            (0x9001, 0x2000),       // ragged remainder
            (0x1000, 0x2000),       // already inside the ceiling
            (0x2000, 0x2000),       // exactly the ceiling
        ] {
            let span = GuestRamRegion {
                gpa_base: 0x1_0000_0000,
                host_va: 0x7f00_0000_0000,
                len,
            };
            let chunks = chunk_span(span, span_max);
            assert!(!chunks.is_empty(), "len={len:#x} produced no chunk");
            let mut walked = 0u64;
            for c in &chunks {
                assert!(c.len > 0, "len={len:#x} produced an empty chunk");
                assert!(
                    c.len <= span_max,
                    "len={len:#x} chunk of {:#x} is past the ceiling {span_max:#x}",
                    c.len
                );
                assert_eq!(
                    c.gpa_base,
                    span.gpa_base + walked,
                    "len={len:#x} chunk does not continue where the last ended"
                );
                assert_eq!(
                    c.host_va,
                    span.host_va + walked,
                    "len={len:#x} host pointer drifted from the GPA"
                );
                walked += c.len;
            }
            assert_eq!(walked, len, "len={len:#x} chunks do not cover the block");
        }
    }

    /// No single import may be longer than the API-derived span the backend
    /// published. This asserts that property about every import rather than
    /// about the count, because the count is a consequence and the length is
    /// the rule.
    #[test]
    fn no_import_is_longer_than_the_span_the_backend_published() {
        const CEILING: u64 = 0x2000_0000;
        with_span_max(0x1000, CEILING, || {
            let mut host = two_spans();
            assert_eq!(standing_refusal(&mut host), None);
            let imports = imports();
            assert_eq!(
                imports.len(),
                8,
                "two 2 GiB blocks at a 512 MiB ceiling are four chunks each"
            );
            for i in &imports {
                assert!(
                    i.len() <= CEILING,
                    "import at {:#x} is {} bytes, past the {CEILING} ceiling",
                    i.gpa_base().expect("RAMBlock imports have a GPA base"),
                    i.len()
                );
            }
            assert_eq!(
                imports.iter().map(|i| i.len()).sum::<u64>(),
                two_spans_bytes(),
                "chunking must not lose or duplicate guest RAM"
            );
        });
    }

    /// Every address the unchunked map resolved must still resolve, including
    /// the ones that now fall in a later chunk — and a hole must still be
    /// refused, which is what says the search did not simply widen.
    ///
    /// The boundary addresses are the point. `partition_point` picks the last
    /// import whose base is `<= gpa`, so a GPA exactly on a chunk base is the
    /// case an off-by-one puts in the previous chunk, where it is one byte past
    /// the end.
    #[test]
    fn a_chunked_block_resolves_at_every_boundary_and_still_refuses_the_hole() {
        const CEILING: u64 = 0x2000_0000;
        with_span_max(0x1000, CEILING, || {
            let mut host = two_spans();
            for gpa in [
                0u64,
                CEILING - 0x1000,
                CEILING,
                CEILING + 0x1000,
                0x8000_0000 - 0x1000,
                0x1_0000_0000,
                0x1_0000_0000 + CEILING,
                0x1_8000_0000 - 0x1000,
            ] {
                let got = reference(&mut host, gpa, 0x1000);
                assert!(
                    got.is_ok(),
                    "gpa {gpa:#x} is guest RAM and did not resolve: {:?}",
                    got.err()
                );
            }
            assert_eq!(
                reference(&mut host, 0x8000_0000, 0x1000).err(),
                Some(MapRefusal::GpaNotInAnyImport { gpa: 0x8000_0000 }),
                "the PCI hole is not guest RAM and chunking must not have covered it"
            );
        });
    }

    /// A guest run that crosses a seam between two chunks of one RAMBlock is
    /// **split**, not refused, and the pieces tile the request exactly.
    ///
    /// A `GuestRef` is an offset into one import, so it cannot describe both
    /// sides of a seam. Refusing would be safe — a named decline, whole window
    /// to the copying rail — and would put a chunk size the guest cannot see
    /// into this device's throughput on roughly one writeback in 250. The
    /// consumers already take a list, and a RAMBlock boundary has always been
    /// able to produce one, so a split here is indistinguishable from a split
    /// the guest's own page plan produced.
    ///
    /// Asserted on the tiling rather than on the count, because "two runs" would
    /// still pass if the second one started in the wrong place.
    #[test]
    fn a_run_crossing_a_chunk_seam_is_split_and_the_pieces_tile_the_window() {
        const CEILING: u64 = 0x2000_0000;
        const PAGE: u64 = 0x1000;
        with_span_max(PAGE, CEILING, || {
            let mut host = two_spans();
            // Two GPA-contiguous pages either side of the first chunk seam.
            let gpas = [CEILING - PAGE, CEILING];
            let runs = references_for_runs(&mut host, &gpas, PAGE, 0, 2 * PAGE)
                .expect("both pages are guest RAM; a seam is not a refusal");

            assert_eq!(runs.len(), 2, "one piece per import the run touches");
            let mut want = 0u64;
            for r in &runs {
                assert_eq!(
                    r.window_offset, want,
                    "a piece does not begin where the last one ended"
                );
                want += r.guest.requested();
            }
            assert_eq!(want, 2 * PAGE, "the pieces do not cover the window");
            assert!(
                runs[0].guest.import().id() != runs[1].guest.import().id(),
                "a split that stays inside one import is not a seam split"
            );
        });
    }

    /// The same run, with the seam moved out of its way, must stay one piece.
    /// Without this the test above would pass on an implementation that split
    /// every run in half.
    #[test]
    fn a_run_inside_one_chunk_is_not_split() {
        const CEILING: u64 = 0x2000_0000;
        const PAGE: u64 = 0x1000;
        with_span_max(PAGE, CEILING, || {
            let mut host = two_spans();
            let gpas = [CEILING + PAGE, CEILING + 2 * PAGE];
            let runs = references_for_runs(&mut host, &gpas, PAGE, 0, 2 * PAGE)
                .expect("both pages are guest RAM");
            assert_eq!(runs.len(), 1, "a run wholly inside one import is one piece");
            assert_eq!(runs[0].guest.requested(), 2 * PAGE);
        });
    }

    /// Run `body` with a granularity and an explicit heap budget latched.
    fn with_budget<R>(align: u64, budget: u64, body: impl FnOnce() -> R) -> R {
        let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        reset();
        latch_import_limits(align, budget, UNBOUNDED);
        let out = body();
        reset();
        forget_import_limits();
        out
    }

    /// The bytes [`two_spans`] asks this device to keep resident. Derived from
    /// the spans rather than written out, so a change to either stays one number.
    fn two_spans_bytes() -> u64 {
        two_spans().0.iter().map(|r| r.len).sum()
    }

    /// A guest larger than the roomiest heap on the host's GPU must not import
    /// at all.
    ///
    /// This is the reported `radv`/`amdgpu` failure: the import succeeds, and
    /// every submission that names it then fails validation in the kernel with
    /// `Not enough memory for command submission`, which arrives as a lost
    /// device. The copying rails are the working configuration on such a host and
    /// this is how the device reaches them without an operator setting a variable.
    #[test]
    fn a_guest_larger_than_every_heap_takes_the_copying_rails() {
        let needed = two_spans_bytes();
        with_budget(0x1000, needed - 1, || {
            let mut host = two_spans();
            assert_eq!(
                standing_refusal(&mut host),
                Some(MapRefusal::ImportExceedsHeap {
                    needed,
                    budget: needed - 1,
                }),
                "a guest past the budget must refuse by name"
            );
            assert_eq!(
                imports().len(),
                0,
                "and must not leave a partial import behind"
            );
        });
    }

    /// The bound is `>`, not `>=`: a guest that exactly fills the roomiest heap
    /// is admitted, because nothing in the contract says an allocation the size
    /// of its heap cannot be made. Pinned because an off-by-one in the safe
    /// direction here silently costs every host whose heap equals its guest.
    #[test]
    fn a_guest_that_exactly_fills_the_roomiest_heap_still_imports() {
        let needed = two_spans_bytes();
        with_budget(0x1000, needed, || {
            let mut host = two_spans();
            assert_eq!(standing_refusal(&mut host), None);
            assert_eq!(imports().len(), 2);
        });
    }

    /// A backend that publishes a granularity beside a zero budget has published
    /// nothing: the pair is withdrawn, and the map reads it as a host that cannot
    /// import. Without this, a zero budget would refuse every guest by name and
    /// blame the guest's size for a backend that answered badly.
    #[test]
    fn a_zero_budget_withdraws_the_granularity_rather_than_refusing_every_guest() {
        with_budget(0x1000, 0, || {
            let mut host = two_spans();
            assert_eq!(
                standing_refusal(&mut host),
                Some(MapRefusal::NoBackendImport),
                "a zero budget is a backend that cannot import, not a guest that is too big"
            );
        });
    }

    /// Warming before a backend has published a granularity must leave the
    /// resolution untaken, so the import a late backend enables is still
    /// available.
    ///
    /// This is the whole hazard in moving the import off the guest's first
    /// draw. `resolve` answers `NoBackendImport` when there is no granularity
    /// and that answer is latched in `MAP` for the rest of the boot, so a warm
    /// taken one instant too early does not merely fail to help — it converts a
    /// host that can import into one that refuses every window for the life of
    /// the VM, and reports it as a host lacking the extension.
    #[test]
    fn warming_before_the_backend_publishes_a_granularity_latches_nothing() {
        with_granularity(None, || {
            let mut host = two_spans();
            warm(&mut host, &NoopExecutor);
            assert!(
                MAP.lock().unwrap_or_else(|p| p.into_inner()).is_none(),
                "a warm with no granularity must not latch a refusal"
            );

            // The backend comes up late; the import must still be available.
            latch_granularity(0x1000);
            warm(&mut host, &NoopExecutor);
            assert_eq!(
                standing_refusal(&mut host),
                None,
                "the late backend's import must still resolve"
            );
            assert_eq!(imports().len(), 2, "both spans import once warmed");
        });
    }

    /// Warming is what the first reference would have done, and doing it twice
    /// changes nothing — the guest's handshake may be replayed, and every later
    /// caller has to find the same answer.
    #[test]
    fn warming_is_idempotent_and_equals_what_the_lazy_path_would_resolve() {
        let lazy = with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            let refusal = standing_refusal(&mut host);
            (refusal, imports().len())
        });
        let warmed = with_granularity(Some(0x1000), || {
            let mut host = two_spans();
            warm(&mut host, &NoopExecutor);
            warm(&mut host, &NoopExecutor);
            let refusal = standing_refusal(&mut host);
            (refusal, imports().len())
        });
        assert_eq!(warmed, lazy);
        assert_eq!(warmed.1, 2);
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
            impl GuestRamProvider for Counting {
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

    /// The protocol handshake publishes each RAMBlock to the executor before
    /// the first draw can demand an import, and does so only once per import.
    #[test]
    fn the_handshake_warm_imports_before_any_draw_references_a_byte() {
        const LEN: u64 = 16 << 20;

        struct OneBlock(u64);
        impl crate::runtime::host::GuestRamProvider for OneBlock {
            fn guest_ram_regions(
                &mut self,
            ) -> Result<
                Vec<reims_vgpu_memory::GuestRamRegion>,
                crate::runtime::host::GuestRamRegionsError,
            > {
                Ok(vec![reims_vgpu_memory::GuestRamRegion {
                    gpa_base: 0,
                    host_va: self.0,
                    len: LEN,
                }])
            }
        }

        with_granularity(Some(4096), || {
            let mut host = OneBlock(0x7f00_0000_0000);
            let executor = RecordingWarmExecutor::default();
            warm(&mut host, &executor);
            assert_eq!(executor.bytes(), LEN);

            warm(&mut host, &executor);
            assert_eq!(executor.bytes(), LEN);
        });
    }
}
