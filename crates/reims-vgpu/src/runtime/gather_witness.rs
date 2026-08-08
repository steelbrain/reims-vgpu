//! Which sampled windows the engine may bind without reading a byte of guest RAM.
//!
//! The three zero-copy sampled producers ([`super::draw::vulkan`]'s
//! linear, type-11 and type-5 rails) hand the engine a
//! [`crate::backend::vulkan::engine::SampledSource::GuestRuns`], and the engine's
//! only byte-moving arm gathers the whole window out of guest RAM into a staging
//! buffer. That arm had no content cache — measured on a driven x86/PCI boot at
//! 360 gathers and **842.4 MB per second**, both figures repeating to the digit
//! across eight consecutive windows, which is the shape of the same unchanged
//! content being re-read every frame rather than of a working set that moves.
//!
//! This module is the cache's witness: it answers "nothing has written these
//! pages since the gather that filled the retained image", and issues a
//! [`GatheredIdentity`] the engine binds on with no compare at all. A *false*
//! answer serves stale pixels, which is a wrong frame that then persists — the
//! failure mode that turned the screen black once already.
//!
//! # The witness takes two halves
//!
//! Neither half alone is sound, because they cover disjoint writers:
//!
//! - the **generation** ([`crate::runtime::host::HostOps::guest_write_gen`], the
//!   hypervisor dirty bitmap) witnesses guest CPU stores, and is defined not to
//!   see writes this device makes;
//! - the **page-exact host-write record**
//!   ([`crate::runtime::host_writes::HostWrites`]) witnesses this device's own
//!   writes into exactly these pages.
//!
//! Both quiet is [`GatherVerdict::Vouched`]: the generation the entry already
//! holds survives, and the gather is skipped. Anything else spends a fresh
//! generation, and the engine's `(key, generation)` lookup misses, so the bytes
//! are read.
//!
//! **A spent generation makes the next lookup miss by construction, and that is
//! the witness working rather than a cache failing.** [`note_gather`] hands both
//! facts back together in a [`GatherOutcome`] for exactly this reason: the
//! identity is what the engine binds and retains on, and the [`GatherVouch`]
//! beside it is whether that identity could ever have named a retained image.
//! An engine that has only the identity cannot tell a compulsory miss from a
//! lost one — it once tried, by asking whether the identity was present at all,
//! and that question has one answer. Every window this witness is asked about
//! gets an entry, so it names every one of them.
//!
//! Verdicts, through [`crate::runtime::drain::note_store_route`]:
//!
//! | route | meaning |
//! |---|---|
//! | `gw_vouched` / `gw_vouched_kb` | both halves quiet — the gather is skipped |
//! | `gw_refused_guest_store` | the hypervisor saw a guest store into the pages |
//! | `gw_refused_host_write` | this device wrote pages of this window |
//! | `gw_unarmed` | no token, or a generation not yet readable — no answer |
//! | `gw_rearm` | the window's page set changed, so nothing to compare against |
//! | `gw_audit_seed` | first fold of this window — expected, and not the alarm |
//! | `gw_audit_restart` | the stride came due and the baseline had been dropped: **the audit declining to compare** |
//! | `gw_audit_ok` | folded under a live baseline and the bytes agreed |
//! | `gw_audit_unsound` | folded under a live baseline and the bytes had moved |
//!
//! # The half that refuses is `gw_refused_host_write`, by 368 to 1
//!
//! A driven x86/PCI Safari drag, quiesced, 166 census windows:
//!
//! ```text
//! gw_vouched             6050
//! gw_refused_host_write  5156
//! gw_refused_guest_store   14
//! gw_unarmed              212
//! gw_rearm                128
//! gw_audit_unsound          0
//! ```
//!
//! The guest hardly writes the windows it samples; something on this device's
//! side is what refuses them, and each refusal costs the next bind a full
//! re-gather — 68 % of that rail's misses on the same boot, against 32 % that
//! were a retained image the cache had dropped. So the cache is not the lever.
//!
//! That reading used to end "`gw_audit_unsound` at 0 says the witness stayed
//! sound throughout." **It says no such thing**, and the next section is why.
//!
//! # The audit has never once compared, and its zero is not a measurement
//!
//! [`ContentAudit`] is the alarm for a writer that escapes both halves. To
//! reach a comparison it needs a fold taken under a vouch and still valid at
//! the next stride bind — and `fold_valid` is dropped by **any single refused
//! bind**, correctly, because a refusal means the bytes were free to move. So
//! a comparison needs [`AUDIT_STRIDE`] *consecutive* vouched binds of one
//! window.
//!
//! At the refusal rates every driven boot of this device measures — of the
//! order of 4 700 refusals against 7 300 vouches — a run of 64 is a coincidence
//! this workload does not produce. Three consecutive driven boots read
//! `gw_audit_ok` **0** against `gw_audit_seed` 163-175: every audit bind was a
//! first fold and the fold was never once checked against a previous one.
//!
//! `gw_audit_unsound` is therefore 0 because the comparison never ran, not
//! because it ran and agreed. A real escaping writer went unnoticed behind that
//! zero on this branch — the GPU-direct GVA Store wrote guest pages without
//! recording them, and the audit was structurally incapable of noticing.
//!
//! **Read `gw_audit_restart` beside it.** While that dominates `gw_audit_ok`,
//! the alarm is not running. The repair is to re-seed the fold on the *refused*
//! binds, where the gather reads those bytes anyway, so a baseline survives to
//! meet the next vouch; it is not done here.
//!
//! # It was the ring, not the writes: `gw_hw_aged` 4275 against `gw_hw_overlap` 5
//!
//! The split below was measured on the next driven boot and the attribution
//! above is **wrong**:
//!
//! ```text
//! gw_hw_quiet          5706
//! gw_hw_aged           4275     (gw_refused_host_write 4204)
//! gw_hw_overlap           5
//! gw_hw_unnamed           0
//! gw_hw_unresolvable      0
//! ```
//!
//! Five binds in 9986 had a recorded write that actually covers the window.
//! Every other refusal is
//! [`crate::runtime::host_writes::HostWrites`]'s ring having dropped the writes
//! the reader is asking about, so it cannot say nothing touched them. This
//! device is *not* writing the windows it samples; it is failing to remember
//! that it did not. 43 % of every witness ask on that boot was refused for that
//! one reason, and each refusal costs a full re-gather.
//!
//! `RING`'s own doc sized it from "~28 host writes a second against ~330 gathers
//! a second, so the usual answer is zero entries to scan". That held for the
//! workload it was measured on and does not hold under compositing. Band the
//! requested reach before choosing a new size — the number wanted is how far
//! back a reader asks, and nothing has measured it yet.
//!
//! **`gw_refused_host_write` is not "this device wrote these pages".**
//! [`crate::runtime::host_writes::HostWrites::wrote_any_since`] answers "written"
//! for four different reasons and only one of them is a write that landed in the
//! window: the other three are its fail-closed rule — a writer that named no
//! pages, a ring too short to still hold the writes being asked about, and a
//! mapping-named write whose page list has since moved. Three of those are
//! bookkeeping this device could fix without changing what it writes at all.
//! The `gw_hw_*` routes below split them, and until a boot reads them the 5156
//! is an upper bound on real overlap rather than a measurement of it:
//!
//! | route | meaning |
//! |---|---|
//! | `gw_hw_quiet` | nothing recorded touched the window — the vouchable case |
//! | `gw_hw_overlap` | a recorded write names one of these pages; the bytes moved |
//! | `gw_hw_unnamed` | a writer could not say where it landed, so all readers assume it |
//! | `gw_hw_aged` | the ring no longer holds the writes this reader asks about |
//! | `gw_hw_unresolvable` | a mapping-named write whose page list cannot be rebuilt |
//!
//! Both halves stay load-bearing and neither may be weakened to raise the vouch
//! rate. Whatever the split says, the repair is to make the record *sharper* —
//! a writer naming its pages rules itself out of windows it never touched —
//! never to let an undecidable read as quiet.
//!
//! # A device-wide `gw_refused_guest_store` is the hypervisor rail, not the guest
//!
//! This counter reads in the low hundreds over a whole driven boot. A boot where
//! it reads in the tens of thousands has not met a guest that started writing its
//! surfaces; it has met a witness that cannot say otherwise, and the difference is
//! worth recognising because the second one latches and the first does not.
//!
//! The shape to look for is a **step**: the per-second rate jumping two orders of
//! magnitude inside one second, across every mapping at once, and never coming
//! back. Per-surface causes cannot do that — only state the whole device shares
//! can, and on this rail that state lives in `reims_vgpu_dirty_harvest`
//! (`hw/display/reims-vgpu-dirty.c`), which reads any tracked page it cannot
//! resolve to a recorded guest-RAM range as written. One such bug is fixed and
//! documented there: the harvest cut its window with a walk that swallowed every
//! page above the first non-RAM byte, and nothing unwound it short of a reboot.
//!
//! It is worth recognising from the other side too, because the same step drives
//! `runtime::draw`'s type-11 sampled rung into
//! `t11rung_resident_refused`, whose merge skips every page the witness claims
//! and so leaves a GPU-side composite reading blank. Twelve recorded boots
//! separated on this counter with no overlap — 155-186 clean against
//! 20 122-34 772 degraded — which makes it the gate for that class as well.
//!
//! # The content fold is now an audit, not the decision
//!
//! A full fold over the window is what *established* the rule above: crossed
//! against the two halves it produced the cell "vouched, and the bytes moved
//! anyway", which condemned three candidate rules in turn before the surviving
//! one read zero across four driven boots.
//!
//! Running it on every bind would defeat the cache it licensed — a skipped
//! gather that still reads every byte to fold them has moved the cost, not
//! removed it. So the fold runs once per
//! [`crate::runtime::gather_witness::AUDIT_STRIDE`] binds of a window and
//! its verdict is a standing alarm rather than an input:
//!
//! | route | meaning |
//! |---|---|
//! | `gw_audit_ok` | folded under a vouch, and the bytes were where it said |
//! | `gw_audit_unsound` | **folded under a vouch, and the bytes had moved** |
//! | `gw_audit_seed` | folded with no trustworthy predecessor to compare against |
//! | `gw_audit_kb` | bytes the audit read — the whole remaining cost of the fold |
//!
//! `gw_audit_unsound` is the one that matters, and it is not only counted: it
//! fails through the always-on log with the window that broke, and drops the
//! vouched generation so the next bind re-gathers. Both holes found while
//! building this witness — a per-mapping rule whose pages aliased, and a writer
//! outside the host-write record — fired tens to hundreds of times per boot, so
//! sampling costs the alarm latency and not its reach.

use std::collections::BTreeMap;

use crate::contract::fnv;

/// Which zero-copy sampled producer built the window.
///
/// The 2x2 below says whether the witness is sound; this says whose gathers it
/// would be sound *for*. The aggregate reading that opened this — 360 gathers and
/// 842.4 MB a second — is the sum over all three rails and has never been split,
/// so which of them to fix is not yet known.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GatherRail {
    /// Linear guest texture addressed through task GVA.
    Linear,
    /// Type-11 mapping-backed sampled bind.
    Type11,
    /// Type-5 serialized IOSurface plane view (the video path).
    Type5,
}

impl GatherRail {
    /// Census names for the rail's gather count and its gathered kilobytes.
    fn names(self) -> (&'static str, &'static str) {
        match self {
            Self::Linear => ("gw_rail_linear", "gw_rail_linear_kb"),
            Self::Type11 => ("gw_rail_t11", "gw_rail_t11_kb"),
            Self::Type5 => ("gw_rail_t5", "gw_rail_t5_kb"),
        }
    }
}

/// Which sampled window a witness entry describes.
///
/// The two shapes are the two ways the producers name a window: a task-GVA span
/// (the linear texture rail, which has no mapping) and a mapping-relative offset
/// (the type-11 and type-5 rails). Those two rails can name the same
/// `(mid, base_off)` for a single-plane surface, and that is harmless — same
/// mapping, same offset and same span is the same bytes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum GatherKey {
    /// A texture window addressed through a task's GVA space.
    TaskGva { task_id: u32, gva: u64 },
    /// A window at a byte offset into a mapping's page list.
    Mapping { mid: u32, base_off: u64 },
}

impl GatherKey {
    /// A 64-bit name for this window in the device-wide sampled-identity
    /// keyspace.
    ///
    /// Collisions across the two shapes, or with any other producer's keys, are
    /// harmless and do not need to be designed out: the engine matches on
    /// `(key, generation)` and generations come from one device-global counter
    /// that issues each value once and never again. The key only has to be
    /// *stable* for one window, so that a window's own binds find each other.
    pub fn content_key(self) -> u64 {
        // FNV-1a over the discriminant and fields. A hash rather than a packing
        // because both shapes carry more than 64 bits. The discriminant is
        // folded first so the two shapes cannot alias each other.
        let mut h = fnv::FNV_OFFSET_BASIS;
        let mut eat = |v: u64| h = fnv::fold_u64(h, v);
        match self {
            Self::TaskGva { task_id, gva } => {
                eat(1);
                eat(task_id as u64);
                eat(gva);
            }
            Self::Mapping { mid, base_off } => {
                eat(2);
                eat(mid as u64);
                eat(base_off);
            }
        }
        h
    }

    /// Whitespace-free rendering for the always-on log, which is parsed by
    /// splitting on spaces.
    fn log_token(self) -> String {
        match self {
            Self::TaskGva { task_id, gva } => format!("gva:{task_id}:{gva:#x}"),
            Self::Mapping { mid, base_off } => format!("map:{mid}:{base_off:#x}"),
        }
    }
}

/// What the last bind of one window observed.
#[derive(Clone, Debug)]
struct Entry {
    /// The exact page set the gather read, in window order. A change here means
    /// the window was re-pointed and there is nothing to compare against.
    gpas: Vec<u64>,
    /// Byte length of the window (a geometry change is also a re-point).
    span: u64,
    /// Tracking token armed over `gpas`, or 0 when the host refused one.
    token: u64,
    /// Generation read at the previous bind; 0 means "was not readable".
    gen: u64,
    /// Content fold from the last audit of this window.
    fold: u128,
    /// Whether `fold` still describes the window's bytes.
    ///
    /// True from the audit that recorded it for as long as every bind since was
    /// [`GatherVerdict::Vouched`] — which is the claim the audit exists to check,
    /// so comparing across that run is exactly the right comparison and a longer
    /// run is a stronger one. A bind the witness refused may have changed the
    /// bytes with nothing reading them, and clears it.
    fold_valid: bool,
    /// Whether this window has ever been folded, latched on the first audit.
    ///
    /// Separate from [`Self::fold_valid`], which answers whether the stored
    /// fold is still a *baseline*. Together they separate the two ways an audit
    /// can find nothing to compare against — never folded, or folded and then
    /// invalidated — which read identically without this and are
    /// [`ContentAudit::Seeded`] and [`ContentAudit::Restarted`] with it.
    fold_seeded: bool,
    /// Binds of this window since its last audit, against [`AUDIT_STRIDE`].
    ///
    /// Per window rather than device-wide: a global stride would audit whichever
    /// window happened to land on the multiple and could starve a busy one
    /// indefinitely, where the alarm's whole job is bounded latency per window.
    binds_since_fold: u32,
    /// `HostWrites::epoch` at the previous bind, against which the page-exact
    /// question "did this device write any of *these pages* since" is asked.
    pages_epoch: u64,
    /// Bind ordinal of the last sight of this window, for LRU eviction.
    last_seen: u64,
    /// Sampled-content generation currently vouched for these bytes.
    ///
    /// Held across binds for as long as both halves of the witness say the bytes
    /// cannot have changed, and replaced the moment either says otherwise. The
    /// engine's sampled cache binds a retained image on `(key, generation)` with
    /// no compare at all, so a generation that outlives its content by one bind
    /// is a wrong picture that then persists.
    generation: u64,
}

/// Per-device witness state: one entry per sampled window seen.
#[derive(Default, Debug)]
pub struct GatherWitness {
    entries: BTreeMap<GatherKey, Entry>,
    /// Monotonic bind ordinal, stamped into [`Entry::last_seen`].
    binds: u64,
}

/// Upper bound on tracked windows.
///
/// Not a memory bound — a hypervisor harvest bound. `reims_vgpu_dirty_harvest`
/// walks every page of every tracked set on the BQL thread at each register write
/// that hands the device work, so each armed window adds its page count to a cost
/// the whole VM pays. A driven Safari boot re-presents on the order of sixty
/// distinct sampled keys, so this sits just above the observed working set rather
/// than wherever memory would run out.
///
/// The first driven boot hit the cap twice during a hard scroll, so the working
/// set does reach it. Overflow evicts the least recently bound window rather than
/// dropping the map: a full drop costs a `gw_rearm` for every live window at once,
/// which is precisely the population whose answers are wanted.
const MAX_TRACKED_WINDOWS: usize = 256;

/// Binds of one window between content audits.
///
/// The fold no longer decides a skip, so its only remaining job is to catch the
/// witness going unsound — and that is a systematic fault rather than a one-off.
/// Both holes found while building this witness repeated tens to hundreds of
/// times per boot, so an audit that sees one bind in `AUDIT_STRIDE` still sees
/// them within seconds.
///
/// The value is the two bounds meeting. A window re-presented at frame rate
/// binds about sixty times a second, so sixty-four bounds the alarm at roughly a
/// second of stale pixels; and one bind in sixty-four is 1.6% of the gathered
/// bytes, about 13 MB/s against the 842 MB/s rail this cache was built to
/// remove. Both the latency and the cost degrade smoothly, so neither edge is
/// fitted to an observation.
pub const AUDIT_STRIDE: u32 = 64;

impl GatherWitness {
    /// Detach every tracking token this witness armed, for release through
    /// [`crate::runtime::host::HostOps::untrack_guest_writes`].
    ///
    /// Returns them rather than releasing them because this type has no
    /// `HostOps` and the crate already has one rail for host state it cannot
    /// free itself: `DeviceState::retired_guest_write_tokens`, drained by
    /// `mapper::flush_retired_views`. The tokens are host resources keyed to
    /// page sets, so dropping the map without this leaves the host dirty-logging
    /// those pages for the life of the process.
    pub fn take_tokens(&mut self) -> Vec<u64> {
        let tokens = self
            .entries
            .values()
            .map(|entry| entry.token)
            .filter(|&token| token != 0)
            .collect();
        self.entries.clear();
        tokens
    }

    /// Arm one window against `token` with nothing else set, so a test can
    /// prove the token is released without driving a gather to create it.
    #[cfg(test)]
    pub fn arm_token_for_test(&mut self, token: u64) {
        self.entries.insert(
            GatherKey::TaskGva {
                task_id: 1,
                gva: 0x1000,
            },
            Entry {
                gpas: vec![0x3000],
                span: 0x1000,
                token,
                gen: 0,
                fold: 0,
                fold_valid: false,
                fold_seeded: false,
                binds_since_fold: 0,
                pages_epoch: 0,
                last_seen: 0,
                generation: 0,
            },
        );
    }

    /// The host-write epoch recorded at the previous bind of `key`, if any.
    fn previous_pages_epoch(&self, key: &GatherKey) -> Option<u64> {
        self.entries.get(key).map(|entry| entry.pages_epoch)
    }

    /// Drop the least recently bound window, releasing its token.
    fn evict_oldest<M: crate::runtime::host::HostOps>(&mut self, host: &mut M) {
        let Some(victim) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_seen)
            .map(|(key, _)| *key)
        else {
            return;
        };
        if let Some(entry) = self.entries.remove(&victim) {
            if entry.token != 0 {
                host.untrack_guest_writes(entry.token);
            }
        }
    }
}

/// Fold `span` bytes of a gathered window into a 128-bit value.
///
/// Word-wise rather than byte-wise, and two accumulators mixed differently so the
/// result is position-sensitive: a fold that only summed words would call any
/// permutation of a window unchanged, and a scrolled tile atlas is exactly a
/// permutation of itself.
///
/// # Safety
/// Every run's `host_ptr` must be a live mapping of at least `len` bytes — the
/// same precondition the gather itself relies on, read at the same point in the
/// draw.
unsafe fn fold_runs(runs: &[crate::backend::vulkan::engine::GuestRun], span: u64) -> u128 {
    let mut a: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut b: u64 = 0xc2b2_ae3d_27d4_eb4f;
    let mut remaining = span;
    for run in runs {
        if remaining == 0 {
            break;
        }
        let n = run.len.min(remaining) as usize;
        remaining -= n as u64;
        // SAFETY: caller's precondition — `host_ptr` is a stable RAMBlock alias
        // valid for at least `run.len` bytes, and `n <= run.len`.
        let bytes = unsafe { std::slice::from_raw_parts(run.host_ptr as *const u8, n) };
        let (words, tail) = bytes.split_at(n & !7);
        for chunk in words.chunks_exact(8) {
            let w = u64::from_le_bytes(chunk.try_into().expect("chunks_exact(8) yields 8 bytes"));
            a = (a ^ w).rotate_left(29).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            b = b.rotate_left(7).wrapping_add(w ^ a);
        }
        for (i, &byte) in tail.iter().enumerate() {
            a ^= (byte as u64) << (8 * i);
        }
        // Fold the run boundary in so two windows with the same bytes split into
        // different runs are still distinguishable.
        b = b.wrapping_mul(0xff51_afd7_ed55_8ccd) ^ (n as u64);
    }
    ((a as u128) << 64) | b as u128
}

/// This device's own answer for one bind: did *we* write these pages?
///
/// Gathered before the witness is touched, because the page-exact question needs
/// the epoch recorded at the previous bind and the ring that answers it is read
/// through the same device state the witness lives in.
///
/// Two coarser counts used to be asked here beside it — the device-global host
/// write sequence and a per-mapping share of it — scoring the two candidate
/// invalidation rules that lost. The global rule invalidates a texture because
/// an unrelated scanout was composited; the per-mapping one read fifteen stale
/// binds a minute, because guest pages are reachable under more than one mapping
/// id. Neither is a rule this device could use, so neither is a count it still
/// takes.
#[derive(Clone, Copy, Debug)]
struct HostWriteCounts {
    /// `HostWrites::epoch()` now, to be recorded for the next bind to ask against.
    pages_epoch: u64,
    /// Whether this device wrote any of this window's pages since the previous
    /// bind, and on what grounds. `None` when there is no previous bind to ask
    /// about.
    ///
    /// Carried as the verdict rather than a `bool` because three of its four
    /// non-quiet values are this device declining to rule the write out rather
    /// than a write that landed here, and the three want different repairs.
    pages_wrote: Option<crate::runtime::host_writes::HostWriteVerdict>,
}

/// The resolved window one gather will read.
///
/// The pages and the host spans over them are both needed and neither implies
/// the other: guest-write tracking registers a page set, and the content fold
/// reads through the coalesced host pointers.
pub struct GatherWindow<'a> {
    /// Page-aligned guest addresses the window covers, in window order.
    pub gpas: &'a [u64],
    /// Coalesced host spans the gather reads, covering `span` bytes in order.
    pub runs: &'a [crate::backend::vulkan::engine::GuestRun],
    /// Byte length of the window.
    pub span: u64,
    /// Guest page size the `gpas` are expressed in.
    pub page_size: usize,
}

/// What the two halves of the witness said about one bind of a window.
///
/// Returned rather than only counted so a test can drive the witness against a
/// host whose writes it controls, and so the census emission is one place instead
/// of five.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GatherVerdict {
    /// First sight of the window, or its page set / span moved. Nothing to
    /// compare against; the entry now holds this bind's answers.
    Rearmed,
    /// No readable generation on one side of the comparison, so the hypervisor
    /// half says nothing at all. Fail closed: nothing is vouched for.
    Unarmed,
    /// Both halves quiet — no guest store into the pages, and no write by this
    /// device either. The gather is skippable and the entry keeps its generation.
    Vouched,
    /// At least one half saw a write. The generation is spent and the bytes are
    /// read.
    ///
    /// Both flags can be set at once, and are counted apart because they name
    /// different work: a guest store is the guest repainting, while a write by
    /// this device is our own writeback landing in pages a sampler also reads.
    Refused {
        /// The hypervisor observed a guest store into these pages.
        guest_wrote: bool,
        /// This device wrote at least one of these pages.
        host_wrote_pages: bool,
    },
}

/// What the content fold said, on the binds where it ran.
///
/// The fold is the audit of [`GatherVerdict::Vouched`], not an input to it —
/// see [`AUDIT_STRIDE`] for why it runs on one bind in sixty-four rather than
/// all of them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContentAudit {
    /// Not due this bind: no byte of the window was read.
    Skipped,
    /// Folded for the first time on this window. There was nothing to compare
    /// against; this bind only records one.
    Seeded,
    /// The stride came due and there *was* a fold, but a refused bind since it
    /// was taken invalidated it, so this bind could only record a new one.
    ///
    /// Split out from [`Self::Seeded`] because the two say opposite things
    /// about whether the alarm is running. A seed is a window being met for
    /// the first time and is expected. This is the audit **declining to
    /// compare**, and where it dominates, [`Self::Disagreed`] reading zero says
    /// nothing at all — which is exactly how it read while a writer that
    /// escaped both halves went unnoticed.
    ///
    /// It is common by construction rather than by accident: the baseline is
    /// dropped by any single refusal, so reaching a comparison needs
    /// [`AUDIT_STRIDE`] consecutive vouched binds of one window. At the
    /// refusal rates a driven boot measures — 4 669 refusals against 7 347
    /// vouches — that is a run this workload never produces, and the fix is to
    /// re-seed the fold on the refused binds, where the gather reads the bytes
    /// anyway.
    Restarted,
    /// Folded under a vouch, and the bytes are where the vouch said they were.
    Agreed,
    /// Folded under a vouch, and the bytes had moved. Some writer reaches these
    /// guest pages without either half of the witness seeing it, so every gather
    /// skipped since the last audit bound a stale image.
    Disagreed,
}

/// One bind's answers: what the witness decided, what the audit found, and the
/// generation the window is left naming.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GatherObservation {
    /// The decision, from the two witness halves alone.
    pub verdict: GatherVerdict,
    /// The check on it, on the binds where the fold ran.
    pub audit: ContentAudit,
    /// The generation this window names *after* the bind — the one the entry
    /// carried in, where it survived, and `fresh_generation` where it did not.
    ///
    /// Returned rather than looked back up so the identity has no absent case to
    /// spell. Reading it back out of the map produced an `Option` that was
    /// `Some` on every path through [`observe`], which is how
    /// `sampled_gather_unvouched` came to be a counter that could not fire.
    pub generation: u64,
    /// Whether [`Self::generation`] is the one the entry carried in or one spent
    /// this bind. Decided beside the assignment that spends it, never
    /// re-derived from [`Self::verdict`] — a `Disagreed` audit vouches and still
    /// spends, so the two do not agree.
    pub vouch: GatherVouch,
}

/// The one way this witness can be wrong.
#[derive(Clone, Copy, Debug)]
pub enum GatherWitnessFault {
    /// Both halves vouched for a window and the content audit found its bytes
    /// moved. Names the window so the writer can be hunted, and the bind count
    /// so the number of stale frames served is bounded rather than guessed.
    VouchedBytesMoved {
        key: GatherKey,
        span: u64,
        binds: u32,
    },
}

impl crate::observe::decline::Decline for GatherWitnessFault {
    fn slug(&self) -> &'static str {
        match self {
            Self::VouchedBytesMoved { .. } => "gather_witness_vouched_bytes_moved",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::VouchedBytesMoved { key, span, binds } => vec![
                ("window", key.log_token()),
                ("span", span.to_string()),
                ("binds", binds.to_string()),
            ],
        }
    }
}

/// One bind's answer to the engine: what to bind on, and whether it is worth
/// anything.
///
/// Both halves are always present. The type exists so they travel together —
/// carrying the identity alone is what let the engine ask "is there an identity"
/// and read the answer as "did the witness vouch".
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatherOutcome {
    /// What the engine looks the retained image up under, and retains under.
    pub identity: GatheredIdentity,
    /// Whether that identity can name an image the cache already holds.
    pub vouch: GatherVouch,
}

/// Record one zero-copy sampled gather against the guest-write witness, and
/// report it to the census.
///
/// Called from the producers with the window already resolved, so it adds a
/// page-set compare and one content fold and changes no behaviour.
///
/// # Why this does not return an `Option`
///
/// It used to, and the `Option` was `Some` on every path: the identity was read
/// back with `vouched_identity`, which answered "is this window tracked", and
/// [`observe`] leaves an entry for every key it is given — the re-point branch
/// inserts one and returns, the overflow evictor never evicts the key it was
/// asked about, and the surviving branch holds a `&mut` to one. The engine spent
/// a boot counting `identity.is_some()` as the witness's verdict and read the
/// resulting zero as "the witness never refused a gather". It cannot refuse
/// through this return value at all; [`GatherVouch`] is where the verdict lives.
#[must_use = "the identity is what lets the engine skip the gather; dropping it \
              silently keeps the copy"]
pub fn note_gather<M: crate::runtime::host::HostOps>(
    state: &mut crate::model::DeviceState,
    host: &mut M,
    rail: GatherRail,
    key: GatherKey,
    window: GatherWindow<'_>,
) -> GatherOutcome {
    use crate::runtime::drain::{note_store_route, note_store_route_n};

    let span = window.span;
    let (rail_count, rail_kb) = rail.names();
    note_store_route(rail_count);
    note_store_route_n(rail_kb, span / 1024);

    // This device's own answer, taken before the witness is touched: the
    // page-exact question needs the epoch recorded at the *previous* bind, which
    // is inside the witness, and the ring that answers it is read through the
    // same device state.
    let counts = HostWriteCounts {
        pages_epoch: state.host_writes.epoch(),
        pages_wrote: state
            .gather_witness
            .previous_pages_epoch(&key)
            .map(|since| state.host_writes.wrote_any_since(state, since, window.gpas)),
    };
    // Report the host-write half's grounds, not just its answer. Three of its
    // four non-quiet values are this device declining to rule a write out rather
    // than one that landed here, and they want different repairs — name the
    // writer's pages, widen the ring, or stop writing the window at all. Taken
    // for every bind that had a previous one to ask about, so the split covers
    // the vouched binds too and `gw_hw_quiet` is the denominator.
    if let Some(verdict) = counts.pages_wrote {
        note_store_route(verdict.route());
    }
    // A generation is issued from the device-global counter and never reused, so
    // it is taken before the witness runs and spent only if the witness refuses
    // to vouch for the previous one. An unspent generation is not a leak: the
    // counter's whole contract is that a value is issued once and never again.
    let fresh = state.next_sampled_content_generation();
    let seen = observe(&mut state.gather_witness, host, key, window, counts, fresh);

    match seen.verdict {
        GatherVerdict::Rearmed => note_store_route("gw_rearm"),
        GatherVerdict::Unarmed => note_store_route("gw_unarmed"),
        GatherVerdict::Vouched => {
            note_store_route("gw_vouched");
            note_store_route_n("gw_vouched_kb", span / 1024);
        }
        GatherVerdict::Refused {
            guest_wrote,
            host_wrote_pages,
        } => {
            if guest_wrote {
                note_store_route("gw_refused_guest_store");
            }
            if host_wrote_pages {
                note_store_route("gw_refused_host_write");
            }
        }
    }
    // `gw_audit_kb` is every byte the fold still reads, so the cost of keeping
    // the alarm is reported in the same units as the gathers it saves.
    if !matches!(seen.audit, ContentAudit::Skipped) {
        note_store_route_n("gw_audit_kb", span / 1024);
    }
    match seen.audit {
        ContentAudit::Skipped => {}
        ContentAudit::Seeded => note_store_route("gw_audit_seed"),
        // The denominator `gw_audit_unsound` never had. Read the two together:
        // while this dominates `gw_audit_ok`, the alarm is not running and a
        // zero from it is not a measurement.
        ContentAudit::Restarted => note_store_route("gw_audit_restart"),
        ContentAudit::Agreed => note_store_route("gw_audit_ok"),
        ContentAudit::Disagreed => {
            note_store_route("gw_audit_unsound");
            // Once per window: a writer escaping both halves escapes them on
            // every bind, and the second line says nothing the first did not.
            // The count above carries the magnitude.
            crate::observe::emit::Emit::decline(
                "gather_witness",
                &GatherWitnessFault::VouchedBytesMoved {
                    key,
                    span,
                    binds: AUDIT_STRIDE,
                },
            )
            .fail_once(key.content_key());
        }
    }
    GatherOutcome {
        identity: GatheredIdentity {
            key: key.content_key(),
            generation: seen.generation,
        },
        vouch: seen.vouch,
    }
}

/// Whether this bind's identity names bytes some earlier gather already moved,
/// or one minted for bytes nothing has ever gathered.
///
/// The distinction decides whether a lookup miss is a fault at all, and it is
/// not recoverable from the identity: a `Fresh` identity is by construction one
/// no cache entry can have been retained under, so it *must* miss and the gather
/// that follows is the witness working. Only a `Vouched` identity that misses
/// says an image was lost.
///
/// Carried beside [`GatheredIdentity`] rather than folded into it because the
/// identity is what the engine *binds on* and this is what it *reports* — a
/// `Fresh` bind still retains under its new identity, which is exactly what lets
/// the next quiet bind hit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GatherVouch {
    /// Both halves said the bytes cannot have moved since the gather that filled
    /// the retained image, so the identity is one the cache may already hold.
    Vouched,
    /// Either half saw a write, the window was re-pointed, or no token could
    /// answer — the generation was spent this bind and names bytes no retained
    /// image was ever built from.
    Fresh,
}

impl GatherVouch {
    /// True only for [`GatherVouch::Vouched`], so a caller cannot spell the
    /// question as "is there an identity" — there always is.
    pub fn is_vouched(self) -> bool {
        matches!(self, Self::Vouched)
    }
}

/// What the engine may bind a retained image on without looking at a byte.
///
/// Produced on **every** bind, not only vouched ones — the generation is what
/// separates the two. Where both halves agree the window's bytes cannot have
/// moved (no guest store into the pages, and no write by this device either) the
/// generation is the one the previous gather retained under, and the engine's
/// lookup hits. Where either half saw a write the generation is spent, so the
/// lookup misses, the bytes are read, and the new identity is what the retain
/// lands under — which is what makes the *following* quiet bind hit.
///
/// [`GatherVouch`] says which of the two this is. Do not reconstruct it from the
/// identity's presence: an absent identity would mean the witness was never
/// asked, and that is not a case [`note_gather`] can return.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatheredIdentity {
    /// Stable name for the window, in the device-wide sampled-identity keyspace.
    pub key: u64,
    /// Generation vouched for these bytes, from
    /// `DeviceState::next_sampled_content_generation`.
    pub generation: u64,
}

/// The witness itself: ask both halves about the last bind of the same window,
/// audit the answer on the stride, and leave the entry describing this bind.
fn observe<M: crate::runtime::host::HostOps>(
    witness: &mut GatherWitness,
    host: &mut M,
    key: GatherKey,
    window: GatherWindow<'_>,
    counts: HostWriteCounts,
    fresh_generation: u64,
) -> GatherObservation {
    let GatherWindow {
        gpas,
        runs,
        span,
        page_size,
    } = window;
    let HostWriteCounts {
        pages_epoch,
        pages_wrote,
    } = counts;

    witness.binds = witness.binds.wrapping_add(1);
    while witness.entries.len() >= MAX_TRACKED_WINDOWS && !witness.entries.contains_key(&key) {
        crate::runtime::drain::note_store_route("gw_window_overflow");
        witness.evict_oldest(host);
    }

    let stale = match witness.entries.get(&key) {
        Some(entry) => entry.gpas != gpas || entry.span != span,
        None => true,
    };
    if stale {
        if let Some(old) = witness.entries.remove(&key) {
            if old.token != 0 {
                host.untrack_guest_writes(old.token);
            }
        }
        let token = host.track_guest_writes(gpas, page_size).unwrap_or(0);
        let gen = if token == 0 {
            0
        } else {
            host.guest_write_gen(token).unwrap_or(0)
        };
        witness.entries.insert(
            key,
            Entry {
                gpas: gpas.to_vec(),
                span,
                token,
                gen,
                // A re-point gathers unconditionally, so folding here would buy
                // nothing the first audit does not: the stride seeds one before
                // there is any vouch for it to check.
                fold: 0,
                fold_valid: false,
                fold_seeded: false,
                binds_since_fold: 0,
                pages_epoch,
                last_seen: witness.binds,
                generation: fresh_generation,
            },
        );
        return GatherObservation {
            verdict: GatherVerdict::Rearmed,
            audit: ContentAudit::Skipped,
            generation: fresh_generation,
            // The other place a generation is assigned, and the only one that
            // assigns unconditionally: a re-pointed window has no previous bind
            // of these pages to have vouched for them.
            vouch: GatherVouch::Fresh,
        };
    }

    let entry = witness
        .entries
        .get_mut(&key)
        .expect("the stale branch above returns for every absent key");
    let gen = if entry.token == 0 {
        0
    } else {
        host.guest_write_gen(entry.token).unwrap_or(0)
    };
    // A generation of 0 on either side is "cannot tell": the token is unarmed,
    // was released with its pages, or has not survived the two harvests the
    // dirty adapter needs before it can answer at all.
    // `pages_wrote == None` cannot happen beside a live entry, and reading a
    // missing answer as quiet would vouch on the strength of not having asked.
    // Taken once: the vouch arm and the refusal arm below want exact complements
    // of this, and two spellings of it is one edit away from a witness that
    // vouches and reports a host write in the same breath.
    let host_quiet = pages_wrote.is_some_and(|seen| !seen.wrote());
    let verdict = if gen == 0 || entry.gen == 0 {
        GatherVerdict::Unarmed
    } else if gen == entry.gen && host_quiet {
        GatherVerdict::Vouched
    } else {
        GatherVerdict::Refused {
            guest_wrote: gen != entry.gen,
            host_wrote_pages: !host_quiet,
        }
    };
    let vouched = matches!(verdict, GatherVerdict::Vouched);

    let audit = if entry.binds_since_fold >= AUDIT_STRIDE {
        // SAFETY: `runs` describe the window this draw is about to gather from,
        // so their pointers are live here for the same reason they are live
        // there. On a vouched bind the gather will be skipped, but the runs were
        // resolved by the same producer in the same call and name the same
        // pages, which the entry's page set is checked against above.
        let fold = unsafe { fold_runs(runs, span) };
        // `fold_valid` is the question "does the stored fold still describe
        // this window", and it is false for two reasons that read identically
        // here and do not mean the same thing: this window has never been
        // folded, or a refusal since the last fold dropped the baseline. The
        // second is the audit declining to compare, and counting it as a seed
        // is what let `gw_audit_unsound=0` read as a clean sweep of a
        // population the fold never looked at.
        let audit = match (vouched && entry.fold_valid, fold == entry.fold) {
            (false, _) if entry.fold_seeded => ContentAudit::Restarted,
            (false, _) => ContentAudit::Seeded,
            (true, true) => ContentAudit::Agreed,
            (true, false) => ContentAudit::Disagreed,
        };
        entry.fold_seeded = true;
        entry.fold = fold;
        entry.fold_valid = true;
        entry.binds_since_fold = 0;
        audit
    } else {
        entry.binds_since_fold += 1;
        // A bind the witness refused may have moved the bytes with nothing
        // reading them, which is precisely when the stored fold stops describing
        // the window.
        entry.fold_valid &= vouched;
        ContentAudit::Skipped
    };

    // Keep the generation only where both halves vouch for the bytes *and* the
    // audit did not just catch them out. A `Disagreed` audit is not only an
    // alarm: the vouch it refutes is live, so dropping the generation here is
    // what stops the next bind serving the stale image again.
    //
    // This one expression decides both what the entry names and what the caller
    // is told the name is worth. Re-deriving the second from `verdict` would get
    // a `Disagreed` audit wrong — that arm vouches and still spends the
    // generation — and a reader comparing the two spellings could not tell which
    // was the rule.
    let kept = vouched && !matches!(audit, ContentAudit::Disagreed);
    if !kept {
        entry.generation = fresh_generation;
    }
    entry.gen = gen;
    entry.pages_epoch = pages_epoch;
    entry.last_seen = witness.binds;
    GatherObservation {
        verdict,
        audit,
        generation: entry.generation,
        vouch: if kept {
            GatherVouch::Vouched
        } else {
            GatherVouch::Fresh
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::vulkan::engine::GuestRun;

    const KEY: GatherKey = GatherKey::Mapping {
        mid: 11,
        base_off: 0,
    };
    const PAGE: usize = 4096;
    const GPAS: [u64; 1] = [8 * PAGE as u64];

    /// A one-page window over `runs`, at `gpas`, judged against a device that has
    /// written nothing.
    fn one_page<'a>(gpas: &'a [u64], runs: &'a [GuestRun]) -> GatherWindow<'a> {
        GatherWindow {
            gpas,
            runs,
            span: PAGE as u64,
            page_size: PAGE,
        }
    }

    /// This device wrote none of the window's pages since the previous bind.
    const QUIET: HostWriteCounts = HostWriteCounts {
        pages_epoch: 1,
        pages_wrote: Some(crate::runtime::host_writes::HostWriteVerdict::Quiet),
    };

    /// One bind, discarding the audit — for the tests that are about the verdict.
    fn verdict<M: crate::runtime::host::HostOps>(
        w: &mut GatherWitness,
        host: &mut M,
        window: GatherWindow<'_>,
        counts: HostWriteCounts,
        gen: u64,
    ) -> GatherVerdict {
        observe(w, host, KEY, window, counts, gen).verdict
    }

    /// Bind `n` times with nothing writing anything, returning the last
    /// observation.
    fn bind_quietly<M: crate::runtime::host::HostOps>(
        w: &mut GatherWitness,
        host: &mut M,
        gpas: &[u64],
        runs: &[GuestRun],
        n: u32,
    ) -> GatherObservation {
        let mut last = None;
        for _ in 0..n {
            last = Some(observe(
                w,
                host,
                KEY,
                one_page(gpas, runs),
                QUIET,
                next_gen(),
            ));
        }
        last.expect("bind_quietly is never called with n == 0")
    }

    /// Bind quietly until the audit next runs, and return that bind.
    ///
    /// Spelled as "until it fires" rather than as a bind count so the tests say
    /// what they mean and do not encode the stride's off-by-ones — the exact
    /// bind an audit lands on is [`AUDIT_STRIDE`]'s business, not theirs.
    fn bind_to_next_audit<M: crate::runtime::host::HostOps>(
        w: &mut GatherWitness,
        host: &mut M,
        gpas: &[u64],
        runs: &[GuestRun],
    ) -> GatherObservation {
        for _ in 0..=2 * AUDIT_STRIDE {
            let seen = observe(w, host, KEY, one_page(gpas, runs), QUIET, next_gen());
            if seen.audit != ContentAudit::Skipped {
                return seen;
            }
        }
        panic!("no audit within two strides, so the fold is never reached at all");
    }

    /// A generation that has never been issued before, as the device's own
    /// counter promises.
    fn next_gen() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(1);
        SEQ.fetch_add(1, Ordering::Relaxed)
    }

    fn run_over(buf: &[u8]) -> GuestRun {
        GuestRun {
            host_ptr: buf.as_ptr() as usize,
            len: buf.len() as u64,
        }
    }

    #[test]
    fn the_fold_sees_a_single_changed_byte_anywhere_in_the_window() {
        let mut buf = vec![7u8; 4096 + 3];
        let base = unsafe { fold_runs(&[run_over(&buf)], buf.len() as u64) };
        for at in [0usize, 1, 8, 1000, 4095, 4096, 4098] {
            let saved = buf[at];
            buf[at] ^= 0x40;
            let moved = unsafe { fold_runs(&[run_over(&buf)], buf.len() as u64) };
            assert_ne!(base, moved, "a flipped byte at {at} folded the same");
            buf[at] = saved;
        }
        assert_eq!(base, unsafe {
            fold_runs(&[run_over(&buf)], buf.len() as u64)
        });
    }

    #[test]
    fn the_fold_is_position_sensitive_so_a_permuted_window_is_not_unchanged() {
        // Distinct bytes at the two swapped indices, or the "permutation" is the
        // identity and the test proves nothing.
        let a: Vec<u8> = (0..512u32).map(|i| (i / 2) as u8).collect();
        let mut b = a.clone();
        assert_ne!(a[0], a[256]);
        b.swap(0, 256);
        assert_ne!(
            unsafe { fold_runs(&[run_over(&a)], a.len() as u64) },
            unsafe { fold_runs(&[run_over(&b)], b.len() as u64) },
            "swapping two words folded the same, so the fold sums rather than orders"
        );
    }

    /// The generation is the whole product of this witness, and its contract is
    /// that it survives exactly as long as the bytes it names.
    ///
    /// Held while both halves vouch, and replaced by every other verdict — the
    /// bytes being unchanged is not the question, because a bind where either
    /// half saw a write is a bind whose bytes nothing has vouched for.
    ///
    /// Asserted on the observation the bind returns rather than by reading the
    /// map back, because that read is what the engine used to do and it cannot
    /// come back absent: every arm here leaves an entry, so an `Option` from it
    /// is `Some` whatever the verdict was. The [`GatherVouch`] beside each
    /// generation is the part that varies, and it is checked at every step.
    #[test]
    fn the_vouched_generation_outlives_a_quiet_bind_and_no_other_kind() {
        let mut host = crate::runtime::host::FakeHost::new();
        let mut w = GatherWitness::default();
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];

        let first = observe(&mut w, &mut host, KEY, one_page(&GPAS, &runs), QUIET, 10);
        assert_eq!((first.generation, first.vouch), (10, GatherVouch::Fresh));

        // Quiet at both halves: the same bytes, so the same generation, and the
        // only bind of the four that names an image an earlier gather filled.
        let quiet = observe(&mut w, &mut host, KEY, one_page(&GPAS, &runs), QUIET, 11);
        assert_eq!((quiet.generation, quiet.vouch), (10, GatherVouch::Vouched));

        // A host write into the pages, with the bytes unchanged. Unchanged is
        // not enough: this device wrote them, so nothing vouches for them.
        let host_wrote = observe(
            &mut w,
            &mut host,
            KEY,
            one_page(&GPAS, &runs),
            HostWriteCounts {
                pages_wrote: Some(crate::runtime::host_writes::HostWriteVerdict::Overlap),
                ..QUIET
            },
            12,
        );
        assert_eq!(
            (host_wrote.generation, host_wrote.vouch),
            (12, GatherVouch::Fresh),
            "a generation survived a write to its own pages"
        );

        // A guest store, likewise.
        buf[3] ^= 0xff;
        host.guest_wrote_page(GPAS[0]);
        let guest_wrote = observe(&mut w, &mut host, KEY, one_page(&GPAS, &runs), QUIET, 13);
        assert_eq!(
            (guest_wrote.generation, guest_wrote.vouch),
            (13, GatherVouch::Fresh)
        );
    }

    /// A window whose bytes and pages both stand still, bound twice: the whole
    /// point of the exercise, and the verdict whose count says what the cache
    /// saves.
    #[test]
    fn a_window_nothing_writes_is_vouched_for_on_the_second_bind() {
        let mut host = crate::runtime::host::FakeHost::new();
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        assert_eq!(
            verdict(&mut w, &mut host, one_page(&GPAS, &runs), QUIET, next_gen()),
            GatherVerdict::Rearmed,
            "first sight has nothing to compare against"
        );
        assert_eq!(
            verdict(&mut w, &mut host, one_page(&GPAS, &runs), QUIET, next_gen()),
            GatherVerdict::Vouched
        );
    }

    /// The hypervisor half saw the store, so the vouch is refused and the bytes
    /// are read — which is what a sound witness looks like on content that really
    /// moved.
    #[test]
    fn a_guest_store_into_the_window_refuses_the_vouch() {
        let mut host = crate::runtime::host::FakeHost::new();
        let mut w = GatherWitness::default();
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        assert_eq!(
            verdict(&mut w, &mut host, one_page(&GPAS, &runs), QUIET, next_gen()),
            GatherVerdict::Rearmed
        );
        buf[100] ^= 0xff;
        host.guest_wrote_page(GPAS[0]);
        assert_eq!(
            verdict(&mut w, &mut host, one_page(&GPAS, &runs), QUIET, next_gen()),
            GatherVerdict::Refused {
                guest_wrote: true,
                host_wrote_pages: false
            }
        );
    }

    /// The whole point of moving the fold onto a stride: a vouched bind reads no
    /// byte of the window at all.
    ///
    /// [`ContentAudit::Skipped`] *is* that statement — it is returned only where
    /// `fold_runs` was not called — so this is the test that would fail if the
    /// fold went back on the per-bind path, and the reason the audit's outcome is
    /// reported rather than kept inside the function.
    #[test]
    fn a_vouched_bind_before_the_stride_reads_none_of_the_window() {
        let mut host = crate::runtime::host::FakeHost::new();
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        // The rearm, then every bind up to but not including the audit.
        let last = bind_quietly(&mut w, &mut host, &GPAS, &runs, AUDIT_STRIDE + 1);
        assert_eq!(last.verdict, GatherVerdict::Vouched);
        assert_eq!(
            last.audit,
            ContentAudit::Skipped,
            "a bind inside the stride folded the window anyway"
        );
        // And the one that lands on the stride does fold, with nothing to compare
        // against yet.
        let due = bind_to_next_audit(&mut w, &mut host, &GPAS, &runs);
        assert_eq!(due.audit, ContentAudit::Seeded);
        // Which then gives the next audit something to check.
        let checked = bind_to_next_audit(&mut w, &mut host, &GPAS, &runs);
        assert_eq!(checked.audit, ContentAudit::Agreed);
    }

    /// A refusal inside a stride leaves the next audit unable to compare, and it
    /// must say so rather than reading as a first fold.
    ///
    /// This is the difference between an alarm that is armed and one that is
    /// not, and until `Restarted` existed the two were one counter. Every
    /// driven boot of this device reports `gw_audit_ok` 0 against
    /// `gw_audit_seed` in the hundreds — because refusals are roughly two in
    /// five binds, and a comparison needs `AUDIT_STRIDE` consecutive vouches of
    /// one window. So `gw_audit_unsound` reading 0 was a check that never ran,
    /// and it read exactly like one that ran and agreed. A real writer escaping
    /// both halves hid behind that zero.
    ///
    /// The test drives one refusal into an otherwise quiet run, which is the
    /// smallest thing that puts the audit in the state the whole workload is
    /// permanently in.
    #[test]
    fn a_refusal_inside_a_stride_makes_the_next_audit_say_it_could_not_compare() {
        let mut host = crate::runtime::host::FakeHost::new();
        let mut w = GatherWitness::default();
        let buf = vec![0x5au8; PAGE];
        let runs = [run_over(&buf)];
        // Rearm, then take a first fold so the window has a baseline at all.
        bind_quietly(&mut w, &mut host, &GPAS, &runs, 1);
        assert_eq!(
            bind_to_next_audit(&mut w, &mut host, &GPAS, &runs).audit,
            ContentAudit::Seeded,
            "the first fold of a window is a seed"
        );
        // With nothing disturbing it, the next audit does compare — this is the
        // case the existing tests live in and the workload never reaches.
        assert_eq!(
            bind_to_next_audit(&mut w, &mut host, &GPAS, &runs).audit,
            ContentAudit::Agreed
        );

        // One refused bind: this device wrote a page of the window. Nothing
        // about the bytes changed, so an audit that still compared would agree
        // — the point is that it is no longer *entitled* to.
        let refused = observe(
            &mut w,
            &mut host,
            KEY,
            one_page(&GPAS, &runs),
            HostWriteCounts {
                pages_epoch: 2,
                pages_wrote: Some(crate::runtime::host_writes::HostWriteVerdict::Overlap),
            },
            next_gen(),
        );
        assert!(
            matches!(
                refused.verdict,
                GatherVerdict::Refused {
                    host_wrote_pages: true,
                    ..
                }
            ),
            "the fixture must actually refuse, or the rest proves nothing"
        );

        assert_eq!(
            bind_to_next_audit(&mut w, &mut host, &GPAS, &runs).audit,
            ContentAudit::Restarted,
            "a dropped baseline must not report as a window being met for the \
             first time — that is the reading that made a dead alarm look clean"
        );
    }

    /// The unsound case, produced deliberately: bytes changed under pages neither
    /// half of the witness saw written. This is the shape a host-side writer into
    /// guest RAM makes, and it is what the audit exists to catch — so if a driven
    /// boot ever reports `gw_audit_unsound`, this test says what that means.
    ///
    /// The audit is a repair as well as an alarm: the generation it refutes is
    /// live, so it must not survive the bind that caught it.
    #[test]
    fn bytes_moving_under_a_vouch_are_caught_by_the_audit_and_cost_the_generation() {
        let mut host = crate::runtime::host::FakeHost::new();
        let mut w = GatherWitness::default();
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        // Rearm, then seed a fold the next audit can compare against.
        bind_quietly(&mut w, &mut host, &GPAS, &runs, 1);
        let seeded = bind_to_next_audit(&mut w, &mut host, &GPAS, &runs);
        assert_eq!(seeded.audit, ContentAudit::Seeded);
        let vouched_gen = seeded.generation;

        // No `guest_wrote_page` and no host write recorded: the bytes move with
        // both halves of the witness none the wiser.
        buf[7] ^= 0xff;
        let caught = bind_to_next_audit(&mut w, &mut host, &GPAS, &runs);
        assert_eq!(
            caught.verdict,
            GatherVerdict::Vouched,
            "the witness is what is being caught out, so it must still be vouching"
        );
        assert_eq!(caught.audit, ContentAudit::Disagreed);
        assert_ne!(
            caught.generation, vouched_gen,
            "the refuted generation survived the audit that refuted it, so the \
             next bind serves the stale image again"
        );
        // The one bind where the verdict and the vouch disagree, and the reason
        // the engine is told the vouch rather than the verdict: this bind
        // vouches and still spends its generation, so an engine deriving
        // "vouched" from the verdict would count a guaranteed miss as a
        // retention failure.
        assert_eq!(
            caught.vouch,
            GatherVouch::Fresh,
            "a generation the audit just spent was reported as one the cache \
             could still be holding an image under"
        );
    }

    /// A host that cannot observe guest writes must never vouch, however still
    /// the bytes are. Fail closed: half a witness is not a witness.
    #[test]
    fn a_host_that_cannot_watch_guest_writes_never_vouches() {
        let mut host = crate::runtime::host::FakeHost::new();
        host.guest_writes_unobservable = true;
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        assert_eq!(
            verdict(&mut w, &mut host, one_page(&GPAS, &runs), QUIET, next_gen()),
            GatherVerdict::Rearmed
        );
        assert_eq!(
            verdict(&mut w, &mut host, one_page(&GPAS, &runs), QUIET, next_gen()),
            GatherVerdict::Unarmed
        );
    }

    /// A window re-pointed at different pages has no predecessor, even though its
    /// key repeats. Comparing across the move would compare two different surfaces.
    #[test]
    fn a_window_whose_pages_move_rearms_rather_than_comparing_across_the_move() {
        let mut host = crate::runtime::host::FakeHost::new();
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let moved = [9 * PAGE as u64];
        assert_eq!(
            verdict(&mut w, &mut host, one_page(&GPAS, &runs), QUIET, next_gen()),
            GatherVerdict::Rearmed
        );
        assert_eq!(
            verdict(
                &mut w,
                &mut host,
                one_page(&moved, &runs),
                QUIET,
                next_gen()
            ),
            GatherVerdict::Rearmed,
            "same key, different pages: nothing to compare"
        );
        assert_eq!(
            verdict(
                &mut w,
                &mut host,
                one_page(&moved, &runs),
                QUIET,
                next_gen()
            ),
            GatherVerdict::Vouched
        );
    }

    /// A refused bind between two audits invalidates the stored fold, because it
    /// may have moved the bytes with nothing reading them.
    ///
    /// Without this the next audit would compare against a fold from before a
    /// legitimate repaint and report `gw_audit_unsound` for a witness that was
    /// right — an alarm that cries wolf is worse than no alarm, since the whole
    /// value of this one is that a nonzero count means something.
    #[test]
    fn a_refused_bind_between_audits_does_not_leave_a_false_alarm_behind() {
        let mut host = crate::runtime::host::FakeHost::new();
        let mut w = GatherWitness::default();
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        // Rearm, seed a fold, and reach a vouched steady state.
        bind_quietly(&mut w, &mut host, &GPAS, &runs, 1);
        assert_eq!(
            bind_to_next_audit(&mut w, &mut host, &GPAS, &runs).audit,
            ContentAudit::Seeded
        );

        // A guest store the hypervisor *does* see, repainting the window. The
        // gather happens, so nothing is stale — but the stored fold is now from
        // before the repaint.
        buf[11] ^= 0xff;
        host.guest_wrote_page(GPAS[0]);
        assert!(matches!(
            observe(
                &mut w,
                &mut host,
                KEY,
                one_page(&GPAS, &runs),
                QUIET,
                next_gen()
            )
            .verdict,
            GatherVerdict::Refused { .. }
        ));

        let next = bind_to_next_audit(&mut w, &mut host, &GPAS, &runs);
        assert_eq!(
            next.audit,
            ContentAudit::Restarted,
            "the audit compared against a fold from before a repaint it knew about"
        );
    }

    #[test]
    fn the_fold_stops_at_span_even_when_the_runs_are_longer() {
        let buf = vec![3u8; 256];
        let short = unsafe { fold_runs(&[run_over(&buf)], 64) };
        let head = vec![3u8; 64];
        assert_eq!(short, unsafe { fold_runs(&[run_over(&head)], 64) });
    }
}
