//! Diagnostic stability witness for sampled guest-resource windows.
//!
//! This combines the decoded resource-validity transition with exact device
//! writes and reports whether both accounts stayed quiet. It is telemetry, not
//! a content-identity contract: the validity transition is a synchronization
//! statement consumed at submission construction, not a version emitted for
//! every CPU write to unified shared storage. The full-content audit has
//! observed bytes move while both accounts reported `Vouched`.
//!
//! That gap is a property of the API rather than a shortfall in these two
//! accounts, so widening them cannot close it. The contract is that a CPU store
//! into shared storage is visible to a later GPU read through a texture
//! aliasing that storage, with no API call announcing the write: no flush, no
//! synchronization, no revalidation. `scripts/metal-oracle/linear-alias.swift`
//! states that as a runnable question, and a native Metal host answers
//! `visible` three times over -- including for stores landing after the
//! resource has already been sampled once.
//!
//! The consequence for this module is worth being blunt about. A guest that
//! rasterizes into its own shared storage is entitled to say nothing, so there
//! is no announcement for any witness to observe, and no pair of accounts
//! assembled from decoded traffic can be sound over that writer. This is
//! therefore an alarm permanently: do not promote it into a correctness gate,
//! and do not add a third account hoping to reach soundness. A rail that
//! samples the guest's live storage has no such question to answer, which is
//! the structural fix; the witness exists to keep the copying rails' exposure
//! measurable until such a rail carries them.
//!
//! The content fold is the check that exposes that missing writer. It is
//! disabled normally and runs on every bind when explicitly enabled. A
//! disagreement is fail-visible without changing shipping behavior. In
//! particular, neither `GatherVouch` nor `GatheredIdentity` may select a
//! retained copied image; direct images are safe because they sample the live
//! storage rather than relying on the witness.
//!
//! Entries are unbounded while their owners are live. Task entries retire with
//! the task and mapping entries with the mapping; device reset clears both.

pub use reims_vgpu_core::{
    fold_runs, AuditDensity, ContentAudit, GatherKey, GatherObservation, GatherOutcome,
    GatherReadings as WitnessReadings, GatherVerdict, GatherVouch, GatherWindow, GatherWitness,
    GatheredIdentity, GuestWriteReach as PendingWrites, StatedGeneration, StatedGuestWrite,
    VouchPolicy, AUDIT_REBASELINE_LIMIT,
};

/// Resolve this process's diagnostic sampling policy at the composition edge.
pub fn audit_density() -> AuditDensity {
    match crate::env::switch(crate::env::GATHER_AUDIT_ALL) {
        crate::env::Switch::On => AuditDensity::EveryBind,
        _ => AuditDensity::default(),
    }
}

/// Resolve whether this process may act on the vouches the contract grants.
///
/// Only `off` withholds. An unset or misspelled value keeps the contract arm,
/// so a typo cannot quietly put a boot on the slow rail and have its timings
/// read as the shipping ones.
pub fn vouch_policy() -> VouchPolicy {
    match crate::env::switch(crate::env::GATHER_VOUCH) {
        crate::env::Switch::Off => VouchPolicy::Withheld,
        _ => VouchPolicy::default(),
    }
}

fn pending_writes_over(
    executor: &dyn crate::runtime::executor::Executor,
    gpas: &[u64],
) -> PendingWrites {
    if !executor.guest_writes_outstanding() {
        return PendingWrites::Quiet;
    }
    executor.guest_writes_reaching(gpas)
}

/// Census name for one reach answer.
///
/// `gw_pending_quiet` and `gw_pending_disjoint` are counted apart because they
/// are reached by different routes and send an investigation to opposite halves
/// of the device. `quiet` means this device held **no** outstanding guest write
/// when the gather ran, so the pages were never examined; `disjoint` means it
/// held one, the pages were examined, and the write lands in none of them. Both
/// license the gather, and a single counter covering both cannot say whether a
/// gather that read stale bytes did so because the debt was never armed or
/// because the footprint comparison was wrong.
fn pending_writes_route(pending: PendingWrites) -> &'static str {
    match pending {
        PendingWrites::Quiet => "gw_pending_quiet",
        PendingWrites::Disjoint => "gw_pending_disjoint",
        PendingWrites::Overlap => "gw_pending_overlap",
        PendingWrites::Unnamed => "gw_pending_unnamed",
    }
}

#[cfg(test)]
fn observe(
    witness: &mut GatherWitness,
    key: GatherKey,
    window: GatherWindow<'_>,
    readings: WitnessReadings,
    fresh_generation: u64,
) -> GatherObservation {
    // SAFETY: tests construct runs over live byte buffers retained across this call.
    unsafe { witness.observe(key, window, readings, fresh_generation) }
}

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
    /// IOSurface texture mapping-backed sampled bind.
    IOSurface,
    /// IOSurface plane view serialized IOSurface plane view (the video path).
    IOSurfacePlaneView,
}

impl GatherRail {
    /// Census names for the rail's gather count and its gathered kilobytes.
    fn names(self) -> (&'static str, &'static str) {
        match self {
            Self::Linear => ("gw_rail_linear", "gw_rail_linear_kb"),
            Self::IOSurface => ("gw_rail_iosurface", "gw_rail_iosurface_kb"),
            Self::IOSurfacePlaneView => ("gw_rail_t5", "gw_rail_t5_kb"),
        }
    }
}

/// Which sampled window a witness entry describes.
///
/// The two shapes are the two ways the producers name a window: a task-GVA span
/// (the linear texture rail, which has no mapping) and a mapping-relative offset
/// (the IOSurface texture and IOSurface plane view rails). Those two rails can name the same
/// `(mid, base_off)` for a single-plane surface, and that is harmless — same
/// mapping, same offset and same span is the same bytes.
fn gather_key_log_token(key: GatherKey) -> String {
    // Whitespace-free rendering for the always-on log, which is parsed by
    // splitting on spaces.
    match key {
        GatherKey::TaskGva {
            task_id,
            resource,
            gva,
        } => format!(
            "gva:{task_id}:{}:{}:{gva:#x}",
            resource.index(),
            resource.generation()
        ),
        GatherKey::Mapping {
            mapping,
            base_offset,
        } => format!("map:{}:{:#x}", mapping.get(), base_offset.get()),
    }
}

/// Guest-declared content generation in the identity space that owns it.
///
/// Mapping-backed resources carry the mapping generation. Task-local GVA
/// resources carry the generation of the resource object whose dirty bit the
/// submission consumed. The variants cannot compare across namespaces, which
/// makes replacing one resource shape with the other a write rather than a
/// coincidental equal integer.
/// What this witness reports when its soundness audit finds stale bytes.
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
                ("window", gather_key_log_token(*key)),
                ("span", span.to_string()),
                ("binds", binds.to_string()),
            ],
        }
    }
}

/// Record one zero-copy sampled gather against the guest-write witness, and
/// report it to the census.
///
/// Called from the producers with the window already resolved and with the
/// decoded resource generation captured in that resource's identity space.
///
/// # `stated_gen` is `Option` because the guest does not always make a statement
///
/// `None` means this device is not *told* about writes to the resource, which
/// is a different fact from being told there were none. Only
/// [`reims_vgpu_protocol::StorageMode::Managed`] obliges the guest to announce
/// a CPU write, so on every other mode the resource's generation sits still
/// while the guest rasterizes into it, and a `Some` built from that generation
/// would be a statement of silence the guest never made. The witness already
/// fails closed on `None` — it reports [`StatedGuestWrite::Unaddressed`], which
/// never vouches — so passing the absence through is all that is required.
///
/// Passing `Some` where the guest is silent is the defect this signature
/// exists to prevent: it costs the guest content, not throughput, and its only
/// instrument is the sampled content audit.
///
/// # Why this does not return an `Option`
///
/// It used to, and the `Option` was `Some` on every path: the identity was read
/// back with `vouched_identity`, which answered "is this window tracked", and
/// [`observe`] leaves an entry for every key it is given — the re-point branch
/// inserts one and returns, and the surviving branch holds a `&mut` to one. The engine spent
/// a boot counting `identity.is_some()` as the witness's verdict and read the
/// resulting zero as "the witness never refused a gather". It cannot refuse
/// through this return value at all; [`GatherVouch`] is where the verdict lives.
#[must_use = "the identity is what lets the engine skip the gather; dropping it \
              silently keeps the copy"]
pub fn note_gather(
    state: &mut crate::runtime::Device,
    rail: GatherRail,
    key: GatherKey,
    stated_gen: Option<StatedGeneration>,
    window: GatherWindow<'_>,
) -> GatherOutcome {
    use crate::runtime::drain::{note_store_route, note_store_route_n};
    let _phase = crate::runtime::sampled_phase::Span::open(
        crate::runtime::sampled_phase::Part::GatherWitness,
    );

    let span = window.span;
    let (rail_count, rail_kb) = rail.names();
    note_store_route(rail_count);
    note_store_route_n(rail_kb, span / 1024);

    // Both writers' accounts, taken before the witness is touched: the
    // page-exact question needs the epoch recorded at the *previous* bind, which
    // is inside the witness, and the ring that answers it is read through the
    // same device state.
    let counts = WitnessReadings {
        pages_epoch: state.content.host_writes.epoch(),
        pages_wrote: state
            .content
            .sampled
            .gather_witness
            .previous_pages_epoch(&key)
            .map(|since| {
                state
                    .content
                    .host_writes
                    .wrote_any_since(since, window.gpas)
            }),
        // The guest's own statement about this resource, captured by the caller
        // in the identity space that owns it, and absent where the guest makes
        // no statement at all.
        stated_gen,
        pending: pending_writes_over(state.executor.as_ref(), window.gpas),
    };
    // Every bind, vouched or not, so the route is a denominator rather than a
    // tally of refusals — the reading wanted is what fraction of binds this
    // device has a copy in flight over, and a count with no denominator cannot
    // say whether a repair moved it.
    note_store_route(pending_writes_route(counts.pending));
    // Report the host-write half's grounds, not just its answer. Three of its
    // four non-quiet values are this device declining to rule a write out rather
    // than one that landed here, and they want different repairs — name the
    // writer's pages, widen the ring, or stop writing the window at all. Taken
    // for every bind that had a previous one to ask about, so the split covers
    // the vouched binds too. `gw_hw_no_writes + gw_hw_disjoint` is the
    // denominator; keeping them separate says whether this question walked
    // the window or the global epoch answered it first.
    if let Some(verdict) = counts.pages_wrote {
        note_store_route(verdict.route());
    }
    // A generation is issued from the device-global counter and never reused, so
    // it is taken before the witness runs and spent only if the witness refuses
    // to vouch for the previous one. An unspent generation is not a leak: the
    // counter's whole contract is that a value is issued once and never again.
    let fresh = state.next_sampled_content_generation();
    // SAFETY: producers resolve these runs for the gather performed by this
    // same bind; they remain live through this synchronous witness call.
    let seen = unsafe {
        state
            .content
            .sampled
            .gather_witness
            .observe(key, window, counts, fresh)
    };

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
        GatherVerdict::Withheld => note_store_route("gw_withheld"),
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
        // The arm holding itself open across a refusal. Costs one fold of a
        // window the gather reads anyway, and the count is what the alarm pays
        // to stay reachable at all.
        ContentAudit::Rebaselined => note_store_route("gw_audit_rebaseline"),
        ContentAudit::Agreed => note_store_route("gw_audit_ok"),
        // Read beside `gw_audit_ok` the same way `gw_audit_restart` is: it is
        // the audit's blind spot, and a boot where it dominates has an alarm
        // that is looking away rather than one that is seeing nothing.
        ContentAudit::Indebted => note_store_route("gw_audit_indebted"),
        ContentAudit::Disagreed => {
            note_store_route("gw_audit_unsound");
            // Once per window: a writer escaping both halves escapes them on
            // every bind, and the second line says nothing the first did not.
            // The count above carries the magnitude.
            let mut emission = crate::observe::emit::Emit::decline(
                "gather_witness",
                &GatherWitnessFault::VouchedBytesMoved {
                    key,
                    span,
                    binds: 1,
                },
            );
            if let GatherKey::TaskGva { resource, .. } = key {
                if let Some((task, object)) = state.task_objects.resources.owner(resource) {
                    emission = emission
                        .field("object", object.get())
                        .field("owner_task", task.get());
                }
                if let Some(StatedGeneration::TaskResource(
                    reims_vgpu_core::ResourceWriteStamp::Resolved { version, .. },
                )) = stated_gen
                {
                    emission = emission.field("content_version", version.get());
                }
            }
            emission.fail_once(key.content_key());
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

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_memory::GuestRun;

    const KEY: GatherKey = GatherKey::Mapping {
        mapping: reims_vgpu_protocol::MappingId::new(11),
        base_offset: reims_vgpu_protocol::ByteOffset::new(0),
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

    /// Neither the guest nor this device wrote the window since the previous
    /// bind.
    const QUIET: WitnessReadings = WitnessReadings {
        pages_epoch: 1,
        pages_wrote: Some(reims_vgpu_core::HostWriteVerdict::NoWrites),
        pending: PendingWrites::Disjoint,
        stated_gen: Some(StatedGeneration::Mapping(0)),
    };

    /// The two reach answers that license a gather must stay tellable apart on
    /// the census, and must license it identically.
    ///
    /// They are reached by different routes -- `Quiet` without the pages being
    /// examined at all, `Disjoint` only after examining them -- and one counter
    /// covering both cannot say whether a gather that returned stale bytes did
    /// so because the write debt was never armed or because the footprint
    /// comparison was wrong. Those two have opposite repairs, which is why the
    /// split is worth a variant rather than a comment.
    ///
    /// The audit half is the half that could regress silently: the admission is
    /// spelled with `matches!`, which `rustc` does not exhaustiveness-check, so
    /// nothing but this test fails if a later variant stops licensing the audit.
    #[test]
    fn a_quiet_ledger_and_a_tested_disjoint_window_are_counted_apart_and_license_alike() {
        assert_ne!(
            pending_writes_route(PendingWrites::Quiet),
            pending_writes_route(PendingWrites::Disjoint),
            "one counter for both routes is the zero-sampled-at-the-wrong-place trap"
        );

        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let bind = |w: &mut GatherWitness, pending| {
            observe(
                w,
                KEY,
                one_page(&GPAS, &runs),
                WitnessReadings { pending, ..QUIET },
                next_gen(),
            )
        };

        for licensing in [PendingWrites::Quiet, PendingWrites::Disjoint] {
            // Fresh witness per arm: the audit is stateful, and a shared one
            // would let the first arm's fold decide the second's outcome.
            let mut w = witness_auditing(AuditDensity::EveryBind);
            let settled = bind_quietly(&mut w, &GPAS, &runs, 4);
            assert_eq!(
                settled.audit,
                ContentAudit::Agreed,
                "control: the audit must actually be running"
            );
            let seen = bind(&mut w, licensing);
            assert_ne!(
                seen.audit,
                ContentAudit::Indebted,
                "{licensing:?} licenses the gather, so the audit may compare across it"
            );
        }

        for withholding in [PendingWrites::Overlap, PendingWrites::Unnamed] {
            let mut w = witness_auditing(AuditDensity::EveryBind);
            let _ = bind_quietly(&mut w, &GPAS, &runs, 4);
            assert_eq!(
                bind(&mut w, withholding).audit,
                ContentAudit::Indebted,
                "{withholding:?} does not license a fold across an unlanded copy"
            );
        }
    }

    /// One bind, discarding the audit — for the tests that are about the verdict.
    fn verdict(
        w: &mut GatherWitness,
        window: GatherWindow<'_>,
        counts: WitnessReadings,
        gen: u64,
    ) -> GatherVerdict {
        observe(w, KEY, window, counts, gen).verdict
    }

    /// Bind `n` times with nothing writing anything, returning the last
    /// observation.
    fn bind_quietly(
        w: &mut GatherWitness,
        gpas: &[u64],
        runs: &[GuestRun],
        n: u32,
    ) -> GatherObservation {
        let mut last = None;
        for _ in 0..n {
            last = Some(observe(w, KEY, one_page(gpas, runs), QUIET, next_gen()));
        }
        last.expect("bind_quietly is never called with n == 0")
    }

    /// Bind quietly until an explicitly enabled every-bind audit next runs.
    fn bind_to_next_audit(
        w: &mut GatherWitness,
        gpas: &[u64],
        runs: &[GuestRun],
    ) -> GatherObservation {
        bind_to_next_audit_with(w, gpas, runs, QUIET)
    }

    fn bind_to_next_audit_with(
        w: &mut GatherWitness,
        gpas: &[u64],
        runs: &[GuestRun],
        counts: WitnessReadings,
    ) -> GatherObservation {
        // First sight rearms, the next bind advances the audit arm, and the
        // third seeds it. Those are states of the witness, not a product-side
        // sampling interval.
        for _ in 0..3 {
            let seen = observe(w, KEY, one_page(gpas, runs), counts, next_gen());
            if seen.audit != ContentAudit::Skipped {
                return seen;
            }
        }
        panic!("an enabled every-bind audit did not reach its next audit state");
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
        let mut w = GatherWitness::default();
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];

        let first = observe(&mut w, KEY, one_page(&GPAS, &runs), QUIET, 10);
        assert_eq!((first.generation, first.vouch), (10, GatherVouch::Fresh));

        // Quiet at both halves: the same bytes, so the same generation, and the
        // only bind of the four that names an image an earlier gather filled.
        let quiet = observe(&mut w, KEY, one_page(&GPAS, &runs), QUIET, 11);
        assert_eq!((quiet.generation, quiet.vouch), (10, GatherVouch::Vouched));

        // A host write into the pages, with the bytes unchanged. Unchanged is
        // not enough: this device wrote them, so nothing vouches for them.
        let host_wrote = observe(
            &mut w,
            KEY,
            one_page(&GPAS, &runs),
            WitnessReadings {
                pages_wrote: Some(reims_vgpu_core::HostWriteVerdict::Overlap),
                ..QUIET
            },
            12,
        );
        assert_eq!(
            (host_wrote.generation, host_wrote.vouch),
            (12, GatherVouch::Fresh),
            "a generation survived a write to its own pages"
        );

        // A guest store, likewise. The resource-table statement advances the
        // resource generation consumed by this bind.
        buf[3] ^= 0xff;
        let guest_wrote = observe(
            &mut w,
            KEY,
            one_page(&GPAS, &runs),
            WitnessReadings {
                stated_gen: Some(StatedGeneration::Mapping(1)),
                ..QUIET
            },
            13,
        );
        assert_eq!(
            (guest_wrote.generation, guest_wrote.vouch),
            (13, GatherVouch::Fresh)
        );
    }

    /// The guest's resource statement answers only where it addresses the
    /// sampled resource.
    ///
    /// Four claims, and the first two are what make the reading usable: a
    /// generation that has not moved is [`StatedGuestWrite::Quiet`] *including at
    /// generation 0*, and one that has moved is [`StatedGuestWrite::Wrote`]. The
    /// third is the fail-closed rule — a window the channel cannot address at
    /// either bind is `Unaddressed` and never quiet, so an absent statement is
    /// not read as a statement of silence. The fourth is that a re-point reports
    /// `Unaddressed` too, because it has no previous bind to compare against and
    /// gathers unconditionally.
    #[test]
    fn the_stated_channel_answers_only_where_the_guest_addresses_it() {
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let stated = |gen: Option<u32>| WitnessReadings {
            stated_gen: gen.map(StatedGeneration::Mapping),
            ..QUIET
        };

        // First sight of the window: a re-point, so no comparison exists yet.
        let first = observe(&mut w, KEY, one_page(&GPAS, &runs), stated(Some(0)), 10);
        assert_eq!(first.stated, StatedGuestWrite::Unaddressed);

        // Generation 0 twice is a real quiet answer, not an absent one — the
        // mapping has been addressed and the guest has written it zero times.
        let quiet = observe(&mut w, KEY, one_page(&GPAS, &runs), stated(Some(0)), 11);
        assert_eq!(quiet.stated, StatedGuestWrite::Quiet);

        // The guest states a CPU write: `resource_validity::apply` bumps the
        // mapping's generation, and the channel reports it.
        let wrote = observe(&mut w, KEY, one_page(&GPAS, &runs), stated(Some(1)), 12);
        assert_eq!(wrote.stated, StatedGuestWrite::Wrote);

        // Settled at the new generation, quiet again.
        let settled = observe(&mut w, KEY, one_page(&GPAS, &runs), stated(Some(1)), 13);
        assert_eq!(settled.stated, StatedGuestWrite::Quiet);

        // The mapping goes away. Fail closed: not quiet, whatever the device
        // write record says, and the bind before it does not become quiet
        // retroactively.
        let gone = observe(&mut w, KEY, one_page(&GPAS, &runs), stated(None), 14);
        assert_eq!(gone.stated, StatedGuestWrite::Unaddressed);
        let still_gone = observe(&mut w, KEY, one_page(&GPAS, &runs), stated(None), 15);
        assert_eq!(still_gone.stated, StatedGuestWrite::Unaddressed);
    }

    /// A resource whose guest never announces cannot freeze its sampled copy.
    ///
    /// This is the consequence the storage-mode gate exists to produce, stated
    /// where it is enforced. A texture in a silent `MTLStorageMode` reaches the
    /// witness with **no** stated generation, however many times it is bound,
    /// because the producer has nothing to state. Every one of those binds must
    /// come back `Fresh` at the generation it was offered — never `Vouched`,
    /// and never at a generation an earlier bind already used.
    ///
    /// A repeated identity is what the defect looked like: the engine memoizes
    /// on `(key, generation)`, so a vouch that holds the generation still makes
    /// the next bind reuse the previous copy. On a CPU-rasterized glyph atlas
    /// that froze the sampled bytes at whatever the atlas held when it was
    /// first gathered, which on the Maps workload was before any type had been
    /// drawn into it.
    #[test]
    fn a_silent_resource_is_never_vouched_and_never_repeats_a_generation() {
        let mut w = GatherWitness::default();
        let buf = vec![0x5au8; PAGE];
        let runs = [run_over(&buf)];
        let silent = WitnessReadings {
            stated_gen: None,
            ..QUIET
        };

        let mut issued = Vec::new();
        for generation in 20..40u64 {
            let seen = observe(&mut w, KEY, one_page(&GPAS, &runs), silent, generation);
            assert_eq!(
                seen.stated,
                StatedGuestWrite::Unaddressed,
                "an absent statement is not a statement of silence"
            );
            assert_eq!(
                seen.vouch,
                GatherVouch::Fresh,
                "a silent resource must re-read its bytes every bind"
            );
            assert_eq!(
                seen.generation, generation,
                "the offered generation must be spent, so the memo cannot hit"
            );
            issued.push(seen.generation);
        }

        let distinct: std::collections::BTreeSet<u64> = issued.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            issued.len(),
            "a repeated identity is what let the frozen atlas survive"
        );
    }

    /// [`VouchPolicy::Withheld`] takes away vouches and takes away nothing else.
    ///
    /// Both halves matter, and only the second is easy to get wrong. If the
    /// ablation reached any other verdict it would stop being a one-variable
    /// arm, and a content difference between the two boots would no longer
    /// isolate the memo — which is the entire reason the switch exists.
    ///
    /// The two witnesses are fed byte-identical readings in the same order, so
    /// any divergence below is the policy and nothing else.
    #[test]
    fn withholding_removes_the_vouch_and_leaves_every_other_verdict_alone() {
        let buf = vec![0x3cu8; PAGE];
        let runs = [run_over(&buf)];
        let withheld = |vouch| {
            GatherWitness::with_policies(reims_vgpu_core::GatherPolicies {
                vouch,
                ..Default::default()
            })
        };

        // A resource that states a generation and holds it still: the contract
        // grants a vouch here, so this is the one bind the policy may change.
        let mut contract = withheld(VouchPolicy::Contract);
        let mut ablated = withheld(VouchPolicy::Withheld);
        for w in [&mut contract, &mut ablated] {
            let _ = observe(w, KEY, one_page(&GPAS, &runs), QUIET, 1);
        }
        let granted = observe(&mut contract, KEY, one_page(&GPAS, &runs), QUIET, 2);
        let declined = observe(&mut ablated, KEY, one_page(&GPAS, &runs), QUIET, 2);

        assert_eq!(granted.verdict, GatherVerdict::Vouched);
        assert_eq!(granted.vouch, GatherVouch::Vouched);
        assert_eq!(
            granted.generation, 1,
            "a vouch holds the identity, which is what lets the memo hit"
        );

        assert_eq!(declined.verdict, GatherVerdict::Withheld);
        assert_eq!(declined.vouch, GatherVouch::Fresh);
        assert_eq!(
            declined.generation, 2,
            "a withheld vouch must spend the offered generation, or the memo hits anyway"
        );
        assert_eq!(
            declined.stated, granted.stated,
            "the contract's own reading of the guest is not the policy's to change"
        );

        // A resource the contract already refuses: both arms must agree
        // exactly, verdict included, or the ablation is masking refusals.
        let silent = WitnessReadings {
            stated_gen: None,
            ..QUIET
        };
        let mut contract = withheld(VouchPolicy::Contract);
        let mut ablated = withheld(VouchPolicy::Withheld);
        for w in [&mut contract, &mut ablated] {
            let _ = observe(w, KEY, one_page(&GPAS, &runs), silent, 1);
        }
        let a = observe(&mut contract, KEY, one_page(&GPAS, &runs), silent, 2);
        let b = observe(&mut ablated, KEY, one_page(&GPAS, &runs), silent, 2);

        assert_eq!(a.verdict, GatherVerdict::Unarmed);
        assert_eq!(b.verdict, a.verdict, "the policy invented a refusal");
        assert_eq!(b.vouch, a.vouch);
        assert_eq!(b.generation, a.generation);
        assert_eq!(b.stated, a.stated);
    }

    /// A witness at a stated audit density, for the two tests that are about the
    /// density rather than about the witness.
    fn witness_auditing(density: AuditDensity) -> GatherWitness {
        GatherWitness::with_policies(reims_vgpu_core::GatherPolicies {
            audit: density,
            ..Default::default()
        })
    }

    /// [`AuditDensity::EveryBind`] judges every bind it can, while the normal
    /// shipping witness performs no byte audit.
    ///
    /// Both arms are asserted because only the pair says the switch does
    /// anything: the dense arm alone would pass against a witness that always
    /// audited, and the disabled arm alone against one that never did.
    ///
    /// Six binds rather than a computed count, and the dense arm is allowed its
    /// first three — a comparison needs a baseline bind and a bind to spend it
    /// on, and the first sight of a window is a rearm that has neither.
    #[test]
    fn every_bind_compares_while_the_shipping_witness_does_not_read_bytes() {
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let compares = |density| {
            let mut w = witness_auditing(density);
            (0..6)
                .map(|_| observe(&mut w, KEY, one_page(&GPAS, &runs), QUIET, next_gen()))
                .filter(|seen| seen.audit == ContentAudit::Agreed)
                .count()
        };
        assert_eq!(
            compares(AuditDensity::Disabled),
            0,
            "the shipping witness must not turn a byte fold into render-thread work"
        );
        assert_eq!(
            compares(AuditDensity::EveryBind),
            3,
            "every bind after the first three must compare against the bind before it"
        );
    }

    /// The reading the switch exists to produce: a writer that escapes **both**
    /// halves of the witness is caught on the very next bind.
    ///
    /// This is the failure a write-channel witness cannot report without reading
    /// the bytes, so the audit is the instrument that makes missed writers
    /// visible.
    /// The bytes here move with no guest store and no recorded host write, which
    /// is exactly the shape of an unrecorded writer.
    ///
    /// The vouch is asserted too: the audit observes the contract without
    /// becoming a second input to it.
    #[test]
    fn an_unrecorded_write_is_reported_without_overriding_the_contract_vouch() {
        let mut w = witness_auditing(AuditDensity::EveryBind);
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let settled = bind_quietly(&mut w, &GPAS, &runs, 4);
        assert_eq!(
            (settled.audit, settled.vouch),
            (ContentAudit::Agreed, GatherVouch::Vouched),
            "the window has to be under a live vouch before the escape means anything"
        );

        // Neither half is told. This is the writer the module's whole soundness
        // argument assumes does not exist.
        buf[2048] ^= 0xff;

        let caught = observe(&mut w, KEY, one_page(&GPAS, &runs), QUIET, 77);
        assert_eq!(
            (caught.verdict, caught.audit),
            (GatherVerdict::Vouched, ContentAudit::Disagreed),
            "both halves vouched and the bytes had moved, which is the alarm's whole purpose"
        );
        assert_eq!(
            (caught.generation, caught.vouch),
            (settled.generation, GatherVouch::Vouched),
            "an instrument must not replace the decoded contract's identity"
        );
    }

    /// The audit declines to compare across a copy this device has submitted and
    /// the GPU has not run — and the *vouch* is untouched by it.
    ///
    /// Both halves matter and they pull opposite ways. The fold is a CPU read of
    /// guest pages and is ordered against that copy by nothing, so comparing
    /// across it reports the device's own queue as a stale image. The gather the
    /// cache elides is a GPU copy on the same queue as the writeback, so it *is*
    /// ordered and the vouch is still sound — making this refuse the vouch too
    /// would cost re-gathers to fix a defect in the instrument.
    ///
    /// `Unnamed` is asserted beside `Overlap` because a footprint nobody could
    /// name is not a proof of disjointness, and reading it as one is how the
    /// blind spot would come back as "we could not tell, so we compared".
    #[test]
    fn an_unlanded_copy_stops_the_audit_comparing_and_leaves_the_vouch_alone() {
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let bind = |w: &mut GatherWitness, pending| {
            observe(
                w,
                KEY,
                one_page(&GPAS, &runs),
                WitnessReadings { pending, ..QUIET },
                next_gen(),
            )
        };

        // Quiet queue: the audit reaches a comparison, which is the control —
        // without it the assertions below would pass against an audit that never
        // ran at all.
        let mut w = witness_auditing(AuditDensity::EveryBind);
        let settled = bind_quietly(&mut w, &GPAS, &runs, 4);
        assert_eq!(
            (settled.verdict, settled.audit),
            (GatherVerdict::Vouched, ContentAudit::Agreed)
        );

        for pending in [PendingWrites::Overlap, PendingWrites::Unnamed] {
            let seen = bind(&mut w, pending);
            assert_eq!(
                seen.audit,
                ContentAudit::Indebted,
                "{pending:?} folded across a copy that has not landed"
            );
            assert_eq!(
                seen.verdict,
                GatherVerdict::Vouched,
                "{pending:?} moved the vouch, which is ordered behind that copy and did not need to"
            );
            assert_eq!(seen.vouch, GatherVouch::Vouched);
        }
    }

    /// A window whose bytes and pages both stand still, bound twice: the whole
    /// point of the exercise, and the verdict whose count says what the cache
    /// saves.
    #[test]
    fn a_window_nothing_writes_is_vouched_for_on_the_second_bind() {
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        assert_eq!(
            verdict(&mut w, one_page(&GPAS, &runs), QUIET, next_gen()),
            GatherVerdict::Rearmed,
            "first sight has nothing to compare against"
        );
        assert_eq!(
            verdict(&mut w, one_page(&GPAS, &runs), QUIET, next_gen()),
            GatherVerdict::Vouched
        );
    }

    /// The guest's resource statement reports the store, so the vouch is
    /// refused and the bytes are read.
    #[test]
    fn a_guest_store_into_the_window_refuses_the_vouch() {
        let mut w = GatherWitness::default();
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        assert_eq!(
            verdict(&mut w, one_page(&GPAS, &runs), QUIET, next_gen()),
            GatherVerdict::Rearmed
        );
        buf[100] ^= 0xff;
        assert_eq!(
            verdict(
                &mut w,
                one_page(&GPAS, &runs),
                WitnessReadings {
                    stated_gen: Some(StatedGeneration::Mapping(1)),
                    ..QUIET
                },
                next_gen()
            ),
            GatherVerdict::Refused {
                guest_wrote: true,
                host_wrote_pages: false
            }
        );
    }

    /// The normal shipping witness reads no byte of the window at all.
    ///
    /// [`ContentAudit::Skipped`] *is* that statement — it is returned only where
    /// `fold_runs` was not called — so this is the test that would fail if the
    /// fold went back on the per-bind path, and the reason the audit's outcome is
    /// reported rather than kept inside the function.
    #[test]
    fn a_shipping_vouched_bind_never_folds_the_window() {
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let last = bind_quietly(&mut w, &GPAS, &runs, 8);
        assert_eq!(last.verdict, GatherVerdict::Vouched);
        assert_eq!(
            last.audit,
            ContentAudit::Skipped,
            "the disabled audit folded the window anyway"
        );
    }

    /// A refusal inside an enabled audit no longer disarms the alarm: the next arm takes
    /// a fresh baseline and the comparison still happens.
    ///
    /// The test drives one refusal into an otherwise quiet run, which is the
    /// smallest state transition that can invalidate an armed baseline.
    #[test]
    fn a_refusal_inside_an_audit_still_leaves_the_alarm_able_to_compare() {
        let mut w = witness_auditing(AuditDensity::EveryBind);
        let buf = vec![0x5au8; PAGE];
        let runs = [run_over(&buf)];
        bind_quietly(&mut w, &GPAS, &runs, 1);
        assert_eq!(
            bind_to_next_audit(&mut w, &GPAS, &runs).audit,
            ContentAudit::Seeded,
            "arming takes the baseline"
        );
        assert_eq!(
            bind_to_next_audit(&mut w, &GPAS, &runs).audit,
            ContentAudit::Agreed,
            "and the very next vouched bind is the one that checks it"
        );

        // One refused bind: this device wrote a page of the window. Nothing
        // about the bytes changed.
        let refused = observe(
            &mut w,
            KEY,
            one_page(&GPAS, &runs),
            WitnessReadings {
                pages_epoch: 2,
                pages_wrote: Some(reims_vgpu_core::HostWriteVerdict::Overlap),
                ..QUIET
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

        // The refused bind rebaselined the instrument from the bytes the gather
        // was about to consume, so the next quiet bind can compare immediately.
        assert_eq!(
            bind_to_next_audit(&mut w, &GPAS, &runs).audit,
            ContentAudit::Agreed,
            "the arm after a refusal did not compare against its fresh baseline"
        );
        assert_eq!(
            bind_to_next_audit(&mut w, &GPAS, &runs).audit,
            ContentAudit::Agreed,
            "and the alarm is running again one bind later"
        );
    }

    /// A refused bind between the baseline and the check must not produce a
    /// `Disagreed` for a witness that was right.
    ///
    /// An alarm that cries wolf is worse than no alarm, since the whole value of
    /// this one is that a nonzero count means something. The refusal is the
    /// witness working — it saw the store — so the baseline is retaken from the
    /// bytes the gather is about to read, and the check that follows compares
    /// across a vouched bind only.
    #[test]
    fn a_refused_bind_between_audits_does_not_leave_a_false_alarm_behind() {
        let mut w = witness_auditing(AuditDensity::EveryBind);
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        bind_quietly(&mut w, &GPAS, &runs, 1);
        assert_eq!(
            bind_to_next_audit(&mut w, &GPAS, &runs).audit,
            ContentAudit::Seeded
        );

        // A stated guest store repaints the window while the audit is armed.
        // The gather happens, so nothing is stale — but the baseline is now
        // from before the repaint.
        buf[11] ^= 0xff;
        let after_store = WitnessReadings {
            stated_gen: Some(StatedGeneration::Mapping(1)),
            ..QUIET
        };
        let refused = observe(&mut w, KEY, one_page(&GPAS, &runs), after_store, next_gen());
        assert!(matches!(refused.verdict, GatherVerdict::Refused { .. }));
        assert_eq!(
            refused.audit,
            ContentAudit::Rebaselined,
            "the armed window retakes its baseline from the repainted bytes"
        );

        assert_eq!(
            bind_to_next_audit_with(&mut w, &GPAS, &runs, after_store).audit,
            ContentAudit::Agreed,
            "comparing across the repaint would have been a false alarm"
        );
    }

    /// An armed window that is only ever refused gives up rather than folding on
    /// every bind.
    ///
    /// The arm costs one fold per refused bind, and the rail it audits moves
    /// 842 MB/s — so a window the witness never vouches for would pull the whole
    /// of it back through the audit. `AUDIT_REBASELINE_LIMIT` bounds that cost.
    #[test]
    fn an_armed_window_that_is_never_vouched_gives_up_instead_of_folding_forever() {
        let mut w = witness_auditing(AuditDensity::EveryBind);
        let buf = vec![0x11u8; PAGE];
        let runs = [run_over(&buf)];
        bind_quietly(&mut w, &GPAS, &runs, 1);
        assert_eq!(
            bind_to_next_audit(&mut w, &GPAS, &runs).audit,
            ContentAudit::Seeded
        );

        let refuse = |w: &mut GatherWitness| {
            observe(
                w,
                KEY,
                one_page(&GPAS, &runs),
                WitnessReadings {
                    pages_epoch: 2,
                    pages_wrote: Some(reims_vgpu_core::HostWriteVerdict::Overlap),
                    ..QUIET
                },
                next_gen(),
            )
            .audit
        };
        for i in 0..AUDIT_REBASELINE_LIMIT {
            assert_eq!(
                refuse(&mut w),
                ContentAudit::Rebaselined,
                "refusal {i} is still inside the arm's budget"
            );
        }
        assert_eq!(
            refuse(&mut w),
            ContentAudit::Restarted,
            "past the budget the arm gives up rather than folding on every bind"
        );
        // And having given up it stops folding, so the cost really is bounded.
        assert_eq!(refuse(&mut w), ContentAudit::Skipped);
    }

    /// The unsound case, produced deliberately: bytes changed under pages neither
    /// half of the witness saw written. This is the shape a host-side writer into
    /// guest RAM makes, and it is what the audit exists to catch — so if a driven
    /// boot ever reports `gw_audit_unsound`, this test says what that means.
    ///
    /// The audit is an alarm, not a repair: the decoded contract remains the
    /// sole source of the generation and vouch.
    #[test]
    fn bytes_moving_under_a_vouch_are_caught_without_costing_the_generation() {
        let mut w = witness_auditing(AuditDensity::EveryBind);
        let mut buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        // Rearm, then seed a fold the next audit can compare against.
        bind_quietly(&mut w, &GPAS, &runs, 1);
        let seeded = bind_to_next_audit(&mut w, &GPAS, &runs);
        assert_eq!(seeded.audit, ContentAudit::Seeded);
        let vouched_gen = seeded.generation;

        // No `guest_wrote_page` and no host write recorded: the bytes move with
        // both halves of the witness none the wiser.
        buf[7] ^= 0xff;
        let caught = bind_to_next_audit(&mut w, &GPAS, &runs);
        assert_eq!(
            caught.verdict,
            GatherVerdict::Vouched,
            "the witness is what is being caught out, so it must still be vouching"
        );
        assert_eq!(caught.audit, ContentAudit::Disagreed);
        assert_eq!(
            caught.generation, vouched_gen,
            "the audit replaced the generation supplied by the decoded contract"
        );
        assert_eq!(
            caught.vouch,
            GatherVouch::Vouched,
            "the audit overrode a contract-backed vouch"
        );
    }

    /// A window with no guest resource statement must never vouch, however
    /// still the bytes are. Fail closed: half a witness is not a witness.
    #[test]
    fn an_unaddressed_resource_statement_never_vouches() {
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let unaddressed = WitnessReadings {
            stated_gen: None,
            ..QUIET
        };
        assert_eq!(
            verdict(&mut w, one_page(&GPAS, &runs), unaddressed, next_gen()),
            GatherVerdict::Rearmed
        );
        assert_eq!(
            verdict(&mut w, one_page(&GPAS, &runs), unaddressed, next_gen()),
            GatherVerdict::Unarmed
        );
    }

    /// A window re-pointed at different pages has no predecessor, even though its
    /// key repeats. Comparing across the move would compare two different surfaces.
    #[test]
    fn a_window_whose_pages_move_rearms_rather_than_comparing_across_the_move() {
        let mut w = GatherWitness::default();
        let buf = vec![0xa5u8; PAGE];
        let runs = [run_over(&buf)];
        let moved = [9 * PAGE as u64];
        assert_eq!(
            verdict(&mut w, one_page(&GPAS, &runs), QUIET, next_gen()),
            GatherVerdict::Rearmed
        );
        assert_eq!(
            verdict(&mut w, one_page(&moved, &runs), QUIET, next_gen()),
            GatherVerdict::Rearmed,
            "same key, different pages: nothing to compare"
        );
        assert_eq!(
            verdict(&mut w, one_page(&moved, &runs), QUIET, next_gen()),
            GatherVerdict::Vouched
        );
    }

    #[test]
    fn the_fold_stops_at_span_even_when_the_runs_are_longer() {
        let buf = vec![3u8; 256];
        let short = unsafe { fold_runs(&[run_over(&buf)], 64) };
        let head = vec![3u8; 64];
        assert_eq!(short, unsafe { fold_runs(&[run_over(&head)], 64) });
    }

    /// Live sampled windows are retained without a capacity eviction.
    #[test]
    fn live_windows_are_not_evicted_by_capacity() {
        let mut w = GatherWitness::default();
        let buf = vec![0x5au8; PAGE];
        let runs = [run_over(&buf)];

        let key_at = |i: u64| GatherKey::Mapping {
            mapping: reims_vgpu_protocol::MappingId::new(11),
            base_offset: reims_vgpu_protocol::ByteOffset::new(i * PAGE as u64),
        };
        let gpas_at = |i: u64| [(64 + i) * PAGE as u64];

        const DISTINCT_WINDOWS: u64 = 512;
        for i in 0..DISTINCT_WINDOWS {
            let gpas = gpas_at(i);
            observe(&mut w, key_at(i), one_page(&gpas, &runs), QUIET, next_gen());
        }
        assert_eq!(w.entry_count(), DISTINCT_WINDOWS as usize);
        assert!((0..DISTINCT_WINDOWS).all(|i| w.contains(&key_at(i))));
    }

    /// Witness entries end with the task or mapping that owns the sampled
    /// window; unrelated live resources survive either end.
    #[test]
    fn resource_lifetime_retirement_releases_only_the_owned_windows() {
        let mut w = GatherWitness::default();
        let buf = vec![0x5au8; PAGE];
        let runs = [run_over(&buf)];
        let keys = [
            GatherKey::TaskGva {
                task_id: 7,
                resource: reims_vgpu_protocol::ResourceId::new(3, 1),
                gva: 0x1000,
            },
            GatherKey::TaskGva {
                task_id: 8,
                resource: reims_vgpu_protocol::ResourceId::new(4, 1),
                gva: 0x1000,
            },
            GatherKey::Mapping {
                mapping: reims_vgpu_protocol::MappingId::new(11),
                base_offset: reims_vgpu_protocol::ByteOffset::new(0),
            },
            GatherKey::Mapping {
                mapping: reims_vgpu_protocol::MappingId::new(12),
                base_offset: reims_vgpu_protocol::ByteOffset::new(0),
            },
        ];
        for key in keys {
            observe(&mut w, key, one_page(&GPAS, &runs), QUIET, next_gen());
        }

        w.retire_task(7);
        w.retire_mapping(11);
        assert_eq!(w.entry_count(), 2);
        assert!(!w.contains(&keys[0]));
        assert!(!w.contains(&keys[2]));
        assert!(w.contains(&keys[1]));
        assert!(w.contains(&keys[3]));
    }
}
