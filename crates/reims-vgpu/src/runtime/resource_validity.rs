//! Apply the guest's per-resource validity quad, from either producer.
//!
//! # Two producers, one record layout
//!
//! The guest states who owns a resource's authoritative bytes with four u8
//! fields — `clear_host_valid | set_host_valid | clear_guest_valid |
//! set_guest_valid` — and emits them from two places:
//!
//! - `pageBacking` → `CmdInvalidateResources` (`0x34`), 8-byte records, one
//!   hardcoded quad (`clear_host + set_guest`).
//! - `AppleParavirtCommandQueue::writeInvalidates` → the resource table inside
//!   every `EXEC_INDIRECT2` payload, 24-byte records, a quad computed per
//!   resource.
//!
//! The record *lengths* differ; the quad does not. Both decode through
//! [`InvalidateValidityOps`] and both land here, so the two paths cannot drift
//! into two different meanings for the same four bytes.
//!
//! # Why `clear_host_valid` has to do more than bump a generation
//!
//! `AppleParavirtResource::shouldInvalidateHost()` is a `lock btr` test-and-clear
//! of the resource's dirty bit plus a sticky flag it also clears, and
//! `writeInvalidates` is its only caller. So "the guest CPU-wrote this resource"
//! is delivered exactly once, in one submission's table, and is never resent.
//!
//! A pending deferred window for that resource holds pixels the device rendered
//! *before* that guest write. Landing it afterwards replaces bytes the guest
//! authored with bytes the guest has just declared stale — a full-extent clobber
//! of the guest's own work. `flush_all_windows_before_fence` cannot see this: it
//! decides *when* a window lands, and the answer here is that it must not land
//! at all. So a `clear_host_valid` drops the window rather than resequencing it.
//!
//! # Order within one quad
//!
//! Clear before set, in wire field order. `0x00000101` — both host bits in one
//! record — occurs in live traffic, and clear-then-set is the only reading under
//! which it is not self-contradictory: the guest wrote the resource, and this
//! submission then rewrites it.
//!
//! # Order between the guest's claim and the device's frame
//!
//! `clear_host_valid` is a statement about a moment, not a standing property.
//! [`writeback_licence`] therefore compares *when* the guest claimed its write
//! against *when* the device last published pixels for that resource, both
//! stamped from `DeviceState::next_validity_seq`. Treating the claim as a latch
//! instead refuses the device's every later frame for that surface; see
//! [`crate::model::ResourceValidity`] for the boot that measured it.

use crate::model::{DeviceState, ResourceValidity};
use crate::runtime::decode::fifo::InvalidateValidityOps;

/// Which producer delivered a quad. Only used to name the counters, so an arm
/// can tell an exec-table statement from an invalidate-command one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValiditySite {
    ExecTable,
    InvalidateResources,
}

impl ValiditySite {
    fn clear_host_route(self) -> &'static str {
        match self {
            Self::ExecTable => "validity_clr_host_exec",
            Self::InvalidateResources => "validity_clr_host_inv",
        }
    }
}

/// What one record changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ValidityOutcome {
    /// Mappings whose `content_generation` this record advanced.
    pub bumped: u32,
    /// The record named no mapping this device holds.
    pub missed: bool,
}

/// Apply one record's quad to whatever mapping state the object id names.
///
/// `task_id` is needed because a table id may be a texture ref rather than a
/// mapping id, and `texture_to_mapping` is per-task. Both are applied when both
/// resolve — the crate carries two registries for one guest object and a
/// statement about that object is a statement about both.
pub fn apply(
    state: &mut DeviceState,
    task_id: u32,
    object_id: u32,
    ops: InvalidateValidityOps,
    site: ValiditySite,
) -> ValidityOutcome {
    let mut out = ValidityOutcome::default();
    if object_id == 0 {
        // `writeInvalidates` skips null resources and id 0; `pageBacking` never
        // emits one. A zero id names nothing to apply to.
        return out;
    }
    let mut targets = vec![object_id];
    if let Some(&mid) = state.texture_to_mapping.get(&(task_id, object_id)) {
        if mid != object_id {
            targets.push(mid);
        }
    }
    let mut hit = false;
    for id in targets {
        if !state.mappings.contains_key(&id) {
            continue;
        }
        hit = true;
        if ops.clear_host_valid != 0 {
            // The guest wrote these pages after our last render into them, so
            // our copy is stale by the guest's own statement and the next read
            // must re-take the guest pages.
            let seq = state.next_validity_seq();
            if let Some(m) = state.mappings.get_mut(&id) {
                m.content_generation = m.content_generation.saturating_add(1);
                // Stamped rather than latched: the claim is about this moment,
                // and the device's next publish into this surface supersedes it.
                m.validity.host_cleared_seq = seq;
                out.bumped = out.bumped.saturating_add(1);
            }
            crate::runtime::drain::note_store_route("validity_gen_bump");
        }
        let Some(m) = state.mappings.get_mut(&id) else {
            continue;
        };
        m.validity = next_validity(m.validity, ops);
    }
    out.missed = !hit;
    if ops.clear_host_valid != 0 {
        // The statement this device used to decode and drop. A buffer has no
        // mapping, so the loop above skipped it and the guest's account of its
        // own write went nowhere — which is the one signal the draw-time buffer
        // gather has no substitute for. Recorded only on the miss, because an
        // object with a mapping already carries `content_generation` and a
        // second spelling of one fact is a divergence waiting to happen.
        if !hit {
            state.buffer_write_gen.note_write(task_id, object_id);
        }
        crate::runtime::drain::note_store_route(site.clear_host_route());
    }
    out
}

/// The quad applied to one validity pair, clear before set.
///
/// Split out from [`apply`] so the transition table is testable without a
/// device: it is the part that has to match the host framework's
/// `setIsHostValid:` / `setIsGuestValid:` semantics, and the part a second
/// producer could silently disagree with.
fn next_validity(prev: ResourceValidity, ops: InvalidateValidityOps) -> ResourceValidity {
    let mut next = prev;
    if ops.clear_host_valid != 0 {
        next.host_valid = false;
        next.host_stated = true;
    }
    if ops.set_host_valid != 0 {
        next.host_valid = true;
        next.host_stated = true;
    }
    if ops.clear_guest_valid != 0 {
        next.guest_valid = false;
        next.guest_stated = true;
    }
    if ops.set_guest_valid != 0 {
        next.guest_valid = true;
        next.guest_stated = true;
    }
    next
}

/// Who wrote a mapping's bytes last, as it bears on landing a deferred
/// writeback into that mapping's guest pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WritebackLicence {
    /// The device published newer pixels after the guest's last claim. The
    /// writeback is owed.
    Licensed,
    /// The guest claimed a CPU write *after* the device's last publish. Landing
    /// our frame would replace the guest's work with a copy it declared stale.
    Superseded,
    /// The guest has never claimed a CPU write to this resource, so there is
    /// nothing to order the device's publish against.
    Unstated,
}

/// Read the licence for one mapping.
///
/// A happens-before between the guest's last `clear_host_valid` and the device's
/// last publish, never a latch on `host_valid`. See [`ResourceValidity`] for the
/// measurement that forced that distinction.
///
/// Pure — the counting is [`writeback_refused`]'s job, which is the caller that
/// stamps `note_store_route`, so a caller that only wants to attribute a write
/// does not inflate the flush census.
fn writeback_licence(state: &DeviceState, mapping_id: u32) -> WritebackLicence {
    licence_of(
        state
            .mappings
            .get(&mapping_id)
            .map(|m| m.validity)
            .unwrap_or_default(),
    )
}

/// [`writeback_licence`] for a caller that already holds the entry.
///
/// The footprint attribution runs on the mapping write path, which has just
/// looked this mapping up; a second lookup of the same map per write buys
/// nothing.
pub fn licence_of(validity: ResourceValidity) -> WritebackLicence {
    if validity.host_cleared_seq == 0 {
        WritebackLicence::Unstated
    } else if validity.host_published_seq > validity.host_cleared_seq {
        WritebackLicence::Licensed
    } else {
        WritebackLicence::Superseded
    }
}

impl WritebackLicence {
    fn route(self) -> &'static str {
        match self {
            Self::Licensed => "validity_wb_licensed",
            Self::Superseded => "validity_wb_superseded",
            Self::Unstated => "validity_wb_unstated",
        }
    }
}

/// Whether a landing writeback must be refused, counting the population as it
/// goes.
///
/// Every landing is counted by verdict whether or not the refusal is enforced,
/// so an armed boot and its control report the same numbers and differ only in
/// whether the write happened.
///
/// `Unstated` never refuses. The safe reading of "the guest never claimed a
/// write" is to deliver the frame: refusing withholds the device's pixels and
/// turns a compositing layer black, which this project has already paid a boot
/// to discover once. `validity_wb_unstated` is what would make tightening that
/// direction provable rather than a guess.
///
/// `Superseded` should be rare, and the reason is worth stating because it makes
/// this counter a standing check rather than a workhorse: the exec table's
/// `clear_host_valid` already drops the mapping's pending windows at the moment
/// it arrives, so a window that survives to a flush with the guest's claim newer
/// than our publish is one that drop did not reach.
///
/// One driven boot with the ordering in place, three `icon-composite` rounds,
/// all CLEAN: `validity_wb_licensed 126`, `validity_wb_unstated 589`,
/// `validity_wb_superseded 0` over 672 `surface_flush`es and 794
/// `clear_host_valid` deliveries. Nothing was withheld. The same workload
/// against the latch this replaced refused 32 % of every landing.
pub fn writeback_refused(state: &DeviceState, mapping_id: u32) -> bool {
    let licence = writeback_licence(state, mapping_id);
    crate::runtime::drain::note_store_route(licence.route());
    licence == WritebackLicence::Superseded
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};

    fn quad(clr_h: u8, set_h: u8, clr_g: u8, set_g: u8) -> InvalidateValidityOps {
        InvalidateValidityOps {
            clear_host_valid: clr_h,
            set_host_valid: set_h,
            clear_guest_valid: clr_g,
            set_guest_valid: set_g,
        }
    }

    /// `0x00000101` — both host bits in one record — is live traffic. Clear
    /// before set is the only reading under which it is not self-contradictory.
    #[test]
    fn a_record_carrying_both_host_bits_ends_host_valid() {
        let after = next_validity(ResourceValidity::default(), quad(1, 1, 0, 0));
        assert!(after.host_valid);
        assert!(after.host_stated);
    }

    /// An op the record does not carry must leave its bit alone, including the
    /// "never stated" flag — otherwise every quad would look like a statement
    /// about all four bits.
    #[test]
    fn an_absent_op_states_nothing() {
        let after = next_validity(ResourceValidity::default(), quad(1, 0, 0, 0));
        assert!(after.host_stated);
        assert!(!after.guest_stated, "guest side was never mentioned");
        assert!(!after.guest_valid);
    }

    /// Pageon's hardcoded quad: the host copy goes stale, the guest pages become
    /// authoritative.
    #[test]
    fn the_pageon_quad_hands_ownership_to_the_guest() {
        let after = next_validity(ResourceValidity::default(), InvalidateValidityOps::PAGEON);
        assert!(!after.host_valid);
        assert!(after.guest_valid);
        assert!(after.host_stated && after.guest_stated);
    }



    /// A texture ref and the mapping it resolves to are one guest resource, so a
    /// statement about the ref has to land on the mapping. One that stopped at
    /// the ref would leave the mapping still claiming host-valid bytes the guest
    /// has just overwritten.
    #[test]
    fn a_statement_about_a_texture_ref_reaches_its_mapping() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.mappings.entry(77).or_default().mapped = true;
        state.mappings.entry(77).or_default().validity.host_valid = true;
        state.texture_to_mapping.insert((4, 12), 77);
        let out = apply(&mut state, 4, 12, quad(1, 0, 0, 0), ValiditySite::ExecTable);
        assert_eq!(out.bumped, 1, "the ref must resolve to its mapping");
        assert!(
            !state.mappings[&77].validity.host_valid,
            "clear_host_valid must reach the mapping the ref names"
        );
    }

    /// An id no registry answers for is reported, not silently skipped.
    #[test]
    fn an_unknown_object_is_reported_as_a_miss() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let out = apply(
            &mut state,
            0,
            4242,
            quad(1, 0, 0, 0),
            ValiditySite::ExecTable,
        );
        assert!(out.missed);
        assert_eq!(out.bumped, 0);
    }

    #[test]
    fn object_id_zero_applies_to_nothing() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.mappings.entry(0).or_default().mapped = true;
        let out = apply(&mut state, 0, 0, quad(1, 0, 0, 0), ValiditySite::ExecTable);
        assert_eq!(out, ValidityOutcome::default());
    }

    /// A mapping the guest has never claimed a write to must not have its
    /// writeback refused. Refusing withholds the device's frame, which is a
    /// compositing layer going black — a strictly worse failure than landing a
    /// frame nobody vouched for.
    #[test]
    fn a_never_claimed_mapping_is_not_refused() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.mappings.entry(5).or_default().mapped = true;
        assert_eq!(writeback_licence(&state, 5), WritebackLicence::Unstated);
        assert!(!writeback_refused(&state, 5));
    }

    /// The gate: the guest claimed a CPU write and nothing has been published
    /// since, so the frame this window holds is older than the guest's bytes.
    #[test]
    fn a_claim_newer_than_our_last_publish_refuses_the_writeback() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.mappings.entry(5).or_default().mapped = true;
        state.note_surface_content_published(5);
        apply(&mut state, 0, 5, quad(1, 0, 0, 0), ValiditySite::ExecTable);
        assert_eq!(writeback_licence(&state, 5), WritebackLicence::Superseded);
        assert!(writeback_refused(&state, 5));
    }

    /// The case that makes this a happens-before and not a latch, and the one a
    /// live boot refuted the latch on: after the guest's claim, the device
    /// renders into the surface again. Its frame is now the newer one and the
    /// writeback is owed. A latch refuses this forever, because nothing in the
    /// protocol re-affirms a resource the guest has stopped writing.
    #[test]
    fn a_publish_after_the_guests_claim_re_earns_the_writeback() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.mappings.entry(5).or_default().mapped = true;
        apply(&mut state, 0, 5, quad(1, 0, 0, 0), ValiditySite::ExecTable);
        assert_eq!(writeback_licence(&state, 5), WritebackLicence::Superseded);
        state.note_surface_content_published(5);
        assert_eq!(writeback_licence(&state, 5), WritebackLicence::Licensed);
        assert!(!writeback_refused(&state, 5));
    }

    /// Writing the mapping's guest pages is a publish too — the same claim about
    /// currency, made by the rail that does not defer.
    #[test]
    fn writing_the_guest_pages_counts_as_a_publish() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.mappings.entry(5).or_default().mapped = true;
        apply(&mut state, 0, 5, quad(1, 0, 0, 0), ValiditySite::ExecTable);
        state.mark_mapping_written(5);
        assert_eq!(writeback_licence(&state, 5), WritebackLicence::Licensed);
    }

    /// A mapping this device does not hold has nothing to order, and the flush
    /// rails' own `map_generation` guard is what refuses those.
    #[test]
    fn an_absent_mapping_reads_as_unstated() {
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert_eq!(writeback_licence(&state, 999), WritebackLicence::Unstated);
    }

    /// The counter must move on every landing, not only on the refusals — a
    /// census that counted only what it blocked could not report a rate.
    #[test]
    fn every_verdict_is_counted() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.mappings.entry(5).or_default().mapped = true;
        let before = crate::runtime::drain::store_route_count("validity_wb_unstated");
        assert!(!writeback_refused(&state, 5));
        assert_eq!(
            crate::runtime::drain::store_route_count("validity_wb_unstated"),
            before + 1
        );
    }

}
