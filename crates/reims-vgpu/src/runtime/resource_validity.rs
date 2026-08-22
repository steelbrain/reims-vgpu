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
//! stamped by the surface registry's single validity timeline. Treating the claim as a latch
//! instead refuses the device's every later frame for that surface; see
//! [`crate::model::ResourceValidity`] for the boot that measured it.

use crate::model::ResourceValidity;
use crate::runtime::decode::fifo::InvalidateValidityOps;
use crate::runtime::Device;
use reims_vgpu_core::{
    CommandExecution, ExecutionOutput, ResolvedCommand, ResolvedResourceState, ResolvedSubmission,
    ResourceStateCompletion,
};

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
/// mapping id, and an IOSurface texture's retained relation is per-task. Both
/// are applied when both resolve: a statement about the task object is a
/// statement about every mapping it names. That candidate set is
/// [`Device::mappings_named_by`], shared with the render-frame ledger so
/// the two cannot disagree about which mappings a reference names.
pub fn apply(
    state: &mut Device,
    task_id: u32,
    object_id: u32,
    ops: InvalidateValidityOps,
    site: ValiditySite,
) -> ValidityOutcome {
    if object_id == 0 {
        // Null resources name no state and are not failed resolutions.
        return ValidityOutcome::default();
    }
    let resource = state.task_objects.resources.identity(task_id, object_id);
    let mappings = state
        .mappings_named_by(task_id, object_id)
        .iter()
        .map(reims_vgpu_protocol::SurfaceId::new)
        .filter(|surface| state.surfaces.mappings.contains_key(&surface.get()))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    // Pre-construction currency is a normalization-side effect: it preserves a
    // guest write when no canonical resource exists yet, but the unresolved
    // object name itself does not cross the execution boundary.
    if ops.clear_host_valid != 0 {
        state
            .content
            .preconstruction_writes
            .note_write(task_id, object_id);
        crate::runtime::drain::note_store_route("buffer_write_gen_bump");
        crate::runtime::drain::note_store_route(site.clear_host_route());
    }
    if ops.clear_guest_valid != 0 {
        note_owed_guest_read(state, task_id, object_id);
    }
    let update = ResolvedResourceState {
        resource,
        mappings,
        ops,
    };
    let expected = update.clone();
    let context = crate::runtime::executor::context_for(state, task_id);
    let submission =
        ResolvedSubmission::<(), ()>::single(context, ResolvedCommand::ResourceState(update));
    let outcome = std::cell::Cell::new(None);
    let completion = reims_vgpu_core::execute_resolved_submission(
        submission,
        |_, ()| -> Result<CommandExecution<()>, std::convert::Infallible> { unreachable!() },
        |_, ()| -> Result<CommandExecution<()>, std::convert::Infallible> { unreachable!() },
        |_, _| -> Result<CommandExecution<_>, std::convert::Infallible> { unreachable!() },
        |_, update| {
            outcome.set(Some(apply_resolved(state, update.clone())));
            Ok(CommandExecution::without_gpu_materialization(
                ResourceStateCompletion { update },
            ))
        },
    )
    .expect("resource-state execution is infallible");
    debug_assert!(matches!(
        completion.output.as_ref(),
        [ExecutionOutput::ResourceState(ResourceStateCompletion {
            update: completed
        })] if *completed == expected
    ));
    outcome
        .get()
        .expect("the single resource-state command was executed")
}

fn apply_resolved(state: &mut Device, update: ResolvedResourceState) -> ValidityOutcome {
    let mut out = ValidityOutcome::default();
    let ops = update.ops;
    if ops.clear_host_valid != 0 {
        // The object graph owns content authority for every constructed
        // resource, including buffers and textures without a SurfaceMappingEntry.
        // Mapping generations below remain cache invalidation witnesses during
        // the migration; they no longer stand in for the resource's version.
        if let Some(resource) = update.resource {
            state
                .task_objects
                .resources
                .note_guest_write_by_id(resource);
        }
    }
    for surface in &update.mappings {
        let id = surface.get();
        let applied = state.apply_surface_validity(*surface, ops);
        debug_assert!(
            applied,
            "resolved live surface disappeared during transition"
        );
        if ops.clear_host_valid != 0 {
            // The guest wrote these pages after our last render into them, so
            // our copy is stale by the guest's own statement and the next read
            // must re-take the guest pages.
            out.bumped = out.bumped.saturating_add(1);
            // A cached frame is a host copy of the resource the guest just
            // declared newer. Removing it here makes every cache consumer obey
            // the decoded validity statement without its own currency rule.
            crate::runtime::surface_cache::forget(state, id);
            crate::runtime::drain::note_store_route("validity_gen_bump");
        }
    }
    out.missed = update.mappings.is_empty();
    out
}

fn note_owed_guest_read(state: &Device, task_id: u32, object_id: u32) {
    // Byte +6 of the exec-table record, and the one op in the quad this
    // device stores without anything reading it.
    //
    // Its name here is not settled. Under this crate's reading the guest is
    // declaring its own copy stale; under the emitting driver's it is the
    // page-off / synchronize-requested flag, the guest explicitly asking for
    // host->guest visibility. The alarm below is deliberately built to be
    // correct under *both*, because the two readings agree about what
    // happens next: the guest is about to look at these guest pages. A frame
    // this device still owes them is unserved guest work either way.
    //
    // Expected to stay at zero, and a firing is the bug — the shape
    // `AGENTS.md` calls a healthy-zero alarm. It matters because the blit
    // rail stopped manufacturing host visibility for every
    // `copyFromTexture:toTexture:`, on the grounds that the command does not
    // carry it; this is the counterpart check that the guest's *explicit*
    // request is not being dropped on the floor at the same time.
    let owed_gva = crate::runtime::writeback_debt::resource_key(state, task_id, object_id)
        .is_some_and(|key| state.content.pending_writebacks.has_gva(key));
    if owed_gva {
        crate::observe::Emit::decline("validity_guest_read", &GuestReadDecline::GvaFrameStillOwed)
            .field("task", task_id)
            .field("object", object_id)
            .fail_once(u64::from(object_id));
        // Both, not either: the decline dedupes per object so a standing
        // problem reads as one line, and the counter is what says whether
        // that line stood for one occurrence or a million.
        crate::runtime::drain::note_store_route("validity_guest_read_frame_owed");
    }
    crate::runtime::drain::note_store_route("validity_clr_guest");
}

/// The guest signalled on byte +6 for a resource this device has not finished
/// delivering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GuestReadDecline {
    /// A GVA writeback is still owed to the transfer backing the guest will read.
    GvaFrameStillOwed,
}

impl crate::observe::Decline for GuestReadDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::GvaFrameStillOwed => "validity_guest_read_gva_frame_owed",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// The quad applied to one validity pair, clear before set.
///
/// Split out from [`apply`] so the transition table is testable without a
/// device: it is the part that has to match the host framework's
/// `setIsHostValid:` / `setIsGuestValid:` semantics, and the part a second
/// producer could silently disagree with.
#[cfg(test)]
fn next_validity(prev: ResourceValidity, ops: InvalidateValidityOps) -> ResourceValidity {
    let mut mapping = crate::model::SurfaceMappingEntry::default();
    mapping.content.validity = prev;
    mapping.content.apply_validity(ops);
    mapping.content.validity
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
fn writeback_licence(state: &Device, mapping_id: u32) -> WritebackLicence {
    licence_of(
        state
            .surfaces
            .mappings
            .get(&mapping_id)
            .map(|m| m.content.validity)
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
pub fn writeback_refused(state: &Device, mapping_id: u32) -> bool {
    let licence = writeback_licence(state, mapping_id);
    crate::runtime::drain::note_store_route(licence.route());
    licence == WritebackLicence::Superseded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};

    fn resource_key(
        state: &Device,
        task_id: u32,
        texture_ref: u32,
    ) -> crate::runtime::writeback_debt::GvaResourceKey {
        crate::runtime::writeback_debt::GvaResourceKey {
            task_id,
            resource: state.register_test_resource(task_id, texture_ref),
        }
    }

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
        let after = next_validity(ResourceValidity::default(), InvalidateValidityOps::PAGE_ON);
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
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        state
            .surfaces
            .mappings
            .entry(77)
            .or_default()
            .lifecycle
            .active = true;
        state
            .surfaces
            .mappings
            .entry(77)
            .or_default()
            .content
            .validity
            .host_valid = true;
        state
            .surfaces
            .mappings
            .entry(77)
            .or_default()
            .content
            .surface_epoch = 9;
        crate::runtime::surface_cache::store(&mut state, 77, 2, 2, vec![0x55; 16]);
        state.fixtures.texture_to_mapping.insert((4, 12), 77);
        let out = apply(&mut state, 4, 12, quad(1, 0, 0, 0), ValiditySite::ExecTable);
        assert_eq!(out.bumped, 1, "the ref must resolve to its mapping");
        assert!(
            !state.surfaces.mappings[&77].content.validity.host_valid,
            "clear_host_valid must reach the mapping the ref names"
        );
        assert_ne!(
            state.surfaces.mappings[&77].content.surface_epoch, 9,
            "the same decoded write must invalidate a resident Store stamp"
        );
        assert!(
            crate::runtime::surface_cache::get(&state, 77, 2, 2).is_none(),
            "a host byte copy must not survive the guest's ownership claim"
        );
    }

    /// Byte +6 arriving while a frame is still owed is the one thing that could
    /// have been broken by the blit rail no longer settling every
    /// `copyFromTexture:toTexture:`: the guest is about to read guest pages this
    /// device has not delivered into.
    ///
    /// Asserted in both directions, because an alarm that cannot stay quiet is
    /// worth as little as one that cannot fire — this one is expected to read
    /// zero on every boot, so a version of it that fired on the ordinary case
    /// would be indistinguishable from the bug it is watching for.
    #[test]
    fn byte_six_against_an_owed_frame_is_an_alarm_and_is_otherwise_quiet() {
        let task_id = 4;
        let texture_ref = 12;
        let debt = |state: &mut Device| {
            let key = resource_key(state, task_id, texture_ref);
            let before = state.resource_write_stamp_for(key.resource).unwrap();
            let _ = state.content.pending_writebacks.arm_gva(
                key,
                crate::runtime::writeback_debt::GvaWritebackDebt {
                    linear: crate::runtime::draw::LinearColorTarget {
                        allocation_gva: 0x4000,
                        allocation_size: 256 * 64,
                        plane_offset: 0,
                        row_stride: 256,
                    },
                    width: 64,
                    height: 64,
                    format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
                    resident_layout: reims_vgpu_core::pixel_format::TexelLayout::Bgra8,
                    generation: 7,
                    content: None,
                    guest_write: before,
                    seq: 0,
                },
            );
        };

        // Nothing owed: the guest may read, and the alarm must not fire.
        let mut quiet = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        quiet
            .surfaces
            .mappings
            .entry(texture_ref)
            .or_default()
            .lifecycle
            .active = true;
        let before_quiet =
            crate::runtime::drain::store_route_count("validity_guest_read_frame_owed");
        apply(
            &mut quiet,
            task_id,
            texture_ref,
            quad(0, 0, 1, 0),
            ValiditySite::ExecTable,
        );
        assert_eq!(
            crate::runtime::drain::store_route_count("validity_guest_read_frame_owed"),
            before_quiet,
            "no frame is owed, so byte +6 is ordinary traffic"
        );

        // A GVA frame owed for the very resource the guest is about to read.
        let mut owed = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        owed.surfaces
            .mappings
            .entry(texture_ref)
            .or_default()
            .lifecycle
            .active = true;
        debt(&mut owed);
        let before_owed =
            crate::runtime::drain::store_route_count("validity_guest_read_frame_owed");
        apply(
            &mut owed,
            task_id,
            texture_ref,
            quad(0, 0, 1, 0),
            ValiditySite::ExecTable,
        );
        assert_eq!(
            crate::runtime::drain::store_route_count("validity_guest_read_frame_owed"),
            before_owed + 1,
            "the guest is about to read pages this device has not delivered"
        );
    }

    /// Mapping ids and task-local texture refs are separate namespaces. When
    /// their integers collide, invalidating the mapping must not hide the guest
    /// write from a host-authoritative GVA resident owned by the texture ref.
    #[test]
    fn a_numeric_mapping_collision_still_invalidates_an_owed_gva_resource() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        let task_id = 4;
        let texture_ref = 12;
        state
            .surfaces
            .mappings
            .entry(texture_ref)
            .or_default()
            .lifecycle
            .active = true;
        let key = resource_key(&state, task_id, texture_ref);
        let before = state.resource_write_stamp_for(key.resource).unwrap();
        let _ = state.content.pending_writebacks.arm_gva(
            key,
            crate::runtime::writeback_debt::GvaWritebackDebt {
                linear: crate::runtime::draw::LinearColorTarget {
                    allocation_gva: 0x4000,
                    allocation_size: 256 * 64,
                    plane_offset: 0,
                    row_stride: 256,
                },
                width: 64,
                height: 64,
                format: reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
                resident_layout: reims_vgpu_core::pixel_format::TexelLayout::Bgra8,
                generation: 7,
                content: None,
                guest_write: before,
                seq: 0,
            },
        );

        let out = apply(
            &mut state,
            task_id,
            texture_ref,
            quad(1, 0, 0, 0),
            ValiditySite::ExecTable,
        );

        assert_eq!(out.bumped, 1, "the colliding mapping still invalidates");
        assert!(
            !state
                .resource_write_stamp(task_id, texture_ref)
                .quiet_since(before),
            "the distinct GVA resource must see the same guest-write declaration"
        );
    }

    /// An id no registry answers for is reported, not silently skipped.
    #[test]
    fn an_unknown_object_is_reported_as_a_miss() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
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
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        state
            .surfaces
            .mappings
            .entry(0)
            .or_default()
            .lifecycle
            .active = true;
        let out = apply(&mut state, 0, 0, quad(1, 0, 0, 0), ValiditySite::ExecTable);
        assert_eq!(out, ValidityOutcome::default());
    }

    /// A mapping the guest has never claimed a write to must not have its
    /// writeback refused. Refusing withholds the device's frame, which is a
    /// compositing layer going black — a strictly worse failure than landing a
    /// frame nobody vouched for.
    #[test]
    fn a_never_claimed_mapping_is_not_refused() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        state
            .surfaces
            .mappings
            .entry(5)
            .or_default()
            .lifecycle
            .active = true;
        assert_eq!(writeback_licence(&state, 5), WritebackLicence::Unstated);
        assert!(!writeback_refused(&state, 5));
    }

    /// The gate: the guest claimed a CPU write and nothing has been published
    /// since, so the frame this window holds is older than the guest's bytes.
    #[test]
    fn a_claim_newer_than_our_last_publish_refuses_the_writeback() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        state
            .surfaces
            .mappings
            .entry(5)
            .or_default()
            .lifecycle
            .active = true;
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
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        state
            .surfaces
            .mappings
            .entry(5)
            .or_default()
            .lifecycle
            .active = true;
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
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        state
            .surfaces
            .mappings
            .entry(5)
            .or_default()
            .lifecycle
            .active = true;
        apply(&mut state, 0, 5, quad(1, 0, 0, 0), ValiditySite::ExecTable);
        state.mark_mapping_written(5);
        assert_eq!(writeback_licence(&state, 5), WritebackLicence::Licensed);
    }

    /// A mapping this device does not hold has nothing to order, and the flush
    /// rails' own `map_generation` guard is what refuses those.
    #[test]
    fn an_absent_mapping_reads_as_unstated() {
        let state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        assert_eq!(writeback_licence(&state, 999), WritebackLicence::Unstated);
    }

    /// The counter must move on every landing, not only on the refusals — a
    /// census that counted only what it blocked could not report a rate.
    #[test]
    fn every_verdict_is_counted() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        state
            .surfaces
            .mappings
            .entry(5)
            .or_default()
            .lifecycle
            .active = true;
        let before = crate::runtime::drain::store_route_count("validity_wb_unstated");
        assert!(!writeback_refused(&state, 5));
        assert_eq!(
            crate::runtime::drain::store_route_count("validity_wb_unstated"),
            before + 1
        );
    }
}
