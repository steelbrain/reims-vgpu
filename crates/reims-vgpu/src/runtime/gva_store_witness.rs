//! Contract-owned currency for copied-out GVA render targets.
//!
//! A named GVA target is a task-local resource. Its `clear_host_valid` validity
//! operation is the guest's notification that CPU writes superseded a host
//! copy, and the core write-generation ledger retains that notification in
//! the resource's canonical generational namespace. Device writes are tracked
//! separately by [`reims_vgpu_core::HostWrites`] because different resource
//! names may alias the same guest pages.
//!
//! A Store records both generations beside the target's exact page footprint.
//! A resident may stand in for those pages only while both remain unchanged.
//! Anonymous GVA targets have no serialized resource lifetime or validity
//! record, so they acquire no entry and conservatively miss this shortcut.

use crate::runtime::Device;
pub use reims_vgpu_core::{GvaStoreWitness, GvaTargetKey, GvaWriteReach, HostWriteVerdict};

/// Stamp a Store after its own page writes have been recorded.
///
/// `guest_write` is the resource content version this Store leaves behind, and
/// it is a parameter rather than a read of the current stamp because the two
/// are not the same value at every call site. `resource_write_stamp_for`
/// answers the resource's *current* content version, and a GPU Store advances
/// that version itself — `record_completed_gpu_store` returns the new one — so
/// a rail that stamps the witness before recording its Store captures the
/// version from before it and every later [`reach`] compares a bumped current
/// against it and answers `GuestWrote`. Such an entry can never answer `Quiet`,
/// which retires the resident and chained-seed shortcuts for that rail
/// entirely while looking exactly like a guest that keeps rewriting its own
/// target.
///
/// So each caller passes the version that belongs to the Store it is recording:
/// the direct-landing rail passes the one its own Store returned, and the
/// copy-out rail passes the current stamp, which nothing advances after it.
pub fn note_store(
    state: &mut Device,
    key: GvaTargetKey,
    gpas: &[u64],
    guest_write: reims_vgpu_core::ResourceWriteStamp,
) {
    let host_epoch = state.content.host_writes.epoch();
    if !state
        .content
        .gva_stores
        .note_store(key, gpas, guest_write, host_epoch)
    {
        crate::runtime::drain::note_store_route("gvaw_unnamed_resource");
        return;
    }
    crate::runtime::drain::note_store_route("gvaw_stamped");
}

/// Retire targets whose physical footprint is no longer owned by the task.
pub fn retire_pages(state: &mut Device, gone: &[u64]) {
    if gone.is_empty() {
        return;
    }
    state.content.gva_stores.retire_pages(gone);
}

/// Compare the Store stamp with the guest's decoded validity statements and
/// this device's exact page writes.
pub fn reach(state: &Device, key: GvaTargetKey) -> GvaWriteReach {
    let Some(now) = state.resource_write_stamp_for(key.resource) else {
        return GvaWriteReach::NoEntry;
    };
    state
        .content
        .gva_stores
        .reach(key, now, &state.content.host_writes)
}

pub fn note_host_reach(state: &Device, key: GvaTargetKey) {
    let Some(distance) = state
        .content
        .gva_stores
        .host_epoch_distance(key, state.content.host_writes.epoch())
    else {
        return;
    };
    crate::runtime::drain::note_store_route(if distance < 64 {
        "gvaw_reach_lt64"
    } else if distance < 512 {
        "gvaw_reach_lt512"
    } else if distance < 4096 {
        "gvaw_reach_lt4k"
    } else {
        "gvaw_reach_ge4k"
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};

    const PAGE: u64 = 1 << PAGE_SHIFT_X86;

    fn state() -> Device {
        Device::new(DeviceId::default(), PAGE_SHIFT_X86)
    }

    /// Stamp with the resource's current content version.
    ///
    /// What every caller did before the version became a parameter, and what
    /// the copy-out rail still does. Returning early on an unnamed resource is
    /// the old behaviour these tests were written against.
    fn note_store_now(state: &mut Device, key: GvaTargetKey, gpas: &[u64]) {
        let Some(guest_write) = state.resource_write_stamp_for(key.resource) else {
            return;
        };
        note_store(state, key, gpas, guest_write);
    }

    fn key(state: &Device, task_id: u32, texture_ref: u32) -> GvaTargetKey {
        GvaTargetKey {
            task_id,
            resource: state.register_test_resource(task_id, texture_ref),
            gva: PAGE,
            generation: 9,
            width: 16,
            height: 16,
            bgra: true,
        }
    }

    /// The witness must be stamped with the version its Store *produced*, not
    /// the one that Store replaced.
    ///
    /// A GPU Store advances the resource's content version itself, so a rail
    /// that stamps before recording its Store captures the earlier version and
    /// every later `reach` compares a bumped current against it. Such an entry
    /// can never read `Quiet`, which silently retires the resident and
    /// chained-seed shortcuts for that whole rail. Both halves are asserted
    /// here so the ordering cannot regress to reading the same either way.
    #[test]
    fn a_store_stamped_with_the_version_it_replaced_can_never_read_quiet() {
        let mut state = state();
        let target = key(&state, 1, 7);
        let replaced = state
            .resource_write_stamp_for(target.resource)
            .expect("a named resource has a content version");
        state
            .task_objects
            .resources
            .note_guest_write_by_id(target.resource);
        let produced = state
            .resource_write_stamp_for(target.resource)
            .expect("a named resource has a content version");
        assert_ne!(
            replaced, produced,
            "the bump this test is about did not happen"
        );

        note_store_now(&mut state, target, &[PAGE]);
        assert_eq!(
            reach(&state, target),
            GvaWriteReach::Quiet,
            "the version the Store leaves behind is the one that reads quiet"
        );

        note_store(&mut state, target, &[PAGE], replaced);
        assert_eq!(
            reach(&state, target),
            GvaWriteReach::GuestWrote,
            "stamping with the superseded version is the defect, and it reads \
             exactly like a guest that rewrote its own target"
        );
    }

    #[test]
    fn decoded_guest_write_invalidates_only_its_resource() {
        let mut state = state();
        let a = key(&state, 1, 7);
        let b = key(&state, 1, 8);
        note_store_now(&mut state, a, &[PAGE]);
        note_store_now(&mut state, b, &[2 * PAGE]);
        state
            .task_objects
            .resources
            .note_guest_write_by_id(a.resource);
        assert_eq!(reach(&state, a), GvaWriteReach::GuestWrote);
        assert_eq!(reach(&state, b), GvaWriteReach::Quiet);
    }

    #[test]
    fn device_write_invalidates_every_alias_of_its_page() {
        let mut state = state();
        let a = key(&state, 1, 7);
        let b = key(&state, 2, 8);
        note_store_now(&mut state, a, &[PAGE]);
        note_store_now(&mut state, b, &[PAGE]);
        state.note_host_wrote_pages(vec![PAGE]);
        assert_eq!(
            reach(&state, a),
            GvaWriteReach::Host(HostWriteVerdict::Overlap)
        );
        assert_eq!(
            reach(&state, b),
            GvaWriteReach::Host(HostWriteVerdict::Overlap)
        );
    }

    #[test]
    fn anonymous_or_unstamped_target_never_reads_quiet() {
        let mut state = state();
        let unnamed = GvaTargetKey {
            task_id: 1,
            resource: reims_vgpu_protocol::ResourceId::new(99, 1),
            gva: PAGE,
            generation: 9,
            width: 16,
            height: 16,
            bgra: true,
        };
        note_store_now(&mut state, unnamed, &[PAGE]);
        assert_eq!(state.content.gva_stores.len(), 0);
        assert_eq!(reach(&state, unnamed), GvaWriteReach::NoEntry);
    }

    #[test]
    fn task_retirement_ends_witness_lifetime() {
        let mut state = state();
        let target = key(&state, 4, 12);
        note_store_now(&mut state, target, &[PAGE]);
        state.content.gva_stores.retire_task(4);
        assert_eq!(reach(&state, target), GvaWriteReach::NoEntry);
    }
}
