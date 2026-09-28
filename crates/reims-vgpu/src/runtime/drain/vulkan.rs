//! The Vulkan rail's completion-stamp publication.
//!
//! This rail can attach the stamp word to the GPU submission the guest work
//! precedes, so the drain worker hands the word to a completion thread instead
//! of blocking on that work. Everything here exists to decide whether that is
//! available for one stamp, and to fall back honestly when it is not.
//!
//! Reached only through [`crate::backend::Backend::order_completion_stamp`]; the
//! drain never names this rail.

use super::note_stamp_direction;
use crate::backend::vulkan::engine;
use crate::model::{stamp_slot_offset, DeviceState};
use crate::runtime::drain::census::note_store_route;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::render_writeback::{settle_guest_writes, SettleSite};

/// What [`stamp_word_order_on_fifo`] found, before it is reduced to the
/// backend-neutral [`crate::backend::StampOrdering`] the drain reads.
///
/// [`Self::CpuReady`] means the caller may publish immediately;
/// [`Self::Declined`] means it must settle through the blocking fallback.
/// Keeping those two distinct is load-bearing: another thread may arm an
/// unrelated guest write after this function observes no preceding work, and a
/// fallback that re-reads the global debt would incorrectly wait for that later
/// work before completing this FIFO. Both collapse to `Settled` at the seam,
/// and the collapse is only safe because the settle happens *here*, on the arm
/// that owes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StampOrder {
    CpuReady,
    Queued,
    Declined,
}

impl StampOrder {
    fn from_debt(guest_access: bool, fifo_pending: bool) -> Self {
        if guest_access || fifo_pending {
            Self::Queued
        } else {
            Self::CpuReady
        }
    }

    fn needs_blocking_fallback(self) -> bool {
        self == Self::Declined
    }
}

fn note_stamp_guest_ref_refusal(refusal: &crate::runtime::guest_ram_map::MapRefusal) {
    use crate::runtime::guest_ram_map::MapRefusal;
    let route = match refusal {
        MapRefusal::NoBackendImport => "stamp_guest_ref_no_backend_import",
        MapRefusal::HostRefused(_) => "stamp_guest_ref_host_refused",
        MapRefusal::NoUsableRegion { .. } => "stamp_guest_ref_no_usable_region",
        MapRefusal::ImportExceedsHeap { .. } => "stamp_guest_ref_import_exceeds_heap",
        MapRefusal::GpaNotInAnyImport { .. } => "stamp_guest_ref_gpa_not_imported",
        MapRefusal::OutsideImport(_) => "stamp_guest_ref_outside_import",
        MapRefusal::Scattered { .. } => "stamp_guest_ref_scattered",
        // The stamp is a CPU store through QEMU's own mapping, never a GPU
        // write, so no stamp site asks `write_refusal`; the arm is for totality.
        MapRefusal::WritesNarrowed => "stamp_guest_ref_writes_narrowed",
    };
    note_store_route(route);
}

/// Queue this stamp behind the FIFO completion point of the guest-memory work
/// it follows, so the drain worker never blocks on that work.
///
/// The word is four bytes inside one page, so the contiguity rule
/// `reference_for_pages` enforces is satisfied by construction — but it is asked
/// rather than assumed, because a stamp page outside an imported RAMBlock is
/// exactly the case that must fall back rather than be written blind.
fn stamp_word_order_on_fifo<H: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut H,
    index: u32,
    value: u32,
) -> StampOrder {
    if crate::config::switch(crate::config::GPU_STAMP) == crate::config::Switch::Off {
        return StampOrder::Declined;
    }
    // A CPU-only packet normally has nothing queued behind it. It must still
    // join an older pending completion on this same FIFO: publishing it now
    // would let the older completion overwrite the slot with a prior value.
    // Reads count just as much as writes: once this stamp moves, the guest may
    // repaint or free pages a preceding command buffer still sources.
    let guest_access = engine::guest_access_outstanding();
    let fifo_pending = engine::stamp_completion::fifo_has_pending_stamp(index);
    if StampOrder::from_debt(guest_access, fifo_pending) == StampOrder::CpuReady {
        return StampOrder::CpuReady;
    }
    let page_size = state.page_size();
    let Some(off) = stamp_slot_offset(index, page_size) else {
        return StampOrder::Declined;
    };
    let gpa = state.pfn_gpa(state.gfx.fifo_base_page) + off;
    let page = gpa & !(page_size - 1);
    let in_page = gpa - page;
    let guest_ref = match crate::runtime::guest_ram_map::reference_for_pages(
        host,
        &[page],
        page_size,
        in_page,
        4,
    ) {
        Ok(guest_ref) => guest_ref,
        Err(refusal) => {
            note_stamp_guest_ref_refusal(&refusal);
            return StampOrder::Declined;
        }
    };
    // The direction check the CPU rail gets from `note_stamp_direction`. Taken
    // before enqueueing because the completion thread owns the next write and
    // reading the word afterward says nothing about what this device promised.
    note_stamp_direction(host, gpa, index, value);
    let queued = match engine::write_completion_stamp(&guest_ref, index, value) {
        Ok(()) => true,
        Err(_) => {
            note_store_route("stamp_gpu_engine_declined");
            false
        }
    };
    if queued && !guest_access {
        note_store_route("stamp_pending_fifo_chained");
    }
    if queued {
        StampOrder::Queued
    } else {
        StampOrder::Declined
    }
}

/// [`crate::backend::Backend::order_completion_stamp`] for this rail.
///
/// The three-way [`StampOrder`] becomes the two-way answer the drain acts on,
/// and this function is where the third case is discharged: a `Declined` owes
/// both the queued stamps ahead of it and every guest read and write this
/// device has outstanding, and pays all three before saying `Settled`.
pub fn order_completion_stamp<H: HostMemory + HostOps>(
    state: &DeviceState,
    host: &mut H,
    index: u32,
    value: u32,
    site: SettleSite,
) -> crate::backend::StampOrdering {
    let order = stamp_word_order_on_fifo(state, host, index, value);
    if order == StampOrder::Queued {
        return crate::backend::StampOrdering::Queued;
    }
    if order.needs_blocking_fallback() {
        // The asynchronous route was required but could not carry the
        // completion, so it must let an older word for this slot land first.
        engine::quiesce_completion_stamps(index);
        // Only this answer may re-read global debt: `CpuReady` already proved
        // the packet had nothing preceding it, and work another thread arms
        // afterward belongs *after* this stamp.
        settle_guest_writes(site);
        engine::quiesce_guest_reads();
    }
    crate::backend::StampOrdering::Settled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cpu_only_stamp_publishes_now_unless_its_fifo_has_an_older_completion() {
        assert_eq!(StampOrder::from_debt(false, false), StampOrder::CpuReady);
        assert_eq!(StampOrder::from_debt(true, false), StampOrder::Queued);
        assert_eq!(StampOrder::from_debt(false, true), StampOrder::Queued);
        assert!(!StampOrder::CpuReady.needs_blocking_fallback());
        assert!(StampOrder::Declined.needs_blocking_fallback());
    }
}
