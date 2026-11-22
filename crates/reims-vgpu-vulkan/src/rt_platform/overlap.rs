//! What a slow guest answer ran over, for reims-vgpu's `graphics::slow_answers`, which is told of
//! each ring-0 answer the host gave a frame or more after its make (reims-vgpu-graphics'
//! `Host::answer_slow`). A diagnostic: which lines the host writes is free under the contract.
//!
//! - **The makes of rt's pool.** rt tells the device's `BlockMappings` (`super::fault`'s
//!   `Allocations`) of each allocation of its pool right after it made it, on the thread that made
//!   it: the block-maker thread's makes ahead of need, a take's make on the take's thread, and
//!   each class's first in the device's make. [`pool_made`] keeps when each of the last
//!   [`KEPT`] ended, on rt's host clock (`rt::host_time`), one clock read and two relaxed atomics a
//!   make. rt tells nothing as a make begins, so an answer counts as over a make where a make ended
//!   during it ([`pool_makes_ended`]): every answer that waited for a make to end, but not one that
//!   ended inside a make.
//! - **The first-use waits of compute pipelines.** rt keeps its last 256 ended makes of a compute
//!   kernel's `VkPipeline` for a dispatch's variant, with when each began and ended, how many
//!   dispatches waited for it (its first-use waits, rt's `Metrics::first_use_variant_waits`) and
//!   when the earliest of them began (rt's `Counters::variant_makes`). [`first_use_waits_met`]
//!   reads them, only for a slow answer, and counts each make's waits once.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

use metal2vulkan_runtime as rt;

use crate::core::sync::Mutex;

/// The makes of rt's pool whose ends [`pool_made`] keeps: the last 16.
pub const KEPT: usize = 16;

/// When each of the last [`KEPT`] makes of rt's pool ended, on rt's host clock, at its number's
/// place modulo [`KEPT`]; 0 where none yet.
static MADE_AT: [AtomicU64; KEPT] = [const { AtomicU64::new(0) }; KEPT];

/// The makes of rt's pool so far, the next one's number.
static MADE: AtomicUsize = AtomicUsize::new(0);

/// The numbers of rt's variant makes whose waits [`first_use_waits_met`] counted, the last 256:
/// as many as rt keeps.
static TOLD: Mutex<VecDeque<u64>> = Mutex::new(VecDeque::new());

/// rt made an allocation of its pool just now (`super::fault`'s `Allocations`, which rt tells of
/// each right after its make).
pub(super) fn pool_made() {
    let at = MADE.fetch_add(1, Relaxed) % KEPT;
    MADE_AT[at].store(rt::host_time(), Relaxed);
}

/// The makes of rt's pool among the last [`KEPT`] that ended within `start..=end` of rt's host
/// clock: how many, and when the latest of them ended (0 where none did).
pub fn pool_makes_ended(start: u64, end: u64) -> (u64, u64) {
    let within = MADE_AT
        .iter()
        .map(|at| at.load(Relaxed))
        .filter(|&at| at != 0 && (start..=end).contains(&at));
    within.fold((0, 0), |(n, latest), at| (n + 1, latest.max(at)))
}

/// The first-use waits of compute pipelines (rt's variant makes that dispatches waited for) that
/// met `start..=end` of rt's host clock, from the earliest wait's start to the make's end, of the
/// process's device, and that no call before counted: their waits. 0 before the device is made.
/// A make still running at `end` has not ended, and is not among them.
pub fn first_use_waits_met(start: u64, end: u64) -> u64 {
    let Some(makes) = super::device::variant_makes() else {
        return 0;
    };
    let mut told = TOLD.lock();
    let mut waits = 0;
    for make in makes {
        let began = if make.first_wait != 0 {
            make.first_wait
        } else {
            make.began
        };
        if make.waits == 0 || began > end || make.ended < start || told.contains(&make.number) {
            continue;
        }
        waits += u64::from(make.waits);
        if told.len() == 256 {
            told.pop_front();
        }
        told.push_back(make.number);
    }
    waits
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A make of rt's pool told now is found ended within a span that holds now, and not within
    /// one that ended before it
    #[test]
    fn a_make_ended_during_a_span_is_found_in_it_and_not_before_it() {
        let before = rt::host_time();
        pool_made();
        let after = rt::host_time();
        let (n, latest) = pool_makes_ended(before, after);
        assert!(n >= 1, "{n} makes ended in {before}..={after}");
        assert!((before..=after).contains(&latest), "{latest}");
        let (_, earlier) = pool_makes_ended(0, before.saturating_sub(1));
        assert!(earlier < before, "{earlier} ended before {before}");
    }
}
