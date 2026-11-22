//! What a slow guest answer ran over, for reims-vgpu's `graphics::slow_answers`, which is told of
//! each ring-0 answer the host gave a frame or more after its make (reims-vgpu-graphics'
//! `Host::answer_slow`). A diagnostic: which lines the host writes is free under the contract.
//!
//! - **The makes of rt's pool.** rt tells the device's `BlockMappings` (`super::fault`'s
//!   `Allocations`) of each allocation of its pool right after it made it, on the thread that made
//!   it: the block-maker thread's makes ahead of need, a take's make on the take's thread, and
//!   each class's first in the device's make. [`pool_made`] keeps the last [`KEPT`]
//!   ([`PoolMake`]): when each began and ended, on rt's host clock (`rt::host_time`), from the
//!   start and the length of its make that rt's `AllocationInfo` gives with its allocation's
//!   length and storage mode, and the thread rt told it on, the one that made it; a clone of the
//!   thread's handle and an uncontended lock a make. An answer counts as over a make where the
//!   make ran during it, from its start to its end ([`pool_makes_met`]): one that began before
//!   the ask, or ended inside the answer, among them. rt tells of a make only once it ended, so a
//!   make still running when the answer is read is not among them.
//! - **The first-use waits of compute pipelines.** rt keeps its last 256 ended makes of a compute
//!   kernel's `VkPipeline` for a dispatch's variant, with when each began and ended, how many
//!   dispatches waited for it (its first-use waits, rt's `Metrics::first_use_variant_waits`) and
//!   when the earliest of them began (rt's `Counters::variant_makes`). [`first_use_waits_met`]
//!   reads them, only for a slow answer, and counts each make's waits once.

use std::collections::VecDeque;
use std::thread::Thread;

use metal2vulkan_runtime as rt;

use crate::core::sync::Mutex;

/// The makes of rt's pool [`pool_made`] keeps: the last 16.
pub const KEPT: usize = 16;

/// A make of rt's pool, as [`pool_made`] keeps it.
#[derive(Clone, Debug)]
pub struct PoolMake {
    /// When it began, on rt's host clock (rt's `AllocationInfo::make_start`).
    pub began: u64,
    /// When it ended, on rt's host clock: its start and its length (rt's
    /// `AllocationInfo::make_nanos`), just before rt told the device's `BlockMappings` of it.
    pub ended: u64,
    /// Its allocation's length, in bytes.
    pub len: u64,
    /// The storage mode of its blocks.
    pub class: rt::MemoryClass,
    /// The thread that made it, which rt told it on.
    pub thread: Thread,
}

/// The last [`KEPT`] makes of rt's pool, the oldest first.
static MADE: Mutex<VecDeque<PoolMake>> = Mutex::new(VecDeque::new());

/// The numbers of rt's variant makes whose waits [`first_use_waits_met`] counted, the last 256:
/// as many as rt keeps.
static TOLD: Mutex<VecDeque<u64>> = Mutex::new(VecDeque::new());

/// rt made an allocation of its pool just now, on this thread, `len` bytes of `class`, in a make
/// that began at `began` of rt's host clock and took `nanos` (`super::fault`'s `Allocations`,
/// which rt tells of each right after its make, with its `AllocationInfo::make_start` and
/// `make_nanos`).
pub(super) fn pool_made(began: u64, nanos: u64, len: u64, class: rt::MemoryClass) {
    let make = PoolMake {
        began,
        ended: began.saturating_add(nanos),
        len,
        class,
        thread: std::thread::current(),
    };
    let mut made = MADE.lock();
    if made.len() == KEPT {
        made.pop_front();
    }
    made.push_back(make);
}

/// The makes of rt's pool among the last [`KEPT`] that ran during `start..=end` of rt's host
/// clock, from their start to their end, in the order they began.
pub fn pool_makes_met(start: u64, end: u64) -> Vec<PoolMake> {
    let made = MADE.lock();
    let mut met: Vec<PoolMake> = (made.iter())
        .filter(|make| make.began <= end && make.ended >= start)
        .cloned()
        .collect();
    drop(made);
    met.sort_by_key(|make| (make.began, make.ended));
    met
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

    /// A make of rt's pool told on a thread is found in a span it ran in, with its start, its
    /// end, its length, its storage mode and the thread it was told on, and not in one that ended
    /// before it began
    #[test]
    fn a_make_that_ran_during_a_span_is_found_in_it_and_not_before_it() {
        const LEN: u64 = 0x123_0000;
        let thread = std::thread::Builder::new().name("reims-vgpu.test-maker".into());
        let began = rt::host_time();
        let told = thread.spawn(move || pool_made(began, 5_000, LEN, rt::MemoryClass::Private));
        told.expect("a thread").join().expect("the make told");
        let met = pool_makes_met(began + 1_000, began + 2_000);
        let make = met.iter().find(|make| make.len == LEN);
        let make = make.unwrap_or_else(|| panic!("{met:?} met {began}"));
        assert_eq!((make.began, make.ended), (began, began + 5_000), "{make:?}");
        assert_eq!(make.class, rt::MemoryClass::Private, "{make:?}");
        assert_eq!(make.thread.name(), Some("reims-vgpu.test-maker"));
        let earlier = pool_makes_met(0, began.saturating_sub(1));
        assert!(earlier.iter().all(|make| make.len != LEN), "{earlier:?}");
        let later = pool_makes_met(began + 5_001, began + 6_000);
        assert!(later.iter().all(|make| make.len != LEN), "{later:?}");
    }

    /// A make that began before a span and ended in it, and one that began in it and ended after
    /// it, both ran during it, and are found in the order they began
    #[test]
    fn makes_over_either_end_of_a_span_are_found_in_the_order_they_began() {
        const LEN: u64 = 0x789_0000;
        let at = rt::host_time();
        pool_made(at + 3_000, 4_000, LEN + 1, rt::MemoryClass::Shared);
        pool_made(at, 2_000, LEN, rt::MemoryClass::Shared);
        let met = pool_makes_met(at + 1_000, at + 4_000);
        let ours: Vec<u64> = (met.iter())
            .filter(|make| (LEN..LEN + 2).contains(&make.len))
            .map(|make| make.len - LEN)
            .collect();
        assert_eq!(ours, [0, 1], "{met:?}");
    }

    /// Of more makes than [`KEPT`], the last [`KEPT`] are kept, in the order they began
    #[test]
    fn the_last_kept_makes_are_found_in_the_order_they_began() {
        const LEN: u64 = 0x456_0000;
        let kept = u64::try_from(KEPT).unwrap();
        let before = rt::host_time();
        for n in 0..kept + 4 {
            pool_made(rt::host_time(), 0, LEN + n, rt::MemoryClass::Shared);
        }
        let met = pool_makes_met(before, rt::host_time());
        let ours: Vec<u64> = (met.iter())
            .filter(|make| (LEN..LEN + 0x1_0000).contains(&make.len))
            .map(|make| make.len - LEN)
            .collect();
        assert!(ours.len() <= KEPT, "{ours:?}");
        assert!(ours.windows(2).all(|w| w[0] < w[1]), "{ours:?}");
        assert_eq!(ours.last(), Some(&(kept + 3)), "{ours:?}");
        let starts: Vec<u64> = met.iter().map(|make| make.began).collect();
        assert!(starts.windows(2).all(|w| w[0] <= w[1]), "{starts:?}");
    }
}
