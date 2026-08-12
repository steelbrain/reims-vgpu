//! When the guest last said it wrote a **buffer** object's bytes.
//!
//! # The signal this recovers
//!
//! [`crate::runtime::resource_validity::apply`] takes the guest's validity quad
//! for one object id and applies it to `DeviceState::mappings`. Buffers have no
//! mapping, so every quad naming one falls out of the loop and is counted as
//! `validity_no_surface` — ~4 700 a second on a driven macos-13
//! sustained-animation boot. The statement is decoded, correct, and discarded.
//!
//! That is the one signal the draw-time buffer gather has no substitute for.
//! `buffer_gather_working_set` measures ~20 800 gathers a second over ~1 900
//! distinct windows, so **91 % of them re-assemble a window this device already
//! assembled** — and whether that is 91 % of a copy that could be skipped or
//! 91 % of a copy that had to be redone depends entirely on whether the bytes
//! moved in between. Recurrence is about keys; this is about bytes.
//!
//! # Why not the hypervisor dirty bitmap, which is already built
//!
//! [`crate::runtime::gather_witness`] answers exactly this question for the
//! sampled rails, soundly, from two halves — the hypervisor dirty bitmap for
//! guest CPU stores and [`crate::runtime::host_writes`] for this device's own.
//! Its `MAX_TRACKED_WINDOWS` is 256, and that bound is **not** about memory: it
//! is a harvest bound, because `reims_vgpu_dirty_harvest` walks every page of
//! every armed set on the BQL thread at each register write that hands the
//! device work. The buffer rail's working set is ~1 900 windows of ~38 pages,
//! so arming it there would put ~72 000 pages into a walk the whole VM waits
//! on. That is not a resize; it is a different cost.
//!
//! The guest's own declaration costs nothing at all — it is already decoded,
//! and this is a `u64` bump on a map keyed by the object it already names.
//!
//! # It is an instrument, and its soundness is not yet established
//!
//! Nothing here decides a skip. A cache built on this would be trusting that
//! `writeInvalidates` and the exec table's validity quad are a **complete**
//! account of guest CPU writes to a buffer's bytes, and that claim has not been
//! tested — a surface's equivalent claim is not complete, which is exactly why
//! the sampled rail carries a hypervisor half as well. What this measures is the
//! *ceiling*: no cache invalidated by the guest's declarations can do better
//! than the clean rate reported here, so a low reading closes the design and a
//! high one is a licence to go and test the soundness, not to assume it.
//!
//! # The bound
//!
//! One entry per `(task, object)` the guest has declared a write to. A task's
//! entries go when the task does, which is the same lifetime `bound_buffers`
//! retires on and the only announcement this device gets. Past [`BufferWriteGens::MAX`] the
//! map is **cleared** rather than partially evicted: forgetting one object would
//! make its next comparison read as clean, which is the direction that reports a
//! cache hit for a window whose bytes moved. A clear makes every comparison read
//! as *unknown*, which is the safe direction, and `buffer_write_gen_reset` says
//! it happened.

use std::collections::HashMap;

/// Per-object write generations for objects this device holds no mapping for.
#[derive(Default, Debug)]
pub struct BufferWriteGens {
    gens: HashMap<(u32, u32), u64>,
    /// Bumped on every clear, so a comparison spanning one is not mistaken for
    /// a comparison that found the same generation twice. A reader stores this
    /// beside the generation and treats a change in it as "unknown".
    epoch: u64,
}

impl BufferWriteGens {
    /// The most `(task, object)` pairs tracked before the map resets.
    ///
    /// A driven boot's `bound_buffers` registry holds ~700 resolutions across
    /// **22** distinct `(task, reference)` pairs, and this is keyed the same way
    /// — by object, not by window — so 4 096 is two orders of magnitude above
    /// the only related number anyone has measured. It bounds a guest that
    /// creates and writes objects without bound, and `buffer_write_gen_reset`
    /// is what says whether it ever binds.
    pub const MAX: usize = 4096;

    /// Record that the guest declared a write to `object_id` under `task_id`.
    ///
    /// Called only for the ids [`crate::runtime::resource_validity::apply`]
    /// found no mapping for: an object with a mapping already has
    /// `content_generation`, and stamping it twice would be two spellings of one
    /// fact.
    pub fn note_write(&mut self, task_id: u32, object_id: u32) {
        if self.gens.len() >= Self::MAX && !self.gens.contains_key(&(task_id, object_id)) {
            self.gens.clear();
            self.epoch = self.epoch.wrapping_add(1);
            crate::runtime::drain::note_store_route("buffer_write_gen_reset");
        }
        let slot = self.gens.entry((task_id, object_id)).or_insert(0);
        *slot = slot.wrapping_add(1);
        // The cross-check that makes the freshness split readable, and it is not
        // optional. A reader compares this generation against a `(task,
        // reference)` pair taken at a draw-time bind, and if those two ids turn
        // out to be different namespaces then **no** comparison ever moves and
        // the split reports ~100 % quiet — a false positive in the direction
        // that licenses a cache serving stale bytes. A boot reading
        // `buffer_gather_freshness quiet_rate=1.000` beside a zero here has
        // measured a wiring fault and not a workload.
        crate::runtime::drain::note_store_route("buffer_write_gen_bump");
    }

    /// What a reader records beside a copy it has just taken, and compares
    /// against later.
    ///
    /// The epoch travels with the generation so a clear cannot be read as
    /// "unchanged": an object with no entry reads `(epoch, 0)`, and after a
    /// clear the epoch differs from every stamp taken before it.
    pub fn stamp(&self, task_id: u32, object_id: u32) -> BufferWriteStamp {
        BufferWriteStamp {
            epoch: self.epoch,
            gen: self.gens.get(&(task_id, object_id)).copied().unwrap_or(0),
        }
    }

    /// Forget one task's objects, because the task's ids no longer name them.
    ///
    /// Retiring by task rather than by object for the reason
    /// [`crate::runtime::bound_buffers`] states about its own registry: mapping
    /// an object id back to what resolved through it is machinery bought with
    /// nothing, and task teardown is rare.
    pub fn retire_task(&mut self, task_id: u32) {
        self.gens.retain(|&(task, _), _| task != task_id);
    }

    /// Entries held.
    ///
    /// Named for what it counts rather than `len`, because this is not a
    /// collection anything iterates and a `len`/`is_empty` pair would suggest it
    /// is.
    pub fn tracked(&self) -> usize {
        self.gens.len()
    }
}

/// One object's write generation as a reader saw it.
///
/// Two of these are comparable only when their epochs agree; see
/// [`BufferWriteGens::stamp`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferWriteStamp {
    epoch: u64,
    gen: u64,
}

impl BufferWriteStamp {
    /// Whether the guest has declared no write to this object between the two
    /// stamps.
    ///
    /// `false` for a pair that straddles a map clear, which is the unknown case
    /// answered in the safe direction — see this module's doc on [`BufferWriteGens::MAX`].
    pub fn quiet_since(self, earlier: Self) -> bool {
        self.epoch == earlier.epoch && self.gen == earlier.gen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generation moves only when the guest says it wrote, which is the
    /// whole of what a cache would key on.
    #[test]
    fn a_declared_write_moves_the_generation_and_nothing_else_does() {
        let mut g = BufferWriteGens::default();
        let before = g.stamp(1, 7);
        assert!(g.stamp(1, 7).quiet_since(before), "no write, no move");
        g.note_write(1, 7);
        assert!(!g.stamp(1, 7).quiet_since(before));
        let after = g.stamp(1, 7);
        assert!(g.stamp(1, 7).quiet_since(after), "still no further write");
    }

    /// A write to one object must not invalidate another's stamp, or the
    /// measurement collapses to "anything was written" and reports a floor of
    /// zero for every window.
    #[test]
    fn a_write_to_another_object_leaves_this_one_quiet() {
        let mut g = BufferWriteGens::default();
        let before = g.stamp(1, 7);
        g.note_write(1, 8);
        g.note_write(2, 7);
        assert!(g.stamp(1, 7).quiet_since(before));
    }

    /// A stamp taken before a clear must not compare equal to one taken after
    /// it. Forgetting an object silently is the direction that reports a hit
    /// for a window whose bytes moved.
    #[test]
    fn a_stamp_that_straddles_a_reset_is_not_quiet() {
        let mut g = BufferWriteGens::default();
        g.note_write(1, 7);
        let before = g.stamp(1, 7);
        for object in 0..(BufferWriteGens::MAX as u32 + 1) {
            g.note_write(9, object);
        }
        assert!(
            !g.stamp(1, 7).quiet_since(before),
            "the map was cleared under this reader, so it cannot say the object was quiet"
        );
    }

    /// A task that goes takes its objects with it, so a later task reusing an
    /// id cannot inherit a stamp that was about something else.
    #[test]
    fn retiring_a_task_forgets_its_objects() {
        let mut g = BufferWriteGens::default();
        g.note_write(1, 7);
        g.note_write(2, 7);
        let before = g.stamp(1, 7);
        g.retire_task(1);
        assert_eq!(g.tracked(), 1, "only task 2's entry remains");
        assert!(
            !g.stamp(1, 7).quiet_since(before),
            "the entry is gone, so its generation reads as 0 and cannot match"
        );
    }

    /// The map stops at its bound rather than tracking a guest that creates
    /// objects without end.
    #[test]
    fn the_map_stops_at_its_bound() {
        let mut g = BufferWriteGens::default();
        for object in 0..(BufferWriteGens::MAX as u32 + 5) {
            g.note_write(1, object);
        }
        assert!(g.tracked() <= BufferWriteGens::MAX);
    }
}
