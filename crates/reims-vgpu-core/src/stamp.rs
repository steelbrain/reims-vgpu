//! Device-local completion-stamp scheduling state.

use std::collections::BTreeMap;

pub use reims_vgpu_protocol::StampWait;

#[derive(Clone, Copy, Default)]
pub struct PendingStamp {
    value: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnmetSource {
    Coalesced,
    Queued,
    Absent,
}

/// Device-local ownership of completion words from coalescing through
/// publication.
///
/// A word can be owed by a drain-local latch, handed to the ordered
/// publication rail, and finally become visible to the guest.  The sequence is
/// advanced at the last transition (or when an asynchronous publication can
/// become visible at any moment), so it is a progress witness for the same
/// state rather than an independent counter.
#[derive(Clone, Default, Debug)]
pub struct CompletionPublications {
    ledger: StampLedger,
    sequence: u64,
    held_timelines: u32,
}

#[derive(Clone, Default, Debug)]
pub struct StampLedger {
    owed: BTreeMap<u32, u32>,
    written: BTreeMap<u32, u32>,
}

fn slot_fits(slot: u32, page_bytes: u64) -> bool {
    u64::from(slot)
        .checked_mul(reims_vgpu_protocol::STAMP_SLOT_LEN)
        .is_some_and(|offset| offset < page_bytes)
}

impl StampLedger {
    pub fn owe(&mut self, slot: u32, value: u32, page_bytes: u64) {
        Self::fold(&mut self.owed, slot, value, page_bytes);
    }

    pub fn wrote(&mut self, slot: u32, value: u32, page_bytes: u64) {
        let slot = slot & reims_vgpu_protocol::STAMP_INDEX_MASK;
        if !slot_fits(slot, page_bytes) {
            return;
        }
        Self::fold(&mut self.written, slot, value, page_bytes);
        if self
            .owed
            .get(&slot)
            .is_some_and(|held| (held.wrapping_sub(value) as i32) <= 0)
        {
            self.owed.remove(&slot);
        }
    }

    pub fn classify(&self, wait: StampWait) -> UnmetSource {
        let slot = wait.slot();
        if self
            .owed
            .get(&slot)
            .is_some_and(|value| wait.satisfied_by(*value))
        {
            return UnmetSource::Coalesced;
        }
        if self
            .written
            .get(&slot)
            .is_some_and(|value| wait.satisfied_by(*value))
        {
            return UnmetSource::Queued;
        }
        UnmetSource::Absent
    }

    fn fold(map: &mut BTreeMap<u32, u32>, slot: u32, value: u32, page_bytes: u64) {
        let slot = slot & reims_vgpu_protocol::STAMP_INDEX_MASK;
        if !slot_fits(slot, page_bytes) {
            return;
        }
        map.entry(slot)
            .and_modify(|held| {
                if (value.wrapping_sub(*held) as i32) > 0 {
                    *held = value;
                }
            })
            .or_insert(value);
    }
}

impl CompletionPublications {
    /// Record a completion word still held by the coalescing rail.
    pub fn owe(&mut self, slot: u32, value: u32, page_bytes: u64) {
        self.ledger.owe(slot, value, page_bytes);
    }

    /// Transfer a completion word from coalescing to the ordered publication
    /// rail.  This does not itself claim that the guest can see the word.
    pub fn hand_to_publication(&mut self, slot: u32, value: u32, page_bytes: u64) {
        self.ledger.wrote(slot, value, page_bytes);
    }

    pub fn classify(&self, wait: StampWait) -> UnmetSource {
        self.ledger.classify(wait)
    }

    /// Record the point after which the guest may observe a completion word.
    pub fn note_may_be_visible(&mut self) {
        self.sequence = self.sequence.wrapping_add(1);
    }

    /// Snapshot used to detect whether retrying held timelines published work.
    pub fn progress(&self) -> u64 {
        self.sequence
    }

    /// Hold one root/child FIFO timeline behind an unmet completion word.
    pub fn hold_timeline(&mut self, bit: u32) {
        self.held_timelines |= bit;
    }

    /// Release one timeline before retrying its FIFO head.
    pub fn release_timeline(&mut self, bit: u32) {
        self.held_timelines &= !bit;
    }

    pub fn held_timelines(&self) -> u32 {
        self.held_timelines
    }
}

impl PendingStamp {
    pub fn latch(&mut self, stamp: u32) {
        self.value = Some(match self.value {
            Some(held) if stamp.wrapping_sub(held) as i32 <= 0 => held,
            _ => stamp,
        });
    }

    pub fn owed(self) -> Option<u32> {
        self.value
    }

    pub fn discharges(self, slot: u32, wait: StampWait) -> bool {
        wait.slot() == slot && self.value.is_some_and(|value| wait.satisfied_by(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_separates_local_and_queued_publication() {
        let page = 4096;
        let wait = StampWait { index: 2, value: 7 };
        let mut ledger = StampLedger::default();
        assert_eq!(ledger.classify(wait), UnmetSource::Absent);
        ledger.owe(2, 7, page);
        assert_eq!(ledger.classify(wait), UnmetSource::Coalesced);
        ledger.wrote(2, 7, page);
        assert_eq!(ledger.classify(wait), UnmetSource::Queued);
    }

    #[test]
    fn completion_publication_progress_moves_only_at_visibility_boundary() {
        let page = 4096;
        let wait = StampWait { index: 2, value: 7 };
        let mut publications = CompletionPublications::default();

        publications.owe(2, 7, page);
        assert_eq!(publications.classify(wait), UnmetSource::Coalesced);
        assert_eq!(publications.progress(), 0);

        publications.hand_to_publication(2, 7, page);
        assert_eq!(publications.classify(wait), UnmetSource::Queued);
        assert_eq!(publications.progress(), 0);

        publications.note_may_be_visible();
        assert_eq!(publications.progress(), 1);
    }

    #[test]
    fn held_timeline_membership_retires_independently() {
        let mut publications = CompletionPublications::default();
        publications.hold_timeline(0b001);
        publications.hold_timeline(0b100);
        assert_eq!(publications.held_timelines(), 0b101);
        publications.release_timeline(0b001);
        assert_eq!(publications.held_timelines(), 0b100);
    }

    #[test]
    fn pending_stamp_keeps_wrapping_later_value() {
        let mut pending = PendingStamp::default();
        pending.latch(u32::MAX);
        pending.latch(0);
        assert_eq!(pending.owed(), Some(0));
    }
}
