//! Byte-bounded content memo with least-recently-used eviction.
//!
//! Replaces the ad-hoc `(BTreeMap, running-byte-total)` pairs that used to
//! `clear()` the whole map the instant a byte/entry cap was crossed. A full
//! clear drops the *entire* hot working set at once, so the next frames
//! re-decode/re-convert every surface that was cached — a cap-cliff hitch (the
//! "120 fps collapses to a handful once a cap blows" class this project guards
//! against). This map instead evicts only the least-recently-touched entries,
//! down to a low-water mark, so the hot set survives a cap crossing and the
//! reclaim cost is amortized across many inserts rather than paid all at once.
//!
//! Recency is bumped on every confirmed hit (`get_touch`) and on `insert`, so a
//! static-but-hot entry (e.g. a wallpaper plane sampled every frame without ever
//! being rewritten) is not evicted ahead of churny one-shot uploads.
//!
//! The map is deliberately value-agnostic: the caller supplies each entry's byte
//! weight at insert time (`entry_bytes`) — the map never introspects `V`.

use std::collections::BTreeMap;

/// A single memo entry plus its LRU bookkeeping.
#[derive(Debug)]
struct Slot<V> {
    value: V,
    /// Byte weight charged against the map's `byte_cap`.
    bytes: usize,
    /// Monotonic access stamp; the largest stamp is the most-recently-touched
    /// entry, the smallest is the eviction victim.
    access: u64,
}

/// Byte-bounded map with least-recently-used eviction. Keeps total live entry
/// bytes at or below `byte_cap`, evicting the coldest entries first.
#[derive(Debug)]
pub struct LruBytesMemo<K: Ord + Clone, V> {
    map: BTreeMap<K, Slot<V>>,
    bytes: usize,
    byte_cap: usize,
    seq: u64,
}

impl<K: Ord + Clone, V> LruBytesMemo<K, V> {
    /// Create an empty memo bounded to `byte_cap` total entry bytes.
    pub fn new(byte_cap: usize) -> Self {
        Self {
            map: BTreeMap::new(),
            bytes: 0,
            byte_cap,
            seq: 0,
        }
    }

    /// Current summed entry bytes (the value the byte cap bounds).
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Live entry count.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Kept because clippy requires it beside `len`, not because a caller
    /// exists; `len_without_is_empty` is a hard error at `-D warnings` on this
    /// crate.
    ///
    /// Removing the `allow` is not a way to delete this method; `len` has
    /// callers, so deleting it fails the clippy arms instead.
    #[allow(
        dead_code,
        reason = "exists to satisfy clippy::len_without_is_empty beside `len`, not for a caller"
    )]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Peek without touching recency — for read paths that hold only `&self`.
    /// Prefer [`Self::get_touch`] on the hot path so hits stay warm.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map.get(key).map(|s| &s.value)
    }

    /// Look up `key` and mark it most-recently-used. Use on the hit path: a hot
    /// entry read every frame (but never rewritten) stays warm and is evicted
    /// only after genuinely colder entries.
    pub fn get_touch(&mut self, key: &K) -> Option<&V> {
        self.seq = self.seq.wrapping_add(1);
        let seq = self.seq;
        let slot = self.map.get_mut(key)?;
        slot.access = seq;
        Some(&slot.value)
    }

    /// Remove `key`, returning its value and reclaiming its bytes.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let slot = self.map.remove(key)?;
        self.bytes = self.bytes.saturating_sub(slot.bytes);
        Some(slot.value)
    }

    /// Insert (or replace) `key` with `value` weighing `entry_bytes`, first
    /// evicting the least-recently-used entries so the result stays within the
    /// byte cap. A replace reclaims the old entry's bytes before charging the
    /// new. A single entry larger than the whole cap is admitted alone (matching
    /// the prior clear-then-insert behavior) rather than rejected.
    pub fn insert(&mut self, key: K, value: V, entry_bytes: usize) {
        if let Some(old) = self.map.remove(&key) {
            self.bytes = self.bytes.saturating_sub(old.bytes);
        }
        self.evict_for(entry_bytes);
        self.seq = self.seq.wrapping_add(1);
        let access = self.seq;
        self.map.insert(
            key,
            Slot {
                value,
                bytes: entry_bytes,
                access,
            },
        );
        self.bytes += entry_bytes;
    }

    /// Evict least-recently-used entries until `incoming` bytes fit under the
    /// low-water mark (7/8 of the cap). Draining to a low-water rather than
    /// exactly to the cap means a steady insert stream evicts in occasional
    /// batches with headroom, not one-for-one on every insert at the boundary.
    /// Stops early once the map empties (an oversized single entry is then
    /// admitted over the cap by the caller).
    fn evict_for(&mut self, incoming: usize) {
        let low_water = self.byte_cap - self.byte_cap / 8;
        if self.bytes + incoming <= low_water {
            return;
        }
        // Coldest-first. Eviction only runs at the cap boundary (never on the
        // steady hot path), so one ordered pass over the keys is acceptable.
        let mut by_access: Vec<(u64, K)> = self
            .map
            .iter()
            .map(|(k, s)| (s.access, k.clone()))
            .collect();
        by_access.sort_unstable_by_key(|(a, _)| *a);
        for (_, k) in by_access {
            if self.bytes + incoming <= low_water {
                break;
            }
            if let Some(slot) = self.map.remove(&k) {
                self.bytes = self.bytes.saturating_sub(slot.bytes);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_tracks_bytes_and_replace_reclaims() {
        let mut m: LruBytesMemo<u32, u8> = LruBytesMemo::new(1000);
        m.insert(1, 0xAA, 100);
        m.insert(2, 0xBB, 200);
        assert_eq!(m.len(), 2);
        assert_eq!(m.bytes(), 300);
        // Replacing a key reclaims the old bytes before charging the new.
        m.insert(1, 0xCC, 50);
        assert_eq!(m.len(), 2);
        assert_eq!(m.bytes(), 250);
        assert_eq!(*m.peek(&1).unwrap(), 0xCC);
    }

    #[test]
    fn remove_reclaims_bytes() {
        let mut m: LruBytesMemo<u32, u8> = LruBytesMemo::new(1000);
        m.insert(1, 0xAA, 100);
        m.insert(2, 0xBB, 200);
        assert_eq!(m.remove(&1), Some(0xAA));
        assert_eq!(m.bytes(), 200);
        assert_eq!(m.remove(&1), None);
        assert_eq!(m.bytes(), 200);
    }

    #[test]
    fn cap_cross_evicts_lru_not_the_whole_map() {
        // cap 800, low-water 700. Fill four 200-byte entries; the fourth insert
        // crosses the cap and must evict the COLDEST (key 1), leaving the rest —
        // never a full clear.
        let mut m: LruBytesMemo<u32, u32> = LruBytesMemo::new(800);
        m.insert(1, 1, 200);
        m.insert(2, 2, 200);
        m.insert(3, 3, 200);
        m.insert(4, 4, 200); // 800 -> would be over low-water; evicts key 1
        assert!(m.peek(&1).is_none(), "coldest entry evicted");
        assert!(m.peek(&2).is_some());
        assert!(m.peek(&3).is_some());
        assert!(m.peek(&4).is_some());
        assert!(m.bytes() <= 800);
        assert!(m.len() >= 3, "eviction is incremental, not a bulk clear");
    }

    #[test]
    fn get_touch_keeps_hot_entry_alive_across_cap_cross() {
        // Key 1 is the "static-but-hot" entry: never rewritten, but read every
        // round via get_touch. Under sustained inserts it must NOT be the victim.
        let mut m: LruBytesMemo<u32, u32> = LruBytesMemo::new(800);
        m.insert(1, 1, 200);
        m.insert(2, 2, 200);
        m.insert(3, 3, 200);
        for k in 4..12u32 {
            // Touch the hot entry right before each cap-crossing insert.
            assert!(m.get_touch(&1).is_some());
            m.insert(k, k, 200);
            assert!(m.peek(&1).is_some(), "hot entry survives round {k}");
        }
        assert!(m.bytes() <= 800);
    }

    #[test]
    fn oversized_single_entry_admitted_alone() {
        let mut m: LruBytesMemo<u32, u32> = LruBytesMemo::new(800);
        m.insert(1, 1, 200);
        m.insert(2, 2, 200);
        // An entry larger than the whole cap evicts everything else, then rides
        // alone (over cap) — same net effect the old clear-then-insert had.
        m.insert(9, 9, 2000);
        assert_eq!(m.len(), 1);
        assert_eq!(*m.peek(&9).unwrap(), 9);
        assert_eq!(m.bytes(), 2000);
    }
}
