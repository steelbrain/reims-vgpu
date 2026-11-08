//! [`KeyHasher`]: the fixed hasher of [`super::LookupMap`] and of the maps `reims-vgpu`'s
//! decoder keys by the guest's IDs.
//!
//! A multiply-and-add hash of each integer written, rotated at the end so that the bits the
//! table indexes by mix every input bit (the multiply moves entropy up, never down). It is
//! fixed: no per-process seed, so a map's order is a function of its keys and its insert and
//! remove history, the same in every run. The maps' keys are the guest's references and IDs,
//! which pick no collisions to slow a lookup down beyond their own table's.

use std::hash::{BuildHasherDefault, Hasher};

use crate::num::ToU64;

/// The odd multiplier, a 64-bit constant with its bits spread (the one `rustc-hash` 2 uses).
const K: u64 = 0xf135_7aea_2e62_a9c5;

/// The hasher; see the module docs.
#[derive(Debug, Clone, Copy, Default)]
pub struct KeyHasher(u64);

/// The maps' hasher builder.
pub type KeyBuild = BuildHasherDefault<KeyHasher>;

impl KeyHasher {
    fn add(&mut self, word: u64) {
        self.0 = self.0.wrapping_add(word).wrapping_mul(K);
    }
}

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }

    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for word in words {
            self.add(u64::from_le_bytes(*word));
        }
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            if let Some(w) = word.get_mut(..rest.len()) {
                w.copy_from_slice(rest);
            }
            // The length tells a tail of zeros from a shorter one.
            self.add(u64::from_le_bytes(word) ^ rest.len().to_u64() << 56);
        }
    }

    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }

    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    fn write_usize(&mut self, i: usize) {
        self.add(i.to_u64());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::BuildHasher;

    /// The same key hashes the same in every hasher, and keys a table tells apart by their low
    /// bits (references counting up, addresses 16 bytes apart) still differ in the low bits of
    /// the hash, which index the table.
    #[test]
    fn keys_spread_over_the_low_bits() {
        let build = KeyBuild::default();
        assert_eq!(build.hash_one(7u32), build.hash_one(7u32));
        for (step, count) in [(1u64, 64u64), (16, 64), (0x4000, 64)] {
            let mut low: Vec<u64> = (0..count).map(|k| build.hash_one(k * step) & 63).collect();
            low.sort_unstable();
            low.dedup();
            assert!(
                low.len() >= 32,
                "keys {step} apart fill {} of 64 buckets",
                low.len()
            );
        }
    }
}
