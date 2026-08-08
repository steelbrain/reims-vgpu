//! The two content hashes the Metal backend's compiled-object caches key on.
//!
//! Both are built from the FNV-1a parameters in [`crate::contract::fnv`]. The
//! constants live there rather than here because callers of theirs are not
//! backends at all — the sampled gather witness and the panic latch both fold
//! with them — and this module was behind `feature = "backend-metal"` when
//! those were written, so they could not name these and wrote the basis and
//! the prime out as literals instead. Its
//! declaration in [`crate::backend`] says why it is no longer gated; the
//! constants stay where they are, because that split is about who can *see*
//! them and moving the module did not change who needs them.
//!
//! # What constrains these numbers
//!
//! This file's doc used to say the hash matched an ObjC `reims_vgpu_hash_bytes`.
//! **No such symbol exists anywhere in this repository, in any language, at any
//! commit** — `git log -S` finds only the commit that added that sentence. Read
//! it as provenance for where the algorithm came from, not as a live
//! cross-check, because there is nothing here to cross-check against.
//!
//! So nothing outside this crate pins these values. The only requirement the
//! tree actually imposes is self-consistency within one process: every producer
//! and consumer of a cache key must fold the same way. That is a weaker
//! obligation than an ABI, and worth stating plainly rather than leaving a
//! reader to infer an external contract that is not there.

use crate::contract::fnv::{fold_bytes, FNV_OFFSET_BASIS, FNV_PRIME};

/// Fold `data` into a 64-bit content hash.
///
/// The trailing length mix is not stock FNV-1a: it makes a run of zero bytes
/// hash differently from a shorter one, which matters because these keys are
/// compared against shader blobs that can share a prefix. That extra step is
/// why this stays here rather than joining [`fold_bytes`] in `contract` — a
/// caller that reached for the shared fold and got this one would silently
/// produce keys from a different keyspace.
pub fn hash_bytes(data: &[u8]) -> u64 {
    let h = fold_bytes(FNV_OFFSET_BASIS, data);
    (h ^ data.len() as u64).wrapping_mul(FNV_PRIME)
}

/// Fold one `u64` into a running hash.
///
/// Not FNV-1a despite the basis its callers seed it with: this is the
/// golden-ratio mix, which combines a whole word per step where FNV-1a takes
/// eight. Cache keys here are built from dozens of small integer fields, so the
/// per-word form is the one they want; the shared basis only means the two
/// hashes in this module start from the same place.
pub fn hash_u64(mut h: u64, v: u64) -> u64 {
    h ^= v
        .wrapping_add(0x9e3779b97f4a7c15)
        .wrapping_add(h << 6)
        .wrapping_add(h >> 2);
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned against a literal, not against a recomputation.
    ///
    /// This test used to re-spell both constants and redo the function's own
    /// final step, so it could only catch an edit to `hash_bytes` — an edit to
    /// a *constant* changed both sides at once and the assertion followed it.
    /// The literal is derivable rather than observed: with no bytes to fold,
    /// `hash_bytes` reduces to FNV-1a over the single zero byte contributed by
    /// the length mix, and FNV-1a of one NUL byte is the published
    /// `0xaf63bd4c8601b7df`. So this pins the basis, the prime, and the length
    /// mix at once, against a value from outside this file.
    #[test]
    fn the_empty_input_hashes_to_fnv1a_of_one_zero_byte() {
        assert_eq!(hash_bytes(b""), 0xaf63_bd4c_8601_b7df);
    }

    #[test]
    fn distinguishes_content() {
        assert_ne!(hash_bytes(b"a"), hash_bytes(b"b"));
    }
}
