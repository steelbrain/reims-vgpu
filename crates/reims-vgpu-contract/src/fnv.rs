//! FNV-1a's published constants and its fold step.
//!
//! # Why a backend-neutral home
//!
//! Several places in this crate fold with FNV-1a, and only one family of them
//! is a backend: the sampled gather witness names a window with it, the panic
//! latch folds an entry point and its raise site into the discriminant it
//! dedupes on, and the Metal backend's compiled-object caches key shaders and
//! descriptors with it. The constants used to be declared inside the Metal
//! backend's own hash module, behind `feature = "backend-metal"` — so the sites
//! outside it *could not* name them even if their authors had looked, and wrote
//! the basis and the prime out as literals instead. (That module is
//! `crate::backend::hash` now, and ungated, for a different reason its own doc
//! gives.)
//!
//! They wrote them out in different shapes. Before this module existed the
//! basis appeared as `0xcbf2_9ce4_8422_2325` at three sites and as
//! `14695981039346656037` at one, and the prime as `0x0000_0100_0000_01b3` at
//! two sites and `0x100_0000_01b3` at a third. No grep finds those spellings
//! together, which is the whole hazard: a hash whose seeds diverge does not
//! fail loudly. It stops sharing cache entries and reads as a cold cache.
//!
//! # What this module does and does not promise
//!
//! It promises the published FNV-1a 64-bit parameters and the exact fold every
//! caller was already performing. It does **not** promise that the callers
//! produce comparable digests: each folds a different sequence, and two add
//! their own finishing steps. Sharing the constants makes them agree on the
//! algorithm, not on the keyspace — which is what was wanted, since the
//! keyspaces are deliberately separate.

/// FNV-1a's 64-bit offset basis: the state every fold starts from.
pub const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a's 64-bit prime.
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Continue an FNV-1a fold over `bytes`.
///
/// Takes the running state rather than seeding it, because two callers fold
/// several values into one digest and a seeding helper would restart them.
/// Start from [`FNV_OFFSET_BASIS`].
pub fn fold_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Continue an FNV-1a fold over `value`'s little-endian bytes.
///
/// Little-endian is not a choice this module gets to make: it is what the
/// runtime caller already did, and its digests name live sampled windows. The
/// byte order is fixed by [`fold_bytes`] seeing the same sequence on every host,
/// so `to_le_bytes` is spelled here rather than `to_ne_bytes`.
pub fn fold_u64(hash: u64, value: u64) -> u64 {
    fold_bytes(hash, &value.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned against the published FNV-1a test vector, not against a
    /// recomputation of this file's own arithmetic.
    ///
    /// FNV-1a of the single byte `0x00` is `0xaf63bd4c8601b7df`, a value from
    /// the algorithm's specification. One assertion therefore pins the basis,
    /// the prime, and the direction of the xor-then-multiply step at once. A
    /// test that recomputed `BASIS ^ 0` and multiplied by `PRIME` would follow
    /// an edit to either constant and could only ever catch an edit to the
    /// loop.
    #[test]
    fn one_zero_byte_hashes_to_the_published_vector() {
        assert_eq!(fold_bytes(FNV_OFFSET_BASIS, &[0u8]), 0xaf63_bd4c_8601_b7df);
    }

    /// The two constants are the published pair, whichever base a reader
    /// happens to have seen them in. Decimal here on purpose: it is the second
    /// spelling that used to live in another file, and holding both forms in
    /// one assertion is what makes them impossible to drift apart.
    #[test]
    fn the_constants_are_the_published_fnv1a_pair() {
        assert_eq!(FNV_OFFSET_BASIS, 14695981039346656037);
        assert_eq!(FNV_PRIME, 1099511628211);
    }

    /// [`fold_u64`] is [`fold_bytes`] over eight little-endian bytes, and its
    /// runtime caller depended on exactly that before it shared this code.
    /// Written as an independent fold rather than by calling `fold_bytes`, so a
    /// change to the byte order inside `fold_u64` is visible here.
    #[test]
    fn a_u64_folds_as_its_little_endian_bytes() {
        let value = 0x0123_4567_89ab_cdef_u64;
        let mut expected = FNV_OFFSET_BASIS;
        for byte in [0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01_u8] {
            expected ^= u64::from(byte);
            expected = expected.wrapping_mul(FNV_PRIME);
        }
        assert_eq!(fold_u64(FNV_OFFSET_BASIS, value), expected);
    }

    /// A fold is order-dependent, so a digest over a sequence names the
    /// sequence and not just its members. Both non-backend callers rely on this
    /// — the gather witness folds a discriminant followed by its window's
    /// fields, and the panic latch an entry point followed by its raise site.
    #[test]
    fn folding_two_values_depends_on_their_order() {
        let a = fold_u64(fold_u64(FNV_OFFSET_BASIS, 1), 2);
        let b = fold_u64(fold_u64(FNV_OFFSET_BASIS, 2), 1);
        assert_ne!(a, b);
    }
}
