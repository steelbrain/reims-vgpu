//! The identity of a shader blob, in its two halves: what a lookup borrows and
//! what an entry retains.
//!
//! Every compiled-object cache in the Metal backend is keyed on a blob the guest
//! supplied — a `.mtlb` library for a function, a kernel library for a compute
//! pipeline state or a texture reflection. Three of them keyed on
//! [`crate::backend::hash::hash_bytes`] of that blob beside its length and
//! retained no copy of it, so `CacheEntry::matches` compared two `u64`s and a
//! length. Two distinct blobs of equal length whose digests collide were one
//! entry: the guest asked for a shader it had submitted and received a compiled
//! object built from somebody else's bytes, with nothing to refuse and nothing
//! to log.
//!
//! # Why this is a fix and not a wider hash
//!
//! The standing argument for the digest was the birthday bound — at the 75
//! distinct shaders a driven boot settles at, a 64-bit collision is around
//! `2e-16`. That arithmetic is right and it is the wrong shape. It prices a
//! failure this device cannot observe if it ever happens, and a wider hash only
//! moves the exponent. Retaining the bytes removes the class.
//!
//! The cost is one copy of each *distinct* blob for the life of the process. The
//! entry beside it already retains an `MTLFunction`, an
//! `MTLComputePipelineState` or an `MTLRenderPipelineState` compiled from those
//! same bytes, so this is a fraction of what the cache holds rather than a
//! doubling of it. `crate::runtime::m2v_cache` made the same trade against the
//! AIR blobs and its `Slot` doc carries the same reasoning.
//!
//! # Why it is here and not under `backend::metal`
//!
//! Neither type names anything from the `metal` crate, and everything under
//! `backend/metal/` is `cfg`-ed out of the arm a Linux host builds — so gating
//! this would put the tests below on the arm nobody can run. That is the reason
//! [`crate::backend::hash`]'s declaration gives for sitting out here, and this is
//! the same case: the identity compare is the thing worth testing and it is pure
//! byte arithmetic.

#![cfg_attr(not(feature = "backend-metal"), allow(dead_code))]

use crate::backend::hash::hash_bytes;
use std::sync::Arc;

/// A blob a caller is asking about, borrowed, beside the digest that buckets it.
///
/// Borrowed rather than owned because a lookup happens once per pipeline build
/// and an owned key would copy the blob to throw away on every hit. Only
/// [`BlobIdentity::of`] takes ownership, once per distinct blob.
#[derive(Clone, Copy)]
pub struct BlobKey<'a> {
    /// Picks the bucket and decides nothing. [`BlobIdentity::is`] is the hit.
    pub hash: u64,
    pub bytes: &'a [u8],
}

impl<'a> BlobKey<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            hash: hash_bytes(bytes),
            bytes,
        }
    }
}

/// The retained half: a cache entry's own copy of the blob it was compiled from.
///
/// `Arc` rather than `Vec` so an entry filed in two caches under one blob — a
/// `.mtlb` is the key of both the function cache and the compute pipeline cache
/// — can share the copy when a caller already holds one.
pub struct BlobIdentity {
    hash: u64,
    bytes: Arc<[u8]>,
}

impl BlobIdentity {
    /// Retain `key`'s bytes. This is the only copy taken on the whole path.
    pub fn of(key: &BlobKey<'_>) -> Self {
        Self {
            hash: key.hash,
            bytes: Arc::from(key.bytes),
        }
    }

    /// Lend this identity back as a lookup key, so an insert re-scans with
    /// exactly what it filed rather than with a second key from the caller.
    pub fn as_key(&self) -> BlobKey<'_> {
        BlobKey {
            hash: self.hash,
            bytes: &self.bytes,
        }
    }

    /// The full identity compare. The digest is checked first only because it
    /// rejects in one word; the bytes are what decide.
    pub fn is(&self, key: &BlobKey<'_>) -> bool {
        self.hash == key.hash && *self.bytes == *key.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two different blobs filed under one digest stay two blobs.
    ///
    /// A natural 64-bit collision is a 2^32 meet-in-the-middle and not something
    /// a test can produce, so build the state one would produce — equal `hash`,
    /// different bytes — and ask the compare. This is the whole of what the
    /// retained copy buys.
    #[test]
    fn a_collided_digest_over_different_bytes_is_not_the_same_blob() {
        let first: Vec<u8> = (0..64u8).collect();
        let second: Vec<u8> = (0..64u8).map(|b| b ^ 0x5a).collect();
        let identity = BlobIdentity::of(&BlobKey::new(&first));

        let collided = BlobKey {
            hash: identity.hash,
            bytes: &second,
        };
        assert!(
            !identity.is(&collided),
            "equal digests over different bytes must not be one entry"
        );
        assert!(identity.is(&BlobKey::new(&first)));
    }

    /// Equal bytes are one blob however the key was built, which is what makes a
    /// second ask a hit rather than a recompile.
    #[test]
    fn the_same_bytes_from_two_slices_are_the_same_blob() {
        let air: Vec<u8> = (0..64u8).map(|b| b.wrapping_mul(7)).collect();
        let copy = air.clone();
        let identity = BlobIdentity::of(&BlobKey::new(&air));
        assert!(identity.is(&BlobKey::new(&copy)));
        assert_eq!(BlobKey::new(&air).hash, BlobKey::new(&copy).hash);
    }

    /// A prefix of a blob is not the blob. The digest folds the length, so this
    /// also fails on the prefilter — the point is that `is` does not need it to.
    #[test]
    fn a_prefix_is_not_the_blob() {
        let air: Vec<u8> = (0..64u8).collect();
        let identity = BlobIdentity::of(&BlobKey::new(&air));
        let short = BlobKey {
            hash: identity.hash,
            bytes: &air[..32],
        };
        assert!(!identity.is(&short));
    }

    /// `as_key` lends back exactly what was retained, which is what lets an
    /// insert re-scan without taking a second key from its caller.
    #[test]
    fn an_identity_lends_back_the_key_it_was_built_from() {
        let air: Vec<u8> = (0..8u8).collect();
        let identity = BlobIdentity::of(&BlobKey::new(&air));
        let lent = identity.as_key();
        assert_eq!(lent.hash, BlobKey::new(&air).hash);
        assert_eq!(lent.bytes, &air[..]);
        assert!(identity.is(&lent));
    }
}
