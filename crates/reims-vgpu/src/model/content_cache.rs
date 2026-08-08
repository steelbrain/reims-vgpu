//! Unbounded content-keyed cache of compiled GPU objects.
//!
//! The sibling of [`super::lru_memo`], and here beside it for the same reason:
//! this is a container, not a backend fact. Nothing in it names a GPU type. It
//! lived inside `backend::metal::cache`, which is behind
//! `feature = "backend-metal"`, so its tests — all of which drive the table with
//! an entry that holds no Metal object at all — were compiled out on every
//! non-Apple host and had never run anywhere a Linux checkout could see. They
//! run on every arm from here, which is the whole point of the location.
//!
//! # Why there is no capacity
//!
//! This held `cap` entries and, once full, overwrote a rotating slot — a clock
//! hand with no reference bit, so the victim was whichever slot came next
//! regardless of whether the guest was still drawing with it.
//!
//! The caps were 96 functions, 64 render pipeline states, 64 compute pipeline
//! states, 64 reflections, 32 samplers and 16 depth-stencil states.
//!
//! **The render-pipeline cap was below what this guest asks for.** The Vulkan
//! arm decodes the same command stream into the same object identities, so its
//! `object_cache_levels` census is a direct reading of the guest's distinct
//! object set. A driven x86 boot, window-drag probe against Safari, settles at:
//!
//! ```text
//!   m2v=75  shaders=75  layouts=33  passes=4  pipelines=92  samplers=14
//!   compute_pipelines=16
//! ```
//!
//! 92 distinct render pipelines against a 64-slot table is not headroom, it is
//! sustained thrash: 28 pipelines more than the table holds, every one of them
//! live, with a rotating hand choosing the victim. On a compositing desktop the
//! object it picks is more often than not one that will be bound again on the
//! next frame, and rebuilding it is `newRenderPipelineStateWithDescriptor:` —
//! a shader compile.
//!
//! So the bound was not protecting the host from the guest; it was capping the
//! guest below what it had already been observed to need. The live entry count
//! here is the number of *distinct* objects the guest has compiled, which is a
//! property of its own program and state set rather than of how long the device
//! has run — the same bound a real driver's pipeline cache has. When a guest
//! genuinely asks for more than the host can build, `newRenderPipelineState…`
//! returns nil and the caller declines with a reason. That is a GPU refusing
//! because its memory is full, which is the behaviour being emulated; silently
//! forgetting an object the guest still has bound is not.
//!
//! Removing the bound removes the linear scan with it. A `find` used to walk
//! every live slot, which was affordable only because the table was small. The
//! entries are indexed by the `u64` prefilter every one of these keys already
//! carries, so a lookup descends an ordered map to one bucket and walks that,
//! and [`CacheEntry::matches`] — the full identity compare — still decides every
//! hit exactly as before.
//!
//! # Why the lookup key is borrowed and the entry is not
//!
//! "The full identity compare" was once a claim about the shape rather than
//! about the keys. `CacheEntry`'s key type had no lifetime, so a key could only
//! hold what a caller could afford to build on every lookup — and for a cache of
//! compiled shader objects that ruled out the shader. Three of the six keys were
//! a 64-bit FNV digest of a blob beside its length, with no copy of the blob
//! retained anywhere, so `matches` compared two digests and a hit was one
//! collision away from handing the guest an `MTLFunction`, an
//! `MTLComputePipelineState` or a reflection built from bytes it never
//! submitted. Nothing would refuse and nothing would log.
//!
//! Splitting the key in two removes the class rather than moving its
//! probability: the entry retains the blob, `Key<'a>` borrows the caller's, and
//! `matches` compares content. The cost is one copy of each *distinct* blob for
//! the life of the process, which is a fraction of the compiled object already
//! retained beside it. `crate::backend::blob::BlobIdentity` is that pair;
//! `crate::runtime::m2v_cache` is the same fix made earlier against its own
//! table.

#![cfg_attr(not(feature = "backend-metal"), allow(dead_code))]

use std::collections::BTreeMap;

/// What makes two entries of one cache the same entry.
///
/// Stated once, beside the entry type. Each cache used to state it twice —
/// once in its `_lookup` scan and once in the re-scan its `_insert` does under
/// the lock — six rules in twelve places, with nothing comparing any pair. One
/// of the twelve was already missing: the reflection cache's insert did not
/// re-scan at all, so two callers that missed the same blob both pushed and the
/// cache carried a duplicate.
pub(crate) trait CacheEntry {
    /// The identity a lookup is made with, **borrowed**.
    ///
    /// Borrowed because the identity of a compiled GPU object is the blob that
    /// produced it, and a lookup happens once per pipeline build: an owned key
    /// would copy that blob to throw away on every hit. The entry retains its
    /// own copy — see [`Self::lookup_key`] — so the two halves are separate
    /// types by construction, and a key that is cheap to build cannot quietly
    /// become the thing the cache stores.
    type Key<'a>;
    /// The identity this entry was filed under, borrowed from the entry's own
    /// retained copy. An insert asks the entry for it rather than taking it a
    /// second time from the caller, so the two cannot disagree.
    fn lookup_key(&self) -> Self::Key<'_>;
    /// The full identity compare. This alone decides a hit; [`Self::bucket`]
    /// only narrows which entries are asked.
    fn matches(&self, key: &Self::Key<'_>) -> bool;
    /// A cheap `u64` that must be equal whenever [`Self::matches`] is true.
    ///
    /// Every key in this crate already carries one — the prefilter hash its
    /// `matches` consults before the byte compare — so this is a projection of
    /// the existing identity rather than a second one. Two keys sharing a bucket
    /// is merely a longer walk; two keys that match but bucket differently is a
    /// lookup that misses, which is why the invariant is stated here rather than
    /// left to each implementor to rediscover.
    fn bucket(key: &Self::Key<'_>) -> u64;
}

/// A process-global content-keyed cache, retained for the life of the process.
///
/// See the module header for why there is no capacity and no replacement rule.
pub(crate) struct ContentCache<E: CacheEntry> {
    /// Bucket (`CacheEntry::bucket`) → the entries filed under it. A bucket
    /// holds more than one entry only on a prefilter-hash collision.
    ///
    /// A `BTreeMap` rather than a `HashMap` for two reasons: its `new` is
    /// `const`, which is what lets the whole table live in the `const fn` the
    /// Metal caches are built from, and the key is already a well-mixed `u64`,
    /// so a handful of integer comparisons beats re-hashing it through
    /// `SipHash`.
    buckets: BTreeMap<u64, Vec<E>>,
}

impl<E: CacheEntry> ContentCache<E> {
    pub(crate) const fn new() -> Self {
        Self {
            buckets: BTreeMap::new(),
        }
    }

    pub(crate) fn find(&self, key: &E::Key<'_>) -> Option<&E> {
        self.buckets
            .get(&E::bucket(key))?
            .iter()
            .find(|e| e.matches(key))
    }

    /// Insert `entry`, unless one with its key arrived between the caller's
    /// [`find`](Self::find) and this call — the lock is released in between
    /// while the caller builds the GPU object, so it can.
    pub(crate) fn insert_unique(&mut self, entry: E) -> &E {
        let bucket = self
            .buckets
            .entry(E::bucket(&entry.lookup_key()))
            .or_default();
        let slot = match bucket.iter().position(|e| e.matches(&entry.lookup_key())) {
            Some(raced) => raced,
            None => {
                bucket.push(entry);
                bucket.len() - 1
            }
        };
        &bucket[slot]
    }

    /// Live entries across every bucket.
    ///
    /// This is the level the `object_cache_levels` census publishes, and the
    /// reading that can falsify the module header's argument: a count still
    /// climbing minutes into a boot means some key is carrying per-frame state
    /// rather than guest state.
    pub(crate) fn len(&self) -> usize {
        self.buckets.values().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stands in for the caches' real lookup keys: a content hash that picks the
    /// bucket, beside the bytes it was taken over, borrowed from the caller.
    #[derive(Clone, Copy)]
    struct ProbeKey<'a> {
        hash: u64,
        bytes: &'a [u8],
    }

    /// An entry with no GPU object in it, so the table itself can be driven
    /// without a device — which is what makes these tests runnable on a host
    /// that has no Metal.
    ///
    /// It retains its own copy of the bytes, exactly as the real entries do, so
    /// `matches` can answer from content rather than from the digest.
    struct Probe {
        hash: u64,
        bytes: Vec<u8>,
        tag: u32,
    }

    impl CacheEntry for Probe {
        type Key<'a> = ProbeKey<'a>;
        fn lookup_key(&self) -> ProbeKey<'_> {
            ProbeKey {
                hash: self.hash,
                bytes: &self.bytes,
            }
        }
        fn matches(&self, key: &ProbeKey<'_>) -> bool {
            self.hash == key.hash && self.bytes == key.bytes
        }
        /// Only the hash, exactly as the real keys do — the content is left to
        /// `matches`, so the collision test below shares one bucket.
        fn bucket(key: &ProbeKey<'_>) -> u64 {
            key.hash
        }
    }

    fn probe(hash: u64, tag: u32) -> Probe {
        Probe {
            hash,
            bytes: vec![0xab; 8],
            tag,
        }
    }

    fn ask(hash: u64, bytes: &[u8]) -> ProbeKey<'_> {
        ProbeKey { hash, bytes }
    }

    /// An insert that races another caller's insert must not add a second copy.
    ///
    /// The lock is released between a caller's `find` and its `insert_unique`,
    /// while it builds the GPU object, so two callers can miss the same key and
    /// both arrive here. Five of the six caches re-scanned for that; the
    /// reflection cache did not, and carried the duplicate.
    #[test]
    fn a_raced_insert_returns_the_entry_already_there() {
        let mut cache: ContentCache<Probe> = ContentCache::new();
        assert_eq!(cache.insert_unique(probe(1, 10)).tag, 10);
        assert_eq!(
            cache.insert_unique(probe(1, 20)).tag,
            10,
            "the loser of the race gets the winner's entry, not its own"
        );
        assert_eq!(
            cache.len(),
            1,
            "and the cache holds one copy of the key, not two"
        );
    }

    /// Nothing is ever displaced. The retired table held `cap` entries and then
    /// overwrote a rotating slot, so on the Metal arm a guest with more distinct
    /// pipeline states than the cap — which the Vulkan arm measures this guest to
    /// have — recompiled objects it was still drawing with. Drive well past every
    /// cap this container used to be built with (96, 64, 32, 16) and assert the
    /// first entry is still served.
    #[test]
    fn every_distinct_key_is_retained_past_every_retired_capacity() {
        let mut cache: ContentCache<Probe> = ContentCache::new();
        for i in 0..1024 {
            cache.insert_unique(probe(i, i as u32));
        }
        assert_eq!(cache.len(), 1024, "the table never displaces an entry");
        let bytes = vec![0xab; 8];
        assert_eq!(
            cache.find(&ask(0, &bytes)).map(|e| e.tag),
            Some(0),
            "the first object compiled is still there after 1023 later ones"
        );
        assert_eq!(cache.find(&ask(1023, &bytes)).map(|e| e.tag), Some(1023));
    }

    /// The content is the key and the digest only picks the bucket: two blobs
    /// whose hashes collide must not share one compiled object.
    ///
    /// This is the hazard the borrowed-key half of [`CacheEntry`] exists for. A
    /// natural 64-bit collision is not something a test can produce, so drive
    /// the state one would produce — two different blobs filed under one bucket
    /// — and ask through the real lookup. A `matches` that trusted the digest
    /// would return the first entry for the second blob's bytes.
    #[test]
    fn a_digest_collision_between_different_blobs_is_not_a_hit() {
        let mut cache: ContentCache<Probe> = ContentCache::new();
        let first = vec![0xab; 8];
        let second = vec![0xcd; 8];
        cache.insert_unique(probe(7, 70));
        assert!(cache.find(&ask(7, &first)).is_some());
        assert!(
            cache.find(&ask(7, &second)).is_none(),
            "a blob that only collides is a miss, not somebody else's object"
        );

        // Both live in bucket 7, and each still resolves to its own entry.
        cache.insert_unique(Probe {
            hash: 7,
            bytes: second.clone(),
            tag: 90,
        });
        assert_eq!(cache.find(&ask(7, &first)).map(|e| e.tag), Some(70));
        assert_eq!(cache.find(&ask(7, &second)).map(|e| e.tag), Some(90));
        assert_eq!(cache.len(), 2);
    }
}
