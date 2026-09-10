//! The hazard half of the dependency compiler: which earlier transactions a
//! newly admitted one must wait for, and why.
//!
//! # What this is and is not
//!
//! This compiles **hazard** edges — the ones that exist because two accesses
//! touch the same memory and at least one of them writes. It does not compile
//! explicit synchronisation. Events and fences can name a point that has not
//! been signalled yet, so they are unresolved prerequisites with their own
//! wait-for graph and their own cycle diagnostic; folding them in here would
//! destroy the one property that makes this graph trivially acyclic.
//!
//! That property is the reason transactions are admitted in ingress order and
//! never re-derived: **every edge this compiler creates points from a newer
//! ingress ordinal to an older one**, so ingress order is a topological order
//! for hazards. It is asserted rather than assumed, on every admission.
//!
//! # Imprecision costs a scan, and the census charges it
//!
//! Accesses are indexed by the thing they name — a backing, a heap — so
//! admitting one compares against the accesses that could possibly conflict
//! rather than against all of them. [`AccessKey::DomainOnly`] is the exception
//! by construction: an access whose participation is unknown could touch
//! anything, so it meets every live access in its domain and it makes every
//! later access meet it.
//!
//! That is the honest cost of rung three, and it is measured
//! ([`Census::domain_only_comparisons`]) rather than hidden, because the point
//! of measuring it is to know what narrowing an access would buy.

use crate::access::{requires_edge, AccessIntent, AccessKey, BackingId};
use crate::identity::{ChannelId, IngressOrdinal};
use std::collections::HashMap;

/// One live access, and the transaction that declared it.
#[derive(Clone, Copy, Debug)]
struct Entry {
    ordinal: IngressOrdinal,
    intent: AccessIntent,
    /// Cleared when the transaction retires. The slot stays so the indices
    /// pointing at it stay valid; compaction is [`DependencyGraph::compact`].
    live: bool,
}

/// What admitting transactions has cost so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    /// Accesses admitted.
    pub accesses: usize,
    /// Hazard edges created.
    pub edges: usize,
    /// Accesses admitted at each rung of the precision ladder, indexed by
    /// `rung - 1`.
    pub by_rung: [usize; 3],
    /// Comparisons that happened only because one side's participation was
    /// unknown. This is what rung three costs, and it is the number that says
    /// what narrowing an access would be worth.
    pub domain_only_comparisons: usize,
    /// Edges created by a comparison in which one side's direction was not
    /// established. Ordering bought with ignorance rather than with knowledge.
    ///
    /// A subset of [`Self::edges`] — never more — so `edges_from_unknown_mode`
    /// over `edges` is a fraction. It used to be charged per *comparison*, so
    /// two accesses of one transaction meeting one earlier ordinal counted
    /// twice against the one edge they made, and the fraction could exceed one.
    pub edges_from_unknown_mode: usize,
}

/// The live hazard state.
///
/// Holds only accesses whose transactions have not retired. Retiring is the
/// caller's obligation and is what keeps this bounded; nothing here evicts on
/// its own, because an eviction would silently drop an edge a later
/// transaction was owed.
#[derive(Debug, Default)]
pub struct DependencyGraph {
    entries: Vec<Entry>,
    /// Inactive entries still present in the candidate indexes.
    retired_entries: usize,
    /// Compaction must not erase the ingress-order contract.
    last_admitted: Option<IngressOrdinal>,
    by_backing: HashMap<BackingId, Vec<usize>>,
    /// Keyed by the heap, not by `HeapId`: the membership generation says
    /// which set a record was written against and never which memory exists, so
    /// indexing on it would file one heap's accesses in several buckets that
    /// never meet. `AccessKey::may_alias` asks the same question.
    by_heap: HashMap<u64, Vec<usize>>,
    by_domain: HashMap<ChannelId, Vec<usize>>,
    domain_only: HashMap<ChannelId, Vec<usize>>,
    by_ordinal: HashMap<IngressOrdinal, Vec<usize>>,
    census: Census,
    /// Buffers `admit` refills instead of allocating.
    ///
    /// The architecture plan's structural zeros include "heap allocations per
    /// steady-state draw", and a graph that built a fresh candidate list per
    /// access and a fresh tree node per edge made one trip into the allocator
    /// per access and one per distinct wait — the cost that spreads evenly
    /// across a profile and so is never attributed to anything. Both are the
    /// same shape every time, so both are kept and reused. The one allocation
    /// left on the path is the wait list `admit` hands back, which its
    /// signature owns.
    scratch: Vec<usize>,
    waits: Vec<IngressOrdinal>,
}

impl DependencyGraph {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    /// Live accesses, for a test or a report. Not a bound anything enforces.
    #[must_use]
    pub fn live_accesses(&self) -> usize {
        self.entries.iter().filter(|e| e.live).count()
    }

    /// Admit one transaction's accesses and return the ordinals it must wait
    /// for.
    ///
    /// The returned set is deduplicated and sorted, because a caller that has
    /// to deduplicate is a caller that will forget to, and because two accesses
    /// of one transaction conflicting with one earlier transaction is one edge
    /// and not two.
    ///
    /// # Panics
    ///
    /// If `ordinal` is not greater than every ordinal already admitted. That is
    /// the ingress-order contract, and violating it would silently produce an
    /// edge pointing forward — which is the one thing this graph promises
    /// cannot happen.
    pub fn admit(
        &mut self,
        ordinal: IngressOrdinal,
        accesses: &[AccessIntent],
    ) -> Vec<IngressOrdinal> {
        assert!(
            self.last_admitted.is_none_or(|last| ordinal > last),
            "transactions are admitted in ingress order; {ordinal:?} arrived after a later one"
        );
        self.last_admitted = Some(ordinal);
        // Rebuild only after enough retirements to amortize the index walk.
        // The threshold controls cleanup cost, never which live accesses survive.
        const MIN_RETIRED_FOR_COMPACTION: usize = 1024;
        if self.retired_entries >= MIN_RETIRED_FOR_COMPACTION
            && self.retired_entries >= self.entries.len() - self.retired_entries
        {
            self.compact();
        }
        // Taken out so the gathering below can borrow the indexes; put back
        // before returning, so the next admission finds the capacity this one
        // grew. Both are cleared here rather than at the end, because a
        // panicking assertion above must not leave stale contents behind.
        let mut waits = std::mem::take(&mut self.waits);
        let mut scratch = std::mem::take(&mut self.scratch);
        waits.clear();
        for intent in accesses {
            scratch.clear();
            self.gather(intent, &mut scratch);
            for &candidate in &scratch {
                let entry = self.entries[candidate];
                if !entry.live || entry.ordinal == ordinal {
                    continue;
                }
                if !requires_edge(&entry.intent, intent) {
                    continue;
                }
                // Kept sorted by insertion, which is both the deduplication
                // and the order the answer is owed in. A set would spell the
                // same thing with an allocation per member.
                let Err(at) = waits.binary_search(&entry.ordinal) else {
                    continue;
                };
                waits.insert(at, entry.ordinal);
                self.census.edges += 1;
                // Charged where the edge is created, not at every
                // comparison that would have created it. Two accesses of
                // one transaction meeting one earlier ordinal are one edge,
                // and counting the second comparison here made a fraction
                // of edges that could exceed one.
                if entry.intent.mode == crate::access::AccessMode::Unknown
                    || intent.mode == crate::access::AccessMode::Unknown
                {
                    self.census.edges_from_unknown_mode += 1;
                }
            }
            self.insert(ordinal, *intent);
        }
        self.scratch = scratch;
        let out = waits.clone();
        self.waits = waits;
        out
    }

    /// The entries an access could possibly conflict with, appended to `out`.
    ///
    /// Indices rather than entries so the borrow ends before `insert` runs, and
    /// deliberately allowed to contain duplicates: a resource that belongs to a
    /// heap is reachable two ways, and the caller already deduplicates by
    /// ordinal.
    ///
    /// Fills a buffer the caller owns rather than returning one, so a warm
    /// graph makes no trip into the allocator per access. See [`Self::scratch`].
    fn gather(&mut self, intent: &AccessIntent, out: &mut Vec<usize>) {
        // Every access meets the domain's unknown-participation accesses.
        if let Some(v) = self.domain_only.get(&intent.domain) {
            self.census.domain_only_comparisons += v.len();
            out.extend_from_slice(v);
        }
        match intent.key {
            AccessKey::DomainOnly => {
                // And an unknown-participation access meets everything in its
                // domain. Both halves are charged to the same counter.
                if let Some(v) = self.by_domain.get(&intent.domain) {
                    self.census.domain_only_comparisons += v.len();
                    out.extend_from_slice(v);
                }
            }
            AccessKey::Heap(h) => {
                if let Some(v) = self.by_heap.get(&h.id) {
                    out.extend_from_slice(v);
                }
            }
            AccessKey::Range(r, _) | AccessKey::Subresource(r, _) | AccessKey::Whole(r) => {
                if let Some(v) = self.by_backing.get(&r.backing) {
                    out.extend_from_slice(v);
                }
                if let Some(h) = r.heap {
                    if let Some(v) = self.by_heap.get(&h.id) {
                        out.extend_from_slice(v);
                    }
                }
            }
        }
    }

    fn insert(&mut self, ordinal: IngressOrdinal, intent: AccessIntent) {
        let idx = self.entries.len();
        self.entries.push(Entry {
            ordinal,
            intent,
            live: true,
        });
        self.census.accesses += 1;
        self.census.by_rung[usize::from(intent.key.rung()) - 1] += 1;
        self.by_domain.entry(intent.domain).or_default().push(idx);
        self.by_ordinal.entry(ordinal).or_default().push(idx);
        match intent.key {
            AccessKey::DomainOnly => self.domain_only.entry(intent.domain).or_default().push(idx),
            AccessKey::Heap(h) => self.by_heap.entry(h.id).or_default().push(idx),
            AccessKey::Range(r, _) | AccessKey::Subresource(r, _) | AccessKey::Whole(r) => {
                self.by_backing.entry(r.backing).or_default().push(idx);
                // Also under its heap, so a heap declaration can find it: a
                // heap-use record names the heap and never its members, so
                // membership has to be reachable from both directions.
                if let Some(h) = r.heap {
                    self.by_heap.entry(h.id).or_default().push(idx);
                }
            }
        }
    }

    /// Mark one transaction's accesses as no longer live.
    ///
    /// Retirement is a completion fact, not a planning one: an access stops
    /// creating edges when the work that declared it has finished, and a caller
    /// that retires early publishes a hazard it still owes.
    pub fn retire(&mut self, ordinal: IngressOrdinal) {
        if let Some(indices) = self.by_ordinal.remove(&ordinal) {
            self.retired_entries += indices.len();
            for idx in indices {
                self.entries[idx].live = false;
            }
        }
    }

    /// Drop retired entries and rebuild the indexes.
    ///
    /// Separate from [`Self::retire`] because retirement is on the completion
    /// path and this is not: an index rebuild in a completion handler is work
    /// charged to the thing that finished rather than to the thing that grew.
    pub fn compact(&mut self) {
        let live: Vec<_> = self.entries.iter().copied().filter(|e| e.live).collect();
        self.entries.clear();
        self.retired_entries = 0;
        self.by_backing.clear();
        self.by_heap.clear();
        self.by_domain.clear();
        self.domain_only.clear();
        self.by_ordinal.clear();
        // The census is a running total across the graph's life and is not
        // rebuilt: compaction is bookkeeping, and it did not admit anything.
        let saved = self.census;
        for e in live {
            self.insert(e.ordinal, e.intent);
        }
        self.census = saved;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{AccessMode, ByteRange, HeapId, ResourceKey, SubresourceRange};
    use std::collections::BTreeSet;

    fn res(backing: u64) -> ResourceKey {
        ResourceKey {
            backing: BackingId(backing),
            heap: None,
        }
    }

    fn intent(key: AccessKey, mode: AccessMode) -> AccessIntent {
        AccessIntent {
            domain: ChannelId(1),
            key,
            mode,
            api_stages: 0,
            input_content_version: None,
            output_content_version: None,
        }
    }

    fn ord(n: u64) -> IngressOrdinal {
        IngressOrdinal(n)
    }

    /// The property the whole graph rests on.
    #[test]
    fn every_edge_points_backwards() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        for n in 1..=8 {
            let mode = if n % 2 == 0 {
                AccessMode::Write
            } else {
                AccessMode::Read
            };
            for w in g.admit(ord(n), &[intent(k, mode)]) {
                assert!(w < ord(n), "edge {w:?} -> {:?} points forward", ord(n));
            }
        }
    }

    #[test]
    fn a_read_waits_for_the_preceding_writer_and_not_for_readers() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        assert!(g.admit(ord(1), &[intent(k, AccessMode::Write)]).is_empty());
        assert_eq!(
            g.admit(ord(2), &[intent(k, AccessMode::Read)]),
            vec![ord(1)]
        );
        assert_eq!(
            g.admit(ord(3), &[intent(k, AccessMode::Read)]),
            vec![ord(1)],
            "a second reader waits for the writer and not for the first reader"
        );
    }

    #[test]
    fn a_write_waits_for_the_preceding_writer_and_every_preceding_reader() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        g.admit(ord(1), &[intent(k, AccessMode::Write)]);
        g.admit(ord(2), &[intent(k, AccessMode::Read)]);
        g.admit(ord(3), &[intent(k, AccessMode::Read)]);
        assert_eq!(
            g.admit(ord(4), &[intent(k, AccessMode::Write)]),
            vec![ord(1), ord(2), ord(3)]
        );
    }

    #[test]
    fn disjoint_ranges_over_one_backing_do_not_meet() {
        let mut g = DependencyGraph::new();
        let lo = AccessKey::Range(
            res(1),
            ByteRange {
                offset: 0,
                length: 64,
            },
        );
        let hi = AccessKey::Range(
            res(1),
            ByteRange {
                offset: 64,
                length: 64,
            },
        );
        g.admit(ord(1), &[intent(lo, AccessMode::Write)]);
        assert!(
            g.admit(ord(2), &[intent(hi, AccessMode::Write)]).is_empty(),
            "two writes to disjoint halves of one buffer are independent"
        );
        assert_eq!(
            g.admit(
                ord(3),
                &[intent(AccessKey::Whole(res(1)), AccessMode::Read)]
            ),
            vec![ord(1), ord(2)],
            "and a whole-backing read meets both of them"
        );
    }

    #[test]
    fn a_heap_declaration_meets_its_members_in_both_directions() {
        let heap = HeapId {
            id: 3,
            membership_generation: 1,
        };
        let member = AccessKey::Whole(ResourceKey {
            backing: BackingId(9),
            heap: Some(heap),
        });
        // Member first, then the heap declaration.
        let mut g = DependencyGraph::new();
        g.admit(ord(1), &[intent(member, AccessMode::Write)]);
        assert_eq!(
            g.admit(ord(2), &[intent(AccessKey::Heap(heap), AccessMode::Read)]),
            vec![ord(1)]
        );
        // Heap declaration first, then the member.
        let mut g = DependencyGraph::new();
        g.admit(ord(1), &[intent(AccessKey::Heap(heap), AccessMode::Write)]);
        assert_eq!(
            g.admit(ord(2), &[intent(member, AccessMode::Read)]),
            vec![ord(1)]
        );
    }

    /// The graph's heap index has to answer the same question
    /// `AccessKey::may_alias` does. Filing accesses under `(heap, membership)`
    /// puts a declaration written before a placement and a member access
    /// written after it in two buckets that never meet, and the write/read
    /// hazard between them disappears — silently, and only when the guest
    /// happens to allocate from the heap in between.
    #[test]
    fn a_placement_between_two_uses_does_not_dissolve_the_heap_edge() {
        let at = |generation| HeapId {
            id: 3,
            membership_generation: generation,
        };
        let member = |generation| {
            AccessKey::Whole(ResourceKey {
                backing: BackingId(9),
                heap: Some(at(generation)),
            })
        };
        let mut g = DependencyGraph::new();
        g.admit(ord(1), &[intent(AccessKey::Heap(at(1)), AccessMode::Write)]);
        assert_eq!(
            g.admit(ord(2), &[intent(member(2), AccessMode::Read)]),
            vec![ord(1)],
            "the resource did not move when something else was placed beside it"
        );
        assert_eq!(
            g.admit(ord(3), &[intent(AccessKey::Heap(at(2)), AccessMode::Read)]),
            vec![ord(1)],
            "and the later declaration still meets the earlier one's write"
        );
    }

    #[test]
    fn unknown_participation_meets_everything_in_its_domain_both_ways() {
        let mut g = DependencyGraph::new();
        g.admit(
            ord(1),
            &[intent(AccessKey::Whole(res(1)), AccessMode::Write)],
        );
        g.admit(
            ord(2),
            &[intent(AccessKey::Whole(res(2)), AccessMode::Write)],
        );
        assert_eq!(
            g.admit(ord(3), &[intent(AccessKey::DomainOnly, AccessMode::Read)]),
            vec![ord(1), ord(2)],
            "an access that could touch anything waits for every writer"
        );
        assert_eq!(
            g.admit(
                ord(4),
                &[intent(AccessKey::Whole(res(3)), AccessMode::Write)]
            ),
            vec![ord(3)],
            "and a later access to an unrelated backing waits for it"
        );
        assert!(g.census().domain_only_comparisons > 0);
    }

    #[test]
    fn separate_domains_produce_no_hazard_edge() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        let mut a = intent(k, AccessMode::Write);
        a.domain = ChannelId(1);
        let mut b = intent(k, AccessMode::Read);
        b.domain = ChannelId(2);
        g.admit(ord(1), &[a]);
        assert!(
            g.admit(ord(2), &[b]).is_empty(),
            "the contract leaves separate submission domains unordered"
        );
    }

    #[test]
    fn a_retired_transaction_stops_creating_edges() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        g.admit(ord(1), &[intent(k, AccessMode::Write)]);
        g.retire(ord(1));
        assert!(g.admit(ord(2), &[intent(k, AccessMode::Read)]).is_empty());
        assert_eq!(g.live_accesses(), 1);
        g.compact();
        assert_eq!(g.live_accesses(), 1, "only the live access survives");
    }

    /// Compaction is bookkeeping. It must not change an answer, and it must not
    /// rewrite the running totals — a census that reset on compaction would
    /// under-report exactly when the graph was busiest.
    #[test]
    fn compaction_changes_no_answer_and_no_total() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        g.admit(ord(1), &[intent(k, AccessMode::Write)]);
        g.admit(ord(2), &[intent(k, AccessMode::Read)]);
        g.retire(ord(1));
        let before = g.census();
        g.compact();
        assert_eq!(g.census(), before);
        assert_eq!(
            g.admit(ord(3), &[intent(k, AccessMode::Write)]),
            vec![ord(2)]
        );
    }

    /// Ordering bought with ignorance is counted apart from ordering bought
    /// with knowledge, so a census can say which one a workload is paying.
    #[test]
    fn an_unknown_direction_is_charged_to_its_own_counter() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        g.admit(ord(1), &[intent(k, AccessMode::Read)]);
        assert_eq!(
            g.admit(ord(2), &[intent(k, AccessMode::Unknown)]),
            vec![ord(1)]
        );
        assert_eq!(g.census().edges, 1);
        assert_eq!(g.census().edges_from_unknown_mode, 1);
    }

    #[test]
    fn subresource_windows_that_do_not_overlap_are_independent() {
        let win = |base_level| {
            AccessKey::Subresource(
                res(1),
                SubresourceRange {
                    base_level,
                    level_count: 1,
                    base_slice: 0,
                    slice_count: 1,
                    plane: 0,
                },
            )
        };
        let mut g = DependencyGraph::new();
        g.admit(ord(1), &[intent(win(0), AccessMode::Write)]);
        assert!(g
            .admit(ord(2), &[intent(win(1), AccessMode::Write)])
            .is_empty());
        assert_eq!(
            g.admit(ord(3), &[intent(win(0), AccessMode::Read)]),
            vec![ord(1)]
        );
    }

    #[test]
    #[should_panic(expected = "admitted in ingress order")]
    fn admitting_out_of_order_is_a_contract_violation_and_not_a_reordering() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        g.admit(ord(2), &[intent(k, AccessMode::Write)]);
        g.admit(ord(1), &[intent(k, AccessMode::Read)]);
    }

    /// Two accesses of one transaction meeting one earlier transaction is one
    /// edge. A caller that had to deduplicate would eventually not.
    #[test]
    fn one_earlier_transaction_produces_one_edge_however_many_accesses_meet_it() {
        let mut g = DependencyGraph::new();
        g.admit(
            ord(1),
            &[intent(AccessKey::Whole(res(1)), AccessMode::Write)],
        );
        let waits = g.admit(
            ord(2),
            &[
                intent(AccessKey::Whole(res(1)), AccessMode::Read),
                intent(
                    AccessKey::Range(
                        res(1),
                        ByteRange {
                            offset: 0,
                            length: 4,
                        },
                    ),
                    AccessMode::Read,
                ),
            ],
        );
        assert_eq!(waits, vec![ord(1)]);
        assert_eq!(g.census().edges, 1);
    }

    /// Two accesses of the *same* transaction never order against each other:
    /// a transaction is one unit, and an intra-transaction edge would be a
    /// transaction waiting for itself.
    #[test]
    fn a_transaction_never_waits_for_itself() {
        let mut g = DependencyGraph::new();
        let k = AccessKey::Whole(res(1));
        let waits = g.admit(
            ord(1),
            &[intent(k, AccessMode::Write), intent(k, AccessMode::Read)],
        );
        assert!(waits.is_empty());
    }
    /// **What a census charges must be a subset of what it charges into.**
    ///
    /// `edges_from_unknown_mode` over `edges` is meant to read as "how much of
    /// this ordering was bought with ignorance". It used to be charged at every
    /// comparison that *would* have made an edge rather than at the one that
    /// did, so two accesses of one transaction meeting one earlier ordinal
    /// counted twice against the single edge they produced and the fraction
    /// could exceed one.
    #[test]
    fn ordering_bought_with_ignorance_is_a_fraction_of_the_ordering() {
        let mut g = DependencyGraph::new();
        let key = AccessKey::Whole(ResourceKey {
            backing: BackingId(1),
            heap: None,
        });
        let unknown = AccessIntent {
            domain: ChannelId(1),
            key,
            mode: AccessMode::Unknown,
            api_stages: 0,
            input_content_version: None,
            output_content_version: None,
        };
        g.admit(IngressOrdinal(1), &[unknown]);
        // Two accesses of one transaction, both conflicting with ordinal 1.
        assert_eq!(
            g.admit(IngressOrdinal(2), &[unknown, unknown]),
            vec![IngressOrdinal(1)],
            "one earlier transaction is one edge"
        );
        let c = g.census();
        assert_eq!(c.edges, 1);
        assert_eq!(c.edges_from_unknown_mode, 1, "and one edge bought with it");
    }

    struct Rng(u64);

    impl Rng {
        const fn new(seed: u64) -> Self {
            Self(seed ^ 0x9E37_79B9_7F4A_7C15)
        }

        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, bound: u64) -> u64 {
            if bound == 0 {
                return 0;
            }
            self.next() % bound
        }
    }

    /// Deliberately tiny pools, so backings, heaps and domains are shared
    /// constantly and every key shape meets every other.
    const BACKINGS: u64 = 3;
    const HEAPS: u64 = 2;
    const CHANNELS: u64 = 2;

    fn some_intent(rng: &mut Rng) -> AccessIntent {
        let backing = BackingId(rng.below(BACKINGS));
        let heap = (rng.below(3) != 0).then(|| HeapId {
            id: rng.below(HEAPS),
            // Varied, because a membership generation must not split a heap's
            // accesses into buckets that never meet.
            membership_generation: rng.below(3),
        });
        let resource = ResourceKey { backing, heap };
        let key = match rng.below(5) {
            0 => AccessKey::Range(
                resource,
                ByteRange {
                    offset: rng.below(4) * 16,
                    length: rng.below(3) * 16 + 16,
                },
            ),
            1 => AccessKey::Whole(resource),
            2 => AccessKey::Heap(HeapId {
                id: rng.below(HEAPS),
                membership_generation: rng.below(3),
            }),
            3 => AccessKey::DomainOnly,
            _ => AccessKey::Whole(resource),
        };
        let mode = match rng.below(4) {
            0 => AccessMode::Read,
            1 => AccessMode::Write,
            2 => AccessMode::ReadWrite,
            _ => AccessMode::Unknown,
        };
        AccessIntent {
            domain: ChannelId(rng.below(CHANNELS) as u32),
            key,
            mode,
            api_stages: 0,
            input_content_version: None,
            output_content_version: None,
        }
    }

    /// **Every edge points backwards, the indexes find exactly what an
    /// all-pairs scan would, and compacting changes no answer.**
    ///
    /// Three shadows in one, and each is deliberately dumber than the thing it
    /// checks. The first is a flat list of live accesses compared pairwise with
    /// [`requires_edge`] — no index at all — so an index that files an access
    /// under the wrong bucket, or under one that never meets its counterpart,
    /// disagrees. The second is the census, recomputed as counts of what the
    /// first one saw. The third is a second graph, compacted at random points:
    /// compaction rebuilds every index from the live entries, so a rebuilt
    /// index that answers differently from an accumulated one is the same
    /// defect seen from the other side.
    #[test]
    fn the_indexes_find_what_an_all_pairs_scan_would_and_compaction_changes_nothing() {
        let mut edges = 0usize;
        let mut accesses = 0usize;
        let mut retirements = 0usize;
        let mut compactions = 0usize;
        let mut transactions_with_waits = 0usize;

        for seed in 0..384u64 {
            let mut rng = Rng::new(seed);
            let mut g = DependencyGraph::new();
            let mut compacted = DependencyGraph::new();
            // Shadow: every access ever admitted, and whether it is still live.
            let mut shadow: Vec<(IngressOrdinal, AccessIntent, bool)> = Vec::new();
            let mut live_ordinals: Vec<IngressOrdinal> = Vec::new();
            let mut ordinal = 0u64;

            for _ in 0..40 {
                if rng.below(6) == 0 && !live_ordinals.is_empty() {
                    // Retire a transaction: its accesses stop creating edges.
                    let i = rng.below(live_ordinals.len() as u64) as usize;
                    let retired = live_ordinals.swap_remove(i);
                    g.retire(retired);
                    compacted.retire(retired);
                    for e in &mut shadow {
                        if e.0 == retired {
                            e.2 = false;
                        }
                    }
                    retirements += 1;
                } else if rng.below(8) == 0 {
                    compacted.compact();
                    compactions += 1;
                } else {
                    ordinal += 1 + rng.below(2);
                    let at = IngressOrdinal(ordinal);
                    let intents: Vec<AccessIntent> = (0..rng.below(3) + 1)
                        .map(|_| some_intent(&mut rng))
                        .collect();

                    // All-pairs, in the same order the module inserts, because
                    // an access of this transaction is visible to its own later
                    // accesses and never produces an edge against itself.
                    let mut expected: BTreeSet<IngressOrdinal> = BTreeSet::new();
                    for intent in &intents {
                        for (o, other, live) in &shadow {
                            if *live && *o != at && requires_edge(other, intent) {
                                expected.insert(*o);
                            }
                        }
                        shadow.push((at, *intent, true));
                    }
                    let expected: Vec<IngressOrdinal> = expected.into_iter().collect();

                    let got = g.admit(at, &intents);
                    assert_eq!(got, expected, "seed {seed}: waits for {at:?}");
                    assert_eq!(
                        compacted.admit(at, &intents),
                        expected,
                        "seed {seed}: a compacted graph answered differently"
                    );
                    assert!(
                        got.iter().all(|w| *w < at),
                        "seed {seed}: an edge points forward"
                    );
                    if !got.is_empty() {
                        transactions_with_waits += 1;
                    }
                    edges += got.len();
                    accesses += intents.len();
                    live_ordinals.push(at);
                }

                // The observers agree with the shadow after every step.
                assert_eq!(
                    g.live_accesses(),
                    shadow.iter().filter(|(_, _, live)| *live).count(),
                    "seed {seed}: live_accesses"
                );
                assert_eq!(
                    compacted.live_accesses(),
                    g.live_accesses(),
                    "seed {seed}: compaction changed how much is live"
                );
                let c = g.census();
                assert_eq!(c.accesses, shadow.len(), "seed {seed}: accesses admitted");
                assert_eq!(
                    c.by_rung.iter().sum::<usize>(),
                    c.accesses,
                    "seed {seed}: the rungs are a partition of the accesses"
                );
                for rung in 1..=3u8 {
                    assert_eq!(
                        c.by_rung[usize::from(rung) - 1],
                        shadow
                            .iter()
                            .filter(|(_, i, _)| i.key.rung() == rung)
                            .count(),
                        "seed {seed}: rung {rung}"
                    );
                }
                assert!(
                    c.edges_from_unknown_mode <= c.edges,
                    "seed {seed}: {} of {} edges bought with ignorance",
                    c.edges_from_unknown_mode,
                    c.edges
                );
                assert_eq!(
                    compacted.census().accesses,
                    c.accesses,
                    "seed {seed}: compaction lost the running total"
                );
            }
        }

        // Non-vacuity: every shape an assertion above depends on reaching.
        assert!(accesses > 8_000, "accesses admitted: {accesses}");
        assert!(edges > 8_000, "hazard edges: {edges}");
        assert!(
            transactions_with_waits > 3_000,
            "transactions that had to wait: {transactions_with_waits}"
        );
        assert!(retirements > 1_000, "transactions retired: {retirements}");
        assert!(compactions > 500, "compactions: {compactions}");
    }
    #[test]
    fn automatic_compaction_bounds_retired_history_and_preserves_live_writer() {
        let mut g = DependencyGraph::new();
        let key = AccessKey::Whole(res(1));
        assert!(g
            .admit(ord(1), &[intent(key, AccessMode::Write)])
            .is_empty());
        for n in 2..=20_000 {
            assert_eq!(
                g.admit(ord(n), &[intent(key, AccessMode::Read)]),
                vec![ord(1)]
            );
            g.retire(ord(n));
            g.retire(ord(n)); // Repeated retirement must not inflate accounting.
            assert_eq!(g.live_accesses(), 1);
            assert!(
                g.entries.len() <= 2048,
                "retired history keeps growing: {}",
                g.entries.len()
            );
        }
        assert_eq!(g.census().accesses, 20_000);
        assert_eq!(g.census().edges, 19_999);
        g.retire(ord(1));
        assert!(g
            .admit(ord(20_001), &[intent(key, AccessMode::Write)])
            .is_empty());
    }

    #[test]
    fn compaction_keeps_all_live_dependencies_even_when_the_live_set_is_large() {
        let mut g = DependencyGraph::new();
        let key = AccessKey::Whole(res(1));
        for n in 1..=4096 {
            assert!(g.admit(ord(n), &[intent(key, AccessMode::Read)]).is_empty());
        }
        for n in 1..=2048 {
            g.retire(ord(n));
        }
        assert_eq!(
            g.admit(ord(4097), &[intent(key, AccessMode::Write)]),
            (2049..=4096).map(ord).collect::<Vec<_>>()
        );
        assert_eq!(g.live_accesses(), 2049);
    }

    #[test]
    #[should_panic(expected = "transactions are admitted in ingress order")]
    fn compaction_cannot_erase_the_ingress_order_high_water_mark() {
        let mut g = DependencyGraph::new();
        let a = intent(AccessKey::Whole(res(1)), AccessMode::Write);
        g.admit(ord(10), &[a]);
        g.retire(ord(10));
        g.compact();
        g.admit(ord(9), &[a]);
    }
}
