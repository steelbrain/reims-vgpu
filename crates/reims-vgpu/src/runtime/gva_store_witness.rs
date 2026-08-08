//! What can be said about a GVA render target's guest pages since the Store
//! that published them.
//!
//! # Why this exists
//!
//! A type-11 render Store publishes into a mapping, and the device can ask
//! whether anything has written that mapping since — the token lives on
//! [`crate::model::MappingEntry`] and
//! [`crate::runtime::mapper::mapping_guest_write_verdict`] reads it. That
//! witness is what licenses the type-11 attachment LOAD elision, which serves a
//! `MTLLoadActionLoad` straight out of the engine resident instead of reading a
//! whole frame back out of guest memory: one driven boot elided **11 484** seeds
//! against 94 uploaded.
//!
//! A **GVA** (type-2/3) render target has no mapping and so had no witness, and
//! its LOAD reads the attachment's full span out of guest pages every time —
//! `load_seed_ok_color` 4 949 per driven Safari-drag boot, blocking on a
//! guest-write settle for 2 923 of them, because those are exactly the pages the
//! previous Store has just written. This is the missing half.
//!
//! # Two writers, two questions, and neither may be dropped
//!
//! [`GvaWriteReach`] answers both:
//!
//! * **The guest CPU** wrote the span. Only the hypervisor's dirty bitmap can
//!   see this, through a token registered over the target's pages.
//! * **This device** wrote the span through some other rail — a compute
//!   writeback, a blit, a type-11 write whose pages alias it. The hypervisor
//!   cannot see those; [`crate::runtime::host_writes`] is the record that can,
//!   and aliasing across id namespaces is real rather than theoretical (see that
//!   module's own doc).
//!
//! What this module does **not** answer is whether a draw has landed in the
//! resident since the Store. That is the engine's `content_epoch`, and the
//! caller compares it, because only the caller knows which resident it means.
//!
//! # Why the key is the whole identity
//!
//! A token registered over one page list says nothing about another, and the
//! guest recycles GVAs. The type-11 twin handles this with an explicit
//! `guest_write_token_gen != map_generation` check that a future writer has to
//! remember. Here the page-set hash *is part of the key*
//! (`draw::vulkan::gva_span_alloc_generation`, the same value the engine
//! registry keys the resident on), so a target whose page list has moved cannot
//! be found under the key its token was armed under. The wrong answer is
//! unreachable rather than guarded; the stale entry becomes an orphan, which is
//! a resource cost the eviction below collects.
//!
//! # Why not [`crate::runtime::gather_witness`]
//!
//! It already owns a bounded, token-holding map keyed by `GatherKey::TaskGva`,
//! and reusing it was the obvious move. Two things stop it, and the first is a
//! live hazard rather than an inelegance. Its entry carries the
//! sampled-content generation the engine binds retained images on, and it keys
//! on `(task_id, gva)` with the *sampled* span — which for a compositing layer
//! that is both rendered into and sampled is the same key at a different span,
//! so the two rails would re-point each other's entries every frame and spend
//! each other's generations. Second, its stamp is overwritten on every bind, so
//! "since the previous gather" cannot express "since the Store".
//!
//! What is shared instead is everything with no such coupling: the
//! [`GuestWriteVerdict`] spelling, the release rail through
//! `DeviceState::retired_guest_write_tokens`, and the eviction shape.
//!
//! # The rule every consumer must obey
//!
//! Everything undecidable answers as written. A host with no dirty bitmap, a
//! token still inside its arming window, a target no Store has stamped, a
//! host-write record that has aged out — all refuse, and a refusal costs a
//! re-read rather than a wrong frame. The one answer that must never be
//! invented is [`GvaWriteReach::Quiet`].

use crate::model::DeviceState;
use crate::runtime::host::HostOps;
use crate::runtime::host_writes::HostWriteVerdict;
use crate::runtime::mapper::GuestWriteVerdict;
use std::collections::BTreeMap;

/// Which GVA render target a witness entry is about: the engine registry's own
/// identity for it, spelled without the backend type so `DeviceState` stays
/// backend-neutral.
///
/// [`GvaTargetKey::of`] is the only constructor that a product path may use, so
/// the two spellings cannot drift into naming different targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct GvaTargetKey {
    pub gva: u64,
    /// Hash of the resolved guest page set — `gva_span_alloc_generation`. Part
    /// of the key, which is what makes a moved page list a different target
    /// rather than a stale answer. Never 0 for a usable key: 0 is that
    /// function's "the walk was short".
    pub generation: u64,
    pub width: u32,
    pub height: u32,
    pub bgra: bool,
}

impl GvaTargetKey {
    /// The key for a `TargetIdentity::Gva`, or `None` for any other identity
    /// kind or an unusable generation.
    ///
    /// The task is deliberately not in the key. Two tasks colliding here would
    /// need identical resolved page sets at the same address and extent, which
    /// is the same physical memory — the same target by every test this device
    /// applies to it.
    #[cfg(feature = "backend-vulkan")]
    pub fn of(identity: &crate::backend::vulkan::engine::TargetIdentity) -> Option<Self> {
        match *identity {
            crate::backend::vulkan::engine::TargetIdentity::Gva {
                gva,
                width,
                height,
                generation,
                bgra,
            } if generation != 0 && gva != 0 => Some(Self {
                gva,
                generation,
                width,
                height,
                bgra,
            }),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct Entry {
    /// Host tracking token over this target's pages. Never 0: an entry is only
    /// inserted once the host has armed one.
    token: u64,
    /// The pages the token watches, kept so the host-write half can be asked
    /// page-exactly rather than device-globally.
    gpas: Vec<u64>,
    /// `guest_write_gen` as it stood at the Store. 0 means "no usable stamp" —
    /// the arming window — and can never equal a live generation, because the
    /// host's first readable one is 1.
    gen_at_store: u64,
    /// `host_writes.epoch()` as it stood at the Store, read *after* the Store
    /// recorded its own write so the Store does not invalidate itself.
    host_epoch_at_store: u64,
    /// Store ordinal, for the eviction order.
    last_seen: u64,
}

/// Per-device witness state: one entry per GVA render target stamped.
#[derive(Default, Debug)]
pub struct GvaStoreWitness {
    entries: BTreeMap<GvaTargetKey, Entry>,
    stores: u64,
}

/// Upper bound on tracked GVA targets.
///
/// A hypervisor harvest bound and not a memory one, for the same reason
/// `gather_witness::MAX_TRACKED_WINDOWS` is and out of the same budget: the shim
/// walks every page of every tracked set on the BQL thread at each register
/// write that hands the device work, so each armed target adds its page count to
/// a cost the whole VM pays. The two constants are halves of one budget and the
/// basis is stated in both.
///
/// The population is the compositing layers a desktop keeps live at once, which
/// is the same order as the sampled working set that sized the other bound.
/// Overflow evicts the least recently stored target rather than dropping the
/// map, so a working set past this degrades one target at a time instead of
/// un-arming every live one at once. `gvaw_evict` counts it, so a boot says
/// whether the bound binds rather than leaving it assumed.
const MAX_TRACKED_TARGETS: usize = 128;

impl GvaStoreWitness {
    /// Detach every tracking token, for release through
    /// [`HostOps::untrack_guest_writes`].
    ///
    /// Returns them rather than releasing them because this type has no
    /// `HostOps`, exactly as `gather_witness::GatherWitness::take_tokens` does;
    /// the crate's one rail for host state it cannot free itself is
    /// `DeviceState::retired_guest_write_tokens`.
    pub fn take_tokens(&mut self) -> Vec<u64> {
        let tokens = self.entries.values().map(|e| e.token).collect();
        self.entries.clear();
        tokens
    }

    /// How many targets are armed. For the tests that bound the map.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn evict_oldest(&mut self, retired: &mut Vec<u64>) {
        let Some(victim) = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_seen)
            .map(|(k, _)| *k)
        else {
            return;
        };
        if let Some(e) = self.entries.remove(&victim) {
            crate::runtime::drain::note_store_route("gvaw_evict");
            retired.push(e.token);
        }
    }
}

/// Record what the two witnesses say at the moment a Store publishes this
/// target, registering its pages for tracking the first time.
///
/// `gpas` must be the target's **complete** page list — the same walk the Store
/// built its destination runs from. A partial list would have the host watch
/// part of the span and report unwritten for the rest, which is the one answer
/// that must never be invented; `store_gva_frame` refuses a short walk before it
/// reaches here, and this refuses an empty one.
///
/// **Call this after the Store has recorded its own write** through
/// `DeviceState::note_host_wrote_pages`. The host-write epoch captured here is
/// what [`reach`] compares against, and capturing it first would have every
/// target permanently invalidated by its own Store.
///
/// The twin of [`crate::runtime::mapper::stamp_guest_write_gen`], with counters
/// named apart on purpose: a rail whose registration silently never happens is
/// indistinguishable from a guest that writes every frame, and the two want
/// opposite fixes.
pub fn note_store<H: HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    key: GvaTargetKey,
    gpas: &[u64],
) {
    if key.generation == 0 || gpas.is_empty() {
        crate::runtime::drain::note_store_route("gvaw_unnamed_alloc");
        return;
    }
    let page_size = state.page_size() as usize;
    // Re-arming an existing key means the same page list by construction — the
    // hash is in the key — so the token is reused and only the stamps move.
    let token = match state.gva_store_witness.entries.get(&key) {
        Some(e) => e.token,
        None => {
            while state.gva_store_witness.entries.len() >= MAX_TRACKED_TARGETS {
                let mut retired = Vec::new();
                state.gva_store_witness.evict_oldest(&mut retired);
                state.retired_guest_write_tokens.extend(retired);
            }
            match host.track_guest_writes(gpas, page_size) {
                Some(t) => t,
                None => {
                    // No dirty bitmap, or the host declined this set. Counted so
                    // a boot can tell a witness that never armed from a guest
                    // that writes every frame.
                    crate::runtime::drain::note_store_route("gvaw_untracked");
                    return;
                }
            }
        }
    };
    let gen_at_store = match host.guest_write_gen(token) {
        Some(g) => {
            crate::runtime::drain::note_store_route("gvaw_armed");
            g
        }
        None => {
            // The host holds the pages but its report is not yet a fact about
            // the guest — the arming window. 0 fails the currency test rather
            // than passing it by default.
            crate::runtime::drain::note_store_route("gvaw_unarmed");
            0
        }
    };
    state.gva_store_witness.stores = state.gva_store_witness.stores.wrapping_add(1);
    let last_seen = state.gva_store_witness.stores;
    let host_epoch_at_store = state.host_writes.epoch();
    state.gva_store_witness.entries.insert(
        key,
        Entry {
            token,
            gpas: gpas.to_vec(),
            gen_at_store,
            host_epoch_at_store,
            last_seen,
        },
    );
}

/// Forget every target whose pages this task owned, handing back their tokens.
///
/// A task teardown takes its page tables with it, so nothing in it names
/// anything any more. There is no task in the key — see [`GvaTargetKey::of`] —
/// so this takes the page list instead: an entry every one of whose pages the
/// caller names as gone is gone. Called with the task's retired page set.
pub fn retire_pages(state: &mut DeviceState, gone: &[u64]) {
    if gone.is_empty() {
        return;
    }
    let doomed: Vec<GvaTargetKey> = state
        .gva_store_witness
        .entries
        .iter()
        .filter(|(_, e)| e.gpas.iter().any(|p| gone.contains(p)))
        .map(|(k, _)| *k)
        .collect();
    for k in doomed {
        if let Some(e) = state.gva_store_witness.entries.remove(&k) {
            state.retired_guest_write_tokens.push(e.token);
        }
    }
}

/// What both witnesses say about `key`'s pages since the Store that stamped
/// them.
///
/// Every variant but [`GvaWriteReach::Quiet`] means "assume written". They are
/// kept apart because "this rail never got started", "the guest rewrites this
/// target every frame" and "the record aged out" are the same refusal and three
/// completely different findings — the same reason the type-11 twin splits its
/// own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GvaWriteReach {
    /// Neither the guest nor another rail of this device has written these
    /// pages since the Store. The only answer that licenses reusing the
    /// resident, and only in combination with the caller's own epoch test.
    Quiet,
    /// The hypervisor's verdict, when it is not clean.
    Guest(GuestWriteVerdict),
    /// This device's own record, when it is not quiet.
    Host(HostWriteVerdict),
}

impl GvaWriteReach {
    /// Census route naming this answer, so a boot can say which half refused.
    /// The two halves keep their own vocabularies rather than being folded into
    /// one, because they are repaired differently.
    pub fn route(self) -> &'static str {
        match self {
            Self::Quiet => "gvaw_quiet",
            Self::Guest(GuestWriteVerdict::Clean) => "gvaw_guest_clean",
            Self::Guest(GuestWriteVerdict::NoMapping) => "gvaw_no_entry",
            Self::Guest(GuestWriteVerdict::NoStamp) => "gvaw_guest_no_stamp",
            Self::Guest(GuestWriteVerdict::Wrote) => "gvaw_guest_wrote",
            Self::Guest(GuestWriteVerdict::Unreadable) => "gvaw_guest_unreadable",
            Self::Host(HostWriteVerdict::Quiet) => "gvaw_host_quiet",
            Self::Host(HostWriteVerdict::Overlap) => "gvaw_host_overlap",
            Self::Host(HostWriteVerdict::Unnamed) => "gvaw_host_unnamed",
            Self::Host(HostWriteVerdict::Aged) => "gvaw_host_aged",
            Self::Host(HostWriteVerdict::Unresolvable) => "gvaw_host_unresolvable",
        }
    }

    /// True only for [`Self::Quiet`], spelled once so a caller cannot express
    /// the question as "not one of the variants I remembered".
    pub fn is_quiet(self) -> bool {
        matches!(self, Self::Quiet)
    }
}

/// Ask both witnesses about `key`.
///
/// The guest half is asked first because it is a word; the host-write half
/// walks a ring and is only reached when the guest half is clean.
pub fn reach<H: HostOps>(state: &DeviceState, host: &H, key: GvaTargetKey) -> GvaWriteReach {
    let Some(e) = state.gva_store_witness.entries.get(&key) else {
        return GvaWriteReach::Guest(GuestWriteVerdict::NoMapping);
    };
    if e.gen_at_store == 0 {
        return GvaWriteReach::Guest(GuestWriteVerdict::NoStamp);
    }
    match host.guest_write_gen(e.token) {
        Some(g) if g == e.gen_at_store => {}
        Some(_) => return GvaWriteReach::Guest(GuestWriteVerdict::Wrote),
        None => return GvaWriteReach::Guest(GuestWriteVerdict::Unreadable),
    }
    let host_verdict = state
        .host_writes
        .wrote_any_since(state, e.host_epoch_at_store, &e.gpas);
    match host_verdict {
        HostWriteVerdict::Quiet => GvaWriteReach::Quiet,
        other => GvaWriteReach::Host(other),
    }
}

/// How far back the host-write record is being asked to reach for `key`, banded.
///
/// `host_writes`'s own doc asks for exactly this distribution before anyone
/// resizes its ring: "a reach that is usually 70 and a reach that is usually
/// 7000 both produce [an `Aged`-dominated] reading, and only one of them is
/// fixed by a bigger ring". This rail asks over a Store-to-LOAD interval, which
/// is the longest reach in the device, so it is the one that should supply the
/// bands.
pub fn note_host_reach(state: &DeviceState, key: GvaTargetKey) {
    let Some(e) = state.gva_store_witness.entries.get(&key) else {
        return;
    };
    let reach = state.host_writes.epoch().saturating_sub(e.host_epoch_at_store);
    crate::runtime::drain::note_store_route(if reach < 64 {
        "gvaw_reach_lt64"
    } else if reach < 512 {
        "gvaw_reach_lt512"
    } else if reach < 4096 {
        "gvaw_reach_lt4k"
    } else {
        "gvaw_reach_ge4k"
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::host::FakeHost;

    /// The x86 guest page, to match the `DeviceState` these tests build.
    const PAGE: u64 = 1u64 << crate::model::PAGE_SHIFT_X86;

    fn device() -> DeviceState {
        DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_X86)
    }

    fn key(generation: u64) -> GvaTargetKey {
        GvaTargetKey {
            gva: 0x1_0000,
            generation,
            width: 64,
            height: 64,
            bgra: true,
        }
    }

    fn armed(state: &mut DeviceState, host: &mut FakeHost, k: GvaTargetKey, gpas: &[u64]) {
        note_store(state, host, k, gpas);
    }

    /// The whole contract of the guest half: a Store stamps, a quiet guest reads
    /// back `Quiet`, and a guest write into a tracked page turns it into a
    /// refusal. A false `Quiet` is a stale frame that then persists, so the
    /// write case is the one that matters.
    #[test]
    fn a_guest_write_after_the_store_stops_the_target_reading_quiet() {
        let mut state = device();
        let mut host = FakeHost::new();
        let pages = [3 * PAGE, 4 * PAGE];
        armed(&mut state, &mut host, key(0xabc), &pages);
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);

        host.guest_wrote_page(4 * PAGE);
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Guest(GuestWriteVerdict::Wrote)
        );

        // A fresh Store re-stamps, so one guest write does not disable the
        // target for the rest of the boot.
        armed(&mut state, &mut host, key(0xabc), &pages);
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);
    }

    /// A write by another rail of this device is invisible to the hypervisor,
    /// and it is the second half. `host_writes` aliasing across id namespaces is
    /// real, so a page-exact ask is what separates "something wrote my span"
    /// from "something wrote somewhere".
    #[test]
    fn this_devices_own_write_into_the_span_stops_it_reading_quiet() {
        let mut state = device();
        let mut host = FakeHost::new();
        let pages = [3 * PAGE, 4 * PAGE];
        armed(&mut state, &mut host, key(0xabc), &pages);
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);

        // Somewhere else: still quiet, because the ask is page-exact.
        state.note_host_wrote_pages(vec![9 * PAGE]);
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);

        state.note_host_wrote_pages(vec![4 * PAGE]);
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Host(HostWriteVerdict::Overlap)
        );
    }

    /// A writer that could not name its pages invalidates everything older, and
    /// an aged-out record cannot rule anything out. Both must refuse: reading
    /// either as quiet is the wrong-frame direction.
    #[test]
    fn an_unnamed_or_aged_host_write_refuses_rather_than_reading_quiet() {
        let mut state = device();
        let mut host = FakeHost::new();
        let pages = [3 * PAGE];
        armed(&mut state, &mut host, key(0xabc), &pages);
        state.note_host_wrote_guest_ram();
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Host(HostWriteVerdict::Unnamed)
        );

        // And the aged case, which is the one a Store-to-LOAD interval will
        // actually hit: push the ring past what it remembers.
        armed(&mut state, &mut host, key(0xabc), &pages);
        for i in 0..256u64 {
            state.note_host_wrote_pages(vec![(100 + i) * PAGE]);
        }
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Host(HostWriteVerdict::Aged)
        );
    }

    /// The Store records its own write into these pages, so a witness that
    /// captured the host epoch before that would find its own Store in the ring
    /// and refuse forever. This is the ordering bug that would make the rail
    /// read zero on every boot with nothing saying why.
    #[test]
    fn a_store_does_not_invalidate_its_own_stamp() {
        let mut state = device();
        let mut host = FakeHost::new();
        let pages = [3 * PAGE];
        // The order `store_gva_frame` uses: record the write, then stamp.
        state.note_host_wrote_pages(pages.to_vec());
        armed(&mut state, &mut host, key(0xabc), &pages);
        assert_eq!(reach(&state, &host, key(0xabc)), GvaWriteReach::Quiet);
    }

    /// The guest recycling a GVA must not inherit the previous allocation's
    /// answer. The page-set hash is in the key, so the old entry is not merely
    /// out-voted — it cannot be reached at all.
    #[test]
    fn a_recycled_address_cannot_reach_the_old_allocations_entry() {
        let mut state = device();
        let mut host = FakeHost::new();
        let pages = [3 * PAGE];
        armed(&mut state, &mut host, key(0xabc), &pages);
        assert_eq!(
            reach(&state, &host, key(0xdef)),
            GvaWriteReach::Guest(GuestWriteVerdict::NoMapping),
            "a different page set at the same address is a different target"
        );
    }

    /// A host that cannot watch pages arms nothing, and a token still inside its
    /// arming window stamps 0. Neither may read as quiet by default.
    #[test]
    fn an_unwatchable_host_and_an_unarmed_token_both_refuse() {
        let mut state = device();
        let mut blind = FakeHost::new();
        blind.guest_writes_unobservable = true;
        armed(&mut state, &mut blind, key(0xabc), &[3 * PAGE]);
        assert_eq!(
            reach(&state, &blind, key(0xabc)),
            GvaWriteReach::Guest(GuestWriteVerdict::NoMapping),
            "nothing is inserted when the host cannot arm"
        );

        let mut warming = FakeHost::new();
        warming.guest_write_startup_window = true;
        armed(&mut state, &mut warming, key(0xabc), &[3 * PAGE]);
        assert_eq!(
            reach(&state, &warming, key(0xabc)),
            GvaWriteReach::Guest(GuestWriteVerdict::NoStamp)
        );
    }

    /// Past the bound the least recently stored target is evicted and its token
    /// handed back, rather than the map growing without limit or being dropped
    /// whole. A leaked token has the whole VM walking its pages at every
    /// doorbell for the life of the process.
    #[test]
    fn the_bound_evicts_the_coldest_target_and_hands_back_its_token() {
        let mut state = device();
        let mut host = FakeHost::new();
        for i in 0..(MAX_TRACKED_TARGETS as u64 + 4) {
            let k = GvaTargetKey {
                gva: 0x1000 * (i + 1),
                generation: 0xabc,
                width: 8,
                height: 8,
                bgra: true,
            };
            armed(&mut state, &mut host, k, &[(i + 3) * PAGE]);
        }
        assert!(state.gva_store_witness.len() <= MAX_TRACKED_TARGETS);
        assert_eq!(
            state.retired_guest_write_tokens.len(),
            4,
            "one token handed back per eviction"
        );
        assert_eq!(
            reach(
                &state,
                &host,
                GvaTargetKey {
                    gva: 0x1000,
                    generation: 0xabc,
                    width: 8,
                    height: 8,
                    bgra: true
                }
            ),
            GvaWriteReach::Guest(GuestWriteVerdict::NoMapping),
            "the coldest target is the one gone"
        );
    }

    /// Every armed token is handed back on reset. Anything else leaves the host
    /// dirty-logging those pages for the life of the process.
    #[test]
    fn every_token_is_handed_back_when_the_witness_is_taken() {
        let mut state = device();
        let mut host = FakeHost::new();
        armed(&mut state, &mut host, key(0xabc), &[3 * PAGE]);
        armed(
            &mut state,
            &mut host,
            GvaTargetKey {
                gva: 0x2_0000,
                ..key(0xabc)
            },
            &[4 * PAGE],
        );
        let freed = state.gva_store_witness.take_tokens();
        assert_eq!(freed.len(), 2);
        assert_eq!(state.gva_store_witness.len(), 0);
    }

    /// Pages a task teardown reclaimed take their targets with them: a GVA in a
    /// dead task names nothing, and its token must not outlive it.
    #[test]
    fn retiring_a_pages_target_hands_back_only_that_targets_token() {
        let mut state = device();
        let mut host = FakeHost::new();
        armed(&mut state, &mut host, key(0xabc), &[3 * PAGE]);
        let other = GvaTargetKey {
            gva: 0x2_0000,
            ..key(0xabc)
        };
        armed(&mut state, &mut host, other, &[4 * PAGE]);
        retire_pages(&mut state, &[3 * PAGE]);
        assert_eq!(state.retired_guest_write_tokens.len(), 1);
        assert_eq!(
            reach(&state, &host, key(0xabc)),
            GvaWriteReach::Guest(GuestWriteVerdict::NoMapping)
        );
        assert_eq!(
            reach(&state, &host, other),
            GvaWriteReach::Quiet,
            "another target's entry is untouched"
        );
    }
}
