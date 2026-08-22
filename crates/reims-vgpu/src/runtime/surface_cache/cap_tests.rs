//! Tests for the surface cache's byte cap and its eviction order.
//!
//! Kept separate from `super::tests` because it was a separate module before
//! the move, and merging two test modules is a judgement about what they are
//! for rather than motion.

use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_X86};

/// One 16x16 BGRA frame — 1 024 bytes, so a cap in the tens of KiB holds a
/// countable number of them.
const W: u32 = 16;
const H: u32 = 16;
const FRAME_BYTES: usize = (W * H * 4) as usize;

fn state_capped(cap: usize) -> Device {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    state.host_replicas.gva_cache_byte_cap = cap;
    state
}

fn store_frame(state: &mut Device, gva: u64, fill: u8) {
    store_gva_owned(state, gva, W, H, vec![fill; FRAME_BYTES], 0, None, true);
}

/// The leak, and the bound.
///
/// This map is keyed by guest *virtual* address and a store at an existing
/// key replaces in place, so growth comes entirely from new GVAs — which is
/// exactly what a resolution change produces. Measured on the rig, 60
/// guest-driven mode changes took it from 26 entries to 354 without it ever
/// decreasing once. Here that shape is reproduced with 400 distinct
/// addresses: uncapped it holds all 400, capped it holds what the cap
/// allows and no more.
#[test]
fn the_byte_cap_bounds_a_map_whose_keys_never_repeat() {
    let uncapped = {
        let mut state = state_capped(usize::MAX);
        for i in 0..400u64 {
            store_frame(&mut state, 0x1000 + i * 0x1000, i as u8);
        }
        state.host_replicas.gva_surfaces.len()
    };
    assert_eq!(
        uncapped, 400,
        "control: without a cap every abandoned address is kept forever"
    );

    let cap = 64 * FRAME_BYTES;
    let mut state = state_capped(cap);
    for i in 0..400u64 {
        store_frame(&mut state, 0x1000 + i * 0x1000, i as u8);
    }
    let bytes: usize = state
        .host_replicas
        .gva_surfaces
        .values()
        .map(|e| e.bgra.len())
        .sum();
    assert!(
        bytes <= cap,
        "capped map holds {bytes} bytes against a {cap}-byte cap"
    );
    assert!(
        state.host_replicas.gva_surfaces.len() < 400,
        "the cap must actually have evicted something"
    );
    assert!(
        state.host_replicas.gva_eviction_witness.evicted > 0,
        "and it must say so: an eviction count of zero is a cap that never engaged"
    );
}

/// The wallpaper property, and the whole reason this is recency and not
/// staleness.
///
/// A wallpaper plane is stored **once** and sampled every frame thereafter.
/// A rule keyed on stores would see the most-wanted entry in the map as its
/// coldest; a rule keyed on translation would evict it too, because this
/// cache is deliberately retained across Unmap, so "does not translate" is
/// that entry's normal state. Touch-on-read is what makes it the hottest
/// thing here instead.
#[test]
fn an_entry_read_every_frame_but_never_rewritten_survives_the_cap() {
    let cap = 16 * FRAME_BYTES;
    let mut state = state_capped(cap);
    let wallpaper = 0x9_0000u64;
    store_frame(&mut state, wallpaper, 0xAB);

    for i in 0..400u64 {
        // Sampled every frame, never rewritten — the read is the only thing
        // keeping it alive.
        assert!(
            get_gva(&state, wallpaper, W, H).is_some(),
            "wallpaper evicted at round {i}"
        );
        touch_gva(&mut state, wallpaper, W, H);
        store_frame(&mut state, 0x100_0000 + i * 0x1000, i as u8);
    }

    let served = get_gva(&state, wallpaper, W, H).expect("wallpaper survives the whole stream");
    assert!(
        served.iter().all(|&b| b == 0xAB),
        "and it is still its own pixels, not a neighbour's"
    );
    assert_eq!(
        state
            .host_replicas
            .gva_eviction_witness
            .wanted
            .load(Relaxed),
        0,
        "no lookup was ever charged to the cap"
    );
}

/// Without the touch, the same entry is evicted — so the assertion above is
/// testing the touch and not merely a map that happens to be small.
///
/// The two tests are a matched pair on one binary: identical cap, identical
/// insert stream, and the only difference is whether the read path reports
/// the use.
#[test]
fn the_same_entry_is_evicted_when_nothing_reports_reading_it() {
    let cap = 16 * FRAME_BYTES;
    let mut state = state_capped(cap);
    let wallpaper = 0x9_0000u64;
    store_frame(&mut state, wallpaper, 0xAB);
    for i in 0..400u64 {
        store_frame(&mut state, 0x100_0000 + i * 0x1000, i as u8);
    }
    assert!(
        get_gva(&state, wallpaper, W, H).is_none(),
        "an entry nothing reports using is exactly what the cap is for"
    );
    assert_eq!(
        state
            .host_replicas
            .gva_eviction_witness
            .wanted
            .load(Relaxed),
        1,
        "and the lookup that then wanted it is charged to the cap, not written off"
    );
}

/// The harm witness must charge the cap for its own misses and nothing
/// else, or the number cannot be read.
///
/// An address that was never cached misses for the ordinary reason, and a
/// cached address asked for at a geometry it never held is not a cap
/// casualty either. Only a lookup that would have hit, for an identity the
/// cap removed, is the cost of capping.
#[test]
fn the_witness_charges_the_cap_only_for_misses_the_cap_caused() {
    let mut state = state_capped(2 * FRAME_BYTES);
    let victim = 0x5_0000u64;
    store_frame(&mut state, victim, 0x11);
    for i in 0..64u64 {
        store_frame(&mut state, 0x300_0000 + i * 0x1000, i as u8);
    }
    assert!(state.host_replicas.gva_eviction_witness.evicted > 0);
    assert!(!state.host_replicas.gva_surfaces.contains_key(&victim));

    // Never cached at all.
    assert!(get_gva(&state, 0xdead_0000, W, H).is_none());
    assert_eq!(
        state
            .host_replicas
            .gva_eviction_witness
            .wanted
            .load(Relaxed),
        0,
        "an address this cache never held is an ordinary miss"
    );

    // Evicted, but asked for at a geometry it never had.
    assert!(get_gva(&state, victim, W * 2, H).is_none());
    assert_eq!(
        state
            .host_replicas
            .gva_eviction_witness
            .wanted
            .load(Relaxed),
        0,
        "the cap did not remove *that* identity"
    );

    // The real thing.
    assert!(get_gva(&state, victim, W, H).is_none());
    assert_eq!(
        state
            .host_replicas
            .gva_eviction_witness
            .wanted
            .load(Relaxed),
        1,
        "a lookup that would have hit but for the cap is the cost of capping"
    );
}

/// A probe is not a read, or one frame's single logical lookup would be
/// counted two or three times.
///
/// The sampled path asks [`has_gva`] first (so it can decide whether there
/// is anything to revalidate) and only then reads. Charging the witness in
/// the shared selection rule would score that frame twice and make the
/// figure uninterpretable — inflated toward reporting harm, which is the
/// direction that wastes a session rather than hiding a bug, but wrong.
#[test]
fn asking_whether_an_entry_exists_is_not_charged_as_harm() {
    let mut state = state_capped(2 * FRAME_BYTES);
    let victim = 0x5_0000u64;
    store_frame(&mut state, victim, 0x11);
    for i in 0..64u64 {
        store_frame(&mut state, 0x300_0000 + i * 0x1000, i as u8);
    }
    assert!(!has_gva(&state, victim, W, H));
    touch_gva(&mut state, victim, W, H);
    assert_eq!(
        state
            .host_replicas
            .gva_eviction_witness
            .wanted
            .load(Relaxed),
        0,
        "probes do not read the pixels, so they are not denied any"
    );
    assert!(get_gva(&state, victim, W, H).is_none());
    assert_eq!(
        state
            .host_replicas
            .gva_eviction_witness
            .wanted
            .load(Relaxed),
        1
    );
}

/// Once the content is back, later misses are a different question.
///
/// Otherwise the witness keeps charging the cap for an identity that has
/// been re-stored since, and `gva_cap_wanted` drifts upward for reasons
/// that have nothing to do with the bound.
#[test]
fn a_store_that_brings_an_evicted_identity_back_stops_charging_the_cap() {
    let mut state = state_capped(2 * FRAME_BYTES);
    let victim = 0x5_0000u64;
    store_frame(&mut state, victim, 0x11);
    for i in 0..64u64 {
        store_frame(&mut state, 0x300_0000 + i * 0x1000, i as u8);
    }
    assert!(!state.host_replicas.gva_surfaces.contains_key(&victim));

    store_frame(&mut state, victim, 0x22);
    assert!(get_gva(&state, victim, W, H).is_some());
    // Evict it again by store pressure, but this time the witness has
    // forgotten it, so the miss below is not the cap's to answer for.
    state.host_replicas.gva_eviction_witness = crate::model::GvaEvictionWitness::default();
    assert!(get_gva(&state, 0x5_1000, W, H).is_none());
    assert_eq!(
        state
            .host_replicas
            .gva_eviction_witness
            .wanted
            .load(Relaxed),
        0
    );
}

/// The ring bound must under-report visibly, never silently.
///
/// It remembers a fixed number of evicted identities, so a long boot
/// evicts more than it can hold. That makes `wanted` a lower bound, and a
/// reader has to be able to tell — `forgotten` is what says so.
#[test]
fn forgetting_an_evicted_key_is_reported_rather_than_swallowed() {
    let mut state = state_capped(2 * FRAME_BYTES);
    let overflow_by = 64u64;
    let n = crate::model::GVA_EVICTION_WITNESS_KEYS as u64 + overflow_by;
    for i in 0..n {
        store_frame(&mut state, 0x1000 + i * 0x1000, i as u8);
    }
    let (evicted, _, forgotten) = state.host_replicas.gva_eviction_witness.counts();
    assert!(evicted > crate::model::GVA_EVICTION_WITNESS_KEYS as u64);
    assert!(
        forgotten > 0,
        "the ring overflowed and the census must be able to say so"
    );

    // The very first address evicted is the one the ring dropped, so a
    // lookup for it is uncounted — which is the point of `forgotten`.
    assert!(get_gva(&state, 0x1000, W, H).is_none());
    assert_eq!(
        state
            .host_replicas
            .gva_eviction_witness
            .wanted
            .load(Relaxed),
        0,
        "uncounted, and `forgotten` is the flag that keeps that honest"
    );
}

/// A store must not be undone by its own cap enforcement.
///
/// Enforcement runs *after* the insert, so a single entry over the
/// low-water mark is the only candidate in an otherwise empty map and
/// evicts itself — the surface is then never cached at all, which is the
/// "refused for being big" behaviour the cap explicitly must not have.
///
/// Reachable in production, not only at test sizes: `MAX_SCANOUT_DIM` is
/// 8192, so one entry may be 256 MiB against a 112 MiB low-water mark.
#[test]
fn an_entry_bigger_than_the_cap_is_admitted_alone_not_evicted_by_its_own_store() {
    let (w, h) = (64u32, 64u32);
    let big = (w * h * 4) as usize;
    let mut state = state_capped(big / 4);
    let gva = 0x8_0000u64;
    store_gva_owned(&mut state, gva, w, h, vec![0x77; big], 0, None, true);

    let served = get_gva(&state, gva, w, h)
        .expect("an oversized entry rides alone rather than being refused");
    assert!(served.iter().all(|&b| b == 0x77));
    assert_eq!(
        state.host_replicas.gva_cache_bytes, big,
        "and the total still describes it"
    );
    assert_eq!(
        state.host_replicas.gva_eviction_witness.evicted, 0,
        "nothing was evicted: there was nothing else to evict"
    );

    // It also does not pin the map forever — a later store at another
    // address makes it an ordinary candidate and the coldest thing present.
    store_frame(&mut state, 0x8_1000, 0x11);
    assert!(
        get_gva(&state, gva, w, h).is_none(),
        "once it is not the store's own key it is an ordinary eviction candidate"
    );
}

/// The cap tests a running total, so that total has to equal the map.
///
/// A second source of truth is exactly how a bound silently stops bounding:
/// under-count and the cap never fires, over-count and it evicts content it
/// never needed to. This drives the three transitions that can break it —
/// a fresh key, a replace at an existing key (which must net to the
/// difference, not double-charge), and an eviction — and holds the total to
/// the real sum at every step. `gva_cap_drift` is the same check, live.
#[test]
fn the_running_byte_total_equals_the_map_after_every_transition() {
    let truth = |state: &Device| -> usize {
        state
            .host_replicas
            .gva_surfaces
            .values()
            .map(|e| e.bgra.len())
            .sum()
    };
    let mut state = state_capped(usize::MAX);
    assert_eq!(state.host_replicas.gva_cache_bytes, 0);

    // Fresh keys.
    for i in 0..8u64 {
        store_frame(&mut state, 0x1000 + i * 0x1000, i as u8);
        assert_eq!(
            state.host_replicas.gva_cache_bytes,
            truth(&state),
            "after insert {i}"
        );
    }
    assert_eq!(state.host_replicas.gva_cache_bytes, 8 * FRAME_BYTES);

    // Replace at an existing key, same size: the total must not move.
    store_frame(&mut state, 0x1000, 0xFF);
    assert_eq!(
        state.host_replicas.gva_cache_bytes,
        8 * FRAME_BYTES,
        "replace double-charged"
    );
    assert_eq!(state.host_replicas.gva_cache_bytes, truth(&state));

    // Replace at an existing key with a *different* geometry: the old bytes
    // are reclaimed and the new ones charged.
    let (w2, h2) = (W * 2, H);
    store_gva_owned(
        &mut state,
        0x1000,
        w2,
        h2,
        vec![0x5A; (w2 * h2 * 4) as usize],
        0,
        None,
        true,
    );
    assert_eq!(
        state.host_replicas.gva_cache_bytes,
        truth(&state),
        "geometry change"
    );

    // Eviction.
    evict_gva(&mut state, 0x1000);
    assert_eq!(
        state.host_replicas.gva_cache_bytes,
        truth(&state),
        "after evict"
    );
    assert_eq!(state.host_replicas.gva_cache_bytes, 7 * FRAME_BYTES);

    // And after the cap itself has run a batch of evictions.
    state.host_replicas.gva_cache_byte_cap = 4 * FRAME_BYTES;
    for i in 0..64u64 {
        store_frame(&mut state, 0x400_0000 + i * 0x1000, i as u8);
        assert_eq!(
            state.host_replicas.gva_cache_bytes,
            truth(&state),
            "under the cap, round {i}"
        );
    }
    assert!(
        state.host_replicas.gva_eviction_witness.evicted > 0,
        "the cap must have run"
    );
}

use std::sync::atomic::Ordering::Relaxed;

/// Store a frame whose bytes the guest's own pages do **not** hold — the shape
/// a render writeback produces on every outcome that did not reach guest RAM.
fn store_unlanded_frame(state: &mut Device, gva: u64, fill: u8) {
    store_gva_owned(state, gva, W, H, vec![fill; FRAME_BYTES], 0, None, false);
}

/// The cap must not evict an entry whose bytes never reached guest RAM.
///
/// A writeback refuses a guest write when the address has been re-pointed, on
/// the argument that the refusal is safe because the caller keeps the content
/// either way, so nothing renderable is lost by refusing. That is a claim about
/// this map. Before
/// `HostSurface::guest_holds_bytes` the cap was free to falsify it: the window
/// exclusion only covers an address whose flush has not run, and a refused flush
/// leaves no window behind.
///
/// Drive far past the cap with unlanded frames and require that every one
/// survives — the map goes over its bound rather than take the only copy.
#[test]
fn the_cap_never_evicts_bytes_the_guest_does_not_have() {
    let mut state = state_capped(FRAME_BYTES * 8);
    for i in 0..64u64 {
        store_unlanded_frame(&mut state, 0x1000 + i * 0x1000, i as u8);
    }
    assert_eq!(
        state.host_replicas.gva_surfaces.len(),
        64,
        "no unlanded entry may be evicted, even eight times over the cap"
    );
    assert!(
        state.host_replicas.gva_cache_bytes > state.host_replicas.gva_cache_byte_cap,
        "the map is deliberately over its cap rather than lossy"
    );
    let (evicted, wanted, _) = state.host_replicas.gva_eviction_witness.counts();
    assert_eq!((evicted, wanted), (0, 0), "and nothing was taken");
}

/// A landed entry is still evictable, and is chosen ahead of unlanded ones.
///
/// The exclusion has to be narrow: if it swallowed the whole cap the bound would
/// stop working for the ordinary case, where the guest's pages hold the same
/// bytes and an eviction costs a re-read.
#[test]
fn the_cap_still_evicts_what_the_guest_can_re_derive() {
    let mut state = state_capped(FRAME_BYTES * 8);
    // Four entries the guest cannot re-derive, pinned by their own truthfulness.
    for i in 0..4u64 {
        store_unlanded_frame(&mut state, 0x1_0000 + i * 0x1000, i as u8);
    }
    // Then a stream of ordinary landed ones, well past the cap.
    for i in 0..32u64 {
        store_frame(&mut state, 0x2_0000 + i * 0x1000, i as u8);
    }
    for i in 0..4u64 {
        assert!(
            state
                .host_replicas
                .gva_surfaces
                .contains_key(&(0x1_0000 + i * 0x1000)),
            "unlanded entry {i} survived the landed stream"
        );
    }
    let (evicted, _, _) = state.host_replicas.gva_eviction_witness.counts();
    assert!(
        evicted > 0,
        "landed entries are still reclaimed, or the cap has stopped working"
    );
    assert!(
        state.host_replicas.gva_surfaces.len() < 36,
        "and the map did not simply keep everything: {}",
        state.host_replicas.gva_surfaces.len()
    );
}

/// A later flush that *does* reach guest RAM makes the entry evictable again.
///
/// Without this the exclusion would be a ratchet: the compute writeback rail
/// caches before it writes, so every one of its entries would enter the map
/// unevictable and stay that way through any number of successful writes.
#[test]
fn landing_in_guest_pages_returns_an_entry_to_the_caps_reach() {
    let mut state = state_capped(FRAME_BYTES * 4);
    let gva = 0x9_0000u64;
    store_unlanded_frame(&mut state, gva, 0xAA);
    note_gva_landed(&mut state, gva);
    for i in 0..32u64 {
        store_frame(&mut state, 0xB_0000 + i * 0x1000, i as u8);
    }
    assert!(
        !state.host_replicas.gva_surfaces.contains_key(&gva),
        "once the guest holds the bytes the entry is an ordinary candidate"
    );
}

/// When the exclusions leave nothing to take, the cap says so.
///
/// A GPU with no free memory refuses; it does not discard a surface the client
/// still holds. The refusal has to be visible or an over-cap map is
/// indistinguishable from a cap that is holding.
#[test]
fn a_cap_with_nothing_evictable_reports_instead_of_going_quiet() {
    let cap_bytes = FRAME_BYTES * 8;
    let capture = crate::observe::FailCapture::start();
    let mut state = state_capped(cap_bytes);
    // 200 stores, of which ~192 land the map over its cap and none is evictable.
    for i in 0..200u64 {
        store_unlanded_frame(&mut state, 0x1000 + i * 0x1000, i as u8);
    }
    assert_eq!(
        state.host_replicas.gva_surfaces.len(),
        200,
        "still nothing evicted"
    );

    let lines: Vec<String> = capture
        .lines()
        .into_iter()
        .filter(|l| l.starts_with("gva_cache_cap "))
        .collect();
    assert!(
        lines
            .first()
            .is_some_and(|l| l.contains("reason=gva_cache_cap_nothing_evictable")
                && l.contains(&format!("cap={cap_bytes}"))),
        "the refusal names itself and the bound it is over: {lines:?}"
    );
    // The latch is on the overshoot's binary magnitude, so the line count is
    // the number of doublings — logarithmic in the overshoot, not one per
    // store. Anything close to 192 here means the dedupe has stopped working
    // and this decline has become a flood on the always-on channel.
    assert!(
        lines.len() <= 16,
        "one line per doubling, not one per over-cap store: {} lines",
        lines.len()
    );
}
