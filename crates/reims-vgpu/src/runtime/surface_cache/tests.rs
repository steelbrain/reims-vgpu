//! Tests for the host surface cache.
//!
//! Out of line with `cap_tests` for the reason the sibling `runtime/` modules
//! that already do this have: colocated, the two were together 1,663 of
//! `surface_cache.rs`'s 2,480 lines — 67%.

use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
use crate::runtime::host::FakeHost;

/// The probe has to separate a reassigned address from one that merely
/// failed to walk, because only the first is evidence the entry is dead.
///
/// `d455c3e`'s finding was that this device routinely asks before the guest
/// has finished mapping, so a failure to translate is a transient state and
/// not a licence to drop content — collapsing `unmapped` into `moved` would
/// build an eviction rule on exactly that mistake.
#[test]
fn the_backing_probe_separates_a_reassigned_address_from_an_unmapped_one() {
    use crate::model::GvaBacking;
    let mut host = FakeHost::new();
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let root_gpa = setup_depth1_task(&mut host, &mut st);

    // Three entries at GVAs 1, 2 and 3 pages in, each recording the page its
    // own PTE currently names (PT_BASE + i, i.e. 5, 6, 7).
    for i in 1..=3u32 {
        let gva = (i as u64) << PAGE_SHIFT_ARM64E;
        store_gva_owned(&mut st, gva, 2, 2, vec![0u8; 2 * 2 * 4], 0, None, true);
        st.host_gva_surfaces.get_mut(&gva).unwrap().backing = Some(GvaBacking {
            task_id: 1,
            first_gpa: ((4 + i) as u64) << PAGE_SHIFT_ARM64E,
        });
    }

    // Nothing touched yet: every backing still agrees with the page table.
    assert_eq!(gva_backing_moved(&st, &host), (0, 0, 3));

    // The guest hands GVA page 2 to a different allocation.
    repoint_pte(&mut host, root_gpa, 2, 12);
    assert_eq!(
        gva_backing_moved(&st, &host),
        (1, 0, 3),
        "a re-pointed PTE is a moved backing"
    );

    // And drops GVA page 3 entirely (PTE 0 = not present).
    repoint_pte(&mut host, root_gpa, 3, 0);
    let (moved, unmapped, checked) = gva_backing_moved(&st, &host);
    assert_eq!(
        (moved, unmapped),
        (1, 1),
        "unmapped must not be folded into moved"
    );
    // The denominator must stay honest, or "nothing moved" and "nothing was
    // examined" become the same reading.
    assert_eq!(checked, 3);
}

/// A guest virtual address means nothing outside the address space it was
/// recorded in, and this cache is keyed by the address alone.
///
/// Serving across tasks hands a render pass another process's picture as the
/// attachment's prior content, and because the matching Store writes the
/// composite back it persists rather than flickering. The freshness probe cannot
/// catch it — `gva_backing_state` walks the page table of the task that *stored*
/// the entry, so it answers `Same` however foreign the asker is, which is why
/// the ownership test has to come first and separately.
#[test]
fn the_seed_door_refuses_an_address_recorded_by_another_task() {
    use crate::model::GvaBacking;
    let mut host = FakeHost::new();
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_depth1_task(&mut host, &mut st);

    let gva = 1u64 << PAGE_SHIFT_ARM64E;
    // `guest_holds_bytes: false` — an entry the guest's pages also hold is
    // refused before ownership is reached, and this test is about ownership.
    store_gva_owned(&mut st, gva, 2, 2, vec![0u8; 2 * 2 * 4], 0, None, false);
    st.host_gva_surfaces.get_mut(&gva).unwrap().backing = Some(GvaBacking {
        task_id: 1,
        first_gpa: 5u64 << PAGE_SHIFT_ARM64E,
    });

    assert_eq!(
        gva_seed_verdict(&st, &host, 1, gva),
        GvaSeedVerdict::Admit,
        "the task that stored it, over pages that have not moved"
    );
    assert_eq!(
        gva_seed_verdict(&st, &host, 2, gva),
        GvaSeedVerdict::OtherTask,
        "another task's identical address is not this entry"
    );
    // The blind spot this exists to cover: the freshness probe is perfectly
    // happy, because it never asked who was asking.
    assert_eq!(
        gva_backing_state(&st, &host, gva),
        GvaBackingState::Same,
        "which is exactly why a zero from that probe was never evidence"
    );
}

/// `Moved` is the one state carrying positive evidence the pixels belong to
/// someone else now, and the door computed it for a counter without acting on
/// it. Refusing costs a guest re-read; serving costs a persistent wrong layer.
#[test]
fn the_seed_door_refuses_a_backing_the_guest_has_re_pointed() {
    use crate::model::GvaBacking;
    let mut host = FakeHost::new();
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let root_gpa = setup_depth1_task(&mut host, &mut st);

    let gva = 2u64 << PAGE_SHIFT_ARM64E;
    // See the sibling test: freshness is only reached for an entry the guest's
    // own pages do not hold.
    store_gva_owned(&mut st, gva, 2, 2, vec![0u8; 2 * 2 * 4], 0, None, false);
    st.host_gva_surfaces.get_mut(&gva).unwrap().backing = Some(GvaBacking {
        task_id: 1,
        first_gpa: 6u64 << PAGE_SHIFT_ARM64E,
    });
    assert_eq!(gva_seed_verdict(&st, &host, 1, gva), GvaSeedVerdict::Admit);

    repoint_pte(&mut host, root_gpa, 2, 12);
    assert_eq!(
        gva_seed_verdict(&st, &host, 1, gva),
        GvaSeedVerdict::Moved,
        "the address now names another allocation's pages"
    );

    repoint_pte(&mut host, root_gpa, 2, 0);
    assert_eq!(
        gva_seed_verdict(&st, &host, 1, gva),
        GvaSeedVerdict::Unmapped,
        "and an address that does not translate cannot establish ownership \
         either — unmapped must not fold into moved"
    );
}

/// Every verdict has to carry a distinct route name, or the counters that price
/// this gate cannot say which arm refused.
#[test]
fn the_seed_verdicts_have_distinct_route_names() {
    let all = [
        GvaSeedVerdict::Admit,
        GvaSeedVerdict::OtherTask,
        GvaSeedVerdict::Moved,
        GvaSeedVerdict::Unmapped,
        GvaSeedVerdict::Unrecorded,
        GvaSeedVerdict::GuestHolds,
    ];
    let mut names: Vec<&str> = all.iter().map(|v| v.route()).collect();
    let distinct = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), distinct, "one route name per verdict");
}

/// The gauge has to separate the two shapes a growing cache can take, or it
/// cannot tell "many small surfaces" from "a few 4K ones" — which is the
/// distinction the no-size-cap question turns on, since a 4K entry is ~4x a
/// 1080p one.
#[test]
fn the_cache_gauge_reports_count_bytes_and_the_largest_entry() {
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    assert_eq!(cache_levels(&st).0, CacheLevel::default(), "empty is zero");

    // Two small entries and one large: 4x4 and 2x2 at RGBA8, then 8x8.
    store(&mut st, 1, 4, 4, vec![0u8; 4 * 4 * 4]);
    store(&mut st, 2, 2, 2, vec![0u8; 2 * 2 * 4]);
    store(&mut st, 3, 8, 8, vec![0u8; 8 * 8 * 4]);

    let (surfaces, _, _) = cache_levels(&st);
    assert_eq!(surfaces.entries, 3);
    assert_eq!(surfaces.bytes, (4 * 4 + 2 * 2 + 8 * 8) * 4);
    // Not the sum and not the newest — the largest single entry.
    assert_eq!(surfaces.largest, 8 * 8 * 4);

    // Eviction is visible, which is what makes an unbounded map detectable:
    // a gauge that only ever rose could not tell growth from churn.
    forget(&mut st, 3);
    let (after, _, _) = cache_levels(&st);
    assert_eq!(after.entries, 2);
    assert_eq!(after.bytes, (4 * 4 + 2 * 2) * 4);
    assert_eq!(after.largest, 4 * 4 * 4);
}

/// A generation must name one content for the life of the device, and the
/// hard case is the one that shipped broken: the entry is *destroyed* in
/// between.
///
/// `evict_gva` runs on every deferred GVA render Store arm, so this
/// sequence is the routine compositor path rather than a corner. With a
/// per-entry counter both stores report generation 1, the engine's sampled
/// cache matches `(gva, 1)` against the image it retained for the first
/// one, and binds the previous content — measured live as
/// `sampled_identity_stale identity_key=0xa4c000 generation=1` over two
/// different 64x64 icons.
///
/// Asserting the two generations differ is the whole property; asserting
/// either value would pin the counter's history instead.
#[test]
fn a_gva_reused_after_eviction_never_repeats_a_generation() {
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let (gva, w, h) = (0xa4_c000u64, 2u32, 2u32);

    let px = vec![0x11; (w * h * 4) as usize];
    store_gva_owned(&mut st, gva, w, h, px, 0, None, true);
    let (_, first) = get_gva_with_gen(&st, gva, w, h).expect("first store");

    evict_gva(&mut st, gva);
    assert!(
        get_gva_with_gen(&st, gva, w, h).is_none(),
        "the arm removes the entry outright"
    );

    let px = vec![0x22; (w * h * 4) as usize];
    store_gva_owned(&mut st, gva, w, h, px, 0, None, true);
    let (bytes, second) = get_gva_with_gen(&st, gva, w, h).expect("second store");

    assert_eq!(bytes[0], 0x22, "the cache holds the new content");
    assert_ne!(
        first, second,
        "same gva, different bytes, same generation: the sampled cache \
         would bind the first store's image for the second store's pixels"
    );
}

/// The same rule across producers. The generations used to live in two
/// namespaces split by a `1 << 32` constant precisely because the counters
/// were independent; one counter removes the constant and the failure mode
/// it was guarding against, so nothing may reintroduce a second source.
#[test]
fn every_host_cache_producer_draws_from_one_generation_source() {
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let px = vec![0u8; 4 * 4 * 4];
    let mut seen = std::collections::HashSet::new();

    store(&mut st, 7, 4, 4, px.clone());
    seen.insert(
        get_from_with_gen(&st.host_surfaces, &7, 4, 4)
            .expect("mid store")
            .1,
    );
    store_texture(&mut st, 3, 9, 4, 4, px.clone(), 0);
    seen.insert(
        get_from_with_gen(&st.host_texture_surfaces, &(3, 9), 4, 4)
            .expect("ref store")
            .1,
    );
    store_gva_owned(&mut st, 0x5000, 4, 4, px, 0, None, true);
    seen.insert(get_gva_with_gen(&st, 0x5000, 4, 4).expect("gva store").1);
    assert!(
        cede_surface_to_resident(&mut st, 7, 4, 4),
        "cession is a state change and must take a generation too"
    );
    seen.insert(st.host_surfaces.get(&7).expect("ceded entry").host_gen);

    assert_eq!(seen.len(), 4, "four stores, four distinct generations");
    assert!(
        !seen.contains(&0),
        "0 is reserved for 'no host content yet'"
    );
}

/// A short readback leaves a live entry empty, and that must not read the
/// same as a supersede.
///
/// `flush_linear_one` may refuse the guest write on drift and call the
/// refusal lossless because "the cache entry keeps the authoritative
/// bytes". When this returned a bare `bool` the caller discarded it, so the
/// two arms below were one value: the frame was gone from both the cache
/// and the guest pages, and nothing said so. The entry staying
/// resident-authoritative afterwards is the proof the bytes did not land.
#[test]
fn a_short_resident_readback_is_distinguishable_from_a_supersede() {
    use crate::contract::pixel_format::MTL_FORMAT_RGBA16_FLOAT;
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let win = LinearWindow {
        task_id: 6,
        texture_ref: 21,
        gva: 0x30_2000,
        pixel_format: MTL_FORMAT_RGBA16_FLOAT,
        width: 4,
        height: 2,
        row_stride: 32,
    };
    assert!(note_linear_texture_resident(&mut st, &win, 2));
    let need = (win.width * win.height * 8) as usize;
    let short = vec![0xabu8; need - 1];
    assert_eq!(
        materialize_linear_resident(&mut st, win.task_id, win.texture_ref, 2, &short),
        Err(LinearMaterializeDecline::ReadbackShort {
            got: need - 1,
            need,
        }),
        "a live entry that could not be filled is a loss, not a supersede"
    );
    assert_eq!(
        linear_texture_resident_gen(&st, &win),
        Some(2),
        "the entry is still resident-authoritative, so the frame is in neither place"
    );
}

/// Deferred linear residency lifecycle: note marks the entry
/// resident-authoritative with empty bytes, the resident getter validates
/// the descriptor exactly, materialize lands bytes and clears the marker,
/// and a plain bytes store also clears it.
#[test]
fn linear_resident_note_materialize_and_store_clear() {
    use crate::contract::pixel_format::{MTL_FORMAT_RGBA16_FLOAT, MTL_FORMAT_RGBA8_UNORM};
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let win = LinearWindow {
        task_id: 6,
        texture_ref: 21,
        gva: 0x30_2000,
        pixel_format: MTL_FORMAT_RGBA16_FLOAT,
        width: 4,
        height: 2,
        row_stride: 32,
    };
    let (task, r) = (win.task_id, win.texture_ref);
    assert!(note_linear_texture_resident(&mut st, &win, 2));
    assert_eq!(linear_texture_resident_gen(&st, &win), Some(2));
    // Bytes consumers see nothing while resident-authoritative.
    assert!(get_linear_texture(&st, &win).is_none());
    // Any descriptor drift invalidates the resident claim.
    assert_eq!(
        linear_texture_resident_gen(
            &st,
            &LinearWindow {
                pixel_format: MTL_FORMAT_RGBA8_UNORM,
                ..win
            }
        ),
        None
    );
    assert_eq!(
        linear_texture_resident_gen(
            &st,
            &LinearWindow {
                gva: win.gva + 0x1000,
                ..win
            }
        ),
        None
    );
    // Materialize with the wrong generation is refused; the right one
    // lands bytes and clears the marker.
    let flushed = vec![0xabu8; (win.width * win.height * 8) as usize];
    assert_eq!(
        materialize_linear_resident(&mut st, task, r, 9, &flushed),
        Err(LinearMaterializeDecline::Superseded { resident_gen: 2 }),
        "the wrong generation is a supersede, not a lost frame"
    );
    assert_eq!(
        materialize_linear_resident(&mut st, task, r, 2, &flushed),
        Ok(())
    );
    assert_eq!(linear_texture_resident_gen(&st, &win), None);
    let got = get_linear_texture(&st, &win).expect("materialized bytes");
    assert!(got.iter().all(|&b| b == 0xab));
    // A later resident note supersedes; a plain store clears again.
    assert!(note_linear_texture_resident(&mut st, &win, 3));
    let px = vec![0x5au8; (win.width * win.height * 8) as usize];
    assert!(store_linear_texture(&mut st, &win, &px));
    assert_eq!(linear_texture_resident_gen(&st, &win), None);
}

/// The two store paths admit exactly the same windows.
///
/// They used to carry a copy each of this test, and the copies were not
/// identical — a divergence would have meant a window the bytes path
/// created but the deferred path refused, or the reverse, with the caller
/// choosing between them for reasons that have nothing to do with the
/// window. Both now ask `storable_bpp`; this fails if either grows its own
/// again.
#[test]
fn both_linear_store_paths_admit_the_same_windows() {
    use crate::contract::pixel_format::MTL_FORMAT_RGBA16_FLOAT;
    let ok = LinearWindow {
        task_id: 6,
        texture_ref: 21,
        gva: 0x30_2000,
        pixel_format: MTL_FORMAT_RGBA16_FLOAT,
        width: 4,
        height: 2,
        row_stride: 32,
    };
    // Every way a window can fail to name storable content, plus the
    // baseline that must still be admitted.
    let cases = [
        ("ok", ok, true),
        (
            "no object",
            LinearWindow {
                texture_ref: 0,
                ..ok
            },
            false,
        ),
        ("no address", LinearWindow { gva: 0, ..ok }, false),
        ("zero width", LinearWindow { width: 0, ..ok }, false),
        ("zero height", LinearWindow { height: 0, ..ok }, false),
        // 4 px × 8 bytes = 32, so 31 cannot hold one row.
        (
            "stride under one tight row",
            LinearWindow {
                row_stride: 31,
                ..ok
            },
            false,
        ),
        (
            "unsized format",
            LinearWindow {
                pixel_format: 0xffff,
                ..ok
            },
            false,
        ),
    ];
    for (name, win, admits) in cases {
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let px = vec![0u8; (win.width.max(1) * win.height.max(1) * 8) as usize];
        assert_eq!(
            store_linear_texture(&mut st, &win, &px),
            admits,
            "bytes store disagrees on {name}"
        );
        let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert_eq!(
            note_linear_texture_resident(&mut st, &win, 7),
            admits,
            "deferred store disagrees on {name}"
        );
    }
    // The one precondition the deferred path holds alone: generation 0 is
    // its "no resident" value, so it can never be stored as one.
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    assert!(!note_linear_texture_resident(&mut st, &ok, 0));
}

/// Task/object deletion of a resident-authoritative entry queues the
/// engine unpin key (the runtime drains `retired_linear_residents`).
#[test]
fn linear_resident_retires_on_task_and_object_delete() {
    use crate::contract::pixel_format::MTL_FORMAT_RGBA16_FLOAT;
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    st.define_task(6, 0x1000, 1);
    let win = LinearWindow {
        task_id: 6,
        texture_ref: 21,
        gva: 0x30_2000,
        pixel_format: MTL_FORMAT_RGBA16_FLOAT,
        width: 4,
        height: 2,
        row_stride: 32,
    };
    assert!(note_linear_texture_resident(&mut st, &win, 2));
    // A pending guest-flush obligation dies with the entry (boot-16 rule:
    // never write guest pages at a lifetime boundary).
    assert!(st.delete_task(6));
    assert_eq!(st.retired_linear_residents.len(), 1);
    let key = st.retired_linear_residents[0];
    assert!(key.is_linear());
    assert_eq!(key.map_generation, 6);
    assert_eq!(key.texture_ref, 21);
    assert_eq!(key.surface_offset, 0x30_2000);
    crate::runtime::render_writeback::retire_linear_residents(&mut st);
    assert!(st.retired_linear_residents.is_empty());

    st.define_task(6, 0x1000, 1);
    st.insert_object(6, 21);
    assert!(note_linear_texture_resident(&mut st, &win, 5));
    assert!(st.delete_object(6, 21));
    assert_eq!(st.retired_linear_residents.len(), 1);
    assert_eq!(st.retired_linear_residents[0].texture_ref, 21);
    // Non-resident entries retire nothing.
    st.retired_linear_residents.clear();
    let px = vec![0u8; 4 * 2 * 8];
    st.insert_object(6, 22);
    assert!(store_linear_texture(
        &mut st,
        &LinearWindow {
            texture_ref: 22,
            gva: 0x40_0000,
            ..win
        },
        &px,
    ));
    assert!(st.delete_object(6, 22));
    assert!(st.retired_linear_residents.is_empty());
}

#[test]
fn store_and_get_roundtrip() {
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let w = 4u32;
    let h = 2u32;
    let mut px = vec![0u8; (w * h * 4) as usize];
    px[0] = 0x11;
    px[1] = 0x22;
    px[2] = 0x33;
    px[3] = 0xff;
    store(&mut st, 7, w, h, px);
    let got = get(&st, 7, w, h).expect("cached");
    assert_eq!(got[0], 0x11);
    assert_eq!(got[3], 0xff);
    assert!(get(&st, 7, 8, 8).is_none());
    forget(&mut st, 7);
    assert!(get(&st, 7, w, h).is_none());
    let _ = HostSurface::default();
}

#[test]
fn texture_and_surface_namespaces_are_separate() {
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let w = 2u32;
    let h = 2u32;
    let mut surface = vec![0u8; 16];
    surface[0] = 1;
    let mut tex = vec![0u8; 16];
    tex[0] = 2;
    store(&mut st, 5, w, h, surface);
    store_texture(&mut st, 3, 5, w, h, tex, 0);
    assert_eq!(get(&st, 5, w, h).unwrap()[0], 1);
    assert_eq!(get_texture(&st, 3, 5, w, h).unwrap()[0], 2);
}

/// Texture object refs are local to their task. Two processes routinely use
/// the same small integer, and replacing or evicting one must not discard the
/// other process's render seed.
#[test]
fn texture_cache_keeps_same_numbered_refs_in_separate_tasks() {
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let (texture_ref, w, h) = (5, 2, 2);
    store_texture(&mut st, 0, texture_ref, w, h, vec![0x11; 16], 0x1000);
    store_texture(&mut st, 1, texture_ref, w, h, vec![0x22; 16], 0x2000);

    assert_eq!(get_texture(&st, 0, texture_ref, w, h).unwrap()[0], 0x11);
    assert_eq!(get_texture(&st, 1, texture_ref, w, h).unwrap()[0], 0x22);

    evict_texture(&mut st, 1, texture_ref);
    assert!(get_texture(&st, 1, texture_ref, w, h).is_none());
    assert_eq!(
        get_texture(&st, 0, texture_ref, w, h).unwrap()[0],
        0x11,
        "evicting a client texture must preserve WindowServer's same-numbered seed"
    );
}

/// The ref cache remembers which address produced its pixels, and a store at
/// a new address replaces that answer rather than leaving the old one.
///
/// This is the only thing that separates "the GVA entry aged out of its byte
/// cap and the ref door is serving the same allocation" from "the guest
/// re-pointed this texture and the ref door is serving the previous
/// allocation's picture". Both look identical at the serve site without it,
/// and only the second is a wrong LOAD seed.
#[test]
fn the_ref_cache_remembers_which_address_produced_its_pixels() {
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let (w, h) = (2u32, 2u32);
    assert_eq!(
        texture_source_gva(&st, 3, 5, w, h),
        None,
        "no entry, no answer — an absent entry must not read as address 0"
    );
    store_texture(&mut st, 3, 5, w, h, vec![1u8; 16], 0x4000);
    assert_eq!(texture_source_gva(&st, 3, 5, w, h), Some(0x4000));
    // The guest re-points the texture and stores again: the recorded address
    // must move with the pixels, or a later serve compares against a stale one.
    store_texture(&mut st, 3, 5, w, h, vec![2u8; 16], 0x9000);
    assert_eq!(texture_source_gva(&st, 3, 5, w, h), Some(0x9000));
    // A producer with no address records none, which is its own case at the
    // serve site: unknowable, neither "same" nor "different".
    store_texture(&mut st, 3, 5, w, h, vec![3u8; 16], 0);
    assert_eq!(texture_source_gva(&st, 3, 5, w, h), Some(0));
    assert_eq!(
        texture_source_gva(&st, 3, 5, w + 1, h),
        None,
        "a geometry the entry does not answer at answers nothing here either"
    );
}

#[test]
fn gva_cache_roundtrip() {
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let gva = 0x2c48000u64;
    let mut px = vec![0u8; 16];
    px[0] = 0xaa;
    store_gva_owned(&mut st, gva, 2, 2, px, 0, None, true);
    assert_eq!(get_gva(&st, gva, 2, 2).unwrap()[0], 0xaa);
    let (got, generation) = get_gva_with_gen(&st, gva, 2, 2).unwrap();
    assert_eq!(got[0], 0xaa);
    assert_eq!(generation, 1);
    assert!(get_gva_with_gen(&st, gva, 4, 1).is_none());

    let mut replacement = vec![0u8; 16];
    replacement[0] = 0xbb;
    store_gva_owned(&mut st, gva, 2, 2, replacement, 0, None, true);
    let (got, generation) = get_gva_with_gen(&st, gva, 2, 2).unwrap();
    assert_eq!(got[0], 0xbb);
    assert_eq!(generation, 2);

    let mut owned = vec![0u8; 16];
    owned[0] = 0xcc;
    store_gva_owned(&mut st, gva, 2, 2, owned, 2, None, true);
    let (got, generation) = get_gva_with_gen(&st, gva, 2, 2).unwrap();
    assert_eq!(got[0], 0xcc);
    assert_eq!(generation, 3);
    evict_gva(&mut st, gva);
    assert!(get_gva(&st, gva, 2, 2).is_none());
}

/// The GVA encode cache is keyed by virtual address alone, at any geometry.
///
/// This is what makes "UnmapMemory retains the encode" work: the retain is
/// the *absence* of an evict on the unmap path, so the cache has to stay
/// readable through page-table churn without anyone re-registering it. The
/// live x86 wallpaper class was a full sky store to `gva=0x2c22000` followed
/// by UnmapMemory + MapMemory2 of the same VA with new PFNs; if the entry
/// did not survive on the VA key alone, the next sample found zero guest
/// pages and an empty cache and the pipe stored a black wipe. No size gate —
/// a 64x48 layer takes the same path as a full-screen one.
#[test]
fn gva_encode_is_keyed_by_address_at_any_size() {
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let gva = 0x2c22000u64;
    // Small layer — same retain path as wallpaper (no W×H gate).
    let w = 64u32;
    let h = 48u32;
    let mut px = vec![0u8; (w * h * 4) as usize];
    for chunk in px.chunks_exact_mut(4) {
        chunk[0] = 185;
        chunk[1] = 126;
        chunk[2] = 81;
        chunk[3] = 255;
    }
    store_gva_owned(&mut st, gva, w, h, px, 0, None, true);
    let got = get_gva(&st, gva, w, h).expect("retained on the VA key");
    assert_eq!(got[0], 185);
    assert_eq!(got[2], 81);
}

/// Regression guard for the host-surface cache-hit validator
/// (`get_from_with_gen` / `get_from`). This decides whether cached pixels
/// are served straight to scanout/present, so every guard clause is
/// load-bearing: serving a differently-sized or truncated entry paints the
/// wrong or a torn surface (residue / framebuffer corruption). Lock:
///  - absent id -> None;
///  - width OR height mismatch -> None (never resize-serve a stale frame);
///  - empty or short-of-`need` bytes -> None (no partial garbage);
///  - exact geom + sufficient bytes -> exactly `need` bytes (over-allocated
///    entries are truncated to the requested extent) plus the entry host_gen;
///  - `get_from` returns the same bytes, dropping only the generation.
#[test]
fn host_surface_cache_hit_validates_geom_and_truncates_to_need() {
    use crate::contract::pixel_format::RGBA8_BPP;
    let (id, w, h) = (7u32, 4u32, 2u32);
    let need = (w * h * RGBA8_BPP) as usize; // 32
    let mut map: std::collections::BTreeMap<u32, HostSurface> = Default::default();

    // Absent id.
    assert_eq!(get_from_with_gen(&map, &id, w, h), None);

    // Store an over-allocated (need + slop) buffer with a distinct host_gen.
    let mut bgra = vec![0xABu8; need + 16];
    bgra[need] = 0xCD; // a byte past `need` must never be returned
    map.insert(
        id,
        HostSurface {
            width: w,
            height: h,
            bgra: std::sync::Arc::new(bgra),
            host_gen: 9,
            ..Default::default()
        },
    );

    // Geometry mismatch on either axis must miss (no resize-serve).
    assert_eq!(get_from_with_gen(&map, &id, w + 1, h), None);
    assert_eq!(get_from_with_gen(&map, &id, w, h + 1), None);

    // Exact hit: exactly `need` bytes (slop truncated) + the entry host_gen.
    let (bytes, gen) = get_from_with_gen(&map, &id, w, h).expect("exact geom must hit");
    assert_eq!(bytes.len(), need, "must truncate to width*height*BPP");
    assert_eq!(gen, 9, "must report the entry host_gen");
    assert!(bytes.iter().all(|&b| b == 0xAB), "no slop byte leaks in");

    // get_from is the same content, generation dropped.
    assert_eq!(get_from(&map, &id, w, h), Some(bytes));

    // Empty bytes -> None even with matching geometry.
    map.get_mut(&id).unwrap().bgra = std::sync::Arc::new(Vec::new());
    assert_eq!(
        get_from_with_gen(&map, &id, w, h),
        None,
        "empty entry misses"
    );

    // Non-empty but short of `need` -> None (truncated store, no partial serve).
    map.get_mut(&id).unwrap().bgra = std::sync::Arc::new(vec![0xABu8; need - 1]);
    assert_eq!(
        get_from_with_gen(&map, &id, w, h),
        None,
        "under-`need` bytes must not be served",
    );
}

/// Ceding a mapping to its resident must stop the cache answering for it —
/// with a miss, never with the frame the Store superseded.
///
/// The whole point of the `skip_readback` rail is that no CPU copy of the new
/// frame exists, so anything still serving the *old* one is serving a frame
/// that is now a layer behind. `capture_present_frame` reads
/// `surface_cache::get` **before** it tries the resident, so a cession that
/// left the bytes in place would pin the display to the pre-Store frame for as
/// long as the rail stayed engaged, with nothing to report it.
///
/// The restore direction is asserted too: the flush writes through
/// `mapping_write::write_bgra8`, whose tail republishes this entry, and that
/// is what ends the cession. A cession that could not be ended would leave the
/// mapping permanently dependent on a resident that only a pin protects.
#[test]
fn a_ceded_surface_serves_a_miss_and_says_it_was_ceded() {
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let (w, h) = (4u32, 4u32);
    let need = (w * h * 4) as usize;
    store(&mut state, 7, w, h, vec![0xA1u8; need]);
    let before = state.host_surfaces.get(&7).map(|e| e.host_gen).unwrap();

    assert!(cede_surface_to_resident(&mut state, 7, w, h));
    assert_eq!(
        get(&state, 7, w, h),
        None,
        "a ceded entry must not serve the frame the Store superseded"
    );
    assert!(
        get_shared(&state, 7, w, h).is_none(),
        "the shared handle must miss wherever the slice does"
    );
    assert!(surface_ceded_to_resident(&state, 7, w, h));
    assert!(
        state.host_surfaces.get(&7).unwrap().host_gen != before,
        "the cession is a state change and must advance host_gen"
    );

    // A geometry this cache would not have stored anyway is refused, so the
    // arm can fail closed rather than leave a live entry contradicting a
    // resident-authoritative window.
    assert!(!cede_surface_to_resident(&mut state, 0, w, h));
    assert!(!cede_surface_to_resident(&mut state, 7, 0, h));
    assert!(!cede_surface_to_resident(
        &mut state,
        7,
        w,
        crate::model::MAX_SCANOUT_DIM + 1
    ));

    // The flush's republish ends it.
    store(&mut state, 7, w, h, vec![0xB2u8; need]);
    assert!(!surface_ceded_to_resident(&state, 7, w, h));
    assert_eq!(get(&state, 7, w, h).map(|b| b[0]), Some(0xB2));
}

/// A ceded entry is not the same thing as a stale-geometry one, and the
/// classifier must not confuse them.
///
/// Both make `get` miss, and folding them together would print
/// `have=4x4` against `want=4x4` on the LOAD-seed decline — a line that reads
/// as a contradiction rather than as the expected cost of the rail.
#[test]
fn cession_is_distinguishable_from_a_stale_geometry_entry() {
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    store(&mut state, 7, 8, 8, vec![0xA1u8; 8 * 8 * 4]);
    assert!(
        !surface_ceded_to_resident(&state, 7, 4, 4),
        "an entry at another geometry is stale, not ceded"
    );
    assert!(
        !surface_ceded_to_resident(&state, 9, 4, 4),
        "an absent entry is absent, not ceded"
    );
    assert!(cede_surface_to_resident(&mut state, 7, 4, 4));
    assert!(surface_ceded_to_resident(&state, 7, 4, 4));
    assert!(
        !surface_ceded_to_resident(&state, 7, 8, 8),
        "the cession is scoped to the geometry it was taken at"
    );
}

/// `get_shared` must hit exactly when `get` hits, including for a stored
/// buffer carrying slop past `width * height * 4`.
///
/// Returning `None` there would be a silent seed loss, and a missing Load
/// seed renders the pass onto a cleared target — a compositing layer going
/// solid black. The shared handle cannot be truncated the way `get`'s slice
/// is, so the slop case pays a copy instead of missing.
#[test]
fn get_shared_hits_wherever_get_hits_and_never_serves_slop() {
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let (w, h) = (4u32, 4u32);
    let need = (w * h * 4) as usize;

    // Exact: shared, and the same bytes `get` serves.
    store(&mut state, 7, w, h, vec![0xA1u8; need]);
    let exact = get_shared(&state, 7, w, h).expect("exact store must hit");
    assert_eq!(exact.len(), need);
    assert_eq!(&exact[..], get(&state, 7, w, h).unwrap());

    // Slop: still hits, truncated to the geometry the caller matched on —
    // the engine rejects a seed whose length is not exactly that.
    let mut slop = vec![0xB2u8; need + 16];
    slop[need] = 0xCD;
    state.host_surfaces.get_mut(&7).unwrap().bgra = std::sync::Arc::new(slop);
    let got = get_shared(&state, 7, w, h).expect("a store with slop must still hit");
    assert_eq!(got.len(), need, "must truncate to width*height*BPP");
    assert!(got.iter().all(|&b| b == 0xB2), "no slop byte leaks in");
    assert_eq!(&got[..], get(&state, 7, w, h).unwrap());

    // Geometry mismatch misses in both, identically.
    assert!(get_shared(&state, 7, w + 1, h).is_none());
    assert!(get(&state, 7, w + 1, h).is_none());
}

/// A depth-1 task page table where root PTE `i` points at PFN `PT_BASE + i`,
/// so a GVA of `i << PAGE_SHIFT_ARM64E` resolves to a page this test can
/// re-point by rewriting one PTE — which is exactly what the guest does when
/// it hands a virtual address to a different allocation.
fn setup_depth1_task(host: &mut FakeHost, state: &mut DeviceState) -> u64 {
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    const DIR_PFN: u32 = 2;
    const ROOT_PFN: u32 = 3;
    const PT_BASE: u32 = 4;
    let dir_gpa = (DIR_PFN as u64) << PAGE_SHIFT_ARM64E;
    let root_gpa = (ROOT_PFN as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], ROOT_PFN);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    for i in 0..16u32 {
        let pfn = PT_BASE + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        let _ = host.write_gpa(root_gpa + (i as u64) * 4, &pte);
    }
    state.define_task(1, 0x1000, DIR_PFN);
    root_gpa
}

fn repoint_pte(host: &mut FakeHost, root_gpa: u64, index: u64, pfn: u32) {
    use crate::contract::endian::st32;
    let mut pte = [0u8; 4];
    st32(&mut pte, pfn);
    let _ = host.write_gpa(root_gpa + index * 4, &pte);
}

/// An address that does not resolve records no backing, rather than a
/// backing of zero.
///
/// This is the shape the dense page list used to guard, carried over to the
/// first-page identity that replaced it. The list kept a `0` slot where a
/// page did not resolve so two mappings with holes in different places
/// could not read as the same one. A single recorded GPA has the sharper
/// version of that hazard: store `0` for an unresolvable page and every
/// unresolvable page compares equal to every other, so a later `Moved`
/// check answers `Same` for two unrelated allocations. `None` is the only
/// honest answer, and `gva_backing_state` reads it as `Unrecorded`.
#[test]
fn an_address_that_does_not_resolve_records_no_backing() {
    let mut host = FakeHost::new();
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let root_gpa = setup_depth1_task(&mut host, &mut st);
    let page = 1u64 << PAGE_SHIFT_ARM64E;
    let (w, h) = (64u32, 64u32);
    let gva = page;
    let pte_index = gva >> PAGE_SHIFT_ARM64E;

    let full = gva_backing(&st, &host, 1, gva, w, h).expect("walk resolves");
    assert_ne!(full.first_gpa, 0, "a resolved page is not the hole value");

    // Punt this page's PTE to an invalid PFN.
    repoint_pte(&mut host, root_gpa, pte_index, 0);
    assert!(
        gva_backing(&st, &host, 1, gva, w, h).is_none(),
        "an unresolvable address must yield no backing, never first_gpa=0"
    );

    // A zero geometry cannot name a span either.
    assert!(gva_backing(&st, &host, 1, gva, 0, h).is_none());
    assert!(gva_backing(&st, &host, 1, 0, w, h).is_none());
}

/// "Cannot tell" is its own answer, and the serve site is where conflating
/// it with "fresh" would cost a frame.
///
/// `the_backing_probe_separates_a_reassigned_address_from_an_unmapped_one`
/// pins Same/Moved/Unmapped through the map-wide sum, which now delegates
/// here, so those need no second statement. What only the per-entry answer
/// can be asked is the case the sum *skips*: an entry whose walk never
/// resolved, and a key that was never stored. The colour LOAD seed asks
/// about one address at a time, so it meets both, and a probe that answered
/// `Same` for either would report a clean result for a question it never
/// asked.
#[test]
fn a_backing_the_probe_cannot_read_is_not_a_fresh_one() {
    let mut host = FakeHost::new();
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_depth1_task(&mut host, &mut st);
    let page = 1u64 << PAGE_SHIFT_ARM64E;
    // 64x64 BGRA8 is exactly one 16 KiB page.
    let (w, h) = (64u32, 64u32);
    let gva = page;

    let backing = gva_backing(&st, &host, 1, gva, w, h).expect("walk resolves");
    store_gva_owned(
        &mut st,
        gva,
        w,
        h,
        vec![0xAB; (w * h * 4) as usize],
        0,
        Some(backing),
        true,
    );
    assert_eq!(
        gva_backing_state(&st, &host, gva),
        GvaBackingState::Same,
        "control: a store whose walk resolved reads its own pages back"
    );

    // Re-store the same key with no backing: the walk did not resolve, so
    // there is nothing to compare and the entry drops out of the census
    // denominator rather than counting as fresh.
    let px = vec![0xCD; (w * h * 4) as usize];
    store_gva_owned(&mut st, gva, w, h, px, 0, None, true);
    assert_eq!(
        gva_backing_state(&st, &host, gva),
        GvaBackingState::Unrecorded
    );
    assert_eq!(
        gva_backing_moved(&st, &host),
        (0, 0, 0),
        "and is not counted"
    );

    assert_eq!(
        gva_backing_state(&st, &host, gva + page),
        GvaBackingState::Unrecorded,
        "a key that was never stored is not an answer about backing"
    );
}

/// An entry the guest's own pages also hold is not a seed source.
///
/// Both copies start equal and only the guest's tracks the guest CPU, which
/// writes its own memory with no device operation at all. So this door can only
/// serve bytes that are the same or older, and "older" here is not a stale read
/// the next frame corrects: a `MTLLoadActionLoad` seed becomes the pass's prior
/// content and the matching Store writes it back into the guest's pages, so
/// whatever the CPU wrote in between is overwritten rather than missed once.
///
/// The refusal is deliberately narrow. An entry the guest's pages do *not* hold
/// is the only copy of those pixels — the writeback was refused and this map is
/// what the page-ownership guard promised would keep them — and it keeps
/// serving.
#[test]
fn the_seed_door_refuses_an_entry_the_guests_own_pages_hold() {
    use crate::model::GvaBacking;
    let mut host = FakeHost::new();
    let mut st = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_depth1_task(&mut host, &mut st);

    let gva = 1u64 << PAGE_SHIFT_ARM64E;
    for guest_holds in [false, true] {
        store_gva_owned(
            &mut st,
            gva,
            2,
            2,
            vec![0u8; 2 * 2 * 4],
            0,
            None,
            guest_holds,
        );
        st.host_gva_surfaces.get_mut(&gva).unwrap().backing = Some(GvaBacking {
            task_id: 1,
            first_gpa: 5u64 << PAGE_SHIFT_ARM64E,
        });
        assert_eq!(
            gva_seed_verdict(&st, &host, 1, gva),
            if guest_holds {
                GvaSeedVerdict::GuestHolds
            } else {
                GvaSeedVerdict::Admit
            },
            "guest_holds_bytes={guest_holds}"
        );
    }
}
