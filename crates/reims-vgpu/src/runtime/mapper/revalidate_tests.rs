//! Tests for mapper revalidation: when a captured page plan is re-checked.
//!
//! Out of line for the reason the sibling `runtime/` modules that already do
//! this have: colocated, this module and `tests` were together 3,038 of
//! `mapper.rs`'s 4,199 lines — 72% — so the IOSurface mapper itself was the
//! quarter of the file that was hardest to find.

use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_X86};
use crate::runtime::host::FakeHost;
use reims_vgpu_paging::geometry::{
    MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
    MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
};

#[test]
fn page_table_revalidation_slow_proxy_threshold_is_explicit() {
    assert!(!revalidate_timing_is_slow(REVALIDATE_SLOW_US - 1));
    assert!(revalidate_timing_is_slow(REVALIDATE_SLOW_US));
}

#[test]
fn fragmented_run_import_slow_proxy_threshold_is_explicit() {
    assert!(!mapping_run_import_is_slow(MAPPING_RUN_IMPORT_SLOW_US - 1));
    assert!(mapping_run_import_is_slow(MAPPING_RUN_IMPORT_SLOW_US));
}

#[test]
fn revalidate_fail_closed_without_internal_and_empty_pages() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();
    state.map_surface(2);
    // Mapped but no MappingInternal and no pages → not writable.
    assert!(!revalidate_mapping_pages(&mut state, &host, 2));
}

#[test]
fn revalidate_reason_disambiguates_the_miss() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();
    // Unknown id → the mapping was never created / already forgotten.
    assert_eq!(
        revalidate_mapping_reason(&mut state, &host, 7),
        Some("revalidate_gone")
    );
    // Mapped, no MappingInternal, no page list. The resolve never ran, so
    // this says nothing about whether the pages exist — which is exactly
    // what the old shared `revalidate_no_pages` slug hid behind a comment
    // calling it a benign (re)wire gap.
    state.map_surface(2);
    assert_eq!(
        revalidate_mapping_reason(&mut state, &host, 2),
        Some("revalidate_no_internal")
    );
    // Unmapped on entry is caught before the resolve and keeps its own slug.
    state.map_surface(3);
    state
        .surfaces
        .mappings
        .get_mut(&3)
        .unwrap()
        .lifecycle
        .active = false;
    assert_eq!(
        revalidate_mapping_reason(&mut state, &host, 3),
        Some("revalidate_unmapped")
    );
    // A resolvable static page list → success (None).
    state.map_surface(4);
    state.surfaces.mappings.get_mut(&4).unwrap().pages.entries =
        vec![(0x100 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    assert_eq!(revalidate_mapping_reason(&mut state, &host, 4), None);
}

#[test]
fn revalidate_accepts_static_page_list_without_internal() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let host = FakeHost::new();
    state.map_surface(4);
    {
        let m = state.surfaces.mappings.get_mut(&4).unwrap();
        m.pages.entries = vec![(0x100 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(revalidate_mapping_pages(&mut state, &host, 4));
}

#[test]
fn mapping_io_still_rejects_non_ram_page_at_map_boundary() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let mid = 6;
    assert!(state.map_surface(mid));
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.pages.entries = vec![(0x7f000 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    let mut byte = [0u8; 1];
    assert!(!read_mapping_bytes(
        &mut state, &mut host, mid, 0, &mut byte,
    ));
    let vouched = vouch_mapping_pages_verdict(&mut state, &host, mid)
        .1
        .expect("no walk to contradict");
    assert!(!write_mapping_bytes(
        &mut state,
        &mut host,
        mid,
        0,
        &[1],
        &vouched
    ));
}

#[test]
fn invalidate_mapping_pages_bumps_map_generation_and_clears() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    state.map_surface(5);
    let import_id = {
        let m = state.surfaces.mappings.get_mut(&5).unwrap();
        m.pages.entries = vec![1];
        let import = std::sync::Arc::new(
            crate::runtime::guest_ram::GuestRamImport::new_host_allocation(0xdead_0000, 4096, 4096)
                .expect("synthetic aligned import"),
        );
        let import_id = import.id();
        let mut view = crate::model::SurfaceHostView::new(
            0xdead,
            4096,
            crate::runtime::guest_ram::GuestPageFootprint::new([0x1000].into(), 4096)
                .expect("synthetic footprint"),
        )
        .expect("valid host view");
        assert!(view.replace_import(import).is_none());
        m.materialization.install(view);
        import_id
    };
    let gen0 = state
        .surfaces
        .mappings
        .get(&5)
        .unwrap()
        .lifecycle
        .generation;
    assert!(state.invalidate_mapping_pages(5).had_page_state);
    let m = state.surfaces.mappings.get(&5).unwrap();
    assert!(m.pages.entries.is_empty());
    assert!(!m.materialization.has_view());
    assert!(m.lifecycle.generation != gen0);
    assert_eq!(
        state.host_materializations.queued_views(),
        vec![(0xdead, 4096)]
    );
    assert_eq!(
        state.host_materializations.queued_guest_imports(),
        vec![import_id]
    );
}

/// Invalidating a mapping's pages drops the host-side copy of those pages.
///
/// The cached frame is keyed by `(mapping_id, geometry)` with no
/// `map_generation` in it, so the bump that disqualifies the resident, the
/// contiguous view and every armed window leaves this entry addressable and
/// still holding pixels read through the page list that just stopped being
/// the surface's. The sampled ladder's host-cache rung then serves it: its
/// only currency test is the guest-write witness, and the token this call
/// retires makes that witness answer `NoStamp`, which is deliberately not
/// treated as evidence of a guest write.
///
/// Without the drop this asserts nothing about a rung — it asserts that a
/// copy of invalidated pages is still readable, which is the whole defect.
#[test]
fn invalidate_mapping_pages_drops_the_host_cache_of_those_pages() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    state.map_surface(5);
    {
        let m = state.surfaces.mappings.get_mut(&5).unwrap();
        m.pages.entries = vec![1];
        m.publish_geometry_for_test(2, 2, 0);
    }
    crate::runtime::surface_cache::store(&mut state, 5, 2, 2, vec![0xab; 2 * 2 * 4]);
    assert!(
        crate::runtime::surface_cache::get_shared(&state, 5, 2, 2).is_some(),
        "the cache entry under test was never stored"
    );

    let _ = state.invalidate_mapping_pages(5);

    assert!(
        crate::runtime::surface_cache::get_shared(&state, 5, 2, 2).is_none(),
        "a host copy of the invalidated pages is still being served"
    );
}

/// A host refusal is derived once per page list, not once per call, and a new
/// page list re-asks the host.
///
/// Before this cache every call on a fragmented mapping rebuilt the page-GPA
/// vector, rescanned it for runs, and emitted a line — 471 757 of them in
/// one 2 900 s boot, the only prefix ever to trip `log_flood_detected`. The
/// line count is therefore the assertion: repeated calls must add none, and
/// the magnitude the old line carried must still be readable as `served=`.
#[test]
fn a_host_refusal_is_cached_but_fragmentation_is_still_offered_to_it() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    let entry =
        |gpa: u64| ((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT | PAGE_ENTRY_VALID;
    // Non-adjacent guest pages — never one packed run.
    let (gpa0, gpa1, gpa2) = (0x1000_0000u64, 0x2000_0000u64, 0x3000_0000u64);
    for gpa in [gpa0, gpa1, gpa2] {
        host.map_range(gpa, page as usize, 0);
    }
    let mid = 9u32;
    state.map_surface(mid);
    state.surfaces.mappings.get_mut(&mid).unwrap().pages.entries = vec![entry(gpa0), entry(gpa1)];

    let cap = crate::observe::FailCapture::start();
    let lines = || -> Vec<String> {
        cap.lines()
            .into_iter()
            .filter(|l| l.starts_with("OFF contig_view_refused"))
            .collect()
    };
    for _ in 0..16 {
        assert!(
            ensure_contig_view(&mut state, &mut host, mid).is_none(),
            "this fixture models a host that declines scattered aliases"
        );
    }
    let first = lines();
    assert_eq!(
        first.len(),
        1,
        "16 calls on one page list must derive (and say) the verdict once: {first:?}"
    );
    assert!(
        first[0].contains(" pages=2 physical_runs=2 "),
        "the derived line keeps its shape: {}",
        first[0]
    );

    // A different page list is a different verdict: the generation bump that
    // retires `contig_ptr` must also retire this.
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.pages.entries = vec![entry(gpa0), entry(gpa1), entry(gpa2)];
        crate::model::DeviceState::bump_page_generation(m);
    }
    assert!(ensure_contig_view(&mut state, &mut host, mid).is_none());
    let after = lines();
    assert_eq!(
        after.len(),
        2,
        "a new page list must re-derive and re-report: {after:?}"
    );
    assert!(
        after[1].contains(" pages=3 physical_runs=3 "),
        "the second line describes the second list: {}",
        after[1]
    );

    // Magnitude survives deduplication: `served` counts every fragmented
    // answer, cached ones included, so it advanced by the 16 calls between
    // the two derivations.
    let served = |l: &str| -> u64 {
        l.rsplit_once("served=")
            .and_then(|(_, v)| v.split_whitespace().next())
            .and_then(|v| v.parse().ok())
            .expect("line carries served=")
    };
    assert_eq!(
        served(&after[1]) - served(&after[0]),
        16,
        "served must count cached answers too: {after:?}"
    );
    assert_eq!(
        host.map_pages_calls, 2,
        "each page-list generation must reach the host exactly once"
    );
}

/// A host without a scattered-alias primitive declines the full view;
/// `write_mapping_bytes` still lands bytes via maximal packed runs.
#[test]
fn multi_import_fragmented_mapping_write() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    // Two non-adjacent guest pages (gap in GPA → not one packed map_pages).
    let gpa0 = 0x1000_0000u64;
    let gpa1 = 0x2000_0000u64;
    host.map_range(gpa0, page as usize, 0);
    host.map_range(gpa1, page as usize, 0);
    let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
    let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
    let mid = 9u32;
    state.map_surface(mid);
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.pages.entries = vec![
            (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    assert!(
        ensure_contig_view(&mut state, &mut host, mid).is_none(),
        "fragmented list must not pack under strict_linux_map"
    );
    let payload = b"FRAG-MULTI-IMPORT-OK!!!!"; // 24 bytes
    let vouched = vouch_mapping_pages_verdict(&mut state, &host, mid)
        .1
        .expect("no walk to contradict");
    assert!(write_mapping_bytes(
        &mut state, &mut host, mid, 0, payload, &vouched
    ));
    // Second page offset = page_size.
    let mut hi = [0u8; 8];
    assert!(read_mapping_bytes(
        &mut state, &mut host, mid, page, &mut hi
    ));
    // Write only touched page 0; page 1 still zero.
    assert_eq!(hi, [0u8; 8]);
    let mut lo = [0u8; 24];
    assert!(read_mapping_bytes(&mut state, &mut host, mid, 0, &mut lo));
    assert_eq!(&lo[..], &payload[..]);
    // Cross-page write spanning the gap.
    let cross = vec![0xABu8; 16];
    let off = page - 8;
    assert!(write_mapping_bytes(
        &mut state, &mut host, mid, off, &cross, &vouched
    ));
    let mut check = [0u8; 16];
    assert!(read_mapping_bytes(
        &mut state, &mut host, mid, off, &mut check
    ));
    assert_eq!(check, [0xABu8; 16]);
}

/// The mapping rail's writes reach `observe::footprint`, and claim exactly
/// the frames they wrote.
///
/// This is the positive control the footprint's own unit tests cannot be:
/// those drive the bit set directly, so they prove the container works and
/// say nothing about whether any rail is wired to it. A footprint fed by no
/// rail reports an empty set, and an empty set answers "this device never
/// wrote that page" to every question — the exoneration that costs nothing
/// to produce and cannot be told from a real one.
///
/// The negatives are the load-bearing half. A mapping's page list is a
/// scatter, so the tempting implementation — mark `[gpa_lo, gpa_hi]` over
/// the write's span — would claim the 64 Ki frames between these two pages,
/// none of which this device can reach through this mapping. Every one of
/// them would then read as a hit for the rest of the boot, against a guest
/// panic that had nothing to do with us.
#[test]
fn a_mapping_write_marks_its_own_frames_and_not_the_gap_between_them() {
    use crate::observe::footprint;

    let _fp = footprint::exclusive_for_tests();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    let gpa0 = 0x1000_0000u64;
    let gpa1 = 0x2000_0000u64;
    host.map_range(gpa0, page as usize, 0);
    host.map_range(gpa1, page as usize, 0);
    let mid = 9u32;
    state.map_surface(mid);
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.pages.entries = vec![
            (((gpa0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (((gpa1 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    let vouched = vouch_mapping_pages_verdict(&mut state, &host, mid)
        .1
        .expect("no walk to contradict");

    // 24 bytes at offset 0: page 0 only.
    assert!(write_mapping_bytes(
        &mut state,
        &mut host,
        mid,
        0,
        b"FOOTPRINT-FIRST-PAGE!!!!",
        &vouched
    ));
    assert!(footprint::wrote_gpa(gpa0), "the destination must be marked");
    assert!(
        !footprint::wrote_gpa(gpa1),
        "a write that never reached page 1 must not claim it"
    );
    assert!(
        !footprint::wrote_gpa((gpa0 + gpa1) / 2),
        "the hull between the two pages belongs to whoever the guest gave it \
         to; claiming it would make every later `pn` in that range a hit"
    );
    assert_eq!(footprint::counts(), (1, 0));

    // Cross-page: now both, and still nothing between them.
    assert!(write_mapping_bytes(
        &mut state,
        &mut host,
        mid,
        page - 8,
        &[0xABu8; 16],
        &vouched
    ));
    assert!(footprint::wrote_gpa(gpa1), "the second page is reached now");
    assert!(!footprint::wrote_gpa((gpa0 + gpa1) / 2));
    assert_eq!(
        footprint::counts(),
        (2, 0),
        "two frames total, not the span between them"
    );
}

/// The same claim for the contiguous-view fast path, which is the one
/// production takes.
///
/// The two paths mark through different code — the fast path resolves the
/// destination from the mapping's page list because it has only a host
/// pointer and an offset, the slow path from each packed run it maps — so a
/// control over one says nothing about the other. The fast path is where
/// nearly all of the ~100 000 mapping writes a driven boot makes actually
/// go, so an unmarked one there is most of the footprint missing.
#[test]
fn a_contiguous_mapping_write_marks_only_the_pages_its_offset_reaches() {
    use crate::observe::footprint;

    let _fp = footprint::exclusive_for_tests();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page = 1u64 << PAGE_SHIFT_X86;
    // Adjacent, so `ensure_contig_view` packs them into one view.
    let gpa0 = 0x3000_0000u64;
    let gpa1 = gpa0 + page;
    host.map_range(gpa0, 2 * page as usize, 0);
    let mid = 11u32;
    state.map_surface(mid);
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.pages.entries = vec![
            (((gpa0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (((gpa1 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    let (_, _, pages) = ensure_contig_view_with_pages(&mut state, &mut host, mid)
        .expect("the fixture must take the fast path or it is testing the other one");
    assert_eq!(&*pages, &[gpa0, gpa1]);
    let (_, _, reused) = ensure_contig_view_with_pages(&mut state, &mut host, mid)
        .expect("the retained view remains live");
    assert!(
        std::sync::Arc::ptr_eq(&pages, &reused),
        "resource synchronization must retain the admitted footprint, not rebuild it"
    );
    let vouched = vouch_mapping_pages_verdict(&mut state, &host, mid)
        .1
        .expect("no walk to contradict");

    // Entirely inside page 1: the offset, not the base, decides the frame.
    assert!(write_mapping_bytes(
        &mut state,
        &mut host,
        mid,
        page + 16,
        &[0x5Au8; 32],
        &vouched
    ));
    assert!(footprint::wrote_gpa(gpa1));
    assert!(
        !footprint::wrote_gpa(gpa0),
        "marking from the mapping's base rather than the write's offset \
         would claim page 0, which this write never touched"
    );
    assert_eq!(footprint::counts(), (1, 0));
}
