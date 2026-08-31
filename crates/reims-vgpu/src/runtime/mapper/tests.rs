//! Tests for the IOSurface mapper capture and page-table resolve.
//!
//! Out of line for the reason the sibling `runtime/` modules that already do
//! this have: colocated, this module and `revalidate_tests` were together 3,038 of
//! `mapper.rs`'s 4,199 lines — 72% — so the IOSurface mapper itself was the
//! quarter of the file that was hardest to find.

use super::selected_within;

fn sel(ranges: Option<&[(u64, u64)]>, lo: u64, hi: u64) -> Vec<(u64, u64)> {
    selected_within(ranges, lo, hi).collect()
}

/// `None` is the whole window and `Some(&[])` is none of it. The two must
/// never collapse: a caller with an explicit list of ranges to write hands
/// over an empty one when there is nothing to write, and reading that as
/// "everything" would make the cheapest landing the most expensive one.
#[test]
fn an_absent_selection_and_an_empty_one_are_opposites() {
    assert_eq!(sel(None, 10, 20), vec![(10, 20)]);
    assert_eq!(sel(Some(&[]), 10, 20), vec![]);
    // An empty window selects nothing either way.
    assert_eq!(sel(None, 10, 10), vec![]);
}

/// Each selected range is clipped to the window, and ranges wholly outside
/// it contribute nothing.
#[test]
fn a_selection_is_clipped_to_the_window() {
    let ranges = [(0u64, 8u64), (16, 24), (32, 40)];
    assert_eq!(sel(Some(&ranges), 4, 20), vec![(4, 8), (16, 20)]);
    assert_eq!(sel(Some(&ranges), 8, 16), vec![]);
    assert_eq!(sel(Some(&ranges), 0, 64), vec![(0, 8), (16, 24), (32, 40)]);
    assert_eq!(sel(Some(&ranges), 40, 64), vec![]);
    // A window inside one range yields exactly that window.
    assert_eq!(sel(Some(&ranges), 18, 22), vec![(18, 22)]);
}

/// The binary search must land on the first range whose *end* is past the
/// window start, not the first whose start is. A range straddling the
/// window's left edge is the case a `start >= lo` search silently drops,
/// and dropping it loses guest pixels rather than failing.
#[test]
fn a_range_straddling_the_window_start_is_not_skipped() {
    let ranges = [(0u64, 100u64), (200, 300)];
    assert_eq!(sel(Some(&ranges), 50, 250), vec![(50, 100), (200, 250)]);
}

/// Queries need not ascend — the walk re-finds its start each time — so a
/// caller may probe page runs in whatever order its page list gives them.
#[test]
fn windows_may_be_queried_out_of_order() {
    let ranges = [(0u64, 8u64), (16, 24), (32, 40)];
    assert_eq!(sel(Some(&ranges), 32, 40), vec![(32, 40)]);
    assert_eq!(sel(Some(&ranges), 0, 8), vec![(0, 8)]);
    assert_eq!(sel(Some(&ranges), 16, 24), vec![(16, 24)]);
}

/// A type-4 surface can be re-pointed by the guest with no packet at all,
/// and this is the only thing that can see it.
///
/// The gap is precise. `revalidate_mapping_reason` re-resolves only when
/// `mapping_internal != 0`; a type-4 surface has none, so that function
/// falls through to `mapped && !page_entries.is_empty()` and answers
/// "resolvable" without checking anything. The guest then re-points the
/// backing in its own page table — no MapMemory2, no UnmapMemory, no
/// ReplacePhysical, so `map_generation` does not move — and the cached list
/// stays trusted. Every deferred flush after that writes a framebuffer into
/// whatever now owns those pages.
///
/// The fixture is exactly that sequence: adopt a list walked through a live
/// task page table, rewire the PTE behind the device's back, and require the
/// answer to flip. The final assertion is the one that keeps the check
/// honest — restore the PTE and it must go back to `true`, because a witness
/// that always says "drifted" would pass the first half and refuse every
/// legitimate write in production.
#[test]
fn the_page_witness_sees_a_rewire_no_packet_announced() {
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::model::{DeviceId, Type4Walk, PAGE_SHIFT_X86};
    use crate::runtime::host::{FakeHost, HostMemory};

    let page = 1u64 << PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data0 = 4u64 << PAGE_SHIFT_X86;
    let data1 = 10u64 << PAGE_SHIFT_X86;
    for gpa in [dir_gpa, root_gpa, data0, data1] {
        host.map_range(gpa, page as usize, 0);
    }
    let st32 = |b: &mut [u8], v: u32| b[..4].copy_from_slice(&v.to_le_bytes());
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &d).unwrap();
    // GVA page 0 of the task translates to `data0`.
    let mut pte = [0u8; 4];
    st32(&mut pte, (data0 >> PAGE_SHIFT_X86) as u32);
    host.write_gpa(root_gpa, &pte).unwrap();

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.define_task(1, page, 2);
    state.map_surface(6);
    {
        let m = state.mappings.get_mut(&6).unwrap();
        m.mapped = true;
        m.page_entries =
            vec![(((data0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        m.type4_walk = Some(Type4Walk {
            task_id: 1,
            backing_pfn: 0,
            map_generation: m.map_generation,
        });
    }
    assert!(
        super::type4_pages_witness(&state, &host, 6) == super::Type4Witness::Verified,
        "the list was just walked from this page table"
    );

    // The guest re-points the backing. Nothing on the wire says so, and
    // `map_generation` is untouched — which is the whole defect.
    let generation_before = state.mappings.get(&6).unwrap().map_generation;
    st32(&mut pte, (data1 >> PAGE_SHIFT_X86) as u32);
    host.write_gpa(root_gpa, &pte).unwrap();
    assert_eq!(
        state.mappings.get(&6).unwrap().map_generation,
        generation_before,
        "no packet arrived, so nothing bumped the incarnation"
    );
    assert!(
        super::type4_pages_witness(&state, &host, 6) == super::Type4Witness::Drifted,
        "a fresh walk names a different page, and a writeback through the \
         cached one lands in whatever the guest gave it to"
    );

    // A latch from a superseded incarnation is not evidence about the list
    // in hand, so it must not be read as drift.
    {
        let m = state.mappings.get_mut(&6).unwrap();
        let mut walk = m.type4_walk.unwrap();
        walk.map_generation = m.map_generation.wrapping_sub(1);
        m.type4_walk = Some(walk);
    }
    assert!(
        super::type4_pages_witness(&state, &host, 6)
            == super::Type4Witness::Unwitnessed("walk_superseded"),
        "a stale latch says nothing about the current list — it must not refuse, \
         and it must not be counted as a verification either"
    );

    // And the check must be able to say yes: put the translation back and
    // the same list is legitimate again. A witness that only ever refuses
    // would pass the assertion above and lose every frame in production.
    {
        let m = state.mappings.get_mut(&6).unwrap();
        let mut walk = m.type4_walk.unwrap();
        walk.map_generation = m.map_generation;
        m.type4_walk = Some(walk);
    }
    st32(&mut pte, (data0 >> PAGE_SHIFT_X86) as u32);
    host.write_gpa(root_gpa, &pte).unwrap();
    assert!(
        super::type4_pages_witness(&state, &host, 6) == super::Type4Witness::Verified,
        "the translation is back where the list says it is"
    );
}

/// "Nothing to check" must not be reported as "checked and fine".
///
/// Four of this witness's exits check no pages at all, and every caller used
/// to collapse them into the same `PagesVerdict::Ours` a full clean re-walk
/// produces. The counters built on that cannot distinguish a guard that
/// passed from one that was never armed — opposite claims about the
/// write-after-free class — and one boot reported `mapw_pages_vouched`
/// 29 002 against `mapw_pages_refused` 0 without being able to say which it
/// meant.
///
/// This pins the split, and pins that it is measurement only: an
/// `Unwitnessed` verdict still issues a `PagesVouched` token, so the write
/// proceeds exactly as before.
#[test]
fn a_mapping_with_nothing_to_check_is_not_counted_as_verified() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;

    let host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.map_surface(6);
    {
        let m = state.mappings.get_mut(&6).unwrap();
        m.mapped = true;
        m.page_entries = vec![(4u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        // No `type4_walk`: nothing ever latched a walk for this mapping.
        m.type4_walk = None;
    }
    assert_eq!(
        super::type4_pages_witness(&state, &host, 6),
        super::Type4Witness::Unwitnessed("no_walk"),
        "a list nothing walked is unwitnessed, not verified"
    );

    // An empty list and an absent mapping are their own states, because a
    // single slug would make four different gaps look like one.
    state.mappings.get_mut(&6).unwrap().page_entries.clear();
    assert_eq!(
        super::type4_pages_witness(&state, &host, 6),
        super::Type4Witness::Unwitnessed("no_walk"),
    );
    assert_eq!(
        super::type4_pages_witness(&state, &host, 999),
        super::Type4Witness::Unwitnessed("no_mapping"),
    );

    // Policy is unchanged: the verdict still hands back a token, so the
    // writers this gates still write. Only the counter differs.
    let verdict = super::mapping_pages_verdict(&mut state, &host, 6);
    assert!(
        matches!(verdict, super::PagesVerdict::Unwitnessed(_)),
        "got {verdict:?}"
    );
    let (verdict, token) = super::vouch_mapping_pages_verdict(&mut state, &host, 6);
    assert!(matches!(verdict, super::PagesVerdict::Unwitnessed(_)));
    assert!(
        token.is_some(),
        "an unwitnessed list still writes — this split is a measurement, \
         not a new refusal"
    );
}

/// A page the task cannot translate at all is a different finding from one
/// that translates somewhere new, and the witness has to say which.
///
/// The failed walk used to be answered with the GVA, to match the identity
/// fallback that built such entries. Both outcomes refuse the write, so the
/// substitution cost nothing in safety — it cost the diagnosis. Every page
/// whose walk failed was written up as `translation_moved`, "the guest
/// re-pointed this surface and no packet said so", when the guest had done
/// nothing at all.
#[test]
fn a_page_the_task_cannot_translate_is_not_reported_as_a_move() {
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::model::{DeviceId, Type4Walk, PAGE_SHIFT_X86};
    use crate::runtime::host::{FakeHost, HostMemory};

    crate::observe::redirect_logs_for_tests();
    let page = 1u64 << PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data0 = 4u64 << PAGE_SHIFT_X86;
    for gpa in [dir_gpa, root_gpa, data0] {
        host.map_range(gpa, page as usize, 0);
    }
    let st32 = |b: &mut [u8], v: u32| b[..4].copy_from_slice(&v.to_le_bytes());
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &d).unwrap();
    let mut pte = [0u8; 4];
    st32(&mut pte, (data0 >> PAGE_SHIFT_X86) as u32);
    host.write_gpa(root_gpa, &pte).unwrap();

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.define_task(1, page, 2);
    state.map_surface(6);
    {
        let m = state.mappings.get_mut(&6).unwrap();
        m.mapped = true;
        m.page_entries =
            vec![(((data0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        m.type4_walk = Some(Type4Walk {
            task_id: 1,
            backing_pfn: 0,
            map_generation: m.map_generation,
        });
    }
    assert!(super::type4_pages_witness(&state, &host, 6) != super::Type4Witness::Drifted);

    // The translation goes away rather than moving: the PTE is cleared, so
    // the walk fails outright.
    let at = std::fs::read_to_string(crate::observe::fail_log_path())
        .unwrap_or_default()
        .len();
    st32(&mut pte, 0);
    host.write_gpa(root_gpa, &pte).unwrap();
    assert!(
        super::type4_pages_witness(&state, &host, 6) == super::Type4Witness::Drifted,
        "a page the table cannot translate must not vouch for a write"
    );
    let body = std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default();
    let fresh: Vec<&str> = body[at.min(body.len())..]
        .lines()
        .filter(|l| l.starts_with("mapping_page_drift "))
        .collect();
    assert!(
        fresh.iter().any(|l| l.contains("reason=no_translation")),
        "the refusal must name the failed walk, got: {fresh:?}"
    );
    assert!(
        !fresh.iter().any(|l| l.contains("reason=translation_moved")),
        "nothing moved — blaming the guest here is the bug: {fresh:?}"
    );
}

/// Every page of a multi-page surface is checked, including the ones in the
/// middle.
///
/// The witness resolves the whole run through one root read and one walk
/// cache rather than descending the table per page, which is what makes the
/// licence check for a 1080p writeback cost one walk instead of two
/// thousand. The hazard that shape introduces is a cache that answers for
/// page N with what it read for page N-1, and the symptom would be silent:
/// a surface the guest re-pointed in the middle would vouch, and the next
/// flush would write a frame into whatever now owns those pages.
///
/// So the page that moves here is deliberately neither the first nor the
/// last. The single-page test above cannot see this class at all.
#[test]
fn a_page_moved_in_the_middle_of_a_run_still_refuses_the_write() {
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::model::{DeviceId, Type4Walk, PAGE_SHIFT_X86};
    use crate::runtime::host::{FakeHost, HostMemory};

    crate::observe::redirect_logs_for_tests();
    const PAGES: usize = 5;
    let page = 1u64 << PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    // Data pages 4..9, plus one more the guest can re-point a PTE at.
    let data_pfn = |i: usize| 4u32 + i as u32;
    let elsewhere_pfn = 4u32 + PAGES as u32;
    for pfn in [2u32, 3].into_iter().chain(4..=elsewhere_pfn) {
        host.map_range(u64::from(pfn) << PAGE_SHIFT_X86, page as usize, 0);
    }
    let st32 = |b: &mut [u8], v: u32| b[..4].copy_from_slice(&v.to_le_bytes());
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &d).unwrap();
    let write_pte = |host: &mut FakeHost, index: usize, pfn: u32| {
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        host.write_gpa(root_gpa + (index as u64) * 4, &pte).unwrap();
    };
    for i in 0..PAGES {
        write_pte(&mut host, i, data_pfn(i));
    }

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.define_task(1, page, 2);
    state.map_surface(6);
    {
        let m = state.mappings.get_mut(&6).unwrap();
        m.mapped = true;
        m.page_entries = (0..PAGES)
            .map(|i| (data_pfn(i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
            .collect();
        m.type4_walk = Some(Type4Walk {
            task_id: 1,
            backing_pfn: 0,
            map_generation: m.map_generation,
        });
    }
    assert_eq!(
        super::type4_pages_witness(&state, &host, 6),
        super::Type4Witness::Verified,
        "a run whose every page still translates where it was walked must vouch"
    );

    // Re-point page 2 of 5 and nothing else. A per-page walk and a cached
    // one must reach the same verdict; only one of them can get this wrong.
    let at = std::fs::read_to_string(crate::observe::fail_log_path())
        .unwrap_or_default()
        .len();
    write_pte(&mut host, 2, elsewhere_pfn);
    assert_eq!(
        super::type4_pages_witness(&state, &host, 6),
        super::Type4Witness::Drifted,
        "a page re-pointed in the middle of the run must refuse the write"
    );
    let body = std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default();
    let fresh: Vec<&str> = body[at.min(body.len())..]
        .lines()
        .filter(|l| l.starts_with("mapping_page_drift "))
        .collect();
    assert!(
        fresh.iter().any(|l| l.contains("reason=translation_moved")),
        "the refusal must name the move, got: {fresh:?}"
    );
    assert!(
        fresh.iter().any(|l| l.contains("page=2/5")),
        "the refusal must name which page moved, got: {fresh:?}"
    );
}

/// A task whose page table is gone refuses rather than vouching on a walk
/// that visited nothing.
///
/// The bulk visitor returns without calling back at all for an inactive
/// task, so "saw no disagreement" and "checked nothing" are the same silence
/// from inside the loop. Counting what it visited is what separates them,
/// and a witness that skipped that count would vouch for every page of every
/// surface the instant a task went away.
#[test]
fn a_walk_that_visits_nothing_is_not_agreement() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::model::{DeviceId, Type4Walk, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;

    crate::observe::redirect_logs_for_tests();
    let host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    // A task id that was never defined: no directory, so nothing to walk.
    state.map_surface(6);
    {
        let m = state.mappings.get_mut(&6).unwrap();
        m.mapped = true;
        m.page_entries = vec![(4u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID; 3];
        m.type4_walk = Some(Type4Walk {
            task_id: 1,
            backing_pfn: 0,
            map_generation: m.map_generation,
        });
    }
    assert_eq!(
        super::type4_pages_witness(&state, &host, 6),
        super::Type4Witness::Drifted,
        "a page list nothing walked must not vouch for a write"
    );
}

/// The dirty bitmap answers in guest physical pages and a writeback lays
/// bytes out in mapping offsets; this is the only place the two meet, so it
/// is tested against a page list that is deliberately not in address order.
///
/// A GPA the mapping does not hold contributes nothing. A token is per page
/// list, but an answer can be taken across a rebind, and inventing an offset
/// for a page this surface does not own would exclude bytes at random.
#[test]
fn mapping_offsets_of_pages_maps_guest_pages_to_surface_offsets() {
    use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    const PAGE: u64 = 1 << PAGE_SHIFT_X86;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    state.map_surface(6);
    // Surface pages 0..4 live at PFNs 0x50, 0x53, 0x51, 0x52 — out of
    // address order, which is what makes the index the answer and not the
    // address.
    let pfns = [0x50u32, 0x53, 0x51, 0x52];
    state.mappings.get_mut(&6).unwrap().page_entries = pfns
        .iter()
        .map(|p| (p << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
        .collect();
    let gpa = |pfn: u32| (pfn as u64) << PAGE_SHIFT_X86;

    assert_eq!(mapping_offsets_of_pages(&state, 6, &[]), vec![]);
    // Surface page 1 is the highest GPA of the four.
    assert_eq!(
        mapping_offsets_of_pages(&state, 6, &[gpa(0x53)]),
        vec![(PAGE, 2 * PAGE)]
    );
    // Adjacent surface pages merge even though their GPAs are not adjacent.
    assert_eq!(
        mapping_offsets_of_pages(&state, 6, &[gpa(0x51), gpa(0x52)]),
        vec![(2 * PAGE, 4 * PAGE)]
    );
    // Non-adjacent stay apart, and a page this surface does not own is
    // ignored rather than placed somewhere.
    assert_eq!(
        mapping_offsets_of_pages(&state, 6, &[gpa(0x50), gpa(0x52), gpa(0x99)]),
        vec![(0, PAGE), (3 * PAGE, 4 * PAGE)]
    );
    assert_eq!(mapping_offsets_of_pages(&state, 7, &[gpa(0x50)]), vec![]);
}

use super::*;
use crate::contract::endian::st32;
use crate::contract::iosurface_pages::{
    MAPPING_INTERNAL_BACKPTR, MAPPING_INTERNAL_EXPECTED_SIZE, MAPPING_INTERNAL_ID,
    MAPPING_INTERNAL_SIZE, PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
};

/// The span is a hull over the *resolvable* entries, in PFN order-independent
/// fashion, and it must not be fooled by an unsorted list or by an invalid
/// entry sitting at either end — those are exactly the shapes a real page
/// list has, and a span that tracked first/last instead of min/max would
/// name a range that does not contain the pages written.
#[test]
fn entry_gpa_span_is_a_hull_over_resolvable_entries() {
    let valid = |pfn: u32| (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    // Unsorted, so first/last != min/max.
    let entries = [valid(9), valid(3), valid(7)];
    assert_eq!(
        entry_gpa_span(&entries, 12),
        Some((3u64 << 12, 9u64 << 12)),
        "min/max, not first/last"
    );
    // An invalid entry is skipped, not treated as GPA 0 — a zero would drag
    // `lo` to the bottom of RAM and make the span claim pages no write can
    // reach, which is the wrong direction for a bound used as evidence.
    let with_hole = [0u32, valid(3), valid(9), 0u32];
    assert_eq!(
        entry_gpa_span(&with_hole, 12),
        Some((3u64 << 12, 9u64 << 12))
    );
    // Page shift is honoured rather than assumed to be 12 (arm64 is 14).
    assert_eq!(entry_gpa_span(&entries, 14), Some((3u64 << 14, 9u64 << 14)));
    // Nothing resolvable ⇒ no span at all, rather than (u64::MAX, 0).
    assert_eq!(entry_gpa_span(&[0, 0], 12), None);
    assert_eq!(entry_gpa_span(&[], 12), None);
}

#[test]
fn plan_adoption_decision_incarnation_semantics() {
    // No condemn: plain pages-changed compare against the live entries.
    assert_eq!(
        plan_adoption_decision(None, &[1, 2], &[1, 2]),
        (false, false, false)
    );
    assert_eq!(
        plan_adoption_decision(None, &[1, 2], &[1, 3]),
        (true, false, false)
    );
    // Condemned + identical plan = stale delete reprieve: the SAME
    // incarnation lives on — no bump, no drop (black-band class).
    assert_eq!(
        plan_adoption_decision(Some(&[1, 2]), &[], &[1, 2]),
        (false, false, true)
    );
    // Condemned + different plan = the backing really died and the id
    // was re-used: bump + drop the old incarnation's windows.
    assert_eq!(
        plan_adoption_decision(Some(&[1, 2]), &[], &[7, 8]),
        (true, true, false)
    );
    // The live (cleared) entries never mask the fingerprint compare.
    assert_eq!(
        plan_adoption_decision(Some(&[1, 2]), &[7, 8], &[1, 2]),
        (false, false, true)
    );
}

#[test]
fn resolve_fail_latch_dedups_per_mapping_and_rearms_on_clear() {
    // Flood guard for the per-present `resolve_mapping_backing` path: a
    // genuinely-broken mapping must log each reason once, re-arm when it
    // resolves, and never bleed across mappings. Unique ids so this never
    // races real mappings across the process-global latch.
    let mid = 0xF00D_0001u32;
    let other = 0xF00D_0002u32;
    clear_resolve_fail(mid);
    clear_resolve_fail(other);
    let seen = |m: u32, r: &'static str| {
        resolve_fail_latch()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&(m, r))
    };
    note_resolve_fail(mid, "iosurface_validate_mapping_id_mismatch", "x".into());
    assert!(seen(mid, "iosurface_validate_mapping_id_mismatch"));
    // A different reason on the same mapping is tracked independently.
    note_resolve_fail(mid, "iosurface_mapper_internal_owner_read", "x".into());
    assert!(seen(mid, "iosurface_mapper_internal_owner_read"));
    // A different mapping is untouched by mid's failures.
    assert!(!seen(other, "iosurface_validate_mapping_id_mismatch"));
    // Clearing mid re-arms both its reasons but leaves `other` alone.
    note_resolve_fail(other, "iosurface_validate_mapping_id_mismatch", "x".into());
    clear_resolve_fail(mid);
    assert!(!seen(mid, "iosurface_validate_mapping_id_mismatch"));
    assert!(!seen(mid, "iosurface_mapper_internal_owner_read"));
    assert!(seen(other, "iosurface_validate_mapping_id_mismatch"));
    clear_resolve_fail(other);
}

#[test]
fn mapper_declines_are_exact_and_log_safe() {
    use crate::observe::Decline;

    let declines = [
        MapperDecline::CaptureMapperXregRead(MemError::XregUnavailable),
        MapperDecline::CaptureRequestTypeXregRead(MemError::XregUnavailable),
        MapperDecline::CaptureInternalXregRead(MemError::XregUnavailable),
        MapperDecline::CaptureRequestTypeMismatch,
        MapperDecline::CaptureInternalZero,
        MapperDecline::CaptureInternalKvaInvalid,
        MapperDecline::CaptureMapperKvaInvalid,
        MapperDecline::DeviceDescriptorRead(MemError::Unmapped),
    ];
    let mut slugs = std::collections::HashSet::new();
    for decline in declines {
        let slug = decline.slug();
        assert!(slug.starts_with("mapper_"));
        assert!(
            slug.bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'),
            "not log-safe: {slug}"
        );
        assert!(slugs.insert(slug), "duplicate mapper decline: {slug}");
    }
    assert_eq!(
        crate::observe::Emit::decline(
            "mapper_capture_fail",
            &MapperDecline::CaptureRequestTypeMismatch,
        )
        .field("mapping", 9)
        .render(),
        "mapper_capture_fail reason=mapper_capture_request_type_mismatch mapping=9"
    );
}

#[test]
fn mapper_boundary_preserves_the_iosurface_check_reason() {
    let status =
        iosurface_pages::Status::ErrInternalRead("iosurface_mapper_internal_mapping_id_read");
    assert_eq!(
        refusal_reason(&status),
        "iosurface_mapper_internal_mapping_id_read"
    );
    assert_eq!(
        crate::observe::Emit::refusal("mapper_resolve_fail", &status)
            .unwrap()
            .field("mapping", 4)
            .render(),
        "mapper_resolve_fail reason=iosurface_mapper_internal_mapping_id_read \
         class=internal_read mapping=4"
    );
}
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SIZE_ARM64E};
use crate::runtime::host::FakeHost;

/// arm64e kernel VA base used by the contract.
const KVA: u64 = 0xfffffe00_10000000;

fn put_u32(h: &mut FakeHost, gpa: u64, v: u32) {
    h.map_range(gpa, 4, 0);
    h.put_u32(gpa, v);
}
fn put_u64(h: &mut FakeHost, gpa: u64, v: u64) {
    h.map_range(gpa, 8, 0);
    let b = v.to_le_bytes();
    let _ = h.write_gpa(gpa, &b);
}

#[test]
fn capture_validates_identity_and_ring() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let ring = 0x7000_0000u64;
    state.iosfc.ring_base = ring;

    // producer=1 → entry 0: MAP mapping_id=7
    let mut entry = [0u8; 16];
    st32(&mut entry[0..], MAPPER_REQUEST_MAP);
    st32(&mut entry[4..], 7);
    host.map_range(ring, 16, 0);
    let _ = host.write_gpa(ring, &entry);

    let internal = KVA;
    let mapper = KVA + 0x1000;
    // MappingInternal identity fields
    put_u64(&mut host, internal + MAPPING_INTERNAL_BACKPTR, mapper);
    put_u32(&mut host, internal + MAPPING_INTERNAL_ID, 7);
    put_u32(
        &mut host,
        internal + MAPPING_INTERNAL_SIZE,
        MAPPING_INTERNAL_EXPECTED_SIZE,
    );

    host.set_xreg(MAPPER_CAPTURE_REG_MAPPER_DEVICE, mapper);
    host.set_xreg(MAPPER_CAPTURE_REG_REQUEST_TYPE, MAPPER_REQUEST_MAP as u64);
    host.set_xreg(MAPPER_CAPTURE_REG_MAPPING_INTERNAL, internal);

    let cap = capture_at_producer(&state, &host, 1).expect("capture");
    assert_eq!(cap.producer, 1);
    assert_eq!(cap.mapping_internal, internal);
    assert!(apply_capture(&mut state, &cap, 7));
    assert_eq!(state.mappings.get(&7).unwrap().mapping_internal, internal);
}

#[test]
fn capture_handoff_mismatch_is_fail_visible_and_latched() {
    // A decoded MAP request whose captured handoff registers disagree with
    // the ring (wrong request-type in the xreg) is a genuine capture miss:
    // the mapping never attaches → downstream black. It must return None,
    // latch its reason once (no per-publish flood), and re-arm on clear.
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let ring = 0x7100_0000u64;
    state.iosfc.ring_base = ring;

    // producer=1 → entry 0: MAP mapping_id=9
    let mut entry = [0u8; 16];
    st32(&mut entry[0..], MAPPER_REQUEST_MAP);
    st32(&mut entry[4..], 9);
    host.map_range(ring, 16, 0);
    let _ = host.write_gpa(ring, &entry);

    let internal = KVA;
    // xreg request-type disagrees with the ring's MAP → handoff mismatch.
    host.set_xreg(MAPPER_CAPTURE_REG_MAPPER_DEVICE, 0);
    host.set_xreg(MAPPER_CAPTURE_REG_REQUEST_TYPE, MAPPER_REQUEST_UNMAP as u64);
    host.set_xreg(MAPPER_CAPTURE_REG_MAPPING_INTERNAL, internal);

    clear_resolve_fail(9);
    let seen = |m: u32, r: &'static str| {
        resolve_fail_latch()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&(m, r))
    };
    assert!(capture_at_producer(&state, &host, 1).is_none());
    assert!(seen(9, "mapper_capture_request_type_mismatch"));
    // A second identical publish must not add a duplicate (still one entry).
    assert!(capture_at_producer(&state, &host, 1).is_none());
    // A clean resolve of the same mapping re-arms the capture reason.
    clear_resolve_fail(9);
    assert!(!seen(9, "mapper_capture_request_type_mismatch"));
}

/// A mapping id 3 whose internal object resolves to exactly one valid page,
/// attached and ready for `resolve_mapping_backing`.
///
/// Shared by the two span tests so they build the footprint the same way.
///
/// `pfn` is a parameter rather than a constant because the emitters dedup
/// through `observe::first_sight`, whose latch is process-global and so is
/// shared by every test in the binary. Two tests resolving the same page
/// would compute the same [`span_first_sight_key`], and whichever ran first
/// would silence the other's line — the very coupling these tests exist to
/// check, reappearing between the tests themselves. Distinct PFNs keep each
/// test's latch its own whatever the run order.
///
/// Returns `(state, host, page_gpa)`.
fn span_fixture(pfn: u32) -> (DeviceState, FakeHost, u64) {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let internal = KVA;
    let mapper = KVA + 0x1000;
    let page_obj = KVA + 0x2000;
    let table = KVA + 0x3000;
    let page_gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;

    put_u64(&mut host, internal + MAPPING_INTERNAL_BACKPTR, mapper);
    put_u32(&mut host, internal + MAPPING_INTERNAL_ID, 3);
    put_u32(
        &mut host,
        internal + MAPPING_INTERNAL_SIZE,
        MAPPING_INTERNAL_EXPECTED_SIZE,
    );
    // page fields: 0x48 points at page_obj which has table ptr at +0xb8
    put_u64(
        &mut host,
        internal + iosurface_pages::MAPPING_INTERNAL_PAGE_FIELD_48,
        page_obj,
    );
    put_u64(
        &mut host,
        internal + iosurface_pages::MAPPING_INTERNAL_PAGE_FIELD_50,
        0,
    );
    put_u64(
        &mut host,
        internal + iosurface_pages::MAPPING_INTERNAL_PAGE_COUNT,
        1,
    );
    put_u64(
        &mut host,
        page_obj + iosurface_pages::MAPPING_PAGE_TABLE_FROM_F48,
        table,
    );
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    put_u32(&mut host, table, entry);
    // one page of guest RAM for the surface
    host.map_range(page_gpa, PAGE_SIZE_ARM64E as usize, 0x55);

    state.mapper_device_kva = mapper;
    assert!(state.attach_mapping_internal(3, internal));
    (state, host, page_gpa)
}

#[test]
fn resolve_builds_page_entries() {
    let pfn = 0x1e88c_u32;
    let (mut state, host, _page_gpa) = span_fixture(pfn);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    // The adopted page list is what bounds every mapping-rail guest write,
    // so a successful resolve must report its guest-physical footprint. An
    // earlier cut of `mapping_gpa_span` keyed on `pages_changed` and was
    // silent for whole live boots while page lists were plainly being
    // adopted; asserting it here is what makes that failure loud in the
    // suite instead of only in a log nobody diffed.
    let cap = crate::observe::sink::FailCapture::start();
    assert!(resolve_mapping_backing(&mut state, &host, 3));
    let m = state.mappings.get(&3).unwrap();
    assert_eq!(m.page_entries.len(), 1);
    assert_eq!(m.page_entries[0], entry);
    let span = cap.one("OFF");
    assert!(
        span.contains("mapping_gpa_span mid=3") && span.contains("pages=1"),
        "resolve must report its adopted footprint, got {span:?}"
    );
    // The page number is what a guest panic prints (`pmap_page_protect()
    // ... pn=0x...`), so it has to be readable without arithmetic.
    assert!(
        span.contains(&format!("pn_lo={pfn:#x}")),
        "span must name the adopted PFN as a page number, got {span:?}"
    );
}

/// The type-4 adoption site must not be able to silence this one.
///
/// Both emitters print `mapping_gpa_span` and both dedup on
/// [`span_first_sight_key`], which is built from the mapping id and the span
/// alone — so for any footprint both sites reach, they compute the *same*
/// key. While they also shared one `first_sight` namespace, whichever
/// arrived first claimed that footprint and the other went permanently
/// quiet for it. The type-4 site wins in practice, so `src=type4` on every
/// line in a boot was a property of the latch, not a finding about where
/// page lists arrive.
///
/// This claims the footprint under the type-4 namespace first and then
/// requires the mapper's line anyway. With one shared namespace the claim
/// below swallows it and `cap.one("OFF")` finds nothing to return.
#[test]
fn the_type4_span_latch_does_not_suppress_the_mapper_span() {
    let pfn = 0x2c4d1_u32;
    let (mut state, host, page_gpa) = span_fixture(pfn);

    // The claim has to be taken *after* `start()`, which drops every latch
    // precisely so no test inherits another's. Taking it before would be
    // undone, and this test would then pass without ever contending.
    let cap = crate::observe::sink::FailCapture::start();

    // The fixture's single page is both ends of the span. Same discriminant
    // the emitter will compute, taken from the shared helper so this cannot
    // drift from the site it is guarding.
    let key = span_first_sight_key(3, page_gpa, page_gpa, state.page_shift);
    assert!(
        crate::observe::first_sight(SPAN_SEEN_TYPE4, key),
        "the type-4 latch must be unclaimed at the start of this test"
    );

    assert!(resolve_mapping_backing(&mut state, &host, 3));
    let span = cap.one("OFF");
    assert!(
        span.contains("mapping_gpa_span mid=3") && span.contains("src=mapper"),
        "the mapper's own adoption must report its footprint even after the \
         type-4 site has claimed the same one, got {span:?}"
    );
    assert!(
        span.contains(&format!("pn_lo={pfn:#x}")),
        "span must name the adopted PFN as a page number, got {span:?}"
    );
}

struct FailingKvaHost {
    inner: FakeHost,
    err: MemError,
}

impl HostMemory for FailingKvaHost {
    fn read_gpa(&self, gpa: u64, buf: &mut [u8]) -> Result<(), MemError> {
        self.inner.read_gpa(gpa, buf)
    }

    fn write_gpa(&mut self, gpa: u64, buf: &[u8]) -> Result<(), MemError> {
        self.inner.write_gpa(gpa, buf)
    }
}

impl HostOps for FailingKvaHost {
    fn mono_ns(&self) -> u64 {
        0
    }

    fn enqueue(&mut self, _action: crate::runtime::host::HostAction) {}

    fn schedule_bh(&mut self) {}

    fn read_kva(&self, _kva: u64, _buf: &mut [u8]) -> Result<(), MemError> {
        Err(self.err)
    }

    fn map_pages(&mut self, gpas: &[u64], page_size: usize) -> Option<usize> {
        self.inner.map_pages(gpas, page_size)
    }

    fn unmap_pages(&mut self, ptr: usize, len: usize) {
        self.inner.unmap_pages(ptr, len);
    }

    fn is_ram_gpa(&self, gpa: u64) -> bool {
        self.inner.is_ram_gpa(gpa)
    }
}

fn assert_revalidate_error_preserves_cached_page_plan(err: MemError) {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let entry = (0x444u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.attach_mapping_internal(3, KVA));
    assert!(state.set_mapping_geom(
        3,
        64,
        64,
        crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
    ));
    {
        let m = state.mappings.get_mut(&3).unwrap();
        m.mapped = true;
        m.page_entries = vec![entry];
        m.page_table_kva = KVA + 0x3000;
    }

    let host = FailingKvaHost {
        inner: FakeHost::new(),
        err,
    };
    clear_resolve_fail(3);
    let log_before = std::fs::read_to_string(crate::observe::fail_log_path())
        .unwrap_or_default()
        .len();
    assert_eq!(revalidate_mapping_reason(&mut state, &host, 3), None);
    let m = state.mappings.get(&3).unwrap();
    assert_eq!(m.page_entries, vec![entry]);
    assert_eq!(m.page_table_kva, KVA + 0x3000);
    let log_after = std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default();
    assert!(
        !log_after[log_before..].contains("mapper_revalidate_fallback"),
        "an expected cached-plan alias fallback must stay silent: {}",
        &log_after[log_before..]
    );
}

#[test]
fn revalidate_no_cpu_preserves_cached_page_plan() {
    assert_revalidate_error_preserves_cached_page_plan(MemError::NoCpu);
}

#[test]
fn revalidate_unmapped_read_preserves_cached_page_plan() {
    assert_revalidate_error_preserves_cached_page_plan(MemError::Unmapped);
}

/// qemu-shim: early page resolve + late geom must re-expand the table.
/// IOSurface PAGE_SIZE is 16 KiB (arm64e). 1440×1080 BGRA needs
/// ALIGN_UP(1440×4,128)×1080 = 6 220 800 bytes ≈ 380 pages; a 1-page stale
/// table must not cover (dual-mid Store after mode switch).
#[test]
fn pages_cover_geom_false_when_table_shorter_than_span() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    assert!(state.attach_mapping_internal(8, KVA));
    {
        let m = state.mappings.get_mut(&8).unwrap();
        m.mapped = true;
        // Stale early resolve: single PAGE_SIZE before geom latched.
        m.page_entries = vec![0x11; 1];
    }
    assert!(state.set_mapping_geom(
        8,
        1440,
        1080,
        crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
    ));
    assert!(
        !pages_cover_geom(&state, 8),
        "1×16KiB page cannot cover 1440×1080 BGRA sample window"
    );
    let host = FakeHost::new();
    let _ = ensure_resolved_for_scanout(&mut state, &host, 8);
    assert!(!pages_cover_geom(&state, 8));
}

#[test]
fn pages_cover_geom_true_for_full_table() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    assert!(state.attach_mapping_internal(3, KVA));
    assert!(state.set_mapping_geom(
        3,
        64,
        64,
        crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
    ));
    // 64×64 BGRA packed bpr 256 → 16 KiB → 1×16KiB page covers.
    {
        let m = state.mappings.get_mut(&3).unwrap();
        m.mapped = true;
        m.page_entries = vec![0x22; 1];
    }
    assert!(pages_cover_geom(&state, 3));
}

/// A page table cannot make a window the guest's own allocation does not
/// contain, however many pages it holds.
///
/// `mapping_span_bound` refuses when the estimated packed span runs past
/// `device_desc.alloc_size` — that refusal is the only place the wire
/// allocation bounds the span. This case is the one where a caller could
/// answer it by calling `packed_span_estimate` directly and getting the
/// rejected span back: the descriptor says 1.5 MiB, the packed 1024² BGRA
/// window is 4 MiB, and the table is deliberately sized to cover the 4 MiB.
/// Falling
/// back would report "covered" for a surface whose own descriptor says it is
/// a third of that size.
#[test]
fn a_generous_table_cannot_cover_a_window_past_the_wire_allocation() {
    use crate::contract::endian::{st32, st64};
    use crate::contract::iosurface_pages::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_BPR, DEVICE_DESC_DIMS, DEVICE_DESC_LEN,
        DEVICE_DESC_PLANE_COUNT,
    };

    let mut desc = vec![0u8; DEVICE_DESC_LEN];
    // 1.5 MiB allocation, single plane, and a bpr too small for 1024 BGRA so
    // the device-surface path refuses and the invent tail is what runs.
    st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], 0x18_0000);
    st64(
        &mut desc[DEVICE_DESC_DIMS..],
        (1024u64 << 8) | (1024u64 << 40),
    );
    st32(&mut desc[DEVICE_DESC_BPR..], 64);
    desc[DEVICE_DESC_PLANE_COUNT] = 0;

    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    assert!(state.attach_mapping_internal(9, KVA));
    assert!(state.set_mapping_device_desc(9, &desc));
    assert!(state.set_mapping_geom(
        9,
        1024,
        1024,
        crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
    ));
    {
        let m = state.mappings.get_mut(&9).unwrap();
        m.mapped = true;
        // 256 × 16 KiB = 4 MiB, exactly the packed 1024² BGRA span, so a
        // fallback to `sample_window` would compare 4 MiB against 4 MiB and
        // pass.
        m.page_entries = vec![0x33; 256];
    }
    assert!(
        !pages_cover_geom(&state, 9),
        "alloc_size 0x18_0000 cannot hold a 4 MiB window; the page count is not the bound"
    );
}

/// 249² Favourites-class tiles fit in 16×16KiB pages; short-table proxy is
/// desktop dual-mid, not tile size alone.
#[test]
fn pages_cover_geom_249_tile_fits_in_sixteen_16k_pages() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    assert!(state.attach_mapping_internal(8, KVA));
    {
        let m = state.mappings.get_mut(&8).unwrap();
        m.mapped = true;
        m.page_entries = vec![0x11; 16];
    }
    assert!(state.set_mapping_geom(
        8,
        249,
        249,
        crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
    ));
    assert!(
        pages_cover_geom(&state, 8),
        "live Favourites pages=16 is enough for 249² BGRA at 16KiB pages"
    );
}

#[test]
fn render_pages_reject_known_device_and_task_control_pages() {
    let mut state = DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
    state.gfx.root_page = 0x120;
    state.child_rings[2].page_gpas = vec![0x330_000];
    state.define_task(1, 0x4000_0000, 0x440);
    assert!(state.set_object_list(1, 0x550, 1024));

    assert_eq!(
        first_control_page_collision(&state, &[0x120_000]),
        Some((0x120_000, "gfx_root"))
    );
    assert_eq!(
        first_control_page_collision(&state, &[0x330_000]),
        Some((0x330_000, "child_fifo"))
    );
    assert_eq!(
        first_control_page_collision(&state, &[0x440_000]),
        Some((0x440_000, "task_directory"))
    );
    assert_eq!(first_control_page_collision(&state, &[0x660_000]), None);
}

/// The main FIFO is as long as the guest declared it, not one page.
///
/// The ring spans `fifo_length` bytes from its base page and is routinely
/// larger than a page, but the guard probed only the first one — so a
/// surface backed by the ring's second page aliased live transport memory
/// and was accepted. `fifo_length` is a decoded guest field with a real
/// consumer (`drain_main_fifo` reads the ring over exactly that extent), so
/// the bound is the guest's own number rather than a chosen one.
#[test]
fn the_main_fifo_is_probed_over_every_page_the_guest_declared() {
    let mut state = DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
    let page = 1u64 << crate::model::PAGE_SHIFT_X86;
    state.gfx.fifo_base_page = 0x220;
    // Four pages of ring, which is what the guest's own length says.
    state.gfx.fifo_length = (4 * page) as u32;
    let base = 0x220u64 << crate::model::PAGE_SHIFT_X86;

    for i in 0..4u64 {
        let gpa = base + i * page;
        assert_eq!(
            first_control_page_collision(&state, &[gpa]),
            Some((gpa, "root_fifo")),
            "page {i} of the declared ring is transport memory"
        );
    }
    // One page past the declared end is not the ring.
    assert_eq!(
        first_control_page_collision(&state, &[base + 4 * page]),
        None
    );
    // The lowest colliding page is the one reported, whatever order the
    // surface names its pages in.
    assert_eq!(
        first_control_page_collision(&state, &[base + 3 * page, base + page]),
        Some((base + page, "root_fifo"))
    );
    // A declared length of zero still guards the base page itself, so a
    // ring the guest has based but not yet sized is not a hole.
    state.gfx.fifo_length = 0;
    assert_eq!(
        first_control_page_collision(&state, &[base]),
        Some((base, "root_fifo"))
    );
    assert_eq!(first_control_page_collision(&state, &[base + page]), None);
}

/// A task's object list lives in that task's GVA space, so its number must
/// not be tested against surface GPAs at all.
///
/// `object_list_pfn` used to be shifted and compared here like the physical
/// regions beside it. Because tasks put their object lists in low pages, a
/// surface page whose *physical* address happens to equal that *virtual*
/// one was rejected outright — real guest work lost to a coincidence, with
/// any genuine alias still invisible. The five regions this function does
/// compare are guest-physical by construction; this one is not.
#[test]
fn an_object_list_gva_is_not_compared_against_surface_physical_pages() {
    let mut state = DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
    state.define_task(1, 0x4000_0000, 0x440);
    assert!(state.set_object_list(1, 0x550, 1024));

    // The exact numeric coincidence: a surface page at the GPA that equals
    // the object list's GVA, and pages across the whole span it spelled out
    // under either the contract's 12-byte slot or the 16 it used to assume.
    for gpa in [0x550_000, 0x551_000, 0x552_000, 0x553_000] {
        assert_eq!(
            first_control_page_collision(&state, &[gpa]),
            None,
            "{gpa:#x} is a GPA; the object list's {:#x} is a GVA and they do not compare",
            (state.tasks[1].object_list_pfn as u64) << state.page_shift
        );
    }
    // The task's directory is a real PFN and must still be caught, so this
    // is not "the task loop stopped checking anything".
    assert_eq!(
        first_control_page_collision(&state, &[0x440_000]),
        Some((0x440_000, "task_directory"))
    );
}

/// Priority order survives the rewrite: a surface colliding with several
/// control structures at once names the same one it always did. The walk is
/// per task, so task 1's directory is reported before task 2's — which a
/// flat "collect every control page then sort" would silently lose.
#[test]
fn a_surface_colliding_with_several_control_structures_names_the_first() {
    let mut state = DeviceState::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
    state.gfx.root_page = 0x120;
    state.gfx.fifo_base_page = 0x220;
    state.iosfc.ring_base = 0x300_000;
    state.child_rings[2].page_gpas = vec![0x330_000];
    state.define_task(1, 0x4000_0000, 0x440);
    state.define_task(2, 0x4000_0000, 0x660);

    let all = [
        0x660_000, 0x440_000, 0x330_000, 0x300_000, 0x220_000, 0x120_000,
    ];
    assert_eq!(
        first_control_page_collision(&state, &all),
        Some((0x120_000, "gfx_root"))
    );
    state.gfx.root_page = 0;
    assert_eq!(
        first_control_page_collision(&state, &all),
        Some((0x220_000, "root_fifo"))
    );
    state.gfx.fifo_base_page = 0;
    assert_eq!(
        first_control_page_collision(&state, &all),
        Some((0x300_000, "iosfc_ring"))
    );
    state.iosfc.ring_base = 0;
    assert_eq!(
        first_control_page_collision(&state, &all),
        Some((0x330_000, "child_fifo"))
    );
    state.child_rings[2].page_gpas.clear();
    assert_eq!(
        first_control_page_collision(&state, &all),
        Some((0x440_000, "task_directory"))
    );
    // Task 1 outranks task 2: the walk is in task order, not address order.
    state.tasks[1].directory_pfn = 0;
    assert_eq!(
        first_control_page_collision(&state, &all),
        Some((0x660_000, "task_directory"))
    );
}

#[test]
fn stable_view_without_a_vulkan_import_unmaps_at_mapping_retirement() {
    let mut state = DeviceState::new(crate::model::DeviceId::default(), 12);
    state.retired_views.push((0x1000, 0x2000));
    let mut host = crate::runtime::FakeHost::new();
    host.stable_map_pages = true;

    super::flush_retired_views(&mut state, &mut host);

    assert!(state.retired_views.is_empty());
    assert_eq!(host.unmap_pages_calls, 1);
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn vulkan_alias_unmaps_only_after_terminal_destruction_is_published() {
    let mut host = crate::runtime::FakeHost::new();

    assert_eq!(super::drain_deferred_unmaps(&mut host), 0);
    assert_eq!(host.unmap_pages_calls, 0);

    crate::backend::vulkan::engine::publish_released_host_alias_for_test((0x1000, 0x2000));
    assert_eq!(super::drain_deferred_unmaps(&mut host), 1);
    assert_eq!(host.unmap_pages_calls, 1);
    assert_eq!(super::drain_deferred_unmaps(&mut host), 0);
    assert_eq!(
        host.unmap_pages_calls, 1,
        "an alias is returned exactly once"
    );
}

#[test]
fn transient_views_still_unmap_at_mapping_retirement() {
    let mut state = DeviceState::new(crate::model::DeviceId::default(), 12);
    state.retired_views.push((0x1000, 0x2000));
    let mut host = crate::runtime::FakeHost::new();

    super::flush_retired_views(&mut state, &mut host);

    assert!(state.retired_views.is_empty());
    assert_eq!(host.unmap_pages_calls, 1);
}
