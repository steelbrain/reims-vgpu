//! Tests for the guest-surface-to-console scanout path.
//!
//! Out of line for the reason the sibling `runtime/` modules that already do
//! this have: colocated, these 1,266 lines were 53% of `scanout.rs`.

use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
use crate::runtime::host::{FakeHost, HostActionKind};
use reims_vgpu_paging::geometry::{
    MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
    MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
};

const ALL_CAPTURE: &[CaptureDecline] = &[
    CaptureDecline::NoMapping,
    CaptureDecline::Unmapped,
    CaptureDecline::NoPages,
    CaptureDecline::NoGeom,
    CaptureDecline::GeomMismatch {
        have_w: 0,
        have_h: 0,
    },
    CaptureDecline::BppUnknown { format: 0 },
    CaptureDecline::TightRowUnknown { format: 0 },
    CaptureDecline::NoSampleWindow,
    CaptureDecline::BprBelowTight { bpr: 0, tight: 0 },
    CaptureDecline::ContigViewNull,
    CaptureDecline::ContigViewShort {
        contig_len: 0,
        span_end: 0,
    },
    CaptureDecline::BaseBeyondSpan {
        base_off: 0,
        span_end: 0,
        contig: true,
    },
    CaptureDecline::MultiReadFailed { len: 0 },
    CaptureDecline::DstOverflow { row: 0 },
    CaptureDecline::ConvertRowOob { row: 0 },
    CaptureDecline::ConvertRowMissing { row: 0 },
    CaptureDecline::ConvertToRgba { format: 0 },
    CaptureDecline::ConvertFromRgba,
    CaptureDecline::DirectRowOob { row: 0 },
    CaptureDecline::DirectRowMissing { row: 0 },
];

/// Every capture reason names the rail that wrote it.
///
/// Bare, three of these belonged to other rails too — `unmapped` and
/// `short_view` to the guest-page import path and `no_mapping` to the IOSurface plane view
/// loader — so a `grep reason=unmapped` over one boot returned a mix of
/// subsystems. The prefix is what makes that grep answerable.
#[test]
fn every_capture_reason_names_its_rail_and_is_distinct() {
    use crate::observe::Decline as _;
    let mut slugs: Vec<&str> = Vec::new();
    for d in ALL_CAPTURE {
        assert!(
            d.slug().starts_with("capture_"),
            "{} is not namespaced to the capture rail",
            d.slug()
        );
        slugs.push(d.slug());
    }
    slugs.sort_unstable();
    let before = slugs.len();
    slugs.dedup();
    assert_eq!(before, slugs.len(), "duplicate CaptureDecline slug");
}

/// **`short_view` was one `if` with three `||`-ed conditions.** A null host
/// pointer, a view shorter than the sample window, and a degenerate window
/// are three different faults with three different fixes, and they reported
/// one name. This is the "N checks behind one status" class, inside a single
/// expression.
#[test]
fn the_three_faults_that_shared_short_view_have_three_names() {
    use crate::observe::Decline as _;
    let names = [
        CaptureDecline::ContigViewNull.slug(),
        CaptureDecline::ContigViewShort {
            contig_len: 4,
            span_end: 8,
        }
        .slug(),
        CaptureDecline::BaseBeyondSpan {
            base_off: 8,
            span_end: 8,
            contig: true,
        }
        .slug(),
    ];
    let mut sorted = names.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 3, "{names:?}");
    assert!(names.iter().all(|n| *n != "capture_short_view"));
}

/// The two row-read sites the old names merged sit on paths whose bounds
/// differ — the converting path slices `tight` bytes, the direct path
/// `min(dst_row, tight)` — so they can fail under different conditions and
/// must not answer to one name.
#[test]
fn the_convert_and_direct_row_paths_do_not_share_a_name() {
    use crate::observe::Decline as _;
    assert_ne!(
        CaptureDecline::ConvertRowOob { row: 1 }.slug(),
        CaptureDecline::DirectRowOob { row: 1 }.slug()
    );
    assert_ne!(
        CaptureDecline::ConvertRowMissing { row: 1 }.slug(),
        CaptureDecline::DirectRowMissing { row: 1 }.slug()
    );
}

/// A capture refusal must carry the numbers behind it — "the view was short"
/// without the two lengths does not say by how much, and the console is black
/// either way.
#[test]
fn a_capture_refusal_carries_its_numbers() {
    use crate::observe::Decline as _;
    assert_eq!(
        CaptureDecline::ContigViewShort {
            contig_len: 4096,
            span_end: 8_294_400,
        }
        .fields(),
        vec![
            ("contig_len", "4096".to_string()),
            ("span_end", "8294400".to_string()),
        ]
    );
    assert_eq!(
        CaptureDecline::GeomMismatch {
            have_w: 800,
            have_h: 600,
        }
        .fields(),
        vec![("have", "800x600".to_string())]
    );
    // Field values are grepped out of a space-separated line.
    for d in ALL_CAPTURE {
        for (k, v) in d.fields() {
            assert!(!k.contains(' '), "{k}");
            assert!(!v.contains(' '), "{}: {v}", d.slug());
        }
    }
}

#[test]
fn missing_mapping_fails_without_latching() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let w = 4u32;
    let h = 2u32;
    let stride = w * 4;
    let mut dst = vec![0xAAu8; (stride * h) as usize];
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 1, &mut dst, stride, w, h, 0),
        ScanoutCopyResult::Failed
    );
    // Destination untouched; generation not latched.
    assert!(dst.iter().all(|&b| b == 0xAA));
    assert!(!state.presentation.present.console_valid());
}

#[test]
fn early_boot_front_formats_and_geometry_barrier() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mapping_id = 7u32;
    assert!(state.map_surface(mapping_id));
    {
        let m = state.surfaces.mappings.get_mut(&mapping_id).unwrap();
        m.lifecycle.active = true;
        m.publish_geometry_for_test(1920, 1080, MTL_FORMAT_BGRA8_UNORM);
        m.pages.entries = vec![(1u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        m.content.guest_page_generation = 3;
    }
    // Live Monterey: first full-frame BGRA8 writeback enqueues at guest geom.
    note_front_buffer_writeback(
        &mut state,
        &mut host,
        mapping_id,
        1920,
        1080,
        MTL_FORMAT_BGRA8_UNORM,
    );
    assert_eq!(host.actions.len(), 1);
    assert_eq!(host.actions[0].kind, HostActionKind::ScanoutUpdate);
    assert_eq!(host.actions[0].a1, 1920);
    assert_eq!(host.actions[0].a2, 1080);
    assert!(state.presentation.present.console_valid());
    host.actions.clear();
    // Same geom → paint again.
    note_front_buffer_writeback(
        &mut state,
        &mut host,
        mapping_id,
        1920,
        1080,
        MTL_FORMAT_BGRA8_UNORM,
    );
    assert_eq!(host.actions.len(), 1);
    host.actions.clear();
    // Different geom → latch only, no paint (archive same_geom; not resized).
    note_front_buffer_writeback(
        &mut state,
        &mut host,
        mapping_id,
        1920,
        24,
        MTL_FORMAT_BGRA8_UNORM,
    );
    assert!(host.actions.is_empty());
    assert_eq!(state.presentation.present.console_width(), 1920);
    assert_eq!(state.presentation.present.console_height(), 1080);
    // Non-front format → ignore.
    note_front_buffer_writeback(&mut state, &mut host, mapping_id, 1920, 1080, 0x9999);
    assert!(host.actions.is_empty());
}

#[test]
fn early_scanout_target_refuses_resize_geom() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    // Established console 1920×1080 after first paint.
    state.presentation.present.establish_console(1920, 1080, 0);
    state.presentation.present.note_early_composite(5);
    assert!(state.map_surface(5));
    {
        let m = state.surfaces.mappings.get_mut(&5).unwrap();
        m.lifecycle.active = true;
        m.publish_geometry_for_test(1440, 1080, MTL_FORMAT_RGBA16_FLOAT);
        m.content.guest_page_generation = 9;
    }
    // The composite front points at a new mode FB, but early gfx_update
    // must not resize — DisplaySwap owns modeChangeHandler sizeInPixels.
    assert!(early_scanout_target(&state).is_none());

    // Same geom re-pull still allowed.
    {
        let m = state.surfaces.mappings.get_mut(&5).unwrap();
        m.publish_geometry_for_test(1920, 1080, MTL_FORMAT_RGBA16_FLOAT);
    }
    let t = early_scanout_target(&state).expect("same-geom target");
    assert_eq!(t, (5, 1920, 1080, 9));
}

/// ClearOnly init present_mapping must not early-paint (keep BAR1).
#[test]
fn early_scanout_target_refuses_clear_only_init() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.presentation.present.establish_console(1920, 1080, 0);
    state.presentation.present.note_early_composite(2);
    state.presentation.present.invalidate_frame_for_test();
    assert!(state.map_surface(2));
    {
        let m = state.surfaces.mappings.get_mut(&2).unwrap();
        m.lifecycle.active = true;
        m.publish_geometry_for_test(1920, 1080, MTL_FORMAT_BGRA8_UNORM);
        m.content.guest_page_generation = 1;
    }
    state.note_surface_clear(2);
    assert!(early_scanout_target(&state).is_none());

    // Composite latch may early-paint.
    state.note_surface_composite(2);
    let t = early_scanout_target(&state).expect("composite early target");
    assert_eq!(t.0, 2);
    assert_eq!((t.1, t.2), (1920, 1080));
}

/// `present_mapping` alone is not a console front, however good it looks.
///
/// Mid 7 here is faultless by every other measure the resolver applies —
/// mapped, page-backed, a front-buffer format, console geometry, and
/// composited rather than cleared. The one thing it is not is the mapping
/// the guest most recently composited into, and that is the whole contract:
/// only `early_front_mapping` names the pre-boundary console. Ranking
/// `present_mapping` behind it, which is what this used to do, served the
/// last writeback of *any* kind with no sentence saying the guest meant it.
#[test]
fn early_scanout_ignores_a_present_mapping_that_is_not_the_composite_front() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.presentation.present.establish_console(1920, 1080, 0);
    state.presentation.present.invalidate_frame_for_test();
    assert!(state.map_surface(7));
    {
        let m = state.surfaces.mappings.get_mut(&7).unwrap();
        m.lifecycle.active = true;
        m.publish_geometry_for_test(1920, 1080, MTL_FORMAT_BGRA8_UNORM);
        m.content.guest_page_generation = 4;
        m.pages.entries = vec![1];
    }
    state.note_surface_composite(7);
    state.presentation.present.note_present_candidate(7);
    assert!(
        early_scanout_target(&state).is_none(),
        "the last writeback of any kind is not a statement that the guest \
         composited into it"
    );

    // Naming it as the composited front is what licenses it.
    state.presentation.present.note_early_composite(7);
    assert_eq!(
        early_scanout_target(&state),
        Some((7, 1920, 1080, 4)),
        "the composited front serves, at the mapping's own content generation"
    );
}

/// Sticky early_front survives ClearOnly present_mapping thrash.
#[test]
fn early_scanout_prefers_sticky_composite_over_clear_present() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.presentation.present.establish_console(1920, 1080, 0);
    state.presentation.present.invalidate_frame_for_test();
    // Logo mid (composite writeback).
    assert!(state.map_surface(1));
    {
        let m = state.surfaces.mappings.get_mut(&1).unwrap();
        m.lifecycle.active = true;
        m.publish_geometry_for_test(1920, 1080, MTL_FORMAT_BGRA8_UNORM);
        m.content.guest_page_generation = 5;
        m.pages.entries = vec![1];
    }
    state.note_surface_composite(1);
    state.presentation.present.note_early_composite(1);
    // Guest ClearOnly flip mid overwrites present_mapping (buffer setup).
    assert!(state.map_surface(2));
    {
        let m = state.surfaces.mappings.get_mut(&2).unwrap();
        m.lifecycle.active = true;
        m.publish_geometry_for_test(1920, 1080, MTL_FORMAT_BGRA8_UNORM);
        m.content.guest_page_generation = 1;
        m.pages.entries = vec![1];
    }
    state.note_surface_clear(2);
    state.presentation.present.note_present_candidate(2);
    let t = early_scanout_target(&state).expect("sticky early front");
    assert_eq!(t.0, 1, "must keep logo mid, not ClearOnly flip");
    assert_eq!(t.3, 5);
}

/// After CmdDisplaySwap, writebacks into the back buffer must not rename
/// `present_mapping` (PGDisplay presents the surface named by DisplaySwap only).
#[test]
fn post_display_swap_writeback_does_not_rename_present_mapping() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    // Front mid from DisplaySwap (ch4 op8).
    state.presentation.present.cross_content_boundary();
    state.presentation.present.begin_present(3);
    state.presentation.present.establish_console(1440, 1080, 0);

    // Back buffer mid=4 receives a full-frame composite writeback.
    assert!(state.map_surface(4));
    {
        let m = state.surfaces.mappings.get_mut(&4).unwrap();
        m.lifecycle.active = true;
        m.publish_geometry_for_test(1440, 1080, MTL_FORMAT_RGBA16_FLOAT);
        m.pages.entries = vec![(1u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        m.content.guest_page_generation = 12;
    }
    note_front_buffer_writeback(
        &mut state,
        &mut host,
        4,
        1440,
        1080,
        MTL_FORMAT_RGBA16_FLOAT,
    );
    // No early paint, no rename of the presented mid.
    assert!(host.actions.is_empty());
    assert_eq!(state.presentation.present.presented_mapping(), 3);
    assert_eq!(state.presentation.present.host_mapping(), 3);
    assert!(early_scanout_target(&state).is_none());
}

/// Fragmented IOSurface page list: paint_mapping multi-imports instead of
/// failing not_contig (live boot class: fullscreen present surfaces).
#[test]
fn paint_mapping_fragmented_pages_multi_import() {
    use crate::model::PAGE_SHIFT_X86;
    use crate::runtime::mapping_write::write_bgra8;
    use reims_vgpu_paging::geometry::{
        MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
        MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
    };

    let mut state = Device::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    let gpa0 = 0x5000_0000u64;
    let gpa1 = 0x6000_0000u64;
    host.map_range(gpa0, page as usize, 0);
    host.map_range(gpa1, page as usize, 0);
    let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
    let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
    let mid = 7u32;
    assert!(state.map_surface(mid));
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![
            (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    let frame = [
        0xCCu8, 0x00, 0x00, 0xFF, 0xCC, 0x00, 0x00, 0xFF, 0xCC, 0x00, 0x00, 0xFF, 0xCC, 0x00, 0x00,
        0xFF,
    ];
    assert!(write_bgra8(&mut state, &mut host, mid, &frame, 8, 2, 2));
    let mut dst = vec![0u8; 16];
    assert!(
        paint_mapping(
            &mut state,
            &mut host,
            mid,
            super::PaintDst {
                bytes: &mut dst,
                stride: 8,
                width: 2,
                height: 2
            },
            crate::runtime::render_writeback::SettleSite::ScanoutPaint
        ),
        "fragmented paint must multi-import, not not_contig"
    );
    assert_eq!(&dst[..], &frame[..]);
    // The fixture must still be fragmented, or this stops testing the
    // multi-import path and silently passes through the packed-view branch.
    // Asserted on the thing `paint_mapping` actually branches on, rather than
    // on a provenance field it used to set as a side effect.
    assert!(
        crate::runtime::mapper::ensure_contig_view(&mut state, &mut host, mid).is_none(),
        "fixture stopped being fragmented"
    );
}

/// The capture buffer is a recycled warm double-buffer, not a fresh 8 MiB
/// alloc per present. Lock that (a) a successful capture recycles the prior
/// retain into `capture_scratch` so the next capture reuses that allocation,
/// and (b) a failed capture returns the scratch untouched and leaves the
/// prior `frame_bgra` retain intact (keep-prior contract).
#[test]
fn capture_recycles_scratch_and_keeps_prior_retain_on_failure() {
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mid = 6u32;
    let pfn = 0x90u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(mid));
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
    }
    assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    state.presentation.present.cross_content_boundary();
    state
        .presentation
        .present
        .set_console_geometry_for_test(2, 2);

    let frame_a = [
        0x10u8, 0x20, 0x30, 0xFF, 0x10, 0x20, 0x30, 0xFF, 0x10, 0x20, 0x30, 0xFF, 0x10, 0x20, 0x30,
        0xFF,
    ];
    assert!(write_bgra8(&mut state, &mut host, mid, &frame_a, 8, 2, 2));
    let gen_a = state
        .surfaces
        .mappings
        .get(&mid)
        .unwrap()
        .content
        .guest_page_generation;
    assert!(capture_present_frame(&mut state, mid, 2, 2, gen_a));
    assert_eq!(
        &state.presentation.present.frame().pixels()[..16],
        &frame_a[..]
    );

    // Second successful capture: the prior retain buffer is recycled into
    // capture_scratch (warm, exactly the frame size — no per-present alloc).
    let frame_b = [
        0x00u8, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00,
        0xFF,
    ];
    assert!(write_bgra8(&mut state, &mut host, mid, &frame_b, 8, 2, 2));
    let gen_b = state
        .surfaces
        .mappings
        .get(&mid)
        .unwrap()
        .content
        .guest_page_generation;
    assert!(capture_present_frame(&mut state, mid, 2, 2, gen_b));
    assert_eq!(
        &state.presentation.present.frame().pixels()[..16],
        &frame_b[..]
    );
    assert_eq!(
        state.presentation.present.frame().scratch_len(),
        16,
        "prior retain recycled as warm scratch of the frame size"
    );

    // Capture failure (unmapped surface) must return the scratch untouched
    // and leave the frame_b retain intact.
    let bad = 99u32;
    assert!(state.map_surface(bad));
    state
        .presentation
        .present
        .set_console_geometry_for_test(2, 2);
    assert!(!capture_present_frame(&mut state, bad, 2, 2, gen_b + 1));
    assert_eq!(
        &state.presentation.present.frame().pixels()[..16],
        &frame_b[..],
        "failed capture must not disturb the prior retain"
    );
    assert_eq!(
        state.presentation.present.frame().scratch_len(),
        16,
        "failed capture recycles its (untouched) scratch"
    );
}

/// After host encode, later guest writebacks must not change scanout until
/// the next DisplaySwap (Apple encodeCurrentFrame + hostPresentCount re-show).
#[test]
fn display_swap_snapshot_stable_against_post_swap_writeback() {
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    let mid = 3u32;
    assert!(state.map_surface(mid));
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
    }
    assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    // Frame A: solid blue-ish BGRA.
    let frame_a = [
        0xCCu8, 0x00, 0x00, 0xFF, 0xCC, 0x00, 0x00, 0xFF, 0xCC, 0x00, 0x00, 0xFF, 0xCC, 0x00, 0x00,
        0xFF,
    ];
    assert!(write_bgra8(&mut state, &mut host, mid, &frame_a, 8, 2, 2));
    let gen = state
        .surfaces
        .mappings
        .get(&mid)
        .unwrap()
        .content
        .guest_page_generation;
    // Present path state as after DisplaySwap (encode pending → first paint).
    state.presentation.present.cross_content_boundary();
    state.presentation.present.begin_present(mid);
    state
        .presentation
        .present
        .set_console_geometry_for_test(2, 2);
    state
        .presentation
        .present
        .set_console_generation_for_test(gen);
    state.presentation.present.mark_frame_encode_pending();
    state.presentation.present.invalidate_frame_for_test();

    // First host paint encodes A.
    let mut dst = vec![0u8; 16];
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, mid, &mut dst, 8, 2, 2, gen),
        ScanoutCopyResult::Painted
    );
    assert_eq!(&dst[..], &frame_a[..]);
    assert!(state.presentation.present.frame().is_valid());
    assert!(!state.presentation.present.frame().encode_pending());

    // Post-encode composite mutates guest pages (mid-pass / next damage).
    let frame_b = [
        0x00u8, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00,
        0xFF,
    ];
    assert!(write_bgra8(&mut state, &mut host, mid, &frame_b, 8, 2, 2));

    // Re-show same present gen still paints frozen A.
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, mid, &mut dst, 8, 2, 2, gen),
        ScanoutCopyResult::Unchanged
    );
    // Force re-blit snapshot (generation match still A after paint).
    state
        .presentation
        .present
        .set_painted_generation_for_test(0);
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, mid, &mut dst, 8, 2, 2, gen),
        ScanoutCopyResult::Painted
    );
    assert_eq!(&dst[..], &frame_a[..]);
}

/// Live class (serial-20260715-054015): early paint latched painted mid/gen
/// (EFI or live black) then capture installed logo+pill into +0x188; with
/// encode_pending cleared at capture, host paint returned Unchanged and
/// QMP stayed on EFI. Capture must force one snapshot blit.
#[test]
fn capture_forces_paint_even_when_painted_mid_gen_already_match() {
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mid = 2u32;
    let pfn = 0x50u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(mid));
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
    }
    assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    // Guest composite: logo-class sparse (non-black BGRA).
    let logo = [
        0x11u8, 0x22, 0x33, 0xFF, 0x11, 0x22, 0x33, 0xFF, 0x11, 0x22, 0x33, 0xFF, 0x11, 0x22, 0x33,
        0xFF,
    ];
    assert!(write_bgra8(&mut state, &mut host, mid, &logo, 8, 2, 2));
    let gen = state
        .surfaces
        .mappings
        .get(&mid)
        .unwrap()
        .content
        .guest_page_generation;

    // Early paint already "painted" this mid@gen (EFI path or prior live
    // paint) — the false Unchanged class without encode_pending.
    state.presentation.present.record_painted_identity(mid, gen);
    state.presentation.present.cross_content_boundary();
    state
        .presentation
        .present
        .set_console_geometry_for_test(2, 2);

    assert!(capture_present_frame(&mut state, mid, 2, 2, gen));
    assert!(
        state.presentation.present.frame().encode_pending(),
        "capture must force next paint of +0x188"
    );
    assert_eq!(
        &state.presentation.present.frame().pixels()[..16],
        &logo[..]
    );

    let mut dst = vec![0u8; 16];
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, mid, &mut dst, 8, 2, 2, gen),
        ScanoutCopyResult::Painted,
        "must blit retain even when painted mid/gen already match"
    );
    assert_eq!(&dst[..], &logo[..]);
    assert!(!state.presentation.present.frame().encode_pending());
    // Second paint: true Unchanged (console holds +0x188).
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, mid, &mut dst, 8, 2, 2, gen),
        ScanoutCopyResult::Unchanged
    );
}

/// The full-frame readback exists for exactly one reason: the DISPLAY needs
/// CPU pixels. Two halves:
///
/// - a resident carrying the current present → NO readback, ever, however
///   long since the last one.
///   The proxies are fed by the GPU reduction instead, so there is no
///   sampling floor forcing a copy any more. `frame_bgra` is dropped rather
///   than left holding the previous readback, while the present metadata
///   advances so `publish_window_frame` exports the fresh resident.
///
///   The buffer still holding the earlier frame used to be this test's
///   evidence that no copy ran, and it was the wrong evidence: that same
///   buffer is what the content verdict and the console blit read as the
///   CURRENT frame. `full_captures` counts readbacks directly, so it answers
///   the question the assertion was actually asking.
/// - no resident carrying → the window blits `frame_bgra`, so the readback
///   runs. This is what keeps the display off any env gate.
#[test]
fn readback_runs_only_for_the_display_never_for_the_proxies() {
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mid = 2u32;
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(mid));
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
    }
    assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));

    // No resident carrying: the display needs pixels, so the readback runs.
    let frame_a = [0x11u8, 0x22, 0x33, 0xFF].repeat(4);
    assert!(write_bgra8(&mut state, &mut host, mid, &frame_a, 8, 2, 2));
    let gen_a = state
        .surfaces
        .mappings
        .get(&mid)
        .unwrap()
        .content
        .guest_page_generation;
    assert!(capture_present_frame(&mut state, mid, 2, 2, gen_a));
    assert_eq!(
        &state.presentation.present.frame().pixels()[..16],
        &frame_a[..],
        "display fallback must read back when no resident carries the frame"
    );
    assert_eq!(state.presentation.present.capture_counts(), (1, 0));

    // A resident carrying: there must be no readback.
    state
        .presentation
        .present
        .set_current_present_resident_carried(true);
    let frame_b = [0x44u8, 0x55, 0x66, 0xFF].repeat(4);
    assert!(write_bgra8(&mut state, &mut host, mid, &frame_b, 8, 2, 2));
    let gen_b = state
        .surfaces
        .mappings
        .get(&mid)
        .unwrap()
        .content
        .guest_page_generation;
    assert_ne!(gen_a, gen_b);
    assert!(capture_present_frame(&mut state, mid, 2, 2, gen_b));
    assert_eq!(
        state.presentation.present.capture_counts().0,
        1,
        "no sampling floor may force a copy once the GPU oracle feeds proxies"
    );
    assert!(
        state.presentation.present.frame().pixels().is_empty(),
        "the readback did not run, so no frame belongs to this present"
    );
    assert_eq!(state.presentation.present.capture_counts().1, 1);
    // Present metadata still advances so the fresh resident gets exported.
    assert_eq!(state.presentation.present.frame().generation(), gen_b);
    assert_eq!(state.presentation.present.frame().mapping(), mid);
    assert!(state.presentation.present.frame().is_valid());
    assert!(state.presentation.present.frame().encode_pending());
}

/// qemu-shim DisplaySwap: guest pages are the single capture source
/// (unified memory). Unreadable pages fail the capture — no mirror exists
/// to invent content from; the prior +0x188 retain covers the console.
#[test]
fn capture_present_reads_pages_and_fails_when_unreadable() {
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let mid = 3u32;
    let pfn = 0x40u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(mid));
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
    }
    assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));

    // Finished composite in pages: solid white BGRA.
    let white = [0xFFu8; 16];
    assert!(write_bgra8(&mut state, &mut host, mid, &white, 8, 2, 2));
    let gen = state
        .surfaces
        .mappings
        .get(&mid)
        .unwrap()
        .content
        .guest_page_generation;
    assert!(capture_present_frame(&mut state, mid, 2, 2, gen));
    assert_eq!(
        &state.presentation.present.frame().pixels()[0..4],
        &[255, 255, 255, 255]
    );

    // Page table unreadable + host-cache evicted → capture fails
    // (guest pages unreadable and no host encode retain).
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.pages.entries.clear();
    }
    crate::runtime::surface_cache::forget(&mut state, mid);
    assert!(!capture_present_frame(&mut state, mid, 2, 2, gen + 1));
}

/// Dual-mid HostAction race (PGDisplay +0x188 / encodeCurrentFrame):
/// DisplaySwap mid3 freezes white, mid4 freezes black (overwrites +0x188).
/// Late HostAction for mid3 still encodes **current** +0x188 (mid4 black) —
/// not recycled live mid3 pages (logo) and not a per-mid white backlog.
#[test]
fn dual_mid_host_action_paints_latest_plus188_not_recycled_pages() {
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    for (mid, pfn) in [(3u32, 0x30u32), (4u32, 0x31u32)] {
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        assert!(state.map_surface(mid));
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
        assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    }

    let white = [0xFFu8; 16];
    let black = [0x00u8; 16];
    let logo = [0xAAu8; 16];
    assert!(write_bgra8(&mut state, &mut host, 3, &white, 8, 2, 2));
    let gen3 = state
        .surfaces
        .mappings
        .get(&3)
        .unwrap()
        .content
        .guest_page_generation;
    // DisplaySwap mid3: +0x188 = white.
    assert!(capture_present_frame(&mut state, 3, 2, 2, gen3));
    state.presentation.present.cross_content_boundary();
    state.presentation.present.begin_present(3);
    state
        .presentation
        .present
        .set_console_geometry_for_test(2, 2);
    state
        .presentation
        .present
        .set_console_generation_for_test(gen3);

    // Guest recycles mid3 after stamp (logo damage / partial composite).
    assert!(write_bgra8(&mut state, &mut host, 3, &logo, 8, 2, 2));

    // DisplaySwap mid4: PGDisplay presentFrame installs named mid4 black.
    assert!(write_bgra8(&mut state, &mut host, 4, &black, 8, 2, 2));
    let gen4 = state
        .surfaces
        .mappings
        .get(&4)
        .unwrap()
        .content
        .guest_page_generation;
    assert!(capture_present_frame(&mut state, 4, 2, 2, gen4));
    state.presentation.present.begin_present(4);
    state
        .presentation
        .present
        .set_console_generation_for_test(gen4);
    assert_eq!(
        state.presentation.present.frame().pixels(),
        &black[..],
        "presentFrame replaces +0x188 with named mid"
    );
    assert_eq!(state.presentation.present.frame().mapping(), 4);

    // Late HostAction for mid3 — encodeCurrentFrame shows current +0x188 (mid4).
    let mut dst = vec![0u8; 16];
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 3, &mut dst, 8, 2, 2, gen3),
        ScanoutCopyResult::Painted
    );
    assert_eq!(
        &dst[..],
        &black[..],
        "late mid3 HostAction must show current +0x188 (mid4)"
    );
    assert_ne!(&dst[..], &logo[..], "must not re-read recycled mid3 pages");
    assert_ne!(&dst[..], &white[..], "must not keep superseded mid3 retain");

    // mid4 HostAction Unchanged after paint.
    let mut dst4 = vec![0u8; 16];
    assert_eq!(
        copy_to_bgra8(&mut state, &mut host, 4, &mut dst4, 8, 2, 2, gen4),
        ScanoutCopyResult::Unchanged
    );
}

/// Capture fail (no pages, no host_cache) returns false; prior +0x188 stays.
#[test]
fn capture_fail_keeps_prior_frame() {
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    assert!(state.map_surface(1));
    {
        let m = state.surfaces.mappings.get_mut(&1).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
    }
    assert!(state.set_mapping_geom(1, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    let mut content = [0u8; 16];
    content[0] = 80;
    content[1] = 80;
    content[2] = 80;
    content[3] = 255;
    assert!(write_bgra8(&mut state, &mut host, 1, &content, 8, 2, 2));
    let gen1 = state
        .surfaces
        .mappings
        .get(&1)
        .unwrap()
        .content
        .guest_page_generation;
    assert!(capture_present_frame(&mut state, 1, 2, 2, gen1));
    assert_eq!(state.presentation.present.frame().pixels(), &content[..]);

    // Mid2 never mapped — capture fails; prior retain intact.
    assert!(state.map_surface(2));
    assert!(state.set_mapping_geom(2, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    assert!(!capture_present_frame(&mut state, 2, 2, 2, 1));
    assert_eq!(state.presentation.present.frame().pixels(), &content[..]);
    assert_eq!(state.presentation.present.frame().mapping(), 1);
}

/// Dual-mid qemu-shim: Clear Store (seed=None) on lagging mid must wipe prior
/// logo before DisplaySwap encode; alternating swap shows each mid's own
/// finished content (not logo bleed under toolbar-only damage).
#[test]
fn dual_mid_clear_store_then_display_swap_both_composites() {
    use crate::runtime::mapping_write::{write_bgra8, write_rgba8_image_changed};

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    // Two 2x2 mids, separate pages.
    for (mid, pfn) in [(3u32, 0x30u32), (4u32, 0x31u32)] {
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        assert!(state.map_surface(mid));
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
        assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
        // Boot logo seed on both.
        let logo = [0xAAu8; 16];
        assert!(write_bgra8(&mut state, &mut host, mid, &logo, 8, 2, 2));
    }
    // mid3: Clear Store full wipe to black (toolbar damage would leave logo
    // with image_changed vs clear seed — seed=None is the Clear contract).
    let clear = [0u8; 16];
    assert!(write_rgba8_image_changed(
        &mut state, &mut host, 3, &clear, None, 2, 2
    ));
    // mid4: full finished frame (white).
    let full = [0xFFu8; 16];
    assert!(write_bgra8(&mut state, &mut host, 4, &full, 8, 2, 2));

    for (mid, expect) in [(3u32, clear.as_slice()), (4u32, full.as_slice())] {
        let gen = state
            .surfaces
            .mappings
            .get(&mid)
            .unwrap()
            .content
            .guest_page_generation;
        state.presentation.present.cross_content_boundary();
        state.presentation.present.begin_present(mid);
        state
            .presentation
            .present
            .set_console_geometry_for_test(2, 2);
        state
            .presentation
            .present
            .set_console_generation_for_test(gen);
        state.presentation.present.mark_frame_encode_pending();
        state.presentation.present.invalidate_frame_for_test();
        let mut dst = vec![0u8; 16];
        assert_eq!(
            copy_to_bgra8(&mut state, &mut host, mid, &mut dst, 8, 2, 2, gen),
            ScanoutCopyResult::Painted
        );
        assert_eq!(
            &dst[..],
            expect,
            "DisplaySwap mid={mid} must encode finished content, not logo"
        );
    }
}

/// Pre-boundary: first same-geom writeback latches present_mapping + paints;
/// a later different-geom front only latches mapping (mode waits DisplaySwap).
#[test]
fn pre_boundary_writeback_latches_present_mapping() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    assert!(!state.presentation.present.content_boundary_crossed());
    assert!(state.map_surface(1));
    {
        let m = state.surfaces.mappings.get_mut(&1).unwrap();
        m.lifecycle.active = true;
        m.publish_geometry_for_test(1920, 1080, MTL_FORMAT_BGRA8_UNORM);
        m.pages.entries = vec![(1u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        m.content.guest_page_generation = 1;
    }
    note_front_buffer_writeback(&mut state, &mut host, 1, 1920, 1080, MTL_FORMAT_BGRA8_UNORM);
    assert_eq!(state.presentation.present.presented_mapping(), 1);
    assert_eq!(host.actions.len(), 1);

    // Mode-switch size: latch new mid, no paint/resize.
    assert!(state.map_surface(3));
    {
        let m = state.surfaces.mappings.get_mut(&3).unwrap();
        m.lifecycle.active = true;
        m.publish_geometry_for_test(1440, 1080, MTL_FORMAT_RGBA16_FLOAT);
        m.pages.entries = vec![(2u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        m.content.guest_page_generation = 2;
    }
    host.actions.clear();
    note_front_buffer_writeback(
        &mut state,
        &mut host,
        3,
        1440,
        1080,
        MTL_FORMAT_RGBA16_FLOAT,
    );
    assert!(host.actions.is_empty());
    assert_eq!(state.presentation.present.presented_mapping(), 3);
    assert_eq!(state.presentation.present.console_width(), 1920);
    assert_eq!(state.presentation.present.console_height(), 1080);
}

/// Regression guard for `present_dims`, the scanout sizing lookup. The
/// blit copies `width*height` from these dims, so their precedence is
/// load-bearing: a present that reads the wrong dimensions blits with the
/// wrong stride/extent -> a torn or clipped scanout. Lock the 3-tier
/// precedence: the mapping's own valid geometry wins; else the retained
/// present dims; else (0, 0). A mapping with `has_geom == false` or a zero
/// axis must NOT be trusted (it falls through to the present dims).
#[test]
fn present_dims_precedence_mapping_then_present_then_zero() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mid = 5u32;

    // No mapping, no present -> (0, 0), never a partial/garbage size.
    assert_eq!(present_dims(&state, mid), (0, 0));

    // Present dims present but no valid mapping geometry -> present dims.
    state
        .presentation
        .present
        .set_console_geometry_for_test(1440, 900);
    assert_eq!(present_dims(&state, mid), (1440, 900));

    assert!(state.map_surface(mid));
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.clear_geometry_for_test(); // geometry not yet valid
    }
    assert_eq!(
        present_dims(&state, mid),
        (1440, 900),
        "has_geom == false must fall through to present dims",
    );

    // A zero axis is not valid mapping geometry either.
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.publish_geometry_for_test(800, 0, 0);
    }
    assert_eq!(
        present_dims(&state, mid),
        (1440, 900),
        "a zero-height mapping must not override present dims",
    );

    // Fully valid mapping geometry wins over the retained present dims.
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.publish_geometry_for_test(800, 600, 0);
    }
    assert_eq!(present_dims(&state, mid), (800, 600));
}

/// Regression guard for the per-row scanout copy (`blit_bgra_buffer`).
///
/// This is the primitive every present ultimately writes pixels through
/// (`blit_present_snapshot` → `copy_to_bgra8`). Its correctness maps
/// directly onto the named framebuffer bugs:
///  - a per-row offset/shear (indexing dst by `src_stride` or vice-versa)
///    is exactly "a/b framebuffer corruption";
///  - copying into a strided dst must land each `width*BPP` row at
///    `y*dst_stride` and leave the row's padding tail untouched — writing
///    into the pad, or reading past the row into the next, corrupts the
///    scanout;
///  - a too-small `src` must be rejected whole (no partial garbage copy),
///    else stale/torn bytes reach the display (residue).
#[test]
fn blit_bgra_buffer_row_offsets_and_bounds_are_exact() {
    let bpp = RGBA8_BPP as usize;
    let (width, height) = (3u32, 2u32);
    let src_stride = width as usize * bpp; // 12
    assert_eq!(src_stride, 12);

    // Distinct per-byte source so any misrouted row/byte is detectable.
    let src: Vec<u8> = (0..(src_stride * height as usize) as u8).collect();

    // 1) Tight dst (dst_stride == src_stride): byte-exact full copy.
    {
        let mut dst = vec![0u8; src.len()];
        assert!(blit_bgra_buffer(
            &src,
            &mut dst,
            src_stride as u32,
            width,
            height
        ));
        assert_eq!(dst, src, "tight blit must be byte-identical");
    }

    // 2) Strided dst (dst_stride > src_stride): each row lands at
    //    y*dst_stride; the trailing pad of every row stays untouched.
    {
        let dst_stride = src_stride + bpp; // one extra pixel of pad/row
        let mut dst = vec![0xEEu8; dst_stride * height as usize];
        assert!(blit_bgra_buffer(
            &src,
            &mut dst,
            dst_stride as u32,
            width,
            height
        ));
        for y in 0..height as usize {
            let doff = y * dst_stride;
            let soff = y * src_stride;
            assert_eq!(
                &dst[doff..doff + src_stride],
                &src[soff..soff + src_stride],
                "row {y} must land at y*dst_stride, not sheared",
            );
            // The pad past the copied width must be preserved (not clobbered,
            // not fed the next row's bytes).
            assert!(
                dst[doff + src_stride..doff + dst_stride]
                    .iter()
                    .all(|&b| b == 0xEE),
                "row {y} padding tail must be left untouched",
            );
        }
    }

    // 3) Undersized src: reject whole, leave dst pristine (no partial copy).
    {
        let short = &src[..src.len() - 1];
        let mut dst = vec![0x11u8; src.len()];
        assert!(
            !blit_bgra_buffer(short, &mut dst, src_stride as u32, width, height),
            "src shorter than width*height*BPP must be refused",
        );
        assert!(
            dst.iter().all(|&b| b == 0x11),
            "a refused blit must not have written any dst byte",
        );
    }

    // 4) Narrower dst than src (dst_stride < src_stride): refuse whole.
    //
    // This case used to assert the opposite — that the blit copied
    // `min(strides)` per row — which is a frame missing its rightmost columns on
    // every row, written with nothing on any channel saying so. A destination
    // that cannot hold the frame is the same kind of answer as a source too
    // short to fill it, three cases up, and it gets the same one.
    {
        let dst_stride = src_stride - bpp; // 8 bytes/row, one pixel short
        let mut dst = vec![0x22u8; dst_stride * height as usize];
        assert!(
            !blit_bgra_buffer(&src, &mut dst, dst_stride as u32, width, height),
            "a dst row shorter than width*BPP must be refused, not truncated",
        );
        assert!(
            dst.iter().all(|&b| b == 0x22),
            "a refused blit must not have written any dst byte",
        );
    }
}

/// The display transaction names one surface, and that surface alone is the
/// capture source. A second compositor member of identical geometry holding
/// different content is not consulted, however fresh it is: the frame that
/// comes out is byte-for-byte the named mid's own pixels, and it is the same
/// frame the named mid produces when no such peer exists at all.
#[test]
fn capture_reads_only_the_named_surface_never_a_same_geometry_peer() {
    use crate::runtime::mapping_write::write_bgra8;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    for (mid, pfn) in [(1u32, 0x40u32), (5u32, 0x41u32)] {
        let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
        host.map_range(gpa, 0x4000, 0);
        let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        assert!(state.map_surface(mid));
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
        assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    }

    // The named mid holds grey; the same-geometry peer holds white and is
    // the fresher of the two by every ordering the model tracks.
    let grey = [0x55u8; 16];
    let white = [0xFFu8; 16];
    assert!(write_bgra8(&mut state, &mut host, 1, &grey, 8, 2, 2));
    let gen1 = state
        .surfaces
        .mappings
        .get(&1)
        .unwrap()
        .content
        .guest_page_generation;
    assert!(write_bgra8(&mut state, &mut host, 5, &white, 8, 2, 2));

    // Capture the named mid with no peer published: the baseline frame.
    assert!(capture_present_frame(&mut state, 1, 2, 2, gen1));
    let alone = state.presentation.present.frame().pixels().to_vec();
    assert_eq!(&alone[..], &grey[..]);

    // Publish a full frame for the peer as well, then re-capture.
    state.note_dense_frame_published(1, 2, 2);
    state.note_dense_frame_published(5, 2, 2);
    // The arrangement has to be one a peer-reading capture would act on, or
    // the assertion below passes for the wrong reason: two surfaces of equal
    // geometry, the peer holding different pixels, and the peer written more
    // recently — mid 5's `write_bgra8` above runs after mid 1's, so it is the
    // later write by program order. (The model tracks no cross-mapping write
    // stamp to assert that with: the one it had existed only to feed a
    // present-staleness census and went with it.)
    let peer = state.surfaces.mappings.get(&5).unwrap();
    assert_eq!(peer.geometry().map(|g| (g.width, g.height)), Some((2, 2)));
    assert_ne!(&grey[..], &white[..]);
    state.presentation.present.clear_frame_pixels_for_test();
    state.presentation.present.invalidate_frame_for_test();
    assert!(capture_present_frame(&mut state, 1, 2, 2, gen1));
    assert_eq!(
        state.presentation.present.frame().pixels(),
        &alone[..],
        "a same-geometry peer must not change the named surface's frame"
    );
    assert_eq!(state.presentation.present.frame().mapping(), 1);
}

/// A light capture must leave no frame behind, because everything
/// downstream reads "is there a frame for this present" off
/// `frame_bgra.is_empty()`.
///
/// A prior CPU-backed present may have populated the buffer. The direct path
/// must clear it so downstream consumers cannot mistake stale bytes for the
/// current resident-carried present.
#[test]
fn a_light_capture_leaves_no_stale_frame_behind() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let (w, h) = (64u32, 64u32);
    // The frame a prior CPU capture leaves: opaque RGB black.
    let stale: Vec<u8> = (0..(w * h) as usize).flat_map(|_| [0, 0, 0, 255]).collect();
    state
        .presentation
        .present
        .replace_frame_pixels_for_test(stale.clone());
    state
        .presentation
        .present
        .set_frame_identity_for_test(0, w, h, 0, 0);

    // Direct present is carrying the display, so this capture takes the
    // light path and reads back nothing.
    state
        .presentation
        .present
        .set_current_present_resident_carried(true);
    let before = state.presentation.present.capture_counts().1;
    assert!(capture_present_frame(&mut state, 1, w, h, 1));
    assert_eq!(
        state.presentation.present.capture_counts().1,
        before.wrapping_add(1),
        "the light path is the one under test"
    );
    assert!(
        state.presentation.present.frame().pixels().is_empty(),
        "a light capture wrote no pixels, so it must not leave {} bytes of an \
         earlier present for the content verdict and the console blit to read \
         as this one",
        state.presentation.present.frame().pixels().len()
    );

    // And the verdict that reads it now says so, instead of reporting the
    // stale frame's colour as this present's.
    use crate::runtime::drain::present_content_verdict;
    assert_eq!(
        present_content_verdict(state.presentation.present.frame().pixels(), 0),
        crate::runtime::drain::PresentContentVerdict::Unsampled
    );
    assert_eq!(
        present_content_verdict(&stale, 0),
        crate::runtime::drain::PresentContentVerdict::Black,
        "the retained frame is what used to be judged"
    );
}

/// The EFI console paint refuses a framebuffer that is device memory, and
/// still paints one that is guest RAM.
///
/// `efi_fb_start` is whatever the guest programmed into 0x1210, and for most of
/// early boot that is the BAR1 GOP framebuffer this device itself exposes.
/// Reading it through the address space cannot work: the QEMU shim sets
/// `MemTxAttrs.memory`, so a translation resolving to device memory fails
/// closed on purpose — a guest page entry aimed at one of our own BARs would
/// otherwise re-enter this device's MMIO handler from inside a Rust call that
/// already holds the device lock.
///
/// So this door is *expected* to be shut, and the caller has a second one
/// (`copy_early_bar1`, which reads BAR1 through the host pointer the C shim
/// registered). Discovering the closure by reading rows until one failed cost a
/// live boot 465 completed reads per attempt on the ~30 Hz early-console
/// cadence, threw all of them away, and put a
/// `mem_qemu_read_gpa_callback_failed` line on the always-on fail channel where
/// it read as a device fault rather than as expected control flow.
///
/// Both directions are asserted, because a pre-flight that simply refused
/// everything would satisfy the first half alone — and this path is what paints
/// the console once the kernel relocates it into system RAM.
#[test]
fn the_efi_console_paint_refuses_a_bar_backed_framebuffer_and_accepts_a_ram_one() {
    let w = crate::model::EFI_BOOT_WIDTH;
    let h = crate::model::EFI_BOOT_HEIGHT;
    let stride = w * RGBA8_BPP;
    let fb = 0x8000_0000u64;
    let span = (h as u64) * (stride as u64);

    let paint = |non_ram: bool| {
        let mut state = Device::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
        state.registers.gfx.efi_fb_start = fb;
        state.registers.gfx.efi_fb_stride = stride;
        let mut host = FakeHost::new();
        // Real bytes either way, so the two cases differ only in how the host
        // classifies the span -- which is what production refuses on.
        host.map_range(fb, span as usize, 0);
        if non_ram {
            host.mark_non_ram(fb, span);
        }
        let mut dst = vec![0u8; (stride as usize) * (h as usize)];
        paint_efi_console(&state, &host, &mut dst, stride, w, h)
    };

    assert!(
        !paint(true),
        "a console framebuffer sitting in device memory must be refused, so the \
         caller falls through to the BAR1 door instead of reading rows that \
         cannot resolve"
    );
    assert!(
        paint(false),
        "a console framebuffer in guest RAM is the relocated-console case this \
         path exists for, and must still paint"
    );
}

/// A framebuffer span with one non-RAM page in the middle is refused, even
/// though both of its endpoints are RAM.
///
/// The pre-flight used to be two `is_ram_gpa` calls on the span's first and
/// last byte, which is a two-point sample of eight megabytes. A driven x86 boot
/// refused a row read 375 rows into the span with both endpoints answering RAM
/// — `address=0x802bf200 len=7680`, exactly `fb + 375 * stride`.
///
/// The fixture discriminates because `FakeHost::read_gpa` serves any mapped
/// range and consults `non_ram` not at all: under the endpoint pre-flight this
/// paint *succeeds*, returning a frame with a row read out of a region the host
/// says is not memory.
/// A row that stops being RAM *during* the copy is named apart from a row the
/// two host doors genuinely disagree about.
///
/// The pre-flight walks the whole span and finds it all RAM; the copy then takes
/// eight megabytes a row at a time while an early-boot guest is relocating its
/// console out of this device's BAR1. The guest is entitled to retract the
/// memory mid-copy and this device cannot close that race — so the read fails,
/// and the question is only which of the two things it says.
///
/// `console_efi_row_vouched_then_refused` is documented as a healthy zero: the
/// pre-flight is the whole reason that arm should be unreachable, so a firing is
/// a defect in one of the two host doors. It had one slug, and the first driven
/// boot to read it read this benign case instead — which is how a healthy-zero
/// alarm stops being one. Asking the walk again about the failing row alone is
/// what separates them, and the two carry different slugs so the benign one
/// cannot spend the other's `fail_once` latch.
#[test]
fn a_console_row_that_leaves_ram_mid_copy_is_not_the_two_doors_disagreeing() {
    let w = crate::model::EFI_BOOT_WIDTH;
    let h = crate::model::EFI_BOOT_HEIGHT;
    let stride = w * RGBA8_BPP;
    let fb = 0x8000_0000u64;
    let span = (h as u64) * (stride as u64);
    let page = 1u64 << crate::model::PAGE_SHIFT_X86;

    let mut state = Device::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
    state.registers.gfx.efi_fb_start = fb;
    state.registers.gfx.efi_fb_stride = stride;

    let mut host = FakeHost::new();
    host.map_range(fb, span as usize, 0);
    // The row the live boot refused. Reading row 0 retracts it, so the
    // pre-flight — which ran before any read — vouched for a span that is no
    // longer all RAM by the time the copy reaches row 867.
    let victim = fb + 867 * stride as u64;
    host.arm_unmap_on_read(fb, stride as u64, victim / page * page, page);

    let mut dst = vec![0u8; (stride as usize) * (h as usize)];
    assert!(
        !paint_efi_console(&state, &host, &mut dst, stride, w, h),
        "a row that stopped being RAM must still refuse the paint"
    );

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(
        log.contains("console_efi_row_left_ram_mid_copy"),
        "a row the guest retracted mid-copy must say so"
    );
    assert!(
        !log.contains("console_efi_row_vouched_then_refused"),
        "it must not be reported as the two host doors disagreeing, which is \
         the arm that is supposed to be a healthy zero"
    );
    assert!(log.contains("row=867"), "the line must name the row");
}

#[test]
fn the_efi_console_paint_refuses_a_span_whose_hole_is_not_at_either_end() {
    let w = crate::model::EFI_BOOT_WIDTH;
    let h = crate::model::EFI_BOOT_HEIGHT;
    let stride = w * RGBA8_BPP;
    let fb = 0x8000_0000u64;
    let span = (h as u64) * (stride as u64);
    let page = 1u64 << crate::model::PAGE_SHIFT_X86;

    use crate::runtime::host::HostPageViews;

    let mut state = Device::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
    state.registers.gfx.efi_fb_start = fb;
    state.registers.gfx.efi_fb_stride = stride;

    let mut host = FakeHost::new();
    host.map_range(fb, span as usize, 0);
    // The row the live boot refused, floored to its page. Deliberately neither
    // the first nor the last page of the span.
    let hole = (fb + 375 * stride as u64) / page * page;
    host.mark_non_ram(hole, page);

    assert!(
        host.is_ram_gpa(fb) && host.is_ram_gpa(fb + span - 1),
        "the fixture must reproduce the trap: both endpoints answer RAM"
    );

    let mut dst = vec![0u8; (stride as usize) * (h as usize)];
    assert!(
        !paint_efi_console(&state, &host, &mut dst, stride, w, h),
        "a span with an interior non-RAM page must be refused before any row is \
         read, not vouched for by its two endpoints"
    );
}
