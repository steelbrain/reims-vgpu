//! Tests for the IOSurface texture mapped-surface writers.
//!
//! Out of line for the reason the sibling `runtime/` modules that already do
//! this have: colocated, these 2,291 lines were 48% of `mapping_write.rs` — the
//! module was half test by line count, and the writers themselves were the
//! harder half to find.

use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
use crate::runtime::host::FakeHost;
use reims_vgpu_paging::geometry::{
    MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
    MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
};

#[test]
fn a_landed_surface_write_returns_the_epoch_it_published() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mapping_id = 30;
    state.map_surface(mapping_id);
    let before = state
        .surfaces
        .mappings
        .get(&mapping_id)
        .unwrap()
        .content
        .surface_epoch;

    let published = note_iosurface_texture_landed(&mut state, mapping_id, 0, 4096)
        .expect("a registered mapping publishes an epoch");
    let mapping = state.surfaces.mappings.get(&mapping_id).unwrap();

    assert_ne!(published, before);
    assert_eq!(published, mapping.content.surface_epoch);
}

/// `mapping_geom_window` puts each measurement in the field of its own name.
///
/// `SurfaceWindow`'s four fields are two `u64`s and two `u32`s, so
/// `base_off`/`span_end` can cross silently and so can `bpr`/`bpp`. The
/// mapping below is chosen so all four read differently, which is what
/// makes a crossing observable at all.
///
/// The row pitch is asserted by its relationships, not as a number: a
/// 4-wide BGRA8 surface reports `bpr = 128` against a tight row of 16,
/// because `iosurface_texture_sample_window` aligns the pitch up. Hard-coding either
/// value would make this a test of that alignment rather than of which
/// field holds what.
#[test]
fn the_surface_window_names_which_measurement_is_which() {
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mid = 30u32;
    state.map_surface(mid);
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.lifecycle.generation = 1;
        m.publish_geometry_for_test(4, 4, MTL_FORMAT_BGRA8_UNORM);
    }
    let rect = Rect {
        origin_x: 0,
        origin_y: 0,
        width: 4,
        height: 4,
    };
    let w = mapping_geom_window(&state, mid, rect).expect("a geometry window");
    assert_eq!(w.bpp, 4, "BGRA8 is four bytes per texel");
    assert!(
        w.bpr >= 4 * w.bpp,
        "a row holds at least the four texels: bpr={} bpp={}",
        w.bpr,
        w.bpp
    );
    assert_ne!(
        w.bpr, w.bpp,
        "the two must differ here or this test could not see them swapped"
    );
    assert_eq!(
        w.span_end - w.base_off,
        u64::from(w.bpr) * 4,
        "the span reaches exactly the four rows the rectangle asked for"
    );
    // A rectangle past the declared extent has no window at all.
    assert!(mapping_geom_window(
        &state,
        mid,
        Rect {
            origin_x: 1,
            ..rect
        }
    )
    .is_none());
}

/// A tight full-page-aligned surface names exactly the pages its bytes
/// occupy, and no more.
///
/// The last page is the one holding the last *texel*, not the one holding
/// `bpr * height`. A plan that rounded up to the row pitch would hand the GPU
/// write access to a page past the surface on every flush of a padded layout,
/// and the guest owns whatever is in it.
#[test]
fn a_tight_window_names_the_pages_its_texels_occupy() {
    // 1920x1080 BGRA8, tight, starting at offset 0 of a 4 KiB-page guest.
    let (page, bpr) = (4096u64, 1920 * 4u32);
    let span = u64::from(bpr) * 1080;
    let plan = plan_guest_window(usize::MAX, page, 0, span, bpr, 1920, RGBA8_BPP)
        .expect("a tight window plans");
    assert_eq!(plan.first_page, 0);
    assert_eq!(plan.last_page, ((span - 1) / page) as usize);
    assert_eq!(plan.in_page, 0);
    assert_eq!(plan.row_length_texels, 1920);
    // Exactly the pages the bytes are in: 1920*4*1080 is a whole number of
    // 4 KiB pages, so the last texel is the last byte of the last one.
    assert_eq!(plan.pages() as u64, span / page);
}

/// A window starting part-way into a page reports that offset, and the page
/// it starts in is the first the guest reference names.
///
/// This is the whole reason the plan exists. The reference starts at a page
/// boundary and the sample window does not, so a copy that took the window's
/// mapping offset as its `bufferOffset` would land the frame `first_page *
/// page_size` bytes early — off the front of the reference entirely for any
/// surface past the first page.
#[test]
fn a_window_starting_inside_a_page_carries_the_offset_and_not_the_mapping_one() {
    let (page, bpr) = (4096u64, 256 * 4u32);
    let base = 3 * page + 512;
    let span = base + u64::from(bpr) * 8;
    let plan = plan_guest_window(usize::MAX, page, base, span, bpr, 256, RGBA8_BPP).expect("plans");
    assert_eq!(plan.first_page, 3);
    assert_eq!(plan.in_page, 512);
    // Not the mapping offset: that is the bug this asserts against.
    assert_ne!(plan.in_page, base);
}

/// Page shift is explicit, so the same window plans differently on the two
/// guests. A helper that assumed 4 KiB would name four times too many pages
/// on arm64 and expose three quarters of a surface it was never asked for.
#[test]
fn the_same_window_spans_fewer_pages_on_a_sixteen_kilobyte_guest() {
    let bpr = 1024 * 4u32;
    let span = u64::from(bpr) * 64;
    let x86 =
        plan_guest_window(usize::MAX, 4096, 0, span, bpr, 1024, RGBA8_BPP).expect("plans on x86");
    let arm = plan_guest_window(usize::MAX, 16384, 0, span, bpr, 1024, RGBA8_BPP)
        .expect("plans on arm64");
    assert_eq!(x86.pages(), arm.pages() * 4);
}

/// A padded guest pitch travels as texels, because that is what
/// `bufferRowLength` is. The inter-row bytes are never named, so the guest's
/// own content in the padding survives the flush — matching the copying
/// rail, which writes row by row and skips it too.
#[test]
fn a_padded_pitch_becomes_a_row_length_in_texels() {
    let bpr = 2048 * 4u32;
    let plan = plan_guest_window(
        usize::MAX,
        4096,
        0,
        u64::from(bpr) * 4,
        bpr,
        1600,
        RGBA8_BPP,
    )
    .expect("plans");
    assert_eq!(plan.row_length_texels, 2048);
}

/// A plane's pitch is a count of **its own** texels, so a wider destination
/// resolves the same byte pitch to fewer of them.
///
/// `bufferRowLength` is what this number becomes, and Vulkan multiplies it by
/// the image's texel size to space the rows. Dividing a half-float plane's byte
/// pitch by four reports twice as many texels as the row holds — a value that
/// passes every validity rule and lands every row after the first at half its
/// true spacing, so the frame arrives sheared into the top half of the window
/// with no refusal anywhere. Both spellings are asserted from one byte pitch,
/// because the defect is the *relation* between them and either one alone reads
/// as correct.
#[test]
fn a_pitch_resolves_to_the_destinations_own_texels() {
    use reims_vgpu_core::pixel_format::RGBA16F_BPP;
    // One tightly-packed row of 256 half-float RGBA texels.
    let bpr = 256 * RGBA16F_BPP;
    let span = u64::from(bpr) * 4;
    let wide = plan_guest_window(usize::MAX, 4096, 0, span, bpr, 256, RGBA16F_BPP)
        .expect("a half-float plane plans");
    assert_eq!(
        wide.row_length_texels, 256,
        "a tight row is exactly the frame's width in the destination's texels"
    );
    let narrow = plan_guest_window(usize::MAX, 4096, 0, span, bpr, 256, RGBA8_BPP)
        .expect("the same bytes as an eight-bit plane plans");
    assert_eq!(
        narrow.row_length_texels,
        wide.row_length_texels * 2,
        "the same byte pitch is twice as many texels at half the width"
    );
    // And a pitch that is whole texels at four bytes but not at eight is
    // refused for the wide destination rather than truncated into one.
    assert_eq!(
        plan_guest_window(usize::MAX, 4096, 0, span, bpr + RGBA8_BPP, 256, RGBA16F_BPP),
        Err(GpuWritebackDecline::PitchNotTexels {
            bpr: bpr + RGBA8_BPP
        })
    );
}

/// Every value a `VkBufferImageCopy` cannot express declines by name rather
/// than being rounded into one it can.
///
/// `bufferOffset` must be a multiple of the texel block size and
/// `bufferRowLength` is counted in texels; a copy submitted with either one
/// wrong is undefined behaviour, not a misplaced frame, so neither may be
/// silently repaired.
#[test]
fn a_geometry_the_copy_cannot_express_declines_by_name() {
    // A row pitch that is not a whole number of texels.
    assert_eq!(
        plan_guest_window(usize::MAX, 4096, 0, 4096, 1023, 1, RGBA8_BPP),
        Err(GpuWritebackDecline::PitchNotTexels { bpr: 1023 })
    );
    // A window starting on an odd byte inside its page.
    assert_eq!(
        plan_guest_window(usize::MAX, 4096, 2, 4096, 4, 1, RGBA8_BPP),
        Err(GpuWritebackDecline::OffsetNotTexelAligned { in_page: 2 })
    );
    // A page list that stops before the window does. Writing anyway would
    // export whatever the shorter list's tail happens to name.
    assert_eq!(
        plan_guest_window(2, 4096, 0, 3 * 4096, 4, 1, RGBA8_BPP),
        Err(GpuWritebackDecline::PageListShort { need: 3, have: 2 })
    );
    // An empty or inverted window has no destination at all.
    assert_eq!(
        plan_guest_window(usize::MAX, 4096, 100, 100, 4, 1, RGBA8_BPP),
        Err(GpuWritebackDecline::NotWritable)
    );
    // A pitch narrower than the frame. Vulkan requires `bufferRowLength` to
    // be zero or at least the extent's width, so this is an invalid copy
    // rather than a tight one — and a plan that let it through would submit
    // it, because nothing else in the path re-derives the row length.
    assert_eq!(
        plan_guest_window(usize::MAX, 4096, 0, 4096, 4 * 8, 9, RGBA8_BPP),
        Err(GpuWritebackDecline::PitchNotTexels { bpr: 32 })
    );
}

/// "This host cannot import" and "these pages would not resolve" are different
/// findings and must not share a name.
///
/// They did, twice, from opposite directions, and the fix for the first is
/// what made the second reachable.
///
/// Originally both `granularity()` returning `None` and any refusal from
/// `guest_ram_map` returned `NoGuestImport`, so a driven x86 boot printed
/// twenty `gpuwb_no_guest_import` lines — one per 1080p mapping — on a host
/// whose `vk_caps` said `host_pointer_import=supported`. The real cause was
/// `Scattered`, reported by `guest_ram_map` exactly once for the whole boot
/// because `report_once` latches on `first_sight`. Ranking the fail channel by
/// `reason=`, the documented way to read that log, put the twenty at the top
/// under a name that contradicted the capability line.
///
/// Splitting them fixed that and left two spellings of "this host cannot
/// import" — a granularity read here and the resolution over in
/// `guest_ram_map` — which is the divergence class in its own right. So the
/// distinction now rides on `via=` rather than on the slug: one variant, one
/// authority (`guest_ram_map::standing_refusal`), and the inner check named on
/// every record.
///
/// The `assert_ne!`s are the regression. What must never come back is two
/// records that a `reason=` ranking cannot tell apart — whichever field
/// carries the difference.
#[test]
fn a_refused_page_list_does_not_report_itself_as_a_host_without_the_import() {
    use crate::observe::Decline;
    use crate::runtime::guest_ram_map::MapRefusal;

    let via = |d: &GpuWritebackDecline| {
        d.fields()
            .into_iter()
            .find(|(k, _)| *k == "via")
            .map(|(_, v)| v)
    };

    let scattered = GpuWritebackDecline::GuestRefRefused {
        refusal: MapRefusal::Scattered {
            pages: 32,
            runs: 9,
            first: 0x39bb_6a000,
        },
    };
    let no_import = GpuWritebackDecline::GuestRefRefused {
        refusal: MapRefusal::NoBackendImport,
    };
    assert_ne!(
        via(&scattered),
        via(&no_import),
        "a refused page list must not read as a host that cannot import"
    );
    assert_eq!(
        via(&no_import).as_deref(),
        Some("guest_ram_map_no_backend_import"),
        "the host-wide statement still names itself, on the record rather than \
         on one line elsewhere in the log"
    );

    // The check that refused, and its own numbers, on this record.
    let fields = scattered.fields();
    assert_eq!(via(&scattered).as_deref(), Some("guest_ram_map_scattered"));
    assert_eq!(
        fields
            .iter()
            .find(|(k, _)| *k == "pages")
            .map(|(_, v)| v.as_str()),
        Some("32")
    );
    // A host-wide fact has nothing per-record to carry beyond its own name.
    assert_eq!(no_import.fields().len(), 1);

    // A different inner check must reach the log differently, or carrying it
    // buys nothing.
    let not_in_import = GpuWritebackDecline::GuestRefRefused {
        refusal: MapRefusal::GpaNotInAnyImport { gpa: 0x1000 },
    };
    assert_ne!(via(&not_in_import), via(&scattered));
}

/// A rect taller than the window it names is refused on **both** storage
/// shapes, and writes nothing past the window.
///
/// `write_rect_raw_at_impl` has three arms that all write guest memory, and
/// the bound used to be on two of them. The per-row fragmented arm reached
/// `mapper::write_mapping_bytes`, which bounds against the whole mapping's
/// page span rather than this plane's window, so an over-tall rect landed in
/// whatever follows the window — on a multi-plane IOSurface, the next plane's
/// pixels — with no fail line. The packed arm refused the same call.
///
/// So the loop over `packed` is the test: an assertion on one shape alone
/// passed throughout, which is how the hole survived. `span_end` is set short
/// of the mapping's real extent on purpose, because that gap is exactly the
/// region the unbounded arm wrote into and the bounded one did not.
#[test]
fn a_rect_taller_than_its_window_is_refused_on_both_storage_shapes() {
    use crate::model::PAGE_SHIFT_X86;
    const PAGE: u64 = 1 << PAGE_SHIFT_X86;
    const W: u32 = 16;
    const BPR: u32 = W * 4;

    for packed in [true, false] {
        let mut state = Device::new(DeviceId(3), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = !packed;
        let base_pfn = 0x60u32;
        host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 4 * PAGE as usize, 0xee);
        let order: Vec<u32> = if packed {
            (0..4).collect()
        } else {
            vec![3, 2, 1, 0]
        };
        let entries: Vec<u32> = order
            .iter()
            .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
            .collect();
        state.map_surface(5);
        state.attach_mapping_internal(5, 0);
        let m = state.surfaces.mappings.get_mut(&5).unwrap();
        m.lifecycle.internal_kva = 1;
        m.pages.entries = entries;

        // The window is one page: 64 rows of 64 bytes. The rect asks for 80.
        let span_end = PAGE;
        let rows = (PAGE / BPR as u64) as u32;
        let over = rows + 16;
        let src = vec![0x5au8; (over as usize) * BPR as usize];

        assert!(
            !write_rect_raw_at(
                &mut state,
                &mut host,
                5,
                SurfaceWindow {
                    base_off: 0,
                    bpr: BPR,
                    span_end,
                    bpp: 4
                },
                Rect {
                    origin_x: 0,
                    origin_y: 0,
                    width: W,
                    height: over
                },
                &src,
                BPR
            ),
            "packed={packed}: a rect past the window's last row must be refused"
        );

        // The bytes after the window still hold the fill the mapping was
        // seeded with, on both shapes.
        let mut after = [0u8; 16];
        assert!(mapper::read_mapping_bytes(
            &mut state, &mut host, 5, span_end, &mut after
        ));
        assert_eq!(
            after, [0xeeu8; 16],
            "packed={packed}: the refused rect must not have written past the window"
        );
    }
}

/// The skip test below fills the frame with one repeated byte, so it proves
/// which *pages* were written and nothing at all about which row went
/// where: a writeback that repeated row 0 sixty-four times, or that shifted
/// every row by one, passes it unchanged. That is not hypothetical — a
/// BGRA8 row no longer passes through the conversion scratch buffer on its
/// way to the guest, so the source offset, the tight row length and the
/// destination stride are now read from three places that used to be two.
///
/// Both storage shapes, because they place rows by different means, and a
/// non-BGRA format so the staged path is exercised beside the direct one.
#[test]
fn a_writeback_lands_every_row_at_its_own_offset() {
    use crate::model::PAGE_SHIFT_X86;
    const PAGE: u64 = 1 << PAGE_SHIFT_X86;
    const W: u32 = 64;
    const H: u32 = 64;
    const BPR: usize = (W * 4) as usize;

    for packed in [true, false] {
        for format in [MTL_FORMAT_BGRA8_UNORM, pixel_format::MTL_FORMAT_RGBA8_UNORM] {
            let mut state = Device::new(DeviceId(2), PAGE_SHIFT_X86);
            let mut host = FakeHost::new();
            host.strict_linux_map = !packed;
            let base_pfn = 0x40u32;
            host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 8 * PAGE as usize, 0);
            let order: Vec<u32> = if packed {
                (0..4).collect()
            } else {
                vec![3, 2, 1, 0]
            };
            let entries: Vec<u32> = order
                .iter()
                .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
                .collect();
            state.map_surface(4);
            state.attach_mapping_internal(4, 0);
            let m = state.surfaces.mappings.get_mut(&4).unwrap();
            m.lifecycle.internal_kva = 1;
            m.pages.entries = entries;
            assert!(state.set_mapping_geom(4, W, H, format));

            // Row y is filled with the byte y in every channel, so a row
            // that lands at the wrong offset is visible as the wrong value
            // and a duplicated row is visible as a repeat. Channel order
            // does not matter to that, which is what lets one frame test
            // both formats.
            let mut frame = vec![0u8; (W * H * 4) as usize];
            for y in 0..H as usize {
                frame[y * BPR..(y + 1) * BPR].fill(y as u8);
            }
            assert!(write_bgra8(&mut state, &mut host, 4, &frame, W * 4, W, H));

            for y in 0..H as usize {
                // Which guest page this row's first byte lives in, and where
                // in it, walking the same page list the mapping declares.
                let off = y * BPR;
                let gpa = ((base_pfn as u64 + order[off / PAGE as usize] as u64) << PAGE_SHIFT_X86)
                    + (off as u64 % PAGE);
                let mut got = [0u8; 4];
                host.read_gpa(gpa, &mut got).unwrap();
                assert!(
                    got.iter().all(|b| *b == y as u8),
                    "packed={packed} fmt={format:#x} row {y} must read {y:#x}, got {got:?}"
                );
            }
        }
    }
}

/// A host write into guest RAM must announce itself so a retained resource
/// reached through an alias cannot keep serving the bytes it replaced.
///
/// The read half is the other half of the contract: reads share the same
/// mapping walk, and a read that moved the record would make every reader
/// re-fetch on account of a reader.
/// A dropped writeback must say which check dropped it.
///
/// The composite surface is the largest frame this device moves, and losing
/// one is a wrong picture that then persists. Sixteen refusal sites used to
/// answer with a bare `false` that the caller rendered as one
/// `reason=write_refused`, so a reader could tell that the frame had been
/// dropped and nothing else.
///
/// `GeometryMoved` is the one this test drives because it is the one a
/// deferred window can reach without anything being broken: the frame is
/// armed at one rect and the surface is re-published at another — which is
/// what an appearance change or a wallpaper switch does. Asserting the
/// *specific* route is the whole point; asserting `!ok` would pass with every
/// site sharing a slug again.
#[test]
fn a_writeback_refused_because_the_geometry_moved_says_so_by_name() {
    use crate::model::PAGE_SHIFT_X86;
    const PAGE: u64 = 1 << PAGE_SHIFT_X86;

    let mut state = Device::new(DeviceId(3), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let base_pfn = 0x40u32;
    host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 8 * PAGE as usize, 0x55);
    state.map_surface(4);
    state.attach_mapping_internal(4, 0);
    let m = state.surfaces.mappings.get_mut(&4).unwrap();
    m.lifecycle.internal_kva = 1;
    m.pages.entries = (0..4)
        .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
        .collect();
    // Latched at 64x64; the frame below is armed at 32x32, as a window armed
    // before a re-publish would be.
    assert!(state.set_mapping_geom(4, 64, 64, MTL_FORMAT_BGRA8_UNORM));

    // Deltas, not absolutes: the route counters are process-global and every
    // other test in this binary shares them, so a sibling reading nonzero
    // says nothing about this write.
    const SIBLINGS: [&str; 3] = [
        "surface_write_mapping_absent",
        "surface_write_pages_not_ours",
        "surface_write_source_short",
    ];
    let before = crate::runtime::drain::store_route_count("surface_write_geometry_moved");
    let before_siblings: Vec<u64> = SIBLINGS
        .iter()
        .map(|r| crate::runtime::drain::store_route_count(r))
        .collect();

    let frame = vec![0xAAu8; 32 * 32 * 4];
    assert!(
        !write_bgra8(&mut state, &mut host, 4, &frame, 32 * 4, 32, 32),
        "a frame whose rect is not the surface's must not land"
    );
    assert_eq!(
        crate::runtime::drain::store_route_count("surface_write_geometry_moved"),
        before + 1,
        "the writeback was dropped without naming the check that dropped it"
    );
    // And no sibling check moved, or the slug is not discriminating.
    for (route, was) in SIBLINGS.iter().zip(before_siblings) {
        assert_eq!(
            crate::runtime::drain::store_route_count(route),
            was,
            "{route} fired for a geometry mismatch"
        );
    }
}

/// The IOSurface texture licence judges the window it is given, not the surface's extent.
///
/// A render Store's destination *is* the surface, so that caller refuses a frame
/// whose rect is not the mapping's latched geometry — the test above drives
/// exactly that, and it stays where it belongs, in the caller. A compute
/// dispatch's destination is not a scanout: writing a sub-rectangle of a surface
/// is ordinary, and the licence resolving its own full-extent window refused
/// every one of them. On a driven macos-13 boot that was 15 of the 19 remaining
/// compute readbacks, all `GeometryMoved`, at extents like 44x26 of a 64x64
/// surface and 128x512 of a 512x512 one.
///
/// So the assertion is that extent is no longer a *term*: a sub-rectangle and a
/// whole-surface destination over the same mapping must reach the same decline,
/// and it must not be `GeometryMoved`. `FakeHost` publishes no guest-RAM import,
/// so both stop at the reference gate — which is downstream of every rule the
/// licence still owns, and therefore says both got through all of them.
#[test]
fn a_iosurface_texture_licence_judges_the_callers_window_and_not_the_surfaces_extent() {
    use crate::model::PAGE_SHIFT_X86;
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    const PAGE: u64 = 1 << PAGE_SHIFT_X86;

    crate::runtime::guest_ram_map::reset();
    crate::runtime::guest_ram::forget_import_limits();
    let mut state = Device::new(DeviceId(9), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let base_pfn = 0x40u32;
    host.map_range(
        (base_pfn as u64) << PAGE_SHIFT_X86,
        16 * PAGE as usize,
        0x55,
    );
    state.map_surface(7);
    state.attach_mapping_internal(7, 0);
    let m = state.surfaces.mappings.get_mut(&7).unwrap();
    m.lifecycle.internal_kva = 1;
    m.pages.entries = (0..16)
        .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
        .collect();
    assert!(state.set_mapping_geom(7, 64, 64, MTL_FORMAT_BGRA8_UNORM));

    let held: reims_vgpu_protocol::StorageImageFormat =
        pixel_format::store_texel_order(MTL_FORMAT_BGRA8_UNORM)
            .expect("BGRA8 has a linear texel")
            .into();
    let dest = |width, height| IOSurfaceDestination {
        mapping_id: 7,
        base_off: 0,
        bpr: 64 * 4,
        span_end: u64::from(height) * 64 * 4,
        width,
        height,
        format: MTL_FORMAT_BGRA8_UNORM,
    };

    let whole = licence_iosurface_texture_surface(&mut state, &mut host, held, &dest(64, 64));
    let part = licence_iosurface_texture_surface(&mut state, &mut host, held, &dest(44, 26));
    for (what, got) in [("the whole surface", whole), ("a sub-rectangle", part)] {
        match got {
            Err(GpuWritebackDecline::GuestRefRefused { .. }) => {}
            other => panic!(
                "{what} must reach the reference gate, and only that gate; got {:?}",
                other.err()
            ),
        }
    }
    crate::runtime::guest_ram_map::reset();
}

/// An IOSurface texture destination follows its mapping allocation, not the amount of
/// unrelated RAM in the VM. A host may be unable to keep every RAMBlock
/// imported while still admitting this exact resource-sized allocation.
#[test]
fn a_iosurface_texture_resource_import_survives_a_whole_ram_map_refusal() {
    use crate::model::PAGE_SHIFT_X86;
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    const PAGE: u64 = 1 << PAGE_SHIFT_X86;
    const MAPPING_PAGES: u64 = 16;

    crate::runtime::guest_ram_map::reset();
    crate::runtime::guest_ram::latch_import_limits(
        PAGE,
        MAPPING_PAGES * PAGE,
        MAPPING_PAGES * PAGE,
    );

    let mut host = FakeHost::new();
    host.stable_map_pages = true;
    let base_pfn = 0x40u32;
    host.map_range(
        (base_pfn as u64) << PAGE_SHIFT_X86,
        (MAPPING_PAGES * PAGE) as usize,
        0x55,
    );
    // This range is not part of the surface. It exists only to put the optional
    // all-RAM import past the published heap capacity.
    host.map_range(0x1000_0000, (2 * MAPPING_PAGES * PAGE) as usize, 0x33);
    assert!(matches!(
        crate::runtime::guest_ram_map::standing_refusal(&mut host),
        Some(crate::runtime::guest_ram_map::MapRefusal::ImportExceedsHeap { .. })
    ));

    let mut state = Device::new(DeviceId(9), PAGE_SHIFT_X86);
    state.map_surface(7);
    state.attach_mapping_internal(7, 0);
    let mapping = state.surfaces.mappings.get_mut(&7).unwrap();
    mapping.lifecycle.internal_kva = 1;
    mapping.pages.entries = (0..MAPPING_PAGES as u32)
        .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
        .collect();
    assert!(state.set_mapping_geom(7, 64, 64, MTL_FORMAT_BGRA8_UNORM));

    let held: reims_vgpu_protocol::StorageImageFormat =
        pixel_format::store_texel_order(MTL_FORMAT_BGRA8_UNORM)
            .expect("BGRA8 has a linear texel")
            .into();
    let licence = licence_iosurface_texture_surface(
        &mut state,
        &mut host,
        held,
        &IOSurfaceDestination {
            mapping_id: 7,
            base_off: 0,
            bpr: 64 * 4,
            span_end: 64 * 64 * 4,
            width: 64,
            height: 64,
            format: MTL_FORMAT_BGRA8_UNORM,
        },
    )
    .expect("the mapping-sized import remains legal");
    assert_eq!(licence.target.runs.len(), 1);
    assert_eq!(licence.target.runs[0].guest.requested(), 64 * 64 * 4);

    crate::runtime::guest_ram_map::reset();
    crate::runtime::guest_ram::forget_import_limits();
}

#[test]
fn writing_guest_pages_moves_the_host_write_record_and_reading_them_does_not() {
    use crate::model::PAGE_SHIFT_X86;
    const PAGE: u64 = 1 << PAGE_SHIFT_X86;
    const W: u32 = 64;
    const H: u32 = 64;

    let mut state = Device::new(DeviceId(3), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let base_pfn = 0x40u32;
    host.map_range((base_pfn as u64) << PAGE_SHIFT_X86, 8 * PAGE as usize, 0x55);
    let entries: Vec<u32> = (0..4)
        .map(|i| ((base_pfn + i) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID)
        .collect();
    state.map_surface(4);
    state.attach_mapping_internal(4, 0);
    let m = state.surfaces.mappings.get_mut(&4).unwrap();
    m.lifecycle.internal_kva = 1;
    m.pages.entries = entries;
    assert!(state.set_mapping_geom(4, W, H, MTL_FORMAT_BGRA8_UNORM));

    let before = state.content.host_writes.epoch();
    let frame = vec![0xAAu8; (W * H * 4) as usize];
    assert!(write_bgra8(&mut state, &mut host, 4, &frame, W * 4, W, H));
    assert_ne!(
        state.content.host_writes.epoch(),
        before,
        "a write into the guest's pages went unannounced"
    );

    let after_write = state.content.host_writes.epoch();
    let mut out = vec![0u8; (W * H * 4) as usize];
    assert!(crate::runtime::mapper::read_mapping_bytes(
        &mut state, &mut host, 4, 0, &mut out
    ));
    assert_eq!(
        state.content.host_writes.epoch(),
        after_write,
        "a read moved the write record, so every reader now invalidates every reader"
    );
}

#[test]
fn write_bumps_generation() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x10u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(3);
    state.attach_mapping_internal(3, 0); // leave internal 0; set pages manually
    let m = state.surfaces.mappings.get_mut(&3).unwrap();
    m.lifecycle.internal_kva = 1;
    m.pages.entries = vec![entry];
    assert!(state.set_mapping_geom(3, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    let src = [0x11u8, 0x22, 0x33, 0x44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    // 2x2 BGRA, stride 8
    assert!(write_bgra8(&mut state, &mut host, 3, &src, 8, 2, 2));
    assert_eq!(
        state
            .surfaces
            .mappings
            .get(&3)
            .unwrap()
            .content
            .guest_page_generation,
        1
    );
}

/// A guest write drops only the storage-residency mirror windows it
/// intersects; disjoint sibling windows (ping-pong canvases) survive.
#[test]
fn mapping_write_invalidates_intersecting_residency_windows_only() {
    use crate::model::ComputeStorageResidencyKey;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x20u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(7);
    state.attach_mapping_internal(7, 0);
    let m = state.surfaces.mappings.get_mut(&7).unwrap();
    m.lifecycle.internal_kva = 1;
    m.pages.entries = vec![entry];
    let window = |surface_offset: u64, span_end: u64| {
        ComputeStorageResidencyKey::surface(
            7,
            state.surfaces.mappings[&7].lifecycle.generation,
            surface_offset,
            32,
            span_end,
            8,
            2,
            MTL_FORMAT_BGRA8_UNORM,
        )
    };
    let hit = window(0, 64);
    let survivor = window(1024, 1088);
    state.content.compute_residency.publish(hit, 5);
    state.content.compute_residency.publish(survivor, 5);
    let vouched = mapper::vouch_mapping_pages_verdict(&mut state, &host, 7)
        .1
        .expect("no walk to contradict");
    assert!(mapper::write_mapping_bytes(
        &mut state, &mut host, 7, 16, &[0u8; 32], &vouched
    ));
    assert!(!state.content.compute_residency.contains(&hit));
    assert!(state.content.compute_residency.contains(&survivor));
}

/// A direct IOSurface texture writeback must not land in a page the guest re-pointed
/// away, and this asserts it in the currency of the bug: the bytes of the
/// page the surface moved to.
///
/// The page-drift witness shipped with exactly one caller — the deferred
/// render flush — so this rail, which writes a full frame of pixels through
/// the mapping's page plan, was unguarded. The crash reports are the
/// receipt: WindowServer aborting in `small_free_list_remove_ptr_no_clear`,
/// and guest-kernel kalloc poison finding whole freed elements filled with
/// `0xff` from offset 0 — opaque white BGRA in memory already handed to
/// somebody else.
///
/// The fixture adopts a list walked through a live task page table, writes one
/// frame, and then re-points the PTE with no lifecycle packet. The second write
/// must refresh the resource's current backing: it lands in `data1`, leaves the
/// recycled `data0` untouched, preserves the logical resource generation, and
/// advances only the page-list generation.
#[test]
fn a_repointed_surface_writes_its_current_backing_and_leaves_the_old_owner_alone() {
    use crate::model::{SurfaceBackingWalk, PAGE_SHIFT_X86};
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    let page = 1u64 << PAGE_SHIFT_X86;
    let mut host = FakeHost::new();
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data0 = 4u64 << PAGE_SHIFT_X86;
    let data1 = 10u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, page as usize, 0);
    host.map_range(root_gpa, page as usize, 0);
    host.map_range(data0, page as usize, 0);
    host.map_range(data1, page as usize, 0);

    let st32 = |b: &mut [u8], v: u32| b[..4].copy_from_slice(&v.to_le_bytes());
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    host.write_gpa(dir_gpa, &d).unwrap();
    let mut pte = [0u8; 4];
    st32(&mut pte, (data0 >> PAGE_SHIFT_X86) as u32);
    host.write_gpa(root_gpa, &pte).unwrap();

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    state.define_task(1, page, 2);
    let mid = 6;
    state.map_surface(mid);
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.pages.entries =
            vec![(((data0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        m.pages.surface_walk = Some(SurfaceBackingWalk {
            task_id: 1,
            backing_pfn: 0,
            page_generation: m.pages.generation,
        });
    }

    // A tight 8x4 BGRA8 frame of opaque white — the payload the crash census
    // reads back out of freed guest memory.
    let (w, h) = (8u32, 4u32);
    let frame = vec![0xffu8; (w * h * RGBA8_BPP) as usize];
    let stride = w * RGBA8_BPP;
    assert!(
        write_bgra8(&mut state, &mut host, mid, &frame, stride, w, h),
        "the list was just walked from this page table, so the frame lands"
    );
    let mut landed = [0u8; 16];
    host.read_gpa(data0, &mut landed).unwrap();
    assert_eq!(landed, [0xffu8; 16], "the first frame reached data0");

    // Now the guest reclaims that page and hands it to something else, which
    // writes its own bytes there — a malloc small-zone region, say, whose
    // free-list pointers live *inside* the freed blocks. `0x5a` stands for
    // them. This is the step the crash reports are about: the corruption
    // lands in the page the surface *left*, not the one it moved to, because
    // the device's cached contig view is a `mach_vm_remap` of the old PFNs
    // and keeps resolving there.
    host.write_gpa(data0, &[0x5au8; 16]).unwrap();

    // And the surface is re-pointed. No MapMemory2, no UnmapMemory, no
    // ReplacePhysical — nothing on the wire, so nothing bumps the
    // incarnation.
    let generation_before = state
        .surfaces
        .mappings
        .get(&mid)
        .unwrap()
        .lifecycle
        .generation;
    st32(&mut pte, (data1 >> PAGE_SHIFT_X86) as u32);
    host.write_gpa(root_gpa, &pte).unwrap();
    assert_eq!(
        state
            .surfaces
            .mappings
            .get(&mid)
            .unwrap()
            .lifecycle
            .generation,
        generation_before,
        "no packet arrived, so nothing bumped the incarnation"
    );

    let page_generation_before = state.surfaces.mappings.get(&mid).unwrap().pages.generation;
    assert!(
        write_bgra8(&mut state, &mut host, mid, &frame, stride, w, h),
        "a complete current-backing walk must refresh and land the frame"
    );
    // The memory assertion comes first deliberately. A return value is this
    // crate's opinion about what it did; `data0` is what the guest will
    // actually find in its heap, and that is the claim the crash reports
    // dispute. Asserting the opinion first would let a rail that returns
    // false *after* writing pass the stronger half unread.
    let mut recycled = [0u8; 16];
    host.read_gpa(data0, &mut recycled).unwrap();
    assert_eq!(
        recycled, [0x5au8; 16],
        "the page the guest took away must still hold its new owner's bytes \
         — this is the guest heap corruption the whole goal is about"
    );
    let mut current = [0u8; 16];
    host.read_gpa(data1, &mut current).unwrap();
    assert_eq!(
        current, [0xffu8; 16],
        "the frame reached the current backing"
    );
    let mapping = state.surfaces.mappings.get(&mid).unwrap();
    assert_eq!(
        mapping.lifecycle.generation, generation_before,
        "physical migration does not create a new logical resource"
    );
    assert_ne!(
        mapping.pages.generation, page_generation_before,
        "page-bound proofs and aliases must move to a new backing generation"
    );
    assert_eq!(
        reims_vgpu_paging::geometry::mapper_entry_gpa(mapping.pages.entries[0], PAGE_SHIFT_X86),
        Some(data1),
        "the cached page list now names the current backing"
    );
}

/// compute_writeback_amplification: fragmented texture writeback imports
/// once per maximal GPA run, not once per image row.
#[test]
fn fragmented_raw_rect_bulk_imports_runs_not_rows() {
    use crate::model::PAGE_SHIFT_X86;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1usize << PAGE_SHIFT_X86;
    let gpa0 = 0x1000_0000u64;
    let gpa1 = 0x2000_0000u64;
    host.map_range(gpa0, page, 0x7e);
    host.map_range(gpa1, page, 0x7e);
    let mid = 19;
    state.map_surface(mid);
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.pages.entries = vec![
            (((gpa0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (((gpa1 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    let src = vec![0x2a; 4 * 16];
    // Contract span ends at the last texel, excluding trailing row
    // padding: (height - 1) * bpr + tight = 3 * 2048 + 16.
    assert!(write_full_rect_raw_at(
        &mut state, &mut host, mid, 0, 2048, 6160, 4, 4, 4, &src, 16,
    ));
    // One full-view attempt plus one successful import per maximal GPA run.
    // The refusal is cached for this page-list generation, so the extra call
    // is constant rather than per row. The old row loop took nine attempts for
    // these four rows and scaled with height.
    assert_eq!(host.map_pages_calls, 3);
    let calls_after_write = host.map_pages_calls;

    let mut row = [0u8; 16];
    assert!(mapper::read_mapping_bytes(
        &mut state, &mut host, mid, 4096, &mut row,
    ));
    assert_eq!(row, [0x2a; 16]);
    assert_eq!(calls_after_write, 3);
}

/// The BGRA row writers reach `observe::footprint`.
///
/// This is the rail the first cut of the footprint's completeness gate
/// missed, and it is the biggest one in the device. These writers never call
/// `mapper::write_mapping_bytes` and never call `HostOps::map_pages`: they
/// take a contig view through `contig_for_write` and poke rows straight into
/// it. A gate that scanned for `map_pages` callers therefore scored this
/// file as reaching guest RAM by no mechanism at all — which was true of the
/// needle and false of the file.
///
/// A missing mark here would have left the footprint empty of nearly every
/// pixel this device writes, and an empty set answers "we never wrote there"
/// to every panic it is asked about.
#[test]
fn a_bgra_row_write_marks_the_footprint_through_its_contig_view() {
    use crate::model::PAGE_SHIFT_X86;
    use crate::observe::footprint;

    let _fp = footprint::exclusive_for_tests();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let page = 1u64 << PAGE_SHIFT_X86;
    // Adjacent so the contig view packs — this is the path production takes.
    let gpa0 = 0x5000_0000u64;
    host.map_range(gpa0, 2 * page as usize, 0);
    let mid = 12u32;
    state.map_surface(mid);
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![
            (((gpa0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            ((((gpa0 + page) >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    assert!(
        mapper::ensure_contig_view(&mut state, &mut host, mid).is_some(),
        "the fixture must take the contig path or it tests the other rail"
    );
    let src = [0xFFu8; 16];
    assert!(write_bgra8(&mut state, &mut host, mid, &src, 8, 2, 2));

    assert!(
        footprint::wrote_gpa(gpa0),
        "the surface's first frame must be in the set"
    );
    assert!(
        !footprint::wrote_gpa(gpa0 + 8 * page),
        "and nothing beyond the surface"
    );
}

/// Linux product: non-packed page list still lands BGRA via multi-import.
#[test]
fn write_bgra8_fragmented_pages_multi_import() {
    use crate::model::PAGE_SHIFT_X86;
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    // 2×2 BGRA needs 16 bytes → one page; use two non-adjacent pages so
    // ensure_contig_view fails and multi-import is forced.
    let gpa0 = 0x3000_0000u64;
    let gpa1 = 0x4000_0000u64;
    host.map_range(gpa0, page as usize, 0);
    host.map_range(gpa1, page as usize, 0);
    let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
    let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
    let mid = 11u32;
    state.map_surface(mid);
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
    let src = [
        0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        0x10,
    ];
    assert!(write_bgra8(&mut state, &mut host, mid, &src, 8, 2, 2));
    let mut first = [0u8; 4];
    assert!(host.read_gpa(gpa0, &mut first).is_ok());
    assert_eq!(&first, &src[..4]);
    assert!(mapper::ensure_contig_view(&mut state, &mut host, mid).is_none());
}

/// The fragmented BGRA write path must land the texels and nothing else —
/// the same contract `write_bgra8_contig_writes_only_inside_the_sample_window`
/// pins on the pointer arm, asserted the same way so the two arms cannot
/// drift apart. Padding after the final row would overrun an exact IOSurface
/// allocation; padding *between* rows is inside the plane but is still
/// content the guest put there and this call never named.
/// The staged full-plane rect write must leave inter-row padding alone, on
/// the same assertion as its two siblings.
///
/// `write_rect_raw_at_impl` has three arms that must land identical guest
/// memory. The contiguous one pokes each row's texel bytes through a raw
/// pointer, the per-row fragmented one writes each row's bytes on its own,
/// and the staged one built a pitch-wide zeroed frame and stored it entire —
/// so every padding byte between rows was zeroed in the guest's pages. That
/// is the defect `write_bgra8_inner` was fixed for, in the sibling function,
/// and this arm's own comment cited the pre-fix behaviour as its
/// justification.
///
/// Asserted as "every byte outside the texel runs is unchanged", the same
/// whole-page comparison the BGRA arms are pinned by, so all three cannot
/// drift apart again. The fixture forces the staged arm two ways: two
/// non-adjacent pages (no contiguous view) and `full_plane`, and a pitch
/// wider than the packed row so `full_tight_direct` cannot take it either.
#[test]
fn write_full_rect_raw_staged_leaves_inter_row_padding_alone() {
    use crate::model::PAGE_SHIFT_X86;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    let gpa0 = 0x3500_0000u64;
    let gpa1 = 0x4600_0000u64;
    host.map_range(gpa0, page as usize, 0xCC);
    host.map_range(gpa1, page as usize, 0xCC);
    let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
    let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
    let mid = 14u32;
    state.map_surface(mid);
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
    assert!(
        mapper::ensure_contig_view(&mut state, &mut host, mid).is_none(),
        "two non-adjacent pages must take the staged path this test is about"
    );

    // 2x2 BGRA8: 8 tight bytes per row at a 128-byte pitch, so 120 bytes of
    // guest content sit between the rows.
    let (w, h, bpp) = (2u32, 2u32, 4u32);
    let tight = (w * bpp) as usize;
    let bpr = 128u32;
    let span_end = (bpr as u64) * (h as u64 - 1) + tight as u64;
    let src = [
        0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        0x10,
    ];
    assert!(write_full_rect_raw_at(
        &mut state,
        &mut host,
        mid,
        0,
        bpr,
        span_end,
        w,
        h,
        bpp,
        &src,
        tight as u32,
    ));

    let mut got = vec![0u8; page as usize];
    assert!(host.read_gpa(gpa0, &mut got).is_ok());
    let mut want = vec![0xCCu8; page as usize];
    want[..tight].copy_from_slice(&src[..tight]);
    want[bpr as usize..bpr as usize + tight].copy_from_slice(&src[tight..]);
    let first_diff = got.iter().zip(want.iter()).position(|(a, b)| a != b);
    assert_eq!(
        first_diff, None,
        "byte {first_diff:?} outside the texel runs was modified"
    );
}

#[test]
fn write_bgra8_fragmented_skips_final_row_padding() {
    use crate::model::PAGE_SHIFT_X86;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    let gpa0 = 0x3500_0000u64;
    let gpa1 = 0x4600_0000u64;
    host.map_range(gpa0, page as usize, 0xCC);
    host.map_range(gpa1, page as usize, 0xCC);
    let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
    let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
    let mid = 13u32;
    state.map_surface(mid);
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
    let src = [
        0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        0x10,
    ];
    assert!(
        mapper::ensure_contig_view(&mut state, &mut host, mid).is_none(),
        "two non-adjacent pages must take the staged path this test is about"
    );
    assert!(write_bgra8(&mut state, &mut host, mid, &src, 8, 2, 2));

    // No device descriptor, so the invented window applies: tight = 2 × 4,
    // bpr = ALIGN_UP(8, ROW_BYTES_ALIGN) = 128, two rows — a pitch wider
    // than the packed row, so there is inter-row padding to get wrong.
    let tight = 8usize;
    let bpr = 128usize;
    let mut got = vec![0u8; page as usize];
    assert!(host.read_gpa(gpa0, &mut got).is_ok());
    let mut want = vec![0xCCu8; page as usize];
    want[..tight].copy_from_slice(&src[..tight]);
    want[bpr..bpr + tight].copy_from_slice(&src[tight..]);
    let first_diff = got.iter().zip(want.iter()).position(|(a, b)| a != b);
    assert_eq!(
        first_diff, None,
        "byte {first_diff:?} outside the sample window was modified"
    );
}

/// The packed-contig BGRA write pokes rows straight into a raw host pointer,
/// so its only bound is the sample window `contig_for_span` validated. The
/// fragmented path's equivalent is checked by
/// `write_bgra8_fragmented_skips_final_row_padding`; this pins the same
/// contract on the pointer path, where an overrun is a write into whatever
/// guest allocation follows rather than a refused import.
///
/// Asserted as "every byte outside the window is unchanged", not just the
/// final row's padding: inter-row padding belongs to the same class and a
/// stride bug hits it first.
#[test]
fn write_bgra8_contig_writes_only_inside_the_sample_window() {
    use crate::model::PAGE_SHIFT_X86;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    let gpa = 0x7300_0000u64;
    host.map_range(gpa, page as usize, 0xCC);
    let pfn = (gpa >> PAGE_SHIFT_X86) as u32;
    let mid = 21u32;
    state.map_surface(mid);
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![(pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    }
    assert!(state.set_mapping_geom(mid, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    // No device descriptor, so the invented window applies: tight = 2 × 4,
    // bpr = ALIGN_UP(8, ROW_BYTES_ALIGN) = 128, two rows.
    let tight = 8usize;
    let bpr = 128usize;
    let src: Vec<u8> = (0..16u8).map(|i| i.wrapping_mul(17)).collect();
    assert!(
        mapper::ensure_contig_view(&mut state, &mut host, mid).is_some(),
        "one packed page must take the contig path this test is about"
    );
    assert!(write_bgra8(&mut state, &mut host, mid, &src, 8, 2, 2));

    let mut got = vec![0u8; page as usize];
    assert!(host.read_gpa(gpa, &mut got).is_ok());
    let mut want = vec![0xCCu8; page as usize];
    want[..tight].copy_from_slice(&src[..tight]);
    want[bpr..bpr + tight].copy_from_slice(&src[tight..]);
    let first_diff = got.iter().zip(want.iter()).position(|(a, b)| a != b);
    assert_eq!(
        first_diff, None,
        "byte {first_diff:?} outside the sample window was modified"
    );
}

/// Fragmented compute staging materializes the sample window once and
/// preserves padded-row addressing across non-contiguous guest pages.
#[test]
fn read_rect_raw_fragmented_pages_with_padded_rows() {
    use crate::model::PAGE_SHIFT_X86;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    let gpa0 = 0x5100_0000u64;
    let gpa1 = 0x6200_0000u64;
    host.map_range(gpa0, page as usize, 0);
    host.map_range(gpa1, page as usize, 0);
    let row0 = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let row1 = [9u8, 10, 11, 12, 13, 14, 15, 16];
    host.write_gpa(gpa0, &row0).unwrap();
    host.write_gpa(gpa1, &row1).unwrap();

    let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
    let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
    let mid = 12u32;
    state.map_surface(mid);
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![
            (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }
    let mut dst = [0u8; 16];
    assert!(read_rect_raw_at(
        &mut state,
        &mut host,
        mid,
        SurfaceWindow {
            base_off: 0,
            bpr: page as u32,
            span_end: page + row1.len() as u64,
            bpp: 4
        },
        Rect {
            origin_x: 0,
            origin_y: 0,
            width: 2,
            height: 2
        },
        &mut dst,
        8
    ));
    assert_eq!(&dst[..8], &row0);
    assert_eq!(&dst[8..], &row1);
}

/// A sub-rectangle of a padded plane over scattered guest pages reads through
/// one page-table walk, not through a plane-sized window.
///
/// This is the source half of every IOSurface texture to linear blit. Before the
/// rectangle shape reached this rail the arm below materialised the *whole*
/// sample window into a fresh zeroed `Vec` and then copied the wanted rows out
/// of it, so a rectangle covering a fraction of the plane still paid for all of
/// it twice. The fixture is deliberately a strict sub-rectangle in both axes
/// with a row stride wider than its rows, which is the shape the old arm could
/// not narrow and the old full-plane-tight special case could not claim.
#[test]
fn a_packed_sub_rectangle_of_a_scattered_plane_reads_through_one_walk() {
    use crate::model::PAGE_SHIFT_X86;
    use crate::runtime::drain::store_route_count;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    // Four pages, none adjacent, so the walk has to split into four runs.
    let gpas = [0x5100_0000u64, 0x6200_0000, 0x4300_0000, 0x7400_0000];
    let bpr = page as u32 / 2;
    let rows = 8u32;
    let bpp = 4u32;
    // Plane bytes, distinct per byte, laid into the pages the mapping names.
    let plane: Vec<u8> = (0..(bpr as u64 * rows as u64))
        .map(|i| (i % 251) as u8)
        .collect();
    for (i, gpa) in gpas.iter().enumerate() {
        host.map_range(*gpa, page as usize, 0);
        let lo = i * page as usize;
        host.write_gpa(*gpa, &plane[lo..lo + page as usize])
            .unwrap();
    }
    let mid = 21u32;
    state.map_surface(mid);
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = gpas
            .iter()
            .map(|gpa| {
                (((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID
            })
            .collect();
    }

    let origin_x = 3u32;
    let origin_y = 2u32;
    let width = 5u32;
    let height = 4u32;
    let rb = (width * bpp) as usize;
    let mut dst = vec![0u8; rb * height as usize];
    let walks = store_route_count("rectrd_rect_walk");
    let windows = store_route_count("rectrd_window_padded_dst");
    assert!(read_rect_raw_at(
        &mut state,
        &mut host,
        mid,
        SurfaceWindow {
            base_off: 0,
            bpr,
            span_end: bpr as u64 * rows as u64,
            bpp,
        },
        Rect {
            origin_x,
            origin_y,
            width,
            height,
        },
        &mut dst,
        rb as u32,
    ));
    assert_eq!(
        store_route_count("rectrd_rect_walk") - walks,
        1,
        "a packed destination must resolve the page table once for the rectangle"
    );
    assert_eq!(
        store_route_count("rectrd_window_padded_dst") - windows,
        0,
        "the plane-sized window arm is for padded destinations only"
    );
    for y in 0..height as usize {
        let src_off = (origin_y as usize + y) * bpr as usize + (origin_x * bpp) as usize;
        assert_eq!(
            &dst[y * rb..(y + 1) * rb],
            &plane[src_off..src_off + rb],
            "row {y} did not land at its texel offset"
        );
    }
}

/// A rect ending past the sample window must be refused the same way and
/// named the same way whichever arm reads it. The bound used to live inside
/// the contig arm, so the fragmented arm — the one a driven x86 boot takes —
/// answered an overrun with a bare `false` from a slice index and no line
/// saying the rect had left the window.
///
/// Run over both arms from one body so the two cannot drift: a single packed
/// page takes the contig arm, two scattered pages take the fragmented one.
#[test]
fn a_rect_past_the_sample_window_is_named_on_both_read_arms() {
    use crate::model::PAGE_SHIFT_X86;

    for scattered in [false, true] {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpa0 = 0x5100_0000u64;
        let gpa1 = 0x6200_0000u64;
        host.map_range(gpa0, page as usize, 0);
        host.map_range(gpa1, page as usize, 0);
        let pfn0 = (gpa0 >> PAGE_SHIFT_X86) as u32;
        let pfn1 = (gpa1 >> PAGE_SHIFT_X86) as u32;
        let mid = 12u32;
        state.map_surface(mid);
        {
            let m = state.surfaces.mappings.get_mut(&mid).unwrap();
            m.lifecycle.active = true;
            m.lifecycle.internal_kva = 1;
            // One page is a packed view; two distant ones cannot be.
            m.pages.entries = if scattered {
                vec![
                    (pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                    (pfn1 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
                ]
            } else {
                vec![(pfn0 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID]
            };
        }
        assert_eq!(
            mapper::ensure_contig_view(&mut state, &mut host, mid).is_some(),
            !scattered,
            "scattered={scattered} must select the arm this iteration is about"
        );

        // The window is one page. Asking for four rows at a one-page pitch
        // puts the fourth row's last byte three pages past its end.
        let mut dst = [0u8; 4 * 8];
        let cap = crate::observe::FailCapture::start();
        let ok = read_rect_raw_at(
            &mut state,
            &mut host,
            mid,
            SurfaceWindow {
                base_off: 0,
                bpr: page as u32,
                span_end: page,
                bpp: 4,
            },
            Rect {
                origin_x: 0,
                origin_y: 0,
                width: 2,
                height: 4,
            },
            &mut dst,
            8,
        );
        let overruns: Vec<String> = cap
            .lines()
            .into_iter()
            .filter(|l| l.contains("reason=read_overrun"))
            .collect();
        assert!(!ok, "scattered={scattered}: the read must refuse");
        assert_eq!(
            overruns.len(),
            1,
            "scattered={scattered}: the refusal must name the bound it broke, \
             not leave the caller's decline to stand for it: {overruns:?}"
        );
    }
}

/// compute_full_tight_scratch: an exact-pitch fragmented compute plane
/// reads and writes directly through the caller's tight buffer. The
/// always-on proxy proves this class is selected on a live dispatch.
///
/// The read half is the rectangle walk and the write half still has its own
/// full-plane-tight arm, so the two proxies differ: a counter for the read, the
/// `full_tight_direct` line for the write. The read's separate special case was
/// retired because the rectangle subsumes it — a tight full plane is a
/// rectangle whose rows happen to touch, and it moves as one piece.
#[test]
fn fragmented_full_tight_rect_uses_direct_mapping_window() {
    use crate::model::PAGE_SHIFT_X86;
    use crate::runtime::drain::store_route_count;

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    let page = 1u64 << PAGE_SHIFT_X86;
    let gpa0 = 0x7100_0000u64;
    let gpa1 = 0x8200_0000u64;
    host.map_range(gpa0, page as usize, 0x31);
    host.map_range(gpa1, page as usize, 0x42);
    let mid = 29;
    assert!(state.map_surface(mid));
    {
        let m = state.surfaces.mappings.get_mut(&mid).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![
            (((gpa0 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
            (((gpa1 >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID,
        ];
    }

    let bpr = page as u32;
    let span = page * 2;
    let mut tight = vec![0u8; span as usize];
    let walks = store_route_count("rectrd_rect_walk");
    assert!(read_rect_raw_at(
        &mut state,
        &mut host,
        mid,
        SurfaceWindow {
            base_off: 0,
            bpr,
            span_end: span,
            bpp: 4
        },
        Rect {
            origin_x: 0,
            origin_y: 0,
            width: bpr / 4,
            height: 2
        },
        &mut tight,
        bpr
    ));
    assert!(tight[..page as usize].iter().all(|&v| v == 0x31));
    assert!(tight[page as usize..].iter().all(|&v| v == 0x42));
    assert_eq!(
        store_route_count("rectrd_rect_walk") - walks,
        1,
        "a tight full plane is a rectangle and must take the one-walk arm"
    );

    tight.fill(0x5a);
    assert!(write_full_rect_raw_at(
        &mut state,
        &mut host,
        mid,
        0,
        bpr,
        span,
        bpr / 4,
        2,
        4,
        &tight,
        bpr,
    ));
    let mut check = vec![0u8; span as usize];
    assert!(mapper::read_mapping_bytes(
        &mut state, &mut host, mid, 0, &mut check,
    ));
    assert_eq!(check, tight);

    let log = std::fs::read_to_string(crate::observe::fail_log_path()).expect("fail log");
    assert!(log.contains(&format!(
        "OFF mapping_write full_tight_direct mid={mid} bytes={span}"
    )));
}

/// A published descriptor that resolves no plane is not the same state as no
/// descriptor at all, and the device must not answer them alike.
///
/// Both used to return the packed window over offset 0. For the second that
/// is the only layout information anyone has; for the first the guest has
/// already said where its planes are and this texture matched two of them —
/// a v0a8 surface's Y and alpha planes share format and geometry, so the
/// scan cannot separate them and plane 0's bytes would be bound for a sample
/// the wire meant for alpha, silently.
#[test]
fn an_ambiguous_descriptor_declines_where_an_absent_one_still_sizes_a_window() {
    use reims_vgpu_core::endian::{st16, st32, st64};
    use reims_vgpu_core::pixel_format::MTL_FORMAT_R8_UNORM;
    use reims_vgpu_protocol::{
        DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_PLANES, DEVICE_DESC_PLANE_COUNT, DEVICE_PLANE_BPE,
        DEVICE_PLANE_BPR, DEVICE_PLANE_DESC_LEN, DEVICE_PLANE_DIMS, DEVICE_PLANE_OFFSET,
        DEVICE_PLANE_SIZE,
    };

    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.map_surface(8);

    // No descriptor yet: geometry came from the IOSurface texture object and
    // the aligned row stands in for the pitch. 4 R8 texels align to 128.
    let m = state.surfaces.mappings.get(&8).expect("mapping");
    assert_eq!(
        iosurface_texture_sample_window(m, 4, 2, MTL_FORMAT_R8_UNORM),
        Some((0, 128, 256)),
        "with nothing published there are no planes to confuse"
    );

    // Publish a v0a8-shaped descriptor: planes 0 and 2 are both R8 4x2.
    let mut desc = vec![0u8; reims_vgpu_protocol::DEVICE_DESC_LEN];
    st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], 0x2000);
    desc[DEVICE_DESC_PLANE_COUNT] = 3;
    let pack = |w: u32, h: u32| ((w as u64 & 0xffffff) << 8) | ((h as u64 & 0xffffff) << 40);
    for (i, (off, w, h, bpe)) in [(512u32, 4u32, 2u32, 1u16), (1024, 2, 1, 2), (1536, 4, 2, 1)]
        .iter()
        .enumerate()
    {
        let p = DEVICE_DESC_PLANES + i * DEVICE_PLANE_DESC_LEN;
        st32(&mut desc[p + DEVICE_PLANE_OFFSET..], *off);
        st32(&mut desc[p + DEVICE_PLANE_SIZE..], 256);
        st64(&mut desc[p + DEVICE_PLANE_DIMS..], pack(*w, *h));
        st32(&mut desc[p + DEVICE_PLANE_BPR..], 64);
        st16(&mut desc[p + DEVICE_PLANE_BPE..], *bpe);
    }
    assert!(state.set_mapping_device_desc(8, &desc));

    let m = state.surfaces.mappings.get(&8).expect("mapping");
    assert_eq!(
        iosurface_texture_sample_window(m, 4, 2, MTL_FORMAT_R8_UNORM),
        None,
        "two planes match and neither is the answer, so nothing is bound"
    );
    // The wire index is the only thing that separates them, and it reaches
    // each of the two directly.
    assert_eq!(
        iosurface_plane_view_sample_window(m, 0, 4, 2, MTL_FORMAT_R8_UNORM).map(|w| w.0),
        Some(512)
    );
    assert_eq!(
        iosurface_plane_view_sample_window(m, 2, 4, 2, MTL_FORMAT_R8_UNORM).map(|w| w.0),
        Some(1536)
    );
    // An index past the plane count resolves nothing rather than falling
    // back onto plane 0's bytes.
    assert_eq!(
        iosurface_plane_view_sample_window(m, 7, 4, 2, MTL_FORMAT_R8_UNORM),
        None
    );
}

/// qemu-shim: guest page write IS the surface content (unified memory) —
/// bytes land in pages and the generation advances; nothing else exists.
#[test]
fn write_bgra8_lands_in_pages_and_bumps_gen() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x18u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(8);
    {
        let m = state.surfaces.mappings.get_mut(&8).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
    }
    assert!(state.set_mapping_geom(8, 2, 2, MTL_FORMAT_BGRA8_UNORM));
    // BGRA red pixel + zeros
    let src = [0x00u8, 0x00, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    assert!(write_bgra8(&mut state, &mut host, 8, &src, 8, 2, 2));
    let m = state.surfaces.mappings.get(&8).unwrap();
    assert_eq!(m.content.guest_page_generation, 1);
    let mut first_px = [0u8; 4];
    assert!(host.read_gpa(gpa, &mut first_px).is_ok());
    assert_eq!(&first_px, &[0x00, 0x00, 0xFF, 0xFF], "pages hold the write");
}

#[test]
fn raw_rows_roundtrip() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x11u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(4);
    let m = state.surfaces.mappings.get_mut(&4).unwrap();
    m.lifecycle.active = true;
    m.lifecycle.internal_kva = 1;
    m.pages.entries = vec![entry];
    assert!(state.set_mapping_geom(4, 2, 2, 0));
    // 2x2 depth32 floats: 1.0, 0.5 / 0.25, 0.0
    let mut src = Vec::new();
    for f in [1.0f32, 0.5, 0.25, 0.0] {
        src.extend_from_slice(&f.to_bits().to_le_bytes());
    }
    assert!(write_raw_rows(&mut state, &mut host, 4, &src, 8, 8, 2, 2));
    let gen = state
        .surfaces
        .mappings
        .get(&4)
        .unwrap()
        .content
        .guest_page_generation;
    assert!(gen >= 1);
    let mut dst = vec![0u8; 16];
    assert!(read_raw_rows(
        &mut state, &mut host, 4, &mut dst, 8, 8, 2, 2
    ));
    assert_eq!(dst, src);
    // Read does not bump generation.
    assert_eq!(
        state
            .surfaces
            .mappings
            .get(&4)
            .unwrap()
            .content
            .guest_page_generation,
        gen
    );
}

/// The read side of the same bound. A rect read whose geometry exceeds what
/// `span_end` allows must be REJECTED, not run past the contig view.
///
/// `contig_for_span` guarantees the view covers `span_end` and nothing more,
/// so an oversized `height` reads whatever is next in the QEMU process —
/// unrelated memory sampled into a texture, or a SIGSEGV that takes the VM
/// down with no guest-side trace. The write side has carried this guard for a
/// while; the read side did not, which is the asymmetry to watch for when a
/// raw-pointer fast path is added beside a checked slow path.
///
/// A correctly-sized read (read_end == span_end) still succeeds.
#[test]
fn oversized_height_rect_read_is_rejected_not_overrun() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x23u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    // A full 16 KiB page, so `contig_for_span` succeeds and the guard — not
    // the view length — is what has to stop the overrun.
    host.map_range(gpa, 0x4000, 0xCC);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(11);
    {
        let m = state.surfaces.mappings.get_mut(&11).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
    }
    // The source allows exactly 2 rows of bpr=8.
    let bpr = 8u32;
    let (width, bpp) = (2u32, 4u32); // row_bytes = 8 == bpr (dense path)
    let span_end = 16u64;

    // 100 rows: read_end = (100-1)*8 + 8 = 800 > 16.
    let mut big = vec![0u8; 100 * bpr as usize];
    let cap = crate::observe::FailCapture::start();
    assert!(
        !read_rect_raw_at(
            &mut state,
            &mut host,
            11,
            SurfaceWindow {
                base_off: 0,
                bpr,
                span_end,
                bpp
            },
            Rect {
                origin_x: 0,
                origin_y: 0,
                width,
                height: 100
            },
            &mut big,
            bpr
        ),
        "an oversized-height read must be rejected"
    );
    assert!(
        cap.one("mapping_read").contains("reason=read_overrun"),
        "the refusal must name itself"
    );
    assert!(
        big.iter().all(|&b| b == 0),
        "a rejected read must not have copied anything into the caller's buffer"
    );
    drop(cap);

    // A correctly-sized 2-row read (read_end == span_end) still succeeds.
    let mut ok = vec![0u8; 2 * bpr as usize];
    assert!(
        read_rect_raw_at(
            &mut state,
            &mut host,
            11,
            SurfaceWindow {
                base_off: 0,
                bpr,
                span_end,
                bpp
            },
            Rect {
                origin_x: 0,
                origin_y: 0,
                width,
                height: 2
            },
            &mut ok,
            bpr
        ),
        "a read whose extent equals span_end must succeed"
    );
    assert_eq!(ok, vec![0xCC; 2 * bpr as usize], "and must read the page");
}

/// A writeback whose source `height` exceeds what the destination `span_end`
/// allows must be REJECTED, not run past the contig view into adjacent guest
/// pages (the trace-less heap smash behind).
/// A correctly-sized write (write_end == span_end) still succeeds.
#[test]
fn oversized_height_writeback_is_rejected_not_overrun() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x21u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    // Map a full 16 KiB page so contig_for_span succeeds; the guard, not the
    // view length, must be what stops the overrun.
    host.map_range(gpa, 0x4000, 0xCC); // 0xCC canary fills the page
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(9);
    {
        let m = state.surfaces.mappings.get_mut(&9).unwrap();
        m.lifecycle.active = true;
        m.lifecycle.internal_kva = 1;
        m.pages.entries = vec![entry];
    }
    // Destination allows exactly 2 rows of bpr=8 (span_end = 2*8 = 16).
    let bpr = 8u32;
    let (width, bpp) = (2u32, 4u32); // row_bytes = 8 == bpr (dense path)
    let span_end = 16u64;
    // Oversized source: 100 rows. write_end = (100-1)*8 + 8 = 800 > 16.
    let big = vec![0x2a; 100 * bpr as usize];
    assert!(
        !write_full_rect_raw_at(
            &mut state, &mut host, 9, 0, bpr, span_end, width, 100, bpp, &big, bpr,
        ),
        "an oversized-height write must be rejected"
    );
    // Nothing past span_end was written — the canary survives at offset 100.
    let mut probe = [0u8; 4];
    assert!(mapper::read_mapping_bytes(
        &mut state, &mut host, 9, 100, &mut probe
    ));
    assert_eq!(
        probe, [0xCC; 4],
        "guest bytes past span_end must be untouched"
    );
    // A correctly-sized 2-row write (write_end == span_end) still succeeds.
    let ok = vec![0x2a; 2 * bpr as usize];
    assert!(
        write_full_rect_raw_at(
            &mut state, &mut host, 9, 0, bpr, span_end, width, 2, bpp, &ok, bpr,
        ),
        "a write whose extent equals span_end must succeed"
    );
}

/// Clear+partial Store: seed=None (full write) must overwrite prior guest
/// content outside the scissor — logo-mid residual when seed=clear skipped.
#[test]
fn clear_store_full_write_overwrites_prior_guest_outside_scissor() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x14u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(7);
    let m = state.surfaces.mappings.get_mut(&7).unwrap();
    m.lifecycle.active = true;
    m.lifecycle.internal_kva = 1;
    m.pages.entries = vec![entry];
    // 4x2 BGRA
    assert!(state.set_mapping_geom(7, 4, 2, MTL_FORMAT_BGRA8_UNORM));
    // Prior guest content: "logo" non-zero all pixels.
    let mut logo = vec![0u8; 4 * 2 * 4];
    for px in logo.chunks_exact_mut(4) {
        px.copy_from_slice(&[0x10, 0x20, 0x30, 0xFF]); // BGRA
    }
    assert!(write_bgra8(&mut state, &mut host, 7, &logo, 16, 4, 2));
    // Metal RT after Clear+partial toolbar: clear everywhere, one red pixel
    // at (1,0) as the drawn strip. Full Store (seed=None).
    let mut rgba = vec![0u8; 4 * 2 * 4]; // clear = zeros RGBA
    rgba[4] = 255; // R
    rgba[4 + 3] = 255; // A
    assert!(write_rgba8_image_changed(
        &mut state, &mut host, 7, &rgba, None, // Clear Store: not image_changed vs clear seed
        4, 2
    ));
    let mut row = vec![0u8; 16];
    assert!(read_rect_raw(
        &mut state,
        &mut host,
        7,
        Rect {
            origin_x: 0,
            origin_y: 0,
            width: 4,
            height: 1
        },
        &mut row,
        16
    ));
    // Outside scissor pixel 0 must be clear (not logo).
    assert_eq!(
        &row[0..4],
        &[0, 0, 0, 0],
        "Clear Store must wipe prior guest"
    );
    // Drawn pixel 1 red in BGRA.
    assert_eq!(&row[4..8], &[0, 0, 255, 255]);
    // Contrast: Load seed=logo + same rgba would leave logo where equal —
    // not tested here; store_seed_policy gates that path.
}

/// The depth/stencil writeback must name its refusals too.
///
/// `write_raw_rows` is the third guest-memory writer in this file and was
/// the last one still answering every refusal with a bare `false`. It is
/// worse placed than the others to be silent: both callers discard its
/// result outright, so nothing above it could report a reason even if it
/// wanted to, and the guest work it drops is a `MTLStoreActionStore` on a
/// depth/stencil attachment - the mapping simply keeps stale bytes and the
/// pass reports success. The colour writeback twenty lines from its caller
/// emits for the analogous condition.
#[test]
fn every_raw_rows_refusal_names_itself() {
    use crate::runtime::drain::store_route_count;
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x14u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(7);
    let m = state.surfaces.mappings.get_mut(&7).unwrap();
    m.lifecycle.active = true;
    m.lifecycle.internal_kva = 1;
    m.pages.entries = vec![entry];
    assert!(state.set_mapping_geom(7, 4, 2, MTL_FORMAT_BGRA8_UNORM));
    let rows = vec![0u8; 4 * 2 * 4];

    // A zero dimension is not a rect.
    let n = store_route_count("surface_write_geometry");
    assert!(!write_raw_rows(
        &mut state, &mut host, 7, &rows, 16, 16, 0, 2
    ));
    assert_eq!(store_route_count("surface_write_geometry"), n + 1);

    // A source pitch that cannot hold one row.
    let n = store_route_count("surface_write_source_stride");
    assert!(!write_raw_rows(
        &mut state, &mut host, 7, &rows, 4, 16, 4, 2
    ));
    assert_eq!(store_route_count("surface_write_source_stride"), n + 1);

    // The source ends before the rows it declares.
    let n = store_route_count("surface_write_source_short");
    assert!(!write_raw_rows(
        &mut state,
        &mut host,
        7,
        &rows[..8],
        16,
        16,
        4,
        2
    ));
    assert_eq!(store_route_count("surface_write_source_short"), n + 1);

    // No such mapping.
    let n = store_route_count("surface_write_mapping_absent");
    assert!(!write_raw_rows(
        &mut state, &mut host, 4242, &rows, 16, 16, 4, 2
    ));
    assert_eq!(store_route_count("surface_write_mapping_absent"), n + 1);

    // The latched geometry is not this frame's.
    let n = store_route_count("surface_write_geometry_moved");
    let big = vec![0u8; 8 * 8 * 4];
    assert!(!write_raw_rows(
        &mut state, &mut host, 7, &big, 32, 32, 8, 8
    ));
    assert_eq!(store_route_count("surface_write_geometry_moved"), n + 1);

    // Unmapped: there is nowhere to write.
    let n = store_route_count("surface_write_mapping_not_resident");
    state
        .surfaces
        .mappings
        .get_mut(&7)
        .unwrap()
        .lifecycle
        .active = false;
    assert!(!write_raw_rows(
        &mut state, &mut host, 7, &rows, 16, 16, 4, 2
    ));
    assert_eq!(
        store_route_count("surface_write_mapping_not_resident"),
        n + 1
    );
}

/// Every refusal in `write_rgba8_image_changed` must name itself.
///
/// This writeback is the guest's own copy of a rendered frame, and it is
/// reached on the live x86/Vulkan sync-store route. Every one of its
/// refusals used to be a bare `false`, so a frame the guest never received
/// left no trace at all — while the sibling writer in this same file
/// answered the identical conditions through `SurfaceWriteRefusal`. The
/// vocabulary was already complete; only this arm did not use it.
///
/// Asserting on the route slugs rather than on the fail lines is what makes
/// this a regression test: `refuse` latches its line per `(check, mapping)`
/// but always counts, so a reverted arm shows up as a counter that stops
/// moving even on a mapping that has already refused once.
#[test]
fn every_rgba8_image_changed_refusal_names_itself() {
    use crate::runtime::drain::store_route_count;
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let pfn = 0x14u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(7);
    let m = state.surfaces.mappings.get_mut(&7).unwrap();
    m.lifecycle.active = true;
    m.lifecycle.internal_kva = 1;
    m.pages.entries = vec![entry];
    assert!(state.set_mapping_geom(7, 4, 2, MTL_FORMAT_BGRA8_UNORM));
    let frame = vec![0u8; 4 * 2 * 4];

    // Each case: (slug, the call that must take that arm).
    let before = |slug: &str| store_route_count(slug);

    // A zero dimension is not a rect.
    let n = before("surface_write_geometry");
    assert!(!write_rgba8_image_changed(
        &mut state, &mut host, 7, &frame, None, 0, 2
    ));
    assert_eq!(store_route_count("surface_write_geometry"), n + 1);

    // The source ends before the frame it declares.
    let n = before("surface_write_source_short");
    assert!(!write_rgba8_image_changed(
        &mut state,
        &mut host,
        7,
        &frame[..4],
        None,
        4,
        2
    ));
    assert_eq!(store_route_count("surface_write_source_short"), n + 1);

    // The seed ends before it: a different buffer, so a different slug.
    let n = before("surface_write_seed_short");
    assert!(!write_rgba8_image_changed(
        &mut state,
        &mut host,
        7,
        &frame,
        Some(&frame[..4]),
        4,
        2
    ));
    assert_eq!(
        store_route_count("surface_write_seed_short"),
        n + 1,
        "a short seed must not be reported as a short source"
    );

    // No such mapping: the surface went away between the arm and the landing.
    let n = before("surface_write_mapping_absent");
    assert!(!write_rgba8_image_changed(
        &mut state, &mut host, 4242, &frame, None, 4, 2
    ));
    assert_eq!(store_route_count("surface_write_mapping_absent"), n + 1);

    // The latched geometry is not the frame's: landing it would skew.
    let n = before("surface_write_geometry_moved");
    let big = vec![0u8; 8 * 8 * 4];
    assert!(!write_rgba8_image_changed(
        &mut state, &mut host, 7, &big, None, 8, 8
    ));
    assert_eq!(store_route_count("surface_write_geometry_moved"), n + 1);

    // Unmapped: there is nowhere to write.
    let n = before("surface_write_mapping_not_resident");
    state
        .surfaces
        .mappings
        .get_mut(&7)
        .unwrap()
        .lifecycle
        .active = false;
    assert!(!write_rgba8_image_changed(
        &mut state, &mut host, 7, &frame, None, 4, 2
    ));
    assert_eq!(
        store_route_count("surface_write_mapping_not_resident"),
        n + 1
    );
}

#[test]
fn rgba8_image_changed_writes_only_diff_spans() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    // 4x2 BGRA: invent bpr 128 → one page.
    let pfn = 0x13u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(6);
    let m = state.surfaces.mappings.get_mut(&6).unwrap();
    m.lifecycle.active = true;
    m.lifecycle.internal_kva = 1;
    m.pages.entries = vec![entry];
    assert!(state.set_mapping_geom(6, 4, 2, MTL_FORMAT_BGRA8_UNORM));
    // Seed: all zeros.
    let seed = vec![0u8; 4 * 2 * 4];
    // Image: one red pixel at (1,0), rest zero.
    let mut rgba = seed.clone();
    rgba[4] = 255; // R
    rgba[4 + 3] = 255; // A
    assert!(write_rgba8_image_changed(
        &mut state,
        &mut host,
        6,
        &rgba,
        Some(&seed),
        4,
        2
    ));
    // Read back first row of mapping (BGRA native).
    let mut row = vec![0u8; 16];
    assert!(read_rect_raw(
        &mut state,
        &mut host,
        6,
        Rect {
            origin_x: 0,
            origin_y: 0,
            width: 4,
            height: 1
        },
        &mut row,
        16
    ));
    // Pixel 1 is red in BGRA: B=0 G=0 R=255 A=255
    assert_eq!(&row[4..8], &[0, 0, 255, 255]);
    assert_eq!(&row[0..4], &[0, 0, 0, 0]);
}

#[test]
fn rect_raw_roundtrip_subregion() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    // 4x2 BGRA needs 4*4=16 tight, aligned bpr = 128 (ROW_BYTES_ALIGN).
    // One page is enough for 2 rows of 128.
    let pfn = 0x12u32;
    let gpa = (pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(gpa, 0x4000, 0);
    let entry = (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
    state.map_surface(5);
    let m = state.surfaces.mappings.get_mut(&5).unwrap();
    m.lifecycle.active = true;
    m.lifecycle.internal_kva = 1;
    m.pages.entries = vec![entry];
    assert!(state.set_mapping_geom(5, 4, 2, MTL_FORMAT_BGRA8_UNORM));
    // Write a 2x1 rect at (1,1): two BGRA pixels.
    let src = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    assert!(write_rect_raw(
        &mut state,
        &mut host,
        5,
        Rect {
            origin_x: 1,
            origin_y: 1,
            width: 2,
            height: 1
        },
        &src,
        8
    ));
    let mut dst = [0u8; 8];
    assert!(read_rect_raw(
        &mut state,
        &mut host,
        5,
        Rect {
            origin_x: 1,
            origin_y: 1,
            width: 2,
            height: 1
        },
        &mut dst,
        8
    ));
    assert_eq!(dst, src);
    // OOB origin fails.
    assert!(!write_rect_raw(
        &mut state,
        &mut host,
        5,
        Rect {
            origin_x: 3,
            origin_y: 0,
            width: 2,
            height: 1
        },
        &src,
        8
    ));
}
