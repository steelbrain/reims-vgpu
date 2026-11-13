use super::*;

fn plane(offset: u64, bytes_per_row: u64, row_bytes: u64, rows: u64) -> PagesPlane {
    PagesPlane {
        offset,
        bytes_per_row,
        row_bytes,
        rows,
    }
}

/// The variant of `over`, its words dropped: what the answers are compared by.
fn kind(over: &Over) -> Over {
    match over {
        Over::Cache(_) => Over::Cache(String::new()),
        other => other.clone(),
    }
}

fn shadow(reason: rt::TextureShadowReason) -> Over {
    over_pages(rt::TexturePlacement::Shadow(reason))
}

/// A texture whose texels run past the pages' buffer (`PastLength`), which no image over the
/// pages reaches, is a shadow a device cache holds in place: its surface needs a cache, where the
/// report's verdicts, which say nothing of the pages' length, kept it over its pages
#[test]
fn texels_past_the_pages_take_a_device_cache() {
    let over = shadow(rt::TextureShadowReason::PastLength);
    assert!(
        matches!(&over, Over::Cache(why) if why.contains("past the pages")),
        "{over:?}"
    );
    let answer = of_surface([of_plane([of_width([Over::InPlace, over])])]);
    assert!(matches!(answer, PagesInPlace::No(_)), "{answer:?}");
}

/// A texture the runtime makes its copy in the device's memory (`DeviceCopy`, a discrete GPU's
/// tracked copy) is no image in place over the pages: a device cache holds it in place, where
/// the report's verdicts, which say nothing of it, kept it over its pages
#[test]
fn a_device_copy_takes_a_device_cache() {
    let over = shadow(rt::TextureShadowReason::DeviceCopy);
    assert!(
        matches!(&over, Over::Cache(why) if why.contains("copy")),
        "{over:?}"
    );
    let answer = of_surface([of_plane([of_width([Over::InPlace, over])])]);
    assert!(matches!(answer, PagesInPlace::No(_)), "{answer:?}");
}

/// A texture that renders into rows with bytes between them that share the device's lines of
/// memory with those bytes, whose check failed for renders of such texels (the Iris Xe's: its
/// stores write a row's last line back whole), is a shadow a device cache holds in place, laid
/// out at a pitch whose rows end on lines: its surface needs a cache, the plane named
#[test]
fn render_targets_of_gapped_rows_in_the_lines_take_a_device_cache() {
    let failed = shadow(rt::TextureShadowReason::CheckFailed {
        access: rt::InPlaceAccess::Renders,
        row_ends: rt::InPlaceRowEnds::InLines,
    });
    assert!(
        matches!(&failed, Over::Cache(why) if why.contains("Renders")),
        "{failed:?}"
    );
    let unchecked = shadow(rt::TextureShadowReason::Unchecked {
        access: rt::InPlaceAccess::Renders,
        row_ends: rt::InPlaceRowEnds::InLines,
    });
    assert!(matches!(unchecked, Over::Cache(_)), "{unchecked:?}");
    let planes = [
        of_plane([of_width([Over::InPlace, Over::InPlace])]),
        of_plane([of_width([Over::InPlace, failed])]),
    ];
    match of_surface(planes) {
        PagesInPlace::No(why) => assert!(why.starts_with("plane 1: "), "{why}"),
        other => panic!("{other:?}"),
    }
}

/// A texture a shader writes whose texels are of a width the device's check stores none of
/// (`StoreWidth`) is a shadow over the pages, and a texture over a device cache, the device's
/// own memory, takes no such check: its surface needs a cache
#[test]
fn texels_of_a_width_the_check_stores_none_of_take_a_device_cache() {
    let over = shadow(rt::TextureShadowReason::StoreWidth { texel_bytes: 3 });
    assert!(
        matches!(&over, Over::Cache(why) if why.contains("3 bytes")),
        "{over:?}"
    );
    let answer = of_surface([of_plane([of_width([Over::InPlace, over])])]);
    assert!(matches!(answer, PagesInPlace::No(_)), "{answer:?}");
}

/// Rows at a pitch no image of the device's takes (`Refused(Pitch)`: RADV lays a linear image's
/// rows at multiples of 256 bytes, CoreAnimation's at multiples of 64), at a place or of an
/// import none takes, or that share bytes, are shadows over the pages that a device cache, laid
/// out at the GPU's pitch from a placed byte, holds in place: each takes a cache
#[test]
fn rows_no_image_over_the_pages_takes_take_a_device_cache() {
    use rt::InPlaceRefusal as Refusal;
    for refusal in [
        Refusal::Pitch,
        Refusal::Placement,
        Refusal::Rows,
        Refusal::Source,
        Refusal::DmaBuf,
        Refusal::Detached,
    ] {
        let over = shadow(rt::TextureShadowReason::Refused(refusal));
        assert!(matches!(over, Over::Cache(_)), "{refusal:?}: {over:?}");
    }
    let policy = shadow(rt::TextureShadowReason::Policy);
    assert!(matches!(policy, Over::Cache(_)), "{policy:?}");
}

/// A texture the device makes no image in place of over any memory (its format and usage,
/// `Refused(Image)`; a texture of levels past the first; a shape no image in place takes) is a
/// shadow over a cache as over the pages: it leaves the surface over its pages. In place and over
/// no host pages are in place, and a pending verdict is pending
#[test]
fn textures_no_image_in_place_takes_leave_the_surface_over_its_pages() {
    let image = shadow(rt::TextureShadowReason::Refused(rt::InPlaceRefusal::Image));
    let levels = shadow(rt::TextureShadowReason::Mipmapped { levels: 2 });
    let shape = over_pages(rt::TexturePlacement::OtherShape);
    assert_eq!([&image, &levels, &shape], [&Over::Neither; 3]);
    assert_eq!(over_pages(rt::TexturePlacement::InPlace), Over::InPlace);
    assert_eq!(
        over_pages(rt::TexturePlacement::NotHostPages),
        Over::InPlace
    );
    assert_eq!(over_pages(rt::TexturePlacement::Pending), Over::Pending);
    assert_eq!(
        of_surface([of_plane([of_width([Over::InPlace, image])])]),
        PagesInPlace::Yes
    );
    assert_eq!(
        of_surface([of_plane([of_width([shape])])]),
        PagesInPlace::Yes
    );
}

/// A width's textures are a cache's where one of them is (a pending one waits, one in place
/// stands aside), else pending where one is, else in place where one is; a plane is in place
/// where some width's are, else pending where some width's are, else a cache's where some
/// width's are: a width the rows allow whose textures lie in place may be the plane's own; and
/// a surface is a cache's where some plane is, else pending where some plane is, else in place
#[test]
fn a_plane_takes_a_cache_only_where_no_width_lies_in_place() {
    let cache = || Over::Cache("c".to_owned());
    assert_eq!(of_width([Over::InPlace, cache()]), cache());
    assert_eq!(of_width([Over::Pending, cache()]), cache());
    assert_eq!(of_width([Over::InPlace, Over::Pending]), Over::Pending);
    assert_eq!(of_width([Over::Neither, Over::InPlace]), Over::InPlace);
    assert_eq!(of_width([Over::Neither, Over::Neither]), Over::Neither);
    assert_eq!(of_width([]), Over::Neither);
    assert_eq!(of_plane([cache(), Over::InPlace]), Over::InPlace);
    assert_eq!(of_plane([cache(), Over::Pending]), Over::Pending);
    assert_eq!(of_plane([Over::Neither, cache()]), cache());
    assert_eq!(of_plane([]), Over::Neither);
    assert_eq!(
        of_surface([Over::InPlace, cache()]),
        PagesInPlace::No("plane 1: c".into())
    );
    assert_eq!(
        of_surface([Over::Pending, Over::InPlace]),
        PagesInPlace::Pending
    );
    assert_eq!(
        of_surface([Over::InPlace, Over::Neither]),
        PagesInPlace::Yes
    );
}

/// The widths a plane is asked as are Metal's that divide its rows' texels, 4 bytes first
#[test]
fn a_plane_is_asked_as_each_width_its_rows_divide_into() {
    let of = |row_bytes| {
        widths(row_bytes)
            .map(|(width, _)| width)
            .collect::<Vec<_>>()
    };
    assert_eq!(of(4000), [4, 8, 16, 2, 1]);
    assert_eq!(of(4004), [4, 2, 1]);
    assert_eq!(of(1001), [1]);
    assert_eq!(of(1002), [2, 1]);
}

/// The report's lines name each class, access and row ends as the runtime writes them; one
/// it does not hold, or a note, is pending
#[test]
fn the_verdicts_are_read_from_the_runtimes_report() {
    let report: Vec<String> = [
        "Pointer: Passed",
        "DmaBuf: Unchecked",
        "Pointer renders: Failed",
        "Pointer writes, rows on lines: Passed",
        "Pointer narrow writes: Failed",
        "Pointer renders, rows on lines: Pending",
        "Pointer: the pass's store changed byte 4095",
    ]
    .map(str::to_owned)
    .into();
    let v = |class, access, on| verdict(&report, class, access, on);
    assert_eq!(v(Class::Pointer, Access::Reads, false), Verdict::Passed);
    assert_eq!(v(Class::DmaBuf, Access::Reads, false), Verdict::Unchecked);
    assert_eq!(v(Class::Pointer, Access::Renders, false), Verdict::Failed);
    assert_eq!(v(Class::Pointer, Access::Writes, true), Verdict::Passed);
    assert_eq!(
        v(Class::Pointer, Access::NarrowWrites, false),
        Verdict::Failed
    );
    assert_eq!(
        v(Class::Pointer, Access::NarrowWrites, true),
        Verdict::Pending
    );
    assert_eq!(v(Class::Pointer, Access::Renders, true), Verdict::Pending);
    assert_eq!(v(Class::Pointer, Access::Writes, false), Verdict::Pending);
    assert_eq!(v(Class::Pointer, Access::Reads, true), Verdict::Pending);
}

/// A shader's writes of a plane's texels are read in the form the runtime's check takes for
/// their width: the narrow writes' verdict for a plane whose rows only 1- or 2-byte texels
/// divide, the wide one's and the narrow one's for a plane whose rows texels of 4 bytes divide
/// too, which is refused only where both are; a texture over a plane keeps its padding, so no
/// form that lets it change is read
#[test]
fn a_shaders_writes_are_read_in_the_form_of_the_planes_widths() {
    let report = |wide: &str, narrow: &str| -> Vec<String> {
        vec![
            "Pointer: Passed".to_owned(),
            "Pointer renders: Passed".to_owned(),
            format!("Pointer writes: {wide}"),
            format!("Pointer narrow writes: {narrow}"),
            "Pointer buffer writes: Failed".to_owned(),
            "Pointer unobserved renders: Failed".to_owned(),
        ]
    };
    let line = 64;
    let refused = |report: &[String], plane: PagesPlane| {
        verdicts(report, line, 0, Class::Pointer, &[plane]).is_err()
    };
    // Rows of 1,001 bytes, 1,024 apart: texels of 1 byte alone, which end inside the lines.
    let narrow = plane(0, 1024, 1001, 4);
    assert!(refused(&report("Passed", "Failed"), narrow));
    assert!(!refused(&report("Failed", "Passed"), narrow));
    // Rows of 1,000 bytes: texels of 1, 2, 4 or 8 bytes.
    let any = plane(0, 1024, 1000, 4);
    assert!(
        !refused(&report("Failed", "Passed"), any),
        "narrow texels in place"
    );
    assert!(
        !refused(&report("Passed", "Failed"), any),
        "wide texels in place"
    );
    assert!(refused(&report("Failed", "Failed"), any));
    assert_eq!(
        verdicts(
            &report("Pending", "Failed"),
            line,
            0,
            Class::Pointer,
            &[any]
        ),
        Ok(true),
        "the wide verdict pending"
    );
}

/// Texels lie on the lines where they start and end on one and so does each row where bytes
/// lie between rows: a BGRA8 plane of 1,000 texels a row (4,000 bytes) at a 4,032-byte pitch
/// ends its rows inside a 64-byte line, one of 1,008 texels at a pitch of 4,032 bytes does
/// not, and rows that follow on with no bytes between need only the plane's ends on lines
#[test]
fn row_ends_are_told_apart_as_the_runtime_tells_them() {
    assert!(!on_lines(0, &plane(0, 4032, 4000, 600), 64));
    assert!(on_lines(0, &plane(0, 4032, 4032, 600), 64));
    assert!(on_lines(0, &plane(0, 64, 64, 3), 64));
    assert!(
        !on_lines(0, &plane(0, 60, 60, 3), 64),
        "the last row ends inside"
    );
    assert!(
        on_lines(0, &plane(0, 60, 60, 16), 64),
        "the plane's ends on lines"
    );
    assert!(
        !on_lines(16, &plane(0, 4096, 4096, 2), 64),
        "a start inside a line"
    );
    assert!(
        on_lines(0, &plane(0, 4000, 4000, 1), 1),
        "lines of one byte"
    );
}

/// Before any import, a surface's pages are answered as lying in place over no import of them
/// exactly where the verdicts of both classes an import of them takes refuse them: wherever a
/// buffer over a view of them (which starts on a page's first byte) imported by their pointer, or
/// as dma-bufs, would be answered in place or pending, the answer before the import leaves it to
/// the import, so a surface kept in place before is kept in place still; and wherever both would
/// be answered no, so is the answer before. Every verdict of every access of both classes, a
/// shader's writes of wide texels and of narrow ones, for rows that end inside the device's lines
/// and for rows on them; and lines that do not divide a page, or none, leave it to the import
/// whatever the verdicts
#[test]
fn pages_lie_in_place_over_no_import_exactly_where_both_classes_refuse_them() {
    let words = ["Pending", "Passed", "Failed", "Unchecked"];
    let classes = [Class::Pointer, Class::DmaBuf];
    let accesses = [
        Access::Reads,
        Access::Renders,
        Access::Writes,
        Access::NarrowWrites,
    ];
    let line = 64;
    let entries: Vec<(Class, Access)> = (classes.iter())
        .flat_map(|&c| accesses.map(move |a| (c, a)))
        .collect();
    for plane in [plane(0, 4032, 4000, 600), plane(64, 4032, 4032, 600)] {
        let planes = std::slice::from_ref(&plane);
        let lines = if on_lines(0, &plane, line) {
            ", rows on lines"
        } else {
            ""
        };
        for combo in 0..4usize.pow(entries.len() as u32) {
            let report: Vec<String> = (entries.iter().enumerate())
                .map(|(k, (class, access))| {
                    let word = words[combo / 4usize.pow(k as u32) % 4];
                    format!("{}{}{lines}: {word}", class.name(), access.name())
                })
                .collect();
            let never = never_in_place(&report, line, planes);
            for start in [0, 7 * GUEST_PAGE] {
                let kept =
                    (classes.iter()).any(|&c| verdicts(&report, line, start, c, planes).is_ok());
                assert_eq!(never.is_none(), kept, "{report:?}, a view at {start:#x}");
            }
        }
    }
    let refused: Vec<String> = (entries.iter())
        .map(|(class, access)| format!("{}{}: Failed", class.name(), access.name()))
        .collect();
    let planes = [plane(0, 4032, 4000, 600)];
    assert!(never_in_place(&refused, line, &planes).is_some());
    assert!(
        never_in_place(&refused, 96, &planes).is_none(),
        "lines across a page's end"
    );
    assert!(never_in_place(&refused, 0, &planes).is_none(), "no lines");
}

/// The runs a span of a surface's bytes lies in, cut to it
#[cfg(target_os = "linux")]
#[test]
#[allow(clippy::single_range_in_vec_init)] // runs of one range
fn the_runs_of_a_span_are_cut_to_it() {
    let runs = [0x1000..0x3000, 0x8000..0x9000, 0x4000..0x5000];
    assert_eq!(cut(&runs, 0x1000..0x2800), [0x2000..0x3000, 0x8000..0x8800]);
    assert_eq!(cut(&runs, 0..0x2000), [0x1000..0x3000]);
    assert_eq!(cut(&runs, 0x3000..0x4000), [0x4000..0x5000]);
    assert_eq!(cut(&runs, 0x4000..0x5000), []);
}

/// The dma-bufs' report lines of every access a plane's textures may take, for rows on lines,
/// with `verdict`.
#[cfg(target_os = "linux")]
fn dma_buf_lines(verdict: &str) -> Vec<String> {
    ["", " renders", " writes", " narrow writes"]
        .iter()
        .map(|access| format!("DmaBuf{access}, rows on lines: {verdict}"))
        .collect()
}

/// On a device whose views are dma-bufs, a surface's pages are answered as lying in place over
/// no import before its view where the dma-bufs' verdicts refuse them, whatever those by
/// pointer answer, or where a plane's pages, counted from the runs the view would map, are no
/// one dma-buf (here pages no file is behind, which the view's import refuses as well); and
/// left to the import where the dma-bufs' verdicts pass and each plane's pages fit one
#[cfg(target_os = "linux")]
#[test]
#[allow(clippy::single_range_in_vec_init)] // pages of one run
fn a_dma_buf_device_refuses_pages_before_their_view_by_the_dma_bufs_alone() {
    use crate::core::num::{ToU64 as _, ToUsize as _};
    let line = 64;
    let page = GUEST_PAGE;
    let planes = [plane(0, 4096, 4096, 4)];
    let report = |dma_buf: &str| -> Vec<String> {
        let mut lines: Vec<String> = ["", " renders", " writes", " narrow writes"]
            .iter()
            .map(|access| format!("Pointer{access}, rows on lines: Passed"))
            .collect();
        lines.extend(dma_buf_lines(dma_buf));
        lines
    };
    let base = crate::os::allocate_aliasable(8 * page)
        .unwrap()
        .addr()
        .to_u64();
    let mapped = [base + 4 * page..base + 6 * page, base..base + 2 * page];
    let passed = report("Passed");
    assert!(never_in_place(&passed, line, &planes).is_none());
    assert_eq!(
        dma_bufs_never_in_place(&passed, line, &planes, &mapped),
        None,
        "the dma-bufs' verdicts pass and the pages fit one"
    );
    let failed = report("Failed");
    assert!(
        never_in_place(&failed, line, &planes).is_none(),
        "those by pointer pass"
    );
    let why = dma_bufs_never_in_place(&failed, line, &planes, &mapped).unwrap();
    assert!(why.contains("imported as dma-bufs"), "{why}");
    let private = vec![0u8; (8 * page).to_usize()];
    let at = (private.as_ptr().addr().to_u64()).next_multiple_of(page);
    let unresolved = [at..at + 4 * page];
    let why = dma_bufs_never_in_place(&passed, line, &planes, &unresolved).unwrap();
    assert!(why.contains("plane 0's pages are no one dma-buf"), "{why}");
    assert!(
        dma_bufs_never_in_place(&passed, 0, &planes, &mapped).is_none(),
        "no lines, pages that fit"
    );
}

/// A fresh surface on a device whose views are dma-bufs needs a device cache whatever its
/// pages where a plane has more pages than one dma-buf takes pieces, each page counted a piece
/// of its own, or where the dma-bufs' verdicts refuse its rows; a plane within the pieces whose
/// verdicts pass, or a device with no lines, may lie in place
#[cfg(target_os = "linux")]
#[test]
fn a_fresh_surface_needs_a_cache_where_its_pages_fit_no_dma_buf_or_its_rows_fail() {
    let line = 64;
    let page = GUEST_PAGE;
    let passed = dma_buf_lines("Passed");
    let fits = [plane(0, page, page, 4)];
    assert_eq!(fresh_dma_bufs_never_in_place(&passed, line, &fits), None);
    assert_eq!(
        fresh_dma_bufs_never_in_place(&passed, 0, &fits),
        None,
        "no lines"
    );
    // The fewest pages, doubling from udmabuf's default list_limit, that one dma-buf of a
    // piece apiece does not take on this host.
    let pages = std::iter::successors(Some(1024u64), |n| n.checked_mul(2))
        .find(|&n| crate::os::one_dma_buf_of_pieces_fits(n, n * page).is_err())
        .unwrap();
    let over = [plane(0, page, page, pages)];
    let why = fresh_dma_bufs_never_in_place(&passed, line, &over).unwrap();
    assert!(
        why.contains("plane 0's pages, each a piece of its own"),
        "{why}"
    );
    let failed = dma_buf_lines("Failed");
    let why = fresh_dma_bufs_never_in_place(&failed, line, &fits).unwrap();
    assert!(why.contains("imported as dma-bufs"), "{why}");
}

/// Over a no-copy buffer of host pages on the device, what the answer takes of each width and
/// usage of a plane is what the runtime's making of that texture over the pages does, as its
/// counts say (in place, a shadow pending the device's check, a shadow of a plan the device
/// makes no image in place of, any other shadow), and the surface's answer is those makings'
/// ([`of_surface`]): planes of rows with no bytes between, with bytes between that end inside the
/// device's lines and on them, at a pitch off 256 bytes (CoreAnimation's multiples of 64, which
/// RADV lays no linear image's rows at), and of rows to the pages' last byte. No plane past the
/// pages is asked: the device refuses an ask whose layout lies past its block's pages
/// (reims-vgpu's `layout_planes`), and on a device that imports the pages as dma-bufs the answer
/// refuses such a plane before the runtime's placement, for the bytes past the pages that no
/// dma-buf holds.
#[test]
fn the_answer_over_the_pages_is_what_the_runtimes_making_does() {
    use crate::core::platform::HostRange;
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::os::vm::HostVm as _;
    use crate::core::raise::OnFault;

    const LENGTH: u64 = 16 * GUEST_PAGE;
    let d = crate::gpu::device::tests::dev();
    assert!(
        d.raw
            .settle_in_place_checks(std::time::Duration::from_secs(60)),
        "the device's check answered"
    );
    let vm = crate::remap::VulkanVm::new();
    let at = crate::os::allocate_aliasable(LENGTH).unwrap();
    // SAFETY: the pages stay mapped until their unmapping below, after the buffer's drop.
    let bytes = unsafe { HostRange::new(at, LENGTH) };
    let buffer = d.new_buffer_over_host_memory(OnFault::Return, bytes, 0, None);
    let buffer = buffer.unwrap().unwrap();
    let made = |plane: &PagesPlane, pixel_format, width, usage| {
        let before = d.raw.work_counts();
        let descriptor = plane_descriptor(pixel_format, plane.row_bytes / width, plane.rows, usage);
        let at = rt::IoSurfacePlane::new(plane.offset, plane.bytes_per_row, false);
        let texture = buffer.raw.new_texture_with_iosurface_plane(&descriptor, at);
        let after = d.raw.work_counts();
        let texture = texture.ok()?;
        drop(texture);
        Some(
            if after.host_textures_in_place > before.host_textures_in_place {
                Over::InPlace
            } else if after.host_texture_shadows_check_pending
                > before.host_texture_shadows_check_pending
            {
                Over::Pending
            } else if after.host_texture_shadows_refused_image
                > before.host_texture_shadows_refused_image
            {
                Over::Neither
            } else if after.shadowed_textures > before.shadowed_textures {
                Over::Cache(String::new())
            } else {
                Over::Neither
            },
        )
    };
    for p in [
        plane(0, 4096, 4096, 8),
        plane(0, 4096, 4000, 8),
        plane(0, 4096, 4032, 8),
        plane(0, 4032, 4000, 8),
        plane(0, 4096, 4096, 16),
    ] {
        let mut by_making = Vec::new();
        for (width, pixel_format) in widths(p.row_bytes) {
            let made: Option<Vec<Over>> = (usages().into_iter())
                .map(|usage| made(&p, pixel_format, width, usage))
                .collect();
            let asked: Result<Vec<Over>, rt::Error> = (usages().into_iter())
                .map(|usage| {
                    let descriptor =
                        plane_descriptor(pixel_format, p.row_bytes / width, p.rows, usage);
                    let at = rt::IoSurfacePlane::new(p.offset, p.bytes_per_row, false);
                    buffer
                        .raw
                        .texture_placement_with_iosurface_plane(&descriptor, at)
                        .map(over_pages)
                })
                .collect();
            let asked = asked.ok().map(|a| a.iter().map(kind).collect::<Vec<_>>());
            assert_eq!(asked, made, "{p:?} as {pixel_format:?}");
            by_making.extend(made.map(of_width));
        }
        let answer = surface_pages_in_place(&buffer.raw, std::slice::from_ref(&p));
        let expected = of_surface([of_plane(by_making)]);
        eprintln!("{p:?}: {answer:?}");
        assert_eq!(
            std::mem::discriminant(&answer),
            std::mem::discriminant(&expected),
            "{p:?}: {answer:?}, the makings' {expected:?}"
        );
    }
    drop(buffer);
    assert!(vm.deallocate(at, LENGTH).is_success());
}
