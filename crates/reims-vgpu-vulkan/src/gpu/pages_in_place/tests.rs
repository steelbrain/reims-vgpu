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

/// The answer of pages laid out as `planes` with every texture asked, as `ask` answers: each
/// width's textures of every one of [`usages`], a width any of whose asks the runtime refuses no
/// width of the plane's ([`of_width`], [`of_plane`], [`of_surface`]). What [`pages_over`] answers,
/// which asks of no texture of a width after its first shadow a cache holds in place.
fn asked_whole(ask: &Ask, planes: &[PagesPlane]) -> PagesInPlace {
    let planar = planes.len() > 1;
    of_surface(planes.iter().map(|p| {
        let at = rt::IoSurfacePlane::new(p.offset, p.bytes_per_row, planar);
        of_plane(widths(p.row_bytes).filter_map(|(width, pixel_format)| {
            let textures: Result<Vec<Over>, rt::Error> = (usages().into_iter())
                .map(|usage| {
                    let descriptor =
                        plane_descriptor(pixel_format, p.row_bytes / width, p.rows, usage);
                    ask(&descriptor, at).map(over_pages)
                })
                .collect();
            textures.ok().map(of_width)
        }))
    }))
}

/// The verdict of `answer` and the plane it names: what the asks' answers are compared by, the
/// words of a shadow's reason dropped.
fn verdict(answer: &PagesInPlace) -> (std::mem::Discriminant<PagesInPlace>, Option<&str>) {
    let plane = match answer {
        PagesInPlace::No(why) => why.split(':').next(),
        _ => None,
    };
    (std::mem::discriminant(answer), plane)
}

/// A width's asks stop at its first shadow a cache holds in place, which decides the width, and
/// the texture that takes every access is asked first: the answer is the one every texture's ask
/// gives ([`asked_whole`]) for each answer the tests here give, of each usage alike and of the two
/// apart, each width alike and the narrowest refused, of a plane and of two; and a width whose
/// texture that takes every access is such a shadow is answered with one ask, as the Iris Xe's
/// renders of rows that end inside its lines are. A refusal of the texture that only reads where
/// the one that takes every access is answered is none the runtime gives ([`plane_over`])
#[test]
fn a_width_s_first_shadow_a_cache_holds_decides_it_as_every_ask_would() {
    use rt::InPlaceRefusal as Refusal;
    use rt::TextureShadowReason as Why;
    let refused = || {
        Err(rt::Error::Invalid(rt::Invalid::new(
            "host_pages_texture_placement",
            rt::Problem::HostPagesBacking,
        )))
    };
    let answers: Vec<fn() -> Placement> = vec![
        || Ok(rt::TexturePlacement::InPlace),
        || Ok(rt::TexturePlacement::NotHostPages),
        || Ok(rt::TexturePlacement::Pending),
        || Ok(rt::TexturePlacement::OtherShape),
        failed,
        || Ok(rt::TexturePlacement::Shadow(Why::Policy)),
        || {
            Ok(rt::TexturePlacement::Shadow(Why::StoreWidth {
                texel_bytes: 2,
            }))
        },
        || Ok(rt::TexturePlacement::Shadow(Why::PastLength)),
        || Ok(rt::TexturePlacement::Shadow(Why::DeviceCopy)),
        || Ok(rt::TexturePlacement::Shadow(Why::Mipmapped { levels: 2 })),
        || Ok(rt::TexturePlacement::Shadow(Why::Refused(Refusal::Image))),
        || Ok(rt::TexturePlacement::Shadow(Why::Refused(Refusal::Pitch))),
        || Ok(rt::TexturePlacement::Shadow(Why::Refused(Refusal::Rows))),
        || Ok(rt::TexturePlacement::Shadow(Why::Refused(Refusal::DmaBuf))),
    ];
    let every = usages()[0];
    assert!(every.contains(TextureUsage::RENDER_TARGET | TextureUsage::SHADER_WRITE));
    let one = [plane(0, 4096, 4000, 8)];
    let two = [plane(0, 4096, 4000, 8), plane(32768, 2048, 2000, 8)];
    let mut compared = 0;
    // Each answer of the texture that takes every access, a refusal too, beside each of the one
    // that only reads, which the runtime refuses only where it refuses the first.
    let firsts = answers.iter().map(|a| Some(*a)).chain([None]);
    for first in firsts {
        let seconds = answers
            .iter()
            .map(|a| Some(*a))
            .chain(first.is_none().then_some(None));
        for second in seconds.collect::<Vec<_>>() {
            // Each width alike, and the narrowest refused with the others alike.
            for narrow_refused in [false, true] {
                let ask = |d: &rt::TextureDescriptor, _: rt::IoSurfacePlane| -> Placement {
                    if narrow_refused && d.pixel_format == PixelFormat::R8Unorm {
                        return refused();
                    }
                    let answer = if d.usage == every { first } else { second };
                    answer.map_or_else(refused, |a| a())
                };
                for planes in [&one[..], &two[..]] {
                    let asks = std::cell::Cell::new(0);
                    let counted = |d: &rt::TextureDescriptor, at| {
                        asks.set(asks.get() + 1);
                        ask(d, at)
                    };
                    let early = pages_over(&counted, planes);
                    let early_asks = asks.replace(0);
                    let whole = asked_whole(&counted, planes);
                    let whole_asks = asks.get();
                    assert_eq!(
                        verdict(&early),
                        verdict(&whole),
                        "{:?} then {:?}, narrowest refused {narrow_refused}, {} planes: {early:?}, \
                         every ask's {whole:?}",
                        first.map(|a| a()),
                        second.map(|a| a()),
                        planes.len()
                    );
                    assert!(early_asks <= whole_asks, "{early_asks} > {whole_asks}");
                    compared += 1;
                }
            }
        }
    }
    assert_eq!(
        compared,
        (answers.len() * answers.len() + answers.len() + 1) * 4
    );
    // The Iris Xe's: a texture that only reads in place, and one that takes every access a shadow
    // the check's refusal of its renders makes, at every width of a plane whose rows allow five:
    // one ask a width, where every texture's asks are two.
    let asks = std::cell::Cell::new(0);
    let iris = |d: &rt::TextureDescriptor, _: rt::IoSurfacePlane| {
        asks.set(asks.get() + 1);
        match d.usage == every {
            true => failed(),
            false => Ok(rt::TexturePlacement::InPlace),
        }
    };
    let answer = pages_over(&iris, &one);
    assert!(
        matches!(&answer, PagesInPlace::No(why) if why.contains("Renders")),
        "{answer:?}"
    );
    assert_eq!(asks.replace(0), 5);
    let _ = asked_whole(&iris, &one);
    assert_eq!(asks.get(), 10);
}

/// The answer before an import for every texture, `answer` of each import: a fake of the
/// runtime's `rt::Device::host_pages_texture_placement`, counting its asks in `asked`.
fn before<'a>(
    answer: impl Fn(rt::HostBacking) -> Placement + 'a,
    asked: &'a std::cell::Cell<u64>,
) -> impl Fn(rt::HostBacking, &rt::TextureDescriptor, rt::IoSurfacePlane) -> Placement + 'a {
    move |backing, _, _| {
        asked.set(asked.get() + 1);
        answer(backing)
    }
}

/// A shadow the check's verdicts make, which a device cache holds in place.
fn failed() -> Placement {
    Ok(rt::TexturePlacement::Shadow(
        rt::TextureShadowReason::CheckFailed {
            access: rt::InPlaceAccess::Renders,
            row_ends: rt::InPlaceRowEnds::InLines,
        },
    ))
}

const POINTER: rt::HostBacking = rt::HostBacking::ImportedByPointer;
const DMA_BUF: rt::HostBacking = rt::HostBacking::ImportedAsDmaBuf;

/// Before any import, a surface's pages take a device cache exactly where the runtime's answer
/// before the import ([`never_over`]) is a shadow a cache holds in place for every import the
/// view may take: by their pointer and as dma-bufs where the device imports pointers (either may
/// take the view), as dma-bufs alone where it does not. An import whose answer is in place or
/// pending leaves the pages to the view's buffer; so does one where no texture of any width lies
/// in place over a cache either (the device makes no image in place of it over any memory)
#[test]
fn before_the_import_pages_take_a_cache_only_where_every_import_refuses_them() {
    let planes = [plane(0, 4096, 4000, 8)];
    let asked = std::cell::Cell::new(0);
    let both = [POINTER, DMA_BUF];
    let refused = before(|_| failed(), &asked);
    let why = never_over(&refused, 64, &both, &planes).expect("both refuse");
    assert!(
        why.contains("pages imported by their pointer, plane 0: ")
            && why.contains("pages imported as dma-bufs, plane 0: "),
        "{why}"
    );
    let pointer_in_place = before(
        |b| match b {
            POINTER => Ok(rt::TexturePlacement::InPlace),
            _ => failed(),
        },
        &asked,
    );
    assert_eq!(never_over(&pointer_in_place, 64, &both, &planes), None);
    let why = never_over(&pointer_in_place, 64, &[DMA_BUF], &planes);
    assert!(
        why.is_some_and(|why| why.starts_with("over pages imported as dma-bufs")),
        "a device whose views are dma-bufs alone"
    );
    let pending = before(
        |b| match b {
            DMA_BUF => Ok(rt::TexturePlacement::Pending),
            _ => failed(),
        },
        &asked,
    );
    assert_eq!(never_over(&pending, 64, &both, &planes), None, "pending");
    let image = before(
        |_| {
            Ok(rt::TexturePlacement::Shadow(
                rt::TextureShadowReason::Refused(rt::InPlaceRefusal::Image),
            ))
        },
        &asked,
    );
    assert_eq!(
        never_over(&image, 64, &both, &planes),
        None,
        "a shadow over a cache too"
    );
}

/// What the report's verdicts never showed, the runtime answers before the import as the buffer's
/// making would: the device's policy, a shader's writes of a width the check stores none of,
/// texels past the pages, rows that share bytes, and a pitch or dma-buf the device's format and
/// limit queries refuse. Each is a shadow a device cache holds in place, so where every import
/// the view may take answers it, the pages take a cache before any view is made
#[test]
fn shadows_the_verdicts_never_showed_take_a_cache_before_the_import() {
    use rt::TextureShadowReason as Why;
    let planes = [plane(0, 4096, 4000, 8)];
    let asked = std::cell::Cell::new(0);
    for reason in [
        Why::Policy,
        Why::StoreWidth { texel_bytes: 2 },
        Why::PastLength,
        Why::Refused(rt::InPlaceRefusal::Rows),
        Why::Refused(rt::InPlaceRefusal::Pitch),
        Why::Refused(rt::InPlaceRefusal::DmaBuf),
    ] {
        let shadow = before(|_| Ok(rt::TexturePlacement::Shadow(reason)), &asked);
        let why = never_over(&shadow, 64, &[POINTER, DMA_BUF], &planes);
        assert!(why.is_some(), "{reason:?}");
        let why = never_over(&shadow, 64, &[DMA_BUF], &planes);
        assert!(why.is_some(), "{reason:?} as dma-bufs alone");
    }
}

/// A texture the runtime answers in place before the pages' import may still be a shadow that
/// only the image or the import decides (`InPlaceRefusal::Pitch`, `Placement`, `Source`,
/// `DmaBuf`: the driver's linear image, the source's dma-bufs): the pages are left to the view's
/// buffer ([`never_over`] answers nothing), whose answer over the buffer then takes the device
/// cache at the making's refusal ([`pages_over`] of the buffer's answers)
#[test]
fn an_answer_in_place_before_the_import_leaves_the_pages_to_the_import() {
    let planes = [plane(0, 4032, 4000, 8)];
    let asked = std::cell::Cell::new(0);
    let in_place = before(|_| Ok(rt::TexturePlacement::InPlace), &asked);
    assert_eq!(
        never_over(&in_place, 64, &[POINTER, DMA_BUF], &planes),
        None
    );
    assert!(asked.get() > 0);
    use rt::InPlaceRefusal as Refusal;
    for refusal in [
        Refusal::Pitch,
        Refusal::Placement,
        Refusal::Source,
        Refusal::DmaBuf,
    ] {
        let over_buffer = |_: &rt::TextureDescriptor, _| {
            Ok(rt::TexturePlacement::Shadow(
                rt::TextureShadowReason::Refused(refusal),
            ))
        };
        let answer = pages_over(&over_buffer, &planes);
        assert!(
            matches!(&answer, PagesInPlace::No(why) if why.starts_with("plane 0: ")),
            "{refusal:?}: {answer:?}"
        );
    }
}

/// An ask the runtime refuses before the import for every texture (a backing it answers nothing
/// of, `Problem::HostPagesBacking`; a length no buffer is made of, `Problem::BufferLength`)
/// leaves the pages to the import: no plane has a width. A refusal of one width alone (a size
/// past the device's limits) makes that width no width of the plane's, the others deciding
#[test]
fn a_refusal_of_every_ask_before_the_import_leaves_the_pages_to_the_import() {
    let planes = [plane(0, 4096, 4000, 8)];
    let asked = std::cell::Cell::new(0);
    let backing = || {
        Err(rt::Error::Invalid(rt::Invalid::new(
            "host_pages_texture_placement",
            rt::Problem::HostPagesBacking,
        )))
    };
    let refused = before(|_| backing(), &asked);
    assert_eq!(never_over(&refused, 64, &[POINTER, DMA_BUF], &planes), None);
    let length = || {
        Err(rt::Error::Nil(rt::Nil::new(
            "new_buffer_with_bytes_no_copy",
            rt::Problem::BufferLength { length: 0 },
        )))
    };
    let refused = before(|_| length(), &asked);
    assert_eq!(never_over(&refused, 64, &[DMA_BUF], &planes), None);
    let narrow_refused = |_: rt::HostBacking, d: &rt::TextureDescriptor, _| match d.pixel_format {
        PixelFormat::R8Unorm => backing(),
        _ => failed(),
    };
    assert!(
        never_over(&narrow_refused, 64, &[POINTER, DMA_BUF], &planes).is_some(),
        "the other widths refuse"
    );
}

/// The runtime is told a view starts at [`VIEW_START`]: where the device's lines divide a page,
/// every page's first byte lies in them as that does, so the answer is the view's; where they do
/// not, the pages are left to the import with nothing asked. Lines of no bytes lie alike from
/// any byte
#[test]
fn lines_that_do_not_divide_a_page_leave_the_pages_to_the_import() {
    let planes = [plane(0, 4096, 4000, 8)];
    let asked = std::cell::Cell::new(0);
    let refused = before(|_| failed(), &asked);
    assert_eq!(never_over(&refused, 96, &[POINTER, DMA_BUF], &planes), None);
    assert_eq!(asked.get(), 0, "nothing asked");
    assert!(never_over(&refused, 0, &[POINTER, DMA_BUF], &planes).is_some());
    assert!(never_over(&refused, 4096, &[POINTER, DMA_BUF], &planes).is_some());
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

/// On a device whose views are dma-bufs, a surface's pages are answered as lying in place over
/// no import before its view where the runtime's answers before an import as dma-bufs refuse
/// them, or where a plane's pages, counted from the runs the view would map, are no one dma-buf
/// (here pages no file is behind, which the view's import refuses as well, asked of nothing); and
/// left to the import where the answers are in place and each plane's pages fit one
#[cfg(target_os = "linux")]
#[test]
#[allow(clippy::single_range_in_vec_init)] // pages of one run
fn a_dma_buf_device_refuses_pages_before_their_view_by_the_dma_bufs_alone() {
    use crate::core::num::{ToU64 as _, ToUsize as _};
    let page = GUEST_PAGE;
    let planes = [plane(0, 4096, 4096, 4)];
    let asked = std::cell::Cell::new(0);
    let in_place = before(|_| Ok(rt::TexturePlacement::InPlace), &asked);
    let refused = before(|_| failed(), &asked);
    let base = crate::os::allocate_aliasable(8 * page)
        .unwrap()
        .addr()
        .to_u64();
    let mapped = [base + 4 * page..base + 6 * page, base..base + 2 * page];
    assert_eq!(
        dma_bufs_never_in_place(&in_place, 64, &planes, &mapped),
        None,
        "the dma-bufs' answers are in place and the pages fit one"
    );
    let why = dma_bufs_never_in_place(&refused, 64, &planes, &mapped).unwrap();
    assert!(why.starts_with("over pages imported as dma-bufs"), "{why}");
    let private = vec![0u8; (8 * page).to_usize()];
    let at = (private.as_ptr().addr().to_u64()).next_multiple_of(page);
    let unresolved = [at..at + 4 * page];
    let asks = asked.get();
    let why = dma_bufs_never_in_place(&in_place, 64, &planes, &unresolved).unwrap();
    assert!(why.contains("plane 0's pages are no one dma-buf"), "{why}");
    assert_eq!(asked.get(), asks, "nothing asked of the runtime");
}

/// A fresh surface on a device whose views are dma-bufs needs a device cache whatever its
/// pages where a plane has more pages than one dma-buf takes pieces, each page counted a piece
/// of its own, or where the runtime's answers before an import as dma-bufs refuse its planes; a
/// plane within the pieces whose answers are in place may lie in place
#[cfg(target_os = "linux")]
#[test]
fn a_fresh_surface_needs_a_cache_where_its_pages_fit_no_dma_buf_or_its_answers_refuse() {
    let page = GUEST_PAGE;
    let asked = std::cell::Cell::new(0);
    let in_place = before(|_| Ok(rt::TexturePlacement::InPlace), &asked);
    let refused = before(|_| failed(), &asked);
    let fits = [plane(0, page, page, 4)];
    assert_eq!(fresh_dma_bufs_never_in_place(&in_place, 64, &fits), None);
    // The fewest pages, doubling from udmabuf's default list_limit, that one dma-buf of a
    // piece apiece does not take on this host.
    let pages = std::iter::successors(Some(1024u64), |n| n.checked_mul(2))
        .find(|&n| crate::os::one_dma_buf_of_pieces_fits(n, n * page).is_err())
        .unwrap();
    let over = [plane(0, page, page, pages)];
    let why = fresh_dma_bufs_never_in_place(&in_place, 64, &over).unwrap();
    assert!(
        why.contains("plane 0's pages, each a piece of its own"),
        "{why}"
    );
    let why = fresh_dma_bufs_never_in_place(&refused, 64, &fits).unwrap();
    assert!(why.starts_with("over pages imported as dma-bufs"), "{why}");
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
        let over_buffer = |descriptor: &rt::TextureDescriptor, at| {
            (buffer.raw).texture_placement_with_iosurface_plane(descriptor, at)
        };
        let whole = asked_whole(&over_buffer, std::slice::from_ref(&p));
        assert_eq!(
            verdict(&answer),
            verdict(&whole),
            "{p:?}: every ask's {whole:?}"
        );
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

/// Before any buffer of host pages is made, the runtime's answer for each texture the answer
/// asks (`rt::Device::host_pages_texture_placement`, given the import the buffer then takes) is
/// the buffer's wherever it is not in place, and where it is in place the buffer's is in place or
/// a refusal only the image or the import decides; it makes no image, and a view's first byte
/// ([`VIEW_START`]) is answered as the buffer's own page-aligned start. So a surface's pages take
/// a device cache before their import ([`never_over`]) only where the answer over the buffer
/// ([`surface_pages_in_place`]) takes one, and leave the rest to it. A backing the runtime answers
/// nothing of before an import is refused (`Problem::HostPagesBacking`) and leaves the pages to
/// the import
#[test]
fn the_answer_before_the_import_is_the_buffers_where_it_decides() {
    use crate::core::num::ToU64 as _;
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
    let start = at.addr().to_u64();
    let line = d.raw.in_place_check_line();
    let backing = match buffer.raw.host_backing() {
        Some(backing @ (POINTER | DMA_BUF)) => backing,
        other => {
            eprintln!(
                "the pages are {other:?}, an import the runtime answers nothing of before it"
            );
            drop(buffer);
            assert!(vm.deallocate(at, LENGTH).is_success());
            return;
        }
    };
    let import_decides = |after: &rt::TexturePlacement| {
        use rt::InPlaceRefusal as R;
        matches!(
            after,
            rt::TexturePlacement::InPlace
                | rt::TexturePlacement::Shadow(rt::TextureShadowReason::Refused(
                    R::Image | R::Pitch | R::Placement | R::Source | R::DmaBuf
                ))
        )
    };
    let images = || d.raw.work_counts().object_churn.vk_images_made_this_thread;
    for p in [
        plane(0, 4096, 4096, 8),
        plane(0, 4096, 4000, 8),
        plane(0, 4096, 4032, 8),
        plane(0, 4032, 4000, 8),
        plane(4096, 3904, 3840, 8),
        plane(0, 4096, 4096, 16),
    ] {
        let plane_at = rt::IoSurfacePlane::new(p.offset, p.bytes_per_row, false);
        for (width, pixel_format) in widths(p.row_bytes) {
            for usage in usages() {
                let descriptor = plane_descriptor(pixel_format, p.row_bytes / width, p.rows, usage);
                let made = images();
                let before = d.raw.host_pages_texture_placement(
                    backing,
                    &descriptor,
                    plane_at,
                    start,
                    LENGTH,
                );
                assert_eq!(images(), made, "{p:?} as {pixel_format:?}: no image made");
                let from_view = d.raw.host_pages_texture_placement(
                    backing,
                    &descriptor,
                    plane_at,
                    VIEW_START,
                    LENGTH,
                );
                assert_eq!(
                    from_view, before,
                    "{p:?} as {pixel_format:?}: from a view's start"
                );
                let after =
                    (buffer.raw).texture_placement_with_iosurface_plane(&descriptor, plane_at);
                eprintln!(
                    "{p:?} as {pixel_format:?} {usage:?}: before {before:?}, after {after:?}"
                );
                match (&before, &after) {
                    (Ok(rt::TexturePlacement::InPlace), Ok(after)) => {
                        assert!(
                            import_decides(after),
                            "{p:?} as {pixel_format:?}: {after:?}"
                        );
                    }
                    _ => assert_eq!(before, after, "{p:?} as {pixel_format:?} {usage:?}"),
                }
            }
        }
        let never = never_over(&before_import(&d.raw, LENGTH), line, &[backing], &[p]);
        let ask = before_import(&d.raw, LENGTH);
        let import = |descriptor: &rt::TextureDescriptor, at| ask(backing, descriptor, at);
        let whole = asked_whole(&import, &[p]);
        let early = pages_over(&import, &[p]);
        assert_eq!(
            verdict(&early),
            verdict(&whole),
            "{p:?}: every ask's {whole:?}"
        );
        let answer = surface_pages_in_place(&buffer.raw, std::slice::from_ref(&p));
        eprintln!("{p:?}: before the import {never:?}, over the buffer {answer:?}");
        if let Some(why) = never {
            assert!(
                matches!(answer, PagesInPlace::No(_)),
                "{p:?}: {why}, then {answer:?}"
            );
        }
    }
    let descriptor = plane_descriptor(PixelFormat::BGRA8Unorm, 1024, 8, usages()[0]);
    let copied = rt::HostBacking::Copied;
    let plane_at = rt::IoSurfacePlane::new(0, 4096, false);
    let refused = d
        .raw
        .host_pages_texture_placement(copied, &descriptor, plane_at, start, LENGTH);
    let backing_refused = |i: &rt::Invalid| i.problem == rt::Problem::HostPagesBacking;
    assert!(
        matches!(&refused, Err(rt::Error::Invalid(i)) if backing_refused(i)),
        "{refused:?}"
    );
    let planes = [plane(0, 4096, 4000, 8)];
    let ask = before_import(&d.raw, LENGTH);
    assert_eq!(never_over(&ask, line, &[copied], &planes), None);
    drop(buffer);
    assert!(vm.deallocate(at, LENGTH).is_success());
}
