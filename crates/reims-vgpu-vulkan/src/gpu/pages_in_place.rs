//! Whether textures over an IOSurface's own pages lie in place there
//! (`GpuDevice::surface_pages_in_place`), asked of a device whose GPU shares the CPU's memory: the
//! graphics lane gives such a surface a device cache only where a texture over its pages would be
//! the runtime's copy of them (a shadow) that a texture over a cache would not be
//! ([`surface_pages_in_place`]).
//!
//! The runtime makes a texture over a no-copy buffer of host pages an image in place where its
//! check of the pages' class passed for each access the texture's usage takes, for texels that
//! lie in the device's lines of memory as the texture's do, and the device makes the image at the
//! buffer's pitch and place; elsewhere a shadow, which each command buffer copies in before its
//! first use and merges back after a write (the runtime's `in_place.rs`). A device cache is the
//! device's own memory, laid out at the GPU's pitch, which the graphics lane copies only at the
//! CPU's locks and unlocks: it holds in place a texture whose shadow over the pages stands for a
//! check that failed or went unchecked for the pages' class, rows at a pitch or a place no image
//! over the pages takes, or texels past the pages. No cache holds one the device makes no image
//! in place of over any memory (its format and usage, a shape no image in place takes): such a
//! texture is a shadow over a cache too ([`Over::Neither`]).
//!
//! Over a buffer of the pages the answer is the runtime's own, plane by plane
//! (`rt::Buffer::texture_placement_with_iosurface_plane`, the making's own decision, which counts
//! and announces nothing): a texture over the plane's rows that only reads, and one that takes
//! every access (read, render, a shader's write). A surface's layout gives no texel width, so each
//! plane is asked as each of Metal's widths that its rows' texels divide into ([`WIDTHS`]), as a
//! texture of that width over it may be made (Metal makes a texture over a plane only of the
//! plane's bytes per element). A plane needs a cache where no width's textures all lie in place
//! over the pages, none is pending, and some width's is a shadow a cache holds in place: a width
//! the rows allow whose textures lie in place may be the plane's own, and in place comes first.
//! The verdicts of narrow and wide texels agree on every device measured (2026-11-12, the
//! check's records on the Iris Xe of homelab-ms01-1 and the Radeon 780M of homelab-um870-1 and
//! homelab-um790-xtx-1: `NarrowWrites` and `Writes` pass and fail alike for every class and row
//! end of host pages), so no width decides a plane apart from the others there.
//!
//! Before any buffer of the pages is made ([`surface_pages_never_in_place`],
//! [`fresh_surface_never_in_place`]) the answer is the runtime's own for pages not imported yet
//! (`rt::Device::host_pages_texture_placement`, given how the import would take them, by their
//! pointer or as dma-bufs, and the buffer's length), asked of each plane, width and usage as a
//! buffer's is ([`pages_over`]). It makes no buffer, import or image, and is the buffer's answer
//! wherever the steps before an import or an image decide it: the device's policy, the check's
//! verdicts for the pages' class, the texture's accesses and how its texels lie in the device's
//! lines, a shader's writes of a width the check stores none of, texels past the pages, rows that
//! share bytes, and what the device's format and limit queries refuse. Where only the image or
//! the import decides (the driver's linear image's pitch and place, the import of the source's
//! dma-bufs) it answers in place, and so do those paths: the pages are then asked again over the
//! buffer of their view ([`surface_pages_in_place`]), which takes a cache where the making
//! refuses. They answer no only where every import the pages may take would be answered no, so a
//! view made would be let go at once.
//!
//! **Measured** 2026-11-13, macOS 26.7.2 guests walking the desktop's apps: on the Radeon 780M
//! (RADV) and over MoltenVK (the M3 Max) a plane at a pitch the driver's linear image does not
//! take (3904 and 4032 bytes, CoreAnimation's multiples of 64) is in place before the import and
//! `Refused(Pitch)` over the buffer, which only the image decides (job m3-max-25006,
//! homelab-um790-xtx-1): the 72 such asks of a walk on homelab-um870-1 (m3-max-24961) still made
//! their view and took their cache over it, and every fresh block deferred there was deferred for
//! its pieces alone. On the Iris Xe of homelab-ms01-1 (ANV) the check refuses renders of rows
//! that end inside its lines for both imports, and the answer before the import deferred 1,615 of
//! 2,349 fresh blocks and answered 12 asks a cache before any view (m3-max-25031), as the report's
//! verdicts had (1,658 of 2,420, and 12: m3-max-24973).

#[cfg(target_os = "linux")]
use crate::core::num::ToU64 as _;
use crate::core::platform::values::{PagesInPlace, PagesPlane};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime as rt;

use crate::answer::plain;

/// The texel widths a texture over a plane may have, with the format each is asked as: Metal's
/// for a plane's elements (`MTLPixelFormat`'s uncompressed sizes), the most common first.
const WIDTHS: [(u64, PixelFormat); 5] = [
    (4, PixelFormat::BGRA8Unorm),
    (8, PixelFormat::RGBA16Float),
    (16, PixelFormat::RGBA32Float),
    (2, PixelFormat::RG8Unorm),
    (1, PixelFormat::R8Unorm),
];

/// The widths of [`WIDTHS`] a plane whose rows hold `row_bytes` bytes of texels may have: those
/// that divide them.
fn widths(row_bytes: u64) -> impl Iterator<Item = (u64, PixelFormat)> {
    (WIDTHS.into_iter()).filter(move |&(width, _)| row_bytes.is_multiple_of(width))
}

/// What a texture over a surface's pages is, as a device cache would change it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Over {
    /// An image in place in the pages.
    InPlace,
    /// A shadow while the device's check has not answered, which moves in place where its
    /// verdicts pass.
    Pending,
    /// A shadow over the pages that a device cache holds in place: why, in words.
    Cache(String),
    /// A shadow over any memory, a cache's too: the device makes no image in place of it.
    Neither,
}

/// What the runtime's `placement` of a texture over a surface's pages is, as a device cache would
/// change it (module docs).
fn over_pages(placement: rt::TexturePlacement) -> Over {
    use rt::TextureShadowReason as Why;
    let reason = match placement {
        rt::TexturePlacement::InPlace | rt::TexturePlacement::NotHostPages => return Over::InPlace,
        rt::TexturePlacement::Pending => return Over::Pending,
        rt::TexturePlacement::Shadow(reason) => reason,
        // No image in place takes a texture of its shape over any buffer; and an answer of the
        // runtime's this does not know says nothing of the pages.
        _ => return Over::Neither,
    };
    let words = match reason {
        Why::Refused(rt::InPlaceRefusal::Image) | Why::Mipmapped { .. } => return Over::Neither,
        Why::Policy => "the device's policy shadows every texture over host pages".to_owned(),
        Why::StoreWidth { texel_bytes } => {
            format!("the device's check stores no texels of {texel_bytes} bytes a shader writes")
        }
        Why::CheckFailed { access, row_ends } => {
            format!("the device's check failed for {access:?} of texels {row_ends:?}")
        }
        Why::Unchecked { access, row_ends } => {
            format!("the device's check made no image for {access:?} of texels {row_ends:?}")
        }
        Why::Refused(refusal) => {
            format!("the device makes no image in place over them: {refusal:?}")
        }
        Why::PastLength => "its texels run past the pages".to_owned(),
        Why::BytesBetweenRows => "bytes the buffer keeps lie between its rows".to_owned(),
        Why::DeviceCopy => "it is the device's copy of the pages".to_owned(),
        other => format!("a shadow: {other:?}"),
    };
    Over::Cache(words)
}

/// What the textures of one width over a plane are, from what each of them is: a shadow a cache
/// holds in place where one is; else pending where one is; else in place where one is (one the
/// device makes in place of no memory, [`Over::Neither`], stands aside); else none in place.
fn of_width(textures: impl IntoIterator<Item = Over>) -> Over {
    let mut answer = Over::Neither;
    for over in textures {
        answer = match (answer, over) {
            (cache @ Over::Cache(_), _) | (_, cache @ Over::Cache(_)) => cache,
            (Over::Pending, _) | (_, Over::Pending) => Over::Pending,
            (Over::InPlace, _) | (_, Over::InPlace) => Over::InPlace,
            (Over::Neither, Over::Neither) => Over::Neither,
        };
    }
    answer
}

/// What a plane is, from what the textures of each width it may have are ([`of_width`]): in
/// place where one width's are; else pending where one width's are; else a shadow a cache holds
/// in place where one width's are; else none the device makes in place.
fn of_plane(widths: impl IntoIterator<Item = Over>) -> Over {
    let mut answer = Over::Neither;
    for over in widths {
        answer = match (answer, over) {
            (Over::InPlace, _) | (_, Over::InPlace) => Over::InPlace,
            (Over::Pending, _) | (_, Over::Pending) => Over::Pending,
            (cache @ Over::Cache(_), _) | (_, cache @ Over::Cache(_)) => cache,
            (Over::Neither, Over::Neither) => Over::Neither,
        };
    }
    answer
}

/// The surface's answer from what each plane is ([`of_plane`]): a cache, naming the first plane
/// whose textures a cache holds in place; else pending where a plane's is; else in place.
fn of_surface(planes: impl IntoIterator<Item = Over>) -> PagesInPlace {
    let mut pending = false;
    for (i, over) in planes.into_iter().enumerate() {
        match over {
            Over::Cache(why) => return PagesInPlace::No(format!("plane {i}: {why}")),
            Over::Pending => pending = true,
            Over::InPlace | Over::Neither => {}
        }
    }
    match pending {
        true => PagesInPlace::Pending,
        false => PagesInPlace::Yes,
    }
}

/// The usages a plane's textures are asked of: one that only reads, and one that takes every
/// access.
fn usages() -> [TextureUsage; 2] {
    let every =
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
    [TextureUsage::SHADER_READ, every]
}

/// The descriptor of a 2D texture of `pixel_format` over a plane's `rows` rows of `width` texels,
/// of `usage`.
fn plane_descriptor(
    pixel_format: PixelFormat,
    width: u64,
    rows: u64,
    usage: TextureUsage,
) -> rt::TextureDescriptor {
    let mut descriptor = rt::TextureDescriptor::default();
    descriptor.pixel_format = pixel_format;
    descriptor.width = width;
    descriptor.height = rows;
    descriptor.usage = usage;
    descriptor
}

/// The runtime's answer of where a texture lies, or its refusal of the texture's arguments.
type Placement = Result<rt::TexturePlacement, rt::Error>;

/// Where the runtime answers that a texture of a descriptor over a plane of the pages lies: over
/// a buffer of them (`rt::Buffer::texture_placement_with_iosurface_plane`), or before any is made
/// (`rt::Device::host_pages_texture_placement`).
type Ask<'a> = dyn Fn(&rt::TextureDescriptor, rt::IoSurfacePlane) -> Placement + 'a;

/// [`Ask`] before any buffer of the pages is made, for each way their import may take them.
type AskBefore<'a> =
    dyn Fn(rt::HostBacking, &rt::TextureDescriptor, rt::IoSurfacePlane) -> Placement + 'a;

/// What the textures over `plane` are (module docs), as `ask` answers: for each width the plane's
/// rows allow, the runtime's placement of a texture of each of [`usages`], `planar` where the
/// surface has planes. A width whose textures the runtime refuses to make (a size past its limits)
/// is no width of the plane's.
fn plane_over(ask: &Ask, plane: &PagesPlane, planar: bool) -> Over {
    let at = rt::IoSurfacePlane::new(plane.offset, plane.bytes_per_row, planar);
    let width = |(width, pixel_format): (u64, PixelFormat)| {
        let placements: Result<Vec<Over>, rt::Error> = (usages().into_iter())
            .map(|usage| {
                let descriptor =
                    plane_descriptor(pixel_format, plane.row_bytes / width, plane.rows, usage);
                ask(&descriptor, at).map(over_pages)
            })
            .collect();
        placements.ok().map(of_width)
    };
    let mut answers = Vec::new();
    for answer in widths(plane.row_bytes).filter_map(width) {
        if answer == Over::InPlace {
            return Over::InPlace;
        }
        answers.push(answer);
    }
    of_plane(answers)
}

/// Whether textures over pages laid out as `planes` lie in place, as `ask` answers for each
/// plane ([`plane_over`], [`of_surface`]). Where `ask` refuses every texture (pages it answers
/// nothing of), each plane has no width, and the answer is yes.
fn pages_over(ask: &Ask, planes: &[PagesPlane]) -> PagesInPlace {
    let planar = planes.len() > 1;
    of_surface(planes.iter().map(|plane| plane_over(ask, plane, planar)))
}

/// Whether textures over the pages `pages` lies over, laid out as `planes` gives, lie in place
/// on `raw` for every access, or would lie in place over no device cache either (module docs):
/// the runtime's answer for each plane.
pub(crate) fn surface_pages_in_place(pages: &rt::Buffer, planes: &[PagesPlane]) -> PagesInPlace {
    let backing = plain(|| pages.host_backing());
    match backing {
        Some(
            rt::HostBacking::ImportedByPointer
            | rt::HostBacking::InHostRam
            | rt::HostBacking::ImportedAsDmaBuf,
        ) => {}
        Some(rt::HostBacking::Copied) => {
            return PagesInPlace::No(
                "its pages are a copy in the device's own memory, which each command buffer \
                 compares"
                    .to_owned(),
            );
        }
        // No host pages under the buffer: nothing of them to place.
        _ => return PagesInPlace::Yes,
    }
    #[cfg(target_os = "linux")]
    if backing == Some(rt::HostBacking::ImportedAsDmaBuf)
        && let Some(start) = plain(|| pages.contents()).map(|at| at.as_ptr().addr().to_u64())
    {
        for (i, plane) in planes.iter().enumerate() {
            let Some(end) = plane.end() else { continue };
            let first = start.saturating_add(plane.offset);
            if let Err(refusal) = crate::os::one_dma_buf_fits(first, end - plane.offset) {
                return PagesInPlace::No(format!(
                    "plane {i}'s pages are no one dma-buf, which a texture over {} binds: \
                     {refusal}",
                    backing_words(rt::HostBacking::ImportedAsDmaBuf)
                ));
            }
        }
    }
    pages_over(
        &|descriptor, at| plain(|| pages.texture_placement_with_iosurface_plane(descriptor, at)),
        planes,
    )
}

/// The pages `backing` names, in words.
fn backing_words(backing: rt::HostBacking) -> &'static str {
    match backing {
        rt::HostBacking::ImportedAsDmaBuf => "pages imported as dma-bufs",
        _ => "pages imported by their pointer",
    }
}

/// The bytes of a page of the guest's, from whose first byte a surface's pages, and the view
/// that maps them, start.
const GUEST_PAGE: u64 = 4096;

/// The host address the runtime is told a view of a surface's pages starts at, before the view
/// is made: a view starts on a page's first byte, and every such byte lies alike in the device's
/// lines where they divide a page ([`never_over`]), as this one does.
const VIEW_START: u64 = 0;

/// The imports a view of a surface's pages on `raw` may take: as dma-bufs alone on a device that
/// imports no shared memory by its pointer (`rt::Device::imports_shared_host_pointers` false:
/// RADV), whose view is a shared mapping of the guest's pages; else by its pointer, or as
/// dma-bufs where the device refuses the pointer (`rt::HostBacking::ImportedAsDmaBuf`).
fn view_imports(raw: &rt::Device) -> &'static [rt::HostBacking] {
    match plain(|| raw.imports_shared_host_pointers()) {
        Ok(false) => &[rt::HostBacking::ImportedAsDmaBuf],
        _ => &[
            rt::HostBacking::ImportedByPointer,
            rt::HostBacking::ImportedAsDmaBuf,
        ],
    }
}

/// The runtime's answer on `raw` before any buffer of a surface's pages is made, a buffer of
/// `length` bytes from a view's first byte ([`VIEW_START`]).
fn before_import(
    raw: &rt::Device,
    length: u64,
) -> impl Fn(rt::HostBacking, &rt::TextureDescriptor, rt::IoSurfacePlane) -> Placement + '_ {
    move |backing, descriptor, plane| {
        plain(|| raw.host_pages_texture_placement(backing, descriptor, plane, VIEW_START, length))
    }
}

/// Why textures over a surface's pages, laid out as `planes` from a page's first byte, would lie
/// in place over no import of them as any of `imports`, as `ask` answers before the import
/// ([`pages_over`] for each): for each import, the first plane whose textures a cache holds in
/// place. `None` where some import's answer is in place or pending, which the buffer then made
/// answers ([`surface_pages_in_place`]); and where the device's lines of `line` bytes do not
/// divide a page, so a view's first byte may lie in them otherwise than [`VIEW_START`] does.
fn never_over(
    ask: &AskBefore,
    line: u64,
    imports: &[rt::HostBacking],
    planes: &[PagesPlane],
) -> Option<String> {
    if line != 0 && !GUEST_PAGE.is_multiple_of(line) {
        return None;
    }
    let mut whys = Vec::new();
    for &backing in imports {
        match pages_over(&|descriptor, at| ask(backing, descriptor, at), planes) {
            PagesInPlace::No(why) => whys.push(format!("over {}, {why}", backing_words(backing))),
            PagesInPlace::Pending | PagesInPlace::Yes => return None,
        }
    }
    (!whys.is_empty()).then(|| whys.join("; "))
}

/// Why textures over an IOSurface's own pages, laid out as `planes` from a page's first byte,
/// would lie in place over no import of them `raw` makes, asked before any is made (module docs):
/// the runtime's answers before the import ([`before_import`]) refuse some plane for every import
/// a view of them may take ([`view_imports`], [`never_over`]), a buffer over `pages`, the view's
/// pages as runs of host addresses, in all. So [`surface_pages_in_place`] answers no over any
/// buffer of them. `None` where some import's answers are in place or pending: the pages' import
/// decides.
///
/// On a device whose views are dma-bufs alone ([`dma_bufs_never_in_place`]) a plane's pages that
/// one dma-buf does not take, counted from `pages` before the view is made, are refused as
/// [`surface_pages_in_place`] refuses them over the view.
pub(crate) fn surface_pages_never_in_place(
    raw: &rt::Device,
    planes: &[PagesPlane],
    pages: &[std::ops::Range<u64>],
) -> Option<String> {
    let length = (pages.iter()).fold(0u64, |sum, run| {
        sum.saturating_add(run.end.saturating_sub(run.start))
    });
    let line = plain(|| raw.in_place_check_line());
    let ask = before_import(raw, length);
    match view_imports(raw) {
        &[rt::HostBacking::ImportedAsDmaBuf] => dma_bufs_never_in_place(&ask, line, planes, pages),
        imports => never_over(&ask, line, imports, planes),
    }
}

/// [`surface_pages_never_in_place`] on a device whose views are dma-bufs, `ask` and `line` its
/// answers: a plane's pages, cut from `pages`, more than one dma-buf takes, as
/// [`surface_pages_in_place`] refuses them over the view; else the dma-bufs' answers
/// ([`never_over`]).
fn dma_bufs_never_in_place(
    ask: &AskBefore,
    line: u64,
    planes: &[PagesPlane],
    pages: &[std::ops::Range<u64>],
) -> Option<String> {
    #[cfg(target_os = "linux")]
    for (i, plane) in planes.iter().enumerate() {
        let Some(end) = plane.end() else { continue };
        let from = plane.offset - plane.offset % GUEST_PAGE;
        let of_plane = cut(pages, from..end.next_multiple_of(GUEST_PAGE));
        if let Err(refusal) = crate::os::one_dma_buf_of_fits(&of_plane) {
            return Some(format!(
                "plane {i}'s pages are no one dma-buf, which a texture over {} binds: {refusal}",
                backing_words(rt::HostBacking::ImportedAsDmaBuf)
            ));
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = pages;
    never_over(ask, line, &[rt::HostBacking::ImportedAsDmaBuf], planes)
}

/// Why textures over a fresh IOSurface's pages, `length` bytes laid out as `planes` from a page's
/// first byte, would lie in place over no import of them `raw` makes, whatever pages the surface
/// comes to have, asked before it has any: as [`surface_pages_never_in_place`] answers, each page
/// counted a piece of its own on a device whose views are dma-bufs
/// ([`fresh_dma_bufs_never_in_place`]). `None` where some import's answers are in place or
/// pending: textures over some such pages may lie in place.
pub(crate) fn fresh_surface_never_in_place(
    raw: &rt::Device,
    planes: &[PagesPlane],
    length: u64,
) -> Option<String> {
    let line = plain(|| raw.in_place_check_line());
    let ask = before_import(raw, length);
    match view_imports(raw) {
        &[rt::HostBacking::ImportedAsDmaBuf] => fresh_dma_bufs_never_in_place(&ask, line, planes),
        imports => never_over(&ask, line, imports, planes),
    }
}

/// [`fresh_surface_never_in_place`] on a device whose views are dma-bufs, `ask` and `line` its
/// answers: a plane of more pages than one dma-buf takes pieces, each page a piece of its own, as
/// a fresh surface's pages come; else the dma-bufs' answers ([`never_over`]). The guest gives a
/// fresh surface's pages as its allocator finds them, a run apiece or near it (**measured**
/// 2026-11-08 on homelab-um870-1, the 780M, an 8 GiB x86-64 macOS 26.7.2 guest scrolling Safari,
/// job m3-max-10438: the 379 WebKit tiles of 1,824 pages each came in 1,379 to 1,824 runs, 1,821
/// the median, each more than udmabuf's default `list_limit` of 1,024).
fn fresh_dma_bufs_never_in_place(
    ask: &AskBefore,
    line: u64,
    planes: &[PagesPlane],
) -> Option<String> {
    #[cfg(target_os = "linux")]
    for (i, plane) in planes.iter().enumerate() {
        let Some(end) = plane.end() else { continue };
        let from = plane.offset - plane.offset % GUEST_PAGE;
        let bytes = end.next_multiple_of(GUEST_PAGE) - from;
        if let Err(refusal) = crate::os::one_dma_buf_of_pieces_fits(bytes / GUEST_PAGE, bytes) {
            return Some(format!(
                "plane {i}'s pages, each a piece of its own as a fresh surface's come, are no \
                 one dma-buf, which a texture over {} binds: {refusal}",
                backing_words(rt::HostBacking::ImportedAsDmaBuf)
            ));
        }
    }
    never_over(ask, line, &[rt::HostBacking::ImportedAsDmaBuf], planes)
}

/// The runs of `pages`, host addresses of a surface's bytes in its order, that its bytes `span`
/// lie in, cut to them, in order.
#[cfg(target_os = "linux")]
fn cut(pages: &[std::ops::Range<u64>], span: std::ops::Range<u64>) -> Vec<std::ops::Range<u64>> {
    let mut out = Vec::new();
    let mut at = 0u64;
    for run in pages {
        let end = at.saturating_add(run.end.saturating_sub(run.start));
        let (from, to) = (span.start.max(at), span.end.min(end));
        if from < to {
            out.push(run.start + (from - at)..run.start + (to - at));
        }
        if end >= span.end {
            break;
        }
        at = end;
    }
    out
}

#[cfg(test)]
mod tests;
