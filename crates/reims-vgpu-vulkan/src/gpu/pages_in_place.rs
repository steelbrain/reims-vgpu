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
//! [`fresh_surface_never_in_place`]) there is no buffer to ask, and the verdicts are read from the
//! runtime's report (`rt::Device::in_place_check_report`), in one place ([`verdict`]), for a
//! read, a render and a shader's write. Those paths answer only where the verdicts refuse, which a
//! buffer's answer would refuse too; what only the image's making refuses (a pitch, a place,
//! texels past the pages) they leave to the answer over the buffer.

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

/// What the textures over `plane` of `pages` are (module docs): for each width the plane's rows
/// allow, the runtime's placement of a texture of each of [`usages`], `planar` where the surface
/// has planes. A width whose textures the runtime refuses to make (a size past its limits) is no
/// width of the plane's.
fn plane_over(pages: &rt::Buffer, plane: &PagesPlane, planar: bool) -> Over {
    let at = rt::IoSurfacePlane::new(plane.offset, plane.bytes_per_row, planar);
    let width = |(width, pixel_format): (u64, PixelFormat)| {
        let placements: Result<Vec<Over>, rt::Error> = (usages().into_iter())
            .map(|usage| {
                let descriptor =
                    plane_descriptor(pixel_format, plane.row_bytes / width, plane.rows, usage);
                plain(|| pages.texture_placement_with_iosurface_plane(&descriptor, at))
                    .map(over_pages)
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

/// The memory a texture over host pages lies in, as the runtime's check tells them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    /// Pages the device imported by their pointer.
    Pointer,
    /// Pages the device imported as dma-bufs, each texture bound to a dma-buf of its own pages.
    DmaBuf,
}

impl Class {
    /// Its name in the runtime's report.
    fn name(self) -> &'static str {
        match self {
            Class::Pointer => "Pointer",
            Class::DmaBuf => "DmaBuf",
        }
    }

    /// The class in words.
    fn words(self) -> &'static str {
        match self {
            Class::Pointer => "pages imported by their pointer",
            Class::DmaBuf => "pages imported as dma-bufs",
        }
    }
}

/// What a texture does with its image, as the runtime's check tells them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Access {
    Reads,
    Renders,
    Writes,
}

impl Access {
    const ALL: [Access; 3] = [Access::Reads, Access::Renders, Access::Writes];

    /// Its words in the runtime's report, after the class.
    fn name(self) -> &'static str {
        match self {
            Access::Reads => "",
            Access::Renders => " renders",
            Access::Writes => " writes",
        }
    }

    /// The access in words.
    fn words(self) -> &'static str {
        match self {
            Access::Reads => "a texture that is read",
            Access::Renders => "a texture that renders",
            Access::Writes => "a texture a shader writes",
        }
    }
}

/// The runtime's check's verdict for a class, access and row ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Pending,
    Passed,
    Failed,
    Unchecked,
}

/// The verdict `report` (`rt::Device::in_place_check_report`) gives textures over `class` that
/// take `access`, their texels lying on the device's lines of memory where `on_lines`: one line
/// of the report each, `"{class}{access}{, rows on lines}: {verdict}"`. One the report does not
/// hold is pending.
fn verdict(report: &[String], class: Class, access: Access, on_lines: bool) -> Verdict {
    let lines = if on_lines { ", rows on lines" } else { "" };
    let key = format!("{}{}{lines}: ", class.name(), access.name());
    let words = ["Pending", "Passed", "Failed", "Unchecked"];
    let found = (report.iter())
        .find_map(|line| line.strip_prefix(&key).filter(|rest| words.contains(rest)));
    match found {
        Some("Passed") => Verdict::Passed,
        Some("Failed") => Verdict::Failed,
        Some("Unchecked") => Verdict::Unchecked,
        _ => Verdict::Pending,
    }
}

/// Whether the texels of `plane`, in a buffer whose first byte lies at `start`, lie on the
/// device's lines of `line` bytes, as the runtime tells them apart: every line holds texels alone
/// or bytes beside them alone, so the texels start and end on a line, and so does each row where
/// bytes lie between rows.
fn on_lines(start: u64, plane: &PagesPlane, line: u64) -> bool {
    let on = |at: u64| at.is_multiple_of(line.max(1));
    let first = start.saturating_add(plane.offset);
    let Some(end) = plane.end() else {
        return false;
    };
    let rows_apart = plane.rows > 1 && plane.bytes_per_row != plane.row_bytes;
    on(first)
        && on(start.saturating_add(end))
        && (!rows_apart || (on(plane.bytes_per_row) && on(first.saturating_add(plane.row_bytes))))
}

/// Whether textures over the pages `pages` lies over, laid out as `planes` gives, lie in place
/// on `raw` for every access, or would lie in place over no device cache either (module docs):
/// the runtime's answer for each plane.
pub(crate) fn surface_pages_in_place(pages: &rt::Buffer, planes: &[PagesPlane]) -> PagesInPlace {
    let class = match plain(|| pages.host_backing()) {
        Some(rt::HostBacking::ImportedByPointer | rt::HostBacking::InHostRam) => Class::Pointer,
        Some(rt::HostBacking::ImportedAsDmaBuf) => Class::DmaBuf,
        Some(rt::HostBacking::Copied) => {
            return PagesInPlace::No(
                "its pages are a copy in the device's own memory, which each command buffer \
                 compares"
                    .to_owned(),
            );
        }
        // No host pages under the buffer: nothing of them to place.
        _ => return PagesInPlace::Yes,
    };
    #[cfg(target_os = "linux")]
    if class == Class::DmaBuf
        && let Some(start) = plain(|| pages.contents()).map(|at| at.as_ptr().addr().to_u64())
    {
        for (i, plane) in planes.iter().enumerate() {
            let Some(end) = plane.end() else { continue };
            let first = start.saturating_add(plane.offset);
            if let Err(refusal) = crate::os::one_dma_buf_fits(first, end - plane.offset) {
                return PagesInPlace::No(format!(
                    "plane {i}'s pages are no one dma-buf, which a texture over {} binds: \
                     {refusal}",
                    class.words()
                ));
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = class;
    let planar = planes.len() > 1;
    of_surface(planes.iter().map(|plane| plane_over(pages, plane, planar)))
}

/// The verdicts `report` gives textures over pages of `class` whose first byte lies at `start`,
/// laid out as `planes`, on a device whose lines of memory are `line` bytes: why they do not lie
/// in place, for the first plane and access whose verdict failed or went unchecked; else whether
/// one is pending.
fn verdicts(
    report: &[String],
    line: u64,
    start: u64,
    class: Class,
    planes: &[PagesPlane],
) -> Result<bool, String> {
    let mut pending = false;
    for (i, plane) in planes.iter().enumerate() {
        let on_lines = on_lines(start, plane, line);
        for access in Access::ALL {
            let refused = match verdict(report, class, access, on_lines) {
                Verdict::Passed => continue,
                Verdict::Pending => {
                    pending = true;
                    continue;
                }
                Verdict::Failed => "found a byte beside the texels changed by",
                Verdict::Unchecked => "made no image for",
            };
            return Err(format!(
                "plane {i}: the device's check {refused} {} over {}, for rows that {} the device's \
                 lines of {line} bytes",
                access.words(),
                class.words(),
                if on_lines {
                    "start and end on"
                } else {
                    "end inside"
                }
            ));
        }
    }
    Ok(pending)
}

/// The bytes of a page of the guest's, from whose first byte a surface's pages, and the view
/// that maps them, start.
const GUEST_PAGE: u64 = 4096;

/// Why textures over an IOSurface's own pages, laid out as `planes` from a page's first byte,
/// would lie in place over no import of them `raw` makes, asked before any is made (module docs):
/// whichever class the import takes, by their pointer or as dma-bufs, a plane's verdict for some
/// access failed or went unchecked, and a copy of the pages holds no texture in place either. So
/// [`surface_pages_in_place`] answers no over any buffer of them: it gives a buffer over pages of
/// either class the verdicts of its class, after, for dma-bufs, a check that only adds refusals,
/// and a view of the pages starts on a page's first byte, where the device's lines start too.
/// `None` where some class's verdicts all passed or are pending, or the device's lines do not
/// divide a page: the pages' import decides.
///
/// A device that imports no shared memory by its pointer (`rt::Device::imports_shared_host_pointers`
/// false: RADV) imports a view, a shared mapping of the guest's pages, as dma-bufs or not at all,
/// so its verdicts are the dma-bufs' alone ([`dma_bufs_never_in_place`]), with the check that one
/// dma-buf takes each plane's pages counted from `pages`, the view's pages, before it is made.
pub(crate) fn surface_pages_never_in_place(
    raw: &rt::Device,
    planes: &[PagesPlane],
    pages: &[std::ops::Range<u64>],
) -> Option<String> {
    let report = plain(|| raw.in_place_check_report());
    let line = plain(|| raw.in_place_check_line());
    if plain(|| raw.imports_shared_host_pointers()).is_ok_and(|takes| !takes) {
        return dma_bufs_never_in_place(&report, line, planes, pages);
    }
    never_in_place(&report, line, planes)
}

/// [`surface_pages_never_in_place`] on a device whose views are dma-bufs, from its `report` and
/// `line`: a plane's pages, cut from `pages`, more than one dma-buf takes, as
/// [`surface_pages_in_place`] refuses them over the view; else the dma-bufs' verdicts as
/// [`never_in_place`] reads a class's, where the device's lines divide a page.
fn dma_bufs_never_in_place(
    report: &[String],
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
                Class::DmaBuf.words()
            ));
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = pages;
    if line == 0 || !GUEST_PAGE.is_multiple_of(line) {
        return None;
    }
    verdicts(report, line, 0, Class::DmaBuf, planes).err()
}

/// Why textures over a fresh IOSurface's pages, laid out as `planes` from a page's first byte,
/// would lie in place over no import of them `raw` makes, whatever pages the surface comes to
/// have, asked before it has any: as [`surface_pages_never_in_place`] answers, each page counted a
/// piece of its own on a device whose views are dma-bufs ([`fresh_dma_bufs_never_in_place`]).
/// `None` where some class's verdicts all passed or are pending, or the device's lines do not
/// divide a page: textures over some such pages may lie in place.
pub(crate) fn fresh_surface_never_in_place(
    raw: &rt::Device,
    planes: &[PagesPlane],
) -> Option<String> {
    let report = plain(|| raw.in_place_check_report());
    let line = plain(|| raw.in_place_check_line());
    if plain(|| raw.imports_shared_host_pointers()).is_ok_and(|takes| !takes) {
        return fresh_dma_bufs_never_in_place(&report, line, planes);
    }
    never_in_place(&report, line, planes)
}

/// [`fresh_surface_never_in_place`] on a device whose views are dma-bufs, from its `report` and
/// `line`: a plane of more pages than one dma-buf takes pieces, each page a piece of its own, as
/// a fresh surface's pages come; else the dma-bufs' verdicts as [`dma_bufs_never_in_place`] reads
/// them. The guest gives a fresh surface's pages as its allocator finds them, a run apiece or near
/// it (**measured** 2026-11-08 on homelab-um870-1, the 780M, an 8 GiB x86-64 macOS 26.7.2 guest
/// scrolling Safari, job m3-max-10438: the 379 WebKit tiles of 1,824 pages each came in 1,379 to
/// 1,824 runs, 1,821 the median, each more than udmabuf's default `list_limit` of 1,024).
fn fresh_dma_bufs_never_in_place(
    report: &[String],
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
                Class::DmaBuf.words()
            ));
        }
    }
    if line == 0 || !GUEST_PAGE.is_multiple_of(line) {
        return None;
    }
    verdicts(report, line, 0, Class::DmaBuf, planes).err()
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

/// [`surface_pages_never_in_place`] from the device's `report` and `line`.
fn never_in_place(report: &[String], line: u64, planes: &[PagesPlane]) -> Option<String> {
    if line == 0 || !GUEST_PAGE.is_multiple_of(line) {
        return None;
    }
    let pointer = verdicts(report, line, 0, Class::Pointer, planes).err()?;
    let dma_buf = verdicts(report, line, 0, Class::DmaBuf, planes).err()?;
    Some(format!("{pointer}; {dma_buf}"))
}

#[cfg(test)]
mod tests;
