//! Whether textures over an IOSurface's own pages lie in place there
//! (`GpuDevice::surface_pages_in_place`), asked of a device whose GPU shares the CPU's memory: the
//! graphics lane gives such a surface a device cache only where they do not
//! ([`surface_pages_in_place`]).
//!
//! The runtime makes a texture over a no-copy buffer of host pages an image in place where its
//! check of the pages' class passed for each access the texture's usage takes, for texels that
//! lie in the device's lines of memory as the texture's do; elsewhere a shadow, which each command
//! buffer copies in before its first use and merges back after a write (the runtime's
//! `in_place.rs`). A surface's textures may take any access, so its pages are in place where each
//! plane's verdicts pass for all three: a read, a render and a shader's write. The class is the
//! buffer's import (`rt::HostBacking`): by pointer (ANV, MoltenVK), or as dma-bufs (RADV), where
//! the runtime binds each texture to a dma-buf of the texture's own pages, so those pages must fit
//! in one ([`crate::os::one_dma_buf_fits`]). A copy of the pages in the device's own memory (the
//! runtime's `copied-host-pages`) holds no texture in place.
//!
//! Not asked: the pitch the device lays a linear image's rows at, which a texture over a device
//! cache meets as one over the pages does; and the width of the texels a shader writes, which the
//! runtime's check covers only for 4, 8 and 16 bytes, as the surface's pixel format is not in its
//! layout: a surface of narrower texels a shader writes stays a shadow over its pages.
//!
//! The verdicts are read from the runtime's report (`rt::Device::in_place_check_report`), in one
//! place ([`verdict`]), until the runtime answers a texture's placement itself.

use crate::core::num::ToU64 as _;
use crate::core::platform::values::{PagesInPlace, PagesPlane};
use metal2vulkan_runtime as rt;

use crate::answer::plain;

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
/// on `raw` for every access (module docs).
pub(crate) fn surface_pages_in_place(
    raw: &rt::Device,
    pages: &rt::Buffer,
    planes: &[PagesPlane],
) -> PagesInPlace {
    let class = match plain(|| pages.host_backing()) {
        Some(rt::HostBacking::ImportedByPointer | rt::HostBacking::InHostRam) => Class::Pointer,
        Some(rt::HostBacking::ImportedAsDmaBuf) => Class::DmaBuf,
        Some(rt::HostBacking::Copied) => {
            return PagesInPlace::No(
                "its pages are a copy in the device's own memory, which holds no texture in place"
                    .to_owned(),
            );
        }
        // No host pages under the buffer: nothing of them to place.
        _ => return PagesInPlace::Yes,
    };
    let Some(start) = plain(|| pages.contents()).map(|at| at.as_ptr().addr().to_u64()) else {
        return PagesInPlace::Yes;
    };
    #[cfg(target_os = "linux")]
    if class == Class::DmaBuf {
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
    let report = plain(|| raw.in_place_check_report());
    let line = plain(|| raw.in_place_check_line());
    match verdicts(&report, line, start, class, planes) {
        Err(why) => PagesInPlace::No(why),
        Ok(true) => PagesInPlace::Pending,
        Ok(false) => PagesInPlace::Yes,
    }
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
pub(crate) fn surface_pages_never_in_place(
    raw: &rt::Device,
    planes: &[PagesPlane],
) -> Option<String> {
    let report = plain(|| raw.in_place_check_report());
    let line = plain(|| raw.in_place_check_line());
    never_in_place(&report, line, planes)
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
mod tests {
    use super::*;

    fn plane(offset: u64, bytes_per_row: u64, row_bytes: u64, rows: u64) -> PagesPlane {
        PagesPlane {
            offset,
            bytes_per_row,
            row_bytes,
            rows,
        }
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
        assert_eq!(v(Class::Pointer, Access::Renders, true), Verdict::Pending);
        assert_eq!(v(Class::Pointer, Access::Writes, false), Verdict::Pending);
        assert_eq!(v(Class::Pointer, Access::Reads, true), Verdict::Pending);
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

    /// Before any import, a surface's pages are answered as lying in place over none exactly
    /// where the verdicts of both classes an import of them takes refuse them: wherever a buffer
    /// over a view of them (which starts on a page's first byte) imported by their pointer, or as
    /// dma-bufs, would be answered in place or pending, the answer before the import leaves it to
    /// the import, so a surface kept in place before is kept in place still; and wherever both
    /// would be answered no, so is the answer before. Every verdict of every access of both
    /// classes, for rows that end inside the device's lines and for rows on them; and lines that
    /// do not divide a page, or none, leave it to the import whatever the verdicts
    #[test]
    fn pages_lie_in_place_over_no_import_exactly_where_both_classes_refuse_them() {
        let words = ["Pending", "Passed", "Failed", "Unchecked"];
        let classes = [Class::Pointer, Class::DmaBuf];
        let line = 64;
        let entries: Vec<(Class, Access)> = (classes.iter())
            .flat_map(|&c| Access::ALL.map(move |a| (c, a)))
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
                    let kept = (classes.iter())
                        .any(|&c| verdicts(&report, line, start, c, planes).is_ok());
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
}
