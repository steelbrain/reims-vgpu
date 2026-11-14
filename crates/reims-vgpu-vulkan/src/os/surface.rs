//! What this platform reads of a surface beyond core's `Surface` trait, written against it and
//! each OS's record of the surfaces it made (`recorded_mut`): a plane's layout and YCbCr facts,
//! the host memory a surface's bytes are over, whether the GPU's writes through it reach that
//! memory, and the alias its buffers lie over.

use std::sync::{Arc, Weak};

use crate::core::num::ToU64;
use crate::core::platform::os::surface::Surface;
use metal2vulkan_runtime as rt;

use super::{OsSurface, host_os};
use crate::gpu::backing::Alias;

/// Plane `plane` of `surface`, as a texture on it reads it: its bytes' offset from the surface's
/// base address, its bytes per row, and whether the surface has planes. A surface without planes
/// is its plane 0. `None` for a plane the surface lacks, of which Metal makes no texture
/// (`textures_backing_records`).
pub(crate) fn plane(surface: &OsSurface, plane: u64) -> Option<rt::IoSurfacePlane> {
    if surface.plane_count() == 0 {
        return (plane == 0).then(|| rt::IoSurfacePlane::new(0, surface.bytes_per_row(), false));
    }
    let (offset, bytes_per_row) = host_os::plane_at(surface, plane)?;
    Some(rt::IoSurfacePlane::new(offset, bytes_per_row, true))
}

/// What a texture over plane `plane` of `surface`, a surface of two planes, reads of its layout,
/// which Metal reads from it when the texture is made: its OSType, its size, and its plane 1's
/// offset from its base address and bytes per row. `None` for a surface of fewer than two planes,
/// over which the runtime then refuses the texture. A texture of a two-plane XR10 format (550,
/// 551) takes these alone, plane 1 its alpha bytes ([`crate::gpu::device`]).
pub(crate) fn plane_facts(surface: &OsSurface, plane: u64) -> Option<rt::IoSurfaceYCbCr> {
    if surface.plane_count() < 2 {
        return None;
    }
    let (chroma, bytes_per_row) = host_os::plane_at(surface, 1)?;
    Some(rt::IoSurfaceYCbCr::new(
        surface.pixel_format(),
        plane,
        [surface.width(), surface.height()],
        chroma,
        bytes_per_row,
    ))
}

/// What a texture of a two-plane YCbCr format the runtime reads
/// ([`crate::gpu::ycbcr::runtime_makes`]) over plane `plane` of `surface` reads of the surface:
/// [`plane_facts`], and its YCbCr matrix as it is now, where it names one
/// (`kIOSurfaceYCbCrMatrix`).
pub(crate) fn ycbcr_facts(surface: &OsSurface, plane: u64) -> Option<rt::IoSurfaceYCbCr> {
    let facts = plane_facts(surface, plane)?;
    Some(match host_os::ycbcr_matrix(surface) {
        Some(matrix) => facts.with_matrix(matrix),
        None => facts,
    })
}

/// The host memory a surface's bytes are over: its allocation in its own mapping (its
/// allocation size from its base address less its `kIOSurfaceOffset`), and, for one
/// `OsSurfaces::create` made over the guest's pages (`kIOSurfaceClientAddressRanges`),
/// those pages as the host maps them, where the guest's other objects over the same pages lie.
/// Each range is a first byte and an end.
///
/// The surface's base address is its offset into its allocation, and a plane's
/// (`IOSurfaceGetBaseAddressOfPlane`) its `kIOSurfacePlaneOffset` into it. A surface over one
/// run of the guest's pages at offset 0 has them as its base address; at another offset, its
/// allocation is another mapping of the pages, the base address that offset into it, and a
/// write through it shows in the pages at that offset (measured on the M3 Max, offsets 0x40,
/// 0x1000 and 0x4000).
pub(crate) fn surface_memory(surface: &OsSurface) -> Vec<std::ops::Range<u64>> {
    let base = surface.base_address().as_ptr::<u8>().addr().to_u64();
    let own = |offset: u64| {
        let first = base.saturating_sub(offset);
        first..first.saturating_add(surface.alloc_size())
    };
    recorded(surface, |made| {
        let pages = made.pages.iter().cloned();
        std::iter::once(own(made.offset)).chain(pages).collect()
    })
    .unwrap_or_else(|| vec![own(0)])
}

/// The host memory of `surface`'s bytes at `offsets`, each a first byte and an end counted from
/// the surface's base address, where [`surface_memory`] has them: from its base address, then in
/// the guest's pages it was made over, in their order, from its offset into them.
pub(crate) fn surface_bytes_memory(
    surface: &OsSurface,
    offsets: &[std::ops::Range<u64>],
) -> Vec<std::ops::Range<u64>> {
    let base = surface.base_address().as_ptr::<u8>().addr().to_u64();
    let own = offsets
        .iter()
        .map(|o| base.saturating_add(o.start)..base.saturating_add(o.end));
    let paged = recorded(surface, |made| {
        let at = |o: &std::ops::Range<u64>| {
            made.offset.saturating_add(o.start)..made.offset.saturating_add(o.end)
        };
        (offsets.iter())
            .flat_map(|o| in_runs(&made.pages, &at(o)))
            .collect::<Vec<_>>()
    });
    own.chain(paged.into_iter().flatten()).collect()
}

/// The host ranges that hold the bytes `offsets` of memory laid out over `runs` in their order
/// (each a host first byte and an end).
pub(crate) fn in_runs(
    runs: &[std::ops::Range<u64>],
    offsets: &std::ops::Range<u64>,
) -> Vec<std::ops::Range<u64>> {
    let mut held = Vec::new();
    let mut at = 0u64;
    for run in runs {
        let length = run.end - run.start;
        let (first, end) = (offsets.start.max(at), offsets.end.min(at + length));
        if first < end {
            held.push(run.start + (first - at)..run.start + (end - at));
        }
        at += length;
        if at >= offsets.end {
            break;
        }
    }
    held
}

/// The offset of `surface`'s bytes into its allocation (`kIOSurfaceOffset`), as
/// `OsSurfaces::create` made it: its base address lies that far into the allocation, a
/// page-aligned mapping of `IOSurfaceGetAllocSize` bytes (measured on the M3 Max at 0x40, 0x1000
/// and 0x4000, over the guest's pages); 0 for a surface made otherwise.
pub(crate) fn surface_offset(surface: &OsSurface) -> u64 {
    recorded(surface, |made| made.offset).unwrap_or(0)
}

/// Whether `surface` was made over the guest's pages (`OsSurfaces::create`).
pub(crate) fn over_guest_pages(surface: &OsSurface) -> bool {
    recorded(surface, |made| !made.pages.is_empty()).unwrap_or(false)
}

/// The alias of the allocation of `surface`, one at an offset into it ([`surface_offset`]), that
/// every buffer over it lies over while one lives: the live one, or a new one, kept for the next:
/// the surface's own view of the guest's pages where it has one (Linux's stand-in over several
/// ranges of them, `own_alias`), borrowed, and otherwise the one `make` makes; `None` where `make`
/// makes none. The buffers then lie in one import of the runtime's, as those over a surface at no
/// offset lie over its base address: a write through one reaches the later commands of its command
/// buffer that read another, as on Metal, and a buffer over the surface can share the range of
/// another over the same first byte (the device's `new_buffer_on_surface`).
pub(crate) fn surface_alias(
    surface: &OsSurface,
    make: impl FnOnce() -> Option<Alias>,
) -> Option<Arc<Alias>> {
    let mut make = Some(|| host_os::own_alias(surface).or_else(make));
    let kept = host_os::recorded_mut(surface, |made| {
        if let Some(alias) = made.alias.upgrade() {
            return Some(alias);
        }
        let alias = Arc::new(make.take()?()?);
        made.alias = Arc::downgrade(&alias);
        Some(alias)
    });
    match kept {
        Some(alias) => alias,
        None => make.take()?().map(Arc::new),
    }
}

/// Whether `surface`'s base address lies in the guest's pages it was made over, so that the CPU's
/// writes through it reach them: as for a surface over one page; one over several, contiguous or
/// not, has a mapping of its own (measured on the M3 Max: one of 4 contiguous pages).
pub(crate) fn base_in_pages(surface: &OsSurface) -> bool {
    let base = surface.base_address().as_ptr::<u8>().addr().to_u64();
    recorded(surface, |made| {
        made.pages.iter().any(|run| run.contains(&base))
    })
    .unwrap_or(false)
}

/// The memory `surface`'s bytes are in, as a diagnostic names it: its own, for one not made over
/// the guest's pages; the guest's pages, where its base address lies in them
/// ([`base_in_pages`]); or a mapping of its own of the guest's pages, as of several pages, or of
/// one run at an offset on macOS, which IOSurface makes, and of several ranges on Linux, the
/// stand-in's view of them.
pub(crate) fn memory_kind(surface: &OsSurface) -> &'static str {
    if !over_guest_pages(surface) {
        "its own memory"
    } else if base_in_pages(surface) {
        "the guest's pages"
    } else {
        "a mapping of its own of the guest's pages"
    }
}

/// What the OS's record of `surface` holds, read by `read`; `None` for a surface it did not
/// record.
fn recorded<T>(surface: &OsSurface, read: impl FnOnce(&MadeOver) -> T) -> Option<T> {
    host_os::recorded_mut(surface, |made| read(made))
}

/// What a surface was made over: the guest's pages, in its bytes' order, merged into runs, and
/// its bytes' offset into its allocation (`kIOSurfaceOffset`); the alias of its allocation its
/// buffers lie over while one lives, where they lie over one ([`surface_alias`]).
#[derive(Debug, Clone, Default)]
pub(crate) struct MadeOver {
    pub(crate) pages: Vec<std::ops::Range<u64>>,
    pub(crate) offset: u64,
    pub(crate) alias: Weak<Alias>,
}

/// The runs of `pages`, each a host first byte and a length, those that follow on in the host's
/// memory merged, each a first byte and an end.
pub(crate) fn page_runs(pages: impl IntoIterator<Item = (u64, u64)>) -> Vec<std::ops::Range<u64>> {
    let mut runs: Vec<std::ops::Range<u64>> = Vec::new();
    for (first, length) in pages {
        let end = first.saturating_add(length);
        match runs.last_mut() {
            Some(run) if run.end == first => run.end = end,
            _ => runs.push(first..end),
        }
    }
    runs
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::core::host::{HostPtr, PhysicalMemoryRange};
    use crate::core::platform::Handle as _;
    use crate::core::platform::os::surface::{IoSurfaces as _, SurfaceSpec};
    use crate::core::platform::os::vm::HostVm as _;

    use super::*;
    use crate::os::{OsSurfaces, OsVm, allocate_aliasable};

    /// The bytes of memory laid out over runs are found in each run they reach, in order
    #[test]
    #[expect(clippy::single_range_in_vec_init, reason = "lists of host ranges")]
    fn in_runs_finds_bytes_in_each_run_they_reach() {
        let runs = [0x10_000..0x14_000, 0x40_000..0x48_000, 0x20_000..0x24_000];
        assert_eq!(in_runs(&runs, &(0x100..0x200)), [0x10_100..0x10_200]);
        assert_eq!(
            in_runs(&runs, &(0x3f00..0x4100)),
            [0x13_f00..0x14_000, 0x40_000..0x40_100]
        );
        assert_eq!(
            in_runs(&runs, &(0x3000..0xd000)),
            [0x13_000..0x14_000, 0x40_000..0x48_000, 0x20_000..0x21_000]
        );
        assert_eq!(in_runs(&runs, &(0x10_000..0x11_000)), []);
        assert_eq!(in_runs(&[], &(0..0x100)), []);
    }

    /// A 'BGRA' surface of `width` × `height`, 4 bytes an element, rows of `width × 4`, made over
    /// fresh pages of the host's (`kIOSurfaceClientAddressRanges`) `offset` bytes into them
    /// (`kIOSurfaceOffset`), and those pages: their first byte and length, which the caller frees
    /// once the surface is gone.
    pub(crate) fn bgra_surface_over_at(
        width: u64,
        height: u64,
        offset: u64,
    ) -> (OsSurface, HostPtr, u64) {
        let page = OsVm::new().page_size();
        let length = (width * height * 4 + offset).next_multiple_of(page);
        let pages = allocate_aliasable(length).unwrap();
        let first = pages.addr().to_u64();
        let spec = SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            offset,
            alloc_size: length,
            width,
            height,
            bytes_per_element: 4,
            bytes_per_row: width * 4,
            element_width: 1,
            element_height: 1,
            client_address_ranges: Some(
                (0..length / page)
                    .map(|i| PhysicalMemoryRange {
                        physical_address: first + i * page,
                        physical_length: page,
                    })
                    .collect(),
            ),
            ..SurfaceSpec::default()
        };
        (OsSurfaces::new().create(&spec).unwrap(), pages, length)
    }

    /// A host-only 'BGRA' surface of `width` × `height`, 4 bytes an element, rows of `width × 4`.
    pub(crate) fn bgra_surface(width: u64, height: u64) -> OsSurface {
        let spec = SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            alloc_size: width * height * 4,
            width,
            height,
            bytes_per_element: 4,
            bytes_per_row: width * 4,
            element_width: 1,
            element_height: 1,
            ..SurfaceSpec::default()
        };
        OsSurfaces::new().create(&spec).unwrap()
    }

    /// The pages of [`bgra_surface_over_at`], freed.
    pub(crate) fn free(pages: HostPtr, length: u64) {
        assert!(
            crate::remap::VulkanVm::new()
                .deallocate(pages, length)
                .is_success()
        );
    }

    /// A lookup of a surface (`lookup_by_port`), a reference of its own, answers what the surface
    /// was made over: its offset and memory
    #[test]
    fn a_lookup_of_a_surface_answers_its_record() {
        let (s, pages, length) = bgra_surface_over_at(16, 16, 0x100);
        let t = s.lookup_by_port().unwrap();
        assert_ne!(t.id(), s.id());
        assert_eq!(surface_offset(&t), 0x100);
        assert_eq!(surface_memory(&t), surface_memory(&s));
        drop((t, s));
        free(pages, length);
    }
}
