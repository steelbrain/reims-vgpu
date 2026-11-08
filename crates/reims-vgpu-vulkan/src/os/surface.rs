//! What this platform reads of a surface beyond core's `Surface` trait, written against it and
//! each OS's record of the surfaces it made (`recorded_mut`): a plane's layout and YCbCr facts,
//! the host memory a surface's bytes are over, whether the GPU's writes through it reach that
//! memory, and the alias its buffers lie over.

use std::sync::{Arc, Weak};

use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::surface::Surface;

use super::{OsSurface, host_os};
use crate::gpu::backing::Alias;

/// Plane `plane` of `surface`, as a texture on it reads it: its bytes' offset from the surface's
/// base address, its bytes per row, and whether the surface has planes. A surface without planes
/// is its plane 0. `None` for a plane the surface lacks, of which Metal makes no texture
/// (`textures_backing_records`). The plane names no `kIOSurfaceClientDirection`, which the
/// runtime refuses: where the GPU's writes through a surface never reach its memory
/// ([`gpu_read_only`]), the buffer under the texture lies over a read-only alias of that memory
/// (the device's `buffer_over_surface_at`), and the GPU's writes never reach it there either.
pub(crate) fn plane(surface: &OsSurface, plane: u64) -> Option<rt::IoSurfacePlane> {
    if surface.plane_count() == 0 {
        return (plane == 0).then(|| rt::IoSurfacePlane::new(0, surface.bytes_per_row(), false));
    }
    let (offset, bytes_per_row) = host_os::plane_at(surface, plane)?;
    Some(rt::IoSurfacePlane::new(offset, bytes_per_row, true))
}

/// One plane of a surface, as a host outside the runtime reads it (the video device, whose
/// codecs read and write the guest's pixel buffers in place): its bytes' offset from the
/// surface's base address, its size in elements, and its bytes per element and per row, as the
/// surface was made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneLayout {
    /// Its bytes' offset from the surface's base address (`IOSurfaceGetBaseAddressOfPlane` less
    /// `IOSurfaceGetBaseAddress`).
    pub offset: u64,
    /// `IOSurfaceGetWidthOfPlane`.
    pub width: u64,
    /// `IOSurfaceGetHeightOfPlane`.
    pub height: u64,
    /// `IOSurfaceGetBytesPerElementOfPlane`.
    pub bytes_per_element: u64,
    /// `IOSurfaceGetBytesPerRowOfPlane`.
    pub bytes_per_row: u64,
}

/// Plane `plane` of `surface`, a surface with planes; `None` for a plane it lacks, and for every
/// plane of a surface without planes, whose layout is the surface's own.
pub fn plane_layout(surface: &OsSurface, plane: u64) -> Option<PlaneLayout> {
    let (offset, bytes_per_row) = host_os::plane_at(surface, plane)?;
    let (width, height, bytes_per_element) = host_os::plane_shape(surface, plane)?;
    Some(PlaneLayout {
        offset,
        width,
        height,
        bytes_per_element,
        bytes_per_row,
    })
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

/// The `kIOSurfaceClientDirection` of a surface core makes over the guest's pages read-only for
/// its clients (core's `iosurface::descriptor`).
pub(crate) const READ_ONLY_DIRECTION: u32 = 2;

/// Whether the GPU's writes through `surface` never reach its memory: it was made over the
/// guest's pages with `kIOSurfaceClientDirection` 2, which Metal maps read-only for the GPU.
///
/// Measured on the M3 Max (macOS 26.7, one case per process, BGRA8Unorm surfaces of 64 by 16, 256
/// by 256 and 1024 by 1024 texels, `tests/client_direction.rs`): through a texture on such a
/// surface, a blit's copy or colour fill and a kernel's writes of one or two texels reach the later
/// commands of their command buffer and never the pages, which later command buffers and the CPU
/// read unchanged; of a row of 64 texels or more, they fail the command buffer with a GPU page
/// fault (`MTLCommandBufferErrorPageFault`), the encoders after them running nothing. So does a
/// blit's copy of a texture of a row of 64 texels into a buffer made on the surface with
/// `newBufferWithIOSurface:`, where one of two texels is dropped as above. A blit's copy from a
/// buffer or colour fill of 1 to 8 texels each way into the texture is dropped at every origin
/// tried (0,0, 1,1 and one far inside, on surfaces of 64 by 16 and 256 by 256 texels); of a row of
/// 16 to 63 texels it faults at some origins and is dropped at others (16 at 0,0 faults, at 1,1 is
/// dropped). A blit's fill of a buffer made on the surface, its copy into one from another buffer,
/// a kernel's stores into one, plain or by an atomic exchange, and a vertex or fragment function's,
/// of 8 bytes to 4 MiB, are dropped as above, the later commands of the command buffer reading what
/// of the write the GPU's caches still hold (all of 4 MiB with nothing else on the GPU, less beside
/// other GPU work, none after 64 MiB of it; a draw's, all, part or none of it, case to case, in one
/// process). A render pass that clears it or draws into it and stores fails its command buffer so;
/// one that stores nothing, or loads, stores and draws nothing, completes, writing nothing.
/// `replaceRegion:` writes through the surface's base address: the page of a surface over one page,
/// which lies there; a surface over several (4, 16 and 256) has a mapping of its own there,
/// read-only, and the write ends the process with a bus error, at every extent measured
/// ([`base_in_pages`]). The key is not read back (`IOSurfaceCopyValue` answers none), so the
/// surfaces made here record it. A surface of its own memory takes the key and the GPU's writes
/// alike; one over the guest's pages takes 3 and the writes land, and none of 0, 1, 4 or 255
/// (`IOSurfaceCreate` answers NULL); core sets 2 alone.
pub(crate) fn gpu_read_only(surface: &OsSurface) -> bool {
    recorded(surface, |made| made.gpu_read_only).unwrap_or(false)
}

/// Dies at `sel`, a command that may write `what` into the memory of a surface whose GPU writes
/// never reach it ([`gpu_read_only`]): Metal's GPU drops some such writes and faults the command
/// buffer at others by their extent, which nothing here can tell before the GPU runs the
/// command, and the runtime's device loses itself at the same fault, so the write is never sent.
/// A blit's fill of a buffer made on the surface, its copy into one from another buffer, and a
/// kernel's, vertex function's or fragment function's stores into one, which Metal drops at
/// every extent measured, as the runtime does, are sent; a tile function's, which nothing
/// measured, are not. So is a blit's copy from a buffer or colour fill into a `BGRA8Unorm` 2D
/// texture made on the surface, of 1 to 8 texels each way inside level 0, on a surface whose rows
/// hold 256 bytes or more, which Metal drops at every origin measured
/// (`gpu::encode_blit::dropped_where_measured`).
pub(crate) fn dropped_write(sel: &'static str, what: &str) -> ! {
    crate::answer::unrepresentable(
        sel,
        &format!(
            "{what} a surface made read-only for the GPU (kIOSurfaceClientDirection 2), whose \
             writes Metal's GPU drops or faults on by their extent"
        ),
    )
}

/// The alias of the allocation of `surface`, one at an offset into it ([`surface_offset`]) or
/// whose GPU writes never reach its memory, read-only then ([`gpu_read_only`]), that every buffer
/// over it lies over while one lives: the live one, or a new one, kept for the next: the surface's
/// own view of the guest's pages where it has one (Linux's stand-in over several ranges of them,
/// `own_alias`), borrowed, and otherwise the one `make` makes; `None` where `make` makes none.
/// The buffers then lie in one import of the runtime's, as those over a surface at no offset
/// whose writes land lie over its base address: a write through one reaches the later commands
/// of its command buffer that read another, as on Metal (`tests/client_direction.rs`), and a
/// buffer over the surface can share the range of another over the same first byte (the device's
/// `new_buffer_on_surface`).
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
/// not, has a mapping of its own (measured on the M3 Max: one of 4 contiguous pages), read-only
/// for a surface of [`gpu_read_only`].
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

/// Writes `bytes` at `offset` in the bytes of `surface`, counted from its first byte, into the
/// guest's pages it was made over, where those bytes lie (from its `kIOSurfaceOffset` in the
/// pages, in their order): what the CPU writes through Metal's texture on such a surface
/// reaches (`replaceRegion:`, measured: `tests/client_direction.rs`). False, writing nothing,
/// for a surface not made over the guest's pages, and for bytes past them.
pub(crate) fn write_surface_bytes(surface: &OsSurface, offset: u64, bytes: &[u8]) -> bool {
    let Some(held) = recorded(surface, |made| {
        let first = made.offset.saturating_add(offset);
        in_runs(
            &made.pages,
            &(first..first.saturating_add(bytes.len().to_u64())),
        )
    }) else {
        return false;
    };
    let total: u64 = held.iter().map(|r| r.end - r.start).sum();
    if total != bytes.len().to_u64() {
        return false;
    }
    let mut from = bytes;
    for run in held {
        let (now, rest) = from.split_at((run.end - run.start).to_usize());
        // SAFETY: `run` lies in the guest's pages the surface was made over, which the host maps
        // readable and writable while the surface lives, as core holds them; `now` is the
        // caller's, apart from them.
        unsafe {
            std::ptr::copy_nonoverlapping(
                now.as_ptr(),
                std::ptr::with_exposed_provenance_mut::<u8>(run.start.to_usize()),
                now.len(),
            );
        }
        from = rest;
    }
    true
}

/// One lend of a surface's bytes at a time in the process ([`lend_surface_bytes`]), so no two
/// lends' slices alias.
static LENDING: reims_vgpu_core::sync::Mutex<()> = reims_vgpu_core::sync::Mutex::new(());

/// Lends `surface`'s bytes, from its first byte to the end of its allocation, to `write`, for a
/// host outside the runtime: the video device's decoders, which write decoded frames into the
/// guest's pixel buffers. What `write` leaves in them is in the surface's memory when this
/// returns.
///
/// - A surface over the guest's pages whose bytes lie in one run of them, in the host's memory,
///   is lent in place, in the pages.
/// - One whose bytes lie in several runs, as the pages of a guest's allocation mostly do, is lent
///   in place on Linux, through its own view of the pages, a mapping of them at its base address
///   (`host_os::base_views_pages`), so that what `write` writes lands in the pages as it writes
///   it. Elsewhere it is lent a copy, gathered from the pages and scattered back into them: on
///   macOS the surface's own mapping of them takes the CPU's writes under IOSurface's lock
///   (reims-vgpu-metal's `os/surface_bytes.rs`), which this lend does not take, and is read-only
///   for a surface made read-only for its clients ([`base_in_pages`]). Bytes past the pages read
///   as zeros and are dropped.
/// - A surface of its own memory is lent from its base address.
///
/// Refused, with why: a protected surface, whose memory the CPU never reaches, and one the guest
/// made read-only for its clients ([`gpu_read_only`]), whose view is read-only. One lend runs at
/// a time in the process, so `write` must not lend another.
pub fn lend_surface_bytes(
    surface: &OsSurface,
    write: &mut dyn FnMut(&mut [u8]),
) -> Result<(), &'static str> {
    let base = surface.base_address().as_ptr::<u8>().addr().to_u64();
    if surface.protection_options() != 0 || base == 0 {
        return Err("a protected surface, whose memory the CPU never reaches");
    }
    if gpu_read_only(surface) {
        return Err("a surface the guest made read-only for its clients");
    }
    let made = recorded(surface, |made| (made.pages.clone(), made.offset));
    let offset = made.as_ref().map_or(0, |(_, offset)| *offset);
    let length = surface.alloc_size().saturating_sub(offset);
    let _lend = LENDING.lock();
    let pages = match made {
        Some((pages, _)) if !pages.is_empty() => pages,
        _ => {
            // SAFETY: a surface of its own memory has `alloc_size` bytes from its allocation's
            // first byte, its base address `offset` before them, readable and writable while the
            // caller holds it; `LENDING` keeps every other lend's slice apart from this one.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    std::ptr::with_exposed_provenance_mut::<u8>(base.to_usize()),
                    length.to_usize(),
                )
            };
            write(bytes);
            return Ok(());
        }
    };
    let held = in_runs(&pages, &(offset..offset.saturating_add(length)));
    let lent = lent_where(surface, &held, length);
    log_first_lend(surface, length, &held, lent);
    if let Some(first) = lent.first(base, &held) {
        // SAFETY: `first` is the first of `length` bytes in the guest's pages the surface was
        // made over, in one run of them or in the surface's own writable view of them, which the
        // host maps readable and writable while the surface lives, as core holds them, and the
        // caller holds the surface; `LENDING` keeps every other lend's slice apart from this one.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                std::ptr::with_exposed_provenance_mut::<u8>(first.to_usize()),
                length.to_usize(),
            )
        };
        write(bytes);
        return Ok(());
    }
    let mut copy = vec![0u8; length.to_usize()];
    let mut at = 0usize;
    for run in &held {
        let len = (run.end - run.start).to_usize();
        // SAFETY: as above, `run` lies in the guest's pages; `copy` is ours, apart from them.
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::with_exposed_provenance::<u8>(run.start.to_usize()),
                copy.as_mut_ptr().add(at),
                len,
            );
        }
        at += len;
    }
    write(&mut copy);
    let mut at = 0usize;
    for run in &held {
        let len = (run.end - run.start).to_usize();
        // SAFETY: as above.
        unsafe {
            std::ptr::copy_nonoverlapping(
                copy.as_ptr().add(at),
                std::ptr::with_exposed_provenance_mut::<u8>(run.start.to_usize()),
                len,
            );
        }
        at += len;
    }
    Ok(())
}

/// Lends `surface`'s bytes, laid out as [`lend_surface_bytes`] lends them, to `read`, for a host
/// outside the runtime that only reads them: the video devices' scaler sources and encoder
/// frames. As [`lend_surface_bytes`], in place where the bytes lie in one run of the guest's
/// pages or in the surface's view of them, and as a copy gathered from them elsewhere, but
/// nothing is written back, so a surface the guest made read-only for its clients is lent too:
/// reading it is what read-only allows. Refused, with why: a protected surface. One lend runs at
/// a time in the process, so `read` must not lend another.
pub fn lend_surface_bytes_for_reading(
    surface: &OsSurface,
    read: &mut dyn FnMut(&[u8]),
) -> Result<(), &'static str> {
    let base = surface.base_address().as_ptr::<u8>().addr().to_u64();
    if surface.protection_options() != 0 || base == 0 {
        return Err("a protected surface, whose memory the CPU never reaches");
    }
    let made = recorded(surface, |made| (made.pages.clone(), made.offset));
    let offset = made.as_ref().map_or(0, |(_, offset)| *offset);
    let length = surface.alloc_size().saturating_sub(offset);
    let _lend = LENDING.lock();
    let pages = match made {
        Some((pages, _)) if !pages.is_empty() => pages,
        _ => {
            // SAFETY: a surface of its own memory has `alloc_size` bytes from its allocation's
            // first byte, its base address `offset` before them, readable while the caller holds
            // it; `LENDING` keeps every other lend's slice apart from this one.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    std::ptr::with_exposed_provenance::<u8>(base.to_usize()),
                    length.to_usize(),
                )
            };
            read(bytes);
            return Ok(());
        }
    };
    let held = in_runs(&pages, &(offset..offset.saturating_add(length)));
    let lent = lent_where(surface, &held, length);
    log_first_lend(surface, length, &held, lent);
    if let Some(first) = lent.first(base, &held) {
        // SAFETY: `first` is the first of `length` bytes in the guest's pages the surface was
        // made over, in one run of them or in the surface's own view of them, which the host maps
        // readable while the surface lives, as core holds them, and the caller holds the surface;
        // `LENDING` keeps every other lend's slice apart from this one.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                std::ptr::with_exposed_provenance::<u8>(first.to_usize()),
                length.to_usize(),
            )
        };
        read(bytes);
        return Ok(());
    }
    let mut copy = vec![0u8; length.to_usize()];
    let mut at = 0usize;
    for run in &held {
        let len = (run.end - run.start).to_usize();
        // SAFETY: as above, `run` lies in the guest's pages; `copy` is ours, apart from them.
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::with_exposed_provenance::<u8>(run.start.to_usize()),
                copy.as_mut_ptr().add(at),
                len,
            );
        }
        at += len;
    }
    read(&copy);
    Ok(())
}

/// Where a lend of a surface over the guest's pages reaches its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lent {
    /// In the one run of the pages they lie in.
    InTheRun,
    /// Through the surface's own view of the pages, at its base address.
    ThroughTheView,
    /// In a copy, gathered from the pages.
    AsACopy,
}

impl Lent {
    /// The first byte lent in place, from the surface's `base` address or the run `held` names;
    /// `None` for a copy.
    fn first(self, base: u64, held: &[std::ops::Range<u64>]) -> Option<u64> {
        match self {
            Lent::InTheRun => held.first().map(|run| run.start),
            Lent::ThroughTheView => Some(base),
            Lent::AsACopy => None,
        }
    }
}

/// Where a lend of `surface`'s `length` bytes, which lie in the runs of the guest's pages
/// `held`, reaches them ([`lend_surface_bytes`]).
fn lent_where(surface: &OsSurface, held: &[std::ops::Range<u64>], length: u64) -> Lent {
    if matches!(held, [run] if run.end - run.start == length) {
        Lent::InTheRun
    } else if host_os::base_views_pages(surface) {
        Lent::ThroughTheView
    } else {
        Lent::AsACopy
    }
}

/// Logs, at the first lend of `surface`'s `length` bytes over the guest's pages, how many runs of
/// them the bytes lie in, `held`, and where they were lent, as a lend's cost depends on it: a copy
/// of the whole surface where they lie in several and no view of them is writable (ours: a guest's
/// 4K frame was copied so up to 2.2 times as slowly as a host buffer's). Once a surface, by its
/// record.
fn log_first_lend(surface: &OsSurface, length: u64, held: &[std::ops::Range<u64>], lent: Lent) {
    let first = host_os::recorded_mut(surface, |made| !std::mem::replace(&mut made.lent, true));
    if first != Some(true) {
        return;
    }
    let sizes = held.iter().map(|run| run.end - run.start);
    let (smallest, largest) = (sizes.clone().min().unwrap_or(0), sizes.max().unwrap_or(0));
    let format = surface.pixel_format().to_be_bytes();
    let format: String = format
        .iter()
        .map(|&b| {
            if b.is_ascii_graphic() {
                char::from(b)
            } else {
                '.'
            }
        })
        .collect();
    log::info!(
        target: "reims_vgpu_vulkan::os",
        "the first lend of the IOSurface at {:#x}, {}x{} '{format}': its {length} bytes lie in {} \
         runs of the guest's pages, of {smallest} to {largest} bytes, and it is lent {}",
        surface.base_address().as_ptr::<u8>().addr(),
        surface.width(),
        surface.height(),
        held.len(),
        match lent {
            Lent::InTheRun => "in place, in the run",
            Lent::ThroughTheView => "in place, through its view of the pages",
            Lent::AsACopy => "as a copy",
        },
    );
}

/// What the OS's record of `surface` holds, read by `read`; `None` for a surface it did not
/// record.
fn recorded<T>(surface: &OsSurface, read: impl FnOnce(&MadeOver) -> T) -> Option<T> {
    host_os::recorded_mut(surface, |made| read(made))
}

/// What a surface was made over: the guest's pages, in its bytes' order, merged into runs, and
/// its bytes' offset into its allocation (`kIOSurfaceOffset`); whether the GPU's writes through
/// it never reach them ([`gpu_read_only`]), the alias of its allocation its buffers lie over
/// while one lives, where they lie over one ([`surface_alias`]), and whether it was lent yet.
#[derive(Debug, Clone, Default)]
pub(crate) struct MadeOver {
    pub(crate) pages: Vec<std::ops::Range<u64>>,
    pub(crate) offset: u64,
    pub(crate) gpu_read_only: bool,
    pub(crate) alias: Weak<Alias>,
    /// Whether its bytes were lent yet ([`log_first_lend`]).
    pub(crate) lent: bool,
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
    use reims_vgpu_core::host::{HostPtr, PhysicalMemoryRange};
    use reims_vgpu_core::platform::Handle as _;
    use reims_vgpu_core::platform::os::surface::{IoSurfaces as _, SurfaceSpec};
    use reims_vgpu_core::platform::os::vm::HostVm as _;

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
        assert!(OsVm::new().deallocate(pages, length).is_success());
    }

    /// A lookup of a surface (`lookup_by_port`), a reference
    /// of its own, answers what the surface was made over: its offset, memory and GPU writes
    #[test]
    fn a_lookup_of_a_surface_answers_its_record() {
        let (s, pages, length) = bgra_surface_over_at(16, 16, 0x100);
        let t = s.lookup_by_port().unwrap();
        assert_ne!(t.id(), s.id());
        assert_eq!(surface_offset(&t), 0x100);
        assert_eq!(surface_memory(&t), surface_memory(&s));
        let page = OsVm::new().page_size();
        let guest = allocate_aliasable(page).unwrap();
        let spec = SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            alloc_size: page,
            width: 16,
            height: 16,
            bytes_per_element: 4,
            bytes_per_row: 64,
            element_width: 1,
            element_height: 1,
            client_direction: Some(READ_ONLY_DIRECTION),
            client_address_ranges: Some(vec![PhysicalMemoryRange {
                physical_address: guest.addr().to_u64(),
                physical_length: page,
            }]),
            ..SurfaceSpec::default()
        };
        let read_only = OsSurfaces::new().create(&spec).unwrap();
        let looked_up = read_only.lookup_by_port().unwrap();
        assert!(gpu_read_only(&read_only) && gpu_read_only(&looked_up));
        assert!(!gpu_read_only(&s) && !gpu_read_only(&t));
        drop((t, s, looked_up, read_only));
        free(pages, length);
        free(guest, page);
    }

    /// The byte at `at` of host memory.
    fn byte_at(at: u64) -> u8 {
        // SAFETY: the tests name bytes of pages they allocated and have not freed.
        unsafe { *std::ptr::with_exposed_provenance::<u8>(at.to_usize()) }
    }

    /// A surface's bytes are lent from its first byte to its allocation's end: in place where
    /// they lie in one run of its pages, through its view of them where they lie in several on
    /// Linux and as a copy gathered from and scattered back into them elsewhere, from its own
    /// memory where it has none; one the guest made read-only is refused for writing, and lent
    /// for reading
    #[test]
    fn a_surface_lends_its_bytes_where_they_lie() {
        // One run, 0x100 bytes into it.
        let (s, pages, length) = bgra_surface_over_at(16, 16, 0x100);
        let first = pages.addr().to_u64();
        let mut lent = 0;
        lend_surface_bytes(&s, &mut |b| {
            lent = b.len();
            b[0] = 0xa5;
            b[b.len() - 1] = 0x5a;
        })
        .unwrap();
        assert_eq!(lent.to_u64(), length - 0x100);
        assert_eq!(byte_at(first + 0x100), 0xa5);
        assert_eq!(byte_at(first + length - 1), 0x5a);
        drop(s);
        free(pages, length);

        // Two pages in reverse order, 0x40 bytes into the first named.
        let page = OsVm::new().page_size();
        let guest = allocate_aliasable(2 * page).unwrap();
        let (low, high) = (guest.addr().to_u64(), guest.addr().to_u64() + page);
        // SAFETY: the low page is ours, allocated above.
        unsafe { *std::ptr::with_exposed_provenance_mut::<u8>(low.to_usize()) = 7 };
        let spec = SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            offset: 0x40,
            alloc_size: 2 * page,
            width: 16,
            height: 16,
            bytes_per_element: 4,
            bytes_per_row: 64,
            element_width: 1,
            element_height: 1,
            client_address_ranges: Some(
                [high, low]
                    .map(|at| PhysicalMemoryRange {
                        physical_address: at,
                        physical_length: page,
                    })
                    .to_vec(),
            ),
            ..SurfaceSpec::default()
        };
        let s = OsSurfaces::new().create(&spec).unwrap();
        let split = (page - 0x40).to_usize();
        let (mut before, mut at, mut landed) = (0, 0, 0);
        lend_surface_bytes(&s, &mut |b| {
            assert_eq!(b.len(), (2 * page - 0x40).to_usize());
            before = b[split];
            at = b.as_ptr().addr();
            b[0] = 1;
            landed = byte_at(high + 0x40);
            b[split] = 2;
        })
        .unwrap();
        assert_eq!(before, 7, "the bytes lent are the pages'");
        let in_view = cfg!(target_os = "linux");
        assert_eq!(
            (at == s.base_address().as_ptr::<u8>().addr(), landed == 1),
            (in_view, in_view),
            "on Linux, lent in place through the surface's view of the pages; else a copy"
        );
        assert_eq!(byte_at(high + 0x40), 1);
        assert_eq!(byte_at(low), 2);
        let read_only = OsSurfaces::new()
            .create(&SurfaceSpec {
                client_direction: Some(READ_ONLY_DIRECTION),
                ..spec
            })
            .unwrap();
        assert!(lend_surface_bytes(&read_only, &mut |_| unreachable!()).is_err());
        let mut gathered = (0, 0, 0);
        lend_surface_bytes_for_reading(&read_only, &mut |b| {
            gathered = (b.len(), b[0], b[split]);
        })
        .unwrap();
        assert_eq!(
            gathered,
            ((2 * page - 0x40).to_usize(), 1, 2),
            "a read-only surface is lent for reading, gathered from its pages"
        );
        drop((s, read_only));
        free(guest, 2 * page);

        // Its own memory.
        let s = bgra_surface(16, 16);
        let mut lent = 0;
        lend_surface_bytes(&s, &mut |b| {
            lent = b.len();
            b.fill(3);
        })
        .unwrap();
        assert_eq!(lent, 16 * 16 * 4);
        let mut read = (0, 0);
        lend_surface_bytes_for_reading(&s, &mut |b| read = (b.len(), b[b.len() - 1])).unwrap();
        assert_eq!(read, (16 * 16 * 4, 3));
    }
}
