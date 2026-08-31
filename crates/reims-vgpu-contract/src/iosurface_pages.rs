//! IOSurface mapper/page-table planning (port of `host/utils/reims-vgpu-iosurface-pages`).

use crate::endian::{ld16, ld32, ld64};
use crate::pixel_format;
use crate::{align_up_u64, checked_add_u64, checked_mul_u64};

pub const U32_SIZE: usize = 4;

/// Minimum typed type-11 object-list descriptor length (geometry prefix).
/// Live blobs are often longer (0x38/0x58) with an unused/constant tail.
/// There is no multi-mip level-record layout on type-11: Metal rejects
/// mipmapped IOSurface textures (`mipmapLevelCount > 1`).
///
/// `TYPE11_` rather than `TEXTURE_DESC_`, which is what these were called.
/// `runtime::decode::resource` declares a `TEXTURE_DESC_WIDTH`, a
/// `TEXTURE_DESC_HEIGHT` and a `TEXTURE_DESC_PIXEL_FORMAT` of its own for the
/// serialized `MTLTextureDescriptor` — a different record, at 60, 64 and 86
/// against the 0x18, 0x1c and 0x16 here. Two records under one set of names, and
/// `runtime::texture` imports from both modules in the same file. Neither value
/// is out of range for the other's record, so picking the wrong import yields a
/// plausible width rather than a bounds failure.
pub const TYPE11_DESC_MIN_LEN: usize = 0x20;
pub const TYPE11_DESC_MAPPING_ID: usize = 0x00;
pub const TYPE11_DESC_OBJECT_REF: usize = 0x10;
pub const TYPE11_DESC_PIXEL_FORMAT: usize = 0x16;
pub const TYPE11_DESC_WIDTH: usize = 0x18;
pub const TYPE11_DESC_HEIGHT: usize = 0x1c;

/// One entry of the guest's mapper request array: `{type, mapping_id, reserved}`.
///
/// The length, the two request types and the three field offsets describe one
/// record and belong together. They did not: `model::regs` carried its own
/// `MAPPER_REQUEST_MAP` / `_UNMAP` / `_ENTRY_LEN` at the same values and none of
/// the offsets, so `runtime::mapper` decoded this record through the constants
/// here while `runtime::drain` decoded the same guest bytes through those.
/// Neither copy could tell the other it had moved.
pub const MAPPER_REQUEST_ENTRY_LEN: usize = 16;
pub const MAPPER_REQUEST_TYPE: usize = 0x00;
pub const MAPPER_REQUEST_MAPPING_ID: usize = 0x04;
pub const MAPPER_REQUEST_RESERVED: usize = 0x08;
pub const MAPPER_REQUEST_MAP: u32 = 1;
pub const MAPPER_REQUEST_UNMAP: u32 = 2;

/*
 * Directed register handoff at do_host_mapping_gated → iosfc producer write:
 * guest leaves mapper device / request type / MappingInternal* in these
 * arm64e xregs (kb + archived reims-vgpu-iosurface-pages format header).
 */
pub const MAPPER_CAPTURE_REG_MAPPER_DEVICE: u32 = 19;
pub const MAPPER_CAPTURE_REG_REQUEST_TYPE: u32 = 21;
pub const MAPPER_CAPTURE_REG_MAPPING_INTERNAL: u32 = 22;

pub const ROW_BYTES_ALIGN: u64 = 128;
pub const DEVICE_PLANE_DESC_LEN: usize = 0x40;
pub const DEVICE_PLANE_OFFSET: usize = 0x08;
pub const DEVICE_PLANE_BASE: usize = 0x0c;
pub const DEVICE_PLANE_SIZE: usize = 0x10;
pub const DEVICE_PLANE_DIMS: usize = 0x14;
pub const DEVICE_PLANE_BPR: usize = 0x1c;
pub const DEVICE_PLANE_BPE: usize = 0x20;

/// Bit position of the width in a `dims` word.
const DIMS_WIDTH_SHIFT: u32 = 8;
/// Bit position of the height in a `dims` word.
const DIMS_HEIGHT_SHIFT: u32 = 40;
/// Both extents are 24 bits, so neither reaches its neighbouring byte.
const DIMS_EXTENT_MASK: u64 = 0x00ff_ffff;

/// The width and height packed into a device-surface or device-plane `dims`
/// word.
///
/// One 64-bit word carrying four fields, one byte and three bytes twice over:
/// element width at byte 0, width at bytes 1–3, element height at byte 4,
/// height at bytes 5–7. This device reads only the two extents; the element
/// sizes are the guest's own subsampling description and nothing here applies
/// them.
///
/// Both record decoders below extract the pair, and until this existed each
/// spelled the shifts itself — along with a test fixture that packs them and an
/// inline literal in another test that packs them again. Four spellings of one
/// layout, in a file whose whole subject is layouts.
pub fn dims_extent(dims: u64) -> (u32, u32) {
    (
        ((dims >> DIMS_WIDTH_SHIFT) & DIMS_EXTENT_MASK) as u32,
        ((dims >> DIMS_HEIGHT_SHIFT) & DIMS_EXTENT_MASK) as u32,
    )
}

pub const DEVICE_DESC_LEN: usize = 0x200;
pub const DEVICE_DESC_PIXEL_FORMAT: usize = 0x04;
pub const DEVICE_DESC_BASE_OFFSET: usize = 0x08;
pub const DEVICE_DESC_ALLOC_SIZE: usize = 0x10;
pub const DEVICE_DESC_DIMS: usize = 0x14;
pub const DEVICE_DESC_BPR: usize = 0x1c;
pub const DEVICE_DESC_BPE: usize = 0x20;
pub const DEVICE_DESC_PLANE_COUNT: usize = 0x24;
pub const DEVICE_DESC_PLANES: usize = 0x40;

#[inline]
pub fn page_size_of(page_shift: u32) -> u64 {
    1u64 << page_shift
}

pub const PAGE_ENTRY_VALID: u32 = 0x1;
pub const PAGE_ENTRY_PFN_SHIFT: u32 = 2;

pub const MAPPING_INTERNAL_BACKPTR: u64 = 0x18;
pub const MAPPING_INTERNAL_ID: u64 = 0x30;
pub const MAPPING_INTERNAL_DESC_PTR: u64 = 0x38;
pub const MAPPING_INTERNAL_SIZE: u64 = 0x40;
pub const MAPPING_INTERNAL_EXPECTED_SIZE: u32 = 0x200;
pub const MAPPING_INTERNAL_PAGE_FIELD_48: u64 = 0x48;
pub const MAPPING_INTERNAL_PAGE_FIELD_50: u64 = 0x50;
pub const MAPPING_INTERNAL_PAGE_COUNT: u64 = 0x70;
pub const MAPPING_PAGE_TABLE_FROM_F48: u64 = 0xb8;
pub const MAPPING_PAGE_TABLE_FROM_F50: u64 = 0x28;

pub const ARM_KERNEL_VA_MASK: u64 = 0xffffff00_00000000;
pub const ARM_KERNEL_VA_BASE: u64 = 0xfffffe00_00000000;
/// x86_64 Darwin canonical kernel half (bits 63:47 all ones in 48-bit VA).
pub const X86_KERNEL_VA_MIN: u64 = 0xffff8000_00000000;

/// Why a mapper/page-table resolve refused, or [`Status::Ok`] if it did not.
///
/// Every variant here names a check this walker actually performs. `ErrArgs`,
/// `ErrOverflow` and `ErrSpanRange` used to sit alongside them and were never
/// constructed: the argument checks are `ErrShortDescriptor`, the arithmetic
/// goes through `checked_*` helpers that fold overflow into the length and
/// page-count refusals, and the span walk refuses as `ErrPageCount`. A refusal
/// class the code cannot reach is a claim the fail log can never substantiate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok,
    ErrShortDescriptor(&'static str),
    ErrNotKernelVa(&'static str),
    ErrInternalRead(&'static str),
    ErrInternalOwner(&'static str),
    ErrInternalMappingId(&'static str),
    ErrInternalSize(&'static str),
    ErrInternalFields(&'static str),
    ErrPageCount(&'static str),
    ErrPageTableRead(&'static str),
    ErrPageEntry(&'static str),
    ErrNoPageTable(&'static str),
}

impl reims_vgpu_observe::Refusal for Status {
    fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Ok => None,
            Self::ErrShortDescriptor(reason)
            | Self::ErrNotKernelVa(reason)
            | Self::ErrInternalRead(reason)
            | Self::ErrInternalOwner(reason)
            | Self::ErrInternalMappingId(reason)
            | Self::ErrInternalSize(reason)
            | Self::ErrInternalFields(reason)
            | Self::ErrPageCount(reason)
            | Self::ErrPageTableRead(reason)
            | Self::ErrPageEntry(reason)
            | Self::ErrNoPageTable(reason) => Some(reason),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        let class = match self {
            Self::Ok => return Vec::new(),
            Self::ErrShortDescriptor(_) => "short_descriptor",
            Self::ErrNotKernelVa(_) => "not_kernel_va",
            Self::ErrInternalRead(_) => "internal_read",
            Self::ErrInternalOwner(_) => "internal_owner",
            Self::ErrInternalMappingId(_) => "internal_mapping_id",
            Self::ErrInternalSize(_) => "internal_size",
            Self::ErrInternalFields(_) => "internal_fields",
            Self::ErrPageCount(_) => "page_count",
            Self::ErrPageTableRead(_) => "page_table_read",
            Self::ErrPageEntry(_) => "page_entry",
            Self::ErrNoPageTable(_) => "no_page_table",
        };
        vec![("class", class.to_string())]
    }
}

/// Memory access callbacks for mapper/page-table reads.
pub trait PagesMemory {
    fn read(&self, address: u64, dst: &mut [u8]) -> bool;
    fn is_kernel_va(&self, address: u64) -> bool {
        guest_kernel_va(address)
    }
    fn is_ram_gpa(&self, _address: u64) -> bool {
        true
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureDescriptor {
    pub mapping_id64: u64,
    pub mapping_id: u32,
    pub object_ref: u32,
    pub pixel_format: u16,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MapperRequestEntry {
    pub request_type: u32,
    pub mapping_id: u32,
    pub reserved: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MapperInternalFields {
    pub internal_kva: u64,
    pub has_mapper_device: bool,
    pub mapper_device_kva: u64,
    pub owner_kva: u64,
    pub mapping_id: u32,
    pub internal_size: u32,
    pub page_field_48: u64,
    pub page_field_50: u64,
    pub raw_page_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageTablePlan {
    pub entries: Vec<u32>,
    pub page_table_kva: u64,
    pub min_size: u64,
    pub required_pages: u64,
    /// Whether the `MappingInternal` field this plan did *not* come through was
    /// populated as well.
    ///
    /// [`build_table_plan`] used to chase `+0x48` then `+0xb8`, and `+0x50`
    /// then `+0x28`, returning the entries of whichever parsed first — the
    /// classic "try both, keep the one that works" ladder. Two driven arm64
    /// boots retired it; see that function. This is what is left of the
    /// measurement, and it stays because it is free: `contract` stays clear of
    /// the observability dependency, so the fact travels in the plan rather
    /// than being emitted here.
    pub candidates: CandidateOutcome,
}

/// What the unused `MappingInternal` page field held on one successful plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CandidateOutcome {
    /// `+0x50` held a kernel VA as well as `+0x48`, which the plan came
    /// through.
    ///
    /// Measured `true` on every one of 223 successful resolves across two
    /// driven arm64 workloads, which is *why* the second chase could go: the
    /// field is populated essentially always, so it never discriminated
    /// anything. A run where this turned mostly `false` would mean the two
    /// fields really are two layouts and the deletion was wrong.
    pub other_field_populated: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceSurfaceRecord {
    pub pixel_format: u32,
    pub base_offset: u32,
    pub alloc_size: u32,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub bytes_per_element: u16,
    pub plane_count: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DevicePlaneRecord {
    pub plane_offset: u32,
    pub plane_base: u32,
    pub plane_size: u32,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub bytes_per_element: u16,
}

pub fn arm_kernel_va(address: u64) -> bool {
    (address & ARM_KERNEL_VA_MASK) == ARM_KERNEL_VA_BASE
}

pub fn x86_kernel_va(address: u64) -> bool {
    address >= X86_KERNEL_VA_MIN
}

/// Guest kernel VA: arm64e TTBR1 window **or** x86_64 Darwin high half.
pub fn guest_kernel_va(address: u64) -> bool {
    arm_kernel_va(address) || x86_kernel_va(address)
}

pub fn span_page_count_shift(min_size: u64, page_shift: u32) -> u64 {
    if min_size == 0 {
        1
    } else {
        ((min_size - 1) >> page_shift) + 1
    }
}

pub fn format_bytes_per_pixel(pixel_format: u16) -> Option<u32> {
    // Match pixel_format storage set / resource-resolve iosurface_bpp.
    pixel_format::bytes_per_pixel(pixel_format)
}

/// Estimated byte reach of a texture of this geometry, for **sizing a mapping's
/// page table** — never for addressing pixels.
///
/// The guest's `sIOSurfaceDeviceDescriptor` is the only thing that knows a
/// surface's real base offset and pitch, and when it has not arrived yet the
/// mapper still has to decide how many pages to walk. This answers that and
/// only that, so it returns a byte count rather than a window: there is no
/// offset and no pitch here to bind, which is the point. A texture bind whose
/// descriptor has not resolved declines by name — see
/// [`sample_window_from_device_desc`].
///
/// `ROW_BYTES_ALIGN` makes the estimate the *tight* row rounded up rather than
/// the tight row itself, so the count errs long. It is not a derivation of what
/// IOSurface actually chose: a surface aligned more coarsely than this reaches
/// further, which is why `alloc_size` bounds the result wherever it is known
/// and why nothing may read a pitch back out of this.
pub fn packed_span_estimate(pixel_format: u16, width: u32, height: u32) -> Option<u64> {
    if width == 0 || height == 0 {
        return None;
    }
    let bpp = format_bytes_per_pixel(pixel_format)?;
    let tight = checked_mul_u64(width as u64, bpp as u64)?;
    let bpr = align_up_u64(tight, ROW_BYTES_ALIGN)?;
    if bpr > u32::MAX as u64 {
        return None;
    }
    checked_mul_u64(bpr, height as u64)
}

pub fn decode_device_surface(bytes: &[u8]) -> Option<DeviceSurfaceRecord> {
    if bytes.len() < DEVICE_DESC_LEN {
        return None;
    }
    let (width, height) = dims_extent(ld64(&bytes[DEVICE_DESC_DIMS..]));
    Some(DeviceSurfaceRecord {
        pixel_format: ld32(&bytes[DEVICE_DESC_PIXEL_FORMAT..]),
        base_offset: ld32(&bytes[DEVICE_DESC_BASE_OFFSET..]),
        alloc_size: ld32(&bytes[DEVICE_DESC_ALLOC_SIZE..]),
        width,
        height,
        bytes_per_row: ld32(&bytes[DEVICE_DESC_BPR..]),
        bytes_per_element: ld16(&bytes[DEVICE_DESC_BPE..]),
        plane_count: bytes[DEVICE_DESC_PLANE_COUNT],
    })
}

pub fn decode_device_plane(bytes: &[u8]) -> Option<DevicePlaneRecord> {
    if bytes.len() < DEVICE_PLANE_DESC_LEN {
        return None;
    }
    let (width, height) = dims_extent(ld64(&bytes[DEVICE_PLANE_DIMS..]));
    Some(DevicePlaneRecord {
        plane_offset: ld32(&bytes[DEVICE_PLANE_OFFSET..]),
        plane_base: ld32(&bytes[DEVICE_PLANE_BASE..]),
        plane_size: ld32(&bytes[DEVICE_PLANE_SIZE..]),
        width,
        height,
        bytes_per_row: ld32(&bytes[DEVICE_PLANE_BPR..]),
        bytes_per_element: ld16(&bytes[DEVICE_PLANE_BPE..]),
    })
}

pub fn device_desc_plane(desc: &[u8], plane_index: u32) -> Option<(DevicePlaneRecord, u32)> {
    if desc.len() < DEVICE_DESC_LEN {
        return None;
    }
    let plane_count = desc[DEVICE_DESC_PLANE_COUNT] as u32;
    // `TYPE4_PLANE_CAP` is `IOSurfaceGetPlaneCount`'s ceiling and lives beside
    // the plane record it bounds; the literal 8 that stood here was the same
    // rule written a second time, in a file that cannot see if the first moves.
    if plane_count == 0
        || plane_count > reims_vgpu_wire::device_desc::TYPE4_PLANE_CAP as u32
        || plane_index >= plane_count
    {
        return None;
    }
    let plane_off = DEVICE_DESC_PLANES + (plane_index as usize) * DEVICE_PLANE_DESC_LEN;
    if plane_off + DEVICE_PLANE_DESC_LEN > desc.len() {
        return None;
    }
    let plane = decode_device_plane(&desc[plane_off..plane_off + DEVICE_PLANE_DESC_LEN])?;
    Some((plane, plane_count))
}

pub fn sample_window_from_device_plane(
    plane: &DevicePlaneRecord,
    pixel_format: u16,
    width: u32,
    height: u32,
) -> Option<(u64, u32, u64)> {
    if width == 0 || height == 0 {
        return None;
    }
    let bpp = format_bytes_per_pixel(pixel_format)?;
    let tight = checked_mul_u64(width as u64, bpp as u64)?;
    if plane.bytes_per_row == 0 || (plane.bytes_per_row as u64) < tight {
        return None;
    }
    let mut surface_off = plane.plane_offset as u64;
    if surface_off == 0 {
        surface_off = plane.plane_base as u64;
    }
    let plane_bytes = checked_mul_u64(plane.bytes_per_row as u64, (height - 1) as u64)?;
    let plane_bytes = checked_add_u64(plane_bytes, tight)?;
    let span_end = checked_add_u64(surface_off, plane_bytes)?;
    if plane.plane_size != 0 && (plane.plane_size as u64) < plane_bytes {
        return None;
    }
    if (plane.width != 0 && plane.width < width) || (plane.height != 0 && plane.height < height) {
        return None;
    }
    Some((surface_off, plane.bytes_per_row, span_end))
}

pub fn sample_window_from_device_surface(
    surf: &DeviceSurfaceRecord,
    pixel_format: u16,
    width: u32,
    height: u32,
) -> Option<(u64, u32, u64)> {
    if width == 0 || height == 0 {
        return None;
    }
    let bpp = format_bytes_per_pixel(pixel_format)?;
    let tight = checked_mul_u64(width as u64, bpp as u64)?;
    if surf.bytes_per_row == 0 || (surf.bytes_per_row as u64) < tight {
        return None;
    }
    if (surf.width != 0 && surf.width < width) || (surf.height != 0 && surf.height < height) {
        return None;
    }
    let rows_span = checked_mul_u64(surf.bytes_per_row as u64, (height - 1) as u64)?;
    let rows_span = checked_add_u64(rows_span, tight)?;
    let span_end = checked_add_u64(surf.base_offset as u64, rows_span)?;
    if surf.alloc_size != 0 && span_end > surf.alloc_size as u64 {
        return None;
    }
    Some((surf.base_offset as u64, surf.bytes_per_row, span_end))
}

/// The window a texture of this geometry occupies inside its IOSurface, taken
/// from the guest's own device descriptor and from nowhere else.
///
/// Returns `(surface_offset, bytes_per_row, span_end)`, or `None` when the
/// descriptor is absent, too short, or names no plane this texture can be. That
/// `None` is the whole contract: the base offset and pitch of an IOSurface are
/// facts the guest owns, and a device that supplies its own when they are
/// missing has bound the wrong bytes with nothing able to notice. Callers
/// decline by name instead — a lost bind is visible, wrong pixels are not.
///
/// Three ways a window is derived, in order:
///
/// - **A wire-carried plane index** (type-5 record `+0x20`) names its plane
///   record directly. It is the only key that separates same-geometry planes:
///   a v0a8 surface's Y plane 0 and alpha plane 2 are both R8 at the luma
///   geometry, so the scan below matches two and takes neither.
/// - **A single-plane surface** uses the surface-level base and pitch.
/// - **A multi-plane surface with no wire index** (type-11) matches width,
///   height and bytes-per-element, and takes the plane only when *exactly one*
///   matches.
pub fn sample_window_from_device_desc(
    desc: Option<&[u8]>,
    plane_index: Option<u32>,
    pixel_format: u16,
    width: u32,
    height: u32,
) -> Option<(u64, u32, u64)> {
    if let Some(desc) = desc {
        if desc.len() >= DEVICE_DESC_LEN {
            if let Some(surf) = decode_device_surface(desc) {
                if let Some(p) = plane_index {
                    if surf.plane_count > 0 {
                        if let Some((cand, _)) = device_desc_plane(desc, p) {
                            if let Some(w) =
                                sample_window_from_device_plane(&cand, pixel_format, width, height)
                            {
                                return Some(w);
                            }
                        }
                    }
                }
                if surf.plane_count == 0 {
                    if let Some(w) =
                        sample_window_from_device_surface(&surf, pixel_format, width, height)
                    {
                        return Some(w);
                    }
                } else if let Some(bpp) = format_bytes_per_pixel(pixel_format) {
                    let mut matches = 0u32;
                    let mut plane = DevicePlaneRecord::default();
                    // The plane record's own cap, not a repeat of its value.
                    // `device_desc_plane` refuses any index at or above it, so a
                    // literal here that drifted from it would either walk indices
                    // that can only miss or stop short of planes the descriptor
                    // holds.
                    for p in 0..surf
                        .plane_count
                        .min(reims_vgpu_wire::device_desc::TYPE4_PLANE_CAP as u8)
                    {
                        if let Some((cand, _)) = device_desc_plane(desc, p as u32) {
                            if cand.width == width
                                && cand.height == height
                                && (cand.bytes_per_element == 0
                                    || cand.bytes_per_element as u32 == bpp)
                            {
                                matches += 1;
                                plane = cand;
                            }
                        }
                    }
                    if matches == 1 {
                        if let Some(w) =
                            sample_window_from_device_plane(&plane, pixel_format, width, height)
                        {
                            return Some(w);
                        }
                    }
                }
            }
        }
    }
    None
}

/// How many bytes of a mapping a texture of this geometry can reach, for sizing
/// its page table.
///
/// The descriptor answers this exactly when it has arrived; otherwise
/// [`packed_span_estimate`] gives a count that errs long. Either way the guest's
/// own allocation bounds it: RE (`allocateBackingHandle`) writes the
/// page-aligned allocation length at type-4 desc `+0`, independent of the plane
/// width/height/pitch filled from the per-plane getters, and we stash it as
/// `device_desc.alloc_size`. A span past that is a span past the pages the guest
/// allocated, so it is refused rather than clamped.
///
/// This is the only place the allocation bound is applied, and it is why nothing
/// may reach for [`packed_span_estimate`] directly and then take the span this
/// refused.
pub fn mapping_span_bound(
    desc: Option<&[u8]>,
    pixel_format: u16,
    width: u32,
    height: u32,
) -> Option<u64> {
    if let Some((_, _, end)) =
        sample_window_from_device_desc(desc, None, pixel_format, width, height)
    {
        return Some(end);
    }
    let end = packed_span_estimate(pixel_format, width, height)?;
    if let Some(desc) = desc {
        if desc.len() >= DEVICE_DESC_LEN {
            if let Some(surf) = decode_device_surface(desc) {
                if surf.alloc_size != 0 && end > surf.alloc_size as u64 {
                    return None;
                }
            }
        }
    }
    Some(end)
}

pub fn entry_gpa_shift(entry: u32, page_shift: u32) -> Option<u64> {
    if (entry & PAGE_ENTRY_VALID) == 0 {
        return None;
    }
    Some(((entry >> PAGE_ENTRY_PFN_SHIFT) as u64) << page_shift)
}

pub fn mapper_request_entry_offset(index: u32) -> u64 {
    (index as u64) * MAPPER_REQUEST_ENTRY_LEN as u64
}

pub fn mapper_request_published_entry_offset(producer: u32) -> Option<u64> {
    if producer == 0 {
        None
    } else {
        Some(mapper_request_entry_offset(producer - 1))
    }
}

pub fn required_entry_count(
    fields: &MapperInternalFields,
    min_size: u64,
    page_shift: u32,
) -> Result<u32, Status> {
    let pages64 = fields.raw_page_count;
    let required_pages = span_page_count_shift(min_size, page_shift);
    // Guest page count is authoritative — no product 4096-page ceiling.
    // Fail only on zero, span coverage, or host-unaddressable entry vectors
    // (process addressability for `Vec<u32>` of entries — not a MiB budget).
    if pages64 == 0 || pages64 < required_pages || pages64 > u32::MAX as u64 {
        return Err(Status::ErrPageCount("iosurface_page_count_invalid"));
    }
    let entry_bytes = pages64.saturating_mul(4);
    if usize::try_from(entry_bytes)
        .ok()
        .filter(|&n| n <= isize::MAX as usize)
        .is_none()
    {
        return Err(Status::ErrPageCount(
            "iosurface_page_count_host_addressability",
        ));
    }
    Ok(pages64 as u32)
}

pub fn decode_texture_descriptor(bytes: &[u8]) -> Result<TextureDescriptor, Status> {
    if bytes.len() < TYPE11_DESC_MIN_LEN {
        return Err(Status::ErrShortDescriptor(
            "iosurface_texture_descriptor_short",
        ));
    }
    Ok(TextureDescriptor {
        mapping_id64: ld64(&bytes[TYPE11_DESC_MAPPING_ID..]),
        mapping_id: ld32(&bytes[TYPE11_DESC_MAPPING_ID..]),
        object_ref: ld32(&bytes[TYPE11_DESC_OBJECT_REF..]),
        pixel_format: ld16(&bytes[TYPE11_DESC_PIXEL_FORMAT..]),
        width: ld32(&bytes[TYPE11_DESC_WIDTH..]),
        height: ld32(&bytes[TYPE11_DESC_HEIGHT..]),
    })
}

pub fn decode_mapper_request_entry(bytes: &[u8]) -> Result<MapperRequestEntry, Status> {
    if bytes.len() < MAPPER_REQUEST_ENTRY_LEN {
        return Err(Status::ErrShortDescriptor("iosurface_mapper_request_short"));
    }
    Ok(MapperRequestEntry {
        request_type: ld32(&bytes[MAPPER_REQUEST_TYPE..]),
        mapping_id: ld32(&bytes[MAPPER_REQUEST_MAPPING_ID..]),
        reserved: ld64(&bytes[MAPPER_REQUEST_RESERVED..]),
    })
}

fn read_u32(mem: &dyn PagesMemory, address: u64) -> Option<u32> {
    let mut bytes = [0u8; 4];
    if address > u64::MAX - 3 {
        return None;
    }
    if !mem.read(address, &mut bytes) {
        return None;
    }
    Some(ld32(&bytes))
}

fn read_u64(mem: &dyn PagesMemory, address: u64) -> Option<u64> {
    let mut bytes = [0u8; 8];
    if address > u64::MAX - 7 {
        return None;
    }
    if !mem.read(address, &mut bytes) {
        return None;
    }
    Some(ld64(&bytes))
}

fn read_u32_at(mem: &dyn PagesMemory, base: u64, offset: u64) -> Option<u32> {
    let address = checked_add_u64(base, offset)?;
    read_u32(mem, address)
}

fn read_u64_at(mem: &dyn PagesMemory, base: u64, offset: u64) -> Option<u64> {
    let address = checked_add_u64(base, offset)?;
    read_u64(mem, address)
}

pub fn read_mapper_identity(
    mem: &dyn PagesMemory,
    internal_kva: u64,
    has_mapper_device: bool,
    mapper_device_kva: u64,
) -> Result<MapperInternalFields, Status> {
    if !mem.is_kernel_va(internal_kva) {
        return Err(Status::ErrNotKernelVa(
            "iosurface_mapper_internal_kva_invalid",
        ));
    }
    if has_mapper_device && !mem.is_kernel_va(mapper_device_kva) {
        return Err(Status::ErrNotKernelVa(
            "iosurface_mapper_device_kva_invalid",
        ));
    }
    let owner_kva = read_u64_at(mem, internal_kva, MAPPING_INTERNAL_BACKPTR).ok_or(
        Status::ErrInternalRead("iosurface_mapper_internal_owner_read"),
    )?;
    let mapping_id = read_u32_at(mem, internal_kva, MAPPING_INTERNAL_ID).ok_or(
        Status::ErrInternalRead("iosurface_mapper_internal_mapping_id_read"),
    )?;
    let internal_size = read_u32_at(mem, internal_kva, MAPPING_INTERNAL_SIZE).ok_or(
        Status::ErrInternalRead("iosurface_mapper_internal_size_read"),
    )?;
    Ok(MapperInternalFields {
        internal_kva,
        has_mapper_device,
        mapper_device_kva,
        owner_kva,
        mapping_id,
        internal_size,
        page_field_48: 0,
        page_field_50: 0,
        raw_page_count: 0,
    })
}

pub fn read_mapper_internal(
    mem: &dyn PagesMemory,
    internal_kva: u64,
    has_mapper_device: bool,
    mapper_device_kva: u64,
) -> Result<MapperInternalFields, Status> {
    let mut fields = read_mapper_identity(mem, internal_kva, has_mapper_device, mapper_device_kva)?;
    fields.page_field_48 = read_u64_at(mem, internal_kva, MAPPING_INTERNAL_PAGE_FIELD_48).ok_or(
        Status::ErrInternalRead("iosurface_mapper_page_field_48_read"),
    )?;
    fields.page_field_50 = read_u64_at(mem, internal_kva, MAPPING_INTERNAL_PAGE_FIELD_50).ok_or(
        Status::ErrInternalRead("iosurface_mapper_page_field_50_read"),
    )?;
    fields.raw_page_count = read_u64_at(mem, internal_kva, MAPPING_INTERNAL_PAGE_COUNT)
        .ok_or(Status::ErrInternalRead("iosurface_mapper_page_count_read"))?;
    Ok(fields)
}

pub fn read_internal_desc_ptr(mem: &dyn PagesMemory, internal_kva: u64) -> Result<u64, Status> {
    let desc_kva = read_u64_at(mem, internal_kva, MAPPING_INTERNAL_DESC_PTR).ok_or(
        Status::ErrInternalRead("iosurface_mapper_device_desc_pointer_read"),
    )?;
    if desc_kva == 0 {
        return Err(Status::ErrInternalFields(
            "iosurface_mapper_device_desc_pointer_zero",
        ));
    }
    if !mem.is_kernel_va(desc_kva) {
        return Err(Status::ErrInternalFields(
            "iosurface_mapper_device_desc_pointer_invalid",
        ));
    }
    Ok(desc_kva)
}

pub fn validate_mapper_internal(
    mem: &dyn PagesMemory,
    expected_mapping_id: u32,
    fields: &MapperInternalFields,
) -> Status {
    if !mem.is_kernel_va(fields.internal_kva) {
        return Status::ErrNotKernelVa("iosurface_validate_internal_kva_invalid");
    }
    if fields.mapping_id != expected_mapping_id {
        return Status::ErrInternalMappingId("iosurface_validate_mapping_id_mismatch");
    }
    if fields.internal_size != MAPPING_INTERNAL_EXPECTED_SIZE {
        return Status::ErrInternalSize("iosurface_validate_internal_size_mismatch");
    }
    if fields.has_mapper_device {
        if !mem.is_kernel_va(fields.mapper_device_kva) {
            return Status::ErrNotKernelVa("iosurface_validate_mapper_device_kva_invalid");
        }
        if fields.owner_kva != fields.mapper_device_kva {
            return Status::ErrInternalOwner("iosurface_validate_internal_owner_mismatch");
        }
    }
    Status::Ok
}

fn read_table_entries(
    mem: &dyn PagesMemory,
    table_kva: u64,
    pages: u32,
    page_shift: u32,
) -> Result<Vec<u32>, Status> {
    let mut entries = Vec::with_capacity(pages as usize);
    for i in 0..pages {
        let entry = read_u32_at(mem, table_kva, (i as u64) * U32_SIZE as u64)
            .ok_or(Status::ErrPageTableRead("iosurface_page_table_entry_read"))?;
        let gpa = entry_gpa_shift(entry, page_shift)
            .ok_or(Status::ErrPageEntry("iosurface_page_table_entry_invalid"))?;
        if !mem.is_ram_gpa(gpa) {
            return Err(Status::ErrPageEntry("iosurface_page_table_gpa_not_ram"));
        }
        entries.push(entry);
    }
    Ok(entries)
}

pub fn build_table_plan(
    mem: &dyn PagesMemory,
    expected_mapping_id: u32,
    fields: &MapperInternalFields,
    min_size: u64,
    page_shift: u32,
) -> Result<PageTablePlan, Status> {
    let st = validate_mapper_internal(mem, expected_mapping_id, fields);
    if st != Status::Ok {
        return Err(st);
    }
    let field_48_populated = mem.is_kernel_va(fields.page_field_48);
    let field_50_populated = mem.is_kernel_va(fields.page_field_50);
    if !field_48_populated && !field_50_populated {
        return Err(Status::ErrInternalFields(
            "iosurface_page_table_fields_invalid",
        ));
    }
    let required_pages = span_page_count_shift(min_size, page_shift);
    let pages = required_entry_count(fields, min_size, page_shift)?;

    // One chase, `+0x48` then `+0xb8`.
    //
    // There used to be a second, `+0x50` then `+0x28`, with the entries of
    // whichever parsed first being returned — a "try both, keep the one that
    // works" ladder, which this project refuses on principle but could not
    // refuse here without evidence, because the alternative reading was that
    // the two fields are two layouts handled side by side.
    //
    // Two driven arm64 boots settled it, over **223 successful resolves** on
    // deliberately different workloads (Safari plus window drags; Finder view
    // switching, System Settings panes, Mission Control and window resizes).
    // Both fields held a kernel VA on every one of them, so the branch really
    // did choose rather than dispatch — and `+0x48` won every one, with the
    // second chase never once carrying a resolve that the first had failed.
    // A branch that chooses, and always chooses the same way, is a fallback.
    //
    // What replaces the fallback is loudness. Every way the `+0x48` chase can
    // fail now reaches `mapper_resolve_fail` under its own slug instead of
    // being silently rescued by a rail nothing has confirmed, and
    // [`CandidateOutcome::other_field_populated`] keeps measuring the premise.
    // The field test above stays: a mapping with only `+0x50` set is still
    // *detected*, and refused by name rather than resolved through the
    // unconfirmed path.
    if !field_48_populated {
        return Err(Status::ErrNoPageTable("iosurface_page_table_only_field_50"));
    }
    let table_kva = match read_u64_at(mem, fields.page_field_48, MAPPING_PAGE_TABLE_FROM_F48) {
        Some(v) if mem.is_kernel_va(v) => v,
        Some(_) => {
            return Err(Status::ErrNoPageTable(
                "iosurface_page_table_pointer_48_invalid",
            ))
        }
        None => {
            return Err(Status::ErrPageTableRead(
                "iosurface_page_table_pointer_48_read",
            ))
        }
    };
    let entries = read_table_entries(mem, table_kva, pages, page_shift)?;
    Ok(PageTablePlan {
        entries,
        page_table_kva: table_kva,
        min_size,
        required_pages,
        candidates: CandidateOutcome {
            other_field_populated: field_50_populated,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gva::PAGE_SHIFT_ARM64E;
    use crate::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    /// The arm64e page size as the `u64` this module's plan arithmetic takes.
    ///
    /// Derived from the one shift rather than restated, exactly as the device's
    /// own `model` copy is: `gva`'s `PAGE_SIZE_ARM64E` is the `u32` a
    /// page-offset mask wants, and widening it here is cheaper than two
    /// constants that can disagree.
    const PAGE_SIZE_ARM64E: u64 = 1u64 << PAGE_SHIFT_ARM64E;
    use reims_vgpu_observe::Refusal;

    use std::collections::HashMap;

    struct MapMem {
        map: HashMap<u64, u8>,
    }
    impl MapMem {
        fn new() -> Self {
            Self {
                map: HashMap::new(),
            }
        }
        fn put_u32(&mut self, a: u64, v: u32) {
            for (i, b) in v.to_le_bytes().iter().enumerate() {
                self.map.insert(a + i as u64, *b);
            }
        }
        fn put_u64(&mut self, a: u64, v: u64) {
            for (i, b) in v.to_le_bytes().iter().enumerate() {
                self.map.insert(a + i as u64, *b);
            }
        }
    }
    impl PagesMemory for MapMem {
        fn read(&self, address: u64, dst: &mut [u8]) -> bool {
            for (i, s) in dst.iter_mut().enumerate() {
                match self.map.get(&(address + i as u64)) {
                    Some(b) => *s = *b,
                    None => return false,
                }
            }
            true
        }
        fn is_kernel_va(&self, address: u64) -> bool {
            arm_kernel_va(address)
        }
    }

    #[test]
    fn status_refusal_separates_control_flow_from_exact_failures() {
        assert_eq!(Status::Ok.refusal(), None);
        assert!(
            reims_vgpu_observe::Emit::refusal("mapper_resolve_fail", &Status::Ok).is_none(),
            "success must not be representable as a failure line"
        );

        let texture = decode_texture_descriptor(&[]).unwrap_err();
        let request = decode_mapper_request_entry(&[]).unwrap_err();
        assert_eq!(
            texture.refusal(),
            Some("iosurface_texture_descriptor_short")
        );
        assert_eq!(request.refusal(), Some("iosurface_mapper_request_short"));
        assert_ne!(
            texture.refusal(),
            request.refusal(),
            "two distinct short-record checks must not collapse to one reason"
        );
        assert_eq!(
            reims_vgpu_observe::Emit::refusal("mapper_resolve_fail", &texture)
                .unwrap()
                .field("mapping", 7)
                .render(),
            "mapper_resolve_fail reason=iosurface_texture_descriptor_short \
             class=short_descriptor mapping=7"
        );
    }

    /// An unreadable `+0x48` pointer is refused by its own name, however good
    /// the other field looks.
    ///
    /// This case used to be the interesting one for a different reason: with
    /// two candidates the question was which failure to *attribute* the refusal
    /// to, and the answer was "the candidate actually walked". With one
    /// candidate there is nothing to outrank, and the case becomes the alarm
    /// instead. A well-formed table sits behind `+0x50` here and this device
    /// deliberately does not go and get it, so if a driven arm64 boot ever
    /// shows this slug the deletion of that chase is what to reconsider.
    #[test]
    fn an_unreadable_chased_pointer_is_refused_by_its_own_name() {
        let internal = ARM_KERNEL_VA_BASE + 0x10_000;
        let field_48 = ARM_KERNEL_VA_BASE + 0x20_000;
        let field_50 = ARM_KERNEL_VA_BASE + 0x30_000;
        let table = ARM_KERNEL_VA_BASE + 0x40_000;
        let mut mem = MapMem::new();

        mem.put_u64(field_50 + MAPPING_PAGE_TABLE_FROM_F50, table);
        mem.put_u32(table, 0);
        let fields = MapperInternalFields {
            internal_kva: internal,
            mapping_id: 3,
            internal_size: MAPPING_INTERNAL_EXPECTED_SIZE,
            page_field_48: field_48,
            page_field_50: field_50,
            raw_page_count: 1,
            ..MapperInternalFields::default()
        };

        let error =
            build_table_plan(&mem, 3, &fields, PAGE_SIZE_ARM64E, PAGE_SHIFT_ARM64E).unwrap_err();
        assert_eq!(
            error.refusal(),
            Some("iosurface_page_table_pointer_48_read")
        );
    }

    /// The page table comes through `+0x48` or it does not come at all.
    ///
    /// The `+0x50` chase that used to stand behind it was retired on 223
    /// successful resolves across two driven arm64 workloads, on which both
    /// fields were always populated and `+0x48` always won. The two cases that
    /// used to be rescued by it are now refusals **by name**, which is the
    /// whole trade: a rail nothing has confirmed no longer answers silently,
    /// and if either refusal ever appears in a driven boot's log it says the
    /// deletion was wrong and names which reading was right.
    #[test]
    fn the_page_table_comes_through_field_48_or_is_refused_by_name() {
        let internal = ARM_KERNEL_VA_BASE + 0x10_000;
        let field_48 = ARM_KERNEL_VA_BASE + 0x20_000;
        let field_50 = ARM_KERNEL_VA_BASE + 0x30_000;
        let table_a = ARM_KERNEL_VA_BASE + 0x40_000;
        let table_b = ARM_KERNEL_VA_BASE + 0x50_000;
        let good_entry = 1u32; // frame 1, which `entry_gpa_shift` accepts
        let base = |page_field_48, page_field_50| MapperInternalFields {
            internal_kva: internal,
            mapping_id: 3,
            internal_size: MAPPING_INTERNAL_EXPECTED_SIZE,
            page_field_48,
            page_field_50,
            raw_page_count: 1,
            ..MapperInternalFields::default()
        };

        // Only `+0x48` populated: a plan, and the census says the other field
        // was empty. On the two measured workloads this never happened.
        let mut mem = MapMem::new();
        mem.put_u64(field_48 + MAPPING_PAGE_TABLE_FROM_F48, table_a);
        mem.put_u32(table_a, good_entry);
        let plan = build_table_plan(
            &mem,
            3,
            &base(field_48, 0),
            PAGE_SIZE_ARM64E,
            PAGE_SHIFT_ARM64E,
        )
        .expect("the chased field alone is a plan");
        assert_eq!(plan.page_table_kva, table_a);
        assert!(!plan.candidates.other_field_populated);

        // Both populated and both parseable: the plan comes through `+0x48`,
        // and `table_b` is never read. This is the shape all 223 measured
        // resolves had.
        let mut mem = MapMem::new();
        mem.put_u64(field_48 + MAPPING_PAGE_TABLE_FROM_F48, table_a);
        mem.put_u64(field_50 + MAPPING_PAGE_TABLE_FROM_F50, table_b);
        mem.put_u32(table_a, good_entry);
        mem.put_u32(table_b, good_entry);
        let plan = build_table_plan(
            &mem,
            3,
            &base(field_48, field_50),
            PAGE_SIZE_ARM64E,
            PAGE_SHIFT_ARM64E,
        )
        .expect("both good is a plan");
        assert_eq!(plan.page_table_kva, table_a, "the chase is `+0x48`");
        assert!(plan.candidates.other_field_populated);

        // Only `+0x50` populated. The field test still sees it — this is not
        // `iosurface_page_table_fields_invalid` — but the chase that used to
        // answer it is gone, so it is refused under a name that says exactly
        // which reading of the two fields it would take to make that wrong.
        let mut mem = MapMem::new();
        mem.put_u64(field_50 + MAPPING_PAGE_TABLE_FROM_F50, table_b);
        mem.put_u32(table_b, good_entry);
        let error = build_table_plan(
            &mem,
            3,
            &base(0, field_50),
            PAGE_SIZE_ARM64E,
            PAGE_SHIFT_ARM64E,
        )
        .expect_err("the deleted chase does not answer this");
        assert_eq!(error.refusal(), Some("iosurface_page_table_only_field_50"));

        // Both populated, `+0x48`'s table unparseable. This is the one shape
        // the fallback was ever load-bearing for, and `earlier_failed` read
        // zero over both driven boots — so it is now a refusal carrying the
        // reason the table failed, rather than a silent rescue.
        let mut mem = MapMem::new();
        mem.put_u64(field_48 + MAPPING_PAGE_TABLE_FROM_F48, table_a);
        mem.put_u64(field_50 + MAPPING_PAGE_TABLE_FROM_F50, table_b);
        mem.put_u32(table_a, 0); // a zero entry is refused
        mem.put_u32(table_b, good_entry);
        let error = build_table_plan(
            &mem,
            3,
            &base(field_48, field_50),
            PAGE_SIZE_ARM64E,
            PAGE_SHIFT_ARM64E,
        )
        .expect_err("no second candidate rescues this any more");
        assert_eq!(
            error.refusal(),
            Some("iosurface_page_table_entry_invalid"),
            "the refusal names why the chased table failed, not that a \
             fallback was missing"
        );
    }

    #[test]
    fn packed_span_estimate_rounds_the_row_up() {
        // 200 BGRA = 800 tight, rounded to 896; the estimate is that row times
        // the height, and it is a byte count with no offset or pitch to bind.
        assert_eq!(
            packed_span_estimate(MTL_FORMAT_BGRA8_UNORM, 200, 100),
            Some(896 * 100)
        );
    }

    #[test]
    fn texture_desc_and_geometry() {
        let mut bytes = [0u8; 0x20];
        bytes[0] = 3; // mapping id
                      // format BGRA at 0x16
        bytes[0x16] = 0x50;
        // width 64 height 32
        bytes[0x18] = 64;
        bytes[0x1c] = 32;
        let d = decode_texture_descriptor(&bytes).unwrap();
        assert_eq!(d.mapping_id, 3);
        assert_eq!(d.pixel_format, MTL_FORMAT_BGRA8_UNORM);
        assert_eq!(d.width, 64);
        assert_eq!(d.height, 32);
    }

    #[test]
    fn entry_gpa_and_span() {
        assert!(entry_gpa_shift(0, PAGE_SHIFT_ARM64E).is_none());
        let e = (5 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        assert_eq!(
            entry_gpa_shift(e, PAGE_SHIFT_ARM64E).unwrap(),
            (5u64) << PAGE_SHIFT_ARM64E
        );
        assert_eq!(span_page_count_shift(0, PAGE_SHIFT_ARM64E), 1);
        assert_eq!(span_page_count_shift(1, PAGE_SHIFT_ARM64E), 1);
        assert_eq!(
            span_page_count_shift(PAGE_SIZE_ARM64E + 1, PAGE_SHIFT_ARM64E),
            2
        );
    }

    #[test]
    fn kernel_va_and_identity() {
        assert!(arm_kernel_va(ARM_KERNEL_VA_BASE + 0x1000));
        assert!(!arm_kernel_va(0x1000));
        assert!(x86_kernel_va(X86_KERNEL_VA_MIN + 0x1000));
        assert!(!x86_kernel_va(0x1000));
        assert!(guest_kernel_va(ARM_KERNEL_VA_BASE + 1));
        assert!(guest_kernel_va(X86_KERNEL_VA_MIN + 1));
        let mut m = MapMem::new();
        let kva = ARM_KERNEL_VA_BASE + 0x10000;
        m.put_u64(kva + MAPPING_INTERNAL_BACKPTR, kva);
        m.put_u32(kva + MAPPING_INTERNAL_ID, 1);
        m.put_u32(kva + MAPPING_INTERNAL_SIZE, MAPPING_INTERNAL_EXPECTED_SIZE);
        let f = read_mapper_identity(&m, kva, false, 0).unwrap();
        assert_eq!(f.mapping_id, 1);
        assert_eq!(validate_mapper_internal(&m, 1, &f), Status::Ok);
    }

    #[test]
    fn property_fuzz_packed_span_estimate() {
        for w in [1u32, 2, 64, 200, 1920] {
            for h in [1u32, 2, 100] {
                if let Some(end) = packed_span_estimate(MTL_FORMAT_BGRA8_UNORM, w, h) {
                    // Errs long: never below the tight extent it stands in for.
                    assert!(end >= w as u64 * 4 * h as u64);
                    assert_eq!(end % 128, 0);
                }
            }
        }
    }

    /// A texture bind takes its window from the guest's descriptor or not at
    /// all. With no descriptor there is nothing to derive a base offset or a
    /// pitch from, and supplying one would be a wrong bind that reads as success
    /// at every layer above — so this declines, and only the page-sizing
    /// estimate answers.
    #[test]
    fn a_bind_without_a_descriptor_declines_where_page_sizing_still_answers() {
        assert!(
            sample_window_from_device_desc(None, None, MTL_FORMAT_BGRA8_UNORM, 200, 100).is_none()
        );
        assert!(
            sample_window_from_device_desc(Some(&[0u8; 4]), None, MTL_FORMAT_BGRA8_UNORM, 200, 100)
                .is_none(),
            "a descriptor shorter than a full record is no descriptor"
        );
        assert_eq!(
            mapping_span_bound(None, MTL_FORMAT_BGRA8_UNORM, 200, 100),
            Some(896 * 100)
        );
    }

    /// The inverse of [`dims_extent`], for building a record to decode.
    ///
    /// Written from the same three constants, so the round trip is over one
    /// declaration of the layout rather than two that could agree while both
    /// being wrong. What pins the layout to the wire is
    /// [`the_dims_word_puts_width_at_bit_eight_and_height_at_bit_forty`], which
    /// uses a literal.
    fn pack_plane_dims(width: u32, height: u32) -> u64 {
        ((width as u64 & DIMS_EXTENT_MASK) << DIMS_WIDTH_SHIFT)
            | ((height as u64 & DIMS_EXTENT_MASK) << DIMS_HEIGHT_SHIFT)
    }

    /// The `dims` layout, against a word written out by hand.
    ///
    /// A round trip through [`pack_plane_dims`] cannot see a shift that moved,
    /// because it moves with it; this literal cannot. 1280 is `0x500` at byte 1
    /// and 720 is `0x2d0` at byte 5, which is the whole claim — and the guard
    /// bytes are the other half of it, since the element sizes share the word
    /// and a mask one bit too wide would read one of them into an extent.
    #[test]
    fn the_dims_word_puts_width_at_bit_eight_and_height_at_bit_forty() {
        assert_eq!(dims_extent(0x0002_d000_0005_0000), (1280, 720));
        assert_eq!(pack_plane_dims(1280, 720), 0x0002_d000_0005_0000);
        // Element width at byte 0 and element height at byte 4, both set to
        // every bit they have; neither may reach an extent.
        assert_eq!(dims_extent(0x0002_d0ff_0005_00ff), (1280, 720));
        // An extent that fills its 24 bits does not spill into the byte above.
        assert_eq!(dims_extent(0xffff_ff00_ffff_ff00), (0xff_ffff, 0xff_ffff));
    }

    /// The page-sizing estimate must not reach past the wire `alloc_size`
    /// (type-4 `length`). RE: allocateBackingHandle writes length@0
    /// independently of plane dims.
    #[test]
    fn mapping_span_bound_rejects_a_span_past_alloc_size() {
        use crate::endian::{st32, st64};
        use crate::pixel_format::MTL_FORMAT_BGRA8_UNORM;

        // Device desc: 1024×1024 dims, alloc only 384*4096 = 0x180000.
        let mut desc = vec![0u8; DEVICE_DESC_LEN];
        st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], 0x18_0000);
        st32(
            &mut desc[DEVICE_DESC_PIXEL_FORMAT..],
            MTL_FORMAT_BGRA8_UNORM as u32,
        );
        let dims = pack_plane_dims(1024, 1024);
        st64(&mut desc[DEVICE_DESC_DIMS..], dims);
        // bpr too small for 1024 BGRA → the device-surface path rejects, so the
        // estimate is what answers and the allocation is what bounds it.
        st32(&mut desc[DEVICE_DESC_BPR..], 64);
        desc[DEVICE_DESC_PLANE_COUNT] = 0;

        // 1024*4096 > alloc → None (fail closed, no height lie).
        assert!(mapping_span_bound(Some(&desc), MTL_FORMAT_BGRA8_UNORM, 1024, 1024).is_none());
        // Within alloc: the estimate stands.
        assert_eq!(
            mapping_span_bound(Some(&desc), MTL_FORMAT_BGRA8_UNORM, 1024, 384),
            Some(384 * 4096)
        );
        // A descriptor this texture cannot be placed against sizes pages but
        // never sources a bind.
        assert!(sample_window_from_device_desc(
            Some(&desc),
            None,
            MTL_FORMAT_BGRA8_UNORM,
            1024,
            384
        )
        .is_none());
    }

    #[test]
    fn the_geometry_scan_picks_a_plane_only_when_exactly_one_matches() {
        use crate::endian::{st16, st32, st64};
        use crate::pixel_format::{MTL_FORMAT_R8_UNORM, MTL_FORMAT_RG8_UNORM};

        let mut desc = vec![0u8; DEVICE_DESC_LEN];
        st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], 0x20000);
        desc[DEVICE_DESC_PLANE_COUNT] = 2;
        // Plane 0: Y 16×8 R8 bpr=64 offset=512 size=512
        let p0 = DEVICE_DESC_PLANES;
        st32(&mut desc[p0 + DEVICE_PLANE_OFFSET..], 512);
        st32(&mut desc[p0 + DEVICE_PLANE_SIZE..], 512);
        st64(&mut desc[p0 + DEVICE_PLANE_DIMS..], pack_plane_dims(16, 8));
        st32(&mut desc[p0 + DEVICE_PLANE_BPR..], 64);
        st16(&mut desc[p0 + DEVICE_PLANE_BPE..], 1);
        // Plane 1: UV 8×4 RG8 bpr=64 offset=1024 size=256
        let p1 = DEVICE_DESC_PLANES + DEVICE_PLANE_DESC_LEN;
        st32(&mut desc[p1 + DEVICE_PLANE_OFFSET..], 1024);
        st32(&mut desc[p1 + DEVICE_PLANE_SIZE..], 256);
        st64(&mut desc[p1 + DEVICE_PLANE_DIMS..], pack_plane_dims(8, 4));
        st32(&mut desc[p1 + DEVICE_PLANE_BPR..], 64);
        st16(&mut desc[p1 + DEVICE_PLANE_BPE..], 2);

        let (off_y, bpr_y, end_y) =
            sample_window_from_device_desc(Some(&desc), None, MTL_FORMAT_R8_UNORM, 16, 8).unwrap();
        assert_eq!(off_y, 512);
        assert_eq!(bpr_y, 64);
        // exclusive last-row end: 512 + 7*64 + 16
        assert_eq!(end_y, 512 + 7 * 64 + 16);

        let (off_uv, bpr_uv, end_uv) =
            sample_window_from_device_desc(Some(&desc), None, MTL_FORMAT_RG8_UNORM, 8, 4).unwrap();
        assert_eq!(off_uv, 1024);
        assert_eq!(bpr_uv, 64);
        assert_eq!(end_uv, 1024 + 3 * 64 + 16);

        // Dims that hit no plane record: zero matches, so nothing is bound.
        assert!(
            sample_window_from_device_desc(Some(&desc), None, MTL_FORMAT_R8_UNORM, 4, 4).is_none()
        );
    }

    /// v0a8 (biplanar video + alpha) shape from the live apple.com hero: the
    /// Y and alpha planes share geometry and bpe, so the geometry scan is
    /// ambiguous by construction — only an explicit wire plane index (type-5
    /// record `+0x20`) separates them.
    #[test]
    fn sample_window_plane_index_selects_among_same_geometry_planes() {
        use crate::endian::{st16, st32, st64};
        use crate::pixel_format::{MTL_FORMAT_R8_UNORM, MTL_FORMAT_RG8_UNORM};

        // Live shape (scaled): Y 946×350 @32 bpr 960; UV 473×175 @336032
        // bpr 960 bpe 2; alpha 946×350 @504992 bpr 960 bpe 1.
        let mut desc = vec![0u8; DEVICE_DESC_LEN];
        st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], 843_776);
        desc[DEVICE_DESC_PLANE_COUNT] = 3;
        let planes = [
            (32u32, 336_000u32, 946u32, 350u32, 960u32, 1u16),
            (336_032, 168_000, 473, 175, 960, 2),
            (504_992, 336_000, 946, 350, 960, 1),
        ];
        for (i, (off, size, w, h, bpr, bpe)) in planes.iter().enumerate() {
            let base = DEVICE_DESC_PLANES + i * DEVICE_PLANE_DESC_LEN;
            st32(&mut desc[base + DEVICE_PLANE_OFFSET..], *off);
            st32(&mut desc[base + DEVICE_PLANE_SIZE..], *size);
            st64(
                &mut desc[base + DEVICE_PLANE_DIMS..],
                pack_plane_dims(*w, *h),
            );
            st32(&mut desc[base + DEVICE_PLANE_BPR..], *bpr);
            st16(&mut desc[base + DEVICE_PLANE_BPE..], *bpe);
        }

        // Indexed selection: each plane record by its wire index.
        let y = sample_window_from_device_desc(Some(&desc), Some(0), MTL_FORMAT_R8_UNORM, 946, 350)
            .unwrap();
        assert_eq!((y.0, y.1), (32, 960));
        let uv =
            sample_window_from_device_desc(Some(&desc), Some(1), MTL_FORMAT_RG8_UNORM, 473, 175)
                .unwrap();
        assert_eq!((uv.0, uv.1), (336_032, 960));
        let a = sample_window_from_device_desc(Some(&desc), Some(2), MTL_FORMAT_R8_UNORM, 946, 350)
            .unwrap();
        assert_eq!((a.0, a.1), (504_992, 960));

        // No index: Y geometry matches plane 0 AND plane 2. Two matches is not
        // "pick the first" — the scan cannot tell them apart, so it declines and
        // the caller reports a lost bind rather than sampling luma for alpha.
        assert!(
            sample_window_from_device_desc(Some(&desc), None, MTL_FORMAT_R8_UNORM, 946, 350)
                .is_none()
        );

        // An index past the plane count names no record, so it resolves nothing
        // — not the geometry scan's answer, and not plane 0's bytes.
        assert!(sample_window_from_device_desc(
            Some(&desc),
            Some(7),
            MTL_FORMAT_R8_UNORM,
            946,
            350
        )
        .is_none());
    }
}
