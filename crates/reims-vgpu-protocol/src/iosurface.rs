//! Semantic IOSurface device-descriptor geometry.

use crate::{align_up_u64, checked_add_u64, checked_mul_u64};

pub const ROW_BYTES_ALIGN: u64 = 128;
pub const DEVICE_PLANE_DESC_LEN: usize = 0x40;
pub const DEVICE_PLANE_OFFSET: usize = 0x08;
pub const DEVICE_PLANE_BASE: usize = 0x0c;
pub const DEVICE_PLANE_SIZE: usize = 0x10;
pub const DEVICE_PLANE_DIMS: usize = 0x14;
pub const DEVICE_PLANE_BPR: usize = 0x1c;
pub const DEVICE_PLANE_BPE: usize = 0x20;

const DIMS_WIDTH_SHIFT: u32 = 8;
const DIMS_HEIGHT_SHIFT: u32 = 40;
const DIMS_EXTENT_MASK: u64 = 0x00ff_ffff;

pub const DEVICE_DESC_LEN: usize = 0x200;
pub const DEVICE_DESC_PIXEL_FORMAT: usize = 0x04;
pub const DEVICE_DESC_BASE_OFFSET: usize = 0x08;
pub const DEVICE_DESC_ALLOC_SIZE: usize = 0x10;
pub const DEVICE_DESC_DIMS: usize = 0x14;
pub const DEVICE_DESC_BPR: usize = 0x1c;
pub const DEVICE_DESC_BPE: usize = 0x20;
pub const DEVICE_DESC_PLANE_COUNT: usize = 0x24;
pub const DEVICE_DESC_PLANES: usize = 0x40;

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

pub fn dims_extent(dims: u64) -> (u32, u32) {
    (
        ((dims >> DIMS_WIDTH_SHIFT) & DIMS_EXTENT_MASK) as u32,
        ((dims >> DIMS_HEIGHT_SHIFT) & DIMS_EXTENT_MASK) as u32,
    )
}

pub fn format_bytes_per_pixel(pixel_format: u16) -> Option<u32> {
    crate::metal_pixel::bytes_per_pixel(pixel_format)
}

pub fn packed_span_estimate(pixel_format: u16, width: u32, height: u32) -> Option<u64> {
    if width == 0 || height == 0 {
        return None;
    }
    let bpp = format_bytes_per_pixel(pixel_format)?;
    let tight = checked_mul_u64(u64::from(width), u64::from(bpp))?;
    let bpr = align_up_u64(tight, ROW_BYTES_ALIGN)?;
    if bpr > u64::from(u32::MAX) {
        return None;
    }
    checked_mul_u64(bpr, u64::from(height))
}

pub fn decode_device_surface(bytes: &[u8]) -> Option<DeviceSurfaceRecord> {
    if bytes.len() < DEVICE_DESC_LEN {
        return None;
    }
    let (width, height) = dims_extent(u64::from_le_bytes(
        bytes[DEVICE_DESC_DIMS..][..8].try_into().ok()?,
    ));
    Some(DeviceSurfaceRecord {
        pixel_format: u32::from_le_bytes(bytes[DEVICE_DESC_PIXEL_FORMAT..][..4].try_into().ok()?),
        base_offset: u32::from_le_bytes(bytes[DEVICE_DESC_BASE_OFFSET..][..4].try_into().ok()?),
        alloc_size: u32::from_le_bytes(bytes[DEVICE_DESC_ALLOC_SIZE..][..4].try_into().ok()?),
        width,
        height,
        bytes_per_row: u32::from_le_bytes(bytes[DEVICE_DESC_BPR..][..4].try_into().ok()?),
        bytes_per_element: u16::from_le_bytes(bytes[DEVICE_DESC_BPE..][..2].try_into().ok()?),
        plane_count: bytes[DEVICE_DESC_PLANE_COUNT],
    })
}

pub fn decode_device_plane(bytes: &[u8]) -> Option<DevicePlaneRecord> {
    if bytes.len() < DEVICE_PLANE_DESC_LEN {
        return None;
    }
    let (width, height) = dims_extent(u64::from_le_bytes(
        bytes[DEVICE_PLANE_DIMS..][..8].try_into().ok()?,
    ));
    Some(DevicePlaneRecord {
        plane_offset: u32::from_le_bytes(bytes[DEVICE_PLANE_OFFSET..][..4].try_into().ok()?),
        plane_base: u32::from_le_bytes(bytes[DEVICE_PLANE_BASE..][..4].try_into().ok()?),
        plane_size: u32::from_le_bytes(bytes[DEVICE_PLANE_SIZE..][..4].try_into().ok()?),
        width,
        height,
        bytes_per_row: u32::from_le_bytes(bytes[DEVICE_PLANE_BPR..][..4].try_into().ok()?),
        bytes_per_element: u16::from_le_bytes(bytes[DEVICE_PLANE_BPE..][..2].try_into().ok()?),
    })
}

pub fn device_desc_plane(desc: &[u8], plane_index: u32) -> Option<(DevicePlaneRecord, u32)> {
    if desc.len() < DEVICE_DESC_LEN {
        return None;
    }
    let plane_count = u32::from(desc[DEVICE_DESC_PLANE_COUNT]);
    if plane_count == 0
        || plane_count > reims_vgpu_wire::device_desc::SURFACE_BACKING_PLANE_CAP as u32
        || plane_index >= plane_count
    {
        return None;
    }
    let plane_off = DEVICE_DESC_PLANES.checked_add(
        usize::try_from(plane_index)
            .ok()?
            .checked_mul(DEVICE_PLANE_DESC_LEN)?,
    )?;
    let plane = decode_device_plane(desc.get(plane_off..plane_off + DEVICE_PLANE_DESC_LEN)?)?;
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
    let tight = checked_mul_u64(
        u64::from(width),
        u64::from(format_bytes_per_pixel(pixel_format)?),
    )?;
    if plane.bytes_per_row == 0 || u64::from(plane.bytes_per_row) < tight {
        return None;
    }
    let surface_offset = if plane.plane_offset == 0 {
        u64::from(plane.plane_base)
    } else {
        u64::from(plane.plane_offset)
    };
    let plane_bytes = checked_add_u64(
        checked_mul_u64(u64::from(plane.bytes_per_row), u64::from(height - 1))?,
        tight,
    )?;
    if (plane.plane_size != 0 && u64::from(plane.plane_size) < plane_bytes)
        || (plane.width != 0 && plane.width < width)
        || (plane.height != 0 && plane.height < height)
    {
        return None;
    }
    Some((
        surface_offset,
        plane.bytes_per_row,
        checked_add_u64(surface_offset, plane_bytes)?,
    ))
}

pub fn sample_window_from_device_surface(
    surface: &DeviceSurfaceRecord,
    pixel_format: u16,
    width: u32,
    height: u32,
) -> Option<(u64, u32, u64)> {
    if width == 0 || height == 0 {
        return None;
    }
    let tight = checked_mul_u64(
        u64::from(width),
        u64::from(format_bytes_per_pixel(pixel_format)?),
    )?;
    if surface.bytes_per_row == 0
        || u64::from(surface.bytes_per_row) < tight
        || (surface.width != 0 && surface.width < width)
        || (surface.height != 0 && surface.height < height)
    {
        return None;
    }
    let rows_span = checked_add_u64(
        checked_mul_u64(u64::from(surface.bytes_per_row), u64::from(height - 1))?,
        tight,
    )?;
    let span_end = checked_add_u64(u64::from(surface.base_offset), rows_span)?;
    if surface.alloc_size != 0 && span_end > u64::from(surface.alloc_size) {
        return None;
    }
    Some((
        u64::from(surface.base_offset),
        surface.bytes_per_row,
        span_end,
    ))
}

pub fn sample_window_from_device_desc(
    desc: Option<&[u8]>,
    plane_index: Option<u32>,
    pixel_format: u16,
    width: u32,
    height: u32,
) -> Option<(u64, u32, u64)> {
    let desc = desc?;
    let surface = decode_device_surface(desc)?;
    if let Some(plane_index) = plane_index {
        if surface.plane_count > 0 {
            let (plane, _) = device_desc_plane(desc, plane_index)?;
            return sample_window_from_device_plane(&plane, pixel_format, width, height);
        }
        // A non-planar surface has exactly one view: plane zero. The explicit
        // plane is guest-declared view state, so a different value cannot be
        // discarded in favor of the whole-surface geometry.
        if plane_index != 0 {
            return None;
        }
    }
    if surface.plane_count == 0 {
        return sample_window_from_device_surface(&surface, pixel_format, width, height);
    }
    let bpp = format_bytes_per_pixel(pixel_format)?;
    let mut matched = None;
    for plane_index in 0..surface
        .plane_count
        .min(reims_vgpu_wire::device_desc::SURFACE_BACKING_PLANE_CAP as u8)
    {
        let (plane, _) = device_desc_plane(desc, u32::from(plane_index))?;
        if plane.width == width
            && plane.height == height
            && (plane.bytes_per_element == 0 || u32::from(plane.bytes_per_element) == bpp)
            && matched.replace(plane).is_some()
        {
            return None;
        }
    }
    sample_window_from_device_plane(&matched?, pixel_format, width, height)
}

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
    if let Some(surface) = desc.and_then(decode_device_surface) {
        if surface.alloc_size != 0 && end > u64::from(surface.alloc_size) {
            return None;
        }
    }
    Some(end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal_pixel::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_R8_UNORM};

    #[test]
    fn descriptor_geometry_is_owned_at_the_semantic_boundary() {
        assert_eq!(dims_extent(0x0002_d000_0005_0000), (1280, 720));
        assert_eq!(
            packed_span_estimate(MTL_FORMAT_BGRA8_UNORM, 200, 100),
            Some(896 * 100)
        );
        assert_eq!(format_bytes_per_pixel(MTL_FORMAT_R8_UNORM), Some(1));
        assert_eq!(decode_device_surface(&[0; DEVICE_DESC_LEN - 1]), None);
    }

    #[test]
    fn an_explicit_nonzero_plane_cannot_alias_a_non_planar_surface() {
        let mut desc = [0u8; DEVICE_DESC_LEN];
        desc[DEVICE_DESC_BASE_OFFSET..DEVICE_DESC_BASE_OFFSET + 4]
            .copy_from_slice(&64u32.to_le_bytes());
        desc[DEVICE_DESC_ALLOC_SIZE..DEVICE_DESC_ALLOC_SIZE + 4]
            .copy_from_slice(&4096u32.to_le_bytes());
        let dims = (4u64 << DIMS_WIDTH_SHIFT) | (4u64 << DIMS_HEIGHT_SHIFT);
        desc[DEVICE_DESC_DIMS..DEVICE_DESC_DIMS + 8].copy_from_slice(&dims.to_le_bytes());
        desc[DEVICE_DESC_BPR..DEVICE_DESC_BPR + 4].copy_from_slice(&16u32.to_le_bytes());

        assert_eq!(
            sample_window_from_device_desc(Some(&desc), Some(0), MTL_FORMAT_BGRA8_UNORM, 4, 4),
            Some((64, 16, 128))
        );
        assert_eq!(
            sample_window_from_device_desc(Some(&desc), Some(1), MTL_FORMAT_BGRA8_UNORM, 4, 4),
            None
        );
    }
}
