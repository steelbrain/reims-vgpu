//! Guest `MTLPixelFormat` vocabulary and storage widths.
//!
//! Raw API ordinals are decoded here. Conversion code may live in a core or
//! backend crate, but those layers must consume these names instead of owning a
//! second pixel-format table.

pub const COMPONENT_COUNT: usize = 4;
pub const COMPONENT_R: usize = 0;
pub const COMPONENT_G: usize = 1;
pub const COMPONENT_B: usize = 2;
pub const COMPONENT_A: usize = 3;

pub const R8_BPP: u32 = 1;
pub const RG8_BPP: u32 = 2;
pub const RGBA8_BPP: u32 = 4;
pub const BGRA8_BPP: u32 = RGBA8_BPP;
pub const R16F_BPP: u32 = 2;
pub const R16_BPP: u32 = 2;
pub const RG16_BPP: u32 = 4;
pub const R32F_BPP: u32 = 4;
pub const RG16F_BPP: u32 = 4;
pub const RGBA16_BPP: u32 = 8;
pub const RGBA16F_BPP: u32 = RGBA16_BPP;
pub const RGBA32_BPP: u32 = 16;
pub const RGBA32F_BPP: u32 = RGBA32_BPP;
pub const R32_BPP: u32 = 4;

pub const MTL_FORMAT_A8_UNORM: u16 = 0x01;
pub const MTL_FORMAT_R8_UNORM: u16 = 0x0a;
pub const MTL_FORMAT_R8_UINT: u16 = 0x0d;
pub const MTL_FORMAT_R16_UNORM: u16 = 0x14;
pub const MTL_FORMAT_R16_FLOAT: u16 = 0x19;
pub const MTL_FORMAT_RG8_UNORM: u16 = 0x1e;
pub const MTL_FORMAT_RG8_UINT: u16 = 0x21;
pub const MTL_FORMAT_R32_UINT: u16 = 0x35;
pub const MTL_FORMAT_R32_SINT: u16 = 0x36;
pub const MTL_FORMAT_R32_FLOAT: u16 = 0x37;
pub const MTL_FORMAT_RG16_UNORM: u16 = 0x3c;
pub const MTL_FORMAT_RG16_FLOAT: u16 = 0x41;
pub const MTL_FORMAT_RGBA8_UNORM: u16 = 0x46;
pub const MTL_FORMAT_RGBA8_UNORM_SRGB: u16 = 0x47;
pub const MTL_FORMAT_RGBA8_UINT: u16 = 0x49;
pub const MTL_FORMAT_RGBA8_SINT: u16 = 0x4a;
pub const MTL_FORMAT_BGRA8_UNORM: u16 = 0x50;
pub const MTL_FORMAT_BGRA8_UNORM_SRGB: u16 = 0x51;
pub const MTL_FORMAT_RGB10A2_UNORM: u16 = 0x5a;
pub const MTL_FORMAT_RGB10A2_UINT: u16 = 0x5b;
pub const MTL_FORMAT_RG11B10_FLOAT: u16 = 0x5c;
pub const MTL_FORMAT_RGB9E5_FLOAT: u16 = 0x5d;
pub const MTL_FORMAT_BGR10A2_UNORM: u16 = 0x5e;
pub const MTL_FORMAT_RGBA16_UNORM: u16 = 0x6e;
pub const MTL_FORMAT_RGBA16_UINT: u16 = 0x71;
pub const MTL_FORMAT_RGBA16_FLOAT: u16 = 0x73;
pub const MTL_FORMAT_RGBA32_UINT: u16 = 0x7b;
pub const MTL_FORMAT_RGBA32_FLOAT: u16 = 0x7d;
pub const MTL_FORMAT_DEPTH16_UNORM: u16 = 250;
pub const MTL_FORMAT_DEPTH32_FLOAT: u16 = 252;
pub const MTL_FORMAT_STENCIL8: u16 = 253;
pub const MTL_FORMAT_DEPTH24_UNORM_STENCIL8: u16 = 255;
pub const MTL_FORMAT_DEPTH32_FLOAT_STENCIL8: u16 = 260;
pub const MTL_FORMAT_X32_STENCIL8: u16 = 261;
pub const MTL_FORMAT_X24_STENCIL8: u16 = 262;

/// Bytes occupied by one full guest storage cell.
///
/// The `X*_Stencil8` values are aspect views of combined depth/stencil cells,
/// so this returns the parent cell width rather than the one-byte stencil plane.
pub const fn bytes_per_pixel(format: u16) -> Option<u32> {
    Some(match format {
        MTL_FORMAT_A8_UNORM | MTL_FORMAT_R8_UNORM | MTL_FORMAT_R8_UINT | MTL_FORMAT_STENCIL8 => {
            R8_BPP
        }
        MTL_FORMAT_R16_FLOAT
        | MTL_FORMAT_RG8_UNORM
        | MTL_FORMAT_RG8_UINT
        | MTL_FORMAT_DEPTH16_UNORM => RG8_BPP,
        MTL_FORMAT_R16_UNORM => R16_BPP,
        MTL_FORMAT_RG16_UNORM => RG16_BPP,
        MTL_FORMAT_RG16_FLOAT => RG16F_BPP,
        MTL_FORMAT_RGBA8_UNORM
        | MTL_FORMAT_RGBA8_UNORM_SRGB
        | MTL_FORMAT_RGBA8_UINT
        | MTL_FORMAT_RGBA8_SINT
        | MTL_FORMAT_BGRA8_UNORM
        | MTL_FORMAT_BGRA8_UNORM_SRGB
        | MTL_FORMAT_R32_UINT
        | MTL_FORMAT_R32_SINT
        | MTL_FORMAT_R32_FLOAT
        | MTL_FORMAT_RGB9E5_FLOAT
        | MTL_FORMAT_RGB10A2_UNORM
        | MTL_FORMAT_RGB10A2_UINT
        | MTL_FORMAT_RG11B10_FLOAT
        | MTL_FORMAT_BGR10A2_UNORM
        | MTL_FORMAT_DEPTH32_FLOAT
        | MTL_FORMAT_DEPTH24_UNORM_STENCIL8
        | MTL_FORMAT_X24_STENCIL8 => RGBA8_BPP,
        MTL_FORMAT_DEPTH32_FLOAT_STENCIL8 | MTL_FORMAT_X32_STENCIL8 => 8,
        MTL_FORMAT_RGBA16_UNORM | MTL_FORMAT_RGBA16_UINT | MTL_FORMAT_RGBA16_FLOAT => RGBA16_BPP,
        MTL_FORMAT_RGBA32_UINT | MTL_FORMAT_RGBA32_FLOAT => RGBA32_BPP,
        _ => return None,
    })
}

pub const fn has_depth_aspect(format: u16) -> bool {
    matches!(
        format,
        MTL_FORMAT_DEPTH16_UNORM
            | MTL_FORMAT_DEPTH32_FLOAT
            | MTL_FORMAT_DEPTH24_UNORM_STENCIL8
            | MTL_FORMAT_DEPTH32_FLOAT_STENCIL8
    )
}

pub const fn has_stencil_aspect(format: u16) -> bool {
    matches!(
        format,
        MTL_FORMAT_STENCIL8
            | MTL_FORMAT_DEPTH24_UNORM_STENCIL8
            | MTL_FORMAT_DEPTH32_FLOAT_STENCIL8
            | MTL_FORMAT_X32_STENCIL8
            | MTL_FORMAT_X24_STENCIL8
    )
}

pub const fn is_srgb(format: u16) -> bool {
    matches!(
        format,
        MTL_FORMAT_RGBA8_UNORM_SRGB | MTL_FORMAT_BGRA8_UNORM_SRGB
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aspect_views_keep_their_parent_cell_width() {
        assert_eq!(bytes_per_pixel(MTL_FORMAT_X24_STENCIL8), Some(4));
        assert_eq!(bytes_per_pixel(MTL_FORMAT_X32_STENCIL8), Some(8));
        assert!(has_stencil_aspect(MTL_FORMAT_X24_STENCIL8));
        assert!(!has_depth_aspect(MTL_FORMAT_X24_STENCIL8));
    }
}
