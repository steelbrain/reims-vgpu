//! Pixel-format helpers matching ObjC `reims_vgpu_storage_image_format` / `reims_vgpu_mtl_pixel_format_bpp`.

use crate::backend::metal::abi::*;
use metal::MTLPixelFormat;

pub fn storage_image_format(format: u32) -> Option<(MTLPixelFormat, usize)> {
    match format {
        REIMS_VGPU_SIMG_RGBA8_UINT => Some((MTLPixelFormat::RGBA8Uint, 4)),
        REIMS_VGPU_SIMG_RGBA8_SINT => Some((MTLPixelFormat::RGBA8Sint, 4)),
        REIMS_VGPU_SIMG_RGBA16_UINT => Some((MTLPixelFormat::RGBA16Uint, 8)),
        REIMS_VGPU_SIMG_RGBA16_FLOAT => Some((MTLPixelFormat::RGBA16Float, 8)),
        REIMS_VGPU_SIMG_RGBA32_FLOAT => Some((MTLPixelFormat::RGBA32Float, 16)),
        REIMS_VGPU_SIMG_RGBA8_UNORM => Some((MTLPixelFormat::RGBA8Unorm, 4)),
        REIMS_VGPU_SIMG_BGRA8_UNORM => Some((MTLPixelFormat::BGRA8Unorm, 4)),
        REIMS_VGPU_SIMG_R16_FLOAT => Some((MTLPixelFormat::R16Float, 2)),
        REIMS_VGPU_SIMG_RG16_FLOAT => Some((MTLPixelFormat::RG16Float, 4)),
        REIMS_VGPU_SIMG_R8_UNORM => Some((MTLPixelFormat::R8Unorm, 1)),
        REIMS_VGPU_SIMG_RG8_UNORM => Some((MTLPixelFormat::RG8Unorm, 2)),
        REIMS_VGPU_SIMG_RGBA32_UINT => Some((MTLPixelFormat::RGBA32Uint, 16)),
        _ => None,
    }
}

/// Bytes per texel for a raw `MTLPixelFormat` value.
///
/// Asks [`crate::contract::pixel_format::bytes_per_pixel`] rather than carrying
/// a table. The two are the same numbering: the contract's wire codes *are*
/// Apple's `MTLPixelFormat` values, so `MTL_FORMAT_BGRA8_UNORM` is `0x50` and
/// `MTLPixelFormat::BGRA8Unorm` is 80.
///
/// This rail used to keep its own switch covering thirteen formats against the
/// contract's twenty-four, agreeing with it on every one they shared. Nothing
/// held the two in step, and the gap was not inert: a depth, stencil or 32-bit
/// single-channel target that the contract sizes correctly returned `None` here
/// and was refused for no reason visible at the call site.
///
/// The `u16` narrowing is the format-code domain, not a truncation: every
/// `MTLPixelFormat` fits, so a wider value names no format and declines.
pub fn mtl_pixel_format_bpp(pixel_format: u32) -> Option<usize> {
    let code = u16::try_from(pixel_format).ok()?;
    crate::contract::pixel_format::bytes_per_pixel(code).map(|bpp| bpp as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_image_formats_report_their_metal_format_and_texel_size() {
        let cases = [
            (REIMS_VGPU_SIMG_RGBA8_UINT, MTLPixelFormat::RGBA8Uint, 4),
            (REIMS_VGPU_SIMG_RGBA8_SINT, MTLPixelFormat::RGBA8Sint, 4),
            (REIMS_VGPU_SIMG_RGBA16_UINT, MTLPixelFormat::RGBA16Uint, 8),
            (REIMS_VGPU_SIMG_RGBA16_FLOAT, MTLPixelFormat::RGBA16Float, 8),
            (
                REIMS_VGPU_SIMG_RGBA32_FLOAT,
                MTLPixelFormat::RGBA32Float,
                16,
            ),
            (REIMS_VGPU_SIMG_RGBA8_UNORM, MTLPixelFormat::RGBA8Unorm, 4),
            (REIMS_VGPU_SIMG_BGRA8_UNORM, MTLPixelFormat::BGRA8Unorm, 4),
            (REIMS_VGPU_SIMG_R16_FLOAT, MTLPixelFormat::R16Float, 2),
            (REIMS_VGPU_SIMG_RG16_FLOAT, MTLPixelFormat::RG16Float, 4),
            (REIMS_VGPU_SIMG_R8_UNORM, MTLPixelFormat::R8Unorm, 1),
            (REIMS_VGPU_SIMG_RG8_UNORM, MTLPixelFormat::RG8Unorm, 2),
            (REIMS_VGPU_SIMG_RGBA32_UINT, MTLPixelFormat::RGBA32Uint, 16),
        ];
        for (wire, metal, bytes) in cases {
            let (actual, actual_bytes) = storage_image_format(wire).expect("mapped format");
            assert_eq!(actual as u64, metal as u64);
            assert_eq!(actual_bytes, bytes);
        }
        assert_eq!(storage_image_format(u32::MAX), None);
        assert_eq!(mtl_pixel_format_bpp(u32::MAX), None);
    }

    #[test]
    fn render_pixel_formats_report_their_byte_widths() {
        let cases = [
            (MTLPixelFormat::A8Unorm, 1),
            (MTLPixelFormat::R8Unorm, 1),
            (MTLPixelFormat::R16Float, 2),
            (MTLPixelFormat::RG8Unorm, 2),
            (MTLPixelFormat::RGBA8Unorm, 4),
            (MTLPixelFormat::RGBA8Unorm_sRGB, 4),
            (MTLPixelFormat::BGRA8Unorm, 4),
            (MTLPixelFormat::BGRA8Unorm_sRGB, 4),
            (MTLPixelFormat::RG16Float, 4),
            (MTLPixelFormat::RGBA16Float, 8),
            (MTLPixelFormat::RGBA16Uint, 8),
            (MTLPixelFormat::RGBA32Float, 16),
            (MTLPixelFormat::RGBA32Uint, 16),
        ];
        for (format, bytes) in cases {
            assert_eq!(mtl_pixel_format_bpp(format as u32), Some(bytes));
        }
    }

    /// The formats this rail's own table did not carry.
    ///
    /// Each one is sized by the contract and was refused here, which is the gap
    /// a second hand-maintained table opens: the widths were never in dispute,
    /// only which of the two tables a caller happened to reach.
    #[test]
    fn formats_the_local_table_used_to_miss_are_sized_by_the_contract() {
        let cases = [
            (MTLPixelFormat::Stencil8, 1),
            (MTLPixelFormat::Depth16Unorm, 2),
            (MTLPixelFormat::R32Uint, 4),
            (MTLPixelFormat::R32Sint, 4),
            (MTLPixelFormat::R32Float, 4),
            (MTLPixelFormat::RGB9E5Float, 4),
            (MTLPixelFormat::Depth32Float, 4),
            (MTLPixelFormat::RGBA8Uint, 4),
            (MTLPixelFormat::RGBA8Sint, 4),
        ];
        for (format, bytes) in cases {
            assert_eq!(
                mtl_pixel_format_bpp(format as u32),
                Some(bytes),
                "{format:?} is sized by the contract and used to be refused here"
            );
        }
    }

    /// The ABI value survives the conversion, and a code naming no format is
    /// refused rather than reinterpreted.
    ///
    /// This used to assert only the first half, because the conversion was a
    /// `transmute` and the second half had no answer — an undeclared ordinal
    /// was undefined behaviour, not a value the test could ask about.
    #[test]
    fn pixel_format_conversion_preserves_the_abi_value_and_refuses_the_rest() {
        use crate::backend::metal::mtl_enum;
        let raw = MTLPixelFormat::BGRA8Unorm_sRGB as u32;
        assert_eq!(
            mtl_enum::pixel_format(raw).map(|f| f as u64),
            Some(raw as u64)
        );
        // 556 is one past `BGR10_XR_SRGB`, the highest format `metal` declares.
        assert_eq!(mtl_enum::pixel_format(556), None);
        assert_eq!(mtl_enum::pixel_format(u32::MAX), None);
    }
}
