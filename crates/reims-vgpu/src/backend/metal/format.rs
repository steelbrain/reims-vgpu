//! Pixel-format helpers matching ObjC `reims_vgpu_storage_image_format` / `reims_vgpu_mtl_pixel_format_bpp`.

use crate::contract::pixel_format::StorageImageSelector;
use metal::MTLPixelFormat;

/// The Metal pixel format and texel width for a contract [`StorageImageSelector`].
///
/// **Total, and it has to be.** This used to match the selector's `u32` ordinal
/// against a list of `REIMS_VGPU_SIMG_*` constants that were hand-copied from the
/// enum's discriminants, returning `None` for anything absent — and it had
/// already drifted: `StorageImageSelector::R32Uint` existed in the contract with
/// no constant and no arm here, so every `R32Uint` storage bind on the whole
/// arm64 pathway refused as `metal_selector_unsupported` while the Vulkan
/// pathway ran it. Nothing could see the gap, because a `u32` match has no
/// coverage for a compiler to check.
///
/// Taking the enum makes the arms exhaustive, so the drift is now a build
/// failure on the Metal arm — which the cross-compiled clippy run reaches from a
/// Linux host, and is the only tool here that does. The `REIMS_VGPU_SIMG_*`
/// constants are gone rather than derived: they were a second spelling of the
/// discriminants with no other reader, and a second spelling that can still be
/// assembled eventually is.
pub fn storage_image_format(selector: StorageImageSelector) -> (MTLPixelFormat, usize) {
    use StorageImageSelector as S;
    match selector {
        S::Rgba8Uint => (MTLPixelFormat::RGBA8Uint, 4),
        S::Rgba8Sint => (MTLPixelFormat::RGBA8Sint, 4),
        S::Rgba16Uint => (MTLPixelFormat::RGBA16Uint, 8),
        S::Rgba16Float => (MTLPixelFormat::RGBA16Float, 8),
        S::Rgba32Float => (MTLPixelFormat::RGBA32Float, 16),
        S::Rgba8Unorm => (MTLPixelFormat::RGBA8Unorm, 4),
        S::Bgra8Unorm => (MTLPixelFormat::BGRA8Unorm, 4),
        S::R16Float => (MTLPixelFormat::R16Float, 2),
        S::Rg16Float => (MTLPixelFormat::RG16Float, 4),
        S::R8Unorm => (MTLPixelFormat::R8Unorm, 1),
        S::Rg8Unorm => (MTLPixelFormat::RG8Unorm, 2),
        S::Rgba32Uint => (MTLPixelFormat::RGBA32Uint, 16),
        S::R32Uint => (MTLPixelFormat::R32Uint, 4),
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

    /// Every selector's Metal format and texel width, written out rather than
    /// derived from the table under test.
    ///
    /// The widths are checked against [`crate::contract::pixel_format`] rather
    /// than only against this list, because the width is the number a staging
    /// buffer is sized by and two tables disagreeing about it is a short read.
    #[test]
    fn storage_image_formats_report_their_metal_format_and_texel_size() {
        use crate::contract::pixel_format as pf;
        use StorageImageSelector as S;

        let cases = [
            (S::Rgba8Uint, MTLPixelFormat::RGBA8Uint, 4, pf::MTL_FORMAT_RGBA8_UINT),
            (S::Rgba8Sint, MTLPixelFormat::RGBA8Sint, 4, pf::MTL_FORMAT_RGBA8_SINT),
            (S::Rgba16Uint, MTLPixelFormat::RGBA16Uint, 8, pf::MTL_FORMAT_RGBA16_UINT),
            (S::Rgba16Float, MTLPixelFormat::RGBA16Float, 8, pf::MTL_FORMAT_RGBA16_FLOAT),
            (S::Rgba32Float, MTLPixelFormat::RGBA32Float, 16, pf::MTL_FORMAT_RGBA32_FLOAT),
            (S::Rgba8Unorm, MTLPixelFormat::RGBA8Unorm, 4, pf::MTL_FORMAT_RGBA8_UNORM),
            (S::Bgra8Unorm, MTLPixelFormat::BGRA8Unorm, 4, pf::MTL_FORMAT_BGRA8_UNORM),
            (S::R16Float, MTLPixelFormat::R16Float, 2, pf::MTL_FORMAT_R16_FLOAT),
            (S::Rg16Float, MTLPixelFormat::RG16Float, 4, pf::MTL_FORMAT_RG16_FLOAT),
            (S::R8Unorm, MTLPixelFormat::R8Unorm, 1, pf::MTL_FORMAT_R8_UNORM),
            (S::Rg8Unorm, MTLPixelFormat::RG8Unorm, 2, pf::MTL_FORMAT_RG8_UNORM),
            (S::Rgba32Uint, MTLPixelFormat::RGBA32Uint, 16, pf::MTL_FORMAT_RGBA32_UINT),
            // Present in the contract with no arm here until 2026-08-10, which
            // cost the arm64 pathway every `R32Uint` storage bind.
            (S::R32Uint, MTLPixelFormat::R32Uint, 4, pf::MTL_FORMAT_R32_UINT),
        ];
        for (selector, metal, bytes, mtl) in cases {
            let (actual, actual_bytes) = storage_image_format(selector);
            assert_eq!(actual as u64, metal as u64);
            assert_eq!(actual_bytes, bytes);
            assert_eq!(
                pf::bytes_per_pixel(mtl),
                Some(bytes as u32),
                "{selector:?} and the pixel table disagree on the texel width"
            );
            assert_eq!(
                pf::storage_selector(mtl),
                Some(selector),
                "{mtl:#x} does not select {selector:?}"
            );
        }
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
