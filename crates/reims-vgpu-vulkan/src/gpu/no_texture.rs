//! The formats Metal makes no view in, from a texture of a format Metal has ([`no_view`]), and
//! those it views a texture's format in with no `PixelFormatView` usage ([`views_without_usage`]),
//! which the reims-graphics device's surface caches take.
//!
//! Measured on Metal on an M3 Max (macOS 26.7, without API validation, one case per process): a
//! view of a texture of any format Metal makes one of, made by any of the view selectors (with or
//! without a resource index, and with any type, levels, slices or swizzle, the source's or not),
//! in a format [`NO_VIEW`] lists is nil, before the view's swizzle, type, levels and slices are
//! checked (a swizzle channel of 9, type 10 or 99, and levels whose end passes `u64::MAX` among
//! them): `MTLPixelFormatUnspecialized` (263), `MTLPixelFormatRGBA16Float_XR` (556),
//! `MTLPixelFormatAnyDepth` (651) and `MTLPixelFormatAnyStencil` (652), and every two-plane YCbCr
//! or RGB format Metal has (those of `_2P` in Metal's names, 500 to 620) but `RGB10A8_2P` and its
//! three siblings (624 to 627), whose views are made from some sources. A view in a one-plane
//! YCbCr format (`_1P`, as 501) or a plane format (as 576, `MTLPixelFormatR10Unorm_X6`) aborts
//! from every source measured.

use metal2vulkan::metal::PixelFormat;

/// The formats a view of a texture of a format the runtime has is nil in, by Metal's raw values,
/// as inclusive ranges: `MTLPixelFormatUnspecialized`, the two-plane YCbCr and RGB formats from
/// `MTLPixelFormatYCBCR8_420_2P` (500) to `MTLPixelFormatYCBCR12_444_2P_PACKED_HLG` (620) that
/// Metal on this host has, `RGBA16Float_XR`, `AnyDepth` and `AnyStencil` (measured: the module's
/// docs).
pub const NO_VIEW: &[(u64, u64)] = &[
    (263, 263),
    (500, 500),
    (502, 503),
    (505, 510),
    (520, 520),
    (522, 523),
    (525, 530),
    (540, 548),
    (556, 556),
    (564, 575),
    (580, 585),
    (591, 602),
    (608, 613),
    (615, 620),
    (651, 652),
];

/// Whether a view in `raw` of a texture of a format the runtime has is nil ([`NO_VIEW`]).
pub fn no_view(raw: u64) -> bool {
    NO_VIEW
        .iter()
        .any(|&(first, last)| (first..=last).contains(&raw))
}

/// Whether Metal views a texture of the raw format `raw` in the raw format `view` with no
/// `PixelFormatView` usage: the same format, its sRGB sibling, or the stencil format of a combined
/// depth and stencil one (`metal2vulkan::metal::format::views_without_usage`). False for a value
/// the runtime has no format of.
pub fn views_without_usage(raw: u64, view: u64) -> bool {
    let format = |raw: u64| {
        u32::try_from(raw)
            .ok()
            .and_then(|r| PixelFormat::try_from(r).ok())
    };
    match (format(raw), format(view)) {
        (Some(pf), Some(view)) => metal2vulkan::metal::format::views_without_usage(pf, view),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    /// A texture is viewed with no `PixelFormatView` usage in its own format and its sRGB
    /// sibling, and in no other format of its texel's size; a raw value of no format in neither.
    #[test]
    fn views_without_usage_are_the_format_and_its_srgb_sibling() {
        use super::views_without_usage;
        // BGRA8Unorm, BGRA8Unorm_sRGB, RGBA8Unorm, R32Uint.
        assert!(views_without_usage(80, 80));
        assert!(views_without_usage(80, 81));
        assert!(views_without_usage(81, 80));
        assert!(!views_without_usage(80, 70));
        assert!(!views_without_usage(80, 53));
        assert!(!views_without_usage(80, 0x7fff_ffff));
    }
}
