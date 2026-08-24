//! `MTLSamplerDescriptor` state → Vulkan sampler state.
//!
//! Two Metal address modes have no exact Vulkan equivalent and both collapse
//! onto `CLAMP_TO_BORDER`; that collapse is stated once here with its reason
//! rather than re-derived wherever a sampler is built.

use ash::vk;

use super::reason::TranslateReason;
use crate::engine::{SamplerAddressMode, SamplerBorderColor, SamplerFilter, SamplerMipFilter};

/// `MTLSamplerMinMagFilter` (SDK numeric values).
pub fn filter(mtl: u32) -> Result<SamplerFilter, TranslateReason> {
    reims_vgpu_protocol::sampler_filter(mtl).map_err(|_| TranslateReason::UnknownSamplerFilter(mtl))
}

/// `MTLSamplerMipFilter` (SDK numeric values).
pub fn mip_filter(mtl: u32) -> Result<SamplerMipFilter, TranslateReason> {
    reims_vgpu_protocol::sampler_mip_filter(mtl)
        .map_err(|_| TranslateReason::UnknownSamplerMipFilter(mtl))
}

/// `MTLSamplerAddressMode` (SDK numeric values).
pub fn address_mode(mtl: u32) -> Result<SamplerAddressMode, TranslateReason> {
    reims_vgpu_protocol::sampler_address_mode(mtl)
        .map_err(|_| TranslateReason::UnknownSamplerAddressMode(mtl))
}

/// `MTLSamplerBorderColor` (SDK numeric values).
pub fn border_color(mtl: u32) -> Result<SamplerBorderColor, TranslateReason> {
    reims_vgpu_protocol::sampler_border_color(mtl)
        .map_err(|_| TranslateReason::UnknownSamplerBorderColor(mtl))
}

pub fn vk_filter(filter: SamplerFilter) -> vk::Filter {
    match filter {
        SamplerFilter::Nearest => vk::Filter::NEAREST,
        SamplerFilter::Linear => vk::Filter::LINEAR,
    }
}

/// Vulkan has no "this image has no mip chain" mip mode — a single-level image
/// simply never samples past level 0 — so `NotMipmapped` takes the cheaper
/// NEAREST mode rather than inventing a third state.
pub fn vk_mipmap_mode(filter: SamplerMipFilter) -> vk::SamplerMipmapMode {
    match filter {
        SamplerMipFilter::NotMipmapped | SamplerMipFilter::Nearest => {
            vk::SamplerMipmapMode::NEAREST
        }
        SamplerMipFilter::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}

/// `ClampToZero` and `ClampToBorderColor` both reach `CLAMP_TO_BORDER`: Vulkan
/// expresses "outside the image reads a fixed colour" only through the border
/// colour, and Metal's `ClampToZero` is that colour being transparent black.
/// The distinction survives in [`SamplerBorderColor`], which the sampler also
/// carries, so nothing is lost by the collapse.
pub fn vk_address_mode(mode: SamplerAddressMode) -> vk::SamplerAddressMode {
    match mode {
        SamplerAddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        SamplerAddressMode::MirrorClampToEdge => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
        SamplerAddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        SamplerAddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        SamplerAddressMode::ClampToZero | SamplerAddressMode::ClampToBorderColor => {
            vk::SamplerAddressMode::CLAMP_TO_BORDER
        }
    }
}

pub fn vk_border_color(color: SamplerBorderColor) -> vk::BorderColor {
    match color {
        SamplerBorderColor::TransparentBlack => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
        SamplerBorderColor::OpaqueBlack => vk::BorderColor::FLOAT_OPAQUE_BLACK,
        SamplerBorderColor::OpaqueWhite => vk::BorderColor::FLOAT_OPAQUE_WHITE,
    }
}

/// The border colour a sampler must bind, given its declared border colour and
/// whether any of its three axes uses `ClampToZero`.
///
/// This is the other half of the `ClampToZero` collapse. Metal's `ClampToZero`
/// means "outside the image reads transparent black" *regardless* of the
/// descriptor's `borderColor` field, so once an axis uses it the border colour
/// is forced — otherwise a sampler that declares opaque white and clamps to
/// zero would fringe white where Metal fringes nothing. Deciding this beside
/// the collapse that makes it necessary keeps the two from being reasoned about
/// separately.
pub fn vk_border_color_with_clamp_to_zero(
    color: SamplerBorderColor,
    address_uses_clamp_to_zero: bool,
) -> vk::BorderColor {
    if address_uses_clamp_to_zero {
        vk::BorderColor::FLOAT_TRANSPARENT_BLACK
    } else {
        vk_border_color(color)
    }
}

// ---------------------------------------------------------------------------
// Sampler state for passes the ENGINE originates
// ---------------------------------------------------------------------------
//
// These are not translations of anything the guest asked for — they are the
// engine's own fixed choices for work it initiates. They live here anyway for
// the reason the whole module exists: `PRESENT_BLIT_FILTER` was spelled
// identically at three separate blit sites, which is three chances for them to
// stop agreeing about how a scaled present is filtered.

/// Filter for the engine's own scaling blits on the present path.
///
/// LINEAR because a present that scales is resampling a finished image for
/// display, where point sampling shows as visible stair-stepping. A
/// non-scaling present hits the 1:1 path and never consults this.
pub const PRESENT_BLIT_FILTER: vk::Filter = vk::Filter::LINEAR;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_sampler_enum_is_total_over_its_sdk_range() {
        for mtl in 0..=1u32 {
            assert!(filter(mtl).is_ok());
        }
        assert_eq!(
            filter(2).unwrap_err(),
            TranslateReason::UnknownSamplerFilter(2)
        );
        for mtl in 0..=2u32 {
            assert!(mip_filter(mtl).is_ok());
            assert!(border_color(mtl).is_ok());
        }
        assert_eq!(
            mip_filter(3).unwrap_err(),
            TranslateReason::UnknownSamplerMipFilter(3)
        );
        assert_eq!(
            border_color(3).unwrap_err(),
            TranslateReason::UnknownSamplerBorderColor(3)
        );
        for mtl in 0..=5u32 {
            assert!(address_mode(mtl).is_ok());
        }
        assert_eq!(
            address_mode(6).unwrap_err(),
            TranslateReason::UnknownSamplerAddressMode(6)
        );
    }

    /// The two collapses are deliberate and are the only ones. Every other
    /// address mode and mip mode must stay distinct, or a repeat becomes a
    /// clamp and a texture tiles wrong.
    #[test]
    fn only_the_two_documented_collapses_exist() {
        let modes: Vec<_> = (0..=5).map(|m| address_mode(m).unwrap()).collect();
        let mut vks: Vec<i32> = modes.iter().map(|m| vk_address_mode(*m).as_raw()).collect();
        vks.sort_unstable();
        let before = vks.len();
        vks.dedup();
        assert_eq!(before - vks.len(), 1, "exactly one address-mode collapse");
        assert_eq!(
            vk_address_mode(SamplerAddressMode::ClampToZero),
            vk_address_mode(SamplerAddressMode::ClampToBorderColor)
        );
        assert_eq!(
            vk_mipmap_mode(SamplerMipFilter::NotMipmapped),
            vk_mipmap_mode(SamplerMipFilter::Nearest)
        );
        assert_ne!(
            vk_mipmap_mode(SamplerMipFilter::Nearest),
            vk_mipmap_mode(SamplerMipFilter::Linear)
        );
    }

    /// `ClampToZero`'s meaning survives the collapse through the border colour
    /// the sampler carries alongside it.
    #[test]
    fn clamp_to_zero_keeps_its_meaning_in_the_border_colour() {
        assert_eq!(
            vk_border_color(SamplerBorderColor::TransparentBlack),
            vk::BorderColor::FLOAT_TRANSPARENT_BLACK
        );
        let mut colors: Vec<i32> = (0..=2)
            .map(|m| vk_border_color(border_color(m).unwrap()).as_raw())
            .collect();
        colors.sort_unstable();
        let before = colors.len();
        colors.dedup();
        assert_eq!(before, colors.len());
    }
}
