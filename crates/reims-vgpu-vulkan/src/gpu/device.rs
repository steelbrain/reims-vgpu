//! What the rt platform's device takes of a Metal device's descriptors and surfaces: the resource
//! options a buffer's or a heap's descriptor holds ([`descriptor_options`]), and the facts of its
//! surface a texture over a surface's plane takes ([`plane_facts_taken`]).

use crate::core::platform::values::OptionSet;

/// A heap descriptor's `resourceOptions` before any setter: `Private` (measured).
pub(crate) const HEAP_DESCRIPTOR_OPTIONS: u64 = 0x20;

/// The resource options a descriptor holds after core's storage properties, in order, from
/// `initial`: a storage mode replaces bits 4 to 7 and a hazard tracking mode bits 8 and 9, as
/// their setters do (measured on a heap descriptor: storage 2 then tracking 2 holds 0x220). A
/// buffer descriptor's value sends no hazard tracking mode.
pub(crate) fn descriptor_options(initial: u64, options: &[OptionSet]) -> u64 {
    options.iter().fold(initial, |held, option| match *option {
        OptionSet::ResourceOptions(options) => options,
        OptionSet::ResourceOptionsOr(bits) => held | bits,
        OptionSet::StorageMode(mode) => (held & !0xf0) | ((mode & 0xf) << 4),
        OptionSet::HazardTrackingMode(mode) => (held & !0x300) | ((mode & 0x3) << 8),
    })
}

/// What a texture over a surface's plane reads of the surface besides the plane
/// ([`plane_facts_taken`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlaneFacts {
    /// The surface's YCbCr facts: its OSType, its size, its plane 1, and its YCbCr matrix.
    YCbCr,
    /// Its OSType, its size and its plane 1 alone.
    Planes,
}

/// The facts of its surface a texture of `pf` over a plane of it takes: the YCbCr facts for a
/// two-plane YCbCr format (`metal2vulkan::metal::format::ycbcr`), the planes'
/// for a two-plane XR10 format ([`two_plane_xr10`]), which Metal reads alike ('b3a8'), and none
/// for any other.
pub(crate) fn plane_facts_taken(pf: metal2vulkan::metal::PixelFormat) -> Option<PlaneFacts> {
    if metal2vulkan::metal::format::ycbcr(pf) {
        Some(PlaneFacts::YCbCr)
    } else if two_plane_xr10(pf) {
        Some(PlaneFacts::Planes)
    } else {
        None
    }
}

/// Whether `pf` is a two-plane XR10 format: `RGB10A8_2P_XR10` (550) or its sRGB twin (551),
/// which Metal makes over a surface's two planes, the color words in plane 0 and the alpha bytes
/// in plane 1.
pub(crate) fn two_plane_xr10(pf: metal2vulkan::metal::PixelFormat) -> bool {
    use metal2vulkan::metal::PixelFormat;
    matches!(
        pf,
        PixelFormat::RGB10A82PXR10 | PixelFormat::RGB10A82PXR10SRGB
    )
}
