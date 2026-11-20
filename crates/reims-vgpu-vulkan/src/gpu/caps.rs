//! The stand-ins of the rt platform's answers (`crate::rt_platform::answers`): Metal's answers on
//! an Apple M3 Max, where no documented fact and no Vulkan query gives one and the guest or the
//! host sees it. What the guest then does with an answer that rt refuses dies at its selector,
//! never with a faked result.

use crate::core::platform::caps::FAMILIES;

/// The stand-ins: Metal's answers on an Apple M3 Max, where no documented fact gives one (the
/// module docs). The registry ID, which Metal answers at run time, is the OS's
/// ([`crate::os::default_registry_id`]).
pub mod stand_in {
    /// `supportsFamily:` of each of core's `FAMILIES`: Apple5 to Apple9 yes, Apple10 and -1009
    /// no.
    pub const FAMILIES: [bool; 7] = [true, true, true, true, true, false, false];
    /// `supportsExtendedXR10Formats`.
    pub const EXTENDED_XR10_FORMATS: bool = true;
    /// `supportsTileShaders` and `supportsNonSquareTileShaders` (private).
    pub const TILE_SHADERS: bool = true;
    /// `supportsImageBlocks` (private).
    pub const IMAGE_BLOCKS: bool = true;
    /// `isRGB10A2GammaSupported`.
    pub const RGB10A2_GAMMA: bool = true;
    /// `supportsFloat16BCubicFiltering`: undocumented, unmeasured.
    pub const FLOAT16_BCUBIC_FILTERING: bool = true;
    /// `supportsTexture2DMultisampleArray`: undocumented.
    pub const TEXTURE_2D_MULTISAMPLE_ARRAY: bool = true;
    /// `supportsPrimitiveType:` of the raw values 5 to 9, no public `MTLPrimitiveType`.
    pub const PRIVATE_PRIMITIVE_TYPES: [bool; 5] = [true, true, true, true, false];
}

/// Metal's `minimumLinearTextureAlignmentForPixelFormat:` on an Apple GPU, the M3 Max's and the
/// M4's, for every pixel format a buffer backs (measured): a stand-in (the module docs). It is
/// also the alignment Metal holds a texture's rows over a buffer or a plane of an IOSurface
/// without planes to, which a device cache laid out at the GPU's pitch has its rows meet
/// (`GpuDevice::linear_rows_pitch`).
pub(crate) const METAL_LINEAR_ALIGNMENT: u64 = 16;

const _: () = assert!(FAMILIES.len() == stand_in::FAMILIES.len());
